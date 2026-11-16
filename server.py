"""DryRun: voice-first AI mock interviewer. Run: uvicorn server:app --reload  then open http://localhost:8000"""
import asyncio
import base64
import json
import logging
import os
import secrets
import tempfile
import time
import uuid
import wave
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from google import genai
from google.auth.transport import requests as google_requests
from google.genai import types
from google.oauth2 import id_token
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")  # before importing store: it reads DATABASE_URL

import judge  # noqa: E402
import store  # noqa: E402
from interview import COMPANY_PACKS, LEVELS, PERSONAS, Interview, Pacer, normalize_topics, speech_stats  # noqa: E402
from progress import progress  # noqa: E402
from report import build_pdf  # noqa: E402

if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    raise SystemExit("Set GEMINI_API_KEY in .env - DryRun cannot run without it.")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")  # unset = local mode: no sign-in, one user
if os.getenv("RENDER") and not (GOOGLE_CLIENT_ID and os.getenv("SESSION_SECRET") and os.getenv("DATABASE_URL")):
    raise SystemExit("On Render, GOOGLE_CLIENT_ID, SESSION_SECRET and DATABASE_URL must all be set.")

LIVE_MODEL = os.getenv("LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-2.5-flash")
MAX_INTERVIEW_MINUTES = float(os.getenv("MAX_INTERVIEW_MINUTES", "30"))
SESSIONS_PER_HOUR = int(os.getenv("SESSIONS_PER_HOUR", "10"))  # per user
INTERVIEWS_PER_DAY = int(os.getenv("INTERVIEWS_PER_DAY", "3" if GOOGLE_CLIENT_ID else "1000"))  # per user
DAILY_INTERVIEW_CAP = int(os.getenv("DAILY_INTERVIEW_CAP", "100"))  # all users together: caps the Gemini bill
MAX_RESUME_BYTES = 5 * 1024 * 1024
MAX_RECORDING_BYTES = 25 * 1024 * 1024  # the browser records at 32 kbps: ~7 MB for 30 minutes
MAX_CODE_CHARS = 20_000
RESUME_MIMES = {"application/pdf", "text/plain"}
RECORDING_TYPES = {"audio/webm", "audio/ogg", "audio/mp4"}

log = logging.getLogger("interviewer")
client = genai.Client(http_options={"api_version": "v1beta"})  # Live API 404s on the SDK default version
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET") or secrets.token_hex(32),
                   https_only=bool(os.getenv("RENDER")), max_age=30 * 86400)
store.init()
LOCAL_USER = "local"
# shortcut: in-process state, so run a single instance (Render's default); move to the DB to scale out.
ACTIVE: set[str] = set()  # interviews with a live call right now: one call per interview


# ---------- Schemas ----------

class NewSession(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    role: str = Field(min_length=1, max_length=100)
    difficulty: str = Field(pattern="^(Beginner|Intermediate|Advanced)$")
    persona: str = Field(pattern="^(" + "|".join(PERSONAS) + ")$")
    job_description: str = Field(default="", max_length=20_000)
    resume_b64: str | None = Field(default=None, max_length=MAX_RESUME_BYTES * 4 // 3 + 4)
    resume_mime: str | None = None
    coding_round: bool = True
    design_round: bool = False
    company: str = Field(default="", max_length=20)  # a COMPANY_PACKS key, "custom", or ""
    company_name: str = Field(default="", max_length=100)  # custom company only
    company_notes: str = Field(default="", max_length=2000)
    challenge: bool = False  # "Survive the bar-raiser"
    mode: Literal["ai", "peer"] = "ai"


class TestCase(BaseModel):
    stdin: str
    expected: str


class Topic(BaseModel):
    topic: str
    focus: str
    difficulty: str
    kind: Literal["discussion", "coding", "design"]
    problem: str  # coding and design topics only: full statement shown on screen
    tests: list[TestCase]  # coding topics only


class Plan(BaseModel):
    candidate_summary: str
    topics: list[Topic]


class Score(BaseModel):
    dimension: str
    score: int | None  # None when the interview never covered this dimension
    justification: str


class QuestionReview(BaseModel):
    question: str
    askedAtSeconds: int | None  # from the transcript timestamps, for replay
    answerSummary: str
    modelAnswer: str
    missing: str


class Evaluation(BaseModel):
    overallFeedback: str
    hiringRecommendation: str
    scores: list[Score]
    questions: list[QuestionReview]


class WeakSpot(BaseModel):
    area: str
    evidence: str
    practice: str


class Insights(BaseModel):
    weakSpots: list[WeakSpot]
    strengths: list[str]


class InsightsRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class GoogleLogin(BaseModel):
    credential: str = Field(min_length=1, max_length=10_000)


# ---------- Helpers ----------

def session_user(session: dict) -> str | None:
    return session.get("uid") if GOOGLE_CLIENT_ID else LOCAL_USER


def current_user(request: Request) -> str:
    uid = session_user(request.session)
    if not uid:
        raise HTTPException(401, "Sign in first")
    return uid


def owned(sid: str, uid: str, peer_ok: bool = False) -> Interview:
    """The interview if uid owns it (or, with peer_ok, was its human interviewer in peer mode)."""
    iv = store.get(sid)
    if not iv or not (iv.user_id == uid or (peer_ok and iv.peer_id and iv.peer_id == uid)):
        raise HTTPException(404, "Unknown session")  # someone else's interview looks the same as a missing one
    return iv


# shortcut: in-memory limiter, resets on restart; the daily quota in the database is the real cap.
_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(key: str, limit: int, what: str) -> None:
    now, q = time.time(), _hits[key]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, f"Limit of {limit} {what} per hour reached, try again later")
    q.append(now)


def check_quota(uid: str) -> None:
    since = time.time() - 86400
    if store.count_since(since, uid) >= INTERVIEWS_PER_DAY:
        raise HTTPException(429, f"You can start {INTERVIEWS_PER_DAY} interviews per day. Try again tomorrow.")
    if store.count_since(since) >= DAILY_INTERVIEW_CAP:
        raise HTTPException(503, "The interviewer is at capacity for today. Try again tomorrow.")


async def generate(contents, schema):
    resp = await client.aio.models.generate_content(
        model=TEXT_MODEL, contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema))
    if not resp.parsed:
        raise HTTPException(502, "The AI returned no usable answer, try again")
    return resp.parsed


# ---------- Auth ----------

@app.get("/api/config")
def config():
    return {"google_client_id": GOOGLE_CLIENT_ID, "run_tests": judge.enabled(), "languages": list(judge.LANGUAGES),
            "companies": [{"id": k, "name": v[0]} for k, v in COMPANY_PACKS.items()]}


@app.post("/api/auth/google")
def google_login(req: GoogleLogin, request: Request):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(400, "Sign-in is not configured")
    try:
        claims = id_token.verify_oauth2_token(req.credential, google_requests.Request(), GOOGLE_CLIENT_ID)
    except ValueError:
        raise HTTPException(401, "Google sign-in failed, try again")
    if not claims.get("email_verified"):
        raise HTTPException(401, "Your Google email is not verified")
    uid = "google:" + claims["sub"]
    store.upsert_user(uid, claims["email"], claims.get("name") or claims["email"], time.time())
    request.session.clear()
    request.session["uid"] = uid
    return {"ok": True}


@app.post("/api/auth/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/api/me")
def me(uid: str = Depends(current_user)):
    if uid == LOCAL_USER:
        return {"id": uid, "email": "", "name": "", "local": True}
    user = store.get_user(uid)
    if not user:
        raise HTTPException(401, "Sign in first")
    return user


# ---------- REST ----------

@app.post("/api/sessions")
async def create_session(req: NewSession, uid: str = Depends(current_user)):
    rate_limit("start:" + uid, SESSIONS_PER_HOUR, "interviews")
    await asyncio.to_thread(check_quota, uid)
    if req.challenge:  # the challenge is always the hardest setting
        req.persona, req.difficulty = "bar_raiser", "Advanced"
    if req.company == "custom":
        company, notes = req.company_name.strip(), req.company_notes.strip()
        if not company:
            raise HTTPException(400, "Enter the company name")
    elif req.company:
        if req.company not in COMPANY_PACKS:
            raise HTTPException(400, "Unknown company pack")
        company, notes = COMPANY_PACKS[req.company]
    else:
        company, notes = "", ""
    coding = ("Make exactly one topic (not the first) a coding problem: kind 'coding', with a self-contained problem "
              "statement solvable in about 10 minutes. The program reads its input from standard input and prints "
              "the answer to standard output; state the exact input and output format and show two examples. Give "
              "6 tests (stdin and expected stdout), the first two being the examples, then edge cases. Double-check "
              "every expected output. "
              if req.coding_round else "Do not include coding problems. ")
    design = ("Make exactly one topic a system design question: kind 'design', with a one-paragraph problem stating "
              "the product and its scale. " if req.design_round else "Do not include system design whiteboard topics. ")
    company_line = (f"The candidate is preparing for {company}. Its hiring bar emphasises: {notes} Include at least "
                    "two behavioral topics that test these values. " if company else "")
    parts: list = [
        f"Plan a {req.difficulty}-level interview for a {req.role} candidate named {req.name}. "
        "Return 6 topics mixing technical and behavioral. Target gaps between the resume and the job description "
        "when both are given, and probe claims on the resume. For each topic give a one-line focus and a difficulty "
        f"from {LEVELS}. {company_line}{coding}{design}Every other topic has kind 'discussion', an empty problem "
        "and no tests. Also write a 2-sentence candidate_summary of their background (or 'No resume provided').\n\n"
        f"Job description:\n{req.job_description or '(none)'}"
    ]
    if req.resume_b64:
        if req.resume_mime not in RESUME_MIMES:
            raise HTTPException(400, "Resume must be PDF or plain text")
        try:
            data = base64.b64decode(req.resume_b64, validate=True)
        except ValueError:
            raise HTTPException(400, "Resume is not valid base64")
        if len(data) > MAX_RESUME_BYTES:
            raise HTTPException(400, "Resume too large (5 MB max)")
        parts.append("Resume:")
        parts.append(types.Part.from_bytes(data=data, mime_type=req.resume_mime))

    plan: Plan = await generate(parts, Plan)
    if not plan.topics:
        raise HTTPException(502, "Could not generate an interview plan, try again")
    iv = Interview(id=uuid.uuid4().hex, name=req.name.strip(), role=req.role.strip(), persona=req.persona,
                   difficulty=req.difficulty, candidate_summary=plan.candidate_summary,
                   topics=normalize_topics([t.model_dump() for t in plan.topics], req.coding_round, req.design_round),
                   user_id=uid, company=company, company_notes=notes, challenge=req.challenge and req.mode == "ai",
                   mode=req.mode)
    await asyncio.to_thread(store.put, iv)
    return {"id": iv.id, "topic_count": len(iv.topics), "max_minutes": MAX_INTERVIEW_MINUTES}


@app.get("/api/sessions")
def list_sessions(uid: str = Depends(current_user)):
    rows = [{"id": iv.id, "name": iv.name, "role": iv.role, "difficulty": iv.difficulty, "persona": iv.persona,
             "created_at": iv.created_at, "ended": iv.ended, "challenge": iv.challenge, "company": iv.company,
             "mode": iv.mode,
             "recommendation": (iv.evaluation or {}).get("hiringRecommendation"),
             "scores": {s["dimension"]: s["score"] for s in (iv.evaluation or {}).get("scores", [])}}
            for iv in store.list_for(uid)]
    return rows


@app.get("/api/sessions/{sid}")
def session_detail(sid: str, uid: str = Depends(current_user)):
    iv = owned(sid, uid, peer_ok=True)
    d = asdict(iv)
    for t in d["topics"]:
        t["tests"] = t.get("tests", [])[:judge.VISIBLE_TESTS]  # hidden tests stay hidden, even afterwards
    return d | {"recording": store.recording_mime(sid), "speech": speech_stats(iv.turns), "owner": iv.user_id == uid}


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str, uid: str = Depends(current_user)):
    owned(sid, uid)
    if sid in ACTIVE:
        raise HTTPException(409, "This interview is still in progress")
    store.delete(sid)
    return {"ok": True}


@app.post("/api/sessions/{sid}/evaluate")
async def evaluate(sid: str, uid: str = Depends(current_user)):
    iv = await asyncio.to_thread(owned, sid, uid, True)
    if iv.evaluation:
        return iv.evaluation
    if sid in ACTIVE:
        raise HTTPException(409, "This interview is still in progress")
    if not any(t["speaker"] == "user" for t in iv.turns):
        raise HTTPException(400, "The candidate did not say anything, nothing to evaluate")
    code = iv.code_for_eval()
    prompt = (
        f"You are a senior hiring manager evaluating a {iv.role} candidate from a voice interview transcript "
        "(speech-to-text, so ignore transcription errors). Captions are split wherever either side paused or "
        "interrupted, so judge communication by the substance and structure of what was said, not by fragmented "
        "lines or filler words. Score Technical Knowledge, Behavioral Fit and Communication from 1-10 with "
        "justifications; if the interview never gave evidence for a dimension, set its score to null and say so. "
        "hiringRecommendation must be Yes, No or Maybe. "
        "For each main question asked, give askedAtSeconds (the [Ns] timestamp where the interviewer asked it), "
        "summarize the candidate's answer, write a concise strong model answer, and state what was missing. "
        + ("For the coding question, judge the code below for correctness, complexity and edge cases inside that "
           "question's review; do not add a separate entry for the code. " if code else "")
        + (f"Judge behavioral fit against {iv.company}'s values: {iv.company_notes} " if iv.company else "")
        + ("The interviewer was a human peer, and the transcript was made from the call audio afterwards. "
           if iv.mode == "peer" else "")
        + "List only questions the interviewer actually asked. "
        + f"\n\nInterviewer notes per topic: {json.dumps(iv.notes)}\n\nTranscript:\n{iv.transcript_text(timestamps=True)}"
        + (f"\n\n{code}" if code else ""))
    iv.evaluation = (await generate(prompt, Evaluation)).model_dump()
    await asyncio.to_thread(store.put, iv)
    return iv.evaluation


@app.get("/api/sessions/{sid}/report.pdf")
def report_pdf(sid: str, uid: str = Depends(current_user)):
    iv = owned(sid, uid, peer_ok=True)
    if not iv.evaluation:
        raise HTTPException(400, "Evaluate the interview first")
    return Response(build_pdf(iv.evaluation, iv.name, iv.role), media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="dryrun_report.pdf"'})


@app.post("/api/sessions/{sid}/recording")
async def upload_recording(sid: str, request: Request, uid: str = Depends(current_user)):
    await asyncio.to_thread(owned, sid, uid)
    mime = request.headers.get("content-type", "").split(";")[0].strip()
    if mime not in RECORDING_TYPES:
        raise HTTPException(400, "Recording must be webm, ogg or mp4 audio")
    if await asyncio.to_thread(store.recording_mime, sid):
        raise HTTPException(409, "This interview already has a recording")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_RECORDING_BYTES:
            raise HTTPException(413, "Recording too large")
    if not body:
        raise HTTPException(400, "Empty recording")
    try:
        await asyncio.to_thread(store.put_recording, sid, mime, bytes(body))
    except Exception:
        log.exception("saving recording for %s failed", sid)
        raise HTTPException(409, "This interview already has a recording")  # lost a race with another upload
    return {"ok": True}


@app.get("/api/sessions/{sid}/recording")
def get_recording(sid: str, uid: str = Depends(current_user)):
    owned(sid, uid, peer_ok=True)
    rec = store.get_recording(sid)
    if not rec:
        raise HTTPException(404, "No recording for this interview")
    return Response(rec[1], media_type=rec[0])


@app.post("/api/sessions/{sid}/share")
def share_session(sid: str, uid: str = Depends(current_user)):
    iv = owned(sid, uid)
    if not iv.evaluation:
        raise HTTPException(400, "Finish the evaluation before sharing")
    if not iv.share_token:
        iv.share_token = secrets.token_urlsafe(16)
        store.share(iv.share_token, sid)
        store.put(iv)
    return {"token": iv.share_token}


@app.delete("/api/sessions/{sid}/share")
def unshare_session(sid: str, uid: str = Depends(current_user)):
    iv = owned(sid, uid)
    store.unshare(sid)
    iv.share_token = ""
    store.put(iv)
    return {"ok": True}


@app.get("/api/share/{token}")
def shared_scorecard(token: str):
    """Public and read-only: scores and the summary only, never the transcript, code or recording."""
    iv = store.shared(token)
    if not iv or not iv.evaluation:
        raise HTTPException(404, "This scorecard is not shared")
    ev = iv.evaluation
    return {"name": iv.name, "role": iv.role, "difficulty": iv.difficulty, "created_at": iv.created_at,
            "company": iv.company, "challenge": iv.challenge, "mode": iv.mode,
            "recommendation": ev.get("hiringRecommendation"), "overallFeedback": ev.get("overallFeedback"),
            "scores": [{"dimension": x["dimension"], "score": x["score"]} for x in ev.get("scores", [])]}


@app.get("/api/progress")
def my_progress(tz: int = 0, uid: str = Depends(current_user)):
    return progress(store.list_for(uid), max(-840, min(840, tz)))


@app.post("/api/insights")
async def insights(req: InsightsRequest, uid: str = Depends(current_user)):
    rate_limit("insights:" + uid, 10, "weak-spot analyses")
    mine = await asyncio.to_thread(store.list_for, uid)
    done = [iv for iv in mine if iv.name.lower() == req.name.strip().lower() and iv.evaluation]
    if not done:
        raise HTTPException(404, "No evaluated interviews for this name yet")
    history = [{"date": time.strftime("%Y-%m-%d", time.localtime(iv.created_at)), "role": iv.role,
                "scores": iv.evaluation["scores"], "notes": iv.notes,
                "missing": [q["missing"] for q in iv.evaluation.get("questions", [])]}
               for iv in sorted(done, key=lambda i: i.created_at)]
    prompt = ("You are an interview coach. Across these mock interviews, find the 3-5 weaknesses that RECUR "
              "(not one-offs), each with evidence from the history and one concrete practice exercise. Then list "
              "2-3 consistent strengths. Be specific and brief.\n\n" + json.dumps(history))
    return (await generate(prompt, Insights)).model_dump()


# ---------- Live voice relay ----------

EMPTY = types.Schema(type="OBJECT", properties={})
TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(name="next_topic", description="Get the next interview topic and difficulty.",
                              parameters=EMPTY),
    types.FunctionDeclaration(
        name="note_answer", description="Privately record how the candidate did on the topic just finished.",
        parameters=types.Schema(type="OBJECT", required=["strength", "gaps"], properties={
            "strength": types.Schema(type="INTEGER", description="1 weak to 5 excellent"),
            "gaps": types.Schema(type="STRING", description="What was missing or wrong"),
        })),
    types.FunctionDeclaration(name="read_code", description="Read the candidate's code editor and latest test results "
                              "(coding topics only).", parameters=EMPTY),
    types.FunctionDeclaration(name="read_diagram", description="Read the candidate's whiteboard as a list of boxes and "
                              "arrows (design topics only).", parameters=EMPTY),
    types.FunctionDeclaration(name="end_interview", description="End the interview after saying goodbye.",
                              parameters=EMPTY),
])]


def run_tool(iv: Interview, fc: types.FunctionCall) -> dict:
    args = fc.args or {}
    if fc.name == "next_topic":
        return iv.next_topic()
    if fc.name == "note_answer":
        return iv.note_answer(args.get("strength"), args.get("gaps", ""))
    if fc.name == "read_code":
        return iv.read_code()
    if fc.name == "read_diagram":
        return iv.read_diagram()
    if fc.name == "end_interview":
        iv.ended = True
        return {"ok": True, "instruction": "Say a one-sentence goodbye."}
    return {"error": f"unknown tool {fc.name}"}


def live_config(iv: Interview, handle: str | None) -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=iv.system_instruction(),
        tools=TOOLS,
        # Pinned to English: auto-detect turned Indian-accented English into Devanagari.
        input_audio_transcription=types.AudioTranscriptionConfig(language_codes=["en-US"]),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=PERSONAS[iv.persona][1]))),
        # Candidates pause to think: wait 1.2 s of silence before treating the answer as finished.
        realtime_input_config=types.RealtimeInputConfig(automatic_activity_detection=types.AutomaticActivityDetection(
            end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW, silence_duration_ms=1200)),
        # Lets audio sessions run past the ~15 min cap by trimming old context.
        context_window_compression=types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow()),
        session_resumption=types.SessionResumptionConfig(handle=handle),
    )


TIME_UP = "(Time is up. Wrap up the current topic now, thank the candidate and call end_interview right after your goodbye.)"


def kickoff_text(iv: Interview, resuming: bool, time_up: bool) -> str:
    if time_up:
        # Reconnected after the time-up nudge: a resumed model tends to forget it and carry on.
        return ("(The connection dropped briefly and is restored. " if resuming else
                f"(The connection dropped. Transcript so far:\n{iv.transcript_text()}\n") + TIME_UP[1:]
    if resuming:
        return "(The connection dropped briefly and is restored. Apologize in a few words and continue where you left off.)"
    if iv.idx < 0:
        # Hand over the first topic directly: the audio model sometimes skips the first next_topic call.
        return ("The candidate has joined. Greet them and begin the interview. First topic "
                f"(already fetched, do not call next_topic for it): {json.dumps(iv.next_topic())}")
    # Reconnected before Gemini gave us a resume handle: rebuild context from our own transcript.
    return ("(The connection dropped and you lost your memory of the conversation. Transcript so far:\n"
            f"{iv.transcript_text()}\n\nCurrent topic: {json.dumps(iv.current_topic)}. Apologize in a few words and "
            "continue the interview from where it stopped. Do not greet the candidate again.)")


def topic_event(iv: Interview) -> dict:
    t = iv.current_topic or {}
    return {"type": "topic", "kind": t.get("kind", "discussion"), "topic": t.get("topic", ""),
            "problem": t.get("problem", ""), "code": iv.code.get(str(iv.idx), ""), "language": iv.language,
            "can_run": judge.enabled() and bool(t.get("tests"))}


MAX_RECONNECTS = 3
SILENCE_NUDGE_SECONDS = 35  # counted from when generation ends; playback runs on for a few seconds after that
WORK_SILENCE_NUDGE_SECONDS = 150  # coding and whiteboard: quiet thinking is expected
MAX_DIAGRAM_CHARS = 10_000
MAX_DIAGRAM_SVG = 500_000
WRAP_UP_SECONDS = 120  # after the time-up nudge, hard stop this much later


@app.websocket("/ws/{sid}")
async def interview_socket(ws: WebSocket, sid: str):
    origin = ws.headers.get("origin", "")
    if origin and urlparse(origin).netloc != ws.headers.get("host"):
        await ws.close(code=4403)  # another site trying to ride the user's cookie
        return
    uid = session_user(ws.session)
    iv = await asyncio.to_thread(store.get, sid) if uid else None
    if not iv or iv.user_id != uid or iv.mode != "ai" or iv.ended or sid in ACTIVE:
        # an ended interview can't be reopened (and re-billed), and one interview gets one call at a time
        await ws.close(code=4404 if not iv or iv.user_id != uid or iv.mode != "ai" else 4409)
        return
    ACTIVE.add(sid)
    await ws.accept()
    iv.started_at = time.time()  # turn timestamps line up with the browser's recording from here
    deadline = iv.started_at + MAX_INTERVIEW_MINUTES * 60
    await ws.send_json({"type": "limits", "max_minutes": MAX_INTERVIEW_MINUTES})
    # Mic audio outlives any single Gemini connection, so buffer it between reconnects.
    audio: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)  # ~5 s
    browser_done = asyncio.Event()

    async def read_browser():
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    return
                if msg.get("bytes"):
                    if audio.full():
                        audio.get_nowait()  # drop oldest rather than block during a reconnect
                    audio.put_nowait(msg["bytes"])
                elif msg.get("text"):
                    try:
                        data = json.loads(msg["text"])
                    except ValueError:
                        continue  # ignore malformed control messages
                    if data.get("type") == "end":
                        return
                    kind = data.get("type")
                    if kind in ("code", "run") and isinstance(data.get("code"), str):
                        lang = data.get("language") if data.get("language") in judge.LANGUAGES else None
                        iv.set_code(data["code"][:MAX_CODE_CHARS], lang)
                        if kind == "run" and not running_tests:
                            asyncio.create_task(run_tests(data["code"][:MAX_CODE_CHARS], lang or iv.language))
                    elif kind == "diagram" and isinstance(data.get("text"), str) and isinstance(data.get("svg"), str):
                        iv.set_diagram(data["text"][:MAX_DIAGRAM_CHARS],
                                       data["svg"] if len(data["svg"]) <= MAX_DIAGRAM_SVG else "")
        finally:
            browser_done.set()

    running_tests = False

    async def run_tests(code: str, language: str):
        nonlocal running_tests
        topic, idx = iv.current_topic or {}, iv.idx
        if not judge.enabled() or topic.get("kind") != "coding" or not topic.get("tests"):
            return
        running_tests = True
        try:
            rate_limit("run:" + uid, 30, "test runs")
            result = await judge.run_tests(code, language, topic["tests"])
            if iv.idx == idx:
                iv.set_tests(result["passed"], result["total"])
            await ws.send_json({"type": "tests", **result})
        except HTTPException as e:
            await ws.send_json({"type": "tests", "error": e.detail})
        except Exception:
            log.exception("test run failed for %s", sid)
            await ws.send_json({"type": "tests", "error": "Could not run the tests right now."})
        finally:
            running_tests = False

    pacer = Pacer()
    ramble_s = PERSONAS[iv.persona][2]

    async def nudge(live):
        while True:
            await asyncio.sleep(1)
            if iv.ended or warned:
                continue  # wrapping up: no more nudges
            busy = (iv.current_topic or {}).get("kind") in ("coding", "design")
            if text := pacer.check(time.time(), WORK_SILENCE_NUDGE_SECONDS if busy else SILENCE_NUDGE_SECONDS,
                                   ramble_s * (2 if busy else 1)):
                await live.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=text)]),
                                               turn_complete=True)

    async def pump_audio(live):
        while True:
            await live.send_realtime_input(audio=types.Blob(data=await audio.get(), mime_type="audio/pcm;rate=16000"))

    warned = False

    async def time_guard(live):
        nonlocal warned
        if not warned:
            await asyncio.sleep(max(0, deadline - time.time()))
            warned = True
            await ws.send_json({"type": "status", "text": "Time is up, the interviewer is wrapping up."})
            await live.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=TIME_UP)]),
                                           turn_complete=True)
        await asyncio.sleep(max(0, deadline + WRAP_UP_SECONDS - time.time()))  # then end, finished or not

    handle: str | None = None

    async def relay(live):
        nonlocal handle
        spoke_after_end = False
        while True:
            async for m in live.receive():
                u = m.session_resumption_update
                if u and u.new_handle:
                    handle = u.new_handle
                if m.tool_call:
                    calls = m.tool_call.function_calls
                    await live.send_tool_response(function_responses=[
                        types.FunctionResponse(id=fc.id, name=fc.name, response=run_tool(iv, fc)) for fc in calls])
                    if any(fc.name == "next_topic" for fc in calls) and iv.current_topic:
                        await ws.send_json(topic_event(iv))
                sc = m.server_content
                if not sc:
                    continue
                if sc.interrupted:
                    await ws.send_json({"type": "interrupted"})
                if sc.model_turn:
                    pacer.ai_spoke()
                    for p in sc.model_turn.parts or []:
                        if p.inline_data and p.inline_data.data:
                            await ws.send_bytes(p.inline_data.data)
                            spoke_after_end |= iv.ended
                for speaker, tr in (("user", sc.input_transcription), ("ai", sc.output_transcription)):
                    if tr and tr.text and (text := iv.add_transcript(speaker, tr.text)):
                        await ws.send_json({"type": "transcript", "speaker": speaker, "text": text})
                        if speaker == "user":
                            pacer.user_spoke(time.time())
                if sc.turn_complete:
                    pacer.ai_done(time.time())
                if sc.turn_complete and spoke_after_end:
                    return

    reader = asyncio.create_task(read_browser())
    error = None
    try:
        for attempt in range(MAX_RECONNECTS + 1):
            resuming = handle is not None
            try:
                async with client.aio.live.connect(model=LIVE_MODEL, config=live_config(iv, handle)) as live:
                    await live.send_client_content(turns=types.Content(role="user", parts=[
                        types.Part(text=kickoff_text(iv, resuming, warned))]), turn_complete=True)
                    await ws.send_json(topic_event(iv))
                    if attempt:
                        await ws.send_json({"type": "status", "text": "Reconnected."})
                    tasks = [asyncio.create_task(c) for c in
                             (pump_audio(live), relay(live), time_guard(live), nudge(live), browser_done.wait())]
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for t in pending:
                        t.cancel()
                    failed = next((e for e in [t.exception() for t in done] if e), None)  # read all: no "never retrieved" noise
                    if not failed:
                        error = None
                        break  # candidate ended it, the goodbye finished, or time ran out
                    raise failed
            except Exception as e:
                error = e
                log.warning("interview %s live connection lost (attempt %d): %r", sid, attempt + 1, e)
                if browser_done.is_set():
                    break
                if attempt < MAX_RECONNECTS:
                    await ws.send_json({"type": "status", "text": "Connection hiccup, reconnecting…"})
    except Exception:
        log.exception("interview %s relay failed", sid)  # browser socket itself broke
    finally:
        reader.cancel()
        iv.ended = True
        ACTIVE.discard(sid)
        await asyncio.to_thread(store.put, iv)
    try:
        if error:
            await ws.send_json({"type": "error", "text": "The interview connection failed and could not be restored."})
        await ws.send_json({"type": "ended"})
        await ws.close()
    except Exception:
        pass  # browser already gone


# ---------- Peer mock interviews: two people talk through us, the AI referees afterwards ----------

class PeerTurn(BaseModel):
    speaker: Literal["interviewer", "candidate"]
    atSeconds: int
    text: str


class PeerReview(BaseModel):
    turns: list[PeerTurn]
    interviewerFeedback: str


PEER_RATE = 16000  # both browsers send 16 kHz mono PCM16
MAX_PEER_ROOMS = int(os.getenv("MAX_PEER_ROOMS", "5"))  # each live room keeps two WAV files on local disk


class Room:
    def __init__(self, iv: Interview):
        self.iv = iv
        self.socks: dict[str, WebSocket] = {}  # role -> socket
        self.start: float | None = None  # when both first connected; audio and transcript times count from here
        self.dir = tempfile.mkdtemp(prefix="peer-")
        self.wavs: dict[str, wave.Wave_write] = {}
        self.frames = {"interviewer": 0, "candidate": 0}
        self.warned = False
        self.final: asyncio.Task | None = None

    def write_audio(self, role: str, pcm: bytes) -> None:
        if self.start is None:
            return
        if role not in self.wavs:
            w = wave.open(os.path.join(self.dir, f"{role}.wav"), "wb")
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(PEER_RATE)
            self.wavs[role] = w
        # Pad silence up to "now" so both files stay aligned with the call's clock.
        gap = int((time.time() - self.start) * PEER_RATE) - len(pcm) // 2 - self.frames[role]
        if gap > 0:
            self.wavs[role].writeframes(b"\0\0" * gap)
            self.frames[role] += gap
        self.wavs[role].writeframes(pcm)
        self.frames[role] += len(pcm) // 2

    async def send(self, role: str, msg: dict) -> None:
        if ws := self.socks.get(role):
            try:
                await ws.send_json(msg)
            except Exception:
                pass  # that side is gone; their own handler cleans up

    async def broadcast_topic(self) -> None:
        iv = self.iv
        await self.send("candidate", topic_event(iv))
        t = iv.current_topic
        plan = {"type": "topic", "done": t is None}
        if t:
            plan |= {"topic": t["topic"], "focus": t["focus"], "difficulty": t["difficulty"], "kind": t.get("kind"),
                     "problem": t.get("problem", ""), "remaining": len(iv.topics) - iv.idx - 1}
        await self.send("interviewer", plan)


ROOMS: dict[str, Room] = {}


async def referee(room: Room) -> None:
    """Transcribe both sides from the recorded audio, then save; the report page runs the usual evaluation."""
    iv = room.iv
    iv.ended = True
    for w in room.wavs.values():
        w.close()
    uploaded = []
    try:
        if room.wavs:
            for role in ("interviewer", "candidate"):
                path = os.path.join(room.dir, f"{role}.wav")
                if os.path.exists(path):
                    uploaded.append((role, await client.aio.files.upload(file=path, config={"mime_type": "audio/wav"})))
            parts: list = [
                "These are the two microphones of a mock job interview between two people, time-aligned from the "
                "start of the call. Write a transcript of the whole conversation in time order: one entry per turn "
                "with the speaker, atSeconds (when the turn starts) and the words spoken (ignore silence and echo of "
                "the other speaker). Then write interviewerFeedback for the human interviewer: question quality, "
                "follow-ups, fairness, time management and one thing to do better, in 3-5 sentences."]
            for role, f in uploaded:
                parts += [f"{role.capitalize()}'s microphone:", types.Part.from_uri(file_uri=f.uri, mime_type="audio/wav")]
            review: PeerReview = await generate(parts, PeerReview)
            iv.turns = [{"speaker": "ai" if t.speaker == "interviewer" else "user", "text": t.text.strip(),
                         "t_ms": max(0, t.atSeconds) * 1000} for t in sorted(review.turns, key=lambda t: t.atSeconds)]
            iv.peer_feedback = review.interviewerFeedback
    except Exception:
        log.exception("peer referee failed for %s", iv.id)
    finally:
        for _, f in uploaded:
            try:
                await client.aio.files.delete(name=f.name)
            except Exception:
                pass
        for name in os.listdir(room.dir):
            os.remove(os.path.join(room.dir, name))
        os.rmdir(room.dir)
        ACTIVE.discard(iv.id)
        ROOMS.pop(iv.id, None)
        await asyncio.to_thread(store.put, iv)


@app.websocket("/ws/peer/{sid}")
async def peer_socket(ws: WebSocket, sid: str):
    origin = ws.headers.get("origin", "")
    if origin and urlparse(origin).netloc != ws.headers.get("host"):
        await ws.close(code=4403)
        return
    uid = session_user(ws.session)
    room = ROOMS.get(sid)
    iv = room.iv if room else (await asyncio.to_thread(store.get, sid) if uid else None)
    if not uid or not iv or iv.mode != "peer":
        await ws.close(code=4404)
        return
    # Local mode has one user, so there the second tab to join becomes the interviewer.
    owner_seat_free = "candidate" not in (room.socks if room else {})
    role = "candidate" if iv.user_id == uid and (uid != LOCAL_USER or owner_seat_free) else "interviewer"
    if role == "interviewer" and iv.peer_id and iv.peer_id != uid:
        await ws.close(code=4404)  # someone else already took the interviewer seat
        return
    if iv.ended or (room and (role in room.socks or room.final)) or (not room and len(ROOMS) >= MAX_PEER_ROOMS):
        await ws.close(code=4409)
        return
    if not room:
        room = ROOMS[sid] = Room(iv)
        ACTIVE.add(sid)
    if role == "interviewer" and not iv.peer_id:
        iv.peer_id = uid
        await asyncio.to_thread(store.put, iv)  # lets them open the report afterwards
    await ws.accept()
    room.socks[role] = ws
    other = "interviewer" if role == "candidate" else "candidate"
    await ws.send_json({"type": "limits", "max_minutes": MAX_INTERVIEW_MINUTES})
    if other in room.socks:
        if room.start is None:
            room.start = iv.started_at = time.time()
            iv.next_topic()
        for r in room.socks:
            await room.send(r, {"type": "peer", "status": "connected"})
        await room.broadcast_topic()
    else:
        await ws.send_json({"type": "peer", "status": "waiting"})

    ending = False
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if room.start and time.time() > room.start + MAX_INTERVIEW_MINUTES * 60:
                if not room.warned:
                    room.warned = True
                    for r in room.socks:
                        await room.send(r, {"type": "status", "text": "Time is up: wrap up in the next two minutes."})
                if time.time() > room.start + MAX_INTERVIEW_MINUTES * 60 + WRAP_UP_SECONDS:
                    ending = True
                    break
            if msg.get("bytes"):
                room.write_audio(role, msg["bytes"])
                if peer := room.socks.get(other):
                    try:
                        await peer.send_bytes(msg["bytes"])
                    except Exception:
                        pass
                continue
            try:
                data = json.loads(msg.get("text") or "")
            except ValueError:
                continue
            kind = data.get("type")
            if kind == "end":
                ending = True
                break
            if role == "interviewer" and room.start:
                if kind == "note":
                    iv.note_answer(data.get("strength"), str(data.get("gaps", ""))[:2000])
                elif kind == "next" and iv.idx < len(iv.topics):
                    iv.next_topic()
                    await room.broadcast_topic()
            elif role == "candidate":
                if kind in ("code", "run") and isinstance(data.get("code"), str):
                    lang = data.get("language") if data.get("language") in judge.LANGUAGES else None
                    iv.set_code(data["code"][:MAX_CODE_CHARS], lang)
                    await room.send("interviewer", {"type": "code_view", "code": data["code"][:MAX_CODE_CHARS],
                                                    "language": iv.language})
                    if kind == "run":
                        asyncio.create_task(peer_tests(room, uid, data["code"][:MAX_CODE_CHARS], lang or iv.language))
                elif kind == "diagram" and isinstance(data.get("text"), str) and isinstance(data.get("svg"), str):
                    svg = data["svg"] if len(data["svg"]) <= MAX_DIAGRAM_SVG else ""
                    iv.set_diagram(data["text"][:MAX_DIAGRAM_CHARS], svg)
                    await room.send("interviewer", {"type": "diagram_view", "svg": svg})
    except Exception:
        log.exception("peer socket %s (%s) failed", sid, role)
    finally:
        room.socks.pop(role, None)
    if not room.socks and room.start is None and not ending:
        ROOMS.pop(sid, None)  # left before the call began (e.g. a page refresh): keep the interview open
        ACTIVE.discard(sid)
        os.rmdir(room.dir)
        return
    if ending or not room.socks:  # someone ended it, or both are gone
        if room.final is None:
            room.final = asyncio.create_task(referee(room))
            for r in list(room.socks):
                await room.send(r, {"type": "status", "text": "Interview over. The AI referee is reviewing the call…"})
        try:
            await ws.send_json({"type": "status", "text": "Interview over. The AI referee is reviewing the call…"})
        except Exception:
            pass
        await asyncio.shield(room.final)
        for r in list(room.socks):
            await room.send(r, {"type": "ended"})
        try:
            await ws.send_json({"type": "ended"})
            await ws.close()
        except Exception:
            pass
    else:
        await room.send(other, {"type": "peer", "status": "left"})


async def peer_tests(room: Room, uid: str, code: str, language: str) -> None:
    iv, topic, idx = room.iv, room.iv.current_topic or {}, room.iv.idx
    if not judge.enabled() or topic.get("kind") != "coding" or not topic.get("tests"):
        return
    try:
        rate_limit("run:" + uid, 30, "test runs")
        result = await judge.run_tests(code, language, topic["tests"])
        if iv.idx == idx:
            iv.set_tests(result["passed"], result["total"])
        for r in ("candidate", "interviewer"):
            await room.send(r, {"type": "tests", **result})
    except HTTPException as e:
        await room.send("candidate", {"type": "tests", "error": e.detail})
    except Exception:
        log.exception("peer test run failed for %s", iv.id)
        await room.send("candidate", {"type": "tests", "error": "Could not run the tests right now."})


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
