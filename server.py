"""Voice-first AI interviewer. Run: uvicorn server:app --reload  then open http://localhost:8000"""
import asyncio
import base64
import json
import logging
import os
import secrets
import time
import uuid
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

import store  # noqa: E402
from interview import LEVELS, PERSONAS, Interview, normalize_topics  # noqa: E402
from report import build_pdf  # noqa: E402

if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    raise SystemExit("Set GEMINI_API_KEY in .env - the voice interviewer cannot run without it.")
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


class Topic(BaseModel):
    topic: str
    focus: str
    difficulty: str
    kind: Literal["discussion", "coding"]
    problem: str  # coding topics only: full problem statement shown on screen


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


def owned(sid: str, uid: str) -> Interview:
    iv = store.get(sid)
    if not iv or iv.user_id != uid:  # someone else's interview looks the same as a missing one
        raise HTTPException(404, "Unknown session")
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
    return {"google_client_id": GOOGLE_CLIENT_ID}


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
    coding = ("Make exactly one topic (not the first) a coding problem: kind 'coding', with a self-contained problem "
              "statement solvable in about 10 minutes, including one example input and output. "
              if req.coding_round else "Do not include coding problems. ")
    parts: list = [
        f"Plan a {req.difficulty}-level interview for a {req.role} candidate named {req.name}. "
        "Return 6 topics mixing technical and behavioral. Target gaps between the resume and the job description "
        "when both are given, and probe claims on the resume. For each topic give a one-line focus and a difficulty "
        f"from {LEVELS}. {coding}Every other topic has kind 'discussion' and an empty problem. "
        "Also write a 2-sentence candidate_summary of their background (or 'No resume provided').\n\n"
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
                   topics=normalize_topics([t.model_dump() for t in plan.topics], req.coding_round), user_id=uid)
    await asyncio.to_thread(store.put, iv)
    return {"id": iv.id, "topic_count": len(iv.topics), "max_minutes": MAX_INTERVIEW_MINUTES}


@app.get("/api/sessions")
def list_sessions(uid: str = Depends(current_user)):
    rows = [{"id": iv.id, "name": iv.name, "role": iv.role, "difficulty": iv.difficulty, "persona": iv.persona,
             "created_at": iv.created_at, "ended": iv.ended,
             "recommendation": (iv.evaluation or {}).get("hiringRecommendation"),
             "scores": {s["dimension"]: s["score"] for s in (iv.evaluation or {}).get("scores", [])}}
            for iv in store.list_for(uid)]
    return rows


@app.get("/api/sessions/{sid}")
def session_detail(sid: str, uid: str = Depends(current_user)):
    return asdict(owned(sid, uid)) | {"recording": store.recording_mime(sid)}


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str, uid: str = Depends(current_user)):
    owned(sid, uid)
    if sid in ACTIVE:
        raise HTTPException(409, "This interview is still in progress")
    store.delete(sid)
    return {"ok": True}


@app.post("/api/sessions/{sid}/evaluate")
async def evaluate(sid: str, uid: str = Depends(current_user)):
    iv = await asyncio.to_thread(owned, sid, uid)
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
        + "List only questions the interviewer actually asked. "
        + f"\n\nInterviewer notes per topic: {json.dumps(iv.notes)}\n\nTranscript:\n{iv.transcript_text(timestamps=True)}"
        + (f"\n\n{code}" if code else ""))
    iv.evaluation = (await generate(prompt, Evaluation)).model_dump()
    await asyncio.to_thread(store.put, iv)
    return iv.evaluation


@app.get("/api/sessions/{sid}/report.pdf")
def report_pdf(sid: str, uid: str = Depends(current_user)):
    iv = owned(sid, uid)
    if not iv.evaluation:
        raise HTTPException(400, "Evaluate the interview first")
    return Response(build_pdf(iv.evaluation, iv.name, iv.role), media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="interview_report.pdf"'})


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
    owned(sid, uid)
    rec = store.get_recording(sid)
    if not rec:
        raise HTTPException(404, "No recording for this interview")
    return Response(rec[1], media_type=rec[0])


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
    types.FunctionDeclaration(name="read_code", description="Read the candidate's code editor (coding topics only).",
                              parameters=EMPTY),
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
            "problem": t.get("problem", ""), "code": iv.code.get(str(iv.idx), "")}


MAX_RECONNECTS = 3
WRAP_UP_SECONDS = 120  # after the time-up nudge, hard stop this much later


@app.websocket("/ws/{sid}")
async def interview_socket(ws: WebSocket, sid: str):
    origin = ws.headers.get("origin", "")
    if origin and urlparse(origin).netloc != ws.headers.get("host"):
        await ws.close(code=4403)  # another site trying to ride the user's cookie
        return
    uid = session_user(ws.session)
    iv = await asyncio.to_thread(store.get, sid) if uid else None
    if not iv or iv.user_id != uid or iv.ended or sid in ACTIVE:
        # an ended interview can't be reopened (and re-billed), and one interview gets one call at a time
        await ws.close(code=4404 if not iv or iv.user_id != uid else 4409)
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
                    if data.get("type") == "code" and isinstance(data.get("code"), str):
                        iv.set_code(data["code"][:MAX_CODE_CHARS])
        finally:
            browser_done.set()

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
                    for p in sc.model_turn.parts or []:
                        if p.inline_data and p.inline_data.data:
                            await ws.send_bytes(p.inline_data.data)
                            spoke_after_end |= iv.ended
                for speaker, tr in (("user", sc.input_transcription), ("ai", sc.output_transcription)):
                    if tr and tr.text and (text := iv.add_transcript(speaker, tr.text)):
                        await ws.send_json({"type": "transcript", "speaker": speaker, "text": text})
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
                             (pump_audio(live), relay(live), time_guard(live), browser_done.wait())]
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


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
