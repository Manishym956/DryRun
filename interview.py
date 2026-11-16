"""Interview state and prompts for DryRun. No I/O here, so it is testable."""
import re
import time
from collections import Counter
from dataclasses import dataclass, field

CONTROL_TOKENS = re.compile(r"<ctrl\d+>")  # the audio model occasionally leaks these into transcripts
LEVELS = ["Beginner", "Intermediate", "Advanced"]

# persona key -> (who the interviewer is, Gemini voice, seconds of uninterrupted answer before they cut in)
PERSONAS = {
    "friendly": ("a warm, encouraging senior engineer who mentors juniors", "Aoede", 180),
    "bar_raiser": ("a rigorous FAANG bar-raiser: polite but demanding, you press hard on vague answers", "Charon", 120),
    "startup_cto": ("a fast-talking, slightly impatient startup CTO who cares about shipping and pragmatic "
                    "trade-offs, and who cuts in when an answer runs long", "Puck", 75),
}

# Public, widely published hiring values: used to flavour questions, not to claim to be the company.
COMPANY_PACKS = {
    "amazon": ("Amazon", "Leadership Principles, especially Customer Obsession, Ownership, Dive Deep, Bias for "
               "Action and Deliver Results. Behavioral answers are expected in STAR form with measurable results."),
    "google": ("Google", "General cognitive ability, role-related knowledge, leadership and 'Googleyness': comfort "
               "with ambiguity, collaboration and humility. Candidates should reason out loud."),
    "meta": ("Meta", "Move fast, focus on long-term impact, be direct. Coding values speed and working code; "
             "behavioral questions focus on conflict, impact and growth."),
    "microsoft": ("Microsoft", "Growth mindset, customer focus and collaboration. Expect stories about learning "
                  "from failure."),
    "netflix": ("Netflix", "Freedom and responsibility, candor, context not control, high performance. Expect "
                "questions on judgement and giving or receiving direct feedback."),
    "stripe": ("Stripe", "Users first, rigor, urgency and focus, clear written thinking. Practical problems "
               "over puzzles."),
}

TOPIC_KINDS = ("coding", "design")  # special rounds; everything else is a discussion


def normalize_topics(topics: list[dict], coding_round: bool, design_round: bool = False) -> list[dict]:
    """Keep at most one coding and one design topic, only for rounds that are on: the planner ignores that."""
    allowed, seen = {"coding": coding_round, "design": design_round}, set()
    for t in topics:
        kind = t.get("kind")
        if allowed.get(kind) and kind not in seen:
            seen.add(kind)
        else:
            t.update(kind="discussion", problem="")
        if t["kind"] != "coding":
            t["tests"] = []
        t.setdefault("tests", [])
    return topics


SILENCE_NUDGE = ("(The candidate has been silent for a while. Gently check in: ask whether they want a moment "
                 "or a small hint. One short sentence.)")
RAMBLE_NUDGE = ("(The candidate has been talking for a long time without pausing. Politely cut in, summarise "
                "what you heard in a few words and steer them to the point or the next question.)")


class Pacer:
    """Decides when the interviewer should break a long silence or cut off a rambling answer."""

    def __init__(self):
        self.ai_done_at: float | None = None  # when the interviewer last finished a turn
        self.user_since: float | None = None  # when the candidate started their current answer
        self.nudged = False
        self.silent_nudges = 0  # in a row: stop after two if the candidate has walked away

    def ai_spoke(self) -> None:
        self.ai_done_at = self.user_since = None
        self.nudged = False

    def ai_done(self, now: float) -> None:
        self.ai_done_at = now

    def user_spoke(self, now: float) -> None:
        self.ai_done_at = None
        self.silent_nudges = 0
        if self.user_since is None:
            self.user_since = now

    def check(self, now: float, silence_s: float, ramble_s: float) -> str | None:
        if self.nudged:
            return None  # once per gap: let the interviewer's own reply reset us
        if self.ai_done_at is not None and now - self.ai_done_at > silence_s and self.silent_nudges < 2:
            self.nudged = True
            self.silent_nudges += 1
            return SILENCE_NUDGE
        if self.user_since is not None and now - self.user_since > ramble_s:
            self.nudged = True
            return RAMBLE_NUDGE
        return None


FILLERS = re.compile(r"\b(?:um+|uh+|uhm|erm|hmm+|you know|i mean|basically|kind of|sort of)\b|,\s*like\s*,", re.I)


def speech_stats(turns: list[dict]) -> dict | None:
    """Filler words and speaking pace of the candidate, from the transcript (speech-to-text, so approximate)."""
    words = timed_words = 0
    seconds = 0.0
    fillers: Counter = Counter()
    for i, t in enumerate(turns):
        if t["speaker"] != "user":
            continue
        n = len(t["text"].split())
        words += n
        if i + 1 < len(turns):  # an answer lasts until the next turn starts; the last one has no end time
            timed_words += n
            seconds += max(0, turns[i + 1]["t_ms"] - t["t_ms"]) / 1000
        for m in FILLERS.finditer(t["text"]):
            fillers[re.sub(r"(.)\1+$", r"\1", m.group(0).lower().strip(" ,"))] += 1
    if not words:
        return None
    return {"words": words, "wpm": round(timed_words / seconds * 60) if seconds >= 10 else None,
            "fillers": dict(fillers.most_common()),
            "fillers_per_100_words": round(100 * sum(fillers.values()) / words, 1)}


@dataclass
class Interview:
    id: str
    name: str
    role: str
    persona: str
    candidate_summary: str
    topics: list[dict]  # [{topic, focus, difficulty, kind: discussion|coding|design, problem, tests}]
    turns: list[dict] = field(default_factory=list)  # [{speaker, text, t_ms}]
    notes: list[dict] = field(default_factory=list)  # [{topic, strength, gaps}]
    idx: int = -1
    ended: bool = False
    evaluation: dict | None = None
    started_at: float = field(default_factory=time.time)  # reset when the call connects; turn t_ms count from here
    code: dict[str, str] = field(default_factory=dict)  # topic index (str, for JSON) -> latest editor contents
    difficulty: str = ""
    created_at: float = field(default_factory=time.time)
    user_id: str = ""
    company: str = ""  # display name of the company pack, if any
    company_notes: str = ""
    challenge: bool = False  # "Survive the bar-raiser": survived when the recommendation is Yes
    language: str = "python"
    test_results: dict[str, dict] = field(default_factory=dict)  # topic index -> {passed, total, language}
    diagram: dict[str, dict] = field(default_factory=dict)  # topic index -> {text, svg}
    mode: str = "ai"  # ai | peer (a human interviewer, the AI only referees)
    peer_id: str = ""  # the human interviewer in peer mode
    peer_feedback: str = ""  # the referee's feedback for the human interviewer
    share_token: str = ""

    def next_topic(self) -> dict:
        self.idx += 1
        if self.idx >= len(self.topics):
            return {"done": True, "instruction": "All topics covered. Thank the candidate briefly, then call end_interview."}
        t = self.topics[self.idx]
        # Adapt: shift planned difficulty by how the last two topics went.
        recent = [n["strength"] for n in self.notes[-2:]]
        avg = sum(recent) / len(recent) if recent else 3
        shift = 1 if avg >= 4 else -1 if avg <= 2 else 0
        base = LEVELS.index(t["difficulty"]) if t["difficulty"] in LEVELS else 1
        level = LEVELS[max(0, min(len(LEVELS) - 1, base + shift))]
        out = {"topic": t["topic"], "focus": t["focus"], "difficulty": level, "kind": t.get("kind", "discussion"),
               "topics_remaining_after_this": len(self.topics) - self.idx - 1}
        if out["kind"] == "coding":
            out["problem"] = t.get("problem", "")
            out["instruction"] = ("The problem is shown on the candidate's screen with a code editor and a Run tests "
                                  "button. Introduce it briefly, let them think aloud, and call read_code to see "
                                  "their code and latest test results before commenting on it.")
        elif out["kind"] == "design":
            out["problem"] = t.get("problem", "")
            out["instruction"] = ("A whiteboard is on the candidate's screen. Ask them to sketch the architecture while "
                                  "they talk. Call read_diagram before commenting on their design.")
        return out

    @property
    def current_topic(self) -> dict | None:
        return self.topics[self.idx] if 0 <= self.idx < len(self.topics) else None

    def _current_kind(self) -> str | None:
        return (self.current_topic or {}).get("kind")

    def set_code(self, code: str, language: str | None = None) -> None:
        if self._current_kind() == "coding":
            self.code[str(self.idx)] = code
            if language:
                self.language = language

    def read_code(self) -> dict:
        code = self.code.get(str(self.idx), "")
        out = {"code": code, "language": self.language} if code.strip() else {"code": "", "note": "The editor is empty so far."}
        if r := self.test_results.get(str(self.idx)):
            out["tests"] = f"{r['passed']}/{r['total']} hidden tests passed on the last run"
        return out

    def set_tests(self, passed: int, total: int) -> None:
        if self._current_kind() == "coding":
            self.test_results[str(self.idx)] = {"passed": passed, "total": total, "language": self.language}

    def set_diagram(self, text: str, svg: str) -> None:
        if self._current_kind() == "design":
            self.diagram[str(self.idx)] = {"text": text, "svg": svg}

    def read_diagram(self) -> dict:
        text = self.diagram.get(str(self.idx), {}).get("text", "")
        return {"diagram": text} if text.strip() else {"diagram": "", "note": "The whiteboard is empty so far."}

    def code_for_eval(self) -> str:
        parts = []
        for i, c in self.code.items():
            if c.strip():
                r = self.test_results.get(i)
                tests = f" (hidden tests: {r['passed']}/{r['total']} passed)" if r else " (tests never run)"
                parts.append(f"Code written for '{self.topics[int(i)]['topic']}' in {self.language}{tests}:\n{c}")
        for i, d in self.diagram.items():
            if d.get("text", "").strip():
                parts.append(f"Whiteboard for '{self.topics[int(i)]['topic']}':\n{d['text']}")
        return "\n\n".join(parts)

    def note_answer(self, strength, gaps: str) -> dict:
        # Topic comes from our own state: the model mislabels it.
        topic = self.topics[self.idx]["topic"] if 0 <= self.idx < len(self.topics) else "General"
        try:
            strength = max(1, min(5, int(strength)))
        except (TypeError, ValueError):
            strength = 3
        self.notes.append({"topic": topic, "strength": strength, "gaps": str(gaps)})
        return {"ok": True}

    def add_transcript(self, speaker: str, text: str) -> str:
        text = CONTROL_TOKENS.sub("", text or "")
        if not text:
            return text
        if self.turns and self.turns[-1]["speaker"] == speaker:
            self.turns[-1]["text"] += text
        else:
            self.turns.append({"speaker": speaker, "text": text,
                               "t_ms": int((time.time() - self.started_at) * 1000)})
        return text

    def transcript_text(self, timestamps: bool = False) -> str:
        return "\n".join((f"[{t['t_ms'] // 1000}s] " if timestamps else "")
                         + f"{'Interviewer' if t['speaker'] == 'ai' else 'Candidate'}: {t['text'].strip()}"
                         for t in self.turns)

    @classmethod
    def from_dict(cls, d: dict) -> "Interview":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})  # tolerate files saved by older versions

    def system_instruction(self) -> str:
        who = PERSONAS.get(self.persona, PERSONAS["friendly"])[0]
        company = (f"\nYou are interviewing in the style of {self.company}, whose hiring bar emphasises: "
                   f"{self.company_notes}\nTie behavioral follow-ups to these values.\n") if self.company else ""
        challenge = ("\nThis is the 'Survive the bar-raiser' challenge: hold a very high bar, push on every vague claim, "
                     "and only give a hint if the candidate is completely stuck.\n") if self.challenge else ""
        return f"""You are {who}, conducting a live VOICE interview with {self.name} for a {self.role} role.

Candidate background: {self.candidate_summary or "unknown"}
{company}{challenge}
Rules:
- Start with a one-sentence greeting, then call next_topic and ask about the topic it returns, at the difficulty it gives.
- Ask ONE question at a time. Keep each turn short (1-3 spoken sentences). Never read lists aloud.
- If an answer is vague, shallow, or hand-wavy, ask a probing follow-up (at most 2 per topic) before moving on.
- If the candidate is stuck or silent, give a small hint rather than the answer.
- When you finish a topic, call note_answer with an honest strength from 1 (weak) to 5 (excellent) and the gaps you noticed, then call next_topic.
- For coding topics: the problem, an editor and a Run tests button are on the candidate's screen. Let them work; ask about
  their approach, complexity and edge cases. Call read_code before giving any feedback on their code, and again when they say they are done.
- For design topics: a whiteboard is on the candidate's screen. Ask about components, data flow, scaling and trade-offs.
  Call read_diagram before commenting on their diagram.
- Messages in parentheses come from the system, not the candidate. Follow them without mentioning them.
- If you are told time is up, wrap up the current topic within a minute and finish the interview.
- Never reveal scores, notes, or the topic list to the candidate.
- When next_topic returns done, thank the candidate briefly and call end_interview."""
