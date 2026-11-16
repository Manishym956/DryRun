"""Interview state and prompts for the voice interviewer. No I/O here, so it is testable."""
import re
import time
from dataclasses import dataclass, field

CONTROL_TOKENS = re.compile(r"<ctrl\d+>")  # the audio model occasionally leaks these into transcripts
LEVELS = ["Beginner", "Intermediate", "Advanced"]

PERSONAS = {
    "friendly": ("a warm, encouraging senior engineer who mentors juniors", "Aoede"),
    "bar_raiser": ("a rigorous FAANG bar-raiser: polite but demanding, you press hard on vague answers", "Charon"),
    "startup_cto": ("a fast-talking startup CTO who cares about shipping and pragmatic trade-offs", "Puck"),
}


def normalize_topics(topics: list[dict], coding_round: bool) -> list[dict]:
    """Keep at most one coding topic (none if the round is off): the planner ignores that instruction."""
    seen = False
    for t in topics:
        if t.get("kind") == "coding" and coding_round and not seen:
            seen = True
        else:
            t.update(kind="discussion", problem="")
    return topics


@dataclass
class Interview:
    id: str
    name: str
    role: str
    persona: str
    candidate_summary: str
    topics: list[dict]  # [{topic, focus, difficulty, kind: discussion|coding, problem}]
    turns: list[dict] = field(default_factory=list)  # [{speaker, text, t_ms}]
    notes: list[dict] = field(default_factory=list)  # [{topic, strength, gaps}]
    idx: int = -1
    ended: bool = False
    evaluation: dict | None = None
    started_at: float = field(default_factory=time.time)  # reset when the call connects; turn t_ms count from here
    code: dict[str, str] = field(default_factory=dict)  # topic index (str, for JSON) -> latest editor contents
    recording: str | None = None  # file name in sessions/
    difficulty: str = ""
    created_at: float = field(default_factory=time.time)

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
            out["instruction"] = ("The problem is shown on the candidate's screen with a code editor. Introduce it briefly, "
                                  "let them think aloud, and call read_code to see their code before commenting on it.")
        return out

    @property
    def current_topic(self) -> dict | None:
        return self.topics[self.idx] if 0 <= self.idx < len(self.topics) else None

    def set_code(self, code: str) -> None:
        if self.current_topic and self.current_topic.get("kind") == "coding":
            self.code[str(self.idx)] = code

    def read_code(self) -> dict:
        code = self.code.get(str(self.idx), "")
        return {"code": code} if code.strip() else {"code": "", "note": "The editor is empty so far."}

    def code_for_eval(self) -> str:
        return "\n\n".join(f"Code written for '{self.topics[int(i)]['topic']}':\n{c}"
                           for i, c in self.code.items() if c.strip())

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
        return f"""You are {who}, conducting a live VOICE interview with {self.name} for a {self.role} role.

Candidate background: {self.candidate_summary or "unknown"}

Rules:
- Start with a one-sentence greeting, then call next_topic and ask about the topic it returns, at the difficulty it gives.
- Ask ONE question at a time. Keep each turn short (1-3 spoken sentences). Never read lists aloud.
- If an answer is vague, shallow, or hand-wavy, ask a probing follow-up (at most 2 per topic) before moving on.
- If the candidate is stuck or silent, give a small hint rather than the answer.
- When you finish a topic, call note_answer with an honest strength from 1 (weak) to 5 (excellent) and the gaps you noticed, then call next_topic.
- For coding topics: the problem and an editor are on the candidate's screen. Let them work; ask about their approach,
  complexity and edge cases. Call read_code before giving any feedback on their code, and again when they say they are done.
- If you are told time is up, wrap up the current topic within a minute and finish the interview.
- Never reveal scores, notes, or the topic list to the candidate.
- When next_topic returns done, thank the candidate briefly and call end_interview."""
