"""Run: python test_interview.py"""
from interview import Interview


def make():
    topics = [{"topic": f"T{i}", "focus": "f", "difficulty": "Intermediate"} for i in range(3)]
    return Interview(id="x", name="A", role="SDE", persona="friendly", candidate_summary="", topics=topics)


def test_difficulty_adapts_and_finishes():
    iv = make()
    assert iv.next_topic()["difficulty"] == "Intermediate"  # no notes yet
    iv.note_answer(5, "")
    assert iv.next_topic()["difficulty"] == "Advanced"  # strong -> harder
    iv.note_answer(1, "")
    iv.note_answer("1", "")  # string from model still parsed
    assert [n["topic"] for n in iv.notes] == ["T0", "T1", "T1"]  # topic taken from state, not the model
    assert iv.next_topic()["difficulty"] == "Beginner"  # weak -> easier
    assert iv.next_topic()["done"] is True


def test_note_answer_clamps_bad_input():
    iv = make()
    iv.note_answer(99, "g")
    iv.note_answer("abc", "g")
    assert [n["strength"] for n in iv.notes] == [5, 3]
    assert iv.notes[0]["topic"] == "General"  # before any topic started


def test_transcript_merges_fragments():
    iv = make()
    for s, t in [("ai", "Hello "), ("ai", "there.<ctrl46>"), ("ai", "<ctrl46>"), ("user", "Hi"), ("ai", "Q?")]:
        iv.add_transcript(s, t)
    assert [t["text"] for t in iv.turns] == ["Hello there.", "Hi", "Q?"]
    assert iv.transcript_text().startswith("Interviewer: Hello there.\nCandidate: Hi")


def test_coding_topic_and_code_tracking():
    iv = make()
    iv.topics[1].update(kind="coding", problem="Reverse a list")
    iv.next_topic()
    iv.set_code("ignored")  # discussion topic: not stored
    t = iv.next_topic()
    assert t["kind"] == "coding" and t["problem"] == "Reverse a list"
    assert iv.read_code()["code"] == ""
    iv.set_code("def rev(x): return x[::-1]")
    assert iv.read_code() == {"code": "def rev(x): return x[::-1]"}
    assert iv.code == {"1": "def rev(x): return x[::-1]"}
    assert "Code written for 'T1'" in iv.code_for_eval()


def test_normalize_topics():
    from interview import normalize_topics
    mk = lambda: [{"kind": k, "problem": "p" if k == "coding" else ""} for k in ["discussion", "coding", "coding"]]
    assert [t["kind"] for t in normalize_topics(mk(), True)] == ["discussion", "coding", "discussion"]
    off = normalize_topics(mk(), False)
    assert [t["kind"] for t in off] == ["discussion"] * 3 and all(t["problem"] == "" for t in off)


def test_roundtrip_and_timestamps():
    from dataclasses import asdict
    iv = make()
    iv.add_transcript("ai", "Hi")
    iv.turns[0]["t_ms"] = 12_500
    assert iv.transcript_text(timestamps=True) == "[12s] Interviewer: Hi"
    d = asdict(iv)
    d["obsolete_field"] = 1
    assert Interview.from_dict(d) == iv


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all tests passed")
