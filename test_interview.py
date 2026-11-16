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
    assert iv.read_code() == {"code": "def rev(x): return x[::-1]", "language": "python"}
    assert iv.code == {"1": "def rev(x): return x[::-1]"}
    assert "Code written for 'T1' in python (tests never run)" in iv.code_for_eval()


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


def test_normalize_design_and_tests():
    from interview import normalize_topics
    ts = [{"kind": "design", "problem": "URL shortener", "tests": [{"stdin": "1", "expected": "1"}]},
          {"kind": "coding", "problem": "p", "tests": [{"stdin": "1", "expected": "1"}]},
          {"kind": "design", "problem": "x"}]
    out = normalize_topics(ts, coding_round=True, design_round=True)
    assert [t["kind"] for t in out] == ["design", "coding", "discussion"]
    assert out[0]["tests"] == [] and len(out[1]["tests"]) == 1 and out[2]["tests"] == []


def test_pacer():
    from interview import RAMBLE_NUDGE, SILENCE_NUDGE, Pacer
    p = Pacer()
    p.ai_done(0)
    assert p.check(20, silence_s=30, ramble_s=60) is None
    assert p.check(31, 30, 60) == SILENCE_NUDGE
    assert p.check(100, 30, 60) is None  # only once per gap
    p.ai_spoke(); p.ai_done(200); p.user_spoke(205)  # candidate answered: no silence nudge
    assert p.check(240, 30, 60) is None
    p.user_spoke(250)  # still the same answer
    assert p.check(266, 30, 60) == RAMBLE_NUDGE
    for t in (300, 400):  # candidate walked away: two silence nudges, then quiet
        p.ai_spoke(); p.ai_done(t)
        assert p.check(t + 31, 30, 60) == SILENCE_NUDGE
    p.ai_spoke(); p.ai_done(500)
    assert p.check(600, 30, 60) is None


def test_speech_stats():
    from interview import speech_stats
    turns = [{"speaker": "ai", "text": "Q?", "t_ms": 0},
             {"speaker": "user", "text": "Um so I mean, like, we umm used basically a cache", "t_ms": 1000},
             {"speaker": "ai", "text": "OK", "t_ms": 31000},
             {"speaker": "user", "text": "I'd like to add more", "t_ms": 32000}]
    st = speech_stats(turns)
    assert st["words"] == 16 and st["wpm"] == 22  # 11 words over 30 s
    assert st["fillers"] == {"um": 2, "i mean": 1, "like": 1, "basically": 1}, st  # "I'd like" is not a filler
    assert speech_stats([{"speaker": "ai", "text": "hi", "t_ms": 0}]) is None


def test_code_tests_and_diagram():
    iv = make()
    iv.topics[0].update(kind="coding", problem="p")
    iv.topics[1].update(kind="design", problem="d")
    iv.next_topic()
    iv.set_code("print(1)", "python")
    iv.set_tests(2, 5)
    iv.set_diagram("ignored", "")  # not a design topic
    assert iv.read_code()["tests"] == "2/5 hidden tests passed on the last run"
    iv.next_topic()
    assert iv.read_diagram()["diagram"] == ""
    iv.set_diagram("LB -> API", "<svg/>")
    assert iv.read_diagram() == {"diagram": "LB -> API"}
    ev = iv.code_for_eval()
    assert "2/5 passed" in ev and "Whiteboard for 'T1'" in ev


def test_progress_streaks_and_badges():
    from progress import progress
    day = 86400
    now = 10 * day + 3600  # day 10, 01:00 UTC
    ivs = []
    for d in (5, 8, 9, 10):
        iv = make(); iv.id = str(d); iv.ended = True; iv.created_at = d * day + 600
        ivs.append(iv)
    ivs[0].challenge = True; ivs[0].evaluation = {"hiringRecommendation": "Yes", "scores": []}
    ivs[1].test_results = {"0": {"passed": 3, "total": 3}}
    p = progress(ivs, 0, now)
    assert (p["streak"], p["best_streak"], p["interviews"]) == (3, 3, 4)
    earned = {b["id"] for b in p["badges"] if b["earned"]}
    assert earned == {"first", "streak3", "green", "survivor"}, earned
    # In UTC-2 (offset +120) every interview and "now" move back a day: days 4, 7, 8, 9, today 9
    assert progress(ivs, 120, now)["streak"] == 3
    assert progress(ivs, 0, 11 * day + 100)["streak"] == 3  # nothing today yet: yesterday keeps the streak
    assert progress(ivs, 0, 12 * day + 100)["streak"] == 0


def test_judge_grading_hides_hidden_tests():
    import asyncio, json
    import httpx
    import judge
    judge.JUDGE0_URL = "https://judge.test"
    def handler(req):
        body = json.loads(req.content)
        out = {"1": "2\n", "2": "4  \n", "3": "7"}.get(body["stdin"], "")
        status = {"id": 5, "description": "Time Limit Exceeded"} if body["stdin"] == "4" else {"id": 3, "description": "Accepted"}
        return httpx.Response(200, json={"stdout": out, "status": status})
    tests = [{"stdin": s, "expected": e} for s, e in [("1", "2"), ("2", "4"), ("3", "6"), ("4", "8")]]
    r = asyncio.run(judge.run_tests("code", "python", tests, transport=httpx.MockTransport(handler)))
    assert (r["passed"], r["total"]) == (2, 4)
    assert r["results"][1]["visible"] and r["results"][1]["passed"]  # trailing spaces ignored
    hidden = r["results"][2]
    assert hidden == {"visible": False, "passed": False, "error": None}  # wrong answer: no input or output leaked
    assert r["results"][3]["error"] == "Time Limit Exceeded"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all tests passed")
