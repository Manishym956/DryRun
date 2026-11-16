"""Streaks and badges, computed from a user's interviews on the fly (nothing extra is stored)."""
from datetime import date, datetime, timedelta, timezone

from interview import Interview


def _local_day(ts: float, tz_offset_min: int) -> date:
    # tz_offset_min follows JavaScript's getTimezoneOffset(): minutes to ADD to local time to get UTC.
    return (datetime.fromtimestamp(ts, timezone.utc) - timedelta(minutes=tz_offset_min)).date()


def streaks(days: set[date], today: date) -> tuple[int, int]:
    """(current streak, best streak) in consecutive days. Today without practice doesn't break it yet."""
    best = run = 0
    prev = None
    for d in sorted(days):
        run = run + 1 if prev and d - prev == timedelta(days=1) else 1
        best, prev = max(best, run), d
    start = today if today in days else today - timedelta(days=1)
    current = 0
    while start - timedelta(days=current) in days:
        current += 1
    return current, best


def _scores(iv: Interview) -> list:
    return [s.get("score") for s in (iv.evaluation or {}).get("scores", [])]


BADGES = [
    # id, name, how to earn it, test over (ended interviews, best streak)
    ("first", "First interview", "Finish your first interview", lambda done, best: len(done) >= 1),
    ("ten", "Ten down", "Finish 10 interviews", lambda done, best: len(done) >= 10),
    ("streak3", "3-day streak", "Practice 3 days in a row", lambda done, best: best >= 3),
    ("streak7", "7-day streak", "Practice 7 days in a row", lambda done, best: best >= 7),
    ("green", "All green", "Pass every hidden test in a coding round",
     lambda done, best: any(r["total"] and r["passed"] == r["total"] for iv in done for r in iv.test_results.values())),
    ("allrounder", "All-rounder", "Score 8+ on every dimension in one interview",
     lambda done, best: any(len(s) >= 3 and all(x is not None and x >= 8 for x in s) for s in map(_scores, done))),
    ("survivor", "Survived the bar-raiser", "Get a 'Yes' in the bar-raiser challenge",
     lambda done, best: any(iv.challenge and (iv.evaluation or {}).get("hiringRecommendation") == "Yes" for iv in done)),
    ("peer", "Peer practice", "Finish a peer mock interview", lambda done, best: any(iv.mode == "peer" for iv in done)),
]


def progress(interviews: list[Interview], tz_offset_min: int = 0, now: float | None = None) -> dict:
    done = [iv for iv in interviews if iv.ended]
    today = _local_day(now if now is not None else datetime.now(timezone.utc).timestamp(), tz_offset_min)
    current, best = streaks({_local_day(iv.created_at, tz_offset_min) for iv in done}, today)
    return {"streak": current, "best_streak": best, "interviews": len(done),
            "badges": [{"id": i, "name": n, "how": h, "earned": bool(test(done, best))} for i, n, h, test in BADGES]}
