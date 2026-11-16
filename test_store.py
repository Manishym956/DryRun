"""Run: python test_store.py  (SQLite in a temp file; set DATABASE_URL to run the same checks on Postgres)"""
import tempfile
import time
import uuid
from pathlib import Path

import store
from interview import Interview

if not store.URL:
    store.LOCAL_DB = Path(tempfile.mkdtemp()) / "test.db"
store.init()
store.init()  # idempotent


def make(uid, created_at=None):
    return Interview(id=uuid.uuid4().hex, name="A", role="SDE", persona="friendly", candidate_summary="",
                     topics=[{"topic": "T", "focus": "f", "difficulty": "Beginner"}], user_id=uid,
                     created_at=created_at or time.time())


def test_roundtrip_ownership_and_counts():
    me, other = "u-" + uuid.uuid4().hex, "u-" + uuid.uuid4().hex
    old = make(me, time.time() - 2 * 86400)
    a, b = make(me), make(other)
    for iv in (old, a, b):
        store.put(iv)
    a.ended = True
    a.evaluation = {"scores": []}
    store.put(a)  # upsert, not duplicate
    assert store.get(a.id) == a
    assert store.get("missing") is None
    assert [iv.id for iv in store.list_for(me)] == [a.id, old.id]  # newest first, only mine
    assert store.count_since(time.time() - 86400, me) == 1
    assert store.count_since(time.time() - 86400) >= 2


def test_recordings_and_delete():
    iv = make("u-" + uuid.uuid4().hex)
    store.put(iv)
    assert store.recording_mime(iv.id) is None
    blob = bytes(range(256)) * 10
    store.put_recording(iv.id, "audio/webm", blob)
    assert store.get_recording(iv.id) == ("audio/webm", blob)
    try:
        store.put_recording(iv.id, "audio/webm", b"x")
        raise AssertionError("second recording should be rejected")
    except AssertionError:
        raise
    except Exception:
        pass
    store.delete(iv.id)
    assert store.get(iv.id) is None and store.get_recording(iv.id) is None


def test_users():
    uid = "google:" + uuid.uuid4().hex
    store.upsert_user(uid, "a@x.com", "A", 1.0)
    store.upsert_user(uid, "b@x.com", "B", 2.0)
    assert store.get_user(uid) == {"id": uid, "email": "b@x.com", "name": "B"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"all store tests passed ({'postgres' if store.URL else 'sqlite'})")
