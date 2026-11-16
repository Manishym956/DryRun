"""Persistence: Postgres when DATABASE_URL is set (Render), else a local SQLite file. Same SQL for both."""
import json
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path

from interview import Interview

URL = os.getenv("DATABASE_URL", "")
LOCAL_DB = Path(__file__).parent / "sessions" / "interviews.db"

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, email TEXT NOT NULL, name TEXT NOT NULL, "
    "created_at DOUBLE PRECISION NOT NULL)",
    "CREATE TABLE IF NOT EXISTS interviews (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
    "created_at DOUBLE PRECISION NOT NULL, data TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS interviews_by_user ON interviews (user_id, created_at)",
    # shortcut: recordings live in the database (~7 MB per 30 min); move to object storage if the DB outgrows its plan.
    "CREATE TABLE IF NOT EXISTS recordings (id TEXT PRIMARY KEY, mime TEXT NOT NULL, data BYTEA NOT NULL)",
    "CREATE TABLE IF NOT EXISTS shares (token TEXT PRIMARY KEY, interview_id TEXT NOT NULL)",
]


def _run(sql: str, args: tuple = (), fetch: bool = False) -> list:
    # shortcut: a new connection per query; add psycopg_pool if traffic makes connects a bottleneck.
    if URL:
        import psycopg
        with psycopg.connect(URL, autocommit=True) as conn:
            cur = conn.execute(sql, args)
            return cur.fetchall() if fetch else []
    conn = sqlite3.connect(LOCAL_DB)
    try:
        cur = conn.execute(sql.replace("%s", "?"), args)
        rows = cur.fetchall() if fetch else []
        conn.commit()
        return rows
    finally:
        conn.close()


def init() -> None:
    if not URL:
        LOCAL_DB.parent.mkdir(exist_ok=True)
    for stmt in SCHEMA:
        _run(stmt)


def upsert_user(uid: str, email: str, name: str, now: float) -> None:
    _run("INSERT INTO users (id, email, name, created_at) VALUES (%s, %s, %s, %s) "
         "ON CONFLICT (id) DO UPDATE SET email = excluded.email, name = excluded.name", (uid, email, name, now))


def get_user(uid: str) -> dict | None:
    rows = _run("SELECT id, email, name FROM users WHERE id = %s", (uid,), fetch=True)
    return dict(zip(("id", "email", "name"), rows[0])) if rows else None


def get(sid: str) -> Interview | None:
    rows = _run("SELECT data FROM interviews WHERE id = %s", (sid,), fetch=True)
    return Interview.from_dict(json.loads(rows[0][0])) if rows else None


def put(iv: Interview) -> None:
    _run("INSERT INTO interviews (id, user_id, created_at, data) VALUES (%s, %s, %s, %s) "
         "ON CONFLICT (id) DO UPDATE SET data = excluded.data", (iv.id, iv.user_id, iv.created_at, json.dumps(asdict(iv))))


def list_for(uid: str) -> list[Interview]:
    rows = _run("SELECT data FROM interviews WHERE user_id = %s ORDER BY created_at DESC", (uid,), fetch=True)
    return [Interview.from_dict(json.loads(r[0])) for r in rows]


def count_since(ts: float, uid: str | None = None) -> int:
    if uid is None:
        return _run("SELECT COUNT(*) FROM interviews WHERE created_at > %s", (ts,), fetch=True)[0][0]
    return _run("SELECT COUNT(*) FROM interviews WHERE created_at > %s AND user_id = %s", (ts, uid), fetch=True)[0][0]


def delete(sid: str) -> None:
    _run("DELETE FROM shares WHERE interview_id = %s", (sid,))
    _run("DELETE FROM recordings WHERE id = %s", (sid,))
    _run("DELETE FROM interviews WHERE id = %s", (sid,))


def share(token: str, sid: str) -> None:
    _run("INSERT INTO shares (token, interview_id) VALUES (%s, %s)", (token, sid))


def unshare(sid: str) -> None:
    _run("DELETE FROM shares WHERE interview_id = %s", (sid,))


def shared(token: str) -> Interview | None:
    rows = _run("SELECT interview_id FROM shares WHERE token = %s", (token,), fetch=True)
    return get(rows[0][0]) if rows else None


def put_recording(sid: str, mime: str, data: bytes) -> None:
    _run("INSERT INTO recordings (id, mime, data) VALUES (%s, %s, %s)", (sid, mime, data))


def recording_mime(sid: str) -> str | None:
    rows = _run("SELECT mime FROM recordings WHERE id = %s", (sid,), fetch=True)
    return rows[0][0] if rows else None


def get_recording(sid: str) -> tuple[str, bytes] | None:
    rows = _run("SELECT mime, data FROM recordings WHERE id = %s", (sid,), fetch=True)
    return (rows[0][0], bytes(rows[0][1])) if rows else None
