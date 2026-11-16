"""Runs candidate code against hidden tests on a Judge0 server. Never run candidate code on our own server."""
import asyncio
import os
from urllib.parse import urlparse

import httpx

JUDGE0_URL = os.getenv("JUDGE0_URL", "").rstrip("/")  # e.g. https://judge0-ce.p.rapidapi.com or a self-hosted Judge0
JUDGE0_KEY = os.getenv("JUDGE0_KEY", "")
LANGUAGES = {"python": 71, "javascript": 63, "java": 62, "cpp": 54}  # Judge0 CE language ids
VISIBLE_TESTS = 2  # the first tests are the examples shown in the problem; the rest stay hidden


def enabled() -> bool:
    return bool(JUDGE0_URL)


def _headers() -> dict:
    if not JUDGE0_KEY:
        return {}
    if "rapidapi" in JUDGE0_URL:
        return {"X-RapidAPI-Key": JUDGE0_KEY, "X-RapidAPI-Host": urlparse(JUDGE0_URL).netloc}
    return {"X-Auth-Token": JUDGE0_KEY}


def same_output(actual: str, expected: str) -> bool:
    norm = lambda s: [line.rstrip() for line in (s or "").strip().splitlines()]
    return norm(actual) == norm(expected)


async def _run_one(client: httpx.AsyncClient, code: str, lang_id: int, test: dict) -> dict:
    r = await client.post(f"{JUDGE0_URL}/submissions", params={"base64_encoded": "false", "wait": "true"},
                          json={"source_code": code, "language_id": lang_id, "stdin": test["stdin"],
                                "cpu_time_limit": 2, "memory_limit": 128_000})
    r.raise_for_status()
    d = r.json()
    status = (d.get("status") or {}).get("description", "Unknown")
    ran = (d.get("status") or {}).get("id") == 3  # 3 = finished normally (we compare output ourselves)
    stdout = d.get("stdout") or ""
    return {"passed": ran and same_output(stdout, test["expected"]), "stdout": stdout[:2000],
            "error": None if ran else (d.get("compile_output") or d.get("stderr") or status)[:2000]}


async def run_tests(code: str, language: str, tests: list[dict], transport=None) -> dict:
    """Returns {passed, total, results}; hidden tests only reveal pass/fail and the kind of error."""
    if language not in LANGUAGES:
        raise ValueError("Unsupported language")
    async with httpx.AsyncClient(timeout=40, headers=_headers(), transport=transport) as client:
        raw = await asyncio.gather(*(_run_one(client, code, LANGUAGES[language], t) for t in tests))
    results = []
    for i, (t, r) in enumerate(zip(tests, raw)):
        if i < VISIBLE_TESTS:
            results.append({"visible": True, "stdin": t["stdin"], "expected": t["expected"], **r})
        else:
            results.append({"visible": False, "passed": r["passed"],
                            "error": r["error"].splitlines()[0][:120] if r["error"] else None})
    return {"passed": sum(r["passed"] for r in raw), "total": len(tests), "results": results}
