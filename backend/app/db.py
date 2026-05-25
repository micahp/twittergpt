"""Tiny SQLite store for jobs. MVP-grade; swap for Postgres in M4."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Optional

from .config import JOBS_DB, ensure_dirs

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,         -- 'train'
    status      TEXT NOT NULL,         -- queued|running|done|failed
    backend     TEXT NOT NULL,
    profile     TEXT,
    dataset_id  TEXT,
    adapter_dir TEXT,
    pid         INTEGER,
    log_path    TEXT,
    error       TEXT,
    meta        TEXT,                  -- JSON blob
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
"""


@contextmanager
def _conn():
    ensure_dirs()
    c = sqlite3.connect(JOBS_DB)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init() -> None:
    with _conn() as c:
        c.executescript(_SCHEMA)


def create_job(kind: str, backend: str, profile: str, dataset_id: str,
               meta: Optional[dict] = None) -> str:
    jid = uuid.uuid4().hex[:12]
    now = time.time()
    with _conn() as c:
        c.execute(
            "INSERT INTO jobs (id, kind, status, backend, profile, dataset_id, meta, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (jid, kind, "queued", backend, profile, dataset_id, json.dumps(meta or {}), now, now),
        )
    return jid


def update_job(jid: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = time.time()
    if "meta" in fields and isinstance(fields["meta"], dict):
        fields["meta"] = json.dumps(fields["meta"])
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), jid))


def get_job(jid: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 50) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
