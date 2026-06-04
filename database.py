import os
import sqlite3
from typing import Optional

DB_PATH = os.getenv("DB_PATH", "sessions.db")


def _connect():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_cost REAL,
                first_message TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()


def save_session(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_cost: Optional[float],
    first_message: str,
    timestamp: str,
):
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions
                (model, prompt_tokens, completion_tokens, total_cost, first_message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (model, prompt_tokens, completion_tokens, total_cost, first_message, timestamp),
        )
        conn.commit()


def get_sessions(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
