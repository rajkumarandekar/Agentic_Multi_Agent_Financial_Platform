"""
SQLite-backed chat history store.
Uses Python built-in sqlite3 — no new packages required.
"""

import json
import os
import sqlite3
import time
import uuid

_DB_PATH = os.path.join(os.path.dirname(__file__), "chats.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call multiple times."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chats (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL DEFAULT 'New Chat',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          TEXT PRIMARY KEY,
                chat_id     TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role        TEXT NOT NULL,
                text        TEXT NOT NULL,
                agent_used  TEXT DEFAULT '',
                agents_used TEXT DEFAULT '[]',
                sources     TEXT DEFAULT '[]',
                created_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_metadata (
                chat_id         TEXT PRIMARY KEY REFERENCES chats(id) ON DELETE CASCADE,
                source_document TEXT DEFAULT '',
                session_id      TEXT DEFAULT ''
            );
        """)


init_db()


def _row(row: sqlite3.Row) -> dict:
    return dict(row)


# ── Chat CRUD ──────────────────────────────────────────────────────────────


def create_chat(session_id: str = "", source_document: str = "") -> dict:
    """Create a new chat record. Returns the chat dict."""
    chat_id = str(uuid.uuid4())
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (chat_id, "New Chat", now, now),
        )
        conn.execute(
            "INSERT INTO chat_metadata (chat_id, source_document, session_id) VALUES (?, ?, ?)",
            (chat_id, source_document, session_id or chat_id),
        )
    return {"id": chat_id, "title": "New Chat", "created_at": now, "updated_at": now}


def list_chats() -> list[dict]:
    """Return all chats ordered by most recently updated."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM chats ORDER BY updated_at DESC"
        ).fetchall()
    return [_row(r) for r in rows]


def get_chat(chat_id: str) -> dict | None:
    """Return a chat with its messages and metadata. None if not found."""
    with _connect() as conn:
        chat_row = conn.execute(
            "SELECT * FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        if not chat_row:
            return None
        chat = _row(chat_row)

        msg_rows = conn.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
            (chat_id,),
        ).fetchall()
        messages = []
        for r in msg_rows:
            m = _row(r)
            m["agents_used"] = json.loads(m.get("agents_used") or "[]")
            m["sources"]     = json.loads(m.get("sources") or "[]")
            messages.append(m)

        meta_row = conn.execute(
            "SELECT * FROM chat_metadata WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        metadata = _row(meta_row) if meta_row else {}

    chat["messages"] = messages
    chat["metadata"] = metadata
    return chat


def rename_chat(chat_id: str, title: str) -> dict | None:
    """Rename a chat. Returns updated record or None if not found."""
    now = time.time()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
            (title.strip() or "New Chat", now, chat_id),
        )
        if cur.rowcount == 0:
            return None
        row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
    return _row(row)


def delete_chat(chat_id: str) -> bool:
    """Delete a chat and cascade to messages + metadata. Returns True if found."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    return cur.rowcount > 0


# ── Messages ───────────────────────────────────────────────────────────────


def save_message(
    chat_id: str,
    role: str,
    text: str,
    agent_used: str = "",
    agents_used: list | None = None,
    sources: list | None = None,
) -> dict:
    """Append a message and touch the parent chat's updated_at."""
    msg_id = str(uuid.uuid4())
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO messages
               (id, chat_id, role, text, agent_used, agents_used, sources, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg_id, chat_id, role, text,
                agent_used,
                json.dumps(agents_used or []),
                json.dumps(sources or []),
                now,
            ),
        )
        conn.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id)
        )
    return {
        "id": msg_id, "chat_id": chat_id, "role": role, "text": text,
        "agent_used": agent_used, "agents_used": agents_used or [],
        "sources": sources or [], "created_at": now,
    }


# ── Metadata ───────────────────────────────────────────────────────────────


def update_metadata(chat_id: str, source_document: str = "", session_id: str = "") -> None:
    """Upsert source_document / session_id for a chat."""
    with _connect() as conn:
        existing = conn.execute(
            "SELECT 1 FROM chat_metadata WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        if existing:
            parts, vals = [], []
            if source_document is not None:
                parts.append("source_document = ?")
                vals.append(source_document)
            if session_id:
                parts.append("session_id = ?")
                vals.append(session_id)
            if parts:
                conn.execute(
                    f"UPDATE chat_metadata SET {', '.join(parts)} WHERE chat_id = ?",
                    (*vals, chat_id),
                )
        else:
            conn.execute(
                "INSERT INTO chat_metadata (chat_id, source_document, session_id) VALUES (?, ?, ?)",
                (chat_id, source_document, session_id),
            )


def update_source_document(chat_id: str, source_document: str) -> None:
    """Persist the document selected for this chat in chat_metadata."""
    with _connect() as conn:
        conn.execute(
            "UPDATE chat_metadata SET source_document = ? WHERE chat_id = ?",
            (source_document, chat_id),
        )


def auto_title(chat_id: str, question: str) -> str:
    """Set chat title from the first user question. Returns the new title."""
    title = question.strip()[:60] or "New Chat"
    rename_chat(chat_id, title)
    return title
