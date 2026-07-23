from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any


class ChatStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_conv ON memories(conversation_id, id)")

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        text = (content or "").strip()
        if not text:
            return
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO messages(conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, role, text, time.time()),
            )

    def load_messages(self, conversation_id: str, limit: int = 80) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [
            {"role": row["role"], "content": row["content"], "created_at": row["created_at"]}
            for row in reversed(rows)
        ]

    def clear_conversation(self, conversation_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM memories WHERE conversation_id = ?", (conversation_id,))

    def remember_from_user_text(self, conversation_id: str, content: str) -> None:
        text = (content or "").strip()
        if not text or len(text) > 120:
            return
        if "?" in text or "？" in text or any(token in text for token in ["谁", "什么人", "什么意思", "啥意思", "是不是"]):
            return
        prefixes = ("我叫", "我是", "我姓", "我的名字", "我的英文名", "我英文名", "我喜欢", "我讨厌", "我不喜欢", "以后记住", "记住")
        identity_patterns = [
            r"我姓[\u4e00-\u9fff]",
            r"我叫[\u4e00-\u9fffA-Za-z0-9_\-]{1,16}",
            r"我的名字是[\u4e00-\u9fffA-Za-z0-9_\-]{1,16}",
            r"我(?:的)?英文名(?:字)?(?:是|叫)[A-Za-z][A-Za-z0-9_\-]{1,20}",
        ]
        if not text.startswith(prefixes) and not any(re.search(pattern, text, flags=re.I) for pattern in identity_patterns):
            return
        existing = {row["content"] for row in self.load_memories(conversation_id, limit=100)}
        if text in existing:
            return
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO memories(conversation_id, kind, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, "user_fact", text, time.time()),
            )

    def remember_active_topic(self, conversation_id: str, topic: str) -> None:
        text = (topic or "").strip()
        if not text:
            return
        payload = json.dumps({"topic": text[:80], "source": "proactive"}, ensure_ascii=False)
        with self.connect() as conn:
            conn.execute("DELETE FROM memories WHERE conversation_id = ? AND kind = ?", (conversation_id, "active_topic"))
            conn.execute(
                "INSERT INTO memories(conversation_id, kind, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, "active_topic", payload, time.time()),
            )

    def load_memories(self, conversation_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT kind, content, created_at
                FROM memories
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [
            {"kind": row["kind"], "content": row["content"], "created_at": row["created_at"]}
            for row in rows
        ]
