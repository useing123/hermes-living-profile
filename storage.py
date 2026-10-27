import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# FTS5 stop words or special punctuation cleaning
_PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)


def _clean_fts_query(query: str) -> str:
    """Prepare a user query for safe SQLite FTS5 matching."""
    cleaned = _PUNCT_RE.sub(" ", query).strip()
    words = [w for w in cleaned.split() if len(w) > 1]
    if not words:
        return ""
    # Join with OR or NEAR for broad but ranked retrieval
    # e.g., 'fastapi OR sqlite OR memory'
    return " OR ".join(words[:15])


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class ProfileStorage:
    """Manages the SQLite database and PROFILE.md for living_profile."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "profile_memory.db"
        self.profile_path = self.data_dir / "PROFILE.md"
        self._lock = threading.Lock()
        self._init_db()
        self._sync_profile_md_to_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            # 1. Full conversation turns ledger with turn_hash and parent_session_id for fork deduplication
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    turn_hash TEXT,
                    parent_session_id TEXT
                )
            """)
            # Migration check: if columns missing from existing table, alter table
            try:
                conn.execute("ALTER TABLE conversation_turns ADD COLUMN turn_hash TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversation_turns ADD COLUMN parent_session_id TEXT")
            except sqlite3.OperationalError:
                pass

            conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON conversation_turns(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_hash ON conversation_turns(turn_hash)")

            # 2. FTS5 index over conversation turns
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_conversation_turns USING fts5(
                    session_id UNINDEXED,
                    role UNINDEXED,
                    content,
                    tokenize='unicode61'
                )
            """)

            # 3. Living profile sections
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profile_sections (
                    heading TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 4. FTS5 index over profile sections
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_profile USING fts5(
                    heading,
                    content,
                    tokenize='unicode61'
                )
            """)
            conn.commit()

    def store_turn(
        self, session_id: str, turn_index: int, role: str, content: str, parent_session_id: Optional[str] = None
    ) -> bool:
        """Append turn to conversation ledger and index in FTS5 with fork deduplication.
        Returns True if stored, False if skipped as duplicate turn from parent fork."""
        if not content or not content.strip():
            return False
        
        t_hash = _sha256(f"{role}:{content.strip()}")
        now = datetime.now(timezone.utc).isoformat()
        
        with self._lock, self._get_connection() as conn:
            # Fork deduplication: if exact turn already exists in this session or its parent, skip
            if parent_session_id:
                cur = conn.execute(
                    "SELECT 1 FROM conversation_turns WHERE turn_hash=? AND session_id IN (?, ?) LIMIT 1",
                    (t_hash, session_id, parent_session_id),
                )
            else:
                cur = conn.execute(
                    "SELECT 1 FROM conversation_turns WHERE turn_hash=? AND session_id=? LIMIT 1",
                    (t_hash, session_id),
                )
            if cur.fetchone():
                return False  # Duplicate turn from forked ancestor, skip!

            cur = conn.execute(
                """
                INSERT INTO conversation_turns (session_id, turn_index, role, content, timestamp, turn_hash, parent_session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, turn_index, role, content, now, t_hash, parent_session_id),
            )
            row_id = cur.lastrowid
            conn.execute(
                """
                INSERT INTO fts_conversation_turns (rowid, session_id, role, content)
                VALUES (?, ?, ?, ?)
                """,
                (row_id, session_id, role, content),
            )
            conn.commit()
            return True

    def parse_profile_md(self, text: str) -> Dict[str, str]:
        """Split a Markdown document into heading -> body sections."""
        sections: Dict[str, str] = {}
        current_heading = "General"
        current_lines: List[str] = []

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                if current_lines:
                    sections[current_heading] = "\n".join(current_lines).strip()
                    current_lines = []
                current_heading = stripped[3:].strip()
            else:
                current_lines.append(line)

        if current_lines:
            sections[current_heading] = "\n".join(current_lines).strip()

        return sections

    def _sync_profile_md_to_db(self) -> None:
        """If PROFILE.md exists on disk and was manually edited, update DB sections."""
        if not self.profile_path.exists():
            default_content = (
                "# Living Agent Profile\n\n"
                "## User & Strategy\n"
                "- Core goals and preferences will be recorded here.\n\n"
                "## Projects & Working Conventions\n"
                "- Active workstreams, preferred tools, and conventions.\n"
            )
            self.profile_path.write_text(default_content, encoding="utf-8")

        try:
            md_text = self.profile_path.read_text(encoding="utf-8")
            parsed = self.parse_profile_md(md_text)
            now = datetime.now(timezone.utc).isoformat()

            with self._lock, self._get_connection() as conn:
                for heading, body in parsed.items():
                    h_hash = _sha256(body)
                    cur = conn.execute(
                        "SELECT content_hash FROM profile_sections WHERE heading=?",
                        (heading,),
                    )
                    row = cur.fetchone()
                    if not row or row["content_hash"] != h_hash:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO profile_sections (heading, content, content_hash, updated_at)
                            VALUES (?, ?, ?, ?)
                            """,
                            (heading, body, h_hash, now),
                        )
                        # Refresh FTS
                        conn.execute("DELETE FROM fts_profile WHERE heading=?", (heading,))
                        conn.execute(
                            "INSERT INTO fts_profile (heading, content) VALUES (?, ?)",
                            (heading, body),
                        )
                conn.commit()
        except Exception as e:
            logger.warning("Failed to sync PROFILE.md to db: %s", e)

    def write_profile_md_from_db(self) -> str:
        """Reconstruct PROFILE.md from SQLite profile_sections table."""
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT heading, content FROM profile_sections ORDER BY heading")
            rows = cur.fetchall()

        lines = ["# Living Agent Profile\n"]
        for r in rows:
            lines.append(f"## {r['heading']}")
            lines.append(f"{r['content']}\n")

        full_md = "\n".join(lines)
        self.profile_path.write_text(full_md, encoding="utf-8")
        return full_md

    def update_section(self, heading: str, content: str) -> bool:
        """Update a section in profile_sections and sync PROFILE.md ($O(1)$ diff)."""
        new_hash = _sha256(content)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT content_hash FROM profile_sections WHERE heading=?", (heading,))
            row = cur.fetchone()
            if row and row["content_hash"] == new_hash:
                return False  # Unchanged! Zero work needed.

            conn.execute(
                """
                INSERT OR REPLACE INTO profile_sections (heading, content, content_hash, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (heading, content, new_hash, now),
            )
            conn.execute("DELETE FROM fts_profile WHERE heading=?", (heading,))
            conn.execute(
                "INSERT INTO fts_profile (heading, content) VALUES (?, ?)",
                (heading, content),
            )
            conn.commit()

        self.write_profile_md_from_db()
        return True

    def append_fact_to_section(self, heading: str, fact_bullet: str) -> bool:
        """Append a single bullet point fact to an existing or new section."""
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT content FROM profile_sections WHERE heading=?", (heading,))
            row = cur.fetchone()
            existing_content = row["content"] if row else ""

        # Dedup: don't add duplicate bullet
        fact_bullet = fact_bullet.strip()
        if not fact_bullet.startswith("- ") and not fact_bullet.startswith("* "):
            fact_bullet = f"- {fact_bullet}"

        if fact_bullet.lower() in existing_content.lower():
            return False

        new_content = f"{existing_content}\n{fact_bullet}".strip() if existing_content else fact_bullet
        return self.update_section(heading, new_content)

    def search_profile(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search structured profile sections via FTS5 with fallback."""
        fts_q = _clean_fts_query(query)
        if fts_q:
            try:
                with self._lock, self._get_connection() as conn:
                    cur = conn.execute(
                        """
                        SELECT heading, content, bm25(fts_profile) as score
                        FROM fts_profile
                        WHERE fts_profile MATCH ?
                        ORDER BY score
                        LIMIT ?
                        """,
                        (fts_q, limit),
                    )
                    rows = cur.fetchall()
                    if rows:
                        return [{"heading": r["heading"], "content": r["content"], "score": r["score"]} for r in rows]
            except sqlite3.OperationalError:
                pass

        # Substring / keyword token fallback if FTS didn't hit (e.g. Russian/multilingual queries)
        tokens = [t.lower() for t in _PUNCT_RE.sub(" ", query).split() if len(t) > 2]
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT heading, content FROM profile_sections")
            matches = []
            for r in cur.fetchall():
                h, c = r["heading"], r["content"]
                text = f"{h} {c}".lower()
                matched_tokens = sum(1 for t in tokens if t in text)
                if matched_tokens > 0:
                    matches.append({"heading": h, "content": c, "score": -matched_tokens})
            if matches:
                matches.sort(key=lambda x: x["score"])
                return matches[:limit]
            
            # If still no match but we have sections, return main GTM/Strategy sections as fallback
            cur = conn.execute("SELECT heading, content FROM profile_sections WHERE heading != 'General' ORDER BY CASE WHEN heading LIKE '%GTM%' OR heading LIKE '%Strategy%' THEN 0 ELSE 1 END LIMIT ?", (limit,))
            return [{"heading": r["heading"], "content": r["content"], "score": 10.0} for r in cur.fetchall()]

    def search_conversations(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search historical conversation turns via FTS5."""
        fts_q = _clean_fts_query(query)
        if not fts_q:
            return []
        try:
            with self._lock, self._get_connection() as conn:
                cur = conn.execute(
                    """
                    SELECT t.session_id, t.role, t.content, t.timestamp, bm25(fts_conversation_turns) as score
                    FROM fts_conversation_turns f
                    JOIN conversation_turns t ON f.rowid = t.id
                    WHERE fts_conversation_turns MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (fts_q, limit),
                )
                return [
                    {
                        "session_id": r["session_id"],
                        "role": r["role"],
                        "content": r["content"],
                        "timestamp": r["timestamp"],
                        "score": r["score"],
                    }
                    for r in cur.fetchall()
                ]
        except sqlite3.OperationalError:
            with self._lock, self._get_connection() as conn:
                cur = conn.execute(
                    """
                    SELECT session_id, role, content, timestamp
                    FROM conversation_turns
                    WHERE content LIKE ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (f"%{query}%", limit),
                )
                return [
                    {
                        "session_id": r["session_id"],
                        "role": r["role"],
                        "content": r["content"],
                        "timestamp": r["timestamp"],
                        "score": 1.0,
                    }
                    for r in cur.fetchall()
                ]

    def delete_session_memories(self, session_id: str) -> int:
        """Purge all conversation turns and FTS index entries for a specific session."""
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT id FROM conversation_turns WHERE session_id=?", (session_id,))
            row_ids = [r["id"] for r in cur.fetchall()]
            if not row_ids:
                return 0
            
            # Delete from FTS5
            for rid in row_ids:
                conn.execute("DELETE FROM fts_conversation_turns WHERE rowid=?", (rid,))
            
            # Delete from turns ledger
            cur_del = conn.execute("DELETE FROM conversation_turns WHERE session_id=?", (session_id,))
            conn.commit()
            return cur_del.rowcount

    def get_all_profile_sections(self) -> List[Dict[str, str]]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT heading, content, updated_at FROM profile_sections ORDER BY heading")
            return [{"heading": r["heading"], "content": r["content"], "updated_at": r["updated_at"]} for r in cur.fetchall()]
