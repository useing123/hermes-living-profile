from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus
from plugins.memory.living_profile.storage import ProfileStorage

logger = logging.getLogger(__name__)

# Heuristic patterns that signal facts worth updating to PROFILE.md
_FACT_TRIGGERS = [
    re.compile(r'\b(my name is|i am|i live in|i work at|i prefer|i want|our strategy|our target|our icp|our stack|we use|we are building|dont use|do not use|always use)\b', re.IGNORECASE),
    re.compile(r'\b(remember that|note that|keep in mind|rule:|decision:|convention:)\b', re.IGNORECASE),
]

_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "profile_view",
            "description": "View the user's living profile (stored in PROFILE.md) or a specific section of it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "heading": {
                        "type": "string",
                        "description": "Optional specific section heading (e.g. 'User & Strategy', 'Projects & Working Conventions'). If omitted, returns the whole living profile.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "profile_search",
            "description": "Search across all past conversation turns and living profile sections using fast FTS5 BM25 keyword matching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (keywords, names, stack choices, decisions).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "profile_update",
            "description": "Update or append a fact to a specific section in the living PROFILE.md.",
            "parameters": {
                "type": "object",
                "properties": {
                    "heading": {
                        "type": "string",
                        "description": "The section heading to update (e.g., 'GTM & Sales', 'Tech Stack', 'User Preferences').",
                    },
                    "fact": {
                        "type": "string",
                        "description": "The bullet point or fact text to store.",
                    },
                },
                "required": ["heading", "fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_control",
            "description": "Control memory recording and privacy for conversations. Actions: 'pause' (stop recording memory for this session), 'resume' (re-enable recording), or 'purge' (delete all memories and turns recorded from this session).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["pause", "resume", "purge"],
                        "description": "The action to perform: 'pause' recording, 'resume' recording, or 'purge' (delete) current session memories.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional specific session ID to purge. Defaults to current session.",
                    }
                },
                "required": ["action"],
            },
        },
    },
]


class LivingProfileMemoryProvider(MemoryProvider):
    """Living Profile Memory Provider:
    - Full turn archiving in SQLite
    - BM25 / FTS5 indexed search across all history
    - Self-compiling living Markdown profile in PROFILE.md
    - Pre-turn automatic prefetch
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self._config = dict(config or {})
        self._storage: Optional[ProfileStorage] = None
        self._session_id: str = ""
        self._parent_session_id: Optional[str] = None
        self._turn_count: int = 0
        self._last_recalled_count: int = 0
        self._last_prefetch_context: str = ""
        self._sync_thread: Optional[threading.Thread] = None
        self._paused_sessions: set[str] = set()

    @property
    def name(self) -> str:
        return "living_profile"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        home = kwargs.get("hermes_home")
        if not home:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
        data_dir = Path(home) / "memories"
        self._storage = ProfileStorage(data_dir)
        self._session_id = session_id
        self._parent_session_id = kwargs.get("parent_session_id")
        self._turn_count = 0
        self._last_recalled_count = 0
        logger.info("LivingProfileMemoryProvider initialized at %s (parent=%s)", data_dir, self._parent_session_id)

    def system_prompt_block(self) -> str:
        if not self._storage:
            return ""
        # Brief instruction letting the model know it has active living profile memory
        return (
            "# Living Profile Memory\n"
            "Active. All conversations are archived in SQLite and searchable via FTS5. "
            "Important context and user preferences are maintained in PROFILE.md. "
            "Relevant memories from past sessions are automatically injected when related to the current turn. "
            "Use `profile_view`, `profile_search`, and `profile_update` to inspect or record durable facts."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Run FTS5 search across profile sections and past conversation turns before turn starts."""
        if not self._storage or not query or len(query.strip()) < 3:
            self._last_recalled_count = 0
            return ""

        profile_matches = self._storage.search_profile(query, limit=2)
        turn_matches = self._storage.search_conversations(query, limit=3)

        # Filter out turns from the current active turn to avoid echoing current message
        sid = session_id or self._session_id
        filtered_turns = [
            t for t in turn_matches
            if not (t["session_id"] == sid and t["content"].strip() == query.strip())
        ]

        total_recalled = len(profile_matches) + len(filtered_turns)
        self._last_recalled_count = total_recalled
        if total_recalled == 0:
            return ""

        context_blocks = []

        if profile_matches:
            prof_lines = ["### Known Profile & Facts:"]
            for m in profile_matches:
                prof_lines.append(f"**[{m['heading']}]**\n{m['content']}")
            context_blocks.append("\n".join(prof_lines))

        if filtered_turns:
            hist_lines = ["### Related Past Conversation Excerpts:"]
            for t in filtered_turns:
                date_str = t['timestamp'][:10] if t.get('timestamp') else "past"
                hist_lines.append(f"- [{date_str} - {t['role']}]: {t['content'][:250]}")
            context_blocks.append("\n".join(hist_lines))

        result = "## Recalled Memory & Profile Context\n" + "\n\n".join(context_blocks)
        self._last_prefetch_context = result
        return result

    def recall_status(self) -> Optional[RecallStatus]:
        if self._last_recalled_count > 0:
            return RecallStatus(
                count=self._last_recalled_count,
                provider_label="LivingProfile",
                glyph="🧠",
            )
        return None

    def sync_turn(
        self, user_content: str, assistant_content: str, *,
        session_id: str = "", messages: Optional[list[dict[str, Any]]] = None,
        turn_author: Optional[dict[str, Any]] = None,
    ) -> None:
        """Persist turn immediately to SQLite and check if facts need auto-extracting."""
        if not self._storage:
            return

        self._turn_count += 1
        sid = session_id or self._session_id

        # Skip recording if session is paused (incognito / do not save)
        if sid in self._paused_sessions:
            logger.info("Memory recording paused for session %s, skipping sync_turn", sid)
            return

        # 1. Store turn immediately into SQLite and FTS5 (synchronous, sub-millisecond)
        if user_content:
            self._storage.store_turn(sid, self._turn_count, "user", user_content, parent_session_id=self._parent_session_id)
        if assistant_content:
            self._storage.store_turn(sid, self._turn_count, "assistant", assistant_content, parent_session_id=self._parent_session_id)

        # 2. Heuristic extraction of durable facts in background thread
        if user_content and any(trigger.search(user_content) for trigger in _FACT_TRIGGERS):
            def _extract():
                try:
                    self._extract_and_update(user_content)
                except Exception as e:
                    logger.debug("Auto-extract failed: %s", e)

            t = threading.Thread(target=_extract, name="living-profile-extract", daemon=True)
            t.start()

    def _extract_and_update(self, text: str) -> None:
        """Heuristic extractor to capture obvious conventions/preferences into PROFILE.md."""
        if not self._storage:
            return
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            lower = line.lower()
            if any(k in lower for k in ["icp", "target", "gtm", "prospect", "sales", "outreach"]):
                self._storage.append_fact_to_section("GTM & Strategy", line)
            elif any(k in lower for k in ["prefer", "stack", "tool", "framework", "sqlite", "python", "fastapi"]):
                self._storage.append_fact_to_section("Tech Stack & Preferences", line)
            elif any(k in lower for k in ["rule", "never", "always", "convention", "must"]):
                self._storage.append_fact_to_section("Working Rules & Constraints", line)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return _TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if not self._storage:
            return json.dumps({"error": "Living profile storage not initialized"})

        if tool_name == "profile_view":
            heading = args.get("heading")
            if heading:
                sections = {s["heading"].lower(): s["content"] for s in self._storage.get_all_profile_sections()}
                content = sections.get(heading.lower())
                if content:
                    return json.dumps({"heading": heading, "content": content})
                return json.dumps({"error": f"Section '{heading}' not found in living profile"})
            else:
                all_sections = self._storage.get_all_profile_sections()
                profile_md = self._storage.profile_path.read_text(encoding="utf-8") if self._storage.profile_path.exists() else ""
                return json.dumps({
                    "profile_markdown": profile_md,
                    "sections_count": len(all_sections),
                    "file_path": str(self._storage.profile_path),
                })

        elif tool_name == "profile_search":
            query = args.get("query", "")
            limit = int(args.get("limit", 5))
            profile_res = self._storage.search_profile(query, limit=limit)
            turn_res = self._storage.search_conversations(query, limit=limit)
            return json.dumps({
                "profile_matches": profile_res,
                "conversation_matches": turn_res,
                "total_found": len(profile_res) + len(turn_res),
            })

        elif tool_name == "profile_update":
            heading = args.get("heading", "General")
            fact = args.get("fact", "")
            if not fact:
                return json.dumps({"error": "fact argument is required"})
            updated = self._storage.append_fact_to_section(heading, fact)
            return json.dumps({
                "success": True,
                "updated": updated,
                "heading": heading,
                "fact": fact,
                "file_path": str(self._storage.profile_path),
            })

        elif tool_name == "memory_control":
            action = args.get("action", "")
            target_sid = args.get("session_id") or self._session_id
            
            if action == "pause":
                self._paused_sessions.add(target_sid)
                return json.dumps({
                    "success": True,
                    "action": "pause",
                    "session_id": target_sid,
                    "message": "Memory recording paused for this session. New turns will not be saved."
                })
            elif action == "resume":
                self._paused_sessions.discard(target_sid)
                return json.dumps({
                    "success": True,
                    "action": "resume",
                    "session_id": target_sid,
                    "message": "Memory recording resumed for this session."
                })
            elif action == "purge":
                deleted_count = self._storage.delete_session_memories(target_sid)
                self._paused_sessions.add(target_sid)  # also pause after purge so subsequent turns don't re-pollute
                return json.dumps({
                    "success": True,
                    "action": "purge",
                    "session_id": target_sid,
                    "deleted_turns": deleted_count,
                    "message": f"Purged {deleted_count} turns from memory for session {target_sid}. Session memory recording is now paused."
                })
            return json.dumps({"error": f"Unknown action: {action}"})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def on_session_switch(
        self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, rewound: bool = False, **kwargs,
    ) -> None:
        """When session forks or switches mid-process, update session and parent lineage."""
        self._session_id = new_session_id
        if parent_session_id:
            self._parent_session_id = parent_session_id
        self._turn_count = 0
        logger.info("LivingProfile switched to session %s (parent=%s)", new_session_id, self._parent_session_id)

    def shutdown(self) -> None:
        pass


def register_memory_provider(ctx):
    ctx.register_memory_provider(LivingProfileMemoryProvider())
