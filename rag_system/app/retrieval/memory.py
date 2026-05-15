"""Conversation memory utilities."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List
from uuid import uuid4


class ConversationMemory:
    """Simple in-memory conversation history store."""

    def __init__(self, max_turns: int = 8) -> None:
        self.max_turns = max_turns
        self.messages: Dict[str, Deque[dict[str, str]]] = defaultdict(deque)

    def ensure_conversation(self, conversation_id: str | None) -> str:
        """Return a usable conversation id."""

        return conversation_id or uuid4().hex

    def add_turn(self, conversation_id: str, role: str, content: str) -> None:
        """Append a turn and trim history."""

        bucket = self.messages[conversation_id]
        bucket.append({"role": role, "content": content})
        while len(bucket) > self.max_turns * 2:
            bucket.popleft()

    def render_history(self, conversation_id: str) -> str:
        """Render compact conversation history."""

        history: List[str] = []
        for item in self.messages.get(conversation_id, []):
            history.append(f"{item['role']}: {item['content']}")
        return "\n".join(history)
