"""
Conversation Store — In-memory conversation state management with optional Redis.
Tracks conversation turns, denomination context, knowledge level, and topics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: Literal["user", "assistant"]
    content: str
    timestamp: float = field(default_factory=time.time)
    rag_context: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    denomination_context: Optional[str] = None
    guardrail_triggered: bool = False


@dataclass
class ConversationState:
    """Full state of a conversation."""
    id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    denomination: str = "Non-denominational"
    knowledge_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    topics_discussed: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


# Maximum number of turns to keep in a conversation window
MAX_CONVERSATION_WINDOW = 30


class ConversationStore:
    """
    In-memory conversation store with optional Redis backend.

    Maintains conversation state across turns with a sliding window
    of MAX_CONVERSATION_WINDOW turns.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self._store: dict[str, ConversationState] = {}
        self._redis = None

        # Try to connect to Redis if URL is provided
        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
            except Exception as e:
                print(f"Redis connection failed, falling back to in-memory: {e}")
                self._redis = None

    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """
        Get a conversation by ID.

        Args:
            conversation_id: The conversation ID.

        Returns:
            ConversationState if found, None otherwise.
        """
        # Try Redis first
        if self._redis:
            try:
                data = self._redis.get(f"conv:{conversation_id}")
                if data:
                    return self._deserialize_conversation(json.loads(data))
            except Exception:
                pass

        return self._store.get(conversation_id)

    def upsert_turn(self, conversation_id: str, turn: ConversationTurn) -> None:
        """
        Add a turn to a conversation, creating it if necessary.

        Implements sliding window: keeps only the last MAX_CONVERSATION_WINDOW turns.

        Args:
            conversation_id: The conversation ID.
            turn: The conversation turn to add.
        """
        conversation = self.get_conversation(conversation_id)

        if conversation is None:
            conversation = ConversationState(
                id=conversation_id,
                denomination=turn.denomination_context or "Non-denominational",
            )

        # Add the turn
        conversation.turns.append(turn)
        conversation.last_active = time.time()

        # Sliding window: keep only last N turns
        if len(conversation.turns) > MAX_CONVERSATION_WINDOW:
            conversation.turns = conversation.turns[-MAX_CONVERSATION_WINDOW:]

        # Update denomination if specified
        if turn.denomination_context:
            conversation.denomination = turn.denomination_context

        # Store
        self._store[conversation_id] = conversation

        # Sync to Redis
        if self._redis:
            try:
                self._redis.setex(
                    f"conv:{conversation_id}",
                    3600 * 24,  # 24 hour expiry
                    json.dumps(self._serialize_conversation(conversation)),
                )
            except Exception:
                pass

    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear a conversation.

        Args:
            conversation_id: The conversation ID to clear.
        """
        if conversation_id in self._store:
            del self._store[conversation_id]

        if self._redis:
            try:
                self._redis.delete(f"conv:{conversation_id}")
            except Exception:
                pass

    def list_conversations(self, limit: int = 10) -> list[dict]:
        """
        List recent conversations.

        Args:
            limit: Maximum number of conversations to return.

        Returns:
            List of conversation summaries.
        """
        conversations = sorted(
            self._store.values(),
            key=lambda c: c.last_active,
            reverse=True,
        )[:limit]

        return [
            {
                "id": c.id,
                "denomination": c.denomination,
                "turn_count": len(c.turns),
                "last_active": c.last_active,
                "created_at": c.created_at,
                "preview": c.turns[-1].content[:100] if c.turns else "",
            }
            for c in conversations
        ]

    def get_conversation_messages(
        self,
        conversation_id: str,
        max_turns: int = 20,
    ) -> list[dict]:
        """
        Get conversation messages formatted for LLM context.

        Args:
            conversation_id: The conversation ID.
            max_turns: Maximum number of turns to return.

        Returns:
            List of message dicts with role and content.
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []

        return [
            {"role": turn.role, "content": turn.content}
            for turn in conversation.turns[-max_turns:]
        ]

    def _serialize_conversation(self, conversation: ConversationState) -> dict:
        """Serialize conversation state to dict for Redis storage."""
        return {
            "id": conversation.id,
            "denomination": conversation.denomination,
            "knowledge_level": conversation.knowledge_level,
            "topics_discussed": conversation.topics_discussed,
            "created_at": conversation.created_at,
            "last_active": conversation.last_active,
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "timestamp": t.timestamp,
                    "guardrail_triggered": t.guardrail_triggered,
                    "denomination_context": t.denomination_context,
                }
                for t in conversation.turns
            ],
        }

    def _deserialize_conversation(self, data: dict) -> ConversationState:
        """Deserialize conversation state from Redis storage."""
        turns = [
            ConversationTurn(
                role=t["role"],
                content=t["content"],
                timestamp=t.get("timestamp", time.time()),
                guardrail_triggered=t.get("guardrail_triggered", False),
                denomination_context=t.get("denomination_context"),
            )
            for t in data.get("turns", [])
        ]

        return ConversationState(
            id=data["id"],
            turns=turns,
            denomination=data.get("denomination", "Non-denominational"),
            knowledge_level=data.get("knowledge_level", "beginner"),
            topics_discussed=data.get("topics_discussed", []),
            created_at=data.get("created_at", time.time()),
            last_active=data.get("last_active", time.time()),
        )
