"""
session_memory.py — Conversation Memory (Redis-backed)

Stores conversation history per customer to provide context
across multiple support interactions.

Falls back to in-memory storage if Redis is unavailable (development mode).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# In-memory fallback store
_in_memory_store: dict[str, list[dict]] = {}


class SessionMemory:
    """
    Manages conversation history for customers.
    
    Uses Redis in production, falls back to in-memory dict in development.
    History is stored as a JSON list of {user, agent, timestamp} dicts.
    """

    MAX_HISTORY_TURNS = 20        # Keep last N turns per customer
    TTL_SECONDS = 86400 * 30      # 30 days retention

    def __init__(self):
        self.redis = None
        self._try_connect_redis()

    def _try_connect_redis(self):
        """Attempt Redis connection, fall back to in-memory on failure."""
        try:
            import redis.asyncio as redis
            from src.utils.config import settings
            self.redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
            logger.info("SessionMemory: Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}), using in-memory fallback")
            self.redis = None

    def _key(self, customer_id: str) -> str:
        return f"autoagent:history:{customer_id}"

    async def get_history(self, customer_id: str) -> list[dict]:
        """
        Retrieve conversation history for a customer.
        
        Returns list of dicts: [{user: str, agent: str, timestamp: str}, ...]
        """
        try:
            if self.redis:
                raw = await self.redis.get(self._key(customer_id))
                if raw:
                    return json.loads(raw)
                return []
            else:
                return _in_memory_store.get(customer_id, [])
        except Exception as e:
            logger.warning(f"Memory get error for {customer_id}: {e}")
            return []

    async def add_turn(
        self,
        customer_id: str,
        user_message: str,
        agent_response: str,
    ) -> None:
        """
        Append a conversation turn to the customer's history.
        
        Args:
            customer_id: Customer identifier
            user_message: What the customer said
            agent_response: What the agent responded
        """
        history = await self.get_history(customer_id)

        history.append({
            "user": user_message,
            "agent": agent_response,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Trim to max history size
        if len(history) > self.MAX_HISTORY_TURNS:
            history = history[-self.MAX_HISTORY_TURNS:]

        try:
            if self.redis:
                await self.redis.setex(
                    self._key(customer_id),
                    self.TTL_SECONDS,
                    json.dumps(history),
                )
            else:
                _in_memory_store[customer_id] = history
        except Exception as e:
            logger.warning(f"Memory set error for {customer_id}: {e}")

    async def clear_history(self, customer_id: str) -> None:
        """Clear conversation history for a customer (e.g., GDPR deletion)."""
        try:
            if self.redis:
                await self.redis.delete(self._key(customer_id))
            else:
                _in_memory_store.pop(customer_id, None)
        except Exception as e:
            logger.warning(f"Memory clear error for {customer_id}: {e}")
