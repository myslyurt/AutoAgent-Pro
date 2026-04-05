"""
triage_agent.py — Triage Agent

Classifies incoming support tickets by:
- Intent: what the customer wants
- Urgency: how time-sensitive it is
- Category: fine-grained topic label
- Sentiment: emotional tone of the message

Uses structured output (JSON mode) for reliable parsing.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Output Schema ─────────────────────────────────────────────────────────────

class TriageResult(BaseModel):
    """Structured output from the triage agent."""

    intent: str = Field(
        description="Primary intent: refund | order_status | technical | billing | general | account_action"
    )
    urgency: str = Field(
        description="Urgency level: low | medium | high | critical"
    )
    category: str = Field(
        description="Specific category label, e.g. 'duplicate_charge', 'missing_package', 'login_issue'"
    )
    sentiment: float = Field(
        description="Sentiment score from -1.0 (very negative) to 1.0 (very positive)",
        ge=-1.0,
        le=1.0,
    )
    confidence: float = Field(
        description="Classification confidence from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="Brief explanation of the classification decision"
    )

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        valid = {"refund", "order_status", "technical", "billing", "general", "account_action"}
        if v not in valid:
            raise ValueError(f"Intent must be one of {valid}, got '{v}'")
        return v

    @field_validator("urgency")
    @classmethod
    def validate_urgency(cls, v: str) -> str:
        valid = {"low", "medium", "high", "critical"}
        if v not in valid:
            raise ValueError(f"Urgency must be one of {valid}, got '{v}'")
        return v


# ─── Triage Agent ──────────────────────────────────────────────────────────────

class TriageAgent:
    """
    Classifies support tickets using a fast, cheap model (gpt-4o-mini).
    Uses JSON mode for reliable structured output.
    """

    SYSTEM_PROMPT = """You are a support ticket triage specialist. Analyze the customer message
and classify it accurately. Always respond with valid JSON matching the schema exactly.

INTENT DEFINITIONS:
- refund: Customer wants money back or reports a charge issue
- order_status: Questions about shipping, delivery, tracking
- technical: App/product not working, bugs, errors
- billing: Subscription, invoicing, payment method questions (not refunds)
- account_action: Password reset, account deletion, data export
- general: Everything else

URGENCY CRITERIA:
- critical: Legal threats, data breach concerns, service completely down, very angry + high-value customer
- high: Significant financial impact, repeated failures, service degraded
- medium: Standard issues with moderate impact
- low: General questions, minor issues, "how to" queries

SENTIMENT GUIDELINES:
- Negative (-1.0 to -0.3): Angry, frustrated, threatening language
- Neutral (-0.3 to 0.3): Factual, calm, neutral tone
- Positive (0.3 to 1.0): Polite, understanding, positive phrasing

Respond ONLY with a JSON object — no markdown, no explanation outside JSON."""

    def __init__(self):
        # Use cheaper, faster model for triage
        self.llm = ChatOpenAI(
            model=settings.TRIAGE_MODEL,   # gpt-4o-mini
            temperature=0.0,               # Deterministic classification
            api_key=settings.OPENAI_API_KEY,
            response_format={"type": "json_object"},
        )

    async def classify(
        self,
        message: str,
        customer_id: str,
        history: list[dict] | None = None,
    ) -> dict:
        """
        Classify a support message.

        Args:
            message: Raw customer message
            customer_id: For context (VIP status, history, etc.)
            history: Previous conversation turns for context

        Returns:
            dict with intent, urgency, category, sentiment, confidence, reasoning
        """
        history_text = ""
        if history:
            recent = history[-3:]  # Last 3 turns for context
            history_text = "\n\nPREVIOUS CONVERSATION:\n" + "\n".join(
                f"Customer: {turn['user']}\nAgent: {turn['agent']}"
                for turn in recent
            )

        prompt = f"""Classify this support ticket:

CUSTOMER ID: {customer_id}
MESSAGE: {message}{history_text}

Respond with JSON:
{{
  "intent": "<one of: refund|order_status|technical|billing|general|account_action>",
  "urgency": "<one of: low|medium|high|critical>",
  "category": "<specific category label>",
  "sentiment": <float -1.0 to 1.0>,
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<brief explanation>"
}}"""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            raw = response.content.strip()

            # Parse and validate with Pydantic
            data = json.loads(raw)
            result = TriageResult(**data)

            logger.debug(
                f"Triage for customer {customer_id}: "
                f"{result.intent}/{result.urgency} "
                f"(confidence={result.confidence:.2f})"
            )

            return result.model_dump()

        except json.JSONDecodeError as e:
            logger.error(f"Triage JSON parse error: {e}. Raw: {raw[:200]}")
            # Return safe defaults on parse failure
            return self._default_result(message)

        except Exception as e:
            logger.exception(f"Triage error: {e}")
            return self._default_result(message)

    def _default_result(self, message: str) -> dict:
        """Safe fallback when classification fails."""
        # Basic keyword-based fallback
        msg_lower = message.lower()
        intent = "general"
        if any(w in msg_lower for w in ["refund", "charge", "money back", "charged twice"]):
            intent = "refund"
        elif any(w in msg_lower for w in ["order", "shipping", "delivery", "tracking"]):
            intent = "order_status"
        elif any(w in msg_lower for w in ["not working", "error", "bug", "crash", "broken"]):
            intent = "technical"

        return {
            "intent": intent,
            "urgency": "medium",
            "category": "unclassified",
            "sentiment": 0.0,
            "confidence": 0.3,
            "reasoning": "Fallback classification due to LLM error",
        }
