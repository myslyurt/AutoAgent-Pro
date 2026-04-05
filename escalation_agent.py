"""
escalation_agent.py — Escalation Agent

Handles the human handoff process when:
- Automated agents cannot resolve the issue
- Quality threshold is not met after retries
- Critical urgency tickets require immediate human attention
- Customer explicitly requests a human agent

Prepares a complete context package for the human agent.
"""

from __future__ import annotations

import json
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EscalationAgent:
    """
    Manages intelligent escalation to human agents.
    
    Creates a rich context package that gives human agents
    full visibility into what the AI already tried.
    """

    SYSTEM_PROMPT = """You are an escalation coordinator. Your job is to:
1. Write a clear, empathetic message to the customer explaining why a human will follow up
2. Create a comprehensive internal summary for the human agent
3. Set appropriate priority and suggested resolution path

Keep the customer message brief, warm, and specific about the expected response time.
The internal summary should be detailed and structured."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=settings.OPENAI_API_KEY,
        )

    async def prepare_handoff(
        self,
        state: dict,
        reason: str,
    ) -> dict:
        """
        Prepare a complete escalation package.

        Args:
            state: Full ticket state from the agent graph
            reason: Why escalation was triggered

        Returns:
            dict with customer_message, internal_summary, priority, routing
        """
        logger.info(
            f"Preparing escalation for ticket {state.get('ticket_id')}: {reason}"
        )

        # Determine expected response time based on urgency
        response_times = {
            "critical": "within 2 hours",
            "high": "within 4 hours",
            "medium": "within 24 hours",
            "low": "within 48 hours",
        }
        urgency = state.get("urgency", "medium")
        eta = response_times.get(urgency, "within 24 hours")

        # Build context for LLM
        actions_summary = ""
        actions = state.get("actions_taken", [])
        if actions:
            successful = [a for a in actions if a.get("status") == "success"]
            failed = [a for a in actions if a.get("status") == "error"]
            actions_summary = f"""
Actions attempted:
- Successful: {[a['tool'] for a in successful]}
- Failed: {[a['tool'] for a in failed]}"""

        prompt = f"""Create an escalation package for this ticket:

TICKET INFO:
- ID: {state.get('ticket_id', 'N/A')}
- Customer: {state.get('customer_id', 'N/A')}
- Intent: {state.get('intent', 'unknown')}
- Urgency: {urgency}
- Sentiment: {state.get('sentiment', 0.0):.2f}
- Escalation reason: {reason}

CUSTOMER MESSAGE:
{state.get('message', '')}

WHAT AI TRIED:
{actions_summary if actions_summary else "No actions were attempted"}
Quality score of draft: {state.get('quality_score', 0.0):.2f}

Write two things:

1. CUSTOMER MESSAGE (warm, empathetic, brief — 2-3 sentences):
Tell them a specialist will help them {eta}. Don't be robotic.

2. INTERNAL SUMMARY (for the human agent — structured):
- Issue summary
- What AI already tried
- Recommended next steps
- Any special considerations

Format:
CUSTOMER_MESSAGE: <message here>
---
INTERNAL_SUMMARY: <summary here>"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])

            content = response.content.strip()

            # Parse the two sections
            customer_msg = ""
            internal_summary = ""

            if "CUSTOMER_MESSAGE:" in content and "INTERNAL_SUMMARY:" in content:
                parts = content.split("---")
                customer_msg = parts[0].replace("CUSTOMER_MESSAGE:", "").strip()
                internal_summary = parts[1].replace("INTERNAL_SUMMARY:", "").strip() if len(parts) > 1 else ""
            else:
                # Fallback: use the whole response as customer message
                customer_msg = content
                internal_summary = f"Escalated: {reason}"

            # Determine routing based on intent and urgency
            routing = self._determine_routing(state)

            result = {
                "customer_message": customer_msg,
                "internal_summary": internal_summary,
                "reason": reason,
                "priority": urgency,
                "routing": routing,
                "escalated_at": datetime.utcnow().isoformat(),
                "context": {
                    "ticket_id": state.get("ticket_id"),
                    "customer_id": state.get("customer_id"),
                    "intent": state.get("intent"),
                    "sentiment": state.get("sentiment"),
                    "actions_attempted": len(actions),
                },
            }

            logger.info(
                f"Escalation prepared: priority={urgency}, routing={routing}"
            )
            return result

        except Exception as e:
            logger.exception(f"Escalation preparation error: {e}")
            return {
                "customer_message": (
                    "We've received your message and a member of our team will "
                    f"review it and respond {eta}. We apologize for any inconvenience."
                ),
                "internal_summary": f"Auto-escalated: {reason}",
                "reason": reason,
                "priority": urgency,
                "routing": "general_support",
                "escalated_at": datetime.utcnow().isoformat(),
                "context": {},
            }

    def _determine_routing(self, state: dict) -> str:
        """Route to the correct human team based on intent and urgency."""
        intent = state.get("intent", "general")
        urgency = state.get("urgency", "medium")

        routing_map = {
            ("refund", "critical"): "billing_manager",
            ("refund", "high"): "billing_senior",
            ("refund", "medium"): "billing_team",
            ("refund", "low"): "billing_team",
            ("technical", "critical"): "engineering_oncall",
            ("technical", "high"): "tech_support_senior",
            ("technical", "medium"): "tech_support",
            ("technical", "low"): "tech_support",
            ("billing", "critical"): "billing_manager",
            ("billing", "high"): "billing_senior",
        }

        return routing_map.get(
            (intent, urgency),
            "general_support"  # Default fallback
        )
