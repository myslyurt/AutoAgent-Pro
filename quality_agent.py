"""
quality_agent.py — Quality Review Agent

Reviews draft responses before they are sent to customers.
Checks for:
- Accuracy: Does the response match the actions taken?
- Completeness: Are all customer questions addressed?
- Tone: Is the tone appropriate for the sentiment?
- Policy compliance: Does the response follow company guidelines?
- Clarity: Is the response easy to understand?

Returns a quality score and specific feedback for improvement.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QualityResult(BaseModel):
    """Structured quality review output."""
    score: float = Field(ge=0.0, le=1.0, description="Overall quality score")
    approved: bool = Field(description="Whether the response is ready to send")
    accuracy_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    tone_score: float = Field(ge=0.0, le=1.0)
    clarity_score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(description="Specific issues found")
    feedback: str = Field(description="Actionable feedback for improvement")


class QualityAgent:
    """
    Reviews agent-generated responses for quality before sending.
    Uses a separate LLM call with a strict evaluation rubric.
    """

    APPROVAL_THRESHOLD = 0.75  # Minimum score to auto-approve

    SYSTEM_PROMPT = """You are a customer support quality assurance specialist.
Your job is to evaluate draft responses before they are sent to customers.

Evaluate responses on these dimensions (0.0 to 1.0):
1. ACCURACY: Does the response accurately reflect what was done? No hallucinations?
2. COMPLETENESS: Does it address ALL parts of the customer's request?
3. TONE: Is the tone empathetic and professional? Appropriate for the situation?
4. CLARITY: Is it easy to understand? Free of jargon? Clear next steps?

APPROVAL CRITERIA:
- Approve if overall score >= 0.75 and no critical issues
- Reject if: contains factual errors, makes promises we can't keep, 
  is dismissive/rude, leaves critical questions unanswered

Respond ONLY with valid JSON — no markdown, no preamble."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=settings.OPENAI_API_KEY,
            response_format={"type": "json_object"},
        )

    async def review(
        self,
        original_message: str,
        draft_response: str,
        intent: str,
        actions_taken: list[dict] | None = None,
    ) -> dict:
        """
        Review a draft response for quality.

        Args:
            original_message: The customer's original message
            draft_response: The agent's draft response
            intent: Ticket intent for context
            actions_taken: List of actions executed (for accuracy check)

        Returns:
            dict with score, approved, feedback, and dimension scores
        """
        actions_summary = ""
        if actions_taken:
            successful = [a for a in actions_taken if a["status"] == "success"]
            failed = [a for a in actions_taken if a["status"] == "error"]
            actions_summary = f"""
ACTIONS SUCCESSFULLY TAKEN: {json.dumps([
    {"tool": a["tool"], "result": a["result"]} for a in successful
], indent=2)}
FAILED ACTIONS: {[a["tool"] for a in failed]}"""

        prompt = f"""CUSTOMER MESSAGE:
{original_message}

INTENT: {intent}
{actions_summary}

DRAFT RESPONSE TO EVALUATE:
{draft_response}

Evaluate this response and respond with JSON:
{{
  "score": <0.0-1.0 overall>,
  "approved": <true/false>,
  "accuracy_score": <0.0-1.0>,
  "completeness_score": <0.0-1.0>,
  "tone_score": <0.0-1.0>,
  "clarity_score": <0.0-1.0>,
  "issues": ["<specific issue 1>", "<specific issue 2>"],
  "feedback": "<actionable feedback for improvement>"
}}"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])

            data = json.loads(response.content)
            result = QualityResult(**data)

            # Override approved if score doesn't meet threshold
            if result.score < self.APPROVAL_THRESHOLD:
                result.approved = False
                if not result.issues:
                    result.issues = [f"Score {result.score:.2f} below threshold {self.APPROVAL_THRESHOLD}"]

            logger.info(
                f"Quality review: score={result.score:.2f}, "
                f"approved={result.approved}, "
                f"issues={len(result.issues)}"
            )

            return result.model_dump()

        except Exception as e:
            logger.exception(f"Quality review error: {e}")
            # On error, return a moderate score — don't block the pipeline
            return {
                "score": 0.6,
                "approved": False,
                "accuracy_score": 0.6,
                "completeness_score": 0.6,
                "tone_score": 0.6,
                "clarity_score": 0.6,
                "issues": ["Quality review failed — manual review recommended"],
                "feedback": "Quality check encountered an error. Please review manually.",
            }
