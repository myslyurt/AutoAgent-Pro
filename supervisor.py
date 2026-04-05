"""
supervisor.py — Supervisor Agent (Main Orchestrator)

The Supervisor is the brain of the multi-agent system. It receives incoming
support tickets, decides which sub-agents to invoke in what order, aggregates
their outputs, and produces the final customer-facing response.

Architecture: ReAct-style reasoning loop using LangGraph StateGraph.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agents.triage_agent import TriageAgent
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.action_agent import ActionAgent
from src.agents.quality_agent import QualityAgent
from src.agents.escalation_agent import EscalationAgent
from src.memory.session_memory import SessionMemory
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.metrics import track_ticket

logger = get_logger(__name__)


# ─── State Schema ─────────────────────────────────────────────────────────────

class TicketState(TypedDict):
    """
    Shared state that flows through the entire agent graph.
    Each agent reads from and writes to this state dict.
    """
    # Core ticket data
    ticket_id: str
    customer_id: str
    message: str                      # Raw customer message
    channel: str                      # email | chat | api

    # Triage outputs
    intent: str                       # refund | order_status | technical | billing | general
    urgency: str                      # low | medium | high | critical
    category: str                     # Granular category label
    sentiment: float                  # -1.0 (very negative) to 1.0 (very positive)

    # Knowledge agent outputs
    retrieved_context: list[str]      # Relevant docs from vector store
    knowledge_confidence: float       # How confident RAG retrieval is

    # Action agent outputs
    actions_taken: list[dict]         # List of tool calls made + results
    action_success: bool

    # Draft response
    draft_response: str

    # Quality review outputs
    quality_score: float              # 0.0 - 1.0
    quality_feedback: str
    response_approved: bool

    # Final decision
    final_response: str
    escalate_to_human: bool
    escalation_reason: str

    # Conversation history (LangGraph managed)
    messages: Annotated[list[BaseMessage], add_messages]

    # Routing control
    next_step: str
    iteration_count: int


# ─── Supervisor Agent ─────────────────────────────────────────────────────────

class SupervisorAgent:
    """
    Orchestrates the multi-agent pipeline using a LangGraph StateGraph.

    Flow:
        START
          └─> triage
                └─> [knowledge + action] (parallel or sequential)
                      └─> draft_response
                            └─> quality_review
                                  └─> [finalize | escalate]
                                        └─> END
    """

    SYSTEM_PROMPT = """You are a supervisor AI that orchestrates a team of specialized
support agents. Your job is to:
1. Analyze the triage results and decide which agents to invoke
2. Synthesize information from multiple agents into coherent responses
3. Ensure the final response is accurate, empathetic, and actionable
4. Never make up information — if agents couldn't find an answer, escalate

Always prioritize customer satisfaction while staying within company policy."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.SUPERVISOR_MODEL,
            temperature=0.1,
            api_key=settings.OPENAI_API_KEY,
        )
        # Initialize all sub-agents
        self.triage = TriageAgent()
        self.knowledge = KnowledgeAgent()
        self.action = ActionAgent()
        self.quality = QualityAgent()
        self.escalation = EscalationAgent()
        self.memory = SessionMemory()

        # Build the agent graph
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct the LangGraph StateGraph for the agent pipeline."""
        builder = StateGraph(TicketState)

        # Register all nodes
        builder.add_node("triage", self._node_triage)
        builder.add_node("knowledge_retrieval", self._node_knowledge)
        builder.add_node("action_execution", self._node_action)
        builder.add_node("draft_response", self._node_draft_response)
        builder.add_node("quality_review", self._node_quality_review)
        builder.add_node("escalate", self._node_escalate)
        builder.add_node("finalize", self._node_finalize)

        # Define edges
        builder.add_edge(START, "triage")

        # After triage: route based on intent
        builder.add_conditional_edges(
            "triage",
            self._route_after_triage,
            {
                "needs_knowledge": "knowledge_retrieval",
                "needs_action": "action_execution",
                "escalate_immediately": "escalate",
            },
        )

        # After knowledge retrieval: check if action also needed
        builder.add_conditional_edges(
            "knowledge_retrieval",
            self._route_after_knowledge,
            {
                "needs_action": "action_execution",
                "draft": "draft_response",
            },
        )

        # After action: draft response
        builder.add_edge("action_execution", "draft_response")

        # Quality gate
        builder.add_edge("draft_response", "quality_review")
        builder.add_conditional_edges(
            "quality_review",
            self._route_after_quality,
            {
                "approved": "finalize",
                "revise": "draft_response",        # Re-draft with feedback
                "escalate": "escalate",
            },
        )

        builder.add_edge("finalize", END)
        builder.add_edge("escalate", END)

        return builder.compile()

    # ─── Node Implementations ─────────────────────────────────────────────────

    async def _node_triage(self, state: TicketState) -> dict:
        """Run the triage agent to classify intent, urgency, and category."""
        logger.info(f"[{state['ticket_id']}] Running triage agent")

        result = await self.triage.classify(
            message=state["message"],
            customer_id=state["customer_id"],
            history=await self.memory.get_history(state["customer_id"]),
        )

        logger.info(
            f"[{state['ticket_id']}] Triage result: "
            f"intent={result['intent']}, urgency={result['urgency']}, "
            f"sentiment={result['sentiment']:.2f}"
        )

        return {
            "intent": result["intent"],
            "urgency": result["urgency"],
            "category": result["category"],
            "sentiment": result["sentiment"],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    async def _node_knowledge(self, state: TicketState) -> dict:
        """Retrieve relevant knowledge from vector store using RAG."""
        logger.info(f"[{state['ticket_id']}] Running knowledge agent")

        result = await self.knowledge.retrieve(
            query=state["message"],
            intent=state["intent"],
            category=state["category"],
            top_k=5,
        )

        logger.info(
            f"[{state['ticket_id']}] Knowledge: "
            f"{len(result['docs'])} docs, confidence={result['confidence']:.2f}"
        )

        return {
            "retrieved_context": result["docs"],
            "knowledge_confidence": result["confidence"],
        }

    async def _node_action(self, state: TicketState) -> dict:
        """Execute real-world actions via tool-calling (refunds, lookups, etc.)."""
        logger.info(f"[{state['ticket_id']}] Running action agent")

        result = await self.action.execute(
            intent=state["intent"],
            message=state["message"],
            customer_id=state["customer_id"],
            context=state.get("retrieved_context", []),
        )

        logger.info(
            f"[{state['ticket_id']}] Actions: "
            f"{len(result['actions_taken'])} actions, success={result['success']}"
        )

        return {
            "actions_taken": result["actions_taken"],
            "action_success": result["success"],
        }

    async def _node_draft_response(self, state: TicketState) -> dict:
        """Generate a customer-facing response using all gathered context."""
        logger.info(f"[{state['ticket_id']}] Drafting response")

        # Build context for the LLM
        context_parts = []
        if state.get("retrieved_context"):
            context_parts.append(
                "KNOWLEDGE BASE CONTEXT:\n" + "\n---\n".join(state["retrieved_context"])
            )
        if state.get("actions_taken"):
            context_parts.append(
                "ACTIONS COMPLETED:\n"
                + json.dumps(state["actions_taken"], indent=2)
            )
        if state.get("quality_feedback"):
            context_parts.append(
                f"QUALITY FEEDBACK (incorporate this): {state['quality_feedback']}"
            )

        prompt = f"""Customer message: {state['message']}

Intent: {state['intent']} | Urgency: {state['urgency']} | Sentiment: {state['sentiment']:.2f}

{chr(10).join(context_parts)}

Write a helpful, empathetic response that:
1. Directly addresses the customer's issue
2. References specific actions taken (if any)
3. Provides clear next steps
4. Matches tone to sentiment (more empathetic for negative sentiment)
5. Is concise (under 200 words unless complexity requires more)

Response:"""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        draft = response.content.strip()

        logger.info(f"[{state['ticket_id']}] Draft generated ({len(draft)} chars)")

        return {"draft_response": draft}

    async def _node_quality_review(self, state: TicketState) -> dict:
        """Quality agent reviews the draft before sending."""
        logger.info(f"[{state['ticket_id']}] Running quality review")

        result = await self.quality.review(
            original_message=state["message"],
            draft_response=state["draft_response"],
            intent=state["intent"],
            actions_taken=state.get("actions_taken", []),
        )

        logger.info(
            f"[{state['ticket_id']}] Quality score: {result['score']:.2f}, "
            f"approved: {result['approved']}"
        )

        return {
            "quality_score": result["score"],
            "quality_feedback": result["feedback"],
            "response_approved": result["approved"],
        }

    async def _node_escalate(self, state: TicketState) -> dict:
        """Prepare escalation package for human agent handoff."""
        logger.info(f"[{state['ticket_id']}] Escalating to human agent")

        result = await self.escalation.prepare_handoff(
            state=state,
            reason=state.get("escalation_reason", "Quality threshold not met"),
        )

        return {
            "escalate_to_human": True,
            "final_response": result["customer_message"],
            "escalation_reason": result["reason"],
        }

    async def _node_finalize(self, state: TicketState) -> dict:
        """Finalize the approved response and persist to memory."""
        logger.info(f"[{state['ticket_id']}] Finalizing response")

        # Store in conversation memory
        await self.memory.add_turn(
            customer_id=state["customer_id"],
            user_message=state["message"],
            agent_response=state["draft_response"],
        )

        # Track metrics
        track_ticket(
            intent=state["intent"],
            urgency=state["urgency"],
            escalated=False,
            quality_score=state["quality_score"],
            iterations=state["iteration_count"],
        )

        return {
            "final_response": state["draft_response"],
            "escalate_to_human": False,
        }

    # ─── Routing Functions ────────────────────────────────────────────────────

    def _route_after_triage(self, state: TicketState) -> str:
        """Decide next step based on triage results."""
        # Critical urgency or very negative sentiment → escalate immediately
        if state["urgency"] == "critical" and state["sentiment"] < -0.7:
            state["escalation_reason"] = "Critical urgency with very negative sentiment"
            return "escalate_immediately"

        # Action-oriented intents need the action agent first
        if state["intent"] in ("refund", "order_status", "account_action"):
            return "needs_action"

        # Everything else: retrieve knowledge first
        return "needs_knowledge"

    def _route_after_knowledge(self, state: TicketState) -> str:
        """After knowledge retrieval, check if action is also needed."""
        if state["intent"] in ("refund", "order_status") and state["knowledge_confidence"] < 0.6:
            return "needs_action"
        return "draft"

    def _route_after_quality(self, state: TicketState) -> str:
        """Route based on quality review result."""
        if state["response_approved"]:
            return "approved"

        # Safety: don't loop forever
        if state.get("iteration_count", 0) >= settings.MAX_AGENT_ITERATIONS:
            state["escalation_reason"] = "Max iterations reached without quality approval"
            return "escalate"

        if state["quality_score"] < 0.4:
            # Very low quality → escalate rather than retry
            state["escalation_reason"] = f"Quality score too low: {state['quality_score']:.2f}"
            return "escalate"

        return "revise"

    # ─── Public Interface ─────────────────────────────────────────────────────

    async def process_ticket(
        self,
        ticket_id: str,
        customer_id: str,
        message: str,
        channel: str = "api",
    ) -> dict:
        """
        Main entry point. Process a support ticket through the full agent pipeline.

        Args:
            ticket_id: Unique ticket identifier
            customer_id: Customer identifier for memory/context lookup
            message: The customer's support message
            channel: Origin channel (email | chat | api)

        Returns:
            dict with final_response, escalate_to_human, and metadata
        """
        logger.info(f"Processing ticket {ticket_id} from customer {customer_id}")

        initial_state: TicketState = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "message": message,
            "channel": channel,
            # Initialize all fields with defaults
            "intent": "",
            "urgency": "medium",
            "category": "",
            "sentiment": 0.0,
            "retrieved_context": [],
            "knowledge_confidence": 0.0,
            "actions_taken": [],
            "action_success": False,
            "draft_response": "",
            "quality_score": 0.0,
            "quality_feedback": "",
            "response_approved": False,
            "final_response": "",
            "escalate_to_human": False,
            "escalation_reason": "",
            "messages": [HumanMessage(content=message)],
            "next_step": "",
            "iteration_count": 0,
        }

        try:
            final_state = await self.graph.ainvoke(initial_state)

            return {
                "ticket_id": ticket_id,
                "status": "escalated" if final_state["escalate_to_human"] else "resolved",
                "response": final_state["final_response"],
                "metadata": {
                    "intent": final_state["intent"],
                    "urgency": final_state["urgency"],
                    "sentiment": final_state["sentiment"],
                    "quality_score": final_state.get("quality_score", 0.0),
                    "actions_taken": final_state.get("actions_taken", []),
                    "escalated": final_state["escalate_to_human"],
                    "escalation_reason": final_state.get("escalation_reason", ""),
                    "iterations": final_state["iteration_count"],
                },
            }

        except Exception as e:
            logger.exception(f"Error processing ticket {ticket_id}: {e}")
            raise
