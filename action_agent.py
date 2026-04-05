"""
action_agent.py — Action Agent (Tool-Calling)

Executes real-world actions on behalf of customers by calling
external tools/APIs. Uses LangChain's tool-calling interface.

Available tools:
- lookup_customer: Get customer profile and history from CRM
- lookup_order: Get order details and status
- process_refund: Issue refund via payment system
- create_ticket: Create escalation ticket in help desk
- send_notification: Send email/SMS to customer
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.tools.crm_tools import lookup_customer, lookup_customer_orders
from src.tools.order_tools import lookup_order, update_order_status
from src.tools.payment_tools import process_refund, get_payment_history
from src.tools.ticket_tools import create_support_ticket, update_ticket_status
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ActionAgent:
    """
    Tool-calling agent that executes actions in external systems.
    
    Uses a ReAct-style loop: Reason → Act → Observe → Repeat.
    Stops when the task is complete or max iterations reached.
    """

    SYSTEM_PROMPT = """You are a support action agent with access to tools for managing 
customer accounts, orders, and payments. 

Your job:
1. Understand what action the customer needs
2. Call the appropriate tools to gather information and execute actions
3. Handle errors gracefully — if a tool fails, try alternatives
4. Never guess at order IDs, customer IDs, or amounts — always look them up first

IMPORTANT:
- Always verify customer identity before processing refunds
- Only process refunds up to $500 without escalation
- Log every action you take for audit trail
- If a tool returns an error, report it honestly — don't fabricate results

Be methodical: gather info first, then act."""

    # Tools available to this agent
    TOOLS = [
        lookup_customer,
        lookup_customer_orders,
        lookup_order,
        update_order_status,
        process_refund,
        get_payment_history,
        create_support_ticket,
        update_ticket_status,
    ]

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.SUPERVISOR_MODEL,
            temperature=0.0,
            api_key=settings.OPENAI_API_KEY,
        ).bind_tools(self.TOOLS)

        # Map tool names to actual functions for execution
        self.tool_map = {tool.name: tool for tool in self.TOOLS}

    async def execute(
        self,
        intent: str,
        message: str,
        customer_id: str,
        context: list[str] | None = None,
    ) -> dict:
        """
        Execute the required actions for a support ticket.

        Args:
            intent: Classified intent (determines which tools to use)
            message: Customer message
            customer_id: Customer identifier
            context: Retrieved knowledge base context

        Returns:
            dict with 'actions_taken' list and 'success' bool
        """
        logger.info(f"ActionAgent executing for intent='{intent}', customer={customer_id}")

        context_text = ""
        if context:
            context_text = "\n\nRELEVANT POLICIES:\n" + "\n".join(context[:2])

        prompt = f"""Customer ID: {customer_id}
Intent: {intent}
Customer message: {message}{context_text}

Execute the necessary actions to resolve this customer's request.
Start by looking up the customer record, then proceed based on what you find."""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        actions_taken = []
        max_iterations = 8
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Ask LLM what to do next
            response = await self.llm.ainvoke(messages)
            messages.append(response)

            # Check if LLM wants to call tools
            if not response.tool_calls:
                logger.info(f"ActionAgent finished after {iteration} iterations")
                break

            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                logger.info(f"Calling tool: {tool_name}({json.dumps(tool_args)[:100]})")

                try:
                    # Execute the tool
                    tool_fn = self.tool_map.get(tool_name)
                    if not tool_fn:
                        raise ValueError(f"Unknown tool: {tool_name}")

                    tool_result = tool_fn.invoke(tool_args)

                    # Record the action
                    actions_taken.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": tool_result,
                        "status": "success",
                    })

                    # Add tool result to message history
                    messages.append(
                        ToolMessage(
                            content=json.dumps(tool_result),
                            tool_call_id=tool_call_id,
                        )
                    )

                    logger.info(f"Tool {tool_name} succeeded")

                except Exception as e:
                    error_msg = f"Tool '{tool_name}' failed: {str(e)}"
                    logger.warning(error_msg)

                    actions_taken.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": None,
                        "status": "error",
                        "error": str(e),
                    })

                    messages.append(
                        ToolMessage(
                            content=json.dumps({"error": str(e), "status": "failed"}),
                            tool_call_id=tool_call_id,
                        )
                    )

        # Determine overall success
        success = any(a["status"] == "success" for a in actions_taken)
        critical_failure = all(a["status"] == "error" for a in actions_taken) if actions_taken else False

        logger.info(
            f"ActionAgent complete: {len(actions_taken)} actions, "
            f"success={success}, critical_failure={critical_failure}"
        )

        return {
            "actions_taken": actions_taken,
            "success": success and not critical_failure,
            "iterations": iteration,
        }
