"""
order_tools.py — Order Management Tools
ticket_tools.py — Help Desk Ticket Tools
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta

from langchain_core.tools import tool

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _is_mock_mode() -> bool:
    return settings.MOCK_APIS


# ─── Order Tools ───────────────────────────────────────────────────────────────

@tool
def lookup_order(order_id: str) -> dict:
    """
    Look up order details and current status.
    
    Args:
        order_id: The order identifier (e.g., 'ORD-4521')
        
    Returns:
        Order details including status, items, shipping info, and timeline
    """
    logger.info(f"Order lookup: order_id={order_id}")
    
    if _is_mock_mode():
        statuses = ["processing", "shipped", "out_for_delivery", "delivered", "cancelled"]
        status = random.choice(statuses)
        
        order = {
            "order_id": order_id,
            "status": status,
            "customer_id": f"cust_{random.randint(1000, 9999)}",
            "items": [
                {
                    "sku": f"SKU-{random.randint(100, 999)}",
                    "name": random.choice(["Premium Widget", "Deluxe Gadget", "Standard Package"]),
                    "quantity": random.randint(1, 3),
                    "unit_price": round(random.uniform(20, 200), 2),
                }
            ],
            "subtotal": round(random.uniform(30, 500), 2),
            "shipping_cost": round(random.uniform(0, 25), 2),
            "total": round(random.uniform(35, 525), 2),
            "payment_status": "paid" if status != "cancelled" else "refunded",
            "shipping_address": {
                "city": "San Francisco",
                "state": "CA",
                "country": "US",
            },
            "created_at": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
        }
        
        # Add tracking for shipped orders
        if status in ("shipped", "out_for_delivery", "delivered"):
            order["tracking"] = {
                "carrier": random.choice(["UPS", "FedEx", "USPS", "DHL"]),
                "tracking_number": f"TRK{random.randint(10000000, 99999999)}",
                "estimated_delivery": (datetime.now() + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d"),
                "last_update": f"Package {status.replace('_', ' ')} - In transit",
            }
        
        return {"status": "found", "order": order}
    
    import httpx
    response = httpx.get(
        f"{settings.ORDER_API_URL}/orders/{order_id}",
        headers={"Authorization": f"Bearer {settings.ORDER_API_KEY}"},
        timeout=10.0,
    )
    if response.status_code == 404:
        return {"status": "not_found", "order": None}
    response.raise_for_status()
    return {"status": "found", "order": response.json()}


@tool
def update_order_status(order_id: str, new_status: str, note: str = "") -> dict:
    """
    Update the status of an order.
    
    Args:
        order_id: The order identifier
        new_status: New status (cancelled | on_hold | expedited)
        note: Internal note explaining the change
        
    Returns:
        Confirmation of the status update
    """
    allowed_statuses = {"cancelled", "on_hold", "expedited"}
    if new_status not in allowed_statuses:
        return {
            "status": "error",
            "error": f"Invalid status '{new_status}'. Allowed: {allowed_statuses}",
        }
    
    logger.info(f"Updating order {order_id} to status '{new_status}': {note}")
    
    if _is_mock_mode():
        return {
            "status": "success",
            "order_id": order_id,
            "previous_status": "processing",
            "new_status": new_status,
            "updated_at": datetime.utcnow().isoformat(),
            "note": note,
        }
    
    import httpx
    response = httpx.patch(
        f"{settings.ORDER_API_URL}/orders/{order_id}",
        json={"status": new_status, "note": note},
        headers={"Authorization": f"Bearer {settings.ORDER_API_KEY}"},
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()


# ─── Ticket Tools ──────────────────────────────────────────────────────────────

@tool
def create_support_ticket(
    customer_id: str,
    subject: str,
    description: str,
    priority: str = "medium",
    category: str = "general",
) -> dict:
    """
    Create a support ticket in the help desk system for human follow-up.
    
    Args:
        customer_id: Customer identifier
        subject: Brief ticket subject line
        description: Full ticket description with context
        priority: Ticket priority (low | medium | high | urgent)
        category: Ticket category for routing
        
    Returns:
        Created ticket ID and URL
    """
    allowed_priorities = {"low", "medium", "high", "urgent"}
    if priority not in allowed_priorities:
        priority = "medium"
    
    logger.info(f"Creating ticket: customer={customer_id}, priority={priority}")
    
    if _is_mock_mode():
        ticket_id = f"TKT-{random.randint(10000, 99999)}"
        return {
            "status": "created",
            "ticket_id": ticket_id,
            "subject": subject,
            "priority": priority,
            "category": category,
            "customer_id": customer_id,
            "url": f"https://support.example.com/tickets/{ticket_id}",
            "created_at": datetime.utcnow().isoformat(),
            "assigned_team": f"{category}_team",
        }
    
    import httpx
    response = httpx.post(
        f"{settings.HELPDESK_API_URL}/tickets",
        json={
            "customer_id": customer_id,
            "subject": subject,
            "description": description,
            "priority": priority,
            "category": category,
        },
        headers={"Authorization": f"Bearer {settings.HELPDESK_API_KEY}"},
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()


@tool
def update_ticket_status(ticket_id: str, status: str, comment: str = "") -> dict:
    """
    Update the status of an existing support ticket.
    
    Args:
        ticket_id: The ticket identifier
        status: New status (open | pending | resolved | closed)
        comment: Internal comment about the status change
        
    Returns:
        Confirmation of the update
    """
    logger.info(f"Updating ticket {ticket_id} → {status}")
    
    if _is_mock_mode():
        return {
            "status": "success",
            "ticket_id": ticket_id,
            "new_status": status,
            "comment": comment,
            "updated_at": datetime.utcnow().isoformat(),
        }
    
    import httpx
    response = httpx.patch(
        f"{settings.HELPDESK_API_URL}/tickets/{ticket_id}",
        json={"status": status, "comment": comment},
        headers={"Authorization": f"Bearer {settings.HELPDESK_API_KEY}"},
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()
