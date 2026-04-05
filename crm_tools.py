"""
crm_tools.py — CRM Integration Tools

LangChain tools for interacting with the Customer Relationship Management system.
In production, these make real HTTP requests to your CRM API.
In development/demo mode, they return realistic mock data.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from langchain_core.tools import tool

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _is_mock_mode() -> bool:
    return settings.MOCK_APIS or not settings.CRM_API_URL


# ─── Mock Data Generator ───────────────────────────────────────────────────────

def _generate_mock_customer(customer_id: str) -> dict:
    """Generate realistic mock customer data for demos."""
    tiers = ["bronze", "silver", "gold", "platinum"]
    tier = random.choice(tiers)
    
    created_days_ago = random.randint(30, 1000)
    total_orders = random.randint(1, 50)
    
    return {
        "customer_id": customer_id,
        "email": f"customer_{customer_id[-4:]}@example.com",
        "name": f"Customer {customer_id[-4:].upper()}",
        "tier": tier,
        "status": "active",
        "total_orders": total_orders,
        "lifetime_value": round(total_orders * random.uniform(30, 200), 2),
        "created_at": (datetime.now() - timedelta(days=created_days_ago)).isoformat(),
        "is_vip": tier in ("gold", "platinum"),
        "open_tickets": random.randint(0, 3),
        "last_order_date": (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
    }


# ─── CRM Tools ────────────────────────────────────────────────────────────────

@tool
def lookup_customer(customer_id: str) -> dict:
    """
    Look up a customer record in the CRM system.
    
    Args:
        customer_id: The unique customer identifier
        
    Returns:
        Customer profile including name, email, tier, order history summary
    """
    logger.info(f"CRM lookup: customer_id={customer_id}")
    
    if _is_mock_mode():
        # Return realistic mock data
        customer = _generate_mock_customer(customer_id)
        logger.debug(f"Mock CRM response for {customer_id}: tier={customer['tier']}")
        return {"status": "found", "customer": customer}
    
    # Production: real API call
    import httpx
    try:
        response = httpx.get(
            f"{settings.CRM_API_URL}/customers/{customer_id}",
            headers={"Authorization": f"Bearer {settings.CRM_API_KEY}"},
            timeout=10.0,
        )
        response.raise_for_status()
        return {"status": "found", "customer": response.json()}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"status": "not_found", "customer": None}
        raise


@tool
def lookup_customer_orders(customer_id: str, limit: int = 5) -> dict:
    """
    Retrieve recent orders for a customer from the CRM.
    
    Args:
        customer_id: The unique customer identifier
        limit: Maximum number of orders to return (default: 5)
        
    Returns:
        List of recent orders with status and amounts
    """
    logger.info(f"CRM order lookup: customer_id={customer_id}, limit={limit}")
    
    if _is_mock_mode():
        orders = []
        statuses = ["delivered", "shipped", "processing", "cancelled", "refunded"]
        
        for i in range(min(limit, random.randint(1, 8))):
            order_num = random.randint(1000, 9999)
            amount = round(random.uniform(15, 500), 2)
            orders.append({
                "order_id": f"ORD-{order_num}",
                "status": random.choice(statuses),
                "amount": amount,
                "currency": "USD",
                "items": random.randint(1, 5),
                "created_at": (datetime.now() - timedelta(days=random.randint(1, 180))).isoformat(),
                "tracking_number": f"TRK{random.randint(10000000, 99999999)}" if random.random() > 0.3 else None,
            })
        
        return {"status": "success", "orders": orders, "total_count": len(orders)}
    
    # Production: real API call
    import httpx
    response = httpx.get(
        f"{settings.CRM_API_URL}/customers/{customer_id}/orders",
        params={"limit": limit},
        headers={"Authorization": f"Bearer {settings.CRM_API_KEY}"},
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()
