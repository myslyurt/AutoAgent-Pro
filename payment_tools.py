"""
payment_tools.py — Payment Processing Tools
order_tools.py — Order Management Tools
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime

from langchain_core.tools import tool

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _is_mock_mode() -> bool:
    return settings.MOCK_APIS or not settings.PAYMENT_API_URL


# ─── Payment Tools ─────────────────────────────────────────────────────────────

@tool
def process_refund(
    customer_id: str,
    order_id: str,
    amount: float,
    reason: str,
) -> dict:
    """
    Process a refund for a customer order.
    
    Args:
        customer_id: Customer identifier for verification
        order_id: The order ID to refund
        amount: Refund amount in USD (must be positive)
        reason: Reason for the refund (for audit trail)
        
    Returns:
        Refund confirmation with transaction ID and timeline
    """
    logger.info(
        f"Processing refund: customer={customer_id}, order={order_id}, "
        f"amount=${amount:.2f}, reason={reason}"
    )
    
    # Validation
    if amount <= 0:
        return {"status": "error", "error": "Refund amount must be positive"}
    if amount > 500:
        return {
            "status": "requires_approval",
            "error": f"Refund of ${amount:.2f} exceeds $500 limit — requires manager approval",
        }
    
    if _is_mock_mode():
        refund_id = f"REF-{uuid.uuid4().hex[:8].upper()}"
        processing_days = random.randint(3, 7)
        
        return {
            "status": "success",
            "refund_id": refund_id,
            "amount": amount,
            "currency": "USD",
            "order_id": order_id,
            "customer_id": customer_id,
            "reason": reason,
            "processing_time": f"{processing_days}-7 business days",
            "estimated_arrival": f"Within {processing_days} business days",
            "created_at": datetime.utcnow().isoformat(),
        }
    
    # Production: real Stripe/payment processor call
    import httpx
    response = httpx.post(
        f"{settings.PAYMENT_API_URL}/refunds",
        json={
            "customer_id": customer_id,
            "order_id": order_id,
            "amount": int(amount * 100),  # Convert to cents
            "reason": reason,
        },
        headers={"Authorization": f"Bearer {settings.PAYMENT_API_KEY}"},
        timeout=15.0,
    )
    response.raise_for_status()
    return response.json()


@tool
def get_payment_history(customer_id: str, limit: int = 10) -> dict:
    """
    Retrieve payment transaction history for a customer.
    
    Args:
        customer_id: Customer identifier
        limit: Maximum transactions to return
        
    Returns:
        List of payment transactions with amounts and statuses
    """
    logger.info(f"Payment history: customer={customer_id}, limit={limit}")
    
    if _is_mock_mode():
        transactions = []
        types = ["charge", "refund", "chargeback", "adjustment"]
        statuses = ["succeeded", "succeeded", "succeeded", "failed", "pending"]
        
        for i in range(min(limit, random.randint(2, 10))):
            amount = round(random.uniform(10, 400), 2)
            txn_type = random.choice(types)
            transactions.append({
                "transaction_id": f"TXN-{uuid.uuid4().hex[:8].upper()}",
                "type": txn_type,
                "amount": amount if txn_type == "charge" else -amount,
                "currency": "USD",
                "status": random.choice(statuses),
                "description": f"Order payment - {txn_type}",
                "created_at": datetime.utcnow().isoformat(),
            })
        
        return {"status": "success", "transactions": transactions}
    
    import httpx
    response = httpx.get(
        f"{settings.PAYMENT_API_URL}/customers/{customer_id}/transactions",
        params={"limit": limit},
        headers={"Authorization": f"Bearer {settings.PAYMENT_API_KEY}"},
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()
