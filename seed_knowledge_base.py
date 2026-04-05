"""
seed_knowledge_base.py — Populate the ChromaDB vector store with sample docs

Run this once before starting the server:
    python scripts/seed_knowledge_base.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.knowledge_agent import KnowledgeAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Sample Knowledge Base Documents ──────────────────────────────────────────

DOCUMENTS = {
    "policies": [
        {
            "id": "refund_policy_001",
            "text": """Refund Policy: Customers may request a full refund within 30 days of purchase. 
Refunds are processed within 3-7 business days and returned to the original payment method. 
For digital products, refunds are available within 14 days if the product has not been downloaded more than 3 times. 
Shipping costs are non-refundable unless the return is due to our error.""",
            "metadata": {"source": "refund_policy", "version": "2.1"},
        },
        {
            "id": "refund_policy_002",
            "text": """Duplicate Charge Policy: If a customer is charged twice for the same order, 
we will refund the duplicate charge immediately upon verification. 
The refund will appear on their statement within 5-10 business days depending on their bank. 
Customers should provide the order number and last 4 digits of their payment card for verification.""",
            "metadata": {"source": "refund_policy", "section": "duplicate_charges"},
        },
        {
            "id": "cancellation_policy",
            "text": """Order Cancellation: Orders can be cancelled within 1 hour of placement if they have not 
yet entered the fulfillment process. Once an order is shipped, it cannot be cancelled. 
Customers may still return the item upon delivery for a full refund (excluding shipping). 
To cancel, use the order management portal or contact support immediately.""",
            "metadata": {"source": "cancellation_policy"},
        },
    ],
    "orders": [
        {
            "id": "shipping_times",
            "text": """Standard Shipping: 5-7 business days within the continental US. 
Expedited Shipping: 2-3 business days. Overnight Shipping: Next business day if ordered before 2 PM EST. 
International orders: 10-21 business days. Tracking numbers are emailed within 24 hours of shipment. 
Weekends and holidays are not counted as business days.""",
            "metadata": {"source": "shipping_faq"},
        },
        {
            "id": "missing_package",
            "text": """Missing Package Policy: If a package shows as delivered but has not been received, 
customers should: (1) Check with neighbors and building management, (2) Wait 48 hours as packages 
can sometimes be marked delivered early, (3) Contact us if still missing after 48 hours. 
We will file a carrier claim and either reship the order or issue a full refund. 
Claims must be filed within 30 days of the delivery date.""",
            "metadata": {"source": "shipping_faq", "section": "missing_packages"},
        },
        {
            "id": "tracking_info",
            "text": """Order Tracking: Use the tracking number provided in your shipment confirmation email 
on the carrier's website (UPS, FedEx, USPS, or DHL). Tracking updates may take up to 24 hours 
after shipment to appear. If your tracking shows no movement for 5+ business days, 
contact our support team for investigation.""",
            "metadata": {"source": "shipping_faq", "section": "tracking"},
        },
    ],
    "billing": [
        {
            "id": "subscription_billing",
            "text": """Subscription Billing: Subscriptions are billed on the same date each month or year. 
You can update your payment method at any time in Account Settings → Billing. 
Invoices are sent to the email on file within 24 hours of each charge. 
Failed payments are retried 3 times over 5 days before the subscription is paused.""",
            "metadata": {"source": "billing_faq"},
        },
        {
            "id": "invoice_questions",
            "text": """Invoice Requests: Business customers can request itemized invoices for any charge. 
Invoices can be downloaded from Account Settings → Billing History. 
For custom invoice formats or purchase orders, contact billing@company.com. 
We support NET-30 payment terms for verified enterprise accounts.""",
            "metadata": {"source": "billing_faq", "section": "invoices"},
        },
    ],
    "technical_docs": [
        {
            "id": "login_issues",
            "text": """Login Troubleshooting: If you cannot log in, try these steps: 
(1) Reset your password using 'Forgot Password' on the login page. 
(2) Clear browser cache and cookies. 
(3) Try a different browser or incognito mode. 
(4) Check if your account email is correct. 
(5) Ensure your account is not locked (3 failed attempts causes a 30-minute lockout). 
Contact support if issues persist after trying all steps.""",
            "metadata": {"source": "tech_support", "category": "authentication"},
        },
        {
            "id": "app_errors",
            "text": """Common App Errors: Error 502/503 indicates a temporary server issue — 
wait 5 minutes and try again. Error 401 means your session has expired — log out and back in. 
Error 429 means you've hit the rate limit — wait 60 seconds. For persistent errors, 
clear the app cache (Settings → Clear Cache) or reinstall the app. 
Report bugs with a screenshot to support@company.com.""",
            "metadata": {"source": "tech_support", "category": "errors"},
        },
    ],
    "general_faq": [
        {
            "id": "contact_support",
            "text": """Contacting Support: Our support team is available Monday-Friday 9 AM to 6 PM EST. 
Email: support@company.com (response within 24 hours). 
Live chat: Available on our website during business hours. 
Phone: 1-800-XXX-XXXX (for urgent issues). 
For fastest response, include your order number and account email in your message.""",
            "metadata": {"source": "general_faq"},
        },
        {
            "id": "account_management",
            "text": """Account Management: You can update your email, password, shipping addresses, 
and payment methods in Account Settings. To delete your account, go to Settings → Privacy → Delete Account. 
Account deletion is permanent and cannot be undone. Your data will be removed within 30 days 
in compliance with GDPR/CCPA regulations.""",
            "metadata": {"source": "general_faq", "section": "account"},
        },
    ],
}


async def seed():
    """Load all documents into ChromaDB collections."""
    agent = KnowledgeAgent()

    total = 0
    for collection_name, docs in DOCUMENTS.items():
        logger.info(f"Seeding collection: {collection_name} ({len(docs)} docs)")
        for doc in docs:
            await agent.add_document(
                text=doc["text"],
                collection_name=collection_name,
                metadata=doc["metadata"],
                doc_id=doc["id"],
            )
            total += 1

    stats = agent.get_collection_stats()
    logger.info(f"✅ Seeded {total} documents across {len(DOCUMENTS)} collections")
    logger.info(f"Collection stats: {stats}")
    return total


if __name__ == "__main__":
    logger.info("Starting knowledge base seeding...")
    count = asyncio.run(seed())
    print(f"\n✅ Successfully seeded {count} documents into ChromaDB!")
    print("You can now start the server: uvicorn src.api.main:app --reload")
