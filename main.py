"""
main.py — FastAPI Application

REST API server for the AutoAgent Pro multi-agent system.
Provides endpoints for:
- Submitting support tickets
- Checking ticket status
- Real-time metrics dashboard
- Webhook integrations (Zendesk, Intercom)
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from src.agents.supervisor import SupervisorAgent
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.metrics import get_metrics_summary

logger = get_logger(__name__)

# Global supervisor instance
supervisor: SupervisorAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources on startup/shutdown."""
    global supervisor
    logger.info("Starting AutoAgent Pro...")
    supervisor = SupervisorAgent()
    logger.info("✅ SupervisorAgent initialized")
    yield
    logger.info("Shutting down AutoAgent Pro...")


# ─── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AutoAgent Pro — AI Support Automation API",
    description="""
    Multi-agent AI system for autonomous customer support automation.
    
    Handles: refunds, order tracking, technical support, billing inquiries.
    Escalates: complex cases, critical urgency, quality failures.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)


# ─── Request / Response Models ─────────────────────────────────────────────────

class TicketRequest(BaseModel):
    """Incoming support ticket request."""
    customer_id: str = Field(..., description="Unique customer identifier", example="cust_abc123")
    message: str = Field(
        ...,
        description="Customer support message",
        min_length=5,
        max_length=5000,
        example="I was charged twice for order #ORD-4521. Please refund the duplicate charge."
    )
    channel: Literal["email", "chat", "api", "webhook"] = Field(
        default="api",
        description="Origin channel"
    )
    ticket_id: str | None = Field(
        default=None,
        description="Optional ticket ID (auto-generated if not provided)"
    )


class TicketResponse(BaseModel):
    """Processed ticket response."""
    ticket_id: str
    status: Literal["resolved", "escalated", "processing"]
    response: str
    metadata: dict
    processed_at: str


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    version: str
    timestamp: str


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """API root — basic info."""
    return {
        "name": "AutoAgent Pro",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/api/v1/tickets", response_model=TicketResponse, tags=["Tickets"])
async def process_ticket(request: TicketRequest):
    """
    Submit a support ticket for AI agent processing.
    
    The multi-agent pipeline will:
    1. Classify intent, urgency, and sentiment (Triage Agent)
    2. Retrieve relevant knowledge (Knowledge Agent)
    3. Execute necessary actions — refunds, lookups (Action Agent)
    4. Review the draft response (Quality Agent)
    5. Return a finalized response or escalate to human
    """
    if not supervisor:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    ticket_id = request.ticket_id or f"TKT-{uuid.uuid4().hex[:8].upper()}"

    logger.info(
        f"Received ticket {ticket_id}: customer={request.customer_id}, "
        f"channel={request.channel}, msg_len={len(request.message)}"
    )

    try:
        result = await supervisor.process_ticket(
            ticket_id=ticket_id,
            customer_id=request.customer_id,
            message=request.message,
            channel=request.channel,
        )

        return TicketResponse(
            ticket_id=result["ticket_id"],
            status=result["status"],
            response=result["response"],
            metadata=result["metadata"],
            processed_at=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.exception(f"Error processing ticket {ticket_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Agent processing error: {str(e)}")


@app.get("/api/v1/metrics", tags=["Analytics"])
async def get_metrics():
    """
    Real-time system metrics dashboard.
    
    Returns resolution rates, quality scores, escalation rates,
    and per-intent breakdown.
    """
    return get_metrics_summary()


@app.post("/api/v1/webhooks/zendesk", tags=["Webhooks"])
async def zendesk_webhook(payload: dict, background_tasks: BackgroundTasks):
    """
    Zendesk webhook endpoint.
    
    Receives new ticket events from Zendesk and processes them
    through the AI agent pipeline asynchronously.
    """
    # Extract ticket data from Zendesk payload format
    zendesk_ticket = payload.get("ticket", {})
    
    if not zendesk_ticket:
        raise HTTPException(status_code=400, detail="Invalid Zendesk payload")

    request = TicketRequest(
        customer_id=str(zendesk_ticket.get("requester_id", "unknown")),
        message=zendesk_ticket.get("description", ""),
        channel="webhook",
        ticket_id=f"ZD-{zendesk_ticket.get('id', uuid.uuid4().hex[:8])}",
    )

    # Process asynchronously to return 200 quickly to Zendesk
    background_tasks.add_task(process_ticket_background, request)

    return {"status": "accepted", "ticket_id": request.ticket_id}


async def process_ticket_background(request: TicketRequest):
    """Background task for webhook processing."""
    try:
        await process_ticket(request)
    except Exception as e:
        logger.exception(f"Background ticket processing failed: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
