"""
test_agents.py — Unit Tests for Agent Components

Tests each agent in isolation with mocked LLM calls.
Run with: pytest tests/ -v
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─── Triage Agent Tests ────────────────────────────────────────────────────────

class TestTriageAgent:
    """Tests for the intent classification agent."""

    @pytest.fixture
    def mock_llm_response(self):
        """Factory for creating mock LLM responses."""
        def _make_response(data: dict):
            mock = MagicMock()
            mock.content = json.dumps(data)
            return mock
        return _make_response

    @pytest.mark.asyncio
    async def test_classifies_refund_intent(self, mock_llm_response):
        """Test that refund-related messages are correctly classified."""
        from src.agents.triage_agent import TriageAgent

        expected = {
            "intent": "refund",
            "urgency": "high",
            "category": "duplicate_charge",
            "sentiment": -0.6,
            "confidence": 0.95,
            "reasoning": "Customer reports duplicate charge"
        }

        with patch("src.agents.triage_agent.ChatOpenAI") as MockLLM:
            mock_instance = MockLLM.return_value
            mock_instance.ainvoke = AsyncMock(return_value=mock_llm_response(expected))

            agent = TriageAgent()
            result = await agent.classify(
                message="I was charged twice for order #1234!",
                customer_id="cust_001",
            )

        assert result["intent"] == "refund"
        assert result["urgency"] == "high"
        assert result["sentiment"] < 0

    @pytest.mark.asyncio
    async def test_classifies_order_status_intent(self, mock_llm_response):
        """Test order tracking classification."""
        from src.agents.triage_agent import TriageAgent

        expected = {
            "intent": "order_status",
            "urgency": "low",
            "category": "delivery_inquiry",
            "sentiment": 0.1,
            "confidence": 0.92,
            "reasoning": "Customer asking about order delivery"
        }

        with patch("src.agents.triage_agent.ChatOpenAI") as MockLLM:
            mock_instance = MockLLM.return_value
            mock_instance.ainvoke = AsyncMock(return_value=mock_llm_response(expected))

            agent = TriageAgent()
            result = await agent.classify(
                message="Where is my order? It's been 5 days.",
                customer_id="cust_002",
            )

        assert result["intent"] == "order_status"
        assert result["urgency"] == "low"

    @pytest.mark.asyncio
    async def test_fallback_on_json_error(self):
        """Test keyword fallback when LLM returns invalid JSON."""
        from src.agents.triage_agent import TriageAgent

        bad_response = MagicMock()
        bad_response.content = "This is not JSON at all"

        with patch("src.agents.triage_agent.ChatOpenAI") as MockLLM:
            mock_instance = MockLLM.return_value
            mock_instance.ainvoke = AsyncMock(return_value=bad_response)

            agent = TriageAgent()
            result = await agent.classify(
                message="I need a refund for my purchase",
                customer_id="cust_003",
            )

        # Should fall back to keyword detection
        assert result["intent"] == "refund"
        assert result["confidence"] == 0.3  # Low confidence for fallback

    @pytest.mark.asyncio
    async def test_validates_intent_field(self, mock_llm_response):
        """Test that invalid intent values are caught by Pydantic validation."""
        from src.agents.triage_agent import TriageAgent

        bad_data = {
            "intent": "INVALID_INTENT",
            "urgency": "medium",
            "category": "test",
            "sentiment": 0.0,
            "confidence": 0.8,
            "reasoning": "test"
        }

        with patch("src.agents.triage_agent.ChatOpenAI") as MockLLM:
            mock_instance = MockLLM.return_value
            mock_instance.ainvoke = AsyncMock(return_value=mock_llm_response(bad_data))

            agent = TriageAgent()
            result = await agent.classify(
                message="help me",
                customer_id="cust_004",
            )

        # Should fall back to defaults on validation error
        assert result["intent"] in {"refund", "order_status", "technical", "billing", "general", "account_action"}


# ─── Quality Agent Tests ───────────────────────────────────────────────────────

class TestQualityAgent:
    """Tests for the response quality review agent."""

    @pytest.mark.asyncio
    async def test_approves_high_quality_response(self):
        """Test that high-quality responses get approved."""
        from src.agents.quality_agent import QualityAgent

        high_quality_result = {
            "score": 0.92,
            "approved": True,
            "accuracy_score": 0.95,
            "completeness_score": 0.90,
            "tone_score": 0.88,
            "clarity_score": 0.94,
            "issues": [],
            "feedback": "Response is clear, accurate, and empathetic."
        }

        mock_response = MagicMock()
        mock_response.content = json.dumps(high_quality_result)

        with patch("src.agents.quality_agent.ChatOpenAI") as MockLLM:
            mock_instance = MockLLM.return_value
            mock_instance.ainvoke = AsyncMock(return_value=mock_response)

            agent = QualityAgent()
            result = await agent.review(
                original_message="I need a refund",
                draft_response="We've processed your refund of $50.00. You'll see it in 3-5 business days.",
                intent="refund",
            )

        assert result["approved"] is True
        assert result["score"] >= 0.75

    @pytest.mark.asyncio
    async def test_rejects_low_quality_response(self):
        """Test that low-quality responses are rejected."""
        from src.agents.quality_agent import QualityAgent

        low_quality_result = {
            "score": 0.45,
            "approved": False,
            "accuracy_score": 0.40,
            "completeness_score": 0.50,
            "tone_score": 0.45,
            "clarity_score": 0.45,
            "issues": ["Doesn't address the refund", "Tone is dismissive"],
            "feedback": "Response does not address the customer's refund request."
        }

        mock_response = MagicMock()
        mock_response.content = json.dumps(low_quality_result)

        with patch("src.agents.quality_agent.ChatOpenAI") as MockLLM:
            mock_instance = MockLLM.return_value
            mock_instance.ainvoke = AsyncMock(return_value=mock_response)

            agent = QualityAgent()
            result = await agent.review(
                original_message="I need a refund",
                draft_response="Please contact us for help.",  # Bad response
                intent="refund",
            )

        assert result["approved"] is False
        assert result["score"] < 0.75
        assert len(result["issues"]) > 0


# ─── Memory Tests ──────────────────────────────────────────────────────────────

class TestSessionMemory:
    """Tests for conversation history management."""

    @pytest.mark.asyncio
    async def test_add_and_retrieve_history(self):
        """Test basic memory read/write cycle."""
        from src.memory.session_memory import SessionMemory

        memory = SessionMemory()  # Will use in-memory fallback

        customer_id = "test_customer_001"
        await memory.add_turn(
            customer_id=customer_id,
            user_message="Hello, I need help",
            agent_response="Of course! How can I help you today?",
        )

        history = await memory.get_history(customer_id)
        assert len(history) == 1
        assert history[0]["user"] == "Hello, I need help"
        assert "timestamp" in history[0]

    @pytest.mark.asyncio
    async def test_history_trimming(self):
        """Test that history is trimmed to MAX_HISTORY_TURNS."""
        from src.memory.session_memory import SessionMemory

        memory = SessionMemory()
        customer_id = "test_customer_002"
        memory.MAX_HISTORY_TURNS = 3  # Low limit for testing

        for i in range(5):
            await memory.add_turn(
                customer_id=customer_id,
                user_message=f"Message {i}",
                agent_response=f"Response {i}",
            )

        history = await memory.get_history(customer_id)
        assert len(history) <= 3  # Should be trimmed

    @pytest.mark.asyncio
    async def test_clear_history(self):
        """Test that clear_history removes all data."""
        from src.memory.session_memory import SessionMemory

        memory = SessionMemory()
        customer_id = "test_customer_003"

        await memory.add_turn(customer_id, "test", "response")
        await memory.clear_history(customer_id)

        history = await memory.get_history(customer_id)
        assert history == []


# ─── Tool Tests ────────────────────────────────────────────────────────────────

class TestTools:
    """Tests for external API integration tools."""

    def test_lookup_customer_mock_mode(self):
        """Test CRM lookup returns valid mock data."""
        from src.tools.crm_tools import lookup_customer

        result = lookup_customer.invoke({"customer_id": "cust_test_001"})

        assert result["status"] == "found"
        assert "customer" in result
        assert result["customer"]["customer_id"] == "cust_test_001"
        assert "tier" in result["customer"]
        assert "email" in result["customer"]

    def test_process_refund_mock_mode(self):
        """Test refund processing in mock mode."""
        from src.tools.payment_tools import process_refund

        result = process_refund.invoke({
            "customer_id": "cust_001",
            "order_id": "ORD-1234",
            "amount": 49.99,
            "reason": "Duplicate charge",
        })

        assert result["status"] == "success"
        assert "refund_id" in result
        assert result["amount"] == 49.99

    def test_refund_amount_validation(self):
        """Test that refunds over $500 require approval."""
        from src.tools.payment_tools import process_refund

        result = process_refund.invoke({
            "customer_id": "cust_001",
            "order_id": "ORD-5678",
            "amount": 600.00,
            "reason": "Full return",
        })

        assert result["status"] == "requires_approval"

    def test_negative_refund_rejected(self):
        """Test that negative refund amounts are rejected."""
        from src.tools.payment_tools import process_refund

        result = process_refund.invoke({
            "customer_id": "cust_001",
            "order_id": "ORD-0001",
            "amount": -10.00,
            "reason": "test",
        })

        assert result["status"] == "error"

    def test_order_lookup_mock_mode(self):
        """Test order lookup returns valid mock data."""
        from src.tools.order_tools import lookup_order

        result = lookup_order.invoke({"order_id": "ORD-9999"})

        assert result["status"] == "found"
        assert result["order"]["order_id"] == "ORD-9999"
        assert "status" in result["order"]
        assert "total" in result["order"]


# ─── Integration Tests ─────────────────────────────────────────────────────────

class TestAPIEndpoints:
    """Integration tests for the FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "AutoAgent Pro"
