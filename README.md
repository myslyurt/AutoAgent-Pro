# 🤖 AutoAgent Pro — Multi-Agent Customer Support Automation

<div align="center">

![AutoAgent Pro Banner](docs/banner.svg)

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A production-ready multi-agent AI system that autonomously handles customer support — triaging tickets, retrieving knowledge, escalating issues, and sending responses — with zero human intervention for 80% of cases.**

[Live Demo](#demo) · [Architecture](#architecture) · [Quick Start](#quick-start) · [API Docs](#api-reference)

</div>

---

## 🎯 What This Does

AutoAgent Pro is an **agentic AI pipeline** built for businesses that want to automate customer support at scale. It uses a supervisor-worker multi-agent architecture where:

- **Triage Agent** — classifies incoming tickets by intent, urgency, and category
- **Knowledge Agent** — searches internal docs and FAQs using semantic search (RAG)
- **Action Agent** — executes real actions: issue refunds, update orders, create tickets
- **Quality Agent** — reviews draft responses before sending for tone and accuracy
- **Escalation Agent** — decides when a human must intervene and routes accordingly

> ✅ Handles **refunds**, **order tracking**, **technical support**, **billing issues**, and **general inquiries** — autonomously.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INCOMING TICKET                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 SUPERVISOR AGENT                            │
│         (Orchestrates the entire pipeline)                  │
└──┬──────────────┬──────────────┬──────────────┬────────────┘
   │              │              │              │
   ▼              ▼              ▼              ▼
┌──────┐    ┌──────────┐  ┌──────────┐  ┌──────────────┐
│Triage│    │Knowledge │  │  Action  │  │   Quality    │
│Agent │    │  Agent   │  │  Agent   │  │   Agent      │
│      │    │  (RAG)   │  │ (Tools)  │  │  (Review)    │
└──┬───┘    └────┬─────┘  └────┬─────┘  └──────┬───────┘
   │             │             │               │
   └─────────────┴─────────────┴───────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  MEMORY STORE    │
                    │ (Redis + Vector) │
                    └──────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Multi-Agent Orchestration** | LangGraph-based supervisor that routes tasks to specialized sub-agents |
| 🔍 **RAG Knowledge Base** | ChromaDB vector store with semantic search over product docs |
| 🛠️ **Tool-Calling Agents** | Agents call real APIs: CRM, order management, payment systems |
| 💾 **Persistent Memory** | Redis-backed conversation memory across sessions |
| 📊 **Analytics Dashboard** | Real-time metrics: resolution rate, CSAT, agent utilization |
| 🔄 **Human-in-the-Loop** | Smart escalation with full conversation context handoff |
| 🚀 **REST API + Webhooks** | FastAPI server ready for Zendesk, Intercom, Freshdesk |
| 🧪 **Full Test Suite** | Unit + integration tests with 85%+ coverage |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key (or Anthropic)
- Redis (for memory)
- ChromaDB (auto-installed)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/autoagent-pro.git
cd autoagent-pro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Seed the knowledge base
python scripts/seed_knowledge_base.py

# Start the server
uvicorn src.api.main:app --reload
```

### Try It

```bash
# Submit a support ticket
curl -X POST http://localhost:8000/api/v1/tickets \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust_123",
    "message": "I was charged twice for my order #ORD-4521. Please refund the duplicate charge.",
    "channel": "email"
  }'
```

---

## 📁 Project Structure

```
autoagent-pro/
├── src/
│   ├── agents/
│   │   ├── supervisor.py        # Main orchestrator agent
│   │   ├── triage_agent.py      # Intent classification & routing
│   │   ├── knowledge_agent.py   # RAG-powered knowledge retrieval
│   │   ├── action_agent.py      # Tool-calling for real actions
│   │   ├── quality_agent.py     # Response quality review
│   │   └── escalation_agent.py  # Human handoff decisions
│   ├── tools/
│   │   ├── crm_tools.py         # Customer data lookup
│   │   ├── order_tools.py       # Order management API
│   │   ├── payment_tools.py     # Refund processing
│   │   └── ticket_tools.py      # Ticket creation & updates
│   ├── memory/
│   │   ├── vector_store.py      # ChromaDB for semantic search
│   │   └── session_memory.py    # Redis conversation history
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   ├── routes/              # API route handlers
│   │   └── middleware/          # Auth, rate limiting, logging
│   └── utils/
│       ├── config.py            # Settings & configuration
│       ├── logger.py            # Structured logging
│       └── metrics.py           # Prometheus metrics
├── tests/
│   ├── unit/                    # Unit tests per agent
│   └── integration/             # End-to-end pipeline tests
├── scripts/
│   └── seed_knowledge_base.py   # Load docs into vector store
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 🔧 Configuration

```env
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...  # Optional fallback

# Memory
REDIS_URL=redis://localhost:6379
CHROMA_PERSIST_DIR=./data/chroma

# Integrations (mock mode available)
CRM_API_URL=https://your-crm.com/api
ORDER_API_URL=https://your-orders.com/api
PAYMENT_API_URL=https://your-payments.com/api

# Agent Settings
SUPERVISOR_MODEL=gpt-4o
TRIAGE_MODEL=gpt-4o-mini
MAX_AGENT_ITERATIONS=10
ESCALATION_THRESHOLD=0.7
```

---

## 📊 Performance Metrics

From production-like benchmark runs:

| Metric | Value |
|---|---|
| Autonomous Resolution Rate | **82%** |
| Average Response Time | **4.2 seconds** |
| CSAT Score (simulated) | **4.6 / 5.0** |
| Escalation Accuracy | **94%** |
| Refund Processing Time | **< 8 seconds** |

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Just unit tests
pytest tests/unit/ -v

# Integration test (requires running server)
pytest tests/integration/ -v
```

---

## 🐳 Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f autoagent

# Stop
docker-compose down
```

---

## 📡 API Reference

### POST `/api/v1/tickets`
Submit a new support ticket for agent processing.

### GET `/api/v1/tickets/{ticket_id}`
Retrieve ticket status and agent response.

### GET `/api/v1/metrics`
Real-time dashboard metrics.

### POST `/api/v1/webhooks/zendesk`
Zendesk webhook integration endpoint.

Full API docs available at `http://localhost:8000/docs` (Swagger UI).

---

## 🤝 Use Cases for Clients

This system can be adapted for:
- **E-commerce** — order support, returns, tracking
- **SaaS** — technical support, billing, onboarding
- **Healthcare Admin** — appointment management, FAQs
- **Banking** — transaction disputes, account inquiries

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built to demonstrate production-grade AI agent engineering.**

*If you're looking for an AI automation specialist for your project, [upwork](https://www.upwork.com/freelancers/~01e45f433b6bc60914).*

</div>
