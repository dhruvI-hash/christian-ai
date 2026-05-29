# ChristianAI — Christianity-Focused AI Assistant

A production-grade, full-stack AI assistant specializing in Christianity, Bible study, and theological exploration. Built with prompt engineering excellence, hallucination prevention, multimodal workflows, and AI safety at its core.

---

## 🏗️ Architecture

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 14 (App Router) + React 18 + TypeScript + TailwindCSS |
| **Backend** | Python 3.11 + FastAPI |
| **Agent Framework** | LangChain agents (`create_agent`) — single tool-calling agent |
| **LLM Providers** | OpenAI GPT-4o, Google Gemini (native), Ollama (local), OpenRouter |
| **Embeddings** | `sentence-transformers` — `all-MiniLM-L6-v2` (384d) |
| **Sparse Search** | BM25 via `rank_bm25` |
| **Vector DBs** | Qdrant Cloud + ChromaDB |
| **Image Gen** | DALL-E 3 via OpenAI SDK |

## ✨ Key Features

- **Hybrid RAG Search**: Dense (sentence-transformers) + Sparse (BM25) with Reciprocal Rank Fusion
- **Scripture Verification**: Every Bible verse citation is verified before being presented
- **3-Layer Safety**: Input guardrail agent → Output verification → Image safety filter
- **Multi-LLM Routing**: Switch between OpenAI, Gemini, and Ollama at runtime
- **Denomination Awareness**: Adapts responses for Catholic, Protestant, Orthodox, Evangelical traditions
- **Hallucination Prevention**: XML system prompt with strict anti-hallucination rules
- **Multimodal**: Christian-themed image generation with theological safety guardrails

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose (optional)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example ../.env
# Edit .env with your API keys
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Docker (Full Stack)

```bash
cp .env.example .env
# Edit .env with your API keys
docker-compose up --build
```

## 🔑 Environment Variables

Copy `.env.example` to `.env` and fill in your API keys. See the file for all available options.

## 📁 Project Structure

```
christianity-ai/
├── backend/          # Python / FastAPI
│   ├── custom_agents/ # LangChain single-agent orchestration + guardrail parsing
│   ├── tools/        # LangChain tools (RAG, scripture verify, image gen)
│   ├── rag/          # Full RAG pipeline (embed, chunk, search)
│   ├── vectordb/     # Qdrant + ChromaDB clients
│   ├── llm/          # Multi-provider LLM router
│   ├── prompts/      # XML system prompts
│   ├── moderation/   # Safety layers
│   ├── memory/       # Conversation store
│   ├── routes/       # FastAPI endpoints
│   └── eval/         # Evaluation dataset + runner
├── frontend/         # Next.js 14 / React
│   └── src/
│       ├── app/      # App Router pages
│       ├── components/
│       ├── hooks/
│       └── api/      # Backend API client
└── docs/             # Architecture + eval docs
```

## 📖 Documentation

- [Architecture Guide](docs/ARCHITECTURE.md)
- [Evaluation Results](docs/EVAL_RESULTS.md)

## 📜 License

MIT
