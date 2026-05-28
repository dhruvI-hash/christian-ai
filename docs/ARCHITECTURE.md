# ChristianAI — Architecture Guide

## System Overview

ChristianAI is a production-grade, full-stack AI assistant specializing in Christianity, Bible study, and theological exploration.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js 14)                        │
│  ┌────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐  │
│  │ Chat UI    │ │ Model Select │ │ Denom Select │ │  RAG Panel  │  │
│  └─────┬──────┘ └──────┬───────┘ └──────┬───────┘ └──────┬──────┘  │
│        └───────────────┴────────────────┴────────────────┘          │
│                              REST API                                │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────┐
│                      Backend (FastAPI + Python)                       │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                     Safety Pipeline                              │ │
│  │  [Input Guardrail] → [Main Agent] → [Output Safety] → Response  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌──────────────┐ ┌───────────────┐ ┌──────────────────────────────┐│
│  │ LLM Router   │ │ Agent (OpenAI │ │ RAG Pipeline                 ││
│  │ ┌──────────┐ │ │ Agents SDK)   │ │ ┌──────────┐ ┌────────────┐ ││
│  │ │ OpenAI   │ │ │ ┌───────────┐ │ │ │ Chunker  │ │ Embedder   │ ││
│  │ │ Gemini   │ │ │ │ RAG Tool  │ │ │ │ (Sem +   │ │ MiniLM-L6  │ ││
│  │ │ Ollama   │ │ │ │ Verify    │ │ │ │  Slide)  │ │ (384d)     │ ││
│  │ └──────────┘ │ │ │ Image     │ │ │ └──────────┘ └────────────┘ ││
│  └──────────────┘ │ └───────────┘ │ │ ┌──────────┐ ┌────────────┐ ││
│                    └───────────────┘ │ │ BM25     │ │ Hybrid     │ ││
│  ┌──────────────┐ ┌───────────────┐ │ │ Sparse   │ │ Search     │ ││
│  │ Memory Store │ │ Safety Layer  │ │ └──────────┘ │ (RRF/Alpha)│ ││
│  │ (Dict+Redis) │ │ (3 layers)   │ │              └────────────┘ ││
│  └──────────────┘ └───────────────┘ └──────────────────────────────┘│
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐│
│  │                    Vector Databases                               ││
│  │  ┌─────────────────────┐  ┌────────────────────────────────────┐ ││
│  │  │ Qdrant Cloud        │  │ ChromaDB (Local)                   │ ││
│  │  │ Dense + Sparse      │  │ Dense + BM25 Rerank                │ ││
│  │  └─────────────────────┘  └────────────────────────────────────┘ ││
│  └──────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────┘
```

## Request Flow

### Chat Request (`POST /api/chat`)

1. **Input Guardrail** — Lightweight gpt-4o-mini agent classifies message safety
2. **If blocked** → Return graceful refusal message (no agent invocation)
3. **Main Agent** — OpenAI Agents SDK agent processes with tools:
   - `rag_tool` — Hybrid search the knowledge base
   - `scripture_verify_tool` — Verify Bible references before citing
   - `image_gen_tool` — Generate images via DALL-E 3 with safety filter
4. **Output Safety** — Scan response for unverified citations
5. **Memory** — Store conversation turn with metadata
6. **Response** — Return structured JSON with citations, safety flags

### RAG Pipeline

1. **Ingest**: File → Markdown → Chunks → Dense Embeddings + BM25 Sparse → Upsert
2. **Search**: Query → Dense Embed + Sparse Encode → Hybrid Search → RRF/Alpha Fusion → Results

## Key Design Decisions

- **Single embedding model** (all-MiniLM-L6-v2) ensures consistency
- **Hybrid search** (dense + sparse) improves recall for theological terminology
- **Guardrail-first** architecture prevents agent invocation on harmful inputs
- **Scripture verification** tool prevents hallucination of Bible references
- **Image safety** filter blocks theological inappropriate imagery before DALL-E call
