# RAG Document Bot + LLM Cost & Performance Monitoring

> **College Project**: A dual-purpose AI system combining **grounded document retrieval (RAG)** with real-time **telemetry observability** (latency, token ledger, per-model cost accounting, and quality tracking).

---

## Architecture Overview

```
01 · INPUT
User asks a question
"What does CGTMSE stand for and who benefits from it?"
       ↓ query
02 · RETRIEVAL
Vector database search (ChromaDB + SentenceTransformers)
Question embedded (all-MiniLM-L6-v2) → compared against pre-indexed 500-char chunks (PDFs, docs) → top matching chunks returned
       ↓ relevant chunks + citations
03 · GENERATION
AI model answers (via local OmniRoute at http://localhost:20128/v1)
Model receives question + retrieved context, generates a grounded answer with citations
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ MONITORING LAYER — RUNS ALONGSIDE EVERY REQUEST                 │
│ 04A · LOGGING: Every call logged (Model, tokens in/out, latency,│
│       cost in USD, success/failure) stored in SQLite db         │
│ 04B · DASHBOARD: Cost / performance view (KPIs, model comparison│
│       table, cost-saving recommendations, request audit log)    │
└─────────────────────────────────────────────────────────────────┘
       ↓
05 · OUTPUT
Answer delivered to user
Grounded, cited response — while cost/performance data quietly accumulates in the background
```

---

## Key Features

1. **Document Ingestion & Text Extraction**:
   - Ingests PDF documents using PyMuPDF (`fitz`).
   - Drag-and-drop web dropzone or automated directory indexing.
   - Text chunking with sliding character window (`chunk_size=500`, `overlap=100`).

2. **ChromaDB Vector Retrieval**:
   - Local persistent vector storage in `chroma_db/`.
   - Embeddings powered by `sentence-transformers/all-MiniLM-L6-v2`.
   - Cosine similarity matching returning top matching document chunks with page numbers and source metadata.

3. **Grounded AI Answers & Citations**:
   - Strictly hallucination-resistant: prompts instruct the LLM to answer using only verified document context.
   - Clear source citations (e.g., `[Source 1: filename.pdf, Page 1]`) with matching percentage scores.

4. **Local OmniRoute Integration (`http://localhost:20128/v1`)**:
   - Compatible with OpenAI client protocol.
   - Live model switching:
     - `Claude Sonnet 4.6` (Fast & Direct)
     - `Claude Opus 4.6` (Deep Reasoning)
     - `Auto / Fast Routing`
     - `Auto / Best Chat`
     - `Auto / Best Coding`
     - `Auto / Best Free`

5. **Full Observability & Cost Monitoring**:
   - **Request Counter**: Total calls, success rates, failed attempts.
   - **Token Ledger**: Prompt tokens, completion tokens, and total tokens per query.
   - **Cost Calculator**: Real-time USD computation based on current model pricing tiers.
   - **Latency Meter**: Sub-second roundtrip timing in milliseconds.
   - **Model Comparison Table**: Side-by-side analysis of cost vs speed across different LLMs.
   - **Optimization Insights**: Automatic recommendations on model selection and token density.
   - **Audit Trail**: SQLite-backed request history with question snippets and status badges.

---

## Project Structure

```
document-rag/
├── documents/                # Stored and uploaded PDF documents
│   └── rag_test_cgtmse_sample.pdf
├── chroma_db/                # Local persistent Chroma vector database
├── logs/                     # SQLite telemetry database (telemetry.db)
├── src/
│   ├── main.py               # FastAPI backend & API routes
│   ├── rag_pipeline.py       # PDF parsing, chunking, embedding, vector retrieval
│   ├── omniroute_client.py   # OmniRoute LLM caller, grounded prompt builder, pricing
│   └── telemetry.py          # SQLite call logger and analytics aggregator
├── static/
│   ├── index.html            # Dual-view UI: Document Chat + Telemetry Dashboard
│   ├── style.css             # Tailored dark-mode CSS with glassmorphism
│   └── app.js                # Interactive frontend controller & live polling
├── tests/
│   └── test_rag.py           # Automated unit & integration tests
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── run.sh                    # One-command launch script
├── build_vector_db.py        # Original baseline database builder (preserved)
├── chunk_pdf.py              # Original baseline chunking script (preserved)
├── read_pdf.py               # Original baseline PDF reader (preserved)
└── test_retrieval.py         # Original baseline retrieval test (preserved)
```

---

## Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (Current environment: Python 3.14)
- OmniRoute running locally on port 20128 (`omniroute serve --daemon --no-open`)

### 2. Environment Setup
The project uses `.env` for configuration. Copy `.env.example` to `.env` if not already present:
```bash
cp .env.example .env
```
Ensure `OMNIROUTE_API_BASE=http://localhost:20128/v1` and your API key is configured.

### 3. Launch the Application
Simply execute the launch script:
```bash
./run.sh
```
Or run directly with uvicorn:
```bash
source .venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## Running the Automated Tests

Run the test suite to verify vector retrieval, chunking formulas, telemetry calculations, and FastAPI endpoints:
```bash
source .venv/bin/activate
python -m unittest tests/test_rag.py
```

All 5 core tests will execute:
- `test_01_existing_retrieval`: Checks ChromaDB similarity search on sample documents.
- `test_02_chunking_logic`: Checks 500-char sliding window and 100-char overlap.
- `test_03_telemetry_logging_and_kpis`: Checks SQLite logging and metrics aggregation.
- `test_04_cost_calculator`: Checks token pricing across model tiers.
- `test_05_fastapi_app_routes`: Checks `/api/health`, `/api/models`, `/api/documents`, and `/api/metrics`.

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the web interface |
| `POST` | `/api/chat` | RAG query: retrieves chunks, queries LLM, logs telemetry |
| `POST` | `/api/upload` | Uploads and indexes a new PDF into ChromaDB |
| `GET` | `/api/models` | Returns available LLM models list |
| `GET` | `/api/documents` | Returns indexed documents and chunk statistics |
| `GET` | `/api/metrics` | Returns telemetry KPIs, model comparison, and audit logs |
| `GET` | `/api/health` | Health check for ChromaDB, OmniRoute, and Embedding model |
