"""
FastAPI Server for RAG Document Bot & LLM Performance/Cost Monitoring
Exposes REST endpoints for:
- PDF upload and vector indexing
- Grounded RAG chat with citations
- Telemetry & performance metrics dashboard
- Available models list and service health

Works both locally (uvicorn) and on Vercel (via api/index.py).
"""

import os
import sys
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Detect Vercel environment
IS_VERCEL = bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENV"))

# Resolve project root (works for both local and Vercel)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration — use /tmp on Vercel (only writable directory)
if IS_VERCEL:
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/tmp/chroma_db")
    TELEMETRY_DB_PATH = os.getenv("TELEMETRY_DB_PATH", "/tmp/telemetry.db")
    DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "/tmp/documents")
else:
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(PROJECT_ROOT, "chroma_db"))
    TELEMETRY_DB_PATH = os.getenv("TELEMETRY_DB_PATH", os.path.join(PROJECT_ROOT, "logs", "telemetry.db"))
    DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", os.path.join(PROJECT_ROOT, "documents"))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "no-think/agy/claude-sonnet-4-6")

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TELEMETRY_DB_PATH) or ".", exist_ok=True)

# Add project root to sys.path if not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag_pipeline import RAGPipeline
from src.omniroute_client import OmniRouteClient
from src.telemetry import TelemetryTracker

# Initialize Core Services
print("[Server] Initializing RAG pipeline...")
rag = RAGPipeline(
    db_path=CHROMA_DB_PATH,
    collection_name=COLLECTION_NAME,
    embedding_model_name=EMBEDDING_MODEL,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

print("[Server] Initializing OmniRoute client...")
llm_client = OmniRouteClient(default_model=DEFAULT_MODEL)

print("[Server] Initializing Telemetry tracker...")
telemetry = TelemetryTracker(db_path=TELEMETRY_DB_PATH)

app = FastAPI(
    title="RAG Document Bot & Monitoring",
    description="Grounded AI Question Answering with Live Latency, Token & Cost Observability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = None
    top_k: Optional[int] = 3

# Endpoints
@app.get("/api/health")
def get_health():
    """Health check for vector DB, OmniRoute, and telemetry."""
    omni_check = llm_client.check_connection()
    rag_stats = rag.get_stats()
    return {
        "status": "healthy" if omni_check.get("connected") else "degraded",
        "vector_db": {
            "connected": True,
            "collection": rag_stats["collection_name"],
            "total_chunks": rag_stats["total_chunks"]
        },
        "omniroute": omni_check,
        "default_model": DEFAULT_MODEL,
        "environment": "vercel" if IS_VERCEL else "local"
    }

@app.get("/api/models")
def get_models():
    """List available LLM models with descriptions."""
    catalog = [
        {"id": "no-think/agy/claude-sonnet-4-6", "name": "Claude Sonnet 4.6 (Fast & Direct)", "tier": "Balanced", "recommended": True},
        {"id": "no-think/agy/claude-opus-4-6-thinking", "name": "Claude Opus 4.6 (Deep Reasoning)", "tier": "Premium", "recommended": False},
        {"id": "auto/fast", "name": "Auto / Fast Routing", "tier": "Fast", "recommended": False},
        {"id": "auto/best-chat", "name": "Auto / Best Chat", "tier": "Balanced", "recommended": False},
        {"id": "auto/best-coding", "name": "Auto / Best Coding", "tier": "Premium", "recommended": False},
        {"id": "auto/best-free", "name": "Auto / Best Free", "tier": "Free", "recommended": False}
    ]
    return {
        "default": DEFAULT_MODEL,
        "models": catalog
    }

@app.get("/api/documents")
def get_documents():
    """List indexed documents, chunk totals, and directory files."""
    stats = rag.get_stats()
    physical_files = []
    if os.path.exists(DOCUMENTS_DIR):
        for f in os.listdir(DOCUMENTS_DIR):
            if f.lower().endswith(".pdf"):
                f_path = os.path.join(DOCUMENTS_DIR, f)
                physical_files.append({
                    "filename": f,
                    "size_bytes": os.path.getsize(f_path),
                    "indexed": f in stats["distinct_sources"] or len(stats["distinct_sources"]) > 0
                })
    return {
        "total_chunks": stats["total_chunks"],
        "collection_name": stats["collection_name"],
        "distinct_sources": stats["distinct_sources"],
        "files": physical_files
    }

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF, saves to documents/, and indexes into ChromaDB."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = os.path.join(DOCUMENTS_DIR, file.filename)
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    try:
        result = rag.ingest_pdf(save_path, filename=file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {str(e)}")

@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    RAG Query Flow:
    1. Embed query & retrieve top-k chunks from ChromaDB.
    2. Construct prompt with citations.
    3. Call OmniRoute LLM.
    4. Log latency, token usage, cost to telemetry.
    5. Return grounded response with citations & telemetry metadata.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    chosen_model = request.model or DEFAULT_MODEL
    top_k = request.top_k or 3

    # Step 1: Retrieval
    retrieved_chunks = rag.retrieve(query, n_results=top_k)

    # Step 2 & 3: Generation & Grounding
    gen_result = llm_client.generate_grounded_answer(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=chosen_model
    )

    # Step 4: Telemetry Logging
    tokens = gen_result.get("tokens", {"prompt": 0, "completion": 0, "total": 0})
    telemetry.log_call(
        query=query,
        model_requested=chosen_model,
        model_resolved=gen_result.get("model_resolved"),
        prompt_tokens=tokens["prompt"],
        completion_tokens=tokens["completion"],
        total_tokens=tokens["total"],
        latency_ms=gen_result.get("latency_ms", 0.0),
        estimated_cost_usd=gen_result.get("cost_usd", 0.0),
        chunks_retrieved=len(retrieved_chunks),
        success=gen_result.get("success", False),
        error_message=gen_result.get("error"),
        response_snippet=gen_result.get("answer", "")[:200]
    )

    # Step 5: Deliver Output
    return {
        "query": query,
        "answer": gen_result.get("answer"),
        "citations": gen_result.get("citations", []),
        "model_requested": chosen_model,
        "model_resolved": gen_result.get("model_resolved"),
        "telemetry": {
            "latency_ms": gen_result.get("latency_ms"),
            "prompt_tokens": tokens["prompt"],
            "completion_tokens": tokens["completion"],
            "total_tokens": tokens["total"],
            "estimated_cost_usd": gen_result.get("cost_usd")
        },
        "success": gen_result.get("success", False),
        "error": gen_result.get("error")
    }

@app.get("/api/metrics")
def get_metrics():
    """Returns aggregated performance, cost, and latency analytics."""
    return telemetry.get_metrics_summary()

# Static Files & UI Mounting
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.api_route("/", methods=["GET", "HEAD"])
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "RAG Document Bot Backend API Running. Static UI not yet created."})
