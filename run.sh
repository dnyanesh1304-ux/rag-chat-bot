#!/bin/bash
# ==============================================================
# RAG Document Bot & LLM Monitoring Launch Script
# ==============================================================

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================================="
echo " Starting RAG Document Bot + LLM Telemetry Platform"
echo "=========================================================="

# 1. Check OmniRoute
if curl -s http://localhost:20128/v1/models > /dev/null 2>&1; then
    echo "✔ OmniRoute server is active on http://localhost:20128/v1"
else
    echo "⏳ Starting local OmniRoute daemon..."
    if command -v omniroute > /dev/null 2>&1; then
        omniroute serve --daemon --no-open
    elif [ -f "/Users/dnyaneshwarraut/.npm-global/bin/omniroute" ]; then
        /Users/dnyaneshwarraut/.npm-global/bin/omniroute serve --daemon --no-open
    else
        echo "⚠ OmniRoute CLI not found in PATH. Ensure OmniRoute is running on port 20128."
    fi
fi

# 2. Check Virtual Environment
if [ -d ".venv" ]; then
    echo "✔ Activating Python virtual environment..."
    source .venv/bin/activate
else
    echo "❌ Virtual environment .venv not found. Please create one with python3 -m venv .venv"
    exit 1
fi

# 3. Check .env configuration
if [ ! -f ".env" ]; then
    echo "⚠ .env not found, copying from .env.example..."
    cp .env.example .env
fi

# 4. Launch FastAPI Application
PORT=8000
echo ""
echo "🚀 Launching DocuRAG Web Server on http://localhost:$PORT"
echo "👉 Open http://localhost:$PORT in your web browser"
echo ""

exec uvicorn src.main:app --host 0.0.0.0 --port $PORT --reload
