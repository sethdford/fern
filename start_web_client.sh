#!/bin/bash
# Quick start script for FERN web client

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║              🌐 FERN Web Client Launcher                          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check API key
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ ERROR: GOOGLE_API_KEY not set!"
    echo ""
    echo "Set it with:"
    echo "  export GOOGLE_API_KEY='your-key-here'"
    echo ""
    exit 1
fi

# Check dependencies
echo "🔍 Checking dependencies..."
python -c "import fastapi" 2>/dev/null || {
    echo "⚠️  FastAPI not installed"
    echo "   Installing: pip install fastapi uvicorn[standard] websockets"
    pip install fastapi "uvicorn[standard]" websockets
}

python -c "import uvicorn" 2>/dev/null || {
    echo "⚠️  uvicorn not installed"
    pip install "uvicorn[standard]"
}

echo "✓ Dependencies OK"
echo ""

# Check models
if [ ! -d "models/csm-1b" ]; then
    echo "⚠️  CSM-1B models not found!"
    echo "   Downloading models (2.9 GB, ~5-10 min)..."
    python scripts/download_models.py
    python scripts/integrate_real_models.py
    echo "✓ Models ready"
    echo ""
fi

# Get IP for display
IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

# Launch web server
echo "🚀 Starting FERN web server..."
echo ""
echo "Access points:"
echo "  • Local:      http://localhost:8000"
echo "  • Network:    http://${IP}:8000"
echo "  • API docs:   http://localhost:8000/docs"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

uvicorn web_client.app:app --host 0.0.0.0 --port 8000

