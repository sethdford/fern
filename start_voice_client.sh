#!/bin/bash
# Quick start script for FERN voice clients

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║              🎙️  FERN Voice Client Launcher                       ║"
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
python -c "import sounddevice" 2>/dev/null || {
    echo "⚠️  sounddevice not installed"
    echo "   Installing: pip install sounddevice soundfile pynput webrtcvad"
    pip install sounddevice soundfile pynput webrtcvad
}

python -c "import soundfile" 2>/dev/null || {
    echo "⚠️  soundfile not installed"
    pip install soundfile
}

python -c "import pynput" 2>/dev/null || {
    echo "⚠️  pynput not installed"
    pip install pynput
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

# Launch client
echo "🚀 Launching Python voice client..."
echo ""
echo "Controls:"
echo "  SPACE - Hold to talk, release to send"
echo "  R     - Reset conversation"
echo "  ESC   - Exit"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

python client_voice.py

