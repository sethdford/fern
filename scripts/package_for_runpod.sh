#!/bin/bash

# Package FERN for RunPod Upload
# This creates a tarball with all necessary files (excluding large cache/venv)

set -e

echo "📦 Packaging FERN for RunPod..."

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Create package name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="fern_runpod_${TIMESTAMP}.tar.gz"

echo "Creating package: $PACKAGE_NAME"

# Create tarball excluding unnecessary files
tar -czf "$PACKAGE_NAME" \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='*.egg-info' \
    --exclude='.pytest_cache' \
    --exclude='.coverage' \
    --exclude='htmlcov' \
    --exclude='models' \
    --exclude='checkpoints' \
    --exclude='*.wav' \
    --exclude='*.mp3' \
    fern/ \
    scripts/ \
    tests/ \
    examples/ \
    requirements.txt \
    setup.py \
    pyproject.toml \
    README.md \
    .cursorrules \
    .coveragerc \
    .editorconfig \
    RUNPOD_DEPLOYMENT.md \
    FINAL_STATUS.md

# Show package info
SIZE=$(du -h "$PACKAGE_NAME" | cut -f1)
echo "✓ Package created: $PACKAGE_NAME ($SIZE)"
echo ""
echo "📤 Upload to RunPod:"
echo ""
echo "1. Launch your RunPod pod"
echo "2. Get SSH connection details from RunPod dashboard"
echo "3. Upload:"
echo "   scp -P PORT $PACKAGE_NAME root@X.X.X.X:/workspace/"
echo ""
echo "4. On RunPod, extract and run setup:"
echo "   cd /workspace"
echo "   tar -xzf $PACKAGE_NAME"
echo "   cd fern"
echo "   bash scripts/runpod_setup.sh"
echo ""
echo "🎉 Done!"

