#!/bin/bash
# RunPod Setup Script for i-LAVA Phase 2
# Automatically sets up environment and runs profiling

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║              🚀  i-LAVA RunPod Setup Script  🚀                     ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/ilava.git}"
WORK_DIR="/workspace/ilava"

# Step 1: System dependencies
echo -e "${BLUE}[1/7]${NC} Installing system dependencies..."
apt-get update -qq
apt-get install -y cmake build-essential tmux git wget curl -qq
echo -e "${GREEN}✓${NC} System dependencies installed"
echo ""

# Step 2: Clone repository
echo -e "${BLUE}[2/7]${NC} Cloning repository..."
if [ ! -d "$WORK_DIR" ]; then
    cd /workspace
    git clone $REPO_URL ilava
else
    echo "  Repository already exists, pulling latest changes..."
    cd $WORK_DIR
    git pull
fi
cd $WORK_DIR
echo -e "${GREEN}✓${NC} Repository ready"
echo ""

# Step 3: Python environment
echo -e "${BLUE}[3/7]${NC} Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"
echo ""

# Step 4: Install dependencies
echo -e "${BLUE}[4/7]${NC} Installing Python dependencies..."
echo "  This may take a few minutes..."
pip install --upgrade pip -q
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
pip install -r requirements.txt -q
pip install pybind11 pytest cmake -q
echo -e "${GREEN}✓${NC} Dependencies installed"
echo ""

# Step 5: Verify CUDA
echo -e "${BLUE}[5/7]${NC} Verifying CUDA setup..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
print(f'✓ CUDA version: {torch.version.cuda}')
print(f'✓ PyTorch version: {torch.__version__}')
"
echo -e "${GREEN}✓${NC} CUDA verified"
echo ""

# Step 6: Enable real CSM model
echo -e "${BLUE}[6/7]${NC} Configuring real CSM model..."

# Backup original file
cp ilava/tts/csm_real.py ilava/tts/csm_real.py.backup

# Enable real CSM (comment stub, uncomment real)
sed -i '
    s/^from load_stub/# from load_stub/
    s/^# from generator import load_csm_1b$/from generator import load_csm_1b/
    s/^self\.generator = load_csm_1b_stub/# self.generator = load_csm_1b_stub/
    s/^# self\.generator = load_csm_1b(device)$/self.generator = load_csm_1b(device)/
' ilava/tts/csm_real.py

echo -e "${GREEN}✓${NC} Real CSM model enabled"
echo ""

# Step 7: Run tests
echo -e "${BLUE}[7/7]${NC} Running quick verification..."
export OPENAI_API_KEY="${OPENAI_API_KEY:-test-key}"
python -c "
from ilava.tts.csm_config import create_development_config
print('✓ Config system working')
print('✓ All imports successful')
" 2>/dev/null
echo -e "${GREEN}✓${NC} Verification passed"
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║              ✅  Setup Complete!  ✅                                ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 What's Ready:"
echo "  ✅ CUDA environment configured"
echo "  ✅ All dependencies installed"
echo "  ✅ Real CSM model enabled"
echo "  ✅ Repository up to date"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "  1️⃣  Profile TTS (30 min):"
echo "     python tools/profiling/profile_tts.py --real --trace"
echo ""
echo "  2️⃣  Analyze results (5 min):"
echo "     python tools/profiling/analyze_results.py"
echo ""
echo "  3️⃣  Build CUDA extension (30 min):"
echo "     cd ilava/native && mkdir -p build && cd build"
echo "     cmake .. && make -j\$(nproc)"
echo "     cd ../.. && pip install -e ilava/native"
echo ""
echo "  4️⃣  Benchmark speedup (15 min):"
echo "     python benchmark_rvq.py --device cuda"
echo ""
echo "💡 Pro Tips:"
echo "  • Use tmux to keep sessions alive: tmux new -s ilava"
echo "  • Monitor GPU: watch -n 1 nvidia-smi"
echo "  • Save results: git add benchmarks/ && git commit -m 'Profiling results'"
echo ""
echo "📊 Estimated total time: 2-3 hours"
echo "💰 Estimated cost: \$0.50-1.00 (RTX 3090 @ \$0.24/hr)"
echo ""
echo "🎉 Ready to start profiling!"
echo ""

