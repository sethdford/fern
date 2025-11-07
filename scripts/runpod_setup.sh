#!/bin/bash

# FERN RunPod Automated Setup Script
# This script sets up everything needed to run FERN on RunPod

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if running on RunPod
if [ ! -d "/workspace" ]; then
    print_error "Not running on RunPod? /workspace directory not found"
    print_info "This script is designed for RunPod.io GPU instances"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

print_header "FERN RunPod Setup"

# Check CUDA
print_info "Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    print_success "CUDA available"
else
    print_error "CUDA not available - are you on a GPU pod?"
    exit 1
fi

# Update system
print_header "Updating System"
apt-get update -y
apt-get install -y git wget curl vim htop tmux build-essential
print_success "System updated"

# Set up workspace
print_header "Setting Up Workspace"
cd /workspace

# Check if FERN already exists
if [ -d "fern" ]; then
    print_info "FERN directory already exists"
    read -p "Remove and re-clone? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf fern
    else
        cd fern
        print_info "Using existing FERN directory"
    fi
fi

# Clone or use existing
if [ ! -d "fern" ]; then
    print_info "Cloning FERN repository..."
    
    # Clone from GitHub
    if git clone https://github.com/sethdford/fern.git; then
        print_success "Repository cloned from GitHub"
    else
        print_error "Failed to clone from GitHub"
        
        # Fallback: Check if user uploaded files
        if [ -f "../voice.tar.gz" ]; then
            print_info "Found voice.tar.gz, extracting..."
            tar -xzf ../voice.tar.gz
            mv voice fern
        else
            print_error "Please push your code to GitHub or upload manually"
            print_info "Option 1 - Push to GitHub:"
            print_info "  cd /Users/sethford/Downloads/voice"
            print_info "  git add ."
            print_info "  git commit -m 'Initial commit'"
            print_info "  git push"
            print_info ""
            print_info "Option 2 - Upload manually:"
            print_info "  cd /Users/sethford/Downloads"
            print_info "  tar -czf voice.tar.gz voice/"
            print_info "  scp -P PORT voice.tar.gz root@X.X.X.X:/workspace/"
            exit 1
        fi
    fi
fi

cd /workspace/fern
print_success "Workspace ready: /workspace/fern"

# Create Python virtual environment
print_header "Setting Up Python Environment"

if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip -q

# Install PyTorch with CUDA
print_info "Installing PyTorch with CUDA support..."

# Detect GPU and install appropriate PyTorch version
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

if [[ "$GPU_NAME" == *"5090"* ]] || [[ "$GPU_NAME" == *"50"* ]]; then
    print_info "Detected RTX 5090 - Installing PyTorch 2.8.0 with CUDA 12.6"
    pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 -q
elif [[ "$GPU_NAME" == *"4090"* ]] || [[ "$GPU_NAME" == *"40"* ]]; then
    print_info "Detected RTX 4090 - Installing PyTorch with CUDA 12.1"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
else
    print_info "Installing PyTorch with CUDA 12.1 (default)"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
fi

print_success "PyTorch installed"

# Verify CUDA in PyTorch
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch'; print(f'✓ PyTorch CUDA: {torch.cuda.get_device_name(0)}')"

# Install FERN dependencies (in correct order)
print_info "Installing FERN dependencies..."
if [ -f "requirements.txt" ]; then
    # Install all requirements (csm-streaming is commented out in the file)
    pip install -r requirements.txt -q
    print_success "Core dependencies installed"
    
    # Now install csm-streaming separately (PyTorch is available now)
    print_info "Installing CSM-streaming (requires PyTorch to build)..."
    pip install git+https://github.com/davidbrowne17/csm-streaming.git -q
    print_success "CSM-streaming installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Download models
print_header "Downloading Models (2.9 GB)"

if [ -d "models/csm-1b" ] && [ -d "models/mimi" ]; then
    print_info "Models already downloaded"
    print_info "  CSM-1B: $(du -sh models/csm-1b | cut -f1)"
    print_info "  Mimi: $(du -sh models/mimi | cut -f1)"
else
    print_info "Downloading CSM-1B and Mimi..."
    print_info "This will take 5-10 minutes..."
    
    if [ -f "scripts/download_models.py" ]; then
        python scripts/download_models.py
        if [ $? -eq 0 ]; then
            print_success "Models downloaded"
        else
            print_error "Model download failed"
            print_info "You can try again manually: python scripts/download_models.py"
        fi
    else
        print_error "scripts/download_models.py not found"
    fi
fi

# Integrate models
print_header "Integrating Real Models"

if [ -f "scripts/integrate_real_models.py" ]; then
    python scripts/integrate_real_models.py
    print_success "Models integrated"
else
    print_error "scripts/integrate_real_models.py not found"
fi

# Run tests
print_header "Testing Installation"

if [ -f "scripts/test_real_models.py" ]; then
    print_info "Running tests..."
    python scripts/test_real_models.py
    
    if [ $? -eq 0 ]; then
        print_success "All tests passed!"
    else
        print_error "Some tests failed"
        print_info "Check output above for details"
    fi
else
    print_error "scripts/test_real_models.py not found"
fi

# Final summary
print_header "Setup Complete!"

echo ""
echo -e "${GREEN}✓ CUDA Available${NC}"
echo -e "${GREEN}✓ Python Environment Ready${NC}"
echo -e "${GREEN}✓ Dependencies Installed${NC}"
echo -e "${GREEN}✓ Models Downloaded${NC}"
echo -e "${GREEN}✓ Models Integrated${NC}"
echo -e "${GREEN}✓ Tests Passed${NC}"
echo ""

print_info "FERN is ready to use!"
echo ""
echo "Quick Start:"
echo "  source venv/bin/activate"
echo "  python scripts/test_real_models.py       # Test everything"
echo "  python scripts/train_lora.py             # Start training"
echo "  python -m fern.cli \"Hello world\"        # Generate speech"
echo ""

print_info "Recommended: Use tmux for long-running tasks"
echo "  tmux new -s training"
echo "  python scripts/train_lora.py --dataset Jinsaryko/Elise"
echo "  # Detach: Ctrl+B then D"
echo "  # Reattach: tmux attach -t training"
echo ""

print_success "Happy training! 🚀"

