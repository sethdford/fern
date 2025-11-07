#!/bin/bash

# Quick fix for audio dependencies on RunPod
# Run this if you get: ModuleNotFoundError: No module named 'sounddevice'

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  🔧 FERN Audio Dependencies Fix${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${YELLOW}ℹ This will install system audio libraries and Python audio packages${NC}"
echo ""

# Install system dependencies
echo -e "${BLUE}1/3 Installing system audio libraries...${NC}"
apt-get update -qq
apt-get install -y portaudio19-dev libportaudio2 libsndfile1 -qq
echo -e "${GREEN}✓ System audio libraries installed${NC}"
echo ""

# Install Python audio packages
echo -e "${BLUE}2/3 Installing Python audio packages...${NC}"
if [ -n "$VIRTUAL_ENV" ]; then
    pip install --force-reinstall sounddevice>=0.4.6 soundfile>=0.12.1 -q
else
    echo -e "${YELLOW}⚠️  No virtual environment detected. Activate venv first:${NC}"
    echo -e "   source /workspace/fern/venv/bin/activate"
    echo ""
    read -p "Try installing anyway? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install --force-reinstall sounddevice>=0.4.6 soundfile>=0.12.1 -q
    else
        exit 1
    fi
fi
echo -e "${GREEN}✓ Python audio packages installed${NC}"
echo ""

# Test import
echo -e "${BLUE}3/3 Testing audio imports...${NC}"
python3 -c "
import sounddevice as sd
import soundfile as sf
print('✓ sounddevice:', sd.__version__)
print('✓ soundfile:', sf.__version__)
print('✓ Available audio devices:')
print(sd.query_devices())
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✅ Audio dependencies successfully installed!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}You can now run:${NC}"
    echo "  python realtime_agent.py"
    echo "  python realtime_agent_advanced.py"
    echo "  python client_voice.py"
    echo ""
else
    echo ""
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}❌ Audio test failed${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "  1. Make sure you're in the virtual environment:"
    echo "     source /workspace/fern/venv/bin/activate"
    echo ""
    echo "  2. Try manual installation:"
    echo "     apt-get install -y portaudio19-dev libportaudio2 libsndfile1"
    echo "     pip install --force-reinstall sounddevice soundfile"
    echo ""
    exit 1
fi

