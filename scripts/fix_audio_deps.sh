#!/bin/bash

# Quick Fix for Audio Dependencies on RunPod
# Run this if you see "ModuleNotFoundError: No module named 'sounddevice'"

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🔧 Fixing Audio Dependencies...${NC}"

# Install system dependencies for audio
echo "Installing system audio libraries..."
apt-get update -y
apt-get install -y portaudio19-dev libportaudio2 libsndfile1 libasound2-dev

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
fi

# Force reinstall audio-related Python packages
echo "Reinstalling Python audio packages..."
pip uninstall -y sounddevice soundfile pynput webrtcvad 2>/dev/null || true
pip install sounddevice>=0.4.6 soundfile>=0.12.1 pynput>=1.7.6 webrtcvad>=2.0.10

echo -e "${GREEN}✓ Audio dependencies fixed!${NC}"

# Test it
echo ""
echo "Testing sounddevice import..."
python3 -c "import sounddevice as sd; print('✓ sounddevice working!'); print(f'  Devices: {len(sd.query_devices())} found')"

echo ""
echo -e "${GREEN}✓ All audio packages installed successfully!${NC}"
echo ""
echo "You can now run:"
echo "  python realtime_agent.py"
echo "  python realtime_agent_advanced.py"
echo "  python client_voice.py"
