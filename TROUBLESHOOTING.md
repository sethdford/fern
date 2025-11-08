# FERN Troubleshooting Guide

## Common Issues and Solutions

### 1. ModuleNotFoundError: No module named 'sounddevice'

**Symptom:**
```
ModuleNotFoundError: No module named 'sounddevice'
```

**Root Cause:**
`sounddevice` requires the PortAudio system library to be installed BEFORE the Python package. If you install `sounddevice` before PortAudio, it compiles without audio support.

**Solution (Quick Fix):**
```bash
cd /workspace/fern
chmod +x scripts/fix_audio_deps.sh
./scripts/fix_audio_deps.sh
```

**Solution (Manual):**
```bash
# 1. Install system dependencies
apt-get update
apt-get install -y portaudio19-dev libportaudio2 libsndfile1 libasound2-dev

# 2. Reinstall Python audio packages
source venv/bin/activate
pip uninstall -y sounddevice soundfile
pip install --force-reinstall --no-cache-dir sounddevice>=0.4.6 soundfile>=0.12.1

# 3. Test it
python3 -c "import sounddevice as sd; print('✓ Working!'); print(f'Devices: {len(sd.query_devices())}')"
```

**Prevention:**
The updated `scripts/runpod_setup.sh` now:
1. Installs PortAudio FIRST
2. Installs all other dependencies
3. Force reinstalls audio packages with `--force-reinstall --no-cache-dir`

---

### 2. OSError: PortAudio library not found

**Symptom:**
```
OSError: PortAudio library not found
```

**Root Cause:**
Same as above - `sounddevice` was compiled without PortAudio.

**Solution:**
Same as issue #1 - run `scripts/fix_audio_deps.sh`

---

### 3. FileNotFoundError: No model files found in models/csm-1b

**Symptom:**
```
FileNotFoundError: No model files found in models/csm-1b
```

**Root Cause:**
CSM-1B and Mimi models haven't been downloaded yet.

**Solution:**
```bash
cd /workspace/fern
source venv/bin/activate

# Download models (2.9 GB, takes 5-10 minutes)
python scripts/download_models.py

# Integrate them
python scripts/integrate_real_models.py

# Test
python scripts/test_real_models.py
```

---

### 4. torch.AcceleratorError: CUDA error: no kernel image is available

**Symptom:**
```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

**Root Cause:**
PyTorch version doesn't support your GPU architecture (e.g., RTX 5090 needs PyTorch 2.5.1+).

**Solution:**
For RTX 5090:
```bash
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

For RTX 4090:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify:**
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

### 5. Error generating response: 404 models/gemini-pro is not found

**Symptom:**
```
Error generating response: 404 models/gemini-pro is not found
```

**Root Cause:**
Google Gemini model names change over time. The code now dynamically detects available models.

**Solution:**
Ensure you have the latest code:
```bash
cd /workspace/fern
git pull
```

The `fern/llm/gemini_manager.py` now auto-detects available models and tries:
1. `gemini-1.5-flash-latest`
2. `gemini-1.5-pro-latest`
3. `gemini-pro`

**Verify:**
```bash
export GOOGLE_API_KEY="your-key"
python3 -c "from fern.llm.gemini_manager import GeminiDialogueManager; gm = GeminiDialogueManager(); print(gm.generate_response('Hello'))"
```

---

### 6. ModuleNotFoundError: No module named 'torchao'

**Symptom:**
```
ModuleNotFoundError: No module named 'torchao'
```

**Root Cause:**
`torchtune` requires `torchao`, but it's not always installed automatically.

**Solution:**
```bash
pip install torchao
```

---

### 7. Git push failed / Permission denied

**Symptom:**
```
git push
Permission denied (publickey).
```

**Root Cause:**
SSH keys not configured for GitHub.

**Solution:**
```bash
# Option 1: Use HTTPS instead
git remote set-url origin https://github.com/sethdford/fern.git
git push

# Option 2: Add SSH key to GitHub
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
# Add this to GitHub Settings > SSH Keys
```

---

### 8. No audio output when running real-time agents

**Symptom:**
Running `realtime_agent.py` but hearing no audio.

**Root Cause:**
Multiple possibilities:
- Audio device not configured
- Volume muted
- Wrong audio device selected

**Solution:**
```bash
# List audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Test audio output
python3 << EOF
import sounddevice as sd
import numpy as np

# Generate 440Hz tone (A4)
fs = 24000
duration = 1.0
t = np.linspace(0, duration, int(fs * duration))
audio = np.sin(2 * np.pi * 440 * t) * 0.3

print("Playing test tone...")
sd.play(audio, fs)
sd.wait()
print("Did you hear a beep?")
EOF
```

**Configure specific device:**
Edit the agent script and set:
```python
sd.default.device = 1  # Change to your device ID
```

---

### 9. Import errors after git pull

**Symptom:**
```
ImportError: cannot import name 'X' from 'fern.Y'
```

**Root Cause:**
New dependencies added or package structure changed.

**Solution:**
```bash
cd /workspace/fern
source venv/bin/activate

# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# If that doesn't work, recreate venv
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
./scripts/runpod_setup.sh  # Will skip already-installed system packages
```

---

### 10. Out of memory errors

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Root Cause:**
Model too large for your GPU.

**Solution:**

**Check GPU memory:**
```bash
nvidia-smi
```

**Reduce batch size:**
```python
# In training scripts
config.batch_size = 1  # Reduce from 4 to 1
```

**Use CPU for TTS (slower but works):**
```python
tts = RealCSMTTS(device="cpu")
```

**Clear CUDA cache:**
```python
import torch
torch.cuda.empty_cache()
```

---

## Prevention: Best Practices

### 1. Always Pull Latest Code
```bash
cd /workspace/fern
git pull
```

### 2. Use Virtual Environments
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies in Order
```bash
# 1. System packages first
apt-get update
apt-get install -y portaudio19-dev libportaudio2 libsndfile1

# 2. PyTorch second
pip install torch torchvision torchaudio --index-url https://...

# 3. Other packages third
pip install -r requirements.txt

# 4. CSM-streaming last
pip install git+https://github.com/davidbrowne17/csm-streaming.git
```

### 4. Set Environment Variables
```bash
# Create .env file
cat > .env << EOF
GOOGLE_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
HF_TOKEN=your-token-here
EOF

# Load in scripts
python3 << EOF
from dotenv import load_dotenv
load_dotenv()
EOF
```

### 5. Test After Installation
```bash
# Quick test
python scripts/test_real_models.py

# Voice client test
python test_gemini.py

# Full pipeline test
python realtime_agent.py
```

---

## Getting Help

### 1. Check Logs
```bash
# Python errors
python script.py 2>&1 | tee error.log

# System logs
dmesg | tail -50

# CUDA logs
nvidia-smi dmon -s u -d 1
```

### 2. Verify Environment
```bash
# Python version
python3 --version

# PyTorch version
python3 -c "import torch; print(torch.__version__)"

# CUDA version
nvidia-smi

# Installed packages
pip list | grep -E "torch|sounddevice|transformers|pydantic"
```

### 3. Run Diagnostics
```bash
# Create diagnostic script
cat > diagnose.py << 'EOF'
import sys
import torch
import sounddevice as sd
import os

print("=== FERN Diagnostic Report ===\n")

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print(f"\nAudio Devices: {len(sd.query_devices())}")
print(sd.query_devices())

print(f"\nEnvironment Variables:")
print(f"  GOOGLE_API_KEY: {'✓ Set' if os.getenv('GOOGLE_API_KEY') else '✗ Missing'}")
print(f"  OPENAI_API_KEY: {'✓ Set' if os.getenv('OPENAI_API_KEY') else '✗ Missing'}")

print("\nTrying imports...")
try:
    from fern.tts.csm_real import RealCSMTTS
    print("  ✓ RealCSMTTS")
except Exception as e:
    print(f"  ✗ RealCSMTTS: {e}")

try:
    from fern.llm.gemini_manager import GeminiDialogueManager
    print("  ✓ GeminiDialogueManager")
except Exception as e:
    print(f"  ✗ GeminiDialogueManager: {e}")

try:
    from fern.asr.whisper_asr import WhisperASR
    print("  ✓ WhisperASR")
except Exception as e:
    print(f"  ✗ WhisperASR: {e}")

print("\n=== End Report ===")
EOF

python diagnose.py
```

---

## Quick Command Reference

```bash
# Fix audio
./scripts/fix_audio_deps.sh

# Reinstall everything
rm -rf venv && python3 -m venv venv && source venv/bin/activate && ./scripts/runpod_setup.sh

# Update code
git pull && pip install --upgrade -r requirements.txt

# Test models
python scripts/test_real_models.py

# Run agent
python realtime_agent_advanced.py

# Start web server
uvicorn web_client.app:app --host 0.0.0.0 --port 8000

# Monitor GPU
watch -n 1 nvidia-smi
```

---

**Last Updated:** 2025-11-08

For more help, see:
- Main README: `README.md`
- Real-time Agent Guide: `REALTIME_AGENT_GUIDE.md`
- Deployment Guide: `DEPLOY.md`
