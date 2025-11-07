# FERN Troubleshooting Guide

## Common Issues & Solutions

### 1. `ModuleNotFoundError: No module named 'sounddevice'`

**Problem**: System audio libraries not installed.

**Solution**:
```bash
# Quick fix (recommended)
bash scripts/fix_audio_deps.sh

# OR manual installation:
# 1. Install system libraries
apt-get update
apt-get install -y portaudio19-dev libportaudio2 libsndfile1

# 2. Reinstall Python packages
source venv/bin/activate
pip install --force-reinstall sounddevice soundfile
```

**Why?** The `sounddevice` Python package requires the PortAudio C library at the system level.

---

### 2. `OSError: PortAudio library not found`

**Problem**: Same as above - PortAudio system library missing.

**Solution**: Follow solution for error #1 above.

---

### 3. `ModuleNotFoundError: No module named 'torchao'`

**Problem**: Missing `torchao` dependency for `torchtune`.

**Solution**:
```bash
source venv/bin/activate
pip install torchao
```

**Why?** `torchtune` (required by CSM-1B) depends on `torchao` for optimizations.

---

### 4. `FileNotFoundError: No model files found in models/csm-1b`

**Problem**: CSM-1B and Mimi model weights not downloaded.

**Solution**:
```bash
# Download models (2.9 GB)
python scripts/download_models.py

# Integrate into FERN
python scripts/integrate_real_models.py

# Test
python scripts/test_real_models.py
```

**Why?** FERN requires real model weights from HuggingFace.

---

### 5. `404 models/gemini-pro is not found`

**Problem**: Gemini model name has changed or is unavailable.

**Solution**: Already auto-fixed! The code now auto-detects available models.

If still failing:
```bash
# Check available models
python -c "
import google.generativeai as genai
genai.configure(api_key='your-key')
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(model.name)
"
```

---

### 6. PyTorch CUDA Capability Mismatch

**Problem**:
```
WARNING: NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible
```

**Solution**: Install correct PyTorch version for your GPU.

```bash
# For RTX 5090 (CUDA 12.6)
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# For RTX 4090 (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For older GPUs (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Why?** Different GPU architectures require different CUDA versions.

---

### 7. `RuntimeError: Index put requires the source and destination dtypes match`

**Problem**: Dtype mismatch when loading CSM-1B model.

**Solution**: Already fixed! The code now auto-converts dtypes.

If still failing:
```bash
# Force CPU fallback
python realtime_agent.py  # Will auto-detect and use CPU if needed
```

---

### 8. No audio output / Can't hear TTS

**Problem**: Audio device configuration.

**Solution**:
```bash
# List available audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Set default output device (replace X with device ID)
export SD_DEVICE=X

# Test audio
python -c "
import sounddevice as sd
import numpy as np
# Play 440Hz tone for 1 second
fs = 44100
t = np.linspace(0, 1, fs)
audio = np.sin(2 * np.pi * 440 * t)
sd.play(audio, fs)
sd.wait()
print('✓ Audio test complete')
"
```

---

### 9. `git clone` fails / Can't push to GitHub

**Problem**: Git authentication.

**Solution**:
```bash
# Option 1: Use HTTPS with Personal Access Token
git clone https://github.com/sethdford/fern.git
# Username: your-github-username
# Password: your-personal-access-token

# Option 2: Use SSH
git clone git@github.com:sethdford/fern.git
# Requires SSH key setup: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
```

---

### 10. RunPod setup script hangs or fails

**Problem**: Network issues, package conflicts, or missing dependencies.

**Solution**:
```bash
# Run setup in verbose mode
bash -x scripts/runpod_setup.sh 2>&1 | tee setup.log

# If specific step fails, run manually:
cd /workspace/fern
source venv/bin/activate

# Install in order:
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install git+https://github.com/davidbrowne17/csm-streaming.git
python scripts/download_models.py
python scripts/integrate_real_models.py
```

---

## Environment Setup Checklist

Before running FERN, ensure:

- ✅ **System**: Linux/Ubuntu (RunPod recommended)
- ✅ **GPU**: NVIDIA with CUDA (RTX 4090/5090 recommended)
- ✅ **Python**: 3.10+ with virtual environment
- ✅ **System Audio**: PortAudio installed (`apt-get install portaudio19-dev`)
- ✅ **PyTorch**: Correct CUDA version for your GPU
- ✅ **API Key**: `GOOGLE_API_KEY` environment variable set
- ✅ **Models**: CSM-1B and Mimi weights downloaded (`models/` directory)
- ✅ **Packages**: All `requirements.txt` installed + `csm-streaming`

---

## Getting Help

### Check Logs
```bash
# Run with debug logging
export LOG_LEVEL=DEBUG
python realtime_agent.py 2>&1 | tee fern.log
```

### Test Components Individually
```bash
# Test Gemini
python test_gemini.py

# Test models
python scripts/test_real_models.py

# Test audio
python -c "import sounddevice; print(sounddevice.query_devices())"

# Test VAD
python -c "from fern.asr.vad_detector import VADDetector; vad = VADDetector(); print('✓ VAD OK')"

# Test ASR
python -c "from fern.asr.whisper_asr import WhisperASR; asr = WhisperASR(); print('✓ ASR OK')"
```

### Quick Fixes Script
```bash
# Runs all common fixes
bash scripts/fix_audio_deps.sh
```

---

## Best Practices

1. **Always use virtual environment**:
   ```bash
   source /workspace/fern/venv/bin/activate
   ```

2. **Update dependencies regularly**:
   ```bash
   git pull
   pip install -r requirements.txt --upgrade
   ```

3. **Use tmux for long-running tasks**:
   ```bash
   tmux new -s fern
   python realtime_agent_advanced.py
   # Detach: Ctrl+B then D
   # Reattach: tmux attach -t fern
   ```

4. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

5. **Check disk space** (models are 2.9 GB):
   ```bash
   df -h /workspace
   du -sh models/
   ```

---

## Still Having Issues?

1. **Check the logs** in the terminal output
2. **Run the test suite**: `pytest tests/`
3. **Verify system requirements** in this guide
4. **Check GitHub Issues**: https://github.com/sethdford/fern/issues
5. **Run quick fixes**: `bash scripts/fix_audio_deps.sh`

---

*This troubleshooting guide covers the most common issues. For FERN-specific problems, see `REALTIME_AGENT_GUIDE.md`.*

