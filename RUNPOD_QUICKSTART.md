# FERN RunPod Quick Start Guide

**⏱️ Total Time: 15-20 minutes**

---

## 📋 Prerequisites

- [ ] RunPod.io account (https://runpod.io/)
- [ ] Payment method added (~$0.50/hour for RTX 4090)
- [ ] SSH key set up (or use password)
- [ ] HuggingFace account (for model downloads)

---

## 🚀 5-Step Deployment

### Step 1: Launch RunPod (2 minutes)

1. Go to https://runpod.io/console/pods
2. Click **"Deploy"**
3. Select **GPU**:
   - **Recommended**: RTX 4090 (24GB) - $0.50/hour
   - **Budget**: RTX 3090 (24GB) - $0.30/hour
   - **Premium**: A100 (40GB) - $2.00/hour
4. Select **Template**: PyTorch or Ubuntu
5. Set **Container Disk**: 50GB
6. Click **"Deploy On-Demand"** (or "Deploy Spot" for 50% savings)

**Wait for pod to start** (~1 minute)

### Step 2: Upload FERN (5 minutes)

From your **local terminal**:

```bash
# Navigate to your FERN directory
cd /Users/sethford/Downloads/voice

# Package is already created: fern_runpod_20251107_044327.tar.gz

# Get SSH details from RunPod dashboard (looks like this):
# ssh root@X.X.X.X -p XXXXX -i ~/.ssh/id_ed25519

# Upload package
scp -P XXXXX fern_runpod_20251107_044327.tar.gz root@X.X.X.X:/workspace/

# This takes 2-5 minutes (128 KB package)
```

### Step 3: Connect & Setup (5 minutes)

```bash
# Connect to RunPod
ssh root@X.X.X.X -p XXXXX -i ~/.ssh/id_ed25519

# Extract package
cd /workspace
tar -xzf fern_runpod_20251107_044327.tar.gz
cd fern

# Run automated setup
bash scripts/runpod_setup.sh

# The script will:
# ✓ Install dependencies
# ✓ Download models (2.9 GB)
# ✓ Integrate real models
# ✓ Run tests
# Takes 5-10 minutes
```

### Step 4: Test Everything (2 minutes)

```bash
# Activate environment
source venv/bin/activate

# Run full test suite
python scripts/test_real_models.py

# Expected output:
# ✓ Mimi Codec: PASSED
# ✓ CSM Generation: PASSED  # Works on GPU!
# ✓ Full Pipeline: PASSED (if OPENAI_API_KEY set)
```

### Step 5: Generate Speech! (30 seconds)

```bash
# Generate your first speech
python -c "
from fern.tts.csm_real import RealCSMTTS
import soundfile as sf

tts = RealCSMTTS(device='cuda')
audio = tts.synthesize('Hello from FERN on RunPod! This is amazing!')

sf.write('hello_runpod.wav', audio.cpu().numpy(), 24000)
print('✓ Generated: hello_runpod.wav')
"

# Download to listen
# From local terminal: scp -P XXXXX root@X.X.X.X:/workspace/fern/hello_runpod.wav ./
```

---

## 🎓 Start Training (Optional)

```bash
# Set up W&B for tracking (optional)
pip install wandb
wandb login

# Set OpenAI key for validation
export OPENAI_API_KEY="sk-..."

# Start training Elise voice (2-4 hours)
tmux new -s training
source venv/bin/activate
python scripts/train_lora.py \
    --dataset Jinsaryko/Elise \
    --batch_size 4 \
    --epochs 10 \
    --use_wandb

# Detach from tmux: Ctrl+B then D
# Reattach later: tmux attach -t training

# Training progress:
# - Epoch 1/10: ~20 minutes
# - Total: 2-4 hours on RTX 4090
# - Final checkpoint: checkpoints/elise-lora/
```

---

## 💰 Cost Estimate

**Testing & Setup:**
- Time: 20 minutes
- Cost: $0.17 (RTX 4090)

**Training (1 run):**
- Time: 3 hours
- Cost: $1.50 (RTX 4090)

**Full Day Development:**
- Time: 8 hours
- Cost: $4.00 (RTX 4090)

**💡 Tip**: Use **Spot instances** for 50% savings!

---

## 🔧 Troubleshooting

### "Connection refused" when SSH-ing
- Wait 30 more seconds for pod to fully start
- Check SSH port number from RunPod dashboard

### "CUDA out of memory"
```bash
# Reduce batch size
python scripts/train_lora.py --batch_size 2
```

### Lost connection during training
```bash
# Always use tmux!
tmux new -s training
# Run your command
# Detach: Ctrl+B then D
# Even if SSH dies, training continues
```

### Need to download models again
```bash
# Set HuggingFace token
export HF_TOKEN="hf_..."

# Re-download
python scripts/download_models.py
```

---

## 💾 Save Your Work

### Download Trained Model

```bash
# From local terminal:
scp -P XXXXX -r root@X.X.X.X:/workspace/fern/checkpoints ./checkpoints_backup
```

### Upload to HuggingFace

```bash
# On RunPod:
pip install huggingface_hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='checkpoints/elise-lora/final',
    repo_id='YOUR_USERNAME/fern-elise-voice',
    token='hf_...'
)
"
```

---

## 🎯 What's Next?

After testing on RunPod:

1. ✅ **Train your voice** - Fine-tune on Elise dataset
2. 🎤 **Test quality** - Generate samples and evaluate
3. 📊 **Benchmark** - Measure latency and quality
4. 🚀 **Deploy API** - Serve via FastAPI
5. 🌐 **Scale up** - Multi-GPU training if needed

---

## 📞 Need Help?

Common commands:

```bash
# Check GPU
nvidia-smi

# Check disk space
df -h

# Check running processes
htop

# Kill stuck process
pkill -f python

# Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} +

# Re-run setup
cd /workspace/fern && bash scripts/runpod_setup.sh
```

---

## 🎊 Success Checklist

After setup, you should have:

- ✅ RunPod connected
- ✅ FERN uploaded
- ✅ Dependencies installed
- ✅ Models downloaded (2.9 GB)
- ✅ Tests passing (3/3)
- ✅ GPU working (nvidia-smi)
- ✅ Speech generated
- ✅ Ready to train!

**You're now running FERN at full GPU speed! 🚀**

---

## 📚 Full Documentation

- **Complete Guide**: `RUNPOD_DEPLOYMENT.md`
- **API Reference**: `README.md`
- **Training Guide**: `LOAD_REAL_MODELS.md`
- **Final Status**: `FINAL_STATUS.md`

---

*Remember to **STOP YOUR POD** when done to avoid charges!*  
(RunPod Dashboard → Your Pod → Stop)

