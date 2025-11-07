# FERN Voice Clients

Two options for real-time voice conversation with FERN:

1. **Python Desktop Client** - Simple, keyboard-driven, works anywhere
2. **Web Client** - Browser-based, modern UI, remote access

---

## 🖥️ Option 1: Python Desktop Client

### Features
- ✅ Push-to-talk recording (SPACE key)
- ✅ Real-time audio playback
- ✅ Conversation history
- ✅ Works locally (no web server needed)
- ✅ Visual terminal feedback
- ✅ Streaming TTS (low latency)

### Quick Start

```bash
# Install dependencies
pip install sounddevice soundfile pynput

# Set API key
export GOOGLE_API_KEY="your-key-here"

# Run the client
python client_voice.py
```

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Hold to record, release to send |
| `R` | Reset conversation |
| `ESC` | Exit |

### Usage Flow

1. **Start**: `python client_voice.py`
2. **Talk**: Press and hold `SPACE`, speak, then release
3. **Listen**: FERN responds with streaming audio
4. **Continue**: Press `SPACE` again for next turn

### Example Session

```
🎙️  FERN Voice Client

Loading models...
  → Gemini LLM...
    ✓ Using gemini-1.5-flash
  → CSM-1B TTS (cuda)...
    ✓ Ready

============================================================
Ready! Press SPACE to talk, ESC to exit
============================================================

🔴 Recording... (release SPACE when done)
⏹️  Processing...
  → Transcribing...
  You: What's the weather today?
  
  → Thinking...
  🤖 FERN: I don't have access to real-time weather data...
  
  → Speaking...
  🔊 Playing...
  ✓ Done!

Press SPACE to talk again
```

### Troubleshooting

**No audio input/output?**
```bash
# List audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Test audio
python -c "import sounddevice as sd; sd.rec(16000, samplerate=16000, channels=1)"
```

**PortAudio library not found?**
```bash
# Ubuntu/Debian
apt-get install -y portaudio19-dev libportaudio2

# macOS
brew install portaudio

# Then reinstall
pip install --force-reinstall sounddevice
```

---

## 🌐 Option 2: Web Client

### Features
- ✅ Modern, responsive UI
- ✅ No installation for end users
- ✅ Remote access (deploy anywhere)
- ✅ REST API + WebSocket support
- ✅ Mobile-friendly
- ✅ Streaming audio responses

### Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn[standard] websockets

# Set API key
export GOOGLE_API_KEY="your-key-here"

# Start server
cd /workspace/fern
uvicorn web_client.app:app --host 0.0.0.0 --port 8000

# Open browser
http://localhost:8000
```

### Server Endpoints

#### REST API

**1. Health Check**
```bash
curl http://localhost:8000/health
```

**2. Text Chat**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello FERN!"}'
```

Response:
```json
{
  "response": "Hello! How can I help you today?",
  "audio_base64": null
}
```

**3. Text-to-Speech**
```bash
curl -X POST http://localhost:8000/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  --output response.wav
```

**4. Full Conversation (Text + Audio)**
```bash
curl -X POST http://localhost:8000/api/chat-with-voice \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a joke"}' | jq
```

Response:
```json
{
  "response": "Why did the programmer quit his job?...",
  "audio_base64": "UklGRi4YAABUQU..."
}
```

**5. Streaming Audio**
```bash
curl "http://localhost:8000/api/stream/Hello%20from%20FERN" \
  --output streaming.wav
```

#### WebSocket (Real-time)

Connect to `ws://localhost:8000/ws`

**Client sends:**
```json
{
  "type": "message",
  "text": "Hello!"
}
```

**Server responds:**
```json
{"type": "response", "text": "Hello! How are you?"}
{"type": "audio_chunk", "data": "UklGRi..."}
{"type": "audio_chunk", "data": "AABUQU..."}
{"type": "complete"}
```

### Deploy on RunPod

**1. Update setup script**
```bash
# Edit scripts/runpod_setup.sh - add at the end:
echo "Starting web server..."
cd /workspace/fern
nohup uvicorn web_client.app:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
echo "✓ Server running on port 8000"
```

**2. Expose port in RunPod**
- Go to RunPod console
- Click "Edit" on your pod
- Add TCP port: 8000
- Access via: `https://<pod-id>-8000.proxy.runpod.net`

**3. Access remotely**
```
https://qcfadv63vojkpe-8000.proxy.runpod.net
```

### Web UI Features

**Chat Interface**
- Type messages or click "Send"
- Press `Enter` to send
- Responses appear with streaming audio
- Auto-scroll to latest message

**Controls**
- 🎤 **Start Recording** - Voice input (coming soon)
- 🗑️ **Clear** - Reset conversation
- Visual status indicators (thinking/speaking/ready)

### Customize the UI

Edit `web_client/index.html`:

```html
<!-- Change colors -->
<style>
  body {
    background: linear-gradient(135deg, #your-color 0%, #another-color 100%);
  }
  
  h1 {
    color: #your-brand-color;
  }
</style>

<!-- Change title -->
<h1>🎙️ Your Assistant Name</h1>

<!-- Change greeting -->
<div class="message assistant-message">
  👋 Your custom greeting here!
</div>
```

### Production Deployment

**Environment Variables**
```bash
export GOOGLE_API_KEY="your-key"
export FERN_DEVICE="cuda"  # or "cpu"
export FERN_HOST="0.0.0.0"
export FERN_PORT="8000"
```

**Run with systemd** (Ubuntu)
```bash
# Create service file
sudo nano /etc/systemd/system/fern.service
```

```ini
[Unit]
Description=FERN Voice Assistant API
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/fern
Environment="GOOGLE_API_KEY=your-key"
ExecStart=/path/to/venv/bin/uvicorn web_client.app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable fern
sudo systemctl start fern
sudo systemctl status fern
```

**Run with Docker**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV GOOGLE_API_KEY=""
EXPOSE 8000

CMD ["uvicorn", "web_client.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t fern-voice .
docker run -p 8000:8000 -e GOOGLE_API_KEY="your-key" fern-voice
```

**NGINX Reverse Proxy**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📊 Performance Comparison

| Feature | Python Client | Web Client |
|---------|---------------|------------|
| **Latency** | 80-100ms first audio | 100-200ms (network) |
| **Setup** | Instant | Requires server |
| **Access** | Local only | Remote OK |
| **UI** | Terminal | Modern web |
| **Mobile** | ❌ No | ✅ Yes |
| **Multi-user** | ❌ No | ✅ Yes |

---

## 🔧 Advanced: Add Voice Input to Web Client

**1. Enable microphone in browser**

Edit `web_client/index.html`, add:

```javascript
let mediaRecorder;
let audioChunks = [];

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await sendAudioToAPI(audioBlob);
            audioChunks = [];
        };
        
        mediaRecorder.start();
        updateStatus('recording', '🔴 Recording...');
        
    } catch (error) {
        alert('Microphone access denied: ' + error.message);
    }
}

async function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        updateStatus('thinking', '🤔 Processing...');
    }
}

async function sendAudioToAPI(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob);
    
    const response = await fetch('/api/transcribe', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    addMessage(data.text, 'user');
    
    // Continue with chat flow...
}
```

**2. Add transcription endpoint**

Edit `web_client/app.py`:

```python
from fern.asr.whisper_asr import WhisperASR

asr = None

def get_asr():
    global asr
    if asr is None:
        asr = WhisperASR(model_size="large-v3-turbo")
    return asr

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe audio to text."""
    try:
        # Read audio file
        audio_bytes = await audio.read()
        
        # Convert to numpy array
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Transcribe
        asr_instance = get_asr()
        text = asr_instance.transcribe(audio_data, sample_rate)
        
        return {"text": text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 🎯 Next Steps

### Python Client
- [ ] Integrate Whisper ASR (replace keyboard input)
- [ ] Add VAD for automatic turn detection
- [ ] Support hotword activation ("Hey FERN")
- [ ] Add GUI with PyQt/Tkinter

### Web Client
- [ ] Add microphone recording
- [ ] Implement WebSocket real-time streaming
- [ ] Add user authentication
- [ ] Support multiple concurrent users
- [ ] Add conversation export (JSON/TXT)
- [ ] Voice activity visualization (waveform)

### Both
- [ ] Add emotion detection in voice
- [ ] Support multiple languages
- [ ] Add voice customization (pitch, speed)
- [ ] Integrate with calendar/email
- [ ] Add memory/context persistence

---

## 🐛 Common Issues

**1. "No module named 'sounddevice'"**
```bash
pip install sounddevice soundfile
```

**2. "PortAudio library not found"**
```bash
# Ubuntu/Debian
apt-get install -y portaudio19-dev

# macOS
brew install portaudio
```

**3. "CUDA out of memory"**
```python
# Reduce batch size or use CPU for TTS
tts = RealCSMTTS(device="cpu")
```

**4. "Connection refused" (web client)**
```bash
# Check if server is running
curl http://localhost:8000/health

# Check firewall
sudo ufw allow 8000
```

**5. "Audio is choppy/laggy"**
- Use smaller chunk sizes: `StreamingTTS(tts, chunk_duration_ms=100)`
- Enable GPU acceleration: `device="cuda"`
- Reduce LLM response length

---

## 📚 API Documentation

Full interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

**Ready to start?** Choose your client and follow the Quick Start guide above! 🚀

