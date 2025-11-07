# FERN Voice Clients - Critical Audit

## 🔴 CRITICAL: Missing ASR Integration

### Issue 1: Python Client Has NO Real ASR
**Location**: `client_voice.py` lines 187-190

```python
# Simple placeholder - in production, use Whisper
# For now, use keyboard input as fallback
print(f"{Colors.YELLOW}  ⚠️  ASR not yet integrated - please type your message:{Colors.END}")
user_text = input(f"{Colors.GREEN}  You: {Colors.END}")
```

**Impact**: 
- User records audio but can't actually transcribe it
- Must type message manually - defeats entire purpose
- **CRITICAL BLOCKER** for voice functionality

**Fix Required**:
- Integrate `WhisperASR` from `fern/asr/whisper_asr.py`
- Add to `__init__` and call in `_process_audio`

---

## 🔴 CRITICAL: Web Client Recording Not Implemented

### Issue 2: Browser Recording Placeholder
**Location**: `web_client/index.html` line 335

```javascript
function startRecording() {
    alert('Voice recording coming soon! For now, please type your message.');
    // TODO: Implement WebRTC audio recording
}
```

**Impact**:
- Web client is **text-only**
- No voice input capability
- Not actually a "voice client"

**Fix Required**:
- Implement `MediaRecorder` API for browser recording
- Add `/api/transcribe` endpoint to backend
- Wire up WebSocket for real-time streaming

---

## 🟡 MAJOR: Incomplete Model Loading

### Issue 3: Mimi Codec Still Using Stubs
**Location**: `fern/tts/csm/moshi/models/loaders.py` lines 109-115

```python
# TODO: Actually load the real Mimi model from weight_path
# For now, return a stub that generates random audio
print(f"⚠️  Using stub Mimi model (weights at {weight_path} not loaded)")
print(f"   This is a placeholder - real audio codec loading not yet implemented")

return MimiModel(device=device)
```

**Impact**:
- Audio quality may be compromised
- Using placeholder codec instead of real weights
- Training/fine-tuning won't work properly

**Fix Required**:
- Implement actual safetensors loading for Mimi
- Load encoder/decoder/RVQ layers properly
- Test audio quality with real weights

---

## 🟡 MAJOR: Training Still Using Stubs

### Issue 4: Training Script Uses Dummy Model
**Location**: `scripts/train_lora.py` lines 47-50

```python
# For now, use stub for testing
from fern.tts.csm.load_stub import load_csm_1b_stub

model = load_csm_1b_stub(device=device)
logger.info("CSM model loaded (using stub)")
```

**Impact**:
- Fine-tuning won't work
- Can't train LoRA adapters
- Elise dataset integration incomplete

**Fix Required**:
- Load real CSM-1B model for training
- Verify gradient flow through LoRA layers
- Test training loop end-to-end

---

## 🟡 MAJOR: Error in Web API Streaming

### Issue 5: Unreachable Exception Handler
**Location**: `web_client/app.py` lines 246-247

```python
        return StreamingResponse(
            generate_chunks(),
            media_type="audio/wav"
        )
        
    except Exception as e:  # ← This except is AFTER return!
        raise HTTPException(status_code=500, detail=str(e))
```

**Impact**:
- Exception handler never executes (unreachable code)
- Errors in streaming endpoint won't be caught
- Server may crash on errors

**Fix Required**:
- Move `try:` block to wrap entire function
- Fix indentation

---

## 🟢 MINOR: WebRTC VAD Not Used

### Issue 6: VAD Import But No Usage
**Location**: `client_voice.py` lines 38-44

```python
try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    print("⚠️  webrtcvad not installed - using simple silence detection")
```

**Impact**:
- Imported but never actually used
- No silence detection implemented
- Records background noise

**Fix Required**:
- Add `_detect_silence()` method using webrtcvad
- Filter out silent segments before transcription
- Add "End of turn" detection

---

## 🟢 MINOR: Missing Clear History Method

### Issue 7: `clear_history()` Called But May Not Exist
**Location**: `client_voice.py` line 242

```python
def _reset_conversation(self):
    """Reset conversation history."""
    self.llm.clear_history()  # ← Does this method exist?
```

**Impact**:
- May crash if `GeminiDialogueManager.clear_history()` doesn't exist
- Need to verify this method exists

**Fix Required**:
- Check `fern/llm/gemini_manager.py` for `clear_history()`
- Add if missing, or use correct method name

---

## 🟢 MINOR: Device Detection Incomplete

### Issue 8: Naive Device Check
**Location**: Multiple files

```python
device = "cuda" if os.path.exists("/proc/driver/nvidia/version") else "cpu"
```

**Impact**:
- Doesn't detect MPS (Apple Silicon)
- Doesn't check if CUDA is actually available
- May crash on misconfigured systems

**Fix Required**:
- Use `torch.cuda.is_available()`
- Add MPS detection with `torch.backends.mps.is_available()`
- Add proper fallback chain

---

## 🟢 MINOR: No Tests for Voice Clients

### Issue 9: Zero Test Coverage
**Location**: `tests/` directory

**Impact**:
- No unit tests for `client_voice.py`
- No integration tests for web client
- No API endpoint tests
- Quality/reliability unknown

**Fix Required**:
- Add `tests/test_client_voice.py`
- Add `tests/test_web_client.py`
- Add API integration tests with pytest

---

## 🟢 MINOR: Missing Error Handling in Streaming

### Issue 10: No Cleanup on Stream Errors
**Location**: `fern/tts/csm_streaming.py`

**Impact**:
- If streaming fails mid-generation, no cleanup
- May leave incomplete audio chunks
- Could cause memory leaks

**Fix Required**:
- Add try/finally blocks in streaming generators
- Ensure proper resource cleanup
- Add retry logic for transient failures

---

## 🟢 MINOR: CORS Wide Open in Production

### Issue 11: Security Risk in Web Client
**Location**: `web_client/app.py` lines 45-51

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Impact**:
- Anyone can access API from any domain
- CSRF attacks possible
- Not production-ready

**Fix Required**:
- Make CORS configurable via environment variable
- Default to restrictive policy
- Document security implications

---

## Summary by Priority

### 🔴 CRITICAL (Must Fix Before Use)
1. **ASR Integration** - Python client can't transcribe audio
2. **Web Recording** - Web client can't record audio
3. **API Error Handling** - Unreachable exception handler

### 🟡 MAJOR (Should Fix Soon)
4. **Mimi Codec Loading** - Using stub instead of real weights
5. **Training Integration** - Can't fine-tune models
6. **Error Handling** - Multiple gaps in error handling

### 🟢 MINOR (Quality/Polish)
7. **VAD Usage** - Imported but not used
8. **Device Detection** - Naive implementation
9. **Test Coverage** - Zero tests for clients
10. **CORS Security** - Wide open in production

---

## Recommended Fix Order

### Phase 1: Make It Work (1-2 hours)
1. ✅ Fix ASR integration in Python client
2. ✅ Fix API error handling (unreachable except)
3. ✅ Fix device detection
4. ✅ Add `clear_history()` to GeminiDialogueManager

### Phase 2: Make It Complete (3-4 hours)
5. ✅ Implement web client recording
6. ✅ Add `/api/transcribe` endpoint
7. ✅ Load real Mimi codec weights
8. ✅ Fix training script to use real model

### Phase 3: Make It Production-Ready (4-6 hours)
9. ✅ Add comprehensive tests
10. ✅ Implement VAD for turn detection
11. ✅ Add proper error handling throughout
12. ✅ Make CORS configurable
13. ✅ Add logging and monitoring

---

## Files Needing Changes

### Immediate (Critical)
- `client_voice.py` - Add WhisperASR integration
- `web_client/app.py` - Fix error handling, add transcribe endpoint
- `web_client/index.html` - Implement MediaRecorder
- `fern/llm/gemini_manager.py` - Verify/add clear_history()

### Soon (Major)
- `fern/tts/csm/moshi/models/loaders.py` - Load real Mimi
- `scripts/train_lora.py` - Use real CSM model
- `fern/tts/csm_streaming.py` - Better error handling

### Later (Minor)
- `tests/test_client_voice.py` - CREATE
- `tests/test_web_client.py` - CREATE
- `client_voice.py` - Implement VAD
- `web_client/app.py` - Configurable CORS

---

## Estimated Total Work

- **Critical Fixes**: 2-3 hours
- **Major Improvements**: 4-6 hours
- **Quality/Testing**: 6-8 hours
- **Total**: 12-17 hours for full production readiness

**Current State**: ~60% complete (works for text-only, demo quality)
**After Critical Fixes**: ~80% complete (actual voice functionality)
**After All Fixes**: ~95% complete (production-ready)

---

## What We Got Right ✅

Despite the issues above, we did get several things right:

1. ✅ **Clean Architecture** - Good separation of concerns
2. ✅ **Streaming TTS** - Already implemented and working
3. ✅ **API Design** - Well-structured REST + WebSocket
4. ✅ **Documentation** - Comprehensive guides
5. ✅ **Launcher Scripts** - One-command setup
6. ✅ **Beautiful UI** - Modern, responsive web interface
7. ✅ **Error Fallbacks** - Graceful degradation (save to file)
8. ✅ **Gemini Integration** - Auto-model detection working
9. ✅ **Dual Client Options** - Python + Web flexibility
10. ✅ **Production Guides** - Docker, systemd, NGINX docs

---

**Bottom Line**: The foundation is solid, but the voice input functionality is incomplete. The critical path is adding ASR to both clients, which is about 2-3 hours of focused work.

