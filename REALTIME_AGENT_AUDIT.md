# Real-Time Agent Audit Report

## Critical Issues Found

### 1. **API Mismatch in `realtime_agent_advanced.py`** ❌ CRITICAL

**Location:** Line 241-244

**Current Code:**
```python
is_complete = self.turn_detector.is_turn_complete(
    recent_audio_frames=frame_arrays,
    conversation_context=context
)
```

**Problem:**
- `HybridTurnDetector.is_turn_complete()` expects 3 arguments:
  - `audio_buffer: np.ndarray`
  - `transcript: str`
  - `conversation_history: List[Dict[str, str]]`
  
- But we're calling it with:
  - `recent_audio_frames` (wrong name, wrong type - list vs single array)
  - `conversation_context` (wrong name)
  - Missing `transcript` argument entirely

**Fix Required:**
```python
# Need to:
# 1. Concatenate recent_audio_frames into single array
# 2. Get current partial transcript
# 3. Pass conversation history
audio_buffer = np.concatenate(frame_arrays)
is_complete = self.turn_detector.is_turn_complete(
    audio_buffer=audio_buffer,
    transcript=self.conversation.last_user_text or "",
    conversation_history=context
)
```

---

### 2. **Missing `get_silence_duration` Method** ❌ CRITICAL

**Location:** `fern/vad/semantic_turn_detector.py`, line 354

**Problem:**
`HybridTurnDetector.is_turn_complete()` calls:
```python
silence_duration = self.vad.get_silence_duration(audio_buffer)
```

But `VADDetector` (from `fern/asr/vad_detector.py`) does **NOT** have a `get_silence_duration` method.

Available methods are:
- `is_speech(audio_frame)`
- `filter_silence(audio, padding_ms)`
- `detect_end_of_turn(audio_frames, silence_duration_ms)`

**Fix Required:**
Add `get_silence_duration()` method to `VADDetector`:

```python
def get_silence_duration(self, audio: np.ndarray) -> float:
    """
    Calculate duration of trailing silence in audio.
    
    Args:
        audio: Audio array
    
    Returns:
        Duration of silence in seconds
    """
    # Split into frames
    num_frames = len(audio) // self.frame_size
    silence_frames = 0
    
    # Count trailing silence frames
    for i in range(num_frames - 1, -1, -1):
        start = i * self.frame_size
        end = start + self.frame_size
        frame = audio[start:end]
        
        if self.is_speech(frame):
            break
        silence_frames += 1
    
    # Convert to seconds
    return (silence_frames * self.frame_duration_ms) / 1000.0
```

---

### 3. **Unused `StreamingTTS` Import** ⚠️ MINOR

**Location:** Both `realtime_agent.py` and `realtime_agent_advanced.py`

**Problem:**
Both files import `StreamingTTS`:
```python
from fern.tts.csm_streaming import StreamingTTS
```

But neither file actually uses it. The agents use `RealCSMTTS` directly and call `.synthesize()`.

**Fix Options:**

**Option A:** Remove the unused import (cleaner)
```python
# Remove this line
from fern.tts.csm_streaming import StreamingTTS
```

**Option B:** Actually use it for streaming (better UX)
```python
# In _load_models():
self.tts = RealCSMTTS(device=self.device, sample_rate=24000)
self.streaming_tts = StreamingTTS(self.tts, chunk_duration_ms=200)

# In _response_worker():
for chunk in self.streaming_tts.synthesize_stream(response):
    sd.play(chunk, samplerate=24000)
    sd.wait()
```

---

### 4. **No Error Handling for Model Loading** ⚠️ MAJOR

**Location:** `realtime_agent.py` line 95, `realtime_agent_advanced.py` line 141

**Problem:**
If models fail to load (e.g., missing models, GPU OOM), the program crashes with no helpful message.

```python
def _load_models(self, api_key: str):
    print(f"{Colors.BLUE}⚙️  Loading AI models...{Colors.END}")
    
    # No try/except around this!
    self.vad = VADDetector(sample_rate=self.sample_rate)
    self.asr = WhisperASR(model_name="large-v3-turbo", device=self.device)
    # ...
```

**Fix Required:**
```python
def _load_models(self, api_key: str):
    print(f"{Colors.BLUE}⚙️  Loading AI models...{Colors.END}")
    
    try:
        # VAD
        print(f"  {Colors.DIM}[1/4] VAD...{Colors.END}", end=" ")
        self.vad = VADDetector(sample_rate=self.sample_rate)
        print(f"{Colors.GREEN}✓{Colors.END}")
        
        # ASR
        print(f"  {Colors.DIM}[2/4] Whisper ASR...{Colors.END}", end=" ")
        self.asr = WhisperASR(model_name="large-v3-turbo", device=self.device)
        print(f"{Colors.GREEN}✓{Colors.END}")
        
        # ... rest of models
        
    except FileNotFoundError as e:
        print(f"\n{Colors.RED}✗ Model files not found: {e}{Colors.END}")
        print(f"{Colors.YELLOW}Run: python scripts/download_models.py{Colors.END}")
        raise
    
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"\n{Colors.RED}✗ GPU out of memory{Colors.END}")
            print(f"{Colors.YELLOW}Try: device='cpu' or smaller model{Colors.END}")
        raise
    
    except Exception as e:
        print(f"\n{Colors.RED}✗ Model loading failed: {e}{Colors.END}")
        raise
```

---

### 5. **Race Condition in Audio Buffer** ⚠️ MAJOR

**Location:** `realtime_agent.py` line 251-259, `realtime_agent_advanced.py` similar

**Problem:**
The audio callback and processing thread both access `self.audio_buffer`:

```python
def _audio_callback(self, indata, frames, time_info, status):
    if self.is_listening and not self.is_speaking:
        self.audio_buffer.append(indata.copy())  # No lock!

def _get_buffered_audio(self) -> np.ndarray:
    with self.audio_lock:
        if not self.audio_buffer:
            return np.array([], dtype=np.float32)
        
        audio = np.concatenate([buf.flatten() for buf in self.audio_buffer])
        self.audio_buffer = []  # With lock
        return audio
```

`_audio_callback` doesn't use the lock, but `_get_buffered_audio` does!

**Fix Required:**
```python
def _audio_callback(self, indata, frames, time_info, status):
    if self.is_listening and not self.is_speaking:
        with self.audio_lock:  # ADD THIS
            self.audio_buffer.append(indata.copy())
```

---

### 6. **Missing Transcription in Turn Detection** ⚠️ MAJOR

**Location:** `realtime_agent_advanced.py`, line 222-249

**Problem:**
The `_detect_turn_completion()` method uses `HybridTurnDetector` which needs a transcript, but it only has the conversation history, not the current partial transcript that's being buffered.

The agent accumulates audio in `self.audio_buffer` but never transcribes it incrementally. By the time we check turn completion, we don't know what the user has said so far!

**Fix Required:**
Either:

**Option A:** Disable semantic detection in turn completion check (use VAD only)
```python
def _detect_turn_completion(self) -> bool:
    # Just use VAD for turn detection
    # Save semantic for final processing
    with self.audio_lock:
        if len(self.audio_buffer) < 30:
            return False
        recent_frames = self.audio_buffer[-30:]
        frame_arrays = [frame.flatten() for frame in recent_frames]
        return self.vad.detect_end_of_turn(frame_arrays, silence_duration_ms=700)
```

**Option B:** Add incremental transcription (complex, adds latency)
```python
def _detect_turn_completion(self) -> bool:
    with self.audio_lock:
        if len(self.audio_buffer) < 30:
            return False
        
        # Get recent audio
        recent_audio = np.concatenate([buf.flatten() for buf in self.audio_buffer[-30:]])
        
        # Quick transcription of recent audio
        partial_transcript = self.asr.transcribe(recent_audio, self.sample_rate)
        
        # Check with hybrid detector
        if self.turn_detector:
            context = [
                {"role": turn["role"], "content": turn["text"]}
                for turn in self.conversation.get_context()
            ]
            
            return self.turn_detector.is_turn_complete(
                audio_buffer=recent_audio,
                transcript=partial_transcript,
                conversation_history=context
            )
```

---

### 7. **No Graceful Shutdown** ⚠️ MINOR

**Location:** Both agents, signal handler

**Problem:**
When Ctrl+C is pressed, the agents set `should_stop = True` but don't wait for threads to finish or close audio streams properly.

**Fix Required:**
```python
def _signal_handler(self, sig, frame):
    print(f"\n\n{Colors.YELLOW}🛑 Shutting down gracefully...{Colors.END}")
    self.should_stop = True
    
    # Stop audio stream
    if hasattr(self, 'audio_stream'):
        self.audio_stream.stop()
        self.audio_stream.close()
    
    # Wait for worker threads
    if hasattr(self, 'transcription_thread'):
        self.transcription_thread.join(timeout=2.0)
    if hasattr(self, 'response_thread'):
        self.response_thread.join(timeout=2.0)
    
    print(f"{Colors.GREEN}✓ Cleanup complete{Colors.END}")
    sys.exit(0)
```

---

## Summary

### Critical (Must Fix Before Use)
1. ✅ API mismatch in `HybridTurnDetector` call
2. ✅ Missing `get_silence_duration()` method in `VADDetector`

### Major (Should Fix Soon)
3. ✅ No error handling for model loading
4. ✅ Race condition in audio buffer
5. ✅ Missing transcription in turn detection

### Minor (Nice to Have)
6. ✅ Unused `StreamingTTS` import
7. ✅ No graceful shutdown

---

## Recommended Fix Order

1. **Add `get_silence_duration()` to `VADDetector`** (fixes critical bug)
2. **Fix API call in `realtime_agent_advanced.py`** (fixes critical bug)
3. **Add lock to `_audio_callback`** (prevents data corruption)
4. **Simplify turn detection to VAD-only** (removes dependency on missing transcript)
5. **Add error handling to model loading** (better UX)
6. **Add graceful shutdown** (clean exit)
7. **Remove or use `StreamingTTS`** (code cleanup)

---

## Testing After Fixes

```bash
# 1. Fix audio deps
./scripts/fix_audio_deps.sh

# 2. Test basic agent
export GOOGLE_API_KEY="your-key"
python realtime_agent.py

# 3. Test advanced agent
python realtime_agent_advanced.py

# 4. Test error handling
# Remove a model file and verify graceful error
rm -rf models/csm-1b/model.safetensors
python realtime_agent.py
# Should see: "Model files not found" with helpful message

# 5. Test interruption
# While agent is speaking, start talking
# Should interrupt and respond to new input
```

---

**Last Updated:** 2025-11-08
**Status:** Ready to fix
**Estimated Fix Time:** 30-45 minutes

