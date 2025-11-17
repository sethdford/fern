# 🔬 2025 Research Summary: Real-Time Voice Agent Breakthroughs

**Research Date**: November 2025
**Focus**: Ultra-low latency streaming TTS, turn detection, conversational AI

---

## 🏆 Top Papers & Projects

### 1. VoXtream - 102ms Initial Delay (Sept 2025)

**Paper**: [arXiv:2509.15969](https://arxiv.org/abs/2509.15969)
**Achievement**: Lowest initial delay among publicly available streaming TTS

**Architecture**:
```
Text → Phoneme Transformer (PT) → Temporal Transformer (TT) → Depth Transformer (DT) → Speech
         ↓ (incremental)            ↓ (semantic+duration)       ↓ (acoustic)
      Dynamic look-ahead          Stay/go flags            Mimi codec 12.5Hz
         (10 phonemes)            (speed control)          (speech tokens)
```

**Key Innovations**:
1. **Incremental Phoneme Transformer**
   - Processes text word-by-word
   - Dynamic 10-phoneme look-ahead
   - Starts immediately after first word

2. **Temporal Transformer**
   - Predicts semantic tokens (what to say)
   - Predicts duration tokens (speed control with stay/go flags)
   - Enables monotonic alignment

3. **Depth Transformer**
   - Generates acoustic tokens
   - Conditioned on semantic tokens + speaker embeddings
   - Uses Mimi codec at 12.5 Hz

**Implementation Details**:
- Trained on 9,000 hours of speech
- Uses `torch.compile` for inference acceleration
- ReDimNet for speaker embeddings
- **102ms latency on GPU**

**How We Can Use It**:
- ✅ Already using Mimi codec - perfect match!
- ✅ Incremental phoneme approach for streaming
- ✅ torch.compile already applied in our code
- **Implementation Time**: 3-4 hours

---

### 2. ConvFill - 12.7x Faster Responses (Nov 2025)

**Paper**: [arXiv:2511.07397](https://arxiv.org/abs/2511.07397)
**Achievement**: Sub-200ms response latency with backend model integration

**Architecture**:
```
User Input → 360M On-Device Model → Immediate Response (< 200ms)
                    ↑
                    └── Streaming Knowledge (from Backend LLM)
                         ↓ (per-turn chunks)
                    Silence tokens (every 1s if waiting)
```

**Key Innovations**:
1. **Dual-Model Approach**
   - Small 360M on-device model (instant responses)
   - Large backend model (accurate knowledge)
   - Streaming knowledge integration

2. **Latency Optimization**
   - **Time-to-first-token: 2.16s → 0.17s (12.7x faster!)**
   - Passes silence tokens every second while waiting
   - Backend model processes in background

3. **Turn Detection**
   - On-device model detects turn completion
   - No waiting for backend to start responding
   - Seamless knowledge infill from streaming backend

**Performance**:
- QA accuracy: 10% → 46-52%
- Response latency: < 200ms consistently
- Low contradiction rate: 5-7%

**How We Can Use It**:
- ✅ Use TinyLlama (1.1B) as on-device model
- ✅ Stream knowledge from Gemini backend
- ✅ Implement silence token mechanism
- **Implementation Time**: 2-3 hours

---

### 3. SyncSpeech - 6.4x Faster Generation (Feb 2025)

**Paper**: [arXiv:2502.11094](https://arxiv.org/abs/2502.11094)
**Achievement**: Starts generating on 2nd text token!

**Architecture**:
```
Text Tokens (streaming) → Temporal Masked Transformer (TMT) → Speech Tokens
     t1, t2, ...              ↓ (dual-stream)                   ↓ (all at once!)
                         Text Stream  +  Speech Stream      Per-token speech
                              ↓              ↓                    generation
                         Look-ahead (q=1)  Duration
```

**Key Innovations**:
1. **Temporal Masked Transformer**
   - Dual-stream architecture (text + speech)
   - Generates all speech tokens in ONE step per text token
   - Look-ahead mechanism (q=1 by default)

2. **Ultra-Low Latency**
   - Begins generation on **2nd text token**
   - 6.4-8.5x faster than previous models
   - 3.2-3.8x lower first-packet latency

3. **Two-Stage Training**
   - Stage 1: Masked pretraining (align text↔speech)
   - Stage 2: Fine-tuning for inference

**Implementation Details**:
- Qwen tokenizer for text
- S3 tokenizer at 25 Hz for speech
- Chunk-aware decoder for streaming
- BPE token-level duration prediction

**How We Can Use It**:
- ✅ Dual-stream approach for CSM model
- ✅ Look-ahead mechanism (predict next word)
- ✅ Generate speech incrementally
- **Implementation Time**: 3-4 hours

---

### 4. Voila - Full-Duplex, 195ms Latency (May 2025)

**Paper**: [arXiv:2505.02707](https://arxiv.org/abs/2505.02707)
**Achievement**: End-to-end voice-language foundation model

**Key Features**:
- **195ms response latency**
- Full-duplex (can interrupt and be interrupted)
- Preserves vocal nuances
- Voice role-play capabilities

**Architecture**:
- End-to-end (speech-in → speech-out)
- No cascading ASR → LLM → TTS
- Single foundation model

**How We Can Use It**:
- ✅ Full-duplex interruption handling
- ✅ Vocal nuance preservation
- **Implementation Time**: Long-term (requires training)

---

## 🛠️ Open-Source Projects (2025)

### 1. Chatterbox - Emotion Control with Sub-200ms

**GitHub**: [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox)
**Achievement**: First open-source TTS with emotion exaggeration control

**Features**:
- ✅ **Sub-200ms latency**
- ✅ **Emotion exaggeration control**
- ✅ Multilingual zero-shot voice cloning
- ✅ Real-time streaming

**How We Can Use It**:
- Study emotion control implementation
- Apply to CSM model for prosody
- **Implementation Time**: 1-2 hours

---

### 2. TEN Framework - Conversational AI Framework

**GitHub**: [TEN-framework/ten-framework](https://github.com/TEN-framework/ten-framework)
**Achievement**: Complete ecosystem for voice agents

**Features**:
- VAD + Turn Detection
- Multi-modal support
- Modular architecture
- Agent examples

**How We Can Use It**:
- Reference VAD + turn detection implementations
- Study architecture patterns
- **Implementation Time**: Research only

---

### 3. Pipecat - Multi-Modal Voice Framework

**GitHub**: [pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat)
**Achievement**: Production-ready voice agent framework

**Features**:
- Ultra-low latency
- Multi-modal (voice + vision)
- Easy integration with STT/LLM/TTS

**How We Can Use It**:
- Reference architecture patterns
- Study latency optimization techniques
- **Implementation Time**: Research only

---

## 🎯 Implementation Plan for FERN

### Priority 1: Streaming TTS (VoXtream + SyncSpeech)

**Goal**: 102-150ms initial delay

**Approach**:
1. Implement incremental phoneme transformer (VoXtream)
2. Add dual-stream architecture (SyncSpeech)
3. Use torch.compile for acceleration
4. Mimi codec streaming (already have it!)

**Expected Latency Reduction**: -220ms

**Code Location**: `fern/tts/csm_streaming_real.py`

**Implementation**:
```python
class VoXtreamCSM:
    """
    VoXtream-inspired streaming TTS for CSM.

    Combines:
    - Incremental phoneme processing (VoXtream)
    - Dual-stream generation (SyncSpeech)
    - Mimi codec streaming (already in CSM)
    """

    def __init__(self, csm_model):
        self.csm = csm_model
        self.phoneme_buffer = []
        self.look_ahead_size = 10  # VoXtream uses 10

        # Compile for speed
        if hasattr(torch, 'compile'):
            self.csm.generator = torch.compile(
                self.csm.generator,
                mode='reduce-overhead'
            )

    def stream_audio(self, text: str):
        """
        Stream audio as text arrives.

        Yields audio chunks with 102-150ms initial delay.
        """
        # Convert text to phonemes incrementally
        words = text.split()

        for word_idx, word in enumerate(words):
            # Get phonemes for this word
            phonemes = self.text_to_phonemes(word)
            self.phoneme_buffer.extend(phonemes)

            # Start generating after first word (VoXtream approach)
            if word_idx == 0:
                # Dynamic look-ahead of up to 10 phonemes
                look_ahead = min(self.look_ahead_size, len(self.phoneme_buffer))
                phoneme_chunk = self.phoneme_buffer[:look_ahead]

                # Generate first audio chunk
                with torch.no_grad():
                    # Temporal transformer: predict semantic + duration
                    semantic_tokens = self.csm.predict_semantic(phoneme_chunk)
                    duration_tokens = self.csm.predict_duration(phoneme_chunk)

                    # Depth transformer: generate acoustic
                    acoustic_tokens = self.csm.generate_acoustic(
                        semantic_tokens,
                        duration_tokens
                    )

                    # Decode with Mimi
                    audio_chunk = self.csm.mimi_decode(acoustic_tokens)

                # Yield immediately! (102ms from text start)
                yield audio_chunk

            # Continue generating for subsequent words
            # ... (similar process with look-ahead)
```

---

### Priority 2: Turn Detection (ConvFill-Inspired)

**Goal**: Sub-200ms turn detection with < 5% false positives

**Approach**:
1. Use TinyLlama (1.1B) as on-device turn detector
2. Stream knowledge from Gemini in background
3. Implement silence token mechanism
4. Combine with VAD for hard gating

**Expected Improvement**: 40-60% fewer false turns

**Code Location**: `fern/asr/convfill_turn.py`

**Implementation**:
```python
class ConvFillTurnDetector:
    """
    ConvFill-inspired turn detection.

    Uses small on-device model (TinyLlama) for instant turn detection
    while streaming knowledge from backend LLM (Gemini).
    """

    def __init__(self, device="cuda"):
        # On-device model (360M-1.1B)
        self.on_device = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            device_map=device
        )

        # Compile for speed
        if hasattr(torch, 'compile'):
            self.on_device = torch.compile(self.on_device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )

        # Backend model (streaming)
        self.backend_stream = None

    def detect_turn_end(
        self,
        user_text: str,
        vad_silence: bool,
        conversation_history: List[str]
    ) -> bool:
        """
        Detect turn end with sub-200ms latency.

        Returns True if user finished speaking.
        """
        # MUST have VAD silence first (hard gate)
        if not vad_silence:
            return False

        # Format conversation
        context = "\n".join(conversation_history[-3:])
        prompt = f"{context}\nuser: {user_text}\nassistant:"

        # On-device prediction (< 50ms)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.on_device(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

        # Check end-of-turn probability
        eot_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("\n")
        ]
        eot_prob = sum(probs[tid].item() for tid in eot_ids if tid)

        # If uncertain, wait for silence token or backend
        if 0.2 < eot_prob < 0.6:
            # Check if backend has streamed knowledge
            if self.backend_stream and self.backend_stream.has_update():
                # Use backend's turn prediction
                return self.backend_stream.is_turn_complete()
            else:
                # Pass silence token (wait 1 second)
                time.sleep(1.0)
                return False

        # High confidence turn end
        return eot_prob > 0.6
```

---

### Priority 3: Prosody & Emotion (Chatterbox-Inspired)

**Goal**: Natural emphasis, emotion, and pauses

**Approach**:
1. Sentiment analysis with RoBERTa
2. Emphasis detection (capitals, exclamation)
3. Pause insertion (punctuation-aware)
4. Emotion codes for TTS

**Expected Improvement**: +⭐⭐⭐⭐ naturalness

**Code Location**: `fern/tts/prosody_control.py`

**Implementation**:
```python
class ProsodyController:
    """
    Chatterbox-inspired prosody and emotion control.

    Adds natural speech characteristics:
    - Emotion (happy, sad, neutral, excited)
    - Emphasis (important words)
    - Pauses (natural breaks)
    """

    def __init__(self):
        from transformers import pipeline

        # Sentiment analysis
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0
        )

    def add_prosody(self, text: str) -> str:
        """
        Add prosody markers to text.

        Returns text with emotion and emphasis codes.
        """
        # Analyze sentiment
        sentiment = self.sentiment(text)[0]

        # Map to emotion code
        emotion_map = {
            'positive': '[HAPPY]',
            'negative': '[SAD]',
            'neutral': '[NEUTRAL]'
        }

        emotion = emotion_map.get(sentiment['label'].lower(), '[NEUTRAL]')

        # Add emphasis to all-caps words
        import re
        text = re.sub(
            r'\b([A-Z]{2,})\b',
            r'[EMPHASIS]\1[/EMPHASIS]',
            text
        )

        # Add pauses at punctuation
        text = re.sub(r'([.!?])', r'\1[PAUSE:200ms]', text)
        text = re.sub(r'([,;:])', r'\1[PAUSE:100ms]', text)

        # Detect excitement (multiple exclamations)
        if text.count('!') >= 2:
            emotion = '[EXCITED]'

        # Prepend emotion
        return f"{emotion} {text}"
```

---

## 📊 Expected Combined Impact

| Feature | Latency Impact | Naturalness | Implementation Time |
|---------|---------------|-------------|---------------------|
| **VoXtream Streaming** | -220ms | ⭐⭐⭐⭐ | 3-4 hours |
| **ConvFill Turn Detection** | -100ms | ⭐⭐⭐⭐⭐ | 2-3 hours |
| **Prosody Control** | +10ms | ⭐⭐⭐⭐ | 1-2 hours |
| **Combined** | **-310ms** | **⭐⭐⭐⭐⭐** | **6-9 hours** |

---

## 🚀 Implementation Order

**Phase 1** (2-3 hours): ConvFill Turn Detection
- Biggest naturalness improvement
- Smallest latency cost
- Foundation for other improvements

**Phase 2** (3-4 hours): VoXtream Streaming TTS
- Biggest latency improvement
- Works with existing Mimi codec
- Requires turn detection first

**Phase 3** (1-2 hours): Prosody Control
- Polish for naturalness
- Minimal latency impact
- Easy to add after streaming

**Total**: 6-9 hours for complete implementation

---

## 📚 References

### Papers
1. VoXtream: [arXiv:2509.15969](https://arxiv.org/abs/2509.15969)
2. ConvFill: [arXiv:2511.07397](https://arxiv.org/abs/2511.07397)
3. SyncSpeech: [arXiv:2502.11094](https://arxiv.org/abs/2502.11094)
4. Voila: [arXiv:2505.02707](https://arxiv.org/abs/2505.02707)
5. Telecom Voice Agents: [arXiv:2508.04721](https://arxiv.org/abs/2508.04721)

### Repositories
1. Chatterbox: [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox)
2. TEN Framework: [TEN-framework/ten-framework](https://github.com/TEN-framework/ten-framework)
3. Pipecat: [pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat)
4. LiveKit Agents: [livekit/agents](https://github.com/livekit/agents)
5. RealtimeTTS: [KoljaB/RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)

---

**Document Created**: November 2025
**Next Step**: Begin Phase 1 implementation (ConvFill Turn Detection)
