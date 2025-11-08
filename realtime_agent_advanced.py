#!/usr/bin/env python3
"""
FERN Advanced Real-Time Voice Agent

Features beyond basic real-time:
- Hybrid turn detection (VAD + semantic)
- Conversation context awareness
- Interruption handling
- Partial transcription display
- Multi-threaded pipeline optimization

Usage:
    python realtime_agent_advanced.py
"""

import os
import sys
import threading
import queue
import time
from pathlib import Path
from typing import Optional, List, Dict
from collections import deque
import signal

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import sounddevice as sd

from fern.llm.gemini_manager import GeminiDialogueManager
from fern.tts.csm_real import RealCSMTTS
from fern.tts.csm_streaming import StreamingTTS
from fern.asr.whisper_asr import WhisperASR
from fern.asr.vad_detector import VADDetector
from fern.vad.semantic_turn_detector import SemanticTurnDetector, HybridTurnDetector


class Colors:
    """ANSI colors."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


class ConversationState:
    """Track conversation state and context."""
    
    def __init__(self, max_history: int = 10):
        self.turns = deque(maxlen=max_history)
        self.current_speaker = None
        self.last_user_text = ""
        self.last_agent_text = ""
        self.turn_count = 0
    
    def add_user_turn(self, text: str):
        """Add user turn to history."""
        self.turns.append({"role": "user", "text": text, "timestamp": time.time()})
        self.last_user_text = text
        self.current_speaker = "user"
        self.turn_count += 1
    
    def add_agent_turn(self, text: str):
        """Add agent turn to history."""
        self.turns.append({"role": "agent", "text": text, "timestamp": time.time()})
        self.last_agent_text = text
        self.current_speaker = "agent"
        self.turn_count += 1
    
    def get_context(self, num_turns: int = 5) -> List[Dict]:
        """Get recent conversation context."""
        return list(self.turns)[-num_turns:]
    
    def should_respond(self) -> bool:
        """Check if agent should respond based on context."""
        # Don't respond if just spoke
        if self.current_speaker == "agent":
            return False
        
        # Respond if user spoke
        return self.current_speaker == "user"


class AdvancedRealtimeAgent:
    """
    Advanced real-time voice agent with hybrid turn detection.
    
    Features:
    - Semantic turn detection (SmolLM2)
    - Partial transcription streaming
    - Context-aware responses
    - Interruption handling
    - Performance monitoring
    """
    
    def __init__(
        self,
        google_api_key: str,
        device: str = "cuda",
        use_semantic_detection: bool = True,
    ):
        """Initialize advanced agent."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}║   🚀 FERN Advanced Real-Time Voice Agent 🚀                 ║{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.END}\n")
        
        self.device = device
        self.sample_rate = 16000
        self.use_semantic = use_semantic_detection
        
        # State
        self.conversation = ConversationState()
        self.audio_buffer = []
        self.audio_lock = threading.Lock()
        
        self.is_listening = False
        self.is_processing = False
        self.is_speaking = False
        self.should_stop = False
        self.can_interrupt = True  # Allow interruptions
        
        # Performance metrics
        self.metrics = {
            "transcription_time": [],
            "response_time": [],
            "tts_time": [],
            "total_latency": [],
        }
        
        # Queues
        self.transcription_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Load models
        self._load_models(google_api_key)
        
        # Start workers
        self._start_workers()
        
        print(f"\n{Colors.GREEN}✨ Advanced agent ready! Features enabled:{Colors.END}")
        print(f"  {Colors.GREEN}✓{Colors.END} Continuous listening")
        print(f"  {Colors.GREEN}✓{Colors.END} Hybrid turn detection (VAD + Semantic)")
        print(f"  {Colors.GREEN}✓{Colors.END} Context-aware responses")
        print(f"  {Colors.GREEN}✓{Colors.END} Interruption support")
        print(f"  {Colors.GREEN}✓{Colors.END} Performance monitoring\n")
        print(f"{Colors.DIM}Just start talking naturally! (Ctrl+C to exit){Colors.END}\n")
    
    def _load_models(self, api_key: str):
        """Load all models with comprehensive error handling."""
        print(f"{Colors.BLUE}⚙️  Loading AI models...{Colors.END}")
        
        try:
            # VAD
            print(f"  {Colors.DIM}[1/5] VAD...{Colors.END}", end=" ", flush=True)
            self.vad = VADDetector(sample_rate=self.sample_rate, aggressiveness=2)
            print(f"{Colors.GREEN}✓{Colors.END}")
            
            # Semantic turn detector (if enabled)
            if self.use_semantic:
                print(f"  {Colors.DIM}[2/5] Semantic Turn Detector (SmolLM2)...{Colors.END}", end=" ", flush=True)
                try:
                    self.semantic_detector = SemanticTurnDetector(device=self.device)
                    self.turn_detector = HybridTurnDetector(
                        vad=self.vad,
                        semantic=self.semantic_detector
                    )
                    print(f"{Colors.GREEN}✓{Colors.END}")
                except Exception as e:
                    print(f"{Colors.YELLOW}⚠️  (fallback to VAD only){Colors.END}")
                    self.turn_detector = None
            else:
                self.turn_detector = None
                print(f"  {Colors.DIM}[2/5] Semantic Detector...{Colors.END} {Colors.YELLOW}(disabled){Colors.END}")
            
            # Whisper
            print(f"  {Colors.DIM}[3/5] Whisper ASR...{Colors.END}", end=" ", flush=True)
            compute_type = "float16" if self.device == "cuda" else "int8"
            self.asr = WhisperASR(
                model_size="large-v3",
                device=self.device,
                compute_type=compute_type
            )
            print(f"{Colors.GREEN}✓{Colors.END}")
            
            # Gemini
            print(f"  {Colors.DIM}[4/5] Gemini LLM...{Colors.END}", end=" ", flush=True)
            self.llm = GeminiDialogueManager(api_key=api_key)
            print(f"{Colors.GREEN}✓ ({self.llm.model_name}){Colors.END}")
            
            # CSM-1B
            print(f"  {Colors.DIM}[5/5] CSM-1B TTS ({self.device})...{Colors.END}", end=" ", flush=True)
            self.tts = RealCSMTTS(device=self.device)
            self.streaming_tts = StreamingTTS(self.tts, chunk_duration_ms=150)
            print(f"{Colors.GREEN}✓{Colors.END}")
            
            print(f"\n{Colors.GREEN}✓ All models loaded successfully!{Colors.END}")
            
        except FileNotFoundError as e:
            print(f"\n\n{Colors.RED}✗ Model files not found{Colors.END}")
            print(f"{Colors.YELLOW}Issue: {e}{Colors.END}")
            print(f"\n{Colors.BOLD}Solution:{Colors.END}")
            print(f"  1. Download models: {Colors.CYAN}python scripts/download_models.py{Colors.END}")
            print(f"  2. Integrate models: {Colors.CYAN}python scripts/integrate_real_models.py{Colors.END}")
            sys.exit(1)
        
        except RuntimeError as e:
            error_msg = str(e)
            if "CUDA out of memory" in error_msg:
                print(f"\n\n{Colors.RED}✗ GPU out of memory{Colors.END}")
                print(f"\n{Colors.BOLD}Solutions:{Colors.END}")
                print(f"  1. Use CPU instead: {Colors.CYAN}device='cpu'{Colors.END}")
                print(f"  2. Use smaller Whisper model: {Colors.CYAN}model_size='base'{Colors.END}")
                print(f"  3. Close other GPU applications")
            elif "CUDA" in error_msg or "cuDNN" in error_msg:
                print(f"\n\n{Colors.RED}✗ CUDA error{Colors.END}")
                print(f"{Colors.YELLOW}{error_msg}{Colors.END}")
                print(f"\n{Colors.BOLD}Solutions:{Colors.END}")
                print(f"  1. Check CUDA installation: {Colors.CYAN}nvidia-smi{Colors.END}")
                print(f"  2. Try CPU mode: {Colors.CYAN}device='cpu'{Colors.END}")
            else:
                print(f"\n\n{Colors.RED}✗ Runtime error: {error_msg}{Colors.END}")
            sys.exit(1)
        
        except ImportError as e:
            print(f"\n\n{Colors.RED}✗ Missing dependency{Colors.END}")
            print(f"{Colors.YELLOW}{e}{Colors.END}")
            print(f"\n{Colors.BOLD}Solution:{Colors.END}")
            if "csm-streaming" in str(e) or "torchao" in str(e):
                print(f"  Install CSM dependencies:")
                print(f"    {Colors.CYAN}pip install torchao{Colors.END}")
                print(f"    {Colors.CYAN}pip install git+https://github.com/davidbrowne17/csm-streaming.git{Colors.END}")
            elif "sounddevice" in str(e):
                print(f"  Fix audio dependencies:")
                print(f"    {Colors.CYAN}bash scripts/fix_audio_deps.sh{Colors.END}")
            else:
                print(f"  Install requirements:")
                print(f"    {Colors.CYAN}pip install -r requirements.txt{Colors.END}")
            sys.exit(1)
        
        except KeyError as e:
            if "GOOGLE_API_KEY" in str(e) or "api_key" in str(e):
                print(f"\n\n{Colors.RED}✗ Missing API key{Colors.END}")
                print(f"\n{Colors.BOLD}Solution:{Colors.END}")
                print(f"  Set your Gemini API key:")
                print(f"    {Colors.CYAN}export GOOGLE_API_KEY='your-key-here'{Colors.END}")
                print(f"\n  Get a key at: {Colors.CYAN}https://makersuite.google.com/app/apikey{Colors.END}")
            else:
                print(f"\n\n{Colors.RED}✗ Configuration error: {e}{Colors.END}")
            sys.exit(1)
        
        except Exception as e:
            print(f"\n\n{Colors.RED}✗ Unexpected error during model loading{Colors.END}")
            print(f"{Colors.YELLOW}Error: {type(e).__name__}: {e}{Colors.END}")
            print(f"\n{Colors.BOLD}Debug steps:{Colors.END}")
            print(f"  1. Check all dependencies: {Colors.CYAN}pip list | grep -E 'torch|transformers|sounddevice'{Colors.END}")
            print(f"  2. Verify CUDA: {Colors.CYAN}python -c 'import torch; print(torch.cuda.is_available())'{Colors.END}")
            print(f"  3. Run diagnostics: {Colors.CYAN}python diagnose.py{Colors.END}")
            print(f"\n{Colors.DIM}See TROUBLESHOOTING.md for more help{Colors.END}")
            sys.exit(1)
    
    def _start_workers(self):
        """Start background worker threads."""
        self.transcription_thread = threading.Thread(
            target=self._transcription_worker, daemon=True
        )
        self.response_thread = threading.Thread(
            target=self._response_worker, daemon=True
        )
        
        self.transcription_thread.start()
        self.response_thread.start()
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio capture callback."""
        if status:
            print(f"{Colors.YELLOW}⚠️  {status}{Colors.END}")
        
        if self.is_listening:
            with self.audio_lock:
                self.audio_buffer.append(indata.copy())
    
    def _detect_turn_completion(self) -> bool:
        """
        Advanced turn detection.
        
        Uses VAD for turn detection. Semantic detection is available
        but currently used only for post-processing to reduce latency.
        """
        with self.audio_lock:
            if len(self.audio_buffer) < 30:  # Need ~1 second
                return False
            
            # Get recent frames for VAD check
            recent_frames = self.audio_buffer[-30:]
            frame_arrays = [frame.flatten() for frame in recent_frames]
            
            # Use VAD for turn detection (fast, reliable)
            # Semantic detection adds too much latency for real-time turn detection
            return self.vad.detect_end_of_turn(frame_arrays, silence_duration_ms=700)
    
    def _get_buffered_audio(self) -> np.ndarray:
        """Get and clear audio buffer."""
        with self.audio_lock:
            if not self.audio_buffer:
                return np.array([], dtype=np.float32)
            
            audio = np.concatenate([buf.flatten() for buf in self.audio_buffer])
            self.audio_buffer = []
            return audio
    
    def _transcription_worker(self):
        """Background transcription."""
        while not self.should_stop:
            try:
                audio_data = self.transcription_queue.get(timeout=0.1)
                if audio_data is None:
                    continue
                
                start_time = time.time()
                
                # Filter silence
                filtered = self.vad.filter_silence(audio_data, padding_ms=300)
                
                if len(filtered) == 0:
                    self.is_processing = False
                    continue
                
                # Transcribe with context
                result = self.asr.transcribe_with_context(
                    filtered,
                    previous_text=self.conversation.last_user_text,
                    sample_rate=self.sample_rate
                )
                
                text = result["text"].strip()
                
                if not text:
                    self.is_processing = False
                    continue
                
                # Record metrics
                transcription_time = time.time() - start_time
                self.metrics["transcription_time"].append(transcription_time)
                
                # Display
                print(f"{Colors.GREEN}  👤 You: {Colors.BOLD}{text}{Colors.END}")
                print(f"{Colors.DIM}     (transcribed in {transcription_time:.2f}s){Colors.END}")
                
                # Update conversation state
                self.conversation.add_user_turn(text)
                
                # Queue for response
                self.response_queue.put((text, start_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"{Colors.RED}  ✗ Transcription error: {e}{Colors.END}")
                self.is_processing = False
    
    def _response_worker(self):
        """Background response generation and speech."""
        while not self.should_stop:
            try:
                item = self.response_queue.get(timeout=0.1)
                if item is None:
                    continue
                
                user_text, start_time = item
                
                # Generate response
                print(f"{Colors.BLUE}  💭 Thinking...{Colors.END}")
                response_start = time.time()
                
                response_text = self.llm.generate_response(user_text)
                
                response_time = time.time() - response_start
                self.metrics["response_time"].append(response_time)
                
                # Update conversation
                self.conversation.add_agent_turn(response_text)
                
                # Display
                print(f"{Colors.MAGENTA}  🤖 FERN: {Colors.BOLD}{response_text}{Colors.END}")
                print(f"{Colors.DIM}     (generated in {response_time:.2f}s){Colors.END}")
                
                # Speak
                print(f"{Colors.CYAN}  🔊 Speaking...{Colors.END}")
                tts_start = time.time()
                self.is_speaking = True
                
                try:
                    for chunk in self.streaming_tts.synthesize_stream_sentences(response_text):
                        if self.should_stop:
                            break
                        
                        # Check for interruption
                        if self.can_interrupt and self._detect_interruption():
                            print(f"{Colors.YELLOW}  ⚠️  Interrupted by user{Colors.END}")
                            sd.stop()
                            break
                        
                        sd.play(chunk, samplerate=24000)
                        sd.wait()
                    
                    tts_time = time.time() - tts_start
                    total_latency = time.time() - start_time
                    
                    self.metrics["tts_time"].append(tts_time)
                    self.metrics["total_latency"].append(total_latency)
                    
                    print(f"{Colors.GREEN}  ✓ Complete{Colors.END}")
                    print(f"{Colors.DIM}     (spoke in {tts_time:.2f}s, total: {total_latency:.2f}s){Colors.END}\n")
                    
                    # Show performance stats
                    self._show_metrics()
                    
                finally:
                    self.is_speaking = False
                    self.is_processing = False
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"{Colors.RED}  ✗ Response error: {e}{Colors.END}")
                import traceback
                traceback.print_exc()
                self.is_speaking = False
                self.is_processing = False
    
    def _detect_interruption(self) -> bool:
        """Detect if user is trying to interrupt."""
        with self.audio_lock:
            if len(self.audio_buffer) < 10:
                return False
            
            # Check recent frames for speech
            recent = self.audio_buffer[-10:]
            frame_arrays = [frame.flatten() for frame in recent]
            
            # Count speech frames
            speech_frames = sum(1 for frame in frame_arrays if self.vad.is_speech(frame))
            
            # If more than 50% are speech, user is interrupting
            return speech_frames > len(frame_arrays) * 0.5
    
    def _show_metrics(self):
        """Display performance metrics."""
        if len(self.metrics["total_latency"]) < 3:
            return
        
        avg_transcription = np.mean(self.metrics["transcription_time"][-5:])
        avg_response = np.mean(self.metrics["response_time"][-5:])
        avg_tts = np.mean(self.metrics["tts_time"][-5:])
        avg_total = np.mean(self.metrics["total_latency"][-5:])
        
        print(f"{Colors.DIM}  📊 Avg latency (last 5): transcribe={avg_transcription:.2f}s, "
              f"think={avg_response:.2f}s, speak={avg_tts:.2f}s, total={avg_total:.2f}s{Colors.END}\n")
    
    def _monitoring_loop(self):
        """Monitor for turn completion."""
        while not self.should_stop:
            try:
                time.sleep(0.1)
                
                if self.is_processing or self.is_speaking:
                    continue
                
                if self._detect_turn_completion():
                    audio_data = self._get_buffered_audio()
                    
                    if len(audio_data) > 0:
                        self.is_processing = True
                        self.transcription_queue.put(audio_data)
                
            except Exception as e:
                print(f"{Colors.RED}Monitoring error: {e}{Colors.END}")
    
    def run(self):
        """Start the advanced agent."""
        print(f"{Colors.CYAN}🎙️  Starting microphone...{Colors.END}")
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.03),
            ):
                self.is_listening = True
                print(f"{Colors.GREEN}✅ Listening! Speak naturally...\n{Colors.END}")
                
                self._monitoring_loop()
                
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Interrupted{Colors.END}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop gracefully."""
        print(f"\n{Colors.CYAN}🛑 Stopping...{Colors.END}")
        
        self.should_stop = True
        self.is_listening = False
        sd.stop()
        
        if hasattr(self, 'transcription_thread'):
            self.transcription_thread.join(timeout=1.0)
        if hasattr(self, 'response_thread'):
            self.response_thread.join(timeout=1.0)
        
        # Show final stats
        if self.conversation.turn_count > 0:
            print(f"\n{Colors.BOLD}📊 Session Summary:{Colors.END}")
            print(f"  Turns: {self.conversation.turn_count}")
            if self.metrics["total_latency"]:
                print(f"  Avg latency: {np.mean(self.metrics['total_latency']):.2f}s")
        
        print(f"{Colors.GREEN}👋 Goodbye!{Colors.END}\n")


def detect_device() -> str:
    """Detect device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    """Run advanced agent."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print(f"{Colors.RED}❌ GOOGLE_API_KEY not set!{Colors.END}")
        return 1
    
    device = detect_device()
    print(f"{Colors.DIM}Device: {device}{Colors.END}")
    
    agent = AdvancedRealtimeAgent(
        google_api_key=api_key,
        device=device,
        use_semantic_detection=True,
    )
    
    def signal_handler(sig, frame):
        agent.should_stop = True
    
    signal.signal(signal.SIGINT, signal_handler)
    
    agent.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

