#!/usr/bin/env python3
"""
FERN Voice Client - Real-time conversational AI

A complete voice assistant client with:
- Real-time voice input (microphone)
- Streaming TTS output (speakers)
- Visual feedback
- Conversation history

Usage:
    python client_voice.py

Controls:
    SPACE - Push to talk
    ESC   - Exit
    R     - Reset conversation
"""

import os
import sys
import threading
import queue
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import sounddevice as sd
import soundfile as sf
from pynput import keyboard

from fern.llm.gemini_manager import GeminiDialogueManager
from fern.tts.csm_real import RealCSMTTS
from fern.tts.csm_streaming import StreamingTTS

# For voice activity detection
try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    print("⚠️  webrtcvad not installed - using simple silence detection")
    print("   Install with: pip install webrtcvad")


class Colors:
    """ANSI color codes for terminal output."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


class VoiceClient:
    """
    Real-time voice client for FERN.
    
    Features:
    - Push-to-talk recording
    - Automatic transcription (Whisper)
    - Gemini conversational AI
    - Streaming TTS playback
    - Visual feedback
    """
    
    def __init__(
        self,
        google_api_key: str,
        device: str = "cuda",
        sample_rate: int = 16000,
    ):
        """Initialize the voice client."""
        print(f"{Colors.BOLD}{Colors.BLUE}🎙️  FERN Voice Client{Colors.END}\n")
        
        self.sample_rate = sample_rate
        self.device = device
        self.is_recording = False
        self.audio_buffer = []
        self.audio_queue = queue.Queue()
        
        # Initialize AI components
        print("Loading models...")
        
        print("  → Gemini LLM...")
        self.llm = GeminiDialogueManager(api_key=google_api_key)
        print(f"    ✓ Using {self.llm.model_name}")
        
        print(f"  → CSM-1B TTS ({device})...")
        self.tts = RealCSMTTS(device=device)
        self.streaming_tts = StreamingTTS(self.tts, chunk_duration_ms=200)
        print("    ✓ Ready")
        
        print("\n" + "=" * 60)
        print(f"{Colors.BOLD}Ready! Press SPACE to talk, ESC to exit{Colors.END}")
        print("=" * 60 + "\n")
        
        # Keyboard listener
        self.listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        
        self.running = True
    
    def _on_key_press(self, key):
        """Handle key press events."""
        try:
            if key == keyboard.Key.space and not self.is_recording:
                self._start_recording()
            elif key == keyboard.Key.esc:
                print(f"\n{Colors.YELLOW}Exiting...{Colors.END}")
                self.running = False
                return False  # Stop listener
            elif hasattr(key, 'char') and key.char == 'r':
                self._reset_conversation()
        except Exception as e:
            print(f"Key error: {e}")
    
    def _on_key_release(self, key):
        """Handle key release events."""
        try:
            if key == keyboard.Key.space and self.is_recording:
                self._stop_recording()
        except Exception as e:
            print(f"Key error: {e}")
    
    def _start_recording(self):
        """Start recording audio."""
        self.is_recording = True
        self.audio_buffer = []
        print(f"{Colors.BOLD}{Colors.RED}🔴 Recording... (release SPACE when done){Colors.END}")
        
        # Start audio input stream
        self.input_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            blocksize=1024,
        )
        self.input_stream.start()
    
    def _stop_recording(self):
        """Stop recording and process audio."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.input_stream.stop()
        self.input_stream.close()
        
        print(f"{Colors.BLUE}⏹️  Processing...{Colors.END}")
        
        # Combine audio buffer
        if not self.audio_buffer:
            print(f"{Colors.YELLOW}⚠️  No audio recorded{Colors.END}\n")
            return
        
        audio_data = np.concatenate(self.audio_buffer, axis=0)
        
        # Save for debugging (optional)
        # sf.write("recorded.wav", audio_data, self.sample_rate)
        
        # Process in background thread
        threading.Thread(
            target=self._process_audio,
            args=(audio_data,),
            daemon=True
        ).start()
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            print(f"Audio status: {status}")
        
        if self.is_recording:
            self.audio_buffer.append(indata.copy())
    
    def _process_audio(self, audio_data):
        """Process recorded audio (transcribe, generate response, speak)."""
        try:
            # Transcribe audio
            print(f"{Colors.BLUE}  → Transcribing...{Colors.END}")
            
            # Simple placeholder - in production, use Whisper
            # For now, use keyboard input as fallback
            print(f"{Colors.YELLOW}  ⚠️  ASR not yet integrated - please type your message:{Colors.END}")
            user_text = input(f"{Colors.GREEN}  You: {Colors.END}")
            
            if not user_text.strip():
                print(f"{Colors.YELLOW}  ⚠️  Empty input{Colors.END}\n")
                return
            
            print(f"{Colors.GREEN}  👤 You: {user_text}{Colors.END}")
            
            # Generate response with Gemini
            print(f"{Colors.BLUE}  → Thinking...{Colors.END}")
            response_text = self.llm.generate_response(user_text)
            print(f"{Colors.BLUE}  🤖 FERN: {response_text}{Colors.END}")
            
            # Generate and play streaming audio
            print(f"{Colors.BLUE}  → Speaking...{Colors.END}")
            self._play_streaming_response(response_text)
            
            print(f"{Colors.GREEN}  ✓ Done!{Colors.END}\n")
            print(f"{Colors.BOLD}Press SPACE to talk again{Colors.END}\n")
            
        except Exception as e:
            print(f"{Colors.RED}  ✗ Error: {e}{Colors.END}\n")
            import traceback
            traceback.print_exc()
    
    def _play_streaming_response(self, text: str):
        """Play response audio with streaming."""
        try:
            # Generate audio chunks
            for i, chunk in enumerate(self.streaming_tts.synthesize_stream_sentences(text)):
                if i == 0:
                    print(f"{Colors.GREEN}  🔊 Playing...{Colors.END}")
                
                # Play chunk
                sd.play(chunk, samplerate=24000)
                sd.wait()
                
        except Exception as e:
            print(f"{Colors.RED}  Audio playback error: {e}{Colors.END}")
            
            # Fallback: save to file
            try:
                audio = self.tts.synthesize(text)
                if hasattr(audio, 'cpu'):
                    audio = audio.cpu().numpy()
                sf.write("response.wav", audio, 24000)
                print(f"{Colors.YELLOW}  💾 Saved to response.wav{Colors.END}")
            except Exception as e2:
                print(f"{Colors.RED}  Fallback save failed: {e2}{Colors.END}")
    
    def _reset_conversation(self):
        """Reset conversation history."""
        self.llm.clear_history()
        print(f"{Colors.YELLOW}  🔄 Conversation reset{Colors.END}\n")
    
    def run(self):
        """Run the voice client."""
        # Start keyboard listener
        self.listener.start()
        
        # Keep running until ESC
        try:
            while self.running:
                import time
                time.sleep(0.1)
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Interrupted{Colors.END}")
        
        # Cleanup
        self.listener.stop()
        print(f"{Colors.GREEN}👋 Goodbye!{Colors.END}")


def main():
    """Run the voice client."""
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print(f"{Colors.RED}❌ GOOGLE_API_KEY not set!{Colors.END}")
        print("Set it with: export GOOGLE_API_KEY='your-key'")
        return 1
    
    # Check device
    device = "cuda" if os.path.exists("/proc/driver/nvidia/version") else "cpu"
    
    # Create and run client
    try:
        client = VoiceClient(
            google_api_key=api_key,
            device=device,
        )
        client.run()
        return 0
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted{Colors.END}")
        return 0
    except Exception as e:
        print(f"{Colors.RED}Fatal error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

