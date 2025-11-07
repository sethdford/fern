"""
FERN Conversational AI Assistant with Gemini.

A complete voice assistant using:
- Gemini 1.5 Flash for fast, intelligent responses
- CSM-1B for natural text-to-speech
- Real-time conversation with context awareness
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fern.llm.gemini_manager import GeminiDialogueManager
from fern.tts.csm_real import RealCSMTTS
import soundfile as sf
import sounddevice as sd
import numpy as np


class GeminiVoiceAssistant:
    """
    Complete conversational AI assistant with Gemini and CSM-1B.
    """
    
    def __init__(
        self,
        google_api_key: str,
        device: str = "cuda",
        personality: str = "friendly",
    ):
        """
        Initialize the voice assistant.
        
        Args:
            google_api_key: Google AI API key
            device: Device for TTS (cuda, mps, cpu)
            personality: Assistant personality (friendly, professional, casual, expert)
        """
        print("🚀 Initializing FERN Voice Assistant with Gemini...")
        
        # Initialize Gemini LLM
        print("  → Loading Gemini 1.5 Flash...")
        self.llm = GeminiDialogueManager(
            api_key=google_api_key,
            model_name="gemini-1.5-flash",  # Fastest
            temperature=0.7,
            max_tokens=150,
            system_prompt=self._get_personality_prompt(personality),
        )
        print("  ✓ Gemini ready!")
        
        # Initialize CSM-1B TTS
        print(f"  → Loading CSM-1B TTS on {device}...")
        self.tts = RealCSMTTS(device=device)
        print("  ✓ TTS ready!")
        
        self.device = device
        self.personality = personality
        
        print("✨ Assistant ready! Let's chat!\n")
    
    def _get_personality_prompt(self, personality: str) -> str:
        """Get system prompt for personality."""
        prompts = {
            "friendly": """You are FERN, a warm and friendly AI voice assistant.
            
You're helpful, conversational, and approachable. You give concise responses 
(1-3 sentences) that are easy to understand when spoken aloud. You occasionally 
use light humor and always maintain a positive, supportive tone.""",
            
            "professional": """You are FERN, a professional AI assistant.
            
You provide clear, accurate information in a polished, business-appropriate tone.
Your responses are concise (1-3 sentences) and well-structured. You maintain 
professionalism while still being approachable.""",
            
            "casual": """You are FERN, a laid-back AI buddy.
            
You're chill, conversational, and easy to talk to. You keep things simple and 
fun, using everyday language. Responses are short (1-3 sentences) and natural, 
like chatting with a friend.""",
            
            "expert": """You are FERN, a knowledgeable AI expert.
            
You provide insightful, accurate information with depth and nuance. While 
authoritative, you remain accessible. Responses are concise but informative 
(1-3 sentences), perfect for spoken delivery.""",
        }
        return prompts.get(personality, prompts["friendly"])
    
    def chat(self, user_text: str, play_audio: bool = True) -> tuple[str, np.ndarray]:
        """
        Process user input and generate voice response.
        
        Args:
            user_text: What the user said/typed
            play_audio: Whether to play the audio automatically
        
        Returns:
            Tuple of (response_text, audio_array)
        """
        print(f"\n👤 You: {user_text}")
        
        # Generate text response with Gemini
        print("  🤔 Thinking...")
        response_text = self.llm.generate_response(user_text)
        print(f"  🤖 FERN: {response_text}")
        
        # Generate speech with CSM-1B
        print("  🎤 Speaking...")
        audio = self.tts.synthesize(response_text)
        
        # Play audio if requested
        if play_audio:
            self._play_audio(audio)
        
        return response_text, audio
    
    def _play_audio(self, audio: np.ndarray):
        """Play audio through speakers."""
        try:
            # Convert to numpy if needed
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            
            # Play audio
            sd.play(audio, samplerate=24000)
            sd.wait()  # Wait until audio finishes
        except Exception as e:
            print(f"  ⚠️ Could not play audio: {e}")
            print("  (Audio generation was successful, playback failed)")
    
    def save_response(self, audio: np.ndarray, filename: str = "response.wav"):
        """Save audio response to file."""
        if hasattr(audio, 'cpu'):
            audio = audio.cpu().numpy()
        sf.write(filename, audio, 24000)
        print(f"  💾 Saved to {filename}")
    
    def clear_history(self):
        """Clear conversation history."""
        self.llm.clear_history()
        print("  🔄 Conversation history cleared")
    
    def get_summary(self) -> str:
        """Get conversation summary."""
        return self.llm.get_conversation_summary()


def main():
    """Run interactive demo."""
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not set!")
        print("Set it with: export GOOGLE_API_KEY='your-key'")
        print("Get your key at: https://makersuite.google.com/app/apikey")
        return
    
    # Initialize assistant
    assistant = GeminiVoiceAssistant(
        google_api_key=api_key,
        device="cuda",  # Change to "cpu" or "mps" if needed
        personality="friendly",  # or "professional", "casual", "expert"
    )
    
    # Demo conversation
    print("═" * 80)
    print("FERN VOICE ASSISTANT DEMO".center(80))
    print("═" * 80)
    print()
    print("Try these examples:")
    print("  • 'Hello! Tell me about yourself.'")
    print("  • 'What's the weather like?' (it will make up a response)")
    print("  • 'Tell me a fun fact about space.'")
    print("  • 'What can you help me with?'")
    print()
    print("Type 'quit' to exit, 'clear' to reset conversation, 'summary' for recap")
    print("─" * 80)
    print()
    
    # Interactive loop
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                assistant.clear_history()
                continue
            
            if user_input.lower() == 'summary':
                summary = assistant.get_summary()
                print(f"\n📋 Summary: {summary}")
                continue
            
            # Generate response
            response_text, audio = assistant.chat(user_input, play_audio=True)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

