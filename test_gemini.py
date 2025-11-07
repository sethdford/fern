import os
import sys
sys.path.insert(0, '.')

from fern.llm.gemini_manager import GeminiDialogueManager
from fern.tts.csm_real import RealCSMTTS
import soundfile as sf

print("🚀 Testing Gemini + FERN...")

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ Set GOOGLE_API_KEY first!")
    print("export GOOGLE_API_KEY='your-key'")
    sys.exit(1)

# Test Gemini
print("\n1. Testing Gemini LLM...")
llm = GeminiDialogueManager(api_key=api_key)
response = llm.generate_response("Say hello in one sentence!")
print(f"   ✓ Gemini: {response}")

# Test TTS
print("\n2. Testing CSM-1B TTS...")
tts = RealCSMTTS(device="cuda")
audio = tts.synthesize(response)
print(f"   ✓ Generated audio: {len(audio)} samples")

# Save audio
sf.write("test_output.wav", audio.cpu().numpy(), 24000)
print(f"   ✓ Saved to: test_output.wav")

print("\n✨ Success! Everything works!")
print("\nTo hear it, download the file:")
print("scp -P PORT root@X.X.X.X:/workspace/fern/test_output.wav ./")

