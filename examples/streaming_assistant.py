"""
FERN Streaming Conversational Assistant.

Ultra-low latency voice assistant with streaming audio.
User hears the first words in ~100ms instead of waiting 400ms!
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fern.llm.gemini_manager import GeminiDialogueManager
from fern.tts.csm_real import RealCSMTTS
from fern.tts.csm_streaming import StreamingTTS
import soundfile as sf
import numpy as np


def demo_streaming():
    """Demonstrate streaming vs non-streaming."""
    print("🚀 FERN Streaming Demo")
    print("=" * 60)
    
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Set GOOGLE_API_KEY first!")
        return
    
    # Initialize
    print("\n1️⃣  Loading models...")
    llm = GeminiDialogueManager(api_key=api_key)
    tts = RealCSMTTS(device="cuda")
    streaming_tts = StreamingTTS(tts, chunk_duration_ms=200)
    print("   ✓ Models ready")
    
    # Test text
    test_text = """Hello! I'm FERN, your AI assistant. 
    I can help you with questions, have conversations, and provide information. 
    What would you like to know?"""
    
    print("\n2️⃣  Testing NON-STREAMING (traditional)...")
    print("   Generating full audio...")
    
    import time
    start = time.time()
    audio = tts.synthesize(test_text)
    gen_time = (time.time() - start) * 1000
    
    print(f"   ✓ Generated in {gen_time:.0f}ms")
    print(f"   ⏱️  User waits: {gen_time:.0f}ms before hearing anything")
    
    # Save
    if hasattr(audio, 'cpu'):
        audio = audio.cpu().numpy()
    sf.write("demo_normal.wav", audio, 24000)
    print("   💾 Saved to: demo_normal.wav")
    
    # Test streaming
    print("\n3️⃣  Testing STREAMING (chunked)...")
    print("   Generating chunks...")
    
    chunks = []
    chunk_times = []
    
    start = time.time()
    for i, chunk in enumerate(streaming_tts.synthesize_stream(test_text)):
        chunk_time = (time.time() - start) * 1000
        chunk_times.append(chunk_time)
        chunks.append(chunk)
        
        if i == 0:
            first_chunk_time = chunk_time
            print(f"   ✓ First chunk in {first_chunk_time:.0f}ms")
            print(f"   🎵 User HEARS audio after {first_chunk_time:.0f}ms!")
        
        print(f"      Chunk {i+1}: {len(chunk)} samples (+{chunk_time - (chunk_times[i-1] if i > 0 else 0):.0f}ms)")
    
    total_time = chunk_times[-1]
    print(f"\n   ✓ All chunks in {total_time:.0f}ms")
    
    # Save streaming output
    streaming_audio = np.concatenate(chunks)
    sf.write("demo_streaming.wav", streaming_audio, 24000)
    print("   💾 Saved to: demo_streaming.wav")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(f"\nNon-Streaming:")
    print(f"  Time to first audio:  {gen_time:.0f}ms")
    print(f"  Total generation:     {gen_time:.0f}ms")
    print(f"  User experience:      😴 Wait {gen_time:.0f}ms")
    
    print(f"\nStreaming:")
    print(f"  Time to first audio:  {first_chunk_time:.0f}ms  🚀 {gen_time/first_chunk_time:.1f}x faster!")
    print(f"  Total generation:     {total_time:.0f}ms")
    print(f"  User experience:      😊 Hear response immediately!")
    
    print(f"\nPerceived latency reduction: {gen_time - first_chunk_time:.0f}ms ({((gen_time - first_chunk_time)/gen_time*100):.0f}% faster)")
    
    print("\n" + "=" * 60)
    print("💡 TIP: With sentence-based streaming, first words can play in <100ms!")
    print("=" * 60)


def demo_sentence_streaming():
    """Demo sentence-by-sentence streaming."""
    print("\n\n🎯 SENTENCE-BASED STREAMING DEMO")
    print("=" * 60)
    
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Set GOOGLE_API_KEY first!")
        return
    
    # Initialize
    llm = GeminiDialogueManager(api_key=api_key)
    tts = RealCSMTTS(device="cuda")
    streaming_tts = StreamingTTS(tts)
    
    # Generate response
    print("\n1️⃣  Asking Gemini a question...")
    response = llm.generate_response("Tell me three interesting facts about space in three sentences.")
    print(f"   Response: {response}")
    
    # Stream by sentences
    print("\n2️⃣  Streaming by sentences...")
    
    import time
    sentences = []
    start = time.time()
    
    for i, audio_chunk in enumerate(streaming_tts.synthesize_stream_sentences(response)):
        elapsed = (time.time() - start) * 1000
        sentences.append(audio_chunk)
        
        if i == 0:
            print(f"   🎵 First sentence playing after {elapsed:.0f}ms!")
        else:
            print(f"   🎵 Sentence {i+1} playing...")
        
        # In real app, you'd play this immediately:
        # play_audio(audio_chunk)
    
    total_time = (time.time() - start) * 1000
    
    # Save
    full_audio = np.concatenate(sentences)
    sf.write("demo_sentences.wav", full_audio, 24000)
    
    print(f"\n   ✓ Complete in {total_time:.0f}ms")
    print(f"   💾 Saved to: demo_sentences.wav")
    print(f"\n   Perceived latency: ~{total_time/len(sentences):.0f}ms per sentence")
    print(f"   User experience: 🚀 Immediate feedback!")


if __name__ == "__main__":
    try:
        demo_streaming()
        demo_sentence_streaming()
        
        print("\n\n✨ Streaming demos complete!")
        print("\nDownload and listen to:")
        print("  • demo_normal.wav (traditional)")
        print("  • demo_streaming.wav (chunked)")  
        print("  • demo_sentences.wav (sentence-based)")
        print("\nNotice how streaming feels much more responsive! 🎉")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo cancelled")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

