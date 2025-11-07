"""Streaming audio demonstration for fern pipeline."""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fern import VoiceToVoicePipeline, FERNConfig
import soundfile as sf


def simulate_audio_playback(audio_chunk: np.ndarray, sample_rate: int = 24000):
    """
    Simulate audio playback (in production, this would play through speakers).
    
    Args:
        audio_chunk: Audio samples to "play"
        sample_rate: Sample rate
    """
    duration_ms = (len(audio_chunk) / sample_rate) * 1000
    print(f"  🔊 Playing chunk: {len(audio_chunk)} samples ({duration_ms:.1f}ms)")


def main():
    """Demonstrate streaming audio generation."""
    print("=" * 60)
    print("FERN Streaming Demo")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Create configuration optimized for streaming
    config = FERNConfig(
        device="cpu",  # Change to "cuda" for GPU
        rvq_iterations=16,  # Low latency configuration
        enable_streaming=True,
        streaming_chunk_size=512,  # Small chunks for low latency
        enable_metrics=True,
        log_level="INFO",
    )
    
    print("\n📋 Streaming Configuration:")
    print(f"  Device: {config.device.value}")
    print(f"  RVQ Iterations: {config.rvq_iterations}")
    print(f"  Chunk Size: {config.streaming_chunk_size} samples")
    print(f"  Expected First Chunk Latency: ~1700-2100ms (CPU)")
    
    # Initialize pipeline
    print("\n🚀 Initializing pipeline...")
    pipeline = VoiceToVoicePipeline(config=config)
    print("✅ Pipeline ready!")
    
    # Create a synthetic audio input for demonstration
    # In production, this would be real voice input
    print("\n" + "=" * 60)
    print("Generating Synthetic Input Audio")
    print("=" * 60)
    
    # Create 3 seconds of synthetic audio (silence with tone)
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simple tone to represent speech
    audio_input = np.sin(2 * np.pi * 200 * t) * 0.1
    audio_input = audio_input.astype(np.float32)
    
    # Save synthetic input
    input_path = "synthetic_input.wav"
    sf.write(input_path, audio_input, sample_rate)
    print(f"✅ Created synthetic input: {input_path} ({duration}s)")
    
    # Demo 1: Streaming conversation
    print("\n" + "=" * 60)
    print("Demo 1: Streaming Audio Generation")
    print("=" * 60)
    
    print("\n📝 Processing: 'Tell me about artificial intelligence'")
    print("🎵 Streaming audio chunks...\n")
    
    # We'll use text input for demonstration
    print("(Using text input for demo - audio file would go through ASR)")
    
    # Process with streaming
    all_chunks = []
    chunk_count = 0
    
    try:
        # Since we're using text input, we'll manually trigger TTS streaming
        response_text = pipeline.llm.generate_response(
            "Tell me about artificial intelligence in one sentence"
        )
        print(f"\n💬 LLM Response: '{response_text}'\n")
        
        # Generate streaming audio
        print("🎵 Generating audio chunks:\n")
        for chunk in pipeline.tts.synthesize_streaming(
            text=response_text,
            chunk_size=config.streaming_chunk_size
        ):
            chunk_count += 1
            all_chunks.append(chunk)
            simulate_audio_playback(chunk, config.tts_sample_rate)
        
        # Combine chunks
        complete_audio = np.concatenate(all_chunks)
        
        print(f"\n✅ Streaming complete!")
        print(f"   Total chunks: {chunk_count}")
        print(f"   Total duration: {len(complete_audio)/config.tts_sample_rate:.2f}s")
        print(f"   Average chunk size: {len(complete_audio)/chunk_count:.0f} samples")
        
        # Save output
        output_path = "streaming_output.wav"
        sf.write(output_path, complete_audio, config.tts_sample_rate)
        print(f"   Saved to: {output_path}")
    
    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
    
    # Demo 2: Compare streaming vs one-shot
    print("\n" + "=" * 60)
    print("Demo 2: Streaming vs One-Shot Comparison")
    print("=" * 60)
    
    test_text = "Hello, this is a test."
    
    # One-shot
    print("\n⏱️  One-shot generation:")
    import time
    start = time.time()
    
    response = pipeline.llm.generate_response(test_text)
    audio_oneshot = pipeline.tts.synthesize(response)
    
    oneshot_time = (time.time() - start) * 1000
    print(f"   Total time: {oneshot_time:.1f}ms")
    print(f"   Audio duration: {len(audio_oneshot)/config.tts_sample_rate*1000:.1f}ms")
    
    # Streaming
    print("\n⚡ Streaming generation:")
    pipeline.clear_context()
    
    start = time.time()
    first_chunk_time = None
    
    response2 = pipeline.llm.generate_response(test_text)
    chunks = []
    
    for i, chunk in enumerate(pipeline.tts.synthesize_streaming(
        response2,
        chunk_size=config.streaming_chunk_size
    )):
        if first_chunk_time is None:
            first_chunk_time = (time.time() - start) * 1000
        chunks.append(chunk)
    
    total_time = (time.time() - start) * 1000
    audio_streaming = np.concatenate(chunks)
    
    print(f"   First chunk latency: {first_chunk_time:.1f}ms")
    print(f"   Total time: {total_time:.1f}ms")
    print(f"   Audio duration: {len(audio_streaming)/config.tts_sample_rate*1000:.1f}ms")
    print(f"   Number of chunks: {len(chunks)}")
    
    # Show benefit
    perceived_latency_reduction = oneshot_time - first_chunk_time
    print(f"\n💡 Streaming reduces perceived latency by: {perceived_latency_reduction:.1f}ms")
    print(f"   ({perceived_latency_reduction/oneshot_time*100:.1f}% improvement)")
    
    print("\n" + "=" * 60)
    print("Streaming demo complete!")
    print("=" * 60)
    
    # Cleanup
    if os.path.exists(input_path):
        os.remove(input_path)


if __name__ == "__main__":
    main()

