"""Basic conversation example using fern pipeline."""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fern import VoiceToVoicePipeline, FERNConfig


def main():
    """Run a basic voice-to-voice conversation."""
    print("=" * 60)
    print("FERN Basic Conversation Example")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Create configuration
    # Use CPU with 16 RVQ iterations for faster processing
    config = FERNConfig(
        device="cpu",  # Change to "cuda" if GPU available
        rvq_iterations=16,  # Lower iterations for lower latency
        enable_streaming=False,  # One-shot generation for simplicity
        enable_metrics=True,
        log_level="INFO",
    )
    
    print("\n📋 Configuration:")
    print(f"  Device: {config.device.value}")
    print(f"  RVQ Iterations: {config.rvq_iterations}")
    print(f"  LLM: {config.llm_model}")
    print(f"  Whisper: {config.whisper_model}")
    
    # Initialize pipeline
    print("\n🚀 Initializing pipeline...")
    pipeline = VoiceToVoicePipeline(config=config)
    
    # Show pipeline info
    info = pipeline.get_pipeline_info()
    print("\n✅ Pipeline ready!")
    print(f"  TTS Model: {info['tts_model']}")
    print(f"  Estimated latency reduction: {info['tts_info']['estimated_latency_reduction']}")
    
    # Example 1: Process audio file
    print("\n" + "=" * 60)
    print("Example 1: Process Audio File")
    print("=" * 60)
    
    # For this example, we'll use text input since we don't have an audio file
    print("\n📝 Using text input (no audio file available)")
    
    text_input = "Hello, how are you today?"
    print(f"\nUser: {text_input}")
    
    output_audio = pipeline.process_text(
        text=text_input,
        output_path="output_example1.wav"
    )
    
    print(f"\n✅ Generated {len(output_audio)} audio samples")
    print(f"   Duration: {len(output_audio)/24000:.2f} seconds")
    print(f"   Saved to: output_example1.wav")
    
    # Example 2: Multiple conversation turns
    print("\n" + "=" * 60)
    print("Example 2: Multi-turn Conversation")
    print("=" * 60)
    
    conversation_turns = [
        "What's the weather like?",
        "Tell me a joke",
        "Thank you!",
    ]
    
    for i, text in enumerate(conversation_turns, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {text}")
        
        output_audio = pipeline.process_text(text)
        
        print(f"Generated: {len(output_audio)} samples ({len(output_audio)/24000:.2f}s)")
    
    # Show conversation history
    print("\n" + "=" * 60)
    print("Conversation History")
    print("=" * 60)
    print(pipeline.get_conversation_history())
    
    # Example 3: Clear context and start fresh
    print("\n" + "=" * 60)
    print("Example 3: Clear Context")
    print("=" * 60)
    
    pipeline.clear_context()
    print("✅ Context cleared")
    
    # New conversation
    text = "Let's start fresh. What's your name?"
    print(f"\nUser: {text}")
    
    output_audio = pipeline.process_text(
        text=text,
        output_path="output_example3.wav"
    )
    
    print(f"\n✅ Generated {len(output_audio)} audio samples")
    print(f"   Saved to: output_example3.wav")
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

