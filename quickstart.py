#!/usr/bin/env python3
"""Quick start script for i-LAVA voice-to-voice pipeline."""

import os
import sys


def check_requirements():
    """Check if requirements are met."""
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    print("✓ OpenAI API key found")
    
    # Check if packages are installed
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            print("✓ Apple Metal (MPS) available")
        else:
            print("⚠ Running on CPU (GPU recommended for best performance)")
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    return True


def run_demo():
    """Run a simple demonstration."""
    print("\n" + "=" * 60)
    print("i-LAVA Quick Start Demo")
    print("=" * 60)
    
    from fern import VoiceToVoicePipeline, FERNConfig
    
    # Determine device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n📋 Configuration:")
    print(f"  Device: {device}")
    print(f"  RVQ Iterations: 16 (low latency)")
    print(f"  Streaming: Enabled")
    
    # Create configuration
    config = FERNConfig(
        device=device,
        rvq_iterations=16,
        enable_streaming=True,
        enable_metrics=True,
        log_level="INFO",
    )
    
    # Initialize pipeline
    print("\n🚀 Initializing pipeline...")
    print("   (This may take a minute on first run)")
    pipeline = VoiceToVoicePipeline(config=config)
    
    # Get pipeline info
    info = pipeline.get_pipeline_info()
    print(f"\n✅ Pipeline ready!")
    print(f"  ASR: {info['asr_model']}")
    print(f"  LLM: {info['llm_model']}")
    print(f"  TTS: {info['tts_model']}")
    
    # Run a simple conversation
    print("\n" + "=" * 60)
    print("Demo Conversation")
    print("=" * 60)
    
    test_inputs = [
        "Hello! How are you today?",
        "What can you help me with?",
        "Thank you!",
    ]
    
    for i, text in enumerate(test_inputs, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {text}")
        
        # Process
        audio = pipeline.process_text(text)
        
        # Show stats
        duration = len(audio) / config.tts_sample_rate
        print(f"Generated: {len(audio):,} samples ({duration:.2f}s)")
    
    # Show conversation history
    print("\n" + "=" * 60)
    print("Conversation Summary")
    print("=" * 60)
    print(pipeline.get_conversation_history())
    
    # Show expected performance
    print("\n" + "=" * 60)
    print("Expected Performance (from paper)")
    print("=" * 60)
    
    if device == "cuda":
        print("GPU Performance (NVIDIA L4):")
        print("  First Chunk Latency: ~640ms")
        print("  Real-Time Factor: 0.48x")
        print("  Expected SNR: 7-9 dB")
    else:
        print("CPU Performance (Apple M4 Max):")
        print("  First Chunk Latency: ~1749ms")
        print("  Real-Time Factor: 0.93x")
        print("  Expected SNR: 15-16 dB")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\n📚 Next steps:")
    print("  - See examples/ directory for more examples")
    print("  - Read USAGE.md for detailed usage guide")
    print("  - Run 'python examples/streaming_demo.py' for streaming")
    print("  - Run 'python examples/benchmark.py' for benchmarks")
    print("  - Run 'pytest' to run tests")


def main():
    """Main entry point."""
    print("=" * 60)
    print("i-LAVA: Low Latency Voice-2-Voice Architecture")
    print("Paper: https://arxiv.org/html/2509.20971v1")
    print("=" * 60)
    
    if not check_requirements():
        print("\n❌ Please fix the requirements above and try again.")
        return 1
    
    print("\n✅ All requirements met!")
    
    try:
        run_demo()
        return 0
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

