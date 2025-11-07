"""Command-line interface for i-LAVA."""

import argparse
import sys
from pathlib import Path

from fern import VoiceToVoicePipeline, FERNConfig
from fern.utils import setup_logging


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="i-LAVA: Low Latency Voice-2-Voice Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process audio file")
    process_parser.add_argument("input", help="Input audio file path")
    process_parser.add_argument("-o", "--output", help="Output audio file path")
    process_parser.add_argument("-d", "--device", default="cuda", choices=["cuda", "cpu", "mps"])
    process_parser.add_argument("-r", "--rvq", type=int, default=16, choices=[16, 20, 24, 32])
    process_parser.add_argument("--no-streaming", action="store_true", help="Disable streaming")
    process_parser.add_argument("--metrics", action="store_true", help="Show performance metrics")
    
    # Text command
    text_parser = subparsers.add_parser("text", help="Process text input")
    text_parser.add_argument("text", help="Text to process")
    text_parser.add_argument("-o", "--output", required=True, help="Output audio file path")
    text_parser.add_argument("-d", "--device", default="cuda", choices=["cuda", "cpu", "mps"])
    text_parser.add_argument("-r", "--rvq", type=int, default=16, choices=[16, 20, 24, 32])
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    benchmark_parser.add_argument("-d", "--device", default="cuda", choices=["cuda", "cpu", "mps"])
    benchmark_parser.add_argument("-r", "--rvq", type=int, nargs="+", default=[16, 20, 24, 32])
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(level="INFO")
    
    if args.command == "process":
        return process_audio(args)
    elif args.command == "text":
        return process_text(args)
    elif args.command == "info":
        return show_info(args)
    elif args.command == "benchmark":
        return run_benchmark(args)
    
    return 0


def process_audio(args):
    """Process audio file."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    config = FERNConfig(
        device=args.device,
        rvq_iterations=args.rvq,
        enable_streaming=not args.no_streaming,
        enable_metrics=args.metrics,
    )
    
    print(f"Processing: {input_path}")
    print(f"Device: {args.device}, RVQ: {args.rvq}")
    
    pipeline = VoiceToVoicePipeline(config=config)
    
    audio, metrics = pipeline.process_audio(
        str(input_path),
        output_path=args.output,
        return_metrics=args.metrics,
    )
    
    if args.output:
        print(f"Output saved to: {args.output}")
    
    if args.metrics and metrics:
        print("\n" + metrics.summary())
    
    return 0


def process_text(args):
    """Process text input."""
    config = FERNConfig(
        device=args.device,
        rvq_iterations=args.rvq,
    )
    
    print(f"Processing text: '{args.text}'")
    print(f"Device: {args.device}, RVQ: {args.rvq}")
    
    pipeline = VoiceToVoicePipeline(config=config)
    
    audio = pipeline.process_text(args.text, output_path=args.output)
    
    print(f"Output saved to: {args.output}")
    print(f"Generated: {len(audio)} samples ({len(audio)/24000:.2f}s)")
    
    return 0


def show_info(args):
    """Show system information."""
    import torch
    from fern.utils import detect_device
    
    print("=" * 60)
    print("i-LAVA System Information")
    print("=" * 60)
    
    print(f"\nPython: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    device_type, device_name = detect_device()
    print(f"\nDevice: {device_name} ({device_type})")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
    
    import os
    api_key_set = "Yes" if os.getenv("OPENAI_API_KEY") else "No"
    print(f"\nOpenAI API Key: {api_key_set}")
    
    return 0


def run_benchmark(args):
    """Run benchmarks."""
    print("=" * 60)
    print("i-LAVA Benchmarks")
    print("=" * 60)
    print(f"\nDevice: {args.device}")
    print(f"RVQ Iterations: {args.rvq}")
    print("\nRunning benchmarks...")
    
    # Import and run benchmark
    from examples.benchmark import run_benchmark
    
    test_texts = [
        "Hello, how are you?",
        "Tell me about the weather today.",
    ]
    
    results = run_benchmark(args.device, args.rvq, test_texts)
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    for result in results:
        print(f"\n{args.device.upper()} - {result['rvq_iterations']} RVQ:")
        print(f"  First Chunk: {result['avg_first_chunk_ms']:.1f} ms")
        print(f"  RTF: {result['avg_rtf_oneshot']:.3f}x")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

