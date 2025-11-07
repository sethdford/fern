"""
Comprehensive TTS profiling for CSM-1B optimization.

This script profiles the full CSM-1B TTS pipeline to identify bottlenecks
and guide optimization efforts.

Usage:
    # With real CSM model (requires GPU and model download)
    python tools/profiling/profile_tts.py --real
    
    # With placeholder (CPU testing)
    python tools/profiling/profile_tts.py --placeholder
    
    # Save detailed trace
    python tools/profiling/profile_tts.py --real --trace

Requirements:
    - For real profiling: CUDA-capable GPU
    - For real profiling: Edit ilava/tts/csm_real.py lines 44-45
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fern.tts.csm_config import CSMConfig, create_development_config, create_production_config
from fern.tts import CSMTTS


def profile_tts_synthesis(
    text: str,
    device: str = 'cuda',
    use_real_csm: bool = True,
    num_iterations: int = 10,
    warmup: int = 3,
) -> Dict:
    """
    Profile TTS synthesis with detailed timing breakdown.
    
    Args:
        text: Text to synthesize
        device: Device to use ('cuda', 'cpu')
        use_real_csm: Use real CSM-1B or placeholder
        num_iterations: Number of profiling iterations
        warmup: Number of warmup iterations
        
    Returns:
        Dictionary with profiling results
    """
    print("=" * 70)
    print("TTS SYNTHESIS PROFILING")
    print("=" * 70)
    print()
    
    # Create config
    if use_real_csm:
        config = create_production_config(device)
    else:
        config = create_development_config(device)
    
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Use Real CSM: {use_real_csm}")
    print(f"  RVQ Iterations: {config.rvq_iterations}")
    print(f"  Text: '{text}'")
    print()
    
    # Initialize TTS
    print("Initializing TTS...")
    try:
        tts = CSMTTS(config=config)
        print("✓ TTS initialized")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize TTS: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for i in range(warmup):
        _ = tts.synthesize(text)
        print(f"  Warmup {i+1}/{warmup} complete")
    print("✓ Warmup complete")
    print()
    
    # Profile
    print(f"Profiling ({num_iterations} iterations)...")
    times = []
    
    for i in range(num_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        audio = tts.synthesize(text)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
        
        print(f"  Iteration {i+1}/{num_iterations}: {elapsed:.2f}ms")
    
    print()
    
    # Statistics
    times = np.array(times)
    stats = {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'p50': float(np.percentile(times, 50)),
        'p95': float(np.percentile(times, 95)),
        'p99': float(np.percentile(times, 99)),
    }
    
    print("Results:")
    print(f"  Mean:   {stats['mean']:.2f} ms")
    print(f"  Std:    {stats['std']:.2f} ms")
    print(f"  Min:    {stats['min']:.2f} ms")
    print(f"  Max:    {stats['max']:.2f} ms")
    print(f"  P50:    {stats['p50']:.2f} ms")
    print(f"  P95:    {stats['p95']:.2f} ms")
    print(f"  P99:    {stats['p99']:.2f} ms")
    print()
    
    # Calculate RTF
    audio_duration_sec = len(audio) / config.sample_rate
    rtf = stats['mean'] / 1000 / audio_duration_sec
    
    print(f"Audio Duration: {audio_duration_sec:.2f}s")
    print(f"Real-Time Factor (RTF): {rtf:.2f}x")
    print()
    
    return {
        'stats': stats,
        'rtf': rtf,
        'audio_duration': audio_duration_sec,
        'config': config.to_dict(),
    }


def profile_with_torch_profiler(
    text: str,
    device: str = 'cuda',
    use_real_csm: bool = True,
    save_trace: bool = True,
) -> None:
    """
    Profile TTS with torch.profiler for detailed analysis.
    
    Args:
        text: Text to synthesize
        device: Device to use
        use_real_csm: Use real CSM-1B or placeholder
        save_trace: Save Chrome trace for visualization
    """
    print("=" * 70)
    print("TORCH PROFILER ANALYSIS")
    print("=" * 70)
    print()
    
    # Create config
    if use_real_csm:
        config = create_production_config(device)
    else:
        config = create_development_config(device)
    
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Use Real CSM: {use_real_csm}")
    print(f"  Text: '{text}'")
    print()
    
    # Initialize TTS
    print("Initializing TTS...")
    tts = CSMTTS(config=config)
    print("✓ TTS initialized")
    print()
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = tts.synthesize(text)
    print("✓ Warmup complete")
    print()
    
    # Profile
    print("Profiling with torch.profiler...")
    print("This may take a minute...")
    print()
    
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == 'cuda' and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        # Profile multiple iterations
        for _ in range(5):
            _ = tts.synthesize(text)
    
    # Print results
    print("=" * 70)
    print("TOP OPERATIONS BY TIME")
    print("=" * 70)
    print()
    
    sort_key = "cuda_time_total" if device == 'cuda' and torch.cuda.is_available() else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=30))
    print()
    
    # Save trace
    if save_trace:
        trace_dir = Path(__file__).parent.parent.parent / 'benchmarks' / 'traces'
        trace_dir.mkdir(parents=True, exist_ok=True)
        
        trace_file = trace_dir / f'tts_trace_{"real" if use_real_csm else "placeholder"}_{device}.json'
        prof.export_chrome_trace(str(trace_file))
        
        print(f"✓ Trace saved to: {trace_file}")
        print(f"  View at: chrome://tracing")
        print()
    
    # Analyze top operations
    print("=" * 70)
    print("BOTTLENECK ANALYSIS")
    print("=" * 70)
    print()
    
    key_averages = prof.key_averages()
    sorted_ops = sorted(key_averages, key=lambda x: getattr(x, sort_key), reverse=True)
    
    total_time = sum(getattr(op, sort_key) for op in sorted_ops)
    
    print("Top 10 operations by time:")
    for i, op in enumerate(sorted_ops[:10], 1):
        op_time = getattr(op, sort_key) / 1000  # Convert to ms
        percentage = (getattr(op, sort_key) / total_time * 100) if total_time > 0 else 0
        print(f"  {i}. {op.key:50s} {op_time:8.2f}ms ({percentage:5.1f}%)")
    
    print()


def save_profiling_report(results: Dict, output_dir: Path) -> None:
    """
    Save profiling report as JSON.
    
    Args:
        results: Profiling results
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / 'tts_profiling_report.json'
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results,
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Report saved to: {report_file}")


def main():
    """Run profiling."""
    parser = argparse.ArgumentParser(description='Profile CSM-1B TTS')
    parser.add_argument('--real', action='store_true', help='Use real CSM-1B (requires GPU)')
    parser.add_argument('--placeholder', action='store_true', help='Use placeholder mode')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--text', default='Hello, this is a test of the text to speech system.', 
                       help='Text to synthesize')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('--trace', action='store_true', help='Save torch profiler trace')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.real:
        use_real_csm = True
    elif args.placeholder:
        use_real_csm = False
    else:
        # Auto-detect
        use_real_csm = torch.cuda.is_available()
    
    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
        use_real_csm = False
    
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                      ║")
    print("║              CSM-1B TTS PROFILING TOOL                               ║")
    print("║              Phase 2.1: Profiling & Analysis                         ║")
    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    if args.device == 'cuda':
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ Running on CPU")
    
    print(f"Mode: {'Real CSM-1B' if use_real_csm else 'Placeholder'}")
    print()
    
    try:
        # Basic profiling
        results = profile_tts_synthesis(
            text=args.text,
            device=args.device,
            use_real_csm=use_real_csm,
            num_iterations=args.iterations,
        )
        
        if results is None:
            print("❌ Profiling failed")
            return 1
        
        # Detailed profiling
        if args.trace or use_real_csm:
            profile_with_torch_profiler(
                text=args.text,
                device=args.device,
                use_real_csm=use_real_csm,
                save_trace=args.trace,
            )
        
        # Save report
        output_dir = Path(__file__).parent.parent.parent / 'benchmarks' / 'profiling'
        save_profiling_report(results, output_dir)
        
        print("=" * 70)
        print("✓ PROFILING COMPLETE")
        print("=" * 70)
        print()
        
        if use_real_csm:
            print("Next steps:")
            print("  1. Review benchmarks/traces/tts_trace_real_*.json in chrome://tracing")
            print("  2. Analyze top operations and identify bottlenecks")
            print("  3. Run: python tools/profiling/analyze_results.py")
            print("  4. Create optimization plan based on data")
        else:
            print("ℹ Placeholder mode - results are not representative")
            print("  Run with --real flag on GPU for accurate profiling")
        
        print()
        return 0
        
    except KeyboardInterrupt:
        print()
        print("❌ Profiling interrupted by user")
        return 1
    except Exception as e:
        print()
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

