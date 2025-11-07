"""
Profiling and benchmarking script for RVQ operations.

This script establishes baseline performance metrics for RVQ decoding
before implementing CUDA optimizations.

Usage:
    python benchmark_rvq.py
"""

import time
import numpy as np
import torch
from typing import List, Dict, Tuple

try:
    from fern.tts.rvq_optimizer import RVQOptimizer
    RVQ_AVAILABLE = True
except ImportError:
    RVQ_AVAILABLE = False
    print("Warning: RVQOptimizer not available")


def benchmark_operation(func, *args, num_iterations: int = 100, warmup: int = 10) -> Dict[str, float]:
    """
    Benchmark a function with warmup and multiple iterations.
    
    Args:
        func: Function to benchmark
        *args: Arguments to pass to function
        num_iterations: Number of benchmark iterations
        warmup: Number of warmup iterations
        
    Returns:
        Dictionary with timing statistics (ms)
    """
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        _ = func(*args)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times.append((time.perf_counter() - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'p50': float(np.percentile(times, 50)),
        'p95': float(np.percentile(times, 95)),
        'p99': float(np.percentile(times, 99)),
    }


def profile_rvq_decoding(
    device: str = 'cuda',
    num_iterations: int = 16,
    num_codebooks: int = 32,
    time_steps: int = 75,
    batch_size: int = 1,
) -> None:
    """
    Profile RVQ decoding operation.
    
    Args:
        device: Device to run on ('cuda', 'cpu')
        num_iterations: Number of RVQ iterations
        num_codebooks: Number of codebooks
        time_steps: Number of time steps
        batch_size: Batch size
    """
    print("=" * 70)
    print("RVQ DECODING BENCHMARK")
    print("=" * 70)
    print()
    
    if not RVQ_AVAILABLE:
        print("❌ RVQOptimizer not available. Cannot benchmark.")
        return
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("❌ CUDA not available. Falling back to CPU.")
        device = 'cpu'
    
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  RVQ Iterations: {num_iterations}")
    print(f"  Codebooks: {num_codebooks}")
    print(f"  Time Steps: {time_steps}")
    print(f"  Batch Size: {batch_size}")
    print()
    
    # Initialize RVQ optimizer
    try:
        optimizer = RVQOptimizer(
            num_iterations=num_iterations,
            num_codebooks=num_codebooks,
            padding_method="none",
        )
    except Exception as e:
        print(f"❌ Failed to initialize RVQOptimizer: {e}")
        return
    
    # Create test data
    codes = torch.randint(
        0, 2048,
        (batch_size, num_codebooks, time_steps),
        device=device,
        dtype=torch.int32
    )
    
    print(f"Test Data Shape: {codes.shape}")
    print(f"Test Data Size: {codes.numel() * codes.element_size() / 1024:.2f} KB")
    print()
    
    # Benchmark decoding
    print("Running benchmark...")
    try:
        stats = benchmark_operation(
            optimizer.decode_rvq_codes,
            codes,
            num_iterations=100,
            warmup=10
        )
        
        print()
        print("Results:")
        print(f"  Mean:   {stats['mean']:.3f} ms")
        print(f"  Std:    {stats['std']:.3f} ms")
        print(f"  Min:    {stats['min']:.3f} ms")
        print(f"  Max:    {stats['max']:.3f} ms")
        print(f"  P50:    {stats['p50']:.3f} ms")
        print(f"  P95:    {stats['p95']:.3f} ms")
        print(f"  P99:    {stats['p99']:.3f} ms")
        print()
        
        # Calculate throughput
        samples_per_sec = (time_steps * 75) / (stats['mean'] / 1000)  # 75Hz = 24kHz / 320
        print(f"Throughput: {samples_per_sec:.0f} samples/sec")
        print(f"Real-time Factor: {samples_per_sec / 24000:.2f}x")
        print()
        
        # Memory usage
        if torch.cuda.is_available() and device == 'cuda':
            mem_allocated = torch.cuda.memory_allocated() / 1024**2
            mem_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"GPU Memory:")
            print(f"  Allocated: {mem_allocated:.2f} MB")
            print(f"  Reserved:  {mem_reserved:.2f} MB")
            print()
        
        # Save baseline
        save_baseline(stats, device, num_iterations, num_codebooks)
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


def save_baseline(stats: Dict[str, float], device: str, iterations: int, codebooks: int) -> None:
    """Save baseline metrics for future comparison."""
    import json
    from pathlib import Path
    
    baseline = {
        'device': device,
        'iterations': iterations,
        'codebooks': codebooks,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'stats': stats,
    }
    
    baseline_file = Path(__file__).parent / 'benchmarks' / 'rvq_baseline.json'
    baseline_file.parent.mkdir(exist_ok=True)
    
    with open(baseline_file, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    print(f"✓ Baseline saved to: {baseline_file}")


def profile_with_torch_profiler(device: str = 'cuda') -> None:
    """
    Profile RVQ with torch.profiler for detailed analysis.
    
    Args:
        device: Device to profile on
    """
    print("=" * 70)
    print("TORCH PROFILER ANALYSIS")
    print("=" * 70)
    print()
    
    if not RVQ_AVAILABLE:
        print("❌ RVQOptimizer not available.")
        return
    
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    optimizer = RVQOptimizer(num_iterations=16, num_codebooks=32)
    codes = torch.randint(0, 2048, (1, 16, 75), device=device, dtype=torch.int32)
    
    print(f"Profiling on {device}...")
    print()
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA if device == 'cuda' else None,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(10):
            _ = optimizer.decode_rvq_codes(codes)
    
    # Print results
    print(prof.key_averages().table(sort_by="cuda_time_total" if device == 'cuda' else "cpu_time_total", row_limit=20))
    print()
    
    # Save trace
    trace_file = Path(__file__).parent / 'benchmarks' / f'rvq_trace_{device}.json'
    trace_file.parent.mkdir(exist_ok=True)
    prof.export_chrome_trace(str(trace_file))
    print(f"✓ Trace saved to: {trace_file}")
    print(f"  View at: chrome://tracing")
    print()


def main():
    """Run all benchmarks."""
    import sys
    
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                      ║")
    print("║              RVQ PERFORMANCE PROFILING & BASELINE                    ║")
    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Detect device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("ℹ CUDA not available, using CPU")
    print()
    
    # Run benchmarks
    try:
        # Basic benchmark
        profile_rvq_decoding(device=device)
        
        # Detailed profiling
        if '--profile' in sys.argv:
            profile_with_torch_profiler(device=device)
        
        print("=" * 70)
        print("✓ Profiling complete!")
        print()
        print("Next steps:")
        print("  1. Review baseline metrics in benchmarks/rvq_baseline.json")
        print("  2. Analyze bottlenecks")
        print("  3. Design CUDA kernel")
        print("  4. Implement optimization")
        print("=" * 70)
        print()
        
    except KeyboardInterrupt:
        print()
        print("❌ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

