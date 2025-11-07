"""Benchmark script to evaluate i-LAVA performance across RVQ configurations."""

import os
import sys
import time
from typing import List, Dict, Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fern import VoiceToVoicePipeline, FERNConfig
from fern.metrics import PerformanceMetrics


def run_benchmark(
    device: str,
    rvq_iterations_list: List[int],
    test_texts: List[str],
) -> List[Dict[str, Any]]:
    """
    Run benchmarks across different RVQ configurations.
    
    Args:
        device: Device to run on (cuda, cpu, mps)
        rvq_iterations_list: List of RVQ iterations to test
        test_texts: List of test texts
        
    Returns:
        List of benchmark results
    """
    results = []
    
    for rvq_iterations in rvq_iterations_list:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {device.upper()} with {rvq_iterations} RVQ Iterations")
        print(f"{'='*60}")
        
        # Create configuration
        config = FERNConfig(
            device=device,
            rvq_iterations=rvq_iterations,
            enable_streaming=True,
            enable_metrics=True,
            log_level="WARNING",  # Reduce noise
        )
        
        # Initialize pipeline
        print("Initializing pipeline...")
        pipeline = VoiceToVoicePipeline(config=config)
        
        # Run tests
        test_results = []
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}/{len(test_texts)}: '{text[:50]}...'")
            
            try:
                # Generate LLM response
                start_llm = time.time()
                response_text = pipeline.llm.generate_response(text)
                llm_time = (time.time() - start_llm) * 1000
                
                # TTS One-shot
                start_tts = time.time()
                audio = pipeline.tts.synthesize(response_text)
                tts_time = (time.time() - start_tts) * 1000
                
                audio_duration = (len(audio) / config.tts_sample_rate) * 1000
                rtf = tts_time / audio_duration if audio_duration > 0 else 0
                
                # TTS Streaming
                start_streaming = time.time()
                first_chunk_time = None
                chunks = []
                chunk_times = []
                last_chunk_time = start_streaming
                
                for chunk in pipeline.tts.synthesize_streaming(
                    response_text,
                    chunk_size=config.streaming_chunk_size
                ):
                    current_time = time.time()
                    
                    if first_chunk_time is None:
                        first_chunk_time = (current_time - start_streaming) * 1000
                    else:
                        inter_chunk = (current_time - last_chunk_time) * 1000
                        chunk_times.append(inter_chunk)
                    
                    chunks.append(chunk)
                    last_chunk_time = current_time
                
                total_streaming_time = (time.time() - start_streaming) * 1000
                
                result = {
                    "text_length": len(text),
                    "response_length": len(response_text),
                    "llm_latency_ms": llm_time,
                    "tts_oneshot_ms": tts_time,
                    "audio_duration_ms": audio_duration,
                    "rtf_oneshot": rtf,
                    "first_chunk_latency_ms": first_chunk_time,
                    "total_streaming_ms": total_streaming_time,
                    "num_chunks": len(chunks),
                    "avg_inter_chunk_ms": np.mean(chunk_times) if chunk_times else 0,
                    "min_inter_chunk_ms": np.min(chunk_times) if chunk_times else 0,
                    "max_inter_chunk_ms": np.max(chunk_times) if chunk_times else 0,
                }
                
                test_results.append(result)
                
                print(f"  ✓ LLM: {llm_time:.1f}ms")
                print(f"  ✓ TTS One-shot: {tts_time:.1f}ms (RTF: {rtf:.3f}x)")
                print(f"  ✓ TTS Streaming: First chunk in {first_chunk_time:.1f}ms")
                print(f"    - Total: {total_streaming_time:.1f}ms")
                print(f"    - Chunks: {len(chunks)}")
                print(f"    - Avg inter-chunk: {result['avg_inter_chunk_ms']:.1f}ms")
            
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        # Aggregate results
        if test_results:
            aggregate = {
                "device": device,
                "rvq_iterations": rvq_iterations,
                "num_tests": len(test_results),
                "avg_llm_latency_ms": np.mean([r["llm_latency_ms"] for r in test_results]),
                "avg_tts_oneshot_ms": np.mean([r["tts_oneshot_ms"] for r in test_results]),
                "avg_rtf_oneshot": np.mean([r["rtf_oneshot"] for r in test_results]),
                "avg_first_chunk_ms": np.mean([r["first_chunk_latency_ms"] for r in test_results]),
                "avg_total_streaming_ms": np.mean([r["total_streaming_ms"] for r in test_results]),
                "avg_chunks": np.mean([r["num_chunks"] for r in test_results]),
                "avg_inter_chunk_ms": np.mean([r["avg_inter_chunk_ms"] for r in test_results]),
                "test_results": test_results,
            }
            
            results.append(aggregate)
        
        # Clear context between configurations
        pipeline.clear_context()
    
    return results


def print_benchmark_results(results: List[Dict[str, Any]]):
    """Print formatted benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    
    # Print table header
    print(f"\n{'Device':<10} {'RVQ':<5} {'LLM (ms)':<12} {'First Chunk':<12} "
          f"{'RTF':<8} {'Avg Chunks':<11} {'Inter-Chunk':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['device']:<10} "
              f"{result['rvq_iterations']:<5} "
              f"{result['avg_llm_latency_ms']:>10.1f}  "
              f"{result['avg_first_chunk_ms']:>10.1f}  "
              f"{result['avg_rtf_oneshot']:>6.3f}  "
              f"{result['avg_chunks']:>9.1f}  "
              f"{result['avg_inter_chunk_ms']:>10.1f}")
    
    print("-" * 80)
    
    # Detailed breakdown
    print("\n" + "=" * 80)
    print("DETAILED METRICS")
    print("=" * 80)
    
    for result in results:
        print(f"\n{result['device'].upper()} - {result['rvq_iterations']} RVQ Iterations:")
        print(f"  Tests conducted: {result['num_tests']}")
        print(f"  LLM Latency: {result['avg_llm_latency_ms']:.1f} ms")
        print(f"  TTS One-shot: {result['avg_tts_oneshot_ms']:.1f} ms")
        print(f"  Real-Time Factor: {result['avg_rtf_oneshot']:.3f}x")
        print(f"  First Chunk Latency: {result['avg_first_chunk_ms']:.1f} ms")
        print(f"  Total Streaming Time: {result['avg_total_streaming_ms']:.1f} ms")
        print(f"  Average Chunks: {result['avg_chunks']:.1f}")
        print(f"  Average Inter-Chunk Latency: {result['avg_inter_chunk_ms']:.1f} ms")
    
    # Comparison to paper results
    print("\n" + "=" * 80)
    print("COMPARISON TO PAPER RESULTS")
    print("=" * 80)
    
    paper_results = {
        "cpu": {
            16: {"first_chunk": 1748.6, "rtf": 0.934},
            20: {"first_chunk": 1855.0, "rtf": 1.143},
            24: {"first_chunk": 2172.3, "rtf": 1.236},
            32: {"first_chunk": 2662.5, "rtf": 1.489},
        },
        "cuda": {
            16: {"first_chunk": 640.9, "rtf": 0.480},
            20: {"first_chunk": 1105.4, "rtf": 0.571},
            24: {"first_chunk": 1172.3, "rtf": 0.574},
            32: {"first_chunk": 1381.9, "rtf": 0.785},
        }
    }
    
    for result in results:
        device = result['device']
        rvq = result['rvq_iterations']
        
        if device in paper_results and rvq in paper_results[device]:
            paper = paper_results[device][rvq]
            print(f"\n{device.upper()} - {rvq} RVQ Iterations:")
            print(f"  Paper First Chunk: {paper['first_chunk']:.1f} ms")
            print(f"  Our First Chunk:   {result['avg_first_chunk_ms']:.1f} ms")
            print(f"  Difference:        {result['avg_first_chunk_ms'] - paper['first_chunk']:+.1f} ms")
            print(f"  Paper RTF:         {paper['rtf']:.3f}x")
            print(f"  Our RTF:           {result['avg_rtf_oneshot']:.3f}x")
            print(f"  Difference:        {result['avg_rtf_oneshot'] - paper['rtf']:+.3f}x")


def main():
    """Run comprehensive benchmarks."""
    print("=" * 80)
    print("i-LAVA Performance Benchmark")
    print("=" * 80)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY not set")
        return
    
    # Test texts of varying lengths
    test_texts = [
        "Hello, how are you?",
        "Tell me about the weather today.",
        "Can you explain what artificial intelligence is in simple terms?",
    ]
    
    # Determine available device
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        print("\n✓ CUDA available, will benchmark GPU")
    else:
        device = "cpu"
        print("\n⚠ CUDA not available, will benchmark CPU only")
    
    # RVQ iterations to test
    rvq_iterations_list = [16, 20, 24, 32]
    
    print(f"\n📊 Benchmark Configuration:")
    print(f"  Device: {device}")
    print(f"  RVQ Iterations: {rvq_iterations_list}")
    print(f"  Test Texts: {len(test_texts)}")
    
    # Run benchmarks
    print("\n🚀 Starting benchmarks...")
    results = run_benchmark(device, rvq_iterations_list, test_texts)
    
    # Print results
    print_benchmark_results(results)
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

