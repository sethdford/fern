"""
Analyze profiling results and generate optimization recommendations.

This script analyzes torch profiler traces and generates a data-driven
optimization plan with ROI estimates.

Usage:
    python tools/profiling/analyze_results.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re

def load_profiling_report(report_path: Path) -> Dict:
    """Load profiling report JSON."""
    with open(report_path) as f:
        return json.load(f)


def analyze_chrome_trace(trace_path: Path) -> Dict:
    """
    Analyze Chrome trace JSON for bottlenecks.
    
    Args:
        trace_path: Path to trace JSON file
        
    Returns:
        Analysis results
    """
    print(f"Analyzing trace: {trace_path.name}")
    
    with open(trace_path) as f:
        trace = json.load(f)
    
    # Parse trace events
    events = trace.get('traceEvents', [])
    
    # Group by operation name
    op_times = {}
    for event in events:
        if event.get('ph') == 'X':  # Complete events
            name = event.get('name', 'unknown')
            dur = event.get('dur', 0) / 1000  # Convert to ms
            
            if name not in op_times:
                op_times[name] = []
            op_times[name].append(dur)
    
    # Calculate statistics
    op_stats = {}
    for name, times in op_times.items():
        if times:
            op_stats[name] = {
                'count': len(times),
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'max_time': max(times),
            }
    
    # Sort by total time
    sorted_ops = sorted(op_stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
    
    return {
        'total_ops': len(op_stats),
        'top_operations': sorted_ops[:20],
    }


def identify_bottlenecks(analysis: Dict) -> List[Dict]:
    """
    Identify optimization opportunities from analysis.
    
    Args:
        analysis: Analysis results
        
    Returns:
        List of bottleneck descriptions with optimization suggestions
    """
    bottlenecks = []
    
    top_ops = analysis.get('top_operations', [])
    
    # Categorize operations
    for op_name, stats in top_ops[:10]:
        bottleneck = {
            'operation': op_name,
            'total_time_ms': stats['total_time'],
            'avg_time_ms': stats['avg_time'],
            'count': stats['count'],
            'category': categorize_operation(op_name),
            'optimization': suggest_optimization(op_name, stats),
        }
        bottlenecks.append(bottleneck)
    
    return bottlenecks


def categorize_operation(op_name: str) -> str:
    """Categorize operation by component."""
    op_lower = op_name.lower()
    
    if 'embedding' in op_lower or 'quantize' in op_lower or 'rvq' in op_lower:
        return 'RVQ/Quantization'
    elif 'attention' in op_lower or 'transformer' in op_lower:
        return 'Transformer'
    elif 'mimi' in op_lower or 'codec' in op_lower:
        return 'Mimi Codec'
    elif 'matmul' in op_lower or 'linear' in op_lower or 'bmm' in op_lower:
        return 'Matrix Operations'
    elif 'conv' in op_lower:
        return 'Convolution'
    elif 'norm' in op_lower:
        return 'Normalization'
    else:
        return 'Other'


def suggest_optimization(op_name: str, stats: Dict) -> str:
    """Suggest optimization for operation."""
    category = categorize_operation(op_name)
    
    optimizations = {
        'RVQ/Quantization': 'CUDA kernel for parallel codebook lookup',
        'Transformer': 'torch.compile, Flash Attention, or KV cache optimization',
        'Mimi Codec': 'ONNX Runtime or C++ implementation',
        'Matrix Operations': 'Ensure using cuBLAS/TensorCores, consider mixed precision',
        'Convolution': 'cuDNN optimization, channel-last format',
        'Normalization': 'Fused kernel, torch.compile',
        'Other': 'Analyze specific operation',
    }
    
    return optimizations.get(category, 'Further analysis needed')


def estimate_speedup_potential(bottlenecks: List[Dict], total_time_ms: float) -> List[Dict]:
    """
    Estimate speedup potential for each optimization.
    
    Args:
        bottlenecks: List of bottlenecks
        total_time_ms: Total execution time
        
    Returns:
        Bottlenecks with speedup estimates
    """
    for bottleneck in bottlenecks:
        time_ms = bottleneck['total_time_ms']
        percentage = (time_ms / total_time_ms * 100) if total_time_ms > 0 else 0
        
        # Estimate potential speedup based on category
        category = bottleneck['category']
        if category == 'RVQ/Quantization':
            potential_speedup = 10.0  # CUDA can be 10x faster
        elif category == 'Transformer':
            potential_speedup = 2.0  # torch.compile ~2x
        elif category == 'Mimi Codec':
            potential_speedup = 6.0  # ONNX ~6x
        elif category == 'Matrix Operations':
            potential_speedup = 1.5  # TensorCores ~1.5x
        else:
            potential_speedup = 1.2  # Modest improvement
        
        # Calculate time saved
        time_saved_ms = time_ms * (1 - 1/potential_speedup)
        total_speedup = 1 / (1 - time_saved_ms / total_time_ms) if total_time_ms > 0 else 1.0
        
        bottleneck['percentage_of_total'] = percentage
        bottleneck['potential_speedup'] = potential_speedup
        bottleneck['time_saved_ms'] = time_saved_ms
        bottleneck['total_pipeline_speedup'] = total_speedup
        bottleneck['roi'] = percentage * (potential_speedup - 1)  # Simple ROI metric
    
    # Sort by ROI
    bottlenecks.sort(key=lambda x: x['roi'], reverse=True)
    
    return bottlenecks


def generate_report(bottlenecks: List[Dict], output_path: Path) -> None:
    """
    Generate optimization report.
    
    Args:
        bottlenecks: Analyzed bottlenecks
        output_path: Output file path
    """
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("PROFILING ANALYSIS REPORT")
    report_lines.append("Phase 2.1: Bottleneck Identification & Optimization Plan")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append("TOP BOTTLENECKS (by ROI)")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    for i, b in enumerate(bottlenecks[:10], 1):
        report_lines.append(f"{i}. {b['operation']}")
        report_lines.append(f"   Category: {b['category']}")
        report_lines.append(f"   Time: {b['total_time_ms']:.2f}ms ({b['percentage_of_total']:.1f}% of total)")
        report_lines.append(f"   Optimization: {b['optimization']}")
        report_lines.append(f"   Potential Speedup: {b['potential_speedup']:.1f}x → saves {b['time_saved_ms']:.2f}ms")
        report_lines.append(f"   Pipeline Speedup: {b['total_pipeline_speedup']:.2f}x")
        report_lines.append(f"   ROI Score: {b['roi']:.1f}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("OPTIMIZATION RECOMMENDATIONS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Group by category
    categories = {}
    for b in bottlenecks:
        cat = b['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(b)
    
    # Recommendations by category
    sorted_cats = sorted(categories.items(), key=lambda x: sum(b['roi'] for b in x[1]), reverse=True)
    
    for i, (cat, ops) in enumerate(sorted_cats[:5], 1):
        total_time = sum(b['total_time_ms'] for b in ops)
        total_roi = sum(b['roi'] for b in ops)
        
        report_lines.append(f"{i}. {cat}")
        report_lines.append(f"   Total Time: {total_time:.2f}ms")
        report_lines.append(f"   Total ROI: {total_roi:.1f}")
        report_lines.append(f"   Optimization: {ops[0]['optimization']}")
        report_lines.append("")
    
    # Write report
    report_text = "\n".join(report_lines)
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    # Also print to console
    print()
    print(report_text)


def main():
    """Run analysis."""
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                      ║")
    print("║           PROFILING ANALYSIS & OPTIMIZATION PLANNER                  ║")
    print("║           Phase 2.1: Data-Driven Optimization                        ║")
    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Find latest trace
    trace_dir = Path(__file__).parent.parent.parent / 'benchmarks' / 'traces'
    
    if not trace_dir.exists() or not any(trace_dir.glob('*.json')):
        print("❌ No profiling traces found!")
        print()
        print("Please run profiling first:")
        print("  python tools/profiling/profile_tts.py --real --trace")
        print()
        return 1
    
    # Find most recent trace
    traces = list(trace_dir.glob('tts_trace_*.json'))
    if not traces:
        print("❌ No TTS traces found!")
        return 1
    
    latest_trace = max(traces, key=lambda p: p.stat().st_mtime)
    print(f"Using trace: {latest_trace.name}")
    print()
    
    # Analyze
    print("Analyzing profiling data...")
    analysis = analyze_chrome_trace(latest_trace)
    
    print(f"✓ Found {analysis['total_ops']} operations")
    print()
    
    # Identify bottlenecks
    print("Identifying bottlenecks...")
    bottlenecks = identify_bottlenecks(analysis)
    print(f"✓ Identified {len(bottlenecks)} potential optimizations")
    print()
    
    # Estimate speedups
    print("Estimating optimization ROI...")
    
    # Get total time from report
    report_dir = Path(__file__).parent.parent.parent / 'benchmarks' / 'profiling'
    report_file = report_dir / 'tts_profiling_report.json'
    
    if report_file.exists():
        report = load_profiling_report(report_file)
        total_time_ms = report['results']['stats']['mean']
    else:
        # Estimate from trace
        total_time_ms = sum(b['total_time_ms'] for b in bottlenecks)
    
    bottlenecks = estimate_speedup_potential(bottlenecks, total_time_ms)
    print("✓ ROI estimates complete")
    print()
    
    # Generate report
    output_dir = Path(__file__).parent.parent.parent / 'benchmarks' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'optimization_plan.txt'
    
    generate_report(bottlenecks, output_file)
    
    print(f"✓ Report saved to: {output_file}")
    print()
    
    # Summary
    top_opt = bottlenecks[0]
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print(f"🎯 Highest Priority: {top_opt['category']}")
    print(f"   Operation: {top_opt['operation']}")
    print(f"   Current Time: {top_opt['total_time_ms']:.2f}ms ({top_opt['percentage_of_total']:.1f}%)")
    print(f"   Potential Speedup: {top_opt['potential_speedup']:.1f}x")
    print(f"   Pipeline Impact: {top_opt['total_pipeline_speedup']:.2f}x faster")
    print(f"   Optimization: {top_opt['optimization']}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

