"""Tests for metrics module."""

import pytest
import numpy as np
import time
from fern.metrics import MetricsTracker, PerformanceMetrics


class TestMetricsTracker:
    """Test MetricsTracker."""
    
    def test_initialization(self):
        """Test metrics tracker initialization."""
        tracker = MetricsTracker(sample_rate=24000)
        assert tracker.sample_rate == 24000
    
    def test_pipeline_timing(self):
        """Test pipeline timing tracking."""
        tracker = MetricsTracker()
        
        tracker.start_pipeline()
        time.sleep(0.1)
        
        # Record first chunk
        chunk = np.random.randn(1000).astype(np.float32)
        tracker.record_chunk(chunk)
        
        metrics = tracker.calculate_metrics()
        
        assert metrics.latency.first_chunk_latency_ms > 0
        assert metrics.chunks.num_chunks == 1
    
    def test_component_timing(self):
        """Test individual component timing."""
        tracker = MetricsTracker()
        
        tracker.start_pipeline()
        
        # ASR
        tracker.start_asr()
        time.sleep(0.05)
        tracker.end_asr()
        
        # LLM
        tracker.start_llm()
        time.sleep(0.05)
        tracker.end_llm()
        
        # TTS
        tracker.start_tts()
        time.sleep(0.05)
        tracker.end_tts()
        
        metrics = tracker.calculate_metrics()
        
        assert metrics.latency.asr_latency_ms > 0
        assert metrics.latency.llm_latency_ms > 0
        assert metrics.latency.tts_latency_ms > 0
    
    def test_chunk_tracking(self):
        """Test audio chunk tracking."""
        tracker = MetricsTracker(sample_rate=24000)
        tracker.start_pipeline()
        
        # Record multiple chunks
        for i in range(5):
            chunk = np.random.randn(1000).astype(np.float32)
            tracker.record_chunk(chunk)
            time.sleep(0.01)
        
        metrics = tracker.calculate_metrics()
        
        assert metrics.chunks.num_chunks == 5
        assert metrics.chunks.avg_chunk_size_ms > 0
        assert len(metrics.chunks.inter_chunk_latencies) == 4  # n-1 inter-chunk latencies
    
    def test_rtf_calculation(self):
        """Test Real-Time Factor calculation."""
        tracker = MetricsTracker(sample_rate=24000)
        tracker.start_pipeline()
        
        # Simulate generation
        time.sleep(0.1)
        
        # Record chunk (1000 samples at 24kHz = ~41.67ms)
        chunk = np.random.randn(1000).astype(np.float32)
        tracker.record_chunk(chunk)
        
        metrics = tracker.calculate_metrics()
        
        # RTF = processing_time / audio_duration
        assert metrics.real_time_factor > 0
    
    def test_snr_calculation(self):
        """Test SNR calculation."""
        tracker = MetricsTracker()
        
        # Create clean and noisy signals
        clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1000))
        noise = np.random.randn(1000) * 0.1
        noisy = clean + noise
        
        snr = tracker.calculate_snr(clean, noisy)
        
        assert snr > 0  # Should have positive SNR
        assert snr < 50  # Reasonable upper bound


class TestPerformanceMetrics:
    """Test PerformanceMetrics."""
    
    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary."""
        metrics = PerformanceMetrics()
        metrics.real_time_factor = 0.5
        
        metrics_dict = metrics.to_dict()
        
        assert "real_time_factor" in metrics_dict
        assert metrics_dict["real_time_factor"] == 0.5
        assert "latency" in metrics_dict
        assert "chunks" in metrics_dict
        assert "quality" in metrics_dict
    
    def test_metrics_summary(self):
        """Test metrics summary generation."""
        metrics = PerformanceMetrics()
        metrics.real_time_factor = 0.693
        metrics.latency.first_chunk_latency_ms = 1021.2
        metrics.chunks.num_chunks = 6
        
        summary = metrics.summary()
        
        assert "Real-Time Factor" in summary
        assert "0.693" in summary
        assert "First Chunk Latency" in summary
        assert "1021.2" in summary

