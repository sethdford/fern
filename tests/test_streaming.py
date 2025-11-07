"""Tests for streaming pipeline."""

import pytest
import numpy as np
import time
from fern.pipeline.streaming import (
    ChunkedAudioBuffer,
    LatencyMonitor,
    AudioStreamMerger,
)


class TestChunkedAudioBuffer:
    """Test ChunkedAudioBuffer."""
    
    def test_initialization(self):
        """Test buffer initialization."""
        buffer = ChunkedAudioBuffer(chunk_size=512)
        assert buffer.chunk_size == 512
        assert len(buffer.buffer) == 0
    
    def test_add_samples(self):
        """Test adding samples and yielding chunks."""
        buffer = ChunkedAudioBuffer(chunk_size=100)
        
        # Add 250 samples (should yield 2 chunks of 100)
        samples = np.random.randn(250).astype(np.float32)
        chunks = list(buffer.add_samples(samples))
        
        assert len(chunks) == 2
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100
        assert len(buffer.buffer) == 50  # Remaining samples
    
    def test_flush(self):
        """Test flushing remaining samples."""
        buffer = ChunkedAudioBuffer(chunk_size=100)
        
        # Add 150 samples
        samples = np.random.randn(150).astype(np.float32)
        chunks = list(buffer.add_samples(samples))
        
        assert len(chunks) == 1
        assert len(buffer.buffer) == 50
        
        # Flush
        remaining = buffer.flush()
        assert remaining is not None
        assert len(remaining) == 50
        assert len(buffer.buffer) == 0
    
    def test_clear(self):
        """Test clearing buffer."""
        buffer = ChunkedAudioBuffer(chunk_size=100)
        
        samples = np.random.randn(50).astype(np.float32)
        list(buffer.add_samples(samples))
        
        buffer.clear()
        assert len(buffer.buffer) == 0


class TestLatencyMonitor:
    """Test LatencyMonitor."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = LatencyMonitor()
        assert monitor.first_chunk_time is None
        assert len(monitor.chunk_times) == 0
    
    def test_first_chunk_timing(self):
        """Test first chunk latency measurement."""
        monitor = LatencyMonitor()
        
        monitor.start()
        time.sleep(0.05)
        
        latency = monitor.record_chunk()
        
        assert monitor.first_chunk_time is not None
        assert latency is None  # First chunk returns None
    
    def test_inter_chunk_timing(self):
        """Test inter-chunk latency measurement."""
        monitor = LatencyMonitor()
        
        monitor.start()
        time.sleep(0.05)
        monitor.record_chunk()  # First chunk
        
        time.sleep(0.02)
        latency = monitor.record_chunk()  # Second chunk
        
        assert latency is not None
        assert latency > 10  # Should be ~20ms
        assert len(monitor.chunk_times) == 1
    
    def test_get_stats(self):
        """Test statistics calculation."""
        monitor = LatencyMonitor()
        
        monitor.start()
        monitor.record_chunk()  # First
        
        for _ in range(5):
            time.sleep(0.01)
            monitor.record_chunk()
        
        stats = monitor.get_stats()
        
        assert "avg_inter_chunk_latency_ms" in stats
        assert "min_inter_chunk_latency_ms" in stats
        assert "max_inter_chunk_latency_ms" in stats
        assert stats["num_chunks"] == 6


class TestAudioStreamMerger:
    """Test AudioStreamMerger."""
    
    def test_initialization(self):
        """Test merger initialization."""
        merger = AudioStreamMerger(
            max_context_duration_ms=5000.0,
            sample_rate=24000
        )
        
        assert merger.max_context_duration_ms == 5000.0
        assert merger.sample_rate == 24000
        assert len(merger.context_audio) == 0
    
    def test_add_audio(self):
        """Test adding audio to context."""
        merger = AudioStreamMerger(
            max_context_duration_ms=1000.0,
            sample_rate=24000
        )
        
        # Add 0.5s of audio
        audio1 = np.random.randn(12000).astype(np.float32)
        merger.add_audio(audio1)
        
        assert len(merger.context_audio) == 12000
    
    def test_context_trimming(self):
        """Test context is trimmed to max duration."""
        merger = AudioStreamMerger(
            max_context_duration_ms=1000.0,  # 1 second
            sample_rate=24000
        )
        
        # Add 2 seconds of audio
        audio = np.random.randn(48000).astype(np.float32)
        merger.add_audio(audio)
        
        # Should be trimmed to 1 second (24000 samples)
        assert len(merger.context_audio) == 24000
    
    def test_get_context(self):
        """Test getting context audio."""
        merger = AudioStreamMerger()
        
        audio = np.random.randn(1000).astype(np.float32)
        merger.add_audio(audio)
        
        context = merger.get_context()
        
        assert len(context) == 1000
        assert np.allclose(context, audio)
        # Should be a copy
        assert context is not merger.context_audio
    
    def test_clear(self):
        """Test clearing context."""
        merger = AudioStreamMerger()
        
        audio = np.random.randn(1000).astype(np.float32)
        merger.add_audio(audio)
        
        merger.clear()
        
        assert len(merger.context_audio) == 0

