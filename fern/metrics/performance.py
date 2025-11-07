"""Performance metrics tracking and calculation for i-LAVA pipeline."""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

import numpy as np
import torch
from scipy import signal

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Latency-related performance metrics."""
    first_chunk_latency_ms: float = 0.0
    avg_inter_chunk_latency_ms: float = 0.0
    min_inter_chunk_latency_ms: float = float('inf')
    max_inter_chunk_latency_ms: float = 0.0
    total_processing_time_ms: float = 0.0
    asr_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "first_chunk_latency_ms": self.first_chunk_latency_ms,
            "avg_inter_chunk_latency_ms": self.avg_inter_chunk_latency_ms,
            "min_inter_chunk_latency_ms": self.min_inter_chunk_latency_ms,
            "max_inter_chunk_latency_ms": self.max_inter_chunk_latency_ms,
            "total_processing_time_ms": self.total_processing_time_ms,
            "asr_latency_ms": self.asr_latency_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "tts_latency_ms": self.tts_latency_ms,
        }


@dataclass
class ChunkMetrics:
    """Audio chunk-related metrics."""
    num_chunks: int = 0
    avg_chunk_size_ms: float = 0.0
    total_audio_duration_ms: float = 0.0
    chunks_per_second: float = 0.0
    chunk_sizes: List[float] = field(default_factory=list)
    inter_chunk_latencies: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_chunks": self.num_chunks,
            "avg_chunk_size_ms": self.avg_chunk_size_ms,
            "total_audio_duration_ms": self.total_audio_duration_ms,
            "chunks_per_second": self.chunks_per_second,
        }


@dataclass
class AudioQualityMetrics:
    """Audio quality metrics."""
    snr_db: Optional[float] = None
    pesq_score: Optional[float] = None
    stoi_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary."""
        return {
            "snr_db": self.snr_db,
            "pesq_score": self.pesq_score,
            "stoi_score": self.stoi_score,
        }


@dataclass
class PerformanceMetrics:
    """Complete performance metrics for i-LAVA pipeline."""
    real_time_factor: float = 0.0
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    chunks: ChunkMetrics = field(default_factory=ChunkMetrics)
    quality: AudioQualityMetrics = field(default_factory=AudioQualityMetrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "real_time_factor": self.real_time_factor,
            "latency": self.latency.to_dict(),
            "chunks": self.chunks.to_dict(),
            "quality": self.quality.to_dict(),
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "i-LAVA Performance Metrics Summary",
            "=" * 60,
            f"Real-Time Factor (RTF):           {self.real_time_factor:.3f}x",
            "",
            "Latency Metrics:",
            f"  First Chunk Latency:            {self.latency.first_chunk_latency_ms:.1f} ms",
            f"  Average Inter-Chunk Latency:    {self.latency.avg_inter_chunk_latency_ms:.1f} ms",
            f"  Min Inter-Chunk Latency:        {self.latency.min_inter_chunk_latency_ms:.1f} ms",
            f"  Max Inter-Chunk Latency:        {self.latency.max_inter_chunk_latency_ms:.1f} ms",
            f"  ASR Latency:                    {self.latency.asr_latency_ms:.1f} ms",
            f"  LLM Latency:                    {self.latency.llm_latency_ms:.1f} ms",
            f"  TTS Latency:                    {self.latency.tts_latency_ms:.1f} ms",
            f"  Total Processing Time:          {self.latency.total_processing_time_ms:.1f} ms",
            "",
            "Chunk Metrics:",
            f"  Number of Chunks:               {self.chunks.num_chunks}",
            f"  Average Chunk Size:             {self.chunks.avg_chunk_size_ms:.1f} ms",
            f"  Total Audio Duration:           {self.chunks.total_audio_duration_ms:.1f} ms",
            f"  Chunks per Second:              {self.chunks.chunks_per_second:.2f}",
            "",
            "Audio Quality Metrics:",
        ]
        
        if self.quality.snr_db is not None:
            lines.append(f"  Signal-to-Noise Ratio (SNR):    {self.quality.snr_db:.3f} dB")
        if self.quality.pesq_score is not None:
            lines.append(f"  PESQ Score:                     {self.quality.pesq_score:.3f}")
        if self.quality.stoi_score is not None:
            lines.append(f"  STOI Score:                     {self.quality.stoi_score:.3f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class MetricsTracker:
    """Tracks and calculates performance metrics for i-LAVA pipeline."""
    
    def __init__(self, sample_rate: int = 24000):
        """
        Initialize metrics tracker.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self._start_time: Optional[float] = None
        self._first_chunk_time: Optional[float] = None
        self._last_chunk_time: Optional[float] = None
        
        self._asr_start: Optional[float] = None
        self._asr_end: Optional[float] = None
        self._llm_start: Optional[float] = None
        self._llm_end: Optional[float] = None
        self._tts_start: Optional[float] = None
        self._tts_end: Optional[float] = None
        
        self._chunk_times: List[float] = []
        self._chunk_sizes: List[float] = []
        self._inter_chunk_latencies: List[float] = []
        
        self._input_audio_duration: float = 0.0
        self._output_audio_duration: float = 0.0
        
        self._reference_audio: Optional[np.ndarray] = None
        self._generated_audio: Optional[np.ndarray] = None
    
    def start_pipeline(self):
        """Mark the start of pipeline processing."""
        self._start_time = time.time()
        logger.debug("Pipeline processing started")
    
    def start_asr(self):
        """Mark the start of ASR processing."""
        self._asr_start = time.time()
    
    def end_asr(self):
        """Mark the end of ASR processing."""
        self._asr_end = time.time()
    
    def start_llm(self):
        """Mark the start of LLM processing."""
        self._llm_start = time.time()
    
    def end_llm(self):
        """Mark the end of LLM processing."""
        self._llm_end = time.time()
    
    def start_tts(self):
        """Mark the start of TTS processing."""
        self._tts_start = time.time()
    
    def end_tts(self):
        """Mark the end of TTS processing."""
        self._tts_end = time.time()
    
    def record_chunk(self, audio_chunk: np.ndarray):
        """
        Record generation of an audio chunk.
        
        Args:
            audio_chunk: Generated audio chunk as numpy array
        """
        current_time = time.time()
        
        if self._first_chunk_time is None:
            self._first_chunk_time = current_time
            logger.debug(f"First chunk generated at {(current_time - self._start_time) * 1000:.1f} ms")
        else:
            # Calculate inter-chunk latency
            inter_chunk_latency = (current_time - self._last_chunk_time) * 1000
            self._inter_chunk_latencies.append(inter_chunk_latency)
        
        self._chunk_times.append(current_time)
        
        # Calculate chunk duration in ms
        chunk_duration_ms = (len(audio_chunk) / self.sample_rate) * 1000
        self._chunk_sizes.append(chunk_duration_ms)
        self._output_audio_duration += chunk_duration_ms
        
        self._last_chunk_time = current_time
    
    def set_input_audio(self, audio: np.ndarray):
        """
        Set input audio for metrics calculation.
        
        Args:
            audio: Input audio as numpy array
        """
        self._input_audio_duration = (len(audio) / self.sample_rate) * 1000
        self._reference_audio = audio
    
    def set_output_audio(self, audio: np.ndarray):
        """
        Set complete output audio for quality metrics.
        
        Args:
            audio: Output audio as numpy array
        """
        self._generated_audio = audio
    
    def calculate_snr(
        self,
        clean_audio: np.ndarray,
        noisy_audio: np.ndarray
    ) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR) in dB.
        
        Implementation based on:
        Kim, C., & Stern, R. M. (2008). Robust Signal-to-Noise Ratio Estimation
        Based on Waveform Amplitude Distribution Analysis.
        
        Args:
            clean_audio: Clean reference audio
            noisy_audio: Noisy audio to evaluate
            
        Returns:
            SNR in dB
        """
        # Ensure same length
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        
        # Calculate noise as difference
        noise = noisy_audio - clean_audio
        
        # Calculate power
        signal_power = np.mean(clean_audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Avoid division by zero
        if noise_power < 1e-10:
            return float('inf')
        
        # Calculate SNR in dB
        snr_db = 10 * np.log10(signal_power / noise_power)
        
        return snr_db
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        metrics = PerformanceMetrics()
        
        # Calculate latency metrics
        if self._start_time and self._first_chunk_time:
            metrics.latency.first_chunk_latency_ms = (
                (self._first_chunk_time - self._start_time) * 1000
            )
        
        if self._inter_chunk_latencies:
            metrics.latency.avg_inter_chunk_latency_ms = np.mean(
                self._inter_chunk_latencies
            )
            metrics.latency.min_inter_chunk_latency_ms = np.min(
                self._inter_chunk_latencies
            )
            metrics.latency.max_inter_chunk_latency_ms = np.max(
                self._inter_chunk_latencies
            )
        
        if self._asr_start and self._asr_end:
            metrics.latency.asr_latency_ms = (self._asr_end - self._asr_start) * 1000
        
        if self._llm_start and self._llm_end:
            metrics.latency.llm_latency_ms = (self._llm_end - self._llm_start) * 1000
        
        if self._tts_start and self._tts_end:
            metrics.latency.tts_latency_ms = (self._tts_end - self._tts_start) * 1000
        
        if self._start_time and self._last_chunk_time:
            metrics.latency.total_processing_time_ms = (
                (self._last_chunk_time - self._start_time) * 1000
            )
        
        # Calculate chunk metrics
        metrics.chunks.num_chunks = len(self._chunk_sizes)
        if self._chunk_sizes:
            metrics.chunks.avg_chunk_size_ms = np.mean(self._chunk_sizes)
            metrics.chunks.chunk_sizes = self._chunk_sizes
        
        metrics.chunks.total_audio_duration_ms = self._output_audio_duration
        metrics.chunks.inter_chunk_latencies = self._inter_chunk_latencies
        
        if metrics.latency.total_processing_time_ms > 0:
            metrics.chunks.chunks_per_second = (
                metrics.chunks.num_chunks /
                (metrics.latency.total_processing_time_ms / 1000)
            )
        
        # Calculate Real-Time Factor (RTF)
        if self._output_audio_duration > 0 and metrics.latency.total_processing_time_ms > 0:
            metrics.real_time_factor = (
                metrics.latency.total_processing_time_ms /
                self._output_audio_duration
            )
        
        # Calculate audio quality metrics
        if self._reference_audio is not None and self._generated_audio is not None:
            try:
                metrics.quality.snr_db = self.calculate_snr(
                    self._reference_audio,
                    self._generated_audio
                )
            except Exception as e:
                logger.warning(f"Failed to calculate SNR: {e}")
        
        logger.info(f"Metrics calculated: RTF={metrics.real_time_factor:.3f}x")
        
        return metrics
    
    def log_metrics(self, metrics: PerformanceMetrics):
        """
        Log metrics to logger.
        
        Args:
            metrics: Metrics to log
        """
        logger.info("\n" + metrics.summary())

