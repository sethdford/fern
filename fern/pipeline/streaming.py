"""Streaming pipeline for real-time audio processing."""

import logging
import time
import queue
import threading
from typing import Optional, Iterator, Callable

import numpy as np

logger = logging.getLogger(__name__)


class StreamingPipeline:
    """
    Streaming pipeline for real-time audio generation.
    
    Manages audio chunk queue and ensures seamless streaming by:
    1. Generating audio chunks asynchronously
    2. Maintaining non-empty queue for smooth playback
    3. Bridging gap between speech end and response start
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        max_queue_size: int = 10,
        prefetch_chunks: int = 2,
    ):
        """
        Initialize streaming pipeline.
        
        Args:
            chunk_size: Size of audio chunks in samples
            max_queue_size: Maximum queue size
            prefetch_chunks: Number of chunks to prefetch before playing
        """
        self.chunk_size = chunk_size
        self.max_queue_size = max_queue_size
        self.prefetch_chunks = prefetch_chunks
        
        self.audio_queue = queue.Queue(maxsize=max_queue_size)
        self.is_generating = False
        self.generation_error: Optional[Exception] = None
        
        logger.info(
            f"Streaming pipeline initialized: "
            f"chunk_size={chunk_size}, prefetch={prefetch_chunks}"
        )
    
    def stream_audio(
        self,
        generate_fn: Callable[[], Iterator[np.ndarray]],
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> Iterator[np.ndarray]:
        """
        Stream audio chunks with asynchronous generation.
        
        Args:
            generate_fn: Function that yields audio chunks
            callback: Optional callback for each chunk
            
        Yields:
            Audio chunks as numpy arrays
        """
        self.audio_queue = queue.Queue(maxsize=self.max_queue_size)
        self.is_generating = True
        self.generation_error = None
        
        # Start generation thread
        generation_thread = threading.Thread(
            target=self._generate_chunks,
            args=(generate_fn,),
            daemon=True
        )
        generation_thread.start()
        
        # Wait for prefetch
        chunks_received = 0
        while chunks_received < self.prefetch_chunks and self.is_generating:
            try:
                chunk = self.audio_queue.get(timeout=5.0)
                chunks_received += 1
                
                if callback:
                    callback(chunk)
                
                yield chunk
            except queue.Empty:
                if not self.is_generating:
                    break
                logger.warning("Prefetch timeout, starting playback anyway")
                break
        
        logger.info(f"Prefetched {chunks_received} chunks, starting playback")
        
        # Stream remaining chunks
        while self.is_generating or not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                
                if callback:
                    callback(chunk)
                
                yield chunk
            except queue.Empty:
                if not self.is_generating:
                    break
                logger.debug("Queue empty, waiting for more chunks...")
        
        # Check for errors
        if self.generation_error:
            logger.error(f"Generation error: {self.generation_error}")
            raise self.generation_error
        
        logger.info("Streaming complete")
    
    def _generate_chunks(self, generate_fn: Callable[[], Iterator[np.ndarray]]):
        """
        Generate audio chunks in background thread.
        
        Args:
            generate_fn: Function that yields audio chunks
        """
        try:
            for chunk in generate_fn():
                # Add chunk to queue (blocking if full)
                self.audio_queue.put(chunk)
        except Exception as e:
            logger.error(f"Chunk generation failed: {e}")
            self.generation_error = e
        finally:
            self.is_generating = False
    
    def clear_queue(self):
        """Clear audio queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.debug("Audio queue cleared")


class ChunkedAudioBuffer:
    """
    Buffer for managing chunked audio processing.
    
    Accumulates audio samples and yields fixed-size chunks.
    """
    
    def __init__(self, chunk_size: int = 512):
        """
        Initialize chunked audio buffer.
        
        Args:
            chunk_size: Size of chunks to yield
        """
        self.chunk_size = chunk_size
        self.buffer = np.array([], dtype=np.float32)
    
    def add_samples(self, samples: np.ndarray) -> Iterator[np.ndarray]:
        """
        Add samples to buffer and yield complete chunks.
        
        Args:
            samples: Audio samples to add
            
        Yields:
            Complete audio chunks
        """
        self.buffer = np.concatenate([self.buffer, samples])
        
        # Yield complete chunks
        while len(self.buffer) >= self.chunk_size:
            chunk = self.buffer[:self.chunk_size]
            self.buffer = self.buffer[self.chunk_size:]
            yield chunk
    
    def flush(self) -> Optional[np.ndarray]:
        """
        Flush remaining samples in buffer.
        
        Returns:
            Remaining samples, or None if empty
        """
        if len(self.buffer) > 0:
            chunk = self.buffer.copy()
            self.buffer = np.array([], dtype=np.float32)
            return chunk
        return None
    
    def clear(self):
        """Clear buffer."""
        self.buffer = np.array([], dtype=np.float32)


class LatencyMonitor:
    """Monitor inter-chunk latency for streaming."""
    
    def __init__(self):
        """Initialize latency monitor."""
        self.chunk_times = []
        self.last_chunk_time: Optional[float] = None
        self.first_chunk_time: Optional[float] = None
        self.start_time: Optional[float] = None
    
    def start(self):
        """Mark start of generation."""
        self.start_time = time.time()
        self.first_chunk_time = None
        self.last_chunk_time = None
        self.chunk_times = []
    
    def record_chunk(self) -> Optional[float]:
        """
        Record chunk generation.
        
        Returns:
            Inter-chunk latency in ms, or None for first chunk
        """
        current_time = time.time()
        
        if self.first_chunk_time is None:
            # First chunk
            self.first_chunk_time = current_time
            self.last_chunk_time = current_time
            
            if self.start_time:
                first_chunk_latency = (current_time - self.start_time) * 1000
                logger.info(f"First chunk latency: {first_chunk_latency:.1f}ms")
                return None
        else:
            # Subsequent chunks
            inter_chunk_latency = (current_time - self.last_chunk_time) * 1000
            self.chunk_times.append(inter_chunk_latency)
            self.last_chunk_time = current_time
            
            return inter_chunk_latency
        
        return None
    
    def get_stats(self) -> dict:
        """
        Get latency statistics.
        
        Returns:
            Dictionary with latency stats
        """
        if not self.chunk_times:
            return {}
        
        return {
            "avg_inter_chunk_latency_ms": np.mean(self.chunk_times),
            "min_inter_chunk_latency_ms": np.min(self.chunk_times),
            "max_inter_chunk_latency_ms": np.max(self.chunk_times),
            "num_chunks": len(self.chunk_times) + 1,  # +1 for first chunk
        }


class AudioStreamMerger:
    """
    Merges multiple audio streams for context-aware generation.
    
    Useful for maintaining audio context across multiple generation calls.
    """
    
    def __init__(self, max_context_duration_ms: float = 5000.0, sample_rate: int = 24000):
        """
        Initialize audio stream merger.
        
        Args:
            max_context_duration_ms: Maximum context duration in milliseconds
            sample_rate: Audio sample rate
        """
        self.max_context_duration_ms = max_context_duration_ms
        self.sample_rate = sample_rate
        self.max_samples = int((max_context_duration_ms / 1000.0) * sample_rate)
        
        self.context_audio = np.array([], dtype=np.float32)
    
    def add_audio(self, audio: np.ndarray):
        """
        Add audio to context.
        
        Args:
            audio: Audio samples to add
        """
        self.context_audio = np.concatenate([self.context_audio, audio])
        
        # Trim to max length
        if len(self.context_audio) > self.max_samples:
            self.context_audio = self.context_audio[-self.max_samples:]
    
    def get_context(self) -> np.ndarray:
        """
        Get current audio context.
        
        Returns:
            Audio context as numpy array
        """
        return self.context_audio.copy()
    
    def clear(self):
        """Clear audio context."""
        self.context_audio = np.array([], dtype=np.float32)

