"""
Voice Activity Detection using WebRTC VAD.

Provides real-time speech detection for better turn-taking.
"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    logger.warning("webrtcvad not installed - using simple energy-based detection")


class VADDetector:
    """
    Voice Activity Detection for turn-taking and silence removal.
    
    Uses WebRTC VAD when available, falls back to energy-based detection.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        aggressiveness: int = 2,
        energy_threshold: float = 0.01,
    ):
        """
        Initialize VAD detector.
        
        Args:
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000)
            frame_duration_ms: Frame duration (10, 20, or 30 ms)
            aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
            energy_threshold: Energy threshold for fallback detector
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.aggressiveness = aggressiveness
        self.energy_threshold = energy_threshold
        
        # Initialize WebRTC VAD if available
        if WEBRTC_AVAILABLE:
            self.vad = webrtcvad.Vad(aggressiveness)
            logger.info(f"WebRTC VAD initialized (aggressiveness={aggressiveness})")
        else:
            self.vad = None
            logger.info(f"Using energy-based VAD (threshold={energy_threshold})")
        
        # Frame settings
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.bytes_per_frame = self.frame_size * 2  # 16-bit PCM
    
    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """
        Detect if audio frame contains speech.
        
        Args:
            audio_frame: Audio frame as numpy array (float32, range [-1, 1])
        
        Returns:
            True if speech detected, False otherwise
        """
        if self.vad is not None:
            # Use WebRTC VAD
            try:
                # Convert float32 to int16 PCM
                audio_int16 = (audio_frame * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                
                # Detect speech
                return self.vad.is_speech(audio_bytes, self.sample_rate)
            
            except Exception as e:
                logger.warning(f"WebRTC VAD error: {e}, falling back to energy")
                return self._energy_based_detection(audio_frame)
        
        else:
            # Fallback to energy-based detection
            return self._energy_based_detection(audio_frame)
    
    def _energy_based_detection(self, audio_frame: np.ndarray) -> bool:
        """
        Simple energy-based speech detection.
        
        Args:
            audio_frame: Audio frame
        
        Returns:
            True if energy exceeds threshold
        """
        energy = np.mean(audio_frame ** 2)
        return energy > self.energy_threshold
    
    def filter_silence(
        self,
        audio: np.ndarray,
        padding_ms: int = 300,
    ) -> np.ndarray:
        """
        Remove silence from audio while keeping speech.
        
        Args:
            audio: Full audio array
            padding_ms: Milliseconds of padding to keep around speech
        
        Returns:
            Filtered audio with silence removed
        """
        padding_frames = int(padding_ms * self.sample_rate / 1000)
        
        # Split into frames
        num_frames = len(audio) // self.frame_size
        speech_frames = []
        
        for i in range(num_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frame = audio[start:end]
            
            if self.is_speech(frame):
                # Include speech frame with padding
                pad_start = max(0, start - padding_frames)
                pad_end = min(len(audio), end + padding_frames)
                speech_frames.append((pad_start, pad_end))
        
        if not speech_frames:
            # No speech detected
            return np.array([], dtype=audio.dtype)
        
        # Merge overlapping ranges
        merged = []
        current_start, current_end = speech_frames[0]
        
        for start, end in speech_frames[1:]:
            if start <= current_end:
                # Overlapping, extend current range
                current_end = max(current_end, end)
            else:
                # Non-overlapping, save current and start new
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        
        # Extract speech segments
        filtered = []
        for start, end in merged:
            filtered.append(audio[start:end])
        
        return np.concatenate(filtered) if filtered else np.array([], dtype=audio.dtype)
    
    def detect_end_of_turn(
        self,
        audio_frames: List[np.ndarray],
        silence_duration_ms: int = 700,
    ) -> bool:
        """
        Detect if user has finished speaking (end of turn).
        
        Args:
            audio_frames: Recent audio frames
            silence_duration_ms: Milliseconds of silence to consider end of turn
        
        Returns:
            True if end of turn detected
        """
        silence_frames_needed = int(
            silence_duration_ms / self.frame_duration_ms
        )
        
        if len(audio_frames) < silence_frames_needed:
            return False
        
        # Check last N frames for silence
        recent_frames = audio_frames[-silence_frames_needed:]
        
        for frame in recent_frames:
            if self.is_speech(frame):
                return False  # Still speaking
        
        return True  # Silence detected


def create_vad(
    sample_rate: int = 16000,
    aggressiveness: int = 2,
) -> VADDetector:
    """
    Convenience function to create VAD detector.
    
    Args:
        sample_rate: Audio sample rate
        aggressiveness: VAD sensitivity (0-3)
    
    Returns:
        VADDetector instance
    
    Example:
        >>> vad = create_vad(sample_rate=16000, aggressiveness=2)
        >>> is_speech = vad.is_speech(audio_frame)
    """
    return VADDetector(
        sample_rate=sample_rate,
        aggressiveness=aggressiveness,
    )

