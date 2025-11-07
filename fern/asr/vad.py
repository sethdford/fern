"""Voice Activity Detection using Silero VAD."""

import logging
from typing import List, Tuple, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Voice Activity Detection using Silero VAD model.
    
    Reference:
    Silero Team. (2024). Silero VAD: pre-trained enterprise-grade 
    Voice Activity Detector (VAD).
    https://github.com/snakers4/silero-vad
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,
    ):
        """
        Initialize Silero VAD.
        
        Args:
            threshold: Speech probability threshold (0-1)
            sampling_rate: Audio sampling rate in Hz
            min_speech_duration_ms: Minimum speech duration in milliseconds
            min_silence_duration_ms: Minimum silence duration in milliseconds
            window_size_samples: Window size for VAD processing
        """
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        
        # Load Silero VAD model
        try:
            self.model, self.utils = self._load_model()
            self.model.eval()
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise
    
    def _load_model(self):
        """Load Silero VAD model from torch hub."""
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        return model, utils
    
    def __call__(
        self,
        audio: np.ndarray,
        return_timestamps: bool = False
    ) -> Tuple[bool, Optional[List[Tuple[float, float]]]]:
        """
        Detect voice activity in audio.
        
        Args:
            audio: Audio samples as numpy array
            return_timestamps: Whether to return speech timestamps
            
        Returns:
            Tuple of (has_speech, timestamps)
            - has_speech: Boolean indicating if speech was detected
            - timestamps: List of (start, end) tuples in seconds (if return_timestamps=True)
        """
        if len(audio) == 0:
            return False, [] if return_timestamps else None
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Get speech timestamps
        speech_timestamps = self.utils[0](
            audio_tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            window_size_samples=self.window_size_samples,
            return_seconds=True
        )
        
        has_speech = len(speech_timestamps) > 0
        
        if return_timestamps:
            timestamps = [
                (segment['start'], segment['end']) 
                for segment in speech_timestamps
            ]
            return has_speech, timestamps
        
        return has_speech, None
    
    def detect_speech_segments(
        self,
        audio: np.ndarray,
        silence_duration_for_end: float = 1.5
    ) -> List[Tuple[float, float]]:
        """
        Detect speech segments with custom silence duration.
        
        Args:
            audio: Audio samples as numpy array
            silence_duration_for_end: Duration of silence to mark end of speech (seconds)
            
        Returns:
            List of (start, end) tuples for speech segments in seconds
        """
        # Save original min_silence_duration
        original_silence_duration = self.min_silence_duration_ms
        
        # Convert silence duration to milliseconds
        self.min_silence_duration_ms = int(silence_duration_for_end * 1000)
        
        try:
            has_speech, timestamps = self(audio, return_timestamps=True)
            return timestamps if timestamps else []
        finally:
            # Restore original setting
            self.min_silence_duration_ms = original_silence_duration
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Simple check if audio contains speech.
        
        Args:
            audio: Audio samples as numpy array
            
        Returns:
            True if speech is detected, False otherwise
        """
        has_speech, _ = self(audio, return_timestamps=False)
        return has_speech
    
    def extract_speech_audio(
        self,
        audio: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Extract only speech segments from audio.
        
        Args:
            audio: Audio samples as numpy array
            
        Returns:
            Tuple of (speech_audio, timestamps)
            - speech_audio: Concatenated speech segments
            - timestamps: Original timestamps of segments
        """
        has_speech, timestamps = self(audio, return_timestamps=True)
        
        if not has_speech or not timestamps:
            return np.array([]), []
        
        # Extract speech segments
        speech_segments = []
        for start, end in timestamps:
            start_sample = int(start * self.sampling_rate)
            end_sample = int(end * self.sampling_rate)
            speech_segments.append(audio[start_sample:end_sample])
        
        # Concatenate all speech segments
        speech_audio = np.concatenate(speech_segments) if speech_segments else np.array([])
        
        return speech_audio, timestamps
    
    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """
        Get speech probability for an audio chunk.
        
        Args:
            audio_chunk: Audio chunk as numpy array
            
        Returns:
            Speech probability (0-1)
        """
        if len(audio_chunk) == 0:
            return 0.0
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_chunk).float()
        
        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sampling_rate).item()
        
        return speech_prob

