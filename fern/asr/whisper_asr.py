"""Automatic Speech Recognition using OpenAI Whisper."""

import logging
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class WhisperASR:
    """
    Automatic Speech Recognition using OpenAI Whisper large-v3 (turbo).
    
    Reference:
    Radford, A., et al. (2022). Robust Speech Recognition via 
    Large-Scale Weak Supervision. arXiv:2212.04356
    """
    
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        chunk_length: int = 30,
        language: Optional[str] = None,
    ):
        """
        Initialize Whisper ASR.
        
        Args:
            model_size: Whisper model size (large-v3 for turbo)
            device: Compute device (cuda, cpu)
            compute_type: Compute precision (float16, int8, float32)
            chunk_length: Chunk length in seconds for processing
            language: Target language code (None for auto-detection)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.chunk_length = chunk_length
        self.language = language
        
        # Load model
        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            logger.info(
                f"Whisper {model_size} loaded on {device} "
                f"with {compute_type} precision"
            )
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False,
        vad_filter: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio samples as numpy array
            sample_rate: Audio sample rate in Hz
            initial_prompt: Optional prompt to guide transcription
            word_timestamps: Whether to include word-level timestamps
            vad_filter: Whether to use VAD filtering
            
        Returns:
            Dictionary containing:
            - text: Transcribed text
            - segments: List of segments with timestamps
            - language: Detected language
        """
        if len(audio) == 0:
            return {
                "text": "",
                "segments": [],
                "language": self.language or "en"
            }
        
        try:
            # Transcribe with chunked processing
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                initial_prompt=initial_prompt,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=True,
            )
            
            # Collect segments
            transcription_segments = []
            full_text = []
            
            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
                
                if word_timestamps:
                    segment_dict["words"] = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability,
                        }
                        for word in segment.words
                    ]
                
                transcription_segments.append(segment_dict)
                full_text.append(segment.text.strip())
            
            # Join text and clean up
            text = " ".join(full_text)
            text = self._clean_transcription(text)
            
            result = {
                "text": text,
                "segments": transcription_segments,
                "language": info.language,
                "language_probability": info.language_probability,
            }
            
            logger.debug(
                f"Transcribed {len(audio)/sample_rate:.2f}s audio: "
                f"'{text[:50]}...'"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def transcribe_streaming(
        self,
        audio_chunks: List[np.ndarray],
        sample_rate: int = 16000,
    ) -> str:
        """
        Transcribe audio chunks in streaming fashion.
        
        Args:
            audio_chunks: List of audio chunks as numpy arrays
            sample_rate: Audio sample rate in Hz
            
        Returns:
            Complete transcription text
        """
        # Concatenate all chunks
        audio = np.concatenate(audio_chunks)
        
        # Transcribe
        result = self.transcribe(audio, sample_rate=sample_rate)
        
        return result["text"]
    
    def _clean_transcription(self, text: str) -> str:
        """
        Clean up transcription text.
        
        Args:
            text: Raw transcription text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove common filler words/sounds (optional)
        # This can be customized based on requirements
        filler_patterns = [
            " um ", " uh ", " hmm ", " ah ",
            " er ", " like ", " you know "
        ]
        
        text_lower = text.lower()
        for pattern in filler_patterns:
            # Only remove if it's a standalone word
            if pattern in text_lower:
                # Use case-insensitive replacement
                import re
                text = re.sub(
                    pattern,
                    " ",
                    text,
                    flags=re.IGNORECASE
                )
        
        # Clean up spacing again
        text = " ".join(text.split())
        
        return text.strip()
    
    def detect_language(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Detect language of audio.
        
        Args:
            audio: Audio samples as numpy array
            
        Returns:
            Dictionary with detected language and probability
        """
        # Use small portion for language detection
        audio_sample = audio[:16000 * 10]  # First 10 seconds
        
        segments, info = self.model.transcribe(
            audio_sample,
            language=None,  # Auto-detect
            beam_size=1,
            best_of=1,
        )
        
        # Consume the iterator
        _ = list(segments)
        
        return {
            "language": info.language,
            "probability": info.language_probability,
        }
    
    def transcribe_with_context(
        self,
        audio: np.ndarray,
        previous_text: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """
        Transcribe audio with context from previous transcription.
        
        Args:
            audio: Audio samples as numpy array
            previous_text: Previous transcription for context
            sample_rate: Audio sample rate in Hz
            
        Returns:
            Transcription result dictionary
        """
        # Use previous text as prompt for context
        initial_prompt = previous_text[-224:] if previous_text else None
        
        return self.transcribe(
            audio,
            sample_rate=sample_rate,
            initial_prompt=initial_prompt,
        )

