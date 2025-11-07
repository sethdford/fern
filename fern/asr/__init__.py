"""Automatic Speech Recognition (ASR) module."""

from fern.asr.whisper_asr import WhisperASR
from fern.asr.vad import SileroVAD

__all__ = ["WhisperASR", "SileroVAD"]

