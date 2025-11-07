"""
i-LAVA: Low Latency Voice-2-Voice Architecture for Agents

Implementation of the paper: https://arxiv.org/html/2509.20971v1
"""

from fern.config import FERNConfig
from fern.pipeline.voice_pipeline import VoiceToVoicePipeline

__version__ = "1.0.0"
__all__ = ["VoiceToVoicePipeline", "FERNConfig"]

