"""Text-to-Speech (TTS) module with CSM-1B."""

from fern.tts.csm_tts import CSMTTS
from fern.tts.rvq_optimizer import RVQOptimizer

__all__ = ["CSMTTS", "RVQOptimizer"]

