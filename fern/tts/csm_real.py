"""Real CSM-1B implementation using csm-streaming."""

import logging
from typing import Optional, Iterator
import numpy as np
import torch

logger = logging.getLogger(__name__)

class RealCSMTTS:
    """
    Real CSM-1B TTS using csm-streaming repository.
    
    This replaces the placeholder implementation with actual CSM-1B model.
    
    Reference: https://github.com/davidbrowne17/csm-streaming
    """
    
    def __init__(
        self,
        device: str = "cuda",
        sample_rate: int = 24000,
        enable_flash_attention: bool = True,
    ):
        """Initialize real CSM-1B TTS."""
        self.device = device
        self.sample_rate = sample_rate
        
        logger.info("Loading real CSM-1B model from csm-streaming...")
        
        try:
            import sys
            import os
            
            # Add CSM directory to path
            csm_path = os.path.join(os.path.dirname(__file__), 'csm')
            if csm_path not in sys.path:
                sys.path.insert(0, csm_path)
            
            from generator import Segment
            from load_real import load_csm_1b_real
            
            self.generator, self.mimi = load_csm_1b_real(device)
            self.Segment = Segment
            
            logger.info("✓ CSM-1B generator loaded successfully!")
            logger.info(f"  Device: {device}")
            logger.info(f"  Sample rate: {sample_rate}Hz")
            
        except ImportError as e:
            logger.error(
                f"Failed to import csm-streaming: {e}\n"
                "Please install: pip install git+https://github.com/davidbrowne17/csm-streaming.git"
            )
            raise
    
    def synthesize(
        self,
        text: str,
        speaker: int = 0,
        context_audio: Optional[np.ndarray] = None,
        context_text: Optional[str] = None,
    ) -> np.ndarray:
        """Synthesize speech from text."""
        # Build context
        context = []
        if context_audio is not None and context_text is not None:
            context.append(self.Segment(
                text=context_text,
                speaker=speaker,
                audio=torch.from_numpy(context_audio).to(self.device)
            ))
        
        # Generate
        audio = self.generator.generate(
            text=text,
            speaker=speaker,
            context=context,
            stream=True  # Internal streaming optimization
        )
        
        return audio.cpu().numpy()
    
    def synthesize_streaming(
        self,
        text: str,
        speaker: int = 0,
        context_audio: Optional[np.ndarray] = None,
        context_text: Optional[str] = None,
        chunk_size: int = 512,
    ) -> Iterator[np.ndarray]:
        """Synthesize speech with streaming."""
        # Build context
        context = []
        if context_audio is not None and context_text is not None:
            context.append(self.Segment(
                text=context_text,
                speaker=speaker,
                audio=torch.from_numpy(context_audio).to(self.device)
            ))
        
        # Stream generation
        for audio_chunk in self.generator.generate_stream(
            text=text,
            speaker=speaker,
            context=context
        ):
            yield audio_chunk.cpu().numpy()
