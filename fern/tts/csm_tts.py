"""Text-to-Speech using CSM-1B (Conversational Speech Model)."""

import logging
import time
from typing import Optional, Dict, Any, List, Iterator, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fern.tts.rvq_optimizer import RVQOptimizer
from fern.tts.csm_config import CSMConfig

logger = logging.getLogger(__name__)


class CSMTTS:
    """
    Text-to-Speech using CSM-1B (Conversational Speech Model).
    
    CSM-1B uses:
    - Llama 3.2 1B as backbone
    - Llama 3.2 100M as decoder
    - Residual Vector Quantization (RVQ) with Mimi tokenizer
    - Context-aware generation (text + audio)
    
    Reference:
    Sesame AI. (2025). CSM (Conversational Speech Model).
    
    Note: This is a wrapper implementation. The actual CSM-1B model
    would need to be obtained from Sesame AI or equivalent source.
    
    Example:
        >>> from fern.tts.csm_config import CSMConfig
        >>> config = CSMConfig(device='cpu', use_real_csm=False)
        >>> tts = CSMTTS(config=config)
        >>> audio = tts.synthesize('Hello world')
    """
    
    def __init__(
        self,
        config: Optional[CSMConfig] = None,
        # Legacy parameters for backward compatibility
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        rvq_iterations: Optional[int] = None,
        rvq_padding_method: Optional[str] = None,
        mimi_codebooks: Optional[int] = None,
        sample_rate: Optional[int] = None,
        enable_torch_compile: Optional[bool] = None,
        enable_cold_start: Optional[bool] = None,
        cold_start_generations: Optional[int] = None,
        use_real_csm: Optional[bool] = None,
    ):
        """
        Initialize CSM-1B TTS.
        
        Args:
            config: CSMConfig object (preferred). If provided, other params ignored.
            model_name: [LEGACY] Model identifier
            device: [LEGACY] Compute device (cuda, cpu, mps)
            rvq_iterations: [LEGACY] Number of RVQ iterations (16, 20, 24, 32)
            rvq_padding_method: [LEGACY] RVQ padding method
            mimi_codebooks: [LEGACY] Number of Mimi codebooks
            sample_rate: [LEGACY] Audio sample rate in Hz
            enable_torch_compile: [LEGACY] Enable torch.compile for optimization
            enable_cold_start: [LEGACY] Perform cold start generations
            cold_start_generations: [LEGACY] Number of cold start generations
            use_real_csm: [LEGACY] Use real CSM-1B (True) or placeholder (False)
        
        Note:
            New code should use config parameter. Legacy parameters are
            supported for backward compatibility.
        
        Example:
            >>> # New style (preferred)
            >>> config = CSMConfig(device='cpu', use_real_csm=False)
            >>> tts = CSMTTS(config=config)
            
            >>> # Old style (still supported)
            >>> tts = CSMTTS(device='cpu', use_real_csm=False)
        """
        # Handle backward compatibility
        if config is None:
            # Create config from individual parameters
            config_kwargs = {}
            if model_name is not None:
                config_kwargs['model_name'] = model_name
            if device is not None:
                config_kwargs['device'] = device
            if rvq_iterations is not None:
                config_kwargs['rvq_iterations'] = rvq_iterations
            if rvq_padding_method is not None:
                config_kwargs['rvq_padding_method'] = rvq_padding_method
            if mimi_codebooks is not None:
                config_kwargs['mimi_codebooks'] = mimi_codebooks
            if sample_rate is not None:
                config_kwargs['sample_rate'] = sample_rate
            if enable_torch_compile is not None:
                config_kwargs['enable_torch_compile'] = enable_torch_compile
            if enable_cold_start is not None:
                config_kwargs['enable_cold_start'] = enable_cold_start
            if cold_start_generations is not None:
                config_kwargs['cold_start_generations'] = cold_start_generations
            if use_real_csm is not None:
                config_kwargs['use_real_csm'] = use_real_csm
            
            config = CSMConfig(**config_kwargs)
        
        # Store config
        self.config = config
        
        # Extract commonly used values for convenience
        self.model_name = config.model_name
        self.device = config.device
        self.sample_rate = config.sample_rate
        self.enable_torch_compile = config.enable_torch_compile
        self.use_real_csm = config.use_real_csm
        self.real_generator = None
        
        # Initialize RVQ optimizer (for placeholder mode)
        self.rvq_optimizer = RVQOptimizer(
            num_iterations=config.rvq_iterations,
            num_codebooks=config.mimi_codebooks,
            padding_method=config.rvq_padding_method,
        )
        
        logger.info(f"Initializing CSM-1B TTS on {config.device}")
        logger.info(f"Mode: {'Real CSM-1B' if config.use_real_csm else 'Placeholder'}")
        
        if not config.use_real_csm:
            logger.info(self.rvq_optimizer.get_configuration_summary())
        
        try:
            # Load models
            self._load_models()
            
            # Perform cold start if enabled and not using real CSM
            if config.enable_cold_start and not config.use_real_csm:
                self._perform_cold_start(config.cold_start_generations)
            
            logger.info("CSM-1B TTS initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize CSM-1B TTS: {e}")
            raise
    
    def _load_models(self):
        """Load CSM-1B models (backbone, decoder, Mimi tokenizer)."""
        try:
            if self.use_real_csm:
                # Load real CSM-1B from csm-streaming
                logger.info("Loading real CSM-1B model...")
                try:
                    from fern.tts.csm_real import RealCSMTTS
                    
                    self.real_generator = RealCSMTTS(
                        device=self.device,
                        sample_rate=self.sample_rate,
                    )
                    logger.info("✓ Real CSM-1B loaded successfully!")
                    return
                    
                except ImportError as e:
                    logger.error(
                        f"Failed to import real CSM: {e}\n"
                        "Falling back to placeholder mode..."
                    )
                    self.use_real_csm = False
                    # Continue with placeholder loading below
            
            # Placeholder implementation
            logger.info("Loading CSM-1B backbone (Llama 3.2 1B)...")
            # self.backbone = load_csm_backbone(self.model_name, device=self.device)
            
            logger.info("Loading CSM-1B decoder (Llama 3.2 100M)...")
            # self.decoder = load_csm_decoder(self.model_name, device=self.device)
            
            logger.info("Loading Mimi tokenizer...")
            # self.mimi_tokenizer = load_mimi_tokenizer(
            #     num_codebooks=self.rvq_optimizer.num_codebooks
            # )
            
            # Apply torch.compile if enabled
            if self.enable_torch_compile and hasattr(torch, 'compile'):
                logger.info("Applying torch.compile optimization...")
                # self.backbone = torch.compile(self.backbone)
                # self.decoder = torch.compile(self.decoder)
            
            # For demonstration, we'll use placeholder models
            self.backbone = None
            self.decoder = None
            self.mimi_tokenizer = None
            
            logger.warning(
                "Using placeholder models. Actual CSM-1B models need to be loaded. "
                "This implementation provides the architecture framework."
            )
        
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def _perform_cold_start(self, num_generations: int = 2):
        """
        Perform cold start generations for optimal performance.
        
        CSM-1B takes at least 2 generations before achieving desired performance.
        
        Args:
            num_generations: Number of cold start generations
        """
        logger.info(f"Performing {num_generations} cold start generations...")
        
        dummy_text = "Hello, this is a warm-up generation."
        
        for i in range(num_generations):
            try:
                # Generate dummy audio
                _ = self._generate_audio(
                    text=dummy_text,
                    context_audio=None,
                    context_text=None,
                )
                logger.debug(f"Cold start generation {i+1}/{num_generations} complete")
            except Exception as e:
                logger.warning(f"Cold start generation {i+1} failed: {e}")
        
        logger.info("Cold start complete")
    
    def _generate_audio(
        self,
        text: str,
        context_audio: Optional[np.ndarray] = None,
        context_text: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate audio from text using CSM-1B.
        
        Args:
            text: Text to synthesize
            context_audio: Optional previous audio for context
            context_text: Optional previous text for context
            
        Returns:
            Generated audio as numpy array
        """
        # This is a placeholder implementation
        # Actual CSM-1B would perform:
        # 1. Text encoding with backbone
        # 2. Context integration (audio + text)
        # 3. RVQ decoding with optimized iterations
        # 4. Mimi detokenization to audio
        
        logger.debug(f"Generating audio for text: '{text[:50]}...'")
        
        # Placeholder: Generate synthetic audio
        # In reality, this would be the CSM-1B forward pass
        duration_seconds = len(text.split()) * 0.3  # Rough estimate
        num_samples = int(duration_seconds * self.sample_rate)
        
        # Generate placeholder audio (sine wave for demonstration)
        t = np.linspace(0, duration_seconds, num_samples)
        audio = np.sin(2 * np.pi * 440 * t) * 0.3  # 440 Hz sine wave
        
        logger.debug(f"Generated {len(audio)} samples ({duration_seconds:.2f}s)")
        
        return audio.astype(np.float32)
    
    def synthesize(
        self,
        text: str,
        context_audio: Optional[np.ndarray] = None,
        context_text: Optional[str] = None,
        streaming: bool = False,
    ) -> np.ndarray:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            context_audio: Optional previous audio for context-aware generation
            context_text: Optional previous text for context
            streaming: Whether to use streaming generation (not used in one-shot)
            
        Returns:
            Generated audio as numpy array
        """
        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return np.array([])
        
        start_time = time.time()
        
        try:
            # Use real CSM if available
            if self.use_real_csm and self.real_generator is not None:
                audio = self.real_generator.synthesize(
                    text=text,
                    context_audio=context_audio,
                    context_text=context_text,
                )
            else:
                # Use placeholder
                audio = self._generate_audio(
                    text=text,
                    context_audio=context_audio,
                    context_text=context_text,
                )
            
            generation_time = (time.time() - start_time) * 1000
            audio_duration = (len(audio) / self.sample_rate) * 1000
            rtf = generation_time / audio_duration if audio_duration > 0 else 0
            
            logger.info(
                f"Generated {audio_duration:.1f}ms audio in {generation_time:.1f}ms "
                f"(RTF: {rtf:.3f}x)"
            )
            
            return audio
        
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            raise
    
    def synthesize_streaming(
        self,
        text: str,
        context_audio: Optional[np.ndarray] = None,
        context_text: Optional[str] = None,
        chunk_size: int = 512,
    ) -> Iterator[np.ndarray]:
        """
        Synthesize speech with streaming output.
        
        Streaming improves perceived latency by yielding audio chunks
        as they are generated, rather than waiting for complete generation.
        
        Args:
            text: Text to synthesize
            context_audio: Optional previous audio for context
            context_text: Optional previous text for context
            chunk_size: Size of audio chunks to yield
            
        Yields:
            Audio chunks as numpy arrays
        """
        if not text.strip():
            logger.warning("Empty text provided for streaming synthesis")
            return
        
        logger.debug(f"Starting streaming synthesis for: '{text[:50]}...'")
        
        start_time = time.time()
        first_chunk = True
        
        try:
            # Use real CSM streaming if available
            if self.use_real_csm and self.real_generator is not None:
                for chunk in self.real_generator.synthesize_streaming(
                    text=text,
                    context_audio=context_audio,
                    context_text=context_text,
                    chunk_size=chunk_size,
                ):
                    if first_chunk:
                        first_chunk_time = (time.time() - start_time) * 1000
                        logger.info(f"First chunk latency: {first_chunk_time:.1f}ms")
                        first_chunk = False
                    yield chunk
                return
            
            # Placeholder: Generate complete audio then chunk it
            audio = self._generate_audio(
                text=text,
                context_audio=context_audio,
                context_text=context_text,
            )
            
            # Yield in chunks
            num_chunks = (len(audio) + chunk_size - 1) // chunk_size
            
            for i in range(num_chunks):
                chunk_start = i * chunk_size
                chunk_end = min((i + 1) * chunk_size, len(audio))
                chunk = audio[chunk_start:chunk_end]
                
                if first_chunk:
                    first_chunk_time = (time.time() - start_time) * 1000
                    logger.info(f"First chunk latency: {first_chunk_time:.1f}ms")
                    first_chunk = False
                
                yield chunk
            
            total_time = (time.time() - start_time) * 1000
            audio_duration = (len(audio) / self.sample_rate) * 1000
            
            logger.info(
                f"Streaming complete: {num_chunks} chunks, "
                f"{audio_duration:.1f}ms audio in {total_time:.1f}ms"
            )
        
        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            raise
    
    def synthesize_batch(
        self,
        texts: List[str],
        context_audios: Optional[List[np.ndarray]] = None,
        context_texts: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """
        Synthesize multiple texts in batch.
        
        Args:
            texts: List of texts to synthesize
            context_audios: Optional list of context audios
            context_texts: Optional list of context texts
            
        Returns:
            List of generated audio arrays
        """
        if context_audios is None:
            context_audios = [None] * len(texts)
        if context_texts is None:
            context_texts = [None] * len(texts)
        
        results = []
        for text, ctx_audio, ctx_text in zip(texts, context_audios, context_texts):
            audio = self.synthesize(
                text=text,
                context_audio=ctx_audio,
                context_text=ctx_text,
            )
            results.append(audio)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and configuration.
        
        Returns:
            Dictionary with model info
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "rvq_iterations": self.rvq_optimizer.num_iterations,
            "rvq_codebooks": self.rvq_optimizer.num_codebooks,
            "rvq_padding": self.rvq_optimizer.padding_method,
            "torch_compile_enabled": self.enable_torch_compile,
            "estimated_latency_reduction": f"{self.rvq_optimizer.estimate_latency_reduction()*100:.0f}%",
            "estimated_quality": self.rvq_optimizer.estimate_quality_impact(),
        }

