"""End-to-end voice-to-voice pipeline integrating ASR, LLM, and TTS."""

import logging
import time
from typing import Optional, Dict, Any, Iterator, Tuple

import numpy as np
import soundfile as sf

from fern.config import FERNConfig, ASRConfig, LLMConfig, TTSConfig, VADConfig
from fern.asr import WhisperASR, SileroVAD
from fern.llm import DialogueManager
from fern.tts import CSMTTS
from fern.metrics import MetricsTracker, PerformanceMetrics
from fern.pipeline.streaming import StreamingPipeline, AudioStreamMerger, LatencyMonitor
from fern.utils import setup_logging, detect_device

logger = logging.getLogger(__name__)


class VoiceToVoicePipeline:
    """
    End-to-end voice-to-voice pipeline for i-LAVA.
    
    Architecture:
    Input Audio → VAD → ASR (Whisper) → LLM (GPT-4o-mini) → TTS (CSM-1B) → Output Audio
                                             ↑                      ↓
                                       Context Storage ← Audio Context
    
    The pipeline processes voice input through:
    1. Voice Activity Detection (VAD) to detect speech segments
    2. Automatic Speech Recognition (ASR) to transcribe speech
    3. Large Language Model (LLM) to generate contextual responses
    4. Text-to-Speech (TTS) to synthesize voice output
    
    Key features:
    - Low latency: First chunk as low as 640.9ms on GPU
    - Context-aware: Maintains conversation history and audio context
    - Streaming: Seamless audio output with chunk-based generation
    - Metrics: Comprehensive performance tracking
    """
    
    def __init__(
        self,
        config: Optional[FERNConfig] = None,
        device: Optional[str] = None,
        rvq_iterations: int = 16,
        enable_streaming: bool = True,
        enable_metrics: bool = True,
        log_level: str = "INFO",
    ):
        """
        Initialize voice-to-voice pipeline.
        
        Args:
            config: FERNConfig object (creates default if None)
            device: Override device (cuda, cpu, mps)
            rvq_iterations: Override RVQ iterations (16, 20, 24, 32)
            enable_streaming: Enable streaming output
            enable_metrics: Enable performance metrics
            log_level: Logging level
        """
        # Setup logging
        setup_logging(level=log_level)
        
        # Create or update config
        if config is None:
            config_dict = {}
            if device:
                config_dict["device"] = device
            if rvq_iterations:
                config_dict["rvq_iterations"] = rvq_iterations
            config_dict["enable_streaming"] = enable_streaming
            config_dict["enable_metrics"] = enable_metrics
            config_dict["log_level"] = log_level
            
            config = FERNConfig(**config_dict)
        
        self.config = config
        self.enable_metrics = enable_metrics
        
        logger.info("=" * 60)
        logger.info("Initializing i-LAVA Voice-to-Voice Pipeline")
        logger.info("=" * 60)
        
        # Detect device if not specified
        if device is None:
            detected_device, device_name = detect_device()
            logger.info(f"Auto-detected device: {device_name}")
        
        # Initialize components
        self._initialize_components()
        
        # Initialize streaming and context management
        self.streaming_pipeline = StreamingPipeline(
            chunk_size=config.streaming_chunk_size,
            prefetch_chunks=2,
        )
        self.audio_merger = AudioStreamMerger(
            max_context_duration_ms=5000.0,
            sample_rate=config.tts_sample_rate,
        )
        
        # Initialize metrics tracker
        if self.enable_metrics:
            self.metrics_tracker = MetricsTracker(sample_rate=config.tts_sample_rate)
        
        logger.info("=" * 60)
        logger.info("i-LAVA Pipeline initialized successfully")
        logger.info("=" * 60)
    
    def _initialize_components(self):
        """Initialize ASR, VAD, LLM, and TTS components."""
        logger.info("Initializing components...")
        
        # Initialize VAD
        logger.info("Loading Silero VAD...")
        vad_config = VADConfig.from_ilava_config(self.config)
        self.vad = SileroVAD(
            threshold=vad_config.threshold,
            sampling_rate=16000,  # VAD uses 16kHz
            min_speech_duration_ms=int(vad_config.min_speech_duration * 1000),
        )
        logger.info("✓ VAD initialized")
        
        # Initialize ASR
        logger.info("Loading Whisper ASR...")
        asr_config = ASRConfig.from_ilava_config(self.config)
        self.asr = WhisperASR(
            model_size=asr_config.model,
            device=asr_config.device,
            compute_type=asr_config.compute_type,
            chunk_length=asr_config.chunk_length,
        )
        logger.info("✓ ASR initialized")
        
        # Initialize LLM
        logger.info("Loading LLM dialogue manager...")
        llm_config = LLMConfig.from_ilava_config(self.config)
        self.llm = DialogueManager(
            api_key=llm_config.api_key,
            model=llm_config.model,
            system_prompt=llm_config.system_prompt,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            max_context_turns=self.config.max_context_length,
        )
        logger.info("✓ LLM initialized")
        
        # Initialize TTS
        logger.info("Loading CSM-1B TTS...")
        tts_config = TTSConfig.from_ilava_config(self.config)
        self.tts = CSMTTS(
            device=tts_config.device,
            rvq_iterations=tts_config.rvq_iterations,
            rvq_padding_method=tts_config.rvq_padding_method.value,
            mimi_codebooks=tts_config.mimi_codebooks,
            sample_rate=tts_config.sample_rate,
            enable_torch_compile=tts_config.enable_torch_compile,
            enable_cold_start=tts_config.enable_cold_start,
            cold_start_generations=tts_config.cold_start_generations,
            use_real_csm=tts_config.use_real_csm,
        )
        logger.info("✓ TTS initialized")
    
    def process_audio(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        return_metrics: bool = True,
    ) -> Tuple[np.ndarray, Optional[PerformanceMetrics]]:
        """
        Process audio file through complete pipeline.
        
        Orchestrates the full V-2-V pipeline following Clean Architecture:
        Load → VAD → ASR → LLM → TTS → Save
        
        Args:
            audio_path: Path to input audio file
            output_path: Optional path to save output audio
            return_metrics: Whether to return performance metrics
            
        Returns:
            Tuple of (output_audio, metrics)
        """
        logger.info(f"Processing audio file: {audio_path}")
        
        # Initialize metrics tracking
        self._initialize_metrics(return_metrics)
        
        # Load and prepare audio
        audio = self._load_audio_file(audio_path)
        if audio is None:
            return np.array([]), None
        
        # Detect speech
        if not self._detect_speech_segments(audio):
            return np.array([]), None
        
        # Transcribe speech to text
        transcribed_text = self._transcribe_audio(audio, return_metrics)
        
        # Generate LLM response
        response_text = self._generate_llm_response(transcribed_text, return_metrics)
        
        # Synthesize speech
        output_audio = self._synthesize_tts(response_text, return_metrics)
        
        # Save output if requested
        self._save_output_audio(output_audio, output_path)
        
        # Finalize and return metrics
        metrics = self._finalize_metrics(return_metrics)
        
        logger.info("Processing complete")
        return output_audio, metrics
    
    def _initialize_metrics(self, return_metrics: bool) -> None:
        """Initialize metrics tracking for pipeline run."""
        if self.enable_metrics and return_metrics:
            self.metrics_tracker.reset()
            self.metrics_tracker.start_pipeline()
    
    def _load_audio_file(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess audio file.
        
        Converts to mono and resamples to 16kHz for ASR/VAD.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio array or None if loading fails
        """
        audio, sample_rate = sf.read(audio_path)
        logger.info(f"Loaded audio: {len(audio)/sample_rate:.2f}s @ {sample_rate}Hz")
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz for ASR/VAD
        if sample_rate != 16000:
            from scipy import signal as scipy_signal
            audio = scipy_signal.resample(
                audio,
                int(len(audio) * 16000 / sample_rate)
            )
        
        if self.enable_metrics:
            self.metrics_tracker.set_input_audio(audio)
        
        return audio
    
    def _detect_speech_segments(self, audio: np.ndarray) -> bool:
        """
        Detect speech in audio using VAD.
        
        Args:
            audio: Audio array
            
        Returns:
            True if speech detected, False otherwise
        """
        logger.info("Step 1: Voice Activity Detection")
        has_speech, speech_segments = self.vad(audio, return_timestamps=True)
        
        if not has_speech:
            logger.warning("No speech detected in audio")
            return False
        
        logger.info(f"Detected {len(speech_segments)} speech segments")
        return True
    
    def _transcribe_audio(self, audio: np.ndarray, return_metrics: bool) -> str:
        """
        Transcribe audio to text using ASR.
        
        Args:
            audio: Audio array
            return_metrics: Whether metrics are being tracked
            
        Returns:
            Transcribed text
        """
        logger.info("Step 2: Automatic Speech Recognition")
        
        if self.enable_metrics and return_metrics:
            self.metrics_tracker.start_asr()
        
        transcription = self.asr.transcribe(audio, sample_rate=16000)
        transcribed_text = transcription["text"]
        
        if self.enable_metrics and return_metrics:
            self.metrics_tracker.end_asr()
        
        logger.info(f"Transcribed: '{transcribed_text}'")
        return transcribed_text
    
    def _generate_llm_response(self, text: str, return_metrics: bool) -> str:
        """
        Generate LLM response to transcribed text.
        
        Args:
            text: Input text
            return_metrics: Whether metrics are being tracked
            
        Returns:
            LLM response text
        """
        logger.info("Step 3: LLM Response Generation")
        
        if self.enable_metrics and return_metrics:
            self.metrics_tracker.start_llm()
        
        response_text = self.llm.generate_response(text)
        
        if self.enable_metrics and return_metrics:
            self.metrics_tracker.end_llm()
        
        logger.info(f"LLM Response: '{response_text}'")
        return response_text
    
    def _synthesize_tts(self, text: str, return_metrics: bool) -> np.ndarray:
        """
        Synthesize speech from text using TTS.
        
        Args:
            text: Text to synthesize
            return_metrics: Whether metrics are being tracked
            
        Returns:
            Synthesized audio array
        """
        logger.info("Step 4: Text-to-Speech Synthesis")
        
        if self.enable_metrics and return_metrics:
            self.metrics_tracker.start_tts()
        
        # Get audio context if enabled
        audio_context = self.audio_merger.get_context() if self.config.include_audio_context else None
        
        # Get previous text for context
        history_messages = self.llm.history.get_messages()
        context_text = history_messages[-2]["content"] if len(history_messages) >= 2 else None
        
        # Synthesize
        output_audio = self.tts.synthesize(
            text=text,
            context_audio=audio_context,
            context_text=context_text,
        )
        
        if self.enable_metrics and return_metrics:
            self.metrics_tracker.end_tts()
            self.metrics_tracker.set_output_audio(output_audio)
        
        # Add to context for next turn
        if self.config.include_audio_context:
            self.audio_merger.add_audio(output_audio)
        
        return output_audio
    
    def _save_output_audio(self, audio: np.ndarray, output_path: Optional[str]) -> None:
        """
        Save output audio to file if path provided.
        
        Args:
            audio: Audio array to save
            output_path: Path to save audio, or None to skip
        """
        if output_path:
            sf.write(output_path, audio, self.config.tts_sample_rate)
            logger.info(f"Output audio saved to: {output_path}")
    
    def _finalize_metrics(self, return_metrics: bool) -> Optional[PerformanceMetrics]:
        """
        Calculate and return final metrics.
        
        Args:
            return_metrics: Whether to calculate metrics
            
        Returns:
            PerformanceMetrics object or None
        """
        metrics = None
        if self.enable_metrics and return_metrics:
            metrics = self.metrics_tracker.calculate_metrics()
            self.metrics_tracker.log_metrics(metrics)
        return metrics
    
    def stream_conversation(
        self,
        audio_path: str,
    ) -> Iterator[np.ndarray]:
        """
        Process audio with streaming output.
        
        Args:
            audio_path: Path to input audio file
            
        Yields:
            Audio chunks as they are generated
        """
        logger.info(f"Starting streaming conversation: {audio_path}")
        
        # Process input through ASR and LLM
        # Load audio
        audio, sample_rate = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz
        if sample_rate != 16000:
            from scipy import signal as scipy_signal
            audio = scipy_signal.resample(
                audio,
                int(len(audio) * 16000 / sample_rate)
            )
        
        # VAD
        has_speech, _ = self.vad(audio, return_timestamps=True)
        if not has_speech:
            logger.warning("No speech detected")
            return
        
        # ASR
        transcription = self.asr.transcribe(audio, sample_rate=16000)
        transcribed_text = transcription["text"]
        logger.info(f"Transcribed: '{transcribed_text}'")
        
        # LLM
        response_text = self.llm.generate_response(transcribed_text)
        logger.info(f"Response: '{response_text}'")
        
        # TTS Streaming
        audio_context = self.audio_merger.get_context() if self.config.include_audio_context else None
        history_messages = self.llm.history.get_messages()
        context_text = history_messages[-2]["content"] if len(history_messages) >= 2 else None
        
        # Create streaming generator
        def tts_generator():
            return self.tts.synthesize_streaming(
                text=response_text,
                context_audio=audio_context,
                context_text=context_text,
                chunk_size=self.config.streaming_chunk_size,
            )
        
        # Stream with pipeline
        for chunk in self.streaming_pipeline.stream_audio(tts_generator):
            yield chunk
        
        logger.info("Streaming complete")
    
    def process_text(
        self,
        text: str,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Process text input directly (skip ASR).
        
        Args:
            text: Input text
            output_path: Optional path to save output audio
            
        Returns:
            Generated audio
        """
        logger.info(f"Processing text: '{text}'")
        
        # LLM
        response_text = self.llm.generate_response(text)
        logger.info(f"Response: '{response_text}'")
        
        # TTS
        audio_context = self.audio_merger.get_context() if self.config.include_audio_context else None
        history_messages = self.llm.history.get_messages()
        context_text = history_messages[-2]["content"] if len(history_messages) >= 2 else None
        
        output_audio = self.tts.synthesize(
            text=response_text,
            context_audio=audio_context,
            context_text=context_text,
        )
        
        # Add to context
        if self.config.include_audio_context:
            self.audio_merger.add_audio(output_audio)
        
        # Save if requested
        if output_path:
            sf.write(output_path, output_audio, self.config.tts_sample_rate)
            logger.info(f"Saved to: {output_path}")
        
        return output_audio
    
    def clear_context(self):
        """Clear conversation and audio context."""
        self.llm.clear_history()
        self.audio_merger.clear()
        logger.info("Context cleared")
    
    def get_conversation_history(self) -> str:
        """Get conversation history summary."""
        return self.llm.get_conversation_summary()
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get pipeline configuration and status.
        
        Returns:
            Dictionary with pipeline info
        """
        return {
            "device": self.config.device.value,
            "asr_model": self.config.whisper_model,
            "llm_model": self.config.llm_model,
            "tts_model": self.config.tts_model,
            "rvq_iterations": self.config.rvq_iterations,
            "streaming_enabled": self.config.enable_streaming,
            "metrics_enabled": self.enable_metrics,
            "tts_info": self.tts.get_model_info(),
        }

