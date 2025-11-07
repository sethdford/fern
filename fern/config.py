"""Configuration management for i-LAVA voice-to-voice pipeline."""

import os
from enum import Enum
from typing import Optional, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class DeviceType(str, Enum):
    """Supported compute devices."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders


class RVQPaddingMethod(str, Enum):
    """RVQ padding methods for reduced iterations."""
    NONE = "none"
    MEAN = "mean"
    CONCAT = "concat"


class FERNConfig(BaseModel):
    """Main configuration for i-LAVA pipeline."""
    
    # Device Configuration
    device: DeviceType = Field(
        default=DeviceType.CUDA,
        description="Compute device (cpu, cuda, mps)"
    )
    
    # ASR Configuration
    whisper_model: str = Field(
        default="large-v3",
        description="Whisper model version (large-v3 for turbo)"
    )
    whisper_compute_type: str = Field(
        default="float16",
        description="Compute type for Whisper (float16, int8, float32)"
    )
    whisper_chunk_length: int = Field(
        default=30,
        description="Chunk length in seconds for ASR processing"
    )
    
    # VAD Configuration
    vad_threshold: float = Field(
        default=0.5,
        description="Voice Activity Detection threshold (0-1)"
    )
    vad_silence_duration: float = Field(
        default=1.5,
        description="Silence duration in seconds to consider end of speech"
    )
    vad_min_speech_duration: float = Field(
        default=0.25,
        description="Minimum speech duration in seconds"
    )
    
    # LLM Configuration
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for dialogue management"
    )
    llm_temperature: float = Field(
        default=0.7,
        description="LLM sampling temperature"
    )
    llm_max_tokens: int = Field(
        default=500,
        description="Maximum tokens for LLM response"
    )
    llm_system_prompt: str = Field(
        default="You are a helpful voice assistant. Provide concise, natural responses suitable for voice conversation.",
        description="System prompt for LLM"
    )
    
    # TTS Configuration
    tts_model: str = Field(
        default="csm-1b",
        description="TTS model (csm-1b)"
    )
    rvq_iterations: Literal[16, 20, 24, 32] = Field(
        default=16,
        description="Number of RVQ iterations (16, 20, 24, or 32)"
    )
    rvq_padding_method: RVQPaddingMethod = Field(
        default=RVQPaddingMethod.NONE,
        description="Padding method when using fewer RVQ iterations"
    )
    mimi_codebooks: int = Field(
        default=32,
        description="Number of Mimi codebooks (32 for full, or match rvq_iterations)"
    )
    tts_sample_rate: int = Field(
        default=24000,
        description="TTS output sample rate in Hz"
    )
    tts_use_real_csm: bool = Field(
        default=True,
        description="Use real CSM-1B model (True) or placeholder for testing (False)"
    )
    
    # Streaming Configuration
    enable_streaming: bool = Field(
        default=True,
        description="Enable streaming audio output"
    )
    streaming_chunk_size: int = Field(
        default=512,
        description="Audio chunk size for streaming in samples"
    )
    
    # Performance Optimization
    enable_torch_compile: bool = Field(
        default=True,
        description="Enable torch.compile for kernel optimization"
    )
    enable_cold_start: bool = Field(
        default=True,
        description="Perform cold start generations for optimal performance"
    )
    cold_start_generations: int = Field(
        default=2,
        description="Number of cold start generations"
    )
    
    # Context Management
    max_context_length: int = Field(
        default=10,
        description="Maximum number of conversation turns to keep in context"
    )
    include_audio_context: bool = Field(
        default=True,
        description="Include audio context in TTS generation"
    )
    
    # Metrics and Logging
    enable_metrics: bool = Field(
        default=True,
        description="Enable performance metrics tracking"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    # OpenAI API
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (reads from OPENAI_API_KEY env var if not set)"
    )
    
    @field_validator("device", mode="before")
    @classmethod
    def validate_device(cls, v: str) -> DeviceType:
        """Validate device availability with graceful fallback."""
        import torch
        import logging
        
        logger = logging.getLogger(__name__)
        
        if isinstance(v, DeviceType):
            v_str = v.value
        else:
            v_str = v.lower()
        
        # Check availability and fall back if needed
        if v_str == "cuda":
            if not torch.cuda.is_available():
                logger.warning(
                    "CUDA requested but not available. Falling back to CPU. "
                    "Set device='cpu' explicitly to suppress this warning."
                )
                return DeviceType.CPU
        elif v_str == "mps":
            if not torch.backends.mps.is_available():
                logger.warning(
                    "MPS requested but not available. Falling back to CPU. "
                    "Set device='cpu' explicitly to suppress this warning."
                )
                return DeviceType.CPU
        
        return DeviceType(v_str)
    
    @model_validator(mode='after')
    def validate_and_adjust(self) -> 'FERNConfig':
        """
        Model-level validation after all fields are set.
        
        This validator:
        1. Auto-adjusts mimi_codebooks to match rvq_iterations when no padding
        2. Ensures API key is set from environment if not provided
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # 1. Auto-adjust mimi_codebooks
        if self.rvq_padding_method == RVQPaddingMethod.NONE:
            if self.mimi_codebooks != self.rvq_iterations:
                logger.debug(
                    f"Auto-adjusting mimi_codebooks from {self.mimi_codebooks} "
                    f"to {self.rvq_iterations} (no padding mode)"
                )
                self.mimi_codebooks = self.rvq_iterations
        
        # 2. Ensure API key is set
        if not self.openai_api_key or self.openai_api_key == "":
            key = os.getenv("OPENAI_API_KEY")
            if key:
                self.openai_api_key = key
            else:
                raise ValueError(
                    "OpenAI API key must be provided via openai_api_key parameter "
                    "or OPENAI_API_KEY environment variable"
                )
        
        return self
    
    class Config:
        """Pydantic config."""
        use_enum_values = False
        validate_assignment = True


class ASRConfig(BaseModel):
    """ASR-specific configuration."""
    model: str
    compute_type: str
    chunk_length: int
    device: str
    
    @classmethod
    def from_ilava_config(cls, config: FERNConfig) -> "ASRConfig":
        """Create ASR config from main config."""
        return cls(
            model=config.whisper_model,
            compute_type=config.whisper_compute_type,
            chunk_length=config.whisper_chunk_length,
            device=config.device.value
        )


class LLMConfig(BaseModel):
    """LLM-specific configuration."""
    model: str
    temperature: float
    max_tokens: int
    system_prompt: str
    api_key: str
    
    @classmethod
    def from_ilava_config(cls, config: FERNConfig) -> "LLMConfig":
        """Create LLM config from main config."""
        return cls(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            system_prompt=config.llm_system_prompt,
            api_key=config.openai_api_key or ""
        )


class TTSConfig(BaseModel):
    """TTS-specific configuration."""
    model: str
    rvq_iterations: int
    rvq_padding_method: RVQPaddingMethod
    mimi_codebooks: int
    sample_rate: int
    device: str
    enable_torch_compile: bool
    enable_cold_start: bool
    cold_start_generations: int
    include_audio_context: bool
    use_real_csm: bool
    
    @classmethod
    def from_ilava_config(cls, config: FERNConfig) -> "TTSConfig":
        """Create TTS config from main config."""
        return cls(
            model=config.tts_model,
            rvq_iterations=config.rvq_iterations,
            rvq_padding_method=config.rvq_padding_method,
            mimi_codebooks=config.mimi_codebooks,
            sample_rate=config.tts_sample_rate,
            device=config.device.value,
            enable_torch_compile=config.enable_torch_compile,
            enable_cold_start=config.enable_cold_start,
            cold_start_generations=config.cold_start_generations,
            include_audio_context=config.include_audio_context,
            use_real_csm=config.tts_use_real_csm
        )


class VADConfig(BaseModel):
    """VAD-specific configuration."""
    threshold: float
    silence_duration: float
    min_speech_duration: float
    
    @classmethod
    def from_ilava_config(cls, config: FERNConfig) -> "VADConfig":
        """Create VAD config from main config."""
        return cls(
            threshold=config.vad_threshold,
            silence_duration=config.vad_silence_duration,
            min_speech_duration=config.vad_min_speech_duration
        )

