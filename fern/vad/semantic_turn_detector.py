"""
Semantic Turn Detection using Small Language Model.

Based on: https://blog.speechmatics.com/semantic-turn-detection

Uses SmolLM2-360M-Instruct to predict end-of-turn probability
based on conversational context, not just silence.

Benefits:
- 60-75% reduction in false positives
- 3x improvement in user satisfaction
- Cost savings (~$91/year per 1000 conversations)
- Runs on CPU (<50ms latency)

Example:
    >>> from fern.vad.semantic_turn_detector import SemanticTurnDetector
    >>> detector = SemanticTurnDetector()
    >>> 
    >>> messages = [
    ...     {"role": "user", "content": "My ID is 123 764"}
    ... ]
    >>> 
    >>> prob = detector.predict_eot_prob(messages)
    >>> print(f"End-of-turn probability: {prob:.2%}")
    >>> 
    >>> if prob > 0.03:
    ...     print("User is done speaking")
    ... else:
    ...     print("User is mid-thought, extend grace period")
"""

import logging
import math
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class SemanticTurnDetector:
    """
    Semantic end-of-turn detector using SmolLM2-360M-Instruct.
    
    Analyzes conversation context to determine if user has finished speaking,
    reducing false positives from basic VAD silence detection.
    
    Args:
        model_name: HuggingFace model name (default: SmolLM2-360M-Instruct)
        threshold: Probability threshold for end-of-turn (default: 0.03)
        max_history: Maximum conversation history to consider (default: 10)
        device: Device to use (default: 'cpu')
        
    Example:
        >>> detector = SemanticTurnDetector()
        >>> 
        >>> messages = [
        ...     {"role": "assistant", "content": "What's your customer ID?"},
        ...     {"role": "user", "content": "It's 123 764"},
        ... ]
        >>> 
        >>> prob = detector.predict_eot_prob(messages)
        >>> is_done = prob > detector.threshold
    """
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        threshold: float = 0.03,
        max_history: int = 10,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.max_history = max_history
        
        # Auto-select device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        
        logger.info(f"Loading Semantic Turn Detector: {model_name}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Threshold: {threshold}")
        logger.info(f"  Max history: {max_history}")
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Semantic Turn Detector loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_eot_prob(
        self,
        messages: List[Dict[str, str]]
    ) -> float:
        """
        Predict probability that user turn is complete.
        
        Args:
            messages: Conversation history as list of dicts with 'role' and 'content'
                Example:
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help?"},
                    {"role": "user", "content": "I need help with"}
                ]
        
        Returns:
            Probability (0.0-1.0) that turn is complete
        """
        # Truncate to max history
        messages = messages[-self.max_history:]
        
        # Convert to ChatML format
        text_input = self._convert_to_chatml(messages)
        
        # Get next token logprobs
        logprobs = self._get_next_token_logprobs(text_input)
        
        # Extract end-of-turn probability
        eot_prob = self._extract_eot_prob(logprobs)
        
        return eot_prob
    
    def _convert_to_chatml(
        self,
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Convert messages to ChatML format.
        
        Format:
        <|im_start|>role<|im_sep|>content<|im_end|>
        
        For the last user message (the one we're checking),
        we DON'T add <|im_end|> because we want the model
        to predict whether it should come next.
        
        Args:
            messages: List of message dicts
        
        Returns:
            Formatted ChatML string
        """
        formatted = ""
        
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            
            formatted += f"<|im_start|>{role}<|im_sep|>{content}"
            
            # Only add <|im_end|> if NOT the last user message
            is_last_user = (i == len(messages) - 1) and (role == "user")
            if not is_last_user:
                formatted += "<|im_end|>"
        
        return formatted
    
    def _get_next_token_logprobs(
        self,
        text: str,
        top_k: int = 20,
    ) -> Dict[str, float]:
        """
        Get log probabilities for next token.
        
        Args:
            text: Input text in ChatML format
            top_k: Number of top tokens to return
        
        Returns:
            Dictionary mapping token string → log probability
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get logits for next token (last position)
        next_token_logits = outputs.logits[0, -1, :]  # (vocab_size,)
        
        # Compute log softmax
        log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        
        # Get top-k
        top_logprobs_vals, top_logprobs_ids = torch.topk(log_probs, top_k)
        
        # Convert to dict
        result = {}
        for i in range(top_k):
            token_id = top_logprobs_ids[i].item()
            token_str = self.tokenizer.decode([token_id])
            logprob = top_logprobs_vals[i].item()
            result[token_str] = logprob
        
        return result
    
    def _extract_eot_prob(
        self,
        logprobs: Dict[str, float],
        target_tokens: Optional[List[str]] = None,
    ) -> float:
        """
        Extract end-of-turn probability from logprobs.
        
        Checks for special tokens and punctuation that indicate EOT.
        
        Args:
            logprobs: Dict of token → log probability
            target_tokens: Tokens to check (default: <|im_end|>, ., ?, !)
        
        Returns:
            Maximum probability among target tokens
        """
        if target_tokens is None:
            target_tokens = ["<|im_end|>", ".", "?", "!"]
        
        max_prob = 0.0
        
        for token_str, logprob in logprobs.items():
            # Strip whitespace for matching
            stripped_token = token_str.strip()
            
            if stripped_token in target_tokens:
                # Convert log prob to prob
                prob = math.exp(logprob)
                max_prob = max(max_prob, prob)
        
        return max_prob
    
    def is_turn_complete(
        self,
        messages: List[Dict[str, str]],
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Determine if turn is complete based on threshold.
        
        Args:
            messages: Conversation history
            threshold: Custom threshold (uses self.threshold if None)
        
        Returns:
            True if turn is complete
        """
        if threshold is None:
            threshold = self.threshold
        
        prob = self.predict_eot_prob(messages)
        
        return prob > threshold


class HybridTurnDetector:
    """
    Hybrid turn detector combining VAD (silence) + Semantic (context).
    
    Strategy:
    1. VAD detects silence
    2. If silence > min_threshold:
       - Run semantic check
       - If EOT prob high → end turn
       - If EOT prob low → extend grace period
    3. If silence > max_threshold → force end
    
    Args:
        vad: VAD instance (e.g., SileroVAD)
        semantic: SemanticTurnDetector instance
        min_silence: Minimum silence before checking semantic (seconds)
        max_silence: Maximum silence before forcing end (seconds)
        eot_threshold: Semantic EOT probability threshold
        
    Example:
        >>> from fern.vad.silero_vad import SileroVAD
        >>> from fern.vad.semantic_turn_detector import (
        ...     SemanticTurnDetector,
        ...     HybridTurnDetector,
        ... )
        >>> 
        >>> vad = SileroVAD()
        >>> semantic = SemanticTurnDetector()
        >>> hybrid = HybridTurnDetector(vad, semantic)
        >>> 
        >>> # Check if turn is complete
        >>> is_complete = hybrid.is_turn_complete(
        ...     audio_buffer=audio,
        ...     transcript="My ID is 123 764",
        ...     conversation_history=[...],
        ... )
    """
    
    def __init__(
        self,
        vad,
        semantic: SemanticTurnDetector,
        min_silence: float = 0.3,
        max_silence: float = 2.0,
        eot_threshold: float = 0.03,
    ):
        self.vad = vad
        self.semantic = semantic
        self.min_silence = min_silence
        self.max_silence = max_silence
        self.eot_threshold = eot_threshold
        
        logger.info("Hybrid Turn Detector initialized")
        logger.info(f"  Min silence: {min_silence}s")
        logger.info(f"  Max silence: {max_silence}s")
        logger.info(f"  EOT threshold: {eot_threshold}")
    
    def is_turn_complete(
        self,
        audio_buffer: 'np.ndarray',
        transcript: str,
        conversation_history: List[Dict[str, str]],
    ) -> bool:
        """
        Determine if user turn is complete.
        
        Args:
            audio_buffer: Recent audio samples
            transcript: Current partial transcript
            conversation_history: Full conversation history
        
        Returns:
            True if turn is complete
        """
        # 1. Check VAD for silence
        silence_duration = self.vad.get_silence_duration(audio_buffer)
        
        # Not enough silence yet
        if silence_duration < self.min_silence:
            return False
        
        # Exceeded max silence → definitely done
        if silence_duration > self.max_silence:
            return True
        
        # 2. In between → use semantic check
        messages = conversation_history + [
            {"role": "user", "content": transcript}
        ]
        
        eot_prob = self.semantic.predict_eot_prob(messages)
        
        # 3. Decide based on probability
        if eot_prob > self.eot_threshold:
            # High probability → user is done
            return True
        else:
            # Low probability → extend grace period
            # Only end if we've hit max silence
            return silence_duration > self.max_silence


# Convenience function for quick testing
def test_semantic_turn_detection():
    """Test semantic turn detection with examples."""
    print("\n" + "=" * 70)
    print("Semantic Turn Detection Test")
    print("=" * 70)
    
    detector = SemanticTurnDetector()
    
    test_cases = [
        {
            "name": "Complete thought",
            "messages": [{"role": "user", "content": "I have a problem with my card"}],
            "expected": "High probability (complete)",
        },
        {
            "name": "Incomplete thought",
            "messages": [{"role": "user", "content": "My ID is 123 764"}],
            "expected": "Low probability (incomplete - recalling info)",
        },
        {
            "name": "Question",
            "messages": [{"role": "user", "content": "Can you help me?"}],
            "expected": "High probability (complete question)",
        },
        {
            "name": "Mid-sentence",
            "messages": [{"role": "user", "content": "I want to"}],
            "expected": "Low probability (clearly incomplete)",
        },
    ]
    
    for test in test_cases:
        prob = detector.predict_eot_prob(test["messages"])
        is_complete = prob > detector.threshold
        
        print(f"\nTest: {test['name']}")
        print(f"  Input: '{test['messages'][0]['content']}'")
        print(f"  EOT Probability: {prob:.4f}")
        print(f"  Is Complete: {is_complete}")
        print(f"  Expected: {test['expected']}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_semantic_turn_detection()

