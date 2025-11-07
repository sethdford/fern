"""Dialogue management using OpenAI GPT-4o-mini."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    role: str  # "user" or "assistant"
    text: str
    audio_context: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None


@dataclass
class ConversationHistory:
    """Manages conversation history."""
    turns: List[ConversationTurn] = field(default_factory=list)
    max_turns: int = 10
    
    def add_turn(self, role: str, text: str, audio_context: Optional[Dict[str, Any]] = None):
        """Add a turn to conversation history."""
        turn = ConversationTurn(
            role=role,
            text=text,
            audio_context=audio_context
        )
        self.turns.append(turn)
        
        # Trim to max turns
        if len(self.turns) > self.max_turns * 2:  # *2 because each exchange has 2 turns
            self.turns = self.turns[-self.max_turns * 2:]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get conversation history as OpenAI messages format."""
        return [
            {"role": turn.role, "content": turn.text}
            for turn in self.turns
        ]
    
    def clear(self):
        """Clear conversation history."""
        self.turns.clear()


class DialogueManager:
    """
    Manages dialogue using OpenAI GPT-4o-mini for context-aware responses.
    
    The LLM serves as an intermediate component between ASR and TTS,
    generating contextually appropriate responses based on conversation history.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful voice assistant. Provide concise, natural responses suitable for voice conversation.",
        temperature: float = 0.7,
        max_tokens: int = 500,
        max_context_turns: int = 10,
    ):
        """
        Initialize Dialogue Manager.
        
        Args:
            api_key: OpenAI API key
            model: LLM model to use (gpt-4o-mini for low latency)
            system_prompt: System prompt for the LLM
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            max_context_turns: Maximum conversation turns to keep in context
        """
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Initialize conversation history
        self.history = ConversationHistory(max_turns=max_context_turns)
        
        logger.info(f"DialogueManager initialized with model: {model}")
    
    def generate_response(
        self,
        user_input: str,
        audio_context: Optional[Dict[str, Any]] = None,
        include_history: bool = True,
    ) -> str:
        """
        Generate LLM response for user input.
        
        Args:
            user_input: User's transcribed speech
            audio_context: Optional audio context (tone, sentiment, etc.)
            include_history: Whether to include conversation history
            
        Returns:
            Generated response text
        """
        if not user_input.strip():
            logger.warning("Empty user input received")
            return ""
        
        try:
            # Add user input to history
            self.history.add_turn("user", user_input, audio_context)
            
            # Build messages
            messages = [{"role": "system", "content": self.system_prompt}]
            
            if include_history:
                messages.extend(self.history.get_messages())
            else:
                messages.append({"role": "user", "content": user_input})
            
            # Generate response
            logger.debug(f"Generating response for: '{user_input[:50]}...'")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
            
            # Extract response text
            response_text = response.choices[0].message.content.strip()
            
            # Add assistant response to history
            self.history.add_turn("assistant", response_text)
            
            logger.debug(f"Generated response: '{response_text[:50]}...'")
            logger.info(
                f"LLM response generated: "
                f"{response.usage.prompt_tokens} prompt tokens, "
                f"{response.usage.completion_tokens} completion tokens"
            )
            
            return response_text
        
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            raise
    
    def generate_response_streaming(
        self,
        user_input: str,
        audio_context: Optional[Dict[str, Any]] = None,
        include_history: bool = True,
    ):
        """
        Generate LLM response with streaming.
        
        Args:
            user_input: User's transcribed speech
            audio_context: Optional audio context
            include_history: Whether to include conversation history
            
        Yields:
            Response text chunks
        """
        if not user_input.strip():
            logger.warning("Empty user input received")
            return
        
        try:
            # Add user input to history
            self.history.add_turn("user", user_input, audio_context)
            
            # Build messages
            messages = [{"role": "system", "content": self.system_prompt}]
            
            if include_history:
                messages.extend(self.history.get_messages())
            else:
                messages.append({"role": "user", "content": user_input})
            
            # Generate response with streaming
            logger.debug(f"Generating streaming response for: '{user_input[:50]}...'")
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            
            # Collect full response for history
            full_response = []
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response.append(content)
                    yield content
            
            # Add complete response to history
            response_text = "".join(full_response)
            self.history.add_turn("assistant", response_text)
            
            logger.debug(f"Streaming response complete: '{response_text[:50]}...'")
        
        except Exception as e:
            logger.error(f"Failed to generate streaming LLM response: {e}")
            raise
    
    def set_system_prompt(self, prompt: str):
        """
        Update system prompt.
        
        Args:
            prompt: New system prompt
        """
        self.system_prompt = prompt
        logger.info("System prompt updated")
    
    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()
        logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the conversation.
        
        Returns:
            Formatted conversation summary
        """
        if not self.history.turns:
            return "No conversation history."
        
        lines = ["Conversation History:", "-" * 40]
        for turn in self.history.turns:
            lines.append(f"{turn.role.upper()}: {turn.text}")
        lines.append("-" * 40)
        
        return "\n".join(lines)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment/tone of user input.
        
        Args:
            text: User input text
            
        Returns:
            Dictionary with sentiment analysis
        """
        try:
            prompt = f"""Analyze the sentiment and tone of the following text.
Provide a brief analysis including:
- Overall sentiment (positive, negative, neutral)
- Emotional tone
- Urgency level (low, medium, high)

Text: "{text}"

Provide response in format:
Sentiment: [sentiment]
Tone: [tone]
Urgency: [urgency]"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            # Parse response
            lines = analysis_text.split("\n")
            analysis = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    analysis[key.strip().lower()] = value.strip()
            
            return analysis
        
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "tone": "unknown", "urgency": "low"}

