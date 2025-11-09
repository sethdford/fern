"""
Gemini LLM integration for FERN conversational AI.

Provides conversational dialogue management using Google's Gemini models.
"""

import logging
import os
from typing import List, Dict, Optional
import google.generativeai as genai

logger = logging.getLogger(__name__)


class GeminiDialogueManager:
    """
    Manages conversational dialogue using Google Gemini.
    
    Features:
    - Fast response times (optimized for conversation)
    - Context-aware responses
    - Configurable personality via system prompts
    - Conversation history management
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-pro",  # Stable model name
        temperature: float = 0.7,
        max_tokens: int = 150,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize Gemini dialogue manager.
        
        Args:
            api_key: Google AI API key (or set GOOGLE_API_KEY env var)
            model_name: Gemini model to use
                - gemini-pro: Stable, fast for conversation (default)
                - gemini-1.5-pro-latest: Latest Pro model
                - models/gemini-1.5-flash-latest: Latest Flash (if available)
            temperature: Response creativity (0.0-1.0)
            max_tokens: Maximum response length
            system_prompt: Custom personality/instructions
        """
        # Get API key
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY env var or pass api_key parameter."
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Try to find an available model
        # The API keeps changing model names, so we try multiple options
        available_models = self._get_available_models()
        
        # Preference order for conversational AI
        model_preferences = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-1.5-pro-latest",
            "gemini-pro",
            model_name,  # Try user's requested model
        ]
        
        # Find first available model
        self.model_name = None
        for candidate in model_preferences:
            if candidate in available_models:
                self.model_name = candidate
                logger.info(f"Using Gemini model: {self.model_name}")
                break
        
        # Fallback to first available if none match
        if not self.model_name and available_models:
            self.model_name = available_models[0]
            logger.warning(f"Using fallback model: {self.model_name}")
        
        if not self.model_name:
            raise ValueError(
                "No compatible Gemini models found. Check your API key and quota."
            )
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Default system prompt for conversational assistant
        self.system_prompt = system_prompt or """You are FERN, a helpful and friendly AI voice assistant.

Key traits:
- Conversational and natural
- Concise responses (1-3 sentences for most queries)
- Warm and engaging tone
- Knowledgeable but humble
- Occasional light humor when appropriate

Remember: Your responses will be spoken aloud, so keep them natural and conversational."""
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 10  # Keep last 10 exchanges
        
        logger.info(f"Initialized Gemini dialogue manager with {self.model_name}")
    
    def _get_available_models(self) -> list[str]:
        """Get list of available Gemini models that support generateContent."""
        try:
            models = []
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    # Extract model name (e.g., "models/gemini-pro" -> "gemini-pro")
                    name = model.name.split('/')[-1] if '/' in model.name else model.name
                    models.append(name)
            return models
        except Exception as e:
            logger.warning(f"Could not list models: {e}")
            # Return common fallbacks
            return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    
    def generate_response(
        self,
        user_message: str,
        include_history: bool = True,
    ) -> str:
        """
        Generate a response to user message.
        
        Args:
            user_message: User's input text
            include_history: Include conversation history in context
        
        Returns:
            AI response text
        """
        try:
            # Build prompt with context
            if include_history and self.conversation_history:
                # Format conversation history
                history_text = self._format_history()
                prompt = f"{self.system_prompt}\n\n{history_text}\n\nUser: {user_message}\n\nAssistant:"
            else:
                prompt = f"{self.system_prompt}\n\nUser: {user_message}\n\nAssistant:"
            
            # Generate response with configuration
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,  # Optimize for low latency
            )
            response = self.model.generate_content(prompt, generation_config=generation_config)
            response_text = response.text.strip()
            
            # Add to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Trim history if too long
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history * 2:]
            
            logger.debug(f"Generated response: {response_text[:100]}...")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, I'm having trouble responding right now. Could you try again?"
    
    def _format_history(self) -> str:
        """Format conversation history for context."""
        formatted = []
        for msg in self.conversation_history[-self.max_history * 2:]:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted.append(f"{role}: {content}")
        return "\n\n".join(formatted)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Cleared conversation history")
    
    def set_system_prompt(self, prompt: str):
        """Update system prompt (personality/instructions)."""
        self.system_prompt = prompt
        logger.info("Updated system prompt")
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation."""
        if not self.conversation_history:
            return "No conversation yet."
        
        summary_prompt = f"""Summarize this conversation in 2-3 sentences:

{self._format_history()}

Summary:"""
        
        try:
            response = self.model.generate_content(summary_prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary."


# Backwards compatibility alias
GeminiManager = GeminiDialogueManager

