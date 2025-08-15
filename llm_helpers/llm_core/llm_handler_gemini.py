import os
from typing import List, Dict, Optional
from .llm_handler import BaseLLMHandler
from .context_handler import ContextHandler
import google.genai as genai
from google.genai import types
import uuid
import logging

logger = logging.getLogger(__name__)

class GeminiLLMHandler(BaseLLMHandler):
    """Google Gemini LLM Handler with configurable settings."""
    
    # Default Gemini-specific settings
    DEFAULT_GEMINI_CONFIG = {
        "model": "gemini-2.0-flash-exp",
    }
    
    def __init__(self, 
                 model: Optional[str] = None,
                 **kwargs):
        """
        Initialize Gemini LLM handler.
        
        Args:
            model: Model name (default: gemini-2.0-flash-exp)
            **kwargs: Additional configuration parameters passed to BaseLLMHandler
        """
        # Initialize base handler with common config
        super().__init__(**kwargs)
        
        self.provider = "gemini"
        # Add Gemini-specific config
        gemini_config = self.DEFAULT_GEMINI_CONFIG.copy()
        if model is not None:
            gemini_config["model"] = model
            
        self.config.update(gemini_config)
        
        # Configure client
        # We can authenticate with Gemini using API key or ADC (Application Default Credentials)
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            logging.info("Authenticating with Gemini using API key")
            self.client = genai.Client(api_key=self.api_key)
        else:
            logging.info("Authenticating with Gemini (VertexAI) using ADC")
            self.client = genai.Client(vertexai=True)
        self.context_handler = ContextHandler()

    def _convert_messages_to_contents(self, messages: List[Dict[str, str]]) -> List[types.Content]:
        """
        Convert messages from the standard format to Gemini Content format.
        
        Args:
            messages: List of messages with 'role' and 'content' keys
            
        Returns:
            List of Content objects for Gemini API
        """
        contents = []
        for message in messages:
            role = message.get("role", "user")
            content_text = message.get("content", "")
            
            # Gemini uses 'user' and 'model' roles
            if role == "assistant":
                role = "model"
            
            content = types.Content(
                role=role,
                parts=[types.Part(text=content_text)]
            )
            contents.append(content)
        
        return contents

    async def send_request(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        include_history: bool = False,
        last_n_turns: Optional[int] = None,
        start_turn: Optional[int] = None,
        end_turn: Optional[int] = None,
        current_message: Optional[str] = None,
        current_metadata: Optional[Dict] = None
    ) -> str:
        # Generate a unique session ID for this request
        session_id = str(uuid.uuid4())
        
        try:
            # Acquire a session slot
            await self.rate_limiter.acquire_session(session_id)
            
            if include_history:
                history = self.context_handler.get_history(
                    last_n_turns=last_n_turns,
                    start_turn=start_turn,
                    end_turn=end_turn
                )
                messages = history + messages
            
            # Convert messages to Gemini Content format
            contents = self._convert_messages_to_contents(messages)
            
            # Prepare generation config
            config = types.GenerateContentConfig(
                temperature=self.config["temperature"],
                max_output_tokens=max_tokens or self.config["max_tokens"],
                top_p=self.config["top_p"]
            )
            
            # Add system instruction if provided
            if system:
                config.system_instruction = types.Content(
                    parts=[types.Part(text=system)]
                )
            
            # Generate content using async client
            response = await self.client.aio.models.generate_content(
                model=self.config["model"],
                contents=contents,
                config=config
            )
            
            # Store current message in history if provided
            if current_message:
                self.context_handler.add_to_history(
                    role="user",
                    content=current_message,
                    metadata=current_metadata
                )
            
            # Extract response text
            response_text = ""
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if part.text:
                        response_text += part.text
            
            if not response_text:
                raise ValueError("No valid response generated by Gemini")
            
            # Store response in history
            self.context_handler.add_to_history(
                role="assistant",
                content=response_text
            )
            
            logger.debug(f"LLM Response: {response_text}")
            return response_text
            
        except Exception as e:
            logger.error(f"Gemini LLM Error: {str(e)}")
            raise
        finally:
            # Always release the session slot
            await self.rate_limiter.release_session(session_id)
