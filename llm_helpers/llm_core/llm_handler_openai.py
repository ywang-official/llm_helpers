import os
from openai import AsyncOpenAI
from typing import List, Dict, Optional
from .llm_handler import BaseLLMHandler
from .context_handler import ContextHandler
from .rate_limiter import rate_limiter
import uuid
import logging

logger = logging.getLogger(__name__)

class OpenAILLMHandler(BaseLLMHandler):
    """OpenAI GPT LLM Handler with configurable settings."""
    
    # Default OpenAI-specific settings
    DEFAULT_OPENAI_CONFIG = {
        "model": "gpt-4o-2024-11-20",
    }
    
    def __init__(self, 
                 model: Optional[str] = None,
                 **kwargs):
        """
        Initialize OpenAI LLM handler.
        
        Args:
            model: Model name (default: gpt-4o-2024-11-20)
            **kwargs: Additional configuration parameters passed to BaseLLMHandler
        """
        # Initialize base handler with common config
        super().__init__(**kwargs)
        
        self.provider = "openai"
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found")
        
        # Add OpenAI-specific config
        openai_config = self.DEFAULT_OPENAI_CONFIG.copy()
        if model is not None:
            openai_config["model"] = model
            
        self.config.update(openai_config)
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.context_handler = ContextHandler()

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
            await rate_limiter.acquire_session(session_id)
            
            if include_history:
                history = self.context_handler.get_history(
                    last_n_turns=last_n_turns,
                    start_turn=start_turn,
                    end_turn=end_turn
                )
                messages = history + messages
            # OpenAI expects 'role' to be 'user', 'assistant', or 'system'
            # and 'content' as the message text

            combined_messages = messages if not system else ([{"role": "system", "content": system}] + messages)

            response = await self.client.chat.completions.create(
                model=self.config["model"],
                messages=combined_messages,
                max_tokens=max_tokens or self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"]
            )

            if current_message:
                self.context_handler.add_to_history(
                    role="user",
                    content=current_message,
                    metadata=current_metadata
                )
            response_text = response.choices[0].message.content
            self.context_handler.add_to_history(
                role="assistant",
                content=response_text
            )
            logger.debug(f"LLM Response: {response_text}")
            return response_text
        except Exception as e:
            logger.error(f"OpenAI LLM Error: {str(e)}")
            raise
        finally:
            # Always release the session slot
            await rate_limiter.release_session(session_id)
