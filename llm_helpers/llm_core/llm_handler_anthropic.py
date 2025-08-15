import os
from typing import List, Dict, Optional
from .llm_handler import BaseLLMHandler
from .context_handler import ContextHandler
from anthropic import AsyncAnthropic
import uuid
import logging

logger = logging.getLogger(__name__)

class AnthropicLLMHandler(BaseLLMHandler):
    """Anthropic Claude LLM Handler with configurable settings."""
    
    # Default Anthropic-specific settings
    DEFAULT_ANTHROPIC_CONFIG = {
        "model": "claude-3-5-sonnet-20241022",
        "api_version": "2023-06-01",
        "base_url": "https://api.anthropic.com/v1/messages"
    }
    
    def __init__(self, 
                 model: Optional[str] = None,
                 api_version: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs):
        """
        Initialize Anthropic LLM handler.
        
        Args:
            model: Model name (default: claude-3-5-sonnet-20241022)
            api_version: API version (default: 2023-06-01)
            base_url: Base URL for API (default: https://api.anthropic.com/v1/messages)
            **kwargs: Additional configuration parameters passed to BaseLLMHandler
        """
        # Initialize base handler with common config
        super().__init__(**kwargs)
        
        self.provider = "anthropic"
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 ANTHROPIC_API_KEY 环境变量")
        
        # Add Anthropic-specific config
        anthropic_config = self.DEFAULT_ANTHROPIC_CONFIG.copy()
        if model is not None:
            anthropic_config["model"] = model
        if api_version is not None:
            anthropic_config["api_version"] = api_version
        if base_url is not None:
            anthropic_config["base_url"] = base_url
            
        self.config.update(anthropic_config)
        self.client = AsyncAnthropic(api_key=self.api_key)
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
            await self.rate_limiter.acquire_session(session_id)
            
            if include_history:
                history = self.context_handler.get_history(
                    last_n_turns=last_n_turns,
                    start_turn=start_turn,
                    end_turn=end_turn
                )
                messages = history + messages
            # Only include 'system' if it is a string
            create_args = dict(
                model=self.config["model"],
                max_tokens=max_tokens or self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                messages=messages,
            )
            if isinstance(system, str):
                create_args["system"] = system
            response = await self.client.messages.create(**create_args)
            if current_message:
                self.context_handler.add_to_history(
                    role="user",
                    content=current_message,
                    metadata=current_metadata
                )
            response_text = response.content[0].text
            self.context_handler.add_to_history(
                role="assistant",
                content=response_text
            )
            logger.debug(f"LLM Response: {response_text}")
            return response_text
        except Exception as e:
            logger.error(f"LLM Error: {str(e)}")
            raise
        finally:
            # Always release the session slot
            await self.rate_limiter.release_session(session_id)