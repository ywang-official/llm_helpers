import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

# 加载环境变量 - 从项目根目录查找 .env 文件
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(env_path)

class BaseLLMHandler(ABC):
    """Base LLM Handler with configurable default settings."""
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_context_tokens": 200000,
        "token_limit_buffer": 1000,
    }
    
    def __init__(self, 
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 max_context_tokens: Optional[int] = None,
                 token_limit_buffer: Optional[int] = None,
                 **kwargs):
        """
        Initialize the LLM handler with configurable parameters.
        
        Args:
            max_tokens: Maximum tokens for response generation
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter
            max_context_tokens: Maximum context window size
            token_limit_buffer: Buffer to avoid hitting token limits
            **kwargs: Additional provider-specific parameters
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Override defaults with provided values
        if max_tokens is not None:
            self.config["max_tokens"] = max_tokens
        if temperature is not None:
            self.config["temperature"] = temperature
        if top_p is not None:
            self.config["top_p"] = top_p
        if max_context_tokens is not None:
            self.config["max_context_tokens"] = max_context_tokens
        if token_limit_buffer is not None:
            self.config["token_limit_buffer"] = token_limit_buffer
            
        # Add any additional kwargs to config
        self.config.update(kwargs)
        
    @abstractmethod
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
        """Send a request to the LLM and return the response text."""
        pass


class LLMHandler:
    """Factory class for creating LLM handlers based on provider."""
    
    @staticmethod
    def create(provider: str = None, **kwargs) -> BaseLLMHandler:
        """
        Create and return an LLM handler for the specified provider.
        
        Args:
            provider (str, optional): LLM provider. Defaults to the LLM_PROVIDER env var or "anthropic".
            **kwargs: Configuration parameters passed to the handler constructor
            
        Returns:
            BaseLLMHandler: Instance of the appropriate LLM handler
            
        Example:
            # Use default settings
            handler = LLMHandler.create("openai")
            
            # Custom configuration
            handler = LLMHandler.create("openai", temperature=0.9, model="gpt-3.5-turbo")
        """
        if provider is None:
            provider = os.getenv("LLM_PROVIDER", None).lower()
        else:
            provider = provider.lower()
            
        if provider == "openai":
            # Import here to avoid circular imports
            from .llm_handler_openai import OpenAILLMHandler
            return OpenAILLMHandler(**kwargs)
        elif provider == "anthropic":
            from .llm_handler_anthropic import AnthropicLLMHandler
            return AnthropicLLMHandler(**kwargs)
        elif provider == "gemini":
            from .llm_handler_gemini import GeminiLLMHandler
            return GeminiLLMHandler(**kwargs)
        else:
            raise ValueError(f"Invalid provider: {provider}")