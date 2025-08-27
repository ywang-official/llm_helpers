import os
from openai import AsyncAzureOpenAI
from typing import List, Dict, Optional
from .llm_handler import BaseLLMHandler
from .context_handler import ContextHandler
import uuid
import logging

logger = logging.getLogger(__name__)

class AzureOpenAILLMHandler(BaseLLMHandler):
    """Azure OpenAI LLM Handler with configurable settings."""
    
    # Default Azure OpenAI-specific settings
    DEFAULT_AZURE_OPENAI_CONFIG = {
        "model": "gpt-4o-2024-11-20",  # This will be treated as deployment name in Azure
        "api_version": "2024-02-15-preview",
    }
    
    def __init__(self, 
                 model: Optional[str] = None,
                 api_version: Optional[str] = None,
                 azure_endpoint: Optional[str] = None,
                 azure_deployment: Optional[str] = None,
                 azure_ad_token: Optional[str] = None,
                 azure_ad_token_provider: Optional[callable] = None,
                 **kwargs):
        """
        Initialize Azure OpenAI LLM handler.
        
        Args:
            model: Model deployment name (default: gpt-4o-2024-11-20)
            api_version: Azure API version (default: 2024-02-15-preview)
            azure_endpoint: Azure OpenAI endpoint URL
            azure_deployment: Azure deployment name (if different from model)
            azure_ad_token: Azure AD token for authentication
            azure_ad_token_provider: Function that returns Azure AD token
            **kwargs: Additional configuration parameters passed to BaseLLMHandler
        """
        # Initialize base handler with common config
        super().__init__(**kwargs)
        
        self.provider = "azure_openai"
        
        # Get API key from environment
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        # Handle Azure AD authentication
        if azure_ad_token is None:
            azure_ad_token = os.getenv("AZURE_OPENAI_AD_TOKEN")
        
        # Require at least one authentication method
        if not self.api_key and not azure_ad_token and not azure_ad_token_provider:
            raise ValueError(
                "Azure OpenAI authentication required. Please set AZURE_OPENAI_API_KEY "
                "environment variable or provide azure_ad_token/azure_ad_token_provider"
            )
        
        # Get Azure endpoint from environment if not provided
        if azure_endpoint is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise ValueError(
                "Azure endpoint required. Please set AZURE_OPENAI_ENDPOINT "
                "environment variable or provide azure_endpoint parameter"
            )
        
        # Get API version from environment if not provided
        if api_version is None:
            api_version = os.getenv("OPENAI_API_VERSION")
        if not api_version:
            api_version = self.DEFAULT_AZURE_OPENAI_CONFIG["api_version"]
        
        # Add Azure OpenAI-specific config
        azure_config = self.DEFAULT_AZURE_OPENAI_CONFIG.copy()
        if model is not None:
            azure_config["model"] = model
        if api_version is not None:
            azure_config["api_version"] = api_version
            
        self.config.update(azure_config)
        
        # Configure Azure OpenAI client
        client_kwargs = {
            "api_version": api_version,
            "azure_endpoint": azure_endpoint,
        }
        
        # Add authentication
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if azure_ad_token:
            client_kwargs["azure_ad_token"] = azure_ad_token
        if azure_ad_token_provider:
            client_kwargs["azure_ad_token_provider"] = azure_ad_token_provider
        
        # Add deployment if specified
        if azure_deployment:
            client_kwargs["azure_deployment"] = azure_deployment
        
        self.client = AsyncAzureOpenAI(**client_kwargs)
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
            
            # Azure OpenAI expects the same format as standard OpenAI
            # but uses deployment names instead of model names
            combined_messages = messages if not system else ([{"role": "system", "content": system}] + messages)

            response = await self.client.chat.completions.create(
                model=self.config["model"],  # This is the deployment name in Azure
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
            
            logger.debug(f"Azure OpenAI LLM Response: {response_text}")
            return response_text
            
        except Exception as e:
            logger.error(f"Azure OpenAI LLM Error: {str(e)}")
            raise
        finally:
            # Always release the session slot
            await self.rate_limiter.release_session(session_id)
