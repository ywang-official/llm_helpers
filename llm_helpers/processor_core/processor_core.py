from typing import Dict, Optional, List, AsyncGenerator, Union, TYPE_CHECKING
import logging
import os
import json
import asyncio
import re

from ..llm_core.llm_handler import LLMHandler, BaseLLMHandler
from ..llm_core.context_handler import ContextHandler
from ..prompt_core.prompt_helper import PromptHelper, ResponseType

if TYPE_CHECKING:
    from ..llm_core.rate_limiter import LLMRateLimiter

logger = logging.getLogger(__name__)

class ProcessorCore:
    def __init__(self, 
                 llm_handler: Optional[BaseLLMHandler] = None, 
                 prompt_helper: Optional[PromptHelper] = None,
                 rate_limiter_config: Optional[Union[Dict, 'LLMRateLimiter']] = None,
                 max_concurrent_sessions: Optional[int] = None,
                 llm_handlers: Optional[Dict[str, BaseLLMHandler]] = None):
        """
        Initialize ProcessorCore with optional rate limiter configuration and multiple handlers.
        
        Args:
            llm_handler: Default LLM handler instance. If None, determines default from llm_handlers
            prompt_helper: Prompt helper instance. If None, creates default
            rate_limiter_config: Rate limiter configuration passed to LLM handler
            max_concurrent_sessions: Max concurrent sessions for rate limiter
            llm_handlers: Dictionary of named LLM handlers {name: handler_instance}
        """
        self.prompt_helper = prompt_helper or PromptHelper()
        
        # Initialize handlers dictionary
        self.llm_handlers: Dict[str, BaseLLMHandler] = {}
        
        # Validate all handlers in llm_handlers first
        if llm_handlers:
            for name, handler in llm_handlers.items():
                if not isinstance(handler, BaseLLMHandler):
                    raise ValueError(f"Handler '{name}' must be an instance of BaseLLMHandler")
        
        # Determine default handler following priority order
        default_handler = None
        
        if llm_handler is not None:
            # Priority 1: Explicitly provided llm_handler
            default_handler = llm_handler
        elif llm_handlers:
            # Priority 2: Handler named "default" in llm_handlers
            if "default" in llm_handlers:
                default_handler = llm_handlers["default"]
            # Priority 3: Single handler in llm_handlers
            elif len(llm_handlers) == 1:
                default_handler = list(llm_handlers.values())[0]
        
        # Priority 4: Create new handler if none found
        if default_handler is None:
            default_handler = LLMHandler.create(
                rate_limiter=rate_limiter_config,
                max_concurrent_sessions=max_concurrent_sessions
            )

        # Set default handler
        self.llm_handlers["default"] = default_handler
        self.llm_handler = default_handler  # Maintain backward compatibility
        
        # Add all remaining handlers from llm_handlers
        if llm_handlers:
            for name, handler in llm_handlers.items():
                self.llm_handlers[lower(name)] = handler

    def add_llm_handler(self, name: str, handler: BaseLLMHandler) -> None:
        """
        Add a new LLM handler with a custom name.
        
        Args:
            name: Custom name for the handler
            handler: LLM handler instance
            
        Raises:
            ValueError: If handler is not a BaseLLMHandler instance
        """
        if not isinstance(handler, BaseLLMHandler):
            raise ValueError(f"Handler must be an instance of BaseLLMHandler")
        self.llm_handlers[name] = handler
        logger.info(f"Added LLM handler '{name}' ({handler.provider})")
    
    def remove_llm_handler(self, name: str) -> bool:
        """
        Remove an LLM handler by name.
        
        Args:
            name: Name of the handler to remove
            
        Returns:
            bool: True if handler was removed, False if not found
            
        Raises:
            ValueError: If trying to remove the default handler
        """
        if name == "default":
            raise ValueError("Cannot remove the default handler")
        
        if name in self.llm_handlers:
            del self.llm_handlers[name]
            logger.info(f"Removed LLM handler '{name}'")
            return True
        return False
    
    def get_llm_handler(self, name: str) -> Optional[BaseLLMHandler]:
        """
        Get an LLM handler by name.
        
        Args:
            name: Name of the handler
            
        Returns:
            BaseLLMHandler: The handler instance, or None if not found
        """
        return self.llm_handlers.get(name)
    
    def list_llm_handlers(self) -> Dict[str, str]:
        """
        List all available LLM handlers.
        
        Returns:
            Dict[str, str]: Dictionary mapping handler names to their provider types
        """
        return {name: handler.provider for name, handler in self.llm_handlers.items()}
    
    def set_default_handler(self, name: str) -> None:
        """
        Set a different handler as the default.
        
        Args:
            name: Name of the handler to set as default
            
        Raises:
            ValueError: If handler name not found
        """
        if name not in self.llm_handlers:
            raise ValueError(f"Handler '{name}' not found")
        
        # Update the default handler reference
        self.llm_handlers["default"] = self.llm_handlers[name]
        self.llm_handler = self.llm_handlers[name]  # Maintain backward compatibility
        logger.info(f"Set handler '{name}' as default")

    async def _clean_json_response(self, raw_response, retry_count=0, max_retries=2):
        """Clean and parse JSON response from LLM."""
        start_index = raw_response.find("__json_start__")
        end_index = raw_response.find("__json_end__")
        response = raw_response
        if start_index != -1:
            if end_index == -1:
                response = response[start_index + len("__json_start__"):]
            else:
                response = response[start_index + len("__json_start__"):end_index]
        else:
            raise ValueError("Invalid response from processor")
        logger.debug(f"Raw response: {repr(response)}")
        # cleaned_json_response = re.sub(r'[\x00-\x1F\x7F]', '', response)
        cleaned_json_response = response

        try:
            cleaned_json_response = json.loads(cleaned_json_response)
            logger.debug(f"Parsed JSON response: {cleaned_json_response}")
        
        except json.JSONDecodeError as e:
            logger.warning(f"Error parsing JSON response: {e}, trying to fix unescaped newlines...")
            cleaned_json_response = cleaned_json_response.replace("\n", "\\n")
            logger.debug(f"Cleaned JSON response: {repr(cleaned_json_response)}")
            try:
                cleaned_json_response = json.loads(cleaned_json_response)
                logger.debug(f"Parsed JSON response: {cleaned_json_response}")
            
            except Exception as e:
                logger.error(f"Error parsing JSON response (attempt {retry_count + 1}): {e}")
                logger.error(f"Response: {raw_response}")
                if retry_count >= max_retries:
                    logger.error(f"Failed to fix JSON after {max_retries + 1} attempts")
                    raise ValueError(f"Invalid JSON response from processor after {max_retries + 1} attempts")
                try:
                    logger.info(f"Attempting to reparse JSON response (attempt {retry_count + 1})")
                    components = {'input': raw_response}
                    response = await self.process(
                        components=components,
                        process_type="__fix_json__",
                    )
                    return await self._clean_json_response(response, retry_count + 1, max_retries)
                except Exception as e:
                    logger.error(f"Error fixing JSON response: {e}")
                    logger.error(f"Response: {raw_response}")
                    raise ValueError("Invalid JSON response from processor")
        return cleaned_json_response

    async def process(
        self,
        components: Dict,
        process_type: str,
        context: Optional[Dict] = None,
        response_type: Optional[ResponseType] = None,
        custom_schema: Optional[Dict] = None,
        handler: Optional[str] = None,
    ) -> str:
        """
        Process a request using the LLM and prompt helper.
        
        Args:
            components: Dictionary of components for prompt building
            process_type: Type of processing to perform
            context: Optional context dictionary
            response_type: Optional response type specification
            custom_schema: Optional custom schema for response formatting
            handler: Optional name of specific handler to use (defaults to "default")
        
        Returns:
            str: The response from the LLM
        """
        # Select handler
        handler_name = handler or "default"
        if handler_name not in self.llm_handlers:
            raise ValueError(f"Handler '{handler_name}' not found. Available handlers: {list(self.llm_handlers.keys())}")
        
        selected_handler = self.llm_handlers[handler_name]
        
        # Get prompt bundle from PromptHelper
        bundle = self.prompt_helper.get_prompt_bundle(
            process_type,
            response_type=response_type,
            custom_schema=custom_schema,
            **components
        )
        config = bundle["config"]
        if hasattr(selected_handler, 'context_handler'):
            selected_handler.context_handler = ContextHandler(
                max_history=config.get("max_history_turns", 10)
            )
        try:
            system_prompt = bundle["system_prompt"]
            user_prompt = bundle["user_prompt"]
            logger.debug(f"System prompt: {system_prompt}")
            logger.debug(f"User prompt: {user_prompt}")
            logger.debug(f"Using handler: {handler_name} ({selected_handler.provider})")
            response = await selected_handler.send_request(
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
                max_tokens=config.get("max_tokens"),
                include_history=config.get("include_history", False),
                last_n_turns=config.get("last_n_turns"),
                start_turn=config.get("start_turn"),
                end_turn=config.get("end_turn"),
                current_message=user_prompt,
                current_metadata=context
            )
            return response
        except Exception as e:
            logger.error(f"processing error during {process_type} with handler '{handler_name}': {str(e)}")
            raise

    async def process_v2(
        self,
        components: Dict,
        process_type: str,
        context: Optional[Dict] = None,
        response_type: Optional[ResponseType] = None,
        custom_schema: Optional[Dict] = None,
        handler: Optional[str] = None,
    ) -> str:
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                response = await self.process(
                    components=components,
                    process_type=process_type,
                    context=context,
                    response_type=response_type,
                    custom_schema=custom_schema,
                    handler=handler,
                )
                cleaned_response = await self._clean_json_response(response)
                if process_type not in cleaned_response:
                    raise ValueError(f"Invalid JSON response from processor, missing main key: {process_type}")
                return cleaned_response[process_type]
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    raise Exception(f"Failed to process after {max_retries} retries: {e}")

    async def parallel_batch_process(
            self,
            batches: List[Dict],
            process_type: str,
            context: Optional[Dict] = None,
            response_type: Optional[ResponseType] = None,
            custom_schema: Optional[Dict] = None,
            handler: Optional[str] = None,
    ) -> List[str]:
        try:
            bundle = self.prompt_helper.get_prompt_bundle(process_type)
            config = bundle["config"]
            max_retries = config.get("max_retries", 5)
            async def process_batch_with_retry(order: int, batch: Dict):
                retries = 0
                while retries < max_retries:
                    try:
                        response = await self.process_v2(
                            components=batch,
                            process_type=process_type,
                            context=context,
                            response_type=response_type,
                            custom_schema=custom_schema,
                            handler=handler,
                        )
                        return (order, response)
                    except Exception as e:
                        retries += 1
                        if retries >= max_retries:
                            raise Exception(f"Failed to process batch after {max_retries} retries: {e}")
                        await asyncio.sleep(2 ** retries)
                        logger.warning(f"Failed to process batch after {retries} retries: {e}")
            tasks = [process_batch_with_retry(order=i, batch=batch) for i, batch in enumerate(batches)]
            unordered_responses = await asyncio.gather(*tasks)
            ordered_responses = [response[1] for response in sorted(unordered_responses, key=lambda x: x[0])]
            return ordered_responses
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise


    async def sequential_batch_process(
            self,
            chunks: List[Dict],
            process_type: str,
            context: Optional[Dict] = None,
            sequential_config: Optional[Dict] = None,
            response_type: Optional[ResponseType] = None,
            custom_schema: Optional[Dict] = None,
            handler: Optional[str] = None,
    ) -> AsyncGenerator[Dict, None]:
        try:
            response = {val: "n/a" for val in sequential_config.values()}
            for chunk in chunks:
                for key, val in sequential_config.items():
                    if val in response:
                        chunk[key] = response.get(val)
                response = await self.process_v2(
                    components=chunk,
                    process_type=process_type,
                    context=context,
                    response_type=response_type,
                    custom_schema=custom_schema,
                    handler=handler,
                )
                yield response
        except Exception as e:
            logger.error(f"Error in sequential processing: {str(e)}")
            raise


    async def sequential_batch_process_all(
            self,
            chunks: List[Dict],
            process_type: str,
            context: Optional[Dict] = None,
            sequential_config: Optional[Dict] = None,
            response_type: Optional[ResponseType] = None,
            custom_schema: Optional[Dict] = None,
            handler: Optional[str] = None,
    ) -> List[Dict]:
        try:
            results = []
            async for result in self.sequential_batch_process(chunks, process_type, context, sequential_config, response_type, custom_schema, handler):
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Error in sequential batch processing: {str(e)}")
            raise