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
                 max_concurrent_sessions: Optional[int] = None):
        """
        Initialize ProcessorCore with optional rate limiter configuration.
        
        Args:
            llm_handler: LLM handler instance. If None, creates default handler
            prompt_helper: Prompt helper instance. If None, creates default
            rate_limiter_config: Rate limiter configuration passed to LLM handler
            max_concurrent_sessions: Max concurrent sessions for rate limiter
        """
        # If no handler provided, create one with rate limiter configuration
        if llm_handler is None:
            self.llm_handler = LLMHandler.create(
                rate_limiter=rate_limiter_config,
                max_concurrent_sessions=max_concurrent_sessions
            )
        else:
            self.llm_handler = llm_handler
            
        self.prompt_helper = prompt_helper or PromptHelper()

    async def _clean_json_response(self, raw_response, retry_count=0, max_retries=3):
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
        cleaned_json_response = re.sub(r'[\x00-\x1F\x7F]', '', response)
        try:
            cleaned_json_response = json.loads(cleaned_json_response)
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
    ) -> str:
        """Process a request using the LLM and prompt helper."""
        # Get prompt bundle from PromptHelper
        bundle = self.prompt_helper.get_prompt_bundle(
            process_type,
            response_type=response_type,
            custom_schema=custom_schema,
            **components
        )
        config = bundle["config"]
        if hasattr(self.llm_handler, 'context_handler'):
            self.llm_handler.context_handler = ContextHandler(
                max_history=config.get("max_history_turns", 10)
            )
        try:
            system_prompt = bundle["system_prompt"]
            user_prompt = bundle["user_prompt"]
            response = await self.llm_handler.send_request(
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
            logger.error(f"processing error during {process_type}: {str(e)}")
            raise

    async def process_v2(
        self,
        components: Dict,
        process_type: str,
        context: Optional[Dict] = None,
        response_type: Optional[ResponseType] = None,
        custom_schema: Optional[Dict] = None,
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
    ) -> List[Dict]:
        try:
            results = []
            async for result in self.sequential_batch_process(chunks, process_type, context, sequential_config, response_type, custom_schema):
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Error in sequential batch processing: {str(e)}")
            raise