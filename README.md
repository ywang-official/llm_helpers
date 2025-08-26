# LLM Helpers

A Python library for LLM interactions with support for multiple providers, context management, and advanced text processing.

## Features

- **Multi-Provider Support**: Works with Anthropic Claude, OpenAI GPT, and Google Gemini models
- **Context Management**: Intelligent conversation history handling with configurable memory
- **Rate Limiting**: Built-in concurrent request management and session handling
- **Advanced Processing**: YAML-based prompt templates and JSON response processing
- **Async Support**: Full async/await support for all operations

## Installation

### Using Poetry (Recommended)

```bash
git clone <your-repo-url>
cd llm_helpers
poetry install
```

### Using pip

```bash
git clone <your-repo-url>
cd llm_helpers
pip install -e .
```

## Configuration

### Environment Variables

Create a `.env` file in your project root with the following variables:

```bash
# LLM Provider API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# LLM Configuration
LLM_PROVIDER=anthropic  # or openai or gemini
NODE_ENV=development    # or production
```

### Configuration Files

The library uses:
- Built-in default configurations (customizable via constructor parameters)
- `PromptHelper` for flexible prompt management and automatic JSON formatting
- `llm_helpers/prompts/sample_prompts.yaml` - Default prompt templates

## Quick Start

### Basic LLM Usage

#### Using Factory Method (Recommended)
```python
import asyncio
from llm_helpers import LLMHandler

async def main():
    # Create handler using factory method (uses LLM_PROVIDER env var)
    handler = LLMHandler.create()
    
    # Or specify provider explicitly
    handler = LLMHandler.create("gemini")  # or "openai", "anthropic"
    
    # Simple message
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    response = await handler.send_request(messages)
    print(response)

asyncio.run(main())
```

#### Provider-Specific Usage

##### Google Gemini
```python
import asyncio
from llm_helpers import GeminiLLMHandler

async def main():
    # Direct instantiation with custom settings
    handler = GeminiLLMHandler(
        model="gemini-2.0-flash-exp",
        temperature=0.8,
        max_tokens=1000
    )
    
    messages = [{"role": "user", "content": "Explain quantum computing"}]
    system_instruction = "You are a helpful science teacher."
    
    response = await handler.send_request(
        messages=messages,
        system=system_instruction
    )
    print(response)

asyncio.run(main())
```

##### OpenAI GPT
```python
import asyncio
from llm_helpers import LLMHandler

async def main():
    handler = LLMHandler.create("openai", model="gpt-4o", temperature=0.7)
    messages = [{"role": "user", "content": "Write a short story"}]
    response = await handler.send_request(messages)
    print(response)

asyncio.run(main())
```

##### Anthropic Claude
```python
import asyncio
from llm_helpers import LLMHandler

async def main():
    handler = LLMHandler.create("anthropic", model="claude-3-5-sonnet-20241022")
    messages = [{"role": "user", "content": "Analyze this text"}]
    response = await handler.send_request(messages)
    print(response)

asyncio.run(main())
```

#### Advanced Processing with Components
```python
import asyncio
from llm_helpers import LLMHandler, ProcessorCore, PromptHelper

async def main():
    # Create handler and processor
    llm_handler = LLMHandler.create("gemini")  # or any provider
    prompt_helper = PromptHelper("path/to/your/prompts")
    processor = ProcessorCore(llm_handler=llm_handler, prompt_helper=prompt_helper)
    
    # Process text with custom components
    components = {
        "demo_input": "Your text content here"
    }
    
    response = await processor.process(
        components=components,
        process_type="sample_string_prompt"
    )
    print(response)

asyncio.run(main())
```

### Prompt Management

```python
import asyncio
from llm_helpers import PromptHelper, ResponseType, LLMHandler

async def main():
    # Create prompt helper
    prompt_helper = PromptHelper()
    
    # Get a complete prompt bundle with automatic JSON formatting
    bundle = prompt_helper.get_prompt_bundle(
        "sample_string_prompt",
        response_type=ResponseType.PARSED_JSON
    )
    
    print(bundle)

main()
```

#### Output
```json
{
    "system_prompt": "This is a sample prompt used as a demo. Your job is to return Hello World, followed by a one-line summary of the input that was provided.\n\n\n\nYour task: return a valid JSON object with the format below. All valid json content should be wrapped with __json_start__ and __json_end__.\n- JSON object with key \"sample_string_prompt\":\n{\n  \"sample_string_prompt\": \"Your response content as a string\"\n}\n\n<expected response>\n__json_start__\n{\n  \"sample_string_prompt\": <your response content here>\n}\n__json_end__\n</expected response>\n\nExamples:\n<example_1_input>\nJohn is 20 years old. John loves sports.\n</example_1_input>\n<example_1_output>\n__json_start__\n{ \"sample_string_prompt\": \"Hello World, John is a sport loving 20 year old.\" }\n__json_end__\n</example_1_output>\n\n<example_2_input>\nBob loves watching spongebob, it\\'s his only purpose in life.\n</example_2_input>\n<example_2_output>\n__json_start__\n{ \"sample_string_prompt\": \"Hello World, Bob is a desparate spongebob fan.\" }\n__json_end__\n</example_2_output>\n",
    "user_prompt": "Please see input below:\n\n",
    "config":{
        "max_history_turns":10,
        "include_history":false,
        "max_tokens":2000
    },
    "metadata":{
        "prompt_key":"sample_string_prompt",
        "response_type":"<ResponseType.RAW_STRING":"raw_string"">",
        "custom_schema":"None"
    }
}
```

### Context Management

```python
from llm_helpers import ContextHandler, DialogueTurn

# Create context handler with memory limit
context = ContextHandler(max_history=10)

# Add conversations
turn_id = context.add_to_history("user", "Hello")
context.add_to_history("assistant", "Hi there!")

# Get conversation history
history = context.get_history(last_n_turns=2)
print(history)
```

### Rate Limiting

```python
import asyncio
from llm_helpers import LLMHandler, LLMRateLimiter

# Each handler has its own rate limiter by default
handler1 = LLMHandler.create("gemini", max_concurrent_sessions=10)
handler2 = LLMHandler.create("gemini", max_concurrent_sessions=5)

# Handlers are isolated - handler1 can have 10 concurrent sessions
# while handler2 is limited to 5, independently

# Share a rate limiter between handlers
shared_limiter = LLMRateLimiter(max_concurrent_sessions=8)
handler3 = LLMHandler.create("gemini", rate_limiter=shared_limiter)
handler4 = LLMHandler.create("openai", rate_limiter=shared_limiter)
# Now handler3 and handler4 share the same 8-session limit

# Manual rate limiter usage (for custom scenarios)
async def controlled_request():
    session_id = "unique_session_id"
    rate_limiter = handler1.rate_limiter  # Use handler's rate limiter
    
    # Acquire session slot
    await rate_limiter.acquire_session(session_id)
    
    try:
        # Your LLM operations here
        response = await handler1.send_request(messages)
    finally:
        # Always release the slot
        await rate_limiter.release_session(session_id)
```

#### ProcessorCore Rate Limiting

```python
from llm_helpers import ProcessorCore, LLMRateLimiter

# ProcessorCore with custom rate limiter configuration
processor = ProcessorCore(
    rate_limiter_config={"max_concurrent_sessions": 15}
)

# ProcessorCore with shared rate limiter
shared_limiter = LLMRateLimiter(max_concurrent_sessions=20)
processor = ProcessorCore(rate_limiter_config=shared_limiter)
```

#### Migration from Global Rate Limiter

**Old approach (deprecated):**
```python
# Global rate limiter - all handlers shared the same limiter
from llm_helpers import rate_limiter  # Global instance
```

**New approach (recommended):**
```python
# Each handler has its own rate limiter instance
handler = LLMHandler.create("gemini", max_concurrent_sessions=10)
# Access the handler's rate limiter
status = await handler.rate_limiter.get_queue_status()
```

## API Reference

### ProcessorCore

Advanced text processing with prompt templates.

#### Methods

- `async process_v2(components: Dict, process_type: str, context: Dict = None) -> str`
- `async parallel_batch_process(batches: List[Dict], process_type: str, context: Dict = None) -> List[str]`
- `async sequential_batch_process(batches: List[Dict], process_type: str, context: Dict = None, sequential_config: Dict = None) -> AsyncGenerator[Dict, None]`

## Configuration Details

### Built-in Configuration

All LLM handlers support these configuration parameters:

**Common Parameters:**
- `max_tokens`: Maximum tokens for response (default: 4096)
- `temperature`: Sampling temperature 0.0-1.0 (default: 0.7)
- `top_p`: Top-p sampling parameter (default: 0.9)
- `max_context_tokens`: Maximum context window (default: 200000)
- `token_limit_buffer`: Buffer to avoid token limits (default: 1000)

**OpenAI-specific:**
- `model`: Model name (default: "gpt-4o-2024-11-20")

**Anthropic-specific:**
- `model`: Model name (default: "claude-3-5-sonnet-20241022")
- `api_version`: API version (default: "2023-06-01")
- `base_url`: API base URL (default: "https://api.anthropic.com/v1/messages")

**Google Gemini-specific:**
- `model`: Model name (default: "gemini-2.0-flash-exp")

**Rate Limiter Configuration (All Handlers):**
- `rate_limiter`: LLMRateLimiter instance or dict config for creating one
- `max_concurrent_sessions`: Max concurrent sessions for rate limiter (if creating new)

**Examples:**
```python
# OpenAI with custom settings
handler = LLMHandler.create("openai", 
                           temperature=0.9, 
                           model="gpt-3.5-turbo",
                           max_tokens=2048)

# Anthropic with custom settings  
handler = LLMHandler.create("anthropic",
                           temperature=0.5,
                           model="claude-3-haiku-20240307")

# Gemini with custom settings
handler = LLMHandler.create("gemini",
                           temperature=0.8,
                           model="gemini-2.0-flash-exp",
                           max_tokens=2048)

# Rate limiter configuration examples
handler = LLMHandler.create("gemini", max_concurrent_sessions=20)
handler = LLMHandler.create("gemini", rate_limiter={"max_concurrent_sessions": 15})

# Shared rate limiter between handlers
shared_limiter = LLMRateLimiter(max_concurrent_sessions=10)
handler1 = LLMHandler.create("gemini", rate_limiter=shared_limiter)
handler2 = LLMHandler.create("openai", rate_limiter=shared_limiter)
```

### Prompt Templates (sample_prompts.yaml)

The library supports YAML-based prompt templates for different processing types. See `llm_helpers/prompts/prompts.yaml` for examples.

## Testing

Run the test suite:

```bash
# Test both providers
python llm_helpers/llm_core/test_llm_providers.py
```

Or with pytest:

```bash
poetry run pytest
```

## Development

### Code Quality

```bash
# Format code
poetry run black llm_helpers/

# Sort imports  
poetry run isort llm_helpers/

# Lint code
poetry run flake8 llm_helpers/

# Type checking
poetry run mypy llm_helpers/
```

## License

Apache V2

## Contributing

Contribute however you like
