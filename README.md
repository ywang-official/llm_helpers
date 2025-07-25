# LLM Helpers

A Python library for LLM interactions with support for multiple providers, context management, and advanced text processing.

## Features

- **Multi-Provider Support**: Works with Anthropic Claude and OpenAI GPT models
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
LLM_PROVIDER=anthropic  # or openai
NODE_ENV=development    # or production
```

### Configuration Files

The library uses:
- Built-in default configurations (customizable via constructor parameters)
- `PromptHelper` for flexible prompt management and automatic JSON formatting
- `llm_helpers/prompts/prompts.yaml` - Default prompt templates

## Quick Start

### Basic LLM Usage

```python
import asyncio
from llm_helpers import LLMHandler

async def main():
    # Create an LLM handler (uses LLM_PROVIDER env var)
    handler = LLMHandler.create()
    
    # Or specify provider explicitly
    handler = LLMHandler.create("anthropic")  # or "openai"
    
    # Custom configuration
    handler = LLMHandler.create("openai", 
                               temperature=0.9, 
                               max_tokens=2048,
                               model="gpt-3.5-turbo")
    
    # Send a simple request
    response = await handler.send_request(
        messages=[{"role": "user", "content": "Hello, how are you?"}]
    )
    print(response)

# Run the async function
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
        "sequential_processing",
        response_type=ResponseType.PARSED_JSON,
        prev_story_summary="A hero begins their journey",
        chunk_window="Chapter 1: The adventure starts"
    )
    
    # Use with LLM
    handler = LLMHandler.create()
    response = await handler.send_request(
        messages=[{"role": "user", "content": bundle["user_prompt"]}],
        system=bundle["system_prompt"],
        **bundle["config"]
    )
    print(response)

asyncio.run(main())
```

### Advanced Processing

```python
import asyncio
from llm_helpers import LLMHandler, ProcessorCore

async def main():
    # Create handler and processor
    handler = LLMHandler.create()
    processor = ProcessorCore(handler)
    
    # Process text with custom components
    components = {
        "selected_parts": "Your text content here",
        "target_language": "English",
        "additional_instructions": "Be concise"
    }
    
    response = await processor.process(
        components=components,
        process_type="sequential_processing"
    )
    print(response)

asyncio.run(main())
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
from llm_helpers import rate_limiter

async def controlled_request():
    session_id = "unique_session_id"
    
    # Acquire session slot
    await rate_limiter.acquire_session(session_id)
    
    try:
        # Your LLM operations here
        pass
    finally:
        # Always release the slot
        await rate_limiter.release_session(session_id)
```

## API Reference

### LLMHandler

Main factory class for creating LLM handlers.

#### Methods

- `LLMHandler.create(provider: str = None) -> BaseLLMHandler`

### BaseLLMHandler

Abstract base class for LLM handlers.

### ContextHandler

Manages conversation history and context.

#### Methods

- `add_to_history(role: str, content: str, metadata: dict = None) -> int`
- `get_history(last_n_turns=None, start_turn=None, end_turn=None) -> List[Dict]`
- `clear_history() -> None`

### PromptHelper

Advanced prompt management with automatic JSON formatting.

#### Methods

- `get_prompt_bundle(prompt_key: str, response_type: ResponseType = None, **kwargs) -> Dict`
- `build_system_prompt(prompt_key: str, response_type: ResponseType = None) -> str`
- `build_user_prompt(prompt_key: str, **kwargs) -> str`
- `add_custom_prompt(prompt_key: str, system_prompt: str, user_prompt: str, config: Dict = None)`

### ProcessorCore

Advanced text processing with prompt templates.

#### Methods

- `async process_v2(components: Dict, process_type: str, context: Dict = None) -> str`
- `async parallel_batch_process(batches: List[Dict], process_type: str, context: Dict = None) -> List[str]`
- `async sequential_batch_process(batches: List[Dict], process_type: str, context: Dict = None, sequential_config: Dict = None) -> AsyncGenerator[Dict, None]`

### LLMRateLimiter

Manages concurrent LLM sessions.

#### Methods

- `async acquire_session(session_id: str) -> None`
- `async release_session(session_id: str) -> None`
- `async get_queue_status() -> Dict`

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
```

### Prompt Templates (prompts.yaml)

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

[Add your license information here]

## Contributing

[Add contributing guidelines here]
