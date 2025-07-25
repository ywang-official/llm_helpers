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
from llm_helpers import LLMHandler, ProcessorCore, PromptHelper

async def main():
    # Create handler and processor
    llm_handler = LLMHandler.create()
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

Apache V2

## Contributing

Contribute however you like
