import asyncio
from llm_helpers.llm_core.llm_handler import LLMHandler

async def test_llm(provider_name):
    handler = LLMHandler.create(provider_name)
    messages = [
        {"role": "user", "content": "Say hello in English."},
    ]
    response = await handler.send_request(messages=messages)
    print(f"{provider_name} response: {response}")
    assert "hello".lower() in response.lower()

if __name__ == "__main__":
    for provider in ["anthropic", "openai"]:
        try:
            asyncio.run(test_llm(provider))
        except Exception as e:
            print(f"Test failed for {provider}: {e}")
