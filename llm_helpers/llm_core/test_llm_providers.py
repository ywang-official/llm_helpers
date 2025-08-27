import asyncio
import os
from llm_helpers.llm_core.llm_handler import LLMHandler

async def test_llm(provider_name):
    """Test a specific LLM provider."""
    try:
        handler = LLMHandler.create(provider_name)
        messages = [
            {"role": "user", "content": "Say hello in English."},
        ]
        response = await handler.send_request(messages=messages)
        print(f"✅ {provider_name} response: {response}")
        assert "hello".lower() in response.lower()
        return True
    except Exception as e:
        print(f"❌ Test failed for {provider_name}: {e}")
        return False

def check_provider_requirements(provider_name):
    """Check if the provider has the required environment variables."""
    requirements = {
        "anthropic": ["ANTHROPIC_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "azure_openai": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
        "gemini": ["GOOGLE_API_KEY"]  # Can also use ADC (Application Default Credentials)
    }
    
    missing_vars = []
    for var in requirements.get(provider_name, []):
        if not os.getenv(var):
            missing_vars.append(var)
    
    return missing_vars

async def test_all_providers():
    """Test all available LLM providers."""
    providers = ["anthropic", "openai", "azure_openai", "gemini"]
    results = {}
    
    print("🧪 Testing LLM Providers")
    print("=" * 50)
    
    for provider in providers:
        print(f"\n📡 Testing {provider}...")
        
        # Check requirements
        missing_vars = check_provider_requirements(provider)
        if missing_vars:
            print(f"⚠️  Skipping {provider}: Missing environment variables: {missing_vars}")
            results[provider] = "skipped"
            continue
        
        # Run test
        success = await test_llm(provider)
        results[provider] = "passed" if success else "failed"
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    for provider, result in results.items():
        status_emoji = {"passed": "✅", "failed": "❌", "skipped": "⚠️"}
        print(f"{status_emoji.get(result, '❓')} {provider}: {result}")
    
    return results

if __name__ == "__main__":
    # Run comprehensive tests
    asyncio.run(test_all_providers())
