#!/usr/bin/env python3
"""
PromptHelper Demo - Showing the new prompt management capabilities
"""

import asyncio
from llm_helpers import PromptHelper, ResponseType, LLMHandler, ProcessorCore

def demo_basic_usage():
    """Demonstrate basic PromptHelper usage."""
    print("🚀 PromptHelper Demo\n")
    
    # Create PromptHelper with default prompts
    print("1. Loading default prompts...")
    prompt_helper = PromptHelper()
    
    print(f"   Loaded prompts: {prompt_helper.get_prompt_keys()}")
    print(f"   Total prompts: {len(prompt_helper.get_prompt_keys())}")
    
    # Show prompt details
    print("\n2. Prompt details:")
    for key, details in prompt_helper.list_prompts().items():
        print(f"   {key}: system={details['has_system_prompt']}, user={details['has_user_prompt']}, config_keys={details['config_keys']}")

def demo_prompt_building():
    """Demonstrate prompt building with different response types."""
    print("\n3. Building prompts with different response types...")

    demo_prompt_building_with_prompt_key("sample_string_prompt")
    demo_prompt_building_with_prompt_key("sample_json_prompt")

def demo_prompt_building_with_prompt_key(prompt_key: str):
    prompt_helper = PromptHelper()
    
    # Raw string response
    print(f"\n   🔸 Raw String Response for '{prompt_key}':")
    system_prompt_raw = prompt_helper.build_system_prompt(
        prompt_key, 
        response_type=ResponseType.RAW_STRING
    )
    print(f"   System prompt length: {len(system_prompt_raw)} chars")
    print(f"   Contains JSON instructions: {'__json_start__' in system_prompt_raw}")
    print(f"   Full System prompt:\n{system_prompt_raw}")
    
    # JSON schema response (default)
    print(f"\n   🔸 JSON Schema Response for '{prompt_key}':")
    system_prompt_json = prompt_helper.build_system_prompt(
        prompt_key, 
        response_type=ResponseType.PARSED_JSON
    )
    print(f"   System prompt length: {len(system_prompt_json)} chars")
    print(f"   Contains JSON instructions: {'__json_start__' in system_prompt_json}")
    
    # Custom schema
    print(f"\n   🔸 Custom Schema Response for '{prompt_key}':")
    custom_schema = {
        "summary": "string",
        "characters": ["character_name"],
        "themes": ["theme1", "theme2"]
    }
    system_prompt_custom = prompt_helper.build_system_prompt(
        prompt_key,
        response_type=ResponseType.PARSED_JSON,
        custom_schema=custom_schema
    )
    print(f"   System prompt length: {len(system_prompt_custom)} chars")
    print(f"   Contains custom schema: {'characters' in system_prompt_custom}")

def demo_user_prompt_building():
    """Demonstrate user prompt building with variable substitution."""
    print("\n4. Building user prompts with variables...")
    
    demo_user_prompt_building_with_prompt_key("sample_string_prompt")
    demo_user_prompt_building_with_prompt_key("sample_json_prompt")

def demo_user_prompt_building_with_prompt_key(prompt_key: str):
    prompt_helper = PromptHelper()
    
    # Build user prompt with variables
    user_prompt = prompt_helper.build_user_prompt(
        prompt_key,
        demo_input="John is 20 years old. John loves sports."
    )
    
    print(f"   User prompt length: {len(user_prompt)} chars")
    print(f"   Contains variables: {'adventure begins' in user_prompt}")
    print(f"   Full User prompt:\n{user_prompt}")

def demo_prompt_bundle():
    """Demonstrate getting complete prompt bundles."""
    print("\n5. Getting complete prompt bundles...")
    
    prompt_helper = PromptHelper()
    
    # Get a complete bundle
    bundle = prompt_helper.get_prompt_bundle(
        "sample_string_prompt"
    )
    
    print(f"   Bundle keys: {list(bundle.keys())}")
    print(f"   System prompt length: {len(bundle['system_prompt'])} chars")
    print(f"   User prompt length: {len(bundle['user_prompt'])} chars")
    print(f"   Config: {bundle['config']}")
    print(f"   Metadata: {bundle['metadata']}")

def demo_custom_prompts():
    """Demonstrate adding custom prompts at runtime."""
    print("\n6. Adding custom prompts...")
    
    prompt_helper = PromptHelper()
    
    # Add a custom prompt
    prompt_helper.add_custom_prompt(
        prompt_key="story_summary",
        system_prompt="You are a master storyteller. Summarize the given story in an engaging way.",
        user_prompt="Please summarize this story: {{story_content}}",
        config={"max_tokens": 500, "temperature": 0.8}
    )
    
    print(f"   Added custom prompt: story_summary")
    print(f"   Total prompts now: {len(prompt_helper.get_prompt_keys())}")
    
    # Use the custom prompt
    bundle = prompt_helper.get_prompt_bundle(
        "story_summary",
        response_type=ResponseType.RAW_STRING,
        story_content="Once upon a time, there was a brave knight..."
    )
    
    print(f"   Custom prompt system length: {len(bundle['system_prompt'])} chars")
    print(f"   Custom prompt user length: {len(bundle['user_prompt'])} chars")

async def demo_with_llm():
    """Demonstrate using PromptHelper with actual LLM."""
    print("\n7. Testing with LLM (if API key available)...")
    
    try:
        # Create LLM handler and prompt helper
        llm_handler = LLMHandler.create()
        prompt_helper = PromptHelper()
        processor = ProcessorCore(llm_handler=llm_handler, prompt_helper=prompt_helper)
        
        # Add a simple custom prompt for testing
        prompt_helper.add_custom_prompt(
            prompt_key="simple_test",
            system_prompt="You are a helpful assistant. Always respond politely.",
            user_prompt="Please say hello and tell me your name: {{user_name}}",
            response_type=ResponseType.RAW_STRING,
            config={"max_tokens": 100, "temperature": 0.7}
        )
        
        # Send to LLM
        response = await processor.process_v2(
            components={"user_name": "Alice"},
            process_type="simple_test",
            response_type=ResponseType.RAW_STRING
        )
        
        print(f"   ✅ LLM Response: {response[:100]}...")
        
    except Exception as e:
        print(f"   ⚠️ LLM test skipped (API key issue): {str(e)[:50]}...")

async def main():
    """Run all demonstrations."""
    demo_basic_usage()
    demo_prompt_building()
    demo_user_prompt_building()
    demo_prompt_bundle()
    demo_custom_prompts()
    await demo_with_llm()
    
    print("\n🎉 PromptHelper demo completed!")
    print("\n✅ Key Features Demonstrated:")
    print("   • Load prompts from YAML files/directories")
    print("   • Automatic JSON formatting for LLM responses") 
    print("   • Support for RAW_STRING and PARSED_JSON response types")
    print("   • Variable substitution in prompt templates")
    print("   • Custom schema support for structured responses")
    print("   • Runtime addition of custom prompts")
    print("   • Complete prompt bundles ready for LLM processing")

if __name__ == "__main__":
    asyncio.run(main()) 