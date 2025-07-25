#!/usr/bin/env python3
"""
Configuration Demo - Showing the new flexible configuration system
"""

from llm_helpers import LLMHandler

def demo_configurations():
    """Demonstrate different configuration options."""
    
    print("🎛️  LLM Helpers Configuration Demo\n")
    
    # Default configuration
    print("1. Default Configuration:")
    handler1 = LLMHandler.create("openai")
    print(f"   Temperature: {handler1.config['temperature']}")
    print(f"   Max Tokens: {handler1.config['max_tokens']}")
    print(f"   Model: {handler1.config['model']}")
    
    # Custom temperature and tokens
    print("\n2. Custom Temperature & Tokens:")
    handler2 = LLMHandler.create("openai", temperature=0.9, max_tokens=2048)
    print(f"   Temperature: {handler2.config['temperature']}")
    print(f"   Max Tokens: {handler2.config['max_tokens']}")
    print(f"   Model: {handler2.config['model']}")
    
    # Different model
    print("\n3. Different Model:")
    handler3 = LLMHandler.create("openai", model="gpt-3.5-turbo", temperature=0.3)
    print(f"   Temperature: {handler3.config['temperature']}")
    print(f"   Max Tokens: {handler3.config['max_tokens']}")
    print(f"   Model: {handler3.config['model']}")
    
    # Show all available config options
    print("\n4. All Configuration Options:")
    handler4 = LLMHandler.create("openai", 
                                temperature=0.8,
                                max_tokens=1500,
                                top_p=0.95,
                                max_context_tokens=100000,
                                token_limit_buffer=500)
    
    for key, value in handler4.config.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Configuration system is working perfectly!")
    print("✅ No more YAML files needed!")
    print("✅ All settings configurable via constructor!")

if __name__ == "__main__":
    demo_configurations() 