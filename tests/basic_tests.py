#!/usr/bin/env python3
"""
Basic tests for llm-helpers package.
This example demonstrates the core functionality without requiring API keys.
"""

import asyncio
from llm_helpers import ContextHandler, LLMRateLimiter

def test_context_handler():
    """Test the context handler functionality."""
    print("Testing ContextHandler...")
    
    # Create context handler
    context = ContextHandler(max_history=5)
    
    # Add some conversations
    turn1 = context.add_to_history("user", "Hello, how are you?")
    turn2 = context.add_to_history("assistant", "I'm doing well, thank you!")
    turn3 = context.add_to_history("user", "What's the weather like?")
    turn4 = context.add_to_history("assistant", "I don't have access to weather data.")
    
    print(f"Added {len(context.dialogue_history)} turns to history")
    
    # Get conversation history
    full_history = context.get_history()
    print(f"Full history: {len(full_history)} messages")
    
    # Get last 2 turns
    recent_history = context.get_history(last_n_turns=2)
    print(f"Recent history (last 2): {len(recent_history)} messages")
    
    # Get specific turn range
    range_history = context.get_history(start_turn=turn1, end_turn=turn2)
    print(f"Range history: {len(range_history)} messages")
    
    print("✅ ContextHandler test passed!\n")

async def test_rate_limiter():
    """Test the rate limiter functionality."""
    print("Testing LLMRateLimiter...")
    
    # Create a rate limiter instance
    limiter = LLMRateLimiter(max_concurrent_sessions=2)
    
    # Test acquiring and releasing sessions
    session_id1 = "test_session_1"
    session_id2 = "test_session_2"
    
    # Acquire sessions
    await limiter.acquire_session(session_id1)
    await limiter.acquire_session(session_id2)
    
    # Check status
    status = await limiter.get_queue_status()
    print(f"Active sessions: {status['active_sessions']}")
    print(f"Waiting sessions: {status['waiting_sessions']}")
    
    # Release sessions
    await limiter.release_session(session_id1)
    await limiter.release_session(session_id2)
    
    final_status = await limiter.get_queue_status()
    print(f"Final active sessions: {final_status['active_sessions']}")
    
    print("✅ LLMRateLimiter test passed!\n")

def test_imports():
    """Test that all main classes can be imported."""
    print("Testing imports...")
    
    try:
        from llm_helpers import (
            LLMHandler, 
            BaseLLMHandler, 
            ContextHandler, 
            DialogueTurn,
            LLMRateLimiter, 
            rate_limiter,
            ProcessorCore
        )
        print("✅ All imports successful!")
        print(f"Available classes: {', '.join(['LLMHandler', 'BaseLLMHandler', 'ContextHandler', 'DialogueTurn', 'LLMRateLimiter', 'ProcessorCore'])}")
        print()
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    return True

async def main():
    """Run all tests."""
    print("🚀 Testing llm-helpers package\n")
    
    # Test imports first
    if not test_imports():
        return
    
    # Test context handler
    test_context_handler()
    
    # Test rate limiter
    await test_rate_limiter()
    
    print("🎉 All tests completed successfully!")
    print("\nNext steps:")
    print("1. Copy .env.example to .env and add your API keys")
    print("2. Run: poetry run python llm_helpers/llm_core/test_llm_providers.py")
    print("3. See README.md for usage examples")

if __name__ == "__main__":
    asyncio.run(main()) 