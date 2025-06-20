#!/usr/bin/env python3
"""
Basic example of using the AIService component for generating AI responses.
This demonstrates initialization, configuration, and simple response generation.
"""

import asyncio
import json
import os
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the AIService
from core.ai_service import AIService
from core.service_registry import ServiceRegistry

async def main():
    # Initialize the service registry (required for service dependencies)
    registry = ServiceRegistry()
    
    # Create and register the AI service
    ai_service = AIService()
    registry.register_service(ai_service)
    
    # Configure the AI service with API keys from environment variables
    # (Replace with your own API keys or use environment variables)
    api_config = {
        "models": [
            {
                "name": "default",
                "type": "openai",
                "model_id": "gpt-3.5-turbo",
                "api_key": os.environ.get("OPENAI_API_KEY")
            },
            {
                "name": "anthropic",
                "type": "anthropic",
                "model_id": "claude-2",
                "api_key": os.environ.get("ANTHROPIC_API_KEY")
            },
            {
                "name": "google",
                "type": "google",
                "model_id": "gemini-pro",
                "api_key": os.environ.get("GOOGLE_API_KEY")
            }
        ],
        "default_model": "default",
        "cache": {
            "backend": "memory",
            "default_ttl": 3600  # Cache for 1 hour by default
        }
    }
    
    # Initialize the service with the configuration
    ai_service.initialize(api_config)
    
    # Basic prompt example
    prompt = "Explain the benefits of AI service optimization in a web scraping framework in 3 bullet points."
    
    # Generate a response using the default model
    print("\n--- Generating response with default model ---")
    response = await ai_service.generate_response(prompt)
    print_ai_response(response)
    
    # Generate a response using a specific model
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\n--- Generating response with Anthropic model ---")
        anthropic_response = await ai_service.generate_response(prompt, model_name="anthropic")
        print_ai_response(anthropic_response)
    
    # Disable caching for a specific request
    print("\n--- Generating response with caching disabled ---")
    no_cache_response = await ai_service.generate_response(prompt, use_cache=False)
    print_ai_response(no_cache_response)
    
    # Add context to affect the response generation
    context = {
        "task_type": "summarization",
        "options": {
            "temperature": 0.3,  # Lower temperature for more focused responses
            "max_tokens": 150    # Limit the response length
        }
    }
    
    print("\n--- Generating response with custom context ---")
    context_response = await ai_service.generate_response(prompt, context=context)
    print_ai_response(context_response)
    
    # Print cache statistics
    print("\n--- Cache Statistics ---")
    print(json.dumps(ai_service.cache.get_stats(), indent=2))
    
    # Shutdown the service
    ai_service.shutdown()

def print_ai_response(response: Dict[str, Any]):
    """Print the AI response in a formatted way."""
    print("\nResponse content:")
    print("-" * 50)
    print(response.get("content", "No content"))
    print("-" * 50)
    
    # Print metadata if available
    if "_metadata" in response:
        print("\nMetadata:")
        for key, value in response["_metadata"].items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())