#!/usr/bin/env python3
"""
Advanced example demonstrating content preprocessing and caching features
of the AI service optimization components.
"""

import asyncio
import os
import logging
import time
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the necessary components
from core.ai_service import AIService
from core.content_processor import ContentProcessor
from core.service_registry import ServiceRegistry

# Sample HTML content for demonstration
SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Sample Article Page</title>
    <meta name="description" content="This is a sample article for testing content processing">
</head>
<body>
    <header>
        <nav>
            <ul>
                <li><a href="/">Home</a></li>
                <li><a href="/about">About</a></li>
                <li><a href="/contact">Contact</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        <article>
            <h1>Understanding AI Service Optimization</h1>
            
            <p>AI service optimization is a critical aspect of building scalable, efficient AI-powered applications.
            As AI becomes more integrated into everyday tools, optimizing these services becomes increasingly important.</p>
            
            <h2>Key Optimization Strategies</h2>
            
            <p>There are several key strategies for optimizing AI services:</p>
            
            <ul>
                <li><strong>Caching responses</strong> to reduce API calls and latency</li>
                <li><strong>Content preprocessing</strong> to minimize token usage</li>
                <li><strong>Intelligent model selection</strong> to balance cost and performance</li>
                <li><strong>Batch processing</strong> to optimize request patterns</li>
                <li><strong>Rule-based alternatives</strong> for common patterns</li>
            </ul>
            
            <p>When implemented properly, these strategies can significantly reduce costs, improve performance,
            and enhance the user experience of AI-powered applications.</p>
            
            <h2>Implementation Considerations</h2>
            
            <p>While implementing these optimization strategies, it's important to consider the trade-offs
            between processing time, memory usage, and accuracy. Different use cases may require different
            optimization priorities.</p>
        </article>
    </main>
    
    <footer>
        <p>&copy; 2025 Example Company</p>
        <nav>
            <ul>
                <li><a href="/privacy">Privacy Policy</a></li>
                <li><a href="/terms">Terms of Service</a></li>
            </ul>
        </nav>
    </footer>
</body>
</html>
"""

LONG_TEXT = """
AI optimization involves multiple techniques to improve performance and reduce costs.
First, caching mechanisms store frequent responses to avoid unnecessary API calls.
Second, preprocessing filters and condenses input content to minimize token usage.
Third, batching combines multiple requests into one to reduce overhead.
Fourth, rule-based alternatives handle simple tasks without using AI.
Fifth, model selection chooses the appropriate model based on complexity and cost.
Sixth, context awareness ensures minimal but sufficient context is provided.
Seventh, asynchronous processing avoids blocking operations.
Eighth, error handling and retry strategies ensure reliability.
Ninth, rate limiting prevents exceeding API quotas.
Tenth, performance monitoring provides insights for further optimization.
When these techniques are combined, they create a robust optimization system.
The benefits include lower costs, faster response times, and improved reliability.
Implementing these optimizations requires careful planning and architecture.
However, the long-term benefits far outweigh the initial investment.
"""

async def demonstrate_content_preprocessing():
    """Demonstrate the content preprocessing capabilities."""
    print("\n=== Content Preprocessing Demonstration ===")
    
    # Initialize the content processor
    processor = ContentProcessor()
    
    # Process HTML content
    print("\n1. HTML Content Extraction")
    print("Original HTML length:", len(SAMPLE_HTML), "characters")
    
    # Extract main content from HTML
    extracted_html = processor.preprocess_html(SAMPLE_HTML, extract_main=True)
    print("Extracted main content length:", len(extracted_html), "characters")
    print("Extracted content preview:")
    print("-" * 50)
    print(extracted_html[:300] + "..." if len(extracted_html) > 300 else extracted_html)
    print("-" * 50)
    
    # Process long text content
    print("\n2. Long Text Processing")
    print("Original text length:", len(LONG_TEXT), "characters")
    
    # Chunk the content
    chunks = processor.chunk_content(LONG_TEXT, max_tokens=200)
    print(f"Split into {len(chunks)} chunks")
    
    # Summarize the content
    summary = processor.summarize_content(LONG_TEXT, ratio=0.3)
    print("Summarized content length:", len(summary), "characters")
    print("Summary preview:")
    print("-" * 50)
    print(summary[:300] + "..." if len(summary) > 300 else summary)
    print("-" * 50)

async def demonstrate_advanced_caching(api_key: str = None):
    """Demonstrate the advanced caching capabilities."""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("No API key provided. Skipping AI service demonstration.")
            return
    
    print("\n=== Advanced Caching Demonstration ===")
    
    # Initialize the service registry
    registry = ServiceRegistry()
    
    # Create and configure the AI service
    ai_service = AIService()
    registry.register_service(ai_service)
    
    # Configure with both memory and disk cache options
    api_config = {
        "models": [
            {
                "name": "default",
                "type": "openai",
                "model_id": "gpt-3.5-turbo",
                "api_key": api_key
            }
        ],
        "default_model": "default",
        "cache": {
            "backend": "memory",  # Options: "memory", "disk"
            "default_ttl": 3600   # 1 hour default TTL
        }
    }
    
    # Initialize the service
    ai_service.initialize(api_config)
    
    # Create a test prompt
    prompt = "What are three key benefits of AI service optimization?"
    
    # First request (cache miss)
    print("\n1. First request (expect cache miss)")
    start_time = time.time()
    response1 = await ai_service.generate_response(prompt)
    elapsed1 = time.time() - start_time
    print(f"Response time: {elapsed1:.2f} seconds")
    print(f"Content: {response1.get('content', '')[:100]}...")
    
    # Cache statistics after first request
    print("\nCache statistics after first request:")
    print(f"Hits: {ai_service.cache.stats['hits']}")
    print(f"Misses: {ai_service.cache.stats['misses']}")
    print(f"Sets: {ai_service.cache.stats['sets']}")
    
    # Second request with same prompt (cache hit)
    print("\n2. Second request with same prompt (expect cache hit)")
    start_time = time.time()
    response2 = await ai_service.generate_response(prompt)
    elapsed2 = time.time() - start_time
    print(f"Response time: {elapsed2:.2f} seconds")
    print(f"Content: {response2.get('content', '')[:100]}...")
    
    # Cache statistics after second request
    print("\nCache statistics after second request:")
    print(f"Hits: {ai_service.cache.stats['hits']}")
    print(f"Misses: {ai_service.cache.stats['misses']}")
    print(f"Sets: {ai_service.cache.stats['sets']}")
    
    # Speed comparison
    if elapsed1 > 0:
        print(f"\nSpeed improvement: {elapsed1/elapsed2:.1f}x faster with caching")
    
    # Different context but same prompt (should use cache but with context-specific key)
    print("\n3. Same prompt with different context (should be cache miss)")
    context = {"task_type": "list", "options": {"temperature": 0.2}}
    start_time = time.time()
    response3 = await ai_service.generate_response(prompt, context=context)
    elapsed3 = time.time() - start_time
    print(f"Response time: {elapsed3:.2f} seconds")
    print(f"Content: {response3.get('content', '')[:100]}...")
    
    # Final cache statistics
    print("\nFinal cache statistics:")
    print(f"Hits: {ai_service.cache.stats['hits']}")
    print(f"Misses: {ai_service.cache.stats['misses']}")
    print(f"Sets: {ai_service.cache.stats['sets']}")
    
    # Shutdown the service
    ai_service.shutdown()

async def demonstrate_html_preprocessing_with_ai(api_key: str = None):
    """Demonstrate HTML preprocessing with AI response generation."""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("No API key provided. Skipping AI with preprocessing demonstration.")
            return
    
    print("\n=== HTML Preprocessing with AI Response Generation ===")
    
    # Initialize the service registry and AI service
    registry = ServiceRegistry()
    ai_service = AIService()
    registry.register_service(ai_service)
    
    # Configure the service
    api_config = {
        "models": [
            {
                "name": "default",
                "type": "openai",
                "model_id": "gpt-3.5-turbo",
                "api_key": api_key
            }
        ],
        "default_model": "default",
        "cache": {"backend": "memory"}
    }
    ai_service.initialize(api_config)
    
    # Generate a prompt based on raw HTML
    prompt = "Summarize the main points of this article in 3 bullet points:"
    
    # First try without preprocessing
    print("\n1. AI response without preprocessing")
    full_prompt = f"{prompt}\n\n{SAMPLE_HTML}"
    start_time = time.time()
    response1 = await ai_service.generate_response(
        full_prompt, 
        use_cache=False
    )
    elapsed1 = time.time() - start_time
    
    # Print results
    print(f"Response time: {elapsed1:.2f} seconds")
    print("Token usage without preprocessing:")
    if "_metadata" in response1:
        print(f"  Input tokens: {response1['_metadata'].get('input_tokens', 'N/A')}")
        print(f"  Output tokens: {response1['_metadata'].get('output_tokens', 'N/A')}")
        print(f"  Estimated cost: ${response1['_metadata'].get('total_cost', 'N/A')}")
    print("\nResponse content:")
    print(response1.get("content", ""))
    
    # Now try with preprocessing
    print("\n2. AI response with preprocessing")
    # Use context to specify content preprocessing
    context = {
        "content_type": "html",
        "preprocess": True,
        "extract_main": True,
        "options": {"temperature": 0.7}
    }
    
    start_time = time.time()
    response2 = await ai_service.generate_response(
        f"{prompt}\n\n{SAMPLE_HTML}", 
        context=context,
        use_cache=False
    )
    elapsed2 = time.time() - start_time
    
    # Print results
    print(f"Response time: {elapsed2:.2f} seconds")
    print("Token usage with preprocessing:")
    if "_metadata" in response2:
        print(f"  Input tokens: {response2['_metadata'].get('input_tokens', 'N/A')}")
        print(f"  Output tokens: {response2['_metadata'].get('output_tokens', 'N/A')}")
        print(f"  Estimated cost: ${response2['_metadata'].get('total_cost', 'N/A')}")
    print("\nResponse content:")
    print(response2.get("content", ""))
    
    # Compare token usage and cost
    if "_metadata" in response1 and "_metadata" in response2:
        tokens1 = response1["_metadata"].get("input_tokens", 0)
        tokens2 = response2["_metadata"].get("input_tokens", 0)
        if tokens1 > 0 and tokens2 > 0:
            reduction = (tokens1 - tokens2) / tokens1 * 100
            print(f"\nToken reduction: {reduction:.1f}%")
        
        cost1 = response1["_metadata"].get("total_cost", 0)
        cost2 = response2["_metadata"].get("total_cost", 0)
        if cost1 > 0 and cost2 > 0:
            cost_reduction = (cost1 - cost2) / cost1 * 100
            print(f"Cost reduction: {cost_reduction:.1f}%")
    
    # Shutdown the service
    ai_service.shutdown()

async def main():
    # Get API key from environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Demonstrate content preprocessing
    await demonstrate_content_preprocessing()
    
    # Demonstrate advanced caching (if API key is available)
    if api_key:
        await demonstrate_advanced_caching(api_key)
        await demonstrate_html_preprocessing_with_ai(api_key)
    else:
        print("\nNo OpenAI API key found in environment variables.")
        print("Set OPENAI_API_KEY to run the complete demonstration.")
        print("Skipping AI service demonstrations that require API access.")

if __name__ == "__main__":
    asyncio.run(main())