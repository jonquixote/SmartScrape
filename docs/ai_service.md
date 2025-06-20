# AI Service Documentation

## Overview

The AI Service module provides a centralized system for optimizing AI model usage within SmartScrape. It manages AI interactions with multiple optimization techniques to reduce costs, improve reliability, and enhance performance.

## Architecture

The AI Service is built around several key components:

![AI Service Architecture](../static/ai_service_architecture.png)

### Core Components

1. **AIService**: The central service interface that manages all AI interactions
2. **AI Models**: Adapters for various LLM providers (OpenAI, Anthropic, Google)
3. **AI Cache**: Advanced caching with context-awareness to reduce duplicate requests
4. **Content Processor**: Pre-processes content to reduce token usage
5. **Model Selector**: Intelligently selects the optimal model for each task
6. **Batch Processor**: Batches compatible requests for efficiency
7. **Rule Engine**: Provides non-AI alternatives for common tasks

### Integration with Service Registry

The AI Service is registered with the core Service Registry, making it accessible to all other components:

```python
from core.service_registry import ServiceRegistry
from core.ai_service import AIService

# Get registry
registry = ServiceRegistry()

# Register AI service
registry.register_service_class(AIService)

# Get service instance with configuration
ai_service = registry.get_service("ai_service", config=ai_config)

# Use the service
response = await ai_service.generate_response("What is machine learning?")
```

## Configuration Options

The AI Service accepts a configuration dictionary with the following options:

### Basic Configuration

```python
ai_config = {
    "default_model": "gpt-3.5-turbo",  # Default model ID
    "models": [
        {
            "name": "gpt-3.5-turbo",   # Model identifier
            "type": "openai",          # Provider type
            "api_key": "sk-...",       # API key
            "model_id": "gpt-3.5-turbo" # Provider-specific model ID
        },
        {
            "name": "claude",
            "type": "anthropic",
            "api_key": "sk-...",
            "model_id": "claude-2"
        }
    ],
    "cache": {
        "backend": "memory",           # Cache backend (memory, disk, redis)
        "default_ttl": 3600,           # Default TTL in seconds
        "max_size": 1000               # Maximum cache entries
    },
    "use_rules": true,                 # Enable rule-based alternatives
    "allow_batching": true,            # Enable request batching 
    "log_usage": true                  # Log token usage for cost tracking
}
```

### Advanced Options

```python
ai_config = {
    # ... basic options ...
    "content_processing": {
        "html_extractors": ["main_content", "article"],
        "default_max_tokens": 4000,
        "summarize_long_content": true,
        "remove_boilerplate": true
    },
    "model_selection": {
        "cost_priority": 5,            # 1-10 (higher = more cost-sensitive)
        "quality_priority": 5,         # 1-10 (higher = prefer better models)
        "speed_priority": 5,           # 1-10 (higher = prefer faster models)
        "auto_upgrade_on_length": true # Upgrade model for long content
    },
    "batching": {
        "max_batch_size": 10,
        "max_waiting_time": 0.5,       # Max seconds to wait for batch
        "max_concurrent_batches": 5
    },
    "error_handling": {
        "max_retries": 3,
        "retry_delay": 1.0,            # Seconds between retries
        "fallback_models": ["gpt-3.5-turbo"],
        "timeout": 30.0                # Request timeout in seconds
    }
}
```

## Main Features

### 1. Context-Aware Caching

The AICache system provides intelligent caching of AI responses based on:

- Normalized prompt text
- Task-specific context
- Model used for generation

```python
# Cache will be automatically used
response1 = await ai_service.generate_response(
    "What is machine learning?",
    context={"task_type": "qa"}
)

# This will use cached result (much faster, no API call)
response2 = await ai_service.generate_response(
    "What is machine learning?",
    context={"task_type": "qa"}
)

# Different context = different cache key
response3 = await ai_service.generate_response(
    "What is machine learning?",
    context={"task_type": "summarization"}
)
```

### 2. Content Preprocessing

The ContentProcessor optimizes content before sending to AI models:

- HTML cleaning and extraction of relevant content
- Long text summarization and chunking
- Token count estimation
- Structured data formatting

```python
# HTML content will be preprocessed automatically
response = await ai_service.generate_response(
    html_content,
    context={
        "content_type": "html",
        "preprocess": True,
        "max_tokens": 2000
    }
)

# Long text will be summarized if needed
response = await ai_service.generate_response(
    long_text,
    context={
        "content_type": "long_text",
        "summarize": True
    }
)
```

### 3. Intelligent Model Selection

The ModelSelector chooses the best model for each request based on:

- Task requirements
- Content length
- Quality vs. cost priorities
- Required model capabilities

```python
# Let the system choose the best model
response = await ai_service.generate_response(
    "Explain quantum computing",
    context={
        "task_type": "explanation",
        "quality_priority": 8,  # Prefer higher quality for explanations
        "cost_priority": 3      # Less concerned about cost
    }
)

# Get model suggestions without making an API call
suggestions = ai_service.model_selector.suggest_models(
    "Perform complex reasoning about this financial data",
    content_length=len(financial_data)
)
```

### 4. Request Batching

The BatchProcessor combines compatible requests to reduce API calls:

- Groups requests by model and task type
- Preserves request priorities
- Handles response distribution

```python
# Enable batching for these requests
tasks = []
for question in questions:
    task = asyncio.create_task(
        ai_service.generate_response(
            question,
            context={
                "task_type": "qa",
                "use_batching": True,
                "batch_group": "qa_batch"
            }
        )
    )
    tasks.append(task)

# Requests will be batched behind the scenes
responses = await asyncio.gather(*tasks)
```

### 5. Rule-Based Alternatives

The RuleEngine provides non-AI alternatives for certain tasks:

- Extracting structured data (emails, prices, dates)
- Simple classification tasks
- Common question answering

```python
# Use rules for simple tasks
response = await ai_service.generate_response(
    "What is the capital of France?",
    context={
        "task_type": "qa",
        "use_rules": True
    }
)

# The response will include metadata if a rule was applied
if "rule_applied" in response:
    print(f"Answered using rule: {response['rule_applied']}")
```

## Cost Management

### Token Usage Tracking

Every response includes metadata about token usage:

```python
response = await ai_service.generate_response("Hello world")
metadata = response["_metadata"]

print(f"Input tokens: {metadata['input_tokens']}")
print(f"Output tokens: {metadata['output_tokens']}")
print(f"Total tokens: {metadata['total_tokens']}")
print(f"Estimated cost: ${metadata['estimated_cost']:.5f}")
```

### Cost Optimization Strategies

1. **Caching**: Eliminate duplicate API calls
2. **Content Preprocessing**: Reduce token count of inputs
3. **Model Selection**: Use cheaper models when appropriate
4. **Rule Engine**: Avoid API calls for simple tasks
5. **Batching**: Combine multiple prompts into fewer API calls

### Usage Reports

Track AI usage and costs with built-in reporting:

```python
# Get usage stats
usage_stats = ai_service.get_usage_stats()

print(f"Total requests: {usage_stats['total_requests']}")
print(f"Cached responses: {usage_stats['cache_hits']}")
print(f"Rule-based responses: {usage_stats['rule_hits']}")
print(f"API calls: {usage_stats['api_calls']}")
print(f"Total tokens: {usage_stats['total_tokens']}")
print(f"Estimated cost: ${usage_stats['estimated_cost']:.2f}")
```

## Performance Optimization

### 1. Parallel Processing

Process multiple AI requests in parallel:

```python
async def process_documents(documents):
    tasks = []
    for doc in documents:
        task = asyncio.create_task(
            ai_service.generate_response(
                doc,
                context={"task_type": "summarization"}
            )
        )
        tasks.append(task)
    
    return await asyncio.gather(*tasks)
```

### 2. Streaming Responses

Get partial responses as they're generated:

```python
async def process_with_streaming():
    async for chunk in ai_service.generate_streaming_response(
        "Write a long essay about climate change",
        context={"task_type": "content_generation"}
    ):
        print(chunk, end="", flush=True)
```

### 3. Progressive Enhancement

Start with fast models and upgrade if needed:

```python
# Start with a basic model
response = await ai_service.generate_response(
    "Analyze this data",
    context={
        "task_type": "analysis",
        "progressive_enhancement": True,
        "quality_threshold": 0.7
    }
)
```

## Error Handling

### 1. Automatic Retries

The service handles transient errors automatically:

```python
# Configure retry behavior
response = await ai_service.generate_response(
    "Important prompt",
    context={
        "retry_count": 3,
        "retry_delay": 1.0,
        "exponential_backoff": True
    }
)
```

### 2. Model Fallbacks

Configure fallback models for reliability:

```python
# Set fallback models for critical requests
response = await ai_service.generate_response(
    "Critical request",
    context={
        "fallback_models": ["gpt-3.5-turbo", "claude-instant"]
    }
)
```

### 3. Graceful Degradation

Apply rule-based alternatives when AI fails:

```python
response = await ai_service.generate_response(
    "What's the capital of Spain?",
    context={
        "use_rules": True,
        "fallback_to_rules": True
    }
)
```

## API Reference

### AIService

#### generate_response

```python
async def generate_response(
    self, 
    prompt: str, 
    context: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    use_cache: bool = True,
    batch_id: Optional[str] = None
) -> Dict[str, Any]
```

Generate a response from an AI model with automatic optimization.

**Parameters:**
- `prompt`: The input text to send to the AI model
- `context`: Additional context that affects processing
- `model_name`: Specific model to use (overrides automatic selection)
- `use_cache`: Whether to check cache before generating
- `batch_id`: Optional batch identifier for grouping requests

**Returns:**
- Dictionary containing:
  - `content`: The generated text
  - `_metadata`: Usage statistics and processing info

#### generate_streaming_response

```python
async def generate_streaming_response(
    self,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None
) -> AsyncIterator[str]
```

Generate a streaming response for real-time output.

**Parameters:**
- Same as `generate_response`

**Returns:**
- Async iterator yielding response chunks as they're generated

#### estimate_tokens

```python
def estimate_tokens(
    self,
    text: str,
    model_name: Optional[str] = None
) -> int
```

Estimate the number of tokens in a text.

**Parameters:**
- `text`: The text to tokenize
- `model_name`: Model to use for tokenization

**Returns:**
- Estimated token count

#### estimate_cost

```python
def estimate_cost(
    self,
    input_text: str,
    output_length: int,
    model_name: Optional[str] = None
) -> float
```

Estimate the cost of a request.

**Parameters:**
- `input_text`: The input prompt
- `output_length`: Expected output length in tokens
- `model_name`: Model to use

**Returns:**
- Estimated cost in USD

## Troubleshooting

### Common Issues

#### Rate Limiting

```
Error generating response: Rate limit exceeded
```

**Solution:** Use request batching and implement exponential backoff.

```python
context = {
    "retry_count": 5,
    "retry_delay": 2.0,
    "exponential_backoff": True,
    "use_batching": True
}
```

#### Context Length Exceeded

```
Error generating response: Maximum context length exceeded
```

**Solution:** Enable content preprocessing and chunking.

```python
context = {
    "content_type": "long_text",
    "preprocess": True,
    "chunk_content": True,
    "max_tokens": 4000
}
```

#### High Costs

**Solution:** Implement multiple cost-saving features.

```python
context = {
    "use_cache": True,
    "use_rules": True,
    "cost_priority": 8,  # Prefer cheaper models
    "preprocess": True,  # Reduce token count
    "use_batching": True # Combine requests
}
```

### Logging

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

### Basic Usage

```python
from core.service_registry import ServiceRegistry
from core.ai_service import AIService

# Configure the service
ai_config = {
    "default_model": "gpt-3.5-turbo",
    "models": [
        {
            "name": "gpt-3.5-turbo",
            "type": "openai",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "model_id": "gpt-3.5-turbo"
        }
    ],
    "cache": {
        "backend": "memory",
        "default_ttl": 3600
    }
}

# Get the service
registry = ServiceRegistry()
registry.register_service_class(AIService)
ai_service = registry.get_service("ai_service", ai_config)

# Use the service
async def main():
    response = await ai_service.generate_response(
        "What is machine learning?",
        context={"task_type": "qa"}
    )
    print(response["content"])

# Run the example
asyncio.run(main())
```

### Advanced Usage

```python
async def process_documents(documents, task_type):
    """Process multiple documents using optimal settings."""
    
    results = []
    for doc in documents:
        # Preprocess to determine characteristics
        doc_length = len(doc)
        
        # Adjust context based on document
        context = {
            "task_type": task_type,
            "content_type": "long_text" if doc_length > 1000 else "text",
            "preprocess": doc_length > 500,
            "use_rules": True,
            "use_cache": True,
            "use_batching": True,
            "quality_priority": 7 if task_type == "summarization" else 5
        }
        
        # Process the document
        response = await ai_service.generate_response(doc, context=context)
        results.append({
            "content": response["content"],
            "metadata": response.get("_metadata", {})
        })
    
    return results
```

## Performance Metrics

Based on our testing, the AI Service optimization techniques provide significant improvements:

| Optimization | Typical Cost Reduction | Response Time Improvement |
|--------------|------------------------|---------------------------|
| Caching | 20-40% | 90-99% for cache hits |
| Content Preprocessing | 10-30% | 5-15% |
| Model Selection | 15-35% | Varies |
| Rule Engine | 5-15% | 80-95% for rule hits |
| Batching | 10-25% | Varies |
| **Combined** | **30-60%** | **Varies by use case** |

## Future Improvements

Planned enhancements for the AI Service:

1. **Enhanced Rule Engine**: More sophisticated rules for common tasks
2. **Distributed Caching**: Redis/Memcached integration for horizontal scaling
3. **Function Calling**: Automatic function integration for structured outputs
4. **Adaptive Learning**: Automatic adjustment of strategies based on performance
5. **Cost Budgeting**: Implement budget limits and alerts