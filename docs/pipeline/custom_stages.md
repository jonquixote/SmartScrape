# Creating Custom Pipeline Stages

This guide explains how to create custom pipeline stages for the SmartScrape Pipeline Architecture. It covers the stage interface, lifecycle hooks, best practices for context interaction, error handling, testing, and performance optimization.

## Pipeline Stage Fundamentals

### The PipelineStage Interface

All pipeline stages must implement the `PipelineStage` interface by inheriting from the base class:

```python
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext

class MyCustomStage(PipelineStage):
    """My custom pipeline stage that does something useful."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Implement the core processing logic for this stage.
        
        Args:
            context: The pipeline context containing shared data
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        # Your implementation here
        return True
```

The `process()` method is the only required method you must implement. It should:

1. Take a `PipelineContext` as input
2. Perform some processing on the data in the context
3. Return `True` if processing was successful, or `False` if it failed

### Basic Stage Structure

A complete custom stage typically follows this structure:

```python
class CustomDataProcessingStage(PipelineStage):
    """A stage that processes data in a specific way."""
    
    def __init__(self, config=None):
        """Initialize the stage with configuration.
        
        Args:
            config: Configuration dictionary for this stage
        """
        super().__init__(config)
        # Initialize any instance variables needed
        self.specific_setting = self.config.get("specific_setting", "default_value")
        
    def validate_input(self, context: PipelineContext) -> bool:
        """Validate that the context contains required inputs.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if not context.get("required_input_key"):
            context.add_error(self.name, "Missing required input: required_input_key")
            return False
        return True
        
    async def process(self, context: PipelineContext) -> bool:
        """Process the input data from the context.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        # Get input from context
        input_data = context.get("required_input_key")
        
        # Process the data
        result = self._process_data(input_data)
        
        # Store the result in the context
        context.set("output_key", result)
        
        return True
        
    def _process_data(self, data):
        """Internal helper method for data processing.
        
        Args:
            data: The data to process
            
        Returns:
            The processed data
        """
        # Implementation details
        return processed_data
        
    def get_config_schema(self) -> dict:
        """Get the JSON schema for this stage's configuration.
        
        Returns:
            dict: The JSON schema describing valid configuration
        """
        return {
            "type": "object",
            "properties": {
                "specific_setting": {
                    "type": "string",
                    "description": "A specific setting for this stage"
                }
            }
        }
```

## Stage Lifecycle and Hooks

The full lifecycle of a pipeline stage includes these potential hooks:

### 1. Initialization

The `__init__` method is called when the stage is instantiated:

```python
def __init__(self, config=None):
    super().__init__(config)
    # Initialize resources, parse configuration
```

### 2. Input Validation

The `validate_input` method is called before `process` to verify inputs:

```python
def validate_input(self, context: PipelineContext) -> bool:
    # Check for required inputs
    # Return False if validation fails
    return True
```

### 3. Processing

The `process` method contains the main logic of the stage:

```python
async def process(self, context: PipelineContext) -> bool:
    # Main processing logic
    return True
```

### 4. Error Handling

The `handle_error` method is called if an exception occurs during processing:

```python
def handle_error(self, context: PipelineContext, error: Exception) -> bool:
    # Handle or log the error
    # Return True if error was handled, False to re-raise
    return False
```

### 5. Metadata

The `get_metadata` method provides information about the stage:

```python
def get_metadata(self) -> dict:
    # Return information about this stage
    # Usually no need to override this
    return {
        "name": self.name,
        "description": self.__doc__,
        "config_schema": self.get_config_schema()
    }
```

### 6. Configuration Schema

The `get_config_schema` method defines the configuration schema:

```python
def get_config_schema(self) -> dict:
    # Return JSON schema for this stage's configuration
    return {}
```

## Context Interaction Patterns

### Reading from Context

To get data from the context:

```python
async def process(self, context: PipelineContext) -> bool:
    # Get a value with a default if it doesn't exist
    value = context.get("key", "default_value")
    
    # Check if a key exists
    if "some_key" in context.data:
        # Do something
        pass
    
    # Get nested data using dot notation (if you implement this)
    nested_value = context.get("parent.child.value")
    
    return True
```

### Writing to Context

To write data to the context:

```python
async def process(self, context: PipelineContext) -> bool:
    # Set a single value
    context.set("output_key", "output_value")
    
    # Update multiple values at once
    context.update({
        "key1": "value1",
        "key2": "value2"
    })
    
    # Create a structured output
    context.set("extraction_result", {
        "title": "Extracted Title",
        "content": "Extracted Content",
        "metadata": {
            "source": "webpage",
            "timestamp": "2025-05-06T12:00:00Z"
        }
    })
    
    return True
```

### Context Namespace Conventions

To avoid key collisions in the context, follow these naming conventions:

1. Use stage-specific prefixes for temporary data:
   ```python
   context.set(f"{self.name}_temp_data", temp_data)
   ```

2. Use standardized keys for common data types:
   ```python
   # Input data
   "url" - Source URL
   "html_content" - Raw HTML content
   "text_content" - Text content
   
   # Processed data
   "extracted_data" - Main extraction result
   "extracted_metadata" - Metadata from extraction
   "normalized_data" - Normalized data
   ```

3. Use structured data for complex outputs:
   ```python
   context.set("product", {
       "name": "Product Name",
       "price": 12.99,
       "features": ["feature1", "feature2"]
   })
   ```

### Context Cleanup

For memory efficiency, consider cleaning up temporary data:

```python
async def process(self, context: PipelineContext) -> bool:
    # Process data
    large_input = context.get("large_input_data")
    result = process_large_data(large_input)
    context.set("processed_result", result)
    
    # Clean up large temporary data if it's no longer needed
    if self.config.get("cleanup_temp_data", True):
        if "large_input_data" in context.data:
            del context.data["large_input_data"]
    
    return True
```

## Error Handling Best Practices

### 1. Use the Built-in Error Tracking

Record errors in the context:

```python
async def process(self, context: PipelineContext) -> bool:
    try:
        result = some_operation_that_might_fail()
    except Exception as e:
        context.add_error(self.name, f"Operation failed: {str(e)}")
        return False
        
    context.set("result", result)
    return True
```

### 2. Implement the Error Handler

Override the `handle_error` method for custom error handling:

```python
def handle_error(self, context: PipelineContext, error: Exception) -> bool:
    if isinstance(error, ConnectionError):
        # Handle connection errors
        context.add_error(self.name, f"Connection failed: {str(error)}")
        context.set("connection_status", "failed")
        return True  # Error handled, don't re-raise
    
    # For other errors, add to context but allow re-raising
    context.add_error(self.name, f"Unexpected error: {str(error)}")
    return False  # Not handled, pipeline will re-raise
```

### 3. Implement Partial Success

For operations where partial success is acceptable:

```python
async def process(self, context: PipelineContext) -> bool:
    items = context.get("items", [])
    success_count = 0
    failure_count = 0
    
    results = []
    for item in items:
        try:
            result = process_item(item)
            results.append(result)
            success_count += 1
        except Exception as e:
            context.add_error(self.name, f"Item processing failed: {str(e)}")
            results.append(None)  # Placeholder for failed item
            failure_count += 1
    
    context.set("processed_results", results)
    context.set("processing_stats", {
        "total": len(items),
        "success": success_count,
        "failure": failure_count
    })
    
    # Return True if at least some items succeeded
    return success_count > 0
```

### 4. Use Try-Except in Validation

Add error handling to validation to prevent crashes:

```python
def validate_input(self, context: PipelineContext) -> bool:
    try:
        # Validate that the input is a valid JSON object
        input_data = context.get("input_data")
        if not isinstance(input_data, dict):
            context.add_error(self.name, "Input data must be a JSON object")
            return False
            
        # Validate required fields
        if "required_field" not in input_data:
            context.add_error(self.name, "Missing required field: required_field")
            return False
            
        return True
    except Exception as e:
        context.add_error(self.name, f"Validation error: {str(e)}")
        return False
```

### 5. Implement Retry Logic

For operations that might succeed with retries:

```python
async def process(self, context: PipelineContext) -> bool:
    max_retries = self.config.get("max_retries", 3)
    retry_delay = self.config.get("retry_delay", 1.0)
    
    for attempt in range(max_retries):
        try:
            result = await self._perform_operation(context)
            context.set("result", result)
            return True
        except TransientError as e:
            # This is a temporary error, we can retry
            if attempt < max_retries - 1:
                context.set("retry_count", attempt + 1)
                await asyncio.sleep(retry_delay)
            else:
                # Max retries exceeded
                context.add_error(self.name, f"Operation failed after {max_retries} attempts: {str(e)}")
                return False
        except Exception as e:
            # Non-transient error, don't retry
            context.add_error(self.name, f"Operation failed: {str(e)}")
            return False
    
    return False
```

## Testing Custom Stages

### 1. Unit Testing Stages

Create unit tests for your custom stages:

```python
import unittest
import asyncio
from core.pipeline.context import PipelineContext
from your_module import YourCustomStage

class TestYourCustomStage(unittest.TestCase):
    def setUp(self):
        # Create a stage instance for testing
        self.stage = YourCustomStage({"test_config": "test_value"})
        
    def test_validate_input_with_valid_data(self):
        # Create a context with valid input
        context = PipelineContext({"required_input": "valid_value"})
        
        # Test validation
        result = self.stage.validate_input(context)
        self.assertTrue(result)
        self.assertFalse(context.has_errors())
        
    def test_validate_input_with_invalid_data(self):
        # Create a context with invalid input
        context = PipelineContext({})  # Missing required input
        
        # Test validation
        result = self.stage.validate_input(context)
        self.assertFalse(result)
        self.assertTrue(context.has_errors())
        
    def test_process_successful(self):
        # Create a context with valid input
        context = PipelineContext({"required_input": "test_data"})
        
        # Run the process method
        result = asyncio.run(self.stage.process(context))
        
        # Check the result
        self.assertTrue(result)
        self.assertEqual(context.get("expected_output"), "expected_value")
        
    def test_process_error_handling(self):
        # Create a context that will cause an error
        context = PipelineContext({"input_that_causes_error": True})
        
        # Run the process method, which should handle the error
        result = asyncio.run(self.stage.process(context))
        
        # Check the result
        self.assertFalse(result)
        self.assertTrue(context.has_errors())
```

### 2. Integration Testing

Test your stage within a pipeline:

```python
def test_stage_in_pipeline(self):
    # Create a pipeline with your stage
    pipeline = Pipeline("test_pipeline")
    pipeline.add_stage(InputStage({"test_input": "test_value"}))
    pipeline.add_stage(self.stage)
    pipeline.add_stage(OutputStage())
    
    # Execute the pipeline
    context = asyncio.run(pipeline.execute())
    
    # Check the pipeline execution
    self.assertFalse(context.has_errors())
    self.assertEqual(context.get("final_output"), "expected_pipeline_output")
```

### 3. Mock External Dependencies

Use mocking to test stages with external dependencies:

```python
from unittest.mock import patch, MagicMock

def test_http_stage_with_mock(self):
    # Create a context
    context = PipelineContext({"url": "https://example.com"})
    
    # Create the stage
    http_stage = HttpInputStage()
    
    # Mock the requests library
    with patch('requests.get') as mock_get:
        # Configure the mock
        mock_response = MagicMock()
        mock_response.text = "<html><body>Test content</body></html>"
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Run the process method
        result = asyncio.run(http_stage.process(context))
        
        # Check the result
        self.assertTrue(result)
        self.assertEqual(context.get("html_content"), 
                        "<html><body>Test content</body></html>")
        
        # Verify the mock was called correctly
        mock_get.assert_called_once_with(
            "https://example.com", 
            headers=http_stage.config.get("headers"), 
            timeout=http_stage.config.get("timeout")
        )
```

### 4. Testing Error Conditions

Test how your stage handles various error conditions:

```python
def test_stage_with_invalid_input(self):
    # Test with invalid input
    context = PipelineContext({"invalid_input": "bad_value"})
    result = self.stage.validate_input(context)
    self.assertFalse(result)
    
def test_stage_with_processing_error(self):
    # Test processing error
    context = PipelineContext({"trigger_error": True})
    result = asyncio.run(self.stage.process(context))
    self.assertFalse(result)
    self.assertTrue(context.has_errors())
    self.assertIn("expected_error_message", 
                 context.metadata["errors"][self.stage.name][0])
```

### 5. Testing Configuration Options

Test different configuration options:

```python
def test_stage_with_different_configs(self):
    test_configs = [
        {"option1": "value1", "option2": "value2"},
        {"option1": "value3"},
        {}  # Empty config (should use defaults)
    ]
    
    for config in test_configs:
        with self.subTest(config=config):
            # Create stage with this config
            stage = YourCustomStage(config)
            
            # Create test context
            context = PipelineContext({"test_input": "test_value"})
            
            # Run the process method
            result = asyncio.run(stage.process(context))
            
            # Check results based on expected behavior for this config
            self.assertTrue(result)
            # Verify expected outputs based on config...
```

## Performance Considerations

### 1. Minimize Data Copying

Avoid unnecessary data copying:

```python
# Less efficient - copies the entire list
def process_inefficient(self, context: PipelineContext) -> bool:
    items = context.get("items", [])
    processed_items = items.copy()  # Unnecessary copy
    
    for i in range(len(processed_items)):
        processed_items[i] = self._process_item(processed_items[i])
        
    context.set("processed_items", processed_items)
    return True

# More efficient - create new list directly
def process_efficient(self, context: PipelineContext) -> bool:
    items = context.get("items", [])
    processed_items = [self._process_item(item) for item in items]
    context.set("processed_items", processed_items)
    return True
```

### 2. Use Asynchronous Operations Effectively

Take advantage of async/await for I/O-bound operations:

```python
async def process(self, context: PipelineContext) -> bool:
    urls = context.get("urls", [])
    
    # Process URLs concurrently with limited concurrency
    semaphore = asyncio.Semaphore(self.config.get("max_concurrent", 5))
    
    async def fetch_url(url):
        async with semaphore:
            # Asynchronous HTTP request
            response = await aiohttp.ClientSession().get(url)
            return await response.text()
    
    # Create tasks for all URLs
    tasks = [fetch_url(url) for url in urls]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    successful_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            context.add_error(self.name, f"Error fetching {urls[i]}: {str(result)}")
        else:
            successful_results.append(result)
    
    context.set("fetched_contents", successful_results)
    return len(successful_results) > 0
```

### 3. Implement Caching

Use caching for expensive operations:

```python
class CachingStage(PipelineStage):
    def __init__(self, config=None):
        super().__init__(config)
        self.cache = {}
        self.cache_size = self.config.get("cache_size", 100)
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour
        
    def _get_cache_key(self, data):
        # Create a cache key from the input data
        if isinstance(data, str):
            return hash(data)
        elif isinstance(data, dict):
            # Create a stable hash from dictionary
            return hash(frozenset(data.items()))
        else:
            return hash(str(data))
            
    async def process(self, context: PipelineContext) -> bool:
        input_data = context.get("input_data")
        cache_key = self._get_cache_key(input_data)
        
        # Check cache
        current_time = time.time()
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if current_time - cache_entry["timestamp"] < self.cache_ttl:
                # Cache hit
                context.set("output_data", cache_entry["data"])
                context.set("cache_hit", True)
                return True
        
        # Cache miss, process the data
        result = await self._process_data(input_data)
        
        # Update cache
        self.cache[cache_key] = {
            "data": result,
            "timestamp": current_time
        }
        
        # Limit cache size (simple LRU by evicting oldest entries)
        if len(self.cache) > self.cache_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        context.set("output_data", result)
        context.set("cache_hit", False)
        return True
        
    async def _process_data(self, data):
        # Expensive processing operation
        pass
```

### 4. Optimize Resource Usage

Be careful with resource-intensive operations:

```python
class MemoryEfficientStage(PipelineStage):
    async def process(self, context: PipelineContext) -> bool:
        large_data = context.get("large_data")
        
        # Process data in chunks to avoid memory spikes
        chunk_size = self.config.get("chunk_size", 1000)
        results = []
        
        for i in range(0, len(large_data), chunk_size):
            chunk = large_data[i:i+chunk_size]
            chunk_result = self._process_chunk(chunk)
            results.extend(chunk_result)
            
            # Give control back to event loop periodically
            await asyncio.sleep(0)
        
        context.set("processed_data", results)
        return True
```

### 5. Profile Your Stages

Use profiling to identify bottlenecks:

```python
class ProfilingStage(PipelineStage):
    async def process(self, context: PipelineContext) -> bool:
        import cProfile
        import pstats
        import io
        
        # Only profile in debug mode
        if self.config.get("enable_profiling", False):
            profiler = cProfile.Profile()
            profiler.enable()
            
        # Execute the actual processing
        result = await self._do_actual_processing(context)
        
        # Finish profiling
        if self.config.get("enable_profiling", False):
            profiler.disable()
            
            # Get profile stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Print top 20 lines
            
            # Store profile info in context
            context.set("profile_info", s.getvalue())
        
        return result
        
    async def _do_actual_processing(self, context):
        # Actual processing logic
        return True
```

## Advanced Stage Techniques

### 1. Composable Stages

Create stages that can be composed together:

```python
class ComposableStage(PipelineStage):
    def __init__(self, config=None):
        super().__init__(config)
        # Create sub-stages
        self.sub_stages = []
        
        # Create sub-stages from configuration
        for sub_config in self.config.get("stages", []):
            stage_type = sub_config.get("type")
            stage_config = sub_config.get("config", {})
            
            # Get stage class from registry or configuration
            stage_class = get_stage_class(stage_type)
            if stage_class:
                self.sub_stages.append(stage_class(stage_config))
        
    async def process(self, context: PipelineContext) -> bool:
        # Process each sub-stage
        for stage in self.sub_stages:
            if not await stage.process(context):
                return False
        return True
```

### 2. Dynamic Stage Configuration

Create stages that adjust their behavior based on context:

```python
class AdaptiveStage(PipelineStage):
    async def process(self, context: PipelineContext) -> bool:
        # Determine processing strategy based on input
        content_type = context.get("content_type")
        
        if content_type == "html":
            return await self._process_html(context)
        elif content_type == "json":
            return await self._process_json(context)
        elif content_type == "text":
            return await self._process_text(context)
        else:
            context.add_error(self.name, f"Unsupported content type: {content_type}")
            return False
            
    async def _process_html(self, context):
        # HTML-specific processing
        pass
        
    async def _process_json(self, context):
        # JSON-specific processing
        pass
        
    async def _process_text(self, context):
        # Text-specific processing
        pass
```

### 3. Pipeline Integration

Create stages that integrate with other pipelines:

```python
class SubPipelineStage(PipelineStage):
    async def process(self, context: PipelineContext) -> bool:
        # Create a sub-pipeline
        sub_pipeline = Pipeline("sub-pipeline")
        
        # Add stages based on configuration
        for stage_config in self.config.get("stages", []):
            # Create and add stage
            # ...
            
        # Extract relevant data from main context
        sub_context_data = {}
        for key in self.config.get("input_keys", []):
            if key in context.data:
                sub_context_data[key] = context.data[key]
        
        # Execute sub-pipeline
        sub_context = await sub_pipeline.execute(sub_context_data)
        
        # Check for errors
        if sub_context.has_errors():
            # Propagate errors to main context
            for source, messages in sub_context.metadata["errors"].items():
                for message in messages:
                    context.add_error(f"sub:{source}", message)
            return False
            
        # Merge results back to main context
        for key in self.config.get("output_keys", []):
            if key in sub_context.data:
                context.set(key, sub_context.data[key])
                
        return True
```

## Conclusion

This guide covered the fundamental aspects of creating custom pipeline stages in the SmartScrape Pipeline Architecture. By following these patterns and best practices, you can create reusable, maintainable, and efficient pipeline stages that extend the system's capabilities.

Remember to:
1. Follow the single responsibility principle
2. Handle errors gracefully
3. Document your stages clearly
4. Write comprehensive tests
5. Consider performance implications

For more examples and templates, see the `examples/pipelines/` directory.