# Pipeline Usage Guide

This guide provides step-by-step instructions for using the SmartScrape Pipeline Architecture to build, configure, and execute data processing pipelines.

## Getting Started

### Prerequisites

Before using the pipeline system, ensure you have:

- Python 3.8 or higher
- Basic understanding of asynchronous programming in Python
- Familiarity with the pipeline architecture concepts (see [architecture.md](architecture.md))

### Installation

The pipeline system is included in the core SmartScrape package. No additional installation is required if you're already using SmartScrape.

## Creating Simple Pipelines

### Basic Pipeline Creation

Here's how to create a simple pipeline with a few stages:

```python
from core.pipeline.pipeline import Pipeline
from core.pipeline.stages.input.http_input import HttpInputStage
from core.pipeline.stages.processing.html_processing import HtmlProcessingStage
from core.pipeline.stages.output.json_output import JsonOutputStage

# Create a new pipeline
pipeline = Pipeline("simple_extraction")

# Add stages to the pipeline
pipeline.add_stage(HttpInputStage({"url": "https://example.com"}))
pipeline.add_stage(HtmlProcessingStage({"remove_scripts": True}))
pipeline.add_stage(JsonOutputStage({"pretty_print": True}))
```

### Method Chaining

You can also use method chaining for a more concise syntax:

```python
pipeline = (
    Pipeline("simple_extraction")
    .add_stage(HttpInputStage({"url": "https://example.com"}))
    .add_stage(HtmlProcessingStage({"remove_scripts": True}))
    .add_stage(JsonOutputStage({"pretty_print": True}))
)
```

### Using the Pipeline Registry

For more dynamic pipeline creation, you can use the registry:

```python
from core.pipeline.registry import PipelineRegistry

# Get the pipeline registry
registry = PipelineRegistry()

# Create a pipeline using registered stage types
pipeline = registry.create_pipeline("simple_extraction")
pipeline.add_stage(registry.create_stage("http_input", {"url": "https://example.com"}))
pipeline.add_stage(registry.create_stage("html_processing", {"remove_scripts": True}))
pipeline.add_stage(registry.create_stage("json_output", {"pretty_print": True}))
```

### Creating from Configuration

For maximum flexibility, you can create pipelines from configuration files:

```python
import json
from core.pipeline.factory import PipelineFactory

# Load pipeline configuration
with open("pipeline_config.json", "r") as f:
    config = json.load(f)

# Create pipeline from configuration
factory = PipelineFactory(registry)
pipeline = factory.create_from_config(config)
```

Example configuration file (`pipeline_config.json`):

```json
{
  "name": "extraction_pipeline",
  "config": {
    "parallel_execution": false,
    "continue_on_error": true
  },
  "stages": [
    {
      "type": "http_input",
      "config": {
        "url": "https://example.com",
        "timeout": 10
      }
    },
    {
      "type": "html_processing",
      "config": {
        "remove_scripts": true,
        "remove_styles": true
      }
    },
    {
      "type": "json_output",
      "config": {
        "pretty_print": true
      }
    }
  ]
}
```

## Configuring Stages

### Stage Configuration Options

Each stage type has its own configuration options. Here are some common patterns:

#### Input Stage Configuration

```python
http_input_config = {
    "url": "https://example.com",
    "method": "GET",
    "headers": {
        "User-Agent": "SmartScrape/1.0"
    },
    "timeout": 10,
    "retry_count": 3,
    "retry_delay": 1.0
}

input_stage = HttpInputStage(http_input_config)
```

#### Processing Stage Configuration

```python
html_processing_config = {
    "remove_scripts": True,
    "remove_styles": True,
    "extract_main_content": True,
    "selector": "main.content",
    "encoding": "utf-8"
}

processing_stage = HtmlProcessingStage(html_processing_config)
```

#### Output Stage Configuration

```python
json_output_config = {
    "pretty_print": True,
    "include_metadata": False,
    "output_path": "results.json",
    "append": False
}

output_stage = JsonOutputStage(json_output_config)
```

### Configuration Inheritance

You can create stage configurations that inherit from default configurations:

```python
# Define default configuration
default_http_config = {
    "method": "GET",
    "timeout": 10,
    "retry_count": 3,
    "headers": {"User-Agent": "SmartScrape/1.0"}
}

# Create specific configuration with overrides
specific_config = {**default_http_config, "url": "https://example.com", "timeout": 20}

# Create stage with the combined configuration
input_stage = HttpInputStage(specific_config)
```

### Dynamic Configuration

You can also build configurations dynamically based on runtime conditions:

```python
def build_http_config(url, is_mobile=False):
    config = {
        "url": url,
        "timeout": 15,
        "retry_count": 3
    }
    
    if is_mobile:
        config["headers"] = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15"
        }
    else:
        config["headers"] = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
    return config

# Use the function to generate configuration
mobile_config = build_http_config("https://example.com", is_mobile=True)
input_stage = HttpInputStage(mobile_config)
```

## Executing Pipelines

### Basic Synchronous Execution

While pipelines are asynchronous internally, you can execute them synchronously:

```python
import asyncio

# Create and configure your pipeline
pipeline = Pipeline("extraction")
# ... add stages ...

# Execute the pipeline synchronously
context = asyncio.run(pipeline.execute())

# Access the results
result = context.get("extracted_data")
print(result)
```

### Asynchronous Execution

For better integration with other async code:

```python
async def process_url(url):
    pipeline = Pipeline("extraction")
    pipeline.add_stage(HttpInputStage({"url": url}))
    pipeline.add_stage(HtmlProcessingStage({"remove_scripts": True}))
    pipeline.add_stage(ContentExtractionStage())
    
    # Execute the pipeline
    context = await pipeline.execute()
    
    return context.get("extracted_data")

# Use in an async context
async def main():
    urls = ["https://example.com", "https://another-example.com"]
    tasks = [process_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    # Process results
    for url, result in zip(urls, results):
        print(f"Extracted data from {url}: {result}")

# Run the async main function
asyncio.run(main())
```

### Providing Initial Data

You can provide initial data to the pipeline:

```python
# Prepare initial data
initial_data = {
    "url": "https://example.com",
    "options": {
        "extract_images": True,
        "extract_links": True
    }
}

# Execute with initial data
context = asyncio.run(pipeline.execute(initial_data))
```

### Pipeline Execution Options

Configure pipeline execution behavior:

```python
# Configure execution options
pipeline = Pipeline(
    "configurable_pipeline",
    {
        "parallel_execution": True,  # Execute independent stages in parallel
        "max_workers": 5,            # Limit parallel execution
        "continue_on_error": True,   # Continue execution if a stage fails
        "timeout": 30,               # Overall pipeline timeout in seconds
        "enable_monitoring": True    # Collect execution metrics
    }
)
```

## Handling Results and Errors

### Accessing Results

After pipeline execution, you can access the results from the context:

```python
context = asyncio.run(pipeline.execute())

# Get the main result
result = context.get("extracted_data")

# Get optional results with default values
images = context.get("extracted_images", [])
links = context.get("extracted_links", [])
```

### Checking for Errors

Always check for errors after pipeline execution:

```python
context = asyncio.run(pipeline.execute())

if context.has_errors():
    # Handle errors
    errors = context.metadata["errors"]
    print(f"Pipeline execution had {len(errors)} errors:")
    
    for source, messages in errors.items():
        for message in messages:
            print(f"  {source}: {message}")
else:
    # Process successful results
    result = context.get("extracted_data")
    print("Extraction successful:", result)
```

### Error Recovery

You can implement error recovery by examining the context:

```python
context = asyncio.run(pipeline.execute())

# Check if a specific stage failed
if "ContentExtractionStage" in context.metadata["errors"]:
    # Try a fallback extraction method
    fallback_pipeline = Pipeline("fallback_extraction")
    fallback_pipeline.add_stage(FallbackExtractionStage())
    
    # Copy data from the original context
    initial_data = {
        "html_content": context.get("html_content"),
        "url": context.get("url")
    }
    
    # Execute fallback pipeline
    fallback_context = asyncio.run(fallback_pipeline.execute(initial_data))
    
    # Merge results if fallback was successful
    if not fallback_context.has_errors():
        result = fallback_context.get("extracted_data")
        print("Fallback extraction successful:", result)
```

## Monitoring and Debugging

### Execution Metrics

The pipeline context collects performance metrics:

```python
context = asyncio.run(pipeline.execute())

# Get execution metrics
metrics = context.get_metrics()

print(f"Pipeline: {metrics['pipeline_name']}")
print(f"Total execution time: {metrics['total_time']:.2f}s")
print(f"Stages executed: {metrics['total_stages']}")
print(f"Successful stages: {metrics['successful_stages']}")

# Stage-specific metrics
print("\nStage metrics:")
for stage_name, stage_metrics in metrics['stages'].items():
    status = stage_metrics['status']
    time = stage_metrics['execution_time']
    print(f"  {stage_name}: {status} in {time:.2f}s")
```

### Detailed Logging

Configure detailed logging for debugging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a logger for pipelines
logger = logging.getLogger('pipeline')
logger.setLevel(logging.DEBUG)

# Add a file handler
file_handler = logging.FileHandler('pipeline_debug.log')
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Create and execute pipeline
pipeline = Pipeline("debugged_pipeline")
# ... add stages ...
context = asyncio.run(pipeline.execute())
```

### Tracing Execution

You can add custom tracing to monitor the pipeline's execution flow:

```python
class TracingStage(PipelineStage):
    async def process(self, context: PipelineContext) -> bool:
        print(f"[TRACE] Executing {self.name}")
        print(f"[TRACE] Context data keys: {list(context.data.keys())}")
        return True

# Add tracing stages between regular stages
pipeline = Pipeline("traced_pipeline")
pipeline.add_stage(HttpInputStage({"url": "https://example.com"}))
pipeline.add_stage(TracingStage({"name": "trace_after_input"}))
pipeline.add_stage(HtmlProcessingStage())
pipeline.add_stage(TracingStage({"name": "trace_after_processing"}))
pipeline.add_stage(ContentExtractionStage())
```

### Debugging Context State

For complex pipelines, it can be helpful to visualize the context state:

```python
class ContextDebugStage(PipelineStage):
    async def process(self, context: PipelineContext) -> bool:
        import json
        
        print(f"\n=== Context State at {self.name} ===")
        
        # Print data (limited size for readability)
        data_preview = {}
        for key, value in context.data.items():
            if isinstance(value, str) and len(value) > 100:
                data_preview[key] = f"{value[:100]}... (len: {len(value)})"
            elif isinstance(value, (list, dict)) and len(value) > 5:
                data_preview[key] = f"{type(value).__name__} with {len(value)} items"
            else:
                data_preview[key] = value
                
        print(json.dumps(data_preview, indent=2, default=str))
        print("=======================================\n")
        return True

# Add debug stages to your pipeline
pipeline.add_stage(ContextDebugStage({"name": "debug_final_state"}))
```

## Performance Optimization Tips

### 1. Use Parallel Execution for Independent Stages

When stages don't depend on each other, enable parallel execution:

```python
pipeline = Pipeline("parallel_pipeline", {"parallel_execution": True})

# These stages will execute in parallel if possible
pipeline.add_stage(ImageExtractionStage())
pipeline.add_stage(LinkExtractionStage())
pipeline.add_stage(TextExtractionStage())
```

### 2. Implement Caching for Expensive Operations

Add caching to stages with expensive operations:

```python
class CachedExtractionStage(ContentExtractionStage):
    def __init__(self, config=None):
        super().__init__(config)
        self.cache = {}
        
    async def process(self, context: PipelineContext) -> bool:
        # Generate cache key from input
        html_content = context.get("html_content", "")
        cache_key = hash(html_content)
        
        # Check cache
        if cache_key in self.cache:
            context.set("extracted_data", self.cache[cache_key])
            return True
            
        # Execute normal processing
        success = await super().process(context)
        
        # Update cache on success
        if success:
            self.cache[cache_key] = context.get("extracted_data")
            
        return success
```

### 3. Optimize Data Flow

Minimize data passing between stages:

```python
# Instead of passing the entire HTML document
class OptimizedHtmlProcessingStage(PipelineStage):
    async def process(self, context: PipelineContext) -> bool:
        html_content = context.get("html_content")
        
        # Extract only what's needed for later stages
        main_content = extract_main_content(html_content)
        
        # Store only what's needed
        context.set("main_content", main_content)
        
        # Optionally clean up large data no longer needed
        if self.config.get("cleanup_html", True):
            del context.data["html_content"]
            
        return True
```

### 4. Use Timeouts for External Operations

Add timeouts to prevent hanging on external operations:

```python
class TimeoutAwareStage(PipelineStage):
    async def process(self, context: PipelineContext) -> bool:
        timeout = self.config.get("timeout", 10)
        
        try:
            # Run operation with timeout
            result = await asyncio.wait_for(
                self._expensive_operation(context),
                timeout=timeout
            )
            context.set("operation_result", result)
            return True
        except asyncio.TimeoutError:
            context.add_error(self.name, f"Operation timed out after {timeout}s")
            return False
            
    async def _expensive_operation(self, context):
        # Expensive or external operation
        pass
```

### 5. Profile and Optimize Bottlenecks

Use the built-in metrics to identify bottlenecks:

```python
# Execute pipeline
context = asyncio.run(pipeline.execute())

# Analyze stage performance
metrics = context.get_metrics()
sorted_stages = sorted(
    metrics['stages'].items(),
    key=lambda x: x[1]['execution_time'],
    reverse=True
)

print("Stage execution times (slowest first):")
for stage_name, stage_metrics in sorted_stages:
    print(f"  {stage_name}: {stage_metrics['execution_time']:.2f}s")
```

### 6. Batch Processing for Better Resource Utilization

Process multiple items in batches:

```python
class BatchProcessingStage(PipelineStage):
    async def process(self, context: PipelineContext) -> bool:
        items = context.get("items", [])
        batch_size = self.config.get("batch_size", 10)
        
        # Process in batches
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            
            # Process batch (potentially in parallel)
            batch_tasks = [self._process_item(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            
            results.extend(batch_results)
            
        context.set("processed_items", results)
        return True
        
    async def _process_item(self, item):
        # Process individual item
        pass
```

## Conclusion

This guide covered the basics of using the SmartScrape Pipeline Architecture for data processing workflows. By following these patterns and best practices, you can create efficient, maintainable, and robust pipelines for a variety of data extraction and transformation tasks.

For more advanced topics, refer to:
- [Custom Stages Guide](custom_stages.md) - Creating your own pipeline stages
- [Pipeline Architecture](architecture.md) - Core concepts and design principles
- Example implementations in the `examples/pipelines/` directory