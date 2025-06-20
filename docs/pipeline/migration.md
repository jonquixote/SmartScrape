# Pipeline Architecture Migration Guide

This guide provides detailed instructions for migrating existing code to the new SmartScrape Pipeline Architecture. It offers a systematic approach to identify candidates for migration, convert them to pipeline stages, ensure backward compatibility, and validate the migration's success.

## 1. Identifying Pipeline Candidates

### Extraction Components

The following existing components are excellent candidates for pipeline migration:

- **Content Extractors**: Classes in `extraction/content_extraction.py` that follow a process-transform-output pattern
- **Normalization Components**: Classes in `extraction/content_normalization.py` that perform data cleaning/standardization
- **Validation Components**: Classes in `extraction/content_validation.py` that verify data quality
- **Multi-Strategy Components**: Classes that try different approaches in sequence (like `MultiStrategyExtractor`)

### Indicators of Good Pipeline Candidates

Look for code with these characteristics:

1. **Sequential Processing**: Code that processes data through distinct steps
2. **Clear Input/Output Boundaries**: Functions with well-defined inputs and outputs
3. **Transformation Logic**: Code that transforms data from one format to another
4. **Conditional Execution**: Logic that chooses different processing paths based on data
5. **Error Handling and Recovery**: Code with robust error management

### Example Candidates

```python
# This method is a good pipeline candidate because it has:
# - Clear input (html_content, url)
# - Multiple processing steps
# - Error handling
# - Defined output
async def extract_content(self, html_content, url, options=None):
    options = options or {}
    try:
        # Step 1: Preprocessing
        normalized_content = self._normalize_html(html_content)
        
        # Step 2: Content detection
        content_type = self._detect_content_type(normalized_content, url)
        
        # Step 3: Extraction based on type
        if content_type == "article":
            result = self._extract_article(normalized_content)
        elif content_type == "listing":
            result = self._extract_listing(normalized_content)
        else:
            result = self._extract_generic(normalized_content)
            
        # Step 4: Postprocessing
        return self._postprocess_result(result)
    except Exception as e:
        self.logger.error(f"Extraction failed: {str(e)}")
        return {"success": False, "error": str(e)}
```

## 2. Step-by-Step Conversion Process

### 2.1 Create Pipeline Stage Stubs

Start by creating stub classes for each pipeline stage:

1. **Identify Logical Processing Steps**: Break down the component into logical steps
2. **Create Stage Classes**: Create a separate pipeline stage for each step
3. **Define Stage Inputs/Outputs**: Document what each stage needs and produces

Example:

```python
# Original extraction logic has these steps:
# 1. Normalize HTML
# 2. Detect content type
# 3. Extract based on type
# 4. Postprocess results

# Convert to these stages:
class HTMLNormalizationStage(PipelineStage):
    """Normalizes HTML content."""
    
    async def process(self, context):
        html_content = context.get("html_content")
        if not html_content:
            context.add_error(self.name, "No HTML content provided")
            return False
            
        normalized_content = self._normalize_html(html_content)
        context.set("normalized_content", normalized_content)
        return True
        
class ContentTypeDetectionStage(PipelineStage):
    # ...

class ContentExtractionStage(PipelineStage):
    # ...

class ResultPostprocessingStage(PipelineStage):
    # ...
```

### 2.2 Implement Pipeline Stage Logic

For each stage:

1. **Move Business Logic**: Transfer the logic from the original component
2. **Adapt to Context**: Update code to use the PipelineContext for data exchange
3. **Add Validation**: Implement input validation using `validate_input()`
4. **Add Error Handling**: Implement proper error handling using `handle_error()`

Example:

```python
class ContentExtractionStage(PipelineStage):
    """Extracts content based on detected content type."""
    
    def validate_input(self, context):
        required = ["normalized_content", "content_type", "url"]
        for field in required:
            if field not in context.data:
                context.add_error(self.name, f"Missing required field: {field}")
                return False
        return True
    
    async def process(self, context):
        try:
            normalized_content = context.get("normalized_content")
            content_type = context.get("content_type")
            
            if content_type == "article":
                result = self._extract_article(normalized_content)
            elif content_type == "listing":
                result = self._extract_listing(normalized_content)
            else:
                result = self._extract_generic(normalized_content)
                
            context.update(result)
            return True
        except Exception as e:
            return self.handle_error(context, e)
```

### 2.3 Create Pipeline Configuration

Define how stages connect in a pipeline:

1. **Create Pipeline Template**: Define a reusable pipeline template
2. **Configure Stage Order**: Set up the correct execution order
3. **Add Configuration Options**: Include configurable parameters

Example:

```python
# core/pipeline/templates/extraction_pipeline.py
class ExtractionPipeline(Pipeline):
    """Standard pipeline for content extraction."""
    
    def __init__(self, name="extraction_pipeline", config=None):
        super().__init__(name, config)
        self._setup_stages()
        
    def _setup_stages(self):
        # Create and add stages in sequence
        self.add_stages([
            HTMLNormalizationStage(self.config.get("normalization", {})),
            ContentTypeDetectionStage(self.config.get("detection", {})),
            ContentExtractionStage(self.config.get("extraction", {})),
            ResultPostprocessingStage(self.config.get("postprocessing", {}))
        ])
```

### 2.4 Register with Pipeline Registry

Make the pipeline available through the registry:

```python
# In pipeline initialization code
registry = PipelineRegistry()
registry.register_pipeline("extraction", ExtractionPipeline)
```

### 2.5 Update Calling Code

Update the code that uses the original component:

```python
# Before:
extractor = ContentExtractor()
result = await extractor.extract_content(html, url)

# After:
pipeline = pipeline_registry.create_pipeline("extraction")
context = await pipeline.execute({
    "html_content": html,
    "url": url
})
result = context.data
```

## 3. Handling Backward Compatibility

### 3.1 Using Adapter Classes

Create adapter classes to maintain backward compatibility:

```python
# Adapter for legacy code to use pipelines
class LegacyExtractorAdapter:
    def __init__(self, pipeline_name="extraction"):
        self.pipeline_registry = PipelineRegistry()
        self.pipeline_name = pipeline_name
        
    async def extract_content(self, html_content, url, options=None):
        pipeline = self.pipeline_registry.create_pipeline(self.pipeline_name)
        context = await pipeline.execute({
            "html_content": html_content,
            "url": url,
            "options": options or {}
        })
        
        if context.has_errors():
            return {"success": False, "errors": context.metadata["errors"]}
        
        return {"success": True, **context.data}
```

### 3.2 Using Feature Flags

Implement feature flags to control the rollout:

```python
# In configuration
config = {
    "use_pipeline_architecture": True,  # Master switch
    "pipeline_components": {
        "extraction": True,
        "validation": False,
        "normalization": True
    }
}

# In controller code
async def process_content(html, url):
    if config["use_pipeline_architecture"] and config["pipeline_components"]["extraction"]:
        # Use pipeline
        pipeline = pipeline_registry.create_pipeline("extraction")
        context = await pipeline.execute({"html_content": html, "url": url})
        return context.data
    else:
        # Use legacy code
        extractor = ContentExtractor()
        return await extractor.extract_content(html, url)
```

### 3.3 Parallel Execution Mode

Run both implementations to compare results:

```python
async def parallel_extraction(html, url):
    # Run both implementations
    pipeline_result = None
    legacy_result = None
    
    # Execute in parallel
    pipeline_task = asyncio.create_task(extract_with_pipeline(html, url))
    legacy_task = asyncio.create_task(extract_with_legacy(html, url))
    
    # Wait for both to complete
    results = await asyncio.gather(pipeline_task, legacy_task, return_exceptions=True)
    
    # Check results and log discrepancies
    if results[0] != results[1]:
        log_discrepancy(results[0], results[1])
        
    # Return the preferred result based on configuration
    return results[0] if config["prefer_pipeline"] else results[1]
```

## 4. Testing Migration Success

### 4.1 Unit Tests for Stages

Create dedicated tests for each stage:

```python
# tests/core/pipeline/stages/test_extraction_stage.py
async def test_content_extraction_stage():
    # Setup
    stage = ContentExtractionStage()
    context = PipelineContext({
        "normalized_content": "<html>test content</html>",
        "content_type": "article",
        "url": "https://example.com/article"
    })
    
    # Execute
    result = await stage.process(context)
    
    # Assert
    assert result is True
    assert "title" in context.data
    assert "content" in context.data
```

### 4.2 Integration Tests for Pipelines

Test complete pipelines:

```python
# tests/core/pipeline/test_extraction_pipeline.py
async def test_extraction_pipeline():
    # Setup
    pipeline = ExtractionPipeline()
    
    # Execute
    context = await pipeline.execute({
        "html_content": load_test_html("article.html"),
        "url": "https://example.com/article"
    })
    
    # Assert
    assert not context.has_errors()
    assert "title" in context.data
    assert "content" in context.data
```

### 4.3 Comparative Tests

Compare results with the original implementation:

```python
# tests/core/pipeline/test_migration.py
async def test_result_equivalence():
    # Setup test data
    html = load_test_html("complex_page.html")
    url = "https://example.com/complex_page"
    
    # Get results from both implementations
    legacy_extractor = ContentExtractor()
    legacy_result = await legacy_extractor.extract_content(html, url)
    
    pipeline = ExtractionPipeline()
    pipeline_context = await pipeline.execute({"html_content": html, "url": url})
    pipeline_result = pipeline_context.data
    
    # Compare results (normalize formats first)
    normalized_legacy = normalize_result(legacy_result)
    normalized_pipeline = normalize_result(pipeline_result)
    
    # Assert equivalence
    assert normalized_pipeline == normalized_legacy
```

### 4.4 Performance Benchmarks

Measure and compare performance:

```python
# tests/benchmarks/test_pipeline_performance.py
async def benchmark_extraction(benchmark):
    html = load_large_test_html()
    url = "https://example.com/large_page"
    
    # Benchmark legacy implementation
    legacy_time = await benchmark_async(
        lambda: legacy_extractor.extract_content(html, url)
    )
    
    # Benchmark pipeline implementation
    pipeline_time = await benchmark_async(
        lambda: pipeline.execute({"html_content": html, "url": url})
    )
    
    print(f"Legacy: {legacy_time:.2f}ms, Pipeline: {pipeline_time:.2f}ms")
    
    # Assert pipeline is not significantly slower
    assert pipeline_time <= legacy_time * 1.2  # Allow 20% overhead
```

## 5. Phased Migration Approach

### Phase 1: Foundation (2 weeks)

1. Implement core pipeline infrastructure
2. Create basic stage implementations
3. Develop testing framework
4. Document architecture and patterns

### Phase 2: First Components (2 weeks)

1. Identify 2-3 simple components for migration
2. Create adapter classes
3. Implement & test pipeline versions
4. Run in parallel with legacy code

### Phase 3: Critical Path Components (3 weeks)

1. Migrate core extraction logic
2. Implement complex pipelines (with branches, etc.)
3. Update controllers to use pipelines
4. Extensive testing & performance tuning

### Phase 4: Complete Migration (3 weeks)

1. Migrate remaining components
2. Remove legacy code paths
3. Optimize pipeline configurations
4. Finalize documentation

### Phase 5: Cleanup & Optimization (2 weeks)

1. Remove compatibility layers
2. Refine pipeline interfaces
3. Optimize performance
4. Conduct final review

## 6. Common Migration Challenges

### 6.1 State Management

**Challenge**: Legacy code often keeps state in instance variables, while pipelines use the context.

**Solution**: 
- Move instance state to context
- Use metadata for tracking state
- Consider context snapshots for complex state

### 6.2 Async/Sync Compatibility

**Challenge**: Pipeline architecture uses async, but legacy code might be sync.

**Solution**:
- Convert sync code to async
- Use asyncio.to_thread() for blocking operations
- Create adapter wrappers to bridge sync/async boundaries

### 6.3 Error Handling Differences

**Challenge**: Different error handling paradigms between legacy and pipeline code.

**Solution**:
- Map exceptions to context errors
- Use PipelineStage.handle_error() consistently
- Adapt error response formats

### 6.4 Configuration Complexity

**Challenge**: Complex configuration mappings between old and new patterns.

**Solution**:
- Create configuration mappers
- Implement schema validation
- Use sensible defaults

## 7. Resources and Tools

- **Pipeline Converter**: Use `scripts/tools/pipeline_converter.py` to assist with migration
- **Adapter Classes**: Find reusable adapters in `core/pipeline/adapters.py`
- **Compatibility Layer**: Use utilities from `core/pipeline/compatibility.py`
- **Documentation**: Refer to the pipeline architecture docs

## 8. Conclusion

Migrating to the pipeline architecture is a significant undertaking, but the benefits in maintainability, testability, and extensibility are substantial. By following this phased approach and leveraging the provided tools, the migration can be accomplished with minimal disruption to existing functionality.

For questions or assistance, please contact the architecture team.