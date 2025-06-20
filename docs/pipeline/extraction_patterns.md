# SmartScrape Extraction Patterns

This document analyzes the common extraction patterns found in the SmartScrape codebase and maps them to the pipeline architecture. It provides guidance for standardizing content extraction operations as pipeline stages.

## 1. Common Extraction Patterns

### 1.1 HTML Cleaning and Preparation

#### Operations
- **HTML Parsing**: Converting raw HTML strings to structured document objects
- **Content Cleaning**: Removing unwanted elements (scripts, ads, navigation, etc.)
- **Character Encoding Handling**: Detecting and normalizing character encodings
- **Lazy-Loading Detection**: Identifying content that requires JavaScript rendering

#### Key Code Examples
```python
# From content_extraction.py
soup = BeautifulSoup(html_content, 'lxml')
# Clean by removing script, style and other non-content tags
for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
    tag.decompose()
```

#### Dependencies
- **Key Libraries**: BeautifulSoup, lxml, html5lib
- **Services**: Usually operates independently without external services

### 1.2 DOM Traversal and Element Selection

#### Operations
- **CSS Selection**: Finding elements with CSS selectors
- **XPath Queries**: Extracting content using XPath
- **Element Hierarchy Analysis**: Traversing parent-child relationships
- **Container Discovery**: Identifying item containers in listings

#### Key Code Examples
```python
# From content_extraction.py
# CSS selection
elements = soup.select(selector)

# XPath usage
from lxml import etree
dom = etree.HTML(html_content)
elements = dom.xpath(xpath)

# Container discovery for listings
containers = self._find_listing_containers(soup)
```

#### Dependencies
- **Key Libraries**: BeautifulSoup, lxml, cssselect
- **Services**: Independent operation

### 1.3 Content Extraction From Elements

#### Operations
- **Text Extraction**: Getting clean text from elements
- **Attribute Extraction**: Retrieving specific attributes (href, src, data-*)
- **Structured Data Extraction**: Finding and parsing JSON-LD, microdata
- **Table Extraction**: Converting HTML tables to structured data
- **List Extraction**: Converting HTML lists to structured data

#### Key Code Examples
```python
# From content_extraction.py
# Basic text extraction
title = elements[0].get_text(strip=True)

# Attribute extraction
image_url = elements[0].get('src')
if not image_url:
    image_url = elements[0].get('data-src')  # Handle lazy-loading

# Process relative URLs
if value and url and value.startswith('/'):
    from urllib.parse import urljoin
    value = urljoin(url, value)

# Structured data extraction
ld_json = soup.find_all('script', {'type': 'application/ld+json'})
for script in ld_json:
    try:
        data = json.loads(script.string)
        structured_items.append(data)
    except:
        pass
```

#### Dependencies
- **Key Libraries**: json, re, urllib
- **Services**: Sometimes requires URL context for resolving relative links

### 1.4 Semantic Content Analysis

#### Operations
- **Main Content Detection**: Identifying the primary content area
- **Content Classification**: Determining content type (article, listing, etc.)
- **Entity Extraction**: Identifying people, places, dates, prices, etc.
- **Topic Classification**: Categorizing content by topic
- **Readability Assessment**: Analyzing content readability

#### Key Code Examples
```python
# From content_analysis.py
# Entity extraction
entities = self._extract_entities(text_content)

# Topic classification
topics = self._classify_topics(text_content, entities)

# Readability assessment
readability = self._assess_readability(text_content)

# Content type detection
if soup.find('article') or soup.find('meta', {'property': 'article:published_time'}):
    return "article"
```

#### Dependencies
- **Key Libraries**: nltk, re, spacy (when available)
- **Services**: May utilize AI services for enhanced analysis

### 1.5 Data Normalization and Validation

#### Operations
- **Type Conversion**: Converting strings to appropriate data types
- **Date/Time Parsing**: Converting varied date formats to standard formats
- **Unit Standardization**: Normalizing units of measurement
- **Data Cleaning**: Trimming, case normalization, special character handling
- **Schema Validation**: Checking data against expected schemas

#### Key Code Examples
```python
# From content_normalization.py
# Date normalization
try:
    import dateutil.parser
    parsed_date = dateutil.parser.parse(date_text)
    normalized_date = parsed_date.isoformat()
except:
    normalized_date = date_text

# Price normalization
price_match = re.search(r'(\$|€|£|¥)?\s*[\d,]+(\.\d{1,2})?', price_text)
if price_match:
    normalized_price = price_match.group(0).strip()

# Schema validation
if schema:
    schema_validator = self._get_service("schema_validator")
    validation_result = schema_validator.validate(extraction_result, schema)
```

#### Dependencies
- **Key Libraries**: dateutil, re, json
- **Services**: May use schema validation services

### 1.6 AI-Assisted Extraction

#### Operations
- **Content Summarization**: Generating summaries of text content
- **Semantic Extraction**: Using AI to understand content meaning
- **Structured Data Generation**: Creating structured data from unstructured content
- **Missing Data Inference**: Filling gaps in extracted data

#### Key Code Examples
```python
# From content_extraction.py
# AI-based extraction
prompt = self._generate_extraction_prompt(content, schema)
response = await ai_service.generate_response(
    prompt=prompt,
    options={
        "model": model,
        "response_format": {"type": "json_object"},
        "temperature": 0.2
    }
)
```

#### Dependencies
- **Key Libraries**: json
- **Services**: Relies on AI models (OpenAI, Google Gemini, etc.)
- **Performance Considerations**: Higher latency, rate limits, costs

### 1.7 Result Aggregation and Formatting

#### Operations
- **Data Merging**: Combining data from multiple extraction methods
- **Prioritization**: Selecting best results when multiple sources exist
- **Format Conversion**: Converting to standard output formats (JSON, CSV)
- **Metadata Addition**: Adding extraction provenance information

#### Key Code Examples
```python
# From content_extraction.py
# Result merging
result.update({
    "success": True,
    "extraction_method": "semantic",
    "semantic_structure": semantic_result.get("semantic_structure", {}),
    "entities": semantic_result.get("entities", {}),
    "keywords": semantic_result.get("keywords", []),
    "title": semantic_result.get("title", result["title"])
})

# Multiple extraction attempts
for strategy_name in strategy_order:
    result = await strategy_func(...)
    if result.get("success", False):
        extraction_results.update(result)
        extraction_results["strategy_used"] = strategy_name
        return extraction_results
```

#### Dependencies
- **Key Libraries**: json
- **Services**: Independent operation

## 2. Mapping to Pipeline Stages

### 2.1 Input Stages

#### HTML Input Stage
- **Functionality**: Fetch HTML content from URLs
- **Common Operations**:
  - URL validation and normalization
  - HTTP request handling
  - Response decoding
  - Compression handling
  - Status code handling
  - Rate limiting
  - Retry logic
  - Proxy management

#### File Input Stage
- **Functionality**: Load content from local files
- **Common Operations**:
  - File reading
  - Format detection
  - Encoding detection
  - Line-by-line streaming for large files

#### Structured Data Input Stage
- **Functionality**: Import data from structured formats
- **Common Operations**:
  - JSON/CSV/XML parsing
  - Database query results

### 2.2 Processing Stages

#### HTML Preparation Stage
- **Functionality**: Clean and prepare HTML for extraction
- **Common Operations**:
  - HTML parsing
  - Tag removal (scripts, styles, etc.)
  - Comment removal
  - Character encoding normalization

#### DOM Query Stage
- **Functionality**: Extract elements using CSS/XPath selectors
- **Common Operations**:
  - CSS selector application
  - XPath query execution
  - Result collection

#### Content Extraction Stage
- **Functionality**: Extract and structure specific content from elements
- **Common Operations**:
  - Text extraction
  - Attribute extraction
  - Structured data parsing
  - Table conversion
  - List processing

#### Semantic Analysis Stage
- **Functionality**: Analyze content for semantic meaning
- **Common Operations**:
  - Entity extraction
  - Topic classification
  - Sentiment analysis
  - Readability scoring

#### AI Extraction Stage
- **Functionality**: Use AI models to extract structured data
- **Common Operations**:
  - Prompt generation
  - AI service integration
  - Response parsing
  - Error handling for AI failures

#### Normalization Stage
- **Functionality**: Clean and standardize extracted data
- **Common Operations**:
  - Type conversion
  - Date/time normalization
  - Unit standardization
  - String cleaning
  - URL absolutization

#### Validation Stage
- **Functionality**: Validate data against schemas or rules
- **Common Operations**:
  - Schema validation
  - Data quality checks
  - Required field verification
  - Type checking
  - Value range validation

### 2.3 Output Stages

#### Data Format Output Stage
- **Functionality**: Convert data to specific output formats
- **Common Operations**:
  - JSON serialization
  - CSV conversion
  - XML generation

#### File Output Stage
- **Functionality**: Save data to files
- **Common Operations**:
  - File writing
  - Encoding setting
  - Append vs. overwrite modes
  - Backup creation

#### Database Output Stage
- **Functionality**: Store data in databases
- **Common Operations**:
  - Connection management
  - Query generation
  - Batch insertion
  - Transaction handling

#### API Output Stage
- **Functionality**: Send data to external APIs
- **Common Operations**:
  - API request formatting
  - Authentication
  - Error handling
  - Response processing

### 2.4 Specialized Stage Types

#### Conditional Extraction Stage
- **Functionality**: Apply different extraction logic based on content characteristics
- **Common Operations**:
  - Content type detection
  - Branch selection
  - Conditional processing

#### Parallel Extraction Stage
- **Functionality**: Apply multiple extraction methods in parallel
- **Common Operations**:
  - Concurrent execution
  - Result merging
  - Best result selection

#### Fallback Extraction Stage
- **Functionality**: Try multiple extraction methods in sequence until success
- **Common Operations**:
  - Sequential attempts
  - Success condition evaluation
  - Error aggregation

#### Pagination Handling Stage
- **Functionality**: Handle multi-page content
- **Common Operations**:
  - Next page detection
  - Page navigation
  - Result aggregation across pages

## 3. Data Flow Requirements

### 3.1 Input/Output Formats

#### Input Stage
- **Input**: 
  - URL, file path, or raw content string
  - Configuration parameters (headers, timeout, etc.)
- **Output**: 
  - `PipelineResponse` containing:
    - Raw content
    - Metadata (content type, encoding, etc.)
    - Status information

#### HTML Processing Stage
- **Input**: 
  - Raw HTML string
  - Processing configuration
- **Output**: 
  - Parsed document object (BeautifulSoup or lxml)
  - Cleaned HTML string

#### Extraction Stage
- **Input**: 
  - Parsed document object
  - Extraction rules (selectors, patterns, etc.)
- **Output**: 
  - Extracted raw data dictionary
  - Extraction metadata (method used, confidence, etc.)

#### Normalization Stage
- **Input**: 
  - Raw extracted data
  - Normalization rules
- **Output**: 
  - Normalized data matching expected types and formats

#### Validation Stage
- **Input**: 
  - Normalized data
  - Validation schema or rules
- **Output**: 
  - Validated data
  - Validation results (success, errors, warnings)

#### Output Stage
- **Input**: 
  - Final processed data
  - Output configuration
- **Output**: 
  - Operation result (success, file path, etc.)

### 3.2 Context Requirements

#### Shared Context Data
- **URL Context**: Required for resolving relative URLs
- **Content Type**: Guides extraction approach
- **Extraction Configuration**: Selectors, patterns, rules
- **Metadata Accumulation**: Each stage should add its metadata
- **Error Collection**: Centralized error tracking

#### State Management
- **Multi-stage Results**: Storing intermediate results
- **Pipeline Metrics**: Tracking performance at each stage
- **Cross-stage Signals**: Allowing stages to communicate
- **Caching**: Storing results for reuse

### 3.3 Error Handling Strategies

#### Transient Errors
- **Strategy**: Automatic retry with backoff
- **Examples**: Network timeouts, rate limits, temporary service unavailability
- **Context Requirements**: Retry count, delay parameters

#### Content-Based Errors
- **Strategy**: Fallback to alternative extraction methods
- **Examples**: Missing expected elements, unexpected content structure
- **Context Requirements**: Available fallback methods, success criteria

#### Critical Errors
- **Strategy**: Fail pipeline or skip stage
- **Examples**: Invalid credentials, permanent service failures
- **Context Requirements**: Error classification, failure policy

#### Data Quality Issues
- **Strategy**: Partial results with warnings
- **Examples**: Incomplete extraction, low confidence results
- **Context Requirements**: Quality thresholds, warning accumulation

## 4. Implementation Recommendations

### 4.1 Stage Configuration Framework

Create a unified configuration framework for stages:

```python
# Stage configuration example
html_input_config = {
    "throttle_rate": 10.0,  # Requests per second
    "max_retries": 3,
    "timeout": 30,
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    },
    "respect_robots_txt": True,
    "proxy_settings": {
        "enabled": False,
        "proxy_url": None
    }
}
```

### 4.2 Core Stage Interfaces

Implement these foundational stage types:

```python
# Base stage interfaces
class InputStage(PipelineStage):
    async def acquire_data(self, request: PipelineRequest, context: PipelineContext) -> Optional[PipelineResponse]:
        """Acquire data from a source."""
        pass
        
class ProcessingStage(PipelineStage):
    async def transform_data(self, data: Any, context: PipelineContext) -> Optional[Any]:
        """Transform input data."""
        pass
        
class OutputStage(PipelineStage):
    async def deliver_output(self, data: Any, context: PipelineContext) -> Optional[Any]:
        """Deliver output to destination."""
        pass
        
class ConditionalStage(PipelineStage):
    async def evaluate_condition(self, context: PipelineContext) -> bool:
        """Evaluate if condition is met."""
        pass
```

### 4.3 Stage Registry Organization

Organize stages in a hierarchical registry:

```
core/pipeline/stages/
├── input/
│   ├── http_input.py
│   ├── file_input.py
│   └── database_input.py
├── processing/
│   ├── html/
│   │   ├── html_preparation.py
│   │   ├── dom_query.py
│   │   └── content_extraction.py
│   ├── semantic/
│   │   ├── entity_extraction.py
│   │   ├── topic_classification.py
│   │   └── sentiment_analysis.py
│   ├── ai/
│   │   ├── ai_extraction.py
│   │   └── ai_enhancement.py
│   ├── normalization/
│   │   ├── type_conversion.py
│   │   ├── date_normalization.py
│   │   └── text_cleaning.py
│   └── validation/
│       ├── schema_validation.py
│       └── data_quality.py
├── output/
│   ├── file_output.py
│   ├── database_output.py
│   └── api_output.py
└── specialized/
    ├── conditional.py
    ├── parallel.py
    └── fallback.py
```

### 4.4 Pipeline Templates

Create standard pipeline templates for common extraction patterns:

```python
# Example extraction pipeline template
def create_standard_extraction_pipeline(config: Dict[str, Any]) -> Pipeline:
    """Create a standard extraction pipeline."""
    pipeline = Pipeline("standard_extraction", config)
    
    # Add input stage
    pipeline.add_stage(HttpInputStage(config.get("input_config")))
    
    # Add processing stages
    pipeline.add_stage(HtmlPreparationStage(config.get("html_config")))
    pipeline.add_stage(ContentExtractionStage(config.get("extraction_config")))
    pipeline.add_stage(NormalizationStage(config.get("normalization_config")))
    pipeline.add_stage(ValidationStage(config.get("validation_config")))
    
    # Add output stage
    pipeline.add_stage(JsonOutputStage(config.get("output_config")))
    
    return pipeline
```

## 5. Backward Compatibility Considerations

### 5.1 Adapters for Existing Code

Create adapter classes to bridge between old and new extraction methods:

```python
# Example adapter
class LegacyExtractorAdapter(PipelineStage):
    """Adapter for legacy ContentExtractor class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.extractor = ContentExtractor(use_stealth_browser=config.get("use_stealth_browser", False))
        
    async def process(self, context: PipelineContext) -> bool:
        try:
            html_content = context.get("html_content")
            url = context.get("url")
            content_type = context.get("content_type")
            
            result = await self.extractor.extract_content(
                html_content=html_content,
                url=url,
                content_type=content_type,
                extraction_params=self.config.get("extraction_params")
            )
            
            context.update(result)
            return result.get("success", False)
        except Exception as e:
            return self.handle_error(context, e)
```

### 5.2 Feature Flags

Implement feature flags to gradually transition to pipeline architecture:

```python
# Example configuration
config = {
    "use_pipeline_architecture": True,
    "fallback_to_legacy": True,
    "parallel_legacy_comparison": False
}
```

### 5.3 Metrics Comparison

Add monitoring to compare results between old and new implementations:

```python
# Example metrics collection
metrics = {
    "pipeline_execution_time": pipeline_time,
    "legacy_execution_time": legacy_time,
    "pipeline_success": pipeline_success,
    "legacy_success": legacy_success,
    "result_difference": calculate_difference(pipeline_result, legacy_result)
}
```

## 6. Performance Considerations

### 6.1 Resource-Intensive Operations

Identify operations that may require special handling:

1. **HTML Parsing**: Memory-intensive for large documents
2. **AI Model Calls**: High latency, potential rate limits
3. **Concurrent Network Requests**: Connection pooling needs
4. **DOM Traversal**: CPU-intensive for complex selectors
5. **Large Content Processing**: Memory management for large data

### 6.2 Optimization Strategies

Implement performance optimizations:

1. **Lazy Execution**: Process only what's needed
2. **Stream Processing**: Handle large content in chunks
3. **Caching**: Store expensive computation results
4. **Partial Extraction**: Extract only required fields
5. **Selective AI Usage**: Use AI only when pattern-based extraction fails
6. **Throttling Controls**: Manage resource usage

### 6.3 Monitoring Points

Add monitoring for key performance indicators:

1. **Stage Execution Time**: Time spent in each stage
2. **Memory Usage**: Peak memory during processing
3. **External Service Latency**: Time waiting for AI, APIs
4. **Cache Hit Rates**: Effectiveness of caching
5. **Error Rates by Stage**: Where failures occur
6. **Extraction Quality Metrics**: Completeness, accuracy

## 7. Implementation Strategy

This section provides guidance on how to transform existing extraction code into pipeline stages.

### 7.1 Extraction Strategy Selection

The `MultiStrategyExtractor` class provides a flexible approach to extraction by trying different strategies in a prioritized order. This pattern can be transformed into a pipeline with conditional branches:

```python
class StrategySelectionStage(ConditionalStage):
    """Determines the best extraction strategy based on content characteristics."""
    
    async def process(self, context: PipelineContext) -> bool:
        content_type = context.get("content_type")
        if not content_type:
            # Determine content type from HTML
            html_content = context.get("html_content")
            soup = BeautifulSoup(html_content, 'lxml')
            content_type = self._determine_content_type(soup, context.get("user_intent"))
            context.set("content_type", content_type)
            
        # Get strategy order based on content type
        strategy_order = self._get_strategy_order(content_type, context.get("user_intent"))
        context.set("strategy_order", strategy_order)
        
        return True
        
    def _determine_content_type(self, soup, user_intent):
        # Implementation from MultiStrategyExtractor._determine_content_type
        pass
        
    def _get_strategy_order(self, content_type, user_intent):
        # Implementation from MultiStrategyExtractor._get_strategy_order
        pass
```

### 7.2 Fallback Mechanism

The fallback extraction pattern is common throughout the codebase. It can be implemented as a specialized pipeline stage:

```python
class FallbackExtractionStage(PipelineStage):
    """Tries multiple extraction methods in sequence until one succeeds."""
    
    async def process(self, context: PipelineContext) -> bool:
        strategy_order = context.get("strategy_order", ["css_selector", "xpath", "content_heuristics", "ai_guided"])
        
        for strategy_name in strategy_order:
            try:
                # Create a sub-context for this strategy attempt
                strategy_context = context.create_child_context()
                
                # Execute the strategy
                strategy_stage = self._get_strategy_stage(strategy_name)
                success = await strategy_stage.process(strategy_context)
                
                # Record the attempt
                context.append("attempts", {
                    "strategy": strategy_name,
                    "success": success
                })
                
                if success:
                    # Copy results from successful strategy to main context
                    context.update(strategy_context.data)
                    context.set("strategy_used", strategy_name)
                    return True
                    
            except Exception as e:
                # Log error and continue to next strategy
                self.logger.error(f"Error with {strategy_name} strategy: {str(e)}")
                context.append("attempts", {
                    "strategy": strategy_name,
                    "success": False,
                    "error": str(e)
                })
                
        # If we get here, all strategies failed
        context.set("error", "All extraction strategies failed")
        return False
        
    def _get_strategy_stage(self, strategy_name):
        # Return the appropriate stage for the strategy
        if strategy_name == "css_selector":
            return CssSelectorExtractionStage(self.config.get("css_config"))
        elif strategy_name == "xpath":
            return XPathExtractionStage(self.config.get("xpath_config"))
        elif strategy_name == "content_heuristics":
            return ContentHeuristicsStage(self.config.get("heuristics_config"))
        elif strategy_name == "ai_guided":
            return AIGuidedExtractionStage(self.config.get("ai_config"))
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
```

### 7.3 Content Type-Specific Extraction

The current code uses different extraction methods based on content type. This can be implemented as conditional branching in a pipeline:

```python
class ContentTypeExtractionStage(ConditionalStage):
    """Extracts content based on detected content type."""
    
    async def process(self, context: PipelineContext) -> bool:
        content_type = context.get("content_type")
        
        if content_type == "article":
            return await self._process_with_stage(ArticleExtractionStage(), context)
        elif content_type == "listing":
            return await self._process_with_stage(ListingExtractionStage(), context)
        elif content_type == "data_table":
            return await self._process_with_stage(TableExtractionStage(), context)
        else:
            # Generic extraction
            return await self._process_with_stage(GenericExtractionStage(), context)
            
    async def _process_with_stage(self, stage, context):
        # Initialize the stage with appropriate configuration
        stage.initialize(self.config.get(f"{stage.name}_config"))
        
        # Process with the stage
        return await stage.process(context)
```

### 7.4 AI-Assisted Extraction Integration

The AI-assisted extraction can be integrated as a pipeline stage that respects rate limits and implements circuit breaking:

```python
class AIExtractionStage(ProcessingStage):
    """Extracts data using AI models with proper error handling."""
    
    async def process(self, context: PipelineContext) -> bool:
        html_content = context.get("html_content")
        url = context.get("url")
        user_intent = context.get("user_intent")
        
        if not html_content:
            context.add_error("ai_extraction", "No HTML content provided")
            return False
            
        try:
            # Get circuit breaker service
            circuit_breaker = context.get_service("circuit_breaker_manager")
            
            # Define extraction function with circuit breaker
            @circuit_breaker.circuit_breaker("ai_extraction")
            async def extract_with_ai():
                return await extract_content_with_ai(
                    html_content=html_content,
                    url=url,
                    user_intent=user_intent,
                    desired_properties=context.get("desired_properties", []),
                    entity_type=user_intent.get("entity_type", "item") if user_intent else "item"
                )
            
            # Execute extraction with circuit breaker protection
            items = await extract_with_ai()
            
            if not items:
                context.add_error("ai_extraction", "AI extraction yielded no results")
                return False
                
            # Store results in context
            context.set("extraction_method", "ai_guided")
            context.set("content_type", "listing" if len(items) > 1 else "detail")
            if len(items) > 1:
                context.set("items", items)
            else:
                context.set("data", items[0])
                
            return True
            
        except Exception as e:
            context.add_error("ai_extraction", str(e))
            return False
```

### 7.5 Migration Path for Existing Code

To gradually migrate the codebase to the pipeline architecture:

1. **Wrap existing extractors** as pipeline stages
2. **Create pipeline configurations** that mirror current extraction flows
3. **Validate pipeline results** against existing extraction methods
4. **Gradually refactor** internal extraction logic into atomic stages
5. **Update controllers** to use pipeline execution

Example of an adapter stage that wraps existing extractor code:

```python
class MultiStrategyExtractorStage(PipelineStage):
    """Pipeline stage that wraps the MultiStrategyExtractor."""
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().initialize(config)
        self.extractor = MultiStrategyExtractor(use_ai=config.get("use_ai", True))
        
    async def process(self, context: PipelineContext) -> bool:
        try:
            result = await self.extractor.extract(
                html_content=context.get("html_content"),
                url=context.get("url"),
                user_intent=context.get("user_intent"),
                content_type=context.get("content_type"),
                extraction_schema=context.get("extraction_schema")
            )
            
            # Update context with extraction results
            context.update(result)
            
            # Track metrics
            context.set_metadata("extraction_time", result.get("extraction_time"))
            context.set_metadata("strategy_used", result.get("strategy_used"))
            
            return result.get("success", False)
            
        except Exception as e:
            return self.handle_error(context, e)
```

## 8. Common Patterns for Testing Pipeline Stages

Effective testing is crucial when migrating to a pipeline architecture:

### 8.1 Unit Testing Individual Stages

```python
async def test_html_preparation_stage():
    # Arrange
    stage = HtmlPreparationStage({"remove_scripts": True})
    context = PipelineContext({
        "html_content": "<html><head><script>alert('test')</script></head><body><p>Content</p></body></html>"
    })
    
    # Act
    success = await stage.process(context)
    
    # Assert
    assert success
    assert "<script>" not in context.get("cleaned_html")
    assert "Content" in context.get("cleaned_html")
```

### 8.2 Integration Testing Pipeline Flows

```python
async def test_extraction_pipeline():
    # Arrange
    pipeline = create_standard_extraction_pipeline({
        "input_config": {"timeout": 10},
        "extraction_config": {"use_css_selectors": True}
    })
    
    # Act
    context = await pipeline.execute({
        "url": "https://example.com/test-page"
    })
    
    # Assert
    assert not context.has_errors()
    assert context.get("extraction_method") is not None
    assert context.get("data") is not None
```

### 8.3 Comparative Testing During Migration

```python
async def test_pipeline_vs_legacy():
    # Arrange
    test_html = load_test_html("product_page.html")
    
    # Legacy approach
    legacy_extractor = MultiStrategyExtractor()
    
    # Pipeline approach
    pipeline = create_extraction_pipeline()
    
    # Act
    legacy_result = await legacy_extractor.extract(test_html, url="https://example.com")
    
    pipeline_context = await pipeline.execute({
        "html_content": test_html,
        "url": "https://example.com"
    })
    pipeline_result = {k: v for k, v in pipeline_context.data.items() 
                      if k not in pipeline_context.metadata}
    
    # Assert
    assert legacy_result["success"] == (not pipeline_context.has_errors())
    assert legacy_result.get("title") == pipeline_result.get("title")
    assert len(legacy_result.get("items", [])) == len(pipeline_result.get("items", []))
```