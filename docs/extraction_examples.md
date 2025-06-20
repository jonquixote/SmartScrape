# SmartScrape Extraction Examples Documentation

This document provides detailed walkthrough examples of the Universal Extraction Framework usage, corresponding to the code examples in `examples/extraction_examples.py`.

## Table of Contents

1. [Basic HTML Extraction](#basic-html-extraction)
   - [Product Information Extraction](#product-information-extraction)
   - [Article Content Extraction](#article-content-extraction)
   - [Listing Data Extraction](#listing-data-extraction)

2. [Pipeline Configuration](#pipeline-configuration)
   - [Standard Pipeline Configuration](#standard-pipeline-configuration)
   - [Custom Pipeline Creation](#custom-pipeline-creation)
   - [Pipeline Options](#pipeline-options)
   - [Handling Pipeline Results](#handling-pipeline-results)

3. [Custom Extractor Implementation](#custom-extractor-implementation)
   - [Creating a Specialized Extractor](#creating-a-specialized-extractor)
   - [Using Custom Extractors](#using-custom-extractors)
   - [Registering Custom Extractors](#registering-custom-extractors)

4. [Schema Usage](#schema-usage)
   - [Defining Custom Schemas](#defining-custom-schemas)
   - [Registering Multiple Schemas](#registering-multiple-schemas)
   - [Validating Against Schemas](#validating-against-schemas)
   - [Generating Schemas from Data](#generating-schemas-from-data)

5. [Extraction Quality](#extraction-quality)
   - [Configuring Quality Evaluation](#configuring-quality-evaluation)
   - [Interpreting Quality Metrics](#interpreting-quality-metrics)
   - [Improving Extraction Quality](#improving-extraction-quality)

## Basic HTML Extraction

### Product Information Extraction

Extract structured data from product pages using pattern-based extraction:

```python
async def extract_product_information(html_content: str) -> Dict[str, Any]:
    # Create service registry and context
    registry = ServiceRegistry()
    context = StrategyContext(registry)
    
    # Initialize required services 
    html_service = registry.get_or_create_service("html_service")
    schema_manager = SchemaManager(context)
    schema_manager.initialize()
    registry.register_service("schema_manager", schema_manager)
    
    # Create pattern extractor
    pattern_extractor = DOMPatternExtractor(context)
    pattern_extractor.initialize()
    
    # Define product schema
    product_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Product title"},
            "price": {"type": "number", "description": "Product price"},
            "description": {"type": "string", "description": "Product description"},
            "images": {"type": "array", "items": {"type": "string"}, "description": "Product images"},
            "sku": {"type": "string", "description": "Product SKU/ID"},
            "brand": {"type": "string", "description": "Product brand"},
            "availability": {"type": "string", "description": "Product availability"}
        },
        "required": ["title", "price"]
    }
    
    # Extract product data
    result = pattern_extractor.extract(
        html_content, 
        schema=product_schema,
        options={"content_type": "html", "clean_html": True}
    )
    
    return result
```

This example:
1. Creates a service registry and strategy context
2. Initializes required services (HTML service, schema manager)
3. Creates and initializes a pattern extractor
4. Defines a product schema with field types and descriptions
5. Extracts product data using the pattern extractor

The extractor identifies patterns in the HTML content that match schema fields and returns structured data.

### Article Content Extraction

Extract article content using AI-powered semantic extraction:

```python
async def extract_article_content(html_content: str) -> Dict[str, Any]:
    # Create service registry and context
    registry = ServiceRegistry()
    context = StrategyContext(registry)
    
    # Initialize required services
    html_service = registry.get_or_create_service("html_service")
    ai_service = registry.get_or_create_service("ai_service")
    
    # Create semantic extractor for article content
    semantic_extractor = AISemanticExtractor(context)
    semantic_extractor.initialize()
    
    # Define article schema
    article_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Article title"},
            "author": {"type": "string", "description": "Article author"},
            "published_date": {"type": "string", "description": "Publication date"},
            "content": {"type": "string", "description": "Main article content"},
            "summary": {"type": "string", "description": "Article summary or excerpt"},
            "categories": {"type": "array", "items": {"type": "string"}, "description": "Article categories"}
        },
        "required": ["title", "content"]
    }
    
    # Extract article data
    result = await semantic_extractor.extract_semantic_content(html_content, article_schema)
    
    return result
```

This example:
1. Creates necessary services including the AI service
2. Creates and initializes a semantic extractor
3. Defines an article schema with relevant fields
4. Uses AI-powered semantic extraction to understand content meaning

The semantic extractor integrates with the AI service to understand content context and extract structured data, which works particularly well for articles and other text-heavy content.

### Listing Data Extraction

Extract product listings by first analyzing structure and then extracting with patterns:

```python
async def extract_listing_data(html_content: str) -> Dict[str, List[Dict[str, Any]]]:
    # Create service registry and context
    registry = ServiceRegistry()
    context = StrategyContext(registry)
    
    # Initialize required services
    html_service = registry.get_or_create_service("html_service")
    
    # Create structural analyzer to identify listing patterns
    structural_analyzer = DOMStructuralAnalyzer(context)
    structural_analyzer.initialize()
    
    # Create pattern extractor for listing data
    pattern_extractor = DOMPatternExtractor(context)
    pattern_extractor.initialize()
    
    # Analyze structure first
    structure_info = structural_analyzer.analyze_structure(html_content)
    
    # Define listing item schema
    listing_item_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Product title"},
            "price": {"type": "number", "description": "Product price"},
            "url": {"type": "string", "description": "Product URL"},
            "image": {"type": "string", "description": "Product image"},
        },
        "required": ["title"]
    }
    
    # Extract listing data with container from structural analysis
    options = {
        "content_type": "html", 
        "clean_html": True,
        "container_selector": structure_info.get("listing_container", "")
    }
    
    result = pattern_extractor.extract(html_content, listing_schema, options)
    
    return {"items": result.get("items", [])}
```

This example demonstrates a two-step approach for listing extraction:
1. First, analyze the page structure to identify the listing container
2. Then, extract listing items using the container information

This approach is particularly effective for product listings, search results, and other pages with repeated structured elements.

## Pipeline Configuration

### Standard Pipeline Configuration

Create and configure a standard extraction pipeline:

```python
async def configure_standard_pipeline() -> None:
    # Create service registry and context
    registry = ServiceRegistry()
    context = StrategyContext(registry)
    
    # Initialize required services
    html_service = registry.get_or_create_service("html_service")
    session_manager = registry.get_or_create_service("session_manager")
    
    # Register required pipeline stages
    pipeline_registry = PipelineRegistry()
    registry.register_service("pipeline_registry", pipeline_registry)
    
    # Register extraction stages
    pipeline_registry.register_stage("structural_analysis", StructuralAnalysisStage)
    pipeline_registry.register_stage("metadata_extraction", MetadataExtractionStage)
    pipeline_registry.register_stage("pattern_extraction", PatternExtractionStage)
    # ... register other stages
    
    # Register a product extraction pipeline configuration
    product_pipeline_config = {
        "name": "ProductExtractionPipeline",
        "stages": [
            {"stage": "structural_analysis"},
            {"stage": "metadata_extraction"},
            {"stage": "pattern_extraction", "config": {"schema": "product_schema"}},
            {"stage": "content_normalization"},
            {"stage": "quality_evaluation"},
            {"stage": "schema_validation", "config": {"schema": "product_schema"}}
        ]
    }
    
    pipeline_registry.register_pipeline_config(product_pipeline_config)
    
    # Create a pipeline factory and instantiate a pipeline
    pipeline_factory = PipelineFactory(pipeline_registry, context)
    pipeline = pipeline_factory.create_pipeline("ProductExtractionPipeline")
    
    # In real usage: result = await pipeline.run({"url": "...", "content": html_content})
```

This example:
1. Sets up the service registry and pipeline registry
2. Registers all the extraction stages
3. Defines a standard product extraction pipeline configuration
4. Creates a pipeline factory and instantiates the pipeline

The pipeline approach allows for flexible, modular extraction workflows that can be configured for different content types.

### Custom Pipeline Creation

Create a specialized pipeline for real estate listings:

```python
async def create_custom_pipeline() -> None:
    # Create service registry and context
    registry = ServiceRegistry()
    context = StrategyContext(registry)
    
    # Initialize pipeline registry
    pipeline_registry = PipelineRegistry()
    registry.register_service("pipeline_registry", pipeline_registry)
    
    # Register extraction stages
    pipeline_registry.register_stage("structural_analysis", StructuralAnalysisStage)
    # ... register other stages
    
    # Define a custom real estate listing pipeline
    real_estate_pipeline_config = {
        "name": "RealEstateListingPipeline",
        "stages": [
            {
                "stage": "structural_analysis",
                "config": {
                    "identify_sections": True,
                    "detect_listing_container": True
                }
            },
            {
                "stage": "pattern_extraction", 
                "config": {
                    "schema": "real_estate_schema",
                    "selector_generation": "auto"
                }
            },
            # ... other stages with specialized configuration
        ]
    }
    
    # Register and create the pipeline
    pipeline_registry.register_pipeline_config(real_estate_pipeline_config)
    pipeline_factory = PipelineFactory(pipeline_registry, context)
    pipeline = pipeline_factory.create_pipeline("RealEstateListingPipeline")
```

This example demonstrates:
1. How to create a domain-specific extraction pipeline
2. How to configure stages with specialized options for the domain
3. The flexibility of the pipeline architecture for different extraction needs

### Pipeline Options

Customize pipeline behavior with options:

```python
async def set_pipeline_options() -> None:
    # Create and register necessary services and stages...
    
    # Define basic pipeline config
    basic_pipeline_config = {
        "name": "BasicExtractionPipeline",
        "stages": [
            {"stage": "structural_analysis"},
            {"stage": "pattern_extraction"},
            {"stage": "content_normalization"}
        ]
    }
    
    pipeline_registry.register_pipeline_config(basic_pipeline_config)
    
    # Create a pipeline with default options
    default_pipeline = pipeline_factory.create_pipeline("BasicExtractionPipeline")
    
    # Create a pipeline with customized options
    custom_options = {
        "structural_analysis": {
            "identify_sections": True,
            "detect_listing_container": True
        },
        "pattern_extraction": {
            "selector_generation": "advanced",
            "extract_attributes": True
        },
        "content_normalization": {
            "standardize_dates": True,
            "locale": "en-US"
        }
    }
    
    custom_pipeline = pipeline_factory.create_pipeline(
        "BasicExtractionPipeline", 
        stage_options=custom_options
    )
```

This example shows:
1. How to create a pipeline with default options
2. How to override options for specific stages
3. How different stages can be configured for specific extraction needs

### Handling Pipeline Results

Process and analyze extraction results from a pipeline:

```python
async def handle_pipeline_results() -> None:
    # In practice, you would run a pipeline and get results
    # Here we analyze a simulated result
    
    # Simulated pipeline result with quality metadata
    pipeline_result = {
        "url": "https://example.com/product/123",
        "extracted_data": {
            "title": "Premium Ergonomic Office Chair",
            "price": 299.99,
            # ... other fields
        },
        "_metadata": {
            "extraction_method": "pattern",
            "extraction_time": "2025-05-10T14:23:45",
            "quality_scores": {
                "completeness": 0.92,
                "consistency": 0.95,
                "confidence": 0.89,
                "overall": 0.92
            },
            "field_confidence": {
                "title": 0.99,
                "price": 0.98,
                # ... confidence for other fields
            },
            "validation_results": {
                "valid": True,
                "missing_required": [],
                "invalid_fields": []
            }
        }
    }
    
    # 1. Check overall extraction success
    quality_scores = pipeline_result.get("_metadata", {}).get("quality_scores", {})
    overall_quality = quality_scores.get("overall", 0)
    
    if overall_quality > 0.8:
        print("High-quality extraction result!")
        # Process automatically
    elif overall_quality > 0.5:
        print("Medium-quality extraction - may need validation")
        # Flag for review
    else:
        print("Low-quality extraction - manual review recommended")
        # Send to manual processing queue
    
    # 2. Check for low-confidence fields
    field_confidence = pipeline_result.get("_metadata", {}).get("field_confidence", {})
    low_confidence_fields = [
        field for field, confidence in field_confidence.items()
        if confidence < 0.8
    ]
    
    # 3. Check validation results
    validation = pipeline_result.get("_metadata", {}).get("validation_results", {})
    # Process based on validation results
```

This example demonstrates:
1. How to analyze pipeline results and quality metrics
2. How to implement decision logic based on extraction quality
3. How to identify specific fields that need attention

## Custom Extractor Implementation

### Creating a Specialized Extractor

Implement a specialized extractor for e-commerce products:

```python
class EcommerceSpecificExtractor(PatternExtractor):
    """Custom extractor specialized for e-commerce product pages."""
    
    def __init__(self, context=None):
        super().__init__(context)
        self.platform_patterns = {
            "shopify": {
                "title": [".product-title", "h1.title"],
                "price": [".price", "[data-product-price]"],
                # ... platform-specific patterns
            },
            "woocommerce": {
                "title": [".product_title", "h1.entry-title"],
                # ... platform-specific patterns
            },
            "magento": {
                "title": [".page-title", "h1.product-name"],
                # ... platform-specific patterns
            }
        }
    
    def can_handle(self, content: Any, content_type: str = "html") -> bool:
        """Check if this is an e-commerce product page."""
        if content_type.lower() != "html":
            return False
        
        # Simple heuristic to detect product pages
        if isinstance(content, str):
            return (
                "product" in content.lower() and 
                ("price" in content.lower() or 
                 "buy" in content.lower() or 
                 "cart" in content.lower())
            )
        
        return False
    
    def extract(self, content: Any, schema: Optional[Dict[str, Any]] = None, 
               options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract product information from an e-commerce page."""
        # ... extraction implementation
        
    def _detect_platform(self, soup) -> Optional[str]:
        """Detect the e-commerce platform from page HTML."""
        # Look for platform-specific indicators
        if soup.select("body.shopify-section") or "shopify" in str(soup):
            return "shopify"
        elif soup.select(".woocommerce") or "woocommerce" in str(soup):
            return "woocommerce"
        elif soup.select("[data-ui-id]") or "Magento" in str(soup):
            return "magento"
        return None
```

This example demonstrates:
1. How to create a specialized extractor by extending the base classes
2. How to implement platform detection for e-commerce sites
3. How to use platform-specific patterns for better extraction

### Using Custom Extractors

Use a custom extractor for specialized extraction:

```python
async def use_custom_extractor() -> None:
    # Create service registry and context
    registry = ServiceRegistry()
    context = StrategyContext(registry)
    
    # Initialize required services
    html_service = registry.get_or_create_service("html_service")
    
    # Create the custom extractor
    ecommerce_extractor = EcommerceSpecificExtractor(context)
    ecommerce_extractor.initialize()
    
    # Register the extractor with the service registry
    registry.register_service("ecommerce_extractor", ecommerce_extractor)
    
    # Sample product HTML
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Premium Ergonomic Office Chair - ErgoComfort</title>
        <meta name="description" content="High-quality ergonomic office chair with lumbar support and adjustable height.">
    </head>
    <body class="woocommerce">
        <!-- Product HTML content -->
    </body>
    </html>
    """
    
    # Use the extractor
    result = ecommerce_extractor.extract(sample_html)
    
    # Process the results
    print(f"Detected platform: {result.get('_metadata', {}).get('detected_platform')}")
    # ... process other extracted fields
```

This example shows:
1. How to initialize and use a custom extractor
2. How to register it with the service registry
3. How to process the specialized extraction results

### Registering Custom Extractors

Register a custom extractor with the fallback framework:

```python
async def register_custom_extractor() -> None:
    # Create service registry and context
    registry = ServiceRegistry()
    context = StrategyContext(registry)
    
    # Initialize the fallback framework
    fallback_registry = ExtractionFallbackRegistry()
    
    # Create and register the custom extractor
    ecommerce_extractor = EcommerceSpecificExtractor(context)
    fallback_registry.register_extractor("EcommerceExtractor", EcommerceSpecificExtractor)
    
    # Create a fallback chain with the custom extractor
    chain = fallback_registry.create_fallback_chain("EcommerceExtractor")
    
    # Use the fallback chain for extraction:
    # result = chain.extract(html_content, schema)
```

This example demonstrates:
1. How to register a custom extractor with the fallback framework
2. How to create a fallback chain that includes the custom extractor
3. How the fallback mechanism can use specialized extractors

## Schema Usage

### Defining Custom Schemas

Define a specialized schema for job listings:

```python
async def define_custom_schema() -> None:
    # Define a custom schema for job listings
    job_listing_schema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string", 
                "description": "Job title or position name"
            },
            "company": {
                "type": "string", 
                "description": "Company offering the job"
            },
            # ... other job listing fields
        },
        "required": ["title", "company", "description"]
    }
    
    # Create schema manager
    schema_manager = SchemaManager()
    schema_manager.initialize()
    
    # Register the schema
    schema_manager._schemas["job_listing"] = job_listing_schema
```

This example shows:
1. How to define a domain-specific schema with field types and descriptions
2. How to specify required fields
3. How to register the schema with the schema manager

### Registering Multiple Schemas

Register multiple schemas for different content types:

```python
async def register_schemas() -> None:
    # Create schema manager
    schema_manager = SchemaManager(context)
    schema_manager.initialize()
    registry.register_service("schema_manager", schema_manager)
    
    # Define schemas for different content types
    schemas = {
        "product": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "price": {"type": "number"},
                # ... product fields
            },
            "required": ["title", "price"]
        },
        "article": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "author": {"type": "string"},
                # ... article fields
            },
            "required": ["title", "content"]
        },
        "recipe": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "ingredients": {"type": "array", "items": {"type": "string"}},
                # ... recipe fields
            },
            "required": ["name", "ingredients", "instructions"]
        }
    }
    
    # Register all schemas
    for name, schema in schemas.items():
        schema_manager._schemas[name] = schema
    
    # Get a schema by name
    product_schema = schema_manager.get_schema("product")
```

This example demonstrates:
1. How to manage multiple schemas for different content types
2. How to register all schemas with the schema manager
3. How to retrieve schemas by name

### Validating Against Schemas

Validate extracted data against a schema:

```python
async def validate_against_schema() -> None:
    # Create schema manager and validator
    schema_manager = SchemaManager(context)
    schema_manager.initialize()
    registry.register_service("schema_manager", schema_manager)
    
    # Register a product schema
    product_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "price": {"type": "number"},
            # ... other fields
            "rating": {"type": "number", "minimum": 0, "maximum": 5},
            "reviews_count": {"type": "integer", "minimum": 0}
        },
        "required": ["title", "price"]
    }
    
    schema_manager._schemas["product"] = product_schema
    
    # Sample extracted data - Valid
    valid_data = {
        "title": "Premium Ergonomic Office Chair",
        "price": 299.99,
        # ... other valid fields
    }
    
    # Sample extracted data - Invalid (missing required field, invalid enum value)
    invalid_data = {
        "title": "Premium Ergonomic Office Chair",
        # Missing required price field
        "availability": "limited", # Invalid enum value
        "rating": 6.5, # Above maximum
        "reviews_count": -10 # Below minimum
    }
    
    # Validate the data
    valid_result = schema_manager.validate(valid_data, "product")
    invalid_result = schema_manager.validate(invalid_data, "product")
    
    # Check validation results
    print(f"Valid data validation: {valid_result.get('_metadata', {}).get('valid', False)}")
    print(f"Invalid data validation: {invalid_result.get('_metadata', {}).get('valid', False)}")
    print(f"Error: {invalid_result.get('_metadata', {}).get('error', 'No error')}")
```

This example shows:
1. How to validate extracted data against a schema
2. How to handle both valid and invalid data
3. How to process validation results

### Generating Schemas from Data

Generate a schema from sample data:

```python
async def generate_schema_from_data() -> None:
    # Create schema manager
    schema_manager = SchemaManager(context)
    schema_manager.initialize()
    
    # Sample data from which to infer a schema
    sample_data = {
        "product_name": "Bluetooth Wireless Earbuds",
        "price": 89.99,
        "currency": "USD",
        "description": "High-quality wireless earbuds with noise cancellation.",
        "rating": 4.5,
        "reviews_count": 1250,
        "in_stock": True,
        "features": [
            "Active noise cancellation",
            "Bluetooth 5.0",
            "8-hour battery life",
            "Water resistant"
        ],
        "technical_specs": {
            "battery": "350mAh",
            "weight": "5.6g per earbud",
            "colors": ["black", "white", "blue"],
            "charging_time": 1.5
        },
        "release_date": "2024-03-15"
    }
    
    # Generate schema from data
    generated_schema = schema_manager.create_schema(sample_data, "audio_product")
    
    # Register the generated schema
    schema_manager._schemas["audio_product"] = generated_schema
```

This example demonstrates:
1. How to automatically generate a schema from sample data
2. How the schema manager infers types and structure
3. How to register the generated schema for future use

## Extraction Quality

### Configuring Quality Evaluation

Configure and use the quality evaluator:

```python
async def configure_quality_evaluation() -> None:
    # Create quality evaluator
    quality_evaluator = QualityEvaluator(context)
    quality_evaluator.initialize({
        "completeness_weight": 0.3,
        "consistency_weight": 0.2,
        "confidence_weight": 0.3,
        "validation_weight": 0.2,
        "min_acceptable_quality": 0.6,
        
        "required_fields": {
            "product": ["title", "price"],
            "article": ["title", "content"],
            "listing": ["items"]
        },
        
        "field_validators": {
            "price": lambda x: isinstance(x, (int, float)) and x > 0,
            "title": lambda x: isinstance(x, str) and len(x) > 3,
            "description": lambda x: isinstance(x, str) and len(x) > 10
        }
    })
    
    # Register the evaluator
    registry.register_service("quality_evaluator", quality_evaluator)
    
    # Sample extracted data to evaluate
    extracted_data = {
        "title": "Premium Ergonomic Office Chair",
        "price": 299.99,
        "description": "High-quality ergonomic office chair with lumbar support and adjustable height.",
        "sku": "EC-123456",
        "brand": "ErgoComfort",
        "color": "",  # Empty field
        "features": []  # Empty list
    }
    
    # Product schema for validation
    product_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "price": {"type": "number"},
            # ... other fields
        },
        "required": ["title", "price", "description"]
    }
    
    # Evaluate quality
    quality_result = quality_evaluator.evaluate(extracted_data, product_schema)
    
    # Process quality evaluation results
    quality_scores = quality_result.get("_quality_scores", {})
    overall_quality = quality_scores.get("overall", 0)
    
    if overall_quality > 0.8:
        print("High-quality extraction result!")
    elif overall_quality > 0.5:
        print("Medium-quality extraction - may need validation")
    else:
        print("Low-quality extraction - manual review recommended")
```

This example shows:
1. How to configure the quality evaluator with weights and thresholds
2. How to define content-specific required fields
3. How to implement custom field validators
4. How to evaluate extraction quality and process the results

### Interpreting Quality Metrics

Analyze and interpret quality metrics:

```python
async def interpret_quality_metrics() -> None:
    # Simulated quality metrics from extraction
    quality_metrics = {
        "completeness": 0.85,
        "consistency": 0.92,
        "confidence": 0.78,
        "validation": 1.0,
        "overall": 0.87,
        "field_metrics": {
            "title": {"confidence": 0.95, "valid": True},
            "price": {"confidence": 0.99, "valid": True},
            "description": {"confidence": 0.85, "valid": True},
            "category": {"confidence": 0.65, "valid": True, "issues": ["unusual_value"]},
            "specifications": {"confidence": 0.45, "valid": False, "issues": ["incomplete", "inconsistent_format"]}
        }
    }
    
    # Interpret overall quality
    overall = quality_metrics.get("overall", 0)
    
    if overall >= 0.9:
        print("Excellent quality (90%+): Highly reliable extraction")
        print("Recommendation: Proceed with automated processing")
    elif overall >= 0.8:
        print("Good quality (80-89%): Reliable with minor issues")
        print("Recommendation: Automated processing with spot checks")
    # ... other quality thresholds
    
    # Categorize fields by quality
    excellent_fields = []
    good_fields = []
    review_fields = []
    problem_fields = []
    
    for field, metrics in field_metrics.items():
        confidence = metrics.get("confidence", 0)
        valid = metrics.get("valid", False)
        issues = metrics.get("issues", [])
        
        if confidence >= 0.9 and valid and not issues:
            excellent_fields.append(field)
        elif confidence >= 0.8 and valid:
            good_fields.append(field)
        elif confidence >= 0.6 or not valid:
            review_fields.append((field, confidence, issues))
        else:
            problem_fields.append((field, confidence, issues))
    
    # Process field categories
    # ...
```

This example demonstrates:
1. How to categorize overall extraction quality
2. How to analyze field-level metrics
3. How to identify fields that need review or improvement

### Improving Extraction Quality

Implement techniques to improve extraction quality:

```python
async def improve_extraction_quality() -> None:
    # Strategies to improve extraction quality
    improvement_strategies = [
        {
            "strategy": "Pre-extraction optimization",
            "techniques": [
                "Clean HTML by removing scripts, styles, and comments",
                "Identify main content area to exclude navigation, footer, etc.",
                "Normalize text formatting before extraction",
                "Fix incomplete or malformed HTML"
            ]
        },
        {
            "strategy": "Multi-extractor approach",
            "techniques": [
                "Use pattern extraction and semantic extraction in parallel",
                "Merge results based on confidence scores",
                "Fall back to alternative extractors when primary fails",
                "Use specialized extractors for specific content types"
            ]
        },
        # ... other strategies
    ]
    
    # Example implementation: Multi-extractor approach with result merging
    
    # Simulated results from different extractors
    pattern_result = {
        "title": "Premium Ergonomic Office Chair",
        "price": 299.99,
        "brand": "ErgoComfort",
        "sku": "EC-123456",
        "_metadata": {
            "confidence": {
                "title": 0.95,
                "price": 0.99,
                "brand": 0.90,
                "sku": 0.99
            }
        }
    }
    
    semantic_result = {
        "title": "Premium Ergonomic Office Chair",
        "price": 299.99,
        "description": "High-quality ergonomic office chair with lumbar support and adjustable height.",
        "features": ["Lumbar support", "Adjustable height", "Ergonomic design", "Premium materials"],
        "_metadata": {
            "confidence": {
                "title": 0.97,
                "price": 0.85,
                "description": 0.93,
                "features": 0.88
            }
        }
    }
    
    # Merge results based on confidence
    merged_result = {}
    merged_confidence = {}
    
    # Combine all keys
    all_fields = set(list(pattern_result.keys()) + list(semantic_result.keys()))
    all_fields.discard("_metadata")  # Don't include metadata in field list
    
    for field in all_fields:
        pattern_confidence = pattern_result.get("_metadata", {}).get("confidence", {}).get(field, 0)
        semantic_confidence = semantic_result.get("_metadata", {}).get("confidence", {}).get(field, 0)
        
        # Choose value from extractor with higher confidence
        if field in pattern_result and field in semantic_result:
            if pattern_confidence >= semantic_confidence:
                merged_result[field] = pattern_result[field]
                merged_confidence[field] = pattern_confidence
            else:
                merged_result[field] = semantic_result[field]
                merged_confidence[field] = semantic_confidence
        elif field in pattern_result:
            merged_result[field] = pattern_result[field]
            merged_confidence[field] = pattern_confidence
        elif field in semantic_result:
            merged_result[field] = semantic_result[field]
            merged_confidence[field] = semantic_confidence
    
    # Add merged confidence to metadata
    merged_result["_metadata"] = {
        "confidence": merged_confidence,
        "extraction_method": "merged",
        "avg_confidence": sum(merged_confidence.values()) / len(merged_confidence) if merged_confidence else 0
    }
```

This example shows:
1. Different strategies for improving extraction quality
2. How to implement a multi-extractor approach
3. How to merge results based on confidence
4. How to track and report merged extraction quality