"""
SmartScrape Universal Extraction Framework Examples

This module demonstrates common usage patterns for the SmartScrape
extraction framework, including basic extraction, pipeline configuration,
custom extractors, schema usage, and quality evaluation.
"""

import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup

# Core imports
from strategies.core.strategy_context import StrategyContext
from core.service_registry import ServiceRegistry

# Extraction imports
from extraction.core.extraction_interface import BaseExtractor, PatternExtractor
from extraction.pattern_extractor import DOMPatternExtractor
from extraction.semantic_extractor import AISemanticExtractor
from extraction.structural_analyzer import DOMStructuralAnalyzer
from extraction.metadata_extractor import CompositeMetadataExtractor
from extraction.content_normalizer_impl import ContentNormalizerImpl
from extraction.quality_evaluator import QualityEvaluator
from extraction.schema_manager import SchemaManager
from extraction.schema_validator import SchemaValidator
from extraction.fallback_framework import ExtractionFallbackChain, create_html_extraction_chain

# Pipeline imports
from core.pipeline.registry import PipelineRegistry
from core.pipeline.factory import PipelineFactory
from extraction.stages.structural_analysis_stage import StructuralAnalysisStage
from extraction.stages.metadata_extraction_stage import MetadataExtractionStage
from extraction.stages.pattern_extraction_stage import PatternExtractionStage
from extraction.stages.semantic_extraction_stage import SemanticExtractionStage
from extraction.stages.content_normalization_stage import ContentNormalizationStage
from extraction.stages.quality_evaluation_stage import QualityEvaluationStage
from extraction.stages.schema_validation_stage import SchemaValidationStage

#############################################
# 1. Basic HTML Extraction Examples
#############################################

async def extract_product_information(html_content: str) -> Dict[str, Any]:
    """
    Extract product information from an HTML product page.
    
    Args:
        html_content: HTML content of a product page
        
    Returns:
        Dictionary containing extracted product information
    """
    print("Example: Extract Product Information")
    
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
    
    # Print the result (in practice, you'd return it)
    print(f"Extracted {len(result.keys()) - 1} product fields:")
    for key, value in result.items():
        if key != "_metadata":
            print(f"  {key}: {value}")
    
    return result


async def extract_article_content(html_content: str) -> Dict[str, Any]:
    """
    Extract article content from an HTML page.
    
    Args:
        html_content: HTML content of an article page
        
    Returns:
        Dictionary containing extracted article information
    """
    print("\nExample: Extract Article Content")
    
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
    
    # Print the result (in practice, you'd return it)
    print(f"Extracted article with {len(result.keys())} fields:")
    for key, value in result.items():
        if key == "content":
            print(f"  {key}: {value[:100]}... ({len(value)} chars)")
        else:
            print(f"  {key}: {value}")
    
    return result


async def extract_listing_data(html_content: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract product listings from an HTML page.
    
    Args:
        html_content: HTML content of a listing page
        
    Returns:
        Dictionary containing list of extracted products
    """
    print("\nExample: Extract Listing Data")
    
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
    
    # Extract listing data
    listing_schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": listing_item_schema
            }
        }
    }
    
    # Use container selector from structural analysis if available
    options = {
        "content_type": "html", 
        "clean_html": True,
        "container_selector": structure_info.get("listing_container", "")
    }
    
    result = pattern_extractor.extract(html_content, listing_schema, options)
    
    # Print the result (in practice, you'd return it)
    items = result.get("items", [])
    print(f"Extracted {len(items)} listing items:")
    for i, item in enumerate(items[:3]):  # Show first 3 items
        print(f"  Item {i+1}:")
        for key, value in item.items():
            print(f"    {key}: {value}")
    
    if len(items) > 3:
        print(f"  ... and {len(items) - 3} more items")
    
    return {"items": items}


#############################################
# 2. Pipeline Configuration Examples
#############################################

async def configure_standard_pipeline() -> None:
    """Configure and use a standard extraction pipeline."""
    print("\nExample: Configure Standard Pipeline")
    
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
    pipeline_registry.register_stage("semantic_extraction", SemanticExtractionStage)
    pipeline_registry.register_stage("content_normalization", ContentNormalizationStage)
    pipeline_registry.register_stage("quality_evaluation", QualityEvaluationStage)
    pipeline_registry.register_stage("schema_validation", SchemaValidationStage)
    
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
    
    # Create a pipeline factory
    pipeline_factory = PipelineFactory(pipeline_registry, context)
    
    # Create a pipeline instance
    pipeline = pipeline_factory.create_pipeline("ProductExtractionPipeline")
    
    print("Successfully configured ProductExtractionPipeline with stages:")
    for stage in pipeline.stages:
        print(f"  - {stage.name}")
    
    # In a real scenario, you would run the pipeline:
    # result = await pipeline.run({"url": "https://example.com/product", "content": html_content})


async def create_custom_pipeline() -> None:
    """Create and register a custom extraction pipeline."""
    print("\nExample: Create Custom Pipeline")
    
    # Create service registry and context
    registry = ServiceRegistry()
    context = StrategyContext(registry)
    
    # Initialize required services
    pipeline_registry = PipelineRegistry()
    registry.register_service("pipeline_registry", pipeline_registry)
    
    # Register extraction stages
    pipeline_registry.register_stage("structural_analysis", StructuralAnalysisStage)
    pipeline_registry.register_stage("metadata_extraction", MetadataExtractionStage)
    pipeline_registry.register_stage("pattern_extraction", PatternExtractionStage)
    pipeline_registry.register_stage("semantic_extraction", SemanticExtractionStage)
    pipeline_registry.register_stage("content_normalization", ContentNormalizationStage)
    pipeline_registry.register_stage("quality_evaluation", QualityEvaluationStage)
    pipeline_registry.register_stage("schema_validation", SchemaValidationStage)
    
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
            {
                "stage": "semantic_extraction",
                "config": {
                    "extract_descriptions": True,
                    "extract_amenities": True,
                    "model": "gpt-4o"
                }
            },
            {
                "stage": "content_normalization",
                "config": {
                    "standardize_addresses": True,
                    "standardize_prices": True,
                    "standardize_areas": True
                }
            },
            {
                "stage": "quality_evaluation",
                "config": {
                    "min_confidence": 0.7,
                    "required_fields": ["address", "price", "bedrooms", "bathrooms", "area"]
                }
            },
            {
                "stage": "schema_validation",
                "config": {
                    "schema": "real_estate_schema",
                    "strict": True
                }
            }
        ]
    }
    
    # Register the custom pipeline
    pipeline_registry.register_pipeline_config(real_estate_pipeline_config)
    
    # Create a pipeline factory
    pipeline_factory = PipelineFactory(pipeline_registry, context)
    
    # Create a pipeline instance
    pipeline = pipeline_factory.create_pipeline("RealEstateListingPipeline")
    
    print("Successfully created custom RealEstateListingPipeline with stages:")
    for i, stage in enumerate(pipeline.stages):
        print(f"  {i+1}. {stage.name} - {stage._config if hasattr(stage, '_config') else 'No config'}")
    
    # In a real scenario, you would run this pipeline:
    # result = await pipeline.run({"url": "https://example.com/property", "content": html_content})


async def set_pipeline_options() -> None:
    """Set and override pipeline options."""
    print("\nExample: Set Pipeline Options")
    
    # Create service registry and context
    registry = ServiceRegistry()
    context = StrategyContext(registry)
    
    # Initialize pipeline registry
    pipeline_registry = PipelineRegistry()
    registry.register_service("pipeline_registry", pipeline_registry)
    
    # Register stages
    pipeline_registry.register_stage("structural_analysis", StructuralAnalysisStage)
    pipeline_registry.register_stage("pattern_extraction", PatternExtractionStage)
    pipeline_registry.register_stage("content_normalization", ContentNormalizationStage)
    
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
    
    # Create a pipeline factory
    pipeline_factory = PipelineFactory(pipeline_registry, context)
    
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
    
    # Show difference
    print("Default pipeline options:")
    for stage in default_pipeline.stages:
        print(f"  - {stage.name}: {stage._config if hasattr(stage, '_config') else '{}'}")
    
    print("\nCustomized pipeline options:")
    for stage in custom_pipeline.stages:
        print(f"  - {stage.name}: {stage._config if hasattr(stage, '_config') else '{}'}")


async def handle_pipeline_results() -> None:
    """Process and handle pipeline extraction results."""
    print("\nExample: Handle Pipeline Results")
    
    # In practice, you would run a pipeline and get real results
    # Here we'll simulate pipeline results
    
    # Simulated pipeline result with quality metadata
    pipeline_result = {
        "url": "https://example.com/product/123",
        "extracted_data": {
            "title": "Premium Ergonomic Office Chair",
            "price": 299.99,
            "description": "High-quality ergonomic office chair with lumbar support and adjustable height.",
            "brand": "ErgoComfort",
            "sku": "EC-123456",
            "rating": 4.7,
            "reviews_count": 253,
            "availability": "in_stock"
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
                "description": 0.95,
                "brand": 0.85,
                "sku": 0.99,
                "rating": 0.97,
                "reviews_count": 0.99,
                "availability": 0.95
            },
            "validation_results": {
                "valid": True,
                "missing_required": [],
                "invalid_fields": []
            }
        }
    }
    
    # Process the pipeline result
    print("Processing pipeline result:")
    
    # 1. Check overall extraction success
    quality_scores = pipeline_result.get("_metadata", {}).get("quality_scores", {})
    overall_quality = quality_scores.get("overall", 0)
    
    if overall_quality > 0.8:
        print("  High-quality extraction result!")
    elif overall_quality > 0.5:
        print("  Medium-quality extraction result - may need validation")
    else:
        print("  Low-quality extraction result - manual review recommended")
    
    # 2. Check for low-confidence fields
    field_confidence = pipeline_result.get("_metadata", {}).get("field_confidence", {})
    low_confidence_fields = []
    
    for field, confidence in field_confidence.items():
        if confidence < 0.8:
            low_confidence_fields.append((field, confidence))
    
    if low_confidence_fields:
        print("  Fields with low confidence:")
        for field, confidence in low_confidence_fields:
            print(f"    - {field}: {confidence:.2f}")
    else:
        print("  All fields have high confidence")
    
    # 3. Check validation results
    validation = pipeline_result.get("_metadata", {}).get("validation_results", {})
    if validation.get("valid", False):
        print("  Data is valid against schema")
    else:
        print("  Data validation issues:")
        if validation.get("missing_required"):
            print(f"    Missing required fields: {', '.join(validation.get('missing_required'))}")
        if validation.get("invalid_fields"):
            print(f"    Invalid fields: {', '.join(validation.get('invalid_fields'))}")
    
    # 4. Extract the data payload
    extracted_data = pipeline_result.get("extracted_data", {})
    print(f"  Extracted {len(extracted_data)} fields from {pipeline_result.get('url')}")
    
    # 5. Process data based on quality
    if overall_quality > 0.7 and validation.get("valid", False):
        print("  Proceeding with automated processing")
        # In a real system, you might store in database, trigger follow-up actions, etc.
    else:
        print("  Flagging for manual review")
        # In a real system, you might add to a review queue


#############################################
# 3. Custom Extractor Implementation
#############################################

class EcommerceSpecificExtractor(PatternExtractor):
    """
    Custom extractor specialized for e-commerce product pages.
    
    This extractor recognizes common e-commerce patterns and
    extracts structured product data from various platforms.
    """
    
    def __init__(self, context=None):
        super().__init__(context)
        self.platform_patterns = {
            "shopify": {
                "title": [".product-title", "h1.title"],
                "price": [".price", "[data-product-price]"],
                "description": [".product-description", "[data-product-description]"],
                "images": [".product-image", "[data-product-image]"]
            },
            "woocommerce": {
                "title": [".product_title", "h1.entry-title"],
                "price": [".price", ".amount"],
                "description": [".woocommerce-product-details__short-description", ".description"],
                "images": [".woocommerce-product-gallery__image", "img.wp-post-image"]
            },
            "magento": {
                "title": [".page-title", "h1.product-name"],
                "price": [".price", ".product-info-price"],
                "description": [".product-info-description", ".value", ".description"],
                "images": [".gallery-image", ".fotorama__img"]
            }
        }
    
    def can_handle(self, content: Any, content_type: str = "html") -> bool:
        """Check if this extractor can handle the given content."""
        if content_type.lower() != "html":
            return False
        
        # Check if this looks like an e-commerce product page
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
        if not self._initialized:
            self.initialize()
        
        options = options or {}
        result = {"_metadata": {}}
        
        try:
            # Parse HTML
            html_service = self._get_service("html_service")
            soup = BeautifulSoup(content, "lxml")
            
            # Detect platform
            platform = self._detect_platform(soup)
            result["_metadata"]["detected_platform"] = platform
            
            # Generate patterns based on detected platform
            if platform and platform in self.platform_patterns:
                patterns = self.platform_patterns[platform]
            else:
                # Fall back to generic patterns
                patterns = self.generate_patterns(soup, schema)
            
            # Match patterns
            extracted_data = self.match_patterns(soup, patterns)
            result.update(extracted_data)
            
            return result
            
        except Exception as e:
            result["_metadata"]["error"] = str(e)
            result["_metadata"]["success"] = False
            return result
    
    def generate_patterns(self, content: Any, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate patterns for extracting data from e-commerce pages."""
        # For simplicity, we'll return some generic e-commerce selectors
        return {
            "title": ["h1", ".product-title", ".product-name", "[itemprop='name']"],
            "price": [".price", ".product-price", "[itemprop='price']"],
            "description": [".description", ".product-description", "[itemprop='description']"],
            "images": [".product-image", ".product-gallery img", "[itemprop='image']"],
            "sku": [".sku", ".product-sku", "[itemprop='sku']"],
            "brand": [".brand", ".product-brand", "[itemprop='brand']"]
        }
    
    def match_patterns(self, content: Any, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Match patterns against e-commerce product pages."""
        result = {}
        
        for field, selectors in patterns.items():
            if isinstance(selectors, list):
                # Try each selector until one works
                for selector in selectors:
                    elements = content.select(selector)
                    if elements:
                        if field == "images":
                            # For images, extract all matches and their src attributes
                            image_urls = []
                            for img in elements:
                                if img.name == "img":
                                    src = img.get("src", "")
                                    if src:
                                        image_urls.append(src)
                                else:
                                    img_tags = img.find_all("img")
                                    for img_tag in img_tags:
                                        src = img_tag.get("src", "")
                                        if src:
                                            image_urls.append(src)
                            
                            if image_urls:
                                result[field] = image_urls
                                break
                        elif field == "price":
                            # For price, convert to number
                            price_text = elements[0].get_text(strip=True)
                            price_text = price_text.replace("$", "").replace(",", "")
                            try:
                                result[field] = float(price_text)
                                break
                            except ValueError:
                                # Try next selector
                                continue
                        else:
                            # For other fields, get text content
                            result[field] = elements[0].get_text(strip=True)
                            break
        
        return result
    
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


async def use_custom_extractor() -> None:
    """Use the custom e-commerce extractor."""
    print("\nExample: Use Custom Extractor")
    
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
    
    # Sample product HTML (simplified for example)
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Premium Ergonomic Office Chair - ErgoComfort</title>
        <meta name="description" content="High-quality ergonomic office chair with lumbar support and adjustable height.">
    </head>
    <body class="woocommerce">
        <div class="product">
            <h1 class="product-title">Premium Ergonomic Office Chair</h1>
            <div class="product-price">$299.99</div>
            <div class="product-description">
                High-quality ergonomic office chair with lumbar support and adjustable height.
            </div>
            <div class="product-gallery">
                <img src="chair-main.jpg" class="product-image" alt="Office Chair">
                <img src="chair-side.jpg" class="product-image" alt="Office Chair Side View">
            </div>
            <div class="product-meta">
                <span class="sku">SKU: EC-123456</span>
                <span class="brand">Brand: ErgoComfort</span>
            </div>
            <button class="add-to-cart">Add to Cart</button>
        </div>
    </body>
    </html>
    """
    
    # Use the extractor
    result = ecommerce_extractor.extract(sample_html)
    
    # Print results
    print(f"Custom e-commerce extractor results:")
    print(f"  Detected platform: {result.get('_metadata', {}).get('detected_platform')}")
    
    for key, value in result.items():
        if key != "_metadata":
            if isinstance(value, list):
                print(f"  {key}: {len(value)} items")
                for i, item in enumerate(value[:2]):
                    print(f"    - {item}")
                if len(value) > 2:
                    print(f"    - ... and {len(value) - 2} more")
            else:
                print(f"  {key}: {value}")


async def register_custom_extractor() -> None:
    """Register a custom extractor with the extraction framework."""
    print("\nExample: Register Custom Extractor")
    
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
    
    print(f"Registered custom extractor with fallback framework")
    print(f"Created fallback chain with {len(chain.extractors)} extractors")
    
    # In practice, you would use this chain for extraction:
    # result = chain.extract(html_content, schema)


#############################################
# 4. Schema Usage Examples
#############################################

async def define_custom_schema() -> None:
    """Define a custom extraction schema."""
    print("\nExample: Define Custom Schema")
    
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
            "location": {
                "type": "string", 
                "description": "Job location (city, state, remote, etc.)"
            },
            "salary": {
                "type": "object",
                "properties": {
                    "min": {"type": "number"},
                    "max": {"type": "number"},
                    "currency": {"type": "string"},
                    "period": {"type": "string", "enum": ["hourly", "daily", "weekly", "monthly", "yearly"]}
                }
            },
            "description": {
                "type": "string", 
                "description": "Full job description"
            },
            "requirements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of job requirements"
            },
            "benefits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of job benefits"
            },
            "application_url": {
                "type": "string", 
                "description": "URL to apply for the job"
            },
            "posted_date": {
                "type": "string", 
                "format": "date", 
                "description": "Date the job was posted"
            },
            "employment_type": {
                "type": "string", 
                "enum": ["full-time", "part-time", "contract", "temporary", "internship"],
                "description": "Type of employment"
            }
        },
        "required": ["title", "company", "description"]
    }
    
    # Create schema manager
    schema_manager = SchemaManager()
    schema_manager.initialize()
    
    # Register the schema
    schema_manager._schemas["job_listing"] = job_listing_schema
    
    print(f"Defined and registered custom job listing schema with fields:")
    for field, definition in job_listing_schema["properties"].items():
        if field in job_listing_schema.get("required", []):
            print(f"  - {field} (required): {definition.get('description', '')}")
        else:
            print(f"  - {field}: {definition.get('description', '')}")


async def register_schemas() -> None:
    """Register multiple schemas with the schema manager."""
    print("\nExample: Register Multiple Schemas")
    
    # Create service registry and context
    registry = ServiceRegistry()
    context = StrategyContext(registry)
    
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
                "description": {"type": "string"},
                # More fields...
            },
            "required": ["title", "price"]
        },
        "article": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "author": {"type": "string"},
                "content": {"type": "string"},
                # More fields...
            },
            "required": ["title", "content"]
        },
        "recipe": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "ingredients": {"type": "array", "items": {"type": "string"}},
                "instructions": {"type": "array", "items": {"type": "string"}},
                "prepTime": {"type": "string"},
                "cookTime": {"type": "string"},
                "servings": {"type": "number"},
                # More fields...
            },
            "required": ["name", "ingredients", "instructions"]
        }
    }
    
    # Register all schemas
    for name, schema in schemas.items():
        schema_manager._schemas[name] = schema
    
    # Verify schemas are registered
    for name in schemas.keys():
        if name in schema_manager._schemas:
            print(f"Schema '{name}' successfully registered")
        else:
            print(f"Failed to register schema '{name}'")
    
    # Get a schema by name
    product_schema = schema_manager.get_schema("product")
    if product_schema:
        print(f"Retrieved product schema with {len(product_schema['properties'])} properties")
    else:
        print("Failed to retrieve product schema")


async def validate_against_schema() -> None:
    """Validate extracted data against a schema."""
    print("\nExample: Validate Against Schema")
    
    # Create service registry and context
    registry = ServiceRegistry()
    context = StrategyContext(registry)
    
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
            "description": {"type": "string"},
            "sku": {"type": "string"},
            "availability": {"type": "string", "enum": ["in_stock", "out_of_stock", "preorder"]},
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
        "description": "High-quality ergonomic office chair with lumbar support and adjustable height.",
        "sku": "EC-123456",
        "availability": "in_stock",
        "rating": 4.7,
        "reviews_count": 253
    }
    
    # Sample extracted data - Invalid (missing required field, invalid enum value)
    invalid_data = {
        "title": "Premium Ergonomic Office Chair",
        # Missing required price field
        "description": "High-quality ergonomic office chair with lumbar support and adjustable height.",
        "sku": "EC-123456",
        "availability": "limited", # Invalid enum value
        "rating": 6.5, # Above maximum
        "reviews_count": -10 # Below minimum
    }
    
    # Validate the data
    valid_result = schema_manager.validate(valid_data, "product")
    invalid_result = schema_manager.validate(invalid_data, "product")
    
    print("Valid data validation result:")
    print(f"  Valid: {valid_result.get('_metadata', {}).get('valid', False)}")
    
    print("\nInvalid data validation result:")
    print(f"  Valid: {invalid_result.get('_metadata', {}).get('valid', False)}")
    print(f"  Error: {invalid_result.get('_metadata', {}).get('error', 'No error')}")


async def generate_schema_from_data() -> None:
    """Generate a schema from sample data."""
    print("\nExample: Generate Schema from Data")
    
    # Create service registry and context
    registry = ServiceRegistry()
    context = StrategyContext(registry)
    
    # Create schema manager
    schema_manager = SchemaManager(context)
    schema_manager.initialize()
    registry.register_service("schema_manager", schema_manager)
    
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
    
    # Print generated schema
    print("Generated schema from sample data:")
    
    if "type" in generated_schema:
        print(f"  Type: {generated_schema['type']}")
    
    if "properties" in generated_schema:
        print("  Properties:")
        for prop_name, prop_def in generated_schema["properties"].items():
            prop_type = prop_def.get("type", "unknown")
            if prop_type == "object" and "properties" in prop_def:
                print(f"    {prop_name} (object):")
                for sub_prop, sub_def in prop_def["properties"].items():
                    print(f"      - {sub_prop}: {sub_def.get('type')}")
            elif prop_type == "array" and "items" in prop_def:
                print(f"    {prop_name} (array of {prop_def['items'].get('type', 'items')})")
            else:
                print(f"    {prop_name}: {prop_type}")
    
    if "required" in generated_schema:
        print(f"  Required fields: {', '.join(generated_schema['required'])}")
    
    # Verify the schema can be used
    schema_manager._schemas["audio_product"] = generated_schema
    print(f"\nSchema 'audio_product' registered with {len(generated_schema.get('properties', {}))} properties")


#############################################
# 5. Extraction Quality Examples
#############################################

async def configure_quality_evaluation() -> None:
    """Configure and use quality evaluation for extraction results."""
    print("\nExample: Configure Quality Evaluation")
    
    # Create service registry and context
    registry = ServiceRegistry()
    context = StrategyContext(registry)
    
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
            "description": {"type": "string"},
            "sku": {"type": "string"},
            "brand": {"type": "string"},
            "color": {"type": "string"},
            "features": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["title", "price", "description"]
    }
    
    # Evaluate quality
    quality_result = quality_evaluator.evaluate(extracted_data, product_schema)
    
    # Print quality evaluation results
    print("Quality evaluation results:")
    
    # Overall quality score
    quality_scores = quality_result.get("_quality_scores", {})
    print(f"  Overall quality score: {quality_scores.get('overall', 0):.2f}")
    
    # Detailed scores
    print("  Detailed scores:")
    for metric, score in quality_scores.items():
        if metric != "overall":
            print(f"    - {metric}: {score:.2f}")
    
    # Field-level quality
    field_quality = quality_result.get("_field_quality", {})
    print("\n  Field-level quality:")
    for field, quality in field_quality.items():
        if isinstance(quality, dict):
            confidence = quality.get("confidence", 0)
            valid = quality.get("valid", False)
            issues = quality.get("issues", [])
            
            status = "✅ Good" if confidence > 0.8 and valid else "⚠️ Needs review"
            print(f"    - {field}: {confidence:.2f} - {status}")
            
            if issues:
                print(f"      Issues: {', '.join(issues)}")
    
    # Any fields needing attention
    low_quality_fields = [
        field for field, quality in field_quality.items()
        if isinstance(quality, dict) and (
            quality.get("confidence", 0) < 0.8 or
            not quality.get("valid", True) or
            quality.get("issues", [])
        )
    ]
    
    if low_quality_fields:
        print("\n  Fields needing attention:")
        for field in low_quality_fields:
            print(f"    - {field}")
    else:
        print("\n  All fields have acceptable quality")


async def interpret_quality_metrics() -> None:
    """Interpret quality metrics from extraction results."""
    print("\nExample: Interpret Quality Metrics")
    
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
    print("Overall extraction quality interpretation:")
    overall = quality_metrics.get("overall", 0)
    
    if overall >= 0.9:
        print("  Excellent quality (90%+): Highly reliable extraction")
        print("  Recommendation: Proceed with automated processing")
    elif overall >= 0.8:
        print("  Good quality (80-89%): Reliable with minor issues")
        print("  Recommendation: Automated processing with spot checks")
    elif overall >= 0.7:
        print("  Acceptable quality (70-79%): Usable but with some issues")
        print("  Recommendation: Review fields with low confidence")
    elif overall >= 0.5:
        print("  Fair quality (50-69%): Significant issues present")
        print("  Recommendation: Manual review recommended before use")
    else:
        print("  Poor quality (<50%): Major extraction problems")
        print("  Recommendation: Re-extraction or manual processing required")
    
    # Interpret individual metrics
    print("\nIndividual quality metrics interpretation:")
    
    completeness = quality_metrics.get("completeness", 0)
    print(f"  Completeness: {completeness:.2f}")
    if completeness < 0.7:
        print("    ⚠️ Missing data: Significant fields may be missing")
    
    consistency = quality_metrics.get("consistency", 0)
    print(f"  Consistency: {consistency:.2f}")
    if consistency < 0.8:
        print("    ⚠️ Inconsistent data: Format or value inconsistencies detected")
    
    confidence = quality_metrics.get("confidence", 0)
    print(f"  Confidence: {confidence:.2f}")
    if confidence < 0.7:
        print("    ⚠️ Low confidence: Extractor uncertain about some fields")
    
    validation = quality_metrics.get("validation", 0)
    print(f"  Validation: {validation:.2f}")
    if validation < 1.0:
        print("    ⚠️ Validation issues: Some fields failed schema validation")
    
    # Field-level interpretation
    print("\nField-level quality interpretation:")
    field_metrics = quality_metrics.get("field_metrics", {})
    
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
    
    if excellent_fields:
        print(f"  Excellent fields: {', '.join(excellent_fields)}")
    
    if good_fields:
        print(f"  Good fields: {', '.join(good_fields)}")
    
    if review_fields:
        print("  Fields needing review:")
        for field, confidence, issues in review_fields:
            print(f"    - {field} ({confidence:.2f})")
            if issues:
                print(f"      Issues: {', '.join(issues)}")
    
    if problem_fields:
        print("  Problem fields (likely incorrect):")
        for field, confidence, issues in problem_fields:
            print(f"    - {field} ({confidence:.2f})")
            if issues:
                print(f"      Issues: {', '.join(issues)}")


async def improve_extraction_quality() -> None:
    """Techniques to improve extraction quality."""
    print("\nExample: Improve Extraction Quality")
    
    # Ways to improve extraction quality
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
        {
            "strategy": "Schema refinement",
            "techniques": [
                "Add more detailed field descriptions for semantic extraction",
                "Provide example values for each field",
                "Add field validators for common formats (emails, prices, etc.)",
                "Use more specific field types (e.g., 'price' instead of just 'number')"
            ]
        },
        {
            "strategy": "Post-extraction normalization",
            "techniques": [
                "Standardize formats for dates, prices, measurements",
                "Clean text fields by removing excess whitespace, HTML, etc.",
                "Validate and fix common format issues",
                "Apply field-specific transformations (lowercase emails, format phone numbers)"
            ]
        },
        {
            "strategy": "Quality-based retry logic",
            "techniques": [
                "Re-extract fields with low confidence",
                "Try different extraction methods for problem fields",
                "Use more powerful AI models for difficult content",
                "Simplify extraction targets when full extraction fails"
            ]
        }
    ]
    
    # Print improvement strategies
    print("Strategies to improve extraction quality:")
    
    for i, strategy in enumerate(improvement_strategies, 1):
        print(f"\n{i}. {strategy['strategy']}:")
        for technique in strategy['techniques']:
            print(f"   - {technique}")
    
    # Example implementation of a quality improvement technique
    print("\nImplementation example: Multi-extractor approach with result merging")
    
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
    
    # Print merged result
    print("Merged extraction result:")
    for field, value in merged_result.items():
        if field != "_metadata":
            confidence = merged_result["_metadata"]["confidence"].get(field, 0)
            source = "Pattern" if field in pattern_result and (field not in semantic_result or pattern_result.get("_metadata", {}).get("confidence", {}).get(field, 0) >= semantic_result.get("_metadata", {}).get("confidence", {}).get(field, 0)) else "Semantic"
            
            if isinstance(value, list):
                print(f"  {field} ({source}, {confidence:.2f}): {len(value)} items")
            else:
                print(f"  {field} ({source}, {confidence:.2f}): {value}")
    
    print(f"  Average confidence: {merged_result['_metadata']['avg_confidence']:.2f}")


#############################################
# Main Entry Point
#############################################

async def main():
    """Run all examples."""
    print("=== SmartScrape Universal Extraction Framework Examples ===\n")
    
    # Basic HTML extraction examples
    sample_product_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Premium Ergonomic Office Chair - ErgoComfort</title>
        <meta name="description" content="High-quality ergonomic office chair with lumbar support and adjustable height.">
    </head>
    <body>
        <div class="product">
            <h1 class="product-title">Premium Ergonomic Office Chair</h1>
            <div class="product-price">$299.99</div>
            <div class="product-description">
                High-quality ergonomic office chair with lumbar support and adjustable height.
            </div>
            <div class="product-gallery">
                <img src="chair-main.jpg" class="product-image" alt="Office Chair">
                <img src="chair-side.jpg" class="product-image" alt="Office Chair Side View">
            </div>
            <div class="product-meta">
                <span class="sku">SKU: EC-123456</span>
                <span class="brand">Brand: ErgoComfort</span>
            </div>
            <button class="add-to-cart">Add to Cart</button>
        </div>
    </body>
    </html>
    """
    
    # Run examples
    await extract_product_information(sample_product_html)
    
    # For brevity, we won't actually run all these examples, 
    # but in real usage they would be called as needed:
    """
    await extract_article_content(sample_article_html)
    await extract_listing_data(sample_listing_html)
    
    await configure_standard_pipeline()
    await create_custom_pipeline()
    await set_pipeline_options()
    await handle_pipeline_results()
    
    await use_custom_extractor()
    await register_custom_extractor()
    
    await define_custom_schema()
    await register_schemas()
    await validate_against_schema()
    await generate_schema_from_data()
    
    await configure_quality_evaluation()
    await interpret_quality_metrics()
    await improve_extraction_quality()
    """
    
    print("\n=== End of Examples ===")


if __name__ == "__main__":
    asyncio.run(main())