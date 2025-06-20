# SmartScrape Extraction Pipeline Implementation Guide

This guide provides practical steps and examples for converting existing extraction code into pipeline stages, with a focus on real code from the SmartScrape codebase.

## Table of Contents

1. [Overview](#overview)
2. [Step-by-Step Migration Approach](#step-by-step-migration-approach)
3. [Converting MultiStrategyExtractor](#converting-multistrategyextractor)
4. [Converting ContentExtractor](#converting-contentextractor)
5. [Handling Fallback Extraction](#handling-fallback-extraction)
6. [Integrating AI-Guided Extraction](#integrating-ai-guided-extraction)
7. [Testing Extraction Pipelines](#testing-extraction-pipelines)
8. [Solving Common Implementation Challenges](#solving-common-implementation-challenges)

## Overview

The SmartScrape extraction code currently follows several patterns:

- **Multi-strategy approach**: Trying different extraction methods until one succeeds
- **Content-type specialization**: Using different extractors based on content type (article, listing, etc.)
- **Fallback mechanisms**: Gracefully degrading from advanced to basic extraction methods
- **AI-guided extraction**: Using AI to enhance extraction capabilities

This guide shows how to convert these patterns into pipeline stages that fit the new architecture while maintaining backward compatibility.

## Step-by-Step Migration Approach

1. **Start with wrappers**: Begin by wrapping existing extractors as pipeline stages
2. **Implement key interfaces**: Create the basic Pipeline, Stage, and Context implementations
3. **Convert simple extractors first**: Start with simpler, self-contained extraction methods
4. **Refactor toward atomic stages**: Gradually break down complex extractors into atomic stages
5. **Implement conditional logic**: Create branching logic for content-type and strategy selection
6. **Add error handling and metrics**: Integrate fallback mechanisms and monitoring

## Converting MultiStrategyExtractor

The `MultiStrategyExtractor` class in `extraction/content_extraction.py` follows a strategy pattern that tries different extraction approaches in sequence. Here's how to convert it into pipeline stages:

### 1. Create a Strategy Selection Stage

```python
# core/pipeline/stages/processing/strategy_selection_stage.py

class StrategySelectionStage(PipelineStage):
    """Determines the optimal extraction strategy based on content characteristics."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.config = config or {}
        
    async def process(self, context: PipelineContext) -> bool:
        html_content = context.get("html_content")
        if not html_content:
            context.add_error(self.name, "No HTML content provided")
            return False
            
        # Create BeautifulSoup object for analysis
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Determine content type if not already set
        content_type = context.get("content_type")
        if not content_type:
            content_type = self._determine_content_type(
                soup=soup, 
                user_intent=context.get("user_intent")
            )
            context.set("content_type", content_type)
            
        # Get optimal strategy order
        strategy_order = self._get_strategy_order(
            content_type=content_type,
            user_intent=context.get("user_intent")
        )
        context.set("strategy_order", strategy_order)
        
        return True
        
    def _determine_content_type(self, soup, user_intent=None):
        """
        Logic from MultiStrategyExtractor._determine_content_type
        """
        # Check for explicit content type in user intent
        if user_intent and "content_type" in user_intent:
            return user_intent["content_type"]
            
        # Detect content type from HTML structure
        # (Implement logic from original method)
        article_indicators = [
            bool(soup.find('article')),
            bool(soup.find('meta', {'property': 'article:published_time'})),
            # ... other indicators
        ]
        
        listing_indicators = [
            len(soup.find_all('li')) > 10,
            # ... other indicators
        ]
        
        # ... other detection code
        
        # Return detected content type
        if any(article_indicators):
            return "article"
        elif any(listing_indicators):
            return "listing"
        else:
            return "generic"
            
    def _get_strategy_order(self, content_type, user_intent=None):
        """
        Logic from MultiStrategyExtractor._get_strategy_order
        """
        # Define default order based on content type
        if content_type == "article":
            default_order = ["content_heuristics", "css_selector", "xpath"]
            if self.config.get("use_ai", True):
                default_order.append("ai_guided")
                
        elif content_type == "listing":
            default_order = ["css_selector", "xpath"]
            if self.config.get("use_ai", True):
                default_order.append("ai_guided")
            default_order.append("content_heuristics")
            
        else:
            default_order = ["css_selector", "xpath"]
            if self.config.get("use_ai", True):
                default_order.append("ai_guided")
            default_order.append("content_heuristics")
            
        # Prioritize AI if requested in user intent
        if user_intent and user_intent.get("use_ai_extraction", False):
            if "ai_guided" in default_order:
                default_order.remove("ai_guided")
            return ["ai_guided"] + default_order
            
        return default_order
```

### 2. Implement Strategy Stages

For each extraction strategy, create a separate stage:

```python
# core/pipeline/stages/processing/css_selector_stage.py

class CssSelectorExtractionStage(PipelineStage):
    """Extracts content using CSS selectors."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Logic from MultiStrategyExtractor._extract_with_css_selectors"""
        soup = context.get("soup")
        if not soup:
            # Parse HTML if not already parsed
            html_content = context.get("html_content")
            if not html_content:
                context.add_error(self.name, "No HTML content provided")
                return False
            soup = BeautifulSoup(html_content, 'lxml')
            context.set("soup", soup)
            
        # Get extraction schema
        extraction_schema = context.get("extraction_schema")
        if not extraction_schema:
            context.add_error(self.name, "No extraction schema provided")
            return False
            
        # Extract content
        try:
            result = {
                "success": True,
                "extraction_method": "css_selector"
            }
            
            # Handle item containers if present
            if "items" in extraction_schema:
                result["items"] = []
                # ... extraction logic from _extract_with_css_selectors
            else:
                # ... direct extraction logic
                
            # Store results in context
            context.update(result)
            return True
            
        except Exception as e:
            context.add_error(self.name, f"CSS extraction error: {str(e)}")
            return False
```

### 3. Create a Pipeline Runner

Create a main stage that orchestrates the extraction strategies:

```python
# core/pipeline/stages/processing/multi_strategy_extraction_stage.py

class MultiStrategyExtractionStage(PipelineStage):
    """Tries multiple extraction strategies in sequence."""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Initialize strategy map
        self.strategies = {
            "css_selector": CssSelectorExtractionStage(self.config.get("css_config")),
            "xpath": XPathExtractionStage(self.config.get("xpath_config")),
            "content_heuristics": ContentHeuristicsStage(self.config.get("heuristics_config")),
        }
        
        # Add AI strategy if enabled
        if self.config.get("use_ai", True):
            self.strategies["ai_guided"] = AIExtractionStage(self.config.get("ai_config"))
        
    async def process(self, context: PipelineContext) -> bool:
        # Ensure we have strategy order
        strategy_order = context.get("strategy_order")
        if not strategy_order:
            # Get it via the strategy selection stage
            selection_stage = StrategySelectionStage(self.config)
            await selection_stage.process(context)
            strategy_order = context.get("strategy_order")
            
        # Initialize attempts tracking
        context.set("attempts", [])
        
        # Try each strategy in order
        for strategy_name in strategy_order:
            if strategy_name not in self.strategies:
                continue
                
            strategy_stage = self.strategies[strategy_name]
            try:
                # Create a child context for strategy attempt
                strategy_context = context.create_child_context()
                
                # Process with strategy
                success = await strategy_stage.process(strategy_context)
                
                # Record attempt
                context.append("attempts", {
                    "strategy": strategy_name,
                    "success": success
                })
                
                if success:
                    # Copy results from strategy to main context
                    context.update(strategy_context.data)
                    context.set("strategy_used", strategy_name)
                    return True
                    
            except Exception as e:
                context.append("attempts", {
                    "strategy": strategy_name,
                    "success": False,
                    "error": str(e)
                })
                
        # All strategies failed
        context.add_error(self.name, "All extraction strategies failed")
        return False
```

## Converting ContentExtractor

The `ContentExtractor` class handles specialized extraction based on content type. Here's how to convert it:

### 1. Create Content Type-Specific Stages

```python
# core/pipeline/stages/processing/article_extraction_stage.py

class ArticleExtractionStage(PipelineStage):
    """Extracts article content using various methods."""
    
    async def process(self, context: PipelineContext) -> bool:
        html_content = context.get("html_content")
        url = context.get("url")
        
        if not html_content:
            context.add_error(self.name, "No HTML content provided")
            return False
            
        # Try multiple extraction methods in sequence
        
        # 1. Try readability extraction
        readability_result = self._extract_with_readability(html_content, url)
        if readability_result.get("success", False):
            context.update(readability_result)
            context.set_metadata("extraction_library", "readability")
            return True
            
        # 2. Try trafilatura
        trafilatura_result = self._extract_with_trafilatura(html_content, url)
        if trafilatura_result.get("success", False):
            context.update(trafilatura_result)
            context.set_metadata("extraction_library", "trafilatura")
            return True
            
        # 3. Try goose
        goose_result = self._extract_with_goose(html_content, url)
        if goose_result.get("success", False):
            context.update(goose_result)
            context.set_metadata("extraction_library", "goose")
            return True
            
        # 4. Try justext
        justext_result = self._extract_with_justext(html_content, url)
        if justext_result.get("success", False):
            context.update(justext_result)
            context.set_metadata("extraction_library", "justext")
            return True
            
        # All methods failed
        context.add_error(self.name, "All article extraction methods failed")
        return False
        
    def _extract_with_readability(self, html_content, url):
        """Logic from ContentExtractor._extract_with_readability"""
        # ... implementation
        
    # ... other extraction method implementations
```

### 2. Create a Content Type Selection Stage

```python
# core/pipeline/stages/processing/content_type_selection_stage.py

class ContentTypeSelectionStage(PipelineStage):
    """Selects the appropriate extraction stage based on content type."""
    
    async def process(self, context: PipelineContext) -> bool:
        content_type = context.get("content_type")
        if not content_type:
            # We need to determine content type
            html_content = context.get("html_content")
            if not html_content:
                context.add_error(self.name, "No HTML content provided")
                return False
                
            # Use soup to determine content type if available
            soup = context.get("soup")
            if not soup:
                soup = BeautifulSoup(html_content, 'lxml')
                
            content_type = self._detect_content_type(html_content, soup, context.get("url"))
            context.set("content_type", content_type)
            
        # Select and run the appropriate extraction stage
        if content_type == "article":
            stage = ArticleExtractionStage(self.config.get("article_config"))
        elif content_type == "listing":
            stage = ListingExtractionStage(self.config.get("listing_config"))
        elif content_type == "data_table":
            stage = TableExtractionStage(self.config.get("table_config"))
        else:
            # Generic content
            stage = GenericExtractionStage(self.config.get("generic_config"))
            
        # Process with selected stage
        return await stage.process(context)
        
    def _detect_content_type(self, html_content, soup, url):
        """Logic from ContentExtractor._detect_content_type"""
        # ... implementation
```

## Handling Fallback Extraction

Fallback extraction is a key pattern in the SmartScrape codebase. Convert it to pipeline stages as follows:

### 1. Create a Fallback Stage

```python
# core/pipeline/stages/processing/fallback_extraction_stage.py

class FallbackExtractionStage(PipelineStage):
    """Provides basic extraction when all other methods fail."""
    
    async def process(self, context: PipelineContext) -> bool:
        # Only run if no success flag or success is False
        if context.get("success", False):
            return True
            
        html_content = context.get("html_content")
        url = context.get("url")
        
        if not html_content:
            context.add_error(self.name, "No HTML content provided")
            return False
            
        try:
            # Basic extraction logic from ContentExtractor._fallback_extraction
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
                
            # Extract metadata
            metadata = {}
            for meta in soup.find_all('meta'):
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                if name and content:
                    metadata[name] = content
                    
            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', {'id': 'content'})
            text_content = soup.get_text(separator=' ', strip=True)
            
            if main_content:
                main_text = main_content.get_text(separator=' ', strip=True)
                main_text = re.sub(r'\s+', ' ', main_text).strip()
                
                if len(main_text) > 200:
                    text_content = main_text
                    
            # Store results
            result = {
                "success": True,
                "title": title,
                "text": text_content,
                "html": str(soup),
                "metadata": metadata,
                "extraction_method": "fallback"
            }
            
            context.update(result)
            context.set_metadata("fallback_used", True)
            
            return True
            
        except Exception as e:
            context.add_error(self.name, f"Fallback extraction failed: {str(e)}")
            return False
```

### 2. Implement Circuit Breaker for External Services

```python
# core/pipeline/stages/processing/external_service_stage.py

class ExternalServiceStage(PipelineStage):
    """Base class for stages that use external services with circuit breaking."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.circuit_name = self.config.get("circuit_name", self.__class__.__name__)
        self.retry_count = self.config.get("retry_count", 3)
        
    async def process(self, context: PipelineContext) -> bool:
        # Get circuit breaker service
        circuit_breaker = context.get_service("circuit_breaker_manager")
        if not circuit_breaker:
            # No circuit breaker, execute directly
            return await self._execute_service_call(context)
            
        try:
            # Function to execute with circuit breaker
            @circuit_breaker.circuit_breaker(self.circuit_name)
            async def execute_with_circuit_breaker():
                return await self._execute_service_call(context)
                
            return await execute_with_circuit_breaker()
            
        except OpenCircuitError as e:
            # Circuit is open, use fallback
            context.add_error(self.name, f"Circuit breaker open: {str(e)}")
            return await self._fallback_execution(context)
            
    @abstractmethod
    async def _execute_service_call(self, context: PipelineContext) -> bool:
        """Execute the primary service call."""
        pass
        
    async def _fallback_execution(self, context: PipelineContext) -> bool:
        """Execute fallback when circuit is open."""
        # Default implementation uses the FallbackExtractionStage
        fallback_stage = FallbackExtractionStage(self.config.get("fallback_config"))
        return await fallback_stage.process(context)
```

## Integrating AI-Guided Extraction

AI-guided extraction needs special handling for error cases and resource management:

```python
# core/pipeline/stages/processing/ai_extraction_stage.py

class AIExtractionStage(ExternalServiceStage):
    """Extracts content using AI with proper error handling."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.circuit_name = "ai_extraction"
        
    async def _execute_service_call(self, context: PipelineContext) -> bool:
        html_content = context.get("html_content")
        url = context.get("url")
        user_intent = context.get("user_intent")
        
        if not html_content:
            context.add_error(self.name, "No HTML content provided")
            return False
            
        if not user_intent:
            context.add_error(self.name, "User intent required for AI extraction")
            return False
            
        try:
            # Get AI service
            ai_service = context.get_service("ai_service")
            if not ai_service:
                context.add_error(self.name, "AI service not available")
                return False
                
            # Extract desired properties
            desired_properties = user_intent.get("properties", [])
            if not desired_properties and context.get("extraction_schema"):
                desired_properties = [
                    field for field in context.get("extraction_schema") if field != "items"
                ]
                
            # Get entity type
            entity_type = user_intent.get("entity_type", "item")
            
            # Call AI extraction
            items = await ai_service.extract_content(
                html_content=html_content,
                url=url,
                user_intent=user_intent,
                desired_properties=desired_properties,
                entity_type=entity_type
            )
            
            if not items:
                context.add_error(self.name, "AI extraction yielded no results")
                return False
                
            # Store results
            context.set("extraction_method", "ai_guided")
            context.set("content_type", "listing" if len(items) > 1 else "detail")
            
            if len(items) > 1:
                context.set("items", items)
            else:
                context.set("data", items[0])
                
            return True
            
        except Exception as e:
            context.add_error(self.name, f"AI extraction error: {str(e)}")
            return False
```

## Testing Extraction Pipelines

### 1. Unit Testing Extraction Stages

```python
# tests/core/pipeline/stages/test_extraction_stages.py

import pytest
from bs4 import BeautifulSoup

from core.pipeline.context import PipelineContext
from core.pipeline.stages.processing.css_selector_stage import CssSelectorExtractionStage

class TestCssSelectorStage:
    @pytest.mark.asyncio
    async def test_basic_extraction(self):
        # Arrange
        html = """
        <html>
            <body>
                <h1 class="title">Test Product</h1>
                <div class="price">$9.99</div>
                <div class="description">Product description</div>
            </body>
        </html>
        """
        
        extraction_schema = {
            "title": [".title"],
            "price": [".price"],
            "description": [".description"]
        }
        
        context = PipelineContext({
            "html_content": html,
            "soup": BeautifulSoup(html, 'lxml'),
            "extraction_schema": extraction_schema
        })
        
        stage = CssSelectorExtractionStage()
        
        # Act
        result = await stage.process(context)
        
        # Assert
        assert result is True
        assert context.get("success") is True
        assert context.get("extraction_method") == "css_selector"
        assert context.get("title") == "Test Product"
        assert context.get("price") == "$9.99"
        assert context.get("description") == "Product description"
```

### 2. Testing a Complete Pipeline

```python
# tests/core/pipeline/test_extraction_pipeline.py

import pytest
from unittest.mock import patch, MagicMock

from core.pipeline.pipeline import Pipeline
from core.pipeline.context import PipelineContext
from core.pipeline.stages.input.http_input import HttpInputStage
from core.pipeline.stages.processing.strategy_selection_stage import StrategySelectionStage
from core.pipeline.stages.processing.multi_strategy_extraction_stage import MultiStrategyExtractionStage
from core.pipeline.stages.processing.fallback_extraction_stage import FallbackExtractionStage
from core.pipeline.stages.output.json_output import JsonOutputStage

class TestExtractionPipeline:
    @pytest.mark.asyncio
    async def test_complete_extraction_flow(self):
        # Arrange
        with open("tests/fixtures/test_product_page.html", "r") as f:
            test_html = f.read()
            
        # Mock HttpInputStage to return test HTML
        mock_http_stage = MagicMock(spec=HttpInputStage)
        mock_http_stage.process = MagicMock(return_value=True)
        mock_http_stage.name = "http_input"
        
        # Create pipeline with real stages
        pipeline = Pipeline("extraction_pipeline")
        pipeline.add_stage(mock_http_stage)
        pipeline.add_stage(StrategySelectionStage())
        pipeline.add_stage(MultiStrategyExtractionStage({"use_ai": False}))
        pipeline.add_stage(FallbackExtractionStage())
        pipeline.add_stage(JsonOutputStage())
        
        # Act
        context = await pipeline.execute({
            "url": "https://example.com/test-product"
        })
        
        # Modify context after http_input stage to inject test HTML
        def side_effect(ctx):
            ctx.set("html_content", test_html)
            return True
            
        mock_http_stage.process.side_effect = side_effect
        
        # Assert
        assert not context.has_errors()
        assert context.get("success") is True
        assert context.get("extraction_method") in ["css_selector", "xpath", "content_heuristics"]
        assert context.get("title") is not None
        assert "json_output" in context.data
```

## Solving Common Implementation Challenges

### 1. Handling Circular Dependencies

```python
# Bad: Direct imports creating circular dependencies
from core.pipeline.stages.processing.css_selector_stage import CssSelectorExtractionStage
from core.pipeline.stages.processing.xpath_stage import XPathExtractionStage

# Good: Using factory pattern and lazy imports
class StageFactory:
    @staticmethod
    def create_stage(stage_type, config=None):
        if stage_type == "css_selector":
            from core.pipeline.stages.processing.css_selector_stage import CssSelectorExtractionStage
            return CssSelectorExtractionStage(config)
        elif stage_type == "xpath":
            from core.pipeline.stages.processing.xpath_stage import XPathExtractionStage
            return XPathExtractionStage(config)
        # ... other stage types
```

### 2. Sharing State Between Stages

```python
# Use context to share data between stages
class FirstStage(PipelineStage):
    async def process(self, context: PipelineContext) -> bool:
        # Store data for next stage
        context.set("shared_data", {"key": "value"})
        return True
        
class SecondStage(PipelineStage):
    async def process(self, context: PipelineContext) -> bool:
        # Access data from previous stage
        shared_data = context.get("shared_data")
        # Use shared_data
        return True
```

### 3. Managing Pipeline Configuration

```python
# Define configuration schema
extraction_pipeline_schema = {
    "type": "object",
    "properties": {
        "input": {
            "type": "object",
            "properties": {
                "timeout": {"type": "number", "default": 10},
                "user_agent": {"type": "string"}
            }
        },
        "extraction": {
            "type": "object",
            "properties": {
                "strategies": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["css", "xpath", "ai", "heuristics"]}
                },
                "use_ai": {"type": "boolean", "default": True}
            }
        }
    }
}

# Validate configuration
def validate_config(config, schema):
    """Validate configuration against schema"""
    # Use jsonschema library to validate
    from jsonschema import validate
    validate(instance=config, schema=schema)
    
# Create pipeline with validated config
def create_extraction_pipeline(config):
    # Validate config
    validate_config(config, extraction_pipeline_schema)
    
    # Create pipeline
    pipeline = Pipeline("extraction_pipeline", config)
    
    # Add stages with relevant config sections
    pipeline.add_stage(HttpInputStage(config.get("input")))
    pipeline.add_stage(StrategySelectionStage(config.get("extraction")))
    pipeline.add_stage(MultiStrategyExtractionStage(config.get("extraction")))
    pipeline.add_stage(FallbackExtractionStage())
    
    return pipeline
```

### 4. Managing External Dependencies

```python
# Gracefully handle external dependencies
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    
class ArticleExtractionStage(PipelineStage):
    async def process(self, context: PipelineContext) -> bool:
        # Try trafilatura if available
        if TRAFILATURA_AVAILABLE:
            result = self._extract_with_trafilatura(context.get("html_content"))
            if result.get("success"):
                context.update(result)
                return True
                
        # Continue with other methods
        # ...
```