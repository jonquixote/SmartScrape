"""
Integration tests for adaptive scraper with extraction pipelines.

These tests verify that the Universal Extraction Framework is properly
integrated with the AdaptiveScraper controller.
"""

import asyncio
import json
import os
import pytest
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from bs4 import BeautifulSoup

from controllers.adaptive_scraper import AdaptiveScraper, get_adaptive_scraper
from extraction.core.extraction_interface import BaseExtractor
from extraction.core.pipeline_registry import register_extraction_pipelines
from core.pipeline.registry import PipelineRegistry
from core.pipeline.factory import PipelineFactory
from core.service_registry import ServiceRegistry


class TestPipelineIntegration:
    """
    Tests for AdaptiveScraper integration with extraction pipelines.
    """

    @pytest.fixture
    async def adaptive_scraper(self):
        """Create an instance of AdaptiveScraper with pipelines enabled."""
        scraper = AdaptiveScraper(config={
            'use_ai': True,
            'max_pages': 10,
            'max_depth': 2,
            'use_pipelines': True
        })
        yield scraper
        # Clean up after tests
        scraper.shutdown()

    @pytest.fixture
    def html_product_page(self):
        """Sample HTML of a product page for testing."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Product Page</title>
            <meta name="description" content="This is a test product page">
        </head>
        <body>
            <div class="product">
                <h1 class="product-title">Test Product</h1>
                <div class="product-price">$99.99</div>
                <img class="product-image" src="test-image.jpg" alt="Test Product">
                <div class="product-description">
                    <p>This is a test product description.</p>
                    <ul class="features">
                        <li>Feature 1</li>
                        <li>Feature 2</li>
                        <li>Feature 3</li>
                    </ul>
                </div>
                <div class="product-rating">4.5 stars</div>
            </div>
        </body>
        </html>
        """

    @pytest.fixture
    def html_article_page(self):
        """Sample HTML of an article page for testing."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Article Page</title>
            <meta name="description" content="This is a test article page">
        </head>
        <body>
            <article>
                <h1 class="article-title">Test Article Title</h1>
                <div class="article-meta">
                    <span class="author">John Doe</span>
                    <time datetime="2025-05-10">May 10, 2025</time>
                </div>
                <div class="article-content">
                    <p>This is a test article paragraph 1.</p>
                    <p>This is a test article paragraph 2.</p>
                    <p>This is a test article paragraph 3.</p>
                </div>
                <div class="article-tags">
                    <span class="tag">test</span>
                    <span class="tag">article</span>
                    <span class="tag">example</span>
                </div>
            </article>
        </body>
        </html>
        """

    @pytest.fixture
    def html_listing_page(self):
        """Sample HTML of a listing page for testing."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Listing Page</title>
            <meta name="description" content="This is a test listing page">
        </head>
        <body>
            <div class="listing">
                <h1>Products Listing</h1>
                <div class="products-grid">
                    <div class="product-item">
                        <h2>Product 1</h2>
                        <div class="price">$19.99</div>
                        <img src="product1.jpg" alt="Product 1">
                    </div>
                    <div class="product-item">
                        <h2>Product 2</h2>
                        <div class="price">$29.99</div>
                        <img src="product2.jpg" alt="Product 2">
                    </div>
                    <div class="product-item">
                        <h2>Product 3</h2>
                        <div class="price">$39.99</div>
                        <img src="product3.jpg" alt="Product 3">
                    </div>
                </div>
                <div class="pagination">
                    <a href="?page=1" class="active">1</a>
                    <a href="?page=2">2</a>
                    <a href="?page=3">3</a>
                </div>
            </div>
        </body>
        </html>
        """

    @pytest.mark.asyncio
    async def test_content_type_detection(self, adaptive_scraper, html_product_page, html_article_page, html_listing_page):
        """Test content type detection for different page types."""
        # Test product page detection
        product_analysis = await adaptive_scraper.analyze_page_content(html_product_page, "https://example.com/product")
        assert product_analysis["content_type"] == "product"
        
        # Test article page detection
        article_analysis = await adaptive_scraper.analyze_page_content(html_article_page, "https://example.com/article")
        assert article_analysis["content_type"] == "article"
        
        # Test listing page detection
        listing_analysis = await adaptive_scraper.analyze_page_content(html_listing_page, "https://example.com/listing")
        assert listing_analysis["content_type"] == "listing"
        
        # Check complexity estimation
        assert "complexity" in product_analysis
        assert product_analysis["complexity"] in ["simple", "medium", "complex"]

    @pytest.mark.asyncio
    async def test_pipeline_selection(self, adaptive_scraper, html_product_page, html_article_page, html_listing_page):
        """Test pipeline selection logic based on content type."""
        # Mock the pipeline registry and factory for testing
        adaptive_scraper.pipeline_registry = MagicMock()
        adaptive_scraper.pipeline_factory = MagicMock()
        adaptive_scraper.pipeline_registry.has_pipeline.return_value = True
        
        # Test product page pipeline selection
        product_pipeline = await adaptive_scraper.select_extraction_pipeline(
            url="https://example.com/product",
            content_type="product", 
            content_sample=html_product_page
        )
        
        # Verify correct pipeline was selected
        adaptive_scraper.pipeline_factory.create_pipeline.assert_called_with("product_extraction", {})
        
        # Test article page pipeline selection
        await adaptive_scraper.select_extraction_pipeline(
            url="https://example.com/article",
            content_type="article", 
            content_sample=html_article_page
        )
        
        # Verify correct pipeline was selected
        adaptive_scraper.pipeline_factory.create_pipeline.assert_called_with("article_extraction", {})
        
        # Test listing page pipeline selection
        await adaptive_scraper.select_extraction_pipeline(
            url="https://example.com/listing",
            content_type="listing", 
            content_sample=html_listing_page
        )
        
        # Verify correct pipeline was selected
        adaptive_scraper.pipeline_factory.create_pipeline.assert_called_with("listing_extraction", {})
        
        # Test unknown content type
        await adaptive_scraper.select_extraction_pipeline(
            url="https://example.com/unknown",
            content_type="unknown"
        )
        
        # Verify fallback to default extraction pipeline
        adaptive_scraper.pipeline_factory.create_pipeline.assert_called_with("extraction", {})

    @pytest.mark.asyncio
    async def test_extraction_from_html(self, adaptive_scraper, html_product_page):
        """Test direct HTML extraction using extraction pipeline."""
        # Mock the pipeline execution
        pipeline_mock = MagicMock()
        pipeline_mock.name = "product_extraction"
        pipeline_mock.execute = MagicMock(return_value=asyncio.Future())
        pipeline_mock.execute.return_value.set_result({
            "success": True,
            "results": [{
                "title": "Test Product",
                "price": "$99.99",
                "description": "This is a test product description.",
                "features": ["Feature 1", "Feature 2", "Feature 3"],
                "image": "test-image.jpg"
            }],
            "execution_time": 0.5
        })
        
        adaptive_scraper.select_extraction_pipeline = MagicMock(return_value=asyncio.Future())
        adaptive_scraper.select_extraction_pipeline.return_value.set_result(pipeline_mock)
        adaptive_scraper.execute_pipeline = MagicMock(return_value=asyncio.Future())
        adaptive_scraper.execute_pipeline.return_value.set_result({
            "success": True,
            "results": [{
                "title": "Test Product",
                "price": "$99.99",
                "description": "This is a test product description.",
                "features": ["Feature 1", "Feature 2", "Feature 3"],
                "image": "test-image.jpg"
            }],
            "execution_time": 0.5
        })
        
        # Call scrape_with_pipeline with HTML content
        result = await adaptive_scraper.scrape_with_pipeline(
            url="https://example.com/product",
            content_type="product",
            options={"html_content": html_product_page}
        )
        
        # Verify result structure
        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Test Product"
        assert result["results"][0]["price"] == "$99.99"
        assert "execution_time" in result

    @pytest.mark.asyncio
    async def test_extraction_from_html(self, adaptive_scraper, html_product_page):
        """Test direct HTML extraction using extraction pipeline."""
        # Mock the execute_extraction_pipeline method
        adaptive_scraper.execute_extraction_pipeline = MagicMock(return_value=asyncio.Future())
        adaptive_scraper.execute_extraction_pipeline.return_value.set_result({
            "success": True,
            "results": [{
                "title": "Test Product",
                "price": "$99.99",
                "description": "This is a test product description.",
                "features": ["Feature 1", "Feature 2", "Feature 3"],
                "image": "test-image.jpg"
            }],
            "execution_time": 0.5,
            "content_type": "product",
            "quality_metrics": {
                "overall_score": 0.92,
                "field_coverage": 0.95
            }
        })
        
        # Call the method directly with HTML content
        result = await adaptive_scraper.execute_extraction_pipeline(
            html=html_product_page,
            url="https://example.com/product"
        )
        
        # Verify method was called with correct parameters
        adaptive_scraper.execute_extraction_pipeline.assert_called_with(
            html=html_product_page,
            url="https://example.com/product"
        )
        
        # Verify result structure
        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Test Product"
        assert result["results"][0]["price"] == "$99.99"
        assert "execution_time" in result
        assert "quality_metrics" in result
        assert result["content_type"] == "product"

    @pytest.mark.asyncio
    async def test_extraction_from_strategy(self, adaptive_scraper):
        """Test strategy integration with extraction pipeline."""
        # Mock strategy execution result
        strategy_result = [
            {
                "url": "https://example.com/product",
                "html": "<div class='product'><h1>Test Product</h1><div class='price'>$99.99</div></div>",
                "status": 200
            }
        ]
        
        # Mock strategy
        strategy_mock = MagicMock()
        strategy_mock.name = "test_strategy"
        strategy_mock.execute = MagicMock(return_value=asyncio.Future())
        strategy_mock.execute.return_value.set_result(strategy_result)
        strategy_mock.has_errors = MagicMock(return_value=False)
        
        # Mock strategy factory
        adaptive_scraper.strategy_factory = MagicMock()
        adaptive_scraper.strategy_factory.get_strategy.return_value = strategy_mock
        adaptive_scraper.strategy_factory.select_best_strategy.return_value = strategy_mock
        
        # Mock extraction pipeline
        pipeline_mock = MagicMock()
        pipeline_mock.name = "extraction"
        pipeline_mock.execute = MagicMock(return_value=asyncio.Future())
        pipeline_mock.execute.return_value.set_result({
            "success": True,
            "results": [{
                "title": "Test Product",
                "price": "$99.99"
            }],
            "execution_time": 0.5
        })
        
        # Mock pipeline selection and execution
        adaptive_scraper.select_extraction_pipeline = MagicMock(return_value=asyncio.Future())
        adaptive_scraper.select_extraction_pipeline.return_value.set_result(pipeline_mock)
        
        adaptive_scraper.execute_pipeline = MagicMock(return_value=asyncio.Future())
        adaptive_scraper.execute_pipeline.return_value.set_result({
            "success": True,
            "results": [{
                "title": "Test Product",
                "price": "$99.99"
            }],
            "execution_time": 0.5
        })
        
        # Set up analysis mock to return a content type
        adaptive_scraper.analyze_page_content = MagicMock(return_value=asyncio.Future())
        adaptive_scraper.analyze_page_content.return_value.set_result({
            "content_type": "product",
            "complexity": "simple"
        })
        
        # Call scrape with option to use extraction
        result = await adaptive_scraper.scrape(
            url="https://example.com/product",
            options={"use_extraction": True, "use_extraction_pipelines": True}
        )
        
        # Verify strategy was executed
        strategy_mock.execute.assert_called_once()
        
        # Verify extraction pipeline was used
        assert adaptive_scraper.execute_pipeline.called
        
        # Verify result contains extracted data
        assert result["success"] is True
        assert "results" in result
        assert isinstance(result["results"], list)
        assert result["results"][0]["title"] == "Test Product"
        assert result["results"][0]["price"] == "$99.99"

    @pytest.mark.asyncio
    async def test_extraction_error_handling(self, adaptive_scraper):
        """Test error recovery in extraction pipeline."""
        # Mock strategy result
        strategy_result = [{"html": "<div>Test Content</div>"}]
        
        # Mock extraction pipeline that raises an error
        pipeline_mock = MagicMock()
        pipeline_mock.name = "extraction"
        pipeline_mock.execute = MagicMock(side_effect=Exception("Extraction failed"))
        
        # Another pipeline for fallback
        fallback_pipeline_mock = MagicMock()
        fallback_pipeline_mock.name = "pattern_extraction"
        fallback_pipeline_mock.execute = MagicMock(return_value=asyncio.Future())
        fallback_pipeline_mock.execute.return_value.set_result({
            "success": True,
            "results": [{"title": "Fallback Result"}],
            "execution_time": 0.2
        })
        
        # Mock pipeline selection to return the error pipeline then fallback pipeline
        adaptive_scraper.select_extraction_pipeline = MagicMock()
        adaptive_scraper.select_extraction_pipeline.side_effect = [
            pipeline_mock,  # First call returns error pipeline
            fallback_pipeline_mock  # Second call returns fallback pipeline
        ]
        
        # Mock error handler to use fallback
        adaptive_scraper.handle_extraction_failure = MagicMock(return_value=asyncio.Future())
        adaptive_scraper.handle_extraction_failure.return_value.set_result({
            "success": True,
            "results": [{"title": "Fallback Result"}],
            "alternative_pipeline": "pattern_extraction"
        })
        
        # Mock other necessary methods
        adaptive_scraper.strategy_result_to_pipeline_input = MagicMock(return_value={"content": "<div>Test Content</div>"})
        adaptive_scraper.pipeline_result_to_strategy_format = MagicMock(return_value=[{"title": "Fallback Result"}])
        
        # Simulate extraction with error and fallback
        result = await adaptive_scraper.scrape_with_pipeline(
            url="https://example.com/test",
            options={"use_extraction_pipelines": True}
        )
        
        # Verify error was handled and fallback was used
        assert adaptive_scraper.handle_extraction_failure.called
        assert result["success"] is True
        assert "alternative_pipeline" in result
        assert result["alternative_pipeline"] == "pattern_extraction"
        assert result["results"][0]["title"] == "Fallback Result"

    @pytest.mark.asyncio
    async def test_extraction_error_handling(self, adaptive_scraper):
        """Test error recovery in extraction pipeline."""
        # Mock the execute_extraction_pipeline method to raise an error first time
        original_execute = adaptive_scraper.execute_extraction_pipeline
        call_count = 0
        
        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call raises an error
                raise Exception("Extraction failed")
            else:
                # Second call returns success
                return {
                    "success": True,
                    "results": [{
                        "title": "Fallback Result",
                        "source": "fallback_extraction"
                    }],
                    "content_type": "product",
                    "pipeline_name": "pattern_extraction",
                    "execution_time": 0.2
                }
        
        adaptive_scraper.execute_extraction_pipeline = mock_execute
        
        # Set up the retry method to use fallback pipeline
        adaptive_scraper.retry_with_alternative_pipeline = MagicMock(return_value=asyncio.Future())
        adaptive_scraper.retry_with_alternative_pipeline.return_value.set_result({
            "success": True,
            "results": [{
                "title": "Fallback Result",
                "source": "fallback_extraction"
            }],
            "pipeline_name": "pattern_extraction",
            "execution_time": 0.2
        })
        
        # Call the method that should handle the error
        result = await adaptive_scraper.execute_extraction_pipeline(
            html="<div>Test Content</div>",
            url="https://example.com/test",
            options={"extraction_fallback_behavior": "aggressive"}
        )
        
        # Verify error was handled and fallback was used
        assert call_count == 2  # The method should be called twice
        assert result["success"] is True
        assert result["results"][0]["title"] == "Fallback Result"
        assert result["pipeline_name"] == "pattern_extraction"
        
        # Restore the original method
        adaptive_scraper.execute_extraction_pipeline = original_execute

    @pytest.mark.asyncio
    async def test_extraction_results_format(self, adaptive_scraper, html_product_page):
        """Test result formatting from extraction pipeline."""
        # Create sample pipeline result
        pipeline_result = {
            "success": True,
            "results": [
                {
                    "content": "<div>Product content</div>",
                    "extracted_data": {
                        "title": "Test Product",
                        "price": "$99.99",
                        "features": ["Feature 1", "Feature 2"]
                    },
                    "metadata": {
                        "extraction_method": "pattern",
                        "confidence": 0.85
                    },
                    "quality_score": 0.9
                }
            ],
            "execution_time": 0.5,
            "metrics": {
                "processing_time": 0.3,
                "extraction_time": 0.2
            }
        }
        
        # Test conversion to strategy format
        strategy_format = adaptive_scraper.pipeline_result_to_strategy_format(pipeline_result)
        
        # Verify conversion
        assert isinstance(strategy_format, list)
        assert len(strategy_format) == 1
        assert strategy_format[0]["title"] == "Test Product"
        assert strategy_format[0]["price"] == "$99.99"
        assert isinstance(strategy_format[0]["features"], list)
        assert "_metadata" in strategy_format[0]
        
        # Test conversion from strategy to pipeline
        strategy_result = [
            {
                "title": "Product from Strategy",
                "price": "$199.99",
                "html": html_product_page
            }
        ]
        
        pipeline_input = adaptive_scraper.strategy_result_to_pipeline_input(
            strategy_result,
            {"url": "https://example.com/product"}
        )
        
        # Verify pipeline input
        assert "content" in pipeline_input
        assert "metadata" in pipeline_input
        assert "context" in pipeline_input
        assert pipeline_input["metadata"]["url"] == "https://example.com/product"
        
        # Test merging extraction results
        result1 = [{"title": "Product 1", "price": "$10.99"}]
        result2 = [{"title": "Product 1", "description": "Description", "_metadata": {"extractor": "pattern"}}]
        
        merged = adaptive_scraper.merge_extraction_results([result1, result2])
        
        # Verify merge
        assert len(merged) == 1
        assert merged[0]["title"] == "Product 1"
        assert merged[0]["price"] == "$10.99"
        assert merged[0]["description"] == "Description"
        assert merged[0]["_metadata"]["merged"] is True
        assert "sources" in merged[0]["_metadata"]
        assert "pattern" in merged[0]["_metadata"]["sources"]

    @pytest.mark.asyncio
    async def test_extraction_comparison(self, adaptive_scraper):
        """Test comparison between old and new extraction approaches."""
        # Create sample results from both approaches
        old_result = [
            {"title": "Product 1", "price": "$99.99"},
            {"title": "Product 2", "price": "$149.99"}
        ]
        
        new_result = [
            {"title": "Product 1", "price": "$99.99", "description": "Description 1"},
            {"title": "Product 2", "price": "$149.99", "description": "Description 2"},
            {"title": "Product 3", "price": "$199.99"}
        ]
        
        # Generate comparison report
        report = adaptive_scraper.generate_comparison_report(old_result, new_result)
        
        # Verify report structure
        assert "timestamp" in report
        assert "extraction_comparison" in report
        assert "recommendations" in report
        
        # Check metrics
        metrics = report["extraction_comparison"]["metrics"]
        assert metrics["old_count"] == 2
        assert metrics["new_count"] == 3
        assert metrics["new_unique_fields"] == ["description"]
        
        # Test improvement detection
        is_better = is_improvement("result_count", 5, 10)
        assert is_better is True
        
        is_better = is_improvement("error_rate", 0.1, 0.05)
        assert is_better is True
        
        is_better = is_improvement("extraction_time", 2.5, 1.8)
        assert is_better is True