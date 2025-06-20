"""
Benchmarks for extraction framework performance.

This module contains performance tests for the extraction framework components,
measuring execution time, memory usage, and resource utilization under
different scenarios.
"""

import time
import pytest
import os
import gc
import asyncio
import psutil
import json
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch
from bs4 import BeautifulSoup
import concurrent.futures

# Add import paths
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from extraction.pattern_extractor import DOMPatternExtractor
from extraction.semantic_extractor import AISemanticExtractor
from extraction.schema_manager import SchemaManager

# Fixtures path
FIXTURES_PATH = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def html_content():
    """Load HTML content from fixture files."""
    contents = {}
    
    for file_name in ["product_page.html", "article_page.html", "listing_page.html", 
                      "minimal_page.html", "malformed_page.html"]:
        file_path = FIXTURES_PATH / file_name
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                contents[file_name] = f.read()
    
    return contents


@pytest.fixture
def schemas():
    """Load schema definitions from fixture files."""
    schemas = {}
    
    for file_name in ["product_schema.json", "article_schema.json", 
                      "listing_schema.json", "generic_schema.json"]:
        file_path = FIXTURES_PATH / file_name
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                schemas[file_name] = json.load(f)
    
    return schemas


@pytest.fixture
def mock_strategy_context():
    """Create a mock strategy context with necessary services."""
    context = MagicMock()
    
    # Mock HTML service
    html_service = MagicMock()
    html_service.clean_html.side_effect = lambda x: x
    html_service.extract_main_content.side_effect = lambda x: x
    html_service.generate_selector.return_value = "div.product"
    
    # Mock AI service
    ai_service = MagicMock()
    ai_service.generate_response.side_effect = lambda **kwargs: {
        "content": json.dumps({"title": "Test Product", "price": {"amount": 99.99}}),
        "_metadata": {"total_tokens": 150}
    }
    
    # Mock error classifier
    error_classifier = MagicMock()
    error_classifier.classify_exception.return_value = {"category": "ContentError"}
    
    # Mock model selector
    model_selector = MagicMock()
    model_selector.select_model.return_value = "test-model"
    
    # Set up context to return mocked services
    context.get_service.side_effect = lambda service_name: {
        "html_service": html_service,
        "ai_service": ai_service,
        "error_classifier": error_classifier,
        "model_selector": model_selector,
        "schema_manager": SchemaManager()
    }.get(service_name)
    
    return context


@pytest.fixture
def pattern_extractor(mock_strategy_context):
    """Create a pattern extractor instance."""
    extractor = DOMPatternExtractor(context=mock_strategy_context)
    extractor.initialize()
    return extractor


@pytest.fixture
def semantic_extractor(mock_strategy_context):
    """Create a semantic extractor instance."""
    extractor = AISemanticExtractor(context=mock_strategy_context)
    extractor.initialize()
    return extractor


def measure_execution_time(func, *args, **kwargs):
    """Measure execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def measure_memory_usage(func, *args, **kwargs):
    """Measure memory usage of a function."""
    # Force garbage collection before measurement
    gc.collect()
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    result = func(*args, **kwargs)
    
    # Force garbage collection after execution
    gc.collect()
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_usage = memory_after - memory_before
    
    return result, memory_usage


class TestExtractionPerformance:
    """Benchmark tests for extraction performance."""
    
    def test_extraction_speed(self, pattern_extractor, html_content, schemas):
        """Test extraction speed for different extractors and content types."""
        results = {}
        
        # Benchmark pattern extractor
        for content_name, content in html_content.items():
            # Skip very large content for basic speed test
            if len(content) > 500000:
                continue
                
            # Get appropriate schema for content type
            schema = None
            if "product" in content_name:
                schema = schemas.get("product_schema.json")
            elif "article" in content_name:
                schema = schemas.get("article_schema.json")
            elif "listing" in content_name:
                schema = schemas.get("listing_schema.json")
            else:
                schema = schemas.get("generic_schema.json")
            
            # Measure pattern extraction time
            _, execution_time = measure_execution_time(
                pattern_extractor.extract, content, schema
            )
            
            results[f"pattern_{content_name}"] = execution_time
        
        # Output results
        for test_name, execution_time in results.items():
            print(f"{test_name}: {execution_time:.4f} seconds")
        
        # Ensure we have some results
        assert len(results) > 0, "No extraction speed results were generated"
    
    def test_memory_usage(self, pattern_extractor, semantic_extractor, html_content, schemas):
        """Test memory consumption during extraction."""
        results = {}
        
        # Test product page with both extractors
        product_content = html_content.get("product_page.html")
        product_schema = schemas.get("product_schema.json")
        
        if product_content and product_schema:
            # Measure pattern extraction memory usage
            _, memory_usage_pattern = measure_memory_usage(
                pattern_extractor.extract, product_content, product_schema
            )
            results["pattern_extractor_product"] = memory_usage_pattern
            
            # Mock the async call for semantic extraction
            with patch.object(semantic_extractor, 'extract_semantic_content', 
                             return_value={"title": "Test Product"}):
                _, memory_usage_semantic = measure_memory_usage(
                    semantic_extractor.extract, product_content, product_schema
                )
                results["semantic_extractor_product"] = memory_usage_semantic
        
        # Output results
        for test_name, memory_usage in results.items():
            print(f"{test_name}: {memory_usage:.2f} MB")
        
        # Ensure we have some results
        assert len(results) > 0, "No memory usage results were generated"
    
    def test_ai_vs_pattern_performance(self, pattern_extractor, semantic_extractor, html_content, schemas):
        """Compare performance between AI-based and pattern-based extraction."""
        # Use the product page for comparison
        product_content = html_content.get("product_page.html")
        product_schema = schemas.get("product_schema.json")
        
        if not product_content or not product_schema:
            pytest.skip("Product page or schema fixture not available")
        
        # Measure pattern extraction performance
        _, pattern_time = measure_execution_time(
            pattern_extractor.extract, product_content, product_schema
        )
        
        # Mock the async call for semantic extraction time measurement
        with patch.object(semantic_extractor, 'extract_semantic_content', 
                         return_value={"title": "Test Product"}):
            _, semantic_time = measure_execution_time(
                semantic_extractor.extract, product_content, product_schema
            )
        
        print(f"Pattern extraction time: {pattern_time:.4f} seconds")
        print(f"Semantic extraction time: {semantic_time:.4f} seconds")
        print(f"Ratio (Semantic/Pattern): {semantic_time/pattern_time:.2f}x")
        
        # No specific assertion, just performance comparison
        
    def test_large_document_performance(self, pattern_extractor, html_content):
        """Test extraction performance with large documents."""
        # Use a large document or combine multiple documents
        combined_content = ""
        for content in html_content.values():
            combined_content += content
        
        # Skip if the combined content is not large enough
        if len(combined_content) < 500000:
            # Duplicate the content to make it larger
            combined_content = combined_content * 3
        
        print(f"Testing with document size: {len(combined_content)/1024/1024:.2f} MB")
        
        # Measure extraction time and memory usage
        _, large_time = measure_execution_time(
            pattern_extractor.extract, combined_content, None
        )
        
        _, large_memory = measure_memory_usage(
            pattern_extractor.extract, combined_content, None
        )
        
        print(f"Large document extraction time: {large_time:.4f} seconds")
        print(f"Large document memory usage: {large_memory:.2f} MB")
        
        # No specific assertion, just performance measurement
    
    @pytest.mark.asyncio
    async def test_concurrent_extraction(self, pattern_extractor, html_content):
        """Test performance of parallel extraction."""
        # Prepare a list of extraction tasks
        contents = list(html_content.values())
        
        # Define extraction task
        def extract_task(content):
            return pattern_extractor.extract(content)
        
        # Sequential extraction for comparison
        sequential_start = time.time()
        sequential_results = []
        for content in contents:
            sequential_results.append(extract_task(content))
        sequential_time = time.time() - sequential_start
        
        # Parallel extraction
        parallel_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            parallel_results = list(executor.map(extract_task, contents))
        parallel_time = time.time() - parallel_start
        
        print(f"Sequential extraction time: {sequential_time:.4f} seconds")
        print(f"Parallel extraction time: {parallel_time:.4f} seconds")
        print(f"Speedup: {sequential_time/parallel_time:.2f}x")
        
        # Ensure we got the same number of results
        assert len(sequential_results) == len(parallel_results)
    
    def test_resource_utilization(self, pattern_extractor, semantic_extractor, html_content):
        """Measure CPU and memory utilization during extraction."""
        # Get a representative sample of content
        product_content = html_content.get("product_page.html", "")
        article_content = html_content.get("article_page.html", "")
        listing_content = html_content.get("listing_page.html", "")
        
        if not product_content or not article_content or not listing_content:
            pytest.skip("Not all required content fixtures are available")
        
        # Create a mix of content to process
        contents = [product_content, article_content, listing_content] * 2
        
        # Initialize metrics
        cpu_percent_before = psutil.cpu_percent(interval=0.1)
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Start time tracking
        start_time = time.time()
        
        # Process all content
        for content in contents:
            pattern_extractor.extract(content)
        
        # Measure elapsed time
        elapsed_time = time.time() - start_time
        
        # Measure resource usage
        cpu_percent_after = psutil.cpu_percent(interval=0.1)
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        cpu_increase = cpu_percent_after - cpu_percent_before
        memory_increase = memory_after - memory_before
        
        print(f"Processing {len(contents)} documents:")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"CPU utilization increase: {cpu_increase:.2f}%")
        print(f"Memory utilization increase: {memory_increase:.2f} MB")
        print(f"Per-document processing time: {elapsed_time/len(contents):.4f} seconds")
        
        # No specific assertions, just resource measurements


# Run this module directly to execute just the benchmarks
if __name__ == "__main__":
    pytest.main(["-v", __file__])
