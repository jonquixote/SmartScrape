"""
Test module for HTML processing pipeline stages.

This module contains tests for the HTMLCleaningStage, ContentExtractionStage, 
and other HTML processing stages in the pipeline.
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Tuple
from bs4 import BeautifulSoup
from lxml.html import fromstring, tostring

from core.pipeline.context import PipelineContext
from core.pipeline.dto import PipelineRequest, PipelineResponse, ResponseStatus
from core.pipeline.stages.processing.html_processing import (
    HTMLCleaningStage,
    ContentExtractionStage,
    CleaningStrategy,
    ExtractionAlgorithm,
    SelectorStrategy
)

# Sample HTML for testing
SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <script type="text/javascript">alert('hello');</script>
    <style>body { color: red; }</style>
    <meta name="description" content="Test description">
</head>
<body>
    <header>
        <nav>
            <ul>
                <li><a href="/">Home</a></li>
                <li><a href="/about">About</a></li>
            </ul>
        </nav>
    </header>
    <main id="main-content">
        <article>
            <h1>Test Article</h1>
            <div class="content">
                <p>This is a test paragraph with <b>bold</b> text.</p>
                <p>Another paragraph with <a href="https://example.com">a link</a>.</p>
            </div>
        </article>
    </main>
    <aside>
        <div class="sidebar">
            <h3>Related</h3>
            <ul>
                <li><a href="/related1">Related 1</a></li>
                <li><a href="/related2">Related 2</a></li>
            </ul>
        </div>
    </aside>
    <footer>
        <p>Copyright 2023</p>
    </footer>
    <!-- Comment that should be removed -->
</body>
</html>
"""

# HTML with problematic content
PROBLEMATIC_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Problematic Page</title>
    <script>malicious code;</script>
</head>
<body>
    <div class="content">
        <iframe src="https://example.com/frame"></iframe>
        <object data="flash.swf"></object>
        <form action="/submit">
            <input type="text" name="username">
            <input type="password" name="password">
            <button type="submit">Submit</button>
        </form>
    </div>
</body>
</html>
"""

# Create test fixtures
@pytest.fixture
def pipeline_context():
    """Create a pipeline context with sample HTML."""
    context = PipelineContext()
    return context

@pytest.fixture
def pipeline_request():
    """Create a pipeline request with sample HTML."""
    return PipelineRequest(
        source="test",
        data={"html": SAMPLE_HTML},
        metadata={"url": "https://example.com/test"}
    )

@pytest.fixture
def problematic_request():
    """Create a pipeline request with problematic HTML."""
    return PipelineRequest(
        source="test",
        data={"html": PROBLEMATIC_HTML},
        metadata={"url": "https://example.com/problematic"}
    )


class TestHTMLCleaningStage:
    """Tests for the HTMLCleaningStage class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test initialization with different configurations."""
        # Default initialization
        stage = HTMLCleaningStage()
        assert stage.strategy == CleaningStrategy.MODERATE
        
        # Custom initialization
        config = {
            "strategy": "LIGHT",
            "remove_elements": ["div", "span"],
            "safe_attrs": ["href", "src"],
            "allow_tags": ["a", "p", "h1"],
            "remove_unknown_tags": True,
            "kill_tags": ["script", "style"]
        }
        stage = HTMLCleaningStage(name="custom_cleaner", config=config)
        assert stage.name == "custom_cleaner"
        assert stage.strategy == CleaningStrategy.LIGHT
        assert "div" in stage.remove_elements
        assert "href" in stage.safe_attrs
        assert "a" in stage.allow_tags
        assert stage.remove_unknown_tags == True
        assert "script" in stage.kill_tags
    
    @pytest.mark.asyncio
    async def test_process_request_with_no_html(self):
        """Test processing a request with no HTML content."""
        stage = HTMLCleaningStage()
        request = PipelineRequest(source="test", data={}, metadata={})
        
        response = await stage.process_request(request, PipelineContext())
        assert response.status == ResponseStatus.ERROR
        assert "No HTML content found" in response.error_message
    
    @pytest.mark.asyncio
    async def test_process_request_with_non_string_html(self):
        """Test processing a request with non-string HTML content."""
        stage = HTMLCleaningStage()
        request = PipelineRequest(source="test", data={"html": 123}, metadata={})
        
        response = await stage.process_request(request, PipelineContext())
        assert response.status == ResponseStatus.ERROR
        assert "HTML content must be a string" in response.error_message
    
    @pytest.mark.asyncio
    @patch("core.pipeline.stages.processing.html_processing.time")
    async def test_clean_html_light_strategy(self, mock_time, pipeline_request):
        """Test HTML cleaning with LIGHT strategy."""
        # Setup time mock for consistent timing metrics
        mock_time.time.side_effect = [0, 1]  # Start, end
        
        config = {"strategy": "LIGHT"}
        stage = HTMLCleaningStage(config=config)
        
        response = await stage.process_request(pipeline_request, PipelineContext())
        
        # Check success
        assert response.status == ResponseStatus.SUCCESS
        
        # Check that HTML is cleaned but not too much
        cleaned_html = response.data["html"]
        assert cleaned_html is not None
        assert isinstance(cleaned_html, str)
        
        # Scripts and styles should be removed
        assert "<script" not in cleaned_html
        assert "<style" not in cleaned_html
        
        # But navigation and other elements should remain
        assert "<nav" in cleaned_html
        assert "<header" in cleaned_html
        assert "<footer" in cleaned_html
        
        # Check metrics
        assert "cleaning_metrics" in response.metadata
        assert response.metadata["cleaning_metrics"]["scripts_removed"] > 0
        assert response.metadata["cleaning_metrics"]["styles_removed"] > 0
        assert response.metadata["cleaning_metrics"]["processing_time"] == 1  # Mocked time difference
    
    @pytest.mark.asyncio
    async def test_clean_html_moderate_strategy(self, pipeline_request):
        """Test HTML cleaning with MODERATE strategy."""
        config = {"strategy": "MODERATE"}
        stage = HTMLCleaningStage(config=config)
        
        response = await stage.process_request(pipeline_request, PipelineContext())
        
        # Check success
        assert response.status == ResponseStatus.SUCCESS
        
        # Check that HTML is cleaned more aggressively
        cleaned_html = response.data["html"]
        
        # Scripts, styles, and navigation should be removed
        assert "<script" not in cleaned_html
        assert "<style" not in cleaned_html
        assert "<nav" not in cleaned_html
        
        # But main content should remain
        assert "<main" in cleaned_html
        assert "<article" in cleaned_html
        assert "<p" in cleaned_html
        assert "test paragraph" in cleaned_html
    
    @pytest.mark.asyncio
    async def test_clean_html_aggressive_strategy(self, pipeline_request):
        """Test HTML cleaning with AGGRESSIVE strategy."""
        config = {"strategy": "AGGRESSIVE"}
        stage = HTMLCleaningStage(config=config)
        
        response = await stage.process_request(pipeline_request, PipelineContext())
        
        # Check success
        assert response.status == ResponseStatus.SUCCESS
        
        # Check that HTML is cleaned very aggressively
        cleaned_html = response.data["html"]
        
        # Most structure elements should be removed
        assert "<header" not in cleaned_html
        assert "<footer" not in cleaned_html
        assert "<aside" not in cleaned_html
        
        # But core content should remain
        assert "test paragraph" in cleaned_html
        assert "bold" in cleaned_html
    
    @pytest.mark.asyncio
    async def test_clean_html_custom_strategy(self, pipeline_request):
        """Test HTML cleaning with CUSTOM strategy."""
        config = {
            "strategy": "CUSTOM",
            "remove_scripts": True,
            "remove_comments": True,
            "remove_styles": True,
            "kill_tags": ["iframe", "object", "embed"],
            "allow_tags": ["p", "h1", "b", "i", "a"],
            "safe_attrs": ["href", "title"],
            "remove_unknown_tags": True
        }
        stage = HTMLCleaningStage(config=config)
        
        response = await stage.process_request(pipeline_request, PipelineContext())
        
        # Check success
        assert response.status == ResponseStatus.SUCCESS
        
        # Check that HTML is cleaned according to custom config
        cleaned_html = response.data["html"]
        
        # Check that only allowed tags remain
        soup = BeautifulSoup(cleaned_html, "html.parser")
        allowed_tags = ["p", "h1", "b", "i", "a", "html", "body"]
        
        for tag in soup.find_all():
            assert tag.name in allowed_tags, f"Tag {tag.name} should have been removed"
            
        # Check that only safe attributes remain
        for tag in soup.find_all("a"):
            for attr in tag.attrs:
                assert attr in ["href", "title"], f"Attribute {attr} should have been removed"
    
    @pytest.mark.asyncio
    async def test_clean_html_problematic_content(self, problematic_request):
        """Test cleaning HTML with problematic content."""
        config = {"strategy": "MODERATE"}
        stage = HTMLCleaningStage(config=config)
        
        response = await stage.process_request(problematic_request, PipelineContext())
        
        # Check success
        assert response.status == ResponseStatus.SUCCESS
        
        # Check that problematic elements are removed
        cleaned_html = response.data["html"]
        assert "<iframe" not in cleaned_html
        assert "<object" not in cleaned_html
        assert "<script" not in cleaned_html
    
    @pytest.mark.asyncio
    @patch("core.pipeline.stages.processing.html_processing.ServiceRegistry")
    async def test_clean_with_html_service(self, mock_registry, pipeline_request):
        """Test cleaning HTML using the HTML service."""
        # Mock HTML service
        mock_html_service = MagicMock()
        mock_html_service.clean_html.return_value = {
            "cleaned_html": "<html><body><p>Service cleaned HTML</p></body></html>",
            "metrics": {
                "elements_removed": 10,
                "attributes_removed": 5,
                "scripts_removed": 1,
                "styles_removed": 1,
                "comments_removed": 1
            }
        }
        
        # Mock registry to return our mock service
        mock_registry_instance = MagicMock()
        mock_registry_instance.get_service.return_value = mock_html_service
        mock_registry.return_value = mock_registry_instance
        
        # Create stage and process request
        stage = HTMLCleaningStage()
        response = await stage.process_request(pipeline_request, PipelineContext())
        
        # Check that the service was used
        mock_html_service.clean_html.assert_called_once()
        
        # Check success
        assert response.status == ResponseStatus.SUCCESS
        assert response.data["html"] == "<html><body><p>Service cleaned HTML</p></body></html>"
        
        # Check metrics
        assert response.metadata["cleaning_metrics"]["elements_removed"] == 10
        assert response.metadata["cleaning_metrics"]["scripts_removed"] == 1


class TestContentExtractionStage:
    """Tests for the ContentExtractionStage class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test initialization with different configurations."""
        # Default initialization
        stage = ContentExtractionStage()
        assert stage.algorithm == ExtractionAlgorithm.DENSITY_BASED
        assert len(stage.selectors) > 0  # Should have default selectors
        
        # Custom initialization
        config = {
            "algorithm": "SEMANTIC_BASED",
            "selectors": ["#content", ".main-article", "article"],
            "min_content_length": 50,
            "require_heading": True,
            "quality_metrics": True,
            "fallback_to_body": False
        }
        stage = ContentExtractionStage(name="custom_extractor", config=config)
        assert stage.name == "custom_extractor"
        assert stage.algorithm == ExtractionAlgorithm.SEMANTIC_BASED
        assert "#content" in stage.selectors
        assert stage.min_content_length == 50
        assert stage.require_heading == True
        assert stage.quality_metrics == True
        assert stage.fallback_to_body == False
    
    @pytest.mark.asyncio
    async def test_process_request_with_no_html(self):
        """Test processing a request with no HTML content."""
        stage = ContentExtractionStage()
        request = PipelineRequest(source="test", data={}, metadata={})
        
        response = await stage.process_request(request, PipelineContext())
        assert response.status == ResponseStatus.ERROR
        assert "No HTML content found" in response.error_message
    
    @pytest.mark.asyncio
    async def test_process_request_with_non_string_html(self):
        """Test processing a request with non-string HTML content."""
        stage = ContentExtractionStage()
        request = PipelineRequest(source="test", data={"html": 123}, metadata={})
        
        response = await stage.process_request(request, PipelineContext())
        assert response.status == ResponseStatus.ERROR
        assert "HTML content must be a string" in response.error_message
    
    @pytest.mark.asyncio
    @patch("core.pipeline.stages.processing.html_processing.ServiceRegistry")
    async def test_extract_with_html_service(self, mock_registry, pipeline_request):
        """Test extracting content using the HTML service."""
        # Mock HTML service
        mock_html_service = MagicMock()
        mock_html_service.extract_main_content.return_value = {
            "content": "<article><h1>Test Article</h1><p>Extracted content</p></article>",
            "metrics": {
                "content_length": 57,
                "extraction_method": "semantic",
                "content_to_code_ratio": 0.45,
                "has_heading": True
            }
        }
        
        # Mock registry to return our mock service
        mock_registry_instance = MagicMock()
        mock_registry_instance.get_service.return_value = mock_html_service
        mock_registry.return_value = mock_registry_instance
        
        # Create stage and process request
        stage = ContentExtractionStage()
        response = await stage.process_request(pipeline_request, PipelineContext())
        
        # Check that the service was used
        mock_html_service.extract_main_content.assert_called_once()
        
        # Check success
        assert response.status == ResponseStatus.SUCCESS
        assert response.data["html"] == "<article><h1>Test Article</h1><p>Extracted content</p></article>"
        
        # Check metrics
        assert response.metadata["extraction_metrics"]["content_length"] == 57
        assert response.metadata["extraction_metrics"]["has_heading"] == True
    
    @pytest.mark.asyncio
    async def test_custom_selector_extraction(self, pipeline_request):
        """Test extraction using custom selectors."""
        config = {
            "algorithm": "CUSTOM_SELECTOR",
            "selectors": ["#main-content", "article", ".content"]
        }
        stage = ContentExtractionStage(config=config)
        
        response = await stage.process_request(pipeline_request, PipelineContext())
        
        # Check success
        assert response.status == ResponseStatus.SUCCESS
        
        # Check that main content was extracted
        extracted_html = response.data["html"]
        assert "<h1>Test Article</h1>" in extracted_html
        assert "test paragraph" in extracted_html
        
        # Check that non-content elements are not included
        assert "<header" not in extracted_html
        assert "<footer" not in extracted_html
        assert "<nav" not in extracted_html
    
    @pytest.mark.asyncio
    async def test_semantic_based_extraction(self, pipeline_request):
        """Test extraction using semantic analysis."""
        config = {
            "algorithm": "SEMANTIC_BASED"
        }
        stage = ContentExtractionStage(config=config)
        
        response = await stage.process_request(pipeline_request, PipelineContext())
        
        # Check success
        assert response.status == ResponseStatus.SUCCESS
        
        # Check that main content was extracted based on semantic elements
        extracted_html = response.data["html"]
        assert "<h1>Test Article</h1>" in extracted_html
        assert "test paragraph" in extracted_html
    
    @pytest.mark.asyncio
    async def test_density_based_extraction(self, pipeline_request):
        """Test extraction using text density analysis."""
        config = {
            "algorithm": "DENSITY_BASED"
        }
        stage = ContentExtractionStage(config=config)
        
        response = await stage.process_request(pipeline_request, PipelineContext())
        
        # Check success
        assert response.status == ResponseStatus.SUCCESS
        
        # Check that content with highest text density was extracted
        extracted_html = response.data["html"]
        assert "test paragraph" in extracted_html
    
    @pytest.mark.asyncio
    async def test_extraction_with_min_content_length(self, pipeline_request):
        """Test extraction with minimum content length requirement."""
        # Set a very high minimum content length
        config = {
            "min_content_length": 10000,
            "fallback_to_body": True
        }
        stage = ContentExtractionStage(config=config)
        
        response = await stage.process_request(pipeline_request, PipelineContext())
        
        # Should still succeed but fallback to body content
        assert response.status == ResponseStatus.SUCCESS
        
        # Should contain more than just the main content
        extracted_html = response.data["html"]
        assert "<body" in extracted_html
    
    @pytest.mark.asyncio
    async def test_extraction_with_heading_requirement(self, pipeline_request):
        """Test extraction with heading requirement."""
        config = {
            "require_heading": True,
            "fallback_to_body": True
        }
        stage = ContentExtractionStage(config=config)
        
        response = await stage.process_request(pipeline_request, PipelineContext())
        
        # Should succeed since main content has a heading
        assert response.status == ResponseStatus.SUCCESS
        
        # Should include the heading
        extracted_html = response.data["html"]
        assert "<h1>Test Article</h1>" in extracted_html


class TestSelectorStrategy:
    """Tests for the SelectorStrategy enum and related functionality."""
    
    def test_selector_strategy_values(self):
        """Test that SelectorStrategy enum has expected values."""
        assert SelectorStrategy.FIRST_MATCH is not None
        assert SelectorStrategy.ALL_MATCHES is not None
        assert SelectorStrategy.PRIORITY_BASED is not None
        assert SelectorStrategy.COMPOSITE is not None
    
    def test_selector_strategy_to_string(self):
        """Test string representation of SelectorStrategy."""
        assert str(SelectorStrategy.FIRST_MATCH) == "SelectorStrategy.FIRST_MATCH"
        assert str(SelectorStrategy.ALL_MATCHES) == "SelectorStrategy.ALL_MATCHES"
        assert str(SelectorStrategy.PRIORITY_BASED) == "SelectorStrategy.PRIORITY_BASED"
        assert str(SelectorStrategy.COMPOSITE) == "SelectorStrategy.COMPOSITE"


if __name__ == "__main__":
    pytest.main()