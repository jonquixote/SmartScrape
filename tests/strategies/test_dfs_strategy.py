"""
Tests for the DFS Strategy implementation.

This module tests the DFS (Depth-First Search) strategy, focusing on its
crawl behavior, URL handling, and integration with the strategy pattern.
"""

import pytest
from unittest.mock import MagicMock, patch
import logging
from typing import Dict, Any, Optional, List, Set
from urllib.parse import urlparse

from strategies.dfs_strategy import DFSStrategy
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_types import StrategyType, StrategyCapability
from strategies.core.strategy_error_handler import StrategyErrorCategory, StrategyErrorSeverity

# Mock HTML content
MOCK_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <meta name="description" content="This is a test page for DFS strategy">
</head>
<body>
    <div id="content">
        <p>This is the main content of the test page.</p>
        <a href="http://example.com/page1">Link 1</a>
        <a href="http://example.com/page2">Link 2</a>
        <a href="http://external.com/page">External Link</a>
        <a href="javascript:void(0)">JavaScript Link</a>
        <a href="#section">Anchor Link</a>
    </div>
</body>
</html>
"""

# Mock Services
class MockURLService:
    def normalize_url(self, url, base_url=None):
        return url
    
    def is_allowed(self, url):
        return "disallowed" not in url

class MockHTMLService:
    def clean_html(self, html):
        return html
    
    def extract_main_content(self, html):
        return f"<div>{html.split('<div')[1].split('</div>')[0]}</div>"
    
    def extract_links(self, html, base_url=None):
        if "page1" in html:
            return [{"url": "http://example.com/subpage1"}, {"url": "http://example.com/subpage2"}]
        elif "page2" in html:
            return [{"url": "http://example.com/subpage3"}, {"url": "http://external.com/page"}]
        else:
            return [
                {"url": "http://example.com/page1"}, 
                {"url": "http://example.com/page2"},
                {"url": "http://external.com/page"}
            ]

# Mock StrategyContext
class MockContext:
    def __init__(self):
        self.service_registry = MagicMock()
        self.service_registry.get_service.side_effect = lambda name: {
            "url_service": MockURLService(),
            "html_service": MockHTMLService(),
        }.get(name)

# Test cases
def test_dfs_strategy_initialization():
    """Test DFSStrategy initializes correctly."""
    context = MockContext()
    dfs = DFSStrategy(context=context)
    
    assert dfs.name == "dfs_strategy"
    assert dfs.url_stack == []
    assert dfs.visited_urls == set()
    assert dfs.results == []
    assert dfs.config['max_depth'] == 3
    assert dfs.config['max_pages'] == 100
    assert dfs.config['include_external'] is False

def test_dfs_strategy_can_handle():
    """Test the can_handle method returns True for HTTP/HTTPS URLs."""
    context = MockContext()
    dfs = DFSStrategy(context=context)
    
    assert dfs.can_handle("http://example.com") is True
    assert dfs.can_handle("https://example.com") is True
    assert dfs.can_handle("ftp://example.com") is False
    assert dfs.can_handle("file:///path/to/file") is False

@patch('strategies.dfs_strategy.DFSStrategy._fetch_url')
def test_dfs_crawl_basic(mock_fetch_url):
    """Test the basic DFS crawl functionality."""
    # Setup
    context = MockContext()
    dfs = DFSStrategy(context=context)
    
    # Mock _fetch_url to return HTML content
    mock_fetch_url.return_value = MOCK_HTML
    
    # Execute crawl
    result = dfs.crawl("http://example.com")
    
    # Assertions
    assert mock_fetch_url.call_count > 0
    assert len(dfs.visited_urls) > 0
    assert "http://example.com" in dfs.visited_urls
    assert result is not None
    assert "results" in result
    assert "metrics" in result
    assert "visited_urls" in result
    
    # Verify metrics
    assert result["metrics"]["pages_visited"] == len(dfs.visited_urls)
    assert isinstance(result["visited_urls"], list)

@patch('strategies.dfs_strategy.DFSStrategy._fetch_url')
def test_dfs_crawl_respects_max_depth(mock_fetch_url):
    """Test that DFS respects the max_depth configuration."""
    # Setup
    context = MockContext()
    dfs = DFSStrategy(context=context)
    
    # Mock _fetch_url to return HTML content
    mock_fetch_url.return_value = MOCK_HTML
    
    # Execute crawl with max_depth=1
    result = dfs.crawl("http://example.com", max_depth=1)
    
    # Assertions
    assert len(dfs.visited_urls) <= 3  # Start URL + 2 direct links
    
    # No URLs should be visited at depth > 1
    for url, depth in dfs.url_stack:
        assert depth <= 1

@patch('strategies.dfs_strategy.DFSStrategy._fetch_url')
def test_dfs_crawl_respects_max_pages(mock_fetch_url):
    """Test that DFS respects the max_pages configuration."""
    # Setup
    context = MockContext()
    dfs = DFSStrategy(context=context)
    
    # Mock _fetch_url to return HTML content
    mock_fetch_url.return_value = MOCK_HTML
    
    # Execute crawl with max_pages=2
    result = dfs.crawl("http://example.com", max_pages=2)
    
    # Assertions
    assert len(dfs.visited_urls) <= 2

@patch('strategies.dfs_strategy.DFSStrategy._fetch_url')
def test_dfs_crawl_external_domains(mock_fetch_url):
    """Test DFS handling of external domains based on configuration."""
    # Setup
    context = MockContext()
    dfs = DFSStrategy(context=context)
    
    # Mock _fetch_url to return HTML content
    mock_fetch_url.return_value = MOCK_HTML
    
    # Test with include_external=False (default)
    result1 = dfs.crawl("http://example.com")
    visited1 = dfs.visited_urls.copy()
    
    # Reset and test with include_external=True
    dfs.visited_urls = set()
    dfs.url_stack = []
    result2 = dfs.crawl("http://example.com", include_external=True)
    visited2 = dfs.visited_urls.copy()
    
    # Assertions
    assert "http://external.com/page" not in visited1
    assert len(visited2) >= len(visited1)  # Should visit more URLs with include_external=True

@patch('strategies.dfs_strategy.DFSStrategy._fetch_url')
def test_dfs_extract(mock_fetch_url):
    """Test the extract method of DFS strategy."""
    # Setup
    context = MockContext()
    dfs = DFSStrategy(context=context)
    
    # Execute extract
    result = dfs.extract(MOCK_HTML, "http://example.com")
    
    # Assertions
    assert result is not None
    assert "data" in result
    assert "confidence" in result
    assert result["data"]["title"] == "Test Page"
    assert result["data"]["description"] == "This is a test page for DFS strategy"
    assert "content_sample" in result["data"]
    assert result["data"]["url"] == "http://example.com"
    assert result["confidence"] == 0.6  # Default confidence for DFS

def test_dfs_extract_error_handling():
    """Test error handling in the extract method."""
    # Setup
    context = MockContext()
    dfs = DFSStrategy(context=context)
    
    # Execute extract with invalid HTML
    result = dfs.extract("invalid html", "http://example.com")
    
    # Assertions
    assert result is None  # Should return None on error
    
    # Verify error was handled
    # Note: This would require inspecting the error handler, which might be complex to test

@patch('strategies.dfs_strategy.DFSStrategy.crawl')
def test_dfs_execute(mock_crawl):
    """Test the execute method that delegates to crawl."""
    # Setup
    context = MockContext()
    dfs = DFSStrategy(context=context)
    mock_crawl.return_value = {"results": [{"url": "http://example.com"}]}
    
    # Execute
    result = dfs.execute("http://example.com", max_depth=2)
    
    # Assertions
    mock_crawl.assert_called_once_with("http://example.com", max_depth=2)
    assert result == {"results": [{"url": "http://example.com"}]}

def test_dfs_extract_links():
    """Test the _extract_links method for legacy compatibility."""
    # Setup
    context = MockContext()
    dfs = DFSStrategy(context=context)
    
    # Execute
    links = dfs._extract_links(MOCK_HTML, "http://example.com")
    
    # Assertions
    assert isinstance(links, list)
    assert len(links) >= 2
    assert "http://example.com/page1" in links
    assert "http://example.com/page2" in links
    assert "javascript:void(0)" not in links  # Should filter JavaScript links
    assert "#section" not in links  # Should filter anchor links
    
    # Test with empty HTML
    links_empty = dfs._extract_links("", "http://example.com")
    assert links_empty == []