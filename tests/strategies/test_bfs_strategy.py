"""
Tests for the BFS strategy implementation.
"""

import pytest
from unittest.mock import MagicMock, patch
import logging
from bs4 import BeautifulSoup

from strategies.bfs_strategy import BFSStrategy
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_types import StrategyType, StrategyCapability


# Mock HTML service for testing
class MockHTMLService:
    def clean_html(self, html):
        return html
    
    def extract_main_content(self, html):
        return html
    
    def extract_links(self, html, base_url=None):
        return [
            {"url": "http://example.com/page1"},
            {"url": "http://example.com/page2"},
            {"url": "http://different.com/external"}
        ]


# Mock URL service for testing
class MockURLService:
    def __init__(self):
        self.queue = MockURLQueue()
    
    def normalize_url(self, url, base_url=None):
        return url
    
    def is_allowed(self, url):
        return "disallowed" not in url
    
    def get_queue(self, name):
        return self.queue


# Mock URL queue for testing
class MockURLQueue:
    def __init__(self):
        self.urls = []
        self.visited = set()
        self.in_progress = set()
    
    def add(self, url):
        if url not in self.urls and url not in self.visited:
            self.urls.append(url)
            self.in_progress.add(url)
    
    def get(self):
        if self.urls:
            return self.urls.pop(0)
        return None
    
    def complete(self, url):
        if url in self.in_progress:
            self.in_progress.remove(url)
            self.visited.add(url)
    
    def is_visited(self, url):
        return url in self.visited
    
    def is_in_progress(self, url):
        return url in self.in_progress


# Create a mock context fixture
@pytest.fixture
def mock_context():
    context = MagicMock(spec=StrategyContext)
    html_service = MockHTMLService()
    url_service = MockURLService()
    
    # Setup mock services
    context.html_service = html_service
    context.url_service = url_service
    context.get_service.side_effect = lambda name: {
        "html_service": html_service,
        "url_service": url_service,
    }.get(name)
    
    # Setup logger
    context.logger = logging.getLogger("test_logger")
    
    return context


# Test cases
def test_bfs_strategy_initialization(mock_context):
    """Test BFS Strategy initialization with context."""
    strategy = BFSStrategy(mock_context)
    
    # Check initialization
    assert strategy.name == "bfs_strategy"
    assert isinstance(strategy.visited_urls, set)
    assert len(strategy.visited_urls) == 0
    assert isinstance(strategy.results, list)
    
    # Check capabilities
    metadata = getattr(BFSStrategy, '_metadata', None)
    assert metadata is not None
    assert metadata.strategy_type == StrategyType.TRAVERSAL
    assert StrategyCapability.ROBOTS_TXT_ADHERENCE in metadata.capabilities
    assert StrategyCapability.LINK_EXTRACTION in metadata.capabilities


def test_bfs_strategy_crawl(mock_context):
    """Test BFS Strategy crawl method."""
    strategy = BFSStrategy(mock_context)
    
    # Mock _fetch_url to return HTML
    strategy._fetch_url = MagicMock(return_value="""
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Test Page</h1>
                <p>This is a test page.</p>
                <a href="http://example.com/page1">Page 1</a>
                <a href="http://example.com/page2">Page 2</a>
                <a href="http://different.com/external">External</a>
            </body>
        </html>
    """)
    
    # Run crawl
    result = strategy.crawl("http://example.com")
    
    # Check result
    assert result is not None
    assert "results" in result
    assert "metrics" in result
    assert "visited_urls" in result
    
    # Should have visited at least one URL
    assert len(result["visited_urls"]) > 0
    assert "http://example.com" in result["visited_urls"]
    
    # Should have queued internal links but not external (by default)
    assert len(mock_context.url_service.queue.urls) == 2
    assert any("page1" in url for url in mock_context.url_service.queue.urls + list(mock_context.url_service.queue.visited))
    assert any("page2" in url for url in mock_context.url_service.queue.urls + list(mock_context.url_service.queue.visited))
    assert not any("external" in url for url in mock_context.url_service.queue.urls + list(mock_context.url_service.queue.visited))


def test_bfs_strategy_extract(mock_context):
    """Test BFS Strategy extract method."""
    strategy = BFSStrategy(mock_context)
    
    html = """
        <html>
            <head>
                <title>Test Title</title>
                <meta name="description" content="Test Description">
            </head>
            <body>
                <h1>Test Page</h1>
                <p>This is a test page with some content.</p>
                <a href="http://example.com/page1">Page 1</a>
            </body>
        </html>
    """
    
    result = strategy.extract(html, "http://example.com")
    
    # Check result structure
    assert result is not None
    assert "data" in result
    assert "confidence" in result
    
    # Check extracted data
    data = result["data"]
    assert data["title"] == "Test Title"
    assert data["description"] == "Test Description"
    assert "This is a test page" in data["content_sample"]
    assert data["url"] == "http://example.com"
    assert data["link_count"] == 3  # From MockHTMLService


def test_bfs_strategy_execute(mock_context):
    """Test BFS Strategy execute method."""
    strategy = BFSStrategy(mock_context)
    
    # Mock crawl method
    strategy.crawl = MagicMock(return_value={"results": ["result1", "result2"]})
    
    # Call execute
    result = strategy.execute("http://example.com")
    
    # Should delegate to crawl
    strategy.crawl.assert_called_once()
    assert result == {"results": ["result1", "result2"]}


def test_bfs_strategy_can_handle(mock_context):
    """Test BFS Strategy can_handle method."""
    strategy = BFSStrategy(mock_context)
    
    # Should handle HTTP/HTTPS URLs
    assert strategy.can_handle("http://example.com") == True
    assert strategy.can_handle("https://example.com") == True
    
    # Should not handle non-HTTP URLs
    assert strategy.can_handle("ftp://example.com") == False
    assert strategy.can_handle("file:///path/to/file") == False


def test_bfs_strategy_with_external_links(mock_context):
    """Test BFS Strategy with external links allowed."""
    strategy = BFSStrategy(mock_context, include_external=True)
    
    # Mock _fetch_url to return HTML
    strategy._fetch_url = MagicMock(return_value="""
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Test Page</h1>
                <a href="http://example.com/page1">Page 1</a>
                <a href="http://different.com/external">External</a>
            </body>
        </html>
    """)
    
    # Run crawl
    result = strategy.crawl("http://example.com")
    
    # Should have queued external links since include_external=True
    queued_urls = mock_context.url_service.queue.urls + list(mock_context.url_service.queue.visited)
    assert any("different.com" in url for url in queued_urls)


def test_bfs_strategy_respects_robots_txt(mock_context):
    """Test BFS Strategy respects robots.txt."""
    strategy = BFSStrategy(mock_context)
    
    # Mock _fetch_url to return HTML
    strategy._fetch_url = MagicMock(return_value="""
        <html>
            <head><title>Test Page</title></head>
            <body>
                <a href="http://example.com/disallowed">Disallowed Page</a>
            </body>
        </html>
    """)
    
    # Run crawl
    result = strategy.crawl("http://example.com")
    
    # Should not have visited or queued disallowed URLs
    all_urls = (mock_context.url_service.queue.urls + 
                list(mock_context.url_service.queue.visited) + 
                list(mock_context.url_service.queue.in_progress))
    assert not any("disallowed" in url for url in all_urls)


def test_bfs_strategy_error_handling(mock_context):
    """Test BFS Strategy error handling."""
    strategy = BFSStrategy(mock_context)
    
    # Mock _fetch_url to raise an exception
    strategy._fetch_url = MagicMock(side_effect=Exception("Test exception"))
    
    # Mock handle_error to track calls
    strategy.handle_error = MagicMock()
    
    # Run crawl
    result = strategy.crawl("http://example.com")
    
    # Should have handled the error
    assert strategy.handle_error.called
    
    # Should still return a result structure
    assert result is not None
    assert "results" in result
    assert "metrics" in result
    assert len(result["results"]) == 0


def test_bfs_strategy_backward_compatibility(mock_context):
    """Test BFS Strategy backward compatibility."""
    strategy = BFSStrategy(mock_context)
    
    # Test legacy _extract_links method
    html = """
        <html>
            <body>
                <a href="http://example.com/page1">Page 1</a>
                <a href="javascript:void(0)">JavaScript Link</a>
                <a href="http://example.com/page2">Page 2</a>
            </body>
        </html>
    """
    
    links = strategy._extract_links(html, "http://example.com")
    
    # Should extract valid links
    assert len(links) == 2
    assert "http://example.com/page1" in links
    assert "http://example.com/page2" in links
    
    # Should not extract JavaScript links
    assert "javascript:void(0)" not in links