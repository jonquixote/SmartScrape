"""
Core Services Test Fixtures

This module provides common pytest fixtures for testing the core services of SmartScrape.
"""
import pytest
from unittest.mock import MagicMock
from bs4 import BeautifulSoup

from core.service_interface import BaseService
from core.service_registry import ServiceRegistry

# Mock Service Fixtures
class MockService(BaseService):
    """A simple mock service implementation for testing."""
    def __init__(self):
        self._initialized = False
        self._config = None
    
    def initialize(self, config=None):
        self._initialized = True
        self._config = config
    
    def shutdown(self):
        self._initialized = False
    
    @property
    def name(self):
        return "mock_service"

class MockURLService(BaseService):
    """Mock URL service for testing."""
    def __init__(self):
        self._initialized = False
        self._config = None
        
        # Mock methods
        self.normalize_url = MagicMock(side_effect=lambda url, base_url=None: url)
        self.is_allowed = MagicMock(return_value=True)
        self.get_crawl_delay = MagicMock(return_value=None)
        self.get_sitemaps = MagicMock(return_value=[])
        self.get_queue = MagicMock()
    
    def initialize(self, config=None):
        self._initialized = True
        self._config = config
    
    def shutdown(self):
        self._initialized = False
    
    @property
    def name(self):
        return "url_service"

class MockHTMLService(BaseService):
    """Mock HTML service for testing."""
    def __init__(self):
        self._initialized = False
        self._config = None
        
        # Mock methods
        self.clean_html = MagicMock(side_effect=lambda html, **kwargs: html)
        self.generate_selector = MagicMock(return_value="div > p")
        self.compare_elements = MagicMock(return_value=0.8)
        self.extract_main_content = MagicMock(side_effect=lambda html: html)
        self.extract_tables = MagicMock(return_value=[])
        self.extract_links = MagicMock(return_value=[])
    
    def initialize(self, config=None):
        self._initialized = True
        self._config = config
    
    def shutdown(self):
        self._initialized = False
    
    @property
    def name(self):
        return "html_service"

@pytest.fixture
def mock_service():
    """Return a simple mock service."""
    return MockService()

@pytest.fixture
def mock_url_service():
    """Return a mock URL service with pre-configured methods."""
    return MockURLService()

@pytest.fixture
def mock_html_service():
    """Return a mock HTML service with pre-configured methods."""
    return MockHTMLService()

@pytest.fixture
def service_registry():
    """Return a clean service registry for each test."""
    registry = ServiceRegistry()
    # Store the original _instance to restore after test
    original_instance = ServiceRegistry._instance
    ServiceRegistry._instance = None
    registry = ServiceRegistry()
    
    yield registry
    
    # Restore original _instance after test
    registry.shutdown_all()
    ServiceRegistry._instance = original_instance

# HTML Test Content Fixtures
@pytest.fixture
def simple_html():
    """Return a simple HTML document for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
        <script>console.log("test");</script>
        <style>.test { color: red; }</style>
    </head>
    <body>
        <div id="main">
            <h1>Test Heading</h1>
            <p class="content">This is test content.</p>
            <ul class="list">
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
        <div id="sidebar">
            <h2>Related</h2>
            <ul>
                <li><a href="/page1">Link 1</a></li>
                <li><a href="https://example.com">External Link</a></li>
            </ul>
        </div>
        <div style="display:none;">Hidden content</div>
        <!-- Comment that should be removed -->
    </body>
    </html>
    """

@pytest.fixture
def simple_soup(simple_html):
    """Return a BeautifulSoup object for the simple HTML."""
    return BeautifulSoup(simple_html, 'lxml')

@pytest.fixture
def table_html():
    """Return HTML with a table for testing table extraction."""
    return """
    <!DOCTYPE html>
    <html>
    <body>
        <h1>Table Test</h1>
        <table id="data-table">
            <caption>Sample Data</caption>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>1</td>
                    <td>Item A</td>
                    <td>10.5</td>
                </tr>
                <tr>
                    <td>2</td>
                    <td>Item B</td>
                    <td>20.75</td>
                </tr>
            </tbody>
        </table>
    </body>
    </html>
    """

@pytest.fixture
def links_html():
    """Return HTML with various link types for testing link extraction."""
    return """
    <!DOCTYPE html>
    <html>
    <body>
        <h1>Links Test</h1>
        <div class="navigation">
            <a href="/">Home</a>
            <a href="/about">About</a>
            <a href="/contact">Contact</a>
        </div>
        <div class="external-links">
            <a href="https://example.com" title="Example Site">Example</a>
            <a href="https://test.com" rel="nofollow">Test Site</a>
            <a href="mailto:test@example.com">Email Us</a>
        </div>
        <div class="product-links">
            <a href="/products/1">Product 1</a>
            <a href="/products/2?ref=home&utm_source=homepage">Product 2</a>
        </div>
    </body>
    </html>
    """

# URL Test Fixtures
@pytest.fixture
def test_urls():
    """Return a list of URLs for testing URL service."""
    return [
        "http://example.com",
        "https://test.com/page",
        "https://shop.example.com/products/123",
        "http://example.com:80",
        "https://example.com:443",
        "http://example.com/path with spaces",
        "http://example.com/?param1=value1&utm_source=test",
        "https://example.com/path/to/resource#fragment",
        "/relative/path",
        "//example.com/protocol-relative",
        "https://example.com/path/../resolved",
        "https://EXAMPLE.COM/path",
    ]

@pytest.fixture
def base_url():
    """Return a base URL for testing relative URL resolution."""
    return "https://example.com/base/"