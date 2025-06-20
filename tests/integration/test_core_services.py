"""
Integration tests for core services of SmartScrape.

These tests verify that the core services work together properly:
- Service registry managing multiple services
- URL service and HTML service working together
- Strategy classes using services correctly
"""
import os
import sys
import pytest
from bs4 import BeautifulSoup

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.service_registry import ServiceRegistry
from core.service_interface import BaseService
from core.url_service import URLService
from core.html_service import HTMLService
from strategies.base_strategy import BaseStrategy, get_crawl_strategy

# Create a simple test service for integration testing
class TestService(BaseService):
    """A test service for integration testing."""
    
    def __init__(self):
        self._initialized = False
        self._config = None
        self.url_service = None
        self.html_service = None
    
    def initialize(self, config=None):
        """Initialize the test service."""
        self._initialized = True
        self._config = config
        
        # Get dependencies from service registry
        registry = ServiceRegistry()
        self.url_service = registry.get_service("url_service")
        self.html_service = registry.get_service("html_service")
    
    def shutdown(self):
        """Shutdown the test service."""
        self._initialized = False
    
    @property
    def name(self):
        """Return the name of the service."""
        return "test_service"
    
    def process_url(self, url, html):
        """Process a URL and HTML content using dependent services."""
        # Use URL service to normalize the URL
        normalized_url = self.url_service.normalize_url(url)
        
        # Use HTML service to clean the HTML
        cleaned_html = self.html_service.clean_html(html)
        
        # Extract links using HTML service
        links = self.html_service.extract_links(cleaned_html, normalized_url)
        
        # Process links with URL service
        normalized_links = [self.url_service.normalize_url(link["url"]) for link in links]
        
        return {
            "url": normalized_url,
            "content": cleaned_html,
            "links": normalized_links
        }


class TestCoreServicesIntegration:
    """Test the integration between various core services."""
    
    def setup_method(self):
        """Set up the test environment."""
        # Reset the service registry for each test
        ServiceRegistry._instance = None
        self.registry = ServiceRegistry()
        
        # Register core services
        self.registry.register_service_class(URLService)
        self.registry.register_service_class(HTMLService)
        self.registry.register_service_class(TestService, dependencies={"url_service", "html_service"})
    
    def teardown_method(self):
        """Clean up the test environment."""
        if ServiceRegistry._instance:
            ServiceRegistry._instance.shutdown_all()
            ServiceRegistry._instance = None
    
    def test_service_registry_management(self):
        """Test that the service registry properly manages multiple services."""
        # Get services from registry
        url_service = self.registry.get_service("url_service")
        html_service = self.registry.get_service("html_service")
        
        # Verify services are initialized and of correct type
        assert url_service.is_initialized
        assert html_service.is_initialized
        assert isinstance(url_service, URLService)
        assert isinstance(html_service, HTMLService)
        
        # Verify services are singletons within the registry
        url_service2 = self.registry.get_service("url_service")
        html_service2 = self.registry.get_service("html_service")
        assert url_service is url_service2
        assert html_service is html_service2
        
        # Test shutdown all services
        self.registry.shutdown_all()
        assert not url_service.is_initialized
        assert not html_service.is_initialized
    
    def test_url_and_html_service_integration(self):
        """Test that URL service and HTML service work together properly."""
        url_service = self.registry.get_service("url_service")
        html_service = self.registry.get_service("html_service")
        
        # Sample HTML with links
        html = """
        <html>
        <body>
            <div id="content">
                <h1>Test Page</h1>
                <p>This is a test page with links.</p>
                <a href="/relative">Relative Link</a>
                <a href="https://example.com">External Link</a>
            </div>
        </body>
        </html>
        """
        
        # Use HTML service to extract links
        links = html_service.extract_links(html, base_url="https://test.com")
        
        # Use URL service to normalize extracted links
        normalized_links = [url_service.normalize_url(link["url"]) for link in links]
        
        # Verify links are properly extracted and normalized
        assert len(links) == 2
        assert "https://test.com/relative" in normalized_links
        assert "https://example.com/" in normalized_links
        
        # Test HTML cleaning and link extraction together
        cleaned_html = html_service.clean_html(html)
        links_from_cleaned = html_service.extract_links(cleaned_html, base_url="https://test.com")
        
        # Verify links are still present after cleaning
        assert len(links_from_cleaned) == 2
    
    def test_dependency_resolution(self):
        """Test that service dependencies are properly resolved."""
        # Get test service with dependencies on URL and HTML services
        test_service = self.registry.get_service("test_service")
        
        # Verify dependencies were properly injected
        assert test_service.url_service is not None
        assert test_service.html_service is not None
        assert isinstance(test_service.url_service, URLService)
        assert isinstance(test_service.html_service, HTMLService)
    
    def test_service_integration_workflow(self):
        """Test a complete workflow using multiple services."""
        test_service = self.registry.get_service("test_service")
        
        # Sample URL and HTML
        url = "https://example.com/page?utm_source=test"
        html = """
        <html>
        <head>
            <script>alert('test');</script>
        </head>
        <body>
            <div id="content">
                <a href="/page1">Link 1</a>
                <a href="https://other.com/page2">Link 2</a>
            </div>
        </body>
        </html>
        """
        
        # Process using the test service which uses URL and HTML services
        result = test_service.process_url(url, html)
        
        # Verify results
        assert result["url"] == "https://example.com/page"  # Tracking params removed
        assert "script" not in result["content"]  # Script removed by HTML service
        assert len(result["links"]) == 2  # Both links processed
        assert "https://example.com/page1" in result["links"]  # Relative link resolved
        assert any(link.startswith("https://other.com/page2") for link in result["links"])  # External link normalized
    
    def test_strategy_classes_using_services(self):
        """Test that strategy classes properly use URL and HTML services."""
        try:
            # Create a BFS strategy which should use the services
            bfs_strategy = get_crawl_strategy("bfs", max_depth=2, max_pages=10)
            
            # Verify strategy has initialized services
            assert hasattr(bfs_strategy, "url_service")
            assert hasattr(bfs_strategy, "html_service")
            assert bfs_strategy.url_service is not None
            assert bfs_strategy.html_service is not None
            
            # Verify services are of the correct types
            assert isinstance(bfs_strategy.url_service, URLService)
            assert isinstance(bfs_strategy.html_service, HTMLService)
            
            # Test that the strategy's get_next_urls method uses the services
            html_content = """
            <html>
            <body>
                <a href="/page1">Page 1</a>
                <a href="https://example.com/page2">Page 2</a>
            </body>
            </html>
            """
            
            # Use the strategy to process URLs
            next_urls = bfs_strategy.get_next_urls(
                url="https://example.com", 
                html=html_content, 
                depth=1, 
                visited=set()
            )
            
            # Verify URLs were processed correctly
            assert len(next_urls) == 2
            urls = [item["url"] for item in next_urls]
            assert "https://example.com/page1" in urls
            assert "https://example.com/page2" in urls
            
        except ImportError as e:
            # Skip test if strategies aren't available
            pytest.skip(f"Could not test strategy classes: {str(e)}")


class TestEndToEndCoreServices:
    """End-to-end test for core services working together."""
    
    def setup_method(self):
        """Set up the test environment."""
        # Reset the service registry
        ServiceRegistry._instance = None
        self.registry = ServiceRegistry()
        
        # Register core services
        self.registry.register_service_class(URLService)
        self.registry.register_service_class(HTMLService)
    
    def teardown_method(self):
        """Clean up the test environment."""
        if ServiceRegistry._instance:
            ServiceRegistry._instance.shutdown_all()
            ServiceRegistry._instance = None
    
    def test_end_to_end_core_flow(self):
        """
        Test an end-to-end flow of the core services:
        - Initialize all services
        - Process a sample URL
        - Extract content using HTML service
        - Verify all components work together
        """
        # 1. Initialize all services
        url_service = self.registry.get_service("url_service")
        html_service = self.registry.get_service("html_service")
        
        # 2. Mock processing a sample URL
        sample_url = "http://example.com/page?utm_source=test"
        
        # Normalize the URL (remove tracking parameters)
        normalized_url = url_service.normalize_url(sample_url)
        assert normalized_url == "http://example.com/page"
        
        # Check if robots.txt allows crawling
        is_allowed = url_service.is_allowed(normalized_url)
        assert isinstance(is_allowed, bool)
        
        # Add URL to a queue for processing
        queue = url_service.get_queue("test_queue")
        added = queue.add(normalized_url)
        assert added
        
        # Get URL from queue
        next_url = queue.get()
        assert next_url == normalized_url
        
        # 3. Mock HTML processing
        sample_html = """
        <html>
        <head>
            <title>Test Page</title>
            <script>alert('test');</script>
            <style>.test { color: red; }</style>
        </head>
        <body>
            <div id="content">
                <h1>Sample Content</h1>
                <p>This is a sample content for testing.</p>
                <a href="/page1">Page 1</a>
                <a href="https://example.com/page2">Page 2</a>
            </div>
            <div style="display:none;">Hidden content</div>
        </body>
        </html>
        """
        
        # Clean the HTML
        cleaned_html = html_service.clean_html(sample_html)
        assert "script" not in cleaned_html
        assert "style" not in cleaned_html
        assert "Hidden content" not in cleaned_html
        assert "Sample Content" in cleaned_html
        
        # Extract the main content
        main_content = html_service.extract_main_content(cleaned_html)
        assert "Sample Content" in main_content
        assert "This is a sample content for testing" in main_content
        
        # Extract links
        links = html_service.extract_links(cleaned_html, base_url=normalized_url)
        assert len(links) == 2
        
        # Process extracted links with URL service
        for link in links:
            # Normalize and add to queue
            norm_link = url_service.normalize_url(link["url"])
            queue.add(norm_link)
        
        # Check queue size (should have 2 new URLs, one original already processed)
        assert queue.size == 2
        
        # Mark original URL as completed
        queue.complete(normalized_url)
        
        # Verify URL is now marked as visited
        assert queue.is_visited(normalized_url)
        
        # 4. Verify URL classification
        url_info = url_service.classify_url("http://example.com/products/item123")
        assert url_info["path_type"] == "product"
        assert not url_info["is_resource"]
        
        # All tests passed, showing core services work together correctly
    
    def test_comprehensive_service_features(self):
        """Test that all features of the core services are properly implemented."""
        # Get services
        url_service = self.registry.get_service("url_service")
        html_service = self.registry.get_service("html_service")
        
        # Test URL Service features
        
        # URL normalization with various inputs
        assert url_service.normalize_url("HTTP://Example.COM/path/../test/") == "http://example.com/test/"
        assert url_service.normalize_url("example.com") == "http://example.com/"  # Default scheme
        assert url_service.normalize_url("/page.html", "https://example.com") == "https://example.com/page.html"  # Base URL
        
        # Tracking parameter removal
        assert url_service.normalize_url("http://example.com/?utm_source=test&param=value") == "http://example.com/?param=value"
        
        # URL classification
        product_url_info = url_service.classify_url("https://example.com/products/item123")
        assert product_url_info["path_type"] == "product"
        
        category_url_info = url_service.classify_url("https://example.com/categories/electronics")
        assert category_url_info["path_type"] == "category"
        
        resource_url_info = url_service.classify_url("https://example.com/images/photo.jpg")
        assert resource_url_info["is_resource"] == True
        
        # Queue management
        queue = url_service.get_queue("custom_queue")
        queue.add("https://example.com/page1")
        queue.add("https://example.com/page2")
        assert queue.size == 2
        next_url = queue.get()
        assert next_url in ["https://example.com/page1", "https://example.com/page2"]
        queue.complete(next_url)
        assert queue.is_visited(next_url)
        
        # Test HTML Service features
        
        # HTML cleaning
        html = """
        <html>
            <head>
                <script>console.log('test');</script>
                <style>.hidden { display: none; }</style>
            </head>
            <body>
                <div hidden>Hidden content</div>
                <div style="display:none">Also hidden</div>
                <div id="main">
                    <h1>Test</h1>
                    <p><!-- Comment -->Visible content</p>
                </div>
            </body>
        </html>
        """
        
        cleaned = html_service.clean_html(html)
        assert "script" not in cleaned
        assert "style" not in cleaned
        assert "Hidden content" not in cleaned
        assert "Also hidden" not in cleaned
        assert "Comment" not in cleaned
        assert "Visible content" in cleaned
        
        # Selector generation
        soup = BeautifulSoup(cleaned, 'lxml')
        h1 = soup.find('h1')
        selector = html_service.generate_selector(h1)
        assert "h1" in selector
        
        p = soup.find('p')
        xpath_selector = html_service.generate_selector(p, method='xpath')
        assert "p" in xpath_selector
        
        # Main content extraction
        main_content = html_service.extract_main_content(cleaned)
        assert "Test" in main_content
        assert "Visible content" in main_content
        
        # Table extraction
        table_html = """
        <table>
            <thead>
                <tr><th>Name</th><th>Value</th></tr>
            </thead>
            <tbody>
                <tr><td>Item 1</td><td>100</td></tr>
                <tr><td>Item 2</td><td>200</td></tr>
            </tbody>
        </table>
        """
        
        tables = html_service.extract_tables(table_html)
        assert len(tables) == 1
        assert tables[0]["headers"] == ["Name", "Value"]
        assert len(tables[0]["rows"]) == 2
        assert "Item 1" in tables[0]["rows"][0]
        
        # Link extraction with base URL
        link_html = """
        <div>
            <a href="/page1">Page 1</a>
            <a href="page2">Page 2</a>
            <a href="https://external.com/page3">External Page</a>
        </div>
        """
        
        links = html_service.extract_links(link_html, base_url="https://example.com/dir/")
        assert len(links) == 3
        
        # Check links are properly resolved
        link_urls = [link["url"] for link in links]
        assert "https://example.com/page1" in link_urls
        assert "https://example.com/dir/page2" in link_urls
        assert "https://external.com/page3" in link_urls
        
        # Check internal/external classification
        assert links[0]["is_internal"] == True  # /page1 is internal to example.com
        assert links[1]["is_internal"] == True  # page2 is internal to example.com
        assert links[2]["is_internal"] == False  # external.com is external


if __name__ == "__main__":
    # Run tests directly if script is executed
    pytest.main(["-xvs", __file__])