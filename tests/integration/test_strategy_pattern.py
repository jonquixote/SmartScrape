import pytest
import logging
import time
from typing import Dict, Any, Optional, List, Set

from core.service_registry import ServiceRegistry
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_factory import StrategyFactory
from strategies.core.strategy_interface import BaseStrategy
from strategies.core.strategy_types import StrategyType, StrategyCapability, strategy_metadata
from strategies.core.composite_strategy import (
    SequentialStrategy, FallbackStrategy, PipelineStrategy
)
from controllers.adaptive_scraper import AdaptiveScraper

# For real-world scenario tests
from strategies.bfs_strategy import BFSStrategy
from strategies.dfs_strategy import DFSStrategy
from strategies.base_strategy_v2 import BaseStrategyV2
from strategies.api_strategy import APIStrategy
from strategies.multi_strategy import MultiStrategy

logger = logging.getLogger(__name__)

# Mock website data for testing
MOCK_WEBSITES = {
    "http://test.example.com/": "<html><body><h1>Test Website</h1><p>This is a test.</p><a href='/page1'>Page 1</a><a href='/page2'>Page 2</a></body></html>",
    "http://test.example.com/page1": "<html><body><h1>Page 1</h1><p>Content of page 1</p><a href='/'>Home</a></body></html>",
    "http://test.example.com/page2": "<html><body><h1>Page 2</h1><p>Content of page 2</p><a href='/'>Home</a><a href='/page3'>Page 3</a></body></html>",
    "http://test.example.com/page3": "<html><body><h1>Page 3</h1><form action='/submit'><input type='text' name='query'><button type='submit'>Search</button></form></body></html>",
    "http://api.example.com/api/products": '{"products": [{"id": 1, "name": "Product 1", "price": 10.99}, {"id": 2, "name": "Product 2", "price": 24.99}]}'
}

# Mock strategies for testing
@strategy_metadata(
    strategy_type=StrategyType.TRAVERSAL,
    capabilities={StrategyCapability.ROBOTS_TXT_ADHERENCE},
    description="Test traversal strategy for integration testing."
)
class TestTraversalStrategy(BaseStrategyV2):
    def __init__(self, context: StrategyContext):
        super().__init__(context)
        self.visited_urls = set()
        self.initialization_called = False
        self.shutdown_called = False
    
    @property
    def name(self) -> str:
        return "test_traversal"
    
    def initialize(self) -> None:
        """Initialize the strategy."""
        self.initialization_called = True
        logger.info(f"Initializing {self.name}")
    
    def shutdown(self) -> None:
        """Shutdown the strategy."""
        self.shutdown_called = True
        logger.info(f"Shutting down {self.name}")
    
    def execute(self, url: str, max_depth: int = 1, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute the strategy, visit and collect data from URLs."""
        result = self.crawl(url, max_depth=max_depth, **kwargs)
        return result
    
    def crawl(self, start_url: str, max_depth: int = 1, **kwargs) -> Optional[Dict[str, Any]]:
        """Crawl from the start URL up to max_depth."""
        # Start with the first URL
        to_visit = [(start_url, 0)]  # (url, depth)
        self.visited_urls = set()
        
        while to_visit:
            url, depth = to_visit.pop(0)
            
            if url in self.visited_urls or depth > max_depth:
                continue
                
            # Add to visited
            self.visited_urls.add(url)
            
            # Get mock content
            if url in MOCK_WEBSITES:
                content = MOCK_WEBSITES[url]
                # Store result
                self._results.append({
                    "url": url,
                    "depth": depth,
                    "title": self._extract_title(content),
                    "links": self._extract_links(content, url)
                })
                
                # Add links to queue if not at max depth
                if depth < max_depth:
                    for link in self._extract_links(content, url):
                        to_visit.append((link["url"], depth + 1))
        
        return {
            "urls_visited": len(self.visited_urls),
            "data_collected": len(self._results)
        }
    
    def _extract_title(self, html_content: str) -> str:
        """Extract title from HTML content."""
        import re
        title_match = re.search(r'<h1>(.*?)</h1>', html_content)
        return title_match.group(1) if title_match else "No title"
    
    def _extract_links(self, html_content: str, base_url: str) -> List[Dict[str, str]]:
        """Extract links from HTML content."""
        import re
        links = []
        # Extract href links using regex
        for href in re.finditer(r'<a href=[\'"]([^\'"]*)[\'"]', html_content):
            url = href.group(1)
            if url.startswith('/'):
                # Convert relative URL to absolute
                domain = base_url.split('//', 1)[0] + '//' + base_url.split('//', 1)[1].split('/', 1)[0]
                url = domain + url
            links.append({"url": url})
        return links


@strategy_metadata(
    strategy_type=StrategyType.EXTRACTION,
    capabilities={StrategyCapability.SCHEMA_EXTRACTION},
    description="Test extraction strategy for integration testing."
)
class TestExtractionStrategy(BaseStrategyV2):
    def __init__(self, context: StrategyContext):
        super().__init__(context)
        self.extraction_count = 0
    
    @property
    def name(self) -> str:
        return "test_extraction"
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Extract data from the given URL."""
        # Get mock content
        if url in MOCK_WEBSITES:
            content = MOCK_WEBSITES[url]
            result = self.extract(content, url, **kwargs)
            return result
        return None
    
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Extract data from the provided HTML content."""
        import re
        self.extraction_count += 1
        
        # Check if it's JSON
        if url.endswith('json') or url.startswith('http://api.'):
            try:
                import json
                data = json.loads(html_content)
                result = {
                    "url": url,
                    "type": "json",
                    "data": data
                }
                self._results.append(result)
                return result
            except json.JSONDecodeError:
                pass
        
        # Extract title
        title_match = re.search(r'<h1>(.*?)</h1>', html_content)
        title = title_match.group(1) if title_match else "No title"
        
        # Extract paragraphs
        paragraphs = re.findall(r'<p>(.*?)</p>', html_content)
        
        # Extract form fields
        form_fields = []
        if re.search(r'<form', html_content):
            input_fields = re.findall(r'<input[^>]*name=[\'"]([^\'"]*)[\'"]', html_content)
            form_fields.extend(input_fields)
        
        result = {
            "url": url,
            "type": "html",
            "title": title,
            "paragraphs": paragraphs,
            "form_fields": form_fields
        }
        
        self._results.append(result)
        return result


# Fixtures

@pytest.fixture
def service_registry():
    """Create and initialize a service registry."""
    registry = ServiceRegistry()
    # Initialize core services
    registry.initialize_all({})
    yield registry
    registry.shutdown_all()


@pytest.fixture
def strategy_context(service_registry):
    """Create a strategy context."""
    return StrategyContext({
        "logger": logger,
        "service_registry": service_registry
    })


@pytest.fixture
def strategy_factory(strategy_context):
    """Create a strategy factory."""
    factory = StrategyFactory(strategy_context)
    # Register test strategies
    factory.register_strategy(TestTraversalStrategy)
    factory.register_strategy(TestExtractionStrategy)
    # Register real strategies
    factory.register_strategy(BFSStrategy)
    factory.register_strategy(DFSStrategy)
    factory.register_strategy(APIStrategy)
    factory.register_strategy(MultiStrategy)
    return factory


@pytest.fixture
def adaptive_scraper(strategy_factory, strategy_context):
    """Create an adaptive scraper."""
    scraper = AdaptiveScraper()
    # Replace the scraper's factory with our test factory
    scraper.strategy_factory = strategy_factory
    scraper.context = strategy_context
    return scraper


# Mock requests for HTTP calls in integration tests
@pytest.fixture(autouse=True)
def mock_requests(monkeypatch):
    """Mock requests to return test data instead of making HTTP calls."""
    def mock_get(url, *args, **kwargs):
        class MockResponse:
            def __init__(self, text, status_code):
                self.text = text
                self.status_code = status_code
                self.content = text.encode('utf-8')
                
            def json(self):
                import json
                return json.loads(self.text)
        
        # Return mock website content if available, otherwise 404
        if url in MOCK_WEBSITES:
            return MockResponse(MOCK_WEBSITES[url], 200)
        return MockResponse("Not found", 404)
    
    # Patch requests.get
    import requests
    monkeypatch.setattr(requests, 'get', mock_get)
    
    # Also patch session.get
    class MockSession:
        def get(self, url, *args, **kwargs):
            return mock_get(url, *args, **kwargs)
    
    monkeypatch.setattr(requests, 'Session', MockSession)


# Tests for Full Strategy Lifecycle

class TestStrategyLifecycle:
    """Test the full strategy lifecycle."""
    
    def test_registration_and_instantiation(self, strategy_factory):
        """Test registering and instantiating strategies."""
        # Test getting registered strategies
        strategy_names = strategy_factory.get_all_strategy_names()
        assert "test_traversal" in strategy_names
        assert "test_extraction" in strategy_names
        
        # Test instantiating a strategy
        traversal = strategy_factory.get_strategy("test_traversal")
        assert isinstance(traversal, TestTraversalStrategy)
        assert traversal.name == "test_traversal"
        
        extraction = strategy_factory.get_strategy("test_extraction")
        assert isinstance(extraction, TestExtractionStrategy)
        assert extraction.name == "test_extraction"
    
    def test_initialization_and_shutdown(self, strategy_factory):
        """Test initializing and shutting down strategies."""
        # Get strategy
        traversal = strategy_factory.get_strategy("test_traversal")
        
        # Initialize
        traversal.initialize()
        assert traversal.initialization_called is True
        
        # Shutdown
        traversal.shutdown()
        assert traversal.shutdown_called is True
    
    def test_execution_and_result_collection(self, strategy_factory):
        """Test strategy execution and result collection."""
        # Get strategy
        traversal = strategy_factory.get_strategy("test_traversal")
        
        # Execute
        result = traversal.execute("http://test.example.com/", max_depth=1)
        
        # Check result
        assert result is not None
        assert result["urls_visited"] > 0
        assert result["data_collected"] > 0
        
        # Check results collection
        results = traversal.get_results()
        assert len(results) > 0
        assert results[0]["url"] == "http://test.example.com/"
        assert results[0]["title"] == "Test Website"
        assert len(results[0]["links"]) == 2


# Tests for Strategy Composition

class TestStrategyComposition:
    """Test strategy composition patterns."""
    
    def test_sequential_execution(self, strategy_factory):
        """Test sequential execution of strategies."""
        # Create strategies
        traversal = strategy_factory.get_strategy("test_traversal")
        extraction = strategy_factory.get_strategy("test_extraction")
        
        # Create sequential strategy
        sequential = SequentialStrategy(strategy_factory.context)
        sequential.add_strategy(traversal)
        sequential.add_strategy(extraction)
        
        # Execute
        result = sequential.execute("http://test.example.com/")
        
        # Check result
        assert result is not None
        assert "test_traversal" in result["child_results"]
        assert "test_extraction" in result["child_results"]
        
        # Check that both strategies were executed
        assert len(traversal.get_results()) > 0
        assert len(extraction.get_results()) > 0
    
    def test_fallback_mechanism(self, strategy_factory):
        """Test fallback mechanism between strategies."""
        # Create strategies - first will fail, second will succeed
        @strategy_metadata(
            strategy_type=StrategyType.EXTRACTION,
            capabilities={StrategyCapability.SCHEMA_EXTRACTION},
            description="Failing strategy for testing fallbacks."
        )
        class FailingStrategy(BaseStrategy):
            def __init__(self, context):
                super().__init__(context)
                self._results = []
            
            @property
            def name(self):
                return "failing_strategy"
            
            def execute(self, url, **kwargs):
                # Always fail
                return None
            
            def crawl(self, start_url, **kwargs):
                return None
            
            def extract(self, html_content, url, **kwargs):
                return None
            
            def get_results(self):
                return self._results
        
        # Register failing strategy
        strategy_factory.register_strategy(FailingStrategy)
        
        # Get strategies
        failing = strategy_factory.get_strategy("failing_strategy")
        extraction = strategy_factory.get_strategy("test_extraction")
        
        # Create fallback strategy
        fallback = FallbackStrategy(strategy_factory.context)
        fallback.add_strategy(failing)
        fallback.add_strategy(extraction)
        
        # Execute
        result = fallback.execute("http://test.example.com/")
        
        # Check result - should be from extraction since failing failed
        assert result is not None
        assert result["url"] == "http://test.example.com/"
        assert result["title"] == "Test Website"
        
        # Check that failing strategy was tried but produced no results
        assert len(failing.get_results()) == 0
        # Check that extraction strategy succeeded
        assert len(extraction.get_results()) > 0
    
    def test_pipeline_processing(self, strategy_factory):
        """Test pipeline processing where output of one strategy feeds into the next."""
        # Create a simple pipeline with two stages
        
        # First stage: gets URLs
        @strategy_metadata(
            strategy_type=StrategyType.TRAVERSAL,
            capabilities={StrategyCapability.ROBOTS_TXT_ADHERENCE},
            description="URL discovery strategy for pipeline testing."
        )
        class URLDiscoveryStrategy(BaseStrategy):
            def __init__(self, context):
                super().__init__(context)
                self._results = []
            
            @property
            def name(self):
                return "url_discovery"
            
            def execute(self, url, **kwargs):
                # Return a list of URLs to visit
                urls = [
                    "http://test.example.com/",
                    "http://test.example.com/page1",
                    "http://test.example.com/page2"
                ]
                result = {
                    "discovered_urls": urls
                }
                self._results.append(result)
                return result
            
            def crawl(self, start_url, **kwargs):
                return self.execute(start_url, **kwargs)
            
            def extract(self, html_content, url, **kwargs):
                return None
            
            def get_results(self):
                return self._results
        
        # Second stage: processes each URL
        @strategy_metadata(
            strategy_type=StrategyType.EXTRACTION,
            capabilities={StrategyCapability.SCHEMA_EXTRACTION},
            description="URL processing strategy for pipeline testing."
        )
        class URLProcessingStrategy(BaseStrategy):
            def __init__(self, context):
                super().__init__(context)
                self._results = []
            
            @property
            def name(self):
                return "url_processing"
            
            def execute(self, url, discovered_urls=None, **kwargs):
                """Process the discovered URLs from the previous stage."""
                if not discovered_urls:
                    return None
                
                processed_data = []
                for url in discovered_urls:
                    if url in MOCK_WEBSITES:
                        processed_data.append({
                            "url": url,
                            "content_length": len(MOCK_WEBSITES[url])
                        })
                
                result = {
                    "processed_urls": len(processed_data),
                    "data": processed_data
                }
                self._results.append(result)
                return result
            
            def crawl(self, start_url, **kwargs):
                return None
            
            def extract(self, html_content, url, **kwargs):
                return None
            
            def get_results(self):
                return self._results
        
        # Register strategies
        strategy_factory.register_strategy(URLDiscoveryStrategy)
        strategy_factory.register_strategy(URLProcessingStrategy)
        
        # Get strategies
        discovery = strategy_factory.get_strategy("url_discovery")
        processing = strategy_factory.get_strategy("url_processing")
        
        # Create pipeline strategy
        pipeline = PipelineStrategy(strategy_factory.context)
        pipeline.add_strategy(discovery)
        pipeline.add_strategy(processing)
        
        # Execute pipeline
        result = pipeline.execute("http://test.example.com/")
        
        # Check result
        assert result is not None
        assert "processed_urls" in result
        assert result["processed_urls"] == 3
        assert len(result["data"]) == 3
        
        # Check that both strategies executed
        assert len(discovery.get_results()) == 1
        assert len(processing.get_results()) == 1
    
    def test_parallel_execution(self, strategy_factory):
        """Test parallel execution of strategies."""
        # This would ideally use a ParallelStrategy, but we can simulate by creating
        # a custom implementation for testing purposes
        
        # Get strategies
        traversal = strategy_factory.get_strategy("test_traversal")
        extraction = strategy_factory.get_strategy("test_extraction")
        
        # Execute them in parallel using threads
        import threading
        
        results = {}
        
        def run_strategy(strategy, url, result_key):
            results[result_key] = strategy.execute(url)
        
        # Create and start threads
        t1 = threading.Thread(target=run_strategy, args=(traversal, "http://test.example.com/", "traversal"))
        t2 = threading.Thread(target=run_strategy, args=(extraction, "http://test.example.com/", "extraction"))
        
        t1.start()
        t2.start()
        
        # Wait for completion
        t1.join()
        t2.join()
        
        # Check results
        assert "traversal" in results
        assert "extraction" in results
        assert results["traversal"]["urls_visited"] > 0
        assert results["extraction"]["title"] == "Test Website"
        
        # Check that both strategies executed
        assert len(traversal.get_results()) > 0
        assert len(extraction.get_results()) > 0


# Tests for Real-World Scenarios

class TestRealWorldScenarios:
    """Test strategy pattern with real-world scenarios."""
    
    def test_multi_page_crawling(self, strategy_factory):
        """Test crawling multiple pages."""
        # Get traversal strategy
        traversal = strategy_factory.get_strategy("test_traversal")
        
        # Execute with depth=2 to crawl multiple pages
        result = traversal.execute("http://test.example.com/", max_depth=2)
        
        # Check result
        assert result["urls_visited"] >= 3  # Should visit at least 3 pages
        
        # Check that it collected data from multiple pages
        results = traversal.get_results()
        urls_visited = {r["url"] for r in results}
        assert "http://test.example.com/" in urls_visited
        assert "http://test.example.com/page1" in urls_visited
        assert "http://test.example.com/page2" in urls_visited
    
    def test_form_interaction(self, strategy_factory):
        """Test form interaction."""
        # Get extraction strategy
        extraction = strategy_factory.get_strategy("test_extraction")
        
        # Execute on a page with a form
        result = extraction.execute("http://test.example.com/page3")
        
        # Check that it extracted form fields
        assert result is not None
        assert "form_fields" in result
        assert len(result["form_fields"]) > 0
        assert "query" in result["form_fields"]
    
    def test_data_extraction(self, strategy_factory):
        """Test extracting data from different types of content."""
        # Get extraction strategy
        extraction = strategy_factory.get_strategy("test_extraction")
        
        # Test extracting from HTML
        html_result = extraction.execute("http://test.example.com/")
        assert html_result["type"] == "html"
        assert html_result["title"] == "Test Website"
        assert len(html_result["paragraphs"]) > 0
        
        # Test extracting from API/JSON
        json_result = extraction.execute("http://api.example.com/api/products")
        assert json_result["type"] == "json"
        assert "products" in json_result["data"]
        assert len(json_result["data"]["products"]) == 2
    
    def test_error_handling_and_recovery(self, strategy_factory):
        """Test error handling and recovery during strategy execution."""
        # Create a strategy that throws errors but can recover
        @strategy_metadata(
            strategy_type=StrategyType.TRAVERSAL,
            capabilities={StrategyCapability.ERROR_HANDLING},
            description="Error-prone strategy for testing recovery."
        )
        class ErrorProneStrategy(BaseStrategy):
            def __init__(self, context):
                super().__init__(context)
                self._results = []
                self.retry_count = 0
                self.errors = []
            
            @property
            def name(self):
                return "error_prone"
            
            def execute(self, url, **kwargs):
                try:
                    if self.retry_count < 2:
                        self.retry_count += 1
                        self.errors.append(f"Error on try {self.retry_count}")
                        raise ValueError(f"Simulated error on try {self.retry_count}")
                    
                    # Success after retry
                    result = {
                        "url": url,
                        "success": True,
                        "retry_count": self.retry_count
                    }
                    self._results.append(result)
                    return result
                except Exception as e:
                    # Log error but continue
                    self.errors.append(str(e))
                    if self.retry_count < 3:
                        # Retry
                        return self.execute(url, **kwargs)
                    return None
            
            def crawl(self, start_url, **kwargs):
                return self.execute(start_url, **kwargs)
            
            def extract(self, html_content, url, **kwargs):
                return None
            
            def get_results(self):
                return self._results
        
        # Register strategy
        strategy_factory.register_strategy(ErrorProneStrategy)
        
        # Get strategy
        error_prone = strategy_factory.get_strategy("error_prone")
        
        # Execute
        result = error_prone.execute("http://test.example.com/")
        
        # Check that it recovered and succeeded
        assert result is not None
        assert result["success"] is True
        assert result["retry_count"] == 2
        assert len(error_prone.errors) == 2


# Tests for Adaptive Scraper

class TestAdaptiveScraper:
    """Test the adaptive scraper with strategy pattern."""
    
    def test_different_site_types(self, adaptive_scraper):
        """Test scraping different site types."""
        # Test HTML site
        html_result = adaptive_scraper.scrape("http://test.example.com/")
        assert html_result is not None
        
        # Test API site
        api_result = adaptive_scraper.scrape("http://api.example.com/api/products")
        assert api_result is not None
        
        # Results should be different based on site type
        assert html_result != api_result
    
    def test_strategy_selection_by_capability(self, adaptive_scraper):
        """Test selecting strategies by capability."""
        # Request a strategy with SCHEMA_EXTRACTION capability
        result = adaptive_scraper.scrape(
            "http://api.example.com/api/products",
            required_capabilities={StrategyCapability.SCHEMA_EXTRACTION}
        )
        
        # Should select the extraction strategy
        assert result is not None
        assert "type" in result
        assert result["type"] == "json"
    
    def test_error_scenarios(self, adaptive_scraper, strategy_factory):
        """Test handling of error scenarios in adaptive scraper."""
        # Register error-prone strategy
        @strategy_metadata(
            strategy_type=StrategyType.TRAVERSAL,
            capabilities={StrategyCapability.ERROR_HANDLING},
            description="Strategy that always fails."
        )
        class AlwaysFailStrategy(BaseStrategy):
            def __init__(self, context):
                super().__init__(context)
                self._results = []
            
            @property
            def name(self):
                return "always_fail"
            
            def execute(self, url, **kwargs):
                # Always fail
                return None
            
            def crawl(self, start_url, **kwargs):
                return None
            
            def extract(self, html_content, url, **kwargs):
                return None
            
            def get_results(self):
                return self._results
        
        strategy_factory.register_strategy(AlwaysFailStrategy)
        
        # Try to scrape with the failing strategy
        result = adaptive_scraper.scrape("http://test.example.com/", strategy_name="always_fail")
        
        # Should return None since the strategy failed
        assert result is None
    
    def test_performance_metrics(self, adaptive_scraper):
        """Test collecting performance metrics during scraping."""
        # Create a strategy with performance tracking
        @strategy_metadata(
            strategy_type=StrategyType.TRAVERSAL,
            capabilities={StrategyCapability.ROBOTS_TXT_ADHERENCE},
            description="Strategy that tracks performance metrics."
        )
        class PerformanceTrackingStrategy(BaseStrategy):
            def __init__(self, context):
                super().__init__(context)
                self._results = []
                self.metrics = []
            
            @property
            def name(self):
                return "performance_tracking"
            
            def execute(self, url, **kwargs):
                # Track start time
                start_time = time.time()
                
                # Simulate work
                time.sleep(0.01)  # Very small sleep for test
                
                # Create result
                duration = time.time() - start_time
                result = {
                    "url": url,
                    "duration_seconds": duration
                }
                self._results.append(result)
                
                # Record metric
                self.metrics.append({
                    "timestamp": time.time(),
                    "operation": "execute",
                    "duration_seconds": duration
                })
                
                return result
            
            def crawl(self, start_url, **kwargs):
                return self.execute(start_url, **kwargs)
            
            def extract(self, html_content, url, **kwargs):
                return None
            
            def get_results(self):
                return self._results
        
        strategy_factory.register_strategy(PerformanceTrackingStrategy)
        
        # Replace factory in adaptive scraper
        adaptive_scraper.strategy_factory = strategy_factory
        
        # Execute through the adaptive scraper
        result = adaptive_scraper.scrape("http://test.example.com/", strategy_name="performance_tracking")
        
        # Check result
        assert result is not None
        assert "duration_seconds" in result
        assert result["duration_seconds"] > 0
        
        # Get strategy to check metrics
        strategy = strategy_factory.get_strategy("performance_tracking")
        assert len(strategy.metrics) > 0
        assert strategy.metrics[0]["operation"] == "execute"
        assert strategy.metrics[0]["duration_seconds"] > 0


if __name__ == "__main__":
    pytest.main(['-v', 'test_strategy_pattern.py'])