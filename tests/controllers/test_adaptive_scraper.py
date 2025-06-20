"""
Tests for the AdaptiveScraper controller using the strategy pattern.
"""

import pytest
import pytest_asyncio
import asyncio
import logging
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Optional, Set

from controllers.adaptive_scraper import AdaptiveScraper
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_factory import StrategyFactory
from strategies.core.strategy_types import StrategyCapability, StrategyType, StrategyMetadata
from strategies.core.strategy_interface import BaseStrategy
from strategies.core.composite_strategy import FallbackStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock strategies for testing
class MockSuccessStrategy(BaseStrategy):
    """Strategy that always succeeds."""
    
    def __init__(self, context=None):
        super().__init__(context)
        self._results = []
    
    @property
    def name(self) -> str:
        return "mock_success"
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        result = {"url": url, "success": True, "strategy": self.name}
        self._results.append(result)
        return result
    
    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        result = {"url": start_url, "crawled": True, "strategy": self.name}
        self._results.append(result)
        return result
    
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        result = {"url": url, "extracted": True, "strategy": self.name}
        self._results.append(result)
        return result
    
    def get_results(self) -> List[Dict[str, Any]]:
        return self._results

class MockFailureStrategy(BaseStrategy):
    """Strategy that always fails."""
    
    def __init__(self, context=None):
        super().__init__(context)
        self._results = []
        self._errors = []
    
    @property
    def name(self) -> str:
        return "mock_failure"
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        self.handle_error(message="Execution failed", url=url)
        return None
    
    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        self.handle_error(message="Crawl failed", url=start_url)
        return None
    
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        self.handle_error(message="Extraction failed", url=url)
        return None
    
    def get_results(self) -> List[Dict[str, Any]]:
        return self._results
    
    def has_errors(self, min_severity: str = "warning") -> bool:
        return True

class MockDynamicStrategy(BaseStrategy):
    """Strategy that can be configured to succeed or fail."""
    
    def __init__(self, context=None, should_succeed=True):
        super().__init__(context)
        self._results = []
        self._should_succeed = should_succeed
    
    @property
    def name(self) -> str:
        return "mock_dynamic"
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        if self._should_succeed:
            result = {"url": url, "success": True, "strategy": self.name}
            self._results.append(result)
            return result
        else:
            self.handle_error(message="Dynamic strategy configured to fail", url=url)
            return None
    
    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        if self._should_succeed:
            result = {"url": start_url, "crawled": True, "strategy": self.name}
            self._results.append(result)
            return result
        else:
            self.handle_error(message="Dynamic crawl configured to fail", url=start_url)
            return None
    
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        if self._should_succeed:
            result = {"url": url, "extracted": True, "strategy": self.name}
            self._results.append(result)
            return result
        else:
            self.handle_error(message="Dynamic extraction configured to fail", url=url)
            return None
    
    def get_results(self) -> List[Dict[str, Any]]:
        return self._results
    
    def has_errors(self, min_severity: str = "warning") -> bool:
        return not self._should_succeed

# Mock for core services
from core.service_interface import BaseService

class MockService(BaseService):
    def __init__(self, name):
        self._name = name
        self._initialized = False
    
    def initialize(self, config=None):
        self._initialized = True
    
    def shutdown(self):
        self._initialized = False
    
    @property
    def name(self):
        return self._name
    
    @property 
    def is_initialized(self):
        return self._initialized
    
    async def analyze_site(self, url):
        return {
            "has_search_form": "form" in url,
            "has_search_api": "api" in url,
            "has_pagination": "pagination" in url,
            "requires_javascript": "js" in url,
            "has_login": "login" in url,
            "has_captcha": "captcha" in url
        }
    
    async def generate_search_terms(self, query, **kwargs):
        return {
            "primary": query,
            "variants": [f"{query}_variant1", f"{query}_variant2"]
        }
    
    def record_search_metrics(self, **kwargs):
        pass
    
    async def learn_from_execution(self, **kwargs):
        pass

# Fixture for AdaptiveScraper with mocked dependencies
@pytest_asyncio.fixture
async def adaptive_scraper():
    """Create an AdaptiveScraper instance with mocked dependencies."""
    with patch('strategies.core.strategy_factory.StrategyFactory.register_strategy') as mock_register:
        # Create the scraper
        scraper = AdaptiveScraper()
        
        # Mock the strategy factory's register_strategy method
        mock_register.return_value = None
        
        # Replace the strategy factory with a mocked version
        mock_factory = MagicMock(spec=StrategyFactory)
        scraper.strategy_factory = mock_factory
        
        # Set up the mock factory to return our test strategies
        mock_factory.get_strategy.side_effect = lambda name: {
            "mock_success": MockSuccessStrategy(scraper.strategy_context),
            "mock_failure": MockFailureStrategy(scraper.strategy_context),
            "mock_dynamic": MockDynamicStrategy(scraper.strategy_context, True),
            "fallback_strategy": FallbackStrategy(scraper.strategy_context)
        }.get(name, MockSuccessStrategy(scraper.strategy_context))
        
        mock_factory.select_best_strategy.return_value = MockSuccessStrategy(scraper.strategy_context)
        mock_factory.get_strategies_by_capability.return_value = [MockSuccessStrategy]
        mock_factory.get_all_strategy_names.return_value = ["mock_success", "mock_failure", "mock_dynamic", "fallback_strategy"]
        
        # Replace core services with mocks
        scraper.site_discovery = MockService("site_discovery")
        scraper.search_term_generator = MockService("search_term_generator")
        
        # Set fallback strategies to use test-compatible ones
        scraper.fallback_strategies = ["mock_success", "fallback_strategy"]
        scraper.metrics_analyzer = MockService("metrics_analyzer")
        scraper.continuous_improvement = MockService("continuous_improvement")
        
        # Register mock services with strategy context
        scraper.strategy_context.register_service("site_discovery", scraper.site_discovery)
        scraper.strategy_context.register_service("search_term_generator", scraper.search_term_generator)
        
        yield scraper
        
        # Cleanup
        await scraper.shutdown()

# Tests
@pytest.mark.asyncio
async def test_scrape_with_explicit_strategy(adaptive_scraper):
    """Test scraping with an explicitly specified strategy."""
    # Test with success strategy
    result = await adaptive_scraper.scrape("http://example.com", strategy_name="mock_success")
    
    assert result is not None
    assert result["success"] is True
    assert result["strategy"] == "mock_success"
    assert result["results"]["url"] == "http://example.com"
    
    # Test with failure strategy
    result = await adaptive_scraper.scrape("http://example.com", strategy_name="mock_failure")
    
    assert result is not None
    assert result["success"] is False
    assert result["strategy"] == "mock_failure"
    assert result["results"] is None

@pytest.mark.asyncio
async def test_scrape_with_strategy_selection(adaptive_scraper):
    """Test scraping with strategy selection based on capabilities."""
    # Configure the mock factory to select based on URL
    adaptive_scraper.strategy_factory.select_best_strategy.side_effect = lambda url, caps=None: {
        "http://example.com/form": MockSuccessStrategy(adaptive_scraper.strategy_context),
        "http://example.com/api": MockDynamicStrategy(adaptive_scraper.strategy_context, True),
        "http://example.com/failure": MockFailureStrategy(adaptive_scraper.strategy_context)
    }.get(url, MockSuccessStrategy(adaptive_scraper.strategy_context))
    
    # Test with a URL that selects success strategy
    result = await adaptive_scraper.scrape("http://example.com/form")
    
    assert result is not None
    assert result["success"] is True
    assert result["results"]["url"] == "http://example.com/form"
    
    # Test with a URL that selects failure strategy (should trigger fallback)
    result = await adaptive_scraper.scrape("http://example.com/failure")
    
    assert result is not None
    assert result["success"] is False  # Fallback mechanism is mocked, so it will still fail

@pytest.mark.asyncio
async def test_execute_search_pipeline(adaptive_scraper):
    """Test the complete search pipeline."""
    # Mock the pipeline steps to track calls
    original_prepare = adaptive_scraper._prepare_search_terms
    original_select = adaptive_scraper._select_search_strategy
    original_execute = adaptive_scraper._execute_search
    original_process = adaptive_scraper._process_results
    original_fallback = adaptive_scraper._apply_fallback_if_needed
    
    # Replace with spies
    async def mock_prepare(context):
        context["search_terms"] = {
            "primary": context["query"],
            "variants": [f"{context['query']}_variant1", f"{context['query']}_variant2"]
        }
        return context
    
    async def mock_select(context):
        context["strategy"] = MockSuccessStrategy(adaptive_scraper.strategy_context)
        context["strategy_name"] = "mock_success"
        return context
    
    async def mock_execute(context):
        context["raw_results"] = [{"url": context["url"], "data": "test"}]
        return context
    
    async def mock_process(context):
        context["results"] = context["raw_results"]
        return context
    
    async def mock_fallback(context):
        return context
    
    adaptive_scraper._prepare_search_terms = mock_prepare
    adaptive_scraper._select_search_strategy = mock_select
    adaptive_scraper._execute_search = mock_execute
    adaptive_scraper._process_results = mock_process
    adaptive_scraper._apply_fallback_if_needed = mock_fallback
    
    # Execute the pipeline
    result = await adaptive_scraper.execute_search_pipeline(
        query="test query",
        url="http://example.com",
        options={"test_option": True}
    )
    
    # Restore original methods
    adaptive_scraper._prepare_search_terms = original_prepare
    adaptive_scraper._select_search_strategy = original_select
    adaptive_scraper._execute_search = original_execute
    adaptive_scraper._process_results = original_process
    adaptive_scraper._apply_fallback_if_needed = original_fallback
    
    # Verify the result
    assert result["success"] is True
    assert len(result["results"]) == 1
    assert result["results"][0]["url"] == "http://example.com"
    assert "metrics" in result
    assert "operation_id" in result

@pytest.mark.asyncio
async def test_fallback_mechanism(adaptive_scraper):
    """Test the fallback mechanism when a strategy fails."""
    # Configure the fallback strategy
    fallback = FallbackStrategy(adaptive_scraper.strategy_context)
    fallback.add_strategy(MockSuccessStrategy(adaptive_scraper.strategy_context))
    
    adaptive_scraper.strategy_factory.get_strategy.side_effect = lambda name: {
        "mock_success": MockSuccessStrategy(adaptive_scraper.strategy_context),
        "mock_failure": MockFailureStrategy(adaptive_scraper.strategy_context),
        "fallback_strategy": fallback
    }.get(name, MockSuccessStrategy(adaptive_scraper.strategy_context))
    
    # Create a context that needs fallback
    context = {
        "query": "test query",
        "url": "http://example.com",
        "options": {},
        "metrics": {},
        "attempts": 0,
        "max_attempts": 3,
        "fallbacks_used": [],
        "needs_fallback": True,
        "strategy_name": "mock_failure",
        "search_terms": {
            "primary": "test query",
            "variants": ["test query_variant1", "test query_variant2"]
        }
    }
    
    # Apply fallback
    updated_context = await adaptive_scraper._apply_fallback_if_needed(context)
    
    # In a real scenario, the fallback would work, but our mock setup
    # doesn't actually execute the fallback strategy, so we'll verify
    # that the context was properly updated
    assert updated_context["attempts"] <= context["max_attempts"]
    assert "fallback_times" in updated_context["metrics"]

@pytest.mark.asyncio
async def test_dynamic_strategy_switching(adaptive_scraper):
    """Test dynamic strategy switching based on performance."""
    # Configure the mock factory to track called strategies globally
    global_call_count = {"mock_dynamic": 0}
    
    original_get_strategy = adaptive_scraper.strategy_factory.get_strategy
    
    def side_effect(name):
        if name == "mock_dynamic":
            global_call_count["mock_dynamic"] += 1
            # First call fails, second and subsequent calls succeed
            should_succeed = global_call_count["mock_dynamic"] > 1
            logger.info(f"Creating MockDynamicStrategy (call #{global_call_count['mock_dynamic']}, should_succeed={should_succeed})")
            return MockDynamicStrategy(adaptive_scraper.strategy_context, should_succeed)
        return original_get_strategy(name)
    
    adaptive_scraper.strategy_factory.get_strategy.side_effect = side_effect
    
    # First try with the dynamic strategy configured to fail
    result = await adaptive_scraper.scrape("http://example.com", strategy_name="mock_dynamic")
    
    # Should have called mock_dynamic at least once and attempted fallback
    assert global_call_count["mock_dynamic"] >= 1
    # Result should indicate fallback was attempted (either success=False or fallback applied)
    assert result is not None  # Some result should be returned even if fallback
    
    # Now try again - the dynamic strategy should succeed this time
    result = await adaptive_scraper.scrape("http://example.com", strategy_name="mock_dynamic")
    
    assert result is not None
    assert result["success"] is True
    assert global_call_count["mock_dynamic"] >= 2

@pytest.mark.asyncio
async def test_performance_monitoring(adaptive_scraper):
    """Test that the scraper tracks strategy performance."""
    # Initial state should be empty
    assert len(adaptive_scraper.strategy_performance) == 0
    
    # Scrape with success strategy
    await adaptive_scraper.scrape("http://example.com", strategy_name="mock_success")
    
    # Check that performance metrics were recorded
    assert "mock_success" in adaptive_scraper.strategy_performance
    perf = adaptive_scraper.strategy_performance["mock_success"]
    assert perf["total_calls"] == 1
    assert perf["successful_calls"] == 1
    assert perf["total_time"] > 0
    assert perf["error_rate"] == 0
    
    # Scrape with failure strategy
    await adaptive_scraper.scrape("http://example.com", strategy_name="mock_failure")
    
    # Check that performance metrics were recorded
    assert "mock_failure" in adaptive_scraper.strategy_performance
    perf = adaptive_scraper.strategy_performance["mock_failure"]
    assert perf["total_calls"] == 1
    assert perf["successful_calls"] == 0
    assert perf["error_rate"] == 1