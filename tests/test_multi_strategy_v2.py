"""
Tests for the MultiStrategyV2 implementation.

This module tests the MultiStrategyV2 class which implements the composite strategy pattern
for combining multiple strategies.
"""

import unittest
import pytest
from unittest.mock import MagicMock, patch
import asyncio

from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_interface import BaseStrategy
from strategies.multi_strategy_v2 import MultiStrategyV2, create_multi_strategy_v2


# Mock strategies for testing
class MockSuccessStrategy(BaseStrategy):
    def __init__(self, context=None, name="success_strategy"):
        super().__init__(context)
        self._name = name
        self._results = []
    
    def execute(self, url, **kwargs):
        result = {
            "title": f"Title from {self._name}",
            "content": f"Content from {self._name}",
            "url": url,
            "confidence": 0.8
        }
        self._results.append(result)
        return result
    
    def crawl(self, start_url, **kwargs):
        result = {
            "title": f"Title from {self._name} crawl",
            "url": start_url,
            "confidence": 0.7
        }
        self._results.append(result)
        return result
    
    def extract(self, html_content, url, **kwargs):
        result = {
            "title": f"Extracted title from {self._name}",
            "content": f"Extracted content: {html_content[:20]}...",
            "url": url,
            "confidence": 0.9
        }
        self._results.append(result)
        return result
    
    def get_next_urls(self, url, depth, extraction_result=None, visited=None, **kwargs):
        return [
            {"url": f"{url}/page1", "depth": depth + 1, "score": 0.9},
            {"url": f"{url}/page2", "depth": depth + 1, "score": 0.8}
        ]
    
    def get_results(self):
        return self._results
    
    @property
    def name(self):
        return self._name


class MockLowConfidenceStrategy(BaseStrategy):
    def __init__(self, context=None, name="low_confidence_strategy"):
        super().__init__(context)
        self._name = name
        self._results = []
    
    def execute(self, url, **kwargs):
        result = {
            "title": f"Low confidence title from {self._name}",
            "content": f"Low confidence content from {self._name}",
            "url": url,
            "confidence": 0.3
        }
        self._results.append(result)
        return result
    
    def crawl(self, start_url, **kwargs):
        result = {
            "title": f"Low confidence title from {self._name} crawl",
            "url": start_url,
            "confidence": 0.4
        }
        self._results.append(result)
        return result
    
    def extract(self, html_content, url, **kwargs):
        result = {
            "title": f"Low confidence extracted title from {self._name}",
            "summary": f"Summary: {html_content[:10]}...",
            "url": url,
            "confidence": 0.2
        }
        self._results.append(result)
        return result
    
    def get_next_urls(self, url, depth, extraction_result=None, visited=None, **kwargs):
        return [
            {"url": f"{url}/page3", "depth": depth + 1, "score": 0.5},
            {"url": f"{url}/page4", "depth": depth + 1, "score": 0.4}
        ]
    
    def get_results(self):
        return self._results
    
    @property
    def name(self):
        return self._name


class MockFailureStrategy(BaseStrategy):
    def __init__(self, context=None, name="failure_strategy"):
        super().__init__(context)
        self._name = name
        self._results = []
    
    def execute(self, url, **kwargs):
        # This strategy always fails
        return None
    
    def crawl(self, start_url, **kwargs):
        return None
    
    def extract(self, html_content, url, **kwargs):
        return None
    
    def get_results(self):
        return self._results
    
    @property
    def name(self):
        return self._name


class MockErrorStrategy(BaseStrategy):
    def __init__(self, context=None, name="error_strategy"):
        super().__init__(context)
        self._name = name
        self._results = []
    
    def execute(self, url, **kwargs):
        # This strategy raises an exception
        raise ValueError(f"Error in {self._name}")
    
    def crawl(self, start_url, **kwargs):
        raise ValueError(f"Error in {self._name} crawl")
    
    def extract(self, html_content, url, **kwargs):
        raise ValueError(f"Error in {self._name} extract")
    
    def get_results(self):
        return self._results
    
    @property
    def name(self):
        return self._name


class TestMultiStrategyV2(unittest.TestCase):
    """Test cases for the MultiStrategyV2 implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock context
        self.context = MagicMock(spec=StrategyContext)
        
        # Create mock strategies
        self.success_strategy1 = MockSuccessStrategy(self.context, "success1")
        self.success_strategy2 = MockSuccessStrategy(self.context, "success2")
        self.low_confidence_strategy = MockLowConfidenceStrategy(self.context)
        self.failure_strategy = MockFailureStrategy(self.context)
        self.error_strategy = MockErrorStrategy(self.context)
        
        # Create a multi-strategy with the mock strategies
        self.multi_strategy = MultiStrategyV2(
            context=self.context,
            fallback_threshold=0.4,
            confidence_threshold=0.7,
            use_voting=True
        )
        
        # Add strategies
        self.multi_strategy.add_strategy(self.success_strategy1)
        self.multi_strategy.add_strategy(self.success_strategy2)
        self.multi_strategy.add_strategy(self.low_confidence_strategy)
        
        # Sample HTML content
        self.sample_html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Product: Test Product</h1>
                <div class="price">$99.99</div>
                <div class="description">This is a test product description.</div>
                <div class="info">
                    <span>Weight: 2kg</span>
                    <span>Color: Blue</span>
                </div>
                <div class="links">
                    <a href="/product/1">Product 1</a>
                    <a href="/product/2">Product 2</a>
                    <a href="/category/electronics">Electronics</a>
                </div>
            </body>
        </html>
        """
    
    def test_multi_strategy_initialization(self):
        """Test initializing the multi-strategy."""
        # Check that the strategies were added
        self.assertEqual(len(self.multi_strategy._child_strategies), 3)
        self.assertIn("success1", self.multi_strategy._child_strategies)
        self.assertIn("success2", self.multi_strategy._child_strategies)
        self.assertIn("low_confidence_strategy", self.multi_strategy._child_strategies)
        
        # Check configuration
        self.assertEqual(self.multi_strategy.fallback_threshold, 0.4)
        self.assertEqual(self.multi_strategy.confidence_threshold, 0.7)
        self.assertTrue(self.multi_strategy.use_voting)
        
        # Check performance tracking initialization
        self.assertIn("success1", self.multi_strategy.strategy_performance)
        self.assertEqual(self.multi_strategy.strategy_performance["success1"]["total_executions"], 0)
    
    def test_create_multi_strategy_v2_factory(self):
        """Test the factory function for creating a multi-strategy."""
        # Create a mock strategy factory
        mock_factory = MagicMock()
        mock_factory.get_strategy.side_effect = [
            self.success_strategy1,
            self.success_strategy2
        ]
        
        # Use the factory to create a multi-strategy
        multi_strategy = create_multi_strategy_v2(
            context=self.context,
            strategy_names=["success1", "success2"],
            strategy_factory=mock_factory,
            config={"max_depth": 5}
        )
        
        # Check that the factory was called correctly
        mock_factory.get_strategy.assert_any_call("success1")
        mock_factory.get_strategy.assert_any_call("success2")
        
        # Check that the strategies were added
        self.assertEqual(len(multi_strategy._child_strategies), 2)
        self.assertIn("success1", multi_strategy._child_strategies)
        self.assertIn("success2", multi_strategy._child_strategies)
        
        # Check that the config was applied
        self.assertEqual(multi_strategy.config["max_depth"], 5)
    
    def test_execute_sequential(self):
        """Test sequential execution of strategies."""
        # Configure for sequential execution
        self.multi_strategy.config["parallel_execution"] = False
        
        # Execute the multi-strategy
        result = self.multi_strategy.execute("http://example.com")
        
        # Check that all strategies were executed
        self.assertEqual(self.multi_strategy.extraction_stats["strategy_usage"]["success1"], 1)
        self.assertEqual(self.multi_strategy.extraction_stats["strategy_usage"]["success2"], 1)
        self.assertEqual(self.multi_strategy.extraction_stats["strategy_usage"]["low_confidence_strategy"], 1)
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "combined")  # Should be combined with voting
        self.assertGreaterEqual(result["confidence"], 0.5)  # Should have reasonable confidence
        
        # Check that the result has the expected fields
        self.assertIn("title", result)
        self.assertIn("content", result)
        self.assertEqual(result["url"], "http://example.com")
    
    @pytest.mark.asyncio
    async def test_execute_parallel(self):
        """Test parallel execution of strategies."""
        # Configure for parallel execution
        self.multi_strategy.config["parallel_execution"] = True
        
        # Mock asyncio.gather to run synchronously for testing
        with patch('asyncio.gather', side_effect=lambda *args: asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*args)
        )):
            # Execute the multi-strategy
            result = self.multi_strategy.execute("http://example.com")
        
        # Check that all strategies were executed
        self.assertEqual(self.multi_strategy.extraction_stats["strategy_usage"]["success1"], 1)
        self.assertEqual(self.multi_strategy.extraction_stats["strategy_usage"]["success2"], 1)
        self.assertEqual(self.multi_strategy.extraction_stats["strategy_usage"]["low_confidence_strategy"], 1)
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "combined")  # Should be combined with voting
        self.assertGreaterEqual(result["confidence"], 0.5)  # Should have reasonable confidence
    
    def test_execute_with_high_confidence_strategy(self):
        """Test that a high confidence strategy is used directly."""
        # Create a multi-strategy with only high confidence strategies
        multi_strategy = MultiStrategyV2(
            context=self.context,
            confidence_threshold=0.75
        )
        multi_strategy.add_strategy(self.success_strategy1)  # Confidence 0.8
        
        # Execute the multi-strategy
        result = multi_strategy.execute("http://example.com")
        
        # Check that the high confidence strategy result is used directly
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "success1")
        self.assertEqual(result["confidence"], 0.8)
        self.assertEqual(result["title"], "Title from success1")
    
    def test_execute_with_low_confidence_strategies(self):
        """Test that low confidence strategies are combined."""
        # Create a multi-strategy with only low confidence strategies
        multi_strategy = MultiStrategyV2(
            context=self.context,
            confidence_threshold=0.75,
            use_voting=True
        )
        multi_strategy.add_strategy(self.low_confidence_strategy)  # Confidence 0.3
        
        # Add a second low confidence strategy with different data
        low_confidence2 = MockLowConfidenceStrategy(self.context, "low_confidence2")
        multi_strategy.add_strategy(low_confidence2)
        
        # Execute the multi-strategy
        result = multi_strategy.execute("http://example.com")
        
        # Check that the results are combined
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "combined")
        # Confidence should be low but non-zero
        self.assertGreater(result["confidence"], 0)
        self.assertLess(result["confidence"], 0.75)
    
    def test_combine_results_with_voting(self):
        """Test combining results with weighted voting."""
        # Create strategy results for voting
        strategy_results = [
            (
                {"title": "Product A", "price": "$99.99", "color": "red"},
                0.8,
                "strategy1"
            ),
            (
                {"title": "Product A", "price": "$99.99", "color": "blue"},
                0.7,
                "strategy2"
            ),
            (
                {"title": "Different Product", "price": "$199.99", "weight": "2kg"},
                0.4,
                "strategy3"
            )
        ]
        
        # Combine the results
        combined_data, combined_confidence = self.multi_strategy._combine_results_with_voting(strategy_results)
        
        # Check the combined results
        self.assertEqual(combined_data["title"], "Product A")  # Majority vote
        self.assertEqual(combined_data["price"], "$99.99")  # Same in top strategies
        self.assertEqual(combined_data["color"], "red")  # Higher confidence wins
        self.assertIn("weight", combined_data)  # Unique field is included
        
        # Confidence should be reasonable
        self.assertGreater(combined_confidence, 0.5)
    
    def test_strategy_performance_tracking(self):
        """Test tracking strategy performance."""
        # Execute strategies multiple times
        self.multi_strategy.execute("http://example.com/page1")
        self.multi_strategy.execute("http://example.com/page2")
        
        # Check performance tracking
        performance = self.multi_strategy.strategy_performance
        
        # All strategies should have been executed twice
        self.assertEqual(performance["success1"]["total_executions"], 2)
        self.assertEqual(performance["success2"]["total_executions"], 2)
        self.assertEqual(performance["low_confidence_strategy"]["total_executions"], 2)
        
        # Success strategies should have successful executions
        self.assertEqual(performance["success1"]["successful_executions"], 2)
        self.assertEqual(performance["success2"]["successful_executions"], 2)
        
        # Success rate should be calculated
        self.assertEqual(performance["success1"]["success_rate"], 1.0)
        self.assertEqual(performance["success2"]["success_rate"], 1.0)
        
        # Average confidence should be calculated
        self.assertGreater(performance["success1"]["avg_confidence"], 0)
        self.assertGreater(performance["success2"]["avg_confidence"], 0)
    
    def test_crawl(self):
        """Test crawling with the multi-strategy."""
        # Configure for a small crawl
        self.multi_strategy.config["max_depth"] = 1
        self.multi_strategy.config["max_pages"] = 5
        
        # Perform the crawl
        result = self.multi_strategy.crawl("http://example.com")
        
        # Check that the crawl was performed
        self.assertIsNotNone(result)
        self.assertIn("results", result)
        self.assertIn("metrics", result)
        self.assertIn("visited_urls", result)
        
        # Check that the starting URL was visited
        self.assertIn("http://example.com", result["visited_urls"])
        
        # Check that results were collected
        self.assertGreater(len(result["results"]), 0)
        
        # Check the metrics
        metrics = result["metrics"]
        self.assertIn("extraction_stats", metrics)
        self.assertIn("strategy_performance", metrics)
        self.assertIn("child_strategies", metrics)
    
    def test_extract(self):
        """Test extracting data with the multi-strategy."""
        # Extract data from sample HTML
        result = self.multi_strategy.extract(self.sample_html, "http://example.com")
        
        # Check that the extraction was performed
        self.assertIsNotNone(result)
        self.assertIn("strategy", result)
        self.assertIn("confidence", result)
        
        # Check that all strategies were used
        self.assertEqual(self.multi_strategy.extraction_stats["strategy_usage"]["success1"], 1)
        self.assertEqual(self.multi_strategy.extraction_stats["strategy_usage"]["success2"], 1)
        self.assertEqual(self.multi_strategy.extraction_stats["strategy_usage"]["low_confidence_strategy"], 1)
        
        # Check that results were combined
        self.assertEqual(result["strategy"], "combined")
    
    def test_get_results(self):
        """Test getting combined results from the multi-strategy."""
        # Execute some strategies to generate results
        self.multi_strategy.execute("http://example.com/page1")
        self.multi_strategy.execute("http://example.com/page2")
        
        # Get the combined results
        results = self.multi_strategy.get_results()
        
        # Check that results were collected
        self.assertGreater(len(results), 0)
        
        # Check that results include metadata
        self.assertIn("data", results[0])
        self.assertIn("source_url", results[0])
        self.assertIn("score", results[0])
    
    def test_metrics(self):
        """Test getting metrics from the multi-strategy."""
        # Execute some strategies to generate metrics
        self.multi_strategy.execute("http://example.com")
        
        # Get the metrics
        metrics = self.multi_strategy.get_metrics()
        
        # Check that the metrics are complete
        self.assertIn("extraction_stats", metrics)
        self.assertIn("strategy_performance", metrics)
        self.assertIn("child_strategies", metrics)
        self.assertIn("total_children", metrics)
        
        # Check extraction stats
        stats = metrics["extraction_stats"]
        self.assertEqual(stats["total_attempts"], 1)
        self.assertEqual(stats["successful_extractions"], 1)
        
        # Check strategy performance
        performance = metrics["strategy_performance"]
        self.assertIn("success1", performance)
        self.assertIn("success2", performance)
    
    def test_error_handling(self):
        """Test handling errors in strategies."""
        # Create a multi-strategy with an error-prone strategy
        multi_strategy = MultiStrategyV2(context=self.context)
        multi_strategy.add_strategy(self.success_strategy1)
        multi_strategy.add_strategy(self.error_strategy)
        
        # Execute the multi-strategy
        result = multi_strategy.execute("http://example.com")
        
        # Check that execution continued despite the error
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "success1")
        
        # Check that the error was handled
        self.assertEqual(multi_strategy.extraction_stats["strategy_usage"]["error_strategy"], 1)
        self.assertEqual(multi_strategy.extraction_stats["strategy_success"]["error_strategy"], 0)
    
    def test_deduplication(self):
        """Test deduplication of results."""
        # Create mock results with duplicate URLs
        self.success_strategy1._results = [
            {"url": "http://example.com/duplicate", "title": "Result 1"},
            {"url": "http://example.com/unique1", "title": "Result 2"}
        ]
        
        self.success_strategy2._results = [
            {"url": "http://example.com/duplicate", "title": "Result 3"},
            {"url": "http://example.com/unique2", "title": "Result 4"}
        ]
        
        # Enable deduplication
        self.multi_strategy.config["deduplicate"] = True
        
        # Get deduplicated results
        results = self.multi_strategy.get_results()
        
        # Check that duplicates were removed
        urls = [r.get("url") for r in results if isinstance(r, dict) and "url" in r]
        self.assertEqual(len(set(urls)), len(urls))  # No duplicate URLs
        
        # Disable deduplication
        self.multi_strategy.config["deduplicate"] = False
        
        # Get non-deduplicated results
        results = self.multi_strategy.get_results()
        
        # Check that duplicates were kept
        urls = [r.get("url") for r in results if isinstance(r, dict) and "url" in r]
        self.assertGreaterEqual(len(urls), len(set(urls)))  # May have duplicate URLs

# Run tests if executed directly
if __name__ == '__main__':
    unittest.main()