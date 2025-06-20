"""
Tests for the composite strategy pattern.

This module tests the various composite strategies:
- CompositeStrategy (base class)
- SequentialStrategy
- FallbackStrategy
- PipelineStrategy
- ParallelStrategy
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, List, Optional, Any

from strategies.core.strategy_interface import BaseStrategy
from strategies.core.strategy_context import StrategyContext
from strategies.core.composite_strategy import (
    CompositeStrategy, SequentialStrategy, FallbackStrategy, 
    PipelineStrategy, ParallelStrategy
)


# Mock strategies for testing
class MockSuccessStrategy(BaseStrategy):
    """A strategy that always succeeds."""
    
    def __init__(self, context=None, name="success_strategy", result=None):
        """Initialize with optional result to return."""
        super().__init__(context)
        self._name = name
        self._result = result or {"success": True, "strategy": name}
        self._results = []
    
    def execute(self, url, **kwargs):
        """Always return success result."""
        result = self._result.copy()
        result["url"] = url
        self._results.append(result)
        return result
    
    def crawl(self, start_url, **kwargs):
        """Always return success result for crawl."""
        result = self._result.copy()
        result["url"] = start_url
        result["crawled"] = True
        self._results.append(result)
        return result
    
    def extract(self, html_content, url, **kwargs):
        """Always return success result for extract."""
        result = self._result.copy()
        result["url"] = url
        result["extracted"] = True
        result["content_length"] = len(html_content)
        self._results.append(result)
        return result
    
    def get_results(self):
        """Return collected results."""
        return self._results
    
    @property
    def name(self):
        """Get strategy name."""
        return self._name


class MockFailureStrategy(BaseStrategy):
    """A strategy that always fails (returns None)."""
    
    def __init__(self, context=None, name="failure_strategy"):
        """Initialize with name."""
        super().__init__(context)
        self._name = name
        self._results = []
    
    def execute(self, url, **kwargs):
        """Always return None."""
        return None
    
    def crawl(self, start_url, **kwargs):
        """Always return None for crawl."""
        return None
    
    def extract(self, html_content, url, **kwargs):
        """Always return None for extract."""
        return None
    
    def get_results(self):
        """Return empty results list."""
        return self._results
    
    @property
    def name(self):
        """Get strategy name."""
        return self._name


class MockErrorStrategy(BaseStrategy):
    """A strategy that always raises an exception."""
    
    def __init__(self, context=None, name="error_strategy"):
        """Initialize with name."""
        super().__init__(context)
        self._name = name
        self._results = []
    
    def execute(self, url, **kwargs):
        """Always raise ValueError."""
        raise ValueError(f"Mock strategy {self._name} error on {url}")
    
    def crawl(self, start_url, **kwargs):
        """Always raise ValueError for crawl."""
        raise ValueError(f"Mock strategy {self._name} crawl error on {start_url}")
    
    def extract(self, html_content, url, **kwargs):
        """Always raise ValueError for extract."""
        raise ValueError(f"Mock strategy {self._name} extract error on {url}")
    
    def get_results(self):
        """Return empty results list."""
        return self._results
    
    @property
    def name(self):
        """Get strategy name."""
        return self._name


class MockTransformStrategy(BaseStrategy):
    """A strategy that transforms its input and passes it along."""
    
    def __init__(self, context=None, name="transform_strategy", transform_key="transformed_by"):
        """Initialize with name and transform key."""
        super().__init__(context)
        self._name = name
        self._transform_key = transform_key
        self._results = []
    
    def execute(self, url, **kwargs):
        """Transform the input."""
        result = kwargs.copy()
        result["url"] = url
        result[self._transform_key] = self._name
        self._results.append(result)
        return result
    
    def crawl(self, start_url, **kwargs):
        """Transform the input for crawl."""
        result = kwargs.copy()
        result["url"] = start_url
        result[self._transform_key] = self._name
        result["crawled"] = True
        self._results.append(result)
        return result
    
    def extract(self, html_content, url, **kwargs):
        """Transform the input for extract."""
        result = kwargs.copy()
        result["url"] = url
        result[self._transform_key] = self._name
        result["html_content"] = f"Transformed by {self._name}: {html_content[:10]}..."
        self._results.append(result)
        return result
    
    def get_results(self):
        """Return collected results."""
        return self._results
    
    @property
    def name(self):
        """Get strategy name."""
        return self._name


# Test fixtures
@pytest.fixture
def mock_context():
    """Create a mock strategy context."""
    return MagicMock(spec=StrategyContext)


# Tests for CompositeStrategy (base class)
class TestCompositeStrategy:
    """Tests for the base CompositeStrategy class."""
    
    def test_add_strategy(self, mock_context):
        """Test adding a strategy to a composite."""
        composite = CompositeStrategy(mock_context)
        strategy = MockSuccessStrategy(mock_context, "test_strategy")
        
        composite.add_strategy(strategy)
        
        assert composite.get_child_strategy("test_strategy") is strategy
        assert len(composite.get_child_strategies()) == 1
    
    def test_add_invalid_strategy(self, mock_context):
        """Test adding an invalid object as a strategy."""
        composite = CompositeStrategy(mock_context)
        
        with pytest.raises(TypeError):
            composite.add_strategy("not_a_strategy")
    
    def test_remove_strategy(self, mock_context):
        """Test removing a strategy from a composite."""
        composite = CompositeStrategy(mock_context)
        strategy1 = MockSuccessStrategy(mock_context, "strategy1")
        strategy2 = MockSuccessStrategy(mock_context, "strategy2")
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        composite.remove_strategy("strategy1")
        
        assert composite.get_child_strategy("strategy1") is None
        assert composite.get_child_strategy("strategy2") is strategy2
        assert len(composite.get_child_strategies()) == 1
    
    def test_get_results(self, mock_context):
        """Test getting combined results from child strategies."""
        composite = CompositeStrategy(mock_context)
        strategy1 = MockSuccessStrategy(mock_context, "strategy1")
        strategy2 = MockSuccessStrategy(mock_context, "strategy2")
        
        # Add results to the strategies
        strategy1.execute("http://example.com/1")
        strategy2.execute("http://example.com/2")
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        results = composite.get_results()
        assert len(results) == 2
        assert any(r["url"] == "http://example.com/1" for r in results)
        assert any(r["url"] == "http://example.com/2" for r in results)
    
    def test_initialize_and_shutdown(self, mock_context):
        """Test initialize and shutdown propagate to child strategies."""
        composite = CompositeStrategy(mock_context)
        strategy1 = MagicMock(spec=BaseStrategy)
        strategy1.name = "strategy1"
        strategy2 = MagicMock(spec=BaseStrategy)
        strategy2.name = "strategy2"
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        composite.initialize()
        strategy1.initialize.assert_called_once()
        strategy2.initialize.assert_called_once()
        
        composite.shutdown()
        strategy1.shutdown.assert_called_once()
        strategy2.shutdown.assert_called_once()


# Tests for SequentialStrategy
class TestSequentialStrategy:
    """Tests for the SequentialStrategy class."""
    
    def test_execute(self, mock_context):
        """Test sequential execution of child strategies."""
        composite = SequentialStrategy(mock_context)
        strategy1 = MockSuccessStrategy(mock_context, "strategy1", {"key1": "value1", "list": [1, 2]})
        strategy2 = MockSuccessStrategy(mock_context, "strategy2", {"key2": "value2", "list": [3, 4]})
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        result = composite.execute("http://example.com")
        
        assert result is not None
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        assert result["list"] == [1, 2, 3, 4]  # Lists are combined
        assert len(strategy1.get_results()) == 1
        assert len(strategy2.get_results()) == 1
    
    def test_execute_with_failure(self, mock_context):
        """Test sequential execution when one strategy fails."""
        composite = SequentialStrategy(mock_context)
        strategy1 = MockSuccessStrategy(mock_context, "strategy1", {"key1": "value1"})
        strategy2 = MockFailureStrategy(mock_context, "strategy2")
        strategy3 = MockSuccessStrategy(mock_context, "strategy3", {"key3": "value3"})
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        composite.add_strategy(strategy3)
        
        result = composite.execute("http://example.com")
        
        assert result is not None
        assert result["key1"] == "value1"
        assert result["key3"] == "value3"
        assert "key2" not in result  # strategy2 failed, so no key2
        assert len(strategy1.get_results()) == 1
        assert len(strategy3.get_results()) == 1
    
    def test_execute_with_error(self, mock_context):
        """Test sequential execution when one strategy raises an exception."""
        composite = SequentialStrategy(mock_context)
        strategy1 = MockSuccessStrategy(mock_context, "strategy1", {"key1": "value1"})
        strategy2 = MockErrorStrategy(mock_context, "strategy2")
        strategy3 = MockSuccessStrategy(mock_context, "strategy3", {"key3": "value3"})
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        composite.add_strategy(strategy3)
        
        result = composite.execute("http://example.com")
        
        assert result is not None
        assert result["key1"] == "value1"
        assert result["key3"] == "value3"
        assert len(strategy1.get_results()) == 1
        assert len(strategy3.get_results()) == 1
    
    def test_execute_all_fail(self, mock_context):
        """Test sequential execution when all strategies fail."""
        composite = SequentialStrategy(mock_context)
        strategy1 = MockFailureStrategy(mock_context, "strategy1")
        strategy2 = MockFailureStrategy(mock_context, "strategy2")
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        result = composite.execute("http://example.com")
        
        assert result is None
    
    def test_crawl(self, mock_context):
        """Test sequential crawling."""
        composite = SequentialStrategy(mock_context)
        strategy1 = MockSuccessStrategy(mock_context, "strategy1", {"key1": "value1", "links": ["link1"]})
        strategy2 = MockSuccessStrategy(mock_context, "strategy2", {"key2": "value2", "links": ["link2"]})
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        result = composite.crawl("http://example.com")
        
        assert result is not None
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        assert result["links"] == ["link1", "link2"]  # Lists are combined
        assert len(strategy1.get_results()) == 1
        assert len(strategy2.get_results()) == 1
    
    def test_extract(self, mock_context):
        """Test sequential extraction."""
        composite = SequentialStrategy(mock_context)
        strategy1 = MockSuccessStrategy(mock_context, "strategy1", {"key1": "value1", "data": ["item1"]})
        strategy2 = MockSuccessStrategy(mock_context, "strategy2", {"key2": "value2", "data": ["item2"]})
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        result = composite.extract("<html>test</html>", "http://example.com")
        
        assert result is not None
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        assert result["data"] == ["item1", "item2"]  # Lists are combined
        assert len(strategy1.get_results()) == 1
        assert len(strategy2.get_results()) == 1


# Tests for FallbackStrategy
class TestFallbackStrategy:
    """Tests for the FallbackStrategy class."""
    
    def test_execute_first_success(self, mock_context):
        """Test fallback to first successful strategy."""
        composite = FallbackStrategy(mock_context)
        strategy1 = MockSuccessStrategy(mock_context, "strategy1", {"key1": "value1"})
        strategy2 = MockSuccessStrategy(mock_context, "strategy2", {"key2": "value2"})
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        result = composite.execute("http://example.com")
        
        assert result is not None
        assert result["key1"] == "value1"
        assert len(strategy1.get_results()) == 1
        assert len(strategy2.get_results()) == 0  # Second strategy never called
    
    def test_execute_second_success(self, mock_context):
        """Test fallback to second strategy when first fails."""
        composite = FallbackStrategy(mock_context)
        strategy1 = MockFailureStrategy(mock_context, "strategy1")
        strategy2 = MockSuccessStrategy(mock_context, "strategy2", {"key2": "value2"})
        strategy3 = MockSuccessStrategy(mock_context, "strategy3", {"key3": "value3"})
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        composite.add_strategy(strategy3)
        
        result = composite.execute("http://example.com")
        
        assert result is not None
        assert result["key2"] == "value2"
        assert len(strategy2.get_results()) == 1
        assert len(strategy3.get_results()) == 0  # Third strategy never called
    
    def test_execute_with_error(self, mock_context):
        """Test fallback when a strategy raises an exception."""
        composite = FallbackStrategy(mock_context)
        strategy1 = MockErrorStrategy(mock_context, "strategy1")
        strategy2 = MockSuccessStrategy(mock_context, "strategy2", {"key2": "value2"})
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        result = composite.execute("http://example.com")
        
        assert result is not None
        assert result["key2"] == "value2"
        assert len(strategy2.get_results()) == 1
    
    def test_execute_all_fail(self, mock_context):
        """Test fallback when all strategies fail."""
        composite = FallbackStrategy(mock_context)
        strategy1 = MockFailureStrategy(mock_context, "strategy1")
        strategy2 = MockErrorStrategy(mock_context, "strategy2")
        strategy3 = MockFailureStrategy(mock_context, "strategy3")
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        composite.add_strategy(strategy3)
        
        result = composite.execute("http://example.com")
        
        assert result is None
    
    def test_crawl(self, mock_context):
        """Test fallback crawling."""
        composite = FallbackStrategy(mock_context)
        strategy1 = MockFailureStrategy(mock_context, "strategy1")
        strategy2 = MockSuccessStrategy(mock_context, "strategy2", {"key2": "value2", "crawled": True})
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        result = composite.crawl("http://example.com")
        
        assert result is not None
        assert result["key2"] == "value2"
        assert result["crawled"] is True
        assert len(strategy2.get_results()) == 1
    
    def test_extract(self, mock_context):
        """Test fallback extraction."""
        composite = FallbackStrategy(mock_context)
        strategy1 = MockFailureStrategy(mock_context, "strategy1")
        strategy2 = MockSuccessStrategy(mock_context, "strategy2", {"key2": "value2", "extracted": True})
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        result = composite.extract("<html>test</html>", "http://example.com")
        
        assert result is not None
        assert result["key2"] == "value2"
        assert result["extracted"] is True
        assert len(strategy2.get_results()) == 1


# Tests for PipelineStrategy
class TestPipelineStrategy:
    """Tests for the PipelineStrategy class."""
    
    def test_execute(self, mock_context):
        """Test pipeline passing results between strategies."""
        composite = PipelineStrategy(mock_context)
        strategy1 = MockTransformStrategy(mock_context, "strategy1", "transform1")
        strategy2 = MockTransformStrategy(mock_context, "strategy2", "transform2")
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        # Initial kwargs to pass to the pipeline
        initial_kwargs = {"initial_param": "initial_value"}
        
        result = composite.execute("http://example.com", **initial_kwargs)
        
        assert result is not None
        assert result["initial_param"] == "initial_value"  # Original param preserved
        assert result["transform1"] == "strategy1"  # First strategy added this
        assert result["transform2"] == "strategy2"  # Second strategy added this
        assert len(strategy1.get_results()) == 1
        assert len(strategy2.get_results()) == 1
    
    def test_execute_empty_pipeline(self, mock_context):
        """Test executing an empty pipeline."""
        composite = PipelineStrategy(mock_context)
        
        result = composite.execute("http://example.com")
        
        assert result is None
    
    def test_execute_stage_fails(self, mock_context):
        """Test pipeline when a stage fails (returns None)."""
        composite = PipelineStrategy(mock_context)
        strategy1 = MockTransformStrategy(mock_context, "strategy1", "transform1")
        strategy2 = MockFailureStrategy(mock_context, "strategy2")
        strategy3 = MockTransformStrategy(mock_context, "strategy3", "transform3")  # Should never be called
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        composite.add_strategy(strategy3)
        
        result = composite.execute("http://example.com")
        
        assert result is None
        assert len(strategy1.get_results()) == 1
        assert len(strategy3.get_results()) == 0  # Never called due to previous failure
    
    def test_execute_stage_errors(self, mock_context):
        """Test pipeline when a stage raises an exception."""
        composite = PipelineStrategy(mock_context)
        strategy1 = MockTransformStrategy(mock_context, "strategy1", "transform1")
        strategy2 = MockErrorStrategy(mock_context, "strategy2")
        strategy3 = MockTransformStrategy(mock_context, "strategy3", "transform3")  # Should never be called
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        composite.add_strategy(strategy3)
        
        result = composite.execute("http://example.com")
        
        assert result is None
        assert len(strategy1.get_results()) == 1
        assert len(strategy3.get_results()) == 0  # Never called due to previous error
    
    def test_extract(self, mock_context):
        """Test pipeline extraction with HTML transformation."""
        composite = PipelineStrategy(mock_context)
        strategy1 = MockTransformStrategy(mock_context, "strategy1", "transform1")
        strategy2 = MockTransformStrategy(mock_context, "strategy2", "transform2")
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        result = composite.extract("<html>original content</html>", "http://example.com")
        
        assert result is not None
        assert "html_content" in result  # HTML was transformed
        assert result["transform1"] == "strategy1"
        assert result["transform2"] == "strategy2"
        assert len(strategy1.get_results()) == 1
        assert len(strategy2.get_results()) == 1


# Tests for ParallelStrategy
@pytest.mark.parametrize("combine_mode", ["merge", "append", "best"])
class TestParallelStrategy:
    """Tests for the ParallelStrategy class."""
    
    def test_execute(self, mock_context, combine_mode):
        """Test parallel execution of strategies."""
        composite = ParallelStrategy(mock_context, max_workers=2, combine_mode=combine_mode)
        strategy1 = MockSuccessStrategy(mock_context, "strategy1", {"key1": "value1", "common": "from1", "list": [1, 2]})
        strategy2 = MockSuccessStrategy(mock_context, "strategy2", {"key2": "value2", "common": "from2", "list": [3, 4]})
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        result = composite.execute("http://example.com")
        
        assert result is not None
        
        if combine_mode == "merge":
            # In merge mode, first strategy wins conflicts
            assert result["key1"] == "value1"
            assert result["key2"] == "value2"
            assert result["common"] == "from1"  # First strategy wins
            assert set(result["list"]) == {1, 2, 3, 4}  # Lists combined
        
        elif combine_mode == "append":
            # In append mode, we get a list of strategy results
            assert "results" in result
            assert len(result["results"]) == 2
            strategies_seen = {r["strategy"] for r in result["results"]}
            assert strategies_seen == {"strategy1", "strategy2"}
        
        elif combine_mode == "best":
            # In best mode, we get the result with most fields
            # Both strategies have same number of fields, so either could win
            assert "key1" in result or "key2" in result
            assert "common" in result
            assert "list" in result
        
        # Both strategies executed
        assert len(strategy1.get_results()) == 1
        assert len(strategy2.get_results()) == 1
    
    def test_execute_with_failure(self, mock_context, combine_mode):
        """Test parallel execution when one strategy fails."""
        composite = ParallelStrategy(mock_context, max_workers=2, combine_mode=combine_mode)
        strategy1 = MockSuccessStrategy(mock_context, "strategy1", {"key1": "value1"})
        strategy2 = MockFailureStrategy(mock_context, "strategy2")
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        result = composite.execute("http://example.com")
        
        assert result is not None
        
        if combine_mode == "merge" or combine_mode == "best":
            assert result["key1"] == "value1"
        
        elif combine_mode == "append":
            assert "results" in result
            assert len(result["results"]) == 1
            assert result["results"][0]["strategy"] == "strategy1"
        
        # Both strategies attempted
        assert len(strategy1.get_results()) == 1
        assert len(strategy2.get_results()) == 0  # No result stored since it failed
    
    def test_execute_all_fail(self, mock_context, combine_mode):
        """Test parallel execution when all strategies fail."""
        composite = ParallelStrategy(mock_context, max_workers=2, combine_mode=combine_mode)
        strategy1 = MockFailureStrategy(mock_context, "strategy1")
        strategy2 = MockErrorStrategy(mock_context, "strategy2")
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        result = composite.execute("http://example.com")
        
        assert result is None
    
    def test_extract(self, mock_context, combine_mode):
        """Test parallel extraction."""
        composite = ParallelStrategy(mock_context, max_workers=2, combine_mode=combine_mode)
        strategy1 = MockSuccessStrategy(mock_context, "strategy1", {"title": "Title 1", "data": ["item1"]})
        strategy2 = MockSuccessStrategy(mock_context, "strategy2", {"price": 10.99, "data": ["item2"]})
        
        composite.add_strategy(strategy1)
        composite.add_strategy(strategy2)
        
        result = composite.extract("<html>test</html>", "http://example.com")
        
        assert result is not None
        
        if combine_mode == "merge":
            assert result["title"] == "Title 1"
            assert result["price"] == 10.99
            assert set(result["data"]) == {"item1", "item2"}
        
        elif combine_mode == "append":
            assert "results" in result
            assert len(result["results"]) == 2
            data_seen = set()
            for r in result["results"]:
                data_seen.update(r["data"]["data"])
            assert data_seen == {"item1", "item2"}
        
        elif combine_mode == "best":
            assert ("title" in result and "data" in result) or ("price" in result and "data" in result)
        
        # Both strategies executed
        assert len(strategy1.get_results()) == 1
        assert len(strategy2.get_results()) == 1