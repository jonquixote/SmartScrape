"""
Tests for the Strategy Fallback Framework.

This module tests the components of the fallback framework, including:
- Fallback conditions
- Strategy result quality assessment
- Fallback strategy chains
- Fallback registry
- Recovery mechanisms
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any, Optional, Set

from strategies.fallback_framework import (
    FallbackCondition, ErrorBasedCondition, AttemptBasedCondition,
    TimeoutCondition, ResultQualityCondition, CompositeCondition,
    StrategyResultQualityAssessor, FallbackStrategyChain, FallbackRegistry,
    merge_strategy_results, extract_best_components, rebuild_from_fragments,
    estimate_result_quality, create_fallback_condition
)
from strategies.core.strategy_interface import BaseStrategy
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_types import StrategyType, StrategyCapability


# Mock strategies for testing
class MockSuccessStrategy(BaseStrategy):
    """Mock strategy that always succeeds."""
    
    def __init__(self, context=None, name="success_strategy", result=None):
        super().__init__(context)
        self._name = name
        self._result = result or {"success": True, "strategy": self._name}
        self._results = []
    
    def execute(self, url, **kwargs):
        """Return success result."""
        result = self._result.copy()
        result["url"] = url
        self._results.append(result)
        return result
    
    def crawl(self, start_url, **kwargs):
        """Return success result for crawl."""
        result = self._result.copy()
        result["url"] = start_url
        result["crawled"] = True
        self._results.append(result)
        return result
    
    def extract(self, html_content, url, **kwargs):
        """Return success result for extract."""
        result = self._result.copy()
        result["url"] = url
        result["extracted"] = True
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
    """Mock strategy that always fails."""
    
    def __init__(self, context=None, name="failure_strategy"):
        super().__init__(context)
        self._name = name
        self._results = []
    
    def execute(self, url, **kwargs):
        """Return None to indicate failure."""
        return None
    
    def crawl(self, start_url, **kwargs):
        """Return None to indicate failure."""
        return None
    
    def extract(self, html_content, url, **kwargs):
        """Return None to indicate failure."""
        return None
    
    def get_results(self):
        """Return collected results."""
        return self._results
    
    @property
    def name(self):
        """Get strategy name."""
        return self._name


class MockErrorStrategy(BaseStrategy):
    """Mock strategy that always raises an exception."""
    
    def __init__(self, context=None, name="error_strategy"):
        super().__init__(context)
        self._name = name
        self._results = []
    
    def execute(self, url, **kwargs):
        """Raise an exception."""
        raise ValueError("Mock strategy error")
    
    def crawl(self, start_url, **kwargs):
        """Raise an exception."""
        raise ValueError("Mock strategy error")
    
    def extract(self, html_content, url, **kwargs):
        """Raise an exception."""
        raise ValueError("Mock strategy error")
    
    def get_results(self):
        """Return collected results."""
        return self._results
    
    @property
    def name(self):
        """Get strategy name."""
        return self._name


class MockPartialStrategy(BaseStrategy):
    """Mock strategy that returns partial results."""
    
    def __init__(self, context=None, name="partial_strategy", partial_result=None):
        super().__init__(context)
        self._name = name
        self._partial_result = partial_result or {"partial": True, "strategy": self._name}
        self._results = []
    
    def execute(self, url, **kwargs):
        """Return partial result."""
        result = self._partial_result.copy()
        result["url"] = url
        self._results.append(result)
        return result
    
    def crawl(self, start_url, **kwargs):
        """Return partial result for crawl."""
        result = self._partial_result.copy()
        result["url"] = start_url
        result["crawled"] = True
        self._results.append(result)
        return result
    
    def extract(self, html_content, url, **kwargs):
        """Return partial result for extract."""
        result = self._partial_result.copy()
        result["url"] = url
        result["extracted"] = True
        self._results.append(result)
        return result
    
    def get_results(self):
        """Return collected results."""
        return self._results
    
    @property
    def name(self):
        """Get strategy name."""
        return self._name


# Fixture for strategy context
@pytest.fixture
def strategy_context():
    """Create a mock strategy context."""
    context = MagicMock(spec=StrategyContext)
    context.get_service.return_value = None
    return context


# Test Fallback Conditions
class TestFallbackConditions:
    """Tests for various fallback conditions."""
    
    def test_error_based_condition(self):
        """Test error-based fallback condition."""
        # Condition with categories
        condition = ErrorBasedCondition(error_categories={"network", "http"})
        
        # Should trigger for matching category
        context = {"error": {"category": "network", "severity": "warning"}}
        assert condition.should_fallback(context) is True
        
        # Should not trigger for non-matching category
        context = {"error": {"category": "parsing", "severity": "warning"}}
        assert condition.should_fallback(context) is False
        
        # Should not trigger if no error
        context = {"result": {"data": "test"}}
        assert condition.should_fallback(context) is False
        
        # Test with severities
        condition = ErrorBasedCondition(error_severities={"error", "critical"})
        
        # Should trigger for matching severity
        context = {"error": {"category": "parsing", "severity": "error"}}
        assert condition.should_fallback(context) is True
        
        # Should not trigger for non-matching severity
        context = {"error": {"category": "parsing", "severity": "warning"}}
        assert condition.should_fallback(context) is False
        
        # Test with error messages
        condition = ErrorBasedCondition(error_messages=["timeout", "connection refused"])
        
        # Should trigger for matching message
        context = {"error": {"message": "Request timeout"}}
        assert condition.should_fallback(context) is True
        
        # Should not trigger for non-matching message
        context = {"error": {"message": "Invalid data"}}
        assert condition.should_fallback(context) is False
    
    def test_attempt_based_condition(self):
        """Test attempt-based fallback condition."""
        condition = AttemptBasedCondition(max_attempts=3)
        
        # Should not trigger if attempts < max
        context = {"attempts": 2}
        assert condition.should_fallback(context) is False
        
        # Should trigger if attempts >= max
        context = {"attempts": 3}
        assert condition.should_fallback(context) is True
        context = {"attempts": 4}
        assert condition.should_fallback(context) is True
        
        # Should not trigger if attempts field is missing
        context = {}
        assert condition.should_fallback(context) is False
    
    def test_timeout_condition(self):
        """Test timeout-based fallback condition."""
        condition = TimeoutCondition(timeout_seconds=1.0)
        
        # Setup context with start time
        start_time = time.time() - 0.5  # 0.5 seconds ago
        context = {"start_time": start_time}
        
        # Should not trigger if elapsed time < timeout
        assert condition.should_fallback(context) is False
        
        # Update start time to be more than timeout ago
        start_time = time.time() - 1.5  # 1.5 seconds ago
        context = {"start_time": start_time}
        
        # Should trigger if elapsed time >= timeout
        assert condition.should_fallback(context) is True
        
        # Should not trigger if start_time field is missing
        context = {}
        assert condition.should_fallback(context) is False
    
    def test_result_quality_condition(self):
        """Test result quality based fallback condition."""
        # Mock assessor that returns predetermined scores
        mock_assessor = MagicMock(spec=StrategyResultQualityAssessor)
        mock_assessor.get_quality_score.side_effect = lambda r: 0.8 if "complete" in r else 0.3
        
        condition = ResultQualityCondition(min_quality_score=0.5, assessor=mock_assessor)
        
        # Should not trigger for high quality result
        context = {"result": {"complete": True, "data": "good result"}}
        assert condition.should_fallback(context) is False
        
        # Should trigger for low quality result
        context = {"result": {"partial": True, "data": "partial result"}}
        assert condition.should_fallback(context) is True
        
        # Should trigger if result is missing
        context = {}
        assert condition.should_fallback(context) is True
    
    def test_composite_condition_or_logic(self):
        """Test composite condition with OR logic."""
        # Create component conditions
        attempt_condition = AttemptBasedCondition(max_attempts=3)
        error_condition = ErrorBasedCondition(error_severities={"critical"})
        
        # Create composite with OR logic
        composite = CompositeCondition(
            [attempt_condition, error_condition],
            logic=CompositeCondition.Logic.OR
        )
        
        # Should trigger if any condition is met
        context = {"attempts": 3}  # Only attempt condition met
        assert composite.should_fallback(context) is True
        
        context = {"error": {"severity": "critical"}}  # Only error condition met
        assert composite.should_fallback(context) is True
        
        context = {"attempts": 3, "error": {"severity": "critical"}}  # Both conditions met
        assert composite.should_fallback(context) is True
        
        context = {"attempts": 2, "error": {"severity": "warning"}}  # No conditions met
        assert composite.should_fallback(context) is False
    
    def test_composite_condition_and_logic(self):
        """Test composite condition with AND logic."""
        # Create component conditions
        attempt_condition = AttemptBasedCondition(max_attempts=3)
        error_condition = ErrorBasedCondition(error_severities={"critical"})
        
        # Create composite with AND logic
        composite = CompositeCondition(
            [attempt_condition, error_condition],
            logic=CompositeCondition.Logic.AND
        )
        
        # Should only trigger if all conditions are met
        context = {"attempts": 3}  # Only attempt condition met
        assert composite.should_fallback(context) is False
        
        context = {"error": {"severity": "critical"}}  # Only error condition met
        assert composite.should_fallback(context) is False
        
        context = {"attempts": 3, "error": {"severity": "critical"}}  # Both conditions met
        assert composite.should_fallback(context) is True
        
        context = {"attempts": 2, "error": {"severity": "warning"}}  # No conditions met
        assert composite.should_fallback(context) is False


# Test Strategy Result Quality Assessor
class TestStrategyResultQualityAssessor:
    """Tests for the StrategyResultQualityAssessor."""
    
    def test_assess_completeness(self):
        """Test completeness assessment."""
        # With required fields
        assessor = StrategyResultQualityAssessor(required_fields=["title", "price", "description"])
        
        # Complete result
        result = {"title": "Product A", "price": "9.99", "description": "A great product"}
        assert assessor.assess_completeness(result) == 1.0
        
        # Partial result (2/3 fields)
        result = {"title": "Product A", "price": "9.99"}
        assert assessor.assess_completeness(result) == 2/3
        
        # Empty result
        result = {}
        assert assessor.assess_completeness(result) == 0.0
        
        # None result
        assert assessor.assess_completeness(None) == 0.0
        
        # Without required fields
        assessor = StrategyResultQualityAssessor()
        
        # Any non-empty result should be 100% complete
        result = {"title": "Product A"}
        assert assessor.assess_completeness(result) == 1.0
        
        # Empty result
        result = {}
        assert assessor.assess_completeness(result) == 0.0
    
    def test_assess_confidence(self):
        """Test confidence assessment."""
        assessor = StrategyResultQualityAssessor()
        
        # Result with explicit confidence
        result = {"title": "Product A", "confidence": 0.75}
        assert assessor.assess_confidence(result) == 0.75
        
        # Result with alternative confidence indicator
        result = {"title": "Product A", "score": 0.8}
        assert assessor.assess_confidence(result) == 0.8
        
        # Result without confidence indicator (should use structure heuristic)
        result = {"title": "Product A", "price": "9.99", "details": {"color": "red", "size": "medium"}}
        confidence = assessor.assess_confidence(result)
        assert 0.0 <= confidence <= 1.0  # Should return some reasonable score
        
        # Empty result
        result = {}
        assert assessor.assess_confidence(result) == 0.0
        
        # None result
        assert assessor.assess_confidence(None) == 0.0
    
    def test_assess_relevance(self):
        """Test relevance assessment."""
        assessor = StrategyResultQualityAssessor()
        
        # Result with explicit relevance
        result = {"title": "Product A", "relevance": 0.6}
        assert assessor.assess_relevance(result) == 0.6
        
        # Result without relevance indicator
        result = {"title": "Product A", "price": "9.99", "description": "A lengthy description about this product"}
        relevance = assessor.assess_relevance(result)
        assert 0.0 <= relevance <= 1.0  # Should return some reasonable score
        
        # Empty result
        result = {}
        assert assessor.assess_relevance(result) == 0.0
        
        # None result
        assert assessor.assess_relevance(None) == 0.0
    
    def test_get_quality_score(self):
        """Test overall quality score calculation."""
        # Create an assessor with custom weights
        assessor = StrategyResultQualityAssessor(
            required_fields=["title", "price"],
            weights={"completeness": 0.5, "confidence": 0.3, "relevance": 0.2}
        )
        
        # Mock the individual assessment methods
        original_assess_completeness = assessor.assess_completeness
        original_assess_confidence = assessor.assess_confidence
        original_assess_relevance = assessor.assess_relevance
        
        assessor.assess_completeness = lambda r: 1.0 if r else 0.0
        assessor.assess_confidence = lambda r: 0.8 if r else 0.0
        assessor.assess_relevance = lambda r: 0.6 if r else 0.0
        
        # Calculate quality score
        result = {"title": "Product A", "price": "9.99"}
        expected_score = 1.0 * 0.5 + 0.8 * 0.3 + 0.6 * 0.2  # 0.86
        assert assessor.get_quality_score(result) == expected_score
        
        # Empty result
        result = {}
        assert assessor.get_quality_score(result) == 0.0
        
        # None result
        assert assessor.get_quality_score(None) == 0.0
        
        # Restore original methods
        assessor.assess_completeness = original_assess_completeness
        assessor.assess_confidence = original_assess_confidence
        assessor.assess_relevance = original_assess_relevance


# Test Fallback Strategy Chain
class TestFallbackStrategyChain:
    """Tests for FallbackStrategyChain."""
    
    def test_success_primary_strategy(self, strategy_context):
        """Test successful execution of primary strategy."""
        # Create strategies
        primary = MockSuccessStrategy(strategy_context, "primary", {"primary": True, "data": "Primary result"})
        fallback1 = MockSuccessStrategy(strategy_context, "fallback1", {"fallback": True, "data": "Fallback result 1"})
        fallback2 = MockSuccessStrategy(strategy_context, "fallback2", {"fallback": True, "data": "Fallback result 2"})
        
        # Create chain
        chain = FallbackStrategyChain(primary, [fallback1, fallback2])
        
        # Execute chain
        result = chain.execute("http://example.com")
        
        # Primary should succeed, fallbacks should not be tried
        assert result == {"primary": True, "data": "Primary result", "url": "http://example.com"}
        assert chain.metrics["attempts"] == 1
        assert chain.metrics["fallbacks_triggered"] == 0
        assert len(primary.get_results()) == 1
        assert len(fallback1.get_results()) == 0
        assert len(fallback2.get_results()) == 0
    
    def test_primary_failure_fallback_success(self, strategy_context):
        """Test fallback to first successful strategy when primary fails."""
        # Create strategies
        primary = MockFailureStrategy(strategy_context, "primary")
        fallback1 = MockSuccessStrategy(strategy_context, "fallback1", {"fallback": True, "data": "Fallback result 1"})
        fallback2 = MockSuccessStrategy(strategy_context, "fallback2", {"fallback": True, "data": "Fallback result 2"})
        
        # Create chain
        chain = FallbackStrategyChain(primary, [fallback1, fallback2])
        
        # Execute chain
        result = chain.execute("http://example.com")
        
        # First fallback should succeed
        assert result == {"fallback": True, "data": "Fallback result 1", "url": "http://example.com"}
        assert chain.metrics["attempts"] == 2  # Primary + first fallback
        assert chain.metrics["fallbacks_triggered"] == 1
        assert chain.metrics["successful_fallbacks"] == 1
        assert len(fallback1.get_results()) == 1
        assert len(fallback2.get_results()) == 0  # Second fallback not tried
    
    def test_primary_error_fallback_success(self, strategy_context):
        """Test fallback when primary raises an exception."""
        # Create strategies
        primary = MockErrorStrategy(strategy_context, "primary")
        fallback1 = MockSuccessStrategy(strategy_context, "fallback1", {"fallback": True, "data": "Fallback result 1"})
        
        # Create chain
        chain = FallbackStrategyChain(primary, [fallback1])
        
        # Execute chain
        result = chain.execute("http://example.com")
        
        # Fallback should succeed
        assert result == {"fallback": True, "data": "Fallback result 1", "url": "http://example.com"}
        assert chain.metrics["attempts"] == 2  # Primary + fallback
        assert chain.metrics["fallbacks_triggered"] == 1
        assert chain.metrics["successful_fallbacks"] == 1
        assert len(fallback1.get_results()) == 1
        
        # Check that error was recorded in metrics
        assert "primary" in chain.metrics["strategy_results"]
        assert chain.metrics["strategy_results"]["primary"]["success"] is False
        assert "error" in chain.metrics["strategy_results"]["primary"]
    
    def test_all_strategies_fail(self, strategy_context):
        """Test when all strategies fail."""
        # Create strategies
        primary = MockFailureStrategy(strategy_context, "primary")
        fallback1 = MockFailureStrategy(strategy_context, "fallback1")
        fallback2 = MockFailureStrategy(strategy_context, "fallback2")
        
        # Create chain
        chain = FallbackStrategyChain(primary, [fallback1, fallback2])
        
        # Mock the recover_with_partial_results method to return None
        chain.recover_with_partial_results = MagicMock(return_value=None)
        
        # Execute chain
        result = chain.execute("http://example.com")
        
        # All strategies should fail
        assert result is None
        assert chain.metrics["attempts"] == 3  # Primary + two fallbacks
        assert chain.metrics["fallbacks_triggered"] == 1
        assert chain.metrics["successful_fallbacks"] == 0
        
        # Check that recover_with_partial_results was called
        chain.recover_with_partial_results.assert_called_once()
    
    def test_conditional_fallback(self, strategy_context):
        """Test conditional fallback based on custom condition."""
        # Create strategies
        primary = MockFailureStrategy(strategy_context, "primary")
        fallback = MockSuccessStrategy(strategy_context, "fallback", {"fallback": True})
        
        # Create a condition that only triggers after 2 attempts
        condition = AttemptBasedCondition(max_attempts=2)
        
        # Create chain with condition
        chain = FallbackStrategyChain(primary, [fallback], condition)
        
        # First execution should not trigger fallback due to condition
        context = {"attempts": 1}  # Less than max_attempts
        
        # Mock should_fallback to use our context
        condition.should_fallback = MagicMock(return_value=False)
        
        # Execute chain
        result = chain.execute("http://example.com")
        
        # Should not trigger fallback
        assert result is None
        assert chain.metrics["fallbacks_triggered"] == 0
        
        # Now mock condition to trigger fallback
        condition.should_fallback = MagicMock(return_value=True)
        
        # Execute chain again
        result = chain.execute("http://example.com")
        
        # Should trigger fallback
        assert result is not None
        assert chain.metrics["fallbacks_triggered"] == 1
    
    def test_recover_with_partial_results(self, strategy_context):
        """Test recovering partial results when all strategies fail."""
        # Create a chain with strategies that will all fail
        primary = MockFailureStrategy(strategy_context, "primary")
        fallback1 = MockFailureStrategy(strategy_context, "fallback1")
        fallback2 = MockFailureStrategy(strategy_context, "fallback2")
        
        chain = FallbackStrategyChain(primary, [fallback1, fallback2])
        
        # Manually add some partial results
        chain.partial_results = [
            {"strategy": "primary", "result": {"field1": "value1"}, "success": False},
            {"strategy": "fallback1", "result": {"field2": "value2"}, "success": False},
            {"strategy": "fallback2", "result": {"field3": "value3"}, "success": False}
        ]
        
        # Mock merge_strategy_results to return a merged result
        with patch('strategies.fallback_framework.merge_strategy_results') as mock_merge:
            mock_merge.return_value = {"field1": "value1", "field2": "value2", "field3": "value3"}
            
            # Call recover_with_partial_results
            result = chain.recover_with_partial_results()
            
            # Should successfully recover
            assert result == {"field1": "value1", "field2": "value2", "field3": "value3"}
            
            # merge_strategy_results should be called with the results
            mock_merge.assert_called_once()


# Test Fallback Registry
class TestFallbackRegistry:
    """Tests for FallbackRegistry."""
    
    def test_register_fallback(self):
        """Test registering fallback strategies."""
        registry = FallbackRegistry()
        
        # Define mock strategy classes
        MockStrategyA = MagicMock()
        MockStrategyA.__name__ = "MockStrategyA"
        MockStrategyB = MagicMock()
        MockStrategyB.__name__ = "MockStrategyB"
        
        # Register fallbacks
        registry.register_fallback(
            StrategyType.TRAVERSAL, 
            MockStrategyA,
            error_categories=["network", "timeout"],
            priority=10,
            metadata={"desc": "Strategy A"}
        )
        
        registry.register_fallback(
            StrategyType.EXTRACTION, 
            MockStrategyB,
            error_categories=["parsing"],
            priority=20,
            metadata={"desc": "Strategy B"}
        )
        
        # Check registered fallbacks by type
        traversal_fallbacks = registry.get_fallbacks_for_type(StrategyType.TRAVERSAL)
        assert len(traversal_fallbacks) == 1
        assert traversal_fallbacks[0] == MockStrategyA
        
        extraction_fallbacks = registry.get_fallbacks_for_type(StrategyType.EXTRACTION)
        assert len(extraction_fallbacks) == 1
        assert extraction_fallbacks[0] == MockStrategyB
        
        # Check metadata
        assert registry._fallback_metadata["MockStrategyA"]["priority"] == 10
        assert registry._fallback_metadata["MockStrategyA"]["error_categories"] == ["network", "timeout"]
        assert registry._fallback_metadata["MockStrategyA"]["desc"] == "Strategy A"
        
        # Register duplicate (should be logged but not added again)
        with patch('strategies.fallback_framework.logger') as mock_logger:
            registry.register_fallback(StrategyType.TRAVERSAL, MockStrategyA)
            mock_logger.warning.assert_called_once()
        
        # Should still be only one fallback for TRAVERSAL
        traversal_fallbacks = registry.get_fallbacks_for_type(StrategyType.TRAVERSAL)
        assert len(traversal_fallbacks) == 1
    
    def test_get_fallbacks_for_type(self):
        """Test getting fallbacks for a strategy type."""
        registry = FallbackRegistry()
        
        # Define mock strategy classes
        MockStrategyA = MagicMock()
        MockStrategyA.__name__ = "MockStrategyA"
        MockStrategyB = MagicMock()
        MockStrategyB.__name__ = "MockStrategyB"
        MockStrategyC = MagicMock()
        MockStrategyC.__name__ = "MockStrategyC"
        
        # Register fallbacks with different priorities
        registry.register_fallback(StrategyType.TRAVERSAL, MockStrategyA, priority=30)
        registry.register_fallback(StrategyType.TRAVERSAL, MockStrategyB, priority=10)
        registry.register_fallback(StrategyType.TRAVERSAL, MockStrategyC, priority=20)
        
        # Get fallbacks for type - should be sorted by priority
        fallbacks = registry.get_fallbacks_for_type(StrategyType.TRAVERSAL)
        assert len(fallbacks) == 3
        assert fallbacks[0] == MockStrategyB  # Priority 10 (lowest, highest priority)
        assert fallbacks[1] == MockStrategyC  # Priority 20
        assert fallbacks[2] == MockStrategyA  # Priority 30 (highest, lowest priority)
        
        # Get fallbacks for non-registered type
        fallbacks = registry.get_fallbacks_for_type(StrategyType.SPECIAL_PURPOSE)
        assert len(fallbacks) == 0
    
    def test_create_fallback_chain(self, strategy_context):
        """Test creating a fallback chain for a strategy."""
        registry = FallbackRegistry()
        
        # Register mock fallback strategies
        registry.register_fallback(StrategyType.TRAVERSAL, MockSuccessStrategy, priority=10)
        registry.register_fallback(StrategyType.TRAVERSAL, MockPartialStrategy, priority=20)
        
        # Create a mock primary strategy with metadata
        primary = MagicMock(spec=BaseStrategy)
        primary.name = "mock_primary"
        primary._metadata = MagicMock()
        primary._metadata.strategy_type = StrategyType.TRAVERSAL
        
        # Mock get_fallbacks_for_type to return our mock classes
        registry.get_fallbacks_for_type = MagicMock(return_value=[
            MockSuccessStrategy, MockPartialStrategy
        ])
        
        # Create fallback chain
        chain = registry.create_fallback_chain(primary, strategy_context)
        
        # Verify chain creation
        assert isinstance(chain, FallbackStrategyChain)
        assert chain.primary_strategy == primary
        assert len(chain.fallback_strategies) == 2
        assert isinstance(chain.fallback_strategies[0], MockSuccessStrategy)
        assert isinstance(chain.fallback_strategies[1], MockPartialStrategy)
    
    def test_suggest_fallbacks(self):
        """Test suggesting fallbacks based on error."""
        registry = FallbackRegistry()
        
        # Define mock strategy classes
        MockStrategyA = MagicMock()
        MockStrategyA.__name__ = "MockStrategyA"
        MockStrategyB = MagicMock()
        MockStrategyB.__name__ = "MockStrategyB"
        MockStrategyC = MagicMock()
        MockStrategyC.__name__ = "MockStrategyC"
        
        # Register fallbacks for different error categories
        registry.register_fallback(
            StrategyType.TRAVERSAL, 
            MockStrategyA,
            error_categories=["network"]
        )
        registry.register_fallback(
            StrategyType.TRAVERSAL, 
            MockStrategyB,
            error_categories=["timeout"]
        )
        registry.register_fallback(
            StrategyType.EXTRACTION, 
            MockStrategyC,
            error_categories=["network"]
        )
        
        # Create a mock primary strategy with metadata
        primary = MagicMock(spec=BaseStrategy)
        primary.name = "mock_primary"
        primary._metadata = MagicMock()
        primary._metadata.strategy_type = StrategyType.TRAVERSAL
        
        # Get suggestions for network error
        error = {"category": "network", "severity": "error"}
        suggestions = registry.suggest_fallbacks(primary, error)
        
        # Should suggest MockStrategyA (matches category) and MockStrategyB (matches type)
        assert len(suggestions) == 2
        assert MockStrategyA in suggestions
        assert MockStrategyB in suggestions
        
        # Get suggestions for EXTRACTION strategy with network error
        primary._metadata.strategy_type = StrategyType.EXTRACTION
        suggestions = registry.suggest_fallbacks(primary, error)
        
        # Should suggest MockStrategyC (matches category and type)
        assert len(suggestions) == 1
        assert MockStrategyC in suggestions


# Test Recovery Mechanisms
class TestRecoveryMechanisms:
    """Tests for recovery utility functions."""
    
    def test_merge_strategy_results(self):
        """Test merging results from multiple strategies."""
        # Simple merge case
        results = [
            {"field1": "value1", "field2": "value2"},
            {"field2": "value2", "field3": "value3"}
        ]
        
        merged = merge_strategy_results(results)
        assert merged["field1"] == "value1"
        assert merged["field2"] == "value2"
        assert merged["field3"] == "value3"
        
        # Merge with different value types
        results = [
            {"field1": "value1", "nested": {"a": 1, "b": 2}},
            {"field2": "value2", "nested": {"b": 2, "c": 3}},
            {"list_field": [1, 2, 3]},
            {"list_field": [3, 4, 5]}
        ]
        
        merged = merge_strategy_results(results)
        assert merged["field1"] == "value1"
        assert merged["field2"] == "value2"
        
        # Nested dictionaries should be recursively merged
        assert merged["nested"]["a"] == 1
        assert merged["nested"]["b"] == 2
        assert merged["nested"]["c"] == 3
        
        # Lists should be combined and deduplicated
        assert set(merged["list_field"]) == {1, 2, 3, 4, 5}
        
        # Empty list should raise ValueError
        with pytest.raises(ValueError):
            merge_strategy_results([])
        
        # Single item should be returned as is
        result = {"field": "value"}
        assert merge_strategy_results([result]) == result
    
    def test_extract_best_components(self):
        """Test extracting best components from multiple results."""
        # Create mock assessor that gives higher scores to longer values
        with patch('strategies.fallback_framework.StrategyResultQualityAssessor') as MockAssessor:
            mock_instance = MagicMock()
            MockAssessor.return_value = mock_instance
            
            # Mock get_quality_score to prefer results with "better" in the values
            def mock_score(result):
                if not isinstance(result, dict):
                    return 0.5
                return 0.8 if any("better" in str(v) for v in result.values()) else 0.4
            
            mock_instance.get_quality_score.side_effect = mock_score
            
            # Test with mixed quality results
            results = [
                {"field1": "value1", "field2": "value2", "field3": "short"},
                {"field1": "better1", "field2": "value2_different", "field4": "unique"},
                {"field3": "better and longer"}
            ]
            
            best = extract_best_components(results)
            
            # Should take best version of each field
            assert best["field1"] == "better1"  # From second result (higher quality)
            assert "field2" in best  # Either version is fine
            assert best["field3"] == "better and longer"  # From third result (higher quality)
            assert best["field4"] == "unique"  # Only in second result
            
            # Empty list should return empty dict
            assert extract_best_components([]) == {}
            
            # Single item should be returned as is
            result = {"field": "value"}
            assert extract_best_components([result]) == result
    
    def test_rebuild_from_fragments(self):
        """Test rebuilding result from fragments."""
        # Create results with different fields
        results = [
            {
                "title": "Product A",
                "price": "$10.99",
                "confidence": {"title": 0.9, "price": 0.7}
            },
            {
                "description": "A great product",
                "features": ["Feature 1", "Feature 2"],
                "confidence": {"description": 0.8, "features": 0.6}
            },
            {
                "price": "$10.95",  # Conflicting price with lower confidence
                "stock": "In Stock",
                "confidence": {"price": 0.5, "stock": 0.9}
            }
        ]
        
        # Patch assess_completeness to return predetermined scores
        with patch('strategies.fallback_framework.StrategyResultQualityAssessor.assess_completeness') as mock_assess:
            mock_assess.side_effect = [0.7, 0.6, 0.5]  # First result is most complete
            
            rebuilt = rebuild_from_fragments(results)
            
            # Should use first result as base
            assert rebuilt["title"] == "Product A"
            assert rebuilt["price"] == "$10.99"  # Higher confidence from first result
            
            # Should add missing fields from other results
            assert rebuilt["description"] == "A great product"
            assert rebuilt["features"] == ["Feature 1", "Feature 2"]
            assert rebuilt["stock"] == "In Stock"
            
            # Should maintain confidence information
            assert rebuilt["confidence"]["title"] == 0.9
            assert rebuilt["confidence"]["price"] == 0.7
            assert rebuilt["confidence"]["description"] == 0.8
            assert rebuilt["confidence"]["features"] == 0.6
            assert rebuilt["confidence"]["stock"] == 0.9
            
            # Empty list should return empty dict
            assert rebuild_from_fragments([]) == {}
            
            # Single item should be returned as is
            result = {"field": "value"}
            assert rebuild_from_fragments([result]) == result
    
    def test_estimate_result_quality(self):
        """Test estimating result quality."""
        with patch('strategies.fallback_framework.StrategyResultQualityAssessor') as MockAssessor:
            mock_instance = MagicMock()
            MockAssessor.return_value = mock_instance
            mock_instance.get_quality_score.return_value = 0.75
            
            result = {"field": "value"}
            quality = estimate_result_quality(result, required_fields=["field"])
            
            assert quality == 0.75
            
            # Should create assessor with required fields
            MockAssessor.assert_called_once()
            call_args = MockAssessor.call_args[1]
            assert call_args["required_fields"] == ["field"]


# Test Create Fallback Condition Helper
class TestCreateFallbackCondition:
    """Tests for the create_fallback_condition helper function."""
    
    def test_create_simple_condition(self):
        """Test creating a simple condition."""
        # Create error-based condition
        condition = create_fallback_condition(
            error_categories={"network", "timeout"},
            error_severities={"error", "critical"}
        )
        
        assert isinstance(condition, ErrorBasedCondition)
        assert condition.error_categories == {"network", "timeout"}
        assert condition.error_severities == {"error", "critical"}
        
        # Create attempt-based condition
        condition = create_fallback_condition(max_attempts=5)
        
        assert isinstance(condition, AttemptBasedCondition)
        assert condition.max_attempts == 5
        
        # Create timeout condition
        condition = create_fallback_condition(timeout_seconds=10.0)
        
        assert isinstance(condition, TimeoutCondition)
        assert condition.timeout_seconds == 10.0
        
        # Create quality condition
        condition = create_fallback_condition(min_quality_score=0.7)
        
        assert isinstance(condition, ResultQualityCondition)
        assert condition.min_quality_score == 0.7
    
    def test_create_composite_condition(self):
        """Test creating a composite condition."""
        # Create composite with OR logic (default)
        condition = create_fallback_condition(
            error_severities={"error"},
            max_attempts=3
        )
        
        assert isinstance(condition, CompositeCondition)
        assert condition.logic == CompositeCondition.Logic.OR
        assert len(condition.conditions) == 2
        assert any(isinstance(c, ErrorBasedCondition) for c in condition.conditions)
        assert any(isinstance(c, AttemptBasedCondition) for c in condition.conditions)
        
        # Create composite with AND logic
        condition = create_fallback_condition(
            error_severities={"error"},
            max_attempts=3,
            logic=CompositeCondition.Logic.AND
        )
        
        assert isinstance(condition, CompositeCondition)
        assert condition.logic == CompositeCondition.Logic.AND
        
        # No parameters should use default
        condition = create_fallback_condition()
        
        assert isinstance(condition, CompositeCondition)
        assert len(condition.conditions) == 2