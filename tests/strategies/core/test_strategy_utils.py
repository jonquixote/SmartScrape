"""
Tests for strategy execution utilities.

These tests verify the functionality of:
- retry_on_failure decorator
- with_timeout decorator
- measure_performance decorator
- log_execution decorator
- require_service decorator
"""

import pytest
import time
import logging
import threading
from unittest.mock import MagicMock, patch
import os
import psutil
from typing import Dict, Any, Optional, List

from strategies.core.strategy_utils import (
    retry_on_failure,
    with_timeout,
    measure_performance,
    log_execution,
    require_service,
    _truncate_value
)

# Test fixture for getting a logger that we can inspect
@pytest.fixture
def mock_logger():
    logger = MagicMock()
    with patch('strategies.core.strategy_utils.logger', logger):
        yield logger


# Mock class to simulate a strategy with a name
class MockStrategy:
    def __init__(self, name="mock_strategy"):
        self.name = name
        self.metrics = {}
    
    def add_metric(self, name, value):
        self.metrics[name] = value


# ========= Test retry_on_failure decorator =========

# Mock functions for testing retry behavior
class MockFailingFunction:
    def __init__(self, fail_count=2, exception_cls=ValueError):
        self.fail_count = fail_count
        self.calls = 0
        self.exception_cls = exception_cls
    
    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise self.exception_cls(f"Call {self.calls} failed")
        return f"Success on call {self.calls}"


def test_retry_on_failure_success(mock_logger):
    """Test retry_on_failure when function eventually succeeds."""
    # Function fails twice, then succeeds on third call
    mock_func = MockFailingFunction(fail_count=2)
    
    # Decorate the function
    @retry_on_failure(max_attempts=4, delay_seconds=0.01, backoff_factor=1)
    def test_function(*args, **kwargs):
        return mock_func(*args, **kwargs)
    
    # Call the function
    result = test_function("arg1", key="value")
    
    # Check the function was called the correct number of times
    assert mock_func.calls == 3
    assert result == "Success on call 3"
    
    # Check logging - should have logged the failures and success
    assert mock_logger.log.call_count >= 2
    
    # Check it preserves function metadata
    assert test_function.__name__ == "test_function"


def test_retry_on_failure_with_strategy_object(mock_logger):
    """Test retry_on_failure when called from a strategy object."""
    mock_func = MockFailingFunction(fail_count=1)
    strategy = MockStrategy("test_strategy")
    
    @retry_on_failure(max_attempts=3, delay_seconds=0.01)
    def strategy_method(self, param):
        return mock_func(param)
    
    # Call the method with the strategy object as self
    result = strategy_method(strategy, "test_param")
    
    assert mock_func.calls == 2
    assert result == "Success on call 2"
    
    # Verify the strategy name is used in logs
    assert any("test_strategy" in str(args) for args, _ in mock_logger.log.call_args_list)


def test_retry_on_failure_max_attempts_reached(mock_logger):
    """Test retry_on_failure when max attempts are reached without success."""
    # Function always fails
    mock_func = MockFailingFunction(fail_count=10)
    
    @retry_on_failure(max_attempts=3, delay_seconds=0.01)
    def always_fails():
        return mock_func()
    
    # Should raise the exception after reaching max attempts
    with pytest.raises(ValueError, match="Call 3 failed"):
        always_fails()
    
    # Check the function was called exactly max_attempts times
    assert mock_func.calls == 3
    
    # Verify logging
    assert mock_logger.log.call_count >= 3  # At least 3 log calls (2 failures + final message)


def test_retry_on_failure_specific_exceptions():
    """Test retry_on_failure with specific exception types."""
    # Function raises ValueError first, then TypeError
    class MixedExceptionFunction:
        def __init__(self):
            self.calls = 0
        
        def __call__(self):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("Value error")
            if self.calls == 2:
                raise TypeError("Type error")
            return "Success"
    
    mock_func = MixedExceptionFunction()
    
    # Retry only on ValueError
    @retry_on_failure(max_attempts=3, delay_seconds=0.01, exceptions=ValueError)
    def test_function():
        return mock_func()
    
    # Should retry after ValueError but not catch TypeError
    with pytest.raises(TypeError, match="Type error"):
        test_function()
    
    assert mock_func.calls == 2  # Called twice (original + 1 retry)


def test_retry_on_failure_backoff(mock_logger):
    """Test that retry_on_failure implements exponential backoff."""
    mock_func = MockFailingFunction(fail_count=2)
    
    with patch('time.sleep') as mock_sleep:
        @retry_on_failure(max_attempts=3, delay_seconds=0.1, backoff_factor=2)
        def test_function():
            return mock_func()
        
        test_function()
        
        # Should have called sleep twice with increasing delays
        assert mock_sleep.call_count == 2
        
        # First delay should be 0.1, second should be 0.1 * 2 = 0.2
        delays = [args[0] for args, _ in mock_sleep.call_args_list]
        assert delays[0] == 0.1
        assert delays[1] == 0.2


# ========= Test with_timeout decorator =========

def test_with_timeout_completes_in_time():
    """Test with_timeout when function completes within the timeout."""
    @with_timeout(timeout_seconds=1.0)
    def quick_function():
        time.sleep(0.1)
        return "done"
    
    result = quick_function()
    assert result == "done"


def test_with_timeout_exceeds_timeout():
    """Test with_timeout when function exceeds the timeout."""
    # Define a function that sleeps longer than the timeout
    @with_timeout(timeout_seconds=0.1)
    def slow_function():
        time.sleep(0.5)  # Longer than timeout
        return "done"
    
    # Should raise TimeoutError
    with pytest.raises(TimeoutError):
        slow_function()


def test_with_timeout_with_strategy_object(mock_logger):
    """Test with_timeout when used with a strategy object."""
    strategy = MockStrategy("timeout_strategy")
    
    @with_timeout(timeout_seconds=0.2)
    def strategy_method(self):
        time.sleep(0.1)
        return "quick result"
    
    result = strategy_method(strategy)
    assert result == "quick result"
    
    # Check that strategy name appears in logs if timeout occurs
    @with_timeout(timeout_seconds=0.1)
    def slow_strategy_method(self):
        time.sleep(0.5)
        return "slow result"
    
    with pytest.raises(TimeoutError):
        slow_strategy_method(strategy)
    
    # Verify the strategy name is in the error message
    error_logs = [args for args, _ in mock_logger.error.call_args_list]
    assert any("timeout_strategy" in str(args) for args in error_logs)


def test_with_timeout_cleanup_callback():
    """Test that cleanup callback is called when timeout occurs."""
    cleanup_mock = MagicMock()
    
    @with_timeout(timeout_seconds=0.1, cleanup_callback=cleanup_mock)
    def slow_with_cleanup():
        time.sleep(0.5)
        return "done"
    
    with pytest.raises(TimeoutError):
        slow_with_cleanup()
    
    # Verify cleanup was called
    cleanup_mock.assert_called_once()


def test_with_timeout_propagates_exceptions():
    """Test that with_timeout propagates exceptions from the function."""
    @with_timeout(timeout_seconds=1.0)
    def function_with_error():
        raise ValueError("Expected error")
    
    with pytest.raises(ValueError, match="Expected error"):
        function_with_error()


# ========= Test measure_performance decorator =========

def test_measure_performance_basic(mock_logger):
    """Test basic functionality of measure_performance decorator."""
    @measure_performance
    def simple_function():
        time.sleep(0.1)
        return "result"
    
    result = simple_function()
    assert result == "result"
    
    # Verify performance metrics were logged
    assert mock_logger.log.call_count >= 1
    
    # Check the log contains performance data
    log_calls = [args[1] for args, _ in mock_logger.log.call_args_list]
    assert any("Performance:" in str(msg) for msg in log_calls)
    assert any("simple_function succeeded in" in str(msg) for msg in log_calls)
    assert any("memory:" in str(msg) for msg in log_calls)


def test_measure_performance_with_strategy(mock_logger):
    """Test measure_performance decorator with a strategy object."""
    strategy = MockStrategy("perf_strategy")
    
    @measure_performance
    def strategy_method(self):
        time.sleep(0.1)
        return "result"
    
    result = strategy_method(strategy)
    assert result == "result"
    
    # Verify the strategy name is included in logs
    log_calls = [args[1] for args, _ in mock_logger.log.call_args_list]
    assert any("perf_strategy" in str(msg) for msg in log_calls)


def test_measure_performance_with_metrics():
    """Test measure_performance adds metrics to strategy if supported."""
    strategy = MockStrategy("metrics_strategy")
    
    @measure_performance
    def strategy_method(self):
        time.sleep(0.1)
        return "result"
    
    strategy_method(strategy)
    
    # Check that metrics were added to the strategy
    assert "strategy_method_execution_time" in strategy.metrics
    assert "strategy_method_memory_delta" in strategy.metrics
    assert strategy.metrics["strategy_method_execution_time"] >= 0.1


def test_measure_performance_with_exception(mock_logger):
    """Test measure_performance handles exceptions correctly."""
    @measure_performance
    def failing_function():
        time.sleep(0.1)
        raise ValueError("Expected error")
    
    with pytest.raises(ValueError, match="Expected error"):
        failing_function()
    
    # Verify the failure was logged
    log_calls = [args[1] for args, _ in mock_logger.log.call_args_list]
    assert any("failing_function failed in" in str(msg) for msg in log_calls)


# ========= Test log_execution decorator =========

def test_log_execution_basic(mock_logger):
    """Test basic functionality of log_execution decorator."""
    @log_execution()
    def simple_function(a, b):
        return a + b
    
    result = simple_function(1, 2)
    assert result == 3
    
    # Verify execution was logged
    assert mock_logger.log.call_count >= 2  # Start and end logs
    
    # Check the logs contain the function name and args
    log_messages = [args[1] for args, _ in mock_logger.log.call_args_list]
    assert any("Executing" in str(msg) and "simple_function" in str(msg) for msg in log_messages)
    assert any("Completed" in str(msg) and "simple_function" in str(msg) for msg in log_messages)


def test_log_execution_with_strategy(mock_logger):
    """Test log_execution decorator with a strategy object."""
    strategy = MockStrategy("log_strategy")
    
    @log_execution()
    def strategy_method(self, a, b):
        return a + b
    
    result = strategy_method(strategy, 1, 2)
    assert result == 3
    
    # Verify the strategy name is included in logs
    log_messages = [args[1] for args, _ in mock_logger.log.call_args_list]
    assert any("log_strategy" in str(msg) for msg in log_messages)


def test_log_execution_with_sensitive_params(mock_logger):
    """Test log_execution masks sensitive parameters."""
    @log_execution(sensitive_params={'password', 'token', 'custom_sensitive'})
    def function_with_sensitive_params(user, password, token, custom_sensitive):
        return "success"
    
    function_with_sensitive_params("user123", "secret_pass", "secret_token", "sensitive_data")
    
    # Check that sensitive parameters were masked
    log_messages = [args[1] for args, _ in mock_logger.log.call_args_list]
    assert any("password" in str(msg) and "***MASKED***" in str(msg) for msg in log_messages)
    assert any("token" in str(msg) and "***MASKED***" in str(msg) for msg in log_messages)
    assert any("custom_sensitive" in str(msg) and "***MASKED***" in str(msg) for msg in log_messages)
    assert any("user123" in str(msg) for msg in log_messages)  # Non-sensitive param should be visible


def test_log_execution_without_args(mock_logger):
    """Test log_execution with log_args=False."""
    @log_execution(log_args=False)
    def function_without_args_logging(a, b):
        return a + b
    
    function_without_args_logging(1, 2)
    
    # Check that args are not in the log
    log_messages = [args[1] for args, _ in mock_logger.log.call_args_list]
    executing_logs = [msg for msg in log_messages if "Executing" in str(msg)]
    assert executing_logs
    assert all("args" not in str(msg) for msg in executing_logs)


def test_log_execution_with_result(mock_logger):
    """Test log_execution with log_result=True."""
    @log_execution(log_result=True)
    def function_with_result_logging(a, b):
        return a + b
    
    function_with_result_logging(1, 2)
    
    # Check that result is in the log
    log_messages = [args[1] for args, _ in mock_logger.log.call_args_list]
    completed_logs = [msg for msg in log_messages if "Completed" in str(msg)]
    assert any("with result: 3" in str(msg) for msg in completed_logs)


def test_log_execution_with_exception(mock_logger):
    """Test log_execution handles exceptions correctly."""
    @log_execution()
    def failing_function():
        raise ValueError("Expected error")
    
    with pytest.raises(ValueError, match="Expected error"):
        failing_function()
    
    # Verify the error was logged
    assert any("Error in" in str(args[1]) and "Expected error" in str(args[1]) 
               for args, _ in mock_logger.log.call_args_list)


# ========= Test require_service decorator =========

class MockServiceContext:
    """Mock context for testing require_service decorator."""
    def __init__(self):
        self.services = {
            "url_service": MagicMock(),
            "html_service": MagicMock()
        }
    
    def get_service(self, service_name):
        if service_name in self.services:
            return self.services[service_name]
        raise KeyError(f"Service '{service_name}' not found")


class StrategyWithServices:
    """Mock strategy class for testing require_service decorator."""
    def __init__(self, context=None):
        self.context = context
        self.name = "service_strategy"
    
    @require_service("url_service", "html_service")
    def method_requiring_services(self, arg):
        return f"Using services with {arg}"
    
    @require_service("url_service", "non_existent_service")
    def method_requiring_missing_service(self):
        return "This should not execute"


def test_require_service_with_available_services():
    """Test require_service when all required services are available."""
    context = MockServiceContext()
    strategy = StrategyWithServices(context)
    
    result = strategy.method_requiring_services("test")
    assert result == "Using services with test"


def test_require_service_with_missing_service():
    """Test require_service when a required service is missing."""
    context = MockServiceContext()
    strategy = StrategyWithServices(context)
    
    with pytest.raises(ValueError) as excinfo:
        strategy.method_requiring_missing_service()
    
    # Check the error message
    assert "Required services missing" in str(excinfo.value)
    assert "non_existent_service" in str(excinfo.value)


def test_require_service_without_context():
    """Test require_service when context is missing."""
    strategy = StrategyWithServices(None)  # No context
    
    with pytest.raises(ValueError) as excinfo:
        strategy.method_requiring_services("test")
    
    assert "requires a context" in str(excinfo.value)


# ========= Test helper functions =========

def test_truncate_value():
    """Test the _truncate_value helper function."""
    # Test string truncation
    long_string = "a" * 1000
    truncated = _truncate_value(long_string)
    assert len(truncated) < 1000
    assert truncated.endswith("... [truncated]")
    
    # Test no truncation for short strings
    short_string = "short"
    assert _truncate_value(short_string) == short_string
    
    # Test list truncation
    long_list = list(range(20))
    truncated = _truncate_value(long_list)
    assert "10 more items" in str(truncated)
    
    # Test dictionary truncation
    long_dict = {i: i for i in range(20)}
    truncated = _truncate_value(long_dict)
    assert "10 more items" in str(truncated)