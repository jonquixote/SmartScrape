"""
Tests for the strategy error handler module.

This module tests:
1. Error category and severity enums
2. StrategyError class
3. StrategyErrorHandlingMixin class
4. Error handling utility functions
"""

import pytest
import time
import datetime
from unittest.mock import MagicMock, patch

from strategies.core.strategy_error_handler import (
    StrategyErrorCategory,
    StrategyErrorSeverity,
    StrategyError,
    ErrorPolicy,
    StrategyErrorHandlingMixin,
    log_error,
    retry_on_network_error,
    wait_on_rate_limit
)


class TestStrategyErrorCategory:
    """Tests for StrategyErrorCategory enum."""

    def test_error_category_values(self):
        """Test that the error category enum has the expected values."""
        assert StrategyErrorCategory.NETWORK.value == "network"
        assert StrategyErrorCategory.PARSING.value == "parsing"
        assert StrategyErrorCategory.EXECUTION.value == "execution"
        assert StrategyErrorCategory.VALIDATION.value == "validation"
        assert StrategyErrorCategory.RESOURCE.value == "resource"


class TestStrategyErrorSeverity:
    """Tests for StrategyErrorSeverity enum."""

    def test_error_severity_values(self):
        """Test that the error severity enum has the expected values."""
        assert StrategyErrorSeverity.INFO.value == "info"
        assert StrategyErrorSeverity.WARNING.value == "warning"
        assert StrategyErrorSeverity.ERROR.value == "error"
        assert StrategyErrorSeverity.CRITICAL.value == "critical"


class TestStrategyError:
    """Tests for StrategyError class."""

    def test_error_initialization(self):
        """Test initializing a strategy error."""
        error = StrategyError(
            message="Test error message",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            strategy_name="test_strategy",
            url="http://example.com"
        )

        assert error.message == "Test error message"
        assert error.category == StrategyErrorCategory.NETWORK
        assert error.severity == StrategyErrorSeverity.ERROR
        assert error.strategy_name == "test_strategy"
        assert error.url == "http://example.com"
        assert error.exception is None
        assert error.retry_count == 0
        assert hasattr(error, "timestamp")
        assert hasattr(error, "error_id")

    def test_error_initialization_with_exception(self):
        """Test initializing a strategy error with an exception."""
        try:
            raise ValueError("Test value error")
        except ValueError as e:
            error = StrategyError(
                message="Error occurred",
                category=StrategyErrorCategory.EXECUTION,
                severity=StrategyErrorSeverity.ERROR,
                strategy_name="test_strategy",
                exception=e
            )

        assert error.message == "Error occurred"
        assert error.exception_type == "ValueError"
        assert error.exception_message == "Test value error"
        assert error.traceback is not None
        assert "Traceback" in error.traceback

    def test_error_with_string_category_and_severity(self):
        """Test initializing a strategy error with string category and severity."""
        error = StrategyError(
            message="Test error",
            category="network",
            severity="warning",
            strategy_name="test_strategy"
        )

        assert error.category == StrategyErrorCategory.NETWORK
        assert error.severity == StrategyErrorSeverity.WARNING

    def test_error_with_invalid_string_category_and_severity(self):
        """Test initializing a strategy error with invalid string category and severity."""
        error = StrategyError(
            message="Test error",
            category="nonexistent_category",
            severity="nonexistent_severity",
            strategy_name="test_strategy"
        )

        assert error.category == StrategyErrorCategory.UNEXPECTED
        assert error.severity == StrategyErrorSeverity.WARNING

    def test_error_string_representation(self):
        """Test string representation of an error."""
        error = StrategyError(
            message="Test error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            strategy_name="test_strategy",
            url="http://example.com"
        )

        error_str = str(error)
        assert "ERROR [network]" in error_str
        assert "Test error" in error_str
        assert "URL: http://example.com" in error_str

    def test_error_to_dict(self):
        """Test converting an error to a dictionary."""
        error = StrategyError(
            message="Test error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            strategy_name="test_strategy",
            url="http://example.com",
            context_data={"request_id": "12345"}
        )

        error_dict = error.to_dict()
        assert error_dict["message"] == "Test error"
        assert error_dict["category"] == "network"
        assert error_dict["severity"] == "error"
        assert error_dict["strategy_name"] == "test_strategy"
        assert error_dict["url"] == "http://example.com"
        assert error_dict["context_data"] == {"request_id": "12345"}
        assert error_dict["recoverable"] is True
        assert error_dict["retry_count"] == 0

    def test_increment_retry(self):
        """Test incrementing the retry count."""
        error = StrategyError(
            message="Test error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            strategy_name="test_strategy"
        )

        assert error.retry_count == 0
        error.increment_retry()
        assert error.retry_count == 1
        error.increment_retry()
        assert error.retry_count == 2


class TestErrorPolicy:
    """Tests for ErrorPolicy class."""

    def test_policy_initialization(self):
        """Test initializing an error policy."""
        categories = {StrategyErrorCategory.NETWORK, StrategyErrorCategory.RESOURCE}
        severities = {StrategyErrorSeverity.WARNING, StrategyErrorSeverity.ERROR}
        handler = lambda error: True
        
        policy = ErrorPolicy(
            categories=categories,
            severities=severities,
            handler=handler,
            max_retries=5,
            log_level=20  # INFO
        )
        
        assert policy.categories == categories
        assert policy.severities == severities
        assert policy.handler == handler
        assert policy.max_retries == 5
        assert policy.log_level == 20

    def test_policy_matches(self):
        """Test policy matching logic."""
        policy = ErrorPolicy(
            categories={StrategyErrorCategory.NETWORK},
            severities={StrategyErrorSeverity.ERROR}
        )
        
        # Matching error
        error1 = StrategyError(
            message="Network error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            strategy_name="test_strategy"
        )
        
        # Non-matching category
        error2 = StrategyError(
            message="Parser error",
            category=StrategyErrorCategory.PARSING,
            severity=StrategyErrorSeverity.ERROR,
            strategy_name="test_strategy"
        )
        
        # Non-matching severity
        error3 = StrategyError(
            message="Network warning",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.WARNING,
            strategy_name="test_strategy"
        )
        
        assert policy.matches(error1) is True
        assert policy.matches(error2) is False
        assert policy.matches(error3) is False
        
        # Empty categories and severities should match any
        policy2 = ErrorPolicy()
        assert policy2.matches(error1) is True
        assert policy2.matches(error2) is True
        assert policy2.matches(error3) is True

    def test_policy_handle(self, caplog):
        """Test policy handle method."""
        # Create a mock handler
        handler = MagicMock(return_value=True)
        
        # Create a policy with the mock handler
        policy = ErrorPolicy(
            handler=handler,
            log_level=20  # INFO
        )
        
        # Create an error
        error = StrategyError(
            message="Test error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            strategy_name="test_strategy"
        )
        
        # Handle the error
        with caplog.at_level(20):  # INFO level
            result = policy.handle(error)
            
        # Check that the handler was called with the error
        handler.assert_called_once_with(error)
        
        # Check that handling was successful
        assert result is True
        
        # Check that the error was logged
        assert "ERROR [network]" in caplog.text
        assert "Test error" in caplog.text

    def test_policy_handle_no_handler(self, caplog):
        """Test policy handle method with no handler."""
        # Create a policy with no handler
        policy = ErrorPolicy(
            log_level=20  # INFO
        )
        
        # Create an error
        error = StrategyError(
            message="Test error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            strategy_name="test_strategy"
        )
        
        # Handle the error
        with caplog.at_level(20):  # INFO level
            result = policy.handle(error)
            
        # Check that handling was successful
        assert result is True
        
        # Check that the error was logged
        assert "ERROR [network]" in caplog.text
        assert "Test error" in caplog.text

    def test_policy_handle_handler_error(self, caplog):
        """Test policy handle method when handler raises an exception."""
        # Create a handler that raises an exception
        def failing_handler(error):
            raise ValueError("Handler error")
        
        # Create a policy with the failing handler
        policy = ErrorPolicy(
            handler=failing_handler,
            log_level=20  # INFO
        )
        
        # Create an error
        error = StrategyError(
            message="Test error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            strategy_name="test_strategy"
        )
        
        # Handle the error
        with caplog.at_level(20):  # INFO level
            result = policy.handle(error)
            
        # Check that handling failed
        assert result is False
        
        # Check that the error was logged
        assert "ERROR [network]" in caplog.text
        assert "Test error" in caplog.text
        assert "Error in error handler: Handler error" in caplog.text


class TestStrategyClass(StrategyErrorHandlingMixin):
    """Test class that uses StrategyErrorHandlingMixin."""
    
    def __init__(self):
        """Initialize the test class."""
        super().__init__()
        self.name = "test_strategy"


class TestStrategyErrorHandlingMixin:
    """Tests for StrategyErrorHandlingMixin class."""

    def test_mixin_initialization(self):
        """Test initializing the error handling mixin."""
        strategy = TestStrategyClass()
        
        assert strategy._errors == []
        assert len(strategy._error_policies) > 0  # Should have some default policies
        assert strategy._default_policy is not None

    def test_create_error(self):
        """Test creating an error through the mixin."""
        strategy = TestStrategyClass()
        
        error = strategy.create_error(
            message="Test error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            url="http://example.com"
        )
        
        assert error in strategy._errors
        assert error.message == "Test error"
        assert error.category == StrategyErrorCategory.NETWORK
        assert error.severity == StrategyErrorSeverity.ERROR
        assert error.strategy_name == "test_strategy"
        assert error.url == "http://example.com"

    def test_handle_error_with_exception(self):
        """Test handling an exception through the mixin."""
        strategy = TestStrategyClass()
        
        # Create an exception
        exception = ValueError("Test exception")
        
        # Handle the exception
        result = strategy.handle_error(
            error=exception,
            category=StrategyErrorCategory.PARSING,
            severity=StrategyErrorSeverity.WARNING,
            url="http://example.com"
        )
        
        # Check that handling was successful
        assert result is True
        
        # Check that the error was added to the list
        assert len(strategy._errors) == 1
        error = strategy._errors[0]
        assert error.message == "Test exception"
        assert error.category == StrategyErrorCategory.PARSING
        assert error.severity == StrategyErrorSeverity.WARNING
        assert error.url == "http://example.com"
        assert error.exception == exception

    def test_handle_error_with_strategy_error(self):
        """Test handling a StrategyError through the mixin."""
        strategy = TestStrategyClass()
        
        # Create a StrategyError
        error = StrategyError(
            message="Test error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            strategy_name="test_strategy"
        )
        
        # Handle the error
        result = strategy.handle_error(error)
        
        # Check that handling was successful
        assert result is True
        
        # The error should have been added to the list
        assert len(strategy._errors) == 1
        assert strategy._errors[0] == error

    def test_handle_error_with_message_only(self):
        """Test handling an error with only a message through the mixin."""
        strategy = TestStrategyClass()
        
        # Handle an error with just a message
        result = strategy.handle_error(
            error=None,
            message="Test message only error",
            category=StrategyErrorCategory.UNEXPECTED,
            severity=StrategyErrorSeverity.INFO
        )
        
        # Check that handling was successful
        assert result is True
        
        # Check that the error was added to the list
        assert len(strategy._errors) == 1
        error = strategy._errors[0]
        assert error.message == "Test message only error"
        assert error.category == StrategyErrorCategory.UNEXPECTED
        assert error.severity == StrategyErrorSeverity.INFO

    def test_get_errors(self):
        """Test getting errors with various filters."""
        strategy = TestStrategyClass()
        
        # Create some errors
        error1 = strategy.create_error(
            message="Network error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            recoverable=True
        )
        
        error2 = strategy.create_error(
            message="Parsing warning",
            category=StrategyErrorCategory.PARSING,
            severity=StrategyErrorSeverity.WARNING,
            recoverable=True
        )
        
        error3 = strategy.create_error(
            message="Resource critical",
            category=StrategyErrorCategory.RESOURCE,
            severity=StrategyErrorSeverity.CRITICAL,
            recoverable=False
        )
        
        # Get all errors
        all_errors = strategy.get_errors()
        assert len(all_errors) == 3
        assert error1 in all_errors
        assert error2 in all_errors
        assert error3 in all_errors
        
        # Filter by category
        network_errors = strategy.get_errors(
            categories={StrategyErrorCategory.NETWORK}
        )
        assert len(network_errors) == 1
        assert error1 in network_errors
        
        # Filter by severity
        warning_errors = strategy.get_errors(
            severities={StrategyErrorSeverity.WARNING}
        )
        assert len(warning_errors) == 1
        assert error2 in warning_errors
        
        # Filter by recoverability
        recoverable_errors = strategy.get_errors(
            recoverable_only=True
        )
        assert len(recoverable_errors) == 2
        assert error1 in recoverable_errors
        assert error2 in recoverable_errors
        assert error3 not in recoverable_errors
        
        # Filter by category and severity
        network_error_errors = strategy.get_errors(
            categories={StrategyErrorCategory.NETWORK},
            severities={StrategyErrorSeverity.ERROR}
        )
        assert len(network_error_errors) == 1
        assert error1 in network_error_errors
        
        # Filter with string values
        string_category_errors = strategy.get_errors(
            categories={"network", "resource"}
        )
        assert len(string_category_errors) == 2
        assert error1 in string_category_errors
        assert error3 in string_category_errors
        
        # Filter with string values
        string_severity_errors = strategy.get_errors(
            severities={"warning", "critical"}
        )
        assert len(string_severity_errors) == 2
        assert error2 in string_severity_errors
        assert error3 in string_severity_errors

    def test_has_errors(self):
        """Test checking if there are errors with at least a minimum severity."""
        strategy = TestStrategyClass()
        
        # Initially there should be no errors
        assert strategy.has_errors() is False
        
        # Create errors with different severities
        strategy.create_error(
            message="Info message",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.INFO
        )
        
        # Check for warnings or higher
        assert strategy.has_errors(StrategyErrorSeverity.WARNING) is False
        
        strategy.create_error(
            message="Warning message",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.WARNING
        )
        
        # Check for warnings or higher
        assert strategy.has_errors(StrategyErrorSeverity.WARNING) is True
        # Check for errors or higher
        assert strategy.has_errors(StrategyErrorSeverity.ERROR) is False
        
        strategy.create_error(
            message="Error message",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR
        )
        
        # Check for errors or higher
        assert strategy.has_errors(StrategyErrorSeverity.ERROR) is True
        # Check for critical
        assert strategy.has_errors(StrategyErrorSeverity.CRITICAL) is False
        
        strategy.create_error(
            message="Critical message",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.CRITICAL
        )
        
        # Check for critical
        assert strategy.has_errors(StrategyErrorSeverity.CRITICAL) is True
        
        # Check with string severity
        assert strategy.has_errors("warning") is True
        assert strategy.has_errors("error") is True
        assert strategy.has_errors("critical") is True

    def test_clear_errors(self):
        """Test clearing all errors."""
        strategy = TestStrategyClass()
        
        # Create some errors
        strategy.create_error(
            message="Error 1",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR
        )
        
        strategy.create_error(
            message="Error 2",
            category=StrategyErrorCategory.PARSING,
            severity=StrategyErrorSeverity.WARNING
        )
        
        # Verify errors exist
        assert len(strategy._errors) == 2
        
        # Clear errors
        strategy.clear_errors()
        
        # Verify errors were cleared
        assert len(strategy._errors) == 0

    def test_get_error_counts(self):
        """Test getting error counts by category and severity."""
        strategy = TestStrategyClass()
        
        # Create some errors
        strategy.create_error(
            message="Network error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            recoverable=True
        )
        
        strategy.create_error(
            message="Another network error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.WARNING,
            recoverable=True
        )
        
        strategy.create_error(
            message="Parsing error",
            category=StrategyErrorCategory.PARSING,
            severity=StrategyErrorSeverity.ERROR,
            recoverable=False
        )
        
        # Get error counts
        counts = strategy.get_error_counts()
        
        # Check total count
        assert counts["total"] == 3
        
        # Check counts by category
        assert counts["by_category"]["network"] == 2
        assert counts["by_category"]["parsing"] == 1
        assert counts["by_category"]["execution"] == 0  # No execution errors
        
        # Check counts by severity
        assert counts["by_severity"]["error"] == 2
        assert counts["by_severity"]["warning"] == 1
        assert counts["by_severity"]["info"] == 0  # No info errors
        assert counts["by_severity"]["critical"] == 0  # No critical errors
        
        # Check recoverable counts
        assert counts["recoverable"] == 2
        assert counts["non_recoverable"] == 1

    def test_get_error_summary(self):
        """Test getting an error summary."""
        strategy = TestStrategyClass()
        
        # Create an error
        strategy.create_error(
            message="Test error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR
        )
        
        # Get error summary
        summary = strategy.get_error_summary()
        
        # Check that the summary contains counts and errors
        assert "counts" in summary
        assert "errors" in summary
        assert len(summary["errors"]) == 1
        assert summary["counts"]["total"] == 1

    @patch("time.sleep")
    def test_retry_with_backoff(self, mock_sleep):
        """Test the retry with backoff handler."""
        strategy = TestStrategyClass()
        
        # Create an error
        error = StrategyError(
            message="Test error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            strategy_name="test_strategy"
        )
        
        # Call retry handler
        result = strategy._retry_with_backoff(error)
        
        # Check that handling was successful
        assert result is True
        
        # Check that sleep was called with the right delay
        mock_sleep.assert_called_once_with(1)  # 2^0
        
        # Check that retry count was incremented
        assert error.retry_count == 1
        
        # Call retry handler again
        result = strategy._retry_with_backoff(error)
        
        # Check that handling was successful
        assert result is True
        
        # Check that sleep was called with the right delay
        mock_sleep.assert_called_with(2)  # 2^1
        
        # Check that retry count was incremented again
        assert error.retry_count == 2

    @patch("time.sleep")
    def test_handle_rate_limit(self, mock_sleep):
        """Test the rate limit handler."""
        strategy = TestStrategyClass()
        
        # Create an error with retry-after context
        error = StrategyError(
            message="Rate limited",
            category=StrategyErrorCategory.RATE_LIMIT,
            severity=StrategyErrorSeverity.WARNING,
            strategy_name="test_strategy",
            context_data={"retry_after": 30}
        )
        
        # Call rate limit handler
        result = strategy._handle_rate_limit(error)
        
        # Check that handling was successful
        assert result is True
        
        # Check that sleep was called with the right delay
        mock_sleep.assert_called_once_with(30)
        
        # Check that retry count was incremented
        assert error.retry_count == 1
        
        # Create an error without retry-after context
        error2 = StrategyError(
            message="Rate limited",
            category=StrategyErrorCategory.RATE_LIMIT,
            severity=StrategyErrorSeverity.WARNING,
            strategy_name="test_strategy"
        )
        
        # Call rate limit handler
        result = strategy._handle_rate_limit(error2)
        
        # Check that handling was successful
        assert result is True
        
        # Check that sleep was called with the default delay
        mock_sleep.assert_called_with(60)
        
        # Check that retry count was incremented
        assert error2.retry_count == 1


class TestUtilityFunctions:
    """Tests for utility error handling functions."""

    def test_log_error(self, caplog):
        """Test the log_error utility function."""
        # Create errors with different severities
        info_error = StrategyError(
            message="Info message",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.INFO,
            strategy_name="test_strategy"
        )
        
        warning_error = StrategyError(
            message="Warning message",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.WARNING,
            strategy_name="test_strategy"
        )
        
        error_error = StrategyError(
            message="Error message",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            strategy_name="test_strategy"
        )
        
        critical_error = StrategyError(
            message="Critical message",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.CRITICAL,
            strategy_name="test_strategy"
        )
        
        # Test logging at different levels
        with caplog.at_level(10):  # DEBUG level
            result = log_error(info_error)
            assert "INFO [network]" in caplog.text
            assert "Info message" in caplog.text
            assert result is True
            
            caplog.clear()
            result = log_error(warning_error)
            assert "WARNING [network]" in caplog.text
            assert "Warning message" in caplog.text
            assert result is True
            
            caplog.clear()
            result = log_error(error_error)
            assert "ERROR [network]" in caplog.text
            assert "Error message" in caplog.text
            assert result is True
            
            caplog.clear()
            result = log_error(critical_error)
            assert "CRITICAL [network]" in caplog.text
            assert "Critical message" in caplog.text
            assert result is True

    @patch("time.sleep")
    def test_retry_on_network_error(self, mock_sleep):
        """Test the retry_on_network_error utility function."""
        # Create a network error
        network_error = StrategyError(
            message="Network error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            strategy_name="test_strategy"
        )
        
        # Create a non-network error
        parsing_error = StrategyError(
            message="Parsing error",
            category=StrategyErrorCategory.PARSING,
            severity=StrategyErrorSeverity.ERROR,
            strategy_name="test_strategy"
        )
        
        # Test retrying a network error
        result = retry_on_network_error(network_error)
        assert result is True
        mock_sleep.assert_called_once_with(1)
        assert network_error.retry_count == 1
        
        # Test retrying the same network error again (retry count should be incremented)
        result = retry_on_network_error(network_error)
        assert result is True
        mock_sleep.assert_called_with(2)
        assert network_error.retry_count == 2
        
        # Test with a non-network error (should not retry)
        result = retry_on_network_error(parsing_error)
        assert result is False
        assert parsing_error.retry_count == 0

    @patch("time.sleep")
    def test_wait_on_rate_limit(self, mock_sleep):
        """Test the wait_on_rate_limit utility function."""
        # Create a rate limit error with retry-after context
        rate_limit_error = StrategyError(
            message="Rate limited",
            category=StrategyErrorCategory.RATE_LIMIT,
            severity=StrategyErrorSeverity.WARNING,
            strategy_name="test_strategy",
            context_data={"retry_after": 30}
        )
        
        # Create a rate limit error without retry-after context
        rate_limit_error2 = StrategyError(
            message="Rate limited",
            category=StrategyErrorCategory.RATE_LIMIT,
            severity=StrategyErrorSeverity.WARNING,
            strategy_name="test_strategy"
        )
        
        # Create a non-rate-limit error
        network_error = StrategyError(
            message="Network error",
            category=StrategyErrorCategory.NETWORK,
            severity=StrategyErrorSeverity.ERROR,
            strategy_name="test_strategy"
        )
        
        # Test waiting for a rate limit error with retry-after
        result = wait_on_rate_limit(rate_limit_error)
        assert result is True
        mock_sleep.assert_called_once_with(30)
        assert rate_limit_error.retry_count == 1
        
        # Test waiting for a rate limit error without retry-after
        result = wait_on_rate_limit(rate_limit_error2)
        assert result is True
        mock_sleep.assert_called_with(60)  # Default wait time
        assert rate_limit_error2.retry_count == 1
        
        # Test with a non-rate-limit error (should not wait)
        result = wait_on_rate_limit(network_error)
        assert result is False
        assert network_error.retry_count == 0