import pytest
import time
import asyncio
import threading
from unittest.mock import MagicMock, patch
import requests
from concurrent.futures import TimeoutError

from core.retry_manager import RetryManager, RetryOutcome
from core.error_classifier import ErrorCategory, ErrorSeverity

class TestRetryManager:
    """Tests for the RetryManager service."""
    
    def setup_method(self):
        """Set up the test environment."""
        self.retry_manager = RetryManager()
        
        # Initialize with test configuration
        test_config = {
            'default_max_attempts': 3,
            'default_backoff_factor': 2.0,
            'default_retry_budget': 10,
            'jitter_factor': 0.1,
            'domains': [
                {
                    'name': 'example.com',
                    'retry_budget': 5,
                    'retry_budget_refill_period': 60,
                    'retry_budget_refill_amount': 2
                }
            ]
        }
        self.retry_manager.initialize(test_config)
        
        # Mock the error classifier for testing
        self.mock_error_classifier = MagicMock()
        self.retry_manager._error_classifier = self.mock_error_classifier
    
    def teardown_method(self):
        """Clean up the test environment."""
        self.retry_manager.shutdown()
    
    def test_initialization(self):
        """Test that the retry manager initializes correctly."""
        assert self.retry_manager.is_initialized
        assert self.retry_manager.name == "retry_manager"
        
        # Test config values loaded correctly
        assert self.retry_manager._default_max_attempts == 3
        assert self.retry_manager._default_backoff_factor == 2.0
        assert self.retry_manager._jitter_factor == 0.1
        
        # Test domain budgets initialized
        assert self.retry_manager._retry_budgets['example.com'] == 5
    
    def test_should_retry_max_attempts(self):
        """Test retry decisions based on max attempts."""
        # Should retry if under max attempts
        assert self.retry_manager.should_retry(attempt=0, context={'max_attempts': 3}) == True
        assert self.retry_manager.should_retry(attempt=1, context={'max_attempts': 3}) == True
        
        # Should not retry if max attempts reached
        assert self.retry_manager.should_retry(attempt=2, context={'max_attempts': 3}) == False
        assert self.retry_manager.should_retry(attempt=3, context={'max_attempts': 3}) == False
    
    def test_should_retry_error_classifier_integration(self):
        """Test integration with error classifier for retry decisions."""
        # Setup error classifier response
        self.mock_error_classifier.classify_exception.return_value = {
            'is_retryable': True
        }
        
        # Should retry based on error classifier
        error = ValueError("Test error")
        assert self.retry_manager.should_retry(attempt=0, error=error) == True
        
        # Should use error classifier with correct parameters
        self.mock_error_classifier.classify_exception.assert_called_with(error, {})
        
        # Test when error classifier says not to retry
        self.mock_error_classifier.classify_exception.return_value = {
            'is_retryable': False
        }
        assert self.retry_manager.should_retry(attempt=0, error=error) == False
    
    def test_should_retry_budget_exhausted(self):
        """Test retry decisions when budget is exhausted."""
        # Set up a domain with exhausted budget
        self.retry_manager._retry_budgets['low-budget.com'] = 0
        
        # Should not retry if budget exhausted
        assert self.retry_manager.should_retry(
            attempt=0, 
            error=ValueError(), 
            context={'domain': 'low-budget.com'}
        ) == False
    
    def test_get_retry_delay_strategies(self):
        """Test different delay calculation strategies."""
        # Test constant delay
        delay = self.retry_manager.get_retry_delay(
            attempt=2,
            context={'delay_strategy': 'constant', 'base_delay': 1.0}
        )
        # With jitter, the delay should be around 1.0
        assert 0.9 <= delay <= 1.1
        
        # Test linear delay
        delay = self.retry_manager.get_retry_delay(
            attempt=2,
            context={'delay_strategy': 'linear', 'base_delay': 1.0}
        )
        # Should be base_delay * (attempt + 1) with jitter
        assert 2.7 <= delay <= 3.3
        
        # Test exponential delay
        delay = self.retry_manager.get_retry_delay(
            attempt=2,
            context={'delay_strategy': 'exponential', 'base_delay': 1.0, 'backoff_factor': 2.0}
        )
        # Should be base_delay * (backoff_factor ** attempt) with jitter
        assert 3.6 <= delay <= 4.4
        
        # Test fibonacci delay
        delay = self.retry_manager.get_retry_delay(
            attempt=3,
            context={'delay_strategy': 'fibonacci', 'base_delay': 1.0}
        )
        # Fibonacci sequence: 1, 2, 3, 5 - at attempt 3, should be around 5
        assert 4.5 <= delay <= 5.5
    
    def test_get_retry_delay_max_and_retry_after(self):
        """Test delay with max_delay and retry-after header."""
        # Test max_delay cap
        delay = self.retry_manager.get_retry_delay(
            attempt=10,  # Would normally result in a very large delay
            context={'delay_strategy': 'exponential', 'base_delay': 1.0, 'max_delay': 5.0}
        )
        assert delay <= 5.0
        
        # Test retry-after honored
        delay = self.retry_manager.get_retry_delay(
            attempt=1,
            context={'retry_after': 10.0, 'base_delay': 1.0}
        )
        # Should use retry_after as it's greater than calculated delay
        assert delay >= 10.0
    
    def test_register_and_get_attempt_count(self):
        """Test tracking attempts for operations."""
        op_id = "test_operation"
        
        # Register a successful attempt
        self.retry_manager.register_attempt(op_id, True)
        counts = self.retry_manager.get_attempt_count(op_id)
        
        assert counts['total_attempts'] == 1
        assert counts['successful_attempts'] == 1
        assert counts['failed_attempts'] == 0
        
        # Register a failed attempt
        error = ValueError("Test error")
        self.retry_manager.register_attempt(op_id, False, error)
        counts = self.retry_manager.get_attempt_count(op_id)
        
        assert counts['total_attempts'] == 2
        assert counts['successful_attempts'] == 1
        assert counts['failed_attempts'] == 1
        assert counts['last_error'] == error
        
        # Reset attempts
        self.retry_manager.reset_attempts(op_id)
        counts = self.retry_manager.get_attempt_count(op_id)
        
        assert counts['total_attempts'] == 0
    
    def test_retry_budget_management(self):
        """Test retry budget management functions."""
        domain = "test-domain.com"
        
        # Initial budget should be default
        initial_budget = self.retry_manager.get_retry_budget(domain)
        assert initial_budget == 10  # Default from test config
        
        # Consume some budget
        remaining = self.retry_manager.consume_retry_budget(domain, 3)
        assert remaining == 7
        
        # Verify budget
        current = self.retry_manager.get_retry_budget(domain)
        assert current == 7
        
        # Check if budget exhausted
        assert self.retry_manager.is_budget_exhausted(domain) == False
        
        # Consume all remaining budget
        self.retry_manager.consume_retry_budget(domain, 7)
        assert self.retry_manager.is_budget_exhausted(domain) == True
        
        # Refill budget
        refilled = self.retry_manager.refill_retry_budget(domain, 5)
        assert refilled == 5
        assert self.retry_manager.is_budget_exhausted(domain) == False
        
        # Refill to default
        refilled = self.retry_manager.refill_retry_budget(domain)
        assert refilled == 10
    
    def test_auto_budget_refill(self):
        """Test automatic budget refill over time."""
        domain = "example.com"  # Domain with config in setup
        
        # Set initial budget to 0
        self.retry_manager._retry_budgets[domain] = 0
        
        # Set last refill time to more than the refill period ago
        self.retry_manager._budget_refill_times[domain] = time.time() - 120  # 2 minutes ago
        
        # Check budget - should auto-refill
        budget = self.retry_manager.get_retry_budget(domain)
        
        # Should have refilled with 2 periods × 2 tokens = 4 tokens
        assert budget == 4
    
    def test_add_jitter(self):
        """Test jitter addition to delays."""
        base_delay = 10.0
        factor = 0.2
        
        # Add jitter with custom factor
        jittered = self.retry_manager.add_jitter(base_delay, factor)
        
        # Should be within ±20% of the base delay
        assert 8.0 <= jittered <= 12.0
        assert jittered != base_delay  # Extremely unlikely to be exactly equal
    
    def test_retry_decorator_success(self):
        """Test basic retry decorator with successful function."""
        call_count = 0
        
        @self.retry_manager.retry(max_attempts=3)
        def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = test_func()
        
        assert result == "success"
        assert call_count == 1  # Should only be called once
    
    def test_retry_decorator_with_retries(self):
        """Test retry decorator with a function that fails initially."""
        call_count = 0
        
        @self.retry_manager.retry(max_attempts=3, base_delay=0.01)  # Small delay for faster test
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Failure {call_count}")
            return "success after retries"
        
        result = test_func()
        
        assert result == "success after retries"
        assert call_count == 3  # Should be called 3 times
    
    def test_retry_decorator_max_attempts_exceeded(self):
        """Test retry decorator when max attempts are exceeded."""
        call_count = 0
        
        @self.retry_manager.retry(max_attempts=3, base_delay=0.01)
        def test_func():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Always fails {call_count}")
        
        with pytest.raises(ValueError) as exc_info:
            test_func()
        
        assert "Always fails 3" in str(exc_info.value)  # Last error message
        assert call_count == 3  # Should be called 3 times
    
    def test_retry_on_exception_type(self):
        """Test retry on specific exception types."""
        call_count = 0
        
        @self.retry_manager.retry(
            max_attempts=3, 
            base_delay=0.01,
            retry_on=ValueError  # Only retry on ValueError
        )
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retryable error")
            if call_count == 2:
                raise TypeError("Non-retryable error")
            return "success"
        
        with pytest.raises(TypeError):
            test_func()
        
        assert call_count == 2  # Called twice (retry once on ValueError, fail on TypeError)
    
    def test_retry_if_custom_condition(self):
        """Test retry with custom condition function."""
        call_count = 0
        
        def retry_on_even_calls(e):
            return call_count % 2 == 0
        
        @self.retry_manager.retry(
            max_attempts=4, 
            base_delay=0.01,
            retry_if=retry_on_even_calls
        )
        def test_func():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Error {call_count}")
        
        with pytest.raises(ValueError) as exc_info:
            test_func()
        
        # Should retry on calls 1, 3 but not 2
        assert call_count == 3
        assert "Error 3" in str(exc_info.value)
    
    def test_retry_unless_custom_condition(self):
        """Test retry with custom unless condition."""
        call_count = 0
        
        def dont_retry_on_special_error(e):
            return "special" in str(e)
        
        @self.retry_manager.retry(
            max_attempts=4, 
            base_delay=0.01,
            retry_unless=dont_retry_on_special_error
        )
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Regular error")
            if call_count == 2:
                raise ValueError("This is a special error")
            return "success"
        
        with pytest.raises(ValueError) as exc_info:
            test_func()
        
        assert "special error" in str(exc_info.value)
        assert call_count == 2  # Should not retry after the special error
    
    def test_async_retry_decorator(self):
        """Test the async retry decorator."""
        call_count = 0
        
        @self.retry_manager.retry_async(max_attempts=3, base_delay=0.01)
        async def test_async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Async failure {call_count}")
            return "async success"
        
        # Run the async function
        import asyncio
        result = asyncio.run(test_async_func())
        
        assert result == "async success"
        assert call_count == 3
    
    def test_conditional_retry_decorator(self):
        """Test the conditional retry decorator."""
        call_count = 0
        
        # Custom condition function that checks both exception and context
        def custom_condition(e, context):
            return (isinstance(e, ValueError) and 
                    context['attempt'] < 2)  # Only retry for the first two attempts
        
        # Custom delay function
        def custom_delay(attempt, e, context):
            return 0.01 * (attempt + 1)  # Just for testing
        
        @self.retry_manager.conditional_retry(
            condition=custom_condition,
            delay_func=custom_delay
        )
        def test_func():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Conditional failure {call_count}")
        
        with pytest.raises(ValueError) as exc_info:
            test_func()
        
        assert "Conditional failure 3" in str(exc_info.value)
        assert call_count == 3  # Should retry twice then fail
    
    def test_retry_with_timeout(self):
        """Test retry with a timeout limit."""
        call_count = 0
        start_time = time.time()
        
        @self.retry_manager.retry_with_timeout(
            timeout=0.5,  # Very short timeout for testing
            max_attempts=10,
            base_delay=0.2
        )
        def test_func():
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)  # Ensure some time passes
            raise ValueError(f"Timeout test {call_count}")
        
        with pytest.raises(TimeoutError):
            test_func()
        
        elapsed = time.time() - start_time
        # Should timeout after ~0.5 seconds
        assert 0.4 <= elapsed <= 1.0  # Allow some leeway for test execution
        # Should not reach max attempts due to timeout
        assert call_count < 10
    
    def test_retry_on_result(self):
        """Test retrying based on the return value."""
        call_count = 0
        
        # Function that returns a "bad" result initially
        @self.retry_manager.retry_on_result(
            result_predicate=lambda result: result < 3  # Retry if result < 3
        )
        def test_func():
            nonlocal call_count
            call_count += 1
            return call_count  # Return value increases each call
        
        result = test_func()
        
        assert result == 3  # Should return when the result is 3
        assert call_count == 3  # Should be called 3 times
    
    def test_retry_on_http_status(self):
        """Test retry predicate for HTTP status codes."""
        # Create a predicate for specific status codes
        predicate = self.retry_manager.retry_on_http_status([429, 503])
        
        # Mock response with 429 status
        mock_error = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_error.response = mock_response
        
        # Should retry on 429
        assert predicate(mock_error, {}) == True
        
        # Should not retry on 404
        mock_response.status_code = 404
        assert predicate(mock_error, {}) == False
        
        # Should use status_code from context if provided
        assert predicate(ValueError(), {'status_code': 503}) == True
        assert predicate(ValueError(), {'status_code': 200}) == False
    
    def test_retry_until_success(self):
        """Test retry until success decorator."""
        call_count = 0
        
        @self.retry_manager.retry_until_success(max_attempts=5)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Not yet successful {call_count}")
            return "finally succeeded"
        
        result = test_func()
        
        assert result == "finally succeeded"
        assert call_count == 3  # Should retry until success on 3rd attempt
    
    def test_retry_with_fallback(self):
        """Test retry with fallback function."""
        main_call_count = 0
        
        def main_func():
            nonlocal main_call_count
            main_call_count += 1
            raise ValueError("Main function always fails")
        
        def fallback_func(*args, **kwargs):
            return "fallback result"
        
        # Create decorated function
        decorated = self.retry_manager.retry_with_fallback(fallback_func)(main_func)
        
        # Execute and check results
        result = decorated()
        
        assert result == "fallback result"
        assert main_call_count == 3  # Default retry count
    
    def test_execute_with_retry(self):
        """Test the imperative execute_with_retry method."""
        call_count = 0
        
        def test_operation(arg1, arg2, kwarg1=None):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Operation failed")
            return f"Success: {arg1}, {arg2}, {kwarg1}"
        
        # Execute with retry
        result, outcome = self.retry_manager.execute_with_retry(
            test_operation,
            "value1", "value2",
            kwarg1="kwvalue",
            max_attempts=3,
            base_delay=0.01
        )
        
        assert result == "Success: value1, value2, kwvalue"
        assert outcome == RetryOutcome.SUCCESS
        assert call_count == 2  # Should succeed on second attempt
        
        # Test failure scenario
        call_count = 0
        def failing_operation():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        result, outcome = self.retry_manager.execute_with_retry(
            failing_operation,
            max_attempts=2,
            base_delay=0.01
        )
        
        assert result is None
        assert outcome == RetryOutcome.FAILED_RETRIES_EXHAUSTED
        assert call_count == 2  # Should try max_attempts times
        
        # Test budget exhaustion
        self.retry_manager._retry_budgets["test.com"] = 0
        call_count = 0
        
        result, outcome = self.retry_manager.execute_with_retry(
            failing_operation,
            domain="test.com",
            max_attempts=3
        )
        
        assert result is None
        assert outcome == RetryOutcome.FAILED_BUDGET_EXHAUSTED
        assert call_count == 1  # Should only call once if budget exhausted
    
    def test_retry_strategies(self):
        """Test the retry delay strategy methods directly."""
        # Constant delay
        delay = self.retry_manager.constant_delay(2.0)
        assert 1.8 <= delay <= 2.2  # With jitter
        
        # Linear delay
        delay = self.retry_manager.linear_delay(attempt=2, base_delay=1.0)
        assert 2.7 <= delay <= 3.3  # base_delay * (attempt + 1) with jitter
        
        # Exponential delay
        delay = self.retry_manager.exponential_delay(attempt=3, base_delay=1.0, factor=2.0)
        assert 7.2 <= delay <= 8.8  # base_delay * (factor ^ attempt) with jitter
        
        # Fibonacci delay
        delay = self.retry_manager.fibonacci_delay(attempt=5, base_delay=1.0)
        # Fibonacci: 1, 1, 2, 3, 5, 8 - so attempt 5 should be ~8
        assert 7.2 <= delay <= 8.8  # With jitter