"""
Strategy error handler module providing robust error handling for strategies.

This module contains:
1. Error category and severity enums for classifying errors
2. StrategyError class for storing error details
3. StrategyErrorHandlingMixin for integrating error handling into strategies
4. Utility functions for common error handling scenarios
"""

import datetime
import inspect
import logging
import traceback
from abc import ABC
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

logger = logging.getLogger(__name__)

class StrategyErrorCategory(Enum):
    """Categories of errors that can occur during strategy execution."""
    NETWORK = "network"           # Connectivity issues, timeouts, DNS failures
    PARSING = "parsing"           # HTML/content processing issues
    EXECUTION = "execution"       # Strategy execution issues
    VALIDATION = "validation"     # Input/output validation issues
    RESOURCE = "resource"         # Resource-related issues (memory, disk, etc.)
    AUTHENTICATION = "authentication"  # Login/authentication failures
    AUTHORIZATION = "authorization"    # Permission/access issues
    RATE_LIMIT = "rate_limit"     # Rate limiting or throttling issues
    UNEXPECTED = "unexpected"     # Unexpected or unclassified errors
    DATA = "data"                 # Data-related issues
    DEPENDENCY = "dependency"     # Issues with dependencies
    CONFIGURATION = "configuration"  # Configuration issues
    BUSINESS_LOGIC = "business_logic"  # Business logic errors


class StrategyErrorSeverity(Enum):
    """Severity levels for strategy errors."""
    INFO = "info"                 # Non-critical information
    WARNING = "warning"           # Potential issues that don't prevent execution
    ERROR = "error"               # Failures that can be recovered from
    CRITICAL = "critical"         # Failures that cannot be recovered from


class StrategyError:
    """
    Comprehensive representation of an error that occurred during strategy execution.
    
    Stores detailed information about the error including its category, severity,
    the original exception, context data, timestamp, and more.
    """
    
    def __init__(self,
                message: str,
                category: Union[str, StrategyErrorCategory],
                severity: Union[str, StrategyErrorSeverity],
                strategy_name: str,
                url: Optional[str] = None,
                exception: Optional[Exception] = None,
                traceback_str: Optional[str] = None,
                context_data: Optional[Dict[str, Any]] = None,
                recoverable: bool = True,
                retry_suggestion: Optional[str] = None):
        """
        Initialize a strategy error.
        
        Args:
            message: Description of the error
            category: Error category
            severity: Error severity
            strategy_name: Name of the strategy that encountered the error
            url: URL being processed when the error occurred
            exception: Original exception
            traceback_str: String representation of the exception traceback
            context_data: Additional context data related to the error
            recoverable: Whether the error is potentially recoverable
            retry_suggestion: Suggestion for retrying/resolving the error
        """
        self.message = message
        
        # Convert string category to enum if needed
        if isinstance(category, str):
            try:
                self.category = StrategyErrorCategory(category)
            except ValueError:
                self.category = StrategyErrorCategory.UNEXPECTED
        else:
            self.category = category
            
        # Convert string severity to enum if needed
        if isinstance(severity, str):
            try:
                self.severity = StrategyErrorSeverity(severity)
            except ValueError:
                self.severity = StrategyErrorSeverity.WARNING
        else:
            self.severity = severity
            
        self.strategy_name = strategy_name
        self.url = url
        self.exception = exception
        self.exception_type = type(exception).__name__ if exception else None
        self.exception_message = str(exception) if exception else None
        
        # Get traceback if not provided but we have an exception
        if not traceback_str and exception:
            self.traceback = traceback.format_exc()
        else:
            self.traceback = traceback_str
            
        self.context_data = context_data or {}
        self.recoverable = recoverable
        self.retry_suggestion = retry_suggestion
        self.timestamp = datetime.datetime.now()
        self.error_id = f"{self.strategy_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Keep track of retry attempts if this error is retried
        self.retry_count = 0
        
    def __str__(self) -> str:
        """String representation of the error."""
        parts = [f"{self.severity.value.upper()} [{self.category.value}]"]
        parts.append(self.message)
        
        if self.url:
            parts.append(f"URL: {self.url}")
            
        if self.exception:
            parts.append(f"Exception: {self.exception_type}: {self.exception_message}")
            
        return " - ".join(parts)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the error
        """
        error_dict = {
            "error_id": self.error_id,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "strategy_name": self.strategy_name,
            "timestamp": self.timestamp.isoformat(),
            "recoverable": self.recoverable,
            "retry_count": self.retry_count
        }
        
        if self.url:
            error_dict["url"] = self.url
            
        if self.exception_type:
            error_dict["exception_type"] = self.exception_type
            error_dict["exception_message"] = self.exception_message
            
        if self.retry_suggestion:
            error_dict["retry_suggestion"] = self.retry_suggestion
            
        if self.context_data:
            error_dict["context_data"] = self.context_data
            
        # Don't include traceback by default as it can be very long
        # Include it if specifically needed
        
        return error_dict
        
    def increment_retry(self) -> None:
        """Increment the retry count for this error."""
        self.retry_count += 1


class ErrorPolicy:
    """
    Defines how to handle errors of specific categories and severities.
    
    Error policies can be attached to strategies to customize error handling behavior.
    """
    
    def __init__(self,
                 categories: Optional[Set[StrategyErrorCategory]] = None,
                 severities: Optional[Set[StrategyErrorSeverity]] = None,
                 handler: Optional[Callable[[StrategyError], bool]] = None,
                 max_retries: int = 3,
                 log_level: int = logging.WARNING):
        """
        Initialize an error policy.
        
        Args:
            categories: Set of error categories this policy applies to
            severities: Set of error severities this policy applies to
            handler: Function to handle errors matching this policy
            max_retries: Maximum retry attempts for errors matching this policy
            log_level: Logging level for errors matching this policy
        """
        self.categories = categories or set()
        self.severities = severities or set()
        self.handler = handler
        self.max_retries = max_retries
        self.log_level = log_level
        
    def matches(self, error: StrategyError) -> bool:
        """
        Check if this policy applies to a given error.
        
        Args:
            error: The error to check
            
        Returns:
            True if this policy applies to the error, False otherwise
        """
        # If no categories specified, matches any category
        category_match = not self.categories or error.category in self.categories
        
        # If no severities specified, matches any severity
        severity_match = not self.severities or error.severity in self.severities
        
        return category_match and severity_match
        
    def handle(self, error: StrategyError) -> bool:
        """
        Handle an error using this policy.
        
        Args:
            error: The error to handle
            
        Returns:
            True if the error was handled successfully, False otherwise
        """
        # Log the error at the appropriate level
        log_message = str(error)
        
        if self.log_level == logging.DEBUG:
            logger.debug(log_message)
        elif self.log_level == logging.INFO:
            logger.info(log_message)
        elif self.log_level == logging.WARNING:
            logger.warning(log_message)
        elif self.log_level == logging.ERROR:
            logger.error(log_message)
        elif self.log_level == logging.CRITICAL:
            logger.critical(log_message)
            
        # If no handler is defined, just log and return True
        if not self.handler:
            return True
            
        # Execute the custom handler
        try:
            return self.handler(error)
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
            return False


class StrategyErrorHandlingMixin:
    """
    Mixin class providing robust error handling capabilities for strategies.
    
    Key features:
    1. Error tracking by category and severity
    2. Configurable error policies
    3. Automatic error handling
    4. Error aggregation and reporting
    """
    
    def __init__(self):
        """Initialize the error handling mixin."""
        self._errors: List[StrategyError] = []
        self._error_policies: List[ErrorPolicy] = []
        self._default_policy = ErrorPolicy(
            handler=lambda error: True,  # Default just logs the error
            log_level=logging.WARNING
        )
        
        # Initialize with some common error policies
        self._init_common_policies()
        
    def _init_common_policies(self) -> None:
        """Initialize common error policies."""
        # Network errors - retry with backoff
        self.register_error_policy(ErrorPolicy(
            categories={StrategyErrorCategory.NETWORK},
            severities={StrategyErrorSeverity.WARNING, StrategyErrorSeverity.ERROR},
            handler=self._retry_with_backoff,
            max_retries=3,
            log_level=logging.WARNING
        ))
        
        # Rate limit errors - wait and retry
        self.register_error_policy(ErrorPolicy(
            categories={StrategyErrorCategory.RATE_LIMIT},
            handler=self._handle_rate_limit,
            max_retries=5,
            log_level=logging.WARNING
        ))
        
        # Critical errors - log and don't retry
        self.register_error_policy(ErrorPolicy(
            severities={StrategyErrorSeverity.CRITICAL},
            handler=lambda error: False,  # Don't retry critical errors
            log_level=logging.ERROR
        ))
        
    def register_error_policy(self, policy: ErrorPolicy) -> None:
        """
        Register an error policy.
        
        Args:
            policy: The error policy to register
        """
        self._error_policies.append(policy)
        
    def create_error(self,
                    message: str,
                    category: Union[str, StrategyErrorCategory],
                    severity: Union[str, StrategyErrorSeverity],
                    url: Optional[str] = None,
                    exception: Optional[Exception] = None,
                    context_data: Optional[Dict[str, Any]] = None,
                    recoverable: bool = True,
                    retry_suggestion: Optional[str] = None) -> StrategyError:
        """
        Create a new strategy error.
        
        Args:
            message: Description of the error
            category: Error category
            severity: Error severity
            url: URL being processed when the error occurred
            exception: Original exception
            context_data: Additional context data related to the error
            recoverable: Whether the error is potentially recoverable
            retry_suggestion: Suggestion for retrying/resolving the error
            
        Returns:
            Created StrategyError
        """
        # Get strategy name from the class instance
        strategy_name = getattr(self, 'name', self.__class__.__name__)
        
        # Create error
        error = StrategyError(
            message=message,
            category=category,
            severity=severity,
            strategy_name=strategy_name,
            url=url,
            exception=exception,
            context_data=context_data,
            recoverable=recoverable,
            retry_suggestion=retry_suggestion
        )
        
        # Add to error list
        self._errors.append(error)
        
        return error
        
    def handle_error(self,
                    error: Union[StrategyError, Exception, None],
                    message: Optional[str] = None,
                    category: Union[str, StrategyErrorCategory] = StrategyErrorCategory.UNEXPECTED,
                    severity: Union[str, StrategyErrorSeverity] = StrategyErrorSeverity.WARNING,
                    url: Optional[str] = None,
                    context_data: Optional[Dict[str, Any]] = None,
                    recoverable: bool = True,
                    retry_suggestion: Optional[str] = None) -> bool:
        """
        Handle an error that occurred during strategy execution.
        
        Args:
            error: The error or exception
            message: Error message (if not using a StrategyError)
            category: Error category (if not using a StrategyError)
            severity: Error severity (if not using a StrategyError)
            url: URL being processed (if not using a StrategyError)
            context_data: Additional context data (if not using a StrategyError)
            recoverable: Whether the error is recoverable (if not using a StrategyError)
            retry_suggestion: Suggestion for retrying (if not using a StrategyError)
            
        Returns:
            True if the error was handled successfully, False otherwise
        """
        # Convert Exception to StrategyError if needed
        if error is not None and not isinstance(error, StrategyError):
            if isinstance(error, Exception):
                # Use provided message or exception string
                error_message = message or str(error)
                
                # Create a StrategyError
                error = self.create_error(
                    message=error_message,
                    category=category,
                    severity=severity,
                    url=url,
                    exception=error,
                    context_data=context_data,
                    recoverable=recoverable,
                    retry_suggestion=retry_suggestion
                )
            else:
                # Unknown error type
                error_message = message or f"Unknown error: {error}"
                
                # Create a StrategyError
                error = self.create_error(
                    message=error_message,
                    category=category,
                    severity=severity,
                    url=url,
                    context_data=context_data,
                    recoverable=recoverable,
                    retry_suggestion=retry_suggestion
                )
        elif error is None:
            # No error provided, create a new one with the message
            if not message:
                message = "Unknown error (no details provided)"
                
            error = self.create_error(
                message=message,
                category=category,
                severity=severity,
                url=url,
                context_data=context_data,
                recoverable=recoverable,
                retry_suggestion=retry_suggestion
            )
            
        # Find matching policy
        handled = False
        for policy in self._error_policies:
            if policy.matches(error):
                if error.retry_count < policy.max_retries or policy.max_retries < 0:
                    handled = policy.handle(error)
                    if handled:
                        break
                        
        # Use default policy if no matching policy found or all policies failed
        if not handled:
            handled = self._default_policy.handle(error)
            
        return handled
        
    def get_errors(self,
                  categories: Optional[Set[Union[str, StrategyErrorCategory]]] = None,
                  severities: Optional[Set[Union[str, StrategyErrorSeverity]]] = None,
                  recoverable_only: bool = False,
                  since: Optional[datetime.datetime] = None) -> List[StrategyError]:
        """
        Get errors matching the specified filters.
        
        Args:
            categories: Filter by error categories
            severities: Filter by error severities
            recoverable_only: Only include recoverable errors
            since: Only include errors that occurred after this time
            
        Returns:
            List of errors matching the filters
        """
        # Convert string categories to enums if needed
        if categories:
            enum_categories = set()
            for category in categories:
                if isinstance(category, str):
                    try:
                        enum_categories.add(StrategyErrorCategory(category))
                    except ValueError:
                        pass  # Skip invalid categories
                else:
                    enum_categories.add(category)
            categories = enum_categories
            
        # Convert string severities to enums if needed
        if severities:
            enum_severities = set()
            for severity in severities:
                if isinstance(severity, str):
                    try:
                        enum_severities.add(StrategyErrorSeverity(severity))
                    except ValueError:
                        pass  # Skip invalid severities
                else:
                    enum_severities.add(severity)
            severities = enum_severities
            
        # Filter errors
        filtered_errors = []
        for error in self._errors:
            # Check category
            if categories and error.category not in categories:
                continue
                
            # Check severity
            if severities and error.severity not in severities:
                continue
                
            # Check recoverability
            if recoverable_only and not error.recoverable:
                continue
                
            # Check timestamp
            if since and error.timestamp < since:
                continue
                
            filtered_errors.append(error)
            
        return filtered_errors
        
    def has_errors(self, min_severity: Union[str, StrategyErrorSeverity] = StrategyErrorSeverity.WARNING) -> bool:
        """
        Check if there are any errors with at least the specified severity.
        
        Args:
            min_severity: Minimum severity level to check for
            
        Returns:
            True if there are errors with at least the specified severity, False otherwise
        """
        # Convert string severity to enum if needed
        if isinstance(min_severity, str):
            try:
                min_severity = StrategyErrorSeverity(min_severity)
            except ValueError:
                min_severity = StrategyErrorSeverity.WARNING
                
        # Get severity levels in order of increasing severity
        severity_levels = [
            StrategyErrorSeverity.INFO,
            StrategyErrorSeverity.WARNING,
            StrategyErrorSeverity.ERROR,
            StrategyErrorSeverity.CRITICAL
        ]
        
        # Find minimum severity level index
        min_idx = severity_levels.index(min_severity)
        
        # Check if there are errors with at least the minimum severity
        for error in self._errors:
            error_idx = severity_levels.index(error.severity)
            if error_idx >= min_idx:
                return True
                
        return False
        
    def clear_errors(self) -> None:
        """Clear all errors."""
        self._errors.clear()
        
    def get_error_counts(self) -> Dict[str, int]:
        """
        Get counts of errors by category and severity.
        
        Returns:
            Dictionary with error counts
        """
        counts = {
            "total": len(self._errors),
            "by_category": {},
            "by_severity": {},
            "recoverable": 0,
            "non_recoverable": 0
        }
        
        # Count by category
        for category in StrategyErrorCategory:
            counts["by_category"][category.value] = 0
            
        # Count by severity
        for severity in StrategyErrorSeverity:
            counts["by_severity"][severity.value] = 0
            
        # Count errors
        for error in self._errors:
            counts["by_category"][error.category.value] += 1
            counts["by_severity"][error.severity.value] += 1
            
            if error.recoverable:
                counts["recoverable"] += 1
            else:
                counts["non_recoverable"] += 1
                
        return counts
        
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all errors encountered.
        
        Returns:
            Dictionary with error summary
        """
        summary = {
            "counts": self.get_error_counts(),
            "errors": [error.to_dict() for error in self._errors]
        }
        
        return summary
        
    # Common error handlers
    def _retry_with_backoff(self, error: StrategyError) -> bool:
        """
        Retry with exponential backoff.
        
        Args:
            error: The error to retry
            
        Returns:
            True if the error was handled and should be retried, False otherwise
        """
        import time
        
        # Calculate backoff delay
        delay = 2 ** error.retry_count  # Exponential backoff
        
        # Log retry attempt
        logger.info(f"Retrying after error: {error}. Attempt {error.retry_count + 1}, waiting {delay}s")
        
        # Increment retry count
        error.increment_retry()
        
        # Wait for backoff period
        time.sleep(delay)
        
        return True
        
    def _handle_rate_limit(self, error: StrategyError) -> bool:
        """
        Handle rate limit errors.
        
        Args:
            error: The rate limit error
            
        Returns:
            True if the error was handled and should be retried, False otherwise
        """
        import time
        
        # Default wait time
        wait_time = 60  # 1 minute
        
        # Check if the error context has a retry-after or similar
        if error.context_data and "retry_after" in error.context_data:
            wait_time = error.context_data["retry_after"]
            
        # Log wait
        logger.info(f"Rate limited. Waiting {wait_time}s before retry. Error: {error}")
        
        # Increment retry count
        error.increment_retry()
        
        # Wait
        time.sleep(wait_time)
        
        return True


# Utility error handling functions that can be used with error policies

def log_error(error: StrategyError) -> bool:
    """
    Log an error with appropriate severity.
    
    Args:
        error: The error to log
        
    Returns:
        True always (indicating error was handled)
    """
    if error.severity == StrategyErrorSeverity.INFO:
        logger.info(str(error))
    elif error.severity == StrategyErrorSeverity.WARNING:
        logger.warning(str(error))
    elif error.severity == StrategyErrorSeverity.ERROR:
        logger.error(str(error))
    elif error.severity == StrategyErrorSeverity.CRITICAL:
        logger.critical(str(error))
        
    return True

def retry_on_network_error(error: StrategyError) -> bool:
    """
    Retry logic for network errors.
    
    Args:
        error: The network error
        
    Returns:
        True if the error should be retried, False otherwise
    """
    import time
    
    # Only retry network errors
    if error.category != StrategyErrorCategory.NETWORK:
        return False
        
    # Calculate delay - simple backoff
    delay = 1 + error.retry_count
    
    # Log retry
    logger.info(f"Network error: {error}. Retrying in {delay}s")
    
    # Increment retry count
    error.increment_retry()
    
    # Wait
    time.sleep(delay)
    
    return True

def wait_on_rate_limit(error: StrategyError) -> bool:
    """
    Wait and retry on rate limit errors.
    
    Args:
        error: The rate limit error
        
    Returns:
        True if the error should be retried, False otherwise
    """
    import time
    
    # Only handle rate limit errors
    if error.category != StrategyErrorCategory.RATE_LIMIT:
        return False
        
    # Parse retry-after if available
    wait_time = 60  # Default: 1 minute
    
    if error.context_data and "retry_after" in error.context_data:
        wait_time = error.context_data["retry_after"]
        
    # Log wait
    logger.info(f"Rate limited. Waiting {wait_time}s before retry.")
    
    # Increment retry count
    error.increment_retry()
    
    # Wait
    time.sleep(wait_time)
    
    return True

def circuit_breaker(error: StrategyError) -> bool:
    """
    Implement circuit breaker pattern for handling cascading failures.
    
    This function can be used with a shared state to implement the circuit breaker pattern:
    - Closed: Allow operations to proceed
    - Open: Block operations for a cooldown period
    - Half-open: Allow a limited number of operations to test if the issue is resolved
    
    Args:
        error: The error to handle
        
    Returns:
        True if the operation should proceed, False if it should be blocked
    """
    # This would need to be integrated with a circuit breaker implementation
    # Here's a simple placeholder
    return False

def fallback_to_cache(error: StrategyError) -> bool:
    """
    Fall back to cached data when live data cannot be retrieved.
    
    Args:
        error: The error to handle
        
    Returns:
        True if fallback succeeded, False otherwise
    """
    # This would need to be integrated with a caching system
    # Here's a simple placeholder
    return False