"""
Enhanced Error Handler for SmartScrape

This module provides comprehensive error classification and handling strategies
for different types of extraction errors.
"""

import logging
import traceback
import time
from enum import Enum
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Classification of extraction error types"""
    NETWORK = "network"
    PARSING = "parsing"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    CONTENT_BLOCKED = "content_blocked"
    JAVASCRIPT_REQUIRED = "javascript_required"
    MEMORY_ERROR = "memory_error"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for an error"""
    url: str
    strategy: str
    timestamp: float
    attempt_number: int
    previous_errors: List[str] = None
    
    def __post_init__(self):
        if self.previous_errors is None:
            self.previous_errors = []

@dataclass
class RecoveryStrategy:
    """Strategy for recovering from an error"""
    retry: bool = False
    backoff_type: str = "fixed"  # "fixed", "exponential", "linear"
    delay: float = 1.0
    max_retries: int = 3
    fallback_strategy: str = None
    use_cache: bool = True
    rotate_user_agent: bool = False
    use_proxy: bool = False
    log_only: bool = False

class ErrorHandler:
    """Comprehensive error handling system"""
    
    def __init__(self):
        self.error_patterns = {
            ErrorType.NETWORK: [
                "connection", "network", "dns", "unreachable", 
                "host", "socket", "refused", "reset"
            ],
            ErrorType.PARSING: [
                "parse", "invalid html", "malformed", "decode", 
                "encoding", "beautifulsoup", "lxml"
            ],
            ErrorType.TIMEOUT: [
                "timeout", "timed out", "deadline exceeded", "time limit",
                "asyncio.timeout", "read timeout"
            ],
            ErrorType.RATE_LIMIT: [
                "rate limit", "too many requests", "429", "throttle",
                "quota exceeded"
            ],
            ErrorType.AUTHENTICATION: [
                "unauthorized", "403", "authentication", "forbidden",
                "login required", "access denied"
            ],
            ErrorType.CONTENT_BLOCKED: [
                "blocked", "captcha", "cloudflare", "bot detected",
                "security check", "access denied"
            ],
            ErrorType.JAVASCRIPT_REQUIRED: [
                "javascript", "js required", "dynamic", "spa",
                "react", "vue", "angular"
            ],
            ErrorType.MEMORY_ERROR: [
                "memory", "out of memory", "memoryerror", "heap",
                "allocation failed"
            ],
            ErrorType.CONFIGURATION: [
                "configuration", "config", "setting", "missing",
                "not found", "import error"
            ]
        }
        
        self.recovery_strategies = {
            ErrorType.NETWORK: RecoveryStrategy(
                retry=True,
                backoff_type="exponential",
                delay=2.0,
                max_retries=3,
                fallback_strategy="requests_html"
            ),
            ErrorType.PARSING: RecoveryStrategy(
                retry=True,
                backoff_type="fixed",
                delay=1.0,
                max_retries=2,
                fallback_strategy="trafilatura"
            ),
            ErrorType.TIMEOUT: RecoveryStrategy(
                retry=True,
                backoff_type="linear",
                delay=5.0,
                max_retries=2,
                fallback_strategy="playwright"
            ),
            ErrorType.RATE_LIMIT: RecoveryStrategy(
                retry=True,
                backoff_type="exponential",
                delay=60.0,
                max_retries=2,
                rotate_user_agent=True
            ),
            ErrorType.AUTHENTICATION: RecoveryStrategy(
                retry=False,
                log_only=True,
                fallback_strategy="public_api"
            ),
            ErrorType.CONTENT_BLOCKED: RecoveryStrategy(
                retry=True,
                backoff_type="exponential",
                delay=10.0,
                max_retries=2,
                rotate_user_agent=True,
                use_proxy=True,
                fallback_strategy="playwright"
            ),
            ErrorType.JAVASCRIPT_REQUIRED: RecoveryStrategy(
                retry=False,
                fallback_strategy="playwright",
                use_cache=False
            ),
            ErrorType.MEMORY_ERROR: RecoveryStrategy(
                retry=True,
                backoff_type="fixed",
                delay=30.0,
                max_retries=1,
                log_only=True
            ),
            ErrorType.CONFIGURATION: RecoveryStrategy(
                retry=False,
                log_only=True
            ),
            ErrorType.UNKNOWN: RecoveryStrategy(
                retry=True,
                backoff_type="fixed",
                delay=5.0,
                max_retries=1,
                fallback_strategy="universal_fallback"
            )
        }
        
        self.error_history: Dict[str, List[Dict]] = {}
    
    def classify_error(self, error: Exception, context: ErrorContext = None) -> ErrorType:
        """Classify error type for appropriate handling"""
        error_msg = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        # Check error message patterns
        for error_type, patterns in self.error_patterns.items():
            if any(pattern in error_msg for pattern in patterns):
                return error_type
        
        # Check error type patterns
        type_patterns = {
            ErrorType.NETWORK: ["connectionerror", "httperror", "sslerror"],
            ErrorType.TIMEOUT: ["timeouterror", "asyncio.timeouterror"],
            ErrorType.MEMORY_ERROR: ["memoryerror", "overflowlocation"],
            ErrorType.PARSING: ["parseerror", "decodeerror", "unicodeerror"]
        }
        
        for error_type, patterns in type_patterns.items():
            if any(pattern in error_type_name for pattern in patterns):
                return error_type
        
        return ErrorType.UNKNOWN
    
    def get_recovery_strategy(self, error_type: ErrorType, context: ErrorContext = None) -> RecoveryStrategy:
        """Get recommended recovery strategy"""
        base_strategy = self.recovery_strategies.get(error_type, self.recovery_strategies[ErrorType.UNKNOWN])
        
        # Adjust strategy based on context
        if context:
            # Reduce retries if we've had many failures
            if context.attempt_number > 3:
                base_strategy.max_retries = max(0, base_strategy.max_retries - 1)
            
            # If we've seen similar errors before, adjust strategy
            url_errors = self.error_history.get(context.url, [])
            recent_errors = [e for e in url_errors if time.time() - e['timestamp'] < 3600]  # Last hour
            
            if len(recent_errors) > 5:
                # Too many recent errors, be more conservative
                base_strategy.delay *= 2
                base_strategy.max_retries = min(1, base_strategy.max_retries)
        
        return base_strategy
    
    def record_error(self, error: Exception, context: ErrorContext, error_type: ErrorType):
        """Record error for analysis and pattern detection"""
        if context.url not in self.error_history:
            self.error_history[context.url] = []
        
        error_record = {
            'timestamp': context.timestamp,
            'error_type': error_type.value,
            'error_message': str(error),
            'strategy': context.strategy,
            'attempt_number': context.attempt_number,
            'traceback': traceback.format_exc()
        }
        
        self.error_history[context.url].append(error_record)
        
        # Keep only recent errors (last 24 hours)
        cutoff_time = time.time() - 86400
        self.error_history[context.url] = [
            e for e in self.error_history[context.url] 
            if e['timestamp'] > cutoff_time
        ]
        
        # Log error with appropriate level
        if error_type in [ErrorType.MEMORY_ERROR, ErrorType.CONFIGURATION]:
            logger.error(f"Critical error for {context.url}: {error_type.value} - {error}")
        elif error_type in [ErrorType.NETWORK, ErrorType.TIMEOUT]:
            logger.warning(f"Recoverable error for {context.url}: {error_type.value} - {error}")
        else:
            logger.info(f"Handled error for {context.url}: {error_type.value} - {error}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        total_errors = sum(len(errors) for errors in self.error_history.values())
        
        error_type_counts = {}
        strategy_failure_counts = {}
        url_failure_counts = {}
        
        for url, errors in self.error_history.items():
            url_failure_counts[url] = len(errors)
            
            for error in errors:
                error_type = error['error_type']
                strategy = error['strategy']
                
                error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
                strategy_failure_counts[strategy] = strategy_failure_counts.get(strategy, 0) + 1
        
        return {
            'total_errors': total_errors,
            'error_types': error_type_counts,
            'strategy_failures': strategy_failure_counts,
            'most_problematic_urls': sorted(
                url_failure_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            'most_common_error_type': max(error_type_counts.items(), key=lambda x: x[1])[0] if error_type_counts else None
        }
    
    def clear_error_history(self, url: str = None):
        """Clear error history for a specific URL or all URLs"""
        if url:
            self.error_history.pop(url, None)
            logger.info(f"Cleared error history for {url}")
        else:
            self.error_history.clear()
            logger.info("Cleared all error history")

# Global error handler instance
error_handler = ErrorHandler()
