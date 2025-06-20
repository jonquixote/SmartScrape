"""
Enhanced Error Handling and Logging Infrastructure for SmartScrape

This module provides comprehensive error handling, logging, and debugging
capabilities for the extraction pipeline as outlined in Phase 5.
"""

import logging
import time
import traceback
import json
import uuid
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import asyncio
import sys
import os

# Configure structured logging
class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class ExtractionLogEntry:
    """Structured log entry for extraction operations."""
    timestamp: str
    component: str
    operation: str
    url: Optional[str]
    method: str
    success: bool
    result_size: int
    execution_time: float
    error: Optional[str]
    metadata: Dict[str, Any]

class ExtractPipelineLogger:
    """
    Comprehensive logging system for extraction pipeline with structured logging,
    error tracking, and performance monitoring.
    """
    
    def __init__(self, component_name: str, log_level: str = "INFO"):
        self.component = component_name
        self.logger = logging.getLogger(f"extraction.{component_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Structured logging storage
        self.log_entries: List[ExtractionLogEntry] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        
        # Setup formatter if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    @contextmanager
    def log_extraction_operation(self, url: str, method: str, metadata: Dict[str, Any] = None):
        """Context manager for logging extraction operations with timing."""
        operation_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        self.log_extraction_attempt(url, method, metadata or {}, operation_id)
        
        try:
            yield operation_id
            execution_time = time.time() - start_time
            self.log_extraction_success(url, method, execution_time, operation_id)
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_extraction_error(url, method, str(e), execution_time, operation_id)
            raise
    
    def log_extraction_attempt(self, url: str, method: str, metadata: Dict[str, Any] = None, operation_id: str = None):
        """Log the start of an extraction attempt."""
        extra_info = {
            "url": url,
            "method": method,
            "operation_id": operation_id,
            "metadata": metadata or {}
        }
        
        self.logger.info(
            f"[{self.component}] Starting {method} extraction for {url} [ID: {operation_id}]",
            extra=extra_info
        )
    
    def log_extraction_success(self, url: str, method: str, execution_time: float, 
                             operation_id: str = None, result_size: int = 0):
        """Log successful extraction with metrics."""
        # Record performance metric
        if method not in self.performance_metrics:
            self.performance_metrics[method] = []
        self.performance_metrics[method].append(execution_time)
        
        # Create structured log entry
        entry = ExtractionLogEntry(
            timestamp=datetime.now().isoformat(),
            component=self.component,
            operation="extraction",
            url=url,
            method=method,
            success=True,
            result_size=result_size,
            execution_time=execution_time,
            error=None,
            metadata={"operation_id": operation_id}
        )
        self.log_entries.append(entry)
        
        self.logger.info(
            f"[{self.component}] ✅ {method} extraction successful for {url} "
            f"({result_size} items in {execution_time:.2f}s) [ID: {operation_id}]"
        )
    
    def log_extraction_error(self, url: str, method: str, error: str, 
                           execution_time: float, operation_id: str = None):
        """Log extraction error with details."""
        # Track error counts
        error_key = f"{method}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Create structured log entry
        entry = ExtractionLogEntry(
            timestamp=datetime.now().isoformat(),
            component=self.component,
            operation="extraction",
            url=url,
            method=method,
            success=False,
            result_size=0,
            execution_time=execution_time,
            error=error,
            metadata={"operation_id": operation_id}
        )
        self.log_entries.append(entry)
        
        self.logger.error(
            f"[{self.component}] ❌ {method} extraction failed for {url} "
            f"after {execution_time:.2f}s: {error} [ID: {operation_id}]"
        )
    
    def log_fallback_trigger(self, url: str, primary_method: str, fallback_method: str, 
                           reason: str, operation_id: str = None):
        """Log when fallback extraction is triggered."""
        self.logger.warning(
            f"[{self.component}] 🔄 Triggering fallback from {primary_method} to {fallback_method} "
            f"for {url}: {reason} [ID: {operation_id}]"
        )
    
    def log_result_validation(self, url: str, valid: bool, issues: List[str], operation_id: str = None):
        """Log result validation outcome."""
        if valid:
            self.logger.info(f"[{self.component}] ✓ Result validation passed for {url} [ID: {operation_id}]")
        else:
            self.logger.warning(
                f"[{self.component}] ⚠️  Result validation failed for {url}: {', '.join(issues)} [ID: {operation_id}]"
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all extraction methods."""
        summary = {}
        for method, times in self.performance_metrics.items():
            if times:
                summary[method] = {
                    "total_calls": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_time": sum(times)
                }
        return summary
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary and statistics."""
        total_errors = sum(self.error_counts.values())
        return {
            "total_errors": total_errors,
            "error_breakdown": dict(self.error_counts),
            "top_errors": sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def export_logs(self, format_type: str = "json") -> str:
        """Export logs in specified format."""
        if format_type.lower() == "json":
            return json.dumps([asdict(entry) for entry in self.log_entries], indent=2)
        elif format_type.lower() == "csv":
            # Simple CSV export
            lines = ["timestamp,component,operation,url,method,success,result_size,execution_time,error"]
            for entry in self.log_entries:
                lines.append(f"{entry.timestamp},{entry.component},{entry.operation},"
                           f"{entry.url},{entry.method},{entry.success},{entry.result_size},"
                           f"{entry.execution_time},{entry.error or ''}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

class ResultValidator:
    """
    Validates extraction results for quality and completeness.
    """
    
    @staticmethod
    def validate_extraction_result(result: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate extraction results and return issues found.
        
        Args:
            result: The extraction result to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check basic structure
        if not isinstance(result, dict):
            issues.append("Result is not a dictionary")
            return False, issues
        
        # Check required fields
        required_fields = ['success', 'data']
        for field in required_fields:
            if field not in result:
                issues.append(f"Missing required field: {field}")
        
        # Check success flag consistency
        if result.get('success') == True:
            if not result.get('data'):
                issues.append("Success=True but no data provided")
            elif isinstance(result['data'], list) and len(result['data']) == 0:
                issues.append("Success=True but data list is empty")
        
        # Check data quality
        if result.get('data'):
            data = result['data']
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if not isinstance(item, dict):
                        issues.append(f"Data item {i} is not a dictionary")
                    elif len(item) == 0:
                        issues.append(f"Data item {i} is empty")
                    elif 'url' not in item and 'content' not in item:
                        issues.append(f"Data item {i} missing both 'url' and 'content' fields")
        
        # Check for common quality indicators
        if result.get('success') and result.get('data'):
            data = result['data']
            if isinstance(data, list) and len(data) > 0:
                # Check if content seems meaningful
                first_item = data[0]
                if isinstance(first_item, dict):
                    content_fields = ['content', 'text', 'title', 'description']
                    has_content = any(field in first_item and 
                                    isinstance(first_item[field], str) and 
                                    len(first_item[field].strip()) > 10 
                                    for field in content_fields)
                    if not has_content:
                        issues.append("Data items appear to lack meaningful content")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def assess_content_quality(content: str) -> Dict[str, Any]:
        """
        Assess the quality of extracted content.
        
        Args:
            content: Text content to assess
            
        Returns:
            Quality assessment metrics
        """
        if not isinstance(content, str):
            return {"quality_score": 0, "issues": ["Content is not a string"]}
        
        content = content.strip()
        if not content:
            return {"quality_score": 0, "issues": ["Content is empty"]}
        
        issues = []
        quality_score = 100
        
        # Length checks
        if len(content) < 50:
            issues.append("Content is very short")
            quality_score -= 30
        elif len(content) < 200:
            issues.append("Content is short")
            quality_score -= 15
        
        # Meaningful content checks
        word_count = len(content.split())
        if word_count < 10:
            issues.append("Very few words")
            quality_score -= 25
        
        # Check for common extraction artifacts
        artifacts = ["javascript", "loading", "please enable", "404", "error"]
        for artifact in artifacts:
            if artifact.lower() in content.lower():
                issues.append(f"Contains potential extraction artifact: {artifact}")
                quality_score -= 15
        
        # Check content diversity (not just repeated patterns)
        unique_words = len(set(content.lower().split()))
        if word_count > 0 and unique_words / word_count < 0.3:
            issues.append("Low word diversity - possible repeated content")
            quality_score -= 20
        
        return {
            "quality_score": max(0, quality_score),
            "word_count": word_count,
            "character_count": len(content),
            "unique_word_ratio": unique_words / word_count if word_count > 0 else 0,
            "issues": issues
        }

class ExtractionMonitor:
    """
    Monitors extraction pipeline performance and health.
    """
    
    def __init__(self):
        self.pipeline_metrics = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "fallback_triggers": 0,
            "average_execution_time": 0.0,
            "start_time": time.time()
        }
        self.method_performance = {}
        self.error_tracking = {}
    
    def record_extraction_attempt(self, method: str):
        """Record an extraction attempt."""
        self.pipeline_metrics["total_extractions"] += 1
        if method not in self.method_performance:
            self.method_performance[method] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "total_time": 0.0
            }
        self.method_performance[method]["attempts"] += 1
    
    def record_extraction_success(self, method: str, execution_time: float):
        """Record a successful extraction."""
        self.pipeline_metrics["successful_extractions"] += 1
        self.method_performance[method]["successes"] += 1
        self.method_performance[method]["total_time"] += execution_time
        
        # Update average execution time
        total_time = sum(perf["total_time"] for perf in self.method_performance.values())
        total_attempts = self.pipeline_metrics["total_extractions"]
        self.pipeline_metrics["average_execution_time"] = total_time / total_attempts if total_attempts > 0 else 0
    
    def record_extraction_failure(self, method: str, error: str, execution_time: float):
        """Record a failed extraction."""
        self.pipeline_metrics["failed_extractions"] += 1
        self.method_performance[method]["failures"] += 1
        self.method_performance[method]["total_time"] += execution_time
        
        # Track error types
        error_type = type(error).__name__ if isinstance(error, Exception) else "Unknown"
        if error_type not in self.error_tracking:
            self.error_tracking[error_type] = 0
        self.error_tracking[error_type] += 1
    
    def record_fallback_trigger(self):
        """Record when fallback is triggered."""
        self.pipeline_metrics["fallback_triggers"] += 1
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the extraction pipeline."""
        total = self.pipeline_metrics["total_extractions"]
        success_rate = (self.pipeline_metrics["successful_extractions"] / total * 100) if total > 0 else 0
        fallback_rate = (self.pipeline_metrics["fallback_triggers"] / total * 100) if total > 0 else 0
        
        # Determine health status
        if success_rate >= 80:
            health = "HEALTHY"
        elif success_rate >= 60:
            health = "WARNING"
        else:
            health = "CRITICAL"
        
        return {
            "health_status": health,
            "success_rate": round(success_rate, 2),
            "fallback_rate": round(fallback_rate, 2),
            "total_extractions": total,
            "average_execution_time": round(self.pipeline_metrics["average_execution_time"], 2),
            "uptime_hours": round((time.time() - self.pipeline_metrics["start_time"]) / 3600, 2),
            "top_errors": sorted(self.error_tracking.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def get_method_performance(self) -> Dict[str, Any]:
        """Get performance breakdown by extraction method."""
        performance = {}
        for method, stats in self.method_performance.items():
            attempts = stats["attempts"]
            success_rate = (stats["successes"] / attempts * 100) if attempts > 0 else 0
            avg_time = (stats["total_time"] / attempts) if attempts > 0 else 0
            
            performance[method] = {
                "attempts": attempts,
                "success_rate": round(success_rate, 2),
                "average_time": round(avg_time, 2),
                "total_successes": stats["successes"],
                "total_failures": stats["failures"]
            }
        return performance

# Global monitor instance
_global_monitor = ExtractionMonitor()

def get_extraction_monitor() -> ExtractionMonitor:
    """Get the global extraction monitor."""
    return _global_monitor

def setup_extraction_logging(component_name: str, log_level: str = "INFO") -> ExtractPipelineLogger:
    """
    Set up extraction logging for a component.
    
    Args:
        component_name: Name of the component
        log_level: Logging level
        
    Returns:
        Configured logger instance
    """
    return ExtractPipelineLogger(component_name, log_level)

# Decorator for automatic logging of extraction functions
def log_extraction_operation(logger: ExtractPipelineLogger):
    """Decorator for automatic logging of extraction operations."""
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            # Extract URL from arguments if available
            url = kwargs.get('url') or (args[1] if len(args) > 1 else 'unknown')
            method = func.__name__
            
            with logger.log_extraction_operation(url, method):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            # Extract URL from arguments if available
            url = kwargs.get('url') or (args[1] if len(args) > 1 else 'unknown')
            method = func.__name__
            
            with logger.log_extraction_operation(url, method):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
