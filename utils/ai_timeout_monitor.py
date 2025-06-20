"""
AI timeout monitoring module for SmartScrape.

This module provides utilities for tracking and reporting on AI service timeouts
to help diagnose and fix timeout-related issues.
"""

import logging
import time
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AITimeoutTracker:
    """Tracks and analyzes AI service timeouts to help prevent indefinite hangs"""
    
    def __init__(self, tracking_file=None):
        """
        Initialize the timeout tracker
        
        Args:
            tracking_file: Optional file path to persist timeout data
        """
        self.tracking_file = tracking_file or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "ai_timeout_stats.json"
        )
        
        # Initialize tracking data
        self.timeout_data = {
            "recent_timeouts": [],
            "domain_stats": {},
            "total_timeouts": 0,
            "total_requests": 0,
            "last_updated": None
        }
        
        # Load existing data if available
        self._load_data()
    
    def record_timeout(self, domain: str, operation: str, timeout_seconds: float, context: Optional[Dict] = None):
        """
        Record an AI service timeout
        
        Args:
            domain: Domain or service identifier
            operation: Operation that timed out (e.g., "extraction", "classification")
            timeout_seconds: How many seconds until timeout occurred
            context: Additional context about the operation
        """
        # Update counters
        self.timeout_data["total_timeouts"] += 1
        
        # Add to recent timeouts list (keep last 100)
        timestamp = time.time()
        timeout_entry = {
            "timestamp": timestamp,
            "domain": domain,
            "operation": operation,
            "timeout_seconds": timeout_seconds,
            "context": context or {}
        }
        
        self.timeout_data["recent_timeouts"].insert(0, timeout_entry)
        if len(self.timeout_data["recent_timeouts"]) > 100:
            self.timeout_data["recent_timeouts"].pop()
        
        # Update domain-specific stats
        if domain not in self.timeout_data["domain_stats"]:
            self.timeout_data["domain_stats"][domain] = {
                "timeout_count": 0,
                "request_count": 0,
                "first_timeout": timestamp,
                "last_timeout": timestamp,
                "operations": {}
            }
        
        # Update domain timeout stats
        domain_stats = self.timeout_data["domain_stats"][domain]
        domain_stats["timeout_count"] += 1
        domain_stats["last_timeout"] = timestamp
        
        # Update operation-specific stats
        if operation not in domain_stats["operations"]:
            domain_stats["operations"][operation] = {
                "timeout_count": 0,
                "avg_timeout_seconds": 0
            }
        
        # Update operation timeout stats
        op_stats = domain_stats["operations"][operation]
        old_count = op_stats["timeout_count"]
        old_avg = op_stats["avg_timeout_seconds"]
        
        # Update average timeout (weighted)
        if old_count == 0:
            op_stats["avg_timeout_seconds"] = timeout_seconds
        else:
            op_stats["avg_timeout_seconds"] = (old_avg * old_count + timeout_seconds) / (old_count + 1)
        
        op_stats["timeout_count"] += 1
        
        # Update last updated timestamp
        self.timeout_data["last_updated"] = timestamp
        
        # Log the timeout
        logger.warning(
            f"AI timeout recorded: {domain} {operation} after {timeout_seconds:.1f}s "
            f"(total: {self.timeout_data['total_timeouts']})"
        )
        
        # Save updated data
        self._save_data()
    
    def record_request(self, domain: str, operation: str):
        """
        Record an AI service request (regardless of outcome)
        
        Args:
            domain: Domain or service identifier
            operation: Operation type
        """
        # Update counters
        self.timeout_data["total_requests"] += 1
        
        # Update domain-specific stats
        if domain not in self.timeout_data["domain_stats"]:
            self.timeout_data["domain_stats"][domain] = {
                "timeout_count": 0,
                "request_count": 0,
                "first_timeout": None,
                "last_timeout": None,
                "operations": {}
            }
        
        # Update domain request count
        self.timeout_data["domain_stats"][domain]["request_count"] += 1
        
        # Update last updated timestamp
        self.timeout_data["last_updated"] = time.time()
        
        # Save periodically (every 50 requests)
        if self.timeout_data["total_requests"] % 50 == 0:
            self._save_data()
    
    def get_timeout_rate(self, domain: Optional[str] = None, 
                        window_seconds: int = 3600) -> float:
        """
        Get the AI service timeout rate over the specified time window
        
        Args:
            domain: Optional domain to get rate for (None for all domains)
            window_seconds: Time window in seconds (default: 1 hour)
        
        Returns:
            Timeout rate as a percentage (0-100)
        """
        now = time.time()
        window_start = now - window_seconds
        
        timeouts_in_window = 0
        total_in_window = 0
        
        # Filter for the specific domain if provided
        for timeout in self.timeout_data["recent_timeouts"]:
            if timeout["timestamp"] >= window_start:
                if domain is None or timeout["domain"] == domain:
                    timeouts_in_window += 1
        
        # If no domain specified, use total requests
        if domain is None:
            # Estimate total requests in window based on overall rate
            if self.timeout_data["total_requests"] > 0:
                total_time = now - self.timeout_data.get("first_request_time", now)
                if total_time > 0:
                    requests_per_second = self.timeout_data["total_requests"] / total_time
                    total_in_window = requests_per_second * min(window_seconds, total_time)
                else:
                    total_in_window = self.timeout_data["total_requests"]
            else:
                return 0.0
        else:
            # For specific domain, use domain request count
            if domain in self.timeout_data["domain_stats"]:
                domain_stats = self.timeout_data["domain_stats"][domain]
                if domain_stats["request_count"] > 0:
                    # Estimate based on domain's request history
                    first_request = domain_stats.get("first_request_time", now)
                    domain_time = now - first_request
                    if domain_time > 0:
                        requests_per_second = domain_stats["request_count"] / domain_time
                        total_in_window = requests_per_second * min(window_seconds, domain_time)
                    else:
                        total_in_window = domain_stats["request_count"]
                else:
                    return 0.0
            else:
                return 0.0
        
        # Calculate rate (avoid division by zero)
        if total_in_window > 0:
            return (timeouts_in_window / total_in_window) * 100.0
        return 0.0
    
    def get_timeout_suggestions(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Suggest optimal timeout settings based on historical data
        
        Args:
            domain: Optional domain to get suggestions for
            
        Returns:
            Dictionary with suggested timeout settings
        """
        suggestions = {
            "global_timeout": 90,  # Default
            "ai_extraction_timeout": 45,
            "ai_response_timeout": 25,
            "explanation": "Default timeout settings"
        }
        
        # Get overall timeout stats
        timeout_rate = self.get_timeout_rate(domain)
        
        # Calculate 95th percentile timeout time if we have data
        p95_timeout = None
        timeout_times = []
        
        for timeout in self.timeout_data["recent_timeouts"]:
            if domain is None or timeout["domain"] == domain:
                timeout_times.append(timeout["timeout_seconds"])
        
        if timeout_times:
            timeout_times.sort()
            if len(timeout_times) >= 20:
                p95_index = int(len(timeout_times) * 0.95)
                p95_timeout = timeout_times[p95_index]
        
        # Adjust suggestions based on timeout rate
        if timeout_rate > 50:
            # Very high timeout rate - use more aggressive timeouts
            suggestions["global_timeout"] = 60
            suggestions["ai_extraction_timeout"] = 30
            suggestions["ai_response_timeout"] = 15
            suggestions["explanation"] = f"Aggressive settings due to high timeout rate ({timeout_rate:.1f}%)"
        elif timeout_rate > 20:
            # High timeout rate - use moderately aggressive timeouts
            suggestions["global_timeout"] = 75
            suggestions["ai_extraction_timeout"] = 35
            suggestions["ai_response_timeout"] = 20
            suggestions["explanation"] = f"Moderately aggressive settings due to elevated timeout rate ({timeout_rate:.1f}%)"
        elif p95_timeout is not None:
            # Use data-driven approach if we have enough data
            suggestions["ai_response_timeout"] = max(10, int(p95_timeout * 1.2))
            suggestions["ai_extraction_timeout"] = max(30, int(suggestions["ai_response_timeout"] * 2))
            suggestions["global_timeout"] = max(60, int(suggestions["ai_extraction_timeout"] * 2))
            suggestions["explanation"] = f"Data-driven settings based on timeout history (95th percentile: {p95_timeout:.1f}s)"
        
        return suggestions
    
    def get_report(self) -> Dict[str, Any]:
        """
        Get a detailed report of timeout statistics
        
        Returns:
            Dictionary with timeout statistics and analysis
        """
        now = time.time()
        
        # Calculate timeout rates for different time windows
        hour_rate = self.get_timeout_rate(window_seconds=3600)
        day_rate = self.get_timeout_rate(window_seconds=86400)
        
        # Get domain-specific stats
        domain_timeout_rates = {}
        for domain, stats in self.timeout_data["domain_stats"].items():
            if stats["request_count"] > 0:
                timeout_rate = (stats["timeout_count"] / stats["request_count"]) * 100.0
                domain_timeout_rates[domain] = timeout_rate
        
        # Sort domains by timeout rate
        problem_domains = sorted(
            [(d, r) for d, r in domain_timeout_rates.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get suggested timeout settings
        suggestions = self.get_timeout_suggestions()
        
        return {
            "total_timeouts": self.timeout_data["total_timeouts"],
            "total_requests": self.timeout_data["total_requests"],
            "overall_timeout_rate": (self.timeout_data["total_timeouts"] / max(1, self.timeout_data["total_requests"])) * 100.0,
            "hourly_timeout_rate": hour_rate,
            "daily_timeout_rate": day_rate,
            "problem_domains": problem_domains[:5],  # Top 5 problem domains
            "recent_timeouts": self.timeout_data["recent_timeouts"][:10],  # Last 10 timeouts
            "suggested_settings": suggestions,
            "last_updated": self.timeout_data["last_updated"]
        }
    
    def _load_data(self):
        """Load timeout tracking data from disk"""
        try:
            if os.path.exists(self.tracking_file):
                with open(self.tracking_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.timeout_data = data
                        logger.info(f"Loaded timeout tracking data: {self.timeout_data['total_timeouts']} timeouts recorded")
        except Exception as e:
            logger.error(f"Error loading timeout tracking data: {e}")
    
    def _save_data(self):
        """Save timeout tracking data to disk"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
            
            with open(self.tracking_file, 'w') as f:
                json.dump(self.timeout_data, f)
        except Exception as e:
            logger.error(f"Error saving timeout tracking data: {e}")

# Singleton instance
timeout_tracker = AITimeoutTracker()

def record_ai_timeout(domain, operation, timeout_seconds, context=None):
    """
    Record an AI service timeout (convenience function)
    
    Args:
        domain: Domain or service identifier
        operation: Operation that timed out
        timeout_seconds: How many seconds until timeout occurred
        context: Additional context about the operation
    """
    timeout_tracker.record_timeout(domain, operation, timeout_seconds, context)

def record_ai_request(domain, operation):
    """
    Record an AI service request (convenience function)
    
    Args:
        domain: Domain or service identifier
        operation: Operation type
    """
    timeout_tracker.record_request(domain, operation)

def get_suggested_timeout_settings(domain=None):
    """
    Get suggested timeout settings (convenience function)
    
    Args:
        domain: Optional domain to get suggestions for
    
    Returns:
        Dictionary with suggested timeout settings
    """
    return timeout_tracker.get_timeout_suggestions(domain)
