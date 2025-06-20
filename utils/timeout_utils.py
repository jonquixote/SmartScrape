"""
Utility functions for timeout handling across the SmartScrape system.
This module provides centralized timeout management to ensure consistent
timeout behavior across different components.
"""

import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def check_approaching_deadline(progress_data: Dict[str, Any], 
                              deadline_key: str = 'deadline', 
                              buffer_seconds: float = 2.0) -> bool:
    """
    Check if we're approaching a deadline in the progress tracking data.
    
    Args:
        progress_data: Dictionary containing progress tracking data
        deadline_key: Key in the progress data containing the deadline timestamp
        buffer_seconds: How many seconds before the deadline to consider "approaching"
        
    Returns:
        True if we're approaching the deadline, False otherwise
    """
    if not progress_data or deadline_key not in progress_data:
        return False
        
    now = time.time()
    deadline = progress_data.get(deadline_key)
    
    if not deadline:
        return False
        
    time_remaining = deadline - now
    
    # If we're within the buffer time of the deadline
    if time_remaining <= buffer_seconds:
        return True
        
    return False

def calculate_adaptive_timeout(base_timeout: float, 
                              retry_attempt: int = 0,
                              min_timeout: float = 5.0,
                              reduction_per_retry: float = 5.0) -> float:
    """
    Calculate an adaptive timeout value based on retry attempts.
    Timeouts get progressively shorter with each retry.
    
    Args:
        base_timeout: The base timeout value in seconds
        retry_attempt: Current retry attempt (0 = first try)
        min_timeout: Minimum timeout value in seconds
        reduction_per_retry: How many seconds to reduce per retry
        
    Returns:
        Adjusted timeout value in seconds
    """
    if retry_attempt <= 0:
        return base_timeout
        
    adjusted_timeout = base_timeout - (retry_attempt * reduction_per_retry)
    return max(min_timeout, adjusted_timeout)

def get_remaining_time(progress_data: Dict[str, Any], 
                      deadline_key: str = 'deadline') -> Optional[float]:
    """
    Get the remaining time until a deadline in seconds.
    
    Args:
        progress_data: Dictionary containing progress tracking data
        deadline_key: Key in the progress data containing the deadline timestamp
        
    Returns:
        Remaining time in seconds, or None if no deadline found
    """
    if not progress_data or deadline_key not in progress_data:
        return None
        
    deadline = progress_data.get(deadline_key)
    
    if not deadline:
        return None
        
    remaining_time = deadline - time.time()
    
    # Don't return negative times
    return max(0.0, remaining_time)
