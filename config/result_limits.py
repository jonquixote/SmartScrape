"""
Result Limits Configuration

Centralized configuration for result count management across the SmartScrape system.
Ensures consistent minimum and maximum result counts while allowing user control.
"""

class ResultLimits:
    """Configuration class for result count limits"""
    
    # Default minimum results to return (unless user requests fewer)
    DEFAULT_MIN_RESULTS = 5
    
    # Default target results when no specific count is requested
    DEFAULT_TARGET_RESULTS = 8
    
    # Default maximum results to prevent overwhelming responses
    DEFAULT_MAX_RESULTS = 50
    
    # Absolute minimum (system never returns fewer than this unless no content exists)
    ABSOLUTE_MIN_RESULTS = 3
    
    # Absolute maximum (system never returns more than this regardless of user request)
    ABSOLUTE_MAX_RESULTS = 100
    
    # Maximum URLs to process in search for results
    MAX_URLS_TO_PROCESS = 20
    
    # Maximum attempts to reach minimum result count
    MAX_RETRY_ATTEMPTS = 3

    @classmethod
    def get_result_limits(cls, user_requested: int = None, min_requested: int = None, max_requested: int = None) -> dict:
        """
        Calculate appropriate result limits based on user preferences and system defaults.
        
        Args:
            user_requested: Number of results user specifically requested
            min_requested: Minimum number of results user wants
            max_requested: Maximum number of results user wants
            
        Returns:
            dict: Contains 'min_results', 'target_results', 'max_results'
        """
        # Determine target results
        if user_requested is not None:
            # User specified exact count
            target = max(cls.ABSOLUTE_MIN_RESULTS, min(user_requested, cls.ABSOLUTE_MAX_RESULTS))
        else:
            # Use default target
            target = cls.DEFAULT_TARGET_RESULTS
        
        # Determine minimum results
        if min_requested is not None:
            min_results = max(cls.ABSOLUTE_MIN_RESULTS, min(min_requested, cls.ABSOLUTE_MAX_RESULTS))
        else:
            # Ensure we meet the default minimum unless user specifically requested fewer
            if user_requested is not None and user_requested < cls.DEFAULT_MIN_RESULTS:
                min_results = user_requested
            else:
                min_results = cls.DEFAULT_MIN_RESULTS
        
        # Determine maximum results
        if max_requested is not None:
            max_results = max(target, min(max_requested, cls.ABSOLUTE_MAX_RESULTS))
        elif user_requested is not None:
            # If user requested specific count, allow some buffer above it
            max_results = min(user_requested + 5, cls.ABSOLUTE_MAX_RESULTS)
        else:
            max_results = cls.DEFAULT_MAX_RESULTS
        
        return {
            'min_results': min_results,
            'target_results': target,
            'max_results': max_results
        }
    
    @classmethod
    def should_fetch_more_results(cls, current_count: int, limits: dict) -> bool:
        """
        Determine if the system should attempt to fetch more results.
        
        Args:
            current_count: Number of results currently found
            limits: Result limits dict from get_result_limits()
            
        Returns:
            bool: True if more results should be fetched
        """
        return current_count < limits['min_results']
    
    @classmethod
    def should_stop_fetching(cls, current_count: int, limits: dict) -> bool:
        """
        Determine if the system should stop fetching more results.
        
        Args:
            current_count: Number of results currently found
            limits: Result limits dict from get_result_limits()
            
        Returns:
            bool: True if fetching should stop
        """
        return current_count >= limits['max_results']
    
    @classmethod
    def apply_final_limits(cls, results: list, limits: dict) -> list:
        """
        Apply final result count limits to a list of results.
        
        Args:
            results: List of results to limit
            limits: Result limits dict from get_result_limits()
            
        Returns:
            list: Limited results list
        """
        if not results:
            return results
        
        # Don't exceed maximum
        if len(results) > limits['max_results']:
            results = results[:limits['max_results']]
        
        return results
