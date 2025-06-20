"""
Strategy interface module defining the abstract classes for all SmartScrape strategies.

This module provides the foundation for the Strategy Pattern implementation in SmartScrape.
It defines abstract base classes that all concrete strategies must implement, ensuring
a consistent interface across different scraping strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable, TypeVar, Type, Set, Union
import logging

# Type variable for better type hinting
T = TypeVar('T')

# Forward declaration for type hinting - avoids circular import
# from strategies.core.strategy_context import StrategyContext

class BaseStrategy(ABC):
    """
    Abstract base class that defines the interface for all SmartScrape strategies.
    All concrete strategies must inherit from this class and implement its abstract methods.
    """

    def __init__(self, context: Optional['StrategyContext'] = None):
        """
        Initialize the strategy with an optional context.
        
        The context provides access to shared services and resources.
        Making it optional allows for temporary instantiation in registry.
        
        Args:
            context: The strategy context containing shared services and configuration
        """
        self.context = context
        # Set up logger - if context is available, use its logger, otherwise create a new one
        if context and hasattr(context, 'logger'):
            self.logger = context.logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize results storage
        self._results: List[Dict[str, Any]] = []
        
        # Flag to track if the strategy has been initialized
        self._initialized = False

    @abstractmethod
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute the strategy for the given URL.
        
        This is the main entry point for running a strategy on a specific URL.
        
        Args:
            url: The URL to process
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Optional dictionary with execution results, or None if execution failed
        """
        pass

    @abstractmethod
    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Crawl from a starting URL using this strategy.
        
        This method implements the traversal logic of the strategy.
        
        Args:
            start_url: The URL to start crawling from
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Optional dictionary with crawl results, or None if crawl failed
        """
        pass

    @abstractmethod
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Extract data from HTML content using this strategy.
        
        This method implements the data extraction logic of the strategy.
        
        Args:
            html_content: The HTML content to extract data from
            url: The URL the content was fetched from
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Optional dictionary with extracted data, or None if extraction failed
        """
        pass

    @abstractmethod
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get all results collected by this strategy.
        
        Returns:
            List of dictionaries containing the collected results
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            String name of the strategy
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the strategy-specific configuration.
        
        Args:
            config: Dictionary containing configuration parameters
            
        Returns:
            True if the configuration is valid, False otherwise
        """
        return True

    def can_handle(self, url: str, **kwargs) -> bool:
        """
        Check if the strategy can handle the given URL and context.
        
        This method allows strategies to determine if they are applicable to a given URL.
        
        Args:
            url: The URL to check
            **kwargs: Additional context parameters
            
        Returns:
            True if the strategy can handle the URL, False otherwise
        """
        return True

    def initialize(self) -> None:
        """
        Initialize the strategy.
        
        This method is called before the strategy is used.
        It should set up any resources needed by the strategy.
        """
        self._initialized = True

    def shutdown(self) -> None:
        """
        Perform cleanup and shutdown operations.
        
        This method is called when the strategy is no longer needed.
        It should release any resources held by the strategy.
        """
        self._initialized = False

    def cleanup(self) -> None:
        """
        Clean up resources used by the strategy.
        
        Similar to shutdown but may be called between operations rather than at the end of lifecycle.
        """
        pass

    def pause(self) -> None:
        """
        Pause strategy execution.
        
        This method allows for temporary suspension of strategy execution.
        """
        pass

    def resume(self) -> None:
        """
        Resume strategy execution after pause.
        
        This method allows for resuming a previously paused strategy.
        """
        pass

    def handle_error(self, 
                    error: Optional[Exception] = None, 
                    message: str = "", 
                    category: str = "general", 
                    severity: str = "warning",
                    url: Optional[str] = None,
                    recoverable: bool = True,
                    strategy_name: Optional[str] = None) -> None:
        """
        Handle an error that occurred during strategy execution.
        
        Args:
            error: The exception that was raised
            message: Error message
            category: Category of the error (e.g., 'network', 'parser')
            severity: Severity level ('info', 'warning', 'error', 'critical')
            url: The URL that was being processed when the error occurred
            recoverable: Whether the error is recoverable
            strategy_name: Name of the strategy (defaults to self.name)
        """
        strategy_name = strategy_name or self.name
        
        # Log the error
        log_message = f"{message} - URL: {url}" if url else message
        
        if severity == "info":
            self.logger.info(log_message)
        elif severity == "warning":
            self.logger.warning(log_message)
        elif severity == "error":
            self.logger.error(log_message)
        elif severity == "critical":
            self.logger.critical(log_message)
        
        # Additional error handling could be implemented here
        # For example, storing errors in the context for later analysis
        if hasattr(self.context, 'record_error') and callable(self.context.record_error):
            self.context.record_error(
                strategy_name=strategy_name,
                url=url,
                message=message,
                error=error,
                category=category,
                severity=severity,
                recoverable=recoverable
            )

    def has_errors(self, min_severity: str = "warning") -> bool:
        """
        Check if the strategy has encountered errors of at least the specified severity.
        
        Args:
            min_severity: Minimum severity level to check for ('info', 'warning', 'error', 'critical')
            
        Returns:
            True if there are errors of at least the specified severity, False otherwise
        """
        # This is a placeholder implementation
        # A real implementation would track errors and check their severity
        return False


class WebScrapingStrategy(BaseStrategy):
    """
    Abstract base class for strategies that interact with web pages.
    
    Extends BaseStrategy with web scraping specific functionality.
    """

    def __init__(self, context: Optional['StrategyContext'] = None):
        """
        Initialize the web scraping strategy with a context.
        
        Args:
            context: The strategy context containing shared services
        """
        super().__init__(context)
        
        # Access common services - these will be obtained from the context
        # If the context doesn't have these services, they will be None
        self.url_service = None
        self.html_service = None
        
        if context:
            # Try to get services from context
            try:
                self.url_service = context.get_service("url_service")
                self.html_service = context.get_service("html_service")
            except (AttributeError, KeyError) as e:
                self.logger.warning(f"Failed to get services: {str(e)}")

    def extract_links(self, html_content: str, base_url: str) -> List[Dict[str, Any]]:
        """
        Extract links from HTML content.
        
        Args:
            html_content: The HTML content to extract links from
            base_url: The base URL for resolving relative links
            
        Returns:
            List of dictionaries with link information (url, text, etc.)
        """
        if self.html_service:
            return self.html_service.extract_links(html_content, base_url)
        else:
            self.logger.warning("HTML service not available, cannot extract links")
            return []

    def clean_html(self, html_content: str) -> str:
        """
        Clean HTML content.
        
        Args:
            html_content: The HTML content to clean
            
        Returns:
            Cleaned HTML content
        """
        if self.html_service:
            return self.html_service.clean_html(html_content)
        else:
            self.logger.warning("HTML service not available, returning original HTML")
            return html_content

    def normalize_url(self, url: str, base_url: Optional[str] = None) -> str:
        """
        Normalize a URL.
        
        Args:
            url: The URL to normalize
            base_url: Optional base URL for resolving relative URLs
            
        Returns:
            Normalized URL
        """
        if self.url_service:
            return self.url_service.normalize_url(url, base_url)
        else:
            self.logger.warning("URL service not available, cannot normalize URL")
            return url

    def is_url_allowed(self, url: str) -> bool:
        """
        Check if a URL is allowed to be crawled.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL is allowed, False otherwise
        """
        if self.url_service:
            return self.url_service.is_allowed(url)
        else:
            self.logger.warning("URL service not available, assuming URL is allowed")
            return True

    def extract_main_content(self, html_content: str) -> str:
        """
        Extract the main content from HTML.
        
        Args:
            html_content: The HTML content to extract from
            
        Returns:
            Extracted main content
        """
        if self.html_service:
            return self.html_service.extract_main_content(html_content)
        else:
            self.logger.warning("HTML service not available, returning original HTML")
            return html_content

    def extract_metadata(self, html_content: str) -> Dict[str, Any]:
        """
        Extract metadata from HTML content.
        
        Args:
            html_content: The HTML content to extract metadata from
            
        Returns:
            Dictionary with extracted metadata
        """
        if self.html_service:
            return self.html_service.extract_metadata(html_content)
        else:
            self.logger.warning("HTML service not available, cannot extract metadata")
            return {}

    def _fetch_url(self, url: str, **kwargs) -> Optional[str]:
        """
        Fetch content from a URL.
        
        This is a utility method that handles common concerns like:
        - URL normalization
        - Robots.txt compliance
        - Rate limiting
        - Error handling
        
        Args:
            url: The URL to fetch
            **kwargs: Additional parameters for the fetch operation
            
        Returns:
            HTML content as string, or None if fetch failed
        """
        if not self.url_service:
            self.logger.error("URL service not available, cannot fetch URL")
            return None

        # Normalize URL
        normalized_url = self.normalize_url(url)
        
        # Check if URL is allowed by robots.txt
        if not self.is_url_allowed(normalized_url):
            self.logger.info(f"URL not allowed by robots.txt: {normalized_url}")
            return None
        
        # Actual fetching would be implemented by the concrete strategy
        # This method provides a hook for common pre/post processing
        
        self.logger.debug(f"Fetching URL: {normalized_url}")
        
        # Placeholder for actual fetch
        # In a real implementation, this would use a session manager
        # and follow rate limiting rules
        return None