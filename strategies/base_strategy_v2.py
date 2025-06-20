"""
BaseStrategyV2 Module - Enhanced base implementation of the WebScrapingStrategy interface.

This module provides an improved base strategy implementation that serves as the foundation
for all web scraping strategies in SmartScrape. It implements common functionality and
utilities that are useful across different types of web scraping strategies.
"""

import logging
import time
import asyncio
from typing import Any, Dict, Optional, List, Set, Tuple, Union
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from strategies.core.strategy_interface import WebScrapingStrategy
from strategies.core.strategy_types import StrategyType, StrategyCapability, strategy_metadata

logger = logging.getLogger(__name__)

@strategy_metadata(
    strategy_type=StrategyType.SPECIAL_PURPOSE,
    capabilities={
        StrategyCapability.ROBOTS_TXT_ADHERENCE,
        StrategyCapability.RATE_LIMITING,
        StrategyCapability.PROXY_SUPPORT,
        StrategyCapability.ERROR_HANDLING,
        StrategyCapability.LINK_EXTRACTION,
        StrategyCapability.CONTENT_NORMALIZATION,
    },
    description="Enhanced base strategy providing common web scraping utilities and integrating with core services."
)
class BaseStrategyV2(WebScrapingStrategy):
    """
    Enhanced base strategy implementation that serves as the foundation for all web scraping strategies.
    
    This class:
    1. Implements all abstract methods from WebScrapingStrategy
    2. Provides utilities for common operations like link extraction
    3. Integrates with core services
    4. Implements robust error handling
    5. Provides progress tracking and metrics
    """

    def __init__(self, context=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base strategy.
        
        Args:
            context: The strategy context containing shared services
            config: Optional configuration dictionary
        """
        super().__init__(context)
        
        # Configuration with defaults
        self.config = {
            'max_depth': 2,
            'max_pages': 100,
            'include_external': False,
            'rate_limit': 1.0,  # seconds between requests
            'timeout': 30,      # request timeout in seconds
            'retry_count': 3,   # number of retries for failed requests
            'handle_javascript': False,
            'follow_redirects': True,
            'respect_robots_txt': True,
            'user_agent': 'SmartScrape Bot'
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Initialize tracking
        self.visited_urls: Set[str] = set()
        self.url_scores: Dict[str, float] = {}
        self.start_time = time.time()
        self._results: List[Dict[str, Any]] = []
        self._metrics = {
            'pages_visited': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'errors': [],
            'start_time': self.start_time,
            'total_time': 0
        }
        
        # Domain information for filtering
        self.main_domain = None
        
        # Error tracking
        self.errors: List[Dict[str, Any]] = []
        self.error_urls: Set[str] = set()
        
        # Session state
        self.is_paused = False
    
    @property
    def name(self) -> str:
        """Get the name of the strategy."""
        return "base_v2"

    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute the strategy for the given URL.
        
        Args:
            url: The URL to process
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Optional dictionary with execution results, or None if execution failed
        """
        # Parse and store the main domain if not already set
        if not self.main_domain:
            self.main_domain = urlparse(url).netloc
        
        # Merge config with kwargs
        config = {**self.config, **kwargs}
        
        # Initialize metrics
        self._metrics['start_time'] = time.time()
        
        try:
            # Get HTML content
            html_content = self._fetch_url(url, **config)
            if not html_content:
                self.handle_error(
                    message=f"Failed to fetch URL: {url}",
                    category="network",
                    severity="error",
                    url=url
                )
                return None
            
            # Extract data
            result = self.extract(html_content, url, **config)
            
            # Update metrics
            self._metrics['pages_visited'] += 1
            if result:
                self._metrics['successful_extractions'] += 1
            else:
                self._metrics['failed_extractions'] += 1
            
            return result
            
        except Exception as e:
            self.handle_error(
                error=e,
                message=f"Error executing strategy: {str(e)}",
                category="execution",
                severity="error",
                url=url
            )
            return None
        finally:
            # Update total time
            self._metrics['total_time'] = time.time() - self._metrics['start_time']

    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Crawl from a starting URL using this strategy.
        
        This provides a basic implementation that can be overridden by specialized strategies.
        
        Args:
            start_url: The URL to start crawling from
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Optional dictionary with crawl results, or None if crawl failed
        """
        # Parse and store the main domain
        self.main_domain = urlparse(start_url).netloc
        
        # Merge config with kwargs
        config = {**self.config, **kwargs}
        max_depth = config.get('max_depth', 2)
        max_pages = config.get('max_pages', 100)
        include_external = config.get('include_external', False)
        
        # Initialize variables for tracking
        self.visited_urls = set()
        queue = [{'url': start_url, 'depth': 0}]
        results = []
        
        while queue and len(self.visited_urls) < max_pages:
            # Check if paused
            if self.is_paused:
                time.sleep(1)
                continue
                
            # Get next URL from queue
            current = queue.pop(0)
            url = current['url']
            depth = current['depth']
            
            # Skip if already visited
            if url in self.visited_urls:
                continue
                
            # Add to visited
            self.visited_urls.add(url)
            
            # Execute strategy on URL
            try:
                # Get HTML content
                html_content = self._fetch_url(url, **config)
                if not html_content:
                    continue
                    
                # Extract data
                result = self.extract(html_content, url, **config)
                if result:
                    result['url'] = url
                    result['depth'] = depth
                    results.append(result)
                    
                # Don't extract links if at max depth
                if depth >= max_depth:
                    continue
                    
                # Extract links
                links = self.extract_links(html_content, url)
                
                # Filter and add links to queue
                for link_info in links:
                    link_url = link_info['url']
                    
                    # Skip if already visited or queued
                    if link_url in self.visited_urls or any(item['url'] == link_url for item in queue):
                        continue
                        
                    # Check domain constraints
                    if not include_external:
                        link_domain = urlparse(link_url).netloc
                        if link_domain != self.main_domain:
                            continue
                            
                    # Add to queue
                    queue.append({
                        'url': link_url,
                        'depth': depth + 1
                    })
                    
            except Exception as e:
                self.handle_error(
                    error=e,
                    message=f"Error crawling URL: {str(e)}",
                    category="crawl",
                    severity="error",
                    url=url
                )
                
        # Update results list
        self._results.extend(results)
        
        return {
            'results': results,
            'metrics': self.get_metrics()
        }

    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Extract data from HTML content.
        
        This provides a basic implementation that can be overridden by specialized strategies.
        
        Args:
            html_content: The HTML content to extract data from
            url: The URL the content was fetched from
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Optional dictionary with extracted data, or None if extraction failed
        """
        try:
            # Clean HTML
            cleaned_html = self.clean_html(html_content)
            
            # Use BeautifulSoup for parsing
            soup = BeautifulSoup(cleaned_html, 'html.parser')
            
            # Extract basic metadata
            title = soup.title.text if soup.title else ""
            description = ""
            if soup.find("meta", attrs={"name": "description"}):
                description = soup.find("meta", attrs={"name": "description"}).get("content", "")
                
            # Extract main content
            main_content = self.extract_main_content(cleaned_html)
            main_text = ""
            if main_content:
                soup_content = BeautifulSoup(main_content, 'html.parser')
                main_text = soup_content.get_text(strip=True)
                # Limit to reasonable size
                if len(main_text) > 1000:
                    main_text = main_text[:1000] + "..."
            
            # Extract links for reference
            links = self.extract_links(cleaned_html, url)
            
            # Create result
            result = {
                'url': url,
                'title': title,
                'description': description,
                'main_content_sample': main_text,
                'link_count': len(links),
                'metadata': self.extract_metadata(cleaned_html),
                'timestamp': time.time()
            }
            
            # Add to results
            self._results.append(result)
            
            return result
            
        except Exception as e:
            self.handle_error(
                error=e,
                message=f"Error extracting data: {str(e)}",
                category="extraction",
                severity="error",
                url=url
            )
            return None

    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get all results collected by this strategy.
        
        Returns:
            List of dictionaries containing the collected results
        """
        return self._results
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about strategy execution.
        
        Returns:
            Dictionary with execution metrics
        """
        # Update total time
        self._metrics['total_time'] = time.time() - self._metrics['start_time']
        return self._metrics
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """
        Add a result to the results list.
        
        Args:
            result: The result to add
        """
        self._results.append(result)
    
    def clear_results(self) -> None:
        """Clear all collected results."""
        self._results = []
    
    def initialize(self) -> None:
        """Initialize the strategy."""
        super().initialize()
        self.start_time = time.time()
        self._metrics['start_time'] = self.start_time
    
    def shutdown(self) -> None:
        """Perform cleanup and shutdown operations."""
        self._metrics['total_time'] = time.time() - self._metrics['start_time']
        super().shutdown()
    
    def pause(self) -> None:
        """Pause strategy execution."""
        self.is_paused = True
    
    def resume(self) -> None:
        """Resume strategy execution after pause."""
        self.is_paused = False
    
    def should_visit(self, url: str) -> bool:
        """
        Check if a URL should be visited.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL should be visited, False otherwise
        """
        # Skip if already visited
        if url in self.visited_urls:
            return False
            
        # Skip if not allowed by robots.txt
        if self.config.get('respect_robots_txt', True) and not self.is_url_allowed(url):
            return False
            
        # Skip external domains if configured
        if not self.config.get('include_external', False):
            url_domain = urlparse(url).netloc
            if self.main_domain and url_domain != self.main_domain:
                return False
                
        return True
    
    def can_handle(self, url: str, **kwargs) -> bool:
        """
        Check if this strategy can handle the given URL.
        
        The base implementation can handle any URL.
        
        Args:
            url: The URL to check
            **kwargs: Additional context parameters
            
        Returns:
            True if the strategy can handle the URL, False otherwise
        """
        return True
    
    async def fetch_url_async(self, url: str, **kwargs) -> Optional[str]:
        """
        Asynchronously fetch a URL.
        
        Args:
            url: The URL to fetch
            **kwargs: Additional parameters for the fetch operation
            
        Returns:
            HTML content as string, or None if fetch failed
        """
        # Implement proper async fetch with error handling
        try:
            # In real implementation, this would use httpx or aiohttp
            # For now we just sleep to simulate network delay
            await asyncio.sleep(self.config.get('rate_limit', 1.0))
            return f"<html><body><h1>Sample HTML for {url}</h1></body></html>"
        except Exception as e:
            self.handle_error(
                error=e,
                message=f"Error fetching URL asynchronously: {str(e)}",
                category="network",
                severity="error",
                url=url
            )
            return None
            
    def handle_cookies(self, html_content: str) -> Dict[str, Any]:
        """
        Handle cookies from HTML content.
        
        Args:
            html_content: The HTML content to extract cookies from
            
        Returns:
            Dictionary with cookie information
        """
        # This would be implemented with proper cookie parsing
        return {}
    
    def handle_pagination(self, html_content: str, url: str) -> List[str]:
        """
        Extract pagination links from HTML content.
        
        Args:
            html_content: The HTML content to extract pagination links from
            url: The URL the content was fetched from
            
        Returns:
            List of pagination URLs
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            pagination_urls = []
            
            # Look for common pagination patterns
            # Pattern 1: numbered pagination links
            pagination_links = soup.find_all('a', href=True, text=lambda t: t and t.isdigit())
            for link in pagination_links:
                page_url = self.normalize_url(link['href'], url)
                pagination_urls.append(page_url)
                
            # Pattern 2: next/prev links
            next_links = soup.find_all('a', href=True, text=lambda t: t and ('next' in t.lower() or '>' in t))
            for link in next_links:
                page_url = self.normalize_url(link['href'], url)
                pagination_urls.append(page_url)
                
            return pagination_urls
            
        except Exception as e:
            self.handle_error(
                error=e,
                message=f"Error extracting pagination links: {str(e)}",
                category="pagination",
                severity="warning",
                url=url
            )
            return []
    
    def score_url(self, url: str, base_url: str) -> float:
        """
        Score a URL based on its relevance.
        
        Args:
            url: The URL to score
            base_url: The base URL for comparison
            
        Returns:
            Score between 0.0 and 1.0
        """
        # This is a simple scoring implementation that can be overridden
        # Default score is 0.5
        score = 0.5
        
        # Same domain gets higher score
        url_domain = urlparse(url).netloc
        base_domain = urlparse(base_url).netloc
        if url_domain == base_domain:
            score += 0.3
            
        # Depth penalty
        url_path = urlparse(url).path
        path_depth = len([p for p in url_path.split('/') if p])
        score -= 0.05 * path_depth
        
        # Bound score between 0 and 1
        return max(0.0, min(1.0, score))
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the strategy-specific configuration.
        
        Args:
            config: Dictionary containing configuration parameters
            
        Returns:
            True if the configuration is valid, False otherwise
        """
        # Check required parameters
        required_params = ['max_depth', 'max_pages']
        for param in required_params:
            if param not in config:
                self.logger.warning(f"Missing required configuration parameter: {param}")
                return False
                
        # Check parameter types
        if not isinstance(config.get('max_depth', 0), int):
            self.logger.warning("max_depth must be an integer")
            return False
            
        if not isinstance(config.get('max_pages', 0), int):
            self.logger.warning("max_pages must be an integer")
            return False
            
        return True