"""
Breadth-First Search Strategy Module

This strategy implements a breadth-first search approach for web crawling.
"""

from typing import Dict, List, Any, Optional, Set
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import logging
import requests

from strategies.base_strategy import BaseStrategy
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_types import StrategyType, StrategyCapability, strategy_metadata
from strategies.core.strategy_error_handler import StrategyErrorCategory, StrategyErrorSeverity

# Keep for backward compatibility
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

logger = logging.getLogger(__name__)

@strategy_metadata(
    strategy_type=StrategyType.TRAVERSAL,
    capabilities={
        StrategyCapability.ROBOTS_TXT_ADHERENCE,
        StrategyCapability.RATE_LIMITING,
        StrategyCapability.LINK_EXTRACTION,
        StrategyCapability.ERROR_HANDLING,
        StrategyCapability.CONTENT_NORMALIZATION
    },
    description="Breadth-First Search strategy that explores all URLs at a given depth before moving to URLs at the next depth level."
)
class BFSStrategy(BaseStrategy):
    """
    Breadth-First Search strategy that explores all URLs at a given depth
    before moving to URLs at the next depth level.
    
    Features:
    - Explores links level by level
    - Guarantees shortest path to content
    - Good for finding content within a specific number of clicks
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the BFS strategy.
        
        Args:
            **kwargs: Additional keyword arguments for configuration
        """
        # Initialize with default config
        max_depth = kwargs.pop('max_depth', 3)
        max_pages = kwargs.pop('max_pages', 100)
        include_external = kwargs.pop('include_external', False)
        user_prompt = kwargs.pop('user_prompt', "")
        filter_chain = kwargs.pop('filter_chain', None)
        context = kwargs.pop('context', None)
        
        super().__init__(
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=include_external,
            user_prompt=user_prompt,
            filter_chain=filter_chain
        )
        
        # Will be initialized during execution
        self.url_queue = None
        self.main_domain = None
        
        # Additional configuration from kwargs
        self.config = kwargs
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Get resource services from context if provided
        if context:
            self.strategy_context = context
            self.url_service = context.get_service("url_service")
            self.html_service = context.get_service("html_service")
            self.session_manager = context.get_session_manager()
            self.proxy_manager = context.get_proxy_manager()
            self.rate_limiter = context.get_rate_limiter()
            self.error_classifier = context.get_error_classifier()
            self.retry_manager = context.get_retry_manager()
            self.circuit_breaker_manager = context.get_circuit_breaker_manager()
    
    @property
    def name(self) -> str:
        """Get the name of the strategy."""
        return "bfs_strategy"
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute the BFS strategy for the given URL.
        
        Args:
            url: The URL to process
            **kwargs: Additional keyword arguments including:
                - extraction_config: Configuration for extraction
                - crawler: Optional crawler instance to use
            
        Returns:
            Dictionary with results, or None if execution failed
        """
        import asyncio
        
        # Get optional parameters from kwargs
        extraction_config = kwargs.get('extraction_config')
        crawler = kwargs.get('crawler')
        
        # Use asyncio to run the async crawl method
        async def _run():
            if crawler:
                return await self._execute_async(crawler, url, extraction_config)
            else:
                # Create a temporary crawler if none provided
                from crawl4ai import AsyncWebCrawler
                async with AsyncWebCrawler() as temp_crawler:
                    return await self._execute_async(temp_crawler, url, extraction_config)
                
        # Check if we're already in an async context
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we reach here, we're in an async context - create a task instead
            import concurrent.futures
            
            # Run in a separate thread to avoid "event loop already running" error
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _run())
                return future.result()
                
        except RuntimeError:
            # No event loop is running, we can use run_until_complete
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                # Create a new event loop if none exists
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            return loop.run_until_complete(_run())

    async def _execute_async(self, crawler, start_url, extraction_config=None):
        """
        Internal async implementation of the BFS strategy.
        
        Args:
            crawler: The crawler instance
            start_url: The URL to process
            extraction_config: Configuration for extraction
            
        Returns:
            Dictionary with results, or None if execution failed
        """
        self.logger.info(f"Starting BFS crawl from {start_url}")
        
        # Initialize tracking
        self.visited_urls = set()
        
        # Get a named URL queue from the URL service
        queue_name = f"bfs_{self.name}"
        self.url_queue = self.url_service.get_queue(queue_name)
        self.main_domain = urlparse(start_url).netloc
        
        # Add the start URL to the queue
        normalized_start_url = self.url_service.normalize_url(start_url)
        self.url_queue.add(normalized_start_url)
        
        # Process URLs in BFS order
        visited_count = 0
        try:
            while visited_count < self.max_pages:
                # Get the next URL from the queue
                current_url = self.url_queue.get()
                
                # If queue is empty, we're done
                if current_url is None:
                    break
                    
                # Skip if URL is not allowed by robots.txt
                if not self.url_service.is_allowed(current_url):
                    self.logger.info(f"Skipping URL (not allowed by robots.txt): {current_url}")
                    self.url_queue.complete(current_url)
                    continue
                    
                self.logger.info(f"Visiting URL: {current_url}")
                
                # Fetch the page - now using our robust error handling and resource management
                try:
                    # Get the domain for resource management
                    domain = urlparse(current_url).netloc
                    
                    # Make the request with full error handling
                    response = self._make_request(current_url)
                    html_content = response.text
                    
                    # Calculate the current depth based on our tracking or default to 0
                    current_depth = 0
                    
                    # Extract data if needed
                    if extraction_config:
                        # Use crawler for extraction with the provided config
                        result = await crawler.arun(url=current_url, config=extraction_config)
                        
                        # Track the result
                        if result.success and result.extracted_content:
                            # Default score of 1.0 since BFS doesn't use scoring
                            self.add_result(current_url, result.extracted_content, current_depth)
                    else:
                        # Basic content extraction
                        try:
                            soup = BeautifulSoup(html_content, 'html.parser')
                            
                            title = soup.title.text if soup.title else ""
                            description = ""
                            if soup.find("meta", attrs={"name": "description"}):
                                description = soup.find("meta", {"name": "description"}).get("content", "")
                            
                            # Extract main content
                            main_content = self.html_service.extract_main_content(html_content)
                            
                            # Basic data
                            data = {
                                "title": title,
                                "description": description,
                                "url": current_url
                            }
                            
                            # Add result
                            self.add_result(current_url, data, current_depth)
                        except Exception as e:
                            # Classify and log the error
                            error_context = {
                                'url': current_url,
                                'operation': 'extract_content',
                                'domain': domain
                            }
                            error_info = self._classify_error(e, error_context)
                            logger.error(f"Error extracting content from {current_url}: {str(e)} - "
                                        f"{error_info.get('category', 'unknown')}")
                    
                    # Don't extract links if we're at the maximum depth
                    if current_depth >= self.max_depth:
                        self.url_queue.complete(current_url)
                        visited_count += 1
                        continue
                        
                    # Extract links using the HTML service with error handling
                    try:
                        links = self.html_service.extract_links(html_content, base_url=current_url)
                        
                        # Queue the links for the next depth level
                        for link_data in links:
                            link = link_data.get('url', '')
                            
                            # Skip non-HTTP links
                            if not link.startswith(('http://', 'https://')):
                                continue
                            
                            # Skip if already visited or queued
                            if (link in self.visited_urls or 
                                (hasattr(self.url_queue, 'is_visited') and self.url_queue.is_visited(link)) or
                                (hasattr(self.url_queue, 'is_in_progress') and self.url_queue.is_in_progress(link))):
                                continue
                                
                            # Check domain constraints
                            if not self.include_external:
                                link_domain = urlparse(link).netloc
                                if link_domain != self.main_domain:
                                    continue
                                    
                            # Normalize the URL before adding
                            normalized_link = self.url_service.normalize_url(link)
                            
                            # Add to queue for the next depth level
                            self.url_queue.add(normalized_link)
                    except Exception as e:
                        # Classify and log the error
                        error_context = {
                            'url': current_url,
                            'operation': 'extract_links',
                            'domain': domain
                        }
                        error_info = self._classify_error(e, error_context)
                        logger.error(f"Error extracting links from {current_url}: {str(e)} - "
                                    f"{error_info.get('category', 'unknown')}")
                
                except Exception as e:
                    # Classify and log the error
                    error_context = {
                        'url': current_url,
                        'operation': 'fetch_url',
                        'domain': urlparse(current_url).netloc
                    }
                    error_info = self._classify_error(e, error_context)
                    logger.error(f"Error fetching {current_url}: {str(e)} - "
                                f"{error_info.get('category', 'unknown')}")
                
                # Mark as completed in the queue
                self.url_queue.complete(current_url)
                self.add_visited(current_url)
                visited_count += 1
        
        except Exception as e:
            # Classify and log the error
            error_context = {
                'url': start_url,
                'operation': 'execute_bfs',
                'domain': self.main_domain
            }
            error_info = self._classify_error(e, error_context)
            logger.error(f"Error in BFS execution: {str(e)} - "
                        f"{error_info.get('category', 'unknown')}")
        
        self.logger.info(f"BFS completed. Visited {visited_count} pages, found {len(self.results)} results.")
        
        # Get resource and error statistics
        resource_stats = self.get_resource_stats()
        error_stats = self.get_error_stats()
        
        return {
            "results": self.results,
            "visited_count": visited_count,
            "resource_stats": resource_stats,
            "error_stats": error_stats
        }

    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Crawl from a starting URL using the BFS strategy.
        
        Args:
            start_url: The URL to start crawling from
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Dictionary with crawl results, or None if crawl failed
        """
        # Delegate to execute method which handles the crawling logic
        return self.execute(start_url, **kwargs)

    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Extract data from HTML content using basic extraction logic.
        
        Args:
            html_content: The HTML content to extract data from
            url: The URL the content was fetched from
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Dictionary with extracted data, or None if extraction failed
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            title = soup.title.text if soup.title else ""
            description = ""
            if soup.find("meta", attrs={"name": "description"}):
                description = soup.find("meta", {"name": "description"}).get("content", "")
            
            # Extract main content if html_service is available
            main_content = ""
            if hasattr(self, 'html_service') and self.html_service:
                main_content = self.html_service.extract_main_content(html_content)
            else:
                # Basic fallback content extraction
                main_content = soup.get_text()[:1000]  # First 1000 chars
            
            return {
                "title": title,
                "description": description,
                "url": url,
                "content": main_content
            }
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            return None

    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get all results collected by this strategy.
        
        Returns:
            List of dictionaries containing extracted data
        """
        return self.results