"""
URL Parameter Search Engine Strategy

This module implements a search engine strategy that uses URL parameters to perform searches.
It analyzes URLs and constructs appropriate parameters for search queries.
"""

import logging
import re
import urllib.parse
from typing import Dict, List, Any, Optional, Tuple, Set
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import httpx
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from strategies.base_strategy import (
    BaseStrategy,
    SearchEngineInterface,
    SearchCapabilityType,
    SearchEngineCapability,
    register_search_engine
)
from strategies.core.strategy_types import StrategyType, StrategyCapability, strategy_metadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("URLParamStrategy")

@strategy_metadata(
    strategy_type=StrategyType.INTERACTION,
    capabilities={
        StrategyCapability.API_INTERACTION,
        StrategyCapability.RATE_LIMITING,
        StrategyCapability.ROBOTS_TXT_ADHERENCE
    },
    description="Search engine that uses URL parameters to perform searches"
)
@register_search_engine
class URLParamSearchEngine(SearchEngineInterface, BaseStrategy):
    """
    Search engine that uses URL parameters to perform searches.
    It analyzes URLs and constructs appropriate parameters for search queries.
    """
    
    def __init__(self, 
                 max_depth: int = 2, 
                 max_pages: int = 100,
                 include_external: bool = False,
                 user_prompt: str = "",
                 filter_chain: Optional[Any] = None):
        """
        Initialize the URL parameter search engine.
        
        Args:
            max_depth: Maximum crawling depth
            max_pages: Maximum number of pages to crawl
            include_external: Whether to include external links
            user_prompt: The user's original request/prompt
            filter_chain: Filter chain to apply to URLs
        """
        BaseStrategy.__init__(
            self,
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=include_external,
            user_prompt=user_prompt,
            filter_chain=filter_chain
        )
        
        self.user_agent = UserAgent().random
        
        # Common search parameter names
        self.search_params = [
            'q', 'query', 'search', 'term', 'keyword', 'keywords', 'text',
            's', 'searchfor', 'search_query', 'search_term', 'k', 'kw',
            'search_text', 'searchterm', 'searchquery', 'find'
        ]
        
        # Known search URL patterns for popular sites
        self.known_patterns = {
            'google.com': {'path': '/search', 'param': 'q'},
            'bing.com': {'path': '/search', 'param': 'q'},
            'duckduckgo.com': {'path': '/', 'param': 'q'},
            'yahoo.com': {'path': '/search', 'param': 'p'},
            'amazon.com': {'path': '/s', 'param': 'k'},
            'ebay.com': {'path': '/sch/i.html', 'param': '_nkw'},
            'walmart.com': {'path': '/search/', 'param': 'q'},
            'target.com': {'path': '/s', 'param': 'searchTerm'},
            'etsy.com': {'path': '/search', 'param': 'q'},
            'homedepot.com': {'path': '/s', 'param': 'keyword'},
            'lowes.com': {'path': '/search', 'param': 'searchTerm'},
            'wayfair.com': {'path': '/keyword.php', 'param': 'keyword'},
            'bestbuy.com': {'path': '/site/searchpage.jsp', 'param': 'st'},
            'costco.com': {'path': '/CatalogSearch', 'param': 'keyword'},
            'zillow.com': {'path': '/homes', 'param': 'searchQueryState'},
            'realtor.com': {'path': '/search', 'param': 'searchQueryState'},
            'redfin.com': {'path': '/search', 'param': 'q'},
            'apartments.com': {'path': '/search', 'param': 'query'},
            'indeed.com': {'path': '/jobs', 'param': 'q'},
            'linkedin.com': {'path': '/jobs/search', 'param': 'keywords'},
            'glassdoor.com': {'path': '/Job', 'param': 'sc.keyword'},
            'monster.com': {'path': '/jobs/search', 'param': 'q'},
            'yelp.com': {'path': '/search', 'param': 'find_desc'},
            'tripadvisor.com': {'path': '/Search', 'param': 'q'},
            'expedia.com': {'path': '/search', 'param': 'searchQuery'},
            'booking.com': {'path': '/searchresults', 'param': 'ss'},
            'hotels.com': {'path': '/search', 'param': 'q-destination'},
            'airbnb.com': {'path': '/s', 'param': 'query'},
            'reddit.com': {'path': '/search', 'param': 'q'},
            'twitter.com': {'path': '/search', 'param': 'q'},
            'facebook.com': {'path': '/search', 'param': 'q'},
            'instagram.com': {'path': '/explore', 'param': 'q'},
            'youtube.com': {'path': '/results', 'param': 'search_query'},
            'quora.com': {'path': '/search', 'param': 'q'},
            'stackoverflow.com': {'path': '/search', 'param': 'q'},
            'github.com': {'path': '/search', 'param': 'q'},
            'wikipedia.org': {'path': '/wiki/Special:Search', 'param': 'search'},
        }
        
    @property
    def name(self) -> str:
        """
        Get the name of the search engine.
        
        Returns:
            str: Engine name
        """
        return "url-param-search-engine"
    
    @property
    def capabilities(self) -> List[SearchEngineCapability]:
        """
        Get the capabilities of this search engine.
        
        Returns:
            List of SearchEngineCapability objects
        """
        return [
            SearchEngineCapability(
                capability_type=SearchCapabilityType.URL_PARAMETER,
                confidence=1.0,
                requires_browser=False,
                description="Can construct and use URL parameters for search queries"
            )
        ]
    
    def get_required_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters required by this search engine.
        
        Returns:
            Dictionary of parameter name to parameter specification
        """
        return {
            "query": {
                "type": "string",
                "description": "The search query to include in URL parameters",
                "required": True
            },
            "param_name": {
                "type": "string",
                "description": "The URL parameter name to use for the search query (if known)",
                "required": False
            },
            "additional_params": {
                "type": "object",
                "description": "Additional URL parameters to include in the search request",
                "required": False
            }
        }
    
    async def can_handle(self, url: str, html: Optional[str] = None) -> Tuple[bool, float]:
        """
        Determine if this search engine can handle the given URL/page.
        
        Args:
            url: The URL to check
            html: Optional HTML content of the page
            
        Returns:
            Tuple of (can_handle, confidence)
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        path = parsed_url.path.lower()
        
        # Check if the URL matches a known pattern
        domain_key = None
        for known_domain in self.known_patterns:
            if known_domain in domain:
                domain_key = known_domain
                break
        
        if domain_key:
            pattern = self.known_patterns[domain_key]
            if pattern['path'] in path:
                return True, 0.9
        
        # Check if URL already has search parameters
        query_params = parse_qs(parsed_url.query)
        for param_name in self.search_params:
            if param_name in query_params:
                return True, 0.8
        
        # If HTML is provided, look for search links
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for links that might be search-related
            search_links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                text = link.get_text(strip=True).lower()
                
                # Skip empty links
                if not href:
                    continue
                
                # Check if link text indicates search
                if any(term in text for term in ['search', 'find', 'look', 'seek']):
                    search_links.append(href)
                
                # Check if href has search parameters
                parsed_href = urlparse(href)
                if any(param in parse_qs(parsed_href.query) for param in self.search_params):
                    search_links.append(href)
            
            if search_links:
                return True, 0.7
        
        # As a fallback, check if the URL's path might be search-related
        if any(search_term in path for search_term in ['search', 'find', 'query', 'look', 'seek']):
            return True, 0.6
        
        # Lower confidence for generic pages (but still may be able to handle)
        if path == '/' or path == '':
            return True, 0.5
        
        return False, 0.0
    
    def _extract_search_path_from_html(self, base_url: str, html: str) -> Optional[str]:
        """
        Extract a potential search path from HTML content.
        
        Args:
            base_url: The base URL of the page
            html: HTML content
            
        Returns:
            Potential search path or None
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for forms that might be search forms
        search_paths = []
        
        for form in soup.find_all('form'):
            action = form.get('action', '')
            form_id = form.get('id', '').lower()
            form_class = ' '.join(form.get('class', [])).lower()
            
            # Look for search indicators
            if any(term in attr for term in ['search', 'find', 'query'] 
                  for attr in [action.lower(), form_id, form_class]):
                if action:
                    search_paths.append(action)
        
        # Look for links with search-related text
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            text = link.get_text(strip=True).lower()
            
            if any(term in text for term in ['search', 'find', 'advanced search']):
                search_paths.append(href)
        
        # If found, choose the best path
        if search_paths:
            # Join with base URL and normalize
            search_paths = [urllib.parse.urljoin(base_url, path) for path in search_paths]
            
            # Parse and extract paths
            parsed_paths = [urlparse(path) for path in search_paths]
            
            # Prefer paths that seem most search-like
            for parsed in parsed_paths:
                path = parsed.path.lower()
                if any(term in path for term in ['/search', '/find', '/results']):
                    return parsed.path
            
            # Otherwise, return the first path
            return parsed_paths[0].path
        
        return None
    
    def _construct_search_url(self, url: str, query: str, param_name: Optional[str] = None,
                              additional_params: Optional[Dict[str, str]] = None) -> str:
        """
        Construct a search URL with the given query.
        
        Args:
            url: Base URL to use
            query: Search query string
            param_name: Parameter name to use (if known)
            additional_params: Additional parameters to include
            
        Returns:
            Full search URL with parameters
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        path = parsed_url.path.lower()
        
        # Parse existing query parameters
        query_params = parse_qs(parsed_url.query)
        
        # Initialize new query parameters with all existing ones (flatten the lists)
        new_params = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}
        
        # Add additional parameters if provided
        if additional_params:
            new_params.update(additional_params)
        
        # Check if we match a known site pattern
        domain_key = None
        for known_domain in self.known_patterns:
            if known_domain in domain:
                domain_key = known_domain
                break
        
        # If known site, use its pattern
        if domain_key:
            pattern = self.known_patterns[domain_key]
            
            # If current path doesn't match expected search path, use the expected path
            if pattern['path'] not in path:
                parsed_url = parsed_url._replace(path=pattern['path'])
            
            # Use the known parameter name
            new_params[pattern['param']] = query
        else:
            # Use provided parameter name if available
            if param_name:
                new_params[param_name] = query
            else:
                # Try to detect which parameter might be for search
                search_param = None
                for param in self.search_params:
                    if param in new_params:
                        search_param = param
                        break
                
                # If found, use it, otherwise use a default
                if search_param:
                    new_params[search_param] = query
                else:
                    new_params['q'] = query  # Default to 'q' as it's most common
        
        # Construct the new URL
        new_query = urlencode(new_params, doseq=True)
        new_url = parsed_url._replace(query=new_query)
        
        return urlunparse(new_url)
    
    async def search(self, query: str, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a URL parameter-based search operation.
        
        Args:
            query: The search query string
            url: The target URL to search on
            params: Additional parameters for the search
            
        Returns:
            Dictionary containing search results and metadata
        """
        params = params or {}
        param_name = params.get('param_name')
        additional_params = params.get('additional_params', {})
        
        logger.info(f"Executing URL parameter search with query '{query}' on {url}")
        
        try:
            # First, check if we need to fetch the page to find search path
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            path = parsed_url.path.lower()
            
            domain_key = None
            for known_domain in self.known_patterns:
                if known_domain in domain:
                    domain_key = known_domain
                    break
            
            search_path_known = domain_key and self.known_patterns[domain_key]['path'] in path
            has_search_params = any(param in parse_qs(parsed_url.query) for param in self.search_params)
            
            # If the current URL doesn't look like a search URL, try to find the search path
            if not search_path_known and not has_search_params and not param_name:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    headers = {"User-Agent": self.user_agent}
                    response = await client.get(url, headers=headers)
                    
                    if response.status_code != 200:
                        return {
                            "success": False,
                            "error": f"Failed to fetch page: {response.status_code}",
                            "results": []
                        }
                    
                    html = response.text
                    search_path = self._extract_search_path_from_html(url, html)
                    
                    if search_path:
                        # Construct a new URL with the search path
                        parsed_url = parsed_url._replace(path=search_path)
                        url = urlunparse(parsed_url)
            
            # Construct the search URL
            search_url = self._construct_search_url(url, query, param_name, additional_params)
            
            # Fetch the search results
            async with httpx.AsyncClient(follow_redirects=True) as client:
                headers = {"User-Agent": self.user_agent}
                response = await client.get(search_url, headers=headers)
                
                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Search request failed: {response.status_code}",
                        "results": []
                    }
                
                results_html = response.text
                results_url = str(response.url)
                
                # Extract results from the page
                results = self._extract_basic_results(results_url, results_html)
                
                return {
                    "success": True,
                    "results_url": results_url,
                    "result_count": len(results),
                    "results": results,
                    "search_url_used": search_url
                }
                
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    def _extract_basic_results(self, url: str, html: str) -> List[Dict[str, Any]]:
        """
        Extract basic results from HTML.
        This is a simple extraction that looks for common result patterns.
        
        Args:
            url: The URL of the results page
            html: The HTML content of the results page
            
        Returns:
            List of extracted results
        """
        results = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for common result container patterns
        result_containers = (
            soup.find_all('div', class_=lambda c: c and any(term in c.lower() for term in ['result', 'item', 'product', 'listing']))
            or soup.find_all('li', class_=lambda c: c and any(term in c.lower() for term in ['result', 'item', 'product', 'listing']))
            or soup.find_all('article')
        )
        
        # If no structured results found, try to extract links
        if not result_containers:
            links = soup.find_all('a', href=True)
            
            # Filter out navigation links, javascript, etc.
            content_links = []
            for link in links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # Skip empty links, anchors, javascript, etc.
                if (not href or not text or href.startswith(('#', 'javascript:', 'mailto:'))
                    or any(nav_term in text.lower() for nav_term in ['login', 'sign up', 'register', 'next', 'previous'])):
                    continue
                
                # Make absolute URL
                href = urllib.parse.urljoin(url, href)
                
                # Skip links to the same page
                if href == url:
                    continue
                
                content_links.append({
                    "title": text,
                    "url": href,
                    "text": text,
                    "source_url": url
                })
            
            # Only return unique links
            seen_urls = set()
            for link in content_links:
                if link['url'] not in seen_urls:
                    seen_urls.add(link['url'])
                    results.append(link)
            
            return results[:20]  # Limit to 20 results
        
        # Process each result container
        for container in result_containers[:20]:  # Limit to first 20 containers
            # Try to extract title and link
            title_elem = (
                container.find('h1') or container.find('h2') or container.find('h3') 
                or container.find('h4') or container.find('h5') or container.find('h6')
                or container.find('strong') or container.find('b')
            )
            
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            # Look for links
            link_elem = container.find('a', href=True)
            link = urllib.parse.urljoin(url, link_elem['href']) if link_elem else ""
            
            # Try to extract description
            desc_elems = container.find_all(['p', 'div'], class_=lambda c: c and any(
                term in c.lower() for term in ['desc', 'summary', 'content', 'text']))
            description = " ".join(elem.get_text(strip=True) for elem in desc_elems) if desc_elems else ""
            
            # If no specific description elements, use all text in the container
            if not description:
                # Skip title text
                all_text = container.get_text(strip=True)
                if title and all_text:
                    # Remove title from all text to get description
                    description = all_text.replace(title, "", 1).strip()
            
            # Only add if we have a title or link
            if title or link:
                results.append({
                    "title": title,
                    "url": link,
                    "description": description[:300] if description else "",  # Limit description length
                    "source_url": url
                })
        
        return results
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute the URL parameter strategy for the given URL.
        
        Args:
            url: The URL to process
            **kwargs: Additional keyword arguments including:
                - query: Search query term
                - extraction_config: Configuration for extraction
                - crawler: Optional crawler instance to use
            
        Returns:
            Dictionary with results, or None if execution failed
        """
        import asyncio
        
        # Get optional parameters from kwargs
        query = kwargs.get('query', kwargs.get('search_term', ''))
        extraction_config = kwargs.get('extraction_config', {})
        crawler = kwargs.get('crawler')
        
        # Use asyncio to run the async method
        async def _run():
            return await self._execute_async(crawler, url, extraction_config, query)
                
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

    async def _execute_async(self, crawler, start_url, extraction_config=None, query=""):
        """
        Internal async implementation of the URL parameter strategy.
        
        Args:
            crawler: The AsyncWebCrawler instance
            start_url: The starting URL
            extraction_config: Optional extraction configuration
            query: Search query term
            
        Returns:
            Dictionary containing the results
        """
        # For this engine, execute is just a wrapper around the search method
        query = extraction_config.get("query", "") if extraction_config else ""
        params = extraction_config or {}
        
        search_results = await self.search(query, start_url, params)
        
        if not search_results.get("success", False):
            return {"results": [], "error": search_results.get("error", "Unknown error")}
        
        # Convert search results to crawler results format
        crawler_results = []
        for idx, result in enumerate(search_results.get("results", [])):
            crawler_results.append({
                "source_url": result.get("source_url", start_url),
                "depth": 1,
                "relevance": 1.0 - (idx * 0.05),  # Decreasing relevance by position
                "data": result
            })
        
        return {"results": crawler_results}
    
    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Crawl from a starting URL using URL parameter strategy.
        
        Args:
            start_url: The URL to start crawling from
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Dictionary with crawl results, or None if crawl failed
        """
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
            
            # Extract basic information
            title = soup.title.string if soup.title else ""
            
            # Try to extract description from meta tags
            description = ""
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc:
                description = meta_desc.get("content", "")
            
            # Extract links for potential URL parameter manipulation
            links = []
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if href and not href.startswith(("#", "javascript:")):
                    links.append({
                        "url": href,
                        "text": a.get_text(strip=True) or "[No Text]",
                        "title": a.get("title", "")
                    })
            
            return {
                "title": title,
                "description": description,
                "url": url,
                "links": links[:20],  # Limit to 20 links
                "method": "url_param_extraction"
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
        if not hasattr(self, '_results'):
            self._results = []
        return self._results