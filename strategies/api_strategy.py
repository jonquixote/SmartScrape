"""
API-based search engine implementation.

This file implements a search engine strategy that uses API endpoints
for efficient and direct search operations.
"""

import logging
import asyncio
import re
import json
import time
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlparse, parse_qs, quote, urlencode, quote_plus

import httpx
from bs4 import BeautifulSoup

from strategies.base_strategy import BaseStrategy
from utils.http_utils import fetch_html, clean_url
from strategies.core.strategy_types import StrategyType, StrategyCapability, strategy_metadata

@strategy_metadata(
    strategy_type=StrategyType.EXTRACTION,
    capabilities={
        StrategyCapability.API_INTERACTION,
        StrategyCapability.DYNAMIC_CONTENT,
        StrategyCapability.RATE_LIMITING
    },
    description="Strategy that identifies and uses API endpoints for efficient data retrieval"
)
class APIStrategy(BaseStrategy):
    """
    API-based search engine that identifies and uses site-specific APIs
    for search operations.
    
    This strategy:
    - Identifies API endpoints for search functionality
    - Supports both REST and GraphQL APIs
    - Handles authentication when necessary
    - Efficiently extracts structured data directly from APIs
    - Adapts to different API response formats
    """
    
    def __init__(self, config=None):
        """
        Initialize the API search strategy.
        
        Args:
            config: Configuration dictionary (optional)
        """
        super().__init__(config or {})
        self.logger = logging.getLogger("APIStrategy")
        
        # Store API patterns for known sites
        self.api_patterns = self._load_api_patterns()
        
        # Store API endpoints found during execution
        self.discovered_endpoints = {}
        
        # Common request headers to mimic a browser
    
    @property
    def name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            String name of the strategy
        """
        return "api_strategy"
    
    def can_handle(self, url: str, **kwargs) -> bool:
        """
        Check if this strategy can handle the given URL.
        API strategy looks for sites with known API endpoints or patterns.
        
        Args:
            url: URL to check
            **kwargs: Additional keyword arguments
            
        Returns:
            Boolean indicating if strategy can handle the URL
        """
        try:
            if not url or not url.startswith(('http://', 'https://')):
                return False
                
            # Check if we have known API patterns for this domain
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            
            # Check for known API-friendly domains
            api_indicators = [
                'api.', '/api/', '/rest/', '/graphql/', '/v1/', '/v2/', 
                'jsonapi', 'endpoints', 'swagger', 'openapi'
            ]
            
            if any(indicator in url.lower() for indicator in api_indicators):
                return True
                
            # Check our known patterns
            if domain in self.api_patterns:
                return True
                
            return False
        except Exception:
            return False
        
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
        self.logger.info(f"Executing API strategy for URL: {url}")
        
        try:
            # Discover API endpoints for the site
            site_domain = urlparse(url).netloc
            endpoints = self._discover_api_endpoints(url)
            
            if not endpoints:
                self.logger.warning(f"No API endpoints found for {url}")
                return None
                
            # Execute API calls based on discovered endpoints
            results = {}
            for endpoint_type, endpoint_info in endpoints.items():
                endpoint_url = endpoint_info.get("url")
                if not endpoint_url:
                    continue
                    
                # Call the API endpoint
                api_result = self._call_api_endpoint(
                    endpoint_url, 
                    endpoint_type,
                    params=kwargs.get("params", {})
                )
                
                if api_result:
                    results[endpoint_type] = api_result
            
            # Process and structure the results
            structured_results = self._process_api_results(results, url)
            
            return {
                "success": bool(structured_results),
                "results": structured_results,
                "api_endpoints": list(endpoints.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Error executing API strategy for {url}: {str(e)}")
            self.handle_error(
                error=e,
                message=f"API strategy execution failed for {url}",
                category="api_execution",
                severity="error",
                url=url
            )
            return None

    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Crawl from a starting URL using this strategy.
        
        This method implements the traversal logic for API-based crawling.
        
        Args:
            start_url: The URL to start crawling from
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Optional dictionary with crawl results, or None if crawl failed
        """
        self.logger.info(f"Starting API-based crawl from URL: {start_url}")
        
        try:
            # Get domain and discover API endpoints
            domain = urlparse(start_url).netloc
            endpoints = self._discover_api_endpoints(start_url)
            
            if not endpoints:
                self.logger.warning(f"No API endpoints found for crawling {start_url}")
                return None
                
            # Track crawled URLs and results
            crawled_urls = set()
            crawl_results = []
            
            # Process the start URL
            result = self.execute(start_url, **kwargs)
            if result and result.get("success"):
                crawled_urls.add(start_url)
                crawl_results.append(result)
            
            # Extract pagination or related data API endpoints
            pagination_endpoints = self._extract_pagination_endpoints(start_url, result)
            
            # Process pagination up to the specified limit
            max_pages = kwargs.get("max_pages", 5)
            current_page = 1
            
            for page_url in pagination_endpoints:
                if current_page >= max_pages:
                    break
                    
                if page_url in crawled_urls:
                    continue
                    
                page_result = self.execute(page_url, **kwargs)
                if page_result and page_result.get("success"):
                    crawled_urls.add(page_url)
                    crawl_results.append(page_result)
                    
                current_page += 1
            
            return {
                "success": bool(crawl_results),
                "crawled_urls": list(crawled_urls),
                "results": crawl_results,
                "page_count": len(crawled_urls)
            }
            
        except Exception as e:
            self.logger.error(f"Error during API crawl for {start_url}: {str(e)}")
            self.handle_error(
                error=e,
                message=f"API crawl failed for {start_url}",
                category="api_crawl",
                severity="error",
                url=start_url
            )
            return None
    
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Extract data from HTML content using this strategy.
        
        For API strategy, this extracts API endpoints from HTML and calls them.
        
        Args:
            html_content: The HTML content to extract data from
            url: The URL the content was fetched from
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Optional dictionary with extracted data, or None if extraction failed
        """
        self.logger.info(f"Extracting API endpoints from HTML for URL: {url}")
        
        try:
            # Parse the HTML to find API endpoints
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Look for API endpoints in script tags
            api_endpoints = self._extract_api_from_scripts(soup, url)
            
            # Look for API endpoints in fetch/ajax calls
            api_endpoints.update(self._extract_api_from_ajax(soup, url))
            
            if not api_endpoints:
                self.logger.warning(f"No API endpoints found in HTML for {url}")
                return None
            
            # Store discovered endpoints
            domain = urlparse(url).netloc
            if domain not in self.discovered_endpoints:
                self.discovered_endpoints[domain] = {}
                
            self.discovered_endpoints[domain].update(api_endpoints)
            
            # Call discovered endpoints
            results = {}
            for endpoint_type, endpoint_info in api_endpoints.items():
                endpoint_url = endpoint_info.get("url")
                if not endpoint_url:
                    continue
                    
                # Call the API endpoint
                api_result = self._call_api_endpoint(
                    endpoint_url, 
                    endpoint_type,
                    params=kwargs.get("params", {})
                )
                
                if api_result:
                    results[endpoint_type] = api_result
            
            return {
                "success": bool(results),
                "api_endpoints": list(api_endpoints.keys()),
                "results": self._process_api_results(results, url)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting API data from HTML for {url}: {str(e)}")
            self.handle_error(
                error=e,
                message=f"API extraction from HTML failed for {url}",
                category="api_extraction",
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
        
        # Common request headers to mimic a browser
        self.default_headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": None,  # Will be set before each request
            "Connection": "keep-alive"
        }
        
        # Set default capabilities
        self.capabilities = {
            "direct_api": True,
            "rest_api": True,
            "graphql": True,
            "authentication": True,
            "response_processing": True
        }
        
    def _load_api_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load known API patterns for common sites"""
        return {
            # E-commerce sites
            "amazon.com": {
                "type": "rest",
                "search_endpoint": "/s/ref=nb_sb_noss",
                "search_params": {"field-keywords": "{search_term}"},
                "response_type": "html"  # Amazon doesn't expose a public API, scrape results from HTML
            },
            "ebay.com": {
                "type": "rest",
                "search_endpoint": "/sch/i.html",
                "search_params": {"_nkw": "{search_term}"},
                "response_type": "html"
            },
            
            # Real estate sites
            "zillow.com": {
                "type": "graphql",
                "search_endpoint": "/graphql",
                "operation_name": "SearchPageDataQuery",
                "search_variable": "searchQueryState",
                "response_type": "json"
            },
            "realtor.com": {
                "type": "rest",
                "search_endpoint": "/api/v1/search",
                "search_params": {"query": "{search_term}"},
                "response_type": "json"
            },
            
            # Job sites
            "indeed.com": {
                "type": "rest",
                "search_endpoint": "/jobs",
                "search_params": {"q": "{search_term}"},
                "response_type": "html"
            },
            
            # Generic patterns (will be tried for unknown sites)
            "generic_rest": [
                {
                    "type": "rest",
                    "search_endpoint": "/api/search",
                    "search_params": {"q": "{search_term}"},
                    "response_type": "json"
                },
                {
                    "type": "rest",
                    "search_endpoint": "/api/v1/search",
                    "search_params": {"query": "{search_term}"},
                    "response_type": "json"
                },
                {
                    "type": "rest",
                    "search_endpoint": "/search",
                    "search_params": {"q": "{search_term}"},
                    "response_type": "json"
                }
            ],
            "generic_graphql": [
                {
                    "type": "graphql",
                    "search_endpoint": "/graphql",
                    "operation_name": "Search",
                    "search_variable": "query",
                    "response_type": "json"
                },
                {
                    "type": "graphql",
                    "search_endpoint": "/api/graphql",
                    "operation_name": "SearchQuery",
                    "search_variable": "searchTerm",
                    "response_type": "json"
                }
            ]
        }
    
    async def search(self, url: str, search_term: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform a search using API endpoints.
        
        Args:
            url: URL to search on
            search_term: Search term to use
            options: Additional options (optional)
            
        Returns:
            Dictionary with search results
        """
        options = options or {}
        self.logger.info(f"Performing API-based search on {url} for term: {search_term}")
        
        # Parse the URL to get the domain
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Set referer for requests
        self.default_headers["Referer"] = url
        
        # Try previously discovered endpoints first
        if domain in self.discovered_endpoints:
            self.logger.info(f"Using previously discovered API endpoint for {domain}")
            api_info = self.discovered_endpoints[domain]
            api_result = await self._try_api(base_url, search_term, api_info, options)
            
            if api_result and api_result.get("success", False):
                return api_result
        
        # Try known patterns for this domain
        known_pattern = self._get_known_pattern(domain)
        if known_pattern:
            self.logger.info(f"Using known API pattern for {domain}")
            api_result = await self._try_api(base_url, search_term, known_pattern, options)
            
            if api_result and api_result.get("success", False):
                # Store this successful endpoint for future use
                self.discovered_endpoints[domain] = known_pattern
                return api_result
        
        # If no known pattern worked, try to discover API endpoints
        self.logger.info(f"No known API pattern for {domain}, attempting discovery")
        api_info = await self._discover_api_endpoints(url, search_term)
        
        if api_info:
            self.logger.info(f"Discovered API endpoint for {domain}")
            api_result = await self._try_api(base_url, search_term, api_info, options)
            
            if api_result and api_result.get("success", False):
                # Store this successful endpoint for future use
                self.discovered_endpoints[domain] = api_info
                return api_result
        
        # Try generic patterns as a last resort
        self.logger.info("Trying generic API patterns")
        for pattern in self.api_patterns.get("generic_rest", []) + self.api_patterns.get("generic_graphql", []):
            api_result = await self._try_api(base_url, search_term, pattern, options)
            
            if api_result and api_result.get("success", False):
                # Store this successful endpoint for future use
                self.discovered_endpoints[domain] = pattern
                return api_result
        
        # If all attempts failed, return failure
        self.logger.warning(f"Failed to find a working API endpoint for {domain}")
        return {
            "success": False,
            "engine": "api",
            "error": "No working API endpoint found"
        }
    
    def _get_known_pattern(self, domain: str) -> Dict[str, Any]:
        """Get a known API pattern for the domain"""
        # Try exact match
        if domain in self.api_patterns:
            return self.api_patterns[domain]
        
        # Try to match by removing subdomains (www.example.com -> example.com)
        domain_parts = domain.split('.')
        if len(domain_parts) > 2:
            main_domain = '.'.join(domain_parts[-2:])
            if main_domain in self.api_patterns:
                return self.api_patterns[main_domain]
        
        return None
    
    async def _discover_api_endpoints(self, url: str, search_term: str) -> Dict[str, Any]:
        """Attempt to discover API endpoints by analyzing the site"""
        self.logger.info(f"Attempting to discover API endpoints for {url}")
        
        try:
            # Fetch the main page
            response = await fetch_html(url)
            if not response:
                return None
            
            html_content = response
            
            # Look for API endpoints in script tags
            api_endpoints = self._extract_api_endpoints_from_scripts(html_content, url)
            
            if api_endpoints:
                self.logger.info(f"Found {len(api_endpoints)} potential API endpoints")
                
                # Try each endpoint with the search term
                for endpoint_info in api_endpoints:
                    # Construct base URL if endpoint is relative
                    endpoint_url = endpoint_info["endpoint"]
                    if not endpoint_url.startswith(("http://", "https://")):
                        parsed_url = urlparse(url)
                        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        endpoint_url = urljoin(base_url, endpoint_url)
                    
                    # Try to use this endpoint for search
                    if endpoint_info["type"] == "rest":
                        # Prepare query parameters
                        params = {}
                        for param in endpoint_info.get("params", []):
                            params[param] = search_term
                        
                        # If no params were found, try some common ones
                        if not params:
                            params = {
                                "q": search_term,
                                "query": search_term,
                                "search": search_term,
                                "term": search_term,
                                "keyword": search_term
                            }
                        
                        # Try each parameter one by one
                        for param_name, param_value in params.items():
                            test_params = {param_name: param_value}
                            try:
                                async with httpx.AsyncClient() as client:
                                    response = await client.get(
                                        endpoint_url,
                                        params=test_params,
                                        headers=self.default_headers,
                                        follow_redirects=True,
                                        timeout=10.0
                                    )
                                    
                                    if response.status_code == 200:
                                        # Check if the response looks like search results
                                        try:
                                            data = response.json()
                                            # Check if response has search-like fields
                                            if self._validate_search_response(data, search_term):
                                                # This looks like a valid search API
                                                return {
                                                    "type": "rest",
                                                    "search_endpoint": endpoint_info["endpoint"],
                                                    "search_params": {param_name: "{search_term}"},
                                                    "response_type": "json"
                                                }
                                        except Exception:
                                            # Not a JSON response, continue
                                            pass
                            except Exception as e:
                                self.logger.debug(f"Error testing endpoint {endpoint_url}: {e}")
                                continue
                    
                    elif endpoint_info["type"] == "graphql":
                        # For GraphQL, try a basic search query
                        try:
                            for var_name in ["query", "search", "searchTerm", "term", "q"]:
                                query = endpoint_info.get("query", "query Search($" + var_name + ": String!) { search(query: $" + var_name + ") { results { id title description } } }")
                                variables = {var_name: search_term}
                                
                                payload = {
                                    "query": query,
                                    "variables": variables
                                }
                                
                                if "operationName" in endpoint_info:
                                    payload["operationName"] = endpoint_info["operationName"]
                                
                                async with httpx.AsyncClient() as client:
                                    response = await client.post(
                                        endpoint_url,
                                        json=payload,
                                        headers={
                                            **self.default_headers,
                                            "Content-Type": "application/json"
                                        },
                                        follow_redirects=True,
                                        timeout=10.0
                                    )
                                    
                                    if response.status_code == 200:
                                        try:
                                            data = response.json()
                                            # Check if response has data field (GraphQL standard)
                                            if "data" in data and data["data"] is not None:
                                                # This looks like a valid GraphQL API
                                                return {
                                                    "type": "graphql",
                                                    "search_endpoint": endpoint_info["endpoint"],
                                                    "operation_name": endpoint_info.get("operationName", "Search"),
                                                    "search_variable": var_name,
                                                    "response_type": "json"
                                                }
                                        except Exception:
                                            # Not a JSON response or another error, continue
                                            pass
                        except Exception as e:
                            self.logger.debug(f"Error testing GraphQL endpoint {endpoint_url}: {e}")
                            continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error discovering API endpoints: {e}")
            return None
    
    def _extract_api_endpoints_from_scripts(self, html_content: str, url: str) -> List[Dict[str, Any]]:
        """Extract potential API endpoints from script tags in HTML"""
        endpoints = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all script tags
        script_tags = soup.find_all('script')
        
        # Look for API endpoints in script content
        for script in script_tags:
            script_content = script.string
            if not script_content:
                continue
            
            # Look for API URL patterns
            rest_patterns = [
                r'(\/api\/[\w\-\/]+)',
                r'(https?:\/\/api\.[\w\-\.]+\.[\w]{2,}[\w\-\/\.]+)',
                r'(\/[\w\-\/]+\.json)',
                r'(\/[\w\-\/]+\/search)',
                r'(\/rest\/[\w\-\/]+)'
            ]
            
            for pattern in rest_patterns:
                matches = re.findall(pattern, script_content)
                for match in matches:
                    # Skip if it's an obvious non-API path
                    if any(skip in match.lower() for skip in ['.js', '.css', '.png', '.jpg', '.gif']):
                        continue
                    
                    # Check if this endpoint already exists
                    if not any(e["endpoint"] == match for e in endpoints):
                        endpoints.append({
                            "type": "rest",
                            "endpoint": match,
                            "params": []
                        })
            
            # Look for parameters in AJAX requests
            param_patterns = [
                r'[\'"]([\w]+)[\'"]:\s*[\'"]{search_term}[\'"]',
                r'[\'"](q|query|search|term|keyword)[\'"]:',
                r'data:\s*{[^}]*([\w]+):\s*searchTerm',
                r'params:\s*{[^}]*([\w]+):\s*searchValue'
            ]
            
            for endpoint in endpoints:
                if endpoint["type"] == "rest":
                    for pattern in param_patterns:
                        param_matches = re.findall(pattern, script_content)
                        endpoint["params"].extend(param_matches)
            
            # Look for GraphQL endpoints
            graphql_patterns = [
                r'(\/graphql)',
                r'(\/api\/graphql)',
                r'(https?:\/\/[\w\-\.]+\.[\w]{2,}\/graphql)'
            ]
            
            for pattern in graphql_patterns:
                matches = re.findall(pattern, script_content)
                for match in matches:
                    # Check if this endpoint already exists
                    if not any(e["endpoint"] == match and e["type"] == "graphql" for e in endpoints):
                        # Try to extract operation name
                        operation_pattern = r'operationName:\s*[\'"](\w+)[\'"]'
                        operation_matches = re.findall(operation_pattern, script_content)
                        operation_name = operation_matches[0] if operation_matches else "Search"
                        
                        # Try to extract query
                        query_pattern = r'query:\s*[\'"]([^\'"]*)[\'"]\s*,'
                        query_matches = re.findall(query_pattern, script_content)
                        query = query_matches[0] if query_matches else None
                        
                        endpoint_info = {
                            "type": "graphql",
                            "endpoint": match,
                            "operationName": operation_name
                        }
                        
                        if query:
                            endpoint_info["query"] = query
                            
                        endpoints.append(endpoint_info)
        
        return endpoints
    
    def _validate_search_response(self, data: Dict[str, Any], search_term: str) -> bool:
        """Check if the response data looks like search results"""
        if not data:
            return False
            
        # Look for common result indicators
        result_indicators = ["results", "items", "data", "docs", "hits", "content", "products", "listings"]
        
        for indicator in result_indicators:
            if indicator in data and isinstance(data[indicator], list) and len(data[indicator]) > 0:
                return True
                
        # Check for nested result arrays
        for key, value in data.items():
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    if inner_key in result_indicators and isinstance(inner_value, list) and len(inner_value) > 0:
                        return True
                    
        # Check if response contains the search term
        if isinstance(data, dict):
            json_str = json.dumps(data).lower()
            if search_term.lower() in json_str:
                # If response contains our search term, it might be relevant
                return True
                
        return False
    
    async def _try_api(self, base_url: str, search_term: str, api_info: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Try to use an API endpoint for search"""
        try:
            api_type = api_info.get("type", "rest")
            
            if api_type == "rest":
                return await self._try_rest_api(base_url, search_term, api_info, options)
            elif api_type == "graphql":
                return await self._try_graphql_api(base_url, search_term, api_info, options)
            else:
                self.logger.warning(f"Unknown API type: {api_type}")
                return {"success": False, "error": f"Unknown API type: {api_type}"}
                
        except Exception as e:
            self.logger.error(f"Error trying API: {e}")
            return {"success": False, "error": str(e)}
    
    async def _try_rest_api(self, base_url: str, search_term: str, api_info: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Try a REST API endpoint for search"""
        search_endpoint = api_info.get("search_endpoint", "/api/search")
        search_params = api_info.get("search_params", {"q": "{search_term}"})
        response_type = api_info.get("response_type", "json")
        
        # Construct the full URL
        if search_endpoint.startswith(("http://", "https://")):
            url = search_endpoint
        else:
            url = urljoin(base_url, search_endpoint)
        
        # Replace {search_term} placeholder in parameters
        params = {}
        for key, value in search_params.items():
            if isinstance(value, str) and "{search_term}" in value:
                params[key] = value.replace("{search_term}", search_term)
            else:
                params[key] = search_term
        
        # Add pagination parameters if provided
        page = options.get("page", 1)
        page_size = options.get("page_size", 20)
        
        pagination_params = {
            "page": page,
            "per_page": page_size,
            "limit": page_size,
            "offset": (page - 1) * page_size,
            "size": page_size
        }
        
        # Only add pagination params that aren't already set
        for key, value in pagination_params.items():
            if key not in params:
                params[key] = value
        
        self.logger.info(f"Trying REST API endpoint: {url} with params: {params}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    params=params,
                    headers=self.default_headers,
                    follow_redirects=True,
                    timeout=15.0
                )
                
                if response.status_code != 200:
                    self.logger.warning(f"API request failed with status code: {response.status_code}")
                    return {"success": False, "error": f"API returned status code {response.status_code}"}
                
                # Process the response based on type
                if response_type == "json":
                    try:
                        data = response.json()
                        results = self._extract_results_from_json(data, search_term)
                        
                        if results:
                            return {
                                "success": True,
                                "engine": "api",
                                "results": results,
                                "url": url,
                                "api_type": "rest"
                            }
                        else:
                            self.logger.warning("No results found in API response")
                            return {"success": False, "error": "No results found in API response"}
                            
                    except json.JSONDecodeError:
                        self.logger.warning("Failed to parse JSON response")
                        return {"success": False, "error": "Failed to parse JSON response"}
                        
                elif response_type == "html":
                    # Extract results from HTML response
                    html_content = response.text
                    results = self._extract_results_from_html(html_content, url, search_term)
                    
                    if results:
                        return {
                            "success": True,
                            "engine": "api",
                            "results": results,
                            "url": url,
                            "api_type": "rest_html"
                        }
                    else:
                        self.logger.warning("No results found in HTML response")
                        return {"success": False, "error": "No results found in HTML response"}
                else:
                    self.logger.warning(f"Unsupported response type: {response_type}")
                    return {"success": False, "error": f"Unsupported response type: {response_type}"}
                    
        except Exception as e:
            self.logger.error(f"Error with REST API request: {e}")
            return {"success": False, "error": str(e)}
    
    async def _try_graphql_api(self, base_url: str, search_term: str, api_info: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Try a GraphQL API endpoint for search"""
        search_endpoint = api_info.get("search_endpoint", "/graphql")
        operation_name = api_info.get("operation_name", "Search")
        search_variable = api_info.get("search_variable", "query")
        
        # Construct the full URL
        if search_endpoint.startswith(("http://", "https://")):
            url = search_endpoint
        else:
            url = urljoin(base_url, search_endpoint)
        
        # If a specific query is provided, use it
        query = api_info.get("query")
        if not query:
            # Use a generic search query
            query = f"""
            query {operation_name}(${search_variable}: String!) {{
                search(query: ${search_variable}) {{
                    results {{
                        id
                        title
                        description
                        url
                        image
                    }}
                }}
            }}
            """
        
        # Set up variables
        variables = {search_variable: search_term}
        
        # Add pagination variables if needed
        page = options.get("page", 1)
        page_size = options.get("page_size", 20)
        
        variables.update({
            "page": page,
            "limit": page_size,
            "offset": (page - 1) * page_size
        })
        
        # Build the request payload
        payload = {
            "query": query,
            "variables": variables
        }
        
        if operation_name:
            payload["operationName"] = operation_name
        
        self.logger.info(f"Trying GraphQL API endpoint: {url} with operation: {operation_name}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        **self.default_headers,
                        "Content-Type": "application/json"
                    },
                    follow_redirects=True,
                    timeout=15.0
                )
                
                if response.status_code != 200:
                    self.logger.warning(f"GraphQL request failed with status code: {response.status_code}")
                    return {"success": False, "error": f"GraphQL API returned status code {response.status_code}"}
                
                try:
                    data = response.json()
                    
                    # Check for GraphQL errors
                    if "errors" in data and data["errors"]:
                        error_messages = [error.get("message", "Unknown GraphQL error") for error in data["errors"]]
                        self.logger.warning(f"GraphQL errors: {', '.join(error_messages)}")
                        return {"success": False, "error": f"GraphQL errors: {', '.join(error_messages)}"}
                    
                    results = self._extract_results_from_graphql(data, search_term)
                    
                    if results:
                        return {
                            "success": True,
                            "engine": "api",
                            "results": results,
                            "url": url,
                            "api_type": "graphql"
                        }
                    else:
                        self.logger.warning("No results found in GraphQL response")
                        return {"success": False, "error": "No results found in GraphQL response"}
                        
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse JSON response from GraphQL")
                    return {"success": False, "error": "Failed to parse JSON response from GraphQL"}
                    
        except Exception as e:
            self.logger.error(f"Error with GraphQL API request: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_results_from_json(self, data: Dict[str, Any], search_term: str) -> Dict[str, Any]:
        """Extract search results from a JSON API response"""
        if not data:
            return None
        
        # Try to find the array of results in the response
        result_arrays = self._find_result_arrays(data)
        
        if not result_arrays:
            self.logger.warning("No result arrays found in JSON response")
            return None
        
        # Use the first array that has results and contains relevant data
        items = []
        result_array = None
        
        for array_path, array in result_arrays:
            if array and len(array) > 0:
                # Score this array on how likely it contains search results
                score = self._score_result_array(array, search_term)
                if score > 10:  # Arbitrary threshold
                    result_array = array
                    break
        
        if not result_array:
            self.logger.warning("No suitable result array found in JSON response")
            return None
        
        # Extract structured results from the array
        for item in result_array[:20]:  # Limit to 20 items
            if not isinstance(item, dict):
                continue
                
            result = {}
            
            # Common field names for different properties
            field_mappings = {
                "title": ["title", "name", "heading", "label", "product_name", "productName"],
                "url": ["url", "link", "href", "permalink", "productUrl", "product_url"],
                "description": ["description", "desc", "summary", "content", "snippet", "abstract"],
                "image": ["image", "thumbnail", "imageUrl", "image_url", "thumb", "photo"],
                "price": ["price", "cost", "amount", "value", "priceValue", "price_value"],
                "id": ["id", "uid", "productId", "product_id", "itemId", "item_id"],
                "rating": ["rating", "stars", "score", "ratingValue", "rating_value"],
                "reviews": ["reviews", "reviewCount", "review_count", "totalReviews", "total_reviews"]
            }
            
            # Map the fields from the item
            for prop, keys in field_mappings.items():
                for key in keys:
                    if key in item:
                        value = item[key]
                        if value is not None:
                            result[prop] = str(value)
                            break
            
            # Only include items that have at least a title or description
            if "title" in result or "description" in result:
                items.append(result)
        
        # Build the complete result object
        result_info = {
            "items": items,
            "total_items": len(items)
        }
        
        # Try to find total count information
        count_keys = ["total", "totalCount", "total_count", "count", "totalResults", "total_results"]
        for key in count_keys:
            if key in data:
                try:
                    result_info["estimated_total"] = int(data[key])
                    break
                except (ValueError, TypeError):
                    pass
        
        return result_info
    
    def _extract_results_from_graphql(self, data: Dict[str, Any], search_term: str) -> Dict[str, Any]:
        """Extract search results from a GraphQL API response"""
        if not data or "data" not in data:
            return None
        
        # Start with the data field (standard GraphQL)
        graphql_data = data["data"]
        
        # Look for search results in different common patterns
        result_paths = [
            ["search", "results"],
            ["search", "items"],
            ["search", "edges"],
            ["searchResults", "results"],
            ["searchResults", "items"],
            ["products", "items"],
            ["products", "edges"],
            ["listings", "items"],
            ["listings", "edges"]
        ]
        
        # Try each path to find results
        result_array = None
        for path in result_paths:
            current = graphql_data
            valid_path = True
            
            for key in path:
                if key in current:
                    current = current[key]
                else:
                    valid_path = False
                    break
            
            if valid_path and isinstance(current, list) and len(current) > 0:
                # This path leads to a non-empty array
                result_array = current
                break
        
        # If no predefined path worked, look for arrays recursively
        if not result_array:
            result_arrays = self._find_result_arrays(graphql_data)
            if result_arrays:
                # Use the first array that seems relevant
                for array_path, array in result_arrays:
                    if array and len(array) > 0:
                        # Score this array on how likely it contains search results
                        score = self._score_result_array(array, search_term)
                        if score > 10:  # Arbitrary threshold
                            result_array = array
                            break
        
        if not result_array:
            self.logger.warning("No result array found in GraphQL response")
            return None
        
        # Process items from the found array
        items = []
        
        # Handle "edges" with "node" pattern (common in GraphQL)
        if "node" in result_array[0]:
            result_array = [item["node"] for item in result_array if "node" in item]
        
        # Extract structured results from the array
        for item in result_array[:20]:  # Limit to 20 items
            if not isinstance(item, dict):
                continue
                
            result = {}
            
            # Common field names for different properties
            field_mappings = {
                "title": ["title", "name", "heading", "label", "product_name", "productName"],
                "url": ["url", "link", "href", "permalink", "productUrl", "product_url"],
                "description": ["description", "desc", "summary", "content", "snippet", "abstract"],
                "image": ["image", "thumbnail", "imageUrl", "image_url", "thumb", "photo"],
                "price": ["price", "cost", "amount", "value", "priceValue", "price_value"],
                "id": ["id", "uid", "productId", "product_id", "itemId", "item_id"],
                "rating": ["rating", "stars", "score", "ratingValue", "rating_value"],
                "reviews": ["reviews", "reviewCount", "review_count", "totalReviews", "total_reviews"]
            }
            
            # Map the fields from the item
            for prop, keys in field_mappings.items():
                for key in keys:
                    if key in item:
                        value = item[key]
                        if value is not None:
                            result[prop] = str(value)
                            break
            
            # Only include items that have at least a title or description
            if "title" in result or "description" in result:
                items.append(result)
        
        # Build the complete result object
        result_info = {
            "items": items,
            "total_items": len(items)
        }
        
        # Try to find total count information from common GraphQL patterns
        count_paths = [
            ["search", "totalCount"],
            ["search", "pageInfo", "totalResults"],
            ["searchResults", "totalCount"],
            ["products", "totalCount"],
            ["listings", "totalCount"]
        ]
        
        for path in count_paths:
            current = graphql_data
            valid_path = True
            
            for key in path:
                if key in current:
                    current = current[key]
                else:
                    valid_path = False
                    break
            
            if valid_path and isinstance(current, (int, str)):
                try:
                    result_info["estimated_total"] = int(current)
                    break
                except (ValueError, TypeError):
                    pass
        
        return result_info
    
    def _extract_results_from_html(self, html_content: str, base_url: str, search_term: str) -> Dict[str, Any]:
        """Extract search results from an HTML response"""
        if not html_content:
            return None
        
        soup = BeautifulSoup(html_content, 'html.parser')
        items = []
        
        # Look for search result containers
        result_containers = []
        
        # Common selectors for search result containers
        result_selectors = [
            ".search-results", ".results", ".products", ".listings",
            "[class*='search-results']", "[class*='product-list']", "[class*='search-list']",
            "[data-testid*='search-results']", "[aria-label*='search results']"
        ]
        
        # Try to find a container first
        container = None
        for selector in result_selectors:
            container = soup.select_one(selector)
            if container:
                break
        
        # If no container found, use the body
        if not container:
            container = soup.body
        
        # Look for individual result items
        item_selectors = [
            ".search-result", ".result", ".product", ".item", ".listing",
            "[class*='result-item']", "[class*='search-item']", "[class*='product-item']",
            "article", ".card"
        ]
        
        for selector in item_selectors:
            results = container.select(selector)
            if results and len(results) >= 3:  # We need at least 3 to be sure these are search results
                result_containers = results
                break
        
        # If no results found with specific selectors, try to find similar siblings
        if not result_containers:
            # Look for parent elements that have multiple similar direct children
            for parent in container.find_all(['div', 'ul', 'ol', 'section']):
                children = parent.find_all(recursive=False)
                if len(children) >= 3 and all(children[0].name == child.name for child in children):
                    result_containers = children
                    break
        
        # Process each result item
        for result_item in result_containers[:20]:  # Limit to 20 items
            item = {}
            
            # Extract title
            title_element = result_item.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']) or result_item.find('a')
            if title_element:
                item["title"] = title_element.get_text(strip=True)
            
            # Extract URL
            link = result_item.find('a')
            if link and link.get('href'):
                item["url"] = urljoin(base_url, link['href'])
            
            # Extract image
            img = result_item.find('img')
            if img and img.get('src'):
                item["image"] = urljoin(base_url, img['src'])
            
            # Extract description
            desc_element = result_item.find('p') or result_item.find(class_=lambda c: c and any(word in c for word in ['desc', 'summary', 'text', 'content']))
            if desc_element and desc_element != title_element:
                item["description"] = desc_element.get_text(strip=True)
            
            # Extract price
            price_element = result_item.find(class_=lambda c: c and 'price' in c) or result_item.find(string=lambda s: s and re.search(r'(\$|€|£)\s*\d+', s))
            if price_element:
                if hasattr(price_element, 'get_text'):
                    price_text = price_element.get_text(strip=True)
                else:
                    price_text = str(price_element).strip()
                
                # Extract price with regex
                price_match = re.search(r'(\$|€|£)\s*(\d+(?:,\d+)*(?:\.\d+)?)', price_text)
                if price_match:
                    item["price"] = price_match.group(0)
            
            # Only include items with at least a title or URL
            if "title" in item or "url" in item:
                items.append(item)
        
        # Build the complete result object
        result_info = {
            "items": items,
            "total_items": len(items)
        }
        
        # Try to extract total count from text
        count_match = re.search(r'(\d+(?:,\d+)*)\s+(?:results|items|products|listings|matches)', soup.get_text())
        if count_match:
            try:
                result_info["estimated_total"] = int(count_match.group(1).replace(',', ''))
            except (ValueError, TypeError):
                pass
        
        return result_info
    
    def _find_result_arrays(self, data: Dict[str, Any], path: List[str] = None) -> List[tuple]:
        """Recursively find arrays in the data that might contain results"""
        if path is None:
            path = []
            
        result_arrays = []
        
        if isinstance(data, list) and len(data) > 0:
            # This is a candidate array
            result_arrays.append((path, data))
            
        elif isinstance(data, dict):
            # Look for arrays in this dictionary
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    sub_arrays = self._find_result_arrays(value, path + [key])
                    result_arrays.extend(sub_arrays)
        
        return result_arrays
    
    def _score_result_array(self, array: List[Dict[str, Any]], search_term: str) -> int:
        """Score an array on how likely it contains search results"""
        if not array or not isinstance(array, list) or len(array) == 0:
            return 0
        
        score = 0
        first_item = array[0]
        
        # Array of objects is more likely to be results
        if isinstance(first_item, dict):
            score += 10
            
            # Check for common result keys
            result_keys = ["id", "title", "name", "description", "url", "link", "image", "price"]
            for key in result_keys:
                if key in first_item:
                    score += 3
            
            # Check if search term appears in any string values
            search_term_lower = search_term.lower()
            for value in first_item.values():
                if isinstance(value, str) and search_term_lower in value.lower():
                    score += 15
                    break
        
        # More items is better for search results
        if len(array) >= 3:
            score += 5
        if len(array) >= 10:
            score += 5
        
        return score