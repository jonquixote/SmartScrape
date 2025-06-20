"""
Site Discovery Module

Provides utilities for discovering website structure, sitemaps, and other key features
of websites to enable intelligent crawling.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
# Additional imports for enhanced sitemap processing
try:
    import lxml.etree
    import requests
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    logging.warning("lxml or requests not available, falling back to basic XML parsing")

# Import enhanced utilities
from utils.html_utils import parse_html, extract_text_fast, find_by_xpath, select_with_css
from utils.retry_utils import with_exponential_backoff
from utils.http_utils import fetch_html

# Import search automation for form detection
from components.search_automation import SearchFormDetector
from core.service_interface import BaseService

class SiteDiscovery(BaseService):
    """
    Site discovery component for finding sitemaps and analyzing website structure.
    """
    
    def __init__(self):
        """Initialize site discovery component."""
        self._initialized = False
        self.search_detector = None
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the service with the given configuration."""
        if self._initialized:
            return
            
        self.search_detector = SearchFormDetector()
        self._initialized = True
        logging.info("SiteDiscovery service initialized")
    
    def shutdown(self) -> None:
        """Perform cleanup and shutdown operations."""
        self._initialized = False
        logging.info("SiteDiscovery service shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "site_discovery"
    
    async def analyze_site(self, url: str) -> Dict[str, Any]:
        """
        Analyze a website to determine its characteristics and capabilities.
        
        Args:
            url: The URL of the website to analyze
            
        Returns:
            Dictionary with site analysis information including:
            - has_search_form: Whether the site has search forms
            - has_search_api: Whether the site appears to have API endpoints
            - has_pagination: Whether the site has pagination
            - requires_javascript: Whether the site requires JavaScript
            - has_login: Whether the site has login functionality
            - has_captcha: Whether the site has CAPTCHA protection
        """
        result = {
            "has_search_form": False,
            "has_search_api": False,
            "has_pagination": False,
            "requires_javascript": False,
            "has_login": False,
            "has_captcha": False,
            "site_type": "unknown",
            "error": None
        }
        
        try:
            # Fetch the website content
            response = await fetch_html(url)
            if not response:
                result["error"] = "Failed to fetch website content"
                return result
            
            # Parse HTML for analysis
            soup = parse_html(response)
            
            # Check for search forms
            search_forms = find_by_xpath(soup, '//form[.//input[@type="search"] or .//input[contains(@name, "search") or contains(@placeholder, "search")] or contains(@action, "search")]')
            if search_forms:
                result["has_search_form"] = True
            
            # Check for pagination indicators
            pagination_selectors = [
                '//a[contains(@class, "page") or contains(@class, "next") or contains(@class, "prev")]',
                '//nav[contains(@class, "pagination")]',
                '//*[contains(text(), "Next") or contains(text(), "Previous") or contains(text(), "Page")]'
            ]
            for selector in pagination_selectors:
                if find_by_xpath(soup, selector):
                    result["has_pagination"] = True
                    break
            
            # Check for JavaScript dependencies
            script_tags = find_by_xpath(soup, '//script')
            js_indicators = ['React', 'Vue', 'Angular', 'spa', 'app.js', 'bundle.js']
            for script in script_tags:
                script_content = script.get_text() if hasattr(script, 'get_text') else str(script)
                if any(indicator in script_content for indicator in js_indicators):
                    result["requires_javascript"] = True
                    break
            
            # Check for login functionality
            login_indicators = [
                '//form[.//input[@type="password"]]',
                '//a[contains(@href, "login") or contains(@href, "signin")]',
                '//*[contains(text(), "Login") or contains(text(), "Sign in")]'
            ]
            for selector in login_indicators:
                if find_by_xpath(soup, selector):
                    result["has_login"] = True
                    break
            
            # Check for CAPTCHA
            captcha_indicators = [
                '//*[contains(@class, "captcha") or contains(@id, "captcha")]',
                '//script[contains(@src, "recaptcha")]'
            ]
            for selector in captcha_indicators:
                if find_by_xpath(soup, selector):
                    result["has_captcha"] = True
                    break
            
            # Check for API endpoints (basic heuristics)
            if '/api/' in response or 'application/json' in response:
                result["has_search_api"] = True
            
            # Determine basic site type
            if result["has_search_form"]:
                result["site_type"] = "search_portal"
            elif result["has_pagination"]:
                result["site_type"] = "listing"
            elif 'blog' in url.lower() or 'news' in url.lower():
                result["site_type"] = "blog"
            elif 'shop' in url.lower() or 'store' in url.lower() or 'buy' in url.lower():
                result["site_type"] = "e-commerce"
            
            return result
            
        except Exception as e:
            logging.error(f"Error analyzing site {url}: {str(e)}")
            result["error"] = str(e)
            return result
    
    @with_exponential_backoff(max_attempts=3)
    async def find_sitemap(self, url: str) -> Dict[str, Any]:
        """
        Find sitemap.xml for a website.
        
        Args:
            url: The URL of the website
            
        Returns:
            Dictionary with sitemap information
        """
        result = {
            "sitemap_found": False,
            "sitemap_url": None,
            "error": None,
            "sitemap_count": 0,
            "urls_count": 0
        }
        
        try:
            # Parse the URL to get the base domain
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # First try robots.txt
            robots_url = f"{base_url}/robots.txt"
            robots_response = await fetch_html(robots_url)
            
            if robots_response:
                # Parse robots.txt content to find Sitemap entries
                sitemap_urls = []
                for line in robots_response.splitlines():
                    if line.lower().startswith("sitemap:"):
                        sitemap_url = line[8:].strip()
                        sitemap_urls.append(sitemap_url)
                
                if sitemap_urls:
                    result["sitemap_found"] = True
                    result["sitemap_url"] = sitemap_urls[0]  # Use the first sitemap
                    result["sitemap_count"] = len(sitemap_urls)
                    return result
            
            # If no sitemap found in robots.txt, try common sitemap locations
            common_sitemap_paths = [
                "/sitemap.xml",
                "/sitemap_index.xml",
                "/sitemap/sitemap.xml",
                "/sitemapindex.xml",
                "/news-sitemap.xml",
                "/post-sitemap.xml",
                "/page-sitemap.xml",
                "/category-sitemap.xml"
            ]
            
            # Try common sitemap paths concurrently for better performance
            tasks = [fetch_html(f"{base_url}{path}") for path in common_sitemap_paths]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    continue
                    
                if response and ("<?xml" in response or "<urlset" in response or "<sitemapindex" in response):
                    sitemap_url = f"{base_url}{common_sitemap_paths[i]}"
                    result["sitemap_found"] = True
                    result["sitemap_url"] = sitemap_url
                    result["sitemap_count"] = 1
                    return result
            
            # As a last resort, check the HTML for sitemap links
            response = await fetch_html(url)
            if response:
                # Use optimized HTML parsing
                soup = parse_html(response)
                
                # Look for sitemap links in HTML using efficient xpath
                sitemap_links = []
                
                # Look for links with "sitemap" in href or text using XPath for better performance
                a_elements = find_by_xpath(soup, '//a[contains(@href, "sitemap") or contains(text(), "sitemap") or contains(text(), "Sitemap")]')
                
                for a in a_elements:
                    href = a.get('href', '')
                    if href:
                        full_url = urljoin(base_url, href)
                        sitemap_links.append(full_url)
                
                if sitemap_links:
                    result["sitemap_found"] = True
                    result["sitemap_url"] = sitemap_links[0]
                    result["sitemap_count"] = len(sitemap_links)
                    return result
            
            # No sitemap found
            result["error"] = "No sitemap found"
            return result
            
        except Exception as e:
            logging.error(f"Error finding sitemap: {str(e)}")
            result["error"] = str(e)
            return result

    @with_exponential_backoff(max_attempts=3)
    async def process_sitemap(self, sitemap_url: str, keywords: List[str] = None, max_urls: int = 500) -> List[Dict[str, Any]]:
        """
        Process a sitemap XML to extract URLs with optimized XML parsing.
        
        Args:
            sitemap_url: URL of the sitemap
            keywords: List of keywords to prioritize URLs
            max_urls: Maximum number of URLs to return
            
        Returns:
            List of URL dictionaries with metadata
        """
        try:
            # Fetch the sitemap with improved retry
            sitemap_content = await fetch_html(sitemap_url)
            
            if not sitemap_content:
                logging.error(f"Failed to fetch sitemap: {sitemap_url}")
                return []
            
            # Check if it's a sitemap index (contains multiple sitemaps)
            if "<sitemapindex" in sitemap_content:
                # Parse sitemap index with efficient exception handling
                try:
                    root = ET.fromstring(sitemap_content)
                    
                    # Find namespace using more robust method
                    ns = {}
                    if '}' in root.tag:
                        ns = {'sm': root.tag.split('}')[0][1:]}
                    
                    # Get all sitemap URLs efficiently
                    sitemap_urls = []
                    for sitemap in root.findall('.//sm:sitemap', ns) if ns else root.findall('.//sitemap'):
                        loc_elem = sitemap.find('./sm:loc', ns) if ns else sitemap.find('./loc')
                        if loc_elem is not None and loc_elem.text:
                            sitemap_urls.append(loc_elem.text.strip())
                    
                    # Process multiple sitemaps concurrently for better performance
                    # Limit to avoid overloading the server
                    concurrent_limit = min(5, len(sitemap_urls))
                    selected_sitemaps = sitemap_urls[:concurrent_limit]
                    
                    tasks = [self.process_sitemap(url, keywords, max_urls // concurrent_limit) 
                             for url in selected_sitemaps]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Combine results, filtering out exceptions
                    all_urls = []
                    for result in results:
                        if isinstance(result, Exception):
                            continue
                        all_urls.extend(result)
                    
                    # Sort and truncate to max_urls
                    all_urls.sort(key=lambda x: x.get("priority", 0.5), reverse=True)
                    return all_urls[:max_urls]
                    
                except ET.ParseError as e:
                    logging.warning(f"XML parsing error in sitemap index: {e}")
                    # Fall back to regex-based extraction
                    url_matches = re.findall(r'<loc>([^<]+)</loc>', sitemap_content)
                    sitemap_urls = [url.strip() for url in url_matches]
                    
                    # Process a limited number of sitemaps with regex fallback
                    all_urls = []
                    for sub_url in sitemap_urls[:3]:
                        sub_results = await self.process_sitemap(sub_url, keywords, max_urls // 3)
                        all_urls.extend(sub_results)
                    
                    return all_urls[:max_urls]
            else:
                # Process a regular sitemap with URLs
                urls = []
                
                try:
                    # Try to parse as XML with efficient exception handling
                    root = ET.fromstring(sitemap_content)
                    
                    # Find namespace using more robust method
                    ns = {}
                    if '}' in root.tag:
                        ns = {'sm': root.tag.split('}')[0][1:]}
                    
                    # Find all URL elements efficiently
                    url_elements = root.findall('.//sm:url', ns) if ns else root.findall('.//url')
                    
                    for url_elem in url_elements:
                        url_data = {}
                        
                        # Get location
                        loc_elem = url_elem.find('./sm:loc', ns) if ns else url_elem.find('./loc')
                        if loc_elem is not None and loc_elem.text:
                            url_data["url"] = loc_elem.text.strip()
                        else:
                            continue  # Skip URLs without location
                        
                        # Get last modified date if available
                        lastmod_elem = url_elem.find('./sm:lastmod', ns) if ns else url_elem.find('./lastmod')
                        if lastmod_elem is not None and lastmod_elem.text:
                            url_data["lastmod"] = lastmod_elem.text.strip()
                        
                        # Get change frequency if available
                        changefreq_elem = url_elem.find('./sm:changefreq', ns) if ns else url_elem.find('./changefreq')
                        if changefreq_elem is not None and changefreq_elem.text:
                            url_data["changefreq"] = changefreq_elem.text.strip()
                        
                        # Get priority if available
                        priority_elem = url_elem.find('./sm:priority', ns) if ns else url_elem.find('./priority')
                        if priority_elem is not None and priority_elem.text:
                            try:
                                url_data["priority"] = float(priority_elem.text.strip())
                            except ValueError:
                                url_data["priority"] = 0.5  # Default priority
                        else:
                            url_data["priority"] = 0.5  # Default priority
                        
                        # Calculate relevance score based on keywords if provided
                        if keywords:
                            url_str = url_data["url"].lower()
                            relevance = sum(0.2 for keyword in keywords if keyword.lower() in url_str)
                            
                            # Adjust priority based on relevance
                            url_data["priority"] = min(1.0, url_data["priority"] + relevance)
                        
                        urls.append(url_data)
                        
                except ET.ParseError as e:
                    logging.warning(f"Failed to parse sitemap as XML: {e}")
                    # Try to extract URLs using regex as a fallback
                    url_matches = re.findall(r'<loc>([^<]+)</loc>', sitemap_content)
                    
                    for url_match in url_matches:
                        url_data = {
                            "url": url_match.strip(),
                            "priority": 0.5  # Default priority
                        }
                        
                        # Calculate relevance score based on keywords if provided
                        if keywords:
                            url_str = url_match.lower()
                            relevance = sum(0.2 for keyword in keywords if keyword.lower() in url_str)
                            
                            # Adjust priority based on relevance
                            url_data["priority"] = min(1.0, url_data["priority"] + relevance)
                        
                        urls.append(url_data)
                
                # Sort by priority (highest first) and limit results
                urls.sort(key=lambda x: x.get("priority", 0.5), reverse=True)
                return urls[:max_urls]
                
        except Exception as e:
            logging.error(f"Error processing sitemap: {str(e)}")
            return []

    @with_exponential_backoff(max_attempts=3)
    async def detect_site_features(self, url: str) -> Dict[str, Any]:
        """
        Detect features of a website such as search forms, login forms, etc.
        
        Args:
            url: The URL of the website
            
        Returns:
            Dictionary with detected features
        """
        features = {
            "has_search": False,
            "search_forms": [],
            "has_login": False,
            "login_forms": [],
            "has_sitemap": False,
            "sitemap_url": None,
            "has_robots": False,
            "robots_url": None,
            "has_api_endpoints": False,
            "api_endpoints": [],
            "javascript_frameworks": [],
            "meta_tags": {},
            "response_time_ms": None
        }
        
        try:
            # Parse the URL to get the base domain
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Record start time for performance metrics
            start_time = asyncio.get_event_loop().time()
            
            # Fetch the page with improved error handling
            html_content = await fetch_html(url)
            
            # Calculate response time
            end_time = asyncio.get_event_loop().time()
            features["response_time_ms"] = round((end_time - start_time) * 1000)
            
            if html_content:
                # Use optimized HTML parsing
                soup = parse_html(html_content)
                
                # Detect search forms using our specialized detector
                search_forms = await self.search_detector.detect_search_forms(html_content)
                if search_forms:
                    features["has_search"] = True
                    features["search_forms"] = [
                        {
                            "type": form.get("type", "standard_form"),
                            "score": form.get("search_relevance_score", 0),
                            "field_count": len(form.get("fields", [])),
                            "id": form.get("id", "")
                        }
                        for form in search_forms[:3]  # Include top 3 most relevant forms
                    ]
                
                # Detect login forms using efficient XPath
                login_forms_xpath = '//form[.//input[@type="password"]]'
                login_forms = find_by_xpath(soup, login_forms_xpath)
                
                if login_forms:
                    features["has_login"] = True
                    for form in login_forms[:2]:  # Include top 2 login forms
                        form_action = form.get('action', '')
                        login_form_data = {
                            "id": form.get('id', ''),
                            "method": form.get('method', 'post').lower(),
                            "url": urljoin(base_url, form_action) if form_action else None
                        }
                        features["login_forms"].append(login_form_data)
                
                # Check for API endpoints in JavaScript using optimized pattern matching
                script_tags = find_by_xpath(soup, '//script[not(@src)]')  # Inline scripts
                api_endpoints = set()
                
                for script in script_tags:
                    script_content = extract_text_fast(script)
                    if script_content:
                        # Look for API endpoint patterns with improved regex
                        api_matches = re.findall(r'["\']((\/api\/|\/rest\/|\/graphql|\/v[0-9]+\/|\/service\/)[^"\'\s]+)["\']', script_content)
                        for match in api_matches:
                            api_endpoint = urljoin(base_url, match[0])
                            api_endpoints.add(api_endpoint)
                
                if api_endpoints:
                    features["has_api_endpoints"] = True
                    features["api_endpoints"] = list(api_endpoints)[:10]  # Limit to top 10 endpoints
                    
                # Detect JavaScript frameworks
                frameworks = []
                
                # Check for React
                if re.search(r'react|reactjs|react-dom', html_content, re.I):
                    frameworks.append("React")
                
                # Check for Angular
                if re.search(r'ng-app|angular|ng-controller', html_content, re.I):
                    frameworks.append("Angular")
                
                # Check for Vue
                if re.search(r'vue|v-bind|v-model|v-if|v-for', html_content, re.I):
                    frameworks.append("Vue")
                
                # Check for jQuery
                if re.search(r'jquery|\$\(', html_content, re.I):
                    frameworks.append("jQuery")
                
                # Check for Bootstrap
                if re.search(r'bootstrap|class="[^"]*btn[^"]*"|class="[^"]*container[^"]*"', html_content, re.I):
                    frameworks.append("Bootstrap")
                
                if frameworks:
                    features["javascript_frameworks"] = frameworks
                
                # Extract meta tags for SEO analysis
                meta_tags = {}
                
                # Title
                title_tag = soup.title
                if title_tag:
                    meta_tags["title"] = extract_text_fast(title_tag)
                
                # Description
                desc_meta = select_with_css(soup, 'meta[name="description"]')
                if desc_meta:
                    meta_tags["description"] = desc_meta.get('content', '')
                
                # Keywords
                keywords_meta = select_with_css(soup, 'meta[name="keywords"]')
                if keywords_meta:
                    meta_tags["keywords"] = keywords_meta.get('content', '')
                
                # OpenGraph tags
                og_title = select_with_css(soup, 'meta[property="og:title"]')
                if og_title:
                    meta_tags["og:title"] = og_title.get('content', '')
                
                og_desc = select_with_css(soup, 'meta[property="og:description"]')
                if og_desc:
                    meta_tags["og:description"] = og_desc.get('content', '')
                
                og_image = select_with_css(soup, 'meta[property="og:image"]')
                if og_image:
                    meta_tags["og:image"] = og_image.get('content', '')
                
                features["meta_tags"] = meta_tags
            
            # Check for robots.txt
            robots_url = f"{base_url}/robots.txt"
            robots_content = await fetch_html(robots_url)
            
            if robots_content:
                features["has_robots"] = True
                features["robots_url"] = robots_url
            
            # Check for sitemap
            sitemap_result = await self.find_sitemap(url)
            if sitemap_result["sitemap_found"]:
                features["has_sitemap"] = True
                features["sitemap_url"] = sitemap_result["sitemap_url"]
            
            return features
            
        except Exception as e:
            logging.error(f"Error detecting site features: {str(e)}")
            return features

    @with_exponential_backoff(max_attempts=3)
    async def detect_graphql_endpoints(self, url: str) -> List[Dict[str, Any]]:
        """
        Detect GraphQL endpoints on a website through multiple detection methods.
        
        Args:
            url: The URL of the website
            
        Returns:
            List of dictionaries containing GraphQL endpoint information
        """
        graphql_endpoints = []
        
        try:
            # Parse the URL to get the base domain
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Fetch the page
            html_content = await fetch_html(url)
            
            if not html_content:
                return graphql_endpoints
                
            # Parse HTML
            soup = parse_html(html_content)
            
            # Method 1: Check for common GraphQL endpoint paths
            common_graphql_paths = [
                "/graphql",
                "/api/graphql",
                "/v1/graphql",
                "/v2/graphql",
                "/query",
                "/api/query",
                "/gql",
                "/api/gql",
                "/graphiql",
                "/playground"
            ]
            
            # Try common GraphQL paths concurrently
            tasks = [self._check_graphql_endpoint(f"{base_url}{path}") for path in common_graphql_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    continue
                    
                if result and result.get("is_graphql", False):
                    endpoint_data = {
                        "url": f"{base_url}{common_graphql_paths[i]}",
                        "detection_method": "common_path",
                        "confidence": result.get("confidence", 0.7),
                        "introspection_enabled": result.get("introspection_enabled", False),
                        "schema_available": result.get("schema_available", False)
                    }
                    graphql_endpoints.append(endpoint_data)
            
            # Method 2: Look for GraphQL URLs in JavaScript
            script_tags = find_by_xpath(soup, '//script')
            
            for script in script_tags:
                # Check src attribute for external scripts
                src = script.get('src', '')
                if src and ('graphql' in src or 'apollo' in src or 'relay' in src):
                    graphql_endpoints.append({
                        "url": urljoin(base_url, src),
                        "detection_method": "script_src",
                        "confidence": 0.5,
                        "resource_type": "javascript_library"
                    })
                
                # Check inline scripts
                script_content = extract_text_fast(script)
                if script_content:
                    # Look for GraphQL client initialization patterns
                    apollo_client = re.search(r'new ApolloClient\(\s*\{\s*uri\s*:\s*[\'"]([^\'"]+)[\'"]', script_content)
                    if apollo_client:
                        endpoint_url = apollo_client.group(1)
                        graphql_endpoints.append({
                            "url": urljoin(base_url, endpoint_url),
                            "detection_method": "apollo_client",
                            "confidence": 0.9,
                            "client_library": "apollo"
                        })
                    
                    # Look for GraphQL operations
                    gql_operations = re.findall(r'(query|mutation)\s+(\w+)?\s*\{', script_content)
                    if gql_operations:
                        # Extract associated endpoint if available
                        endpoint_match = re.search(r'(graphql|gql).*?[\'"]([^\'"]+)[\'"]', script_content)
                        endpoint_url = endpoint_match.group(2) if endpoint_match else "/graphql"
                        graphql_endpoints.append({
                            "url": urljoin(base_url, endpoint_url),
                            "detection_method": "gql_operations",
                            "confidence": 0.8,
                            "operations_detected": [op[0] for op in gql_operations[:5]]
                        })
            
            # Method 3: Check network requests for GraphQL patterns
            # This would typically require JavaScript execution which we can't do here directly
            # But we can check for GraphQL related comments or patterns
            
            # Look for GraphQL schema definition files
            schema_patterns = [
                'schema.graphql',
                'schema.json',
                'graphql/schema',
                'graphql-schema'
            ]
            
            for pattern in schema_patterns:
                matches = re.findall(f'[\'"]([^\'"]*/({pattern})[^\'"]*)[\'"]', html_content)
                for match in matches:
                    graphql_endpoints.append({
                        "url": urljoin(base_url, match[0]),
                        "detection_method": "schema_file",
                        "confidence": 0.6,
                        "resource_type": "schema_definition"
                    })
            
            # Deduplicate endpoints based on URL
            unique_endpoints = {}
            for endpoint in graphql_endpoints:
                url = endpoint["url"]
                if url not in unique_endpoints or endpoint["confidence"] > unique_endpoints[url]["confidence"]:
                    unique_endpoints[url] = endpoint
            
            return list(unique_endpoints.values())
            
        except Exception as e:
            logging.error(f"Error detecting GraphQL endpoints: {str(e)}")
            return graphql_endpoints
    
    async def _check_graphql_endpoint(self, url: str) -> Dict[str, Any]:
        """
        Test if a URL is likely a GraphQL endpoint by sending an introspection query.
        
        Args:
            url: The URL to test
            
        Returns:
            Dictionary with GraphQL endpoint information
        """
        result = {
            "is_graphql": False,
            "confidence": 0.0,
            "introspection_enabled": False,
            "schema_available": False
        }
        
        # Basic introspection query to test if it's a GraphQL endpoint
        introspection_query = {
            "query": "{__typename}"
        }
        
        try:
            # Send a POST request with the introspection query
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=introspection_query, timeout=5) as response:
                    if response.status in (200, 400, 405):  # GraphQL endpoints typically return 200 or 400 for invalid queries
                        content_type = response.headers.get('Content-Type', '')
                        is_json = 'application/json' in content_type
                        
                        if is_json:
                            response_data = await response.json()
                            
                            # Check if it looks like a GraphQL response
                            if 'data' in response_data or 'errors' in response_data:
                                result["is_graphql"] = True
                                result["confidence"] = 0.8
                                
                                # If we got data back, introspection is enabled
                                if 'data' in response_data and response_data['data'] and '__typename' in response_data['data']:
                                    result["introspection_enabled"] = True
                                    result["confidence"] = 1.0
                        
                        # Even without JSON, some GraphQL endpoints might return errors in different formats
                        elif response.status == 400:
                            result["is_graphql"] = True
                            result["confidence"] = 0.6
            
            # Try introspection query if basic query didn't confirm GraphQL
            if not result["introspection_enabled"] and result["is_graphql"]:
                # Full introspection query to get schema
                schema_query = {
                    "query": """
                        {
                          __schema {
                            queryType {
                              name
                            }
                          }
                        }
                    """
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=schema_query, timeout=5) as response:
                        if response.status == 200:
                            response_data = await response.json()
                            
                            if ('data' in response_data and 
                                '__schema' in response_data.get('data', {}) and 
                                'queryType' in response_data.get('data', {}).get('__schema', {})):
                                result["schema_available"] = True
                                result["introspection_enabled"] = True
                                result["confidence"] = 1.0
                
        except Exception as e:
            logging.debug(f"Error testing GraphQL endpoint {url}: {str(e)}")
            # Don't mark as error, this is an expected case for non-GraphQL endpoints
            pass
            
        return result

    async def extract_urls_from_sitemap(self, sitemap_url: str, max_urls_per_sitemap: int = 100) -> List[str]:
        """
        Extract URLs from a sitemap using robust XML parsing with lxml.
        
        This method implements the enhanced sitemap parsing from the gameplan,
        with proper namespace handling and support for sitemap index files.
        
        Args:
            sitemap_url: URL of the sitemap to process
            max_urls_per_sitemap: Maximum number of URLs to extract
            
        Returns:
            List of URLs extracted from the sitemap
        """
        try:
            if LXML_AVAILABLE:
                # Use lxml for robust XML parsing
                import lxml.etree as ET_lxml
                import requests
                
                # Use requests for robust HTTP handling
                response = requests.get(sitemap_url, timeout=30)
                response.raise_for_status()
                
                root = ET_lxml.fromstring(response.content)
                urls = []
                
                # Handle sitemap index files
                if 'sitemapindex' in root.tag:
                    for sitemap in root.xpath('//s:sitemap/s:loc', namespaces={'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}):
                        sub_urls = await self.extract_urls_from_sitemap(sitemap.text, max_urls_per_sitemap // 10)
                        urls.extend(sub_urls)
                else:
                    # Regular sitemap
                    for url in root.xpath('//s:url/s:loc', namespaces={'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}):
                        urls.append(url.text)
                
                return urls[:max_urls_per_sitemap]
            else:
                # Fallback to existing process_sitemap method
                logging.info(f"Using fallback sitemap processing for {sitemap_url}")
                sitemap_data = await self.process_sitemap(sitemap_url, max_urls=max_urls_per_sitemap)
                return [item['url'] for item in sitemap_data if 'url' in item][:max_urls_per_sitemap]
        except Exception as e:
            logging.error(f"Error extracting URLs from sitemap {sitemap_url}: {e}")
            return []
