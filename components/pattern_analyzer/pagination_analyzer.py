"""
Pagination Pattern Analyzer Module

This module provides functionality to detect and analyze pagination patterns
on websites, including next/prev links, page number links, and infinite scroll mechanisms.
"""

import re
import logging
import json
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse, parse_qs, urlunparse, urlencode
from bs4 import BeautifulSoup, Tag

from components.pattern_analyzer.base_analyzer import PatternAnalyzer, get_registry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PaginationAnalyzer")

class PaginationAnalyzer(PatternAnalyzer):
    """
    Analyzer for detecting and analyzing pagination patterns on web pages.
    This includes next/prev links, numeric pagination, load more buttons,
    and infinite scroll triggers.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the pagination analyzer.
        
        Args:
            confidence_threshold: Minimum confidence level for pattern detection
        """
        super().__init__(confidence_threshold)
        
        # Common text patterns in pagination elements
        self.pagination_text_patterns = [
            re.compile(r'next\s*page', re.I),
            re.compile(r'next\s*»', re.I),
            re.compile(r'»|›|⟩|▶', re.I),  # Common next symbols
            re.compile(r'previous\s*page', re.I),
            re.compile(r'«\s*previous', re.I),
            re.compile(r'«|‹|⟨|◀', re.I),  # Common previous symbols
            re.compile(r'load\s*more', re.I),
            re.compile(r'show\s*more', re.I),
            re.compile(r'more\s*results', re.I),
            re.compile(r'see\s*more', re.I)
        ]
        
        # Common class and ID patterns for pagination elements
        self.pagination_attr_patterns = [
            re.compile(r'pag(e|ing|ination)', re.I),
            re.compile(r'next-?page', re.I),
            re.compile(r'prev-?page', re.I),
            re.compile(r'pages', re.I),
            re.compile(r'load-?more', re.I),
            re.compile(r'more-?(content|results)', re.I)
        ]
        
        # Common selectors for pagination containers
        self.pagination_container_selectors = [
            '.pagination',
            '.pager',
            '.pages',
            '.page-numbers',
            'nav.pagination',
            'ul.pagination',
            'div.pagination',
            '.paginator',
            '#pagination',
            '[role="navigation"]',
            '.load-more',
            '.infinite-scroll',
            '.next-prev'
        ]
    
    async def analyze(self, html: str, url: str) -> Dict[str, Any]:
        """
        Analyze a page to detect pagination patterns.
        
        Args:
            html: HTML content of the page
            url: URL of the page
            
        Returns:
            Dictionary with detected pagination patterns
        """
        logger.info(f"Analyzing pagination patterns on {url}")
        soup = self.parse_html(html)
        domain = self.get_domain(url)
        
        # Parse current URL to understand its structure
        parsed_url = urlparse(url)
        url_query_params = parse_qs(parsed_url.query)
        
        # Results will contain all detected pagination patterns
        results = {
            "pagination_type": "none",
            "patterns": [],
            "current_page": None,
            "total_pages": None,
            "next_page_url": None,
            "prev_page_url": None,
            "page_urls": [],
            "confidence_score": 0.0
        }
        
        # First, check for common pagination containers
        pagination_containers = []
        for selector in self.pagination_container_selectors:
            containers = soup.select(selector)
            pagination_containers.extend(containers)
        
        # If no containers found, try a more general approach
        if not pagination_containers:
            pagination_containers = self._find_potential_pagination_containers(soup)
        
        # Analyze each potential pagination container
        for container in pagination_containers:
            pagination_data = self._analyze_pagination_container(container, url, url_query_params)
            
            # If this container has higher confidence than previous ones, use it
            if pagination_data["confidence_score"] > results["confidence_score"]:
                # Keep original patterns list and merge with new patterns
                patterns = results["patterns"]
                patterns.append(pagination_data)
                
                # Update with most confident pagination data
                results = pagination_data
                results["patterns"] = patterns
        
        # If no containers found but URL has page parameter, 
        # try to infer pagination pattern from URL
        if results["pagination_type"] == "none" and self._url_has_page_param(url_query_params):
            url_pagination_data = self._analyze_url_pagination(url, url_query_params)
            
            if url_pagination_data["confidence_score"] > results["confidence_score"]:
                results = url_pagination_data
                results["patterns"] = [url_pagination_data]
        
        # Check for infinite scroll indicators if no traditional pagination found
        if results["pagination_type"] == "none":
            infinite_scroll_data = self._detect_infinite_scroll(soup, url)
            
            if infinite_scroll_data["confidence_score"] > results["confidence_score"]:
                results = infinite_scroll_data
                results["patterns"] = [infinite_scroll_data]
        
        # Register the pagination pattern in the global registry if confident enough
        if results["confidence_score"] >= self.confidence_threshold:
            get_registry().register_pattern(
                pattern_type="pagination",
                url=url,
                pattern_data=results,
                confidence=results["confidence_score"]
            )
            logger.info(f"Registered pagination pattern for {domain}")
        
        return results
    
    def _find_potential_pagination_containers(self, soup: BeautifulSoup) -> List[Tag]:
        """
        Find potential pagination containers using heuristics.
        
        Args:
            soup: BeautifulSoup object with parsed HTML
            
        Returns:
            List of potential pagination container elements
        """
        potential_containers = []
        
        # Look for elements with pagination-related text
        for pattern in self.pagination_text_patterns:
            elements = soup.find_all(string=pattern)
            for element in elements:
                parent = element.parent
                if parent:
                    # Walk up a few levels to find a suitable container
                    container = parent
                    for _ in range(3):  # Try up to 3 levels up
                        if container:
                            potential_containers.append(container)
                            container = container.parent
        
        # Look for elements with pagination-related attributes
        for attr in ['class', 'id']:
            for pattern in self.pagination_attr_patterns:
                elements = soup.find_all(lambda tag: tag.get(attr) and 
                                      pattern.search(str(tag.get(attr))))
                potential_containers.extend(elements)
        
        # Look for groupings of numeric links that might be page numbers
        numeric_links = []
        for a in soup.find_all('a', href=True):
            try:
                link_text = a.get_text().strip()
                if link_text.isdigit() and int(link_text) > 0:
                    numeric_links.append(a)
            except:
                continue
                
        # If we found at least 3 numeric links, they might be page numbers
        if len(numeric_links) >= 3:
            # Group by parent to find most likely pagination container
            parent_groups = {}
            for link in numeric_links:
                parent = link.parent
                if parent not in parent_groups:
                    parent_groups[parent] = []
                parent_groups[parent].append(link)
            
            # Add parents with multiple numeric children as potential containers
            for parent, links in parent_groups.items():
                if len(links) >= 3:
                    potential_containers.append(parent)
                    
                    # Also add grandparent as it might be the main container
                    grandparent = parent.parent
                    if grandparent:
                        potential_containers.append(grandparent)
        
        return list(set(potential_containers))
    
    def _analyze_pagination_container(self, container: Tag, url: str, 
                                    url_query_params: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze a potential pagination container.
        
        Args:
            container: BeautifulSoup container element
            url: Current page URL
            url_query_params: Parsed query parameters from the URL
            
        Returns:
            Dictionary with pagination analysis data
        """
        # Default result with low confidence
        result = {
            "pagination_type": "none",
            "current_page": None,
            "total_pages": None,
            "next_page_url": None,
            "prev_page_url": None,
            "page_urls": [],
            "confidence_score": 0.0,
            "container_selector": self._generate_container_selector(container)
        }
        
        evidence_points = []
        
        # Extract links from the container
        links = container.find_all('a', href=True)
        if not links:
            return result
        
        # Check for 'next' and 'prev' links
        next_link = None
        prev_link = None
        page_links = []
        
        for link in links:
            href = link.get('href', '')
            link_text = link.get_text().strip()
            link_classes = link.get('class', [])
            
            # Convert classes to string for easier searching
            link_class_str = ' '.join(link_classes) if isinstance(link_classes, list) else link_classes
            
            # Check for next link indicators
            if (re.search(r'next\s*page|next\s*»|»|›|⟩|▶', link_text, re.I) or 
                re.search(r'next|next-page', link_class_str, re.I) or
                'next' in link.get('rel', []) or
                link.find('img', alt=re.compile(r'next', re.I))):
                next_link = link
                
            # Check for prev link indicators
            elif (re.search(r'previous\s*page|«\s*previous|«|‹|⟨|◀', link_text, re.I) or 
                 re.search(r'prev|previous|prev-page', link_class_str, re.I) or
                 'prev' in link.get('rel', []) or
                 link.find('img', alt=re.compile(r'prev|previous', re.I))):
                prev_link = link
                
            # Check for page number links
            elif link_text.isdigit():
                page_links.append((int(link_text), href))
                
            # Check for Load More button
            elif re.search(r'load\s*more|show\s*more|more\s*results|see\s*more', link_text, re.I):
                result["pagination_type"] = "load_more"
                result["next_page_url"] = urljoin(url, href)
                evidence_points.append(0.8)  # High confidence for explicit load more
        
        # If we found next/prev links, this is likely a next/prev pagination pattern
        if next_link or prev_link:
            result["pagination_type"] = "next_prev"
            evidence_points.append(0.9)  # High confidence for explicit next/prev
            
            if next_link:
                result["next_page_url"] = urljoin(url, next_link.get('href', ''))
                
            if prev_link:
                result["prev_page_url"] = urljoin(url, prev_link.get('href', ''))
        
        # If we found page number links, this is a numeric pagination pattern
        if page_links:
            # If we already identified as next/prev, upgrade to "full" pagination
            if result["pagination_type"] == "next_prev":
                result["pagination_type"] = "full_pagination"
                evidence_points.append(0.95)  # Very high confidence for full pagination
            else:
                result["pagination_type"] = "numeric"
                evidence_points.append(0.85)  # High confidence for numeric pagination
            
            # Sort page links by page number
            page_links.sort(key=lambda x: x[0])
            
            # Extract page URLs
            result["page_urls"] = [urljoin(url, href) for _, href in page_links]
            
            # Try to determine current page and total pages
            current_page = self._find_current_page(container, links, url_query_params)
            if current_page:
                result["current_page"] = current_page
                
            # Total pages can be inferred from highest page number link
            if page_links:
                result["total_pages"] = page_links[-1][0]
        
        # Look for additional evidence in container attributes
        container_classes = container.get('class', [])
        container_class_str = ' '.join(container_classes) if isinstance(container_classes, list) else container_classes
        container_id = container.get('id', '')
        
        if re.search(r'pag(e|ing|ination)', container_class_str + ' ' + container_id, re.I):
            evidence_points.append(0.8)
            
        if container.name == 'nav':
            evidence_points.append(0.7)
            
        if container.get('aria-label') and re.search(r'pag(e|ing|ination)', container.get('aria-label'), re.I):
            evidence_points.append(0.8)
            
        # Check for ARIA roles
        if container.get('role') == 'navigation':
            evidence_points.append(0.6)
        
        # Calculate overall confidence score
        result["confidence_score"] = self.calculate_confidence(evidence_points)
        
        return result
    
    def _find_current_page(self, container: Tag, links: List[Tag], 
                         url_query_params: Dict[str, List[str]]) -> Optional[int]:
        """
        Determine the current page number from pagination elements.
        
        Args:
            container: BeautifulSoup pagination container
            links: List of links in the container
            url_query_params: Parsed query parameters from the URL
            
        Returns:
            Current page number or None if not found
        """
        # First look for elements with active/current classes or aria-current attribute
        active_elements = container.select(
            '.active, .current, [aria-current="page"], [class*="current"], [class*="active"]'
        )
        
        for element in active_elements:
            # If it's a link or contains text that's a number, use that
            try:
                text = element.get_text().strip()
                if text.isdigit():
                    return int(text)
            except:
                pass
                
        # Next, look for unlinked text that's just a number between links
        for element in container.contents:
            if isinstance(element, Tag) and element.name != 'a':
                try:
                    text = element.get_text().strip()
                    if text.isdigit():
                        return int(text)
                except:
                    pass
        
        # If we couldn't find it visually, try to infer from URL
        return self._infer_current_page_from_url(url_query_params)
    
    def _infer_current_page_from_url(self, url_query_params: Dict[str, List[str]]) -> Optional[int]:
        """
        Infer the current page number from URL query parameters.
        
        Args:
            url_query_params: Parsed query parameters from the URL
            
        Returns:
            Current page number or None if not found
        """
        # Common page parameter names
        page_param_names = ['page', 'p', 'pg', 'paged', 'pagenum', 'page_num', 'page-num']
        
        for param in page_param_names:
            if param in url_query_params:
                try:
                    return int(url_query_params[param][0])
                except:
                    pass
        
        # If no page parameter is present, assume it's page 1
        return 1
    
    def _url_has_page_param(self, url_query_params: Dict[str, List[str]]) -> bool:
        """
        Check if URL has a page parameter.
        
        Args:
            url_query_params: Parsed query parameters from the URL
            
        Returns:
            True if URL has a page parameter
        """
        page_param_names = ['page', 'p', 'pg', 'paged', 'pagenum', 'page_num', 'page-num']
        return any(param in url_query_params for param in page_param_names)
    
    def _analyze_url_pagination(self, url: str, url_query_params: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze URL structure to infer pagination pattern.
        
        Args:
            url: Current page URL
            url_query_params: Parsed query parameters from the URL
            
        Returns:
            Dictionary with pagination analysis data based on URL
        """
        # Default result with moderate confidence
        result = {
            "pagination_type": "url_parameter",
            "current_page": None,
            "total_pages": None,
            "next_page_url": None,
            "prev_page_url": None,
            "page_urls": [],
            "confidence_score": 0.7,  # Moderate confidence for URL parameter detection
            "container_selector": None
        }
        
        # Try to find the page parameter
        parsed_url = urlparse(url)
        page_param = None
        current_page = 1
        
        page_param_names = ['page', 'p', 'pg', 'paged', 'pagenum', 'page_num', 'page-num']
        
        for param in page_param_names:
            if param in url_query_params:
                page_param = param
                try:
                    current_page = int(url_query_params[param][0])
                    break
                except:
                    continue
        
        if page_param and current_page:
            result["current_page"] = current_page
            
            # Generate next page URL
            next_page_query = url_query_params.copy()
            next_page_query[page_param] = [str(current_page + 1)]
            next_page_url = parsed_url._replace(query=urlencode(next_page_query, doseq=True))
            result["next_page_url"] = urlunparse(next_page_url)
            
            # Generate previous page URL if not on page 1
            if current_page > 1:
                prev_page_query = url_query_params.copy()
                prev_page_query[page_param] = [str(current_page - 1)]
                prev_page_url = parsed_url._replace(query=urlencode(prev_page_query, doseq=True))
                result["prev_page_url"] = urlunparse(prev_page_url)
            
            # Generate some example page URLs
            for page_num in range(1, min(6, current_page + 3)):
                page_query = url_query_params.copy()
                page_query[page_param] = [str(page_num)]
                page_url = parsed_url._replace(query=urlencode(page_query, doseq=True))
                result["page_urls"].append(urlunparse(page_url))
        
        return result
    
    def _detect_infinite_scroll(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Detect infinite scroll pagination patterns.
        
        Args:
            soup: BeautifulSoup object with parsed HTML
            url: Current page URL
            
        Returns:
            Dictionary with infinite scroll pagination data
        """
        # Default result with low confidence
        result = {
            "pagination_type": "none",
            "current_page": 1,  # Infinite scroll usually starts at page 1
            "total_pages": None,  # Usually unknown for infinite scroll
            "next_page_url": None,
            "prev_page_url": None,
            "page_urls": [],
            "confidence_score": 0.0,
            "container_selector": None
        }
        
        evidence_points = []
        
        # Look for infinite scroll indicators in scripts
        scripts = soup.find_all('script')
        for script in scripts:
            script_text = script.string if script.string else ''
            
            # Look for common infinite scroll libraries
            if script_text and any(term in script_text.lower() for term in 
                                ['infinitescroll', 'infinite-scroll', 'endless', 
                                 'load more', 'loadmore', 'waypoint']):
                evidence_points.append(0.7)
                result["pagination_type"] = "infinite_scroll"
                
            # Look for AJAX pagination related code
            if script_text and re.search(r'(load|fetch)(More|Next|Additional)', script_text):
                evidence_points.append(0.6)
                result["pagination_type"] = "infinite_scroll"
        
        # Look for common infinite scroll containers
        infinite_containers = soup.select('.infinite-scroll, .endless, [data-infinite-scroll]')
        if infinite_containers:
            evidence_points.append(0.7)
            result["pagination_type"] = "infinite_scroll"
            result["container_selector"] = self._generate_container_selector(infinite_containers[0])
        
        # Look for load more button (these often trigger infinite loading)
        load_more_buttons = soup.select(
            'button[class*="load-more"], a[class*="load-more"], .load-more, '
            '[class*="loadMore"], [id*="load-more"], [id*="loadMore"]'
        )
        
        if load_more_buttons:
            evidence_points.append(0.8)
            result["pagination_type"] = "load_more"
            result["container_selector"] = self._generate_container_selector(load_more_buttons[0])
            
            # If it's a link, extract the URL
            for button in load_more_buttons:
                if button.name == 'a' and button.get('href'):
                    result["next_page_url"] = urljoin(url, button.get('href'))
                    break
        
        # Try to find data attributes that might contain pagination info
        elements_with_data = soup.select('[data-page], [data-next-page], [data-current-page]')
        for element in elements_with_data:
            # Look for current page indicator
            if element.get('data-page') and element.get('data-page').isdigit():
                result["current_page"] = int(element.get('data-page'))
                evidence_points.append(0.6)
                
            # Look for total pages
            if element.get('data-total-pages') and element.get('data-total-pages').isdigit():
                result["total_pages"] = int(element.get('data-total-pages'))
                evidence_points.append(0.6)
                
            # Look for next page URL
            if element.get('data-next-page-url'):
                result["next_page_url"] = urljoin(url, element.get('data-next-page-url'))
                evidence_points.append(0.7)
                
            if result["pagination_type"] == "none":
                result["pagination_type"] = "infinite_scroll"
        
        # Try to infer API endpoint for infinite scroll from script content
        if result["pagination_type"] in ["infinite_scroll", "load_more"] and not result["next_page_url"]:
            api_endpoint = self._extract_api_endpoint(soup)
            if api_endpoint:
                result["next_page_url"] = urljoin(url, api_endpoint)
                evidence_points.append(0.6)
        
        # Calculate overall confidence score
        result["confidence_score"] = self.calculate_confidence(evidence_points)
        
        return result
    
    def _extract_api_endpoint(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Try to extract API endpoint for infinite scroll from script content.
        
        Args:
            soup: BeautifulSoup object with parsed HTML
            
        Returns:
            API endpoint URL or None if not found
        """
        # Look for AJAX URLs or API endpoints in script tags
        for script in soup.find_all('script'):
            if not script.string:
                continue
                
            script_text = script.string.lower()
            
            # Look for URL patterns that might be API endpoints
            # for pagination/infinite scroll
            endpoint_patterns = [
                r'(?:url|endpoint|api):\s*[\'"]([^\'"]*/(?:page|scroll|load-?more|api|items|products|posts|results)[^\'"]*)[\'"]',
                r'\.ajax\(\s*{\s*url:\s*[\'"]([^\'"]*)[\'"]',
                r'fetch\([\'"]([^\'"]*(?:page|items|products|posts|results)[^\'"]*)[\'"]'
            ]
            
            for pattern in endpoint_patterns:
                matches = re.findall(pattern, script_text)
                if matches:
                    return matches[0]
        
        return None
    
    def _generate_container_selector(self, container: Tag) -> str:
        """
        Generate a CSS selector for a container.
        
        Args:
            container: BeautifulSoup container element
            
        Returns:
            CSS selector string
        """
        # If the container has an ID, that's the most reliable selector
        if container.get('id'):
            return f"#{container['id']}"
            
        # If the container has classes, use those
        if container.get('class'):
            class_selector = '.'.join(container['class'])
            return f"{container.name}.{class_selector}"
            
        # If the container has a role attribute
        if container.get('role'):
            return f"{container.name}[role='{container['role']}']"
            
        # Fallback: try to generate a selector based on container position
        parent = container.parent
        if parent and parent.name != 'body':
            if parent.get('id'):
                return f"#{parent['id']} > {container.name}"
            elif parent.get('class'):
                class_selector = '.'.join(parent['class'])
                return f".{class_selector} > {container.name}"
        
        # Last resort: use nth-of-type
        siblings = list(container.find_previous_siblings(container.name)) + [container]
        return f"{container.name}:nth-of-type({len(siblings)})"