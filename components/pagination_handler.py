"""
Pagination Handler Module

Handles detection and navigation of pagination on websites.
"""

import re
import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Union, Tuple
from urllib.parse import urljoin, urlparse, parse_qs, urlunparse, urlencode
from bs4 import BeautifulSoup

# Import our enhanced utilities
from utils.html_utils import parse_html, extract_text_fast, find_by_xpath, select_with_css
from utils.retry_utils import with_exponential_backoff
from utils.http_utils import fetch_html

# Import playwright for complex pagination handling
from playwright.async_api import async_playwright, Page

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levellevelname)s - %(message)s')
logger = logging.getLogger("PaginationHandler")

class PaginationHandler:
    """
    Handles the detection and navigation of pagination on websites.
    
    This class provides methods to detect the type of pagination used on a website
    and generate URLs for navigating between pages.
    """
    
    def __init__(self, max_depth: int = 10):
        """
        Initialize the PaginationHandler.
        
        Args:
            max_depth: Maximum depth for pagination traversal, to prevent endless crawling
        """
        # Common patterns for pagination elements
        self.common_patterns = {
            "next_button_text": [
                "next", "next page", "next »", "»", "›", ">", "→", 
                "next >>", "forward", "more", "load more"
            ],
            "next_button_classes": [
                "next", "pagination-next", "next-page", "arrow-next",
                "next-btn", "right", "forward", "nextlink", "load-more"
            ],
            "pagination_container_classes": [
                "pagination", "pager", "pages", "page-numbers", 
                "paginate", "pagenav", "page-nav", "page-link"
            ]
        }
        
        # Add depth tracking for deep crawling
        self.max_depth = max_depth
        self.visited_pages = set()
        self.pagination_patterns = {}
        
        # Interactive pagination mode flag
        self.interactive_mode = False
        self.playwright_browser = None
        
    @with_exponential_backoff(max_attempts=3)
    async def detect_pagination_type(self, html_content: str, current_url: str) -> Dict[str, Any]:
        """
        Detect the type of pagination used on a page.
        
        Args:
            html_content: HTML content of the page
            current_url: Current URL of the page
            
        Returns:
            Dictionary with pagination information
        """
        # Create a BeautifulSoup object with optimized lxml parser
        soup = parse_html(html_content)
        
        # Initialize result with default values
        result = {
            "has_pagination": False,
            "pagination_type": None,
            "current_page": None,
            "next_page_url": None,
            "last_page": None,
            "total_pages": None,
            "pagination_base_url": None,
            "pagination_param": None
        }
        
        # Try different methods to detect pagination
        
        # Method 1: Look for a "next" button
        next_url = self._find_next_button(soup, current_url)
        if next_url:
            result["has_pagination"] = True
            result["pagination_type"] = "next_button"
            result["next_page_url"] = next_url
            
            # Try to extract pagination base information
            pagination_info = self._extract_pagination_params(current_url, next_url)
            if pagination_info:
                result["pagination_base_url"] = pagination_info["base_url"]
                result["pagination_param"] = pagination_info["param"]
                result["current_page"] = pagination_info["current_page"]
                
            return result
            
        # Method 2: Look for URL pattern with page numbers
        pagination_info = self._detect_url_pattern_pagination(current_url)
        if pagination_info["has_pagination"]:
            result.update(pagination_info)
            
            # Generate next page URL if it's not the last page
            if result["current_page"] is not None and result["current_page"] < result.get("total_pages", float('inf')):
                next_page = result["current_page"] + 1
                result["next_page_url"] = self._generate_pagination_url(
                    result["pagination_base_url"], 
                    result["pagination_param"], 
                    next_page
                )
                
            return result
            
        # Method 3: Look for classic numbered pagination
        pagination_info = self._detect_numbered_pagination(soup, current_url)
        if pagination_info["has_pagination"]:
            result.update(pagination_info)
            return result
            
        # Method 4: Look for "Load More" or infinite scroll buttons
        load_more_info = self._detect_load_more(soup, current_url)
        if load_more_info["has_pagination"]:
            result.update(load_more_info)
            return result
            
        # Method 5: Look for AJAX-based pagination
        ajax_info = self._detect_ajax_pagination(soup, current_url)
        if ajax_info["has_pagination"]:
            result.update(ajax_info)
            return result
            
        # Method 6: Look for interactive pagination that requires JavaScript
        js_info = self._detect_js_pagination(soup, current_url)
        if js_info["has_pagination"]:
            result.update(js_info)
            return result
            
        return result
        
    def _find_next_button(self, soup, current_url: str) -> Optional[str]:
        """
        Find a "next page" button in the HTML using optimized selectors.
        
        Args:
            soup: BeautifulSoup object with lxml parser
            current_url: Current URL for resolving relative links
            
        Returns:
            URL of the next page, or None if not found
        """
        # Optimized CSS selector for common next buttons
        next_button_selectors = [
            'a.next', 'a.pagination-next', 'a.next-page', 'a.arrow-next',
            'a[rel="next"]', 'a[aria-label*="next"]', 'a[title*="next"]',
            'a:has(span.next)', 'a:has(svg.next-icon)', 'a:has(i.fa-arrow-right)',
            'a:contains("next")', 'a:contains("Next")', 'a:contains("›")', 'a:contains("»")'
        ]
        
        # Try dedicated XPath for next page links first (faster)
        next_buttons = find_by_xpath(soup, '//a[contains(@class, "next") or contains(text(), "Next") or contains(text(), "next") or @rel="next"]')
        
        for button in next_buttons:
            href = button.get('href')
            if href:
                return urljoin(current_url, href)
        
        # Try CSS selectors if XPath didn't work
        for selector in next_button_selectors:
            elements = select_with_css(soup, selector)
            if elements:
                href = elements[0].get('href')
                if href:
                    return urljoin(current_url, href)
                    
        # Look for link/meta tags with "next" rel attribute
        next_rel = soup.find('link', {'rel': 'next'})
        if next_rel and next_rel.has_attr('href'):
            return urljoin(current_url, next_rel['href'])
            
        return None
    
    def _detect_url_pattern_pagination(self, url: str) -> Dict[str, Any]:
        """
        Detect pagination based on URL pattern.
        
        Args:
            url: URL to analyze
            
        Returns:
            Dictionary with pagination information
        """
        result = {
            "has_pagination": False,
            "pagination_type": None,
            "current_page": None,
            "pagination_base_url": None,
            "pagination_param": None
        }
        
        # Parse the URL
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        
        # Common page parameter names
        page_param_names = ['page', 'p', 'pg', 'pagina', 'pagenum', 'pageNumber', 'pp']
        
        # Check if any of the common page parameters are present
        for param in page_param_names:
            if param in query_params:
                try:
                    page_num = int(query_params[param][0])
                    
                    # Create a base URL by removing the page parameter
                    new_params = {k: v for k, v in query_params.items() if k != param}
                    base_parsed = parsed._replace(query=urlencode(new_params, doseq=True))
                    
                    result["has_pagination"] = True
                    result["pagination_type"] = "url_pattern"
                    result["current_page"] = page_num
                    result["pagination_base_url"] = urlunparse(base_parsed)
                    result["pagination_param"] = param
                    
                    return result
                except ValueError:
                    # Not a numeric page parameter
                    continue
                    
        # Check for page number in URL path segments
        path_parts = parsed.path.split('/')
        
        # Common page path patterns
        page_path_patterns = [
            ('page', r'page/(\d+)/?$'),  # WordPress style: /page/2/
            ('p', r'p/(\d+)/?$'),        # Short form: /p/2/
            ('', r'/(\d+)/?$')           # Just a number at the end: /2/
        ]
        
        for param, pattern in page_path_patterns:
            match = re.search(pattern, parsed.path)
            if match:
                try:
                    page_num = int(match.group(1))
                    
                    # Create a base URL by removing the page part from the path
                    path_prefix = re.sub(pattern, '', parsed.path)
                    if not path_prefix.endswith('/'):
                        path_prefix += '/'
                        
                    base_parsed = parsed._replace(path=path_prefix)
                    
                    result["has_pagination"] = True
                    result["pagination_type"] = "path_pattern"
                    result["current_page"] = page_num
                    result["pagination_base_url"] = urlunparse(base_parsed)
                    result["pagination_param"] = param if param else "page_number"
                    
                    return result
                except ValueError:
                    continue
                    
        return result
        
    def _detect_numbered_pagination(self, soup: BeautifulSoup, current_url: str) -> Dict[str, Any]:
        """
        Detect numbered pagination.
        
        Args:
            soup: BeautifulSoup object
            current_url: Current URL for resolving relative links
            
        Returns:
            Dictionary with pagination information
        """
        result = {
            "has_pagination": False,
            "pagination_type": None,
            "current_page": None,
            "next_page_url": None,
            "total_pages": None
        }
        
        # Look for common pagination container elements
        pagination_elements = []
        
        # Find by common class names
        for class_pattern in self.common_patterns["pagination_container_classes"]:
            elements = soup.find_all(class_=lambda c: c and class_pattern in c.lower())
            pagination_elements.extend(elements)
            
        # If we found potential pagination containers, analyze them
        if pagination_elements:
            # Extract page information from each container
            for element in pagination_elements:
                # Look for links with page numbers
                page_links = []
                
                for a in element.find_all('a'):
                    href = a.get('href')
                    text = a.get_text().strip()
                    
                    # Skip if no href or text
                    if not href or not text:
                        continue
                        
                    # Skip if it's a navigation link
                    lower_text = text.lower()
                    if any(nav in lower_text for nav in ['prev', 'next', 'first', 'last']):
                        if 'next' in lower_text:
                            result["next_page_url"] = urljoin(current_url, href)
                        continue
                        
                    # Try to parse the text as a number
                    try:
                        page_num = int(text)
                        page_links.append((page_num, href))
                    except ValueError:
                        continue
                        
                # If we found numbered page links
                if page_links:
                    result["has_pagination"] = True
                    result["pagination_type"] = "numbered"
                    
                    # Find the highest page number
                    max_page = max(page_links, key=lambda x: x[0])[0]
                    result["total_pages"] = max_page
                    
                    # Try to determine which page is current
                    # First look for links with active/current class
                    active_page = None
                    
                    for li in element.find_all('li'):
                        class_list = li.get('class', [])
                        classes = ' '.join(class_list).lower() if class_list else ""
                        
                        if 'active' in classes or 'current' in classes or 'selected' in classes:
                            # Found the active page
                            try:
                                active_page = int(li.get_text().strip())
                                break
                            except ValueError:
                                pass
                                
                    if active_page:
                        result["current_page"] = active_page
                    else:
                        # If we couldn't find active page, look at the URL
                        parsed = urlparse(current_url)
                        for page_num, link in page_links:
                            parsed_link = urlparse(urljoin(current_url, link))
                            if parsed.path == parsed_link.path and parsed.query == parsed_link.query:
                                result["current_page"] = page_num
                                break
                                
                    # If we still don't have a current page, set it to 1 by default
                    if result["current_page"] is None:
                        result["current_page"] = 1
                        
                    # If we don't have a next_page_url yet, try to generate one
                    if not result["next_page_url"] and result["current_page"] < result["total_pages"]:
                        next_page = result["current_page"] + 1
                        
                        # Look for the link to the next page in our page_links
                        for page_num, href in page_links:
                            if page_num == next_page:
                                result["next_page_url"] = urljoin(current_url, href)
                                break
                                
                    break  # Stop after finding the first valid pagination container
                    
        return result
        
    def _detect_load_more(self, soup: BeautifulSoup, current_url: str) -> Dict[str, Any]:
        """
        Detect "Load More" buttons or infinite scroll mechanism.
        
        Args:
            soup: BeautifulSoup object
            current_url: Current URL for resolving relative links
            
        Returns:
            Dictionary with pagination information
        """
        result = {
            "has_pagination": False,
            "pagination_type": None,
            "next_page_url": None
        }
        
        # Common "Load More" button text patterns
        load_more_patterns = [
            "load more", "show more", "view more", "more items", "more results",
            "see more", "more products", "more listings", "next page", "next"
        ]
        
        # Look for buttons/links with "Load More" text
        for element in soup.find_all(['a', 'button', 'div', 'span']):
            text = element.get_text().strip().lower()
            
            if any(pattern in text for pattern in load_more_patterns):
                # Found a potential "Load More" element
                
                # Check if it's an anchor with href
                if element.name == 'a' and element.has_attr('href'):
                    href = element['href']
                    result["has_pagination"] = True
                    result["pagination_type"] = "load_more"
                    result["next_page_url"] = urljoin(current_url, href)
                    return result
                    
                # Check for data attributes that might contain a URL
                for attr in ['data-url', 'data-href', 'data-link', 'data-next']:
                    if element.has_attr(attr):
                        result["has_pagination"] = True
                        result["pagination_type"] = "load_more"
                        result["next_page_url"] = urljoin(current_url, element[attr])
                        return result
                        
                # Check for a parent form with action
                parent_form = element.find_parent('form')
                if parent_form and parent_form.has_attr('action'):
                    result["has_pagination"] = True
                    result["pagination_type"] = "load_more"
                    result["next_page_url"] = urljoin(current_url, parent_form['action'])
                    return result
                    
        # Look for JavaScript pagination indicators
        scripts = soup.find_all('script')
        
        # Common patterns in JavaScript for pagination
        js_pagination_patterns = [
            r'var\s+totalPages\s*=\s*(\d+)',
            r'var\s+currentPage\s*=\s*(\d+)',
            r'totalPages\s*[:=]\s*(\d+)',
            r'currentPage\s*[:=]\s*(\d+)',
            r'next_page_url\s*[:=]\s*[\'"](.+?)[\'"]',
        ]
        
        current_page = None
        total_pages = None
        next_page_url = None
        
        for script in scripts:
            script_text = script.string
            if not script_text:
                continue
                
            # Look for current page and total pages in JavaScript
            for pattern in js_pagination_patterns:
                match = re.search(pattern, script_text)
                if match:
                    if 'currentPage' in pattern:
                        try:
                            current_page = int(match.group(1))
                        except ValueError:
                            pass
                    elif 'totalPages' in pattern:
                        try:
                            total_pages = int(match.group(1))
                        except ValueError:
                            pass
                    elif 'next_page_url' in pattern:
                        next_page_url = match.group(1)
                        
        # If we found pagination info in JavaScript
        if current_page is not None or total_pages is not None or next_page_url is not None:
            result["has_pagination"] = True
            result["pagination_type"] = "javascript"
            
            if current_page is not None:
                result["current_page"] = current_page
                
            if total_pages is not None:
                result["total_pages"] = total_pages
                
            if next_page_url:
                result["next_page_url"] = urljoin(current_url, next_page_url)
                
        return result
        
    def _detect_ajax_pagination(self, soup: BeautifulSoup, current_url: str) -> Dict[str, Any]:
        """
        Detect AJAX-based pagination mechanisms that might not be detected by other methods.
        
        Args:
            soup: BeautifulSoup object
            current_url: Current URL for resolving relative links
            
        Returns:
            Dictionary with pagination information
        """
        result = {
            "has_pagination": False,
            "pagination_type": None,
            "next_page_url": None,
            "current_page": None,
            "ajax_data": None
        }
        
        # Look for elements with data attributes related to pagination
        pagination_data_attributes = [
            'data-page', 'data-current-page', 'data-next-page', 
            'data-pagination', 'data-paged', 'data-page-num'
        ]
        
        for element in soup.find_all(lambda tag: any(tag.has_attr(attr) for attr in pagination_data_attributes)):
            # Found an element with pagination data attributes
            page_data = {}
            
            for attr in pagination_data_attributes:
                if element.has_attr(attr):
                    page_data[attr] = element[attr]
            
            if page_data:
                result["has_pagination"] = True
                result["pagination_type"] = "ajax"
                result["ajax_data"] = page_data
                
                # Try to determine current page
                for attr in ['data-current-page', 'data-page', 'data-page-num']:
                    if attr in page_data:
                        try:
                            result["current_page"] = int(page_data[attr])
                            break
                        except ValueError:
                            pass
                
                # Look for next page information
                if element.has_attr('data-next-page-url'):
                    result["next_page_url"] = urljoin(current_url, element['data-next-page-url'])
                
                return result
        
        # Look for JavaScript variables with pagination info
        scripts = soup.find_all('script')
        js_patterns = [
            r'var\s+ajaxurl\s*=\s*[\'"](.+?)[\'"]',
            r'ajax_url\s*[:=]\s*[\'"](.+?)[\'"]',
            r'pagination_url\s*[:=]\s*[\'"](.+?)[\'"]',
        ]
        
        for script in scripts:
            script_text = script.string
            if not script_text:
                continue
            
            for pattern in js_patterns:
                match = re.search(pattern, script_text)
                if match:
                    ajax_url = match.group(1)
                    result["has_pagination"] = True
                    result["pagination_type"] = "ajax_script"
                    result["next_page_url"] = urljoin(current_url, ajax_url)
                    
                    # Look for current page info in the same script
                    current_page_match = re.search(r'current_page\s*[:=]\s*(\d+)', script_text)
                    if current_page_match:
                        try:
                            result["current_page"] = int(current_page_match.group(1))
                        except ValueError:
                            pass
                    
                    return result
        
        return result
        
    def _detect_js_pagination(self, soup, current_url: str) -> Dict[str, Any]:
        """
        Detect JavaScript-dependent pagination that would need browser interaction.
        
        Args:
            soup: BeautifulSoup object
            current_url: Current URL for context
            
        Returns:
            Dictionary with pagination information oriented toward interactive navigation
        """
        result = {
            "has_pagination": False,
            "pagination_type": None,
            "requires_interaction": False,
            "interaction_selectors": []
        }
        
        # Look for JavaScript pagination indicators that would require a browser
        js_pagination_indicators = [
            # Event listeners on pagination elements
            'onclick="', 'addEventListener', 'click()', 'onclick=',
            
            # AJAX pagination functions
            'loadMoreResults', 'loadNextPage', 'fetchPage', 'getPaginatedData',
            
            # Infinite scroll indicators
            'infiniteScroll', 'infinite-scroll', 'data-infinite-scroll',
            
            # Dynamic content loading
            'Vue.', 'React.', 'ReactDOM.', 'angular.', 'ko.applyBindings'
        ]
        
        # Check for JavaScript event handlers on potential pagination elements
        pagination_elements = []
        
        # Find elements with pagination-related classes
        for class_name in self.common_patterns["pagination_container_classes"]:
            elements = select_with_css(soup, f'[class*="{class_name}"]')
            pagination_elements.extend(elements)
            
        # Find elements with pagination-related text
        for text in self.common_patterns["next_button_text"]:
            elements = find_by_xpath(soup, f'//a[contains(text(), "{text}")]')
            pagination_elements.extend(elements)
        
        # Check these elements for JavaScript event attributes
        interactive_selectors = []
        for element in pagination_elements:
            # Check for JavaScript event attributes
            js_attrs = ['onclick', 'onmousedown', 'onmouseup', 'onmouseover']
            has_js_attr = False
            
            for attr in js_attrs:
                if element.has_attr(attr):
                    has_js_attr = True
                    break
                    
            # Check for JavaScript framework data attributes
            if not has_js_attr:
                all_attrs = element.attrs
                for attr_name in all_attrs:
                    if attr_name.startswith('data-') or attr_name.startswith('ng-') or attr_name.startswith('v-'):
                        has_js_attr = True
                        break
                        
            if has_js_attr:
                # Create a selector for this element for later interactive use
                selector = self._create_element_selector(element)
                if selector:
                    interactive_selectors.append(selector)
        
        # Check for scripts with pagination functions
        scripts = soup.find_all('script')
        has_js_pagination = False
        
        for script in scripts:
            script_text = script.string
            if not script_text:
                continue
                
            for indicator in js_pagination_indicators:
                if indicator in script_text:
                    has_js_pagination = True
                    break
                    
            if has_js_pagination:
                break
                
        # If we found JS pagination indicators or interactive pagination elements
        if has_js_pagination or interactive_selectors:
            result["has_pagination"] = True
            result["pagination_type"] = "interactive_js"
            result["requires_interaction"] = True
            result["interaction_selectors"] = interactive_selectors
            
        return result
    
    def _create_element_selector(self, element) -> Optional[str]:
        """
        Create a CSS selector for an element that can be used with Playwright.
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            CSS selector string or None if can't create a reliable selector
        """
        # Try ID selector (most specific)
        if element.has_attr('id'):
            return f"#{element['id']}"
            
        # Try data-testid or similar test attributes
        for attr in ['data-testid', 'data-test-id', 'data-test', 'data-cy', 'data-automation-id']:
            if element.has_attr(attr):
                return f"[{attr}='{element[attr]}']"
                
        # Try class with tag
        if element.has_attr('class'):
            class_list = ' '.join(element['class'])
            if class_list:
                return f"{element.name}.{'.'.join(element['class'])}"
                
        # Try other attributes
        for attr in ['aria-label', 'title', 'name', 'placeholder']:
            if element.has_attr(attr):
                return f"{element.name}[{attr}='{element[attr]}']"
                
        # If it's a button or a with text, use text content
        if element.name in ['a', 'button'] and element.string:
            text = element.string.strip()
            if text:
                return f"{element.name}:contains('{text}')"
                
        # Fallback: use tag name (less reliable)
        return element.name
    
    async def setup_interactive_mode(self, crawler=None):
        """
        Set up interactive mode for JavaScript-dependent pagination using Playwright.
        
        Args:
            crawler: Optional crawler instance with playwright browser
            
        Returns:
            Success status
        """
        try:
            if crawler and hasattr(crawler, 'browser'):
                # Reuse the crawler's browser if available
                self.playwright_browser = crawler.browser
                self.interactive_mode = True
                logger.info("Using existing browser for interactive pagination")
                return True
                
            # Otherwise, we'll need to create a new browser
            try:
                from playwright.async_api import async_playwright
                
                # Launch a browser
                playwright = await async_playwright().start()
                self.playwright_browser = await playwright.chromium.launch(headless=True)
                self.interactive_mode = True
                logger.info("Started new browser for interactive pagination")
                return True
                
            except ImportError:
                logger.error("Playwright not available. Install with: pip install playwright")
                self.interactive_mode = False
                return False
                
        except Exception as e:
            logger.error(f"Error setting up interactive mode: {str(e)}")
            self.interactive_mode = False
            return False
    
    async def navigate_pagination_interactive(self, 
                                           url: str, 
                                           next_page_selector: str = None,
                                           max_pages: int = 5) -> List[Dict[str, Any]]:
        """
        Navigate pagination interactively using Playwright for JavaScript-dependent pages.
        
        Args:
            url: Starting URL
            next_page_selector: CSS selector for next page button (if known)
            max_pages: Maximum number of pages to navigate
            
        Returns:
            List of page results with HTML content and URLs
        """
        if not self.interactive_mode:
            success = await self.setup_interactive_mode()
            if not success:
                logger.error("Failed to set up interactive mode for pagination")
                return []
                
        try:
            results = []
            
            # Create a new page
            page = await self.playwright_browser.new_page()
            
            # Navigate to the starting URL with timeout and fallback
            success = False
            for wait_condition in ["domcontentloaded", "load", "networkidle"]:
                try:
                    await page.goto(url, wait_until=wait_condition, timeout=15000)
                    logger.info(f"Successfully navigated to {url} using {wait_condition}")
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"Navigation failed with {wait_condition}: {str(e)}")
                    
            if not success:
                raise Exception(f"Failed to navigate to {url} with all wait conditions")
            
            # Get the first page content
            html_content = await page.content()
            current_url = page.url
            
            # Add first page to results
            results.append({
                "url": current_url,
                "html": html_content
            })
            
            # Find next button selector if not provided
            if not next_page_selector:
                # Parse the HTML with lxml for faster detection
                soup = parse_html(html_content)
                
                # Try to find JS pagination indicators
                js_pagination = self._detect_js_pagination(soup, current_url)
                
                if js_pagination["has_pagination"] and js_pagination["interaction_selectors"]:
                    next_page_selector = js_pagination["interaction_selectors"][0]
                else:
                    # Fallback to common selectors
                    for selector in [
                        'a.next', 'button.next', '[aria-label="Next page"]', 
                        'a:has-text("Next")', 'button:has-text("Next")',
                        'a:has-text("next")', 'button:has-text("next")'
                    ]:
                        if await page.locator(selector).count() > 0:
                            next_page_selector = selector
                            break
            
            # Navigate through pagination
            page_count = 1
            
            while next_page_selector and page_count < max_pages:
                try:
                    # Wait for selector to be visible
                    next_button = page.locator(next_page_selector)
                    is_visible = await next_button.is_visible()
                    
                    if not is_visible:
                        logger.info(f"Next button not visible. Pagination complete after {page_count} pages.")
                        break
                        
                    # Check if the button is disabled
                    is_disabled = await next_button.get_attribute("disabled")
                    aria_disabled = await next_button.get_attribute("aria-disabled")
                    
                    if is_disabled or aria_disabled == "true":
                        logger.info(f"Next button is disabled. Pagination complete after {page_count} pages.")
                        break
                        
                    # Click the next button
                    await next_button.click()
                    
                    # Wait for navigation to complete
                    await page.wait_for_load_state("networkidle")
                    
                    # Get the new page content
                    html_content = await page.content()
                    current_url = page.url
                    
                    # Add to results
                    results.append({
                        "url": current_url,
                        "html": html_content
                    })
                    
                    page_count += 1
                    
                except Exception as e:
                    logger.error(f"Error during interactive pagination: {str(e)}")
                    break
            
            # Close the page
            await page.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in interactive pagination: {str(e)}")
            return []
    
    async def close_interactive_mode(self):
        """
        Close the browser used for interactive pagination.
        """
        if self.playwright_browser:
            try:
                await self.playwright_browser.close()
                self.playwright_browser = None
                self.interactive_mode = False
                logger.info("Closed browser for interactive pagination")
            except Exception as e:
                logger.error(f"Error closing browser: {str(e)}")
                
    async def deep_pagination_urls(self, 
                                 starting_url: str, 
                                 html_content: str, 
                                 max_pages: int = None,
                                 interactive: bool = False,
                                 crawler=None) -> List[Dict[str, Any]]:
        """
        Generate pagination URLs for deep crawling, with support for interactive JavaScript pagination.
        
        Args:
            starting_url: URL of the starting page
            html_content: HTML content of the starting page
            max_pages: Maximum number of pages to generate (None for using max_depth)
            interactive: Whether to use interactive mode for JavaScript pagination
            crawler: Optional crawler instance with playwright browser
            
        Returns:
            List of dictionaries with URLs and HTML content in traversal order
        """
        if max_pages is None:
            max_pages = self.max_depth
            
        # Reset tracking for this deep pagination session
        self.visited_pages = set()
        
        # Get initial pagination info
        pagination_info = await self.detect_pagination_type(html_content, starting_url)
        
        # If no pagination detected or interactive mode requested, check for JavaScript pagination
        if (not pagination_info["has_pagination"] or interactive) and \
           pagination_info.get("pagination_type") == "interactive_js":
            
            # Set up interactive mode if needed
            if not self.interactive_mode:
                await self.setup_interactive_mode(crawler)
                
            if self.interactive_mode:
                # Use interactive navigation for JavaScript-dependent pagination
                next_selector = None
                if pagination_info.get("interaction_selectors"):
                    next_selector = pagination_info["interaction_selectors"][0]
                    
                results = await self.navigate_pagination_interactive(
                    starting_url, 
                    next_page_selector=next_selector,
                    max_pages=max_pages
                )
                
                # First item is the starting page, so skip it
                if results and len(results) > 1:
                    return results[1:]
                return []
        
        if not pagination_info["has_pagination"]:
            return []
            
        # For non-interactive pagination, collect URLs and HTML
        results = []
        queue = [(starting_url, html_content, 0)]  # (url, html_content, depth)
        
        # Store the pagination pattern for this site
        domain = urlparse(starting_url).netloc
        self.pagination_patterns[domain] = pagination_info["pagination_type"]
        
        # Process the queue
        while queue and len(results) < max_pages:
            current_url, current_html, depth = queue.pop(0)
            
            if current_url in self.visited_pages:
                continue
                
            self.visited_pages.add(current_url)
            
            if current_url != starting_url:
                results.append({
                    "url": current_url,
                    "html": current_html
                })
                
            if depth >= self.max_depth:
                continue
                
            # Get pagination info for current page
            page_info = await self.detect_pagination_type(current_html, current_url)
            
            if page_info["has_pagination"] and page_info["next_page_url"]:
                next_url = page_info["next_page_url"]
                
                # Fetch the next page's HTML content
                try:
                    next_html = await fetch_html(next_url)
                    queue.append((next_url, next_html, depth + 1))
                except Exception as e:
                    logger.error(f"Error fetching next page {next_url}: {str(e)}")
                
        return results
        
    async def generate_all_pagination_urls(self, starting_url: str, html_content: str) -> List[str]:
        """
        Generate all pagination URLs for a given starting page.
        
        Args:
            starting_url: URL of the starting page
            html_content: HTML content of the starting page
            
        Returns:
            List of pagination URLs
        """
        pagination_info = await self.detect_pagination_type(html_content, starting_url)
        
        if not pagination_info["has_pagination"]:
            return []
            
        urls = []
        
        # Handle different pagination types
        if pagination_info["pagination_type"] in ["url_pattern", "path_pattern"] and \
           pagination_info["pagination_base_url"] and pagination_info["pagination_param"]:
            
            # If we know the total pages, generate all URLs
            if pagination_info.get("total_pages"):
                # Generate URLs for all pages
                for page in range(1, pagination_info["total_pages"] + 1):
                    urls.append(self._generate_pagination_url(
                        pagination_info["pagination_base_url"],
                        pagination_info["pagination_param"],
                        page
                    ))
            else:
                # We don't know the total pages, but we can generate a few
                # to allow for exploration
                current_page = pagination_info.get("current_page", 1)
                
                # Generate URLs for a few pages before and after
                for page in range(max(1, current_page - 3), current_page + 7):
                    urls.append(self._generate_pagination_url(
                        pagination_info["pagination_base_url"],
                        pagination_info["pagination_param"],
                        page
                    ))
                    
        # For deep crawling of complex paginations, use the deep pagination method
        elif pagination_info["pagination_type"] in ["next_button", "ajax", "ajax_script", "load_more"]:
            return await self.deep_pagination_urls(starting_url, html_content)
        
        return urls
        
    def get_pagination_pattern(self, domain: str) -> Optional[str]:
        """
        Get the detected pagination pattern for a domain.
        
        Args:
            domain: The domain to get pattern for
            
        Returns:
            Pagination pattern type or None if not detected
        """
        return self.pagination_patterns.get(domain)
        
    def reset_tracking(self):
        """
        Reset the pagination handler's tracking data.
        """
        self.visited_pages = set()
    
    async def handle_js_pagination(self, url: str, page_selectors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Handle JavaScript-dependent pagination using Playwright.
        
        Args:
            url: URL to navigate to
            page_selectors: Optional list of CSS selectors for pagination elements
            
        Returns:
            Dictionary with pagination results including content
        """
        logger.info(f"Handling JavaScript pagination for URL: {url}")
        
        # Initialize Playwright if not already running
        if not self.playwright_browser:
            try:
                playwright = await async_playwright().start()
                browser = await playwright.chromium.launch(headless=True)
                self.playwright_browser = browser
                logger.info("Initialized Playwright browser for pagination")
            except Exception as e:
                logger.error(f"Failed to initialize Playwright: {e}")
                return {"success": False, "error": str(e)}
        
        result = {
            "success": False,
            "content": None,
            "url": url,
            "next_url": None,
            "pages_processed": 0
        }
        
        try:
            # Create a new context and page
            context = await self.playwright_browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            
            # Enable stealth mode if available
            try:
                # Try to import the playwright_stealth module
                from playwright_stealth import stealth_async
                page = await context.new_page()
                await stealth_async(page)
                logger.info("Stealth mode enabled for Playwright pagination")
            except ImportError:
                # Fall back to regular page if stealth isn't available
                page = await context.new_page()
                logger.info("Stealth mode not available, using regular Playwright page")
            
            # Navigate to the URL with progressive timeout strategy
            success = False
            for attempt, (wait_condition, timeout_ms) in enumerate([
                ("domcontentloaded", 10000),  # First attempt: fast DOM load
                ("load", 15000),              # Second attempt: full page load
                ("networkidle", 20000)        # Third attempt: network idle (reduced timeout)
            ], 1):
                try:
                    logger.info(f"Navigation attempt {attempt}: {wait_condition} with {timeout_ms}ms timeout")
                    await page.goto(url, wait_until=wait_condition, timeout=timeout_ms)
                    logger.info(f"Successfully navigated to {url} using {wait_condition}")
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"Navigation attempt {attempt} failed with {wait_condition}: {str(e)}")
                    if attempt == 3:
                        logger.error(f"All navigation attempts failed for {url}")
                        raise
            
            if not success:
                raise Exception("Failed to navigate to URL with all timeout strategies")
            
            # Get the page content
            content = await page.content()
            result["content"] = content
            result["success"] = True
            result["pages_processed"] = 1
            
            # Try to find pagination elements if selectors provided
            if page_selectors:
                for selector in page_selectors:
                    try:
                        # Check if the selector exists
                        if await page.locator(selector).count() > 0:
                            logger.info(f"Found pagination element with selector: {selector}")
                            # Click the element
                            await page.click(selector)
                            # Wait for page to load after click with timeout
                            try:
                                await page.wait_for_load_state("networkidle", timeout=10000)
                            except Exception as e:
                                logger.warning(f"Networkidle wait timed out after pagination click: {e}")
                                # Fallback to shorter wait
                                await page.wait_for_load_state("domcontentloaded", timeout=5000)
                            # Update the result with the new URL and content
                            result["next_url"] = page.url
                            result["content"] = await page.content()
                            break
                    except Exception as e:
                        logger.warning(f"Failed to interact with selector {selector}: {e}")
            
            # Close the context
            await context.close()
            
        except Exception as e:
            logger.error(f"Error handling JavaScript pagination: {e}")
            result["error"] = str(e)
            
        return result
    
    async def auto_paginate(self, start_url: str, max_pages: int = 5) -> List[Dict[str, Any]]:
        """
        Automatically paginate through a site, detecting and following pagination.
        
        Args:
            start_url: URL to start pagination from
            max_pages: Maximum number of pages to process
            
        Returns:
            List of pages processed with content
        """
        logger.info(f"Auto-paginating starting from URL: {start_url}")
        
        results = []
        current_url = start_url
        page_count = 0
        
        while current_url and page_count < max_pages:
            # Fetch the page
            html_content = await fetch_html(current_url)
            if not html_content:
                logger.warning(f"Failed to fetch HTML for URL: {current_url}")
                break
                
            # Store current page result
            page_result = {
                "url": current_url,
                "content": html_content,
                "page_number": page_count + 1
            }
            results.append(page_result)
            page_count += 1
            
            # Detect pagination type
            pagination_info = await self.detect_pagination_type(html_content, current_url)
            
            if not pagination_info["has_pagination"] or not pagination_info.get("next_page_url"):
                # Check if this might be JavaScript pagination
                js_indicators = self._check_js_pagination_indicators(html_content)
                
                if js_indicators:
                    logger.info(f"Detected JavaScript pagination on {current_url}")
                    # Try with Playwright
                    js_result = await self.handle_js_pagination(current_url, js_indicators.get("selectors"))
                    
                    if js_result["success"] and js_result["next_url"] and js_result["next_url"] != current_url:
                        # Found next page via JavaScript
                        current_url = js_result["next_url"]
                        # Update the last page's content with what we got from Playwright
                        results[-1]["content"] = js_result["content"]
                        continue
                    
                # No more pages detected
                logger.info(f"No more pages detected after page {page_count}")
                break
                
            # Move to next page
            current_url = pagination_info["next_page_url"]
            logger.info(f"Moving to next page: {current_url}")
            
            # Avoid loops
            if current_url in [r["url"] for r in results]:
                logger.warning(f"Pagination loop detected at URL: {current_url}")
                break
                
            # Wait a bit to be nice to the server
            await asyncio.sleep(1)
            
        return results
    
    def _check_js_pagination_indicators(self, html_content: str) -> Optional[Dict[str, Any]]:
        """
        Check for JavaScript pagination indicators in HTML content.
        
        Args:
            html_content: HTML content to check
            
        Returns:
            Dictionary with JS pagination indicators if found, None otherwise
        """
        soup = parse_html(html_content)
        
        # Common indicators for JS pagination
        js_indicators = {
            "load_more_buttons": [
                '[data-role="load-more"]', 
                '[class*="load-more"]',
                '[class*="loadMore"]',
                'button:contains("Load More")',
                'a:contains("Load More")',
                'button:contains("Show More")',
                'a:contains("Show More")'
            ],
            "infinite_scroll": [
                '[data-role="infinite-scroll"]',
                '[class*="infinite-scroll"]',
                '[class*="infiniteScroll"]'
            ],
            "next_buttons": [
                'a[class*="next"]',
                'button[class*="next"]',
                'a[aria-label*="Next"]',
                'button[aria-label*="Next"]'
            ]
        }
        
        # Check for the presence of indicators
        selectors = []
        
        # Load more buttons
        for selector in js_indicators["load_more_buttons"]:
            elements = select_with_css(soup, selector)
            if elements:
                selectors.append(selector)
                
        # Infinite scroll triggers
        for selector in js_indicators["infinite_scroll"]:
            elements = select_with_css(soup, selector)
            if elements:
                selectors.append(selector)
                
        # Next page buttons
        for selector in js_indicators["next_buttons"]:
            elements = select_with_css(soup, selector)
            if elements:
                selectors.append(selector)
                
        if selectors:
            return {
                "has_js_pagination": True,
                "selectors": selectors
            }
            
        return None
        
    def _create_element_selector(self, element) -> Optional[str]:
        """
        Create a CSS selector for an element for later interactive use.
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            CSS selector as string or None if not determinable
        """
        # Try id first
        if element.has_attr('id'):
            return f"#{element['id']}"
            
        # Try class
        if element.has_attr('class'):
            class_str = '.'.join(element['class'])
            if class_str:
                return f".{class_str}"
                
        # Try common attributes
        for attr in ['data-testid', 'data-id', 'data-role', 'name']:
            if element.has_attr(attr):
                return f"[{attr}='{element[attr]}']"
                
        # Try tag + text
        tag_name = element.name
        text = element.get_text().strip()
        if text and len(text) < 50:  # Only use text if it's reasonably short
            return f"{tag_name}:contains('{text}')"
            
        # Give up and return None
        return None
