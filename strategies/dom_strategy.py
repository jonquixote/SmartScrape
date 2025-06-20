"""
DOM-manipulation search engine implementation.

This file implements a search engine strategy that uses DOM manipulation
for sophisticated interactions with complex search interfaces.
"""

import logging
import asyncio
import re
import json
import time
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlparse, quote

from playwright.async_api import Page, async_playwright
from bs4 import BeautifulSoup

from strategies.base_strategy import BaseStrategy
from utils.html_utils import extract_text_fast
from utils.http_utils import clean_url
from components.search_automation import SearchAutomator, PlaywrightSearchHandler

try:
    from playwright_stealth import stealth_async
except ImportError:
    logging.warning("playwright_stealth not installed, DOM strategy may be less effective")
    async def stealth_async(page):
        logging.info("Using dummy stealth_async function")
        return

from strategies.core.strategy_types import StrategyType, StrategyCapability, strategy_metadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@strategy_metadata(
    strategy_type=StrategyType.EXTRACTION,
    capabilities={
        StrategyCapability.DYNAMIC_CONTENT,
        StrategyCapability.JAVASCRIPT_EXECUTION,
        StrategyCapability.LINK_EXTRACTION
    },
    description="Strategy that manipulates the DOM to extract information from web pages"
)
class DOMStrategy(BaseStrategy):
    """
    DOM-based search engine that uses advanced browser automation for
    sophisticated interaction with complex search interfaces.
    
    This strategy:
    - Uses Playwright for full browser automation
    - Implements stealth mode to avoid bot detection
    - Handles complex interactions with search interfaces
    - Supports modern JavaScript-heavy sites
    - Handles dynamic content loading
    """
    
    def __init__(self, config=None, context=None):
        """
        Initialize the DOM search strategy.
        
        Args:
            config: Configuration dictionary (optional)
            context: Strategy context (optional)
        """
        super().__init__(config or {})
        self.logger = logging.getLogger("DOMStrategy")
        self.playwright_handler = None
        self.search_automator = SearchAutomator()
        self.context = context
        
        # Set default capabilities
        self.capabilities = {
            "search_forms": True,
            "javascript_support": True,
            "ajax_support": True,
            "complex_interactions": True,
            "stealth_mode": True,
            "handle_popups": True
        }
    
    @property
    def name(self) -> str:
        """Return the name of the strategy."""
        return "dom_strategy"
    
    def can_handle(self, url: str, **kwargs) -> bool:
        """
        Check if this strategy can handle the given URL.
        DOM strategy can handle most URLs with complex interactions.
        
        Args:
            url: URL to check
            **kwargs: Additional keyword arguments
            
        Returns:
            Boolean indicating if strategy can handle the URL
        """
        try:
            # DOM strategy can handle most URLs, especially those requiring interaction
            # Exclude obviously problematic URLs
            if not url or not url.startswith(('http://', 'https://')):
                return False
                
            # Check for file downloads or non-web content
            if any(ext in url.lower() for ext in ['.pdf', '.doc', '.zip', '.exe', '.dmg']):
                return False
                
            return True
        except Exception:
            return False
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute the DOM strategy for the given URL.
        
        Args:
            url: The URL to process
            **kwargs: Additional keyword arguments including:
                - search_term: Term to search for
                - options: Additional options for execution
                - context: Strategy context
            
        Returns:
            Dictionary with results, or None if execution failed
        """
        import asyncio
        
        # Get optional parameters from kwargs  
        context = kwargs.get('context', self.context)
        
        # Use asyncio to run the async method
        async def _run():
            return await self._execute_async(context, url, **kwargs)
                
        # Check if we're already in an async context
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we reach here, we're in an async context - create a task instead
            import concurrent.futures
            import threading
            
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

    async def _execute_async(self, context, url, **kwargs) -> Dict[str, Any]:
        """
        Internal async implementation of the DOM strategy.
        
        Args:
            context: Strategy context
            url: URL to process
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with execution results
        """
        search_term = kwargs.get("search_term", "")
        options = kwargs.get("options", {})
        
        if search_term:
            return await self.search(url, search_term, options)
        else:
            # If no search term, just extract data from the URL
            return await self._extract_data_from_url(url, options)
            
    async def _extract_data_from_url(self, url: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract data from a URL without searching.
        
        Args:
            url: URL to extract data from
            options: Additional options
            
        Returns:
            Dictionary with extraction results
        """
        options = options or {}
        self.logger.info(f"Extracting data from URL: {url}")
        
        # Implementation similar to search but without the search step
        async with async_playwright() as p:
            # Initialize browser
            browser_type = options.get("browser_type", "chromium")
            browser_instance = None
            
            if browser_type == "firefox":
                browser_instance = await p.firefox.launch(
                    headless=options.get("headless", True)
                )
            elif browser_type == "webkit":
                browser_instance = await p.webkit.launch(
                    headless=options.get("headless", True)
                )
            else:
                browser_instance = await p.chromium.launch(
                    headless=options.get("headless", True),
                    args=[
                        '--disable-http2',  # Disable HTTP/2 to avoid protocol errors
                        '--disable-web-security',
                        '--disable-features=VizDisplayCompositor',
                        '--disable-dev-shm-usage',
                        '--no-sandbox'
                    ]
                )
            
            # Create context and page
            context = await browser_instance.new_context(
                viewport={"width": options.get("viewport_width", 1280), 
                          "height": options.get("viewport_height", 800)},
                user_agent=options.get("user_agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36")
            )
            
            try:
                page = await context.new_page()
                
                # Apply stealth mode
                if options.get("stealth_mode", True):
                    try:
                        await stealth_async(page)
                    except Exception as e:
                        self.logger.warning(f"Failed to apply stealth mode: {e}")
                
                # Navigate to the URL
                await page.goto(url, wait_until="domcontentloaded")
                await self._wait_for_page_load(page)
                
                # Handle popups
                if options.get("handle_popups", True):
                    await self._handle_common_popups(page)
                
                # Extract content
                content = await page.content()
                soup = BeautifulSoup(content, "html.parser")
                
                # Extract main content
                main_content = extract_text_fast(soup)
                
                # Extract links
                links = []
                for a in soup.find_all("a", href=True):
                    href = a.get("href", "")
                    if href and not href.startswith(("#", "javascript:")):
                        abs_url = urljoin(url, href)
                        links.append({
                            "url": abs_url,
                            "text": a.get_text(strip=True) or "[No Text]",
                            "title": a.get("title", "")
                        })
                
                return {
                    "success": True,
                    "engine": "dom",
                    "url": page.url,
                    "title": await page.title(),
                    "content": main_content,
                    "links": links[:50]  # Limit to 50 links
                }
                
            except Exception as e:
                self.logger.error(f"Error in DOM extraction: {str(e)}")
                return {
                    "success": False,
                    "engine": "dom",
                    "error": str(e)
                }
            finally:
                # Cleanup
                await context.close()
                await browser_instance.close()

    async def search(self, url: str, search_term: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform a search using DOM manipulation.
        
        Args:
            url: URL to search on
            search_term: Search term to use
            options: Additional options (optional)
            
        Returns:
            Dictionary with search results
        """
        options = options or {}
        self.logger.info(f"Performing DOM-based search on {url} for term: {search_term}")
        
        async with async_playwright() as p:
            # Set up browser with appropriate options
            browser_type = options.get("browser_type", "chromium")
            browser_instance = None
            
            if browser_type == "firefox":
                browser_instance = await p.firefox.launch(
                    headless=options.get("headless", True)
                )
            elif browser_type == "webkit":
                browser_instance = await p.webkit.launch(
                    headless=options.get("headless", True)
                )
            else:
                browser_instance = await p.chromium.launch(
                    headless=options.get("headless", True),
                    args=[
                        '--disable-http2',  # Disable HTTP/2 to avoid protocol errors
                        '--disable-web-security',
                        '--disable-features=VizDisplayCompositor',
                        '--disable-dev-shm-usage',
                        '--no-sandbox'
                    ]
                )
            
            # Configure context with appropriate options
            context = await browser_instance.new_context(
                viewport={"width": options.get("viewport_width", 1280), 
                          "height": options.get("viewport_height", 800)},
                user_agent=options.get("user_agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36")
            )
            
            try:
                # Create a new page
                page = await context.new_page()
                
                # Apply stealth mode to avoid detection
                if options.get("stealth_mode", True):
                    try:
                        await stealth_async(page)
                        self.logger.info("Applied stealth mode")
                    except Exception as e:
                        self.logger.warning(f"Failed to apply stealth mode: {e}")
                
                # Navigate to the page
                self.logger.info(f"Navigating to {url}")
                await page.goto(url, wait_until="domcontentloaded")
                
                # Wait for the page to be fully loaded
                await self._wait_for_page_load(page)
                
                # Handle cookies/popups if enabled
                if options.get("handle_popups", True):
                    await self._handle_common_popups(page)
                
                # Find and use search functionality
                search_result = await self._perform_search(page, search_term, options)
                
                if search_result.get("success", False):
                    # Extract results from the search results page
                    results = await self._extract_results(page, search_term, options)
                    
                    # Handle pagination if requested
                    if options.get("handle_pagination", True):
                        page_results = await self._handle_pagination(page, search_term, options)
                        if page_results:
                            results["pagination_results"] = page_results
                            
                            # Merge results from all pages if requested
                            if options.get("merge_results", True):
                                all_results = results.get("items", [])
                                for page_result in page_results:
                                    all_results.extend(page_result.get("items", []))
                                
                                # Deduplicate results
                                unique_results = []
                                seen_urls = set()
                                for item in all_results:
                                    item_url = item.get("url", "")
                                    if item_url and item_url not in seen_urls:
                                        seen_urls.add(item_url)
                                        unique_results.append(item)
                                
                                results["items"] = unique_results
                                results["total_items"] = len(unique_results)
                                results["paginated"] = True
                    
                    return {
                        "success": True,
                        "engine": "dom",
                        "results": results,
                        "url": page.url
                    }
                else:
                    self.logger.warning("Search was not successful")
                    return {
                        "success": False,
                        "engine": "dom",
                        "error": search_result.get("reason", "Search failed"),
                        "url": page.url
                    }
                    
            except Exception as e:
                self.logger.error(f"Error in DOM search: {str(e)}")
                return {
                    "success": False,
                    "engine": "dom",
                    "error": str(e)
                }
            finally:
                # Clean up resources
                await context.close()
                await browser_instance.close()
    
    def can_handle(self, url: str, **kwargs) -> bool:
        """
        Check if this strategy can handle the given URL.
        DOM strategy can handle most URLs with complex interactions.
        
        Args:
            url: URL to check
            **kwargs: Additional keyword arguments
            
        Returns:
            Boolean indicating if strategy can handle the URL
        """
        try:
            # DOM strategy can handle most URLs, especially those requiring interaction
            # Exclude obviously problematic URLs
            if not url or not url.startswith(('http://', 'https://')):
                return False
                
            # Check for file downloads or non-web content
            if any(ext in url.lower() for ext in ['.pdf', '.doc', '.zip', '.exe', '.dmg']):
                return False
                
            return True
        except Exception:
            return False
    
    async def _wait_for_page_load(self, page: Page) -> None:
        """Wait for the page to load completely"""
        try:
            # Wait for networkidle (no network requests for 500ms)
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception as e:
            self.logger.warning(f"Network idle timeout: {e}")
            
        try:
            # Additional wait for any animations or delayed content
            await page.wait_for_timeout(1000)  # 1 second extra wait
        except Exception as e:
            self.logger.warning(f"Timeout error: {e}")
    
    async def _handle_common_popups(self, page: Page) -> None:
        """Handle common popups like cookie notices and modals"""
        # Common selectors for cookie notices and popups
        dismiss_selectors = [
            "[id*='cookie'] button", 
            "[class*='cookie'] button",
            "[id*='consent'] button", 
            "[class*='consent'] button",
            "button:has-text('Accept')",
            "button:has-text('I Accept')",
            "button:has-text('Accept All')",
            "button:has-text('I Agree')",
            "button:has-text('Agree')",
            "button:has-text('Continue')",
            "button:has-text('Close')",
            "button:has-text('Got It')",
            "button:has-text('OK')",
            "button:has-text('Ok')",
            "button:has-text('Dismiss')",
            ".modal .close",
            "#modal .close",
            "[class*='popup'] .close",
            "[class*='modal'] .close",
            "[class*='dialog'] .close"
        ]
        
        for selector in dismiss_selectors:
            try:
                # Check if element exists before trying to click
                is_visible = await page.is_visible(selector, timeout=1000)
                if is_visible:
                    await page.click(selector)
                    self.logger.info(f"Dismissed popup using selector: {selector}")
                    await page.wait_for_timeout(500)  # Wait for popup to disappear
            except Exception as e:
                # Ignore errors for selectors that don't exist
                continue
    
    async def _perform_search(self, page: Page, search_term: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to find and use search functionality on the page"""
        self.logger.info(f"Trying to find search functionality for term: {search_term}")
        
        # Try multiple approaches to find and use search
        
        # 1. First try using the search automator
        self.logger.info("Attempting search with search automator")
        search_forms = await self.search_automator.detect_search_forms(await page.content(), page.url)
        
        if search_forms:
            best_form = search_forms[0]  # Use the first (highest scored) form
            self.logger.info(f"Found search form with score {best_form.get('search_relevance_score', 0)}")
            result = await self.search_automator.submit_search_form(best_form, search_term, page)
            
            if result.get("success", False):
                self.logger.info(f"Successfully used search form")
                # Wait for results to load
                await self._wait_for_page_load(page)
                return result
            
        # 2. If search automator failed, try direct Playwright interactions
        self.logger.info("Attempting direct search form interaction")
        search_input_selectors = [
            'input[type="search"]',
            'input[name="q"]',
            'input[name="query"]',
            'input[name="search"]',
            'input[name="s"]',
            'input[placeholder*="search" i]',
            'input[placeholder*="find" i]',
            'input[aria-label*="search" i]',
            'input[class*="search" i]',
            'input[id*="search" i]'
        ]
        
        for selector in search_input_selectors:
            try:
                is_visible = await page.is_visible(selector, timeout=1000)
                if is_visible:
                    self.logger.info(f"Found search input with selector: {selector}")
                    await page.fill(selector, search_term)
                    await page.press(selector, "Enter")
                    await self._wait_for_page_load(page)
                    return {"success": True, "method": "direct_input", "selector": selector}
            except Exception as e:
                self.logger.debug(f"Error with selector {selector}: {e}")
                continue
        
        # 3. Try common search buttons
        search_button_selectors = [
            'button[type="submit"]',
            'button[aria-label*="search" i]',
            'button[class*="search" i]',
            'button[id*="search" i]',
            '.search-button',
            '#search-button',
            'button:has-text("Search")',
            'button:has-text("Find")'
        ]
        
        for selector in search_button_selectors:
            try:
                is_visible = await page.is_visible(selector, timeout=1000)
                if is_visible:
                    self.logger.info(f"Found search button with selector: {selector}")
                    await page.click(selector)
                    await self._wait_for_page_load(page)
                    # Check if URL changed, which often indicates successful search
                    return {"success": True, "method": "search_button", "selector": selector}
            except Exception as e:
                self.logger.debug(f"Error with button selector {selector}: {e}")
                continue
        
        # 4. Try to manipulate URL directly
        if options.get("try_url_search", True):
            self.logger.info("Attempting URL-based search")
            current_url = page.url
            parsed_url = urlparse(current_url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Common URL patterns for search
            search_url_patterns = [
                f"{base_url}/search?q={quote(search_term)}",
                f"{base_url}/search?query={quote(search_term)}",
                f"{base_url}/search?keyword={quote(search_term)}",
                f"{base_url}/search?term={quote(search_term)}",
                f"{base_url}?s={quote(search_term)}",
                f"{base_url}?q={quote(search_term)}"
            ]
            
            for search_url in search_url_patterns:
                try:
                    self.logger.info(f"Trying search URL: {search_url}")
                    await page.goto(search_url, wait_until="domcontentloaded")
                    await self._wait_for_page_load(page)
                    
                    # Validate that this looks like a search results page
                    page_has_results = await self._validate_search_results_page(page, search_term)
                    if page_has_results:
                        return {"success": True, "method": "url_search", "url": search_url}
                except Exception as e:
                    self.logger.debug(f"Error with URL search {search_url}: {e}")
                    continue
        
        self.logger.warning("Failed to find usable search functionality")
        return {"success": False, "reason": "Could not find usable search functionality"}
    
    async def _validate_search_results_page(self, page: Page, search_term: str) -> bool:
        """Check if the current page appears to be a search results page"""
        # Get page content
        content = await page.content()
        content_lower = content.lower()
        search_term_lower = search_term.lower()
        
        # Check 1: Does the page contain the search term?
        if search_term_lower not in content_lower:
            return False
        
        # Check 2: Does the page have results-related text?
        result_indicators = ["result", "found", "search", "matching", "showing"]
        has_result_text = any(indicator in content_lower for indicator in result_indicators)
        
        # Check 3: Does the URL suggest search results?
        url = page.url.lower()
        search_url_indicators = ["search", "find", "results", "q=", "query=", "s="]
        has_search_url = any(indicator in url for indicator in search_url_indicators)
        
        # Check 4: Does the page have multiple similar elements (likely results)?
        has_results = await page.evaluate("""() => {
            // Look for repeated similar elements
            const resultSelectors = [
                '.search-result', '.result', '.product', '.item', '.listing',
                '[class*="result-item"]', '[class*="search-item"]', '[class*="product-item"]', 
                'article', '.card'
            ];
            
            for (const selector of resultSelectors) {
                const elements = document.querySelectorAll(selector);
                if (elements.length >= 3) {
                    return true;
                }
            }
            
            // Look for lists with multiple items
            const lists = document.querySelectorAll('ul, ol');
            for (const list of lists) {
                const items = list.querySelectorAll('li');
                if (items.length >= 3) {
                    return true;
                }
            }
            
            return false;
        }""")
        
        # Page is likely a results page if it meets at least 2 of our criteria
        criteria_met = [search_term_lower in content_lower, has_result_text, has_search_url, has_results]
        return sum(criteria_met) >= 2
    
    async def _extract_results(self, page: Page, search_term: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Extract search results from the current page"""
        self.logger.info("Extracting search results from page")
        
        # Get domain to resolve relative URLs
        domain = urlparse(page.url).netloc
        base_url = f"https://{domain}"
        
        # Extract results using JavaScript
        js_results = await page.evaluate("""() => {
            // Try to identify result containers
            const resultContainerSelectors = [
                '.search-results', '.results', '.products', '.listings',
                '[class*="search-results"]', '[class*="product-list"]', '[class*="search-list"]',
                '[data-testid*="search-results"]', '[aria-label*="search results"]'
            ];
            
            let resultContainer = null;
            for (const selector of resultContainerSelectors) {
                const container = document.querySelector(selector);
                if (container) {
                    resultContainer = container;
                    break;
                }
            }
            
            // If no specific container found, use the entire body
            if (!resultContainer) {
                resultContainer = document.body;
            }
            
            // Item selectors - try to find repeated items
            const itemSelectors = [
                '.search-result', '.result', '.product', '.item', '.listing',
                '[class*="result-item"]', '[class*="search-item"]', '[class*="product-item"]',
                'article', '.card', 'li'
            ];
            
            let resultItems = [];
            for (const selector of itemSelectors) {
                const items = resultContainer.querySelectorAll(selector);
                if (items.length >= 3) {  // We need at least 3 items to consider this a valid result list
                    resultItems = Array.from(items);
                    break;
                }
            }
            
            // If no clear item pattern, try to find similar siblings
            if (resultItems.length === 0) {
                // Look for parent elements that have multiple similar direct children
                const parents = document.querySelectorAll('div, ul, ol, section');
                for (const parent of parents) {
                    const children = parent.children;
                    if (children.length >= 3) {
                        // Check if first few children have similar structure
                        const firstTagName = children[0].tagName;
                        const secondTagName = children[1].tagName;
                        const thirdTagName = children[2].tagName;
                        
                        if (firstTagName === secondTagName && firstTagName === thirdTagName) {
                            resultItems = Array.from(children);
                            break;
                        }
                    }
                }
            }
            
            // Extract data from each result item
            return resultItems.slice(0, 20).map(item => {
                // Find title: look for heading or first link with text
                let title = "";
                const heading = item.querySelector('h1, h2, h3, h4, h5, h6');
                if (heading) {
                    title = heading.textContent.trim();
                } else {
                    const firstLink = item.querySelector('a');
                    if (firstLink) {
                        title = firstLink.textContent.trim();
                    }
                }
                
                // Find URL: look for first link
                let url = "";
                const link = item.querySelector('a');
                if (link && link.href) {
                    url = link.href;
                }
                
                // Find image: look for img tag
                let image = "";
                const img = item.querySelector('img');
                if (img && img.src) {
                    image = img.src;
                }
                
                // Find description: look for paragraphs or divs with description classes
                let description = "";
                const descElement = item.querySelector('p, [class*="desc"], [class*="summary"], [class*="text"]');
                if (descElement) {
                    description = descElement.textContent.trim();
                }
                
                // Find price: look for elements with price classes or $ sign
                let price = "";
                const priceSelectors = [
                    '[class*="price"]', 
                    '[data-testid*="price"]', 
                    '*:not(meta):contains("$")'
                ];
                
                for (const selector of priceSelectors) {
                    try {
                        const priceElement = item.querySelector(selector);
                        if (priceElement) {
                            // Try to extract price with $ symbol
                            const priceText = priceElement.textContent.trim();
                            const priceMatch = priceText.match(/\\$[\\d,]+(?:\\.\\d{2})?/);
                            if (priceMatch) {
                                price = priceMatch[0];
                                break;
                            } else if (priceText) {
                                price = priceText;
                                break;
                            }
                        }
                    } catch (e) {
                        // Ignore errors with invalid selectors
                        continue;
                    }
                }
                
                return {
                    title,
                    url,
                    image,
                    description,
                    price
                };
            }).filter(item => item.title || item.url);  // Only keep items with at least a title or URL
        }""")
        
        # Get the result count if available
        result_count_info = await page.evaluate("""() => {
            // Look for result count information
            const resultCounts = [];
            
            // Common patterns for result counts
            const countMatches = document.body.innerText.match(/([0-9,]+)\\s*(results|homes|properties|items|products|listings|found)/i);
            if (countMatches && countMatches[1]) {
                resultCounts.push(parseInt(countMatches[1].replace(/,/g, '')));
            }
            
            // Look for elements that might indicate result counts
            const countElements = document.querySelectorAll('[class*="result-count"], [class*="count"], [class*="total"]');
            for (const el of countElements) {
                const text = el.innerText;
                const matches = text.match(/([0-9,]+)/);
                if (matches && matches[1]) {
                    resultCounts.push(parseInt(matches[1].replace(/,/g, '')));
                }
            }
            
            return {
                resultCounts,
                pageTitle: document.title,
                hasResultWords: document.body.innerText.match(/result|found|match|showing|displayed/i) !== null
            };
        }""")
        
        # Clean up the URLs (make them absolute)
        for item in js_results:
            if "url" in item and item["url"]:
                item["url"] = urljoin(page.url, item["url"])
            if "image" in item and item["image"]:
                item["image"] = urljoin(page.url, item["image"])
        
        # Determine total result count
        result_count = None
        if result_count_info.get("resultCounts") and result_count_info["resultCounts"]:
            result_count = max(result_count_info["resultCounts"])
        
        # Return the structured results
        return {
            "items": js_results,
            "total_items": len(js_results),
            "estimated_total": result_count,
            "page_title": result_count_info.get("pageTitle", ""),
            "page_url": page.url
        }
    
    async def _handle_pagination(self, page: Page, search_term: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle pagination to extract results from multiple pages"""
        self.logger.info("Checking for pagination")
        
        # Maximum pages to process
        max_pages = options.get("max_pages", 3)
        
        # Detect pagination type
        pagination_type = await page.evaluate("""() => {
            // Check for numbered pagination
            const hasNumberedPagination = 
                document.querySelector('.pagination, .pager, nav[aria-label*="pagination" i]') !== null ||
                document.querySelectorAll('a[href*="page="], a[href*="/page/"]').length > 0;
            
            // Check for next button pagination
            const hasNextButton = 
                document.querySelector('a:text("Next"), a[aria-label="Next"], ' +
                                     'a.next, .next > a, [class*="pagination"] a[rel="next"]') !== null;
            
            // Check for "Load More" button
            const hasLoadMore = 
                document.querySelector('button:text("Load More"), ' +
                                     'button:text("Show More"), ' +
                                     'a:text("Load More"), ' +
                                     '.load-more, .show-more, .more-results') !== null;
            
            // Check for infinite scroll
            const hasInfiniteScroll = 
                document.querySelectorAll('[class*="infinite"], [class*="scroll-container"]').length > 0 ||
                document.querySelector('div[data-scroll], #infinite-scroll') !== null;
            
            // Return an object with all pagination information
            return {
                hasNumberedPagination,
                hasNextButton,
                hasLoadMore,
                hasInfiniteScroll,
                type: hasNumberedPagination ? "numbered" : 
                      hasNextButton ? "next_button" : 
                      hasLoadMore ? "load_more" : 
                      hasInfiniteScroll ? "infinite_scroll" : "none"
            };
        }""")
        
        if pagination_type["type"] == "none":
            self.logger.info("No pagination detected")
            return []
        
        self.logger.info(f"Detected pagination type: {pagination_type['type']}")
        
        # Save current page URL to avoid loops
        current_url = page.url
        visited_urls = set([current_url])
        
        # Store results from pagination
        pagination_results = []
        
        # Process pagination based on type
        if pagination_type["type"] == "numbered" or pagination_type["type"] == "next_button":
            for page_num in range(2, max_pages + 1):  # Start from page 2 since we already have page 1
                has_next = await self._go_to_next_page(page, page_num, pagination_type["type"])
                if not has_next or page.url in visited_urls:
                    break
                
                visited_urls.add(page.url)
                await self._wait_for_page_load(page)
                
                # Extract results from this page
                page_results = await self._extract_results(page, search_term, options)
                if page_results and page_results.get("items"):
                    self.logger.info(f"Extracted {len(page_results['items'])} results from page {page_num}")
                    pagination_results.append(page_results)
                else:
                    break
        
        elif pagination_type["type"] == "load_more":
            for click_num in range(1, max_pages):
                has_more = await self._click_load_more(page)
                if not has_more:
                    break
                
                await self._wait_for_page_load(page)
                
                # Extract results after loading more
                page_results = await self._extract_results(page, search_term, options)
                if page_results and page_results.get("items"):
                    self.logger.info(f"Extracted {len(page_results['items'])} results after {click_num} 'load more' clicks")
                    pagination_results.append(page_results)
                else:
                    break
        
        elif pagination_type["type"] == "infinite_scroll":
            for scroll_num in range(1, max_pages):
                has_more = await self._trigger_infinite_scroll(page)
                if not has_more:
                    break
                
                await self._wait_for_page_load(page)
                
                # Extract results after scrolling
                page_results = await self._extract_results(page, search_term, options)
                if page_results and page_results.get("items"):
                    self.logger.info(f"Extracted {len(page_results['items'])} results after {scroll_num} scrolls")
                    pagination_results.append(page_results)
                else:
                    break
        
        return pagination_results
    
    async def _go_to_next_page(self, page: Page, page_num: int, pagination_type: str) -> bool:
        """Go to the next page in the pagination"""
        try:
            if pagination_type == "numbered":
                # Try to find and click the specific page number
                next_page_selectors = [
                    f'a:text("{page_num}")',
                    f'a[aria-label="Page {page_num}"]',
                    f'[class*="pagination"] a:text("{page_num}")'
                ]
                
                for selector in next_page_selectors:
                    if await page.is_visible(selector, timeout=1000):
                        await page.click(selector)
                        await page.wait_for_load_state("networkidle", timeout=10000)
                        return True
                        
                # If specific page number not found, try next button
                return await self._click_next_button(page)
            else:
                # Use next button navigation
                return await self._click_next_button(page)
        except Exception as e:
            self.logger.warning(f"Error navigating to next page: {e}")
            return False
    
    async def _click_next_button(self, page: Page) -> bool:
        """Click the Next button in pagination"""
        next_button_selectors = [
            'a:text("Next")',
            'a[aria-label="Next"]',
            '[class*="next"]',
            'a.next',
            '.pagination .next',
            'button:text("Next")',
            'a[rel="next"]',
            'svg[aria-label="Next"]'
        ]
        
        try:
            for selector in next_button_selectors:
                if await page.is_visible(selector, timeout=1000):
                    await page.click(selector)
                    await page.wait_for_load_state("networkidle", timeout=10000)
                    return True
            
            return False
        except Exception as e:
            self.logger.warning(f"Error clicking next button: {e}")
            return False
    
    async def _click_load_more(self, page: Page) -> bool:
        """Click 'Load More' button"""
        load_more_selectors = [
            'button:text("Load More")',
            'button:text("Show More")',
            'a:text("Load More")',
            'a:text("Show More")',
            '.load-more',
            '.show-more',
            '.more-results',
            '[class*="load-more"]',
            '[class*="show-more"]'
        ]
        
        try:
            for selector in load_more_selectors:
                if await page.is_visible(selector, timeout=1000):
                    # Store page height to check if content was actually loaded
                    old_height = await page.evaluate("document.body.scrollHeight")
                    
                    await page.click(selector)
                    await page.wait_for_timeout(2000)  # Wait for new content to load
                    
                    # Check if the page got taller (new content loaded)
                    new_height = await page.evaluate("document.body.scrollHeight")
                    return new_height > old_height
            
            return False
        except Exception as e:
            self.logger.warning(f"Error clicking load more: {e}")
            return False
    
    async def _trigger_infinite_scroll(self, page: Page) -> bool:
        """Trigger infinite scroll by scrolling to the bottom of the page"""
        try:
            # Get current page height
            old_height = await page.evaluate("document.body.scrollHeight")
            
            # Scroll to the bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            
            # Wait for possible loading
            await page.wait_for_timeout(2000)
            
            # Check if page height increased (content loaded)
            new_height = await page.evaluate("document.body.scrollHeight")
            
            return new_height > old_height
        except Exception as e:
            self.logger.warning(f"Error triggering infinite scroll: {e}")
            return False
    
    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Crawl a URL and extract data.
        
        Args:
            start_url: Starting URL to crawl
            **kwargs: Additional arguments
            
        Returns:
            Crawled data or None
        """
        try:
            return self.execute(start_url, **kwargs)
        except Exception as e:
            self.logger.error(f"Error crawling {start_url}: {e}")
            return None
    
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Extract data from HTML content.
        
        Args:
            html_content: HTML content to extract from
            url: Source URL
            **kwargs: Additional arguments
            
        Returns:
            Extracted data or None
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract basic data from HTML
            extracted_data = {
                'url': url,
                'title': soup.title.string if soup.title else '',
                'text': extract_text_fast(html_content),
                'links': [a.get('href') for a in soup.find_all('a', href=True)],
                'timestamp': time.time(),
                'method': 'dom_extraction'
            }
            
            # Store result
            if not hasattr(self, '_results'):
                self._results = []
            self._results.append(extracted_data)
            
            return extracted_data
        except Exception as e:
            self.logger.error(f"Error extracting from HTML: {e}")
            return None
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get all collected results.
        
        Returns:
            List of result dictionaries
        """
        if not hasattr(self, '_results'):
            self._results = []
        return self._results