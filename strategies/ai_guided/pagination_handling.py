"""
Pagination Handling Module

Provides utilities for detecting and handling different types of pagination
across websites to enable comprehensive content crawling.
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from urllib.parse import urlparse, urljoin, parse_qs, urlencode, urlunparse
from bs4 import BeautifulSoup

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PaginationHandler")

class PaginationHandler:
    """
    Handles detection and traversal of paginated content.
    
    This class:
    - Identifies pagination patterns on a website
    - Facilitates navigation through multi-page content
    - Adapts to different pagination implementations (numbered, infinite scroll, load more)
    - Manages state across paginated requests
    """
    
    def __init__(self, max_pages: int = 10, delay_between_pages: float = 1.0):
        """
        Initialize the PaginationHandler.
        
        Args:
            max_pages: Maximum number of pages to traverse
            delay_between_pages: Delay in seconds between page requests
        """
        self.max_pages = max_pages
        self.delay_between_pages = delay_between_pages
        self.known_pagination_patterns = self._load_known_patterns()
        
        # State management for pagination tracking
        self.pagination_state = {}
        self.visited_pagination_urls = set()
        self.pagination_sessions = {}
        
        # JavaScript pagination detection settings
        self.js_pagination_snippets = self._load_js_detection_snippets()
        self.js_pagination_indicators = self._load_js_pagination_indicators()
        
    def _load_known_patterns(self) -> Dict[str, List[str]]:
        """
        Load known pagination patterns for different types of sites.
        
        Returns:
            Dictionary of pagination patterns by site type
        """
        return {
            "numbered": [
                ".pagination", ".pager", ".pages", 
                "[class*='pagination']", "[class*='pager']",
                ".page-numbers", ".page-navigation",
                "[role='navigation']", "[aria-label*='pagination']"
            ],
            "next_link": [
                "a[rel='next']", "a:contains('Next')", "a:contains('next')",
                "a:contains('»')", "a:contains('→')",
                ".next", ".next-page", "[class*='next']",
                "a[aria-label*='Next']", "button:contains('Next')"
            ],
            "load_more": [
                "button:contains('Load More')", "a:contains('Load More')",
                "button:contains('Show More')", "a:contains('Show More')",
                "[class*='load-more']", "[id*='load-more']",
                "[data-role*='load-more']", "[onclick*='load']"
            ],
            "infinite_scroll": [
                "[data-infinite-scroll]", "[data-load-more]", "[data-lazy-load]",
                "[class*='infinite']", "[id*='infinite']",
                "script:contains('infiniteScroll')", "script:contains('InfiniteScroll')",
                "[data-page]", "[data-scroll]"
            ]
        }
    
    def _load_js_detection_snippets(self) -> List[Dict[str, Any]]:
        """
        Load JavaScript snippets that help detect different pagination mechanisms.
        
        Returns:
            List of objects containing JavaScript detection snippets and metadata
        """
        return [
            {
                "name": "scroll_event_listeners",
                "description": "Detects scroll event listeners that might indicate infinite scroll",
                "code": """
                    const scrollListeners = [];
                    const originalAddEventListener = EventTarget.prototype.addEventListener;
                    
                    // Check existing listeners on window and document
                    if (window._paginationInfo) {
                        return window._paginationInfo;
                    }
                    
                    // Create an object to store pagination information
                    window._paginationInfo = {
                        hasScrollListeners: false,
                        scrollListenerCount: 0,
                        hasInfiniteScroll: false,
                        hasLoadMoreButton: false,
                        loadMoreSelectors: [],
                        paginationSelectors: [],
                        detectedPaginationType: null
                    };
                    
                    // Check for infinite scroll libraries
                    const infiniteScrollLibraries = [
                        'infiniteScroll', 
                        'InfiniteScroll', 
                        'infinite-scroll',
                        'waypoints', 
                        'jscroll'
                    ];
                    
                    // Check if any infinite scroll libraries are defined
                    for (const lib of infiniteScrollLibraries) {
                        if (typeof window[lib] !== 'undefined' || 
                            document.querySelector(`script[src*="${lib}"]`)) {
                            window._paginationInfo.hasInfiniteScroll = true;
                            window._paginationInfo.detectedPaginationType = 'infinite_scroll';
                            break;
                        }
                    }
                    
                    // Look for load more buttons
                    const loadMoreSelectors = [
                        'button:contains("Load More")', 
                        'a:contains("Load More")',
                        'button:contains("Show More")', 
                        'a:contains("Show More")',
                        '[class*="load-more"]', 
                        '[id*="load-more"]',
                        '[data-role*="load-more"]', 
                        '[onclick*="load"]'
                    ];
                    
                    for (const selector of loadMoreSelectors) {
                        try {
                            const elements = document.querySelectorAll(selector);
                            if (elements.length > 0) {
                                window._paginationInfo.hasLoadMoreButton = true;
                                window._paginationInfo.loadMoreSelectors.push(selector);
                                window._paginationInfo.detectedPaginationType = 'load_more';
                            }
                        } catch (e) {
                            // Skip invalid selectors
                        }
                    }
                    
                    // Look for pagination elements
                    const paginationSelectors = [
                        '.pagination', 
                        '.pager', 
                        '.pages',
                        '[class*="pagination"]', 
                        '[class*="pager"]',
                        '.page-numbers', 
                        '.page-navigation',
                        '[role="navigation"]', 
                        '[aria-label*="pagination"]'
                    ];
                    
                    for (const selector of paginationSelectors) {
                        try {
                            const elements = document.querySelectorAll(selector);
                            if (elements.length > 0) {
                                window._paginationInfo.paginationSelectors.push(selector);
                                if (!window._paginationInfo.detectedPaginationType) {
                                    window._paginationInfo.detectedPaginationType = 'numbered';
                                }
                            }
                        } catch (e) {
                            // Skip invalid selectors
                        }
                    }
                    
                    // Count scroll event listeners
                    ['scroll', 'touchmove'].forEach(eventType => {
                        const listeners = getEventListeners(window, eventType) || [];
                        window._paginationInfo.scrollListenerCount += listeners.length;
                        window._paginationInfo.hasScrollListeners = window._paginationInfo.hasScrollListeners || (listeners.length > 0);
                        
                        // Also check document for scroll listeners
                        const docListeners = getEventListeners(document, eventType) || [];
                        window._paginationInfo.scrollListenerCount += docListeners.length;
                        window._paginationInfo.hasScrollListeners = window._paginationInfo.hasScrollListeners || (docListeners.length > 0);
                    });
                    
                    // If we have scroll listeners but no detected pagination type yet, assume infinite scroll
                    if (window._paginationInfo.hasScrollListeners && window._paginationInfo.scrollListenerCount > 0 && !window._paginationInfo.detectedPaginationType) {
                        window._paginationInfo.hasInfiniteScroll = true;
                        window._paginationInfo.detectedPaginationType = 'infinite_scroll';
                    }
                    
                    return window._paginationInfo;
                """,
                "result_processing": "json"
            },
            {
                "name": "network_request_monitor",
                "description": "Sets up monitoring for API requests that might be used for pagination",
                "code": """
                    // Create a tracker for API requests if not already created
                    if (!window._apiRequestTracker) {
                        window._apiRequestTracker = {
                            requests: [],
                            originalFetch: window.fetch,
                            originalXhrOpen: XMLHttpRequest.prototype.open,
                            originalXhrSend: XMLHttpRequest.prototype.send
                        };
                        
                        // Override fetch
                        window.fetch = function(url, options) {
                            const timestamp = Date.now();
                            const promise = window._apiRequestTracker.originalFetch.apply(this, arguments);
                            
                            // Store request information
                            window._apiRequestTracker.requests.push({
                                type: 'fetch',
                                url: url.toString(),
                                timestamp,
                                options
                            });
                            
                            return promise;
                        };
                        
                        // Override XMLHttpRequest
                        let currentXhr;
                        XMLHttpRequest.prototype.open = function(method, url) {
                            currentXhr = {
                                type: 'xhr',
                                method,
                                url: url.toString(),
                                timestamp: Date.now()
                            };
                            window._apiRequestTracker.originalXhrOpen.apply(this, arguments);
                        };
                        
                        XMLHttpRequest.prototype.send = function() {
                            if (currentXhr) {
                                window._apiRequestTracker.requests.push(currentXhr);
                                currentXhr = null;
                            }
                            window._apiRequestTracker.originalXhrSend.apply(this, arguments);
                        };
                    }
                    
                    // Return current API requests
                    const paginationRelatedRequests = window._apiRequestTracker.requests.filter(req => {
                        const url = req.url.toLowerCase();
                        return url.includes('page=') || 
                               url.includes('limit=') || 
                               url.includes('offset=') ||
                               url.includes('api') || 
                               url.includes('load') || 
                               url.includes('next');
                    });
                    
                    return {
                        totalRequests: window._apiRequestTracker.requests.length,
                        paginationRequests: paginationRelatedRequests.length,
                        recentRequests: paginationRelatedRequests.slice(-5)
                    };
                """,
                "result_processing": "json"
            },
            {
                "name": "mutation_observer",
                "description": "Sets up mutation observer to detect DOM changes after scroll",
                "code": """
                    // If we already have an observer set up, return its data
                    if (window._mutationData) {
                        return {
                            observedMutations: window._mutationData.mutationCount,
                            contentAdded: window._mutationData.contentAdded,
                            significantChanges: window._mutationData.significantChanges
                        };
                    }
                    
                    // Set up storage for mutation data
                    window._mutationData = {
                        mutationCount: 0,
                        contentAdded: false,
                        significantChanges: false,
                        observer: null
                    };
                    
                    // Create mutation observer
                    const observer = new MutationObserver((mutations) => {
                        for (const mutation of mutations) {
                            window._mutationData.mutationCount++;
                            
                            // Check for added nodes
                            if (mutation.addedNodes && mutation.addedNodes.length > 0) {
                                for (const node of mutation.addedNodes) {
                                    // Only count element nodes
                                    if (node.nodeType === Node.ELEMENT_NODE) {
                                        // Check if this is a significant content addition
                                        if (node.textContent && node.textContent.length > 100) {
                                            window._mutationData.contentAdded = true;
                                        }
                                        
                                        // Check for significant elements that might be content cards
                                        const isContentCard = node.classList && 
                                            (node.classList.contains('item') || 
                                             node.classList.contains('card') ||
                                             node.classList.contains('product') ||
                                             node.classList.contains('post'));
                                             
                                        if (isContentCard) {
                                            window._mutationData.significantChanges = true;
                                        }
                                        
                                        // Check if it contains a lot of child elements (likely a content card)
                                        if (node.querySelectorAll('*').length > 10) {
                                            window._mutationData.significantChanges = true;
                                        }
                                    }
                                }
                            }
                        }
                    });
                    
                    // Start observing
                    observer.observe(document.body, { 
                        childList: true, 
                        subtree: true 
                    });
                    
                    // Store observer for later
                    window._mutationData.observer = observer;
                    
                    // Scroll to trigger any lazy loading
                    window.scrollTo(0, document.body.scrollHeight);
                    
                    // Return initial state (to be checked again after scroll)
                    return {
                        observedMutations: window._mutationData.mutationCount,
                        contentAdded: window._mutationData.contentAdded,
                        significantChanges: window._mutationData.significantChanges
                    };
                """,
                "result_processing": "json" 
            }
        ]
    
    def _load_js_pagination_indicators(self) -> Dict[str, List[str]]:
        """
        Load indicators used to identify JavaScript-based pagination mechanisms.
        
        Returns:
            Dictionary of JavaScript pagination indicators by type
        """
        return {
            "infinite_scroll_libraries": [
                "infinitescroll", "infinite-scroll", "jscroll", "waypoints",
                "ias", "jquery.infinitescroll", "ajax-scroll", "lazy-load"
            ],
            "infinite_scroll_functions": [
                "loadMore", "loadNextPage", "appendItems", "fetchNextPage",
                "getMoreContent", "loadAdditionalItems", "scrollHandler"
            ],
            "ajax_pagination_indicators": [
                "data-page", "data-next-page", "data-load-more", "data-url",
                "data-remote", "data-pagination", "data-infinite", "data-lazy"
            ],
            "spa_pagination_patterns": [
                "#/page/", "?page=", "&page=", "?p=", "&p=", "?offset=", "&offset="
            ]
        }
        
    def create_pagination_session(self, session_id: str, start_url: str, site_type: str = "unknown") -> Dict[str, Any]:
        """
        Create a new pagination session for tracking state across multiple paginated requests.
        
        Args:
            session_id: Unique identifier for the session
            start_url: Starting URL for pagination
            site_type: Type of site (affects pagination strategy)
            
        Returns:
            Dictionary containing session information
        """
        if session_id in self.pagination_sessions:
            logger.warning(f"Session {session_id} already exists. Overwriting.")
        
        # Initialize session state
        session = {
            "session_id": session_id,
            "start_url": start_url,
            "site_type": site_type,
            "current_url": start_url,
            "current_page": 1,
            "total_pages": None,
            "pagination_type": None,
            "page_parameter": None,
            "visited_urls": set([start_url]),
            "pagination_pattern": None,
            "results_count": 0,
            "last_processed_at": None,
            "error_count": 0,
            "status": "created",
            "history": [],
            # Add JavaScript-specific pagination state information
            "js_pagination_detected": False,
            "js_pagination_type": None,
            "js_detection_results": {}
        }
        
        self.pagination_sessions[session_id] = session
        logger.info(f"Created pagination session {session_id} starting at {start_url}")
        
        return session
    
    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current state for a pagination session.
        
        Args:
            session_id: ID of the pagination session
            
        Returns:
            Session state dictionary or None if session doesn't exist
        """
        if session_id not in self.pagination_sessions:
            logger.error(f"Session {session_id} not found")
            return None
            
        return self.pagination_sessions[session_id]
    
    def update_session_state(self, session_id: str, **kwargs) -> bool:
        """
        Update the state of a pagination session.
        
        Args:
            session_id: ID of the pagination session
            **kwargs: Key-value pairs to update in the session state
            
        Returns:
            Boolean indicating success
        """
        if session_id not in self.pagination_sessions:
            logger.error(f"Cannot update session {session_id}: not found")
            return False
            
        session = self.pagination_sessions[session_id]
        
        # Update session with provided values
        for key, value in kwargs.items():
            if key == "visited_urls" and isinstance(value, str):
                # If adding a single URL to visited_urls
                session["visited_urls"].add(value)
            elif key == "history" and isinstance(value, dict):
                # If adding a history entry
                session["history"].append(value)
            else:
                # General case - direct update
                session[key] = value
                
        logger.debug(f"Updated session {session_id} state: {', '.join(kwargs.keys())}")
        return True
    
    def get_next_page(self, session_id: str) -> Optional[str]:
        """
        Get the URL for the next page in the pagination sequence.
        
        Args:
            session_id: ID of the pagination session
            
        Returns:
            URL of the next page or None if no more pages
        """
        if session_id not in self.pagination_sessions:
            logger.error(f"Session {session_id} not found")
            return None
            
        session = self.pagination_sessions[session_id]
        
        # Check if we've reached the maximum pages
        if session["current_page"] >= self.max_pages:
            logger.info(f"Session {session_id} reached max pages limit ({self.max_pages})")
            return None
            
        # Check if we've reached the total pages (if known)
        if session["total_pages"] and session["current_page"] >= session["total_pages"]:
            logger.info(f"Session {session_id} reached end of pagination ({session['total_pages']} pages)")
            return None
            
        # If we have a pagination pattern, construct the next URL
        if session["pagination_pattern"] and session["page_parameter"]:
            next_page_num = session["current_page"] + 1
            
            if "{{page}}" in session["pagination_pattern"]:
                next_url = session["pagination_pattern"].replace("{{page}}", str(next_page_num))
                return next_url
                
        # If the session has a precalculated next URL, use it
        return session.get("next_url")
    
    def track_page_state(self, session_id: str, url: str, html_content: str) -> Dict[str, Any]:
        """
        Track the state of a page for pagination purposes.
        
        Args:
            session_id: ID of the pagination session
            url: URL of the current page
            html_content: HTML content of the current page
            
        Returns:
            Updated page state information
        """
        if session_id not in self.pagination_sessions:
            logger.error(f"Session {session_id} not found")
            return {}
            
        session = self.pagination_sessions[session_id]
        
        # Update session with the current URL
        session["current_url"] = url
        session["visited_urls"].add(url)
        
        # Detect pagination information for this page
        pagination_info = self.detect_pagination_type(html_content, url)
        
        # Update session with pagination information
        if pagination_info["has_pagination"]:
            session["pagination_type"] = pagination_info["pagination_type"]
            session["next_url"] = pagination_info["next_page_url"]
            session["current_page"] = pagination_info["current_page"]
            
            if pagination_info["total_pages"]:
                session["total_pages"] = pagination_info["total_pages"]
                
            if pagination_info["page_parameter"]:
                session["page_parameter"] = pagination_info["page_parameter"]
                
            # Attempt to detect a pattern if we don't have one yet
            if not session["pagination_pattern"] and session["next_url"]:
                pattern = self._infer_pagination_pattern(url, session["next_url"], session["page_parameter"])
                session["pagination_pattern"] = pattern
                
        # Add to history
        history_entry = {
            "url": url,
            "page_number": session["current_page"],
            "has_pagination": pagination_info["has_pagination"],
            "next_url": pagination_info.get("next_page_url"),
        }
        session["history"].append(history_entry)
        
        logger.info(f"Session {session_id}: Processed page {session['current_page']}, next URL: {session.get('next_url', 'None')}")
        
        return session
    
    def _infer_pagination_pattern(self, current_url: str, next_url: str, page_param: Optional[str]) -> Optional[str]:
        """
        Infer a pattern for pagination URLs based on current and next URLs.
        
        Args:
            current_url: Current page URL
            next_url: Next page URL
            page_param: Page parameter name if known
            
        Returns:
            Pagination pattern with {{page}} placeholder, or None if pattern can't be determined
        """
        if not next_url:
            return None
            
        try:
            # If we know the page parameter, use it to create a pattern
            if page_param:
                parsed = urlparse(next_url)
                query_params = parse_qs(parsed.query)
                
                if page_param in query_params:
                    # Create a template with the page parameter
                    query_params[page_param] = ["{{page}}"]
                    new_query = urlencode(query_params, doseq=True)
                    
                    # Replace the query string in the URL
                    pattern_parts = list(parsed)
                    pattern_parts[4] = new_query  # index 4 is the query string
                    return urlunparse(tuple(pattern_parts))
            
            # Try to find differences between current and next URLs
            current_parsed = urlparse(current_url)
            next_parsed = urlparse(next_url)
            
            # Check for page numbers in the path
            current_path = current_parsed.path.split('/')
            next_path = next_parsed.path.split('/')
            
            if len(current_path) == len(next_path):
                for i in range(len(current_path)):
                    if current_path[i] != next_path[i]:
                        # Check if this segment is a number
                        if next_path[i].isdigit() and (current_path[i].isdigit() or current_path[i] == ''):
                            # Replace with a placeholder
                            next_path[i] = "{{page}}"
                            
                            # Reconstruct the URL
                            pattern_parts = list(next_parsed)
                            pattern_parts[2] = '/'.join(next_path)  # index 2 is the path
                            return urlunparse(tuple(pattern_parts))
                            
                        # Check for page prefix (e.g., "page1", "page2")
                        elif (next_path[i].startswith('page') and next_path[i][4:].isdigit() and
                              (current_path[i].startswith('page') or current_path[i] == '')):
                            # Replace with a placeholder but keep the prefix
                            next_path[i] = "page{{page}}"
                            
                            # Reconstruct the URL
                            pattern_parts = list(next_parsed)
                            pattern_parts[2] = '/'.join(next_path)  # index 2 is the path
                            return urlunparse(tuple(pattern_parts))
                            
            # If pattern not found, return None
            return None
            
        except Exception as e:
            logger.error(f"Error inferring pagination pattern: {str(e)}")
            return None
    
    async def detect_pagination_type(self, html_content: str, url: str) -> Dict[str, Any]:
        """
        Detect the type of pagination used on the page.
        
        Args:
            html_content: HTML content of the page
            url: URL of the page
            
        Returns:
            Dictionary with pagination information
        """
        pagination_info = {
            "has_pagination": False,
            "pagination_type": None,
            "next_page_url": None,
            "current_page": 1,
            "total_pages": None,
            "page_parameter": None,
            # Add JavaScript pagination detection fields
            "js_pagination_detected": False,
            "js_pagination_type": None,
            "js_indicators": []
        }
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Check for standard pagination indicators first
        for pagination_type, selectors in self.known_pagination_patterns.items():
            for selector in selectors:
                try:
                    elements = soup.select(selector)
                    if elements:
                        pagination_info["has_pagination"] = True
                        pagination_info["pagination_type"] = pagination_type
                        break
                except Exception as e:
                    logger.debug(f"Error checking selector {selector}: {str(e)}")
                    
            if pagination_info["has_pagination"]:
                break
        
        # Check for JavaScript pagination indicators even if standard pagination was found
        js_pagination_result = self._detect_js_pagination_indicators(html_content, soup)
        
        # If JavaScript pagination was detected, update pagination info
        if js_pagination_result["detected"]:
            pagination_info["js_pagination_detected"] = True
            pagination_info["js_pagination_type"] = js_pagination_result["pagination_type"]
            pagination_info["js_indicators"] = js_pagination_result["indicators"]
            
            # If standard pagination wasn't detected but JavaScript pagination was, use the JavaScript pagination type
            if not pagination_info["has_pagination"]:
                pagination_info["has_pagination"] = True
                pagination_info["pagination_type"] = js_pagination_result["pagination_type"]
                
        # If pagination found, extract additional information
        if pagination_info["has_pagination"]:
            # Extract next page URL if available
            next_page_url = self._extract_next_page_url(soup, url, pagination_info["pagination_type"])
            pagination_info["next_page_url"] = next_page_url
            
            # Try to determine current page and total pages
            current_page, total_pages = self._extract_page_numbers(soup, url)
            pagination_info["current_page"] = current_page
            pagination_info["total_pages"] = total_pages
            
            # Try to determine the URL parameter used for pagination
            page_param = self._extract_page_parameter(url, next_page_url)
            pagination_info["page_parameter"] = page_param
            
        return pagination_info
    
    def _detect_js_pagination_indicators(self, html_content: str, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Detect JavaScript-based pagination mechanisms by analyzing HTML content.
        
        Args:
            html_content: HTML content of the page
            soup: BeautifulSoup object of the page
            
        Returns:
            Dictionary with JavaScript pagination detection results
        """
        result = {
            "detected": False,
            "pagination_type": None,
            "indicators": [],
            "confidence": 0.0
        }
        
        # Check for script tags with pagination-related code
        scripts = soup.find_all('script')
        script_content = ""
        for script in scripts:
            if script.string:
                script_content += script.string.lower() + " "
                
        # Check for infinite scroll libraries
        for library in self.js_pagination_indicators["infinite_scroll_libraries"]:
            if library.lower() in script_content:
                result["detected"] = True
                result["pagination_type"] = "infinite_scroll"
                result["indicators"].append(f"Infinite scroll library: {library}")
                result["confidence"] += 0.3
                
        # Check for infinite scroll functions
        for func in self.js_pagination_indicators["infinite_scroll_functions"]:
            if f"function {func.lower()}(" in script_content or f"{func.lower()}:" in script_content:
                result["detected"] = True
                result["pagination_type"] = "infinite_scroll"
                result["indicators"].append(f"Infinite scroll function: {func}")
                result["confidence"] += 0.2
                
        # Check for event listeners in inline scripts
        if "addeventlistener('scroll'" in script_content or "addeventlistener(\"scroll\"" in script_content:
            result["detected"] = True
            result["pagination_type"] = "infinite_scroll"
            result["indicators"].append("Scroll event listener")
            result["confidence"] += 0.25
            
        # Check for AJAX pagination indicators
        for indicator in self.js_pagination_indicators["ajax_pagination_indicators"]:
            elements = soup.find_all(attrs={indicator: True})
            if elements:
                result["detected"] = True
                result["pagination_type"] = "load_more" if "load" in indicator else "infinite_scroll"
                result["indicators"].append(f"AJAX pagination indicator: {indicator}")
                result["confidence"] += 0.2
                
        # Check for SPA pagination patterns in the URL
        for pattern in self.js_pagination_indicators["spa_pagination_patterns"]:
            if pattern in html_content:
                result["detected"] = True
                result["pagination_type"] = "spa_pagination"
                result["indicators"].append(f"SPA pagination pattern: {pattern}")
                result["confidence"] += 0.2
                
        # Check for load more buttons
        load_more_buttons = soup.find_all(['button', 'a'], string=lambda s: s and ('load more' in s.lower() or 'show more' in s.lower()))
        if load_more_buttons:
            result["detected"] = True
            result["pagination_type"] = "load_more"
            result["indicators"].append(f"Load more button: {len(load_more_buttons)} found")
            result["confidence"] += 0.3
            
        # Normalize confidence
        result["confidence"] = min(1.0, result["confidence"])
        
        return result
    
    async def detect_js_pagination_with_browser(self, 
                                              crawler: AsyncWebCrawler, 
                                              url: str) -> Dict[str, Any]:
        """
        Use browser automation to detect JavaScript-based pagination by executing detection scripts.
        
        Args:
            crawler: AsyncWebCrawler instance
            url: URL to check for JS-based pagination
            
        Returns:
            Dictionary with JavaScript pagination detection results
        """
        result = {
            "detected": False,
            "pagination_type": None,
            "indicators": [],
            "details": {}
        }
        
        try:
            logger.info(f"Detecting JavaScript pagination for {url}")
            
            # Configure crawler to use JavaScript
            config = CrawlerRunConfig(
                use_javascript=True,
                max_execution_time=15  # Allow 15 seconds to execute detection scripts
            )
            
            # Execute each detection script sequentially
            for script_config in self.js_pagination_snippets:
                script_name = script_config["name"]
                logger.debug(f"Running JS detection script: {script_name}")
                
                # Configure crawler with the detection script
                detection_config = CrawlerRunConfig(
                    use_javascript=True,
                    javascript_snippets=[script_config["code"]],
                    max_execution_time=10
                )
                
                # Execute script
                script_result = await crawler.arun(url=url, config=detection_config)
                
                if not script_result.success:
                    logger.warning(f"Failed to execute {script_name} script: {script_result.error}")
                    continue
                    
                # Process the script result
                try:
                    if script_config["result_processing"] == "json":
                        # Extract JSON from the script result
                        script_data = json.loads(script_result.javascript_result)
                    else:
                        script_data = script_result.javascript_result
                        
                    # Store script results
                    result["details"][script_name] = script_data
                    
                    # Process specific scripts
                    if script_name == "scroll_event_listeners":
                        if script_data.get("hasScrollListeners") or script_data.get("hasInfiniteScroll"):
                            result["detected"] = True
                            result["indicators"].append("Scroll event listeners detected")
                            
                            if script_data.get("detectedPaginationType"):
                                result["pagination_type"] = script_data["detectedPaginationType"]
                            else:
                                result["pagination_type"] = "infinite_scroll"
                    
                    elif script_name == "network_request_monitor":
                        if script_data.get("paginationRequests", 0) > 0:
                            result["detected"] = True
                            result["indicators"].append(f"Pagination-related network requests: {script_data.get('paginationRequests', 0)}")
                            result["pagination_type"] = "api_based"
                    
                    elif script_name == "mutation_observer":
                        if script_data.get("contentAdded") or script_data.get("significantChanges"):
                            result["detected"] = True
                            result["indicators"].append("Content mutations detected after scroll")
                            result["pagination_type"] = "infinite_scroll"
                            
                except Exception as e:
                    logger.error(f"Error processing {script_name} result: {str(e)}")
            
            # After running all detection scripts, check for evidence of JavaScript pagination
            if result["detected"]:
                logger.info(f"JavaScript pagination detected: {result['pagination_type']} with indicators: {result['indicators']}")
            else:
                logger.info("No JavaScript pagination detected")
                
            return result
            
        except Exception as e:
            logger.error(f"Error in JavaScript pagination detection: {str(e)}")
            return {"detected": False, "error": str(e)}
    
    async def process_paginated_results_with_state(
        self, 
        crawler: AsyncWebCrawler,
        session_id: str,
        content_processor: callable,
        max_pages: Optional[int] = None,
        adaptive_delay: bool = False
    ) -> List<Dict[str, Any]]:
        """
        Process all pages of results using session state management.
        
        Args:
            crawler: AsyncWebCrawler instance
            session_id: ID of the pagination session
            content_processor: Function to process content of each page
            max_pages: Maximum pages to process (overrides instance setting if provided)
            adaptive_delay: Whether to adjust delay based on server response times
            
        Returns:
            List of processed results from all pages
        """
        if session_id not in self.pagination_sessions:
            logger.error(f"Session {session_id} not found")
            return []
            
        session = self.pagination_sessions[session_id]
        all_results = []
        page_count = 0
        max_pages_to_process = max_pages if max_pages is not None else self.max_pages
        
        # Start with current URL from session
        current_url = session["current_url"]
        
        logger.info(f"Beginning pagination processing for session {session_id} from {current_url}")
        session["status"] = "processing"
        
        # Keep track of response times for adaptive delay
        response_times = []
        
        while current_url and page_count < max_pages_to_process:
            # Skip already visited URLs 
            if (current_url in session["visited_urls"] and page_count > 0) or session["js_pagination_detected"]:
                logger.info(f"Skipping already visited URL: {current_url}")
                current_url = self.get_next_page(session_id)
                continue
                
            # Record start time for response time tracking
            start_time = asyncio.get_event_loop().time()
            
            # Fetch the current page
            page_result = await crawler.afetch(current_url)
            
            # Calculate response time
            response_time = asyncio.get_event_loop().time() - start_time
            response_times.append(response_time)
            
            if not page_result.success:
                logger.error(f"Failed to fetch page: {current_url}")
                # Update session with error
                self.update_session_state(session_id, 
                                        error_count=session["error_count"] + 1, 
                                        history={"url": current_url, "error": "Failed to fetch"})
                                        
                # If too many consecutive errors, break
                if session["error_count"] >= 3:
                    logger.error(f"Too many errors in session {session_id}, stopping pagination")
                    session["status"] = "error"
                    break
                    
                # Try next page
                current_url = self.get_next_page(session_id)
                continue
                
            # Process the content of this page
            page_data = await content_processor(page_result.url, page_result.html)
            
            # Update pagination state based on this page
            self.track_page_state(session_id, page_result.url, page_result.html)
            
            # Add results from this page
            if page_data:
                all_results.extend(page_data)
                logger.info(f"Processed page {page_count + 1}, found {len(page_data)} items")
                self.update_session_state(session_id, results_count=session["results_count"] + len(page_data))
                
            # Get the next page URL from session state
            current_url = self.get_next_page(session_id)
            page_count += 1
            
            # Update session state
            self.update_session_state(session_id, current_page=page_count + 1)
            
            if current_url:
                logger.info(f"Moving to next page: {current_url}")
                
                # Apply delay between requests to avoid overwhelming the server
                delay = self.delay_between_pages
                
                # Adjust delay based on server response times if adaptive delay is enabled
                if adaptive_delay and len(response_times) >= 2:
                    avg_response_time = sum(response_times[-3:]) / min(3, len(response_times))
                    # If response time is increasing, increase delay
                    if response_times[-1] > avg_response_time * 1.2:
                        delay = delay * 1.5
                        logger.info(f"Increasing delay to {delay:.2f}s due to slower server response")
                    # If response time is decreasing, decrease delay (but not below base delay)
                    elif response_times[-1] < avg_response_time * 0.8:
                        delay = max(delay * 0.8, self.delay_between_pages)
                        logger.info(f"Decreasing delay to {delay:.2f}s due to faster server response")
                
                await asyncio.sleep(delay)
            
        # Update final session state
        self.update_session_state(
            session_id, 
            status="completed" if current_url is None else "paused",
            last_processed_at=asyncio.get_event_loop().time()
        )
        
        logger.info(f"Pagination processing complete for session {session_id}. Processed {page_count} pages, found {len(all_results)} total items")
        
        return all_results
        
    async def process_paginated_results(
        self, 
        crawler: AsyncWebCrawler,
        start_url: str, 
        site_analysis: Dict[str, Any],
        content_processor: callable,
        max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process all pages of results by following pagination links.
        
        Args:
            crawler: AsyncWebCrawler instance
            start_url: Starting URL 
            site_analysis: Site structure analysis
            content_processor: Function to process content of each page
            max_pages: Maximum pages to process (overrides instance setting if provided)
            
        Returns:
            List of processed results from all pages
        """
        all_results = []
        current_url = start_url
        page_count = 0
        max_pages_to_process = max_pages if max_pages is not None else self.max_pages
        
        logger.info(f"Beginning pagination processing from {start_url}")
        
        while current_url and page_count < max_pages_to_process:
            # Fetch the current page
            page_result = await crawler.afetch(current_url)
            
            if not page_result.success:
                logger.error(f"Failed to fetch page: {current_url}")
                break
                
            # Process the content of this page
            page_data = await content_processor(page_result.url, page_result.html)
            
            # Add results from this page
            if page_data:
                all_results.extend(page_data)
                logger.info(f"Processed page {page_count + 1}, found {len(page_data)} items")
            
            # Detect pagination and find next page URL
            pagination_info = await self.detect_pagination_type(page_result.html, current_url)
            
            # Break if no pagination or no next page
            if not pagination_info["has_pagination"] or not pagination_info["next_page_url"]:
                logger.info("No further pagination detected")
                break
                
            # Update for next iteration
            next_url = pagination_info["next_page_url"]
            if next_url == current_url:
                logger.info("Next URL same as current URL, stopping pagination")
                break
                
            current_url = next_url
            page_count += 1
            
            logger.info(f"Moving to next page: {current_url}")
            
            # Apply delay between requests to avoid overwhelming the server
            await asyncio.sleep(self.delay_between_pages)
            
        logger.info(f"Pagination processing complete. Processed {page_count + 1} pages, found {len(all_results)} total items")
        
        return all_results
        
    async def handle_infinite_scroll(
        self, 
        crawler: AsyncWebCrawler,
        url: str, 
        content_processor: callable,
        max_iterations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Handle infinite scroll pagination by simulating scroll events.
        
        Args:
            crawler: AsyncWebCrawler instance
            url: URL to crawl
            content_processor: Function to process content
            max_iterations: Maximum number of scroll iterations
            
        Returns:
            List of processed results from all scroll operations
        """
        all_results = []
        
        logger.info(f"Handling infinite scroll for {url}")
        
        # Configure crawler to use JavaScript for infinite scroll
        config = CrawlerRunConfig(
            use_javascript=True,
            javascript_snippets=[
                "window.scrollTo(0, document.body.scrollHeight);",
                "await new Promise(resolve => setTimeout(resolve, 2000));"
            ],
            max_execution_time=10  # 10 seconds to allow content to load after scroll
        )
        
        # Initial page load
        page_result = await crawler.arun(url=url, config=config)
        
        if not page_result.success:
            logger.error(f"Failed to fetch initial page: {url}")
            return all_results
            
        # Process initial content
        initial_data = await content_processor(page_result.url, page_result.html)
        if initial_data:
            all_results.extend(initial_data)
            
        # Track content length to detect when no new content is loaded
        previous_content_length = len(page_result.html)
        
        # Perform scroll operations
        for i in range(max_iterations):
            logger.info(f"Scroll iteration {i + 1}")
            
            # Execute scroll with JavaScript
            scroll_result = await crawler.arun(url=url, config=config)
            
            if not scroll_result.success:
                logger.error(f"Failed during scroll iteration {i + 1}")
                break
                
            # Check if content length increased (indicator of new content)
            current_content_length = len(scroll_result.html)
            content_difference = current_content_length - previous_content_length
            
            if content_difference < 1000:  # Arbitrary threshold to detect no new content
                logger.info(f"No significant new content loaded after scroll, stopping")
                break
                
            # Process new content
            new_data = await content_processor(scroll_result.url, scroll_result.html)
            
            # Add new results, avoiding duplicates
            if new_data:
                # Simple deduplication by URL if results have a URL field
                existing_urls = {result.get('url', '') for result in all_results}
                new_items = [item for item in new_data 
                           if item.get('url', '') not in existing_urls]
                
                all_results.extend(new_items)
                logger.info(f"Scroll iteration {i + 1} yielded {len(new_items)} new items")
                
            # Update for next iteration
            previous_content_length = current_content_length
            
            # Delay between scroll operations
            await asyncio.sleep(self.delay_between_pages)
            
        logger.info(f"Infinite scroll handling complete. Found {len(all_results)} total items")
        
        return all_results
        
    async def handle_infinite_scroll_with_state(
        self, 
        crawler: AsyncWebCrawler,
        session_id: str,
        content_processor: callable,
        max_iterations: int = 5,
        scroll_detection_threshold: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Handle infinite scroll pagination with state management.
        
        Args:
            crawler: AsyncWebCrawler instance
            session_id: ID of the pagination session
            content_processor: Function to process content
            max_iterations: Maximum number of scroll iterations
            scroll_detection_threshold: Minimum content length change to detect new content
            
        Returns:
            List of processed results from all scroll operations
        """
        if session_id not in self.pagination_sessions:
            logger.error(f"Session {session_id} not found")
            return []
            
        session = self.pagination_sessions[session_id]
        url = session["current_url"]
        all_results = []
        
        logger.info(f"Handling infinite scroll for session {session_id} at {url}")
        session["status"] = "processing"
        session["pagination_type"] = "infinite_scroll"
        
        # Configure crawler to use JavaScript for infinite scroll
        config = CrawlerRunConfig(
            use_javascript=True,
            javascript_snippets=[
                "window.scrollTo(0, document.body.scrollHeight);",
                "await new Promise(resolve => setTimeout(resolve, 2000));"
            ],
            max_execution_time=10  # 10 seconds to allow content to load after scroll
        )
        
        # Track previous scroll positions if continuing from a previous session
        scroll_iteration = session.get("scroll_iteration", 0)
        previous_content_length = session.get("previous_content_length", 0)
        
        # If this is a new session, do initial page load
        if scroll_iteration == 0:
            page_result = await crawler.arun(url=url, config=config)
            
            if not page_result.success:
                logger.error(f"Failed to fetch initial page: {url}")
                self.update_session_state(session_id, status="error", error_count=session["error_count"] + 1)
                return all_results
                
            # Process initial content
            initial_data = await content_processor(page_result.url, page_result.html)
            if initial_data:
                all_results.extend(initial_data)
                self.update_session_state(session_id, results_count=len(initial_data))
                
            # Store content length for comparison
            previous_content_length = len(page_result.html)
            self.update_session_state(
                session_id, 
                scroll_iteration=1, 
                previous_content_length=previous_content_length
            )
            scroll_iteration = 1
            
        # Perform scroll operations
        starting_iteration = scroll_iteration
        for i in range(starting_iteration, max_iterations + 1):
            logger.info(f"Scroll iteration {i}")
            
            # Execute scroll with JavaScript
            scroll_result = await crawler.arun(url=url, config=config)
            
            if not scroll_result.success:
                logger.error(f"Failed during scroll iteration {i}")
                self.update_session_state(session_id, error_count=session["error_count"] + 1)
                break
                
            # Check if content length increased (indicator of new content)
            current_content_length = len(scroll_result.html)
            content_difference = current_content_length - previous_content_length
            
            # Update history
            self.update_session_state(session_id, 
                                     history={
                                         "iteration": i,
                                         "content_length": current_content_length,
                                         "content_difference": content_difference
                                     })
            
            if content_difference < scroll_detection_threshold:
                logger.info(f"No significant new content loaded after scroll, stopping")
                self.update_session_state(session_id, status="completed", no_more_content=True)
                break
                
            # Process new content
            new_data = await content_processor(scroll_result.url, scroll_result.html)
            
            # Add new results, avoiding duplicates
            if new_data:
                # Simple deduplication by URL if results have a URL field
                existing_urls = {result.get('url', '') for result in all_results}
                new_items = [item for item in new_data 
                           if item.get('url', '') not in existing_urls]
                
                all_results.extend(new_items)
                logger.info(f"Scroll iteration {i} yielded {len(new_items)} new items")
                
                self.update_session_state(
                    session_id, 
                    results_count=session["results_count"] + len(new_items)
                )
                
            # Update for next iteration
            previous_content_length = current_content_length
            self.update_session_state(
                session_id, 
                scroll_iteration=i + 1, 
                previous_content_length=previous_content_length,
                current_page=i + 1  # For consistency with regular pagination
            )
            
            # Delay between scroll operations
            await asyncio.sleep(self.delay_between_pages)
            
        # Set final status
        if session["status"] != "completed":
            self.update_session_state(
                session_id,
                status="paused" if scroll_iteration < max_iterations else "completed", 
                last_processed_at=asyncio.get_event_loop().time()
            )
            
        logger.info(f"Infinite scroll handling complete for session {session_id}. Found {len(all_results)} total items")
        
        return all_results
    
    async def handle_js_pagination(self, 
                                 crawler: AsyncWebCrawler,
                                 url: str,
                                 content_processor: callable,
                                 js_pagination_type: str,
                                 max_interactions: int = 5) -> List[Dict[str, Any]]:
        """
        Handle JavaScript-based pagination by simulating user interactions.
        
        Args:
            crawler: AsyncWebCrawler instance
            url: URL to crawl
            content_processor: Function to process content
            js_pagination_type: Type of JavaScript pagination detected (infinite_scroll, load_more, spa_pagination)
            max_interactions: Maximum number of pagination interactions
            
        Returns:
            List of processed results from all pages
        """
        all_results = []
        
        logger.info(f"Handling JavaScript pagination of type '{js_pagination_type}' for {url}")
        
        # Initial page load
        config = CrawlerRunConfig(
            use_javascript=True,
            max_execution_time=10
        )
        
        page_result = await crawler.arun(url=url, config=config)
        
        if not page_result.success:
            logger.error(f"Failed to fetch initial page: {url}")
            return all_results
            
        # Process initial content
        initial_data = await content_processor(page_result.url, page_result.html)
        if initial_data:
            all_results.extend(initial_data)
            
        # Keep track of the current content length to detect new content
        previous_content_length = len(page_result.html)
        content_fingerprint = self._generate_content_fingerprint(page_result.html)
        
        # Handle different types of JS pagination
        for i in range(max_interactions):
            logger.info(f"JavaScript pagination interaction {i + 1}")
            
            # Prepare JS snippet based on pagination type
            if js_pagination_type == "infinite_scroll":
                # For infinite scroll, simulate scrolling to the bottom
                js_snippet = """
                    window.scrollTo(0, document.body.scrollHeight);
                    await new Promise(resolve => setTimeout(resolve, 3000));
                    window.scrollTo(0, document.body.scrollHeight);
                    await new Promise(resolve => setTimeout(resolve, 1000));
                """
            elif js_pagination_type == "load_more":
                # For load more buttons, find and click the button
                js_snippet = """
                    function findLoadMoreButton() {
                        // Common load more button selectors
                        const selectors = [
                            'button:contains("Load More")', 
                            'a:contains("Load More")',
                            'button:contains("Show More")', 
                            'a:contains("Show More")',
                            '[class*="load-more"]', 
                            '[id*="load-more"]',
                            '[data-role*="load-more"]', 
                            '[class*="loadmore"]',
                            '[class*="showmore"]',
                            'button:contains("more")',
                            'a.more'
                        ];
                        
                        // Try each selector
                        for (const selector of selectors) {
                            try {
                                const elements = document.querySelectorAll(selector);
                                for (const el of elements) {
                                    if (el.offsetParent !== null) { // Check if visible
                                        return el;
                                    }
                                }
                            } catch (e) {
                                // Skip invalid selectors
                            }
                        }
                        
                        return null;
                    }
                    
                    const loadMoreButton = findLoadMoreButton();
                    if (loadMoreButton) {
                        loadMoreButton.click();
                        await new Promise(resolve => setTimeout(resolve, 3000));
                        return { clicked: true, buttonText: loadMoreButton.innerText };
                    } else {
                        return { clicked: false, error: "No load more button found" };
                    }
                """
            elif js_pagination_type == "spa_pagination":
                # For SPA pagination, look for next page link and click it
                js_snippet = """
                    function findNextPageLink() {
                        // Common next page link selectors
                        const selectors = [
                            'a[rel="next"]',
                            'a:contains("Next")', 
                            'a:contains("next")',
                            'a:contains("»")', 
                            'a:contains("→")',
                            '.next', 
                            '.next-page',
                            '[class*="next"]',
                            'a[aria-label*="Next"]'
                        ];
                        
                        // Try each selector
                        for (const selector of selectors) {
                            try {
                                const elements = document.querySelectorAll(selector);
                                for (const el of elements) {
                                    if (el.offsetParent !== null) { // Check if visible
                                        return el;
                                    }
                                }
                            } catch (e) {
                                // Skip invalid selectors
                            }
                        }
                        
                        return null;
                    }
                    
                    const nextLink = findNextPageLink();
                    if (nextLink) {
                        nextLink.click();
                        await new Promise(resolve => setTimeout(resolve, 3000));
                        return { clicked: true, linkText: nextLink.innerText };
                    } else {
                        return { clicked: false, error: "No next page link found" };
                    }
                """
            else:
                # Default to infinite scroll behavior
                js_snippet = """
                    window.scrollTo(0, document.body.scrollHeight);
                    await new Promise(resolve => setTimeout(resolve, 3000));
                """
                
            # Configure crawler with the pagination interaction script
            interaction_config = CrawlerRunConfig(
                use_javascript=True,
                javascript_snippets=[js_snippet],
                max_execution_time=15
            )
            
            # Execute the interaction
            interaction_result = await crawler.arun(url=url, config=interaction_config)
            
            if not interaction_result.success:
                logger.error(f"Failed during JavaScript pagination interaction {i + 1}")
                break
                
            # Check if content has changed
            current_content_length = len(interaction_result.html)
            new_content_fingerprint = self._generate_content_fingerprint(interaction_result.html)
            
            content_difference = current_content_length - previous_content_length
            logger.info(f"Content length change: {content_difference} bytes")
            
            # If content hasn't changed significantly, stop
            if content_difference < 1000 and new_content_fingerprint == content_fingerprint:
                logger.info(f"No significant new content loaded after interaction, stopping")
                break
                
            # Process new content
            new_data = await content_processor(interaction_result.url, interaction_result.html)
            
            # Add new results, avoiding duplicates
            if new_data:
                # Simple deduplication by URL if results have a URL field
                existing_urls = {result.get('url', '') for result in all_results}
                new_items = [item for item in new_data 
                           if item.get('url', '') not in existing_urls]
                
                all_results.extend(new_items)
                logger.info(f"JavaScript pagination interaction {i + 1} yielded {len(new_items)} new items")
                
                # If no new items were found despite content changes, we might be seeing
                # non-relevant content (like ads or recommendations)
                if len(new_items) == 0 and i >= 2:
                    logger.info("No new items found despite content changes, may be loading non-relevant content")
                    break
            else:
                logger.info(f"No new items found in interaction {i + 1}")
                # If we've tried a couple times with no results, stop
                if i >= 1:
                    break
                
            # Update for next iteration
            previous_content_length = current_content_length
            content_fingerprint = new_content_fingerprint
            
            # Delay between interactions
            await asyncio.sleep(self.delay_between_pages)
            
        logger.info(f"JavaScript pagination handling complete. Found {len(all_results)} total items")
        
        return all_results
    
    def _generate_content_fingerprint(self, html_content: str) -> str:
        """
        Generate a fingerprint of the content to detect meaningful changes.
        
        Args:
            html_content: HTML content to fingerprint
            
        Returns:
            String fingerprint of the content
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script, style and other non-content elements
            for element in soup(['script', 'style', 'meta', 'link', 'noscript']):
                element.extract()
                
            # Get all visible text
            text_content = soup.get_text(separator=' ', strip=True)
            
            # Get a sample of the text (first 10000 chars)
            sample = text_content[:10000]
            
            # Hash the sample
            hash_object = hashlib.md5(sample.encode())
            return hash_object.hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating content fingerprint: {str(e)}")
            # Fallback to a simple hash of the whole content
            hash_object = hashlib.md5(html_content.encode())
            return hash_object.hexdigest()