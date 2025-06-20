"""
AJAX handler module for SmartScrape search automation.

This module provides functionality for handling AJAX responses and waiting strategies
for asynchronous content updates in search interfaces.
"""

import logging
import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Callable, Awaitable
import re

class AJAXHandler:
    """
    Handles AJAX requests and responses in search interfaces.
    
    This class provides:
    - Waiting strategies for AJAX-loaded content
    - Response handling for search results
    - Detection of AJAX-based pagination
    - Support for infinite scroll and incremental loading
    """
    
    def __init__(self):
        """Initialize the AJAX handler."""
        self.logger = logging.getLogger("AJAXHandler")
        # Track observed network requests
        self.observed_requests = []
        # Track ajax response content
        self.ajax_responses = []
        # Register common AJAX patterns for detection
        self.ajax_patterns = [
            r'\.json(\?|$)', 
            r'/api/',
            r'/graphql',
            r'/search',
            r'/query',
            r'/fetch',
            r'/ajax',
            r'/data'
        ]
    
    async def setup_network_monitoring(self, page):
        """
        Set up network monitoring to track AJAX requests and responses.
        
        Args:
            page: Playwright page object
        """
        self.observed_requests = []
        self.ajax_responses = []
        
        # Track requests
        async def on_request(request):
            if request.resource_type in ['xhr', 'fetch']:
                request_data = {
                    'url': request.url,
                    'method': request.method,
                    'time': time.time(),
                    'resource_type': request.resource_type,
                    'is_navigation': request.is_navigation_request(),
                }
                self.observed_requests.append(request_data)
        
        # Track responses
        async def on_response(response):
            try:
                # Only track XHR/fetch responses
                if response.request.resource_type in ['xhr', 'fetch']:
                    # Try to get response as JSON
                    try:
                        response_json = await response.json()
                        response_data = {
                            'url': response.url,
                            'status': response.status,
                            'time': time.time(),
                            'content_type': response.headers.get('content-type', ''),
                            'json_data': response_json,
                            'is_json': True
                        }
                        self.ajax_responses.append(response_data)
                    except:
                        # Not JSON, get as text if reasonable size
                        try:
                            if 'json' not in response.headers.get('content-type', ''):
                                # Skip non-JSON responses for efficiency
                                return
                                
                            text = await response.text()
                            # Only store reasonably sized responses
                            if len(text) < 1000000:  # 1MB limit
                                response_data = {
                                    'url': response.url,
                                    'status': response.status,
                                    'time': time.time(),
                                    'content_type': response.headers.get('content-type', ''),
                                    'text_data': text,
                                    'is_json': False
                                }
                                self.ajax_responses.append(response_data)
                        except:
                            # Skip responses we can't process
                            pass
            except Exception as e:
                self.logger.warning(f"Error processing response: {str(e)}")
        
        # Register listeners
        page.on('request', on_request)
        page.on('response', on_response)
        
        # Return unregister function for cleanup
        def unregister():
            page.remove_listener('request', on_request)
            page.remove_listener('response', on_response)
        
        return unregister
        
    async def wait_for_ajax_complete(self, page, timeout=5000, min_stable_period=500, additional_wait=1000):
        """
        Wait for AJAX requests to complete with an intelligent waiting strategy.
        
        Args:
            page: Playwright page object
            timeout: Maximum time to wait in ms
            min_stable_period: Minimum time with no activity to consider complete in ms
            additional_wait: Extra time to wait after stability in ms
            
        Returns:
            Boolean indicating whether the wait completed successfully
        """
        self.logger.info(f"Waiting for AJAX requests to complete (timeout: {timeout}ms)")
        
        try:
            # Setup network monitoring
            unregister = await self.setup_network_monitoring(page)
            
            # Convert ms to seconds for asyncio
            timeout_sec = timeout / 1000
            min_stable_sec = min_stable_period / 1000
            additional_wait_sec = additional_wait / 1000
            
            start_time = time.time()
            last_activity_time = start_time
            network_idle = False
            
            # Monitor for network activity
            while time.time() - start_time < timeout_sec:
                # Check if we have new requests in the last check period
                current_time = time.time()
                recent_requests = [r for r in self.observed_requests if r['time'] > last_activity_time]
                
                if recent_requests:
                    # Still have activity
                    last_activity_time = current_time
                    network_idle = False
                    await asyncio.sleep(0.1)  # Short sleep to prevent CPU spinning
                else:
                    if not network_idle:
                        # Just became idle
                        network_idle = True
                        network_idle_start = current_time
                        
                    # Check if we've been idle long enough
                    if current_time - network_idle_start >= min_stable_sec:
                        # Network has been idle for min_stable_period
                        # Add a small additional wait for any delayed processing
                        await asyncio.sleep(additional_wait_sec)
                        break
                    
                    await asyncio.sleep(0.1)
            
            # Clean up listeners
            unregister()
            
            # Return success if we didn't time out
            success = time.time() - start_time < timeout_sec
            self.logger.info(f"AJAX wait {'completed' if success else 'timed out'} after {round((time.time() - start_time) * 1000)}ms")
            return success
            
        except Exception as e:
            self.logger.error(f"Error waiting for AJAX: {str(e)}")
            return False
            
    async def wait_for_element_with_ajax(self, page, selector, timeout=10000):
        """
        Wait for an element to appear, accounting for AJAX loading.
        
        Args:
            page: Playwright page object
            selector: CSS selector for the element to wait for
            timeout: Maximum time to wait in ms
            
        Returns:
            Boolean indicating whether the element appeared
        """
        self.logger.info(f"Waiting for element '{selector}' with AJAX awareness (timeout: {timeout}ms)")
        
        try:
            # Setup network monitoring
            unregister = await self.setup_network_monitoring(page)
            
            start_time = time.time()
            timeout_sec = timeout / 1000
            element_found = False
            
            # Try to locate the element immediately
            try:
                element = await page.wait_for_selector(selector, timeout=100)
                if element:
                    element_found = True
                    self.logger.info(f"Element '{selector}' found immediately")
            except:
                # Element not found immediately, will wait with AJAX awareness
                pass
            
            # If not found, wait with AJAX awareness
            if not element_found:
                while time.time() - start_time < timeout_sec:
                    # Check for element
                    try:
                        element = await page.wait_for_selector(selector, timeout=500)
                        if element:
                            element_found = True
                            self.logger.info(f"Element '{selector}' found after waiting")
                            break
                    except:
                        # Element still not found
                        pass
                    
                    # If we observed AJAX activity, wait for it to complete
                    if self.observed_requests:
                        recent_requests = [r for r in self.observed_requests 
                                          if r['time'] > time.time() - 2]  # Last 2 seconds
                        if recent_requests:
                            # Wait for current AJAX to complete
                            await self.wait_for_ajax_complete(page, timeout=3000)
                    
                    # Short pause before trying again
                    await asyncio.sleep(0.5)
            
            # Clean up listeners
            unregister()
            
            return element_found
            
        except Exception as e:
            self.logger.error(f"Error in wait_for_element_with_ajax: {str(e)}")
            return False
    
    async def wait_for_condition(self, page, condition_fn, timeout=10000, check_interval=100):
        """
        Wait for a custom condition to be true, with awareness of AJAX activity.
        
        Args:
            page: Playwright page object
            condition_fn: Async function that returns True when condition is met
            timeout: Maximum time to wait in ms
            check_interval: Interval between condition checks in ms
            
        Returns:
            Boolean indicating whether the condition was met
        """
        self.logger.info(f"Waiting for custom condition with AJAX awareness (timeout: {timeout}ms)")
        
        try:
            # Setup network monitoring
            unregister = await self.setup_network_monitoring(page)
            
            start_time = time.time()
            timeout_sec = timeout / 1000
            check_interval_sec = check_interval / 1000
            
            while time.time() - start_time < timeout_sec:
                # Check condition
                try:
                    condition_met = await condition_fn(page)
                    if condition_met:
                        self.logger.info("Custom condition met")
                        break
                except Exception as e:
                    self.logger.warning(f"Error checking condition: {str(e)}")
                
                # If we observed AJAX activity, wait for it to settle
                recent_requests = [r for r in self.observed_requests 
                                  if r['time'] > time.time() - 2]  # Last 2 seconds
                if recent_requests:
                    # Brief wait for AJAX to progress
                    await asyncio.sleep(check_interval_sec)
                else:
                    # No recent activity, wait a bit longer before next check
                    await asyncio.sleep(check_interval_sec * 2)
            
            # Clean up listeners
            unregister()
            
            # Check final condition
            condition_met = await condition_fn(page)
            
            # Return success if condition met
            success = condition_met
            self.logger.info(f"Condition wait {'succeeded' if success else 'timed out'} after {round((time.time() - start_time) * 1000)}ms")
            return success
            
        except Exception as e:
            self.logger.error(f"Error in wait_for_condition: {str(e)}")
            return False
    
    async def handle_ajax_search_results(self, page, search_term):
        """
        Handle and extract search results from AJAX responses.
        
        Args:
            page: Playwright page object
            search_term: Search term used
            
        Returns:
            Dictionary with search results extracted from AJAX responses
        """
        self.logger.info(f"Handling AJAX search results for term: {search_term}")
        
        try:
            # Wait for AJAX search results to load
            await self.wait_for_ajax_complete(page, timeout=8000)
            
            # Sort responses by time (newest first)
            recent_responses = sorted(self.ajax_responses, key=lambda x: x['time'], reverse=True)
            
            # Initialize result container
            result = {
                'search_term': search_term,
                'found_results': False,
                'result_count': 0,
                'results': [],
                'response_url': None
            }
            
            # Look for search results in JSON responses
            for response in recent_responses:
                if response.get('is_json', False):
                    json_data = response.get('json_data', {})
                    
                    # Search for result arrays in the response
                    result_candidates = self._find_result_arrays(json_data)
                    
                    for path, array in result_candidates:
                        # Skip empty arrays
                        if not array:
                            continue
                            
                        # Check if this looks like search results
                        relevance_score = self._score_result_relevance(array, search_term)
                        
                        if relevance_score > 10:
                            # Found potential search results
                            extracted_results = self._extract_results(array)
                            
                            if extracted_results:
                                result['found_results'] = True
                                result['result_count'] = len(array)
                                result['results'] = extracted_results
                                result['response_url'] = response['url']
                                # Only use the first good result set
                                break
                
                # If we found results, stop processing
                if result['found_results']:
                    break
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error handling AJAX search results: {str(e)}")
            return {
                'search_term': search_term,
                'found_results': False,
                'error': str(e)
            }
    
    def _find_result_arrays(self, data, current_path=None, max_depth=5):
        """Recursively find arrays in the data that might contain results"""
        if current_path is None:
            current_path = []
            
        # Stop at max depth to prevent stack overflow
        if len(current_path) >= max_depth:
            return []
            
        results = []
        
        if isinstance(data, list) and data:
            # This is an array - might be results
            results.append((current_path, data))
            
            # Also check for nested arrays in the first item if appropriate
            if data and isinstance(data[0], (dict, list)):
                child_results = self._find_result_arrays(data[0], current_path + ['[0]'], max_depth)
                results.extend(child_results)
                
        elif isinstance(data, dict):
            # Look for keys that might indicate result arrays
            result_key_indicators = [
                'results', 'items', 'data', 'hits', 'documents', 'list',
                'properties', 'products', 'posts', 'articles', 'content',
                'collection', 'search'
            ]
            
            for key, value in data.items():
                # Check for array values directly under keys that suggest results
                if isinstance(value, list) and any(indicator in key.lower() for indicator in result_key_indicators):
                    results.append((current_path + [key], value))
                
                # Recurse into this value
                child_results = self._find_result_arrays(value, current_path + [key], max_depth)
                results.extend(child_results)
                
        return results
    
    def _score_result_relevance(self, array, search_term):
        """Score an array on how likely it is to contain search results"""
        if not array or not isinstance(array, list):
            return 0
            
        score = 0
        
        # Basic indicators that this might be results
        if len(array) > 0:
            score += 5  # Non-empty array
        if len(array) > 1:
            score += 5  # Multiple items
            
        # Check first item structure
        first_item = array[0] if array else None
        if isinstance(first_item, dict):
            # Items are objects - good sign
            score += 10
            
            # Check for common result fields
            field_indicators = [
                # Generic result fields
                'id', 'title', 'name', 'description', 'url', 'link',
                # Domain-specific fields
                'price', 'image', 'thumbnail', 'category', 'date',
                'author', 'location', 'rating', 'reviews'
            ]
            
            matched_fields = [field for field in field_indicators if field in first_item]
            score += len(matched_fields) * 2
            
            # Check if search term appears in values
            search_term_lower = search_term.lower()
            for key, value in first_item.items():
                if isinstance(value, str) and search_term_lower in value.lower():
                    score += 15
                    break
                    
        return score
    
    def _extract_results(self, items, max_results=20):
        """Extract useful information from result items"""
        results = []
        
        # Limit to reasonable number
        items = items[:max_results]
        
        for item in items:
            if not isinstance(item, dict):
                continue
                
            # Extract common fields
            result = {}
            
            # Field mappings for extraction
            field_mappings = {
                'title': ['title', 'name', 'product_name', 'headline', 'subject'],
                'description': ['description', 'summary', 'content', 'text', 'snippet', 'abstract'],
                'url': ['url', 'link', 'href', 'uri', 'web_url', 'product_url'],
                'id': ['id', 'uid', 'item_id', 'product_id'],
                'image': ['image', 'thumbnail', 'photo', 'picture', 'image_url'],
                'price': ['price', 'cost', 'amount', 'value', 'price_amount']
            }
            
            # Extract fields using mappings
            for target_field, source_fields in field_mappings.items():
                for field in source_fields:
                    if field in item:
                        result[target_field] = item[field]
                        break
            
            # Include result if we extracted anything useful
            if result:
                results.append(result)
                
        return results
    
    async def detect_infinite_scroll(self, page, timeout=8000):
        """
        Detect if a page uses infinite scroll for loading more results.
        
        Args:
            page: Playwright page object
            timeout: Maximum time to check in ms
            
        Returns:
            Dict with detection results
        """
        self.logger.info("Detecting infinite scroll behavior")
        
        try:
            # Setup network monitoring
            unregister = await self.setup_network_monitoring(page)
            
            # Get initial page height and scroll position
            initial_height = await page.evaluate("document.body.scrollHeight")
            
            # Scroll to 80% of page height
            await page.evaluate(f"window.scrollTo(0, {initial_height * 0.8})")
            
            # Wait briefly
            await asyncio.sleep(1)
            
            # Check if page height increased or new XHR requests were triggered
            new_height = await page.evaluate("document.body.scrollHeight")
            height_increased = new_height > initial_height
            
            # Check for AJAX activity related to content loading
            recent_requests = [r for r in self.observed_requests 
                              if r['time'] > time.time() - 2]  # Last 2 seconds
            
            ajax_triggered = any(self._is_content_request(req['url']) for req in recent_requests)
            
            # Clean up listeners
            unregister()
            
            is_infinite_scroll = height_increased and ajax_triggered
            
            return {
                'has_infinite_scroll': is_infinite_scroll,
                'height_changed': height_increased,
                'ajax_triggered': ajax_triggered,
                'initial_height': initial_height,
                'new_height': new_height
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting infinite scroll: {str(e)}")
            return {'has_infinite_scroll': False, 'error': str(e)}
    
    def _is_content_request(self, url):
        """Check if a URL is likely for content loading"""
        # Common patterns for content loading requests
        url_lower = url.lower()
        content_patterns = [
            'page=', 'offset=', 'limit=', 'from=', 'size=',  # Pagination params
            'next', 'more', 'additional', 'load', 'scroll',  # Content loading keywords
            '/api/', '/data/', '/content/', '/items/', '/products/',  # API paths
            'results', 'listing', 'feed', 'stream'  # Content types
        ]
        
        return any(pattern in url_lower for pattern in content_patterns)
    
    async def handle_load_more_button(self, page, max_clicks=3):
        """
        Detect and click on 'load more' buttons to reveal additional content.
        
        Args:
            page: Playwright page object
            max_clicks: Maximum number of times to click the button
            
        Returns:
            Dict with results of the operation
        """
        self.logger.info(f"Looking for 'load more' buttons (max clicks: {max_clicks})")
        
        try:
            # Common selectors for "load more" buttons
            load_more_selectors = [
                'button:has-text("Load more")',
                'button:has-text("Show more")',
                'a:has-text("Load more")',
                'a:has-text("Show more")',
                '.load-more',
                '.show-more',
                '[data-testid="load-more"]',
                '[aria-label="Load more"]',
                '[class*="loadMore"]',
                '[class*="showMore"]',
                '[class*="pagination"] button',
                'button.more',
                'button.load'
            ]
            
            # Track clicks
            clicks = 0
            found_button = False
            
            # Try each selector
            for selector in load_more_selectors:
                # Check if selector exists
                button_count = await page.locator(selector).count()
                
                if button_count > 0:
                    found_button = True
                    self.logger.info(f"Found 'load more' button with selector: {selector}")
                    
                    # Click up to max_clicks times
                    while clicks < max_clicks:
                        # Setup monitoring before click
                        unregister = await self.setup_network_monitoring(page)
                        
                        # Click the button
                        try:
                            await page.click(selector)
                            clicks += 1
                            self.logger.info(f"Clicked 'load more' button ({clicks}/{max_clicks})")
                            
                            # Wait for AJAX to complete
                            await self.wait_for_ajax_complete(page, timeout=5000)
                            
                            # Clean up listeners
                            unregister()
                            
                            # Check if button is still visible
                            still_visible = await page.locator(selector).is_visible()
                            if not still_visible:
                                self.logger.info("Button no longer visible, stopping clicks")
                                break
                                
                            # Wait briefly before next click
                            await asyncio.sleep(1)
                            
                        except Exception as e:
                            self.logger.warning(f"Error clicking button: {str(e)}")
                            unregister()  # Ensure cleanup
                            break
                    
                    # If we found and clicked a button, stop checking other selectors
                    if clicks > 0:
                        break
            
            return {
                'found_button': found_button,
                'clicks_performed': clicks,
                'max_clicks': max_clicks,
                'more_content_loaded': clicks > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error handling load more button: {str(e)}")
            return {'found_button': False, 'error': str(e)}
    
    async def wait_for_search_results(self, page, search_term, timeout=10000):
        """
        Wait for search results to appear, considering both DOM and AJAX updates.
        
        Args:
            page: Playwright page object
            search_term: Search term used
            timeout: Maximum time to wait in ms
            
        Returns:
            Dict with results of the waiting operation
        """
        self.logger.info(f"Waiting for search results for term: {search_term} (timeout: {timeout}ms)")
        
        try:
            # Setup network monitoring
            unregister = await self.setup_network_monitoring(page)
            
            # Common result container selectors
            result_selectors = [
                '.search-results', '.results', '#search-results', '#results',
                '[data-testid="search-results"]', '[aria-label="Search results"]',
                '.product-list', '.product-grid', '.items', '.listings',
                'article', '.article', '.post', '.card'
            ]
            
            # Wait for any of the selectors to appear
            start_time = time.time()
            timeout_sec = timeout / 1000
            results_found = False
            matching_selector = None
            
            while time.time() - start_time < timeout_sec:
                # Check each selector
                for selector in result_selectors:
                    try:
                        # Check if selector exists and is visible
                        is_visible = await page.locator(selector).first.is_visible()
                        if is_visible:
                            results_found = True
                            matching_selector = selector
                            self.logger.info(f"Found search results with selector: {selector}")
                            break
                    except:
                        # Selector not found, continue to next
                        pass
                
                # If results found, break the loop
                if results_found:
                    break
                
                # If we have seen AJAX responses, check them for results
                if self.ajax_responses:
                    # Check the most recent response
                    recent_responses = sorted(self.ajax_responses, key=lambda x: x['time'], reverse=True)
                    for response in recent_responses:
                        if response.get('is_json', False):
                            json_data = response.get('json_data', {})
                            result_candidates = self._find_result_arrays(json_data)
                            
                            for path, array in result_candidates:
                                if array and len(array) > 0:
                                    relevance_score = self._score_result_relevance(array, search_term)
                                    if relevance_score > 10:
                                        results_found = True
                                        matching_selector = "AJAX_RESPONSE"
                                        self.logger.info("Found search results in AJAX response")
                                        break
                        
                        if results_found:
                            break
                
                # If results found from AJAX, break the loop
                if results_found:
                    break
                
                # Wait briefly before checking again
                await asyncio.sleep(0.3)
            
            # Clean up listeners
            unregister()
            
            # Final check for results
            if not results_found:
                # Try once more with longer selectors list
                additional_selectors = [
                    'ul li', '.row', '.grid', '.list', '.collection', 
                    'div[class*="item"]', 'div[class*="product"]', 'div[class*="result"]'
                ]
                
                for selector in additional_selectors:
                    try:
                        count = await page.locator(selector).count()
                        if count > 3:  # Multiple items usually indicate results
                            results_found = True
                            matching_selector = selector
                            self.logger.info(f"Found likely search results with selector: {selector}")
                            break
                    except:
                        pass
            
            search_time = (time.time() - start_time) * 1000
            self.logger.info(f"Search results wait {'completed' if results_found else 'timed out'} after {round(search_time)}ms")
            
            return {
                'results_found': results_found,
                'search_time_ms': round(search_time),
                'matching_selector': matching_selector,
                'is_ajax_result': matching_selector == "AJAX_RESPONSE"
            }
            
        except Exception as e:
            self.logger.error(f"Error waiting for search results: {str(e)}")
            return {'results_found': False, 'error': str(e)}