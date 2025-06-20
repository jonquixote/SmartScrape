"""
Autocomplete and typeahead interfaces for SmartScrape search automation.

This module handles interaction with dynamic search interfaces that provide
suggestions as users type, including autocomplete, typeahead, and search
suggestion components.
"""

import logging
import asyncio
import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from urllib.parse import urlencode, urlparse, parse_qs

from bs4 import BeautifulSoup
from playwright.async_api import Page, Request, Response, ElementHandle

from utils.network_utils import extract_api_pattern


class AutocompleteHandler:
    """
    Handles interaction with autocomplete and typeahead interfaces.
    
    This class:
    - Detects autocomplete interfaces on a page
    - Monitors network requests to identify suggestion APIs
    - Interacts with autocomplete UI elements
    - Extracts suggestion data for further processing
    """
    
    def __init__(self):
        """Initialize the autocomplete handler."""
        self.logger = logging.getLogger("AutocompleteHandler")
        
        # Network request patterns for common autocomplete APIs
        self.common_api_patterns = [
            r'.*/(autocomplete|typeahead|suggest|search|query).*',
            r'.*/api/.*(suggest|search|autocomplete).*'
        ]
        
        # Common response data paths for suggestion data
        self.suggestion_data_paths = [
            ['suggestions'],
            ['results'],
            ['data', 'suggestions'],
            ['data', 'results'],
            ['response', 'suggestions'],
            ['response', 'results'],
            ['items'],
            ['matches'],
        ]
    
    async def detect_autocomplete(self, page: Page) -> Dict[str, Any]:
        """
        Detect autocomplete interfaces on a page.
        
        Args:
            page: Page object for browser interaction
            
        Returns:
            Dictionary with autocomplete information
        """
        self.logger.info("Detecting autocomplete interfaces")
        
        # Detect search inputs
        search_inputs = await self._detect_search_inputs(page)
        
        # For each input, check if it has autocomplete functionality
        autocomplete_results = {}
        for input_id, input_info in search_inputs.items():
            self.logger.debug(f"Checking for autocomplete on input: {input_id}")
            
            # Skip inputs that are unlikely to be search
            if input_info.get("confidence", 0) < 0.3:
                continue
                
            # Test this input for autocomplete behavior
            autocomplete_info = await self._test_autocomplete_input(page, input_info)
            
            if autocomplete_info.get("has_autocomplete", False):
                autocomplete_results[input_id] = autocomplete_info
                
        self.logger.info(f"Found {len(autocomplete_results)} autocomplete interfaces")
        
        return {
            "has_autocomplete": len(autocomplete_results) > 0,
            "autocomplete_inputs": autocomplete_results
        }
    
    async def _detect_search_inputs(self, page: Page) -> Dict[str, Dict[str, Any]]:
        """
        Detect search inputs on a page.
        
        Args:
            page: Page object for browser interaction
            
        Returns:
            Dictionary of search input information
        """
        return await page.evaluate('''
            () => {
                const inputs = {};
                
                // Find all input elements
                document.querySelectorAll('input, textarea, [contenteditable="true"]').forEach(el => {
                    // Skip hidden inputs
                    if (el.type === 'hidden' || 
                        el.style.display === 'none' || 
                        el.style.visibility === 'hidden' ||
                        el.offsetParent === null) {
                        return;
                    }
                    
                    // Compute a confidence score that this is a search input
                    let confidence = 0;
                    
                    // Check element attributes
                    if (el.type === 'search') confidence += 0.5;
                    if (el.name && el.name.toLowerCase().includes('search')) confidence += 0.4;
                    if (el.id && el.id.toLowerCase().includes('search')) confidence += 0.4;
                    if (el.placeholder && 
                        (el.placeholder.toLowerCase().includes('search') || 
                         el.placeholder.toLowerCase().includes('find'))) {
                        confidence += 0.4;
                    }
                    
                    // Check for autocomplete attributes
                    if (el.getAttribute('autocomplete') === 'off') confidence += 0.2;
                    if (el.getAttribute('aria-autocomplete') === 'list') confidence += 0.3;
                    if (el.getAttribute('role') === 'combobox') confidence += 0.3;
                    if (el.getAttribute('aria-expanded') !== null) confidence += 0.2;
                    if (el.getAttribute('aria-owns') || el.getAttribute('aria-controls')) confidence += 0.3;
                    
                    // Check if inside a form
                    const parentForm = el.closest('form');
                    if (parentForm) {
                        // Check form attributes
                        if (parentForm.id && parentForm.id.toLowerCase().includes('search')) confidence += 0.3;
                        if (parentForm.className && parentForm.className.toLowerCase().includes('search')) confidence += 0.3;
                        if (parentForm.action && parentForm.action.toLowerCase().includes('search')) confidence += 0.3;
                        
                        // Check for search button in the form
                        const searchButton = parentForm.querySelector('button[type="submit"], input[type="submit"]');
                        if (searchButton) {
                            const buttonText = searchButton.innerText || searchButton.value || '';
                            if (buttonText.toLowerCase().includes('search') || 
                                buttonText.toLowerCase().includes('find')) {
                                confidence += 0.4;
                            }
                            
                            // Check if button has a search icon
                            if (searchButton.querySelector('i.fa-search, svg, img[src*="search"]')) {
                                confidence += 0.3;
                            }
                        }
                    }
                    
                    // Check surrounding context
                    const parentEl = el.parentElement;
                    if (parentEl) {
                        // Check parent classes and IDs
                        if (parentEl.id && parentEl.id.toLowerCase().includes('search')) confidence += 0.2;
                        if (parentEl.className && parentEl.className.toLowerCase().includes('search')) confidence += 0.2;
                        
                        // Check for search icons nearby
                        if (parentEl.querySelector('i.fa-search, svg, img[src*="search"]')) {
                            confidence += 0.3;
                        }
                        
                        // Check for "Search" text nearby
                        const nearbyText = parentEl.innerText || '';
                        if (nearbyText.toLowerCase().includes('search') || 
                            nearbyText.toLowerCase().includes('find')) {
                            confidence += 0.2;
                        }
                    }
                    
                    // Only add if confidence exceeds threshold
                    if (confidence > 0.1) {
                        const id = el.id || ('input_' + Math.random().toString(36).substr(2, 9));
                        inputs[id] = {
                            element_type: el.tagName.toLowerCase(),
                            id: el.id || null,
                            name: el.name || null,
                            type: el.type || null,
                            placeholder: el.placeholder || null,
                            selector: el.id ? `#${el.id}` : null,
                            confidence: confidence,
                            location: {
                                x: el.getBoundingClientRect().left,
                                y: el.getBoundingClientRect().top,
                                visible: el.getBoundingClientRect().height > 0 && el.getBoundingClientRect().width > 0
                            }
                        };
                        
                        // Add CSS selector if ID not available
                        if (!el.id) {
                            // Try to build a reliable selector
                            const path = [];
                            let currentEl = el;
                            
                            while (currentEl && currentEl !== document.body) {
                                let selector = currentEl.tagName.toLowerCase();
                                
                                if (currentEl.id) {
                                    selector = `${selector}#${currentEl.id}`;
                                    path.unshift(selector);
                                    break;
                                } else if (currentEl.className) {
                                    const classes = currentEl.className.split(' ')
                                        .filter(c => c.trim() !== '')
                                        .map(c => c.trim());
                                    
                                    if (classes.length > 0) {
                                        selector += '.' + classes.join('.');
                                    }
                                }
                                
                                // Add position among siblings if needed
                                const siblings = Array.from(currentEl.parentElement.children)
                                    .filter(el => el.tagName === currentEl.tagName);
                                
                                if (siblings.length > 1) {
                                    const index = siblings.indexOf(currentEl);
                                    selector += `:nth-of-type(${index + 1})`;
                                }
                                
                                path.unshift(selector);
                                currentEl = currentEl.parentElement;
                            }
                            
                            inputs[id].selector = path.join(' > ');
                        }
                    }
                });
                
                return inputs;
            }
        ''')
    
    async def _test_autocomplete_input(self, page: Page, input_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test if an input has autocomplete functionality.
        
        Args:
            page: Page object for browser interaction
            input_info: Information about the input to test
            
        Returns:
            Dictionary with autocomplete information
        """
        selector = input_info.get("selector")
        if not selector:
            return {"has_autocomplete": False, "reason": "No selector available"}
            
        try:
            # Check if the input element exists and is interactable
            is_visible = await page.evaluate(f'''
                () => {{
                    const el = document.querySelector("{selector}");
                    if (!el) return false;
                    const rect = el.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0 && 
                           el.style.display !== 'none' && 
                           el.style.visibility !== 'hidden';
                }}
            ''')
            
            if not is_visible:
                return {"has_autocomplete": False, "reason": "Input not visible/interactable"}
            
            # Set up network request monitoring before interacting
            detected_requests = []
            
            async def handle_request(request):
                url = request.url
                if any(re.match(pattern, url) for pattern in self.common_api_patterns):
                    method = request.method
                    headers = request.headers
                    post_data = request.post_data
                    
                    detected_requests.append({
                        "url": url,
                        "method": method,
                        "headers": dict(headers),
                        "post_data": post_data
                    })
            
            # Start monitoring network requests
            page.on("request", handle_request)
            
            # Track detected suggestion elements
            suggestion_container = None
            
            # Attempt to interact with the input
            try:
                # Clear the input and type a generic search term
                await page.click(selector)
                await page.fill(selector, "")
                await page.type(selector, "a", delay=100)  # Type slowly to trigger autocomplete
                
                # Wait for potential autocomplete results 
                await asyncio.sleep(1)
                
                # Check for visible suggestion containers
                suggestion_elements = await self._detect_suggestion_elements(page)
                
                # If suggestions visible, store container info
                if suggestion_elements.get("has_suggestions", False):
                    suggestion_container = suggestion_elements.get("container_info")
                    
                # Type another character to see if suggestions update
                await page.type(selector, "p", delay=100)
                await asyncio.sleep(1)
                
                # Check again for suggestions
                updated_suggestions = await self._detect_suggestion_elements(page)
                suggestions_updated = updated_suggestions.get("has_suggestions", False)
                
                # Clean up
                await page.fill(selector, "")
                
                # Stop monitoring requests
                page.remove_listener("request", handle_request)
                
                # Analyze results
                has_network_suggestions = len(detected_requests) > 0
                has_ui_suggestions = suggestion_container is not None
                
                result = {
                    "has_autocomplete": has_network_suggestions or has_ui_suggestions,
                    "network_requests": detected_requests if has_network_suggestions else [],
                    "suggestions_container": suggestion_container,
                    "suggestions_updated": suggestions_updated,
                    "detection_method": "network" if has_network_suggestions else "ui" if has_ui_suggestions else None
                }
                
                # If we have network requests, analyze them
                if has_network_suggestions:
                    api_patterns = extract_api_pattern(
                        [req["url"] for req in detected_requests]
                    )
                    
                    result["api_patterns"] = api_patterns
                
                return result
                
            except Exception as e:
                self.logger.warning(f"Error testing input {selector}: {str(e)}")
                return {"has_autocomplete": False, "reason": f"Error: {str(e)}"}
                
        except Exception as e:
            self.logger.warning(f"Error evaluating visibility of {selector}: {str(e)}")
            return {"has_autocomplete": False, "reason": f"Error: {str(e)}"}
    
    async def _detect_suggestion_elements(self, page: Page) -> Dict[str, Any]:
        """
        Detect autocomplete suggestion elements on the page.
        
        Args:
            page: Page object for browser interaction
            
        Returns:
            Dictionary with suggestion element information
        """
        return await page.evaluate('''
            () => {
                // Look for visible elements that might be suggestion containers
                const possibleContainers = [
                    // Common autocomplete container selectors
                    '.autocomplete-suggestions',
                    '.autocomplete-results',
                    '.typeahead-results',
                    '.search-suggestions',
                    '.search-results',
                    '[role="listbox"]',
                    '.dropdown-menu:not(.hidden):not([style*="display: none"])',
                    '.suggestions',
                    '.results',
                    '.ui-autocomplete',
                    '.ui-menu',
                    '.tt-menu'
                ];
                
                let container = null;
                
                // Check each potential container
                for (const selector of possibleContainers) {
                    const elements = document.querySelectorAll(selector);
                    
                    for (const el of elements) {
                        // Check if visible
                        const rect = el.getBoundingClientRect();
                        const isVisible = rect.width > 0 && 
                                         rect.height > 0 && 
                                         window.getComputedStyle(el).display !== 'none' && 
                                         window.getComputedStyle(el).visibility !== 'hidden';
                        
                        if (isVisible) {
                            // Check if it contains list items
                            const items = el.querySelectorAll('li, .item, .suggestion, .result, [role="option"]');
                            
                            if (items.length > 0) {
                                const itemTexts = Array.from(items).map(item => item.innerText.trim())
                                    .filter(text => text.length > 0);
                                
                                container = {
                                    selector: selector,
                                    item_count: items.length,
                                    item_texts: itemTexts.slice(0, 5), // First 5 items for sample
                                    position: {
                                        x: rect.left,
                                        y: rect.top,
                                        width: rect.width,
                                        height: rect.height
                                    }
                                };
                                
                                break;
                            }
                        }
                    }
                    
                    if (container) break;
                }
                
                // If no container found with predefined selectors, look for any newly visible element
                if (!container) {
                    // This would work better with before/after snapshots, but we'll do a basic check
                    document.querySelectorAll('div, ul, ol').forEach(el => {
                        if (el.children.length < 2) return; // Skip containers with too few children
                        
                        const rect = el.getBoundingClientRect();
                        const isVisible = rect.width > 0 && 
                                         rect.height > 0 && 
                                         window.getComputedStyle(el).display !== 'none' && 
                                         window.getComputedStyle(el).visibility !== 'hidden';
                                         
                        if (isVisible && rect.top > 0) { // Only consider elements below the viewport top
                            // Check if it looks like a list of suggestions
                            const childrenSimilar = Array.from(el.children).every(child => 
                                child.tagName === el.children[0].tagName);
                                
                            if (childrenSimilar && el.children.length >= 2) {
                                // Build a selector for this element
                                let selector = el.tagName.toLowerCase();
                                if (el.id) {
                                    selector += `#${el.id}`;
                                } else if (el.className) {
                                    const classes = el.className.split(' ')
                                        .filter(c => c.trim() !== '')
                                        .map(c => c.trim());
                                    
                                    if (classes.length > 0) {
                                        selector += '.' + classes.join('.');
                                    }
                                }
                                
                                const itemTexts = Array.from(el.children).map(child => child.innerText.trim())
                                    .filter(text => text.length > 0);
                                
                                container = {
                                    selector: selector,
                                    item_count: el.children.length,
                                    item_texts: itemTexts.slice(0, 5), // First 5 items for sample
                                    position: {
                                        x: rect.left,
                                        y: rect.top,
                                        width: rect.width,
                                        height: rect.height
                                    },
                                    detection_method: "heuristic"
                                };
                            }
                        }
                    });
                }
                
                return {
                    has_suggestions: container !== null,
                    container_info: container
                };
            }
        ''')
    
    async def interact_with_autocomplete(self, page: Page, input_info: Dict[str, Any], 
                                       search_term: str, 
                                       select_suggestion: bool = True) -> Dict[str, Any]:
        """
        Interact with an autocomplete interface.
        
        Args:
            page: Page object for browser interaction
            input_info: Information about the autocomplete input
            search_term: Search term to type
            select_suggestion: Whether to select a suggestion
            
        Returns:
            Dictionary with interaction results
        """
        self.logger.info(f"Interacting with autocomplete for search term: {search_term}")
        
        selector = input_info.get("selector")
        if not selector:
            return {"success": False, "reason": "No selector available"}
            
        # Results storage
        results = {
            "success": False,
            "search_term": search_term,
            "suggestions": [],
            "selected_suggestion": None,
            "api_requests": []
        }
        
        # Monitor network requests
        detected_requests = []
        detected_responses = []
        
        async def handle_request(request):
            url = request.url
            if any(re.match(pattern, url) for pattern in self.common_api_patterns):
                method = request.method
                headers = request.headers
                post_data = request.post_data
                
                req_data = {
                    "url": url,
                    "method": method,
                    "headers": dict(headers),
                    "post_data": post_data
                }
                
                detected_requests.append(req_data)
        
        async def handle_response(response):
            request = response.request
            url = request.url
            
            if any(re.match(pattern, url) for pattern in self.common_api_patterns):
                try:
                    # Try to get response as JSON
                    response_body = None
                    try:
                        response_body = await response.json()
                    except:
                        try:
                            text = await response.text()
                            response_body = json.loads(text)
                        except:
                            response_body = await response.text()
                    
                    resp_data = {
                        "url": url,
                        "status": response.status,
                        "headers": dict(response.headers),
                        "body": response_body
                    }
                    
                    detected_responses.append(resp_data)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing response: {str(e)}")
        
        # Start monitoring network
        page.on("request", handle_request)
        page.on("response", handle_response)
        
        try:
            # Focus and clear the input
            await page.click(selector)
            await page.fill(selector, "")
            
            # Type the search term character by character
            for char in search_term:
                await page.type(selector, char, delay=100)  # Type slowly to trigger suggestions
                await asyncio.sleep(0.1)
            
            # Wait for suggestions to appear
            await asyncio.sleep(1)
            
            # Detect suggestions
            suggestion_elements = await self._detect_suggestion_elements(page)
            
            if suggestion_elements.get("has_suggestions", False):
                suggestions_container = suggestion_elements.get("container_info", {})
                suggestion_texts = suggestions_container.get("item_texts", [])
                
                results["suggestions"] = suggestion_texts
                results["suggestions_container"] = suggestions_container
                
                # Process API responses for more suggestion data
                api_suggestions = self._extract_suggestions_from_responses(detected_responses)
                if api_suggestions:
                    results["api_suggestions"] = api_suggestions
                    
                    # Merge UI and API suggestions if both available
                    all_suggestions = list(suggestion_texts)
                    for sugg in api_suggestions:
                        if isinstance(sugg, str) and sugg not in all_suggestions:
                            all_suggestions.append(sugg)
                        elif isinstance(sugg, dict) and sugg.get("text") not in all_suggestions:
                            all_suggestions.append(sugg.get("text"))
                    
                    results["all_suggestions"] = all_suggestions
                
                # Select a suggestion if requested
                if select_suggestion and suggestion_texts:
                    # Get container selector
                    container_selector = suggestions_container.get("selector")
                    
                    if container_selector:
                        # Try to click the first suggestion
                        suggestion_selector = f"{container_selector} li:first-child, {container_selector} [role='option']:first-child, {container_selector} .item:first-child, {container_selector} .suggestion:first-child"
                        
                        try:
                            await page.click(suggestion_selector)
                            results["selected_suggestion"] = suggestion_texts[0] if suggestion_texts else None
                        except Exception as e:
                            self.logger.warning(f"Error clicking suggestion: {str(e)}")
                            # Fallback: press down arrow and Enter
                            await page.keyboard.press("ArrowDown")
                            await asyncio.sleep(0.2)
                            await page.keyboard.press("Enter")
                            results["selected_suggestion"] = "Selected via keyboard"
            
            # Store network activity
            results["api_requests"] = detected_requests
            results["api_responses"] = [
                {k: v for k, v in resp.items() if k != 'body'} 
                for resp in detected_responses
            ]
            
            # Mark success
            results["success"] = True
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error interacting with autocomplete: {str(e)}")
            return {"success": False, "reason": f"Error: {str(e)}"}
            
        finally:
            # Stop monitoring network
            page.remove_listener("request", handle_request)
            page.remove_listener("response", handle_response)
    
    def _extract_suggestions_from_responses(self, responses: List[Dict[str, Any]]) -> List[Any]:
        """
        Extract suggestion data from API responses.
        
        Args:
            responses: List of API response data
            
        Returns:
            List of extracted suggestions
        """
        all_suggestions = []
        
        for response in responses:
            body = response.get("body")
            if not body:
                continue
                
            # Try to find suggestions in the response
            suggestions = self._find_suggestions_in_json(body)
            
            if suggestions:
                all_suggestions.extend(suggestions)
                
        return all_suggestions
    
    def _find_suggestions_in_json(self, data: Any, max_depth: int = 3) -> List[Any]:
        """
        Recursively find suggestions in JSON data.
        
        Args:
            data: JSON data to search
            max_depth: Maximum recursion depth
            
        Returns:
            List of extracted suggestions
        """
        if max_depth <= 0:
            return []
            
        # If data is a list, it might be suggestions directly
        if isinstance(data, list):
            # If list of strings, it's likely suggestions
            if all(isinstance(item, str) for item in data):
                return data
                
            # If list of dicts with text/value properties, extract those
            if all(isinstance(item, dict) for item in data):
                suggestion_keys = ['text', 'value', 'name', 'title', 'label', 'suggestion']
                
                # Check if all dicts have at least one suggestion key
                if all(any(key in item for key in suggestion_keys) for item in data):
                    return data
                    
                # Recursively check deeper
                suggestions = []
                for item in data:
                    suggestions.extend(self._find_suggestions_in_json(item, max_depth - 1))
                    
                return suggestions
                
        # If data is a dict, check common paths
        elif isinstance(data, dict):
            # Try common paths first
            for path in self.suggestion_data_paths:
                curr = data
                valid_path = True
                
                for key in path:
                    if key in curr:
                        curr = curr[key]
                    else:
                        valid_path = False
                        break
                        
                if valid_path and isinstance(curr, list):
                    return curr
            
            # Try all keys if common paths failed
            for key, value in data.items():
                if isinstance(value, list) and key.lower() in [
                    'suggestions', 'results', 'items', 'matches', 'options'
                ]:
                    return value
                    
                # Recursively check deeper
                suggestions = self._find_suggestions_in_json(value, max_depth - 1)
                if suggestions:
                    return suggestions
                    
        return []


class TypeaheadHandler:
    """
    Handles interaction with typeahead interfaces.
    
    This class specializes in handling advanced typeahead components like
    Twitter Typeahead, Select2, Chosen.js, and other rich autocomplete widgets.
    """
    
    def __init__(self):
        """Initialize the typeahead handler."""
        self.logger = logging.getLogger("TypeaheadHandler")
        self.autocomplete_handler = AutocompleteHandler()
        
        # Library-specific selectors
        self.library_selectors = {
            "twitter-typeahead": {
                "input": ".tt-input, .typeahead",
                "menu": ".tt-menu, .tt-dropdown-menu",
                "suggestion": ".tt-suggestion"
            },
            "select2": {
                "input": ".select2-search__field",
                "container": ".select2-container",
                "dropdown": ".select2-dropdown",
                "result": ".select2-results__option"
            },
            "chosen": {
                "container": ".chosen-container",
                "dropdown": ".chosen-drop",
                "search": ".chosen-search input",
                "result": ".chosen-results li"
            },
            "autocomplete-js": {
                "menu": ".autocomplete-suggestions",
                "suggestion": ".autocomplete-suggestion"
            },
            "jquery-ui": {
                "menu": ".ui-autocomplete",
                "item": ".ui-menu-item"
            }
        }
    
    async def detect_typeahead_libraries(self, page: Page) -> Dict[str, Any]:
        """
        Detect typeahead library usage on a page.
        
        Args:
            page: Page object for browser interaction
            
        Returns:
            Dictionary with detected libraries
        """
        self.logger.info("Detecting typeahead libraries")
        
        libraries = await page.evaluate('''
            () => {
                const libraries = {};
                
                // Check for library globals
                if (window.jQuery && window.jQuery.fn) {
                    if (window.jQuery.fn.typeahead) {
                        libraries['twitter-typeahead'] = {
                            source: 'jQuery plugin',
                            elements: document.querySelectorAll('.twitter-typeahead, .typeahead').length
                        };
                    }
                    
                    if (window.jQuery.fn.select2) {
                        libraries['select2'] = {
                            source: 'jQuery plugin',
                            elements: document.querySelectorAll('.select2, .select2-container').length
                        };
                    }
                    
                    if (window.jQuery.fn.chosen) {
                        libraries['chosen'] = {
                            source: 'jQuery plugin',
                            elements: document.querySelectorAll('.chosen-container').length
                        };
                    }
                    
                    if (window.jQuery.ui && window.jQuery.ui.autocomplete) {
                        libraries['jquery-ui'] = {
                            source: 'jQuery UI',
                            elements: document.querySelectorAll('.ui-autocomplete').length
                        };
                    }
                }
                
                // Check for library elements
                if (document.querySelectorAll('.twitter-typeahead, .tt-input').length > 0) {
                    libraries['twitter-typeahead'] = libraries['twitter-typeahead'] || {
                        source: 'DOM elements',
                        elements: document.querySelectorAll('.twitter-typeahead, .tt-input').length
                    };
                }
                
                if (document.querySelectorAll('.select2, .select2-container').length > 0) {
                    libraries['select2'] = libraries['select2'] || {
                        source: 'DOM elements',
                        elements: document.querySelectorAll('.select2, .select2-container').length
                    };
                }
                
                if (document.querySelectorAll('.chosen-container').length > 0) {
                    libraries['chosen'] = libraries['chosen'] || {
                        source: 'DOM elements',
                        elements: document.querySelectorAll('.chosen-container').length
                    };
                }
                
                if (document.querySelectorAll('.ui-autocomplete, .ui-menu').length > 0) {
                    libraries['jquery-ui'] = libraries['jquery-ui'] || {
                        source: 'DOM elements',
                        elements: document.querySelectorAll('.ui-autocomplete, .ui-menu').length
                    };
                }
                
                if (document.querySelectorAll('.autocomplete-suggestions').length > 0) {
                    libraries['autocomplete-js'] = {
                        source: 'DOM elements',
                        elements: document.querySelectorAll('.autocomplete-suggestions').length
                    };
                }
                
                // Check for script tags
                const scriptSrcs = Array.from(document.querySelectorAll('script'))
                    .map(script => script.src || '');
                
                if (scriptSrcs.some(src => src.includes('typeahead') || src.includes('bloodhound'))) {
                    libraries['twitter-typeahead'] = libraries['twitter-typeahead'] || {
                        source: 'Script tag',
                        elements: document.querySelectorAll('.twitter-typeahead, .tt-input').length
                    };
                }
                
                if (scriptSrcs.some(src => src.includes('select2'))) {
                    libraries['select2'] = libraries['select2'] || {
                        source: 'Script tag',
                        elements: document.querySelectorAll('.select2, .select2-container').length
                    };
                }
                
                if (scriptSrcs.some(src => src.includes('chosen'))) {
                    libraries['chosen'] = libraries['chosen'] || {
                        source: 'Script tag',
                        elements: document.querySelectorAll('.chosen-container').length
                    };
                }
                
                return libraries;
            }
        ''')
        
        self.logger.info(f"Detected typeahead libraries: {list(libraries.keys())}")
        
        return {
            "has_typeahead_libraries": len(libraries) > 0,
            "detected_libraries": libraries
        }
    
    async def interact_with_typeahead(self, page: Page, library_name: str, 
                                    search_term: str, selector: str = None,
                                    select_suggestion: bool = True) -> Dict[str, Any]:
        """
        Interact with a typeahead interface for a specific library.
        
        Args:
            page: Page object for browser interaction
            library_name: Name of the typeahead library
            search_term: Search term to type
            selector: Optional specific selector for the input
            select_suggestion: Whether to select a suggestion
            
        Returns:
            Dictionary with interaction results
        """
        self.logger.info(f"Interacting with {library_name} typeahead for search term: {search_term}")
        
        # Get library-specific selectors
        library_info = self.library_selectors.get(library_name)
        if not library_info:
            return {"success": False, "reason": f"Library '{library_name}' not supported"}
            
        # Determine input selector
        input_selector = selector if selector else library_info.get("input")
        if not input_selector:
            return {"success": False, "reason": "No input selector available"}
            
        try:
            # Check if input exists
            input_exists = await page.evaluate(f'''
                () => {{
                    return document.querySelector("{input_selector}") !== null;
                }}
            ''')
            
            if not input_exists:
                self.logger.warning(f"Input element not found with selector: {input_selector}")
                
                # Try to find the appropriate element based on library
                if library_name == "select2":
                    # For Select2, we might need to click the container first
                    container_selector = library_info.get("container")
                    container_exists = await page.evaluate(f'''
                        () => {{
                            return document.querySelector("{container_selector}") !== null;
                        }}
                    ''')
                    
                    if container_exists:
                        await page.click(container_selector)
                        await asyncio.sleep(0.5)
                        
                        # Try to find the input again
                        input_exists = await page.evaluate(f'''
                            () => {{
                                return document.querySelector("{input_selector}") !== null;
                            }}
                        ''')
                        
                        if not input_exists:
                            return {"success": False, "reason": "Input element not found after clicking container"}
                    else:
                        return {"success": False, "reason": "Neither input nor container found"}
                
                elif library_name == "chosen":
                    # For Chosen, we need to click the container to open the dropdown
                    container_selector = library_info.get("container")
                    container_exists = await page.evaluate(f'''
                        () => {{
                            return document.querySelector("{container_selector}") !== null;
                        }}
                    ''')
                    
                    if container_exists:
                        await page.click(container_selector)
                        await asyncio.sleep(0.5)
                        
                        # Look for the search input within the dropdown
                        search_selector = library_info.get("search")
                        input_exists = await page.evaluate(f'''
                            () => {{
                                return document.querySelector("{search_selector}") !== null;
                            }}
                        ''')
                        
                        if input_exists:
                            input_selector = search_selector
                        else:
                            return {"success": False, "reason": "Search input not found after opening dropdown"}
                    else:
                        return {"success": False, "reason": "Container element not found"}
                
                else:
                    return {"success": False, "reason": "Input element not found"}
            
            # Interact with the input
            await page.click(input_selector)
            await page.fill(input_selector, "")
            
            # Type the search term character by character
            for char in search_term:
                await page.type(input_selector, char, delay=100)
                await asyncio.sleep(0.1)
            
            # Wait for suggestions to appear
            await asyncio.sleep(1)
            
            # Get suggestion container and items based on library
            suggestion_container = library_info.get("menu") or library_info.get("dropdown")
            suggestion_item = library_info.get("suggestion") or library_info.get("result") or library_info.get("item")
            
            # Check for suggestions
            suggestions = await page.evaluate(f'''
                () => {{
                    const container = document.querySelector("{suggestion_container}");
                    if (!container) return [];
                    
                    const items = container.querySelectorAll("{suggestion_item}");
                    return Array.from(items).map(item => item.innerText.trim())
                        .filter(text => text.length > 0);
                }}
            ''')
            
            # Select a suggestion if requested
            selected_suggestion = None
            if select_suggestion and suggestions.length > 0:
                try:
                    first_item_selector = f"{suggestion_container} {suggestion_item}"
                    await page.click(first_item_selector)
                    selected_suggestion = suggestions[0]
                except Exception as e:
                    self.logger.warning(f"Error clicking suggestion: {str(e)}")
                    # Fallback: press down arrow and Enter
                    await page.keyboard.press("ArrowDown")
                    await asyncio.sleep(0.2)
                    await page.keyboard.press("Enter")
                    selected_suggestion = "Selected via keyboard"
            
            return {
                "success": True,
                "search_term": search_term,
                "suggestions": suggestions,
                "selected_suggestion": selected_suggestion,
                "library": library_name
            }
            
        except Exception as e:
            self.logger.error(f"Error interacting with typeahead: {str(e)}")
            return {"success": False, "reason": f"Error: {str(e)}"}


class SearchSuggestionAPI:
    """
    Handles interaction with search suggestion APIs.
    
    This class:
    - Identifies suggestion API endpoints
    - Constructs API requests for suggestions
    - Parses API responses for suggestion data
    - Provides a cache for suggestion API results
    """
    
    def __init__(self):
        """Initialize the search suggestion API handler."""
        self.logger = logging.getLogger("SearchSuggestionAPI")
        
        # Cache for API responses
        self.suggestion_cache = {}
        
        # Common API patterns
        self.api_patterns = [
            r'.*/(autocomplete|typeahead|suggest|search|query).*',
            r'.*/api/.*(suggest|search|autocomplete).*'
        ]
    
    async def extract_api_from_page(self, page: Page, test_term: str = "a") -> Optional[Dict[str, Any]]:
        """
        Extract suggestion API details from page interactions.
        
        Args:
            page: Page object for browser interaction
            test_term: Term to use for testing API detection
            
        Returns:
            Dictionary with API details or None if not found
        """
        self.logger.info("Extracting suggestion API from page")
        
        # Detect autocomplete inputs
        autocomplete_handler = AutocompleteHandler()
        autocomplete_info = await autocomplete_handler.detect_autocomplete(page)
        
        if not autocomplete_info.get("has_autocomplete", False):
            self.logger.info("No autocomplete interfaces detected")
            return None
            
        # Get the most promising autocomplete input
        autocomplete_inputs = autocomplete_info.get("autocomplete_inputs", {})
        if not autocomplete_inputs:
            return None
            
        best_input_id = max(
            autocomplete_inputs.keys(),
            key=lambda k: autocomplete_inputs[k].get("confidence", 0)
        )
        
        best_input = autocomplete_inputs[best_input_id]
        
        # Interact with the input to capture API requests
        interaction_result = await autocomplete_handler.interact_with_autocomplete(
            page, best_input, test_term, select_suggestion=False
        )
        
        if not interaction_result.get("success", False):
            self.logger.warning("Failed to interact with autocomplete")
            return None
            
        # Extract API information from requests
        api_requests = interaction_result.get("api_requests", [])
        if not api_requests:
            self.logger.info("No API requests detected during interaction")
            return None
            
        # Find the most promising API request
        best_request = self._find_best_api_request(api_requests, test_term)
        if not best_request:
            return None
            
        # Try to extract the API pattern
        url = best_request.get("url", "")
        parsed_url = urlparse(url)
        
        # Extract query parameters
        query_params = parse_qs(parsed_url.query)
        
        # Find the query parameter
        query_param = self._find_query_param(query_params, test_term)
        
        # Build API template
        api_template = self._build_api_template(url, query_param, test_term)
        
        return {
            "api_url": url,
            "api_template": api_template,
            "method": best_request.get("method", "GET"),
            "headers": best_request.get("headers", {}),
            "post_data": best_request.get("post_data"),
            "query_param": query_param,
            "path": parsed_url.path,
            "host": parsed_url.netloc
        }
    
    def _find_best_api_request(self, requests: List[Dict[str, Any]], test_term: str) -> Optional[Dict[str, Any]]:
        """
        Find the best API request from a list of requests.
        
        Args:
            requests: List of request data
            test_term: Test term used for interaction
            
        Returns:
            Best request or None if none found
        """
        if not requests:
            return None
            
        # Score each request based on likelihood of being a suggestion API
        scored_requests = []
        
        for req in requests:
            url = req.get("url", "")
            method = req.get("method", "").upper()
            score = 0
            
            # Prefer GET requests for suggestion APIs
            if method == "GET":
                score += 2
            elif method == "POST":
                score += 1
                
            # Check URL for suggestion-related terms
            url_lower = url.lower()
            for term in ["suggest", "autocomplete", "typeahead", "search", "query", "complete"]:
                if term in url_lower:
                    score += 3
                    
            # Check if URL contains the test term
            if test_term.lower() in url_lower:
                score += 5
                
            # Check query parameters for the test term
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            
            for param, values in query_params.items():
                param_lower = param.lower()
                
                # Preferred parameter names
                if param_lower in ["q", "query", "term", "search", "text", "input"]:
                    score += 2
                    
                # Check if param contains test term
                for value in values:
                    if test_term.lower() in value.lower():
                        score += 4
                        
            # Add to scored list
            scored_requests.append((score, req))
            
        # Sort by score (descending)
        scored_requests.sort(reverse=True, key=lambda x: x[0])
        
        # Return the highest scoring request if score > 0
        if scored_requests and scored_requests[0][0] > 0:
            return scored_requests[0][1]
            
        return None
    
    def _find_query_param(self, query_params: Dict[str, List[str]], test_term: str) -> Optional[str]:
        """
        Find the parameter that likely contains the search query.
        
        Args:
            query_params: Dictionary of query parameters
            test_term: Test term used for interaction
            
        Returns:
            Parameter name or None if not found
        """
        # First check if any param value contains the test term
        for param, values in query_params.items():
            for value in values:
                if test_term.lower() in value.lower():
                    return param
                    
        # If not found, try common parameter names
        common_params = ["q", "query", "term", "search", "text", "input", "keyword"]
        for param in common_params:
            if param in query_params:
                return param
                
        # If still not found, return first param or None
        return next(iter(query_params.keys())) if query_params else None
    
    def _build_api_template(self, url: str, query_param: Optional[str], test_term: str) -> str:
        """
        Build an API template URL with placeholder for search term.
        
        Args:
            url: Original URL
            query_param: Query parameter name
            test_term: Test term used in the original URL
            
        Returns:
            Template URL with {query} placeholder
        """
        if not query_param:
            return url
            
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        # Replace the value of query_param with a placeholder
        if query_param in query_params:
            query_params[query_param] = ["{query}"]
            
        # Rebuild the query string
        new_query = urlencode(query_params, doseq=True)
        
        # Rebuild the URL
        parts = list(parsed_url)
        parts[4] = new_query  # Replace query string
        
        return urlparse.urlunparse(parts)
    
    async def get_suggestions(self, page: Page, api_info: Dict[str, Any], 
                             query: str) -> List[Any]:
        """
        Get suggestions from a search suggestion API.
        
        Args:
            page: Page object for browser interaction
            api_info: API information
            query: Search query
            
        Returns:
            List of suggestions
        """
        # Check cache first
        cache_key = f"{api_info.get('api_template')}::{query}"
        if cache_key in self.suggestion_cache:
            return self.suggestion_cache[cache_key]
            
        # Prepare API request
        api_template = api_info.get("api_template")
        method = api_info.get("method", "GET")
        headers = api_info.get("headers", {})
        
        # Replace {query} in template
        url = api_template.replace("{query}", query)
        
        try:
            # Make the request
            if method.upper() == "GET":
                response = await page.evaluate(f'''
                    async () => {{
                        try {{
                            const response = await fetch("{url}", {{
                                method: "GET",
                                headers: {json.dumps(headers)}
                            }});
                            
                            const data = await response.json();
                            return data;
                        }} catch (error) {{
                            return {{ error: error.toString() }};
                        }}
                    }}
                ''')
            else:
                # For POST requests, we need to handle the body
                post_data = api_info.get("post_data", "")
                
                # Replace query in post data if it's a string
                if isinstance(post_data, str):
                    post_data = post_data.replace(api_info.get("query_param", ""), query)
                    
                response = await page.evaluate(f'''
                    async () => {{
                        try {{
                            const response = await fetch("{url}", {{
                                method: "POST",
                                headers: {json.dumps(headers)},
                                body: "{post_data}"
                            }});
                            
                            const data = await response.json();
                            return data;
                        }} catch (error) {{
                            return {{ error: error.toString() }};
                        }}
                    }}
                ''')
            
            # Check for error
            if response and "error" in response:
                self.logger.warning(f"Error fetching suggestions: {response['error']}")
                return []
                
            # Extract suggestions from response
            autocomplete_handler = AutocompleteHandler()
            suggestions = autocomplete_handler._find_suggestions_in_json(response)
            
            # Cache results
            self.suggestion_cache[cache_key] = suggestions
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error getting suggestions: {str(e)}")
            return []


# Register components in __init__.py
__all__ = [
    'AutocompleteHandler',
    'TypeaheadHandler',
    'SearchSuggestionAPI'
]