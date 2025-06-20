"""
Search coordinator module for SmartScrape.

This module orchestrates the various search components to provide a unified
search interface that combines multiple search strategies.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urlparse

# Import search components
from components.search.browser_interaction import BrowserInteraction
from components.search.ajax_handler import AJAXHandler
from components.search.form_detection import SearchFormDetector
from components.search.api_detection import APIParameterAnalyzer


class SearchCoordinator:
    """
    Coordinates search operations across multiple search strategies.
    
    This class serves as a component-level coordinator for all search-related functionality,
    managing the execution of various search strategies, consolidating results, and handling
    error recovery.
    """
    
    def __init__(self, config=None):
        """
        Initialize the search coordinator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger("SearchCoordinator")
        self.config = config or {}
        
        # Initialize component dependencies
        self.browser_interaction = BrowserInteraction(config=self.config)
        self.ajax_handler = AJAXHandler()
        self.form_detector = SearchFormDetector()
        self.api_analyzer = APIParameterAnalyzer()
        
        # Track active search operations
        self.active_search = None
        self.search_cancel_requested = False
        
    async def search(self, url: str, search_term: str, domain_type: str = "general",
                   max_retries: int = 2, preferred_method: str = None) -> Dict[str, Any]:
        """
        Perform a search operation using the most appropriate method.
        
        This method attempts various search strategies in order of preference and likelihood
        of success for the given website type.
        
        Args:
            url: Target website URL to search
            search_term: Search term to use
            domain_type: Type of domain (e.g., "e_commerce", "real_estate")
            max_retries: Maximum number of retry attempts
            preferred_method: Optional preferred search method
            
        Returns:
            Dictionary with search results and metadata
        """
        self.logger.info(f"Initiating search on {url} for term: '{search_term}'")
        self.search_cancel_requested = False
        
        # Reset active search
        self.active_search = {
            'url': url,
            'search_term': search_term,
            'domain_type': domain_type,
            'status': 'in_progress',
            'strategy_attempts': []
        }
        
        # Prioritize search strategies based on domain type and config
        strategies = self._prioritize_strategies(url, domain_type, preferred_method)
        
        # Initialize browser if needed for most strategies
        browser = await self.browser_interaction.initialize_browser()
        context = await self.browser_interaction.create_context(browser)
        page = await self.browser_interaction.new_page(context)
        
        # Track the best result
        best_result = None
        
        # Try each strategy in priority order
        for strategy_name in strategies:
            if self.search_cancel_requested:
                self.logger.info("Search cancelled by user request")
                await self.browser_interaction.close()
                return {
                    'success': False,
                    'cancelled': True,
                    'url': url,
                    'search_term': search_term
                }
            
            self.logger.info(f"Trying search strategy: {strategy_name}")
            
            # Record this attempt
            self.active_search['strategy_attempts'].append({
                'strategy': strategy_name,
                'status': 'in_progress'
            })
            
            try:
                result = await self._execute_strategy(
                    strategy_name, url, search_term, page, domain_type, max_retries
                )
                
                # Update attempt status
                current_attempt = self.active_search['strategy_attempts'][-1]
                current_attempt['status'] = 'success' if result.get('success') else 'failed'
                current_attempt['result'] = result
                
                # If successful, this is our result
                if result.get('success'):
                    best_result = result
                    break
                    
            except Exception as e:
                self.logger.error(f"Error executing {strategy_name} strategy: {str(e)}")
                # Update attempt status
                current_attempt = self.active_search['strategy_attempts'][-1]
                current_attempt['status'] = 'error'
                current_attempt['error'] = str(e)
        
        # Close browser resources
        await self.browser_interaction.close()
        
        # If we didn't find a successful result, return the last attempt
        if not best_result:
            last_attempt = self.active_search['strategy_attempts'][-1] if self.active_search['strategy_attempts'] else None
            
            result = {
                'success': False,
                'url': url,
                'search_term': search_term,
                'error': 'All search strategies failed'
            }
            
            if last_attempt and 'result' in last_attempt:
                # Include details from the last attempt
                result.update({
                    'last_strategy': last_attempt['strategy'],
                    'last_error': last_attempt['result'].get('error')
                })
            
            return result
            
        # Update search status
        self.active_search['status'] = 'completed'
        
        return best_result
    
    async def cancel_search(self) -> Dict[str, Any]:
        """
        Cancel an ongoing search operation.
        
        Returns:
            Dictionary with cancellation result
        """
        self.logger.info("Cancelling search operation")
        self.search_cancel_requested = True
        
        # Close browser resources
        await self.browser_interaction.close()
        
        return {
            'success': True,
            'cancelled': True,
            'message': 'Search operation cancelled'
        }
    
    def _prioritize_strategies(self, url: str, domain_type: str, 
                             preferred_method: str = None) -> List[str]:
        """
        Prioritize search strategies based on domain type and URL characteristics.
        
        Args:
            url: Target website URL
            domain_type: Type of domain
            preferred_method: Optional preferred method
            
        Returns:
            List of strategy names in priority order
        """
        # Base strategies
        all_strategies = [
            'direct_form_submission',
            'browser_interaction',
            'api_request',
            'ajax_intercept',
            'url_manipulation',
            'javascript_injection'
        ]
        
        # If preferred method is specified, prioritize it
        if preferred_method and preferred_method in all_strategies:
            # Move preferred method to the front
            all_strategies.remove(preferred_method)
            all_strategies.insert(0, preferred_method)
            return all_strategies
        
        # Domain-specific optimizations
        if domain_type == 'e_commerce':
            # E-commerce sites often have better API and AJAX handling
            return [
                'api_request',
                'ajax_intercept',
                'direct_form_submission',
                'browser_interaction',
                'url_manipulation',
                'javascript_injection'
            ]
            
        elif domain_type == 'real_estate':
            # Real estate sites often rely on forms and AJAX
            return [
                'direct_form_submission',
                'ajax_intercept',
                'browser_interaction',
                'api_request',
                'url_manipulation',
                'javascript_injection'
            ]
            
        elif domain_type == 'job_listings':
            # Job sites often use forms and URL parameters
            return [
                'direct_form_submission',
                'url_manipulation',
                'ajax_intercept',
                'browser_interaction',
                'api_request',
                'javascript_injection'
            ]
            
        # Default ordering
        return all_strategies
    
    async def _execute_strategy(self, strategy: str, url: str, search_term: str, page, 
                              domain_type: str, max_retries: int) -> Dict[str, Any]:
        """
        Execute a specific search strategy.
        
        Args:
            strategy: Strategy name to execute
            url: Target URL
            search_term: Search term
            page: Browser page object
            domain_type: Type of domain
            max_retries: Maximum retry attempts
            
        Returns:
            Dictionary with strategy execution results
        """
        # Navigate to the URL if we're at a different page
        if page.url != url:
            await self.browser_interaction.navigate(url, page)
        
        # Execute the requested strategy
        if strategy == 'direct_form_submission':
            return await self._execute_form_strategy(url, search_term, page, max_retries)
            
        elif strategy == 'browser_interaction':
            return await self._execute_browser_strategy(url, search_term, page, max_retries)
            
        elif strategy == 'api_request':
            return await self._execute_api_strategy(url, search_term, page, max_retries)
            
        elif strategy == 'ajax_intercept':
            return await self._execute_ajax_strategy(url, search_term, page, max_retries)
            
        elif strategy == 'url_manipulation':
            return await self._execute_url_strategy(url, search_term, page, max_retries)
            
        elif strategy == 'javascript_injection':
            return await self._execute_js_strategy(url, search_term, page, max_retries)
            
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")
    
    async def _execute_form_strategy(self, url: str, search_term: str, page, 
                                   max_retries: int) -> Dict[str, Any]:
        """
        Execute form-based search strategy.
        
        Args:
            url: Target URL
            search_term: Search term
            page: Browser page object
            max_retries: Maximum retry attempts
            
        Returns:
            Search result dictionary
        """
        # Get the page HTML
        html_content = await page.content()
        
        # Detect search forms
        domain_type = self._detect_domain_type(url)
        forms = await self.form_detector.detect_search_forms(html_content, domain_type)
        
        if not forms:
            return {
                'success': False,
                'error': 'No search forms found',
                'strategy': 'direct_form_submission'
            }
        
        # Try each form in order of relevance
        for form_data in forms:
            # Find the form using the selector
            form_selector = form_data.get('selector', '')
            if not form_selector:
                continue
                
            # Find search input field
            input_field = None
            for field in form_data.get('inputs', []):
                if field.get('is_search_input'):
                    input_field = field
                    break
            
            if not input_field:
                continue
                
            # Construct input selector
            input_selector = None
            if input_field.get('id'):
                input_selector = f"#{input_field['id']}"
            elif input_field.get('name'):
                input_selector = f"{form_selector} input[name='{input_field['name']}']"
            else:
                input_selector = f"{form_selector} input[type='text'], {form_selector} input[type='search']"
            
            # Fill the form
            fill_success = await self.browser_interaction.fill_form(input_selector, search_term, page)
            
            if fill_success:
                # Submit the form
                submit_success = await self.browser_interaction.submit_form(form_selector, page)
                
                if submit_success:
                    # Wait for navigation or AJAX results
                    await self.browser_interaction.wait_for_navigation(page)
                    
                    # Optional: Wait for AJAX content
                    await asyncio.sleep(1)
                    
                    # Return success result
                    return {
                        'success': True,
                        'strategy': 'direct_form_submission',
                        'url': page.url,
                        'search_term': search_term,
                        'form_selector': form_selector,
                        'input_selector': input_selector
                    }
        
        # If we've tried all forms without success
        return {
            'success': False,
            'error': 'Failed to submit forms',
            'strategy': 'direct_form_submission',
            'forms_found': len(forms)
        }
    
    async def _execute_browser_strategy(self, url: str, search_term: str, page, 
                                      max_retries: int) -> Dict[str, Any]:
        """
        Execute browser-based search strategy using more advanced interaction.
        
        Args:
            url: Target URL
            search_term: Search term
            page: Browser page object
            max_retries: Maximum retry attempts
            
        Returns:
            Search result dictionary
        """
        # This uses the browser interaction's built-in search capabilities
        result = await self.browser_interaction.perform_search(search_term, page)
        
        # Enhance the result with additional metadata
        result['strategy'] = 'browser_interaction'
        
        return result
    
    async def _execute_api_strategy(self, url: str, search_term: str, page, 
                                  max_retries: int) -> Dict[str, Any]:
        """
        Execute API-based search strategy.
        
        Args:
            url: Target URL
            search_term: Search term
            page: Browser page object
            max_retries: Maximum retry attempts
            
        Returns:
            Search result dictionary
        """
        # Get the page content
        html_content = await page.content()
        
        # Analyze for API endpoints
        api_params = await self.api_analyzer.analyze(page.url, html_content)
        
        if not api_params.get('has_api', False):
            return {
                'success': False,
                'error': 'No API endpoints detected',
                'strategy': 'api_request'
            }
        
        # Get the detected API endpoint
        api_endpoint = api_params.get('api_endpoint')
        if not api_endpoint:
            return {
                'success': False,
                'error': 'API endpoint not found',
                'strategy': 'api_request'
            }
        
        # Format the API request with the search term
        request_params = api_params.get('request_params', {})
        search_param_name = api_params.get('search_param_name', 'q')
        
        # Add the search term to the parameters
        request_params[search_param_name] = search_term
        
        # Make the API request
        try:
            # Use browser's fetch API to make the request
            response = await page.evaluate(f'''
                async () => {{
                    try {{
                        const response = await fetch("{api_endpoint}", {{
                            method: "{api_params.get('method', 'GET')}",
                            headers: {str(api_params.get('headers', {})).replace("'", '"')},
                            body: {str(request_params).replace("'", '"') if api_params.get('method') == 'POST' else 'null'}
                        }});
                        
                        const contentType = response.headers.get('content-type') || '';
                        
                        if (contentType.includes('application/json')) {{
                            return {{
                                status: response.status,
                                ok: response.ok,
                                data: await response.json()
                            }};
                        }} else {{
                            return {{
                                status: response.status,
                                ok: response.ok,
                                data: await response.text()
                            }};
                        }}
                    }} catch (error) {{
                        return {{
                            error: error.toString(),
                            ok: false
                        }};
                    }}
                }}
            ''')
            
            if not response.get('ok', False):
                return {
                    'success': False,
                    'error': f"API request failed: {response.get('error', 'Unknown error')}",
                    'strategy': 'api_request',
                    'status': response.get('status')
                }
            
            # Return the successful result
            return {
                'success': True,
                'strategy': 'api_request',
                'url': page.url,
                'search_term': search_term,
                'api_endpoint': api_endpoint,
                'response': response.get('data')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"API request error: {str(e)}",
                'strategy': 'api_request'
            }
    
    async def _execute_ajax_strategy(self, url: str, search_term: str, page, 
                                   max_retries: int) -> Dict[str, Any]:
        """
        Execute AJAX interception search strategy.
        
        Args:
            url: Target URL
            search_term: Search term
            page: Browser page object
            max_retries: Maximum retry attempts
            
        Returns:
            Search result dictionary
        """
        # Register AJAX request monitoring
        intercepted_requests = []
        
        await page.route('**/*.json', lambda route: asyncio.create_task(self._intercept_request(route, intercepted_requests)))
        await page.route('**/api/**', lambda route: asyncio.create_task(self._intercept_request(route, intercepted_requests)))
        await page.route('**/search*', lambda route: asyncio.create_task(self._intercept_request(route, intercepted_requests)))
        
        # Find a search interface
        search_interfaces = await self.browser_interaction.find_search_interface(page)
        
        if not search_interfaces.get('success') or search_interfaces.get('count', 0) == 0:
            # Clean up request monitoring
            await page.unroute('**/*.json')
            await page.unroute('**/api/**')
            await page.unroute('**/search*')
            
            return {
                'success': False,
                'error': 'No search interfaces found',
                'strategy': 'ajax_intercept'
            }
        
        # Try using the first interface
        interface = search_interfaces['interfaces'][0]
        
        # Interaction differs based on interface type
        interaction_success = False
        
        if interface['type'] == 'form':
            # Handle form-based interface with AJAX monitoring
            for input_field in interface.get('inputs', []):
                input_type = input_field.get('type', '')
                if input_type in ['search', 'text', '']:
                    # Try to construct a selector for this input
                    if input_field.get('id'):
                        input_selector = f"#{input_field['id']}"
                        
                        # Fill the input
                        fill_success = await self.browser_interaction.fill_form(input_selector, search_term, page)
                        if fill_success:
                            # Submit the form without page navigation
                            submit_success = await page.evaluate(f'''
                                (() => {{
                                    const form = document.querySelector("{interface['selector']}");
                                    if (!form) return false;
                                    
                                    // Override form submission to prevent navigation
                                    const originalSubmit = form.submit;
                                    form.submit = function() {{
                                        const event = new Event('submit', {{bubbles: true, cancelable: true}});
                                        form.dispatchEvent(event);
                                        return false;
                                    }};
                                    
                                    // Try to find a submit button
                                    const submitButton = form.querySelector('button[type="submit"], input[type="submit"]');
                                    if (submitButton) {{
                                        submitButton.click();
                                    }} else {{
                                        form.dispatchEvent(new Event('submit', {{bubbles: true, cancelable: true}}));
                                    }}
                                    
                                    return true;
                                }})()
                            ''')
                            
                            interaction_success = submit_success
                            break
                    
        elif interface['type'] == 'input':
            # Fill the standalone input
            fill_success = await self.browser_interaction.fill_form(interface['selector'], search_term, page)
            if fill_success:
                # Just update the input value and dispatch events
                key_press = await page.evaluate(f'''
                    (() => {{
                        const input = document.querySelector("{interface['selector']}");
                        if (!input) return false;
                        
                        // Dispatch keyboard events
                        const enterEvent = new KeyboardEvent('keydown', {{
                            bubbles: true,
                            cancelable: true,
                            key: 'Enter',
                            code: 'Enter',
                            keyCode: 13
                        }});
                        
                        input.dispatchEvent(enterEvent);
                        return true;
                    }})()
                ''')
                
                interaction_success = key_press
        
        elif interface['type'] == 'button':
            # Look for a nearby input
            input_found = await page.evaluate(f'''
                (() => {{
                    const button = document.querySelector("{interface['selector']}");
                    if (!button) return false;
                    
                    // Look for inputs near this button
                    let container = button.parentElement;
                    for (let i = 0; i < 3; i++) {{
                        if (!container) break;
                        
                        const inputs = container.querySelectorAll('input[type="search"], input[type="text"]');
                        if (inputs.length > 0) {{
                            // Found an input - focus and fill it
                            const input = inputs[0];
                            input.focus();
                            input.value = "{search_term}";
                            input.dispatchEvent(new Event('input', {{bubbles: true}}));
                            input.dispatchEvent(new Event('change', {{bubbles: true}}));
                            return true;
                        }}
                        
                        container = container.parentElement;
                    }}
                    
                    return false;
                }})()
            ''')
            
            if input_found:
                # Click the button
                click_success = await self.browser_interaction.click_element(interface['selector'], page)
                interaction_success = click_success
        
        # Wait for AJAX responses
        await asyncio.sleep(2)
        
        # Clean up request monitoring
        await page.unroute('**/*.json')
        await page.unroute('**/api/**')
        await page.unroute('**/search*')
        
        # Check if we intercepted any search-related requests
        search_requests = []
        for req in intercepted_requests:
            # Check if the request is likely related to search
            url_lower = req['url'].lower()
            if 'search' in url_lower or 'query' in url_lower or 'find' in url_lower or search_term.lower() in url_lower:
                search_requests.append(req)
        
        if not search_requests:
            return {
                'success': False,
                'error': 'No search-related AJAX requests intercepted',
                'strategy': 'ajax_intercept',
                'interaction_success': interaction_success
            }
        
        # Process the intercepted search requests
        await self.ajax_handler.process_response(page, search_requests[-1])
        
        # Check for results in the page content after AJAX processing
        has_results = await page.evaluate(f'''
            (() => {{
                const pageText = document.body.textContent.toLowerCase();
                return pageText.includes("{search_term.lower()}") && 
                       (pageText.includes("result") || 
                        pageText.includes("found") || 
                        document.querySelectorAll(".result, .search-result, .product, .listing").length > 0);
            }})()
        ''')
        
        if has_results:
            return {
                'success': True,
                'strategy': 'ajax_intercept',
                'url': page.url,
                'search_term': search_term,
                'ajax_url': search_requests[-1]['url'],
                'intercepted_requests': len(search_requests)
            }
        
        return {
            'success': False,
            'error': 'AJAX requests intercepted but no results found',
            'strategy': 'ajax_intercept',
            'ajax_url': search_requests[-1]['url'] if search_requests else None
        }
    
    async def _execute_url_strategy(self, url: str, search_term: str, page, 
                                  max_retries: int) -> Dict[str, Any]:
        """
        Execute URL manipulation search strategy.
        
        Args:
            url: Target URL
            search_term: Search term
            page: Browser page object
            max_retries: Maximum retry attempts
            
        Returns:
            Search result dictionary
        """
        # Parse the base URL
        parsed_url = urlparse(url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Common URL patterns for search
        search_url_patterns = [
            f"{base_domain}/search?q={search_term}",
            f"{base_domain}/search?query={search_term}",
            f"{base_domain}/search?term={search_term}",
            f"{base_domain}/search?keyword={search_term}",
            f"{base_domain}/search?s={search_term}",
            f"{base_domain}/search/{search_term}",
            f"{base_domain}?s={search_term}",
            f"{base_domain}?q={search_term}",
            f"{base_domain}/find?q={search_term}"
        ]
        
        # Try each URL pattern
        for search_url in search_url_patterns:
            # Navigate to the search URL
            navigation_result = await self.browser_interaction.navigate(search_url, page)
            
            if not navigation_result.get('success'):
                continue
                
            # Check if the page seems to contain search results
            has_results = await page.evaluate(f'''
                (() => {{
                    const pageText = document.body.textContent.toLowerCase();
                    return pageText.includes("{search_term.lower()}") && 
                           (pageText.includes("result") || 
                            pageText.includes("found") || 
                            document.querySelectorAll(".result, .search-result, .product, .listing, .item").length > 0);
                }})()
            ''')
            
            if has_results:
                return {
                    'success': True,
                    'strategy': 'url_manipulation',
                    'url': page.url,
                    'search_term': search_term,
                    'search_url_pattern': search_url
                }
        
        return {
            'success': False,
            'error': 'No successful URL pattern found',
            'strategy': 'url_manipulation',
            'patterns_tried': len(search_url_patterns)
        }
    
    async def _execute_js_strategy(self, url: str, search_term: str, page, 
                                 max_retries: int) -> Dict[str, Any]:
        """
        Execute JavaScript injection search strategy.
        
        Args:
            url: Target URL
            search_term: Search term
            page: Browser page object
            max_retries: Maximum retry attempts
            
        Returns:
            Search result dictionary
        """
        # Try to inject JavaScript to perform the search
        search_success = await page.evaluate(f'''
            (() => {{
                // Look for global search objects
                const searchObjects = [];
                
                // Common search function names
                const searchFunctionNames = [
                    'search', 'doSearch', 'performSearch', 'executeSearch', 
                    'runSearch', 'searchFor', 'find', 'query'
                ];
                
                // Check for search functions on window
                for (const name of searchFunctionNames) {{
                    if (typeof window[name] === 'function') {{
                        try {{
                            // Try to call the function with the search term
                            window[name]("{search_term}");
                            return true;
                        }} catch (e) {{
                            console.log(`Error calling ${name}:`, e);
                        }}
                    }}
                }}
                
                // Check for search functions on common objects
                const commonObjects = ['Search', 'SearchManager', 'searchManager', 'searchApi', 'searchClient'];
                for (const objName of commonObjects) {{
                    if (window[objName]) {{
                        const obj = window[objName];
                        
                        for (const name of searchFunctionNames) {{
                            if (typeof obj[name] === 'function') {{
                                try {{
                                    // Try to call the object method
                                    obj[name]("{search_term}");
                                    return true;
                                }} catch (e) {{
                                    console.log(`Error calling ${objName}.${name}:`, e);
                                }}
                            }}
                        }}
                    }}
                }}
                
                // Try more aggressive approach by finding any input and programmatically submitting
                const inputs = Array.from(document.querySelectorAll('input'));
                const searchInputs = inputs.filter(input => {{
                    const attrs = (input.id || '') + ' ' + (input.name || '') + ' ' + 
                                (input.className || '') + ' ' + (input.placeholder || '');
                    return attrs.toLowerCase().includes('search');
                }});
                
                if (searchInputs.length > 0) {{
                    const input = searchInputs[0];
                    input.value = "{search_term}";
                    input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    
                    // Try to find and click a nearby button
                    let searchButton = null;
                    
                    // Look in the parent elements
                    let parent = input.parentElement;
                    for (let i = 0; i < 3 && parent && !searchButton; i++) {{
                        const buttons = Array.from(parent.querySelectorAll('button, input[type="submit"], .btn, [role="button"]'));
                        searchButton = buttons.find(btn => btn.textContent.toLowerCase().includes('search'));
                        
                        if (!searchButton && buttons.length > 0) {{
                            // Just use the first button
                            searchButton = buttons[0];
                        }}
                        
                        parent = parent.parentElement;
                    }}
                    
                    if (searchButton) {{
                        searchButton.click();
                        return true;
                    }}
                    
                    // Try submitting the form if the input is in a form
                    const form = input.closest('form');
                    if (form) {{
                        form.submit();
                        return true;
                    }}
                    
                    // Trigger enter key on the input
                    const enterEvent = new KeyboardEvent('keydown', {{
                        bubbles: true,
                        cancelable: true,
                        key: 'Enter',
                        code: 'Enter',
                        keyCode: 13
                    }});
                    
                    input.dispatchEvent(enterEvent);
                    return true;
                }}
                
                return false;
            }})()
        ''')
        
        if not search_success:
            return {
                'success': False,
                'error': 'JavaScript injection failed to trigger search',
                'strategy': 'javascript_injection'
            }
        
        # Wait for any resulting page changes
        await asyncio.sleep(2)
        await self.browser_interaction.wait_for_navigation(page)
        
        # Check if the page now contains search results
        has_results = await page.evaluate(f'''
            (() => {{
                const pageText = document.body.textContent.toLowerCase();
                return pageText.includes("{search_term.lower()}") && 
                       (pageText.includes("result") || 
                        pageText.includes("found") || 
                        document.querySelectorAll(".result, .search-result, .product, .listing").length > 0);
            }})()
        ''')
        
        if has_results:
            return {
                'success': True,
                'strategy': 'javascript_injection',
                'url': page.url,
                'search_term': search_term
            }
        
        return {
            'success': False,
                'error': 'JavaScript injection executed but no results found',
                'strategy': 'javascript_injection'
        }
    
    async def _intercept_request(self, route, intercepted_requests):
        """
        Intercept and record network requests for AJAX strategy.
        
        Args:
            route: Playwright route object
            intercepted_requests: List to store intercepted requests
        """
        request = route.request
        
        # Record the request
        intercepted_requests.append({
            'url': request.url,
            'method': request.method,
            'headers': request.headers,
            'timestamp': asyncio.get_event_loop().time()
        })
        
        # Continue the request normally
        await route.continue_()
    
    def _detect_domain_type(self, url: str) -> str:
        """
        Detect domain type from URL.
        
        Args:
            url: URL to analyze
            
        Returns:
            Domain type string
        """
        # Re-using existing domain detection logic
        url_lower = url.lower()
        
        # Real estate domain indicators
        real_estate_indicators = [
            'real', 'estate', 'property', 'properties', 'home', 'homes', 'house', 'realty',
            'broker', 'apartment', 'rent', 'realtor', 'zillow', 'trulia', 'redfin', 'listing'
        ]
        
        # E-commerce domain indicators
        ecommerce_indicators = [
            'shop', 'store', 'buy', 'cart', 'checkout', 'product', 'mall', 'market',
            'amazon', 'ebay', 'price', 'order', 'purchase', 'catalog', 'retail'
        ]
        
        # Job listing domain indicators
        job_indicators = [
            'job', 'career', 'employ', 'hire', 'recruit', 'resume', 'cv', 'indeed',
            'linkedin', 'position', 'vacancy', 'work', 'glassdoor', 'staffing'
        ]
        
        # Check URL against indicators
        if any(indicator in url_lower for indicator in real_estate_indicators):
            return 'real_estate'
        elif any(indicator in url_lower for indicator in ecommerce_indicators):
            return 'e_commerce'
        elif any(indicator in url_lower for indicator in job_indicators):
            return 'job_listings'
        
        # Default to general if no specific domain detected
        return 'general'