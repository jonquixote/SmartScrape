"""
Search orchestrator component for managing and coordinating search operations.

This module provides the central orchestration for search operations, coordinating
between form detection, interaction, and results processing.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
import time
import random

from components.search.form_detection import SearchFormDetector
from components.search.browser_interaction import BrowserInteraction
from components.search.ajax_handler import AJAXHandler
from components.search.api_detection import APIParameterAnalyzer

class SearchCoordinator:
    """
    Coordinates search operations across different components.
    
    This class orchestrates the complex interactions between various search components:
    - Form detection and selection
    - Form interaction and submission
    - API detection and utilization
    - AJAX response handling
    - Result extraction and processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the search coordinator.
        
        Args:
            config: Optional configuration parameters
        """
        self.logger = logging.getLogger("SearchCoordinator")
        self.config = config or {}
        
        # Initialize sub-components
        self.form_detector = SearchFormDetector(self.config.get('form_detection', {}))
        self.browser_interaction = BrowserInteraction(self.config.get('browser_interaction', {}))
        self.ajax_handler = AJAXHandler(self.config.get('ajax_handler', {}))
        self.api_analyzer = APIParameterAnalyzer(self.config.get('api_analyzer', {}))
        
        # Internal state
        self.last_search_results = None
        self.search_history = []
        self.current_page = None
        
    async def initialize_browser(self) -> None:
        """Initialize the browser for interactive searches."""
        browser = await self.browser_interaction.initialize_browser()
        context = await self.browser_interaction.create_context(browser)
        self.current_page = await self.browser_interaction.new_page(context)
    
    async def close(self) -> None:
        """Close all resources."""
        await self.browser_interaction.close()
        self.current_page = None
    
    async def execute_search(self, url: str, search_term: str) -> Dict[str, Any]:
        """
        Execute a search operation on a website.
        
        This is the main entry point for search operations. It orchestrates:
        1. Navigation to the target URL
        2. Detection of search interfaces
        3. Interaction with the best search interface
        4. Processing of search results
        
        Args:
            url: Target website URL
            search_term: Search term to use
            
        Returns:
            Dictionary with search results and metadata
        """
        if not self.current_page:
            await self.initialize_browser()
        
        # Record this search in history
        self.search_history.append({
            'url': url,
            'search_term': search_term,
            'timestamp': time.time()
        })
        
        # Navigate to the target URL
        navigation_result = await self.browser_interaction.navigate(url, self.current_page)
        
        if not navigation_result.get('success', False):
            self.logger.error(f"Failed to navigate to {url}: {navigation_result.get('error')}")
            return {
                'success': False,
                'error': navigation_result.get('error', 'Navigation failed'),
                'url': url,
                'search_term': search_term
            }
        
        # Detect search forms
        self.logger.info(f"Detecting search interfaces on {url}")
        html_content = await self.current_page.content()
        search_forms = await self.form_detector.detect_search_forms(html_content, url)
        
        # If no forms found through static analysis, try browser-based detection
        if not search_forms or len(search_forms) == 0:
            self.logger.info("No search forms found through static analysis, trying browser-based detection")
            browser_interfaces = await self.browser_interaction.find_search_interface(self.current_page)
            
            if browser_interfaces.get('success', False) and browser_interfaces.get('count', 0) > 0:
                self.logger.info(f"Found {browser_interfaces.get('count')} search interfaces through browser detection")
                
                # Perform the search using browser interaction
                search_result = await self.browser_interaction.perform_search(search_term, self.current_page)
                
                if search_result.get('success', False):
                    # Extract results
                    extracted_content = await self.browser_interaction.extract_page_content(self.current_page)
                    
                    # Process the search results
                    processed_results = await self._process_search_results(extracted_content)
                    
                    self.last_search_results = {
                        'success': True,
                        'method': 'browser_interaction',
                        'url': search_result.get('url'),
                        'search_term': search_term,
                        'results': processed_results
                    }
                    
                    return self.last_search_results
                else:
                    self.logger.warning(f"Browser search failed: {search_result.get('error')}")
            else:
                self.logger.warning("No search interfaces found through browser detection")
        
        # Try API detection
        self.logger.info("Analyzing page for API endpoints")
        api_endpoints = await self.api_analyzer.detect_api_endpoints(html_content, self.current_page)
        
        if api_endpoints and len(api_endpoints) > 0:
            self.logger.info(f"Found {len(api_endpoints)} potential API endpoints")
            
            # Try to use the first API endpoint
            api_result = await self.api_analyzer.execute_api_search(
                search_term, 
                api_endpoints[0], 
                self.current_page
            )
            
            if api_result.get('success', False):
                # Process API results
                processed_results = await self._process_api_results(api_result.get('data', {}))
                
                self.last_search_results = {
                    'success': True,
                    'method': 'api',
                    'url': url,
                    'search_term': search_term,
                    'results': processed_results
                }
                
                return self.last_search_results
            else:
                self.logger.warning(f"API search failed: {api_result.get('error')}")
        
        # As a last resort, try search URL patterns
        self.logger.info("Trying common search URL patterns")
        search_urls = [
            f"{url}search?q={search_term.replace(' ', '+')}",
            f"{url}search?query={search_term.replace(' ', '+')}",
            f"{url}?s={search_term.replace(' ', '+')}"
        ]
        
        for search_url in search_urls:
            navigation_result = await self.browser_interaction.navigate(search_url, self.current_page)
            
            if navigation_result.get('success', False):
                # Check if the page looks like search results
                content = await self.current_page.content()
                if search_term.lower() in content.lower() and ('result' in content.lower() or 'search' in content.lower()):
                    # Extract content from the page
                    extracted_content = await self.browser_interaction.extract_page_content(self.current_page)
                    
                    # Process the search results
                    processed_results = await self._process_search_results(extracted_content)
                    
                    self.last_search_results = {
                        'success': True,
                        'method': 'url_pattern',
                        'url': search_url,
                        'search_term': search_term,
                        'results': processed_results
                    }
                    
                    return self.last_search_results
        
        # If all methods fail
        self.logger.error(f"All search methods failed for {url} with term '{search_term}'")
        return {
            'success': False,
            'error': 'All search methods failed',
            'url': url,
            'search_term': search_term
        }
    
    async def _process_search_results(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process extracted search results from HTML content.
        
        Args:
            content: Extracted page content
            
        Returns:
            List of processed search result items
        """
        results = []
        
        # Extract search results from links
        links = content.get('links', [])
        for link in links:
            # Skip navigation links, social media, etc.
            url = link.get('url', '')
            text = link.get('text', '')
            
            if (not text or text == '[No Text]' or 
                'javascript:' in url or 
                '#' in url or 
                'facebook.com' in url or 
                'twitter.com' in url or
                'instagram.com' in url or
                'login' in url.lower() or
                'signin' in url.lower()):
                continue
            
            results.append({
                'title': text,
                'url': url,
                'type': 'link'
            })
        
        # Extract content from page text if available
        if 'text' in content:
            # Basic content extraction logic - could be enhanced
            paragraphs = content['text'].split('\n')
            for paragraph in paragraphs:
                if len(paragraph.strip()) > 100:  # Only substantial paragraphs
                    results.append({
                        'content': paragraph.strip(),
                        'type': 'content'
                    })
        
        return results
    
    async def _process_api_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process search results from API data.
        
        Args:
            data: API response data
            
        Returns:
            List of processed search result items
        """
        results = []
        
        # Process based on common API response patterns
        if isinstance(data, list):
            # Array of results
            for item in data:
                if isinstance(item, dict):
                    result = {}
                    
                    # Look for common keys in API responses
                    for title_key in ['title', 'name', 'heading', 'label']:
                        if title_key in item:
                            result['title'] = item[title_key]
                            break
                    
                    for url_key in ['url', 'link', 'href']:
                        if url_key in item:
                            result['url'] = item[url_key]
                            break
                    
                    for content_key in ['description', 'content', 'text', 'body', 'summary']:
                        if content_key in item:
                            result['content'] = item[content_key]
                            break
                    
                    result['type'] = 'api_item'
                    results.append(result)
        elif isinstance(data, dict):
            # Object with results array
            for results_key in ['results', 'items', 'data', 'content', 'response']:
                if results_key in data and isinstance(data[results_key], list):
                    return await self._process_api_results(data[results_key])
            
            # Handle single result object
            result = {}
            
            # Look for common keys in API responses
            for title_key in ['title', 'name', 'heading', 'label']:
                if title_key in data:
                    result['title'] = data[title_key]
                    break
            
            for url_key in ['url', 'link', 'href']:
                if url_key in data:
                    result['url'] = data[url_key]
                    break
            
            for content_key in ['description', 'content', 'text', 'body', 'summary']:
                if content_key in data:
                    result['content'] = data[content_key]
                    break
            
            result['type'] = 'api_item'
            results.append(result)
        
        return results
    
    def get_best_search_method(self, url: str) -> str:
        """
        Determine the best search method for a particular URL based on past results.
        
        Args:
            url: Target website URL
            
        Returns:
            String indicating the recommended search method
        """
        # Analyze search history to determine best method
        relevant_searches = [s for s in self.search_history if s['url'] == url]
        
        if not relevant_searches:
            return 'auto'  # No history, try automatic detection
        
        # Count success by method
        method_counts = {}
        for search in relevant_searches:
            if search.get('success', False):
                method = search.get('method', 'auto')
                method_counts[method] = method_counts.get(method, 0) + 1
        
        if not method_counts:
            return 'auto'  # No successful methods, try automatic
        
        # Return most successful method
        return max(method_counts.items(), key=lambda x: x[1])[0]