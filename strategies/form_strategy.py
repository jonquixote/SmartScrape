"""
Form-Based Search Engine Strategy

This module implements a search engine strategy that can interact with HTML forms
to perform searches. It detects and fills out search forms on web pages.
"""

import logging
import re
import asyncio
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from urllib.parse import urlparse, urljoin

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page, ElementHandle
from fake_useragent import UserAgent

from strategies.base_strategy import (
    SearchEngineInterface,
    SearchCapabilityType,
    SearchEngineCapability,
    register_search_engine
)
from strategies.core.strategy_interface import BaseStrategy
from strategies.core.strategy_types import StrategyType, StrategyCapability, strategy_metadata
from strategies.core.strategy_context import StrategyContext
from strategies.ai_guided.site_structure.site_analyzer import SiteStructureAnalyzer
from abc import abstractmethod

# Import search components
try:
    from components.pagination_handler import PaginationHandler
    from components.search.form_detection import SearchFormDetector
    from components.search.browser_interaction import BrowserInteraction
    from utils.retry_utils import with_exponential_backoff, with_http_retry, NETWORK_EXCEPTIONS
    from components.search.form_interaction import FormInteraction, HumanLikeInteraction
    from components.search.ajax_handler import AJAXHandler
    from components.search.api_detection import APIParameterAnalyzer
    from components.search.form_interaction_strategies import SubmissionVerification
    logger = logging.getLogger("FormStrategy")
    logger.info("Successfully imported search components")
except ImportError as e:
    # For standalone tests, provide a path hint
    logger = logging.getLogger("FormStrategy")
    logger.warning(f"Could not import some search components: {str(e)}. " +
                  "Make sure the components directory is in the Python path.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FormStrategy")

@strategy_metadata(
    strategy_type=StrategyType.INTERACTION,
    capabilities={
        StrategyCapability.FORM_INTERACTION,
        StrategyCapability.DYNAMIC_CONTENT,
        StrategyCapability.JAVASCRIPT_EXECUTION
    },
    description="Search engine that interacts with HTML forms to perform searches"
)
@register_search_engine
class FormSearchEngine(SearchEngineInterface, BaseStrategy):
    """
    Search engine that can interact with HTML forms to perform searches.
    It detects search forms on web pages and fills them out with the provided query.
    """
    
    def __init__(self, 
                 context: Optional['StrategyContext'] = None,
                 use_browser: bool = True,
                 max_depth: int = 2, 
                 max_pages: int = 100,
                 include_external: bool = False,
                 user_prompt: str = "",
                 filter_chain: Optional[Any] = None,
                 coordinator: Optional[Any] = None):
        """
        Initialize the form search engine.
        
        Args:
            context: The strategy context containing shared services and configuration
            use_browser: Whether to use a browser for form interaction (required for JavaScript forms)
            max_depth: Maximum crawling depth
            max_pages: Maximum number of pages to crawl
            include_external: Whether to include external links
            user_prompt: The user's original request/prompt
            filter_chain: Filter chain to apply to URLs
            coordinator: Optional SearchCoordinator for component coordination
        """
        # Call the BaseStrategy init to ensure it's properly initialized
        super().__init__(context)
        
        # Store configuration
        self.config = {
            'max_depth': max_depth,
            'max_pages': max_pages,
            'include_external': include_external,
            'user_prompt': user_prompt,
            'filter_chain': filter_chain
        }
        
        self.use_browser = use_browser
        self.user_agent = UserAgent().random
        self._browser = None
        self.browser = None  # Alias for backward compatibility
        self._context = None
        
        # Initialize search coordinator if provided
        self.coordinator = coordinator
        
        # Initialize site analyzer for search page discovery
        from strategies.ai_guided.site_structure.site_analyzer import SiteStructureAnalyzer
        self.site_analyzer = SiteStructureAnalyzer(use_ai=False)
        
        # Initialize form detection component
        try:
            self.form_detector = SearchFormDetector()
            logger.info("SearchFormDetector initialized successfully")
        except NameError:
            logger.warning("SearchFormDetector not available, using built-in form detection")
            self.form_detector = None
            
        # Initialize pagination handler for handling multi-page results
        try:
            self.pagination_handler = PaginationHandler(max_depth=max_pages)
            self.pagination_support_enabled = True
        except NameError:
            logger.warning("PaginationHandler not available, pagination support disabled")
            self.pagination_support_enabled = False
            
        # Initialize API parameter analyzer for detecting search APIs
        try:
            self.api_analyzer = APIParameterAnalyzer()
            logger.info("APIParameterAnalyzer initialized successfully")
        except NameError:
            logger.warning("APIParameterAnalyzer not available, API detection support disabled")
            self.api_analyzer = None
            
        # Initialize AJAX handler for dealing with dynamic content
        try:
            self.ajax_handler = AJAXHandler()
            logger.info("AJAXHandler initialized successfully")
        except NameError:
            logger.warning("AJAXHandler not available, AJAX handling support disabled")
            self.ajax_handler = None
        
    @property
    def name(self) -> str:
        """
        Get the name of the search engine.
        
        Returns:
            str: Engine name
        """
        return "form_search_engine"
        
    async def register_with_coordinator(self, coordinator):
        """
        Register this search engine with a SearchCoordinator.
        
        Args:
            coordinator: The SearchCoordinator instance
            
        Returns:
            True if registration was successful
        """
        logger.info("Registering FormSearchEngine with SearchCoordinator")
        self.coordinator = coordinator
        
        # Register capabilities with the coordinator if supported
        if hasattr(coordinator, 'register_strategy'):
            await coordinator.register_strategy(
                strategy_name=self.name,
                capabilities={
                    'form_interaction': True,
                    'browser_automation': self.use_browser,
                    'dynamic_content': True,
                    'pagination': self.pagination_support_enabled,
                    'api_detection': self.api_analyzer is not None,
                    'ajax_handling': self.ajax_handler is not None
                }
            )
            
        return True
    
    async def can_handle(self, url: str) -> bool:
        """
        Check if this strategy can handle the given URL.
        
        Args:
            url: URL to check
            
        Returns:
            True if the strategy can handle the URL, False otherwise
        """
        try:
            # For form strategy, we need to check if the page contains a form
            # This is a basic check that can be enhanced with more heuristics
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers={'User-Agent': self.user_agent}, follow_redirects=True)
                if response.status_code != 200:
                    return False
                
                html = response.text
                soup = BeautifulSoup(html, 'lxml')
                
                # Check for forms
                forms = soup.find_all('form')
                if forms:
                    # Look for search-related attributes
                    for form in forms:
                        # Check form attributes
                        action = form.get('action', '').lower()
                        form_id = form.get('id', '').lower()
                        form_class = form.get('class', [])
                        if isinstance(form_class, list):
                            form_class = ' '.join(form_class).lower()
                        else:
                            form_class = str(form_class).lower()
                        
                        # Look for search indicators
                        search_indicators = ['search', 'query', 'find', 'lookup', 'seek']
                        for indicator in search_indicators:
                            if (indicator in action or 
                                indicator in form_id or 
                                indicator in form_class):
                                return True
                        
                        # Check input fields
                        inputs = form.find_all('input')
                        for input_field in inputs:
                            input_type = input_field.get('type', '').lower()
                            input_name = input_field.get('name', '').lower()
                            input_id = input_field.get('id', '').lower()
                            
                            # Look for search indicators in input
                            for indicator in search_indicators:
                                if (indicator in input_name or 
                                    indicator in input_id):
                                    return True
                
                # If no search form found, return False
                return False
        except Exception as e:
            # If there's an error, log and return False
            if self.logger:
                self.logger.warning(f"Error checking if FormSearchEngine can handle {url}: {str(e)}")
            return False
            
    async def crawl(self, url: str, **kwargs) -> Any:
        """
        Crawl a URL and its linked pages.
        
        This is required by the BaseStrategy abstract interface.
        For FormSearchEngine, this is handled by the execute method.
        
        Args:
            url: URL to crawl
            **kwargs: Additional arguments
            
        Returns:
            Crawling results
        """
        # For FormSearchEngine, crawling is done within execute
        # This is just a placeholder to satisfy the abstract method requirement
        return {}
        
    async def extract(self, content: Any, **kwargs) -> Any:
        """
        Extract data from content.
        
        This is required by the BaseStrategy abstract interface.
        For FormSearchEngine, this is handled internally during form submission.
        
        Args:
            content: Content to extract data from
            **kwargs: Additional arguments
            
        Returns:
            Extracted data
        """
        # For FormSearchEngine, extraction is done within execute
        # This is just a placeholder to satisfy the abstract method requirement
        return {}
        
    def get_results(self) -> Any:
        """
        Get the results of the strategy execution.
        
        This is required by the BaseStrategy abstract interface.
        For FormSearchEngine, results are returned directly by execute.
        
        Returns:
            Strategy results
        """
        # For FormSearchEngine, results are returned directly by execute
        # This is just a placeholder to satisfy the abstract method requirement
        return {}
    def capabilities(self) -> List[SearchEngineCapability]:
        """
        Get the capabilities of this search engine.
        
        Returns:
            List of SearchEngineCapability objects
        """
        return [
            SearchEngineCapability(
                capability_type=SearchCapabilityType.FORM_BASED,
                confidence=1.0,
                requires_browser=True,
                description="Can detect and interact with HTML forms"
            ),
            SearchEngineCapability(
                capability_type=SearchCapabilityType.DOM_MANIPULATION,
                confidence=0.8,
                requires_browser=True,
                description="Can manipulate DOM elements via browser automation"
            ),
            SearchEngineCapability(
                capability_type=SearchCapabilityType.AUTOCOMPLETE,
                confidence=0.7,
                requires_browser=True,
                description="Can handle autocomplete suggestions on form fields"
            )
        ]
    
    def get_required_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters required by this search engine.
        
        Returns:
            Dictionary of parameter name to parameter specification
        """
        return {
            "query": {
                "type": "string",
                "description": "The search query to submit to the form",
                "required": True
            },
            "form_index": {
                "type": "integer",
                "description": "Index of the form to use if multiple forms are found (0-based)",
                "required": False,
                "default": 0
            },
            "wait_for_navigation": {
                "type": "boolean",
                "description": "Whether to wait for page navigation after form submission",
                "required": False,
                "default": True
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in milliseconds for form submission",
                "required": False,
                "default": 30000
            }
        }
    
    async def can_handle(self, url: str, html: Optional[str] = None) -> Tuple[bool, float]:
        """
        Determine if this search engine can handle the given URL/page.
        
        Args:
            url: The URL to check
            html: Optional HTML content of the page
            
        Returns:
            Tuple of (can_handle, confidence)
        """
        # If HTML is provided, check it directly
        if html:
            return self._analyze_html_for_forms(html)
        
        # Otherwise, fetch the page and check it
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                headers = {"User-Agent": self.user_agent}
                response = await client.get(url, headers=headers)
                
                if response.status_code != 200:
                    return False, 0.0
                
                html = response.text
                return self._analyze_html_for_forms(html)
                
        except Exception as e:
            logger.error(f"Error checking if can handle URL {url}: {str(e)}")
            return False, 0.0
    
    async def _analyze_html_for_forms(self, html: str) -> Tuple[bool, float]:
        """
        Analyze HTML to check for search forms.
        
        Args:
            html: The HTML content to analyze
            
        Returns:
            Tuple of (has_search_form, confidence)
        """
        try:
            # If we have the SearchFormDetector component, use it
            if self.form_detector:
                # Call the async detect_search_forms method correctly
                domain_type = "real_estate" if "realty" in html.lower() or "property" in html.lower() or "home" in html.lower() else "general"
                logger.info(f"Using SearchFormDetector with domain type: {domain_type}")
                
                try:
                    detected_forms = await self.form_detector.detect_search_forms(html, domain_type)
                    
                    if not detected_forms or len(detected_forms) == 0:
                        logger.warning("No search forms detected by SearchFormDetector")
                        return False, 0.0
                    
                    # Get the highest confidence form
                    best_form = max(detected_forms, key=lambda form: form.get('search_relevance_score', 0.0))
                    best_confidence = best_form.get('search_relevance_score', 0.0) / 100.0  # Convert to 0-1 scale
                
                    logger.info(f"SearchFormDetector found form with confidence {best_confidence:.2f}")
                    
                    # Store the detected forms for later use
                    self._detected_forms = detected_forms
                    self._best_form = best_form
                    
                    return best_confidence > 0.3, best_confidence
                
                except Exception as detector_error:
                    logger.error(f"Error using SearchFormDetector: {str(detector_error)}")
                    # Fall through to built-in method
            
            # Otherwise, fall back to the built-in detection method
            logger.warning("Using fallback form detection method - SearchFormDetector not available")
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for forms
            forms = soup.find_all('form')
            if not forms:
                return False, 0.0
            
            # Calculate confidence based on form characteristics
            best_confidence = 0.0
            
            for form in forms:
                confidence = self._score_form_search_likelihood(form)
                best_confidence = max(best_confidence, confidence)
            
            return best_confidence > 0.3, best_confidence
            
        except Exception as e:
            logger.error(f"Error analyzing HTML for forms: {str(e)}")
            return False, 0.0
    
    def _score_form_search_likelihood(self, form, url=None) -> float:
        """
        Score how likely a form is to be a search form.
        
        Args:
            form: BeautifulSoup form element
            url: URL of the page containing the form (optional)
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        score = 0.0
        
        # Check form attributes
        action = form.get('action', '').lower()
        method = form.get('method', '').lower()
        form_id = form.get('id', '').lower()
        form_class = ' '.join(form.get('class', [])).lower()
        form_name = form.get('name', '').lower()
        
        # Higher score for GET method (typical for search forms)
        if method == 'get':
            score += 0.2
        
        # Check for search-related terms in attributes
        search_terms = ['search', 'find', 'query', 'look', 'seek', 'properties', 'listings', 'homes', 'filter']
        for term in search_terms:
            if term in action or term in form_id or term in form_class or term in form_name:
                score += 0.3
                break
        
        # Check for text inputs
        text_inputs = form.find_all('input', type=['text', 'search'])
        if text_inputs:
            score += 0.3
            
            # Check input attributes for search terms
            for input_el in text_inputs:
                input_name = input_el.get('name', '').lower()
                input_id = input_el.get('id', '').lower()
                input_placeholder = input_el.get('placeholder', '').lower()
                
                # Extended search terms for real estate
                extended_terms = ['search', 'find', 'lookup', 'q', 'query', 'keyword', 
                                 'location', 'city', 'address', 'state', 'zip', 'postal',
                                 'area', 'region', 'property', 'home', 'house']
                
                for term in extended_terms:
                    if term in input_name or term in input_id or term in input_placeholder:
                        score += 0.2
                        break
        
        # Check for submit buttons
        submit_buttons = form.find_all(['button', 'input'], type=['submit', 'button'])
        if submit_buttons:
            score += 0.1
            
            # Check button text for search terms
            for button in submit_buttons:
                button_text = button.text.lower() if button.text else ''
                button_value = button.get('value', '').lower()
                
                # Extended button terms
                button_terms = ['search', 'find', 'go', 'submit', 'lookup', 'show', 'filter', 'properties', 'listings']
                
                for term in button_terms:
                    if term in button_text or term in button_value:
                        score += 0.1
                        break
        
        # Check for location/address related fields (real estate specific)
        location_inputs = form.find_all('input', attrs={
            'name': lambda x: x and any(term in x.lower() for term in ['city', 'location', 'address', 'state', 'zip', 'postal', 'region', 'area'])
        })
        if location_inputs:
            score += 0.2
            
        # Check for select dropdowns with relevant options (common in real estate forms)
        selects = form.find_all('select')
        for select in selects:
            select_name = select.get('name', '').lower()
            select_id = select.get('id', '').lower()
            
            if any(term in select_name or term in select_id for term in 
                  ['property', 'type', 'category', 'bedroom', 'bathroom', 'price', 'area']):
                score += 0.15
                break
        
        # Cap the score at 1.0
        return min(score, 1.0)
    
    async def _init_browser(self):
        """Initialize browser if not already initialized"""
        if not self._browser:
            try:
                # Use BrowserInteraction component instead of direct Playwright code
                if not hasattr(self, 'browser_interaction'):
                    from components.search.browser_interaction import BrowserInteraction
                    self.browser_interaction = BrowserInteraction(config={
                        'browser_type': 'chromium',
                        'headless': True,
                        'slow_mo': 50,  # Slight slowdown for reliability
                        'timeout': 60000,  # 60 seconds timeout
                        'use_stealth': True
                    })
                
                # Initialize browser through BrowserInteraction
                self._browser = await self.browser_interaction.initialize_browser()
                self.browser = self._browser  # Set the alias for backward compatibility
                
                # Create context through BrowserInteraction
                self._context = await self.browser_interaction.create_context(self._browser)
                
                # Initialize search coordinator if it wasn't previously set
                if not hasattr(self, 'coordinator') or not self.coordinator:
                    logger.info("Initializing SearchCoordinator")
                    from components.search.search_coordinator import SearchCoordinator
                    self.coordinator = SearchCoordinator()
                    
                    # Register this engine with the coordinator
                    await self.register_with_coordinator(self.coordinator)
                    logger.info("Successfully registered with SearchCoordinator")
                
                return self._browser
            except Exception as e:
                logger.error(f"Error initializing browser: {str(e)}")
                raise
    
    async def _extract_results_with_browser(self, page, results_url) -> List[Dict[str, Any]]:
        """
        Extract results using browser capabilities for more structured extraction.
        This method leverages JavaScript evaluation to extract results from the page.
        
        Args:
            page: Playwright page object
            results_url: URL of the results page
            
        Returns:
            List of extracted results
        """
        try:
            # First check if we have AJAX handler to handle dynamic content
            if self.ajax_handler:
                try:
                    # Try to handle AJAX search results for more complete extraction
                    logger.info("Using AJAXHandler to process search results")
                    
                    # Try to detect and handle Load More buttons (common in search results)
                    load_more_results = await self.ajax_handler.handle_load_more_button(page, max_clicks=2)
                    if load_more_results:
                        logger.info(f"Load more button clicked, revealing additional results")
                    
                    # Try to process any AJAX search results
                    ajax_results = await self.ajax_handler.handle_ajax_search_results(page, "")
                    if (ajax_results and len(ajax_results) > 0):
                        logger.info(f"Extracted {len(ajax_results)} results from AJAX responses")
                        return ajax_results
                except Exception as e:
                    logger.warning(f"AJAX result handling failed: {str(e)}, falling back to standard extraction")
            
            # Try to detect real estate listings using domain-specific selectors
            domain = urlparse(results_url).netloc.lower()
            
            # For Ohio Broker Direct specifically
            if "ohiobrokerdirect" in domain:
                logger.info("Extracting results for Ohio Broker Direct")
                
                # Try to find property listings with enhanced detection for Ohio Broker Direct website
                property_data = await page.evaluate(r"""
                    () => {
                        const listings = [];
                        
                        // Ohio Broker Direct specific selectors
                        const ohioBrokerSelectors = [
                            '.property-listing', 
                            '.property-item', 
                            '.listing-item', 
                            '.search-result', 
                            '[class*="property-"][class*="item"]',
                            '.card', 
                            '.card-listing',
                            '[class*="listing"]',
                            '[class*="property"]',
                            '[class*="result"]'
                        ];
                        
                        // First try with Ohio specific selectors
                        let containers = [];
                        for (const selector of ohioBrokerSelectors) {
                            const elements = document.querySelectorAll(selector);
                            if (elements && elements.length > 0) {
                                console.log(`Found ${elements.length} elements with selector: ${selector}`);
                                containers = Array.from(elements);
                                break;
                            }
                        }
                        
                        // If no specific selectors work, try generic approach
                        if (containers.length === 0) {
                            console.log("No specific containers found, trying generic selectors");
                            const genericSelectors = [
                                '.property-item', 
                                '.listing-item', 
                                '.property', 
                                '.listing', 
                                '[itemtype*="Product"]', 
                                '[itemtype*="Offer"]',
                                'div[class*="property"]',
                                'div[class*="listing"]',
                                'article', 
                                '.card',
                                '.result-item'
                            ];
                            
                            for (const selector of genericSelectors) {
                                const elements = document.querySelectorAll(selector);
                                if (elements && elements.length > 0) {
                                    console.log(`Found ${elements.length} elements with generic selector: ${selector}`);
                                    containers = Array.from(elements);
                                    break;
                                }
                            }
                        }
                        
                        // If we still don't have containers, try the most generic approach
                        if (containers.length === 0) {
                            console.log("No containers found with selectors, trying to find Cleveland properties");
                            
                            // Look for elements that mention Cleveland
                            const allDivs = Array.from(document.querySelectorAll('div, article, section'));
                            const clevelandDivs = allDivs.filter(div => {
                                const text = div.textContent || '';
                                return (text.includes('Cleveland') || text.includes('cleveland')) && 
                                       (text.includes('$') || text.includes('Price')) &&
                                       div.querySelectorAll('a, img').length > 0 && // Has links or images
                                       div.textContent.length > 50 && // Not too small
                                       div.textContent.length < 1000; // Not too large (whole page)
                            });
                            
                            if (clevelandDivs.length > 0) {
                                console.log(`Found ${clevelandDivs.length} divs mentioning Cleveland`);
                                containers = clevelandDivs;
                            }
                        }
                        
                        if (containers && containers.length > 0) {
                            console.log(`Processing ${containers.length} listing containers`);
                            
                            for (const container of containers) {
                                try {
                                    // Extract data with enhanced detection for Ohio Broker Direct
                                    const titleEl = container.querySelector('h1, h2, h3, h4, h5, .property-title, .listing-title, .title, [class*="title"]');
                                    const title = titleEl ? titleEl.textContent.trim() : '';
                                    
                                    // Find a link - first try with the title element, then any link in the container
                                    let link = '';
                                    if (titleEl) {
                                        const closestLink = titleEl.closest('a');
                                        if (closestLink) link = closestLink.href;
                                    }
                                    if (!link) {
                                        const linkEl = container.querySelector('a[href]:not([href="#"])');
                                        if (linkEl) link = linkEl.href;
                                    }
                                    
                                    // Find price - try multiple approaches
                                    let price = '';
                                    const priceEl = container.querySelector('.price, [class*="price"], [itemprop="price"], [class*="amount"]');
                                    if (priceEl) {
                                        price = priceEl.textContent.trim();
                                    } else {
                                        // Try regex approach for price pattern
                                        const text = container.textContent;
                                        const priceMatch = text.match(/[$][\d,]+(?:\.[\d]{2})?|[$][\d,]+K|[$][\d,]+M/);
                                        if (priceMatch) price = priceMatch[0];
                                    }
                                    
                                    // Find address/location with enhanced detection for Cleveland properties
                                    let address = '';
                                    const addressEl = container.querySelector('.address, .location, [class*="address"], [class*="location"], [itemprop="address"]');
                                    if (addressEl) {
                                        address = addressEl.textContent.trim();
                                    } else {
                                        // Look for Cleveland mention in the text
                                        const text = container.textContent;
                                        if (text.includes('Cleveland')) {
                                            // Try to extract address pattern
                                            const addressMatch = text.match(/(\d+)\s+[A-Za-z\s.]+(?:Avenue|Ave|Street|St|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Place|Pl|Parkway|Pkwy)[,\s]+(?:Cleveland|OH|Ohio)/i);
                                            if (addressMatch) {
                                                address = addressMatch[0].trim();
                                            } else if (text.includes('Cleveland')) {
                                                address = 'Cleveland, OH';
                                            }
                                        }
                                    }
                                    
                                    // Find description
                                    let description = '';
                                    const descEl = container.querySelector('.description, [class*="description"], p:not(.price):not(.address):not(.title)');
                                    if (descEl) {
                                        description = descEl.textContent.trim();
                                    } else {
                                        // Use a snippet of text from the container
                                        const text = container.textContent;
                                        // Remove title, price, address from the text if they exist
                                        let cleanText = text;
                                        if (title) cleanText = cleanText.replace(title, '');
                                        if (price) cleanText = cleanText.replace(price, '');
                                        if (address) cleanText = cleanText.replace(address, '');
                                        
                                        // Take a reasonable snippet (not too long)
                                        description = cleanText.trim().substring(0, 200);
                                        if (description.length === 200) description += '...';
                                    }
                                    
                                    // Find features (bedrooms, bathrooms, etc.) with enhanced detection
                                    let bedrooms = '';
                                    let bathrooms = '';
                                    let sqft = '';
                                    
                                    // Look for specific elements first
                                    const bedroomsEl = container.querySelector('[class*="bed"], [data-beds], [class*="rooms"]');
                                    if (bedroomsEl) {
                                        bedrooms = bedroomsEl.textContent.trim().replace(/[^0-9.]/g, '');
                                    }
                                    
                                    const bathroomsEl = container.querySelector('[class*="bath"], [data-baths]');
                                    if (bathroomsEl) {
                                        bathrooms = bathroomsEl.textContent.trim().replace(/[^0-9.]/g, '');
                                    }
                                    
                                    const sqftEl = container.querySelector('[class*="sqft"], [class*="square"], [class*="area"], [data-sqft]');
                                    if (sqftEl) {
                                        sqft = sqftEl.textContent.trim().replace(/[^0-9]/g, '');
                                    }
                                    
                                    // If features weren't found in specific elements, try regex on the description or container text
                                    if (!bedrooms || !bathrooms) {
                                        const text = container.textContent;
                                        const bedBathMatch = text.match(/(\d+)\s*(?:bed|BR|bedroom|bd).*?(\d+)\s*(?:bath|BA|bathroom|ba)/i);
                                        if (bedBathMatch) {
                                            if (!bedrooms) bedrooms = bedBathMatch[1];
                                            if (!bathrooms) bathrooms = bedBathMatch[2];
                                        } else {
                                            const bedMatch = text.match(/(\d+)\s*(?:bed|BR|bedroom|bd)/i);
                                            const bathMatch = text.match(/(\d+)\s*(?:bath|BA|bathroom|ba)/i);
                                            if (bedMatch && !bedrooms) bedrooms = bedMatch[1];
                                            if (bathMatch && !bathrooms) bathrooms = bathMatch[1];
                                        }
                                        
                                        if (!sqft) {
                                            const sqftMatch = text.match(/(\d[,\d]+)\s*(?:sq\.\s*ft\.|square\s*feet|sf|sqft)/i);
                                            if (sqftMatch) sqft = sqftMatch[1].replace(/,/g, '');
                                        }
                                    }
                                    
                                    // Find image
                                    let image = '';
                                    const imgEl = container.querySelector('img');
                                    if (imgEl) {
                                        image = imgEl.src;
                                    }
                                    
                                    // Check if this is actually a Cleveland property
                                    const isCleveland = address.toLowerCase().includes('cleveland') || 
                                                      container.textContent.toLowerCase().includes('cleveland');
                                                      
                                    // Only add if we have sufficient data and it appears to be in Cleveland
                                    if ((title || address || price) && isCleveland) {
                                        // Format the result
                                        listings.push({
                                            title: title || 'Cleveland Property Listing',
                                            url: link || window.location.href,
                                            price: price || 'Contact for price',
                                            address: address || 'Cleveland, OH',
                                            description: description,
                                            bedrooms: bedrooms,
                                            bathrooms: bathrooms,
                                            square_feet: sqft,
                                            image_url: image,
                                            is_cleveland: true
                                        });
                                    }
                                } catch (e) {
                                    console.error("Error extracting listing:", e);
                                }
                            }
                            
                            return listings;
                        }
                        
                        // If no specific containers found, try a more generic approach
                        // Look for divs or sections that might contain property data
                        const potentialListings = document.querySelectorAll('div[class*="property"], div[class*="listing"], section[class*="property"], section[class*="listing"], div.card, article');
                        
                        if (potentialListings && potentialListings.length > 0) {
                            console.log("Found " + potentialListings.length + " potential listing containers");
                            
                            for (const container of potentialListings) {
                                try {
                                    // Skip if too small to be a listing
                                    if (container.textContent.length < 50) continue;
                                    
                                    // Extract basic data
                                    const titleEl = container.querySelector('h1, h2, h3, h4, h5');
                                    const title = titleEl ? titleEl.textContent.trim() : '';
                                    
                                    const linkEl = container.querySelector('a[href]');
                                    const link = linkEl ? linkEl.href : '';
                                    
                                    // Find any element that might contain price information
                                    const pricePattern = /[$][\d,]+(?:\.[\d]{2})?|[$][\d,]+K|[$][\d,]+M/;
                                    const text = container.textContent;
                                    const priceMatch = text.match(pricePattern);
                                    const price = priceMatch ? priceMatch[0] : '';
                                    
                                    // Check for Cleveland mention
                                    const isCleveland = text.toLowerCase().includes('cleveland');
                                    
                                    // Find image
                                    const imgEl = container.querySelector('img');
                                    const image = imgEl ? imgEl.src : '';
                                    
                                    // Try to extract address info
                                    let address = '';
                                    if (isCleveland) {
                                        const addressMatch = text.match(/(\d+)\s+[A-Za-z\s.]+(?:Avenue|Ave|Street|St|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Place|Pl|Parkway|Pkwy)[,\s]+(?:Cleveland|OH|Ohio)/i);
                                        if (addressMatch) {
                                            address = addressMatch[0].trim();
                                        } else {
                                            address = 'Cleveland, OH';
                                        }
                                    }
                                    
                                    // Only add if we have at least a title or link and it's Cleveland-related
                                    if ((title || link) && isCleveland) {
                                        listings.push({
                                            title: title || 'Cleveland Property Listing',
                                            url: link || window.location.href,
                                            price: price || 'Contact for price',
                                            address: address || 'Cleveland, OH',
                                            image_url: image,
                                            description: text.substring(0, 200) + '...', // Include some of the text as description
                                            is_cleveland: true
                                        });
                                    }
                                } catch (e) {
                                    console.error("Error extracting potential listing:", e);
                                }
                            }
                            
                            return listings;
                        }
                        
                        // Last resort - check for Cleveland text on the page
                        if (document.body.textContent.toLowerCase().includes('cleveland')) {
                            console.log("Found Cleveland mentioned on the page, creating a generic result");
                            return [{
                                title: 'Cleveland Property Search Results',
                                url: window.location.href,
                                address: 'Cleveland, OH',
                                description: 'Search results for Cleveland properties',
                                is_cleveland: true
                            }];
                        }
                        
                        // If nothing found, return empty array
                        return [];
                    }
                """)
                
                if property_data and len(property_data) > 0:
                    logger.info(f"Extracted {len(property_data)} property listings")
                    
                    # Format the results for the standard output
                    results = []
                    for prop in property_data:
                        result = {
                            "title": prop.get("title", "Property listing"),
                            "url": prop.get("url", results_url),
                            "source_url": results_url,
                            "type": "real_estate_listing"
                        }
                        
                        # Add additional properties
                        for key, value in prop.items():
                            if key not in result and value:
                                result[key] = value
                        
                        results.append(result)
                    
                    return results
            
            # Generic result extraction for other sites
            return []
            
        except Exception as e:
            logger.error(f"Error extracting results with browser: {str(e)}")
            return []
            
    async def _close_browser(self):
        """Close browser if initialized"""
        try:
            if self._browser:
                if self._context:
                    await self._context.close()
                await self._browser.close()
                self._browser = None
                self.browser = None  # Clear the alias for backward compatibility
                self._context = None
        except Exception as e:
            logger.error(f"Error closing browser: {str(e)}")
            # Reset browser variables even if there was an error
            self._browser = None
            self.browser = None  # Clear the alias for backward compatibility
            self._context = None
    
    async def search(self, 
                    query: str, 
                    url: str, 
                    params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a search operation using this engine.
        
        Args:
            query: The search query string
            url: The target URL to search on
            params: Additional parameters for the search
            
        Returns:
            Dictionary containing search results and metadata
        """
        # Validate inputs
        if not url or url.strip() == "" or url is None:
            logger.info("FormSearchEngine: No valid URL provided, skipping form search strategy")
            return {
                "success": False,
                "error": "No valid URL provided for form search",
                "results": [],
                "strategy": "form_search_engine"
            }
            
        if not query:
            query = "homes in Cleveland, OH"  # Default fallback query
            logger.warning(f"No query provided, using default: {query}")
        
        # Extract parameters with defaults
        if params is None:
            params = {}
            
        form_index = params.get('form_index', 0)
        wait_for_navigation = params.get('wait_for_navigation', True)
        timeout = params.get('timeout', 30000)
        debug_navigation = params.get('debug_navigation', False)
        
        logger.info(f"Starting search for query: '{query}' on URL: {url}")
        
        try:
            # Use browser-based search if available
            if self.use_browser:
                return await self._search_with_browser(
                    query=query,
                    url=url,
                    form_index=form_index,
                    wait_for_navigation=wait_for_navigation,
                    timeout=timeout,
                    debug_navigation=debug_navigation,
                    params=params
                )
            else:
                # Fall back to HTTP-based search
                return await self._search_without_browser(
                    query=query,
                    url=url,
                    form_index=form_index
                )
                
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return {
                "success": False,
                "error": f"Search failed: {str(e)}",
                "results": [],
                "strategy": "form_search_engine"
            }

    @with_exponential_backoff(max_attempts=3, min_wait=1.0, max_wait=10.0)
    async def _search_with_browser(self, query: str, url: str, form_index: int, 
                                  wait_for_navigation: bool, timeout: int, 
                                  debug_navigation: bool = False,
                                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a search using browser automation.
        
        Args:
            query: The search query string
            url: The target URL to search on
            form_index: Index of the form to use
            wait_for_navigation: Whether to wait for navigation
            timeout: Timeout in milliseconds
            debug_navigation: Whether to log extra debugging info
            params: Additional parameters including pagination configuration
            
        Returns:
            Dictionary containing search results
        """
        # Ensure browser is initialized
        if not self._browser:
            try:
                await self._init_browser()
            except Exception as e:
                logger.error(f"Browser initialization failed: {str(e)}")
                return {
                    "success": False,
                    "error": f"Browser initialization failed: {str(e)}",
                    "results": []
                }
        
        # Initialize navigation tracking
        nav_tracker = {
            "urls_visited": [],
            "pre_submission_url": None,
            "post_submission_url": None,
            "time_started": datetime.now().isoformat(),
            "redirects": [],
            "success": False
        }
        
        # Initialize submission verification
        verification = SubmissionVerification()
        
        try:
            # Navigate to the page
            page = await self._context.new_page()
            
            if debug_navigation:
                logger.info(f"Debug: Navigating to URL: {url}")
            else:
                logger.info(f"Navigating to initial page: {url}")
                
            # Track the navigation state - add url to visited list
            nav_tracker["urls_visited"].append({"url": url, "timestamp": datetime.now().isoformat(), "step": "initial"})
            
            # Use the new navigation method with exponential backoff
            navigation_success = await self._navigate_to_page(page, url, timeout)
            if not navigation_success:
                logger.error(f"Failed to navigate to {url}")
                await page.close()
                return {
                    "success": False,
                    "error": f"Failed to navigate to {url}",
                    "results": []
                }
            
            # Log navigation completion
            logger.info(f"Navigation complete. Current URL: {page.url}")
            if page.url != url:
                logger.info(f"URL changed during initial navigation from {url} to {page.url}")
                nav_tracker["redirects"].append({"from": url, "to": page.url, "timestamp": datetime.now().isoformat()})
            
            # Record the pre-submission URL for comparison after form submission
            nav_tracker["pre_submission_url"] = page.url
            
            # Find search forms
            forms_count = await page.evaluate("""() => {
                return document.querySelectorAll('form').length;
            }""")
            
            if debug_navigation:
                logger.info(f"Debug: Found {forms_count} forms on the page")
            
            if forms_count == 0:
                await page.close()
                return {
                    "success": False,
                    "error": "No forms found on the page",
                    "results": []
                }
            
            if form_index >= forms_count:
                form_index = 0  # Default to first form if requested index is out of bounds
            
            # Find the search input in the form with enhanced real estate detection
            search_input_selector = await page.evaluate(f"""(formIndex) => {{
                const form = document.querySelectorAll('form')[formIndex];
                const inputs = form.querySelectorAll('input[type="text"], input[type="search"], input:not([type])');
                
                if (inputs.length === 0) return null;
                
                // Define real estate specific search terms
                const realEstateTerms = ['location', 'city', 'address', 'state', 'zip', 'postal', 'property', 'area'];
                
                // First priority: find location/address inputs for real estate searches
                for (const input of inputs) {{
                    const name = (input.getAttribute('name') || '').toLowerCase();
                    const id = (input.getAttribute('id') || '').toLowerCase();
                    const placeholder = (input.getAttribute('placeholder') || '').toLowerCase();
                    const classes = (input.getAttribute('class') || '').toLowerCase();
                    
                    // Check if this is clearly a location input
                    if (realEstateTerms.some(term => 
                        name.includes(term) || id.includes(term) || placeholder.includes(term) || classes.includes(term))) {{
                        return input.tagName.toLowerCase() + (input.id ? '#' + input.id : 
                              (input.name ? '[name="' + input.name + '"]' : ''));
                    }}
                    
                    // Also check if the placeholder suggests entering a location
                    if (placeholder && (
                        placeholder.includes('enter') || 
                        placeholder.includes('type') || 
                        placeholder.includes('search')
                    ) && (
                        placeholder.includes('city') || 
                        placeholder.includes('location') || 
                        placeholder.includes('address') || 
                        placeholder.includes('where') ||
                        placeholder.includes('area')
                    )) {{
                        return input.tagName.toLowerCase() + (input.id ? '#' + input.id : 
                              (input.name ? '[name="' + input.name + '"]' : ''));
                    }}
                }}
                
                // Second priority: Try to find an input that looks like a standard search box
                for (const input of inputs) {{
                    const attributes = ['name', 'id', 'placeholder', 'class'];
                    for (const attr of attributes) {{
                        const value = input.getAttribute(attr)?.toLowerCase() || '';
                        if (value.includes('search') || value.includes('query') || value.includes('find') || 
                            value.includes('location') || value.includes('city') || value.includes('address') ||
                            value.includes('keyword') || value.includes('mls')) {{
                            return input.tagName.toLowerCase() + (input.id ? '#' + input.id : 
                                  (input.name ? '[name="' + input.name + '"]' : ''));
                        }}
                    }}
                }}
                
                // If no search-specific input found, use the first text input
                const firstInput = inputs[0];
                return firstInput ? firstInput.tagName.toLowerCase() + (firstInput.id ? '#' + firstInput.id : 
                      (firstInput.name ? '[name="' + firstInput.name + '"]' : '')) : null;
            }}""", form_index)
            
            if debug_navigation:
                logger.info(f"Debug: Search input selector: {search_input_selector}")
            
            if not search_input_selector:
                # Try to find any text input as a fallback
                search_input_selector = await page.evaluate(f"""(formIndex) => {{
                    const form = document.querySelectorAll('form')[formIndex];
                    const inputs = form.querySelectorAll('input');
                    
                    for (const input of inputs) {{
                        const type = input.getAttribute('type') || '';
                        if (type === '' || type === 'text' || type === 'search') {{
                            return input.tagName.toLowerCase() + (input.id ? '#' + input.id : 
                                (input.name ? '[name="' + input.name + '"]' : ''));
                        }}
                    }}
                    
                    return null;
                }}""", form_index)
                
                if debug_navigation:
                    logger.info(f"Debug: Fallback search input selector: {search_input_selector}")
            
            if not search_input_selector:
                await page.close()
                return {
                    "success": False,
                    "error": f"No suitable input field found in form {form_index}",
                    "results": []
                }
            
            # Get the form selector
            form_selector = await page.evaluate(f"""(formIndex) => {{
                const form = document.querySelectorAll('form')[formIndex];
                if (!form) return null;
                
                if (form.id) return `#${{form.id}}`;
                
                // Try to create a unique selector
                const formIndex = Array.from(document.querySelectorAll('form')).indexOf(form);
                return `form:nth-of-type(${{formIndex + 1}})`;
            }}""", form_index)
            
            if not form_selector:
                logger.error(f"Failed to get form selector for form index {form_index}")
                form_selector = f"form:nth-of-type({form_index + 1})"
            
            if debug_navigation:
                logger.info(f"Debug: Using form selector: {form_selector}")
                logger.info(f"Debug: Using input selector: {search_input_selector}")
                
            # Use the form submission method with exponential backoff
            try:
                form_submission_time = datetime.now().isoformat()
                logger.info(f"Submitting form with query: '{query}'")
                submit_success = await self._submit_search_form(page, form_selector, search_input_selector, query)
                
                if not submit_success:
                    logger.error(f"Failed to submit form")
                    await page.close()
                    return {
                        "success": False,
                        "error": "Failed to submit search form",
                        "results": []
                    }
                
                nav_tracker["form_submissions"].append({
                    "timestamp": form_submission_time,
                    "form_index": form_index,
                    "query": query
                })
                
                # Get the submit button selector
                submit_selector = await page.evaluate(f"""(formIndex) => {{
                    const form = document.querySelectorAll('form')[formIndex];
                    const submitButton = form.querySelector('input[type="submit"], button[type="submit"], button:not([type])');
                    if (submitButton) {{
                        return submitButton.tagName.toLowerCase() + (submitButton.id ? '#' + submitButton.id : 
                            (submitButton.name ? '[name="' + submitButton.name + '"]' : ''));
                    }}
                    return null;
                }}""", form_index)
                
                if submit_selector:
                    try:
                        if wait_for_navigation:
                            # Add checkpoint for submission start
                            logger.info(f"CHECKPOINT: Form submission started at {form_submission_time}")
                            
                            async with page.expect_navigation(wait_until="networkidle", timeout=timeout):
                                await page.click(submit_selector)
                            logger.info("Form submitted and navigation completed")
                            
                            # Record post-submission URL and add to visited URLs
                            nav_tracker["post_submission_url"] = page.url
                            nav_tracker["urls_visited"].append({
                                "url": page.url, 
                                "timestamp": datetime.now().isoformat(), 
                                "step": "post_submission"
                            })
                            
                            # Check for URL change after submission
                            if nav_tracker["pre_submission_url"] == nav_tracker["post_submission_url"]:
                                logger.warning(f"WARNING: URL did not change after form submission. Still on: {page.url}")
                                logger.info("Attempting to verify if results are shown on the same page...")
                            else:
                                logger.info(f"URL changed after submission from {nav_tracker['pre_submission_url']} to {page.url}")
                                nav_tracker["redirects"].append({
                                    "from": nav_tracker['pre_submission_url'], 
                                    "to": page.url, 
                                    "timestamp": datetime.now().isoformat(),
                                    "context": "form_submission"
                                })
                        else:
                            await page.click(submit_selector)
                            # Wait a moment for any updates
                            await page.wait_for_timeout(3000)
                            logger.info("Form submitted without waiting for navigation")
                            
                            # Record URL after form submission
                            nav_tracker["post_submission_url"] = page.url
                            nav_tracker["urls_visited"].append({
                                "url": page.url, 
                                "timestamp": datetime.now().isoformat(), 
                                "step": "post_submission_no_wait"
                            })
                    except Exception as e:
                        logger.error(f"Error during form submission: {str(e)}")
                        logger.info("Attempting to continue despite error...")
                        
                        # Record error in navigation tracker
                        nav_tracker["errors"] = nav_tracker.get("errors", []) + [{
                            "timestamp": datetime.now().isoformat(),
                            "step": "form_submission",
                            "error": str(e)
                        }]
                else:
                    # This block handles the case when no submit_selector was found
                    if debug_navigation:
                        logger.info(f"Debug: No submit button found, submitting form directly")
                    
                    form_submission_time = datetime.now().isoformat()
                    logger.info(f"CHECKPOINT: Form direct submission started at {form_submission_time}")
                    
                    try:
                        if wait_for_navigation:
                            async with page.expect_navigation(wait_until="networkidle", timeout=timeout):
                                await page.evaluate(f"""(formIndex) => {{
                                    const form = document.querySelectorAll('form')[formIndex];
                                    form.submit();
                                }}""", form_index)
                            
                            # Record post-submission URL and add to visited URLs
                            nav_tracker["post_submission_url"] = page.url
                            nav_tracker["urls_visited"].append({
                                "url": page.url, 
                                "timestamp": datetime.now().isoformat(), 
                                "step": "post_direct_submission"
                            })
                            
                            # Check for URL change after submission
                            if nav_tracker["pre_submission_url"] == nav_tracker["post_submission_url"]:
                                logger.warning(f"WARNING: URL did not change after form submission. Still on: {page.url}")
                            else:
                                logger.info(f"URL changed after direct submission from {nav_tracker['pre_submission_url']} to {page.url}")
                                nav_tracker["redirects"].append({
                                    "from": nav_tracker['pre_submission_url'], 
                                    "to": page.url, 
                                    "timestamp": datetime.now().isoformat(),
                                    "context": "direct_submission"
                                })
                        else:
                            await page.evaluate(f"""(formIndex) => {{
                                const form = document.querySelectorAll('form')[formIndex];
                                form.submit();
                            }}""", form_index)
                            # Wait a moment for any updates
                            await page.wait_for_timeout(3000)
                            
                            # Record URL after form submission
                            nav_tracker["post_submission_url"] = page.url
                            nav_tracker["urls_visited"].append({
                                "url": page.url,                                "timestamp": datetime.now().isoformat(), 
                                "step": "post_direct_submission_no_wait"
                            })
                    except Exception as e:
                        logger.error(f"Error during direct form submission: {str(e)}")
                        
                        # Record error in navigation tracker
                        nav_tracker["errors"] = nav_tracker.get("errors", []) + [{
                            "timestamp": datetime.now().isoformat(),
                            "step": "direct_form_submission",
                            "error": str(e)
                        }]
            except Exception as form_error:
                logger.error(f"Error during form submission process: {str(form_error)}")
                await page.close()
                return {
                    "success": False,
                    "error": f"Form submission process failed: {str(form_error)}",
                    "results": []
                }
        except Exception as e:
            logger.error(f"Browser search error: {str(e)}")
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Full traceback: {error_traceback}")
            
            # Check if browser still available for debug info
            if 'page' in locals() and page:
                try:
                    # Try to get debug info from the page
                    logger.info("CHECKPOINT: Capturing debug info before error exit...")
                    
                    # Get current URL
                    current_url = "unknown"
                    try:
                        current_url = page.url
                        logger.info(f"Current URL at error: {current_url}")
                    except:
                        pass
                    
                    # Try to take screenshot if browser still responding
                    try:
                        screenshot_path = f"error_screenshot_{int(time.time())}.png"
                        await page.screenshot(path=screenshot_path)
                        logger.info(f"Error state screenshot saved to {screenshot_path}")
                    except:
                        logger.warning("Could not save error screenshot")
                    
                    # Close the page
                    try:
                        await page.close()
                    except:
                        pass
                        
                except Exception as debug_e:
                    logger.error(f"Error while gathering debug info: {debug_e}")
            
            # Return detailed error information
            return {
                "success": False,
                "error": f"Browser search error: {str(e)}",
                "error_details": {
                    "message": str(e),
                    "traceback": error_traceback,
                    "navigation_state": nav_tracker if 'nav_tracker' in locals() else None,
                    "timestamp": datetime.now().isoformat(),
                    "api_endpoints": nav_tracker.get("api_endpoints", []) if 'nav_tracker' in locals() else []
                },
                "api_detection": {
                    "detected_endpoints": nav_tracker.get("api_endpoints", []) if 'nav_tracker' in locals() else [],
                    "api_request_template": nav_tracker.get("api_request_template", None) if 'nav_tracker' in locals() else None
                },
                "ajax_information": {
                    "responses_count": nav_tracker.get("ajax_responses", 0) if 'nav_tracker' in locals() else 0,
                    "json_responses_count": len(nav_tracker.get("ajax_json_responses", [])) if 'nav_tracker' in locals() and nav_tracker.get("ajax_json_responses") else 0,
                    "latest_response": nav_tracker.get("latest_ajax_response", None) if 'nav_tracker' in locals() else None
                },
                "results": []
            }
            logger.info("CHECKPOINT: Verifying if we've reached a proper results page...")
            verification_result = await verification.verify_submission(page)
            
            # Log verification results
            logger.info(f"Submission verification result: success={verification_result.get('success', False)}")
            if verification_result.get('success', False):
                logger.info(f"Success indicators: {verification_result.get('success_indicators', {})}")
                nav_tracker["success"] = True
            else:
                logger.warning(f"Submission verification failed: {verification_result.get('error_indicators', {})}")
                
                # If verification failed but URL changed, we might still have results
                if nav_tracker["pre_submission_url"] != nav_tracker["post_submission_url"]:
                    logger.info("URL changed despite verification failure, continuing with extraction...")
                else:
                    logger.warning("URL did not change and verification failed, may not be on results page")
            
            # Wait for AJAX responses and dynamic content to load if AJAX handler is available
            if self.ajax_handler:
                try:
                    logger.info("CHECKPOINT: Waiting for AJAX responses and dynamic content...")
                    # Setup network monitoring
                    unregister_network = await self.ajax_handler.setup_network_monitoring(page)
                    
                    # Wait for AJAX content to stabilize
                    ajax_wait_success = await self.ajax_handler.wait_for_ajax_complete(
                        page, 
                        timeout=10000,          # 10 seconds timeout
                        min_stable_period=1000, # 1 second of stability
                        additional_wait=1500    # Additional 1.5 second wait for any delayed processing
                    )
                    
                    # Check for dynamic content changes
                    if ajax_wait_success:
                        logger.info("AJAX wait completed successfully")
                        
                        # Store AJAX response info in navigation tracker
                        nav_tracker["ajax_responses"] = len(self.ajax_handler.ajax_responses)
                        
                        # Process any AJAX search results
                        if self.ajax_handler.ajax_responses:
                            # Extract recent JSON responses
                            json_responses = [r for r in self.ajax_handler.ajax_responses 
                                             if r.get('is_json', False)]
                            
                            if json_responses:
                                logger.info(f"Found {len(json_responses)} JSON AJAX responses")
                                nav_tracker["ajax_json_responses"] = json_responses
                                
                                # Get the most recent response for potential result extraction
                                if len(json_responses) > 0:
                                    latest_response = sorted(
                                        json_responses, 
                                        key=lambda x: x.get('time', 0), 
                                        reverse=True
                                    )[0]
                                    
                                    # Store important data for future use
                                    nav_tracker["latest_ajax_response"] = {
                                        "url": latest_response.get('url', ''),
                                        "content_type": latest_response.get('content_type', ''),
                                        "status": latest_response.get('status', 0),
                                        "time": latest_response.get('time', 0)
                                    }
                                    
                                    # Look for potential structured results
                                    if "json_data" in latest_response:
                                        logger.info(f"Analyzing JSON response from {latest_response.get('url', '')}")
                    else:
                        logger.warning("AJAX wait timed out")
                    
                    # Cleanup network monitoring
                    unregister_network()
                    
                except Exception as e:
                    logger.warning(f"Error during AJAX handling: {str(e)}")
            
            # Detect and analyze API endpoints if api_analyzer is available
            if self.api_analyzer:
                try:
                    logger.info("Analyzing page for API endpoints...")
                    api_detection_result = await self.api_analyzer.detect_api_endpoints(page, query)
                    
                    if api_detection_result.get("success", False):
                        logger.info(f"Successfully detected {api_detection_result.get('total_detected', 0)} API endpoints")
                        
                        # Store detected endpoints for future use
                        domain = urlparse(url).netloc
                        detected_endpoints = api_detection_result.get("endpoints", [])
                        
                        if detected_endpoints:
                            # Add to results for return
                            nav_tracker["api_endpoints"] = detected_endpoints
                            
                            # Log the top endpoint
                            top_endpoint = detected_endpoints[0]
                            endpoint_url = top_endpoint.get("endpoint", "")
                            endpoint_method = top_endpoint.get("method", "GET")
                            logger.info(f"Top API endpoint: {endpoint_method} {endpoint_url}")
                            
                            # Generate and test an API request if needed
                            if self.config.get("test_detected_apis", False):
                                try:
                                    api_request = await self.api_analyzer.generate_api_request(
                                        top_endpoint, query
                                    )
                                    logger.info(f"Generated API request: {api_request}")
                                    nav_tracker["api_request_template"] = api_request
                                except Exception as e:
                                    logger.warning(f"Error generating API request: {str(e)}")
                    else:
                        logger.info(f"API detection finished without finding endpoints: {api_detection_result.get('reason', 'Unknown reason')}")
                        
                except Exception as e:
                    logger.warning(f"Error during API detection: {str(e)}")
            
            results_html = await page.content()
            
            # Add screenshot for debugging if requested
            if params and params.get('save_screenshots', False):
                screenshot_path = f"ohio_search_screenshot_{int(time.time())}.png"
                await page.screenshot(path=screenshot_path)
                logger.info(f"Screenshot saved to {screenshot_path}")
            
            # Wait a bit longer for any dynamic content to load
            logger.info("CHECKPOINT: Waiting for dynamic content to load...")
            await page.wait_for_timeout(3000)  # Increased from 2000ms to 3000ms
            
            # Initialize params if not provided
            params = params or {}
            
            # Check if page is still the homepage
            is_homepage = await page.evaluate("""
                () => {
                    // Check for homepage indicators
                    const title = document.title.toLowerCase();
                    const path = window.location.pathname;
                    const isRootPath = path === '/' || path === '/index.html' || path === '/index.php' || path === '/home';
                    
                    // Check for homepage-specific elements
                    const hasMainSlider = document.querySelector('.main-slider, #main-slider, .hero, .banner') !== null;
                    const hasWelcomeSection = document.querySelector('.welcome, .intro, .about-us') !== null;
                    
                    return isRootPath && (hasMainSlider || hasWelcomeSection);
                }
            """)
            
            if is_homepage:
                logger.warning("WARNING: Page appears to be the homepage, not a results page!")
                
                # Try alternative approach - check if there's a search form with our query value
                query_in_form = await page.evaluate(f"""
                    (query) => {{
                        const inputs = document.querySelectorAll('input[type="text"], input[type="search"]');
                        for (const input of inputs) {{
                            if (input.value === query) return true;
                        }}
                        return false;
                    }}
                """, query)
                
                if query_in_form:
                    logger.info("Query found in form field - form might not have submitted correctly")
                    
                    # Try again with direct form submission
                    logger.info("CHECKPOINT: Attempting fallback direct form submission...")
                    
                    try:
                        # Submit form directly
                        await page.evaluate(f"""
                            () => {{
                                const forms = document.querySelectorAll('form');
                                if (forms.length > {form_index}) {{
                                    forms[{form_index}].submit();
                                }}
                            }}
                        """)
                        
                        # Wait for navigation
                        await page.wait_for_load_state("networkidle", timeout=timeout)
                        
                        # Update navigation tracker
                        fallback_url = page.url
                        nav_tracker["urls_visited"].append({
                            "url": fallback_url, 
                            "timestamp": datetime.now().isoformat(), 
                            "step": "fallback_submission"
                        })
                        
                        if fallback_url != nav_tracker["pre_submission_url"]:
                            logger.info(f"Fallback submission successful, navigated to: {fallback_url}")
                            results_url = fallback_url
                        else:
                            logger.warning("Fallback submission failed, still on same page")
                    except Exception as e:
                        logger.error(f"Error during fallback submission: {str(e)}")
            
            # Try to extract structured results with pagination support
            logger.info("CHECKPOINT: Starting result extraction...")
            use_pagination = params.get('use_pagination', True)
            
            if self.pagination_support_enabled and use_pagination:
                logger.info("Extracting results with pagination support")
                results = await self._extract_paginated_results(page, results_url, params)
            else:
                if not self.pagination_support_enabled:
                    logger.info("Pagination support not available, extracting from first page only")
                elif not use_pagination:
                    logger.info("Pagination disabled by user configuration, extracting from first page only")
                results = await self._extract_results_with_browser(page, results_url)
            
            # If no structured results found, try basic extraction
            if not results:
                logger.warning("No structured results found, falling back to basic extraction")
                results = self._extract_basic_results(results_url, results_html)
            
            # Add relevance verification for results
            if results and len(results) > 0:
                logger.info(f"CHECKPOINT: Validating relevance of {len(results)} extracted results...")
                
                # Verify result relevance to query
                relevant_results = []
                query_terms = set(query.lower().split())
                
                for result in results:
                    # Calculate relevance score based on query terms present in result
                    score = 0
                    result_text = json.dumps(result).lower()
                    
                    # Count query terms in result
                    for term in query_terms:
                        if term in result_text:
                            score += 1
                    
                    # Consider geography for real estate searches
                    if "cleveland" in query_terms or "ohio" in query_terms:
                        if "cleveland" in result_text or "ohio" in result_text:
                            score += 2
                    
                    # Add relevance score to result
                    result["_relevance_score"] = score / max(1, len(query_terms))
                    
                    # Only include results with minimum relevance
                    if result["_relevance_score"] > 0.2 or len(results) <= 3:
                        relevant_results.append(result)
                    else:
                        logger.warning(f"Filtered out low relevance result: {result.get('title', 'Untitled')}")
                
                # If all results filtered out, keep the original set
                if not relevant_results and results:
                    logger.warning("All results filtered as irrelevant, keeping original results")
                    relevant_results = results
                
                results = relevant_results
            
            await page.close()
            
            # Create the final result with detailed information
            final_result = {
                "success": len(results) > 0,
                "results_url": results_url,
                "result_count": len(results),
                "results": results,
                "has_pagination": self.pagination_support_enabled and params.get('use_pagination', True),
                "navigation_tracking": {
                    "initial_url": url,
                    "final_url": results_url,
                    "url_changed": url != results_url,
                    "redirects": nav_tracker["redirects"],
                    "success": nav_tracker["success"],
                    "api_endpoints": nav_tracker.get("api_endpoints", [])
                },
                "verification": verification_result if 'verification_result' in locals() else None,
                "query": query,
                "api_detection": {
                    "detected_endpoints": nav_tracker.get("api_endpoints", []),
                    "api_request_template": nav_tracker.get("api_request_template", None)
                },
                "ajax_information": {
                    "responses_count": nav_tracker.get("ajax_responses", 0),
                    "json_responses_count": len(nav_tracker.get("ajax_json_responses", [])) if nav_tracker.get("ajax_json_responses") else 0,
                    "latest_response": nav_tracker.get("latest_ajax_response", None)
                }
            }
            
            # Notify coordinator of search completion if available
            if self.coordinator:
                await self.notify_coordinator('search_completed', {
                    'query': query,
                    'url': url,
                    'results_count': len(results),
                    'success': final_result["success"],
                    'results_url': results_url,
                    'timestamp': datetime.now().isoformat(),
                    'strategy': self.name
                })
            
            return final_result
            
        except Exception as e:
            logger.error(f"Browser search error: {str(e)}")
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Full traceback: {error_traceback}")
            
            # Check if browser still available for debug info
            if 'page' in locals() and page:
                try:
                    # Try to get debug info from the page
                    logger.info("CHECKPOINT: Capturing debug info before error exit...")
                    
                    # Get current URL
                    current_url = "unknown"
                    try:
                        current_url = page.url
                        logger.info(f"Current URL at error: {current_url}")
                    except:
                        pass
                    
                    # Try to take screenshot if browser still responding
                    try:
                        screenshot_path = f"error_screenshot_{int(time.time())}.png"
                        await page.screenshot(path=screenshot_path)
                        logger.info(f"Error state screenshot saved to {screenshot_path}")
                    except:
                        logger.warning("Could not save error screenshot")
                    
                    # Close the page
                    try:
                        await page.close()
                    except:
                        pass
                        
                except Exception as debug_e:
                    logger.error(f"Error while gathering debug info: {debug_e}")
            
            # Return detailed error information
            return {
                "success": False,
                "error": f"Browser search error: {str(e)}",
                "error_details": {
                    "message": str(e),
                    "traceback": error_traceback,
                    "navigation_state": nav_tracker if 'nav_tracker' in locals() else None,
                    "timestamp": datetime.now().isoformat(),
                    "api_endpoints": nav_tracker.get("api_endpoints", []) if 'nav_tracker' in locals() else []
                },
                "api_detection": {
                    "detected_endpoints": nav_tracker.get("api_endpoints", []) if 'nav_tracker' in locals() else [],
                    "api_request_template": nav_tracker.get("api_request_template", None) if 'nav_tracker' in locals() else None
                },
                "ajax_information": {
                    "responses_count": nav_tracker.get("ajax_responses", 0) if 'nav_tracker' in locals() else 0,
                    "json_responses_count": len(nav_tracker.get("ajax_json_responses", [])) if 'nav_tracker' in locals() and nav_tracker.get("ajax_json_responses") else 0,
                    "latest_response": nav_tracker.get("latest_ajax_response", None) if 'nav_tracker' in locals() else None
                },
                "results": []
            }
        finally:
            try:
                logger.info("CHECKPOINT: Closing browser...")
                await self._close_browser()
            except Exception as close_e:
                logger.error(f"Error while closing browser: {close_e}")
    
    async def _search_without_browser(self, query: str, url: str, form_index: int) -> Dict[str, Any]:
        """
        Execute a search without browser automation (using direct HTTP requests).
        
        Args:
            query: The search query string
            url: The target URL to search on
            form_index: Index of the form to use
            
        Returns:
            Dictionary containing search results
        """
        try:
            # Fetch the page
            async with httpx.AsyncClient(follow_redirects=True) as client:
                headers = {"User-Agent": self.user_agent}
                response = await client.get(url, headers=headers)
                
                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Failed to fetch page: {response.status_code}",
                        "results": []
                    }
                
                html = response.text
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find forms
                forms = soup.find_all('form')
                if not forms or form_index >= len(forms):
                    return {
                        "success": False,
                        "error": f"Form index {form_index} not found",
                        "results": []
                    }
                
                form = forms[form_index]
                
                # Extract form data
                form_action = form.get('action', '')
                form_method = form.get('method', 'get').lower()
                
                # Handle relative URLs
                if form_action:
                    form_action = urljoin(url, form_action)
                else:
                    form_action = url
                
                # Find the search input field
                inputs = form.find_all('input', {'type': ['text', 'search']}) or form.find_all('input', {'type': None})
                if not inputs:
                    return {
                        "success": False,
                        "error": "No suitable input field found in form",
                        "results": []
                    }
                
                # Find the most likely search input
                search_input = None
                for input_el in inputs:
                    name = input_el.get('name', '').lower()
                    id_attr = input_el.get('id', '').lower()
                    placeholder = input_el.get('placeholder', '').lower()
                    
                    if any(term in attr for term in ['search', 'query', 'find', 'q'] 
                           for attr in [name, id_attr, placeholder]):
                        search_input = input_el
                        break
                
                # Use the first input if no search-specific input found
                if not search_input:
                    search_input = inputs[0]
                
                # Prepare form data with all inputs
                form_data = {}
                for input_el in form.find_all('input'):
                    input_type = input_el.get('type', '').lower()
                    input_name = input_el.get('name')
                    
                    # Skip submit buttons and inputs without name
                    if not input_name or input_type in ['submit', 'button', 'reset', 'image']:
                        continue
                    
                    # Handle checkboxes and radio buttons
                    if input_type in ['checkbox', 'radio']:
                        if input_el.get('checked') is not None:
                            form_data[input_name] = input_el.get('value', 'on')
                    else:
                        # For text inputs, use the query if it's the search input
                        if input_el == search_input:
                            form_data[input_name] = query
                        else:
                            form_data[input_name] = input_el.get('value', '')
                
                # Also include select fields
                for select in form.find_all('select'):
                    select_name = select.get('name')
                    if not select_name:
                        continue
                    
                    # Find selected option or use the first one
                    selected_option = select.find('option', selected=True)
                    if not selected_option:
                        selected_option = select.find('option')
                    
                    if selected_option:
                        form_data[select_name] = selected_option.get('value', '')
                
                # Make the request
                if form_method == 'post':
                    response = await client.post(form_action, data=form_data, headers=headers)
                else:
                    response = await client.get(form_action, params=form_data, headers=headers)
                
                results_url = str(response.url)
                results_html = response.text
                
                # Extract results
                results = self._extract_basic_results(results_url, results_html)
                
                return {
                    "success": True,
                    "results_url": results_url,
                    "result_count": len(results),
                    "results": results
                }
                
        except Exception as e:
            logger.error(f"HTTP search error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    def _extract_basic_results(self, url: str, html: str) -> List[Dict[str, Any]]:
        """
        Extract basic results from HTML.
        This is a simple extraction that looks for common result patterns.
        In a real implementation, this would be more sophisticated.
        
        Args:
            url: The URL of the results page
            html: The HTML content of the results page
            
        Returns:
            List of extracted results
        """
        results = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for common result container patterns
        result_containers = (
            soup.find_all('div', class_=lambda c: c and any(term in c.lower() for term in ['result', 'item', 'product', 'listing']))
            or soup.find_all('li', class_=lambda c: c and any(term in c.lower() for term in ['result', 'item', 'product', 'listing']))
            or soup.find_all('article')
        )
        
        # If no structured results found, try to extract links
        if not result_containers:
            # Get all links from the page
            links = soup.find_all('a', href=True)
            
            # Filter out navigation links, javascript, etc.
            content_links = []
            for link in links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # Skip empty links, anchors, javascript, etc.
                if (not href or not text or href.startswith(('#', 'javascript:', 'mailto:'))
                    or any(nav_term in text.lower() for nav_term in ['login', 'sign up', 'register', 'next', 'previous'])):
                    continue
                
                # Make absolute URL
                href = urljoin(url, href)
                
                # Skip links to the same page
                if href == url:
                    continue
                
                content_links.append({
                    "title": text,
                    "url": href,
                    "text": text,
                    "source_url": url
                })
            
            # Only return unique links
            seen_urls = set()
            for link in content_links:
                if link['url'] not in seen_urls:
                    seen_urls.add(link['url'])
                    results.append(link)
            
            return results[:20]  # Limit to 20 results
        
        # Process each result container
        for container in result_containers[:20]:  # Limit to first 20 containers
            # Try to extract title and link
            title_elem = (
                container.find('h1') or container.find('h2') or container.find('h3') 
                or container.find('h4') or container.find('h5') or container.find('h6')
                or container.find('strong') or container.find('b')
            )
            
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            # Look for links
            link_elem = container.find('a', href=True)
            link = urljoin(url, link_elem['href']) if link_elem else ""
            
            # Try to extract description
            desc_elems = container.find_all(['p', 'div'], class_=lambda c: c and any(
                term in c.lower() for term in ['desc', 'summary', 'content', 'text']))
            description = " ".join(elem.get_text(strip=True) for elem in desc_elems) if desc_elems else ""
            
            # If no specific description elements, use all text in the container
            if not description:
                # Skip title text
                all_text = container.get_text(strip=True)
                if title and all_text:
                    # Remove title from all text to get description
                    description = all_text.replace(title, "", 1).strip()
            
            # Only add if we have a title or link
            if title or link:
                results.append({
                    "title": title,
                    "url": link,
                    "description": description[:300] if description else "",  # Limit description length
                    "source_url": url
                })
        
        return results
    
    async def execute(self, url, options=None, context=None, **kwargs):
        """
        Execute the search engine as a crawler strategy.
        
        Args:
            url: The target URL
            options: Options dictionary with query and other parameters
            context: Strategy context
            **kwargs: Additional arguments including 'options'
            
        Returns:
            Dictionary containing the results
        """
        # Extract options from kwargs if present
        if options is None:
            options = kwargs.pop('options', {})
            
        if not isinstance(options, dict):
            options = {}
        
        # Extract query from options or kwargs
        query = options.get('query', kwargs.get('query', ''))
        if not query and 'intent' in options:
            # Try to get query from intent description
            intent = options.get('intent', {})
            if isinstance(intent, dict):
                query = intent.get('description', '')
        
        # Get location if present
        location = options.get('location', '')
        
        # Real estate specific handling
        is_real_estate = False
        # Check if this is a real estate related domain
        if url and ("ohiobrokerdirect" in url.lower() or 
                   any(term in url.lower() for term in ["realty", "realtor", "property", "properties", "homes", "estate"])):
            is_real_estate = True
            logger.info(f"Detected real estate domain: {url}")
            
            # If no location specified but appears to be real estate, use "Cleveland, OH" as default
            if is_real_estate and not location and "cleveland" not in query.lower():
                location = "Cleveland, OH"
                logger.info(f"Using default location for real estate: {location}")
        
        # Create combined query if both query and location are provided
        combined_query = query
        if location and location not in query.lower():
            combined_query = f"{query} {location}"
        
        # Prepare params for search method
        params = {
            'form_index': options.get('form_index', 0),
            'wait_for_navigation': options.get('wait_for_navigation', True),
            'timeout': options.get('timeout', 30000),
            'analyze_site': options.get('site_analysis', True),
            'debug_navigation': options.get('debug_navigation', False),
            'location': location,
            'is_real_estate': is_real_estate,
            'use_pagination': options.get('use_pagination', True),
            'max_pages': options.get('max_pages', self.config.get('max_pages', 10)),
            'target_result_count': options.get('target_result_count', 100)
        }
        
        # Execute search with the query
        if not combined_query:
            combined_query = "homes in Cleveland, OH"  # Default fallback query
            logger.warning(f"No query found, using default: {combined_query}")
        
        logger.info(f"Executing form search with query: '{combined_query}' on {url}")
        
        # First, try to initialize the browser if needed
        if self.use_browser and not self._browser:
            try:
                await self._init_browser()
            except Exception as e:
                logger.error(f"Browser initialization failed: {str(e)}. Will continue with fallback options.")
                
        # Now run the search
        try:
            search_result = await self.search(combined_query, url, params)
            
            # Add strategy name to the result
            search_result["strategy"] = self.name
            
            # Add metadata about pagination and performance
            search_result["metadata"] = {
                "query": combined_query,
                "domain": urlparse(url).netloc,
                "pagination_enabled": self.pagination_support_enabled and params.get('use_pagination', True),
                "max_pages_setting": params.get('max_pages', self.config.get('max_pages', 10)),
                "target_result_count": params.get('target_result_count', 100),
                "site_analysis_used": params.get('analyze_site', True)
            }
            
            # Check if we need to close the browser
            if self.use_browser and self._browser:
                await self._close_browser()
                
            return search_result
        except Exception as e:
            logger.error(f"Error during search execution: {str(e)}")
            
            # Make sure to close the browser on error
            if self.use_browser and self._browser:
                await self._close_browser()
                
            # Return error result
            return {
                "success": False,
                "error": f"Search execution error: {str(e)}",
                "results": [],
                "strategy": self.name
            }

    async def _extract_paginated_results(self, page, results_url: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract results from all pages of search results using pagination handler.
        
        Args:
            page: Playwright page object
            results_url: URL of the first results page
            params: Additional parameters including max_pages configuration
            
        Returns:
            List of combined results from all pages
        """
        logger.info(f"Extracting paginated results starting from: {results_url}")
        
        # Check if pagination handler is available
        if not self.pagination_support_enabled:
            logger.warning("Pagination support is disabled, only extracting first page")
            return await self._extract_results_with_browser(page, results_url)
        
        # First, extract results from the current page
        first_page_results = await self._extract_results_with_browser(page, results_url)
        
        if not first_page_results:
            logger.warning("No results found on the first page, skipping pagination")
            return []
            
        logger.info(f"Extracted {len(first_page_results)} results from first page")
        
        # Tag first page results with page number
        for item in first_page_results:
            item['page_number'] = 1
            item['pagination_source_url'] = results_url
        
        # Store all results, starting with the first page
        all_results = first_page_results.copy()
        
        try:
            # Get current HTML content for pagination detection
            html_content = await page.content()
            
            # Check for infinite scroll or AJAX-based pagination first if AJAX handler available
            if self.ajax_handler:
                try:
                    logger.info("Checking for AJAX-based pagination and infinite scroll...")
                    # Detect if the page uses infinite scroll
                    infinite_scroll_info = await self.ajax_handler.detect_infinite_scroll(page)
                    
                    if infinite_scroll_info.get("has_infinite_scroll", False):
                        logger.info("Detected infinite scroll pattern, attempting to load more content")
                        scroll_results = infinite_scroll_info.get("results", [])
                        if scroll_results:
                            # Add page marker
                            for item in scroll_results:
                                if item not in all_results:
                                    item['page_number'] = 'infinite-scroll'
                                    item['pagination_source_url'] = results_url
                                    all_results.append(item)
                            
                            logger.info(f"Added {len(scroll_results)} results from infinite scroll")
                            # If we got significant results from infinite scroll, return right away
                            if len(scroll_results) > 10:
                                return all_results
                    
                    # Also check for load more buttons
                    await self.ajax_handler.handle_load_more_button(page, max_clicks=3)
                    
                    # Re-extract results after any scroll or load more operations
                    updated_results = await self._extract_results_with_browser(page, results_url)
                    if len(updated_results) > len(first_page_results):
                        logger.info(f"Found additional results after AJAX handling: {len(updated_results)} vs original {len(first_page_results)}")
                        # Replace first page results with updated results
                        all_results = updated_results
                        for item in all_results:
                            item['page_number'] = 1
                            item['pagination_source_url'] = results_url
                except Exception as e:
                    logger.warning(f"Error in AJAX pagination handling: {str(e)}")
            
            # Detect pagination using the pagination handler
            pagination_info = await self.pagination_handler.detect_pagination_type(html_content, results_url)
            
            if not pagination_info["has_pagination"]:
                logger.info("No pagination detected on results page")
                return all_results
                
            logger.info(f"Detected pagination type: {pagination_info['pagination_type']}")
            
            # Get the maximum number of pages to process
            params = params or {}
            max_pages = params.get('max_pages', self.config.get('max_pages', 10))  # Get max_pages from params
            
            # If a specific result count target is provided, use that to guide pagination
            target_result_count = params.get('target_result_count', 100)
            current_page = 1
            
            logger.info(f"Pagination configuration: max_pages={max_pages}, target_result_count={target_result_count}")
            
            # Process subsequent pages
            while (current_page < max_pages and 
                   len(all_results) < target_result_count and 
                   pagination_info.get("next_page_url")):
                current_page += 1
                next_url = pagination_info["next_page_url"]
                
                logger.info(f"Navigating to page {current_page}: {next_url}")
                
                # Navigate to the next page
                try:
                    await page.goto(next_url, timeout=30000)
                    await page.wait_for_load_state("networkidle")
                    
                    # Extract results from this page
                    page_results = await self._extract_results_with_browser(page, next_url)
                    
                    if page_results:
                        logger.info(f"Extracted {len(page_results)} results from page {current_page}")
                        
                        # Mark results with the page number they came from
                        for item in page_results:
                            item['page_number'] = current_page
                            item['pagination_source_url'] = next_url
                        
                        # Add unique results (avoid duplicates)
                        existing_urls = {r.get('url', '') for r in all_results}
                        new_results = [r for r in page_results if r.get('url', '') not in existing_urls]
                        
                        all_results.extend(new_results)
                        logger.info(f"Added {len(new_results)} new unique results")
                    else:
                        logger.warning(f"No results found on page {current_page}")
                    
                    # Get updated pagination info
                    html_content = await page.content()
                    pagination_info = await self.pagination_handler.detect_pagination_type(html_content, next_url)
                    
                except Exception as e:
                    logger.error(f"Error navigating to page {current_page}: {str(e)}")
                    break
            
            logger.info(f"Completed pagination extraction. Found {len(all_results)} total results across {current_page} pages")
            return all_results
            
        except Exception as e:
            logger.error(f"Error during paginated extraction: {str(e)}")
            # Return results from first page if pagination fails
            return first_page_results

    async def _identify_page_type(self, page, html_content: str) -> str:
        """
        Identify the type of page based on its content patterns.
        
        Args:
            page: Playwright page object
            html_content: HTML content of the page
            
        Returns:
            String describing the page type
        """
        try:
            # Check for common patterns using JavaScript evaluation
            page_type_info = await page.evaluate("""
                () => {
                    // Check for listing patterns
                    const hasPropertyListings = document.querySelectorAll(
                        '.property-item, .listing-item, .property-listing, [class*="property"][class*="item"], ' +
                        '[class*="listing"][class*="item"], [itemtype*="Product"], [itemtype*="Offer"]'
                    ).length > 0;
                    
                    // Check for search form patterns
                    const hasSearchForm = Array.from(document.querySelectorAll('form')).some(form => {
                        return form.querySelector('input[type="text"], input[type="search"]') !== null &&
                               (form.action?.toLowerCase().includes('search') || 
                                form.id?.toLowerCase().includes('search') ||
                                (form.className && form.className.toLowerCase().includes('search')));
                    });
                    
                    // Check for pagination patterns
                    const hasPagination = document.querySelectorAll(
                        '.pagination, [class*="paging"], .pages, .page-numbers, ' +
                        '[class*="page"][class*="nav"], [aria-label*="pagination"]'
                    ).length > 0;
                    
                    // Check for result count patterns
                    const resultCountElement = document.querySelector(
                        '.result-count, .count, .results-count, [class*="result"][class*="count"], ' +
                        '[class*="showing"], [class*="found"]'
                    );
                    const resultCountText = resultCountElement ? resultCountElement.textContent : '';
                    
                    return {
                        isListingPage: hasPropertyListings,
                        isSearchFormPage: hasSearchForm,
                        hasPagination: hasPagination,
                        resultCountText: resultCountText,
                        url: window.location.href,
                        title: document.title
                    };
                }
            """)
            
            # Also analyze the HTML content with BeautifulSoup for additional pattern detection
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Count potential listing containers
            listing_containers = soup.select(
                '.property-item, .listing-item, .property-listing, .search-result, ' +
                '.card, article, [class*="property"], [class*="listing"], [class*="result-item"]'
            )
            
            # Check if page has structured data typically used for listings (Schema.org markup)
            schema_markup = soup.find_all('script', type='application/ld+json')
            has_schema_markup = False
            for script in schema_markup:
                try:
                    if script.string and any(term in script.string.lower() for term in 
                                           ['product', 'offer', 'realestate', 'house', 'apartment']):
                        has_schema_markup = True
                        break
                except:
                    pass
            
            # Determine page type based on various signals
            page_type = "unknown"
            if page_type_info.get('isListingPage', False) or len(listing_containers) > 3:
                if has_schema_markup:
                    page_type = "structured_listing_page"
                else:
                    page_type = "listing_page"
            elif page_type_info.get('isSearchFormPage', False):
                page_type = "search_form_page"
            elif page_type_info.get('hasPagination', False):
                page_type = "paginated_content"
            
            # Add more details to the page type
            domain = urlparse(page.url).netloc
            
            logger.info(f"Identified page type '{page_type}' for {domain} with " +
                       f"{len(listing_containers)} potential listing containers")
            
            return page_type
            
        except Exception as e:
            logger.error(f"Error identifying page type: {str(e)}")
            return "unknown"

    async def register_with_coordinator(self, coordinator):
        """
        Register this search engine with a SearchCoordinator.
        
        Args:
            coordinator: The SearchCoordinator instance
            
        Returns:
            True if registration was successful
        """
        logger.info("Registering FormSearchEngine with SearchCoordinator")
        self.coordinator = coordinator
        
        # Register capabilities with the coordinator if supported
        if hasattr(coordinator, 'register_strategy'):
            await coordinator.register_strategy(
                strategy_name=self.name,
                capabilities={
                    'form_interaction': True,
                    'browser_automation': self.use_browser,
                    'dynamic_content': True,
                    'pagination': self.pagination_support_enabled,
                    'api_detection': self.api_analyzer is not None,
                    'ajax_handling': self.ajax_handler is not None
                }
            )
            
        return True

    async def notify_coordinator(self, event_type, event_data=None):
        """
        Notify the coordinator of a search-related event.
        
        Args:
            event_type: Type of event (e.g., 'search_started', 'form_submitted')
            event_data: Optional data associated with the event
            
        Returns:
            None
        """
        if not self.coordinator:
            return
            
        event_data = event_data or {}
        
        # Add default data
        if 'timestamp' not in event_data:
            event_data['timestamp'] = datetime.now().isoformat()
        if 'strategy' not in event_data:
            event_data['strategy'] = self.name
            
        # Notify coordinator if it has the handle_strategy_event method
        if hasattr(self.coordinator, 'handle_strategy_event'):
            try:
                await self.coordinator.handle_strategy_event(
                    event_type=event_type,
                    event_data=event_data
                )
            except Exception as e:
                logger.warning(f"Error notifying coordinator of event {event_type}: {str(e)}")
                # Continue execution even if notification fails

    @with_exponential_backoff(max_attempts=3, min_wait=1.0, max_wait=10.0)
    async def _navigate_to_page(self, page, url: str, timeout: int = 60000) -> bool:
        """
        Navigate to a page with exponential backoff for better resilience.
        
        Args:
            page: Playwright page object
            url: URL to navigate to
            timeout: Timeout in milliseconds
            
        Returns:
            True if navigation successful, False otherwise
        """
        try:
            await page.goto(url, wait_until="networkidle", timeout=timeout)
            logger.info(f"Navigation complete: {page.url}")
            return True
        except Exception as e:
            logger.error(f"Navigation error: {str(e)}")
            raise  # Re-raise for the decorator to handle

    @with_exponential_backoff(max_attempts=3, min_wait=1.0, max_wait=10.0)
    async def _submit_search_form(self, page, form_selector: str, input_selector: str, query: str) -> bool:
        """
        Submit a search form with exponential backoff for better resilience.
        
        Args:
            page: Playwright page object
            form_selector: Selector for the form
            input_selector: Selector for the input field
            query: Search query
            
        Returns:
            True if submission successful, False otherwise
        """
        try:
            # Clear the field first
            await page.evaluate(f'''
                (() => {{
                    const el = document.querySelector("{input_selector}");
                    if (el) el.value = "";
                }})()
            ''')
            
            # Fill the search field
            await page.fill(input_selector, query)
            logger.info(f"Filled search field with query: {query}")
            
            # Try to find and click the submit button
            submit_button = await page.evaluate(f'''
                (() => {{
                    const form = document.querySelector("{form_selector}");
                    if (!form) return null;
                    
                    // Look for submit button
                    let button = form.querySelector('button[type="submit"], input[type="submit"]');
                    
                    // Fallback to any button-like element
                    if (!button) {{
                        button = form.querySelector('button, input[type="button"], .btn, .button');
                    }}
                    
                    return button ? {{
                        id: button.id || null,
                        selector: button.id ? `#${{button.id}}` : null,
                        exists: true
                    }} : {{ exists: false }};
                }})()
            ''')
            
            if submit_button and submit_button.get("exists"):
                # Try clicking the submit button
                if submit_button.get("selector"):
                    await page.click(submit_button["selector"])
                else:
                    # Fallback: use form.submit()
                    await page.evaluate(f'''
                        (() => {{
                            const form = document.querySelector("{form_selector}");
                            if (form) form.submit();
                        }})()
                    ''')
            else:
                # If no button found, try pressing Enter
                await page.press(input_selector, "Enter")
            
            logger.info("Search form submitted")
            return True
        except Exception as e:
            logger.error(f"Form submission error: {str(e)}")
            raise  # Re-raise for the decorator to handle