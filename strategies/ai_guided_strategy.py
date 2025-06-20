"""
AI-guided strategy implementation for SmartScrape.

This file implements a strategy that uses AI assistance to handle complex search scenarios
and adapt to challenging websites. It leverages large language models to understand website
structure and develop search strategies on the fly.
"""

import logging
import asyncio
import re
import json
import time
import os
import requests
from typing import Dict, List, Any, Optional, Set, TYPE_CHECKING
from urllib.parse import urljoin, urlparse, quote

if TYPE_CHECKING:
    from core.strategy_context import StrategyContext

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    logging.warning("google-generativeai not installed, AI-guided strategy will use fallback methods")

# Check if timeout utilities are available
try:
    from utils.timeout_utils import calculate_adaptive_timeout, get_remaining_time, check_approaching_deadline
    from utils.ai_timeout_monitor import AITimeoutTracker
    
    # Define a function to record AI service requests
    def record_ai_request(service_name, operation_type):
        """Record an AI service request for timeout monitoring"""
        try:
            tracker = AITimeoutTracker()
            tracker.record_request(service_name, operation_type)
        except Exception as e:
            logging.warning(f"Error recording AI request: {e}")
    
    # Define a function to record AI service timeouts
    def record_ai_timeout(service_name, operation_type, timeout_seconds):
        """Record an AI service timeout for monitoring"""
        try:
            tracker = AITimeoutTracker()
            tracker.record_timeout(service_name, operation_type, timeout_seconds)
        except Exception as e:
            logging.warning(f"Error recording AI timeout: {e}")
            
    # Define a function to get suggested timeout settings
    def get_suggested_timeout_settings():
        """Get data-driven timeout settings based on historical performance"""
        try:
            tracker = AITimeoutTracker()
            return tracker.get_timeout_suggestions()
        except Exception as e:
            logging.warning(f"Error getting timeout suggestions: {e}")
            return {}
    
    TIMEOUT_UTILS_AVAILABLE = True
except ImportError:
    TIMEOUT_UTILS_AVAILABLE = False
    
    # Create dummy functions that do nothing if the modules aren't available
    def record_ai_request(service_name, operation_type):
        pass
        
    def record_ai_timeout(service_name, operation_type, timeout_seconds):
        pass
        
    def get_suggested_timeout_settings():
        return {}

from playwright.async_api import Page, async_playwright
from bs4 import BeautifulSoup

from strategies.base_strategy import BaseStrategy
from strategies.dom_strategy import DOMStrategy
from strategies.core.strategy_types import StrategyType, StrategyCapability, strategy_metadata
from components.search_automation import SearchAutomator
from ai_helpers.prompt_generator import PromptGenerator
from strategies.mixins.error_handling_mixin import CircuitOpenError

logger = logging.getLogger(__name__)

@strategy_metadata(
    strategy_type=StrategyType.SPECIAL_PURPOSE,
    capabilities={
        StrategyCapability.AI_ASSISTED,
        StrategyCapability.FORM_INTERACTION, 
        StrategyCapability.API_INTERACTION,
        StrategyCapability.JAVASCRIPT_EXECUTION,
        StrategyCapability.PAGINATION_HANDLING,
        StrategyCapability.DYNAMIC_CONTENT,
        StrategyCapability.ROBOTS_TXT_ADHERENCE,
        StrategyCapability.ERROR_HANDLING,
        StrategyCapability.SCHEMA_EXTRACTION,
    },
    description="AI-guided strategy that uses machine learning models to analyze websites, develop dynamic search strategies, and adapt to complex search interfaces."
)
class AIGuidedStrategy(BaseStrategy):
    """
    AI-guided strategy that uses advanced AI models to analyze and interact with complex search interfaces.
    
    This strategy:
    - Uses large language models to analyze website structure
    - Develops custom search approaches for challenging sites
    - Combines multiple search techniques based on context
    - Identifies and adapts to non-standard search patterns
    - Refines search approach based on initial results
    """
    
    def __init__(self, context: Optional['StrategyContext'] = None, **kwargs):
        """
        Initialize the AI-guided strategy with robust resource management and error handling.
        
        Args:
            context: The strategy context containing shared services and configuration
            **kwargs: Additional keyword arguments for configuration
        """
        # Initialize with default config
        max_depth = kwargs.pop('max_depth', 2)
        max_pages = kwargs.pop('max_pages', 100)
        include_external = kwargs.pop('include_external', False)
        user_prompt = kwargs.pop('user_prompt', "")
        filter_chain = kwargs.pop('filter_chain', None)
        
        # Initialize the base strategy with extracted parameters
        super().__init__(
            max_depth=max_depth,
            max_pages=max_pages, 
            include_external=include_external,
            user_prompt=user_prompt,
            filter_chain=filter_chain
        )
        
        # Store the context separately since BaseStrategy doesn't use it
        self.context = context
        
        # Initialize results storage for BaseStrategy interface compliance
        self._results = []
        
        # Use data-driven timeout suggestions if available
        suggested_timeouts = {}
        if TIMEOUT_UTILS_AVAILABLE:
            try:
                suggested_timeouts = get_suggested_timeout_settings()
                if suggested_timeouts:
                    logger.info(f"Using data-driven timeout settings: {suggested_timeouts.get('explanation', 'Based on historical data')}")
            except Exception as e:
                logger.warning(f"Error getting suggested timeout settings: {e}")
        
        # Default configuration
        self.config = {
            'use_ai_model': kwargs.get('use_ai_model', True),
            'fallback_to_dom': kwargs.get('fallback_to_dom', True),
            'handle_pagination': kwargs.get('handle_pagination', True),
            'cache_instructions': kwargs.get('cache_instructions', True),
            'model_name': kwargs.get('model_name', 'gemini-pro'),
            'temperature': kwargs.get('temperature', 0.2),
            'max_ai_retries': kwargs.get('max_ai_retries', 3),
            'circuit_breaker_threshold': kwargs.get('circuit_breaker_threshold', 5),
            'circuit_breaker_reset': kwargs.get('circuit_breaker_reset', 300),  # 5 minutes
            'proxy_enabled': kwargs.get('proxy_enabled', True),
            # Timeout configurations - use suggested values if available
            'global_timeout': kwargs.get('global_timeout', suggested_timeouts.get('global_timeout', 90)),
            'ai_extraction_timeout': kwargs.get('ai_extraction_timeout', suggested_timeouts.get('ai_extraction_timeout', 45)),
            'ai_response_timeout': kwargs.get('ai_response_timeout', suggested_timeouts.get('ai_response_timeout', 25)),
            'search_form_timeout': kwargs.get('search_form_timeout', 30),
            'http_request_timeout': kwargs.get('http_request_timeout', 20)
        }
        
        # Set up prompt generator and search automator
        self.prompt_generator = PromptGenerator()
        self.search_automator = SearchAutomator()
        
        # Create a DOM strategy as fallback
        self.dom_strategy = DOMStrategy()
        
        # Cache for AI-generated instructions
        self.site_instructions_cache = {}
        
        # Initialize HTML service
        self.html_service = self._get_html_service_from_context()
        
        # Try to initialize the AI model if available
        self.ai_model = None
        self.ai_service = None
        
        # Get AI service from registry if available
        registry = self._get_service_registry()
        if registry:
            try:
                self.ai_service = registry.get_service("ai_service")
                logger.info("Using AI service from service registry")
            except (KeyError, AttributeError):
                logger.warning("AI service not available in registry")
        
        if not self.ai_service and GOOGLE_AI_AVAILABLE and self.config.get('use_ai_model', True):
            self._setup_ai_model()

    def _get_service_registry(self):
        """Get the service registry instance."""
        try:
            from core.service_registry import ServiceRegistry
            return ServiceRegistry()
        except ImportError:
            logger.warning("Could not import ServiceRegistry")
            return None
            
    def _get_html_service_from_context(self):
        """
        Get the HTML service from the strategy context or service registry.
        
        Returns:
            HTML service instance or None if not available
        """
        # First try to get HTML service from context if available
        if self.context:
            try:
                return self.context.get_service("html_service")
            except (KeyError, AttributeError) as e:
                logger.debug(f"Could not get HTML service from context: {e}")
        
        # Fallback to service registry
        registry = self._get_service_registry()
        if registry:
            try:
                return registry.get_service("html_service")
            except (KeyError, AttributeError) as e:
                logger.debug(f"Could not get HTML service from registry: {e}")
        
        # If all else fails, return None
        logger.warning("HTML service not available - some functionality may be limited")
        return None
            
    def _setup_ai_model(self) -> None:
        """Set up a direct AI model for guidance if no AI service is available."""
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                
                # Configure the model
                generation_config = {
                    "temperature": self.config.get('temperature', 0.2),
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }
                
                # Use Gemini Pro model for text generation
                model_name = self.config.get('model_name', 'gemini-pro')
                self.ai_model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config
                )
                logger.info(f"Successfully initialized {model_name} model")
            else:
                logger.warning("No Google API key found in environment variables")
        except Exception as e:
            # Use our enhanced error handling
            error_context = {
                'operation': 'ai_model_setup',
                'model': self.config.get('model_name', 'gemini-pro')
            }
            error_info = self._classify_error(e, error_context)
            logger.error(f"Error setting up AI model: {str(e)} - {error_info.get('category', 'unknown')}")
    
    async def execute(self, crawler, start_url, extraction_config=None):
        """
        Execute the AI-guided strategy on a URL with a global timeout to prevent indefinite hangs.
        """
        # Use a reasonable default timeout, but allow override from config
        default_timeout = 90  # 90 seconds default (1.5 minutes)
        timeout_seconds = self.config.get('global_timeout', default_timeout)
        
        # Ensure the timeout is reasonable (between 30 seconds and 5 minutes)
        timeout_seconds = max(30, min(timeout_seconds, 300))
        
        # Log the timeout setting
        logger.info(f"Using global timeout of {timeout_seconds} seconds for {start_url}")
        
        # Track execution progress for reporting
        self.execution_progress = {
            'status': 'starting',
            'phase': 'initialization',
            'start_time': time.time(),
            'timeout_at': time.time() + timeout_seconds,
            'url': start_url
        }
        
        # Cancel flag for coordinated cancellation of subtasks
        self.cancel_execution = False
        
        try:
            # Create a task for the execution
            task = asyncio.create_task(self._execute_inner(crawler, start_url, extraction_config))
            
            # Set up a timeout task
            try:
                result = await asyncio.wait_for(task, timeout=timeout_seconds)
                
                # Ensure the completed flag is set
                if isinstance(result, dict) and 'completed' not in result:
                    result['completed'] = True
                
                # Update progress
                self.execution_progress['status'] = 'completed'
                self.execution_progress['end_time'] = time.time()
                
                return result
                
            except asyncio.TimeoutError:
                # Set cancel flag to notify any running subtasks
                self.cancel_execution = True
                
                # Update progress
                self.execution_progress['status'] = 'timeout'
                self.execution_progress['end_time'] = time.time()
                
                # Log timeout with details about what was happening
                current_phase = self.execution_progress.get('phase', 'unknown')
                logger.error(f"AI-guided strategy timed out after {timeout_seconds} seconds for {start_url} during {current_phase} phase")
                
                # Get any partial results that might have been collected
                resource_stats = self.get_resource_stats()
                error_stats = self.get_error_stats()
                
                # Prepare error result with progress information
                error_result = {
                    "error": f"Timeout after {timeout_seconds} seconds during {current_phase} phase",
                    "resource_stats": resource_stats,
                    "error_stats": error_stats,
                    "url": start_url,
                    "completed": False,
                    "progress": self.execution_progress,
                    "timestamp": time.time(),
                    "results": self.results if hasattr(self, 'results') and self.results else []
                }
                
                # Display timeout message if not running in adaptive scraper
                if not crawler or not hasattr(crawler, 'is_adaptive_scraper') or not crawler.is_adaptive_scraper:
                    print(f"\n{'='*60}")
                    print(f"SCRAPING TIMEOUT FOR: {start_url}")
                    print(f"{'='*60}")
                    print(f"Timeout occurred during: {current_phase}")
                    
                    # Show any partial results if available
                    if hasattr(self, 'results') and self.results:
                        print(f"Partial results available: {len(self.results)} items found before timeout")
                    else:
                        print("No results due to timeout.")
                    
                    # Provide guidance for timeout remediation
                    print(f"\nPossible solutions:")
                    print(f"1. Increase the global_timeout setting (current: {timeout_seconds}s)")
                    print(f"2. Try again later when the service is less busy")
                    print(f"3. Use a more specific search query to reduce processing time")
                    if current_phase == 'ai_extraction':
                        print(f"4. Consider using a different extraction strategy")
                    
                    print(f"{'='*60}\n")
                
                return error_result
        except Exception as e:
            # Handle any other exceptions during execution
            logger.error(f"Error in AI-guided strategy execution: {str(e)}")
            
            # Update progress
            self.execution_progress['status'] = 'error'
            self.execution_progress['error'] = str(e)
            self.execution_progress['end_time'] = time.time()
            
            # Get any stats that are available
            resource_stats = self.get_resource_stats() if hasattr(self, 'get_resource_stats') else {}
            error_stats = self.get_error_stats() if hasattr(self, 'get_error_stats') else {}
            
            error_result = {
                "error": str(e),
                "resource_stats": resource_stats,
                "error_stats": error_stats,
                "url": start_url,
                "completed": False,
                "progress": self.execution_progress,
                "timestamp": time.time(),
                "results": self.results if hasattr(self, 'results') and self.results else []
            }
            
            # Display error message if not running in adaptive scraper
            if not crawler or not hasattr(crawler, 'is_adaptive_scraper') or not crawler.is_adaptive_scraper:
                print(f"\n{'='*60}")
                print(f"SCRAPING ERROR FOR: {start_url}")
                print(f"{'='*60}")
                print(f"Error: {str(e)}")
                print(f"{'='*60}\n")
            
            return error_result

    async def _execute_inner(self, crawler, start_url, extraction_config=None):
        """
        Execute the AI-guided strategy on a URL.
        
        This method implements the main entry point for the strategy,
        analyzing the page and extracting data with AI assistance.
        
        Args:
            crawler: The crawler instance
            start_url: The URL to process
            extraction_config: Configuration for extraction
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing AI-guided strategy for URL: {start_url}")
        
        # Parse and store the main domain
        self.main_domain = urlparse(start_url).netloc
        
        # Initialize results if not already there
        if not hasattr(self, 'results'):
            self.results = []
        
        try:
            # Make the request with full error handling and resource management
            try:
                # Get the domain for resource management
                domain = urlparse(start_url).netloc
                
                # Update progress tracking
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'rate_limiting'
                    self.execution_progress['domain'] = domain
                
                # Handle rate limiting for the domain
                self._handle_rate_limiting(domain)
                
                # Check if execution has been cancelled
                if hasattr(self, 'cancel_execution') and self.cancel_execution:
                    logger.warning(f"Execution cancelled for {start_url}")
                    return {"cancelled": True, "url": start_url, "timestamp": time.time()}
                
                # Update progress
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'circuit_breaker_check'
                
                # Check circuit breaker
                if not self._check_circuit_breaker(domain):
                    logger.warning(f"Circuit breaker open for {domain}, skipping request")
                    raise CircuitOpenError(f"Circuit breaker open for {domain}")
                
                # Check if execution has been cancelled
                if hasattr(self, 'cancel_execution') and self.cancel_execution:
                    logger.warning(f"Execution cancelled for {start_url}")
                    return {"cancelled": True, "url": start_url, "timestamp": time.time()}
                
                # First, check if this is a site that requires search to get listings
                search_required = False
                search_terms = None
                
                # Update progress
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'extraction_config_analysis'
                
                # If we have specific keywords or location from extraction config, use them
                if extraction_config:
                    if "keywords" in extraction_config:
                        search_terms = extraction_config["keywords"]
                    elif "extract_description" in extraction_config and "Cleveland" in extraction_config["extract_description"]:
                        search_terms = ["Cleveland, OH"]
                    elif "location_data" in extraction_config and extraction_config["location_data"].get("city"):
                        city = extraction_config["location_data"].get("city")
                        state = extraction_config["location_data"].get("state")
                        if city and state:
                            search_terms = [f"{city}, {state}"]
                
                # Check if execution has been cancelled
                if hasattr(self, 'cancel_execution') and self.cancel_execution:
                    logger.warning(f"Execution cancelled for {start_url}")
                    return {"cancelled": True, "url": start_url, "timestamp": time.time()}
                
                # Update progress
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'search_form_detection'
                
                # Try to find and use search forms to get better results
                search_result = None
                try:
                    search_form_timeout = self.config.get('search_form_timeout', 45)  # Default 45 seconds for search form operation
                    search_form_task = self._find_and_use_search_forms(start_url, search_terms)
                    search_result = await asyncio.wait_for(search_form_task, timeout=search_form_timeout)
                    
                    if hasattr(self, 'execution_progress'):
                        self.execution_progress['search_form_completed'] = True
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Search form detection timed out after {search_form_timeout} seconds, continuing with direct request")
                    if hasattr(self, 'execution_progress'):
                        self.execution_progress['search_form_timeout'] = True
                except Exception as e:
                    logger.warning(f"Error in search form detection: {e}, continuing with direct request")
                    if hasattr(self, 'execution_progress'):
                        self.execution_progress['search_form_error'] = str(e)
                
                # Check if execution has been cancelled
                if hasattr(self, 'cancel_execution') and self.cancel_execution:
                    logger.warning(f"Execution cancelled for {start_url}")
                    return {"cancelled": True, "url": start_url, "timestamp": time.time()}
                
                # Update progress
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'html_content_retrieval'
                
                # If search was successful, use the results HTML
                if search_result and search_result.get('success') and search_result.get('html_content'):
                    logger.info(f"Successfully used search forms, using search results content")
                    html_content = search_result.get('html_content')
                    
                    # Record success for circuit breaker
                    self._record_success(domain)
                else:
                    # Make the direct request using our robust implementation
                    logger.info(f"Making HTTP request to {start_url}...")
                    
                    # Update progress
                    if hasattr(self, 'execution_progress'):
                        self.execution_progress['phase'] = 'direct_http_request'
                    
                    # Use requests directly for troubleshooting
                    import requests
                    try:
                        # First try with direct requests to see if it's a connection issue
                        direct_response = requests.get(
                            start_url, 
                            headers={
                                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                            },
                            timeout=30
                        )
                        logger.info(f"Direct request to {start_url} succeeded with status {direct_response.status_code}")
                    except Exception as direct_req_error:
                        logger.warning(f"Direct request failed, will try with _make_request: {str(direct_req_error)}")
                    
                    # Check if execution has been cancelled
                    if hasattr(self, 'cancel_execution') and self.cancel_execution:
                        logger.warning(f"Execution cancelled for {start_url}")
                        return {"cancelled": True, "url": start_url, "timestamp": time.time()}
                    
                    # Now try the regular method
                    response = self._make_request(start_url)
                    if not response or not hasattr(response, 'text'):
                        logger.error(f"Invalid response from _make_request for URL: {start_url}")
                        raise ValueError(f"Invalid response from _make_request for URL: {start_url}")
                    html_content = response.text
                    current_url = start_url  # Track the URL we're actually processing
                    logger.info(f"Successfully retrieved HTML content ({len(html_content)} bytes) from {start_url}")
                    
                    # Record success for circuit breaker
                    self._record_success(domain)
                
                # Check if execution has been cancelled
                if hasattr(self, 'cancel_execution') and self.cancel_execution:
                    logger.warning(f"Execution cancelled for {start_url}")
                    return {"cancelled": True, "url": start_url, "timestamp": time.time()}
                
                # Update progress
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'property_listing_detection'
                
                # First, check if this is a potential real estate listings page
                property_listings = None
                if hasattr(self, 'user_prompt') and self.user_prompt:
                    real_estate_terms = ['home', 'house', 'property', 'real estate', 'listing']
                    # Make sure user_prompt is a string before calling .lower()
                    if isinstance(self.user_prompt, str):
                        should_check_listings = any(term in self.user_prompt.lower() for term in real_estate_terms)
                    elif isinstance(self.user_prompt, dict):
                        # Handle dictionary user_prompt
                        user_prompt_text = str(self.user_prompt)
                        should_check_listings = any(term in user_prompt_text.lower() for term in real_estate_terms)
                    else:
                        should_check_listings = False
                        
                    if should_check_listings:
                        logger.info("Checking for property listings in the page...")
                    
                    # Set a timeout for property detection to avoid hanging
                    try:
                        property_detection_task = self._detect_real_estate_listings(html_content)
                        if isinstance(property_detection_task, list):  # Not a coroutine
                            property_listings = property_detection_task
                        else:  # It's a coroutine
                            property_listings = await asyncio.wait_for(property_detection_task, timeout=30)
                    except asyncio.TimeoutError:
                        logger.warning("Property listing detection timed out, continuing with AI-guided extraction")
                        if hasattr(self, 'execution_progress'):
                            self.execution_progress['property_detection_timeout'] = True
                    except Exception as e:
                        logger.warning(f"Error in property listing detection: {e}, continuing with AI-guided extraction")
                    
                    if property_listings and len(property_listings) > 0:
                        logger.info(f"Found {len(property_listings)} property listings directly")
                        # Add each property listing as a separate result
                        for idx, listing in enumerate(property_listings):
                            listing_result = {
                                'extraction_method': 'property_detection',
                                'data': listing,
                                'confidence': 0.9,
                                'timestamp': time.time()
                            }
                            self.add_result(start_url, listing_result, 0)
                            
                        logger.info(f"Added {len(property_listings)} property listings to results")
                
                # Check if execution has been cancelled
                if hasattr(self, 'cancel_execution') and self.cancel_execution:
                    logger.warning(f"Execution cancelled for {start_url}")
                    
                    # Return any results we've collected so far
                    return {
                        "cancelled": True, 
                        "url": start_url, 
                        "results": self.results,
                        "timestamp": time.time()
                    }
                
                # Update progress
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'ai_guided_extraction'
                
                # If we didn't find property listings directly, try AI-guided extraction
                if not property_listings or len(property_listings) == 0:
                    # Set a timeout for the AI extraction to prevent hanging
                    try:
                        # Use a shorter timeout for AI extraction (45 seconds instead of 60)
                        ai_extraction_timeout = self.config.get('ai_extraction_timeout', 45)
                        
                        # Log the timeout setting
                        logger.info(f"Setting AI extraction timeout to {ai_extraction_timeout} seconds")
                        
                        # Create task and track it
                        extraction_task = self._extract_with_ai_guidance(html_content, start_url, self.user_prompt, self.config)
                        
                        # Update progress tracking
                        if hasattr(self, 'execution_progress'):
                            self.execution_progress['ai_extraction_start'] = time.time()
                            self.execution_progress['ai_extraction_deadline'] = time.time() + ai_extraction_timeout
                        
                        # Wait for the extraction with timeout
                        result = await asyncio.wait_for(extraction_task, timeout=ai_extraction_timeout)
                        
                        if result:
                            # Add to results
                            self.add_result(start_url, result, 0)
                            logger.info(f"Successfully extracted data from {start_url} using AI guidance")
                            # Log a summary of what we got
                            try:
                                data = result.get('data', [])
                                if isinstance(data, list):
                                    if data and len(data) > 0 and isinstance(data[0], dict):
                                        # Get keys from first item if it's a list of dicts
                                        data_keys = data[0].keys()
                                        logger.info(f"Extracted {len(data)} items with fields: {', '.join(data_keys) if data_keys else 'none'}")
                                    else:
                                        logger.info(f"Extracted {len(data)} items")
                                elif isinstance(data, dict):
                                    # Handle legacy dict format
                                    data_keys = data.keys()
                                    logger.info(f"Extracted data fields: {', '.join(data_keys) if data_keys else 'none'}")
                                else:
                                    logger.info(f"Extracted data: {type(data)}")
                            except Exception as e:
                                logger.warning(f"Could not summarize extracted data: {e}")
                        else:
                            logger.warning(f"No data extracted from {start_url} using AI guidance")
                    except asyncio.TimeoutError:
                        logger.warning(f"AI-guided extraction timed out after {ai_extraction_timeout} seconds, continuing with results gathered so far")
                        if hasattr(self, 'execution_progress'):
                            self.execution_progress['ai_extraction_timeout'] = True
                            self.execution_progress['ai_extraction_timeout_time'] = time.time()
                    except Exception as e:
                        logger.error(f"Error in AI-guided extraction: {e}")
                        # Record the error in progress tracking
                        if hasattr(self, 'execution_progress'):
                            self.execution_progress['ai_extraction_error'] = str(e)
                            self.execution_progress['ai_extraction_error_time'] = time.time()
                
                # Update progress
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'finalization'
                
                # Get resource and error statistics
                resource_stats = self.get_resource_stats()
                error_stats = self.get_error_stats()
                
                # Prepare results
                final_results = {
                    "results": self.results,
                    "resource_stats": resource_stats,
                    "error_stats": error_stats,
                    "url": start_url,
                    "completed": True,
                    "timestamp": time.time()
                }
                
                # Print summary to terminal if not part of adaptive scraper
                if not crawler or not hasattr(crawler, 'is_adaptive_scraper') or not crawler.is_adaptive_scraper:
                    print(f"\n{'='*60}")
                    print(f"SCRAPING RESULTS FOR: {start_url}")
                    print(f"{'='*60}")
                    print(f"Extracted {len(self.results)} items")
                    print(f"Total requests: {resource_stats.get('total_requests', 0)}")
                    print(f"Success rate: {resource_stats.get('success_rate', 0):.2f}%")
                    print(f"Errors: {error_stats.get('total_errors', 0)}")
                    print(f"{'='*60}\n")
                
                return final_results
                
            except CircuitOpenError as e:
                logger.warning(f"Circuit breaker prevented request to {start_url}: {str(e)}")
                
                # Try to use cached results or fallback method
                if self.config.get('fallback_to_dom', True):
                    logger.info(f"Using DOM strategy as fallback for {start_url}")
                    
                    # Update progress
                    if hasattr(self, 'execution_progress'):
                        self.execution_progress['phase'] = 'fallback_dom_strategy'
                    
                    # Check if dom_strategy.execute is async before awaiting
                    if asyncio.iscoroutinefunction(self.dom_strategy.execute):
                        dom_result = await self.dom_strategy.execute(start_url, crawler=crawler, extraction_config=extraction_config)
                    else:
                        dom_result = self.dom_strategy.execute(start_url, crawler=crawler, extraction_config=extraction_config)
                    return dom_result
                
                raise
                
            except Exception as e:
                # Get domain for error context
                domain = urlparse(start_url).netloc
                
                # Classify and log the error
                error_context = {
                    'url': start_url,
                    'operation': 'execute',
                    'domain': domain
                }
                error_info = self._classify_error(e, error_context)
                
                # Record failure for circuit breaker
                self._record_failure(domain)
                
                logger.error(f"Error executing AI-guided strategy: {str(e)} - "
                           f"{error_info.get('category', 'unknown')}")
                
                # Try fallback method
                if self.config.get('fallback_to_dom', True):
                    logger.info(f"Using DOM strategy as fallback for {start_url}")
                    
                    # Update progress
                    if hasattr(self, 'execution_progress'):
                        self.execution_progress['phase'] = 'fallback_dom_strategy'
                    
                    # Check if dom_strategy.execute is async before awaiting
                    if asyncio.iscoroutinefunction(self.dom_strategy.execute):
                        dom_result = await self.dom_strategy.execute(start_url, crawler=crawler, extraction_config=extraction_config)
                    else:
                        dom_result = self.dom_strategy.execute(start_url, crawler=crawler, extraction_config=extraction_config)
                    return dom_result
                
                raise
                
        except Exception as e:
            # Final catch-all
            error_context = {
                'url': start_url,
                'operation': 'execute_outer'
            }
            error_info = self._classify_error(e, error_context)
            logger.error(f"Unrecoverable error in AI-guided strategy: {str(e)} - "
                       f"{error_info.get('category', 'unknown')}")
            
            # Get resource and error statistics even in case of failure
            resource_stats = self.get_resource_stats()
            error_stats = self.get_error_stats()
            
            return {
                "error": str(e),
                "error_category": error_info.get('category', 'unknown'),
                "resource_stats": resource_stats,
                "error_stats": error_stats,
                "results": self.results if hasattr(self, 'results') and self.results else [],
                "completed": False
            }
    
    async def _extract_with_ai_guidance(self, html_content: str, url: str, user_intent: str, 
                                      config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract data from HTML content using AI guidance with error handling and retry.
        
        Args:
            html_content: HTML content to extract from
            url: URL the content was fetched from
            user_intent: User's intent or search query
            config: Configuration parameters
            
        Returns:
            Dictionary with extracted data or None if extraction failed
        """
        domain = urlparse(url).netloc
        max_retries = config.get('max_ai_retries', 3)
        
        # Track AI extraction progress if we have execution progress tracking
        if hasattr(self, 'execution_progress'):
            # Calculate extraction deadline based on outer deadline or specific extraction timeout
            ai_extraction_timeout = config.get('ai_extraction_timeout', 45)  # 45 seconds for AI extraction
            extraction_deadline = time.time() + ai_extraction_timeout
            
            # If there's already a global deadline, use the earlier of the two
            if 'timeout_at' in self.execution_progress:
                global_deadline = self.execution_progress.get('timeout_at')
                extraction_deadline = min(extraction_deadline, global_deadline)
            
            self.execution_progress['ai_extraction'] = {
                'start_time': time.time(),
                'attempt': 0,
                'max_retries': max_retries,
                'deadline': extraction_deadline
            }
            
            # Log the timeout setting
            time_remaining = extraction_deadline - time.time()
            logger.info(f"AI extraction timeout set to {time_remaining:.1f} seconds remaining for {url}")
        
        # Check if execution has been cancelled
        if hasattr(self, 'cancel_execution') and self.cancel_execution:
            logger.warning(f"Cancelling AI-guided extraction for {url}")
            return None
        
        # Initialize extraction context
        extraction_context = {
            'domain': domain,
            'url': url,
            'max_retries': max_retries,
            'current_retry': 0
        }
        
        primary_ai_items_log = None # For Log Point 1.1
        fallback_item_log = None # For Log Point 1.2
        final_data_to_return_log = [] # For Log Point 1.3

        for attempt in range(1, max_retries + 1):
            # Check if execution has been cancelled
            if hasattr(self, 'cancel_execution') and self.cancel_execution:
                logger.warning(f"Cancelling AI extraction for {url}")
                return None
            
            # Check if we're approaching our extraction deadline
            if hasattr(self, 'execution_progress') and 'ai_extraction' in self.execution_progress:
                now = time.time()
                deadline = self.execution_progress['ai_extraction'].get('deadline')
                if deadline and now > deadline - 2:  # If less than 2 seconds remaining
                    logger.warning(f"AI extraction approaching deadline for {url}, stopping early")
                    return await self._fallback_extraction(html_content, url, config)
            
            # Update extraction attempt in progress tracking
            if hasattr(self, 'execution_progress') and 'ai_extraction' in self.execution_progress:
                self.execution_progress['ai_extraction']['current_retry'] = attempt
                self.execution_progress['ai_extraction']['phase'] = 'starting'
            
            try:
                # Add a timeout for the circuit breaker check
                if hasattr(self, 'execution_progress') and 'ai_extraction' in self.execution_progress:
                    self.execution_progress['ai_extraction']['phase'] = 'circuit_breaker_check'
                
                # Check AI service circuit breaker if using AI service
                if self.ai_service:
                    ai_domain = "ai_service"  # Generic domain for AI service
                    if not self._check_circuit_breaker(ai_domain):
                        logger.warning(f"AI service circuit breaker is open, using fallback extraction")
                        return await self._fallback_extraction(html_content, url, config)
                
                # Check if execution has been cancelled
                if hasattr(self, 'cancel_execution') and self.cancel_execution:
                    logger.warning(f"Cancelling AI extraction for {url}")
                    return None
                
                # Update progress
                if hasattr(self, 'execution_progress') and 'ai_extraction' in self.execution_progress:
                    self.execution_progress['ai_extraction']['phase'] = 'html_cleaning'
                
                # Clean HTML with error handling and timeout
                try:
                    # Set timeout for HTML cleaning
                    if hasattr(self.html_service, 'clean_html'):
                        if asyncio.iscoroutinefunction(self.html_service.clean_html):
                            # If it's an async function
                            cleaned_html = await asyncio.wait_for(
                                self.html_service.clean_html(html_content), 
                                timeout=10
                            )
                        else:
                            # If it's a regular function
                            cleaned_html = self.html_service.clean_html(html_content)
                    else:
                        cleaned_html = html_content
                except asyncio.TimeoutError:
                    logger.warning(f"HTML cleaning timed out for {url}, using original content")
                    cleaned_html = html_content
                except Exception as e:
                    logger.warning(f"Error using HTML service to clean HTML: {str(e)}. Using original content.")
                    cleaned_html = html_content
                
                # Check if execution has been cancelled
                if hasattr(self, 'cancel_execution') and self.cancel_execution:
                    logger.warning(f"Cancelling AI extraction for {url}")
                    return None
                
                # Use AI to determine the best extraction approach
                if user_intent and (self.ai_service or self.ai_model):
                    # Update progress
                    if hasattr(self, 'execution_progress') and 'ai_extraction' in self.execution_progress:
                        self.execution_progress['ai_extraction']['phase'] = 'determining_approach'
                    
                    try:
                        # Set timeout for determining extraction approach
                        extraction_approach_task = self._determine_extraction_approach(cleaned_html, url, user_intent)
                        extraction_approach = await asyncio.wait_for(extraction_approach_task, timeout=20)
                        
                        # Check if execution has been cancelled
                        if hasattr(self, 'cancel_execution') and self.cancel_execution:
                            logger.warning(f"Cancelling AI extraction for {url}")
                            return None
                        
                        if extraction_approach.get('method') == 'schema':
                            # Update progress
                            if hasattr(self, 'execution_progress') and 'ai_extraction' in self.execution_progress:
                                self.execution_progress['ai_extraction']['phase'] = 'schema_extraction'
                            
                            # Extract structured data using schema.org markup
                            schema_service = self._get_schema_service()
                            
                            if schema_service:
                                # Set timeout for schema extraction
                                try:
                                    if hasattr(schema_service, 'extract_schemas'):
                                        if asyncio.iscoroutinefunction(schema_service.extract_schemas):
                                            # If it's an async function
                                            schema_results = await asyncio.wait_for(
                                                schema_service.extract_schemas(cleaned_html, url),
                                                timeout=15
                                            )
                                        else:
                                            # If it's a regular function
                                            schema_results = schema_service.extract_schemas(cleaned_html, url)
                                        
                                        if schema_results:
                                            return {
                                                'url': url,
                                                'extraction_method': 'ai_guided_schema',
                                                'data': schema_results,
                                                'confidence': 0.9,
                                                'timestamp': time.time()
                                            }
                                except asyncio.TimeoutError:
                                    logger.warning(f"Schema extraction timed out for {url}")
                                except Exception as e:
                                    logger.warning(f"Error during schema extraction: {e}")
                        
                        elif extraction_approach.get('method') == 'ai_direct':
                            # Check if execution has been cancelled
                            if hasattr(self, 'cancel_execution') and self.cancel_execution:
                                logger.warning(f"Cancelling AI extraction for {url}")
                                return None # Ensure we return None if cancelled
                                
                            # Update progress
                            if hasattr(self, 'execution_progress') and 'ai_extraction' in self.execution_progress:
                                self.execution_progress['ai_extraction']['phase'] = 'ai_direct_extraction'
                            
                            # Use AI to directly extract data based on user intent
                            # Set timeout for direct AI extraction
                            try:
                                direct_extraction_task = self._extract_data_with_ai(cleaned_html, url, user_intent, extraction_approach.get('fields'))
                                extracted_data = await asyncio.wait_for(direct_extraction_task, timeout=30)
                                
                                primary_ai_items_log = extracted_data # Log Point 1.1
                                self.logger.info(f"STRATEGY_LOG ({url}): Primary AI extraction yielded: {primary_ai_items_log}")

                                if extracted_data:
                                    final_data_to_return_log = extracted_data if isinstance(extracted_data, list) else [extracted_data]
                                    self.logger.info(f"STRATEGY_LOG_FINAL ({url}): Data being returned by strategy (after primary AI): {final_data_to_return_log}")
                                    return {
                                        'url': url,
                                        'extraction_method': 'ai_direct',
                                        'data': final_data_to_return_log,
                                        'confidence': 0.8,
                                        'timestamp': time.time()
                                    }
                            except asyncio.TimeoutError:
                                logger.warning(f"Direct AI extraction timed out for {url}")
                            except Exception as e:
                                logger.error(f"Error during direct AI extraction: {e}")
                        
                        # If AI approach is unclear or fails, try fallback
                        logger.warning(f"AI approach unclear or failed for {url}, trying fallback.")
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Determining extraction approach timed out for {url}")
                    except Exception as e:
                        logger.error(f"Error determining extraction approach: {e}")
                
                # Fallback extraction if other methods fail or are not applicable
                # This is where Log Point 1.2 will be effectively captured
                fallback_result = await self._fallback_extraction(html_content, url, config) # Pass config here
                fallback_item_log = fallback_result.get('data') if fallback_result else None # Log Point 1.2
                self.logger.info(f"STRATEGY_LOG ({url}): Fallback extraction yielded: {fallback_item_log}")

                if fallback_result and fallback_result.get('data'):
                    final_data_to_return_log = fallback_result['data'] if isinstance(fallback_result['data'], list) else [fallback_result['data']]
                    self.logger.info(f"STRATEGY_LOG_FINAL ({url}): Data being returned by strategy (after fallback): {final_data_to_return_log}")
                    return fallback_result # Fallback result is already a dict with 'data' key
                
            except CircuitOpenError as e:
                logger.warning(f"AI service circuit breaker is open during attempt {attempt}: {str(e)}")
                if attempt == max_retries:
                    logger.error(f"AI service circuit breaker remained open after {max_retries} retries for {url}")
                    # Fallback if circuit breaker remains open
                    fb_res = await self._fallback_extraction(html_content, url, config) # Pass config here
                    fallback_item_log = fb_res.get('data') if fb_res else None # Log Point 1.2
                    self.logger.info(f"STRATEGY_LOG ({url}): Fallback extraction (due to circuit breaker) yielded: {fallback_item_log}")
                    if fb_res and fb_res.get('data'):
                        final_data_to_return_log = fb_res['data'] if isinstance(fb_res['data'], list) else [fb_res['data']]
                        self.logger.info(f"STRATEGY_LOG_FINAL ({url}): Data being returned by strategy (after fallback due to circuit breaker): {final_data_to_return_log}")
                        return fb_res
                    return None
                await asyncio.sleep(2 * attempt)  # Exponential backoff
                continue
            except Exception as e:
                error_context = {
                    'url': url,
                    'operation': 'ai_extraction_attempt',
                    'attempt': attempt,
                    'domain': domain
                }
                error_info = self._classify_error(e, error_context)
                logger.error(f"Error during AI extraction attempt {attempt} for {url}: {str(e)} - "
                               f"{error_info.get('category', 'unknown')}")
                
                # Record failure for AI service circuit breaker
                if self.ai_service:
                    self._record_failure("ai_service")
                
                if attempt == max_retries:
                    logger.error(f"AI extraction failed after {max_retries} retries for {url}")
                    # Fallback if all retries fail
                    fb_res_retry = await self._fallback_extraction(html_content, url, config) # Pass config here
                    fallback_item_log = fb_res_retry.get('data') if fb_res_retry else None # Log Point 1.2
                    self.logger.info(f"STRATEGY_LOG ({url}): Fallback extraction (due to max retries) yielded: {fallback_item_log}")
                    if fb_res_retry and fb_res_retry.get('data'):
                        final_data_to_return_log = fb_res_retry['data'] if isinstance(fb_res_retry['data'], list) else [fb_res_retry['data']]
                        self.logger.info(f"STRATEGY_LOG_FINAL ({url}): Data being returned by strategy (after fallback due to max retries): {final_data_to_return_log}")
                        return fb_res_retry
                    return None # Ensure None is returned if all fails
        
        # If loop completes without returning, it means all attempts failed
        self.logger.error(f"AI-guided extraction failed for {url} after all attempts.")
        # Final fallback attempt if nothing else worked
        final_fallback_result = await self._fallback_extraction(html_content, url, config) # Pass config here
        fallback_item_log = final_fallback_result.get('data') if final_fallback_result else None # Log Point 1.2
        self.logger.info(f"STRATEGY_LOG ({url}): Final fallback extraction yielded: {fallback_item_log}")
        if final_fallback_result and final_fallback_result.get('data'):
            final_data_to_return_log = final_fallback_result['data'] if isinstance(final_fallback_result['data'], list) else [final_fallback_result['data']]
            self.logger.info(f"STRATEGY_LOG_FINAL ({url}): Data being returned by strategy (after final fallback): {final_data_to_return_log}")
            return final_fallback_result

        self.logger.info(f"STRATEGY_LOG_FINAL ({url}): No data being returned by strategy (all attempts failed). Final data log: {final_data_to_return_log}")
        return None

    async def _fallback_extraction(self, html_content: str, url: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Fallback to DOM-based or basic content extraction if AI guidance fails.
        """
        logger.info(f"Using fallback extraction for {url}")
        # Try to use the ContentExtractor's fallback first, as it's more robust
        if hasattr(self, 'content_extractor') and hasattr(self.content_extractor, '_fallback_extraction'):
            try:
                # Ensure context is passed if available and needed by content_extractor's fallback
                # The content_extractor._fallback_extraction is expected to return a single dictionary item
                # or None, not a list of items.
                fallback_item = self.content_extractor._fallback_extraction(html_content, url, self.context)
                if fallback_item and isinstance(fallback_item, dict):
                    # The item from content_extractor's fallback is the data itself.
                    # We need to wrap it in the standard strategy result format.
                    return {
                        'url': url,
                        'extraction_method': 'content_extractor_fallback',
                        'data': [fallback_item], # Ensure data is a list of items
                        'confidence': 0.5, # Lower confidence for fallback
                        'timestamp': time.time()
                    }
                elif fallback_item: # If it's not a dict, log a warning
                    logger.warning(f"ContentExtractor fallback for {url} returned unexpected type: {type(fallback_item)}")
            except Exception as e:
                logger.error(f"Error in ContentExtractor fallback for {url}: {e}")
        
        # If ContentExtractor fallback fails or is not available, try DOM strategy
        if config.get('fallback_to_dom', True):
            try:
                logger.info(f"Attempting DOM strategy as secondary fallback for {url}")
                # Create a dummy crawler or pass None if the DOM strategy can handle it
                # The DOM strategy's execute method might expect a crawler instance.
                # For simplicity here, we'll assume it can run with html_content directly
                # or we adapt its call if necessary.
                # This part might need adjustment based on DOMStrategy's actual interface.
                # For now, let's assume we can't directly call it without a full crawler setup.
                # So, we'll simulate a basic extraction.
                soup = BeautifulSoup(html_content, 'html.parser')
                title = soup.title.string if soup.title else 'No title found'
                # This is a very basic fallback, actual DOM strategy is more complex
                dom_data = [{'title': title, 'url': url, 'source': 'dom_basic_fallback'}]
                return {
                    'url': url,
                    'extraction_method': 'dom_basic_fallback',
                    'data': dom_data,
                    'confidence': 0.4,
                    'timestamp': time.time()
                }
            except Exception as e:
                logger.error(f"Error in DOM strategy fallback for {url}: {e}")
        
        logger.warning(f"All fallback extraction methods failed for {url}")
        return None

    # Abstract method implementations required by BaseStrategy interface
    
    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Crawl a website starting from the given URL using AI-guided techniques.
        
        This method implements the abstract crawl method from BaseStrategy.
        It uses AI to analyze the website structure and discover relevant pages.
        
        Args:
            start_url: The URL to start crawling from
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing crawl results or None if failed
        """
        try:
            logger.info(f"AI-guided crawl starting from: {start_url}")
            
            # Use the existing execute method which handles the full AI-guided process
            # The execute method already performs intelligent crawling based on AI analysis
            result = self.execute(start_url, **kwargs)
            
            if result and result.get("success"):
                # Store the result in _results for get_results() method
                crawl_result = {
                    "url": start_url,
                    "method": "ai_guided_crawl",
                    "success": True,
                    "timestamp": time.time(),
                    "data": result.get("data", []),
                    "analysis": result.get("analysis", {}),
                    "strategy_used": result.get("strategy_used", "ai_guided")
                }
                self._results.append(crawl_result)
                return crawl_result
            else:
                logger.warning(f"AI-guided crawl failed for {start_url}")
                return None
                
        except Exception as e:
            logger.error(f"Error in AI-guided crawl: {e}")
            return None
    
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Extract data from HTML content using AI-guided techniques.
        
        This method implements the abstract extract method from BaseStrategy.
        It uses AI to understand content structure and extract relevant information.
        
        Args:
            html_content: The HTML content to extract data from
            url: The URL the content was fetched from
            **kwargs: Additional keyword arguments including extraction prompt
            
        Returns:
            Dictionary containing extracted data or None if failed
        """
        try:
            logger.info(f"AI-guided extraction from URL: {url}")
            
            # Get extraction prompt from kwargs or use default
            extraction_prompt = kwargs.get("extraction_prompt", self.user_prompt or "Extract relevant information")
            
            # Use the existing _extract_with_ai_guidance method
            extraction_result = self._extract_with_ai_guidance(
                html_content, 
                url, 
                extraction_prompt
            )
            
            if extraction_result and extraction_result.get("success"):
                # Store the result in _results for get_results() method
                extract_result = {
                    "url": url,
                    "method": "ai_guided_extract",
                    "success": True,
                    "timestamp": time.time(),
                    "data": extraction_result.get("data", []),
                    "extraction_method": extraction_result.get("extraction_method", "ai_guidance"),
                    "confidence": extraction_result.get("confidence", 0.5)
                }
                self._results.append(extract_result)
                return extract_result
            else:
                logger.warning(f"AI-guided extraction failed for {url}")
                return None
                
        except Exception as e:
            logger.error(f"Error in AI-guided extraction: {e}")
            return None
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get all results collected by this strategy.
        
        This method implements the abstract get_results method from BaseStrategy.
        Returns all data collected during crawling and extraction operations.
        
        Returns:
            List of dictionaries containing all collected results
        """
        try:
            # Return a copy of the results to prevent external modification
            return self._results.copy()
        except Exception as e:
            logger.error(f"Error getting results: {e}")
            return []
    
    @property 
    def name(self) -> str:
        """
        Get the name of this strategy.
        
        Returns:
            String name identifier for this strategy
        """
        return "ai_guided"
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute the AI-guided strategy synchronously.
        
        This is the required synchronous execute method from BaseStrategy.
        It delegates to the async implementation when needed.
        
        Args:
            url: The URL to crawl
            **kwargs: Additional arguments including crawler, extraction_config, etc.
            
        Returns:
            Dictionary containing the results or None if execution failed
        """
        try:
            # Extract parameters that match the async signature
            crawler = kwargs.get('crawler')
            extraction_config = kwargs.get('extraction_config')
            
            if crawler:
                # If we have a crawler, use the async implementation
                import asyncio
                try:
                    # Check if we're already in an async context
                    if asyncio.get_event_loop().is_running():
                        # We're in an async context, need to create a new event loop in a thread
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run, 
                                self.execute_async(crawler, url, extraction_config)
                            )
                            return future.result()
                    else:
                        # We can run the async method directly
                        return asyncio.run(self.execute_async(crawler, url, extraction_config))
                except Exception as e:
                    logger.error(f"Error in async execution for {url}: {e}")
                    return None
            else:
                # Fallback: basic synchronous execution without crawler
                logger.warning(f"No crawler provided, using basic synchronous execution for {url}")
                try:
                    # Make a basic request and return simple result
                    response = self._make_request(url)
                    if response and hasattr(response, 'text'):
                        return {
                            "success": True,
                            "url": url,
                            "method": "basic_sync",
                            "timestamp": time.time(),
                            "data": {"html_length": len(response.text)},
                            "strategy_used": "ai_guided_basic"
                        }
                    else:
                        return None
                except Exception as e:
                    logger.error(f"Error in basic sync execution for {url}: {e}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error in AIGuidedStrategy.execute for {url}: {e}")
            return None

    async def execute_async(self, crawler, start_url, extraction_config=None):
        """
        Execute the AI-guided strategy on a URL with a global timeout to prevent indefinite hangs.
        
        This is the main async implementation that handles the full AI-guided process.
        """
        # Use a reasonable default timeout, but allow override from config
        default_timeout = 90  # 90 seconds default (1.5 minutes)
        timeout_seconds = self.config.get('global_timeout', default_timeout)
        
        # Ensure the timeout is reasonable (between 30 seconds and 5 minutes)
        timeout_seconds = max(30, min(timeout_seconds, 300))
        
        # Log the timeout setting
        logger.info(f"Using global timeout of {timeout_seconds} seconds for {start_url}")
        
        # Track execution progress for reporting
        self.execution_progress = {
            'status': 'starting',
            'phase': 'initialization',
            'start_time': time.time(),
            'timeout_at': time.time() + timeout_seconds,
            'url': start_url
        }
        
        # Cancel flag for coordinated cancellation of subtasks
        self.cancel_execution = False
        
        try:
            # Create a task for the execution
            task = asyncio.create_task(self._execute_inner(crawler, start_url, extraction_config))
            
            # Set up a timeout task
            try:
                result = await asyncio.wait_for(task, timeout=timeout_seconds)
                
                # Ensure the completed flag is set
                if isinstance(result, dict) and 'completed' not in result:
                    result['completed'] = True
                
                # Update progress
                self.execution_progress['status'] = 'completed'
                self.execution_progress['end_time'] = time.time()
                
                return result
                
            except asyncio.TimeoutError:
                # Set cancel flag to notify any running subtasks
                self.cancel_execution = True
                
                # Update progress
                self.execution_progress['status'] = 'timeout'
                self.execution_progress['end_time'] = time.time()
                
                # Log timeout with details about what was happening
                current_phase = self.execution_progress.get('phase', 'unknown')
                logger.error(f"AI-guided strategy timed out after {timeout_seconds} seconds for {start_url} during {current_phase} phase")
                
                # Get any partial results that might have been collected
                resource_stats = self.get_resource_stats()
                error_stats = self.get_error_stats()
                
                # Prepare error result with progress information
                error_result = {
                    "error": f"Timeout after {timeout_seconds} seconds during {current_phase} phase",
                    "resource_stats": resource_stats,
                    "error_stats": error_stats,
                    "url": start_url,
                    "completed": False,
                    "progress": self.execution_progress,
                    "timestamp": time.time(),
                    "results": self.results if hasattr(self, 'results') and self.results else []
                }
                
                # Display timeout message if not running in adaptive scraper
                if not crawler or not hasattr(crawler, 'is_adaptive_scraper') or not crawler.is_adaptive_scraper:
                    print(f"\n{'='*60}")
                    print(f"SCRAPING TIMEOUT FOR: {start_url}")
                    print(f"{'='*60}")
                    print(f"Timeout occurred during: {current_phase}")
                    
                    # Show any partial results if available
                    if hasattr(self, 'results') and self.results:
                        print(f"Partial results available: {len(self.results)} items found before timeout")
                    else:
                        print("No results due to timeout.")
                    
                    # Provide guidance for timeout remediation
                    print(f"\nPossible solutions:")
                    print(f"1. Increase the global_timeout setting (current: {timeout_seconds}s)")
                    print(f"2. Try again later when the service is less busy")
                    print(f"3. Use a more specific search query to reduce processing time")
                    if current_phase == 'ai_extraction':
                        print(f"4. Consider using a different extraction strategy")
                    
                    print(f"{'='*60}\n")
                
                return error_result
        except Exception as e:
            # Handle any other exceptions during execution
            logger.error(f"Error in AI-guided strategy execution: {str(e)}")
            
            # Update progress
            self.execution_progress['status'] = 'error'
            self.execution_progress['error'] = str(e)
            self.execution_progress['end_time'] = time.time()
            
            # Get any stats that are available
            resource_stats = self.get_resource_stats() if hasattr(self, 'get_resource_stats') else {}
            error_stats = self.get_error_stats() if hasattr(self, 'get_error_stats') else {}
            
            error_result = {
                "error": str(e),
                "resource_stats": resource_stats,
                "error_stats": error_stats,
                "url": start_url,
                "completed": False,
                "progress": self.execution_progress,
                "timestamp": time.time(),
                "results": self.results if hasattr(self, 'results') and self.results else []
            }
            
            # Display error message if not running in adaptive scraper
            if not crawler or not hasattr(crawler, 'is_adaptive_scraper') or not crawler.is_adaptive_scraper:
                print(f"\n{'='*60}")
                print(f"SCRAPING ERROR FOR: {start_url}")
                print(f"{'='*60}")
                print(f"Error: {str(e)}")
                print(f"{'='*60}\n")
            
            return error_result

    async def _execute_inner(self, crawler, start_url, extraction_config=None):
        """
        Execute the AI-guided strategy on a URL.
        
        This method implements the main entry point for the strategy,
        analyzing the page and extracting data with AI assistance.
        
        Args:
            crawler: The crawler instance
            start_url: The URL to process
            extraction_config: Configuration for extraction
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing AI-guided strategy for URL: {start_url}")
        
        # Parse and store the main domain
        self.main_domain = urlparse(start_url).netloc
        
        # Initialize results if not already there
        if not hasattr(self, 'results'):
            self.results = []
        
        try:
            # Make the request with full error handling and resource management
            try:
                # Get the domain for resource management
                domain = urlparse(start_url).netloc
                
                # Update progress tracking
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'rate_limiting'
                    self.execution_progress['domain'] = domain
                
                # Handle rate limiting for the domain
                self._handle_rate_limiting(domain)
                
                # Check if execution has been cancelled
                if hasattr(self, 'cancel_execution') and self.cancel_execution:
                    logger.warning(f"Execution cancelled for {start_url}")
                    return {"cancelled": True, "url": start_url, "timestamp": time.time()}
                
                # Update progress
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'circuit_breaker_check'
                
                # Check circuit breaker
                if not self._check_circuit_breaker(domain):
                    logger.warning(f"Circuit breaker open for {domain}, skipping request")
                    raise CircuitOpenError(f"Circuit breaker open for {domain}")
                
                # Check if execution has been cancelled
                if hasattr(self, 'cancel_execution') and self.cancel_execution:
                    logger.warning(f"Execution cancelled for {start_url}")
                    return {"cancelled": True, "url": start_url, "timestamp": time.time()}
                
                # First, check if this is a site that requires search to get listings
                search_required = False
                search_terms = None
                
                # Update progress
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'extraction_config_analysis'
                
                # If we have specific keywords or location from extraction config, use them
                if extraction_config:
                    if "keywords" in extraction_config:
                        search_terms = extraction_config["keywords"]
                    elif "extract_description" in extraction_config and "Cleveland" in extraction_config["extract_description"]:
                        search_terms = ["Cleveland, OH"]
                    elif "location_data" in extraction_config and extraction_config["location_data"].get("city"):
                        city = extraction_config["location_data"].get("city")
                        state = extraction_config["location_data"].get("state")
                        if city and state:
                            search_terms = [f"{city}, {state}"]
                
                # Check if execution has been cancelled
                if hasattr(self, 'cancel_execution') and self.cancel_execution:
                    logger.warning(f"Execution cancelled for {start_url}")
                    return {"cancelled": True, "url": start_url, "timestamp": time.time()}
                
                # Update progress
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'search_form_detection'
                
                # Try to find and use search forms to get better results
                search_result = None
                try:
                    search_form_timeout = self.config.get('search_form_timeout', 45)  # Default 45 seconds for search form operation
                    search_form_task = self._find_and_use_search_forms(start_url, search_terms)
                    search_result = await asyncio.wait_for(search_form_task, timeout=search_form_timeout)
                    
                    if hasattr(self, 'execution_progress'):
                        self.execution_progress['search_form_completed'] = True
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Search form detection timed out after {search_form_timeout} seconds, continuing with direct request")
                    if hasattr(self, 'execution_progress'):
                        self.execution_progress['search_form_timeout'] = True
                except Exception as e:
                    logger.warning(f"Error in search form detection: {e}, continuing with direct request")
                    if hasattr(self, 'execution_progress'):
                        self.execution_progress['search_form_error'] = str(e)
                
                # Check if execution has been cancelled
                if hasattr(self, 'cancel_execution') and self.cancel_execution:
                    logger.warning(f"Execution cancelled for {start_url}")
                    return {"cancelled": True, "url": start_url, "timestamp": time.time()}
                
                # Update progress
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'html_content_retrieval'
                
                # If search was successful, use the results HTML
                if search_result and search_result.get('success') and search_result.get('html_content'):
                    logger.info(f"Successfully used search forms, using search results content")
                    html_content = search_result.get('html_content')
                    
                    # Record success for circuit breaker
                    self._record_success(domain)
                else:
                    # Make the direct request using our robust implementation
                    logger.info(f"Making HTTP request to {start_url}...")
                    
                    # Update progress
                    if hasattr(self, 'execution_progress'):
                        self.execution_progress['phase'] = 'direct_http_request'
                    
                    # Use requests directly for troubleshooting
                    import requests
                    try:
                        # First try with direct requests to see if it's a connection issue
                        direct_response = requests.get(
                            start_url, 
                            headers={
                                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                            },
                            timeout=30
                        )
                        logger.info(f"Direct request to {start_url} succeeded with status {direct_response.status_code}")
                    except Exception as direct_req_error:
                        logger.warning(f"Direct request failed, will try with _make_request: {str(direct_req_error)}")
                    
                    # Check if execution has been cancelled
                    if hasattr(self, 'cancel_execution') and self.cancel_execution:
                        logger.warning(f"Execution cancelled for {start_url}")
                        return {"cancelled": True, "url": start_url, "timestamp": time.time()}
                    
                    # Now try the regular method
                    response = self._make_request(start_url)
                    if not response or not hasattr(response, 'text'):
                        logger.error(f"Invalid response from _make_request for URL: {start_url}")
                        raise ValueError(f"Invalid response from _make_request for URL: {start_url}")
                    html_content = response.text
                    current_url = start_url  # Track the URL we're actually processing
                    logger.info(f"Successfully retrieved HTML content ({len(html_content)} bytes) from {start_url}")
                    
                    # Record success for circuit breaker
                    self._record_success(domain)
                
                # Check if execution has been cancelled
                if hasattr(self, 'cancel_execution') and self.cancel_execution:
                    logger.warning(f"Execution cancelled for {start_url}")
                    return {"cancelled": True, "url": start_url, "timestamp": time.time()}
                
                # Update progress
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'property_listing_detection'
                
                # First, check if this is a potential real estate listings page
                property_listings = None
                if hasattr(self, 'user_prompt') and self.user_prompt:
                    real_estate_terms = ['home', 'house', 'property', 'real estate', 'listing']
                    # Make sure user_prompt is a string before calling .lower()
                    if isinstance(self.user_prompt, str):
                        should_check_listings = any(term in self.user_prompt.lower() for term in real_estate_terms)
                    elif isinstance(self.user_prompt, dict):
                        # Handle dictionary user_prompt
                        user_prompt_text = str(self.user_prompt)
                        should_check_listings = any(term in user_prompt_text.lower() for term in real_estate_terms)
                    else:
                        should_check_listings = False
                        
                    if should_check_listings:
                        logger.info("Checking for property listings in the page...")
                    
                    # Set a timeout for property detection to avoid hanging
                    try:
                        property_detection_task = self._detect_real_estate_listings(html_content)
                        if isinstance(property_detection_task, list):  # Not a coroutine
                            property_listings = property_detection_task
                        else:  # It's a coroutine
                            property_listings = await asyncio.wait_for(property_detection_task, timeout=30)
                    except asyncio.TimeoutError:
                        logger.warning("Property listing detection timed out, continuing with AI-guided extraction")
                        if hasattr(self, 'execution_progress'):
                            self.execution_progress['property_detection_timeout'] = True
                    except Exception as e:
                        logger.warning(f"Error in property listing detection: {e}, continuing with AI-guided extraction")
                    
                    if property_listings and len(property_listings) > 0:
                        logger.info(f"Found {len(property_listings)} property listings directly")
                        # Add each property listing as a separate result
                        for idx, listing in enumerate(property_listings):
                            listing_result = {
                                'extraction_method': 'property_detection',
                                'data': listing,
                                'confidence': 0.9,
                                'timestamp': time.time()
                            }
                            self.add_result(start_url, listing_result, 0)
                            
                        logger.info(f"Added {len(property_listings)} property listings to results")
                
                # Check if execution has been cancelled
                if hasattr(self, 'cancel_execution') and self.cancel_execution:
                    logger.warning(f"Execution cancelled for {start_url}")
                    
                    # Return any results we've collected so far
                    return {
                        "cancelled": True, 
                        "url": start_url, 
                        "results": self.results,
                        "timestamp": time.time()
                    }
                
                # Update progress
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'ai_guided_extraction'
                
                # If we didn't find property listings directly, try AI-guided extraction
                if not property_listings or len(property_listings) == 0:
                    # Set a timeout for the AI extraction to prevent hanging
                    try:
                        # Use a shorter timeout for AI extraction (45 seconds instead of 60)
                        ai_extraction_timeout = self.config.get('ai_extraction_timeout', 45)
                        
                        # Log the timeout setting
                        logger.info(f"Setting AI extraction timeout to {ai_extraction_timeout} seconds")
                        
                        # Create task and track it
                        extraction_task = self._extract_with_ai_guidance(html_content, start_url, self.user_prompt, self.config)
                        
                        # Update progress tracking
                        if hasattr(self, 'execution_progress'):
                            self.execution_progress['ai_extraction_start'] = time.time()
                            self.execution_progress['ai_extraction_deadline'] = time.time() + ai_extraction_timeout
                        
                        # Wait for the extraction with timeout
                        result = await asyncio.wait_for(extraction_task, timeout=ai_extraction_timeout)
                        
                        if result:
                            # Add to results
                            self.add_result(start_url, result, 0)
                            logger.info(f"Successfully extracted data from {start_url} using AI guidance")
                            # Log a summary of what we got
                            try:
                                data = result.get('data', [])
                                if isinstance(data, list):
                                    if data and len(data) > 0 and isinstance(data[0], dict):
                                        # Get keys from first item if it's a list of dicts
                                        data_keys = data[0].keys()
                                        logger.info(f"Extracted {len(data)} items with fields: {', '.join(data_keys) if data_keys else 'none'}")
                                    else:
                                        logger.info(f"Extracted {len(data)} items")
                                elif isinstance(data, dict):
                                    # Handle legacy dict format
                                    data_keys = data.keys()
                                    logger.info(f"Extracted data fields: {', '.join(data_keys) if data_keys else 'none'}")
                                else:
                                    logger.info(f"Extracted data: {type(data)}")
                            except Exception as e:
                                logger.warning(f"Could not summarize extracted data: {e}")
                        else:
                            logger.warning(f"No data extracted from {start_url} using AI guidance")
                    except asyncio.TimeoutError:
                        logger.warning(f"AI-guided extraction timed out after {ai_extraction_timeout} seconds, continuing with results gathered so far")
                        if hasattr(self, 'execution_progress'):
                            self.execution_progress['ai_extraction_timeout'] = True
                            self.execution_progress['ai_extraction_timeout_time'] = time.time()
                    except Exception as e:
                        logger.error(f"Error in AI-guided extraction: {e}")
                        # Record the error in progress tracking
                        if hasattr(self, 'execution_progress'):
                            self.execution_progress['ai_extraction_error'] = str(e)
                            self.execution_progress['ai_extraction_error_time'] = time.time()
                
                # Update progress
                if hasattr(self, 'execution_progress'):
                    self.execution_progress['phase'] = 'finalization'
                
                # Get resource and error statistics
                resource_stats = self.get_resource_stats()
                error_stats = self.get_error_stats()
                
                # Prepare results
                final_results = {
                    "results": self.results,
                    "resource_stats": resource_stats,
                    "error_stats": error_stats,
                    "url": start_url,
                    "completed": True,
                    "timestamp": time.time()
                }
                
                # Print summary to terminal if not part of adaptive scraper
                if not crawler or not hasattr(crawler, 'is_adaptive_scraper') or not crawler.is_adaptive_scraper:
                    print(f"\n{'='*60}")
                    print(f"SCRAPING RESULTS FOR: {start_url}")
                    print(f"{'='*60}")
                    print(f"Extracted {len(self.results)} items")
                    print(f"Total requests: {resource_stats.get('total_requests', 0)}")
                    print(f"Success rate: {resource_stats.get('success_rate', 0):.2f}%")
                    print(f"Errors: {error_stats.get('total_errors', 0)}")
                    print(f"{'='*60}\n")
                
                return final_results
                
            except CircuitOpenError as e:
                logger.warning(f"Circuit breaker prevented request to {start_url}: {str(e)}")
                
                # Try to use cached results or fallback method
                if self.config.get('fallback_to_dom', True):
                    logger.info(f"Using DOM strategy as fallback for {start_url}")
                    
                    # Update progress
                    if hasattr(self, 'execution_progress'):
                        self.execution_progress['phase'] = 'fallback_dom_strategy'
                    
                    # Check if dom_strategy.execute is async before awaiting
                    if asyncio.iscoroutinefunction(self.dom_strategy.execute):
                        dom_result = await self.dom_strategy.execute(start_url, crawler=crawler, extraction_config=extraction_config)
                    else:
                        dom_result = self.dom_strategy.execute(start_url, crawler=crawler, extraction_config=extraction_config)
                    return dom_result
                
                raise
                
            except Exception as e:
                # Get domain for error context
                domain = urlparse(start_url).netloc
                
                # Classify and log the error
                error_context = {
                    'url': start_url,
                    'operation': 'execute',
                    'domain': domain
                }
                error_info = self._classify_error(e, error_context)
                
                # Record failure for circuit breaker
                self._record_failure(domain)
                
                logger.error(f"Error executing AI-guided strategy: {str(e)} - "
                           f"{error_info.get('category', 'unknown')}")
                
                # Try fallback method
                if self.config.get('fallback_to_dom', True):
                    logger.info(f"Using DOM strategy as fallback for {start_url}")
                    
                    # Update progress
                    if hasattr(self, 'execution_progress'):
                        self.execution_progress['phase'] = 'fallback_dom_strategy'
                    
                    # Check if dom_strategy.execute is async before awaiting
                    if asyncio.iscoroutinefunction(self.dom_strategy.execute):
                        dom_result = await self.dom_strategy.execute(start_url, crawler=crawler, extraction_config=extraction_config)
                    else:
                        dom_result = self.dom_strategy.execute(start_url, crawler=crawler, extraction_config=extraction_config)
                    return dom_result
                
                raise
                
        except Exception as e:
            logger.error(f"Unexpected error in _execute_inner: {str(e)}")
            return {
                "error": str(e),
                "url": start_url,
                "completed": False,
                "timestamp": time.time(),
                "results": []
            }
    
    def can_handle(self, url: str, **kwargs) -> bool:
        """
        Check if this strategy can handle the given URL.
        
        Args:
            url: URL to check
            **kwargs: Additional keyword arguments
            
        Returns:
            Boolean indicating if strategy can handle the URL
        """
        try:
            # AI-guided strategy can handle most URLs, especially complex ones
            if not url or not url.startswith(('http://', 'https://')):
                return False
                
            # Exclude file downloads
            if any(ext in url.lower() for ext in ['.pdf', '.doc', '.zip', '.exe', '.dmg']):
                return False
                
            return True
        except Exception:
            return False
