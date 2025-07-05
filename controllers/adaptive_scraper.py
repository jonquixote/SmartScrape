# __future__ import must be the first non-comment line
from __future__ import annotations

import logging
# Configure logging very early, before other project imports that might use it.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AdaptiveScraper")

import asyncio
import re
import time
import json
import os
import uuid
import traceback
from typing import Optional, Dict, Any, List, Callable, Tuple, Pattern, TYPE_CHECKING

from bs4 import BeautifulSoup

# --- Type Hinting Imports --- #
if TYPE_CHECKING:
    from core.ai_service import AIService as _AIService_Hint
    from ai_helpers.intent_parser import IntentParser as _IntentParser_Hint

# --- Runtime Imports --- #
try:
    from core.ai_service import AIService as AIService_runtime_class
except ImportError:
    AIService_runtime_class = None
    logger.info("AIService_runtime_class not imported during initial try-except.")

try:
    from ai_helpers.intent_parser import IntentParser as IntentParser_runtime_class
except ImportError:
    IntentParser_runtime_class = None
    logger.info("IntentParser_runtime_class not imported during initial try-except.")

# Service and component imports (ensure all other necessary project imports are here)
from core.service_registry import ServiceRegistry
# ... other existing core imports ...

# Conditional import for AIService and IntentParser for type hinting
if TYPE_CHECKING:
    from core.ai_service import AIService
    from ai_helpers.intent_parser import IntentParser

from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup

# Import URL validation
from utils.url_validator import URLValidator, enhanced_url_validation_and_discovery

# Import HTTP utilities for direct HTML fetching
from utils.http_utils import fetch_html

from ai_helpers.intent_parser import IntentParser, get_intent_parser  # Ensure global getter is imported
from core.service_registry import ServiceRegistry
from core.service_interface import BaseService
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_factory import StrategyFactory
from strategies.core.strategy_types import StrategyCapability, StrategyType, StrategyMetadata
from strategies.core.composite_strategy import SequentialStrategy, FallbackStrategy, PipelineStrategy

# Import circuit breaker from core instead of redefining it
from core.circuit_breaker import CircuitBreaker, CircuitBreakerManager, CircuitState

# Strategy imports
from strategies.ai_guided_strategy import AIGuidedStrategy
from strategies.multi_strategy import MultiStrategy, create_multi_strategy
from strategies.form_strategy import FormSearchEngine as FormStrategy
from strategies.url_param_strategy import URLParamSearchEngine as URLParamStrategy
from strategies.dom_strategy import DOMStrategy
from strategies.api_strategy import APIStrategy

# Component imports
from components.pattern_analyzer.base_analyzer import PatternAnalyzer, get_registry
from extraction.schema_extraction import SchemaExtractor, create_schema_from_intent
from extraction.content_extraction import MultiStrategyExtractor
from components.search_automation import SearchAutomator
from components.search_orchestrator import SearchOrchestrator
from components.search_term_generator import SearchTermGenerator
from components.site_discovery import SiteDiscovery
from components.domain_intelligence import DomainIntelligence
from utils.metrics_analyzer import MetricsAnalyzer
from utils.retry_utils import with_exponential_backoff as with_retry, with_async_retry
from utils.continuous_improvement import ContinuousImprovementSystem

# Pipeline imports
from extraction.pipeline_registry import PipelineRegistry
from core.pipeline.factory import PipelineFactory
from core.pipeline.pipeline import Pipeline
from core.pipeline.context import PipelineContext
from core.pipeline.stage import PipelineStage
from crawl4ai import AsyncWebCrawler

# Extraction framework imports
from extraction.pipeline_registry import register_built_in_pipelines as register_extraction_pipelines
from extraction.core.extraction_interface import BaseExtractor
from extraction.pattern_extractor import DOMPatternExtractor
from extraction.semantic_extractor import AISemanticExtractor
from extraction.structural_analyzer import DOMStructuralAnalyzer
from extraction.metadata_extractor import MetadataExtractorImpl
from extraction.content_normalizer_impl import ContentNormalizerImpl
from extraction.quality_evaluator_impl import QualityEvaluatorImpl
from extraction.schema_manager import SchemaManager

from components.pattern_analyzer.default_analyzer import DefaultPatternAnalyzer

# Attempt to import AIService, handle if not found
try:
    from core.ai_service import AIService # Actual class
except ImportError:
    # logger is now defined before this usage
    logger.warning("AIService module not found. AI-related functionalities might be limited.")

# Default AI Model Configuration (can be overridden by a config file)
DEFAULT_AVAILABLE_MODELS = {
    "gemini-2.0-flash-lite": {"score": 125, "available": True, "api_key_env": "GOOGLE_API_KEY"},
    "gemini-2.0-flash": {"score": 120, "available": True, "api_key_env": "GOOGLE_API_KEY"},
    "gemini-1.5-flash": {"score": 100, "available": True, "api_key_env": "GOOGLE_API_KEY"},
    "gemini-1.5-pro": {"score": 115, "available": True, "api_key_env": "GOOGLE_API_KEY"},
    "gpt-4-turbo": {"score": 110, "available": False, "api_key_env": "OPENAI_API_KEY"}, # Example, assuming not available by default
    "gpt-3.5-turbo": {"score": 90, "available": True, "api_key_env": "OPENAI_API_KEY"}
        }

class AdaptiveScraper:
    """
    Controller class for adaptive scraping operations.

    This class coordinates multiple scraping strategies, implements
    execution pipelines with fallbacks, and manages resources efficiently.
    """

    async def _discover_target_sites(self, discovery_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discover target sites based on the query and search terms.
        """
        try:
            query = discovery_context.get("query", "")
            search_terms = discovery_context.get("search_terms", [])
            
            # Use the site discovery component if available
            if hasattr(self, 'site_discovery') and self.site_discovery:
                # Use the analyze_site method to analyze known relevant sites
                analyzed_urls = []
                base_urls = await self._generate_search_urls(search_terms)
                
                for url in base_urls[:3]:  # Analyze first 3 URLs
                    try:
                        analysis = await self.site_discovery.analyze_site(url)
                        if analysis.get("reachable", False):
                            analyzed_urls.append(url)
                    except Exception as e:
                        logger.warning(f"Failed to analyze site {url}: {e}")
                        continue
                
                return {"urls": analyzed_urls if analyzed_urls else base_urls}
            
            # Fallback: generate common search URLs
            return {"urls": await self._generate_search_urls(search_terms)}
            
        except Exception as e:
            logger.error(f"Site discovery error: {e}")
            return {"urls": [], "error": str(e)}
    
    async def _generate_search_urls(self, search_terms: List[str]) -> List[str]:
        """
        Generate content URLs for educational and news sites instead of search engines.
        """
        urls = []
        
        # Take the first search term or join multiple terms
        query = search_terms[0] if search_terms else "news"
        encoded_query = query.replace(" ", "+")
        
        # Educational and content sites instead of search engines
        content_sites = [
            f"https://en.wikipedia.org/wiki/Special:Search?search={encoded_query}",
            f"https://www.reddit.com/search/?q={encoded_query}",
            f"https://stackoverflow.com/search?q={encoded_query}",
        ]
        
        # For Tesla news specifically, add relevant news sites
        if "tesla" in query.lower():
            tesla_sites = [
                "https://www.teslarati.com/",
                "https://electrek.co/guides/tesla/",
                "https://www.reuters.com/business/autos-transportation/",
                "https://techcrunch.com/category/transportation/",
                "https://www.theverge.com/transportation",
            ]
            urls.extend(tesla_sites)
        
        # For general news queries, add news sites
        if "news" in query.lower():
            news_sites = [
                "https://www.reuters.com/",
                "https://www.bbc.com/news",
                "https://www.cnn.com/",
                "https://techcrunch.com/",
                "https://www.theverge.com/",
            ]
            urls.extend(news_sites)
        
        urls.extend(content_sites)
        
        # Remove duplicates and return first 10
        unique_urls = list(dict.fromkeys(urls))
        return unique_urls[:10]

    def __init__(self, config: Optional[Dict[str, Any]] = None, service_registry: Optional[ServiceRegistry] = None):
        logger.info("AdaptiveScraper __init__ called.")
        self._initialized = False
        self.config = config or {}
        self.service_registry = service_registry or ServiceRegistry()

        # Initialize AIService (Moved to be first)
        self.ai_service: Optional[_AIService_Hint] = None # Initialize to None
        try:
            self.ai_service = self.service_registry.get_service("ai_service")
            if self.ai_service:
                logger.info("AIService successfully retrieved from service registry.")
            else:
                logger.warning("AIService was retrieved as None from service registry. Will attempt creation if class is available.")
                if AIService_runtime_class is None:
                     logger.warning("AIService_runtime_class not available, cannot create. AI functionalities will be unavailable.")
        except KeyError:
            logger.info("AIService not found in service registry. Attempting to create and register.")
            if AIService_runtime_class and callable(AIService_runtime_class):
                try:
                    # Try creating AIService without config parameter first
                    try:
                        ai_service_instance = AIService_runtime_class()
                    except TypeError:
                        # If that fails, try with config parameter
                        ai_service_config = self.config.get('ai_service_config')
                        ai_service_instance = AIService_runtime_class(ai_service_config)
                    
                    if hasattr(ai_service_instance, 'initialize') and callable(getattr(ai_service_instance, 'initialize')):
                        ai_service_config = self.config.get('ai_service_config')
                        ai_service_instance.initialize(config=ai_service_config)
                    self.service_registry.register_service("ai_service", ai_service_instance)
                    self.ai_service = ai_service_instance
                    logger.info("AIService created and registered successfully using runtime class.")
                except Exception as e:
                    logger.error(f"Error creating/initializing AIService using runtime class: {e}. AI functionalities may be unavailable.", exc_info=True)
                    self.ai_service = None 
            else:
                 logger.warning("AIService_runtime_class is not available or not callable. AI functionalities will be unavailable.")
                 self.ai_service = None
        except Exception as e: 
            logger.error(f"An unexpected error occurred during AIService retrieval/creation: {e}", exc_info=True)
            self.ai_service = None

        # Initialize IntentParser (Now second, uses self.ai_service)
        self.intent_parser: Optional[_IntentParser_Hint] = None 
        try:
            retrieved_intent_parser = self.service_registry.get_service("intent_parser")
            if retrieved_intent_parser:
                if IntentParser_runtime_class and isinstance(retrieved_intent_parser, IntentParser_runtime_class):
                    self.intent_parser = retrieved_intent_parser
                    logger.info("IntentParser retrieved from service registry and type matched.")
                elif not IntentParser_runtime_class: 
                    self.intent_parser = retrieved_intent_parser
                    logger.info("IntentParser retrieved from service registry (runtime type check skipped as class unavailable).")
                else: 
                    self.intent_parser = retrieved_intent_parser 
                    logger.warning(f"Retrieved intent_parser type mismatch. Expected {IntentParser_runtime_class}, got {type(retrieved_intent_parser)}. Assigned retrieved instance anyway.")
        except KeyError:
            logger.info("IntentParser not found in registry. Attempting to create and register new instance.")
            if IntentParser_runtime_class and callable(IntentParser_runtime_class):
                try:
                    new_intent_parser = IntentParser_runtime_class(ai_service=self.ai_service)
                    if hasattr(new_intent_parser, 'initialize') and callable(getattr(new_intent_parser, 'initialize')):
                        new_intent_parser.initialize() 
                    self.service_registry.register_service("intent_parser", new_intent_parser)
                    self.intent_parser = new_intent_parser
                    logger.info("New IntentParser instance created and registered using runtime class.")
                except Exception as e:
                    logger.error(f"Failed to create/register IntentParser using runtime class: {e}", exc_info=True)
            elif 'get_intent_parser' in globals() and callable(get_intent_parser):
                logger.warning("IntentParser_runtime_class not available/callable or failed. Attempting fallback to global get_intent_parser.")
                try:
                    import inspect
                    sig = inspect.signature(get_intent_parser)
                    if 'ai_service' in sig.parameters:
                        self.intent_parser = get_intent_parser(ai_service=self.ai_service)
                    else:
                        self.intent_parser = get_intent_parser()
                    self.service_registry.register_service("intent_parser", self.intent_parser)
                    logger.info("IntentParser created via global getter and registered.")
                except Exception as e:
                    logger.error(f"Failed to create IntentParser via global getter: {e}", exc_info=True)
            else:
                logger.warning("IntentParser_runtime_class and global get_intent_parser are unavailable. Intent parsing features may be disabled.")
        except Exception as e: 
            logger.error(f"An unexpected error occurred during IntentParser retrieval/creation: {e}", exc_info=True)
            self.intent_parser = None

        # Initialize other components that may depend on ai_service or intent_parser
        self.strategy_factory = StrategyFactory(self.service_registry)
        
        # Initialize extraction framework components
        try:
            # Initialize core extraction components
            self.structural_analyzer = DOMStructuralAnalyzer()
            self.metadata_extractor = MetadataExtractorImpl()
            self.content_normalizer = ContentNormalizerImpl()
            self.quality_evaluator = QualityEvaluatorImpl()
            
            # Initialize pattern extractor with AI service if available
            try:
                self.pattern_extractor = DOMPatternExtractor(ai_service=self.ai_service)
            except TypeError:
                self.pattern_extractor = DOMPatternExtractor()
                if hasattr(self.pattern_extractor, 'ai_service'):
                    self.pattern_extractor.ai_service = self.ai_service
            
            # Initialize semantic extractor with AI service if available
            try:
                self.semantic_extractor = AISemanticExtractor(ai_service=self.ai_service)
            except TypeError:
                self.semantic_extractor = AISemanticExtractor()
                if hasattr(self.semantic_extractor, 'ai_service'):
                    self.semantic_extractor.ai_service = self.ai_service

            logger.info("Extraction framework components initialized successfully")
        except Exception as e:
            logger.warning(f"Error initializing extraction framework components: {e}")
            # Set fallback empty objects
            self.structural_analyzer = None
            self.metadata_extractor = None
            self.content_normalizer = None
            self.quality_evaluator = None
            self.pattern_extractor = None
            self.semantic_extractor = None
        
        # Register essential strategies with the strategy factory
        try:
            # Register AI-guided strategy
            if hasattr(self.strategy_factory, 'register_strategy'):
                from strategies.ai_guided_strategy import AIGuidedStrategy
                self.strategy_factory.register_strategy(AIGuidedStrategy)
                
                # Register multi-strategy
                from strategies.multi_strategy import MultiStrategy
                self.strategy_factory.register_strategy(MultiStrategy)
                
                # Register DOM strategy
                from strategies.dom_strategy import DOMStrategy
                self.strategy_factory.register_strategy(DOMStrategy)
                
                # Register form strategy
                from strategies.form_strategy import FormSearchEngine
                self.strategy_factory.register_strategy(FormSearchEngine)
                
                logger.info("Essential scraping strategies registered successfully")
            elif hasattr(self.strategy_factory, 'register'):
                # Alternative registration method
                from strategies.ai_guided_strategy import AIGuidedStrategy
                ai_strategy = AIGuidedStrategy(ai_service=self.ai_service)
                self.strategy_factory.register("ai_guided", ai_strategy)
                
                from strategies.multi_strategy import MultiStrategy
                multi_strategy = MultiStrategy(ai_service=self.ai_service)
                self.strategy_factory.register("multi_strategy", multi_strategy)
                
                from strategies.dom_strategy import DOMStrategy
                dom_strategy = DOMStrategy()
                self.strategy_factory.register("dom_strategy", dom_strategy)
                
                from strategies.form_strategy import FormSearchEngine
                form_strategy = FormSearchEngine()
                self.strategy_factory.register("form_strategy", form_strategy)
                
                logger.info("Essential scraping strategies registered via register method")
        except Exception as e:
            logger.warning(f"Error registering strategies: {e}")

        # Ensure components like SchemaExtractor can handle self.ai_service being None
        try:
            self.schema_extractor = SchemaExtractor(ai_service=self.ai_service)
        except TypeError:
            # If SchemaExtractor doesn't accept ai_service parameter, create without it
            self.schema_extractor = SchemaExtractor()
            if hasattr(self.schema_extractor, 'ai_service'):
                self.schema_extractor.ai_service = self.ai_service
        self.content_extractor = MultiStrategyExtractor(use_ai=bool(self.ai_service))
        # Set the ai_service and strategy_factory as attributes if the object supports it
        if hasattr(self.content_extractor, 'ai_service'):
            self.content_extractor.ai_service = self.ai_service
        if hasattr(self.content_extractor, 'strategy_factory'):
            self.content_extractor.strategy_factory = self.strategy_factory
        
        self.pattern_analyzer_registry = get_registry()
        if not self.pattern_analyzer_registry.patterns: 
            logger.info("Pattern analyzer registry is empty. Registering DefaultPatternAnalyzer.")
            # Ensure DefaultPatternAnalyzer is imported or defined
            if 'DefaultPatternAnalyzer' in globals() and callable(DefaultPatternAnalyzer):
                default_analyzer = DefaultPatternAnalyzer()
                # Register a pattern instead of the analyzer itself
                self.pattern_analyzer_registry.register_pattern(
                    pattern_type="default",
                    url="*",  # wildcard for all URLs
                    pattern_data={"type": "default", "analyzer": default_analyzer.name},
                    confidence=0.7
                )
                logger.info("DefaultPatternAnalyzer registered in pattern registry.")
            else:
                logger.warning("DefaultPatternAnalyzer not available to register.")

        self.metrics_analyzer = MetricsAnalyzer()
        self.continuous_improvement = ContinuousImprovementSystem()
        self.jobs: Dict[str, Dict[str, Any]] = {}

        self.circuit_breaker_manager = CircuitBreakerManager()
        
        # Initialize search pipeline steps
        self.search_pipeline_steps = [
            self._prepare_search_terms,
            self._select_search_strategy,
            self._execute_search,
            self._process_results,
            self._apply_fallback_if_needed
        ]
        
        self.pipeline_registry = PipelineRegistry()
        if 'register_extraction_pipelines' in globals() and callable(register_extraction_pipelines):
            register_extraction_pipelines()  # Call without arguments
        self.pipeline_factory = PipelineFactory(self.pipeline_registry)

        # Initialize HTML service for robust content fetching
        try:
            from core.html_service import HTMLService
            self.html_service = HTMLService()
            logger.info("HTML service initialized successfully")
        except ImportError:
            logger.warning("HTML service not available, will use fallback HTTP fetching")
            self.html_service = None
        except Exception as e:
            logger.warning(f"Error initializing HTML service: {e}")
            self.html_service = None

        # Pass self.ai_service (which could be None)
        try:
            self.site_discovery = SiteDiscovery(ai_service=self.ai_service)
        except TypeError:
            self.site_discovery = SiteDiscovery()
            if hasattr(self.site_discovery, 'ai_service'):
                self.site_discovery.ai_service = self.ai_service
                
        try:
            self.domain_intelligence = DomainIntelligence(ai_service=self.ai_service)
        except TypeError:
            self.domain_intelligence = DomainIntelligence()
            if hasattr(self.domain_intelligence, 'ai_service'):
                self.domain_intelligence.ai_service = self.ai_service

        try:
            self.search_term_generator = SearchTermGenerator(ai_service=self.ai_service)
        except TypeError:
            self.search_term_generator = SearchTermGenerator()
            if hasattr(self.search_term_generator, 'ai_service'):
                self.search_term_generator.ai_service = self.ai_service
        
        # Register search_term_generator as a service
        self.service_registry.register_service("search_term_generator", self.search_term_generator)
        try:
            self.search_orchestrator = SearchOrchestrator(
                default_engines=["form-search-engine", "url-param-search-engine"],
                cache_results=True,
                use_fallbacks=True,
                fallback_threshold=0.4,
                performance_tracking=True
            )
        except TypeError:
            # Try without parameters if the constructor doesn't accept them
            self.search_orchestrator = SearchOrchestrator()
            
        # Set the components as attributes if the object supports them
        if hasattr(self.search_orchestrator, 'site_discovery'):
            self.search_orchestrator.site_discovery = self.site_discovery
        if hasattr(self.search_orchestrator, 'domain_intelligence'):
            self.search_orchestrator.domain_intelligence = self.domain_intelligence
        if hasattr(self.search_orchestrator, 'search_term_generator'):
            self.search_orchestrator.search_term_generator = self.search_term_generator
        if hasattr(self.search_orchestrator, 'ai_service'):
            self.search_orchestrator.ai_service = self.ai_service
                
        try:
            self.search_automator = SearchAutomator(config=self.config)
        except TypeError:
            # Try without config parameter
            self.search_automator = SearchAutomator()
            
        # Set the components as attributes if the object supports them
        if hasattr(self.search_automator, 'orchestrator'):
            self.search_automator.orchestrator = self.search_orchestrator
        if hasattr(self.search_automator, 'ai_service'):
            self.search_automator.ai_service = self.ai_service
        
        self.url_validator = URLValidator()
        self.schema_manager = SchemaManager()

        # Call BaseService's initialize if AdaptiveScraper inherits from it and it has initialize
        if hasattr(super(), 'initialize') and callable(getattr(super(), 'initialize')):
            try:
                super().initialize(config) 
            except Exception as e:
                logger.error(f"Error calling super().initialize(): {e}", exc_info=True)
        
            # Initialize and set up ExtractionCoordinator relationship
            self._setup_extraction_coordinator()
            
            # Initialize UniversalHunter for intelligent hunting
            self.hunter = None  # Will be initialized on first use
            
            self._initialized = True
            logger.info("AdaptiveScraper __init__ completed.")
            logger.info(f"AdaptiveScraper __init__: hasattr(self, 'process_user_request') after init: {hasattr(self, 'process_user_request')}")
    
    def _setup_extraction_coordinator(self):
        """Set up the relationship with ExtractionCoordinator to avoid circular imports."""
        try:
            from controllers.extraction_coordinator import get_extraction_coordinator
            coordinator = get_extraction_coordinator()
            if coordinator and hasattr(coordinator, 'set_adaptive_scraper'):
                coordinator.set_adaptive_scraper(self)
                logger.info("ExtractionCoordinator relationship established")
        except Exception as e:
            logger.warning(f"Could not set up ExtractionCoordinator relationship: {e}")

    async def process_user_request(self, user_query: str, session_id: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processes a user's query to understand intent, gather information, and return structured data.
        This is the primary entry point for the AdaptiveScraper's functionality.
        UPDATED: Attempts to use UniversalHunter for intelligent hunting, falls back to SimpleScraper.
        """
        options = options or {}
        logger.info(f"AdaptiveScraper received user request: '{user_query[:100]}' (Session: {session_id}) Options: {options}")

        # Check if intelligent hunting is enabled (default True for /scrape-intelligent endpoint)
        enable_intelligent_hunting = options.get('enable_intelligent_hunting', True)
        
        if enable_intelligent_hunting:
            try:
                # Try using UniversalHunter for intelligent hunting
                return await self._process_with_intelligent_hunting(user_query, session_id, options)
            except Exception as e:
                logger.warning(f"UniversalHunter failed: {e}. Falling back to SimpleScraper.")
                # Fall back to SimpleScraper if UniversalHunter fails

        try:
            # Import SimpleScraper for single-pass scraping (fallback)
            from controllers.simple_scraper import SimpleScraper
            
            # Use SimpleScraper with proper context manager for session handling
            logger.info(f"Using SimpleScraper for query: '{user_query[:100]}'")
            
            async with SimpleScraper() as simple_scraper:
                result = await simple_scraper.scrape_query(user_query)
            
            # Ensure consistent result format - SimpleScraper returns 'items' and 'status'
            if result.get("status") == "success":
                return {
                    "success": True,
                    "data": result.get("items", []),  # SimpleScraper uses 'items' not 'results'
                    "metadata": {
                        "query": user_query,
                        "session_id": session_id,
                        "scraper_type": "SimpleScraper",
                        "urls_processed": len(result.get("urls_processed", [])),
                        "total_items": result.get("total_items", 0),
                        "processing_time": 0  # SimpleScraper doesn't track timing yet
                    },
                    "sources": result.get("urls_processed", [])
                }
            else:
                return {
                    "success": False,
                    "error": f"SimpleScraper returned status: {result.get('status', 'unknown')}",
                    "data": [],
                    "metadata": {
                        "query": user_query,
                        "session_id": session_id,
                        "scraper_type": "SimpleScraper"
                    }
                }
                
        except ImportError:
            logger.warning("SimpleScraper not available, falling back to legacy method")
            # Fall back to original method if SimpleScraper not available
            return await self._process_user_request_legacy(user_query, session_id, options)
        except Exception as e:
            logger.error(f"Error in SimpleScraper processing: {e}")
            # Fall back to original method on error
            return await self._process_user_request_legacy(user_query, session_id, options)

    async def _process_with_intelligent_hunting(self, user_query: str, session_id: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Use UniversalHunter for intelligent content hunting and extraction.
        """
        import time
        start_time = time.time()
        
        logger.info(f"Using UniversalHunter for intelligent hunting: '{user_query[:100]}'")
        
        try:
            # Use UniversalHunter for intelligent hunting (now fixed!)
            from intelligence.universal_hunter import UniversalHunter
            import aiohttp
            
            # Create an aiohttp session for UniversalHunter
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as http_session:
                # Create UniversalHunter with the aiohttp session
                hunter = UniversalHunter(session=http_session)
                
                # Get URLs using the enhanced DuckDuckGo generator with fallback
                start_url = options.get('start_url') if options else None
                if start_url:
                    urls = [start_url]
                    logger.info(f"Using provided start_url: {start_url}")
                else:
                    urls = await self._discover_hunting_urls(user_query, options)
                    logger.info(f"Discovered {len(urls)} URLs with enhanced fallback system")
                
                # Determine result count configuration
                min_results = options.get('min_results', 10) if options else 10
                max_results = options.get('max_results', 50) if options else 50
                target_results = options.get('target_results', min_results) if options else min_results
                
                # Use configuration defaults if not specified
                if not options or ('min_results' not in options and 'max_results' not in options and 'target_results' not in options):
                    from config import DEFAULT_MIN_RESULTS, DEFAULT_MAX_RESULTS, DEFAULT_TARGET_RESULTS
                    min_results = DEFAULT_MIN_RESULTS
                    max_results = DEFAULT_MAX_RESULTS
                    target_results = DEFAULT_TARGET_RESULTS
                
                logger.info(f"Result configuration - Min: {min_results}, Target: {target_results}, Max: {max_results}")
                
                # Select the best URLs for hunting (not just the first ones)
                hunting_url_count = min(len(urls), 15)  # Process up to 15 URLs for better results
                hunting_urls = urls[:hunting_url_count]
                
                logger.info(f"🎯 Using top {len(hunting_urls)} URLs for hunting")
                for i, url in enumerate(hunting_urls[:5], 1):  # Log first 5 URLs
                    logger.info(f"   🔍 Hunt URL {i}: {url}")
                
                # Use UniversalHunter to process the discovered URLs with specified result count
                hunting_targets = await hunter.hunt(user_query, hunting_urls, target_results)
                
                logger.info(f"🎯 UniversalHunter returned {len(hunting_targets)} targets")
                
                # Convert hunting targets to results format
                all_results = []
                for i, target in enumerate(hunting_targets):
                    logger.info(f"   Target {i+1}: {target.title[:50]}... (Score: {target.quality_score:.2f})")
                    result = {
                        'title': target.title,
                        'content': target.content_preview,
                        'url': target.url,
                        'relevance_score': target.relevance_score,
                        'quality_score': target.quality_score,
                        'content_type': target.content_type,
                        'extraction_method': target.extraction_method,
                        'metadata': target.metadata
                    }
                    all_results.append(result)
                
                logger.info(f"💫 Converted {len(all_results)} targets to results")
                
                # Ensure we have at least the minimum number of results
                if len(all_results) < min_results:
                    logger.info(f"Only {len(all_results)} results found, need {min_results}. Attempting to get more...")
                    # Try to get more results with more aggressive hunting
                    if len(urls) > 10:
                        additional_targets = await hunter.hunt(user_query, urls[10:20], min_results - len(all_results), direct_urls=True)
                        for target in additional_targets:
                            if len(all_results) >= min_results:
                                break
                            result = {
                                'title': target.title,
                                'content': target.content_preview,
                                'url': target.url,
                                'relevance_score': target.relevance_score,
                                'quality_score': target.quality_score,
                                'content_type': target.content_type,
                                'extraction_method': target.extraction_method,
                                'metadata': target.metadata
                            }
                            all_results.append(result)
                
                # Limit to max_results if exceeded
                if len(all_results) > max_results:
                    all_results = all_results[:max_results]
                    logger.info(f"Limited results to {max_results} as requested")
                
                processing_time = time.time() - start_time
                
                # Format the result
                logger.info(f"🚀 Final result: success=True, {len(all_results)} results")
                return {
                    "success": True,
                    "results": all_results,  # Use 'results' key for consistency 
                    "data": all_results,
                    "metadata": {
                        "query": user_query,
                        "session_id": session_id,
                        "scraper_type": "UniversalHunter",
                        "intelligent_hunting": True,
                        "urls_processed": len(urls),
                        "total_items": len(all_results),
                        "processing_time": processing_time,
                        "targets_found": len(hunting_targets),
                        "extraction_strategy": "intelligent_universal_hunting",
                        "result_configuration": {
                            "min_results": min_results,
                            "max_results": max_results,
                            "target_results": target_results,
                            "achieved_results": len(all_results)
                        }
                    },
                    "sources": urls[:len(all_results)]
                }
                
        except ImportError as e:
            logger.warning(f"UniversalHunter not available: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error in UniversalHunter processing: {e}")
            raise e

    async def _discover_hunting_urls(self, query: str, options: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Discover URLs for hunting using aggressive multi-source search strategy
        """
        try:
            # First, try the aggressive multi-source discovery
            from components.aggressive_url_discovery import AggressiveURLDiscovery
            
            discovery_engine = AggressiveURLDiscovery()
            
            # Determine query type for better targeting
            query_type = self._detect_query_type(query)
            
            # Get target URL count based on result requirements
            min_results = options.get('min_results', 10) if options else 10
            target_url_count = max(min_results * 3, 50)  # Get 3x more URLs than needed results
            
            logger.info(f"🚀 Aggressive URL discovery: targeting {target_url_count} URLs for query type '{query_type}'")
            
            # Discover URLs aggressively
            discovered_urls = await discovery_engine.discover_urls(
                query=query,
                max_urls=target_url_count,
                query_type=query_type
            )
            
            # Convert to simple URL list
            urls = [url_obj.url for url_obj in discovered_urls]
            
            if len(urls) >= 20:
                logger.info(f"✅ Aggressive discovery successful: {len(urls)} URLs")
                return urls
            else:
                logger.warning(f"⚠️ Aggressive discovery yielded only {len(urls)} URLs, falling back...")
                
        except Exception as e:
            logger.warning(f"Aggressive URL discovery failed: {e}")
            logger.info("Falling back to enhanced DuckDuckGo system...")
        
        # Fallback to enhanced DuckDuckGo system
        try:
            # Import DuckDuckGo URL generator
            from components.duckduckgo_url_generator import DuckDuckGoURLGenerator
            
            url_generator = DuckDuckGoURLGenerator()
            
            # Determine how many URLs we need
            min_results = options.get('min_results', 10) if options else 10
            url_count = max(min_results * 2, 30)  # Get 2x more URLs than needed results
            
            logger.info(f"🔍 Enhanced DuckDuckGo discovery: targeting {url_count} URLs")
            
            # Generate URLs with enhanced approach
            url_scores = url_generator.generate_urls_enhanced(query, max_urls=url_count)
            
            # Extract URLs from scores
            urls = [score.url for score in url_scores if score.url]
            
            if len(urls) >= 10:
                logger.info(f"✅ Enhanced DuckDuckGo successful: {len(urls)} URLs")
                return urls
            else:
                logger.warning(f"⚠️ Enhanced DuckDuckGo yielded only {len(urls)} URLs, trying emergency fallback...")
                
        except Exception as e:
            logger.warning(f"Enhanced DuckDuckGo discovery failed: {e}")
        
        # Emergency fallback - create search URLs directly
        emergency_urls = self._create_emergency_search_urls(query)
        logger.info(f"🆘 Emergency fallback: {len(emergency_urls)} URLs")
        
        return emergency_urls

    def _create_emergency_search_urls(self, query: str) -> List[str]:
        """
        Create emergency direct content URLs as last resort fallback
        NEVER return search engine result pages - only direct content URLs
        """
        # Detect query type for better targeting
        query_type = self._detect_query_type(query)
        
        # Direct content URLs - NO SEARCH ENGINE RESULT PAGES
        if query_type == 'tech':
            emergency_urls = [
                'https://techcrunch.com/',
                'https://www.theverge.com/',
                'https://arstechnica.com/',
                'https://www.wired.com/',
                'https://venturebeat.com/',
                'https://www.engadget.com/',
                'https://mashable.com/tech',
                'https://www.cnet.com/news/',
                'https://www.technologyreview.com/',
                'https://spectrum.ieee.org/'
            ]
        elif query_type == 'finance':
            emergency_urls = [
                'https://finance.yahoo.com/news/',
                'https://www.marketwatch.com/latest-news',
                'https://www.cnbc.com/business/',
                'https://fortune.com/section/finance/',
                'https://www.businessinsider.com/',
                'https://seekingalpha.com/news',
                'https://www.fool.com/investing/',
                'https://www.wsj.com/news/business',
                'https://www.reuters.com/business/',
                'https://www.bloomberg.com/news/'
            ]
        elif query_type == 'science':
            emergency_urls = [
                'https://www.nature.com/news',
                'https://www.sciencedaily.com/news/',
                'https://www.newscientist.com/',
                'https://www.scientificamerican.com/',
                'https://phys.org/news/',
                'https://www.space.com/news',
                'https://www.livescience.com/',
                'https://www.popsci.com/',
                'https://www.sciencenews.org/',
                'https://www.eurekalert.org/'
            ]
        else:  # news or general
            emergency_urls = [
                'https://www.reuters.com/world/',
                'https://www.bbc.com/news',
                'https://www.npr.org/sections/news/',
                'https://www.theguardian.com/international',
                'https://www.nytimes.com/',
                'https://www.washingtonpost.com/',
                'https://www.cnn.com/',
                'https://abcnews.go.com/',
                'https://www.cbsnews.com/news/',
                'https://www.nbcnews.com/'
            ]
        
        return emergency_urls

    def _detect_query_type(self, query: str) -> str:
        """
        Detect the type of query for better URL discovery targeting
        
        Args:
            query: The search query
            
        Returns:
            Query type string ('news', 'tech', 'finance', 'academic', 'general')
        """
        query_lower = query.lower()
        
        # News-related keywords
        news_keywords = ['news', 'latest', 'recent', 'breaking', 'update', 'today', 'yesterday', 
                        'this week', 'announcement', 'report', 'press release']
        
        # Tech-related keywords  
        tech_keywords = ['ai', 'artificial intelligence', 'machine learning', 'tech', 'technology',
                        'software', 'programming', 'code', 'development', 'algorithm', 'data',
                        'computer', 'digital', 'cyber', 'blockchain', 'crypto']
                        
        # Finance-related keywords
        finance_keywords = ['stock', 'market', 'finance', 'investment', 'trading', 'price',
                           'economy', 'economic', 'financial', 'earnings', 'revenue', 'profit']
                           
        # Academic/research keywords
        academic_keywords = ['research', 'study', 'paper', 'journal', 'academic', 'university',
                            'scholar', 'analysis', 'breakthrough', 'discovery', 'findings']
        
        # Count keyword matches
        if any(keyword in query_lower for keyword in news_keywords):
            return 'news'
        elif any(keyword in query_lower for keyword in tech_keywords):
            return 'tech'
        elif any(keyword in query_lower for keyword in finance_keywords):
            return 'finance' 
        elif any(keyword in query_lower for keyword in academic_keywords):
            return 'academic'
        else:
            return 'general'

    def _format_hunting_results(self, hunting_targets: List, query: str) -> List[Dict[str, Any]]:
        """
        Format hunting targets into the expected data structure
        """
        formatted_results = []
        
        for target in hunting_targets:
            try:
                # Extract data from hunting target
                result = {
                    "title": getattr(target, 'title', 'No title'),
                    "content": getattr(target, 'content_preview', 'No content'),  # Use content_preview
                    "url": getattr(target, 'url', 'Unknown URL'),
                    "relevance_score": getattr(target, 'relevance_score', 0.0),
                    "quality_score": getattr(target, 'quality_score', 0.0),
                    "extraction_type": "intelligent_hunting",
                    "metadata": getattr(target, 'metadata', {})
                }
                formatted_results.append(result)
            except Exception as e:
                logger.warning(f"Error formatting hunting target: {e}")
                
        logger.info(f"Formatted {len(formatted_results)} hunting results for query: {query}")
        return formatted_results

    async def _process_user_request_legacy(self, user_query: str, session_id: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Legacy implementation of process_user_request - kept as fallback.
        """
        options = options or {}
        logger.info(f"Using legacy AdaptiveScraper processing for: '{user_query[:100]}' (Session: {session_id})")

        if not self.ai_service:
            logger.error("AI service is not available. Cannot process user request.")
            return {"error": "AI service not configured or failed to initialize.", "status_code": 503}
        
        if not self.intent_parser:
            logger.warning("Intent parser is not available. Proceeding without intent parsing if possible, or failing if critical.")
            return {"error": "Intent parser not configured or failed to initialize.", "status_code": 503}

        try:
            # Step 1: Parse user intent to understand what they want to scrape
            logger.info(f"Parsing intent for query: '{user_query[:100]}'")
            intent_result = await self.intent_parser.parse_intent(user_query)
            
            if not intent_result or intent_result.get("error"):
                logger.error(f"Intent parsing failed: {intent_result.get('error') if intent_result else 'No result'}")
                return {"error": "Failed to understand the query intent.", "status_code": 400}
            
            # Step 2: Extract target URLs from intent or discover them
            target_urls = intent_result.get("target_urls", [])
            search_terms = intent_result.get("search_terms", [user_query])
            
            # If no specific URLs provided, use site discovery to find relevant sites
            if not target_urls:
                logger.info("No target URLs provided, using site discovery...")
                try:
                    # Use the site discovery component to find relevant URLs
                    discovery_context = {
                        "query": user_query,
                        "search_terms": search_terms,
                        "intent": intent_result
                    }
                    discovered = await self._discover_target_sites(discovery_context)
                    target_urls = discovered.get("urls", [])
                    
                    if not target_urls:
                        # Fallback: generate URLs based on search terms
                        target_urls = await self._generate_search_urls(search_terms)
                        
                except Exception as e:
                    logger.warning(f"Site discovery failed, using fallback URL generation: {e}")
                    target_urls = await self._generate_search_urls(search_terms)
            
            if not target_urls:
                logger.error("No target URLs could be determined for the query")
                return {"error": "Could not determine relevant websites to scrape for your query.", "status_code": 400}
            
            logger.info(f"Target URLs determined: {target_urls[:3]}{'...' if len(target_urls) > 3 else ''}")
            
            # Step 3: Execute scraping pipeline for each URL
            all_results = []
            successful_scrapes = 0
            
            for url in target_urls[:5]:  # Limit to 5 URLs to avoid overwhelming results
                try:
                    logger.info(f"Scraping URL: {url}")
                    # Use comprehensive extraction method
                    scrape_result = await self._comprehensive_extraction(url, user_query, intent_result, options)
                    
                    if scrape_result.get("success") and scrape_result.get("data"):
                        all_results.extend(scrape_result["data"])
                        successful_scrapes += 1
                        logger.info(f"Successfully scraped {len(scrape_result['data'])} items from {url}")
                    else:
                        logger.warning(f"Scraping failed or returned no results for {url}: {scrape_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")
                    continue
            
            # Step 4: Process and format results
            if not all_results:
                logger.warning("No results obtained from any target URLs")
                return {
                    "query_received": user_query,
                    "extracted_data": {
                        "title": f"No Results Found for '{user_query}'",
                        "summary": "No relevant information could be extracted from the target websites.",
                        "items_found": [],
                        "urls_attempted": target_urls,
                        "successful_scrapes": 0
                    },
                    "status": "success",
                    "message": "Query processed but no results found."
                }
            
            # Limit and format results
            limited_results = all_results[:20]  # Limit to 20 items
            
            result_data = {
                "query_received": user_query,
                "extracted_data": {
                    "title": f"Results for '{user_query}'",
                    "summary": f"Found {len(limited_results)} relevant items from {successful_scrapes} websites.",
                    "items_found": limited_results,
                    "total_results": len(all_results),
                    "urls_scraped": successful_scrapes,
                    "search_terms": search_terms
                },
                "status": "success",
                "message": f"Successfully extracted {len(limited_results)} items from {successful_scrapes} sources."
            }
            
            logger.info(f"Successfully processed user request for: '{user_query[:100]}' - {len(limited_results)} results")
            return result_data

        except Exception as e:
            logger.error(f"Error processing user request '{user_query[:100]}': {e}", exc_info=True)
            return {"error": f"An unexpected error occurred while processing your request: {str(e)}", "status_code": 500}

    # ===========================
    # INTELLIGENT EXTRACTION SYSTEM WITH COORDINATOR
    # ===========================

    async def _crawl4ai_intelligent_extraction(self, url: str, user_query: str, intent_result: Dict[str, Any], 
                                             options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        SIMPLIFIED: Basic content extraction using HTTP requests
        
        This method was originally complex with Crawl4AI integration, but is now simplified
        to provide basic content extraction without the recursive complexity.
        """
        try:
            import aiohttp
            from bs4 import BeautifulSoup
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract basic content
                        articles = []
                        
                        # Look for common content patterns
                        selectors = ['article', '.article', '.post', '.content', 'h1, h2, h3']
                        
                        for selector in selectors:
                            elements = soup.select(selector)[:3]  # Limit to 3 items
                            
                            for element in elements:
                                title = self._extract_title_from_element(element)
                                content = self._extract_content_from_element(element)
                                
                                if title and len(title) > 10:  # Valid title
                                    articles.append({
                                        'title': title,
                                        'content': content[:300],  # Limit content
                                        'url': url,
                                        'extraction_method': 'simplified_crawl4ai'
                                    })
                                    
                            if articles:
                                break  # Stop after finding content
                                
                        logger.info(f"Simplified extraction found {len(articles)} items from {url}")
                        return articles
                        
        except Exception as e:
            logger.error(f"Simplified extraction failed for {url}: {e}")
            
        return []
        
    def _extract_title_from_element(self, element) -> str:
        """Extract title from DOM element"""
        try:
            # Try common title patterns
            for tag in ['h1', 'h2', 'h3', '.title', '.headline']:
                title_elem = element.find(tag) if hasattr(element, 'find') else element.select_one(tag)
                if title_elem:
                    return title_elem.get_text().strip()
                    
            # Fallback to element text
            return element.get_text().strip()[:100]
        except:
            return ""
            
    def _extract_content_from_element(self, element) -> str:
        """Extract content from DOM element"""
        try:
            # Remove script and style elements
            for script in element(["script", "style"]):
                script.decompose()
                
            return element.get_text().strip()
        except:
            return ""

    async def _comprehensive_extraction(self, url: str, user_query: str, intent_result: Dict[str, Any], 
                                      options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        DISABLED: This method was causing recursive DuckDuckGo calls via ExtractionCoordinator.
        Now redirects to basic extraction to avoid the recursion loops.
        """
        logger.warning("🚫 _comprehensive_extraction() DISABLED - redirecting to basic extraction")
        logger.warning("🚫 This method was calling ExtractionCoordinator which caused recursive DuckDuckGo loops")
        
        # Redirect to basic extraction instead
        return await self._basic_comprehensive_extraction(url, user_query, intent_result, options)

    async def _get_extraction_coordinator(self):
        """Get or create the ExtractionCoordinator instance."""
        try:
            # Try to import and get the extraction coordinator
            from controllers.extraction_coordinator import get_extraction_coordinator
            return get_extraction_coordinator()
        except ImportError as e:
            logger.warning(f"ExtractionCoordinator not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting ExtractionCoordinator: {e}")
            return None

    async def _fallback_to_crawl4ai_strategy(self, url: str, user_query: str, intent_result: Dict[str, Any], 
                                           options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback to using UniversalCrawl4AIStrategy directly for intelligent extraction.
        """
        try:
            logger.info(f"Using UniversalCrawl4AIStrategy as fallback for {url}")
            
            # Try to import and use the UniversalCrawl4AIStrategy
            from strategies.universal_crawl4ai_strategy import UniversalCrawl4AIStrategy
            
            # Create strategy instance with enhanced context
            crawl4ai_strategy = UniversalCrawl4AIStrategy(
                context={
                    "query": user_query,
                    "intent_analysis": intent_result,
                    "progressive_collection": True,
                    "ai_pathfinding": True
                }
            )
            
            # Execute the strategy with intelligent options
            strategy_options = {
                **options,
                "user_prompt": user_query,
                "intent_analysis": intent_result,
                "max_pages": options.get("max_pages", 5),
                "progressive_collection": True,
                "early_relevance_termination": True
            }
            
            strategy_result = await crawl4ai_strategy.search(
                query=user_query,
                url=url,
                context=strategy_options
            )
            
            if strategy_result.get("success", False):
                # Format Crawl4AI results to our expected format
                formatted_data = self._format_crawl4ai_results(strategy_result, intent_result)
                
                return {
                    "success": True,
                    "url": url,
                    "extraction_method": "crawl4ai_strategy",
                    "data": formatted_data,
                    "metadata": {
                        "crawl4ai_strategy_used": True,
                        "extraction_timestamp": time.time(),
                        "user_query": user_query,
                        "intent_type": intent_result.get("intent_type", "unknown"),
                        "strategy_metadata": strategy_result.get("metadata", {}),
                        "pages_processed": strategy_result.get("pages_processed", 1),
                        "average_relevance": strategy_result.get("average_relevance", 0.5)
                    }
                }
            else:
                logger.warning(f"Crawl4AI strategy failed for {url}, using basic extraction")
                return await self._basic_comprehensive_extraction(url, user_query, intent_result, options)
                
        except ImportError as e:
            logger.warning(f"UniversalCrawl4AIStrategy not available: {e}")
            return await self._basic_comprehensive_extraction(url, user_query, intent_result, options)
        except Exception as e:
            logger.error(f"Error in Crawl4AI strategy fallback for {url}: {e}")
            return await self._basic_comprehensive_extraction(url, user_query, intent_result, options)

    def _format_coordinated_results(self, extracted_data: Dict[str, Any], intent_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format results from ExtractionCoordinator into our expected format.
        """
        try:
            formatted_results = []
            
            # Extract the actual data items from coordinated results
            if isinstance(extracted_data, dict):
                data_items = extracted_data.get("data", [])
                if not isinstance(data_items, list):
                    data_items = [data_items] if data_items else []
            else:
                data_items = extracted_data if isinstance(extracted_data, list) else []
            
            for item in data_items:
                if isinstance(item, dict):
                    # Enhance item with additional metadata
                    formatted_item = {
                        **item,
                        "_extraction_metadata": {
                            "extraction_method": "extraction_coordinator",
                            "extraction_timestamp": time.time(),
                            "intent_type": intent_result.get("intent_type", "unknown"),
                            "data_quality": self._assess_item_quality(item),
                            "relevance_score": self._calculate_item_relevance_score(item, intent_result)
                        }
                    }
                    formatted_results.append(formatted_item)
            
            logger.info(f"Formatted {len(formatted_results)} items from ExtractionCoordinator")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error formatting coordinated results: {e}")
            return []

    def _format_crawl4ai_results(self, strategy_result: Dict[str, Any], intent_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format results from UniversalCrawl4AIStrategy into our expected format.
        """
        try:
            formatted_results = []
            
            # Extract data from different possible result structures
            aggregated_content = strategy_result.get("aggregated_content", [])
            structured_data = strategy_result.get("structured_data", {})
            
            # Process aggregated content from multiple pages
            for page_content in aggregated_content:
                if isinstance(page_content, dict):
                    formatted_item = {
                        **page_content,
                        "_extraction_metadata": {
                            "extraction_method": "crawl4ai_strategy",
                            "extraction_timestamp": time.time(),
                            "intent_type": intent_result.get("intent_type", "unknown"),
                            "data_quality": self._assess_item_quality(page_content),
                            "relevance_score": page_content.get("relevance_score", 0.5)
                        }
                    }
                    formatted_results.append(formatted_item)
            
            # Process structured data if available
            if structured_data and isinstance(structured_data, dict):
                if structured_data.get("key_information"):
                    for info_item in structured_data["key_information"]:
                        formatted_item = {
                            "content": info_item,
                            "type": "key_information",
                            "_extraction_metadata": {
                                "extraction_method": "crawl4ai_ai_consolidation",
                                "extraction_timestamp": time.time(),
                                "intent_type": intent_result.get("intent_type", "unknown"),
                                "data_quality": "high",
                                "relevance_score": structured_data.get("confidence_score", 0.7)
                            }
                        }
                        formatted_results.append(formatted_item)
            
            logger.info(f"Formatted {len(formatted_results)} items from Crawl4AI strategy")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error formatting Crawl4AI results: {e}")
            return []

    def _assess_item_quality(self, item: Dict[str, Any]) -> str:
        """
        Assess the quality of an extracted item.
        """
        try:
            quality_score = 0
            
            # Check for title/name
            if any(key in item for key in ['title', 'name', 'headline']):
                quality_score += 1
            
            # Check for description/content
            if any(key in item for key in ['description', 'content', 'text']):
                content = str(item.get('description', '') or item.get('content', '') or item.get('text', ''))
                if len(content) > 50:
                    quality_score += 1
                if len(content) > 200:
                    quality_score += 1
            
            # Check for structured fields
            structured_fields = ['url', 'image', 'price', 'rating', 'date', 'author']
            structured_count = sum(1 for field in structured_fields if field in item)
            quality_score += min(structured_count, 2)
            
            # Determine quality level
            if quality_score >= 4:
                return "high"
            elif quality_score >= 2:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "unknown"

    def _calculate_item_relevance_score(self, item: Dict[str, Any], intent_result: Dict[str, Any]) -> float:
        """
        Calculate relevance score for an extracted item based on intent.
        """
        try:
            base_score = 0.5
            
            # Get query terms and intent type
            query = intent_result.get("query", "").lower()
            intent_type = intent_result.get("intent_type", "").lower()
            
            # Create searchable text from item
            searchable_text = ""
            for key, value in item.items():
                if isinstance(value, str) and not key.startswith('_'):
                    searchable_text += f" {value}"
            searchable_text = searchable_text.lower()
            
            # Query term matching
            if query:
                query_terms = query.split()
                matching_terms = sum(1 for term in query_terms if term in searchable_text)
                if query_terms:
                    base_score += (matching_terms / len(query_terms)) * 0.3
            
            # Intent-specific scoring
            intent_keywords = {
                "news": ["news", "article", "story", "breaking", "report"],
                "product": ["product", "price", "buy", "shop", "review"],
                "restaurant": ["restaurant", "menu", "food", "dining", "cuisine"],
                "real_estate": ["property", "house", "apartment", "rent", "buy"]
            }
            
            if intent_type in intent_keywords:
                matching_intent_keywords = sum(1 for keyword in intent_keywords[intent_type] 
                                             if keyword in searchable_text)
                base_score += (matching_intent_keywords / len(intent_keywords[intent_type])) * 0.2
            
            return min(1.0, base_score)
            
        except Exception:
            return 0.5

    async def _basic_comprehensive_extraction(self, url: str, user_query: str, intent_result: Dict[str, Any], 
                                      options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced comprehensive extraction using Crawl4AI and intelligent fallbacks.
        
        This method:
        1. Uses Crawl4AI for smart content extraction
        2. Falls back to direct HTML analysis
        3. Applies intelligent content filtering and ranking
        4. Returns actual content, not just metadata
        
        Args:
            url: URL to scrape
            user_query: Original user query for context
            intent_result: Parsed intent information
            options: Additional extraction options
            
        Returns:
            Dictionary containing extracted data and metadata
        """
        options = options or {}
        logger.info(f"Starting comprehensive extraction for {url}")
        
        extraction_result = {
            "success": False,
            "url": url,
            "extraction_method": "comprehensive",
            "data": [],
            "metadata": {
                "extraction_stages": [],
                "extraction_timestamp": time.time(),
                "user_query": user_query,
                "intent_type": intent_result.get("intent_type", "unknown")
            }
        }
        
        try:
            # Stage 1: Crawl4AI-based extraction (primary method)
            logger.debug(f"Stage 1: Crawl4AI extraction for {url}")
            crawl4ai_data = await self._crawl4ai_intelligent_extraction(url, user_query, intent_result, options)
            if crawl4ai_data and len(crawl4ai_data) > 0:
                extraction_result["data"].extend(crawl4ai_data)
                extraction_result["metadata"]["extraction_stages"].append("crawl4ai_success")
                extraction_result["extraction_method"] = "crawl4ai_intelligent"
                logger.info(f"Crawl4AI extraction successful for {url}, extracted {len(crawl4ai_data)} items")
            
            # Stage 2: Fallback HTML extraction if Crawl4AI didn't work well
            if len(extraction_result["data"]) < 2:  # If Crawl4AI didn't extract enough meaningful content
                logger.debug(f"Stage 2: Fallback HTML extraction for {url}")
                html_content = await self._fetch_html_robustly(url, options)
                if html_content:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Try intelligent content extraction
                    fallback_data = await self._intelligent_fallback_extraction(soup, html_content, user_query, intent_result)
                    if fallback_data and len(fallback_data) > 0:
                        extraction_result["data"].extend(fallback_data)
                        extraction_result["metadata"]["extraction_stages"].append("intelligent_fallback_success")
                        logger.info(f"Intelligent fallback extraction successful for {url}, extracted {len(fallback_data)} items")
            
            # Stage 3: Final component-based extraction if still insufficient
            if len(extraction_result["data"]) < 1:
                logger.debug(f"Stage 3: Component-based extraction for {url}")
                if 'html_content' not in locals():
                    html_content = await self._fetch_html_robustly(url, options)
                if html_content:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    component_data = await self._execute_component_extraction(soup, user_query, intent_result)
                    if component_data and len(component_data) > 0:
                        extraction_result["data"].extend(component_data)
                        extraction_result["metadata"]["extraction_stages"].append("component_extraction_success")
            
            # Stage 4: Data quality assessment and filtering
            if extraction_result["data"]:
                logger.debug(f"Stage 4: Data quality assessment for {url}")
                # Filter and rank by relevance to user query
                filtered_data = await self._filter_and_rank_extracted_data(extraction_result["data"], user_query, intent_result)
                extraction_result["data"] = filtered_data
                extraction_result["success"] = True
                extraction_result["metadata"]["extraction_stages"].append("quality_filtering_complete")
                extraction_result["metadata"]["final_items_count"] = len(extraction_result["data"])
                
                logger.info(f"Comprehensive extraction completed successfully for {url} - {len(extraction_result['data'])} final items")
            else:
                extraction_result["metadata"]["extraction_stages"].append("no_data_extracted")
                logger.warning(f"No data extracted for {url} despite trying all methods")
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive extraction for {url}: {e}", exc_info=True)
            extraction_result["error"] = f"Comprehensive extraction error: {str(e)}"
            return extraction_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive extraction for {url}: {e}", exc_info=True)
            extraction_result["error"] = f"Comprehensive extraction error: {str(e)}"
            return extraction_result

    async def _robust_html_fetch(self, url: str, options: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Robustly fetch HTML content using multiple methods with fallbacks.
        
        Args:
            url: URL to fetch
            options: Fetch options
            
        Returns:
            Tuple of (html_content, fetch_metadata)
        """
        fetch_metadata = {
            "fetch_method": None,
            "response_status": None,
            "content_length": 0,
            "fetch_time": time.time()
        }
        
        try:
            # Try using the HTML service first
            if hasattr(self, 'html_service') and self.html_service:
                try:
                    logger.debug(f"Attempting HTML service fetch for {url}")
                    html_response = await self.html_service.get_html(url, options)
                    if html_response and html_response.get("html"):
                        fetch_metadata["fetch_method"] = "html_service"
                        fetch_metadata["response_status"] = html_response.get("status_code", 200)
                        fetch_metadata["content_length"] = len(html_response["html"])
                        return html_response["html"], fetch_metadata
                except Exception as e:
                    logger.debug(f"HTML service fetch failed for {url}: {e}")
            
            # Fallback to direct HTTP request
            logger.debug(f"Attempting direct HTTP fetch for {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        fetch_metadata["fetch_method"] = "direct_http"
                        fetch_metadata["response_status"] = response.status
                        fetch_metadata["content_length"] = len(html_content)
                        return html_content, fetch_metadata
                    else:
                        logger.warning(f"HTTP request failed with status {response.status} for {url}")
                        
        except Exception as e:
            logger.error(f"All fetch methods failed for {url}: {e}")
        
        return "", fetch_metadata

    async def _analyze_content_structure(self, soup: BeautifulSoup, html_content: str, 
                                       intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the structure and characteristics of the content.
        
        Args:
            soup: BeautifulSoup parsed HTML
            html_content: Raw HTML content
            intent_result: Parsed intent information
            
        Returns:
            Structure analysis results
        """
        try:
            # Use structural analyzer if available
            if hasattr(self, 'structural_analyzer') and self.structural_analyzer:
                logger.debug("Using structural analyzer for content analysis")
                try:
                    if hasattr(self.structural_analyzer, 'analyze_structure'):
                        analysis = self.structural_analyzer.analyze_structure(soup)
                        if analysis and isinstance(analysis, dict):
                            return analysis
                    elif hasattr(self.structural_analyzer, 'analyze'):
                        analysis = self.structural_analyzer.analyze(soup)
                        if analysis and isinstance(analysis, dict):
                            return analysis
                except Exception as e:
                    logger.debug(f"Structural analyzer failed, using fallback: {e}")
            
            # Basic structure analysis fallback
            analysis = {
                "content_type": "html",
                "has_structured_data": False,
                "has_forms": False,
                "has_pagination": False,
                "semantic_structure": {},
                "content_patterns": [],
                "page_type": "unknown",
                "content_density": "medium"
            }
            
            # Check for structured data
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            microdata_items = soup.find_all(attrs={'itemtype': True})
            og_meta = soup.find_all('meta', property=re.compile(r'^og:'))
            
            if json_ld_scripts or microdata_items or og_meta:
                analysis["has_structured_data"] = True
                analysis["structured_data_types"] = []
                if json_ld_scripts:
                    analysis["structured_data_types"].append("json_ld")
                if microdata_items:
                    analysis["structured_data_types"].append("microdata")
                if og_meta:
                    analysis["structured_data_types"].append("open_graph")
            
            # Check for forms
            forms = soup.find_all('form')
            if forms:
                analysis["has_forms"] = True
                analysis["form_count"] = len(forms)
            
            # Check for pagination indicators
            pagination_indicators = soup.find_all(text=re.compile(r'next|previous|page \d+|more|load more', re.IGNORECASE))
            pagination_links = soup.find_all('a', href=re.compile(r'page=\d+|p=\d+', re.IGNORECASE))
            if pagination_indicators or pagination_links:
                analysis["has_pagination"] = True
            
            # Identify semantic structure
            semantic_tags = ['article', 'section', 'main', 'header', 'footer', 'nav', 'aside', 'figure']
            for tag in semantic_tags:
                elements = soup.find_all(tag)
                if elements:
                    analysis["semantic_structure"][tag] = len(elements)
            
            # Analyze content density
            text_content = soup.get_text()
            html_length = len(html_content)
            text_length = len(text_content.strip())
            
            if html_length > 0:
                density_ratio = text_length / html_length
                if density_ratio > 0.3:
                    analysis["content_density"] = "high"
                elif density_ratio < 0.1:
                    analysis["content_density"] = "low"
                else:
                    analysis["content_density"] = "medium"
            
            # Identify page type based on content and intent
            intent_type = intent_result.get("intent_type", "").lower()
            
            # Check for common page types
            if soup.find_all(['article']) and len(soup.find_all(['h1', 'h2'])) > 0:
                analysis["page_type"] = "article"
            elif soup.find_all(['form']) and soup.find_all(['input']):
                analysis["page_type"] = "form"
            elif len(soup.find_all(['li'])) > 10 or len(soup.find_all(class_=re.compile(r'item|result|product', re.I))) > 5:
                analysis["page_type"] = "listing"
            elif soup.find('title') and 'search' in soup.find('title').get_text().lower():
                analysis["page_type"] = "search_results"
            elif soup.find_all(attrs={'itemtype': re.compile(r'Product', re.I)}):
                analysis["page_type"] = "product"
            
            # Identify content patterns based on intent and structure
            if "product" in intent_type or analysis["page_type"] == "product":
                analysis["content_patterns"].append("product_listing")
            elif "news" in intent_type or "article" in intent_type or analysis["page_type"] == "article":
                analysis["content_patterns"].append("article_content")
            elif "search" in intent_type or analysis["page_type"] == "search_results":
                analysis["content_patterns"].append("search_results")
            elif analysis["page_type"] == "listing":
                analysis["content_patterns"].append("listing_content")
            
            # Add metadata about analysis quality
            analysis["analysis_confidence"] = "high" if analysis["has_structured_data"] else "medium"
            analysis["extraction_complexity"] = "low" if analysis["has_structured_data"] else "medium"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing content structure: {e}")
            return {
                "error": str(e),
                "content_type": "html",
                "page_type": "unknown",
                "analysis_confidence": "low"
            }

    async def _execute_extraction_pipeline(self, html_content: str, soup: BeautifulSoup,
                                         structure_analysis: Dict[str, Any], intent_result: Dict[str, Any],
                                         options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute appropriate extraction pipeline based on content analysis.
        
        Args:
            html_content: Raw HTML content
            soup: Parsed BeautifulSoup object
            structure_analysis: Content structure analysis
            intent_result: Parsed intent information
            options: Extraction options
            
        Returns:
            Pipeline extraction results
        """
        try:
            # Check if pipeline factory is available
            if not hasattr(self, 'pipeline_factory') or not self.pipeline_factory:
                logger.debug("Pipeline factory not available, skipping pipeline extraction")
                return {"success": False, "error": "Pipeline factory not available"}
            
            # Determine appropriate pipeline based on structure analysis and intent
            pipeline_type = self._select_extraction_pipeline(structure_analysis, intent_result, options)
            
            if not pipeline_type:
                logger.debug("No suitable pipeline found for content")
                return {"success": False, "error": "No suitable pipeline found"}
            
            logger.debug(f"Selected extraction pipeline: {pipeline_type}")
            
            # Create pipeline context
            context = {
                "html_content": html_content,
                "soup": soup,
                "structure_analysis": structure_analysis,
                "intent_result": intent_result,
                "options": options
            }
            
            # Execute pipeline
            try:
                pipeline = self.pipeline_factory.create_pipeline(pipeline_type, context)
                if pipeline:
                    result = await pipeline.execute(context)
                    if result and result.get("data"):
                        return {
                            "success": True,
                            "data": result["data"],
                            "pipeline_type": pipeline_type,
                            "metadata": result.get("metadata", {})
                        }
            except Exception as e:
                logger.debug(f"Pipeline execution failed: {e}")
                
            return {"success": False, "error": f"Pipeline {pipeline_type} execution failed"}
            
        except Exception as e:
            logger.debug(f"Error in pipeline extraction: {e}")
            return {"success": False, "error": f"Pipeline extraction error: {str(e)}"}

    def _select_extraction_pipeline(self, structure_analysis: Dict[str, Any], 
                                  intent_result: Dict[str, Any], options: Dict[str, Any]) -> Optional[str]:
        """
        Select the most appropriate extraction pipeline based on analysis.
        
        Args:
            structure_analysis: Content structure analysis
            intent_result: Parsed intent information
            options: Extraction options
            
        Returns:
            Pipeline type name or None
        """
        # Check if structured data pipeline is appropriate
        if structure_analysis.get("has_structured_data"):
            return "structured_data_pipeline"
        
        # Check intent-based pipeline selection
        intent_type = intent_result.get("intent_type", "").lower()
        if "product" in intent_type and "product_listing" in structure_analysis.get("content_patterns", []):
            return "product_extraction_pipeline"
        elif "news" in intent_type or "article" in intent_type:
            return "article_extraction_pipeline"
        elif "search" in intent_type:
            return "search_results_pipeline"
        
        # Default to general content pipeline
        return "general_content_pipeline"

    async def _execute_strategy_extraction(self, url: str, html_content: str, soup: BeautifulSoup,
                                         structure_analysis: Dict[str, Any], intent_result: Dict[str, Any],
                                         options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute strategy-based extraction using registered strategies.
        
        Args:
            url: URL being scraped
            html_content: Raw HTML content
            soup: Parsed BeautifulSoup object
            structure_analysis: Content structure analysis
            intent_result: Parsed intent information
            options: Extraction options
            
        Returns:
            Strategy extraction results
        """
        try:
            # Check if strategy factory is available
            if not hasattr(self, 'strategy_factory') or not self.strategy_factory:
                logger.debug("Strategy factory not available, skipping strategy extraction")
                return {"success": False, "error": "Strategy factory not available"}
            
            # Select appropriate strategy
            strategy_type = self._select_extraction_strategy(structure_analysis, intent_result, options)
            
            if not strategy_type:
                logger.debug("No suitable strategy found for content")
                return {"success": False, "error": "No suitable strategy found"}
            
            logger.debug(f"Selected extraction strategy: {strategy_type}")
            
            # Try to get and execute the strategy
            try:
                strategy = self.strategy_factory.get_strategy(strategy_type)
                if strategy and hasattr(strategy, 'extract'):
                    context = {
                        "url": url,
                        "html_content": html_content,
                        "soup": soup,
                        "structure_analysis": structure_analysis,
                        "intent_result": intent_result,
                        "options": options
                    }
                    
                    result = await strategy.extract(context)
                    if result and result.get("data"):
                        return {
                            "success": True,
                            "data": result["data"],
                            "strategy_type": strategy_type,
                            "metadata": result.get("metadata", {})
                        }
            except Exception as e:
                logger.debug(f"Strategy execution failed: {e}")
                
            return {"success": False, "error": f"Strategy {strategy_type} execution failed"}
            
        except Exception as e:
            logger.debug(f"Error in strategy extraction: {e}")
            return {"success": False, "error": f"Strategy extraction error: {str(e)}"}

    def _select_extraction_strategy(self, structure_analysis: Dict[str, Any], 
                                  intent_result: Dict[str, Any], options: Dict[str, Any]) -> Optional[str]:
        """
        Select the most appropriate extraction strategy.
        
        Args:
            structure_analysis: Content structure analysis
            intent_result: Parsed intent information
            options: Extraction options
            
        Returns:
            Strategy name or None
        """
        # Check if AI-guided strategy is appropriate and available
        if hasattr(self, 'ai_service') and self.ai_service and intent_result.get("complexity", "low") == "high":
            return "ai_guided"
        
        # Check for form-based strategy
        if structure_analysis.get("has_forms"):
            return "form_search_engine"
        
        # Check for DOM-based strategy for complex structures
        if structure_analysis.get("semantic_structure"):
            return "dom_strategy"
        
        # Default to multi-strategy for comprehensive coverage
        return "multi_strategy"

    async def _execute_component_extraction(self, soup: BeautifulSoup, html_content: str,
                                          structure_analysis: Dict[str, Any], intent_result: Dict[str, Any],
                                          options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute component-based extraction using built-in extractors.
        
        Args:
            soup: Parsed BeautifulSoup object
            html_content: Raw HTML content
            structure_analysis: Content structure analysis
            intent_result: Parsed intent information
            options: Extraction options
            
        Returns:
            Component extraction results
        """
        try:
            extraction_data = []
            component_type = "mixed"
            
            # Try structured content extraction first
            structured_data = self._extract_structured_content(soup)
            if structured_data and len(structured_data) > 0:
                extraction_data.extend(structured_data)
                component_type = "structured"
                logger.debug(f"Found {len(structured_data)} structured items")
            
            # Try semantic content extraction
            if not extraction_data:
                semantic_data = self._extract_semantic_content(soup)
                if semantic_data and len(semantic_data) > 0:
                    extraction_data.extend(semantic_data)
                    component_type = "semantic"
                    logger.debug(f"Found {len(semantic_data)} semantic items")
            
            # Try pattern-based extraction based on intent
            if not extraction_data:
                intent_type = intent_result.get("intent_type", "").lower()
                if "product" in intent_type:
                    pattern_data = self._extract_product_patterns(soup)
                    if pattern_data and len(pattern_data) > 0:
                        extraction_data.extend(pattern_data)
                        component_type = "product_patterns"
                        logger.debug(f"Found {len(pattern_data)} product items")
                elif "list" in intent_type or "search" in intent_type:
                    pattern_data = self._extract_listing_patterns(soup)
                    if pattern_data and len(pattern_data) > 0:
                        extraction_data.extend(pattern_data)
                        component_type = "listing_patterns"
                        logger.debug(f"Found {len(pattern_data)} listing items")
            
            # Final fallback to generic content extraction
            if not extraction_data:
                generic_data = self._extract_generic_content(soup)
                if generic_data and len(generic_data) > 0:
                    extraction_data.extend(generic_data)
                    component_type = "generic"
                    logger.debug(f"Found {len(generic_data)} generic items")
            
            if extraction_data:
                return {
                    "success": True,
                    "data": extraction_data,
                    "component_type": component_type,
                    "metadata": {
                        "extraction_method": "component_based",
                        "component_types_used": [component_type],
                        "item_count": len(extraction_data)
                    }
                }
            else:
                return {"success": False, "error": "No content extracted by any component method"}
                
        except Exception as e:
            logger.error(f"Error in component extraction: {e}")
            return {"success": False, "error": f"Component extraction error: {str(e)}"}

    async def _post_process_extraction_results(self, data: List[Dict[str, Any]], 
                                             intent_result: Dict[str, Any], structure_analysis: Dict[str, Any],
                                             options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Post-process extraction results for quality and relevance.
        
        Args:
            data: Raw extracted data
            intent_result: Parsed intent information
            structure_analysis: Content structure analysis
            options: Processing options
            
        Returns:
            Processed and enhanced data
        """
        try:
            if not data:
                return data
            
            processed_data = []
            
            for item in data:
                # Skip empty or invalid items
                if not item or not isinstance(item, dict):
                    continue
                
                # Basic data cleaning
                cleaned_item = self._clean_extracted_item(item)
                
                # Intent-based relevance filtering
                if self._is_relevant_to_intent(cleaned_item, intent_result):
                    # Add metadata
                    cleaned_item["_metadata"] = {
                        "extraction_timestamp": time.time(),
                        "relevance_score": self._calculate_relevance_score(cleaned_item, intent_result),
                        "data_quality": self._assess_data_quality(cleaned_item)
                    }
                    processed_data.append(cleaned_item)
            
            # Sort by relevance score if available
            if processed_data and all("_metadata" in item for item in processed_data):
                processed_data.sort(key=lambda x: x["_metadata"].get("relevance_score", 0), reverse=True)
            
            # Limit results to avoid overwhelming response
            max_results = options.get("max_results", 50)
            return processed_data[:max_results]
            
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            return data  # Return original data on error

    def _clean_extracted_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize an extracted item."""
        cleaned = {}
        for key, value in item.items():
            if key.startswith("_"):  # Skip private metadata keys
                continue
            if isinstance(value, str):
                # Clean whitespace and HTML entities
                cleaned_value = re.sub(r'\s+', ' ', value).strip()
                if cleaned_value:
                    cleaned[key] = cleaned_value
            elif value is not None:
                cleaned[key] = value
        return cleaned

    def _is_relevant_to_intent(self, item: Dict[str, Any], intent_result: Dict[str, Any]) -> bool:
        """Check if an item is relevant to the user's intent."""
        try:
            # If no specific intent, accept all items
            if not intent_result.get("keywords") and not intent_result.get("intent_type"):
                return True
            
            # Check for keyword matches
            keywords = intent_result.get("keywords", [])
            if keywords:
                item_text = " ".join(str(v).lower() for v in item.values() if isinstance(v, str))
                for keyword in keywords:
                    if keyword.lower() in item_text:
                        return True
            
            return True  # Default to include item
        except Exception:
            return True

    def _calculate_relevance_score(self, item: Dict[str, Any], intent_result: Dict[str, Any]) -> float:
        """Calculate a relevance score for an item."""
        try:
            score = 0.5  # Base score
            
            keywords = intent_result.get("keywords", [])
            if keywords:
                item_text = " ".join(str(v).lower() for v in item.values() if isinstance(v, str))
                matches = sum(1 for keyword in keywords if keyword.lower() in item_text)
                score += (matches / len(keywords)) * 0.5
            
            # Bonus for having title or name
            if any(key in item for key in ["title", "name", "heading"]):
                score += 0.1
            
            # Bonus for having URL or link
            if any(key in item for key in ["url", "link", "href"]):
                score += 0.1
            
            return min(score, 1.0)
        except Exception:
            return 0.5

    def _assess_data_quality(self, item: Dict[str, Any]) -> str:
        """Assess the quality of extracted data."""
        try:
            if len(item) >= 3 and all(len(str(v)) > 10 for v in item.values() if isinstance(v, str)):
                return "high"
            elif len(item) >= 2:
                return "medium"
            else:
                return "low"
        except Exception:
            return "unknown"

    # ===========================
    # FALLBACK EXTRACTION SYSTEM
    # ===========================
    
    async def _apply_fallback_extraction(self, url: str, html_content: str, 
                                       options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply comprehensive fallback extraction when primary methods fail.
        
        This method coordinates multiple extraction approaches in order of specificity:
        1. Structured content extraction (JSON-LD, microdata)
        2. Semantic content extraction (semantic HTML patterns)
        3. Generic content extraction (basic HTML parsing)
        4. Final content fallback (minimal extraction)
        
        Args:
            url: URL being scraped
            html_content: Raw HTML content
            options: Extraction options
            
        Returns:
            Extracted content with metadata
        """
        logger.info(f"Applying fallback extraction for {url}")
        
        try:
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Initialize results container
            fallback_results = {
                "success": False,
                "url": url,
                "extraction_method": "fallback",
                "data": [],
                "metadata": {
                    "fallback_steps": [],
                    "extraction_timestamp": time.time(),
                    "fallback_reason": options.get("fallback_reason", "primary_extraction_failed")
                }
            }
            
            # Step 1: Try structured content extraction
            try:
                structured_data = self._extract_structured_content(soup)
                if structured_data and len(structured_data) > 0:
                    fallback_results["data"] = structured_data
                    fallback_results["success"] = True
                    fallback_results["extraction_method"] = "structured_fallback"
                    fallback_results["metadata"]["fallback_steps"].append("structured_content_success")
                    logger.info(f"Structured fallback extraction successful for {url}")
                    return fallback_results
                else:
                    fallback_results["metadata"]["fallback_steps"].append("structured_content_failed")
            except Exception as e:
                logger.warning(f"Structured content extraction failed: {e}")
                fallback_results["metadata"]["fallback_steps"].append(f"structured_content_error: {str(e)}")
            
            # Step 2: Try semantic content extraction
            try:
                semantic_data = self._extract_semantic_content(soup)
                if semantic_data and len(semantic_data) > 0:
                    fallback_results["data"] = semantic_data
                    fallback_results["success"] = True
                    fallback_results["extraction_method"] = "semantic_fallback"
                    fallback_results["metadata"]["fallback_steps"].append("semantic_content_success")
                    logger.info(f"Semantic fallback extraction successful for {url}")
                    return fallback_results
                else:
                    fallback_results["metadata"]["fallback_steps"].append("semantic_content_failed")
            except Exception as e:
                logger.warning(f"Semantic content extraction failed: {e}")
                fallback_results["metadata"]["fallback_steps"].append(f"semantic_content_error: {str(e)}")
            
            # Step 3: Try generic content extraction
            try:
                generic_data = self._extract_generic_content(soup)
                if generic_data and len(generic_data) > 0:
                    fallback_results["data"] = generic_data
                    fallback_results["success"] = True
                    fallback_results["extraction_method"] = "generic_fallback"
                    fallback_results["metadata"]["fallback_steps"].append("generic_content_success")
                    logger.info(f"Generic fallback extraction successful for {url}")
                    return fallback_results
                else:
                    fallback_results["metadata"]["fallback_steps"].append("generic_content_failed")
            except Exception as e:
                logger.warning(f"Generic content extraction failed: {e}")
                fallback_results["metadata"]["fallback_steps"].append(f"generic_content_error: {str(e)}")
            
            # Step 4: Final content fallback
            try:
                final_data = self._extract_content_fallback(soup)
                if final_data:
                    fallback_results["data"] = [final_data] if isinstance(final_data, dict) else final_data
                    fallback_results["success"] = True
                    fallback_results["extraction_method"] = "final_fallback"
                    fallback_results["metadata"]["fallback_steps"].append("final_fallback_success")
                    logger.info(f"Final fallback extraction successful for {url}")
                    return fallback_results
                else:
                    fallback_results["metadata"]["fallback_steps"].append("final_fallback_failed")
            except Exception as e:
                logger.warning(f"Final fallback extraction failed: {e}")
                fallback_results["metadata"]["fallback_steps"].append(f"final_fallback_error: {str(e)}")
            
            # If all fallback methods failed
            fallback_results["error"] = "All fallback extraction methods failed"
            fallback_results["metadata"]["fallback_steps"].append("all_fallbacks_exhausted")
            logger.error(f"All fallback extraction methods failed for {url}")
            
            return fallback_results
            
        except Exception as e:
            logger.error(f"Error in fallback extraction system for {url}: {e}")
            return {
                "success": False,
                "error": f"Fallback extraction system error: {str(e)}",
                "url": url,
                "data": [],
                "metadata": {
                    "fallback_steps": ["system_error"],
                    "extraction_timestamp": time.time(),
                    "error_details": str(e)
                }
            }
    
    def _extract_structured_content(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract structured data from HTML using JSON-LD, microdata, and schema.org.
        
        Args:
            soup: BeautifulSoup parsed HTML
            
        Returns:
            List of structured data items
        """
        structured_items = []
        
        try:
            # Extract JSON-LD data
            json_ld_scripts = soup.find_all('script', {'type': 'application/ld+json'})
            for script in json_ld_scripts:
                try:
                    if script.string:
                        data = json.loads(script.string)
                        if isinstance(data, list):
                            structured_items.extend(data)
                        elif isinstance(data, dict):
                            structured_items.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON-LD: {e}")
                    continue
            
            # Extract microdata
            microdata_items = soup.find_all(attrs={"itemscope": True})
            for item in microdata_items:
                microdata_obj = self._parse_microdata_item(item)
                if microdata_obj:
                    structured_items.append(microdata_obj)
            
            # Extract Open Graph data
            og_data = self._extract_open_graph_data(soup)
            if og_data:
                structured_items.append({
                    "@type": "OpenGraph",
                    **og_data
                })
            
            # Extract Twitter Card data
            twitter_data = self._extract_twitter_card_data(soup)
            if twitter_data:
                structured_items.append({
                    "@type": "TwitterCard", 
                    **twitter_data
                })
            
            logger.debug(f"Extracted {len(structured_items)} structured data items")
            return structured_items
            
        except Exception as e:
            logger.error(f"Error extracting structured content: {e}")
            return []
    
    def _extract_semantic_content(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract content using semantic HTML patterns and common content structures.
        
        Args:
            soup: BeautifulSoup parsed HTML
            
        Returns:
            List of semantically extracted content items
        """
        semantic_items = []
        
        try:
            # Extract articles using semantic HTML5 tags
            articles = soup.find_all('article')
            for article in articles:
                article_data = self._parse_article_element(article)
                if article_data:
                    semantic_items.append(article_data)
            
            # Extract content from main elements
            main_elements = soup.find_all('main')
            for main in main_elements:
                main_data = self._parse_main_element(main)
                if main_data:
                    semantic_items.append(main_data)
            
            # Extract product information
            product_data = self._extract_product_patterns(soup)
            semantic_items.extend(product_data)
            
            # Extract listing patterns
            listing_data = self._extract_listing_patterns(soup)
            semantic_items.extend(listing_data)
            
            # Extract navigation and content sections
            section_data = self._extract_section_patterns(soup)
            semantic_items.extend(section_data)
            
            logger.debug(f"Extracted {len(semantic_items)} semantic content items")
            return semantic_items
            
        except Exception as e:
            logger.error(f"Error extracting semantic content: {e}")
            return []
    
    def _extract_generic_content(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract content using generic HTML parsing patterns.
        
        Args:
            soup: BeautifulSoup parsed HTML
            
        Returns:
            List of generically extracted content items
        """
        generic_items = []
        
        try:
            # Extract headings and associated content
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            for heading in headings:
                heading_data = self._parse_heading_section(heading)
                if heading_data:
                    generic_items.append(heading_data)
            
            # Extract table data
            tables = soup.find_all('table')
            for table in tables:
                table_data = self._parse_table_element(table)
                if table_data:
                    generic_items.append(table_data)
            
            # Extract list data
            lists = soup.find_all(['ul', 'ol'])
            for list_elem in lists:
                list_data = self._parse_list_element(list_elem)
                if list_data:
                    generic_items.append(list_data)
            
            # Extract form data
            forms = soup.find_all('form')
            for form in forms:
                form_data = self._parse_form_element(form)
                if form_data:
                    generic_items.append(form_data)
            
            # Extract div containers with content
            content_divs = soup.find_all('div', class_=re.compile(r'content|main|body|article|post', re.I))
            for div in content_divs:
                div_data = self._parse_content_div(div)
                if div_data:
                    generic_items.append(div_data)
            
            logger.debug(f"Extracted {len(generic_items)} generic content items")
            return generic_items
            
        except Exception as e:
            logger.error(f"Error extracting generic content: {e}")
            return []
    
    def _extract_content_fallback(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Progressive fallback content extraction with multiple strategies.
        
        Args:
            soup: BeautifulSoup parsed HTML
            
        Returns:
            Comprehensive extraction result using progressive fallback
        """
        try:
            logger.debug("Starting progressive fallback extraction")
            
            # Level 1: Try structured data extraction (simplified)
            structured_data = {}
            try:
                json_scripts = soup.find_all('script', type='application/ld+json')
                for script in json_scripts:
                    try:
                        if script.string:
                            data = json.loads(script.string)
                            if isinstance(data, dict):
                                structured_data.update(data)
                            elif isinstance(data, list) and data:
                                structured_data.update(data[0])
                    except json.JSONDecodeError:
                        continue
            except Exception:
                pass
            
            if structured_data:
                logger.debug("Found structured data, using as primary source")
                return {
                    "content_type": "structured_data",
                    "data": structured_data,
                    "metadata": {"extraction_method": "structured_data_fallback"}
                }
            
            # Level 2: Try semantic HTML extraction (simplified)
            semantic_content = []
            try:
                semantic_tags = ['article', 'section', 'main']
                for tag_name in semantic_tags:
                    elements = soup.find_all(tag_name)
                    for element in elements:
                        heading = element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                        heading_text = heading.get_text(strip=True) if heading else ""
                        
                        # Clean element and get text
                        for script in element(['script', 'style']):
                            script.extract()
                        content_text = element.get_text(separator=' ', strip=True)
                        
                        if content_text and len(content_text) > 50:
                            semantic_content.append({
                                'type': tag_name,
                                'heading': heading_text,
                                'content': content_text[:500],
                                'score': len(content_text)
                            })
            except Exception:
                pass
            
            if semantic_content:
                logger.debug(f"Found {len(semantic_content)} semantic content sections")
                best_content = max(semantic_content, key=lambda x: x.get('score', 0))
                return {
                    "content_type": "semantic",
                    "title": best_content.get('heading', ''),
                    "content": best_content.get('content', ''),
                    "all_sections": semantic_content,
                    "metadata": {"extraction_method": "semantic_fallback"}
                }
            
            # Level 3: Try generic content extraction (simplified)
            generic_content = []
            try:
                headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                for heading in headings:
                    heading_text = heading.get_text(strip=True)
                    if not heading_text:
                        continue
                    
                    content_parts = []
                    next_elem = heading.find_next_sibling()
                    
                    while next_elem and len(content_parts) < 3:
                        if next_elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            break
                        
                        if next_elem.name == 'p':
                            text = next_elem.get_text(strip=True)
                            if text and len(text) > 20:
                                content_parts.append(text)
                        
                        next_elem = next_elem.find_next_sibling()
                    
                    if content_parts:
                        generic_content.append({
                            'type': 'heading_section',
                            'heading': heading_text,
                            'content': ' '.join(content_parts)[:400],
                            'level': heading.name,
                            'score': len(content_parts) * 10
                        })
            except Exception:
                pass
            
            if generic_content:
                logger.debug(f"Found {len(generic_content)} generic content sections")
                best_content = max(generic_content, key=lambda x: x.get('score', 0))
                return {
                    "content_type": "generic",
                    "title": best_content.get('heading', ''),
                    "content": best_content.get('content', ''),
                    "all_sections": generic_content,
                    "metadata": {"extraction_method": "generic_fallback"}
                }
            
            # Level 4: Final basic extraction
            logger.debug("Using final basic extraction")
            
            # Extract basic page information
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = re.sub(r'\s+', ' ', title_tag.get_text(strip=True))
            
            # Extract meta description
            description = ""
            meta_desc = soup.find('meta', attrs={'name': 'description'}) or \
                       soup.find('meta', attrs={'property': 'og:description'})
            if meta_desc:
                description = re.sub(r'\s+', ' ', meta_desc.get('content', ''))
            
            # Extract main text content
            for script in soup(["script", "style", "noscript"]):
                script.extract()
            
            text_content = re.sub(r'\s+', ' ', soup.get_text(separator=' ', strip=True))
            
            # Extract links and images
            links = []
            for link in soup.find_all('a', href=True):
                link_text = link.get_text(strip=True)
                if link_text and len(link_text) > 0:
                    links.append({"text": link_text, "href": link['href']})
            
            images = []
            for img in soup.find_all('img', src=True):
                alt_text = img.get('alt', '')
                images.append({"src": img['src'], "alt": alt_text})
            
            fallback_result = {
                "content_type": "basic_fallback",
                "title": title,
                "description": description,
                "text_content": text_content[:1000] + "..." if len(text_content) > 1000 else text_content,
                "links": links[:10],
                "images": images[:5],
                "metadata": {
                    "extraction_method": "final_fallback",
                    "content_length": len(text_content),
                    "links_count": len(links),
                    "images_count": len(images),
                    "has_title": bool(title),
                    "has_description": bool(description)
                }
            }
            
            logger.debug("Progressive fallback extraction completed")
            return fallback_result
            
        except Exception as e:
            logger.error(f"Error in fallback extraction: {e}")
            return {
                "content_type": "error_fallback",
                "title": "",
                "description": "",
                "text_content": "",
                "error": str(e),
                "metadata": {"extraction_method": "error_fallback"}
            }

    # Helper methods for fallback extraction
    def _extract_open_graph_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract Open Graph metadata from HTML."""
        og_data = {}
        try:
            for meta in soup.find_all('meta', property=re.compile(r'^og:')):
                property_name = meta.get('property', '').replace('og:', '')
                content = meta.get('content', '')
                if property_name and content:
                    og_data[property_name] = content
            return og_data
        except Exception as e:
            logger.error(f"Error extracting Open Graph data: {e}")
            return {}

    def _extract_twitter_card_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract Twitter Card metadata from HTML."""
        twitter_data = {}
        try:
            for meta in soup.find_all('meta', attrs={'name': re.compile(r'^twitter:')}):
                name = meta.get('name', '').replace('twitter:', '')
                content = meta.get('content', '')
                if name and content:
                    twitter_data[name] = content
            return twitter_data
        except Exception as e:
            logger.error(f"Error extracting Twitter Card data: {e}")
            return {}

    def _parse_microdata_item(self, item) -> Optional[Dict[str, Any]]:
        """Parse a microdata item from HTML."""
        try:
            microdata_obj = {}
            
            # Get item type
            item_type = item.get('itemtype')
            if item_type:
                microdata_obj['@type'] = item_type
            
            # Get item properties
            properties = {}
            for prop in item.find_all(attrs={'itemprop': True}):
                prop_name = prop.get('itemprop')
                prop_value = ""
                
                if prop.name in ['meta']:
                    prop_value = prop.get('content', '')
                elif prop.name in ['img']:
                    prop_value = prop.get('src', '')
                elif prop.name in ['a']:
                    prop_value = prop.get('href', '')
                else:
                    prop_value = prop.get_text(strip=True)
                
                if prop_name and prop_value:
                    properties[prop_name] = prop_value
            
            if properties:
                microdata_obj.update(properties)
                return microdata_obj
            
        except Exception as e:
            logger.error(f"Error parsing microdata item: {e}")
        
        return None

    def _parse_article_element(self, article) -> Optional[Dict[str, Any]]:
        """Parse an HTML5 article element."""
        try:
            article_data = {
                "content_type": "article",
                "title": "",
                "content": "",
                "metadata": {}
            }
            
            # Extract title from h1, h2, or header
            title_elem = article.find(['h1', 'h2', 'header'])
            if title_elem:
                article_data["title"] = title_elem.get_text(strip=True)
            
            # Extract main content
            content_text = article.get_text(separator=' ', strip=True)
            article_data["content"] = content_text
            
            # Extract metadata
            time_elem = article.find('time')
            if time_elem:
                article_data["metadata"]["published_time"] = time_elem.get('datetime', time_elem.get_text(strip=True))
            
            author_elem = article.find('[rel="author"]') or article.find('.author')
            if author_elem:
                article_data["metadata"]["author"] = author_elem.get_text(strip=True)
            
            return article_data if article_data["content"] else None
            
        except Exception as e:
            logger.error(f"Error parsing article element: {e}")
            return None

    def _parse_main_element(self, main) -> Optional[Dict[str, Any]]:
        """Parse an HTML5 main element."""
        try:
            main_data = {
                "content_type": "main_content",
                "content": main.get_text(separator=' ', strip=True),
                "metadata": {
                    "element_type": "main"
                }
            }
            return main_data if main_data["content"] else None
        except Exception as e:
            logger.error(f"Error parsing main element: {e}")
            return None

    def _extract_product_patterns(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract product information using common patterns."""
        products = []
        try:
            # Look for common product selectors
            product_selectors = [
                '[itemtype*="Product"]',
                '.product',
                '.item',
                '[data-product]',
                '.product-item'
            ]
            
            for selector in product_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    product_data = self._parse_product_element(elem)
                    if product_data:
                        products.append(product_data)
            
            return products
        except Exception as e:
            logger.error(f"Error extracting product patterns: {e}")
            return []

    def _parse_product_element(self, elem) -> Optional[Dict[str, Any]]:
        """Parse a product element."""
        try:
            product_data = {
                "content_type": "product",
                "name": "",
                "price": "",
                "description": "",
                "metadata": {}
            }
            
            # Extract product name
            name_elem = elem.find(['h1', 'h2', 'h3', '.title', '.name', '[itemprop="name"]'])
            if name_elem:
                product_data["name"] = name_elem.get_text(strip=True)
            
            # Extract price
            price_elem = elem.find(['.price', '[itemprop="price"]', '.cost'])
            if price_elem:
                product_data["price"] = price_elem.get_text(strip=True)
            
            # Extract description
            desc_elem = elem.find(['.description', '[itemprop="description"]', '.summary'])
            if desc_elem:
                product_data["description"] = desc_elem.get_text(strip=True)
            
            return product_data if product_data["name"] else None
        except Exception as e:
            logger.error(f"Error parsing product element: {e}")
            return None

    def _extract_listing_patterns(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract listing information using common patterns."""
        listings = []
        try:
            # Look for common listing selectors
            listing_selectors = [
                '.listing-item',
                '.list-item',
                '[data-listing]',
                '.result-item'
            ]
            
            for selector in listing_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    listing_data = self._parse_listing_element(elem)
                    if listing_data:
                        listings.append(listing_data)
            
            return listings
        except Exception as e:
            logger.error(f"Error extracting listing patterns: {e}")
            return []

    def _parse_listing_element(self, elem) -> Optional[Dict[str, Any]]:
        """Parse a listing element."""
        try:
            listing_data = {
                "content_type": "listing",
                "title": "",
                "content": elem.get_text(separator=' ', strip=True),
                "metadata": {}
            }
            
            # Extract title
            title_elem = elem.find(['h1', 'h2', 'h3', '.title'])
            if title_elem:
                listing_data["title"] = title_elem.get_text(strip=True)
            
            return listing_data if listing_data["content"] else None
        except Exception as e:
            logger.error(f"Error parsing listing element: {e}")
            return None

    def _extract_section_patterns(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract section information using semantic patterns."""
        sections = []
        try:
            # Look for semantic section elements
            for section in soup.find_all('section'):
                section_data = self._parse_section_element(section)
                if section_data:
                    sections.append(section_data)
            
            return sections
        except Exception as e:
            logger.error(f"Error extracting section patterns: {e}")
            return []

    def _parse_section_element(self, section) -> Optional[Dict[str, Any]]:
        """Parse a section element."""
        try:
            section_data = {
                "content_type": "section",
                "title": "",
                "content": section.get_text(separator=' ', strip=True),
                "metadata": {}
            }
            
            # Extract section title
            title_elem = section.find(['h1', 'h2', 'h3', 'header'])
            if title_elem:
                section_data["title"] = title_elem.get_text(strip=True)
            
            return section_data if section_data["content"] else None
        except Exception as e:
            logger.error(f"Error parsing section element: {e}")
            return None

    def _parse_heading_section(self, heading) -> Optional[Dict[str, Any]]:
        """Parse a heading and its associated content."""
        try:
            heading_data = {
                "content_type": "heading_section",
                "title": heading.get_text(strip=True),
                "level": heading.name,
                "content": "",
                "metadata": {}
            }
            
            # Get content following the heading
            content_parts = []
            next_elem = heading.next_sibling
            
            while next_elem and next_elem.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if hasattr(next_elem, 'get_text'):
                    text = next_elem.get_text(strip=True)
                    if text:
                        content_parts.append(text)
                next_elem = next_elem.next_sibling
            
            heading_data["content"] = ' '.join(content_parts)
            return heading_data if heading_data["title"] else None
        except Exception as e:
            logger.error(f"Error parsing heading section: {e}")
            return None

    def _parse_table_element(self, table) -> Optional[Dict[str, Any]]:
        """Parse a table element."""
        try:
            table_data = {
                "content_type": "table",
                "headers": [],
                "rows": [],
                "metadata": {}
            }
            
            # Extract headers
            header_row = table.find('thead') or table.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                table_data["headers"] = headers
            
            # Extract rows
            rows = table.find_all('tr')[1:] if table.find('thead') else table.find_all('tr')[1:]
            for row in rows:
                row_data = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                if row_data:
                    table_data["rows"].append(row_data)
            
            return table_data if table_data["headers"] or table_data["rows"] else None
        except Exception as e:
            logger.error(f"Error parsing table element: {e}")
            return None

    def _parse_list_element(self, list_elem) -> Optional[Dict[str, Any]]:
        """Parse a list element (ul/ol)."""
        try:
            list_data = {
                "content_type": "list",
                "type": list_elem.name,
                "items": [],
                "metadata": {}
            }
            
            # Extract list items
            for li in list_elem.find_all('li', recursive=False):
                item_text = li.get_text(strip=True)
                if item_text:
                    list_data["items"].append(item_text)
            
            return list_data if list_data["items"] else None
        except Exception as e:
            logger.error(f"Error parsing list element: {e}")
            return None

    def _parse_form_element(self, form) -> Optional[Dict[str, Any]]:
        """Parse a form element."""
        try:
            form_data = {
                "content_type": "form",
                "action": form.get('action', ''),
                "method": form.get('method', 'GET'),
                "fields": [],
                "metadata": {}
            }
            
            # Extract form fields
            for input_elem in form.find_all(['input', 'select', 'textarea']):
                field_info = {
                    "type": input_elem.get('type', input_elem.name),
                    "name": input_elem.get('name', ''),
                    "placeholder": input_elem.get('placeholder', ''),
                    "required": input_elem.has_attr('required')
                }
                form_data["fields"].append(field_info)
            
            return form_data if form_data["fields"] else None
        except Exception as e:
            logger.error(f"Error parsing form element: {e}")
            return None

    def _parse_content_div(self, div) -> Optional[Dict[str, Any]]:
        """Parse a content div element."""
        try:
            content_text = div.get_text(separator=' ', strip=True)
            if len(content_text) < 50:  # Skip short content divs
                return None
            
            div_data = {
                "content_type": "content_div",
                "content": content_text,
                "class": ' '.join(div.get('class', [])),
                "metadata": {
                    "content_length": len(content_text)
                }
            }
            
            return div_data if div_data["content"] else None
        except Exception as e:
            logger.error(f"Error parsing content div: {e}")
            return None


    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the service with the given configuration."""
        if self._initialized:
            return
        
        # Apply additional configuration if provided
        if config:
            self.config.update(config)
        
        # Initialize pipeline components if enabled and not already done
        if self.use_pipelines and not self._pipeline_components_initialized:
            self._initialize_pipeline_components()
        
        # AI service is already initialized in _register_core_services
        
        self._initialized = True
        logger.info("AdaptiveScraper service initialized")

    async def startup(self) -> None:
        """Asynchronous startup initialization for components that require async setup."""
        try:
            logger.info("Starting AdaptiveScraper async initialization...")
            
            # Initialize any async components here if needed
            # For now, just ensure initialization is complete
            if not self._initialized:
                self.initialize()
            
            logger.info("AdaptiveScraper startup completed successfully")
        except Exception as e:
            logger.error(f"Error during AdaptiveScraper startup: {e}")
            raise

    def shutdown(self) -> None:
        """Perform cleanup and shutdown operations."""
        # Clean up UniversalHunter session
        if hasattr(self, 'hunter') and self.hunter is not None:
            try:
                if hasattr(self.hunter, 'shutdown'):
                    self.hunter.shutdown()
                self.hunter = None
                logger.info("UniversalHunter cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up UniversalHunter: {e}")
        
        # Clean up hunter HTTP session
        if hasattr(self, 'hunter_session') and self.hunter_session is not None:
            try:
                # Note: In a real async context, we'd await this
                # For now, we'll close it synchronously
                if not self.hunter_session.closed:
                    # This is a sync call, which is fine for shutdown
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If we're in an event loop, we can't await
                            # Just close it and let the loop cleanup
                            self.hunter_session.close()
                        else:
                            loop.run_until_complete(self.hunter_session.close())
                    except:
                        self.hunter_session.close()
                self.hunter_session = None
                logger.info("UniversalHunter HTTP session closed")
            except Exception as e:
                logger.warning(f"Error closing UniversalHunter session: {e}")
        
        self._initialized = False
        logger.info("AdaptiveScraper service shut down")

    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "adaptive_scraper"

    @property
    def is_initialized(self) -> bool:
        """Return whether the service is initialized."""
        return self._initialized

    def get_registered_services(self) -> Dict[str, Any]:
        """Get all registered services from the service registry."""
        return self.service_registry._services if hasattr(self.service_registry, '_services') else {}

    def get_service(self, service_name: str) -> Any:
        """Get a specific service from the service registry."""
        return self.service_registry.get_service(service_name)

    def get_ai_service(self) -> Optional[Any]:
        """Get or create the AI service."""
        try:
            # Try to get existing AI service from registry
            if "ai_service" in self.service_registry._services:
                return self.service_registry.get_service("ai_service")
            
            # Try to create new AI service
            from core.ai_service import AIService
            ai_service = AIService()
            ai_service.initialize()
            return ai_service
            
        except ImportError:
            logger.warning("AIService not available, using mock AI service")
            # Return a mock AI service for compatibility
            from core.service_interface import BaseService
            
            class MockAIService(BaseService):
                def __init__(self):
                    self._initialized = False
                
                def initialize(self, config=None):
                    self._initialized = True
                
                def shutdown(self):
                    self._initialized = False
                
                @property
                def name(self):
                    return "ai_service"
                    
                async def process_with_ai(self, *args, **kwargs):
                    return {"success": False, "error": "Mock AI service"}
            
            mock_ai = MockAIService()
            mock_ai.initialize()
            return mock_ai
            
        except Exception as e:
            logger.error(f"Failed to create AI service: {e}")
            return None

    async def scrape(self, url: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main scraping method for backwards compatibility and simple usage.
        
        Args:
            url: The URL to scrape
            options: Optional scraping options
            
        Returns:
            Dict containing the scraping results
        """
        try:
            # Use the execute_search_pipeline method as the main scraping logic
            query = options.get('query', 'general content') if options else 'general content'
            return await self.execute_search_pipeline(query, url, options)
        except Exception as e:
            logger.error(f"Error in scrape method: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "results": []
            }

    # === PUBLIC API METHODS ===
    # These methods serve as the main entry points for the AdaptiveScraper
    
    async def scrape_data(
        self,
        url: str,
        query: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main async scraping method that extracts structured data from a URL.
        
        Args:
            url: The URL to scrape
            query: Optional search query or extraction intent
            options: Optional configuration parameters
                - max_results: Maximum number of results to extract
                - timeout: Request timeout in seconds
                - strategy: Preferred extraction strategy
                - schema: Data schema for structured extraction
                - fallback_enabled: Whether to use fallback strategies
        
        Returns:
            Dict containing extracted data with structure:
            {
                "items": [...],  # List of extracted items
                "metadata": {...},  # Extraction metadata
                "extraction_time": float,  # Time taken
                "source_url": str,  # Original URL
                "success": bool  # Whether extraction succeeded
            }
        """
        start_time = time.time()
        options = options or {}
        
        try:
            logger.info(f"Starting scrape_data for URL: {url}")
            
            # Validate URL
            if not url or not isinstance(url, str):
                raise ValueError("URL must be a non-empty string")
            
            # Use pipeline-based extraction
            pipeline_result = await self.scrape_with_pipeline(
                url=url,
                query=query,
                extraction_options=options
            )
            
            # Format result according to expected API structure
            if pipeline_result.get("success"):
                items = pipeline_result.get("results", [])
                
                # Ensure items is a list
                if not isinstance(items, list):
                    items = [items] if items else []
                
                return {
                    "items": items,
                    "metadata": {
                        "source_url": url,
                        "query": query,
                        "extraction_method": pipeline_result.get("pipeline_type", "adaptive"),
                        "total_items": len(items),
                        **pipeline_result.get("metadata", {})
                    },
                    "extraction_time": time.time() - start_time,
                    "source_url": url,
                    "success": True
                }
            else:
                # Fallback to basic extraction
                logger.warning(f"Pipeline extraction failed for {url}, trying fallback")
                return await self._scrape_data_fallback(url, query, options, start_time)
                
        except Exception as e:
            logger.error(f"Error in scrape_data for {url}: {e}")
            return await self._scrape_data_fallback(url, query, options, start_time)
    
    async def _scrape_data_fallback(
        self, 
        url: str, 
        query: Optional[str], 
        options: Dict[str, Any], 
        start_time: float
    ) -> Dict[str, Any]:
        """Fallback scraping method when pipeline extraction fails."""
        try:
            # Direct extraction without recursion - fetch HTML and extract basic content
            html_content = await self._fetch_html_content(url, options)
            
            if html_content:
                # Try to extract basic content directly
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Extract basic information
                    title = soup.title.string if soup.title else ""
                    
                    # Extract text content
                    text_content = soup.get_text()
                    
                    # If query is provided, try to find relevant content
                    relevant_content = text_content[:1000]  # Limit content
                    if query:
                        # Simple relevance check
                        query_lower = query.lower()
                        sentences = text_content.split('.')
                        relevant_sentences = [s for s in sentences if query_lower in s.lower()]
                        if relevant_sentences:
                            relevant_content = '. '.join(relevant_sentences[:5])
                    
                    items = [{
                        "title": title.strip(),
                        "content": relevant_content.strip(),
                        "url": url,
                        "extraction_method": "fallback_direct"
                    }]
                    
                    return {
                        "items": items,
                        "metadata": {
                            "source_url": url,
                            "extraction_method": "fallback_direct",
                            "total_items": len(items)
                        },
                        "extraction_time": time.time() - start_time,
                        "source_url": url,
                        "success": True
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to parse HTML content: {e}")
            
            # Last resort: basic web crawling
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                if result.success and result.extracted_content:
                    items = [{
                        "title": result.metadata.get("title", ""),
                        "content": result.extracted_content[:1000],  # Limit content
                        "url": url
                    }]
                    
                    return {
                        "items": items,
                        "metadata": {
                            "source_url": url,
                            "extraction_method": "basic_crawl",
                            "total_items": 1
                        },
                        "extraction_time": time.time() - start_time,
                        "source_url": url,
                        "success": True
                    }
            
            # If all else fails, return empty result
            return {
                "items": [],
                "metadata": {"source_url": url, "extraction_method": "failed"},
                "extraction_time": time.time() - start_time,
                "source_url": url,
                "success": False
            }
            
        except Exception as e:
            logger.error(f"Fallback scraping failed for {url}: {e}")
            return {
                "items": [],
                "metadata": {"source_url": url, "error": str(e)},
                "extraction_time": time.time() - start_time,
                "source_url": url,
                "success": False,
                "error": str(e)
            }
    
    async def extract_with_schema(
        self,
        url: str,
        schema: Dict[str, Any],
        query: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract data from a URL using a predefined schema.
        
        Args:
            url: The URL to scrape
            schema: Data schema defining the structure to extract
            query: Optional search query
            options: Additional extraction options
        
        Returns:
            Dict containing schema-structured extracted data
        """
        start_time = time.time()
        options = options or {}
        
        try:
            logger.info(f"Starting schema extraction for URL: {url}")
            
            # Add schema to extraction options
            extraction_options = {
                **options,
                "schema": schema,
                "structured_extraction": True
            }
            
            # Use pipeline extraction with schema
            pipeline_result = await self.execute_extraction_pipeline(
                url=url,
                content=None,  # Will be fetched by pipeline
                config={
                    "query": query,
                    "schema": schema,
                    **extraction_options
                }
            )
            
            if pipeline_result.get("success"):
                items = pipeline_result.get("results", [])
                
                # Validate items against schema
                validated_items = []
                for item in (items if isinstance(items, list) else [items]):
                    if isinstance(item, dict):
                        # Basic schema validation - ensure required fields exist
                        validated_item = {}
                        for field_name, field_config in schema.get("properties", {}).items():
                            if field_name in item:
                                validated_item[field_name] = item[field_name]
                            elif schema.get("required", []) and field_name in schema["required"]:
                                validated_item[field_name] = None  # Required but missing
                        validated_items.append(validated_item)
                
                return {
                    "items": validated_items,
                    "metadata": {
                        "source_url": url,
                        "schema_applied": True,
                        "total_items": len(validated_items),
                        "extraction_time": time.time() - start_time
                    },
                    "schema": schema,
                    "success": True
                }
            else:
                raise Exception(f"Pipeline extraction failed: {pipeline_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Schema extraction failed for {url}: {e}")
            
            # Fallback: try regular extraction and map to schema
            try:
                regular_result = await self.scrape_data(url, query, options)
                if regular_result.get("success") and regular_result.get("items"):
                    # Attempt to map regular results to schema
                    mapped_items = []
                    for item in regular_result["items"]:
                        mapped_item = {}
                        for field_name in schema.get("properties", {}).keys():
                            # Simple field mapping - look for similar keys
                            for key, value in item.items():
                                if field_name.lower() in key.lower() or key.lower() in field_name.lower():
                                    mapped_item[field_name] = value
                                    break
                        if mapped_item:
                            mapped_items.append(mapped_item)
                    
                    return {
                        "items": mapped_items,
                        "metadata": {
                            "source_url": url,
                            "schema_applied": True,
                            "extraction_method": "fallback_mapping",
                            "total_items": len(mapped_items)
                        },
                        "schema": schema,
                        "success": True
                    }
            except Exception as fallback_error:
                logger.error(f"Schema extraction fallback failed: {fallback_error}")
            
            return {
                "items": [],
                "metadata": {"source_url": url, "error": str(e)},
                "schema": schema,
                "success": False,
                "error": str(e)
            }
    
    async def extract_raw_content(
        self,
        url: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract raw content from a URL without structured parsing.
        
        Args:
            url: The URL to extract content from
            options: Extraction options
                - include_metadata: Whether to include page metadata
                - content_type: Type of content to extract (text, html, etc.)
        
        Returns:
            Dict containing raw content and metadata
        """
        start_time = time.time()
        options = options or {}
        
        try:
            logger.info(f"Starting raw content extraction for URL: {url}")
            
            # Use web crawler for raw content extraction
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=10,
                    extraction_strategy="NoExtractionStrategy" if options.get("content_type") == "html" else "LLMExtractionStrategy"
                )
                
                if result.success:
                    content_data = {
                        "content": result.extracted_content or result.cleaned_html,
                        "title": result.metadata.get("title", ""),
                        "url": url,
                        "success": True
                    }
                    
                    if options.get("include_metadata", True):
                        content_data["metadata"] = {
                            "word_count": len((result.extracted_content or "").split()),
                            "extraction_method": "raw",
                            "page_title": result.metadata.get("title", ""),
                            "page_description": result.metadata.get("description", ""),
                            "content_length": len(result.extracted_content or result.cleaned_html or ""),
                            "extraction_time": time.time() - start_time,
                            **result.metadata
                        }
                    
                    return content_data
                else:
                    raise Exception(f"Failed to extract content: {result.error_message}")
                    
        except Exception as e:
            logger.error(f"Raw content extraction failed for {url}: {e}")
            return {
                "content": "",
                "title": "",
                "url": url,
                "success": False,
                "error": str(e),
                "metadata": {
                    "extraction_method": "failed",
                    "extraction_time": time.time() - start_time
                }
            }
    
    async def batch_scrape(
        self,
        urls: List[str],
        query: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Scrape multiple URLs in batch with concurrency control.
        
        Args:
            urls: List of URLs to scrape
            query: Optional search query for all URLs
            options: Batch processing options
                - max_results_per_url: Maximum results per URL
                - timeout: Per-URL timeout
                - max_concurrent: Maximum concurrent requests
                - fail_fast: Whether to stop on first failure
        
        Returns:
            Dict containing batch results and summary
        """
        start_time = time.time()
        options = options or {}
        
        if not urls:
            return {
                "results": [],
                "summary": {"total_urls": 0, "successful": 0, "failed": 0},
                "success": True
            }
        
        max_concurrent = options.get("max_concurrent", 5)
        fail_fast = options.get("fail_fast", False)
        timeout = options.get("timeout", 30)
        
        logger.info(f"Starting batch scrape of {len(urls)} URLs with max_concurrent={max_concurrent}")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        successful = 0
        failed = 0
        
        async def scrape_single_url(url: str, index: int) -> Dict[str, Any]:
            async with semaphore:
                try:
                    logger.debug(f"Processing URL {index+1}/{len(urls)}: {url}")
                    
                    # Create per-URL options
                    url_options = {
                        **options,
                        "timeout": timeout
                    }
                    if "max_results_per_url" in options:
                        url_options["max_results"] = options["max_results_per_url"]
                    
                    # Use asyncio.wait_for for timeout control
                    result = await asyncio.wait_for(
                        self.scrape_data(url, query, url_options),
                        timeout=timeout
                    )
                    
                    return {
                        "url": url,
                        "index": index,
                        "success": result.get("success", True),
                        "data": result,
                        "error": None
                    }
                    
                except asyncio.TimeoutError:
                    return {
                        "url": url,
                        "index": index,
                        "success": False,
                        "data": None,
                        "error": f"Timeout after {timeout}s"
                    }
                except Exception as e:
                    return {
                        "url": url,
                        "index": index,
                        "success": False,
                        "data": None,
                        "error": str(e)
                    }
        
        try:
            # Process URLs concurrently
            tasks = [scrape_single_url(url, i) for i, url in enumerate(urls)]
            
            if fail_fast:
                # Process one by one, stopping on first failure
                for task in tasks:
                    result = await task
                    results.append(result)
                    
                    if result["success"]:
                        successful += 1
                    else:
                        failed += 1
                        logger.warning(f"URL failed: {result['url']} - {result['error']}")
                        break  # Stop on first failure
            else:
                # Process all URLs, collecting all results
                completed_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(completed_results):
                    if isinstance(result, Exception):
                        results.append({
                            "url": urls[i],
                            "index": i,
                            "success": False,
                            "data": None,
                            "error": str(result)
                        })
                        failed += 1
                    else:
                        results.append(result)
                        if result["success"]:
                            successful += 1
                        else:
                            failed += 1
            
            # Sort results by original index to maintain order
            results.sort(key=lambda x: x["index"])
            
            total_time = time.time() - start_time
            logger.info(f"Batch scrape completed: {successful} successful, {failed} failed in {total_time:.2f}s")
            
            return {
                "results": results,
                "summary": {
                    "total_urls": len(urls),
                    "successful": successful,
                    "failed": failed,
                    "success_rate": successful / len(urls) if urls else 0,
                    "total_time": total_time,
                    "avg_time_per_url": total_time / len(urls) if urls else 0
                },
                "success": True,
                "query": query,
                "options": options
            }
            
        except Exception as e:
            logger.error(f"Batch scrape failed: {e}")
            return {
                "results": results,
                "summary": {
                    "total_urls": len(urls),
                    "successful": successful,
                    "failed": failed + (len(urls) - len(results)),
                    "error": str(e)
                },
                "success": False,
                "error": str(e)
            }
    
    async def scrape_with_config(
        self,
        url: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Alternative scraping method with config-based parameter structure.
        
        Args:
            url: The URL to scrape
            config: Configuration dictionary containing query and options
        
        Returns:
            Dict containing extracted data
        """
        config = config or {}
        query = config.get("query")
        options = {k: v for k, v in config.items() if k != "query"}
        
        logger.info(f"Starting scrape (alternative API) for URL: {url}")
        
        # Delegate to main scrape_data method
        result = await self.scrape_data(url, query, options)
        
        # Ensure compatibility with expected format
        if result.get("success"):
            return {
                "items": result.get("items", []),
                "metadata": result.get("metadata", {}),
                "url": url,
                "success": True
            }
        else:
            return {
                "items": [],
                "metadata": {"error": result.get("error", "Unknown error")},
                "url": url,
                "success": False
            }
    
    async def _scrape_with_strategy(
        self,
        url: str,
        strategy_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Internal method to scrape using a specific strategy.
        
        Args:
            url: The URL to scrape
            strategy_name: Name of the strategy to use
            config: Strategy configuration
        
        Returns:
            Dict containing extraction results
        """
        config = config or {}
        
        try:
            logger.info(f"Scraping {url} with strategy: {strategy_name}")
            
            # Get strategy from factory or registry
            strategy = None
            
            # Try to get from strategy factory
            try:
                if hasattr(self, 'strategy_factory'):
                    strategy = await self.strategy_factory.create_strategy(
                        strategy_name,
                        config
                    )
            except Exception as e:
                logger.debug(f"Could not create strategy from factory: {e}")
            
            # Try direct strategy instantiation as fallback
            if not strategy:
                strategy_classes = {
                    "ai_guided": AIGuidedStrategy,
                    "dom": DOMStrategy,
                    "api": APIStrategy,
                    "form": FormStrategy,
                    "url_param": URLParamStrategy
                }
                
                strategy_class = strategy_classes.get(strategy_name.lower())
                if strategy_class:
                    strategy = strategy_class(config)
                else:
                    raise ValueError(f"Unknown strategy: {strategy_name}")
            
            # Execute strategy
            if hasattr(strategy, 'execute_async'):
                result = await strategy.execute_async(url, config)
            elif hasattr(strategy, 'execute'):
                # Wrap sync execute in async
                result = await asyncio.get_event_loop().run_in_executor(
                    None, strategy.execute, url, config
                )
            else:
                raise AttributeError(f"Strategy {strategy_name} has no execute method")
            
            return {
                "items": result.get("items", []) if isinstance(result, dict) else [result],
                "strategy": strategy_name,
                "success": True,
                "metadata": {
                    "strategy_used": strategy_name,
                    "url": url
                }
            }
            
        except Exception as e:
            logger.error(f"Strategy {strategy_name} failed for {url}: {e}")
            return {
                "items": [],
                "strategy": strategy_name,
                "success": False,
                "error": str(e),
                "metadata": {
                    "strategy_used": strategy_name,
                    "url": url,
                    "error": str(e)
                }
            }


            # Don't fail initialization if strategy registration fails
            pass

    def _register_core_services(self) -> None:
        """Register core services with the service registry."""
        # Instantiate core services
        self.search_orchestrator = SearchOrchestrator()
        self.site_discovery = SiteDiscovery()
        self.domain_intelligence = DomainIntelligence()
        self.metrics_analyzer = MetricsAnalyzer()
        self.continuous_improvement = ContinuousImprovementSystem()
        self.pattern_analyzer = DefaultPatternAnalyzer()
        self.intent_parser = get_intent_parser()

        # Create additional required services
        from core.url_service import URLService
        from core.html_service import HTMLService
        from core.session_manager import SessionManager
        from core.ai_service import AIService

        self.url_service = URLService()
        self.html_service = HTMLService()
        self.session_manager = SessionManager()
        
        # Register intent_parser first so SearchTermGenerator can use it
        if "intent_parser" not in self.service_registry._services:
            self.service_registry.register_service("intent_parser", self.intent_parser)
        
        # Register schema_extractor service
        if "schema_extractor" not in self.service_registry._services:
            try:
                from extraction.schema_extraction import SchemaExtractor
                self.schema_extractor = SchemaExtractor()
                self.service_registry.register_service("schema_extractor", self.schema_extractor)
                logger.info("SchemaExtractor registered successfully.")
            except ImportError as e:
                logger.warning(f"SchemaExtractor not available: {e}, using fallback MockSchemaExtractor.")
                from core.service_interface import BaseService
                class MockSchemaExtractor(BaseService):
                    def __init__(self):
                        self._initialized = False
                    def initialize(self, config=None):
                        self._initialized = True
                    def shutdown(self):
                        self._initialized = False
                    @property
                    def name(self):
                        return "schema_extractor"
                    async def extract(self, html, url, schema, context=None):
                        return {"success": False, "error": "Mock schema extractor"}
                mock_schema = MockSchemaExtractor()
                self.service_registry.register_service("schema_extractor", mock_schema)
        
        # Now create SearchTermGenerator with access to service registry
        self.search_term_generator = SearchTermGenerator(service_registry=self.service_registry)
        # Initialize the SearchTermGenerator to ensure nlp and other attributes are properly set
        self.search_term_generator.initialize()
        
        # Use lazy initialization for AI service to avoid initialization order issues
        self.ai_service = None  # Will be set when first needed
        logger.info("AdaptiveScraper initialized with lazy AI service loading")

        # Register with service registry (with existence checks to prevent duplicates)
        if "search_orchestrator" not in self.service_registry._services:
            self.service_registry.register_service("search_orchestrator", self.search_orchestrator)
        if "search_term_generator" not in self.service_registry._services:
            self.service_registry.register_service("search_term_generator", self.search_term_generator)
        if "site_discovery" not in self.service_registry._services:
            self.service_registry.register_service("site_discovery", self.site_discovery)
        if "domain_intelligence" not in self.service_registry._services:
            self.service_registry.register_service("domain_intelligence", self.domain_intelligence)
        if "metrics_analyzer" not in self.service_registry._services:
            self.service_registry.register_service("metrics_analyzer", self.metrics_analyzer)
        if "continuous_improvement" not in self.service_registry._services:
            # Initialize the ContinuousImprovementSystem with proper configuration
            improvement_config = {
                'results_dir': os.path.join(os.getcwd(), 'test_results'),
                'patterns_dir': os.path.join(os.getcwd(), 'extraction_strategies', 'extraction_profiles'),
                'strategies_dir': os.path.join(os.getcwd(), 'strategies')
            }
            self.continuous_improvement.initialize(improvement_config)
            self.service_registry.register_service("continuous_improvement", self.continuous_improvement)
        if "pattern_analyzer" not in self.service_registry._services:
            self.service_registry.register_service("pattern_analyzer", self.pattern_analyzer)
        if "url_service" not in self.service_registry._services:
            self.service_registry.register_service("url_service", self.url_service)
        if "html_service" not in self.service_registry._services:
            self.service_registry.register_service("html_service", self.html_service)
        if "session_manager" not in self.service_registry._services:
            self.service_registry.register_service("session_manager", self.session_manager)
        if "ai_service" not in self.service_registry._services:
            # Register a placeholder that will be replaced with actual AI service when needed
            ai_service = self.get_ai_service()
            if ai_service:
                self.service_registry.register_service("ai_service", ai_service)

        # Register ProxyManager with fallback
        if "proxy_manager" not in self.service_registry._services:
            try:
                from core.proxy_manager import ProxyManager
                from config import PROXY_ENABLED, PROXY_LIST
                
                self.proxy_manager = ProxyManager()
                
                # Configure proxy providers based on environment settings
                proxy_config = {
                    'enable_health_checks': False,  # Disable health checks to avoid extra errors
                    'providers': {}
                }
                
                # Add default static provider if proxies are enabled and available
                if PROXY_ENABLED and PROXY_LIST:
                    proxy_config['providers']['default'] = {
                        'type': 'static',
                        'proxies': PROXY_LIST
                    }
                    logger.info(f"ProxyManager configured with {len(PROXY_LIST)} proxies from environment")
                elif PROXY_ENABLED:
                    logger.warning("PROXY_ENABLED is True but no PROXY_LIST provided - ProxyManager will have no proxies")
                
                self.proxy_manager.initialize(proxy_config)
                self.service_registry.register_service("proxy_manager", self.proxy_manager)
                logger.info("ProxyManager registered successfully.")
            except ImportError:
                logger.warning("ProxyManager not available, using fallback MockProxyManager.")
                from core.service_interface import BaseService
                class MockProxyManager(BaseService):
                    def __init__(self):
                        self._initialized = False
                    def initialize(self, config=None):
                        self._initialized = True
                    def shutdown(self):
                        self._initialized = False
                    @property
                    def name(self):
                        return "proxy_manager"
                    @property
                    def is_initialized(self):
                        return self._initialized
                    def get_all_proxies(self): return []
                    def get_proxy(self, *args, **kwargs): return None
                mock_proxy = MockProxyManager()
                mock_proxy.initialize()
                self.service_registry.register_service("proxy_manager", mock_proxy)
            except Exception as e:
                logger.error(f"Failed to initialize or register ProxyManager: {e}")
                # Fallback to MockProxyManager if any other error occurs during init
                from core.service_interface import BaseService
                class MockProxyManagerOnError(BaseService): # Different name to avoid conflict if ImportError also happened
                    def __init__(self):
                        self._initialized = False
                    def initialize(self, config=None):
                        self._initialized = True
                    def shutdown(self):
                        self._initialized = False
                    @property
                    def name(self):
                        return "proxy_manager"
                    @property
                    def is_initialized(self):
                        return self._initialized
                    def get_all_proxies(self): return []
                    def get_proxy(self, *args, **kwargs): return None
                mock_proxy = MockProxyManagerOnError()
                mock_proxy.initialize()
                self.service_registry.register_service("proxy_manager", mock_proxy)
        else:
            logger.debug("ProxyManager already registered, skipping registration.")

        # Register AIModelManager
        if "ai_model_manager" not in self.service_registry._services:
            try:
                from core.ai.model_manager import AIModelManager # Assuming this is the location
                from core.service_interface import BaseService
                
                # Create a wrapper to make AIModelManager compatible with BaseService
                class AIModelManagerWrapper(BaseService):
                    def __init__(self, available_models, default_model_name, config_manager=None):
                        self._initialized = False
                        self.ai_model_manager = AIModelManager(available_models, default_model_name, config_manager)
                    
                    def initialize(self, config=None):
                        self._initialized = True
                    
                    def shutdown(self):
                        self._initialized = False
                    
                    @property
                    def name(self):
                        return "ai_model_manager"
                    
                    @property
                    def is_initialized(self):
                        return self._initialized
                    
                    # Delegate all other methods to the wrapped manager
                    def __getattr__(self, name):
                        return getattr(self.ai_model_manager, name)
                
                # Create a simple config manager if one doesn't exist
                config_manager = getattr(self, 'config_manager', None)
                if config_manager is None:
                    # Create a simple config manager that can access environment variables
                    class SimpleConfigManager:
                        def get_api_key(self, key_name):
                            import os
                            return os.getenv(key_name)
                    
                    self.config_manager = SimpleConfigManager()
                    config_manager = self.config_manager
                    logger.info("Created SimpleConfigManager for AIModelManager API key access.")

                self.ai_model_manager = AIModelManagerWrapper(
                    available_models=self.available_models,
                    default_model_name=self.default_model_name,
                    config_manager=config_manager # Pass config manager for API keys
                )
                self.ai_model_manager.initialize()
                self.service_registry.register_service("ai_model_manager", self.ai_model_manager)
                logger.info("AIModelManager registered successfully.")
            except ImportError:
                logger.warning("AIModelManager not available. AI features will be limited.")
                # Optionally register a mock/fallback AIModelManager
                from core.service_interface import BaseService
                class MockAIModelManager(BaseService):
                    def __init__(self):
                        self._initialized = False
                    def initialize(self, config=None):
                        self._initialized = True
                    def shutdown(self):
                        self._initialized = False
                    @property
                    def name(self):
                        return "ai_model_manager"
                    def get_model(self, model_name=None): return None
                    def select_model(self, requirements=None): return None
                mock_ai = MockAIModelManager()
                mock_ai.initialize()
                self.service_registry.register_service("ai_model_manager", mock_ai)
            except Exception as e:
                logger.error(f"Failed to initialize or register AIModelManager: {e}")
                from core.service_interface import BaseService
                class MockAIModelManagerOnError(BaseService):
                    def __init__(self):
                        self._initialized = False
                    def initialize(self, config=None):
                        self._initialized = True
                    def shutdown(self):
                        self._initialized = False
                    @property
                    def name(self):
                        return "ai_model_manager"
                    def get_model(self, model_name=None): return None
                    def select_model(self, requirements=None): return None
                mock_ai = MockAIModelManagerOnError()
                mock_ai.initialize()
                self.service_registry.register_service("ai_model_manager", mock_ai)
        else:
            logger.debug("AIModelManager already registered, skipping registration.")

        logger.debug("Core services registered")

        # **NEW: Register ExtractionCoordinator and UniversalIntentAnalyzer as critical missing components**
        if "extraction_coordinator" not in self.service_registry._services:
            try:
                # Create BaseService wrapper for ExtractionCoordinator without immediate import
                from core.service_interface import BaseService
                class ExtractionCoordinatorService(BaseService):
                    def __init__(self):
                        self._initialized = False
                        self.coordinator = None
                    
                    def initialize(self, config=None):
                        # Lazy import to avoid circular dependency during initialization
                        from controllers.extraction_coordinator import ExtractionCoordinator
                        self.coordinator = ExtractionCoordinator()
                        self._initialized = True
                        logger.info("ExtractionCoordinator initialized successfully")
                    
                    def shutdown(self):
                        if self.coordinator and hasattr(self.coordinator, 'shutdown'):
                            import asyncio
                            try:
                                if asyncio.iscoroutinefunction(self.coordinator.shutdown):
                                    # Try to get running event loop, if none create one
                                    try:
                                        loop = asyncio.get_running_loop()
                                        task = loop.create_task(self.coordinator.shutdown())
                                        # Don't wait for completion during shutdown to avoid blocking
                                    except RuntimeError:
                                        # No running loop, run shutdown directly
                                        asyncio.run(self.coordinator.shutdown())
                                else:
                                    self.coordinator.shutdown()
                            except Exception as e:
                                logger.warning(f"Error shutting down ExtractionCoordinator: {e}")
                        self._initialized = False
                    
                    @property
                    def name(self):
                        return "extraction_coordinator"
                    
                    def __getattr__(self, name):
                        # Delegate all other methods to the wrapped coordinator
                        if self.coordinator:
                            return getattr(self.coordinator, name)
                        raise AttributeError(f"ExtractionCoordinator not initialized")
                
                extraction_coordinator_service = ExtractionCoordinatorService()
                # Don't initialize immediately to avoid circular imports
                self.service_registry.register_service("extraction_coordinator", extraction_coordinator_service)
                logger.info("ExtractionCoordinator service registered (will initialize on first use)")
            except ImportError as e:
                logger.warning(f"ExtractionCoordinator not available: {e}")
            except Exception as e:
                logger.error(f"Failed to register ExtractionCoordinator: {e}")

        if "universal_intent_analyzer" not in self.service_registry._services:
            try:
                from components.universal_intent_analyzer import UniversalIntentAnalyzer
                # Create BaseService wrapper for UniversalIntentAnalyzer
                from core.service_interface import BaseService
                class UniversalIntentAnalyzerService(BaseService):
                    def __init__(self):
                        self._initialized = False
                        self.analyzer = None
                    
                    def initialize(self, config=None):
                        self.analyzer = UniversalIntentAnalyzer()
                        self._initialized = True
                        logger.info("UniversalIntentAnalyzer initialized successfully")
                    
                    def shutdown(self):
                        if self.analyzer and hasattr(self.analyzer, 'shutdown'):
                            try:
                                self.analyzer.shutdown()
                            except Exception as e:
                                logger.warning(f"Error shutting down UniversalIntentAnalyzer: {e}")
                        self._initialized = False
                    
                    @property
                    def name(self):
                        return "universal_intent_analyzer"
                    
                    def __getattr__(self, name):
                        # Delegate all other methods to the wrapped analyzer
                        if self.analyzer:
                            return getattr(self.analyzer, name)
                        raise AttributeError(f"UniversalIntentAnalyzer not initialized")
                
                intent_analyzer_service = UniversalIntentAnalyzerService()
                intent_analyzer_service.initialize()
                self.service_registry.register_service("universal_intent_analyzer", intent_analyzer_service)
                logger.info("UniversalIntentAnalyzer registered successfully in service registry")
            except ImportError as e:
                logger.warning(f"UniversalIntentAnalyzer not available: {e}")
            except Exception as e:
                logger.error(f"Failed to register UniversalIntentAnalyzer: {e}")

    # Search Pipeline Methods
    
    async def _prepare_search_terms(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare search terms from user query and context.
        
        Args:
            context: Current execution context with query, url, and options
            
        Returns:
            Updated context with search_terms added
        """
        try:
            query = context.get("query", "")
            if not query:
                context["search_terms"] = {"primary": "", "variants": []}
                return context
            
            # Get search term generator service
            search_term_generator = self.service_registry.get_service("search_term_generator")
            
            if search_term_generator:
                # Generate search terms with variations
                search_terms = await search_term_generator.generate_search_terms(
                    query, 
                    site_url=context.get("url"),
                    num_variations=context.get("options", {}).get("num_variations", 3)
                )
            else:
                # Fallback - create basic search terms
                search_terms = {
                    "primary": query,
                    "variants": [f"{query} -site", f'"{query}"', f"{query} info"]
                }
            
            context["search_terms"] = search_terms
            logger.debug(f"Prepared search terms: {search_terms}")
            return context
            
        except Exception as e:
            logger.error(f"Error preparing search terms: {e}")
            context["search_terms"] = {"primary": context.get("query", ""), "variants": []}
            return context
    
    async def _select_search_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the best strategy for the current search context.
        
        Args:
            context: Current execution context
            
        Returns:
            Updated context with strategy and strategy_name added
        """
        try:
            url = context.get("url", "")
            options = context.get("options", {})
            required_capabilities = options.get("required_capabilities", set())
            strategy_name = options.get("strategy_name")
            
            # If specific strategy requested, use it
            if strategy_name:
                strategy = self.strategy_factory.get_strategy(strategy_name)
                if strategy:
                    context["strategy"] = strategy
                    context["strategy_name"] = strategy_name
                    logger.debug(f"Using requested strategy: {strategy_name}")
                    return context
            
            # Select strategy based on URL and capabilities
            strategy = self.strategy_factory.select_best_strategy(
                url=url,
                capabilities=required_capabilities
            )
            
            if strategy:
                context["strategy"] = strategy
                context["strategy_name"] = getattr(strategy, 'name', 'unknown')
                logger.debug(f"Selected strategy: {context['strategy_name']}")
            else:
                # Fallback to a default strategy
                fallback_strategies = getattr(self, 'fallback_strategies', ['multi_strategy'])
                for fallback_name in fallback_strategies:
                    fallback_strategy = self.strategy_factory.get_strategy(fallback_name)
                    if fallback_strategy:
                        context["strategy"] = fallback_strategy
                        context["strategy_name"] = fallback_name
                        logger.warning(f"Using fallback strategy: {fallback_name}")
                        break
            
            return context
            
        except Exception as e:
            logger.error(f"Error selecting search strategy: {e}")
            context["error"] = str(e)
            return context
    
    async def _execute_search(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the selected strategy with the search terms.
        
        Args:
            context: Current execution context with strategy and search_terms
            
        Returns:
            Updated context with raw_results added
        """
        try:
            strategy = context.get("strategy")
            if not strategy:
                raise ValueError("No strategy available for execution")
            
            url = context.get("url", "")
            search_terms = context.get("search_terms", {})
            options = context.get("options", {})
            
            # Execute the strategy
            if hasattr(strategy, 'execute'):
                # Check if execute method is async
                if asyncio.iscoroutinefunction(strategy.execute):
                    result = await strategy.execute(
                        url=url,
                        search_terms=search_terms,
                        **options
                    )
                else:
                    # Call synchronous method
                    result = strategy.execute(
                        url=url,
                        search_terms=search_terms,
                        **options
                    )
                
                if result:
                    context["raw_results"] = result if isinstance(result, list) else [result]
                else:
                    context["raw_results"] = []
                    
            elif hasattr(strategy, 'crawl'):
                # Use crawl method for traversal strategies
                # Check if crawl method is async
                if asyncio.iscoroutinefunction(strategy.crawl):
                    result = await strategy.crawl(
                        start_url=url,
                        **options
                    )
                else:
                    # Call synchronous method
                    result = strategy.crawl(
                        start_url=url,
                        **options
                    )
                context["raw_results"] = result if isinstance(result, list) else [result] if result else []
            else:
                logger.warning(f"Strategy {context.get('strategy_name')} has no execute or crawl method")
                context["raw_results"] = []
            
            logger.debug(f"Strategy execution complete. Results count: {len(context.get('raw_results', []))}")
            return context
            
        except Exception as e:
            logger.error(f"Error executing search: {e}")
            context["error"] = str(e)
            context["raw_results"] = []
            return context
    
    async def _process_results(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and normalize the raw results from strategy execution.
        
        Args:
            context: Current execution context with raw_results
            
        Returns:
            Updated context with processed results
        """
        try:
            raw_results = context.get("raw_results", [])
            
            if not raw_results:
                context["results"] = []
                return context
            
            processed_results = []
            
            for result in raw_results:
                if isinstance(result, dict):
                    # Normalize the result structure
                    normalized_result = self._normalize_result_item(result, context)
                    if normalized_result:
                        processed_results.append(normalized_result)
                elif result:
                    # Handle non-dict results
                    processed_results.append({
                        "data": result,
                        "url": context.get("url", ""),
                        "strategy": context.get("strategy_name", "unknown")
                    })
            
            context["results"] = processed_results
            logger.debug(f"Processed {len(processed_results)} results")
            return context
            
        except Exception as e:
            logger.error(f"Error processing results: {e}")
            context["results"] = []
            return context
    
    async def _apply_fallback_if_needed(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply fallback strategies if the primary strategy failed or produced poor results.
        
        Args:
            context: Current execution context
            
        Returns:
            Updated context with fallback applied if needed
        """
        try:
            # Check if fallback is needed
            results = context.get("results", [])
            error = context.get("error")
            needs_fallback = context.get("needs_fallback", False)
            attempts = context.get("attempts", 0)
            max_attempts = context.get("max_attempts", 3)
            
            # Record metrics
            if "metrics" not in context:
                context["metrics"] = {}
            
            fallback_times = context["metrics"].get("fallback_times", [])
            
            # Determine if fallback is needed
            should_fallback = (
                needs_fallback or 
                error or 
                len(results) == 0 or
                attempts < max_attempts
            )
            
            if should_fallback and attempts < max_attempts:
                context["attempts"] = attempts + 1
                fallback_times.append(time.time())
                context["metrics"]["fallback_times"] = fallback_times
                
                # Get fallback strategies
                fallback_strategies = getattr(self, 'fallback_strategies', ['multi_strategy'])
                used_strategies = context.get("fallbacks_used", [])
                
                # Find an unused fallback strategy
                for fallback_name in fallback_strategies:
                    if fallback_name not in used_strategies:
                        fallback_strategy = self.strategy_factory.get_strategy(fallback_name)
                        if fallback_strategy:
                            # Update context for fallback
                            context["strategy"] = fallback_strategy
                            context["strategy_name"] = fallback_name
                            context["fallbacks_used"] = used_strategies + [fallback_name]
                            
                            logger.info(f"Applying fallback strategy: {fallback_name}")
                            
                            # Re-execute with fallback strategy
                            context = await self._execute_search(context)
                            context = await self._process_results(context)
                            break
            
            return context
            
        except Exception as e:
            logger.error(f"Error applying fallback: {e}")
            return context
    
    def _normalize_result_item(self, item: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalize a single result item to a consistent format.
        
        Args:
            item: Raw result item
            context: Current execution context
            
        Returns:
            Normalized result item or None if invalid
        """
        try:
            if not isinstance(item, dict):
                return None
            
            # Extract basic fields
            normalized = {
                "url": item.get("url", context.get("url", "")),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "strategy": context.get("strategy_name", "unknown"),
                "timestamp": time.time()
            }
            
            # Copy other relevant fields
            for key, value in item.items():
                if key not in normalized and value is not None:
                    normalized[key] = value
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing result item: {e}")
            return None
    
    async def execute_search_pipeline(
        self, 
        query: str, 
        url: str, 
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete search pipeline.
        
        Args:
            query: Search query
            url: Target URL
            options: Additional options
            
        Returns:
            Pipeline execution result
        """
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Initialize context
            context = {
                "query": query,
                "url": url,
                "options": options or {},
                "operation_id": operation_id,
                "start_time": start_time,
                "attempts": 0,
                "max_attempts": 3,
                "fallbacks_used": [],
                "metrics": {}
            }
            
            # Execute pipeline steps
            for step in self.search_pipeline_steps:
                context = await step(context)
                if context.get("error"):
                    break
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Prepare final result
            result = {
                "success": not bool(context.get("error")),
                "results": context.get("results", []),
                "operation_id": operation_id,
                "execution_time": execution_time,
                "strategy": context.get("strategy_name", "unknown"),
                "metrics": context.get("metrics", {}),
                "attempts": context.get("attempts", 1)
            }
            
            if context.get("error"):
                result["error"] = context["error"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing search pipeline: {e}")
            return {
                "success": False,
                "results": [],
                "operation_id": operation_id,
                "execution_time": time.time() - start_time,
                "error": str(e),
                "strategy": "unknown",
                "metrics": {},
                "attempts": 1
            }
    
    def _initialize_pipeline_components(self) -> None:
        """Initialize pipeline components for data processing."""
        try:
            # Set up pipeline registry if not already done
            if not hasattr(self, 'pipeline_registry'):
                try:
                    from extraction.pipeline_registry import PipelineRegistry
                    self.pipeline_registry = PipelineRegistry()
                except ImportError:
                    logger.warning("PipelineRegistry not available")
                    self.pipeline_registry = None
            
            # Set up pipeline factory
            if not hasattr(self, 'pipeline_factory'):
                try:
                    from core.pipeline.factory import PipelineFactory
                    self.pipeline_factory = PipelineFactory()
                except ImportError:
                    logger.warning("PipelineFactory not available")
                    self.pipeline_factory = None
            
            # Initialize pipeline performance tracking
            if not hasattr(self, 'pipeline_performance'):
                self.pipeline_performance = {}
                
            if not hasattr(self, 'pipeline_performance_lock'):
                self.pipeline_performance_lock = asyncio.Lock()
            
            self._pipeline_components_initialized = True
            logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline components: {e}")
            self._pipeline_components_initialized = False

    def _process_structured_data(self, result: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process structured data results based on user intent.
        
        This method takes raw structured data and transforms it into a format
        that's useful for the user based on their search intent.
        
        Args:
            result: Raw result containing structured data
            user_intent: User's search intent with entity type and target properties
            
        Returns:
            Dict containing processed results focused on user-relevant content
        """
        try:
            logger.debug(f"Processing structured data with intent: {user_intent.get('entity_type', 'unknown')}")
            
            # Check if this is structured data (list of items)
            if not self._is_structured_data(result):
                # Not structured data, fall back to intent-based extraction
                return self._extract_intent_based_data(result, user_intent)
            
            # Extract the structured items
            items = []
            if isinstance(result.get('results'), list):
                items = result['results']
            elif isinstance(result.get('data'), list):
                items = result['data']
            elif isinstance(result.get('items'), list):
                items = result['items']
            else:
                # Try to find any list in the result
                for key, value in result.items():
                    if isinstance(value, list) and len(value) > 0:
                        items = value
                        break
            
            if not items:
                logger.warning("No structured items found in result")
                return {
                    "success": False,
                    "error": "No structured data found",
                    "results": []
                }
            
            # Get target properties from user intent
            target_properties = user_intent.get('properties', [])
            entity_type = user_intent.get('entity_type', 'item')
            
            # Process each item
            processed_items = []
            for item in items:
                processed_item = self._extract_relevant_fields(item, target_properties, entity_type)
                if processed_item:
                    processed_items.append(processed_item)
            
            # Return structured result
            return {
                "success": True,
                "results": processed_items,
                "total_count": len(processed_items),
                "entity_type": entity_type,
                "url": result.get("url", ""),
                "strategy": result.get("strategy", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Error processing structured data: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }

    def _extract_intent_based_data(self, result: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data from content based on user intent using AI assistance when available.
        
        This method analyzes raw content and extracts specific information
        that matches the user's search intent.
        
        Args:
            result: Raw result containing content to extract from
            user_intent: User's search intent with entity type and target properties
            
        Returns:
            Dict containing extracted data relevant to user intent
        """
        try:
            logger.debug(f"Extracting intent-based data for: {user_intent.get('target_item', 'unknown')}")
            
            # Get content from result
            content = result.get('content', '')
            url = result.get('url', '')
            
            # Check for error content first
            if self._is_error_content(content):
                return {
                    "success": False,
                    "error": "Access denied or blocked content",
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "url": url
                }
            
            # Try AI-assisted extraction if available
            ai_extracted = self._try_ai_extraction(content, user_intent)
            if ai_extracted and ai_extracted.get('success'):
                return ai_extracted
            
            # Fall back to rule-based extraction
            return self._rule_based_extraction(content, user_intent, url)
            
        except Exception as e:
            logger.error(f"Error in intent-based data extraction: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": result.get('content', '')[:200] + "..." if result.get('content') else "No content"
            }

    # Additional helper methods for the main extraction methods
    
    def _is_structured_data(self, result: Dict[str, Any]) -> bool:
        """Check if the result contains structured data (list of items)."""
        try:
            # Check for common structured data patterns
            for key in ['results', 'data', 'items']:
                value = result.get(key)
                if isinstance(value, list) and len(value) > 0:
                    # Check if items look structured (have multiple fields)
                    first_item = value[0]
                    if isinstance(first_item, dict) and len(first_item) > 2:
                        return True
            return False
        except Exception:
            return False

    def _extract_relevant_fields(self, item: Dict[str, Any], target_properties: List[str], entity_type: str) -> Dict[str, Any]:
        """Extract relevant fields from a structured item based on target properties."""
        try:
            extracted = {}
            
            # Map common field variations
            field_mappings = {
                'title': ['title', 'name', 'heading', 'label'],
                'price': ['price', 'cost', 'amount', 'value'],
                'address': ['address', 'location', 'place'],
                'phone': ['phone', 'telephone', 'contact', 'number'],
                'email': ['email', 'mail', 'contact_email'],
                'description': ['description', 'desc', 'summary', 'details'],
                'bedrooms': ['bedrooms', 'beds', 'bedroom_count'],
                'bathrooms': ['bathrooms', 'baths', 'bathroom_count'],
                'sqft': ['sqft', 'square_feet', 'area', 'size']
            }
            
            # Extract each target property
            for prop in target_properties:
                value = None
                
                # Try direct match first
                if prop in item:
                    value = item[prop]
                else:
                    # Try mapped variations
                    possible_fields = field_mappings.get(prop, [prop])
                    for field in possible_fields:
                        if field in item:
                            value = item[field]
                            break
                
                if value is not None and str(value).strip():
                    extracted[prop] = value
            
            # Always include some basic fields if available
            for basic_field in ['title', 'url', 'id']:
                if basic_field in item and basic_field not in extracted:
                    extracted[basic_field] = item[basic_field]
            
            return extracted if extracted else None
            
        except Exception as e:
            logger.error(f"Error extracting relevant fields: {e}")
            return None

    def _is_error_content(self, content: str) -> bool:
        """Check if content indicates an error or blocked access."""
        if not content:
            return False
        
        content_lower = content.lower()
        error_indicators = [
            'access denied',
            'verify you are human',
            'blocked',
            'forbidden',
            'captcha',
            'please verify',
            'security check',
            'not authorized'
        ]
        
        return any(indicator in content_lower for indicator in error_indicators)

    def _try_ai_extraction(self, content: str, user_intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try AI-assisted extraction if AI service is available."""
        try:
            # Check if AI service is available and initialized
            ai_service = self.get_ai_service()
            if not ai_service or not getattr(ai_service, '_initialized', False):
                return None
            
            # For now, return None to fall back to rule-based extraction
            # TODO: Implement AI extraction when AI service is fully operational
            return None
            
        except Exception as e:
            logger.debug(f"AI extraction not available: {e}")
            return None

    def _rule_based_extraction(self, content: str, user_intent: Dict[str, Any], url: str = "") -> Dict[str, Any]:
        """Perform rule-based extraction from content."""
        try:
            from bs4 import BeautifulSoup
            import re
            
            extracted = {
                "success": True,
                "url": url,
                "extraction_method": "rule_based"
            }
            
            # Parse HTML if content looks like HTML
            if '<' in content and '>' in content:
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract title
                title_elem = soup.find(['h1', 'h2', 'h3', 'title'])
                if title_elem:
                    extracted['title'] = title_elem.get_text().strip()
                
                # Extract price using regex
                price_pattern = r'\$[\d,]+(?:\.\d{2})?'
                price_match = re.search(price_pattern, content)
                if price_match:
                    extracted['price'] = price_match.group()
                
                # Extract phone numbers
                phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
                phone_match = re.search(phone_pattern, content)
                if phone_match:
                    extracted['phone'] = phone_match.group()
                
                # Extract text content
                text_content = soup.get_text()
                if text_content and len(text_content.strip()) > 50:
                    extracted['content'] = text_content.strip()[:500]  # Limit content length
            else:
                # Plain text content
                extracted['content'] = content[:500] if content else ""
            
            # Add any specific extractions based on entity type
            entity_type = user_intent.get('entity_type', '')
            if entity_type == 'property':
                extracted.update(self._extract_property_specific_data(content))
            
            return extracted
            
        except Exception as e:
            logger.error(f"Error in rule-based extraction: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": content[:200] + "..." if content else "No content"
            }

    def _extract_property_specific_data(self, content: str) -> Dict[str, Any]:
        """Extract property-specific data like bedrooms, bathrooms, etc."""
        import re
        
        property_data = {}
        
        try:
            # Extract bedrooms
            bed_pattern = r'(\d+)\s*(?:bed|bedroom|br)\b'
            bed_match = re.search(bed_pattern, content, re.IGNORECASE)
            if bed_match:
                property_data['bedrooms'] = int(bed_match.group(1))
            
            # Extract bathrooms
            bath_pattern = r'(\d+(?:\.\d+)?)\s*(?:bath|bathroom|ba)\b'
            bath_match = re.search(bath_pattern, content, re.IGNORECASE)
            if bath_match:
                property_data['bathrooms'] = float(bath_match.group(1))
            
            # Extract square footage
            sqft_pattern = r'([\d,]+)\s*(?:sq\.?\s*ft|square\s*feet|sqft)\b'
            sqft_match = re.search(sqft_pattern, content, re.IGNORECASE)
            if sqft_match:
                sqft_str = sqft_match.group(1).replace(',', '')
                property_data['sqft'] = int(sqft_str)
            
        except Exception as e:
            logger.debug(f"Error extracting property data: {e}")
        
        return property_data

    # =============================================================================
    # EXTRACTION PIPELINE METHODS
    # =============================================================================

    async def execute_extraction_pipeline(
        self, 
        url: Optional[str] = None,
        html: Optional[str] = None,
        pipeline_type: str = "default_extraction",
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute an extraction pipeline with the given input.
        Enhanced with Phase 5 monitoring and validation from gameplan.
        
        Args:
            url: URL to extract from (optional if html is provided)
            html: Pre-fetched HTML content (optional if url is provided)
            pipeline_type: Type of pipeline to use
            options: Optional extraction and execution options
            
        Returns:
            Extraction results with success flag, extracted data, and metadata
        """
        start_time = time.time()
        execution_id = f"{pipeline_type}_{int(start_time)}"
        
        try:
            # Ensure we have either URL or HTML
            if not url and not html:
                return {
                    "success": False,
                    "error": "Either url or html must be provided",
                    "execution_time": 0.0,
                    "execution_id": execution_id
                }
            
            # Log pipeline start (Phase 5 enhancement)
            self._log_extraction_attempt(url or "html_content", pipeline_type, options)
            
            # Initialize pipeline components if needed
            if not self._pipeline_components_initialized:
                self._initialize_pipeline_components()
            
            # Use the standalone pipeline executor for now
            try:
                from extraction.pipeline_executor import execute_extraction_pipeline
                
                # Prepare execution options
                exec_options = options or {}
                
                # If we have HTML but no URL, we need to handle this case
                if html and not url:
                    url = exec_options.get("url", "about:blank")
                
                # Execute pipeline with monitoring (Phase 5 enhancement)
                result = await self._execute_monitored_pipeline(
                    pipeline_type, url, html, exec_options, execution_id
                )
                
                # Validate results (Phase 5 enhancement)
                is_valid, validation_issues = self._validate_extraction_result(result)
                if not is_valid:
                    self._log_fallback_trigger(
                        url, pipeline_type, "fallback_pipeline", 
                        f"Validation failed: {', '.join(validation_issues)}"
                    )
                    result = await self._execute_fallback_pipeline(url, html, exec_options, execution_id)
                
                # Ensure result has required structure
                if not isinstance(result, dict):
                    result = {"success": False, "error": "Invalid pipeline result"}
                
                # Add execution metadata
                execution_time = time.time() - start_time
                result["execution_time"] = execution_time
                result["execution_id"] = execution_id
                result["pipeline_type"] = pipeline_type
                
                # Ensure success flag exists
                if "success" not in result:
                    result["success"] = "error" not in result
                
                # Log final result (Phase 5 enhancement)
                result_size = len(result.get('data', [])) if isinstance(result.get('data'), list) else (1 if result.get('data') else 0)
                self._log_extraction_result(
                    url, result.get('success', False), result_size, 
                    pipeline_type, result.get('error'), execution_time
                )
                
                return result
                
            except ImportError as e:
                logger.warning(f"Pipeline executor not available, using fallback: {e}")
                return await self._execute_extraction_fallback(url, html, options)
                
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error executing extraction pipeline: {e}"
            logger.error(error_msg)
            
            # Log error result (Phase 5 enhancement)
            self._log_extraction_result(
                url or "html_content", False, 0, pipeline_type, error_msg, execution_time
            )
            
            return {
                "success": False,
                "error": error_msg,
                "execution_time": execution_time,
                "execution_id": execution_id,
                "pipeline_type": pipeline_type
            }

    async def _execute_extraction_fallback(
        self, 
        url: Optional[str], 
        html: Optional[str], 
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fallback extraction when pipeline executor is not available.
        
        Args:
            url: URL to extract from
            html: HTML content
            options: Extraction options
            
        Returns:
            Basic extraction results
        """
        try:
            # Basic content extraction using BeautifulSoup
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract basic content
                title = soup.find('title')
                title_text = title.get_text().strip() if title else ""
                
                # Extract price information
                price_selectors = ['.price', '.cost', '[data-price]', '.amount']
                price = ""
                for selector in price_selectors:
                    price_elem = soup.select_one(selector)
                    if price_elem:
                        price = price_elem.get_text().strip()
                        break
                
                # Extract description
                desc_selectors = ['.description', '.summary', '.content', 'p']
                description = ""
                for selector in desc_selectors:
                    desc_elem = soup.select_one(selector)
                    if desc_elem:
                        description = desc_elem.get_text().strip()[:500]
                        break
                
                return {
                    "success": True,
                    "results": [{
                        "title": title_text,
                        "price": price,
                        "description": description,
                        "url": url
                    }],
                    "content_type": "fallback",
                    "extraction_method": "fallback_beautifulsoup"
                }
            else:
                return {
                    "success": False,
                    "error": "No HTML content provided for fallback extraction"
                }
                
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return {
                "success": False,
                "error": f"Fallback extraction failed: {str(e)}"
            }

    async def select_extraction_pipeline(
        self,
        url: str,
        content_type: Optional[str] = None,
        content_sample: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enhanced pipeline selection with multi-factor analysis.
        
        This method implements Phase 3 of the gameplan: Pipeline Orchestration Improvement
        
        Args:
            url: URL being processed
            content_type: Expected content type
            content_sample: Sample of content for analysis
            options: Selection options
            
        Returns:
            Pipeline name/type to use
        """
        try:
            # Priority 1: Explicit content type
            if content_type:
                pipeline_map = {
                    'sitemap': 'sitemap_extraction',
                    'feed': 'feed_extraction', 
                    'api': 'api_extraction',
                    'product': 'product_extraction',
                    'listing': 'listing_extraction',
                    'article': 'article_extraction',
                    'news': 'article_extraction',
                    'blog': 'article_extraction',
                    'ecommerce': 'product_extraction',
                    'real_estate': 'listing_extraction'
                }
                return pipeline_map.get(content_type, 'html_extraction')
            
            # Priority 2: Content analysis
            if content_sample:
                detected_type = await self._analyze_content_type(content_sample, url)
                if detected_type:
                    return f"{detected_type}_extraction"
            
            # Priority 3: URL pattern analysis
            url_patterns = {
                'sitemap': ['sitemap', 'site-map'],
                'feed': ['rss', 'feed', 'atom'],
                'api': ['api/', '/v1/', '/v2/', 'rest/', 'graphql'],
                'product': ['product/', 'item/', '/p/', 'shop/', 'store/'],
                'listing': ['search', 'listing', 'directory', 'browse', 'category'],
                'article': ['article/', 'blog/', 'news/', 'post/']
            }
            
            url_lower = url.lower()
            for content_type, patterns in url_patterns.items():
                if any(pattern in url_lower for pattern in patterns):
                    return f"{content_type}_extraction"
            
            # Default fallback
            return "html_extraction"
            
        except Exception as e:
            logger.error(f"Error selecting extraction pipeline: {e}")
            return "html_extraction"

    async def _analyze_content_type(self, content: str, url: str) -> Optional[str]:
        """
        Enhanced content type analysis with multi-factor analysis (Phase 4 implementation).
        
        This method implements sophisticated multi-factor analysis including:
        - Structured data analysis
        - DOM patterns analysis
        - URL patterns analysis
        - Text patterns analysis
        - Form patterns analysis
        
        Args:
            content: HTML or text content
            url: Source URL
            
        Returns:
            Detected content type with highest confidence
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Multi-factor analysis
            factors = [
                self._analyze_structured_data(soup),
                self._analyze_dom_patterns(soup),
                self._analyze_url_patterns(url),
                self._analyze_text_patterns(soup.get_text()),
                self._analyze_form_patterns(soup)
            ]
            
            # Combine analysis results
            content_types = {}
            for factor in factors:
                if factor:  # Skip None results
                    for content_type, confidence in factor.items():
                        content_types[content_type] = content_types.get(content_type, 0) + confidence
            
            # Select highest confidence type
            if content_types:
                best_type = max(content_types.items(), key=lambda x: x[1])
                # Only return if confidence is above threshold (adjusted for practical use)
                if best_type[1] >= 1.0:  # At least 1 strong factor or 2 weaker factors
                    logger.debug(f"Content type detected: {best_type[0]} (confidence: {best_type[1]:.2f})")
                    return best_type[0]
            
            logger.debug("No content type detected with sufficient confidence")
            return None
            
        except Exception as e:
            logger.debug(f"Error in enhanced content type analysis: {e}")
            return None

    def _analyze_structured_data(self, soup: BeautifulSoup) -> Dict[str, float]:
        """Analyze structured data (JSON-LD, microdata, RDFa) for content type hints."""
        try:
            type_scores = {}
            
            # JSON-LD analysis
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        schema_type = data.get('@type', '').lower()
                        if 'product' in schema_type:
                            type_scores['product'] = type_scores.get('product', 0) + 1.5
                        elif 'article' in schema_type or 'newsarticle' in schema_type:
                            type_scores['article'] = type_scores.get('article', 0) + 1.5
                        elif 'itemlist' in schema_type or 'realestate' in schema_type:
                            type_scores['listing'] = type_scores.get('listing', 0) + 1.5
                        elif 'organization' in schema_type or 'website' in schema_type:
                            type_scores['company'] = type_scores.get('company', 0) + 1.0
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            # Microdata analysis
            microdata_items = soup.find_all(attrs={'itemtype': True})
            for item in microdata_items:
                itemtype = item.get('itemtype', '').lower()
                if 'product' in itemtype:
                    type_scores['product'] = type_scores.get('product', 0) + 1.0
                elif 'article' in itemtype:
                    type_scores['article'] = type_scores.get('article', 0) + 1.0
                elif 'offer' in itemtype:
                    type_scores['product'] = type_scores.get('product', 0) + 0.8
            
            # Open Graph analysis
            og_type = soup.find('meta', property='og:type')
            if og_type:
                og_content = og_type.get('content', '').lower()
                if 'product' in og_content:
                    type_scores['product'] = type_scores.get('product', 0) + 1.0
                elif 'article' in og_content:
                    type_scores['article'] = type_scores.get('article', 0) + 1.0
            
            return type_scores
            
        except Exception as e:
            logger.debug(f"Error in structured data analysis: {e}")
            return {}

    def _analyze_dom_patterns(self, soup: BeautifulSoup) -> Dict[str, float]:
        """Analyze DOM patterns and CSS classes for content type indicators."""
        try:
            type_scores = {}
            
            # Product page patterns
            product_selectors = [
                '.price', '.product-price', '.cost', '.amount',
                '.add-to-cart', '.buy-now', '.purchase',
                '.product-details', '.product-info', '.item-details',
                '.sku', '.model', '.brand'
            ]
            product_score = sum(1 for selector in product_selectors if soup.select(selector))
            if product_score > 0:
                type_scores['product'] = product_score * 0.3
            
            # Article page patterns  
            article_selectors = [
                'article', '.article', '.post', '.content',
                '.author', '.byline', '.published', '.date',
                '.article-body', '.post-content', '.entry-content'
            ]
            article_score = sum(1 for selector in article_selectors if soup.select(selector))
            if article_score > 0:
                type_scores['article'] = article_score * 0.3
            
            # Listing page patterns
            listing_selectors = [
                '.listing', '.results', '.items', '.grid',
                '.property', '.listing-item', '.search-result',
                '.card', '.tile', '.item-card'
            ]
            listing_score = sum(1 for selector in listing_selectors if soup.select(selector))
            if listing_score > 0:
                type_scores['listing'] = listing_score * 0.3
            
            # Company/homepage patterns
            company_selectors = [
                '.hero', '.banner', '.services', '.about',
                '.team', '.contact', '.footer', 'nav'
            ]
            company_score = sum(1 for selector in company_selectors if soup.select(selector))
            if company_score >= 3:  # Typical company page has multiple of these
                type_scores['company'] = company_score * 0.2
            
            # Form patterns
            forms = soup.find_all('form')
            if len(forms) > 2:  # Multiple forms suggest interactive/application page
                type_scores['form'] = len(forms) * 0.4
            
            return type_scores
            
        except Exception as e:
            logger.debug(f"Error in DOM pattern analysis: {e}")
            return {}

    def _analyze_url_patterns(self, url: str) -> Dict[str, float]:
        """Analyze URL patterns for content type hints."""
        try:
            type_scores = {}
            url_lower = url.lower()
            
            # Product URL patterns
            product_patterns = [
                '/product/', '/item/', '/p/', '/shop/',
                '/buy/', '/catalog/', '/store/'
            ]
            for pattern in product_patterns:
                if pattern in url_lower:
                    type_scores['product'] = type_scores.get('product', 0) + 0.8
            
            # Article URL patterns
            article_patterns = [
                '/article/', '/blog/', '/post/', '/news/',
                '/story/', '/read/', '/content/'
            ]
            for pattern in article_patterns:
                if pattern in url_lower:
                    type_scores['article'] = type_scores.get('article', 0) + 0.8
            
            # Listing URL patterns
            listing_patterns = [
                '/search/', '/results/', '/listings/', '/browse/',
                '/category/', '/properties/', '/homes/', '/cars/'
            ]
            for pattern in listing_patterns:
                if pattern in url_lower:
                    type_scores['listing'] = type_scores.get('listing', 0) + 0.8
            
            # API patterns
            api_patterns = ['/api/', '/rest/', '/graphql/', '.json', '.xml']
            for pattern in api_patterns:
                if pattern in url_lower:
                    type_scores['api'] = type_scores.get('api', 0) + 1.0
            
            # Sitemap patterns
            if 'sitemap' in url_lower:
                type_scores['sitemap'] = 2.0
            
            # Feed patterns
            feed_patterns = ['/feed/', '/rss/', '/atom/', '.rss', '.xml']
            for pattern in feed_patterns:
                if pattern in url_lower:
                    type_scores['feed'] = type_scores.get('feed', 0) + 1.0
            
            return type_scores
            
        except Exception as e:
            logger.debug(f"Error in URL pattern analysis: {e}")
            return {}

    def _analyze_text_patterns(self, text_content: str) -> Dict[str, float]:
        """Analyze text content patterns for content type indicators."""
        try:
            type_scores = {}
            text_lower = text_content.lower()
            
            # Product indicators
            product_keywords = [
                'price', 'buy now', 'add to cart', 'purchase', 'order',
                'shipping', 'delivery', 'warranty', 'specifications',
                'features', 'reviews', 'rating', 'stars', 'sku', 'model'
            ]
            product_count = sum(1 for keyword in product_keywords if keyword in text_lower)
            if product_count >= 3:
                type_scores['product'] = product_count * 0.2
            
            # Article indicators
            article_keywords = [
                'published', 'author', 'written by', 'posted',
                'article', 'story', 'news', 'opinion', 'analysis',
                'paragraph', 'conclusion', 'introduction'
            ]
            article_count = sum(1 for keyword in article_keywords if keyword in text_lower)
            if article_count >= 2:
                type_scores['article'] = article_count * 0.3
            
            # Listing indicators
            listing_keywords = [
                'results found', 'items', 'showing', 'page',
                'filter', 'sort by', 'view all', 'load more',
                'per page', 'total', 'bedrooms', 'bathrooms'
            ]
            listing_count = sum(1 for keyword in listing_keywords if keyword in text_lower)
            if listing_count >= 2:
                type_scores['listing'] = listing_count * 0.3
            
            # Company/About indicators
            company_keywords = [
                'about us', 'our mission', 'our team', 'contact us',
                'services', 'solutions', 'company', 'founded',
                'headquarters', 'employees', 'clients'
            ]
            company_count = sum(1 for keyword in company_keywords if keyword in text_lower)
            if company_count >= 2:
                type_scores['company'] = company_count * 0.2
            
            return type_scores
            
        except Exception as e:
            logger.debug(f"Error in text pattern analysis: {e}")
            return {}

    def _analyze_form_patterns(self, soup: BeautifulSoup) -> Dict[str, float]:
        """Analyze form patterns for content type indicators."""
        try:
            type_scores = {}
            
            forms = soup.find_all('form')
            if not forms:
                return type_scores
            
            for form in forms:
                # Search form indicators
                search_inputs = form.find_all('input', attrs={'type': ['search', 'text']})
                search_keywords = ['search', 'query', 'q', 'keyword']
                
                for inp in search_inputs:
                    name = inp.get('name', '').lower()
                    placeholder = inp.get('placeholder', '').lower()
                    if any(keyword in name or keyword in placeholder for keyword in search_keywords):
                        type_scores['search'] = type_scores.get('search', 0) + 0.5
                
                # Contact form indicators
                contact_inputs = form.find_all('input', attrs={'type': ['email', 'tel']})
                contact_keywords = ['email', 'phone', 'message', 'contact']
                
                for inp in contact_inputs:
                    name = inp.get('name', '').lower()
                    if any(keyword in name for keyword in contact_keywords):
                        type_scores['contact'] = type_scores.get('contact', 0) + 0.3
                
                # Login form indicators
                login_inputs = form.find_all('input', attrs={'type': ['password']})
                if login_inputs:
                    type_scores['login'] = type_scores.get('login', 0) + 0.8
                
                # E-commerce form indicators
                ecommerce_inputs = form.find_all('input', {'name': re.compile(r'(quantity|cart|price|checkout)', re.I)})
                if ecommerce_inputs:
                    type_scores['product'] = type_scores.get('product', 0) + 0.7
            
            return type_scores
            
        except Exception as e:
            logger.debug(f"Error in form pattern analysis: {e}")
            return {}

    async def scrape_with_pipeline(
        self,
        url: str,
        query: Optional[str] = None,
        content_type: Optional[str] = None,
        extraction_options: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Scrape a URL using the extraction pipeline framework.
        
        Args:
            url: URL to scrape
            query: Query or extraction intent
            content_type: Expected content type
            extraction_options: Extraction-specific options
            options: General scraping options
            
        Returns:
            Scraping results with extracted data
        """
        start_time = time.time()
        
        try:
            # Merge options
            options = options or {}
            extraction_options = extraction_options or {}
            all_options = {**options, **extraction_options}
            
            # Include query in options if provided
            if query:
                all_options['query'] = query
            
            # Get HTML content if provided, otherwise fetch it
            html_content = all_options.get("html_content")
            
            if not html_content:
                # Fetch HTML content directly to avoid infinite recursion
                html_content = await self._fetch_html_content(url, all_options)
                if not html_content:
                    return {
                        "success": False,
                        "error": "Failed to fetch HTML content",
                        "execution_time": time.time() - start_time
                    }
            
            # Select appropriate pipeline
            pipeline_type = await self.select_extraction_pipeline(
                url=url,
                content_type=content_type,
                content_sample=html_content[:1000] if html_content else None,
                options=all_options
            )
            
            # Execute extraction pipeline
            extraction_result = await self.execute_extraction_pipeline(
                url=url,
                html=html_content,
                pipeline_type=pipeline_type,
                options=all_options
            )
            
            # Add metadata
            extraction_result["pipeline_used"] = pipeline_type
            extraction_result["total_execution_time"] = time.time() - start_time
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"Error in scrape_with_pipeline: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }

    async def _extract_from_final_url(self, url: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract content from a final target URL using various strategies.
        
        This method coordinates the extraction process by:
        1. Trying Crawl4AI strategy first if schema is available
        2. Falling back to template-based extraction
        3. Using cascading fallback strategies if needed
        
        Args:
            url: The target URL to extract content from
            options: Extraction options including schema, strategy preferences, etc.
            
        Returns:
            Dict containing success status, extracted data, and metadata
        """
        if options is None:
            options = {}
            
        try:
            # Log extraction attempt
            logger.info(f"Starting content extraction for URL: {url}")
            
            # Strategy 1: Use Crawl4AI if schema is provided and AI extraction is enabled
            if options.get("use_universal_crawl4ai_strategy") and options.get("schema"):
                logger.info(f"Attempting Crawl4AI extraction for {url}")
                result = await self._execute_crawl4ai_strategy(url, None, options.get("schema"), options)
                if result.get('success') and result.get('data'):
                    logger.info(f"Crawl4AI extraction successful for {url}, got {len(result.get('data', []))} items")
                    return result
                else:
                    logger.warning(f"Crawl4AI extraction failed for {url}: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
                    return await self._extract_content_fallback(url, options)
                    
            # Strategy 2: Fetch HTML content and apply extraction templates
            html_content = await self._fetch_html_content(url, options)
            if not html_content:
                return {"success": False, "error": "Failed to fetch HTML content", "data": []}
            
            # Try template-based extraction
            logger.info(f"Attempting template-based extraction for {url}")
            template_result = await self._apply_extraction_template(html_content, url, options)
            if template_result.get('success') and template_result.get('data'):
                logger.info(f"Template extraction successful for {url}")
                return template_result
            
            # Strategy 3: Fallback to cascading extraction methods
            logger.info(f"Attempting fallback extraction methods for {url}")
            fallback_result = await self._apply_fallback_extraction(html_content, url, options)
            return fallback_result
            
        except Exception as e:
            logger.error(f"Error in _extract_from_final_url for {url}: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e), "data": []}

    async def _fetch_html_content(self, url: str, options: Dict[str, Any] = None) -> Optional[str]:
        """
        Fetch HTML content from a URL with static and dynamic content support.
        
        Args:
            url: The URL to fetch content from
            options: Fetch options including timeout, headers, etc.
            
        Returns:
            HTML content as string or None if failed
        """
        if options is None:
            options = {}
            
        try:
            # Try static fetch first using fetch_html utility
            logger.debug(f"Fetching HTML content for {url}")
            html_content = await fetch_html(url, timeout=options.get('timeout', 30))
            
            if html_content and len(html_content.strip()) > 100:
                # Check if content seems complete (not just a loading page)
                if not self._is_content_sparse(html_content):
                    return html_content
                elif options.get('allow_dynamic_content_fetching', False):
                    logger.info(f"Content appears sparse, attempting dynamic fetch for {url}")
                    return await self._fetch_dynamic_content(url)
                else:
                    return html_content
            else:
                logger.warning(f"Fetched content too short or empty for {url}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching HTML content for {url}: {e}")
            return None

    def _is_content_sparse(self, html_content: str) -> bool:
        """
        Check if HTML content appears sparse or is likely a loading page.
        
        Args:
            html_content: The HTML content to analyze
            
        Returns:
            True if content appears sparse, False otherwise
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text().strip()
            
            # Check various indicators of sparse content
            sparse_indicators = [
                len(text_content) < 500,  # Very little text
                'loading' in text_content.lower(),
                'javascript' in text_content.lower() and len(text_content) < 1000,
                text_content.count('\n') < 10,  # Very few line breaks
                len(soup.find_all(['p', 'div', 'article', 'section'])) < 5  # Few content elements
            ]
            
            return sum(sparse_indicators) >= 2  # If 2+ indicators, consider sparse
        except Exception:
            return False

    async def _fetch_dynamic_content(self, url: str) -> Optional[str]:
        """
        Fetch content from a URL that requires JavaScript execution.
        
        Args:
            url: The URL to fetch dynamic content from
            
        Returns:
            HTML content after JavaScript execution or None if failed
        """
        try:
            # Try using playwright if available
            try:
                from playwright.async_api import async_playwright
                
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    try:
                        page = await browser.new_page()
                        await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                        await page.wait_for_timeout(3000)  # Wait for dynamic content
                        content = await page.content()
                        return content
                    finally:
                        await browser.close()
                        
            except ImportError:
                logger.warning("Playwright not available, cannot fetch dynamic content")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching dynamic content for {url}: {e}")
            return None

    async def _apply_extraction_template(self, html_content: str, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply extraction templates based on content type and URL patterns.
        
        Args:
            html_content: The HTML content to extract from
            url: The source URL
            options: Extraction options
            
        Returns:
            Dict containing extraction results
        """
        try:
            # Detect content type
            content_type = await self._detect_content_type(html_content, url)
            logger.debug(f"Detected content type: {content_type} for {url}")
            
            # Apply appropriate extraction template
            if content_type == 'product':
                return await self._extract_product_content(html_content, url, options)
            elif content_type == 'article':
                return await self._extract_article_content(html_content, url, options)
            elif content_type == 'listing':
                return await self._extract_listing_content(html_content, url, options)
            else:
                return await self._extract_generic_content(html_content, url, options)
                
        except Exception as e:
            logger.error(f"Error applying extraction template for {url}: {e}")
            return {"success": False, "error": str(e), "data": []}

    async def _detect_content_type(self, html_content: str, url: str) -> str:
        """
        Detect the content type of a page based on HTML content and URL patterns.
        
        Args:
            html_content: The HTML content to analyze
            url: The source URL
            
        Returns:
            Detected content type as string
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            url_lower = url.lower()
            
            # URL pattern analysis
            if any(pattern in url_lower for pattern in ['product/', 'item/', '/p/', 'shop/', 'store/']):
                return 'product'
            elif any(pattern in url_lower for pattern in ['article/', 'blog/', '/post/', '/news/']):
                return 'article'
            elif any(pattern in url_lower for pattern in ['search', 'listing', 'directory', 'browse', 'category']):
                return 'listing'
            
            # Content analysis
            if soup.find(['article', '.article', '.post', '.blog-post']) or \
               soup.find(['time', '.date', '.publish-date', '.author']):
                return 'article'
            
            # Check for product indicators
            if soup.find(['.price', '.buy-button', '.add-to-cart', '.product']) or \
               soup.find('meta', {'property': 'product:price'}):
                return 'product'
            
            # Check for listing indicators
            if len(soup.select('.search-result, .listing-item, .result-item')) >= 2:
                return 'listing'
            
            # Check for table/data indicators
            if soup.find('table') and len(soup.find_all('tr')) > 3:
                return 'data_table'
            
            return 'generic'
        except Exception:
            return 'unknown'

    async def _extract_product_content(self, html_content: str, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Extract product-specific content from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract product information
            product_data = {
                'url': url,
                'title': self._extract_title_from_soup(soup),
                'price': self._extract_product_price(soup),
                'description': self._extract_product_description(soup),
                'images': self._extract_product_images(soup),
                'rating': self._extract_product_rating(soup),
                'availability': self._extract_product_availability(soup)
            }
            
            # Remove None values
            product_data = {k: v for k, v in product_data.items() if v is not None}
            
            if len(product_data) > 2:  # More than just URL and title
                return {"success": True, "data": [product_data]}
            else:
                return {"success": False, "error": "Insufficient product data extracted", "data": []}
                
        except Exception as e:
            logger.error(f"Error extracting product content: {e}")
            return {"success": False, "error": str(e), "data": []}

    async def _extract_article_content(self, html_content: str, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Extract article-specific content from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract article information
            article_data = {
                'url': url,
                'title': self._extract_title_from_soup(soup),
                'content': self._extract_article_text(soup),
                'author': self._extract_article_author(soup),
                'publish_date': self._extract_article_date(soup),
                'summary': self._extract_article_summary(soup)
            }
            
            # Remove None values
            article_data = {k: v for k, v in article_data.items() if v is not None}
            
            if len(article_data) > 2:  # More than just URL and title
                return {"success": True, "data": [article_data]}
            else:
                return {"success": False, "error": "Insufficient article data extracted", "data": []}
                
        except Exception as e:
            logger.error(f"Error extracting article content: {e}")
            return {"success": False, "error": str(e), "data": []}

    async def _extract_listing_content(self, html_content: str, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Extract listing/search results content from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find repeating patterns that might be listings
            items = []
            
            # Common listing selectors
            listing_selectors = [
                '.item', '.listing', '.result', '.card', '.entry',
                '[class*="item"]', '[class*="listing"]', '[class*="result"]'
            ]
            
            for selector in listing_selectors:
                elements = soup.select(selector)
                if len(elements) > 1:  # Must have multiple items to be a listing
                    for elem in elements[:20]:  # Limit to 20 items
                        item_data = {"type": "listing_item", "url": url}
                        
                        # Extract title
                        title_elem = elem.find(['h1', 'h2', 'h3', 'h4', 'a'])
                        if title_elem:
                            item_data["title"] = title_elem.get_text(strip=True)
                        
                        # Extract link if available
                        link_elem = elem.find('a')
                        if link_elem and link_elem.get('href'):
                            item_data["link"] = urljoin(url, link_elem['href'])
                        
                        # Extract content
                        content = elem.get_text(strip=True)
                        if content:
                            item_data["content"] = content[:300]
                        
                        if len(item_data) > 2:  # More than just type and url
                            items.append(item_data)
                    
                    if items:
                        break  # Found listings, no need to try other selectors
            
            return {
                "success": len(items) > 0,
                "data": items,
                "metadata": {"extraction_method": "listing_extraction", "items_count": len(items)}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Listing extraction failed: {str(e)}", "data": []}
    
    async def _process_article_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process article/blog post extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch article content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            article_data = {
                "url": url,
                "type": "article"
            }
            
            # Extract title
            title_selectors = ['h1', '.title', '.headline', 'header h1', 'article h1']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text(strip=True):
                    article_data["title"] = title_elem.get_text(strip=True)
                    break
            
            # Extract article content
            content_selectors = ['article', '.content', '.post-content', '.entry-content', 'main']
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Clean up content
                    for script in content_elem(['script', 'style']):
                        script.decompose()
                    content_text = content_elem.get_text(strip=True)
                    if len(content_text) > 100:  # Minimum content length
                        article_data["content"] = content_text[:2000]  # Limit content length
                        break
            
            # Extract author if available
            author_selectors = ['.author', '.byline', '[rel="author"]']
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem and author_elem.get_text(strip=True):
                    article_data["author"] = author_elem.get_text(strip=True)
                    break
            
            return {
                "success": len(article_data) > 2,  # More than just url and type
                "data": [article_data],
                "metadata": {"extraction_method": "article_extraction", "items_count": 1}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Article extraction failed: {str(e)}", "data": []}
    
    async def _process_html_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process general HTML extraction as fallback."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch HTML content", "data": []}
            
            # Use existing content extraction logic
            return await self._extract_content_fallback_async(url, html, options)
            
        except Exception as e:
            return {"success": False, "error": f"HTML extraction failed: {str(e)}", "data": []}
    
    async def _extract_content_fallback_async(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
                """
                High-level async content fallback extraction method.
                
                Args:
                    url: URL being scraped
                    html: HTML content to parse
                    options: Extraction options
                    
                Returns:
                    Extraction result dictionary
                """
                try:
                    logger.info(f"Applying content fallback extraction for {url}")
                    
                    # Use the existing fallback extraction method
                    result = await self._apply_fallback_extraction(url, html, options.get('fallback_reason', 'content_extraction_failed'))
                    
                    if result and result.get('success'):
                        logger.info(f"Content fallback extraction successful for {url}")
                        return result
                    else:
                        logger.warning(f"Content fallback extraction failed for {url}")
                        return {
                            "success": False,
                            "error": "Content fallback extraction failed",
                            "data": [],
                            "metadata": {
                                "url": url,
                                "extraction_method": "content_fallback_failed"
                            }
                        }
                        
                except Exception as e:
                    logger.error(f"Error in content fallback extraction for {url}: {e}")
                    return {
                        "success": False,
                        "error": f"Content fallback extraction error: {str(e)}",
                        "data": [],
                        "metadata": {
                            "url": url,
                            "extraction_method": "content_fallback_error",
                            "error_details": str(e)
                        }
                    }
        
            # ============================================================================
    
    # ============================================================================
    # PHASE 5: ERROR HANDLING & DEBUGGING ENHANCEMENTS
    # ============================================================================
    
    def _validate_extraction_result(self, result: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate extraction results and return issues found.
        
        Based on gameplan Phase 5 specifications for comprehensive result validation.
        """
        issues = []
        
        try:
            # Check basic structure
            if not isinstance(result, dict):
                issues.append("Result is not a dictionary")
                return False, issues
            
            # Check required fields
            required_fields = ['success', 'data']
            for field in required_fields:
                if field not in result:
                    issues.append(f"Missing required field: {field}")
            
            # Check success flag consistency
            if result.get('success') == True:
                if not result.get('data'):
                    issues.append("Success=True but no data provided")
                elif isinstance(result['data'], list) and len(result['data']) == 0:
                    issues.append("Success=True but data list is empty")
            
            # Check data quality
            if result.get('data'):
                data = result['data']
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if not isinstance(item, dict):
                            issues.append(f"Data item {i} is not a dictionary")
                        elif len(item) == 0:
                            issues.append(f"Data item {i} is empty")
                        else:
                            # Check for meaningful content
                            has_content = any(
                                key in item and item[key] and str(item[key]).strip()
                                for key in ['title', 'content', 'text', 'description', 'name', 'url']
                            )
                            if not has_content:
                                issues.append(f"Data item {i} lacks meaningful content")
            
            # Check metadata if present
            if 'metadata' in result:
                metadata = result['metadata']
                if not isinstance(metadata, dict):
                    issues.append("Metadata is not a dictionary")
                elif 'extraction_method' not in metadata:
                    issues.append("Metadata missing extraction_method")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            return False, issues
    
    def _log_extraction_attempt(self, url: str, method: str, options: Dict[str, Any] = None):
        """Log extraction attempt with detailed context."""
        self.logger.info(
            f"Starting {method} extraction for {url}",
            extra={
                "component": "AdaptiveScraper",
                "url": url,
                "method": method,
                "options": options or {},
                "timestamp": time.time()
            }
        )
    
    def _log_extraction_result(self, url: str, success: bool, result_size: int = 0, 
                             method: str = None, error: str = None, 
                             execution_time: float = None):
        """Log extraction result with comprehensive details."""
        if success:
            self.logger.info(
                f"Extraction successful for {url}, {result_size} items extracted via {method}",
                extra={
                    "component": "AdaptiveScraper",
                    "url": url,
                    "success": True,
                    "result_size": result_size,
                    "method": method,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
            )
        else:
            self.logger.error(
                f"Extraction failed for {url} via {method}: {error}",
                extra={
                    "component": "AdaptiveScraper",
                    "url": url,
                    "success": False,
                    "method": method,
                    "error": error,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
            )
    
    def _log_fallback_trigger(self, url: str, primary_method: str, 
                            fallback_method: str, reason: str):
        """Log fallback strategy trigger with context."""
        self.logger.warning(
            f"Triggering fallback from {primary_method} to {fallback_method} for {url}: {reason}",
            extra={
                "component": "AdaptiveScraper",
                "url": url,
                "primary_method": primary_method,
                "fallback_method": fallback_method,
                "reason": reason,
                "timestamp": time.time()
            }
        )
    
    async def _execute_monitored_pipeline(self, pipeline_type: str, url: str, 
                                        html: str, options: Dict[str, Any], 
                                        execution_id: str) -> Dict[str, Any]:
        """
        Execute pipeline with comprehensive monitoring.
        
        Based on gameplan Phase 3 specifications for pipeline execution monitoring.
        """
        start_time = time.time()
        
        try:
            self._log_extraction_attempt(url, pipeline_type, options)
            
            # Execute the actual pipeline
            if pipeline_type == "sitemap_extraction":
                result = await self._process_sitemap_extraction([url], options or {})
            elif pipeline_type == "feed_extraction":
                result = await self._process_feed_extraction(url, options or {})
            elif pipeline_type == "api_extraction":
                result = await self._process_api_extraction(url, options or {})
            elif pipeline_type == "product_extraction":
                result = await self._process_product_extraction(url, html, options or {})
            elif pipeline_type == "listing_extraction":
                result = await self._process_listing_extraction(url, html, options or {})
            elif pipeline_type == "article_extraction":
                result = await self._process_article_extraction(url, html, options or {})
            else:
                # Default HTML extraction
                result = await self._process_html_extraction(url, html, options or {})
            
            execution_time = time.time() - start_time
            result_size = len(result.get('data', [])) if isinstance(result.get('data'), list) else (1 if result.get('data') else 0)
            
            self._log_extraction_result(
                url, result.get('success', False), result_size, 
                pipeline_type, result.get('error'), execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Pipeline execution error: {str(e)}"
            
            self._log_extraction_result(
                url, False, 0, pipeline_type, error_msg, execution_time
            )
            
            return {
                "success": False,
                "error": error_msg,
                "data": [],
                "metadata": {
                    "pipeline_type": pipeline_type,
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "error_details": traceback.format_exc()
                }
            }
    
    async def _execute_fallback_pipeline(self, url: str, html: str, 
                                       options: Dict[str, Any], 
                                       execution_id: str) -> Dict[str, Any]:
        """Execute fallback pipeline when primary extraction fails."""
        try:
            self._log_fallback_trigger(url, "primary_pipeline", "fallback_pipeline", 
                                     "Primary pipeline validation failed")
            
            # Try fallback extraction methods
            fallback_result = await self._extract_content_fallback_async(url, html, options)
            
            # Validate fallback result
            is_valid, issues = self._validate_extraction_result(fallback_result)
            if not is_valid:
                self.logger.warning(f"Fallback extraction also produced invalid results: {issues}")
                
                # Final fallback - basic URL info
                return {
                    "success": True,
                    "data": [{
                        "url": url,
                        "title": "Content extraction failed",
                        "content": "Unable to extract meaningful content",
                        "extraction_method": "final_fallback"
                    }],
                    "metadata": {
                        "pipeline_type": "final_fallback",
                        "execution_id": execution_id,
                        "validation_issues": issues
                    }
                }
            
            return fallback_result
            
        except Exception as e:
            self.logger.error(f"Fallback pipeline failed: {e}")
            return {
                "success": False,
                "error": f"All pipelines failed: {str(e)}",
                "data": [],
                "metadata": {
                    "pipeline_type": "failed_fallback",
                    "execution_id": execution_id,
                    "error_details": traceback.format_exc()
                }
            }
    
    # Helper methods for specific pipeline types referenced in _execute_monitored_pipeline
    async def _process_feed_extraction(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process RSS/Atom feed extraction."""
        try:
            # Use existing feed extraction logic or implement basic version
            html_content = await self._fetch_html_content(url, options)
            if not html_content:
                return {"success": False, "error": "Failed to fetch feed content", "data": []}
            
            # Basic feed parsing
            soup = BeautifulSoup(html_content, 'xml')
            items = []
            
            # RSS format
            for item in soup.find_all('item')[:10]:  # Limit to 10 items
                title = item.find('title')
                link = item.find('link')
                description = item.find('description')
                
                items.append({
                    "title": title.text if title else "No title",
                    "url": link.text if link else url,
                    "content": description.text if description else "No description",
                    "type": "feed_item"
                })
            
            # Atom format fallback
            if not items:
                for entry in soup.find_all('entry')[:10]:
                    title = entry.find('title')
                    link = entry.find('link')
                    summary = entry.find('summary')
                    
                    items.append({
                        "title": title.text if title else "No title",
                        "url": link.get('href') if link else url,
                        "content": summary.text if summary else "No summary",
                        "type": "feed_item"
                    })
            
            return {
                "success": len(items) > 0,
                "data": items,
                "metadata": {"extraction_method": "feed_extraction", "items_count": len(items)}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Feed extraction failed: {str(e)}", "data": []}
    
    async def _process_api_extraction(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process API endpoint extraction."""
        try:
            # Fetch JSON data
            html_content = await self._fetch_html_content(url, options)
            if not html_content:
                return {"success": False, "error": "Failed to fetch API content", "data": []}
            
            # Try to parse as JSON
            try:
                data = json.loads(html_content)
                
                # Handle different JSON structures
                items = []
                if isinstance(data, list):
                    items = data[:20]  # Limit to 20 items
                elif isinstance(data, dict):
                    # Look for common list fields
                    for key in ['data', 'items', 'results', 'records']:
                        if key in data and isinstance(data[key], list):
                            items = data[key][:20]
                            break
                    
                    # If no list found, treat as single item
                    if not items:
                        items = [data]
                
                return {
                    "success": len(items) > 0,
                    "data": items,
                    "metadata": {"extraction_method": "api_extraction", "items_count": len(items)}
                }
                
            except json.JSONDecodeError:
                return {"success": False, "error": "Invalid JSON response", "data": []}
            
        except Exception as e:
            return {"success": False, "error": f"API extraction failed: {str(e)}", "data": []}
    
    async def _process_product_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process product page extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch product content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract product information
            product_data = {
                "url": url,
                "type": "product"
            }
            
            # Try to find title
            title_selectors = ['h1', '.product-title', '.title', '[data-testid*="title"]']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text(strip=True):
                    product_data["title"] = title_elem.get_text(strip=True)
                    break
            
            # Try to find price
            price_selectors = ['.price', '.cost', '[data-testid*="price"]', '.amount']
            for selector in price_selectors:
                price_elem = soup.select_one(selector)
                if price_elem and price_elem.get_text(strip=True):
                    product_data["price"] = price_elem.get_text(strip=True)
                    break
            
            # Try to find description
            desc_selectors = ['.description', '.product-description', '.details']
            for selector in desc_selectors:
                desc_elem = soup.select_one(selector)
                if desc_elem and desc_elem.get_text(strip=True):
                    product_data["description"] = desc_elem.get_text(strip=True)[:500]
                    break
            
            return {
                "success": len(product_data) > 2,  # More than just url and type
                "data": [product_data],
                "metadata": {"extraction_method": "product_extraction", "items_count": 1}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Product extraction failed: {str(e)}", "data": []}
    
    async def _process_listing_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process listing/directory page extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch listing content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find repeating patterns that might be listings
            items = []
            
            # Common listing selectors
            listing_selectors = [
                '.item', '.listing', '.result', '.card', '.entry',
                '[class*="item"]', '[class*="listing"]', '[class*="result"]'
            ]
            
            for selector in listing_selectors:
                elements = soup.select(selector)
                if len(elements) > 1:  # Must have multiple items to be a listing
                    for elem in elements[:20]:  # Limit to 20 items
                        item_data = {"type": "listing_item", "url": url}
                        
                        # Extract title
                        title_elem = elem.find(['h1', 'h2', 'h3', 'h4', 'a'])
                        if title_elem:
                            item_data["title"] = title_elem.get_text(strip=True)
                        
                        # Extract link if available
                        link_elem = elem.find('a')
                        if link_elem and link_elem.get('href'):
                            item_data["link"] = urljoin(url, link_elem['href'])
                        
                        # Extract content
                        content = elem.get_text(strip=True)
                        if content:
                            item_data["content"] = content[:300]
                        
                        if len(item_data) > 2:  # More than just type and url
                            items.append(item_data)
                    
                    if items:
                        break  # Found listings, no need to try other selectors
            
            return {
                "success": len(items) > 0,
                "data": items,
                "metadata": {"extraction_method": "listing_extraction", "items_count": len(items)}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Listing extraction failed: {str(e)}", "data": []}
    
    async def _process_article_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process article/blog post extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch article content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            article_data = {
                "url": url,
                "type": "article"
            }
            
            # Extract title
            title_selectors = ['h1', '.title', '.headline', 'header h1', 'article h1']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text(strip=True):
                    article_data["title"] = title_elem.get_text(strip=True)
                    break
            
            # Extract article content
            content_selectors = ['article', '.content', '.post-content', '.entry-content', 'main']
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Clean up content
                    for script in content_elem(['script', 'style']):
                        script.decompose()
                    content_text = content_elem.get_text(strip=True)
                    if len(content_text) > 100:  # Minimum content length
                        article_data["content"] = content_text[:2000]  # Limit content length
                        break
            
            # Extract author if available
            author_selectors = ['.author', '.byline', '[rel="author"]']
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem and author_elem.get_text(strip=True):
                    article_data["author"] = author_elem.get_text(strip=True)
                    break
            
            return {
                "success": len(article_data) > 2,  # More than just url and type
                "data": [article_data],
                "metadata": {"extraction_method": "article_extraction", "items_count": 1}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Article extraction failed: {str(e)}", "data": []}
    
    async def _process_html_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process general HTML extraction as fallback."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch HTML content", "data": []}
            
            # Use existing content extraction logic
            return await self._extract_content_fallback_async(url, html, options)
            
        except Exception as e:
            return {"success": False, "error": f"HTML extraction failed: {str(e)}", "data": []}
    
    # ============================================================================
    # PHASE 5: ERROR HANDLING & DEBUGGING ENHANCEMENTS
    # ============================================================================
    
    def _validate_extraction_result(self, result: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate extraction results and return issues found.
        
        Based on gameplan Phase 5 specifications for comprehensive result validation.
        """
        issues = []
        
        try:
            # Check basic structure
            if not isinstance(result, dict):
                issues.append("Result is not a dictionary")
                return False, issues
            
            # Check required fields
            required_fields = ['success', 'data']
            for field in required_fields:
                if field not in result:
                    issues.append(f"Missing required field: {field}")
            
            # Check success flag consistency
            if result.get('success') == True:
                if not result.get('data'):
                    issues.append("Success=True but no data provided")
                elif isinstance(result['data'], list) and len(result['data']) == 0:
                    issues.append("Success=True but data list is empty")
            
            # Check data quality
            if result.get('data'):
                data = result['data']
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if not isinstance(item, dict):
                            issues.append(f"Data item {i} is not a dictionary")
                        elif len(item) == 0:
                            issues.append(f"Data item {i} is empty")
                        else:
                            # Check for meaningful content
                            has_content = any(
                                key in item and item[key] and str(item[key]).strip()
                                for key in ['title', 'content', 'text', 'description', 'name', 'url']
                            )
                            if not has_content:
                                issues.append(f"Data item {i} lacks meaningful content")
            
            # Check metadata if present
            if 'metadata' in result:
                metadata = result['metadata']
                if not isinstance(metadata, dict):
                    issues.append("Metadata is not a dictionary")
                elif 'extraction_method' not in metadata:
                    issues.append("Metadata missing extraction_method")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            return False, issues
    
    def _log_extraction_attempt(self, url: str, method: str, options: Dict[str, Any] = None):
        """Log extraction attempt with detailed context."""
        self.logger.info(
            f"Starting {method} extraction for {url}",
            extra={
                "component": "AdaptiveScraper",
                "url": url,
                "method": method,
                "options": options or {},
                "timestamp": time.time()
            }
        )
    
    def _log_extraction_result(self, url: str, success: bool, result_size: int = 0, 
                             method: str = None, error: str = None, 
                             execution_time: float = None):
        """Log extraction result with comprehensive details."""
        if success:
            self.logger.info(
                f"Extraction successful for {url}, {result_size} items extracted via {method}",
                extra={
                    "component": "AdaptiveScraper",
                    "url": url,
                    "success": True,
                    "result_size": result_size,
                    "method": method,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
            )
        else:
            self.logger.error(
                f"Extraction failed for {url} via {method}: {error}",
                extra={
                    "component": "AdaptiveScraper",
                    "url": url,
                    "success": False,
                    "method": method,
                    "error": error,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
            )
    
    def _log_fallback_trigger(self, url: str, primary_method: str, 
                            fallback_method: str, reason: str):
        """Log fallback strategy trigger with context."""
        self.logger.warning(
            f"Triggering fallback from {primary_method} to {fallback_method} for {url}: {reason}",
            extra={
                "component": "AdaptiveScraper",
                "url": url,
                "primary_method": primary_method,
                "fallback_method": fallback_method,
                "reason": reason,
                "timestamp": time.time()
            }
        )
    
    async def _execute_monitored_pipeline(self, pipeline_type: str, url: str, 
                                        html: str, options: Dict[str, Any], 
                                        execution_id: str) -> Dict[str, Any]:
        """
        Execute pipeline with comprehensive monitoring.
        
        Based on gameplan Phase 3 specifications for pipeline execution monitoring.
        """
        start_time = time.time()
        
        try:
            self._log_extraction_attempt(url, pipeline_type, options)
            
            # Execute the actual pipeline
            if pipeline_type == "sitemap_extraction":
                result = await self._process_sitemap_extraction([url], options or {})
            elif pipeline_type == "feed_extraction":
                result = await self._process_feed_extraction(url, options or {})
            elif pipeline_type == "api_extraction":
                result = await self._process_api_extraction(url, options or {})
            elif pipeline_type == "product_extraction":
                result = await self._process_product_extraction(url, html, options or {})
            elif pipeline_type == "listing_extraction":
                result = await self._process_listing_extraction(url, html, options or {})
            elif pipeline_type == "article_extraction":
                result = await self._process_article_extraction(url, html, options or {})
            else:
                # Default HTML extraction
                result = await self._process_html_extraction(url, html, options or {})
            
            execution_time = time.time() - start_time
            result_size = len(result.get('data', [])) if isinstance(result.get('data'), list) else (1 if result.get('data') else 0)
            
            self._log_extraction_result(
                url, result.get('success', False), result_size, 
                pipeline_type, result.get('error'), execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Pipeline execution error: {str(e)}"
            
            self._log_extraction_result(
                url, False, 0, pipeline_type, error_msg, execution_time
            )
            
            return {
                "success": False,
                "error": error_msg,
                "data": [],
                "metadata": {
                    "pipeline_type": pipeline_type,
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "error_details": traceback.format_exc()
                }
            }
    
    async def _execute_fallback_pipeline(self, url: str, html: str, 
                                       options: Dict[str, Any], 
                                       execution_id: str) -> Dict[str, Any]:
        """Execute fallback pipeline when primary extraction fails."""
        try:
            self._log_fallback_trigger(url, "primary_pipeline", "fallback_pipeline", 
                                     "Primary pipeline validation failed")
            
            # Try fallback extraction methods
            fallback_result = await self._extract_content_fallback_async(url, html, options)
            
            # Validate fallback result
            is_valid, issues = self._validate_extraction_result(fallback_result)
            if not is_valid:
                self.logger.warning(f"Fallback extraction also produced invalid results: {issues}")
                
                # Final fallback - basic URL info
                return {
                    "success": True,
                    "data": [{
                        "url": url,
                        "title": "Content extraction failed",
                        "content": "Unable to extract meaningful content",
                        "extraction_method": "final_fallback"
                    }],
                    "metadata": {
                        "pipeline_type": "final_fallback",
                        "execution_id": execution_id,
                        "validation_issues": issues
                    }
                }
            
            return fallback_result
            
        except Exception as e:
            self.logger.error(f"Fallback pipeline failed: {e}")
            return {
                "success": False,
                "error": f"All pipelines failed: {str(e)}",
                "data": [],
                "metadata": {
                    "pipeline_type": "failed_fallback",
                    "execution_id": execution_id,
                    "error_details": traceback.format_exc()
                }
            }
    
    # Helper methods for specific pipeline types referenced in _execute_monitored_pipeline
    async def _process_feed_extraction(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process RSS/Atom feed extraction."""
        try:
            # Use existing feed extraction logic or implement basic version
            html_content = await self._fetch_html_content(url, options)
            if not html_content:
                return {"success": False, "error": "Failed to fetch feed content", "data": []}
            
            # Basic feed parsing
            soup = BeautifulSoup(html_content, 'xml')
            items = []
            
            # RSS format
            for item in soup.find_all('item')[:10]:  # Limit to 10 items
                title = item.find('title')
                link = item.find('link')
                description = item.find('description')
                
                items.append({
                    "title": title.text if title else "No title",
                    "url": link.text if link else url,
                    "content": description.text if description else "No description",
                    "type": "feed_item"
                })
            
            # Atom format fallback
            if not items:
                for entry in soup.find_all('entry')[:10]:
                    title = entry.find('title')
                    link = entry.find('link')
                    summary = entry.find('summary')
                    
                    items.append({
                        "title": title.text if title else "No title",
                        "url": link.get('href') if link else url,
                        "content": summary.text if summary else "No summary",
                        "type": "feed_item"
                    })
            
            return {
                "success": len(items) > 0,
                "data": items,
                "metadata": {"extraction_method": "feed_extraction", "items_count": len(items)}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Feed extraction failed: {str(e)}", "data": []}
    
    async def _process_api_extraction(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process API endpoint extraction."""
        try:
            # Fetch JSON data
            html_content = await self._fetch_html_content(url, options)
            if not html_content:
                return {"success": False, "error": "Failed to fetch API content", "data": []}
            
            # Try to parse as JSON
            try:
                data = json.loads(html_content)
                
                # Handle different JSON structures
                items = []
                if isinstance(data, list):
                    items = data[:20]  # Limit to 20 items
                elif isinstance(data, dict):
                    # Look for common list fields
                    for key in ['data', 'items', 'results', 'records']:
                        if key in data and isinstance(data[key], list):
                            items = data[key][:20]
                            break
                    
                    # If no list found, treat as single item
                    if not items:
                        items = [data]
                
                return {
                    "success": len(items) > 0,
                    "data": items,
                    "metadata": {"extraction_method": "api_extraction", "items_count": len(items)}
                }
                
            except json.JSONDecodeError:
                return {"success": False, "error": "Invalid JSON response", "data": []}
            
        except Exception as e:
            return {"success": False, "error": f"API extraction failed: {str(e)}", "data": []}
    
    async def _process_product_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process product page extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch product content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract product information
            product_data = {
                "url": url,
                "type": "product"
            }
            
            # Try to find title
            title_selectors = ['h1', '.product-title', '.title', '[data-testid*="title"]']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text(strip=True):
                    product_data["title"] = title_elem.get_text(strip=True)
                    break
            
            # Try to find price
            price_selectors = ['.price', '.cost', '[data-testid*="price"]', '.amount']
            for selector in price_selectors:
                price_elem = soup.select_one(selector)
                if price_elem and price_elem.get_text(strip=True):
                    product_data["price"] = price_elem.get_text(strip=True)
                    break
            
            # Try to find description
            desc_selectors = ['.description', '.product-description', '.details']
            for selector in desc_selectors:
                desc_elem = soup.select_one(selector)
                if desc_elem and desc_elem.get_text(strip=True):
                    product_data["description"] = desc_elem.get_text(strip=True)[:500]
                    break
            
            return {
                "success": len(product_data) > 2,  # More than just url and type
                "data": [product_data],
                "metadata": {"extraction_method": "product_extraction", "items_count": 1}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Product extraction failed: {str(e)}", "data": []}
    
    async def _process_listing_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process listing/directory page extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch listing content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find repeating patterns that might be listings
            items = []
            
            # Common listing selectors
            listing_selectors = [
                '.item', '.listing', '.result', '.card', '.entry',
                '[class*="item"]', '[class*="listing"]', '[class*="result"]'
            ]
            
            for selector in listing_selectors:
                elements = soup.select(selector)
                if len(elements) > 1:  # Must have multiple items to be a listing
                    for elem in elements[:20]:  # Limit to 20 items
                        item_data = {"type": "listing_item", "url": url}
                        
                        # Extract title
                        title_elem = elem.find(['h1', 'h2', 'h3', 'h4', 'a'])
                        if title_elem:
                            item_data["title"] = title_elem.get_text(strip=True)
                        
                        # Extract link if available
                        link_elem = elem.find('a')
                        if link_elem and link_elem.get('href'):
                            item_data["link"] = urljoin(url, link_elem['href'])
                        
                        # Extract content
                        content = elem.get_text(strip=True)
                        if content:
                            item_data["content"] = content[:300]
                        
                        if len(item_data) > 2:  # More than just type and url
                            items.append(item_data)
                    
                    if items:
                        break  # Found listings, no need to try other selectors
            
            return {
                "success": len(items) > 0,
                "data": items,
                "metadata": {"extraction_method": "listing_extraction", "items_count": len(items)}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Listing extraction failed: {str(e)}", "data": []}
    
    async def _process_article_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process article/blog post extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch article content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            article_data = {
                "url": url,
                "type": "article"
            }
            
            # Extract title
            title_selectors = ['h1', '.title', '.headline', 'header h1', 'article h1']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text(strip=True):
                    article_data["title"] = title_elem.get_text(strip=True)
                    break
            
            # Extract article content
            content_selectors = ['article', '.content', '.post-content', '.entry-content', 'main']
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Clean up content
                    for script in content_elem(['script', 'style']):
                        script.decompose()
                    content_text = content_elem.get_text(strip=True)
                    if len(content_text) > 100:  # Minimum content length
                        article_data["content"] = content_text[:2000]  # Limit content length
                        break
            
            # Extract author if available
            author_selectors = ['.author', '.byline', '[rel="author"]']
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem and author_elem.get_text(strip=True):
                    article_data["author"] = author_elem.get_text(strip=True)
                    break
            
            return {
                "success": len(article_data) > 2,  # More than just url and type
                "data": [article_data],
                "metadata": {"extraction_method": "article_extraction", "items_count": 1}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Article extraction failed: {str(e)}", "data": []}
    
    async def _process_html_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process general HTML extraction as fallback."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch HTML content", "data": []}
            
            # Use existing content extraction logic
            return await self._extract_content_fallback_async(url, html, options)
            
        except Exception as e:
            return {"success": False, "error": f"HTML extraction failed: {str(e)}", "data": []}
    
    # ============================================================================
    # PHASE 5: ERROR HANDLING & DEBUGGING ENHANCEMENTS
    # ============================================================================
    
    def _validate_extraction_result(self, result: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate extraction results and return issues found.
        
        Based on gameplan Phase 5 specifications for comprehensive result validation.
        """
        issues = []
        
        try:
            # Check basic structure
            if not isinstance(result, dict):
                issues.append("Result is not a dictionary")
                return False, issues
            
            # Check required fields
            required_fields = ['success', 'data']
            for field in required_fields:
                if field not in result:
                    issues.append(f"Missing required field: {field}")
            
            # Check success flag consistency
            if result.get('success') == True:
                if not result.get('data'):
                    issues.append("Success=True but no data provided")
                elif isinstance(result['data'], list) and len(result['data']) == 0:
                    issues.append("Success=True but data list is empty")
            
            # Check data quality
            if result.get('data'):
                data = result['data']
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if not isinstance(item, dict):
                            issues.append(f"Data item {i} is not a dictionary")
                        elif len(item) == 0:
                            issues.append(f"Data item {i} is empty")
                        else:
                            # Check for meaningful content
                            has_content = any(
                                key in item and item[key] and str(item[key]).strip()
                                for key in ['title', 'content', 'text', 'description', 'name', 'url']
                            )
                            if not has_content:
                                issues.append(f"Data item {i} lacks meaningful content")
            
            # Check metadata if present
            if 'metadata' in result:
                metadata = result['metadata']
                if not isinstance(metadata, dict):
                    issues.append("Metadata is not a dictionary")
                elif 'extraction_method' not in metadata:
                    issues.append("Metadata missing extraction_method")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            return False, issues
    
    def _log_extraction_attempt(self, url: str, method: str, options: Dict[str, Any] = None):
        """Log extraction attempt with detailed context."""
        self.logger.info(
            f"Starting {method} extraction for {url}",
            extra={
                "component": "AdaptiveScraper",
                "url": url,
                "method": method,
                "options": options or {},
                "timestamp": time.time()
            }
        )
    
    def _log_extraction_result(self, url: str, success: bool, result_size: int = 0, 
                             method: str = None, error: str = None, 
                             execution_time: float = None):
        """Log extraction result with comprehensive details."""
        if success:
            self.logger.info(
                f"Extraction successful for {url}, {result_size} items extracted via {method}",
                extra={
                    "component": "AdaptiveScraper",
                    "url": url,
                    "success": True,
                    "result_size": result_size,
                    "method": method,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
            )
        else:
            self.logger.error(
                f"Extraction failed for {url} via {method}: {error}",
                extra={
                    "component": "AdaptiveScraper",
                    "url": url,
                    "success": False,
                    "method": method,
                    "error": error,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
            )
    
    def _log_fallback_trigger(self, url: str, primary_method: str, 
                            fallback_method: str, reason: str):
        """Log fallback strategy trigger with context."""
        self.logger.warning(
            f"Triggering fallback from {primary_method} to {fallback_method} for {url}: {reason}",
            extra={
                "component": "AdaptiveScraper",
                "url": url,
                "primary_method": primary_method,
                "fallback_method": fallback_method,
                "reason": reason,
                "timestamp": time.time()
            }
        )
    
    async def _execute_monitored_pipeline(self, pipeline_type: str, url: str, 
                                        html: str, options: Dict[str, Any], 
                                        execution_id: str) -> Dict[str, Any]:
        """
        Execute pipeline with comprehensive monitoring.
        
        Based on gameplan Phase 3 specifications for pipeline execution monitoring.
        """
        start_time = time.time()
        
        try:
            self._log_extraction_attempt(url, pipeline_type, options)
            
            # Execute the actual pipeline
            if pipeline_type == "sitemap_extraction":
                result = await self._process_sitemap_extraction([url], options or {})
            elif pipeline_type == "feed_extraction":
                result = await self._process_feed_extraction(url, options or {})
            elif pipeline_type == "api_extraction":
                result = await self._process_api_extraction(url, options or {})
            elif pipeline_type == "product_extraction":
                result = await self._process_product_extraction(url, html, options or {})
            elif pipeline_type == "listing_extraction":
                               result = await self._process_listing_extraction(url, html, options or {})
            elif pipeline_type == "article_extraction":
                result = await self._process_article_extraction(url, html, options or {})
            else:
                # Default HTML extraction
                result = await self._process_html_extraction(url, html, options or {})
            
            execution_time = time.time() - start_time
            result_size = len(result.get('data', [])) if isinstance(result.get('data'), list) else (1 if result.get('data') else 0)
            
            self._log_extraction_result(
                url, result.get('success', False), result_size, 
                pipeline_type, result.get('error'), execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Pipeline execution error: {str(e)}"
            
            self._log_extraction_result(
                url, False, 0, pipeline_type, error_msg, execution_time
            )
            
            return {
                "success": False,
                "error": error_msg,
                "data": [],
                "metadata": {
                    "pipeline_type": pipeline_type,
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "error_details": traceback.format_exc()
                }
            }
    
    async def _execute_fallback_pipeline(self, url: str, html: str, 
                                       options: Dict[str, Any], 
                                       execution_id: str) -> Dict[str, Any]:
        """Execute fallback pipeline when primary extraction fails."""
        try:
            self._log_fallback_trigger(url, "primary_pipeline", "fallback_pipeline", 
                                     "Primary pipeline validation failed")
            
            # Try fallback extraction methods
            fallback_result = await self._extract_content_fallback_async(url, html, options)
            
            # Validate fallback result
            is_valid, issues = self._validate_extraction_result(fallback_result)
            if not is_valid:
                self.logger.warning(f"Fallback extraction also produced invalid results: {issues}")
                
                # Final fallback - basic URL info
                return {
                    "success": True,
                    "data": [{
                        "url": url,
                        "title": "Content extraction failed",
                        "content": "Unable to extract meaningful content",
                        "extraction_method": "final_fallback"
                    }],
                    "metadata": {
                        "pipeline_type": "final_fallback",
                        "execution_id": execution_id,
                        "validation_issues": issues
                    }
                }
            
            return fallback_result
            
        except Exception as e:
            self.logger.error(f"Fallback pipeline failed: {e}")
            return {
                "success": False,
                "error": f"All pipelines failed: {str(e)}",
                "data": [],
                "metadata": {
                    "pipeline_type": "failed_fallback",
                    "execution_id": execution_id,
                    "error_details": traceback.format_exc()
                }
            }
    
    # Helper methods for specific pipeline types referenced in _execute_monitored_pipeline
    async def _process_feed_extraction(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process RSS/Atom feed extraction."""
        try:
            # Use existing feed extraction logic or implement basic version
            html_content = await self._fetch_html_content(url, options)
            if not html_content:
                return {"success": False, "error": "Failed to fetch feed content", "data": []}
            
            # Basic feed parsing
            soup = BeautifulSoup(html_content, 'xml')
            items = []
            
            # RSS format
            for item in soup.find_all('item')[:10]:  # Limit to 10 items
                title = item.find('title')
                link = item.find('link')
                description = item.find('description')
                
                items.append({
                    "title": title.text if title else "No title",
                    "url": link.text if link else url,
                    "content": description.text if description else "No description",
                    "type": "feed_item"
                })
            
            # Atom format fallback
            if not items:
                for entry in soup.find_all('entry')[:10]:
                    title = entry.find('title')
                    link = entry.find('link')
                    summary = entry.find('summary')
                    
                    items.append({
                        "title": title.text if title else "No title",
                        "url": link.get('href') if link else url,
                        "content": summary.text if summary else "No summary",
                        "type": "feed_item"
                    })
            
            return {
                "success": len(items) > 0,
                "data": items,
                "metadata": {"extraction_method": "feed_extraction", "items_count": len(items)}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Feed extraction failed: {str(e)}", "data": []}
    
    async def _process_api_extraction(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process API endpoint extraction."""
        try:
            # Fetch JSON data
            html_content = await self._fetch_html_content(url, options)
            if not html_content:
                return {"success": False, "error": "Failed to fetch API content", "data": []}
            
            # Try to parse as JSON
            try:
                data = json.loads(html_content)
                
                # Handle different JSON structures
                items = []
                if isinstance(data, list):
                    items = data[:20]  # Limit to 20 items
                elif isinstance(data, dict):
                    # Look for common list fields
                    for key in ['data', 'items', 'results', 'records']:
                        if key in data and isinstance(data[key], list):
                            items = data[key][:20]
                            break
                    
                    # If no list found, treat as single item
                    if not items:
                        items = [data]
                
                return {
                    "success": len(items) > 0,
                    "data": items,
                    "metadata": {"extraction_method": "api_extraction", "items_count": len(items)}
                }
                
            except json.JSONDecodeError:
                return {"success": False, "error": "Invalid JSON response", "data": []}
            
        except Exception as e:
            return {"success": False, "error": f"API extraction failed: {str(e)}", "data": []}
    
    async def _process_product_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process product page extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch product content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract product information
            product_data = {
                "url": url,
                "type": "product"
            }
            
            # Try to find title
            title_selectors = ['h1', '.product-title', '.title', '[data-testid*="title"]']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text(strip=True):
                    product_data["title"] = title_elem.get_text(strip=True)
                    break
            
            # Try to find price
            price_selectors = ['.price', '.cost', '[data-testid*="price"]', '.amount']
            for selector in price_selectors:
                price_elem = soup.select_one(selector)
                if price_elem and price_elem.get_text(strip=True):
                    product_data["price"] = price_elem.get_text(strip=True)
                    break
            
            # Try to find description
            desc_selectors = ['.description', '.product-description', '.details']
            for selector in desc_selectors:
                desc_elem = soup.select_one(selector)
                if desc_elem and desc_elem.get_text(strip=True):
                    product_data["description"] = desc_elem.get_text(strip=True)[:500]
                    break
            
            return {
                "success": len(product_data) > 2,  # More than just url and type
                "data": [product_data],
                "metadata": {"extraction_method": "product_extraction", "items_count": 1}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Product extraction failed: {str(e)}", "data": []}
    
    async def _process_listing_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process listing/directory page extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch listing content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find repeating patterns that might be listings
            items = []
            
            # Common listing selectors
            listing_selectors = [
                '.item', '.listing', '.result', '.card', '.entry',
                '[class*="item"]', '[class*="listing"]', '[class*="result"]'
            ]
            
            for selector in listing_selectors:
                elements = soup.select(selector)
                if len(elements) > 1:  # Must have multiple items to be a listing
                    for elem in elements[:20]:  # Limit to 20 items
                        item_data = {"type": "listing_item", "url": url}
                        
                        # Extract title
                        title_elem = elem.find(['h1', 'h2', 'h3', 'h4', 'a'])
                        if title_elem:
                            item_data["title"] = title_elem.get_text(strip=True)
                        
                        # Extract link if available
                        link_elem = elem.find('a')
                        if link_elem and link_elem.get('href'):
                            item_data["link"] = urljoin(url, link_elem['href'])
                        
                        # Extract content
                        content = elem.get_text(strip=True)
                        if content:
                            item_data["content"] = content[:300]
                        
                        if len(item_data) > 2:  # More than just type and url
                            items.append(item_data)
                    
                    if items:
                        break  # Found listings, no need to try other selectors
            
            return {
                "success": len(items) > 0,
                "data": items,
                "metadata": {"extraction_method": "listing_extraction", "items_count": len(items)}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Listing extraction failed: {str(e)}", "data": []}
    
    async def _process_article_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process article/blog post extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch article content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            article_data = {
                "url": url,
                "type": "article"
            }
            
            # Extract title
            title_selectors = ['h1', '.title', '.headline', 'header h1', 'article h1']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text(strip=True):
                    article_data["title"] = title_elem.get_text(strip=True)
                    break
            
            # Extract article content
            content_selectors = ['article', '.content', '.post-content', '.entry-content', 'main']
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Clean up content
                    for script in content_elem(['script', 'style']):
                        script.decompose()
                    content_text = content_elem.get_text(strip=True)
                    if len(content_text) > 100:  # Minimum content length
                        article_data["content"] = content_text[:2000]  # Limit content length
                        break
            
            # Extract author if available
            author_selectors = ['.author', '.byline', '[rel="author"]']
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem and author_elem.get_text(strip=True):
                    article_data["author"] = author_elem.get_text(strip=True)
                    break
            
            return {
                "success": len(article_data) > 2,  # More than just url and type
                "data": [article_data],
                "metadata": {"extraction_method": "article_extraction", "items_count": 1}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Article extraction failed: {str(e)}", "data": []}
    
    async def _process_html_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process general HTML extraction as fallback."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch HTML content", "data": []}
            
            # Use existing content extraction logic
            return await self._extract_content_fallback_async(url, html, options)
            
        except Exception as e:
            return {"success": False, "error": f"HTML extraction failed: {str(e)}", "data": []}
    
    # ============================================================================
    # PHASE 5: ERROR HANDLING & DEBUGGING ENHANCEMENTS
    # ============================================================================
    
    def _validate_extraction_result(self, result: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate extraction results and return issues found.
        
        Based on gameplan Phase 5 specifications for comprehensive result validation.
        """
        issues = []
        
        try:
            # Check basic structure
            if not isinstance(result, dict):
                issues.append("Result is not a dictionary")
                return False, issues
            
            # Check required fields
            required_fields = ['success', 'data']
            for field in required_fields:
                if field not in result:
                    issues.append(f"Missing required field: {field}")
            
            # Check success flag consistency
            if result.get('success') == True:
                if not result.get('data'):
                    issues.append("Success=True but no data provided")
                elif isinstance(result['data'], list) and len(result['data']) == 0:
                    issues.append("Success=True but data list is empty")
            
            # Check data quality
            if result.get('data'):
                data = result['data']
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if not isinstance(item, dict):
                            issues.append(f"Data item {i} is not a dictionary")
                        elif len(item) == 0:
                            issues.append(f"Data item {i} is empty")
                        else:
                            # Check for meaningful content
                            has_content = any(
                                key in item and item[key] and str(item[key]).strip()
                                for key in ['title', 'content', 'text', 'description', 'name', 'url']
                            )
                            if not has_content:
                                issues.append(f"Data item {i} lacks meaningful content")
            
            # Check metadata if present
            if 'metadata' in result:
                metadata = result['metadata']
                if not isinstance(metadata, dict):
                    issues.append("Metadata is not a dictionary")
                elif 'extraction_method' not in metadata:
                    issues.append("Metadata missing extraction_method")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            return False, issues
    
    def _log_extraction_attempt(self, url: str, method: str, options: Dict[str, Any] = None):
        """Log extraction attempt with detailed context."""
        self.logger.info(
            f"Starting {method} extraction for {url}",
            extra={
                "component": "AdaptiveScraper",
                "url": url,
                "method": method,
                "options": options or {},
                "timestamp": time.time()
            }
        )
    
    def _log_extraction_result(self, url: str, success: bool, result_size: int = 0, 
                             method: str = None, error: str = None, 
                             execution_time: float = None):
        """Log extraction result with comprehensive details."""
        if success:
            self.logger.info(
                f"Extraction successful for {url}, {result_size} items extracted via {method}",
                extra={
                    "component": "AdaptiveScraper",
                    "url": url,
                    "success": True,
                    "result_size": result_size,
                    "method": method,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
            )
        else:
            self.logger.error(
                f"Extraction failed for {url} via {method}: {error}",
                extra={
                    "component": "AdaptiveScraper",
                    "url": url,
                    "success": False,
                    "method": method,
                    "error": error,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
            )
    
    def _log_fallback_trigger(self, url: str, primary_method: str, 
                            fallback_method: str, reason: str):
        """Log fallback strategy trigger with context."""
        self.logger.warning(
            f"Triggering fallback from {primary_method} to {fallback_method} for {url}: {reason}",
            extra={
                "component": "AdaptiveScraper",
                "url": url,
                "primary_method": primary_method,
                "fallback_method": fallback_method,
                "reason": reason,
                "timestamp": time.time()
            }
        )
    
    async def _execute_monitored_pipeline(self, pipeline_type: str, url: str, 
                                        html: str, options: Dict[str, Any], 
                                        execution_id: str) -> Dict[str, Any]:
        """
        Execute pipeline with comprehensive monitoring.
        
        Based on gameplan Phase 3 specifications for pipeline execution monitoring.
        """
        start_time = time.time()
        
        try:
            self._log_extraction_attempt(url, pipeline_type, options)
            
            # Execute the actual pipeline
            if pipeline_type == "sitemap_extraction":
                result = await self._process_sitemap_extraction([url], options or {})
            elif pipeline_type == "feed_extraction":
                result = await self._process_feed_extraction(url, options or {})
            elif pipeline_type == "api_extraction":
                result = await self._process_api_extraction(url, options or {})
            elif pipeline_type == "product_extraction":
                result = await self._process_product_extraction(url, html, options or {})
            elif pipeline_type == "listing_extraction":
                result = await self._process_listing_extraction(url, html, options or {})
            elif pipeline_type == "article_extraction":
                result = await self._process_article_extraction(url, html, options or {})
            else:
                # Default HTML extraction
                result = await self._process_html_extraction(url, html, options or {})
            
            execution_time = time.time() - start_time
            result_size = len(result.get('data', [])) if isinstance(result.get('data'), list) else (1 if result.get('data') else 0)
            
            self._log_extraction_result(
                url, result.get('success', False), result_size, 
                pipeline_type, result.get('error'), execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Pipeline execution error: {str(e)}"
            
            self._log_extraction_result(
                url, False, 0, pipeline_type, error_msg, execution_time
            )
            
            return {
                "success": False,
                "error": error_msg,
                "data": [],
                "metadata": {
                    "pipeline_type": pipeline_type,
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "error_details": traceback.format_exc()
                }
            }
    
    async def _execute_fallback_pipeline(self, url: str, html: str, 
                                       options: Dict[str, Any], 
                                       execution_id: str) -> Dict[str, Any]:
        """Execute fallback pipeline when primary extraction fails."""
        try:
            self._log_fallback_trigger(url, "primary_pipeline", "fallback_pipeline", 
                                     "Primary pipeline validation failed")
            
            # Try fallback extraction methods
            fallback_result = await self._extract_content_fallback_async(url, html, options)
            
            # Validate fallback result
            is_valid, issues = self._validate_extraction_result(fallback_result)
            if not is_valid:
                self.logger.warning(f"Fallback extraction also produced invalid results: {issues}")
                
                # Final fallback - basic URL info
                return {
                    "success": True,
                    "data": [{
                        "url": url,
                        "title": "Content extraction failed",
                        "content": "Unable to extract meaningful content",
                        "extraction_method": "final_fallback"
                    }],
                    "metadata": {
                        "pipeline_type": "final_fallback",
                        "execution_id": execution_id,
                        "validation_issues": issues
                    }
                }
            
            return fallback_result
            
        except Exception as e:
            self.logger.error(f"Fallback pipeline failed: {e}")
            return {
                "success": False,
                "error": f"All pipelines failed: {str(e)}",
                "data": [],
                "metadata": {
                    "pipeline_type": "failed_fallback",
                    "execution_id": execution_id,
                    "error_details": traceback.format_exc()
                }
            }
    
    # Helper methods for specific pipeline types referenced in _execute_monitored_pipeline
    async def _process_feed_extraction(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process RSS/Atom feed extraction."""
        try:
            # Use existing feed extraction logic or implement basic version
            html_content = await self._fetch_html_content(url, options)
            if not html_content:
                return {"success": False, "error": "Failed to fetch feed content", "data": []}
            
            # Basic feed parsing
            soup = BeautifulSoup(html_content, 'xml')
            items = []
            
            # RSS format
            for item in soup.find_all('item')[:10]:  # Limit to 10 items
                title = item.find('title')
                link = item.find('link')
                description = item.find('description')
                
                items.append({
                    "title": title.text if title else "No title",
                    "url": link.text if link else url,
                    "content": description.text if description else "No description",
                    "type": "feed_item"
                })
            
            # Atom format fallback
            if not items:
                for entry in soup.find_all('entry')[:10]:
                    title = entry.find('title')
                    link = entry.find('link')
                    summary = entry.find('summary')
                    
                    items.append({
                        "title": title.text if title else "No title",
                        "url": link.get('href') if link else url,
                        "content": summary.text if summary else "No summary",
                        "type": "feed_item"
                    })
            
            return {
                "success": len(items) > 0,
                "data": items,
                "metadata": {"extraction_method": "feed_extraction", "items_count": len(items)}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Feed extraction failed: {str(e)}", "data": []}
    
    async def _process_api_extraction(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process API endpoint extraction."""
        try:
            # Fetch JSON data
            html_content = await self._fetch_html_content(url, options)
            if not html_content:
                return {"success": False, "error": "Failed to fetch API content", "data": []}
            
            # Try to parse as JSON
            try:
                data = json.loads(html_content)
                
                # Handle different JSON structures
                items = []
                if isinstance(data, list):
                    items = data[:20]  # Limit to 20 items
                elif isinstance(data, dict):
                    # Look for common list fields
                    for key in ['data', 'items', 'results', 'records']:
                        if key in data and isinstance(data[key], list):
                            items = data[key][:20]
                            break
                    
                    # If no list found, treat as single item
                    if not items:
                        items = [data]
                
                return {
                    "success": len(items) > 0,
                    "data": items,
                    "metadata": {"extraction_method": "api_extraction", "items_count": len(items)}
                }
                
            except json.JSONDecodeError:
                return {"success": False, "error": "Invalid JSON response", "data": []}
            
        except Exception as e:
            return {"success": False, "error": f"API extraction failed: {str(e)}", "data": []}
    
    async def _process_product_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process product page extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch product content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract product information
            product_data = {
                "url": url,
                "type": "product"
            }
            
            # Try to find title
            title_selectors = ['h1', '.product-title', '.title', '[data-testid*="title"]']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text(strip=True):
                    product_data["title"] = title_elem.get_text(strip=True)
                    break
            
            # Try to find price
            price_selectors = ['.price', '.cost', '[data-testid*="price"]', '.amount']
            for selector in price_selectors:
                price_elem = soup.select_one(selector)
                if price_elem and price_elem.get_text(strip=True):
                    product_data["price"] = price_elem.get_text(strip=True)
                    break
            
            # Try to find description
            desc_selectors = ['.description', '.product-description', '.details']
            for selector in desc_selectors:
                desc_elem = soup.select_one(selector)
                if desc_elem and desc_elem.get_text(strip=True):
                    product_data["description"] = desc_elem.get_text(strip=True)[:500]
                    break
            
            return {
                "success": len(product_data) > 2,  # More than just url and type
                "data": [product_data],
                "metadata": {"extraction_method": "product_extraction", "items_count": 1}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Product extraction failed: {str(e)}", "data": []}
    
    async def _process_listing_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process listing/directory page extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch listing content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find repeating patterns that might be listings
            items = []
            
            # Common listing selectors
            listing_selectors = [
                '.item', '.listing', '.result', '.card', '.entry',
                '[class*="item"]', '[class*="listing"]', '[class*="result"]'
            ]
            
            for selector in listing_selectors:
                elements = soup.select(selector)
                if len(elements) > 1:  # Must have multiple items to be a listing
                    for elem in elements[:20]:  # Limit to 20 items
                        item_data = {"type": "listing_item", "url": url}
                        
                        # Extract title
                        title_elem = elem.find(['h1', 'h2', 'h3', 'h4', 'a'])
                        if title_elem:
                            item_data["title"] = title_elem.get_text(strip=True)
                        
                        # Extract link if available
                        link_elem = elem.find('a')
                        if link_elem and link_elem.get('href'):
                            item_data["link"] = urljoin(url, link_elem['href'])
                        
                        # Extract content
                        content = elem.get_text(strip=True)
                        if content:
                            item_data["content"] = content[:300]
                        
                        if len(item_data) > 2:  # More than just type and url
                            items.append(item_data)
                    
                    if items:
                        break  # Found listings, no need to try other selectors
            
            return {
                "success": len(items) > 0,
                "data": items,
                "metadata": {"extraction_method": "listing_extraction", "items_count": len(items)}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Listing extraction failed: {str(e)}", "data": []}
    
    async def _process_article_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process article/blog post extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch article content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            article_data = {
                "url": url,
                "type": "article"
            }
            
            # Extract title
            title_selectors = ['h1', '.title', '.headline', 'header h1', 'article h1']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text(strip=True):
                    article_data["title"] = title_elem.get_text(strip=True)
                    break
            
            # Extract article content
            content_selectors = ['article', '.content', '.post-content', '.entry-content', 'main']
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Clean up content
                    for script in content_elem(['script', 'style']):
                        script.decompose()
                    content_text = content_elem.get_text(strip=True)
                    if len(content_text) > 100:  # Minimum content length
                        article_data["content"] = content_text[:2000]  # Limit content length
                        break
            
            # Extract author if available
            author_selectors = ['.author', '.byline', '[rel="author"]']
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem and author_elem.get_text(strip=True):
                    article_data["author"] = author_elem.get_text(strip=True)
                    break
            
            return {
                "success": len(article_data) > 2,  # More than just url and type
                "data": [article_data],
                "metadata": {"extraction_method": "article_extraction", "items_count": 1}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Article extraction failed: {str(e)}", "data": []}
    
    async def _process_html_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process general HTML extraction as fallback."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch HTML content", "data": []}
            
            # Use existing content extraction logic
            return await self._extract_content_fallback_async(url, html, options)
            
        except Exception as e:
            return {"success": False, "error": f"HTML extraction failed: {str(e)}", "data": []}
    
    # ============================================================================
    # PHASE 5: ERROR HANDLING & DEBUGGING ENHANCEMENTS
    # ============================================================================
    
    def _validate_extraction_result(self, result: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate extraction results and return issues found.
        
        Based on gameplan Phase 5 specifications for comprehensive result validation.
        """
        issues = []
        
        try:
            # Check basic structure
            if not isinstance(result, dict):
                issues.append("Result is not a dictionary")
                return False, issues
            
            # Check required fields
            required_fields = ['success', 'data']
            for field in required_fields:
                if field not in result:
                    issues.append(f"Missing required field: {field}")
            
            # Check success flag consistency
            if result.get('success') == True:
                if not result.get('data'):
                    issues.append("Success=True but no data provided")
                elif isinstance(result['data'], list) and len(result['data']) == 0:
                    issues.append("Success=True but data list is empty")
            
            # Check data quality
            if result.get('data'):
                data = result['data']
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if not isinstance(item, dict):
                            issues.append(f"Data item {i} is not a dictionary")
                        elif len(item) == 0:
                            issues.append(f"Data item {i} is empty")
                        else:
                            # Check for meaningful content
                            has_content = any(
                                key in item and item[key] and str(item[key]).strip()
                                for key in ['title', 'content', 'text', 'description', 'name', 'url']
                            )
                            if not has_content:
                                issues.append(f"Data item {i} lacks meaningful content")
            
            # Check metadata if present
            if 'metadata' in result:
                metadata = result['metadata']
                if not isinstance(metadata, dict):
                    issues.append("Metadata is not a dictionary")
                elif 'extraction_method' not in metadata:
                    issues.append("Metadata missing extraction_method")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            return False, issues
    
    def _log_extraction_attempt(self, url: str, method: str, options: Dict[str, Any] = None):
        """Log extraction attempt with detailed context."""
        self.logger.info(
            f"Starting {method} extraction for {url}",
            extra={
                "component": "AdaptiveScraper",
                "url": url,
                "method": method,
                "options": options or {},
                "timestamp": time.time()
            }
        )
    
    def _log_extraction_result(self, url: str, success: bool, result_size: int = 0, 
                             method: str = None, error: str = None, 
                             execution_time: float = None):
        """Log extraction result with comprehensive details."""
        if success:
            self.logger.info(
                f"Extraction successful for {url}, {result_size} items extracted via {method}",
                extra={
                    "component": "AdaptiveScraper",
                    "url": url,
                    "success": True,
                    "result_size": result_size,
                    "method": method,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
            )
        else:
            self.logger.error(
                f"Extraction failed for {url} via {method}: {error}",
                extra={
                    "component": "AdaptiveScraper",
                    "url": url,
                    "success": False,
                    "method": method,
                    "error": error,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
            )
    
    def _log_fallback_trigger(self, url: str, primary_method: str, 
                            fallback_method: str, reason: str):
        """Log fallback strategy trigger with context."""
        self.logger.warning(
            f"Triggering fallback from {primary_method} to {fallback_method} for {url}: {reason}",
            extra={
                "component": "AdaptiveScraper",
                "url": url,
                "primary_method": primary_method,
                "fallback_method": fallback_method,
                "reason": reason,
                "timestamp": time.time()
            }
        )
    
    async def _execute_monitored_pipeline(self, pipeline_type: str, url: str, 
                                        html: str, options: Dict[str, Any], 
                                        execution_id: str) -> Dict[str, Any]:
        """
        Execute pipeline with comprehensive monitoring.
        
        Based on gameplan Phase 3 specifications for pipeline execution monitoring.
        """
        start_time = time.time()
        
        try:
            self._log_extraction_attempt(url, pipeline_type, options)
            
            # Execute the actual pipeline
            if pipeline_type == "sitemap_extraction":
                result = await self._process_sitemap_extraction([url], options or {})
            elif pipeline_type == "feed_extraction":
                result = await self._process_feed_extraction(url, options or {})
            elif pipeline_type == "api_extraction":
                result = await self._process_api_extraction(url, options or {})
            elif pipeline_type == "product_extraction":
                result = await self._process_product_extraction(url, html, options or {})
            elif pipeline_type == "listing_extraction":
                result = await self._process_listing_extraction(url, html, options or {})
            elif pipeline_type == "article_extraction":
                result = await self._process_article_extraction(url, html, options or {})
            else:
                # Default HTML extraction
                result = await self._process_html_extraction(url, html, options or {})
            
            execution_time = time.time() - start_time
            result_size = len(result.get('data', [])) if isinstance(result.get('data'), list) else (1 if result.get('data') else 0)
            
            self._log_extraction_result(
                url, result.get('success', False), result_size, 
                pipeline_type, result.get('error'), execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Pipeline execution error: {str(e)}"
            
            self._log_extraction_result(
                url, False, 0, pipeline_type, error_msg, execution_time
            )
            
            return {
                "success": False,
                "error": error_msg,
                "data": [],
                "metadata": {
                    "pipeline_type": pipeline_type,
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "error_details": traceback.format_exc()
                }
            }
    
    async def _execute_fallback_pipeline(self, url: str, html: str, 
                                       options: Dict[str, Any], 
                                       execution_id: str) -> Dict[str, Any]:
        """Execute fallback pipeline when primary extraction fails."""
        try:
            self._log_fallback_trigger(url, "primary_pipeline", "fallback_pipeline", 
                                     "Primary pipeline validation failed")
            
            # Try fallback extraction methods
            fallback_result = await self._extract_content_fallback_async(url, html, options)
            
            # Validate fallback result
            is_valid, issues = self._validate_extraction_result(fallback_result)
            if not is_valid:
                self.logger.warning(f"Fallback extraction also produced invalid results: {issues}")
                
                # Final fallback - basic URL info
                return {
                    "success": True,
                    "data": [{
                        "url": url,
                        "title": "Content extraction failed",
                        "content": "Unable to extract meaningful content",
                        "extraction_method": "final_fallback"
                    }],
                    "metadata": {
                        "pipeline_type": "final_fallback",
                        "execution_id": execution_id,
                        "validation_issues": issues
                    }
                }
            
            return fallback_result
            
        except Exception as e:
            self.logger.error(f"Fallback pipeline failed: {e}")
            return {
                "success": False,
                "error": f"All pipelines failed: {str(e)}",
                "data": [],
                "metadata": {
                    "pipeline_type": "failed_fallback",
                    "execution_id": execution_id,
                    "error_details": traceback.format_exc()
                }
            }
    
    # Helper methods for specific pipeline types referenced in _execute_monitored_pipeline
    async def _process_feed_extraction(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process RSS/Atom feed extraction."""
        try:
            # Use existing feed extraction logic or implement basic version
            html_content = await self._fetch_html_content(url, options)
            if not html_content:
                return {"success": False, "error": "Failed to fetch feed content", "data": []}
            
            # Basic feed parsing
            soup = BeautifulSoup(html_content, 'xml')
            items = []
            
            # RSS format
            for item in soup.find_all('item')[:10]:  # Limit to 10 items
                title = item.find('title')
                link = item.find('link')
                description = item.find('description')
                
                items.append({
                    "title": title.text if title else "No title",
                    "url": link.text if link else url,
                    "content": description.text if description else "No description",
                    "type": "feed_item"
                })
            
            # Atom format fallback
            if not items:
                for entry in soup.find_all('entry')[:10]:
                    title = entry.find('title')
                    link = entry.find('link')
                    summary = entry.find('summary')
                    
                    items.append({
                        "title": title.text if title else "No title",
                        "url": link.get('href') if link else url,
                        "content": summary.text if summary else "No summary",
                        "type": "feed_item"
                    })
            
            return {
                "success": len(items) > 0,
                "data": items,
                "metadata": {"extraction_method": "feed_extraction", "items_count": len(items)}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Feed extraction failed: {str(e)}", "data": []}
    
    async def _process_api_extraction(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process API endpoint extraction."""
        try:
            # Fetch JSON data
            html_content = await self._fetch_html_content(url, options)
            if not html_content:
                return {"success": False, "error": "Failed to fetch API content", "data": []}
            
            # Try to parse as JSON
            try:
                data = json.loads(html_content)
                
                # Handle different JSON structures
                items = []
                if isinstance(data, list):
                    items = data[:20]  # Limit to 20 items
                elif isinstance(data, dict):
                    # Look for common list fields
                    for key in ['data', 'items', 'results', 'records']:
                        if key in data and isinstance(data[key], list):
                            items = data[key][:20]
                            break
                    
                    # If no list found, treat as single item
                    if not items:
                        items = [data]
                
                return {
                    "success": len(items) > 0,
                    "data": items,
                    "metadata": {"extraction_method": "api_extraction", "items_count": len(items)}
                }
                
            except json.JSONDecodeError:
                return {"success": False, "error": "Invalid JSON response", "data": []}
            
        except Exception as e:
            return {"success": False, "error": f"API extraction failed: {str(e)}", "data": []}
    
    async def _process_product_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process product page extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch product content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract product information
            product_data = {
                "url": url,
                "type": "product"
            }
            
            # Try to find title
            title_selectors = ['h1', '.product-title', '.title', '[data-testid*="title"]']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text(strip=True):
                    product_data["title"] = title_elem.get_text(strip=True)
                    break
            
            # Try to find price
            price_selectors = ['.price', '.cost', '[data-testid*="price"]', '.amount']
            for selector in price_selectors:
                price_elem = soup.select_one(selector)
                if price_elem and price_elem.get_text(strip=True):
                    product_data["price"] = price_elem.get_text(strip=True)
                    break
            
            # Try to find description
            desc_selectors = ['.description', '.product-description', '.details']
            for selector in desc_selectors:
                desc_elem = soup.select_one(selector)
                if desc_elem and desc_elem.get_text(strip=True):
                    product_data["description"] = desc_elem.get_text(strip=True)[:500]
                    break
            
            return {
                "success": len(product_data) > 2,  # More than just url and type
                "data": [product_data],
                "metadata": {"extraction_method": "product_extraction", "items_count": 1}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Product extraction failed: {str(e)}", "data": []}
    
    async def _process_listing_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process listing/directory page extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch listing content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find repeating patterns that might be listings
            items = []
            
            # Common listing selectors
            listing_selectors = [
                '.item', '.listing', '.result', '.card', '.entry',
                '[class*="item"]', '[class*="listing"]', '[class*="result"]'
            ]
            
            for selector in listing_selectors:
                elements = soup.select(selector)
                if len(elements) > 1:  # Must have multiple items to be a listing
                    for elem in elements[:20]:  # Limit to 20 items
                        item_data = {"type": "listing_item", "url": url}
                        
                        # Extract title
                        title_elem = elem.find(['h1', 'h2', 'h3', 'h4', 'a'])
                        if title_elem:
                            item_data["title"] = title_elem.get_text(strip=True)
                        
                        # Extract link if available
                        link_elem = elem.find('a')
                        if link_elem and link_elem.get('href'):
                            item_data["link"] = urljoin(url, link_elem['href'])
                        
                        # Extract content
                        content = elem.get_text(strip=True)
                        if content:
                            item_data["content"] = content[:300]
                        
                        if len(item_data) > 2:  # More than just type and url
                            items.append(item_data)
                    
                    if items:
                        break  # Found listings, no need to try other selectors
            
            return {
                "success": len(items) > 0,
                "data": items,
                "metadata": {"extraction_method": "listing_extraction", "items_count": len(items)}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Listing extraction failed: {str(e)}", "data": []}
    
    async def _process_article_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process article/blog post extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch article content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            article_data = {
                "url": url,
                "type": "article"
            }
            
            # Extract title
            title_selectors = ['h1', '.title', '.headline', 'header h1', 'article h1']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text(strip=True):
                    article_data["title"] = title_elem.get_text(strip=True)
                    break
            
            # Extract article content
            content_selectors = ['article', '.content', '.post-content', '.entry-content', 'main']
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Clean up content
                    for script in content_elem(['script', 'style']):
                        script.decompose()
                    content_text = content_elem.get_text(strip=True)
                    if len(content_text) > 100:  # Minimum content length
                        article_data["content"] = content_text[:2000]  # Limit content length
                        break
            
            # Extract author if available
            author_selectors = ['.author', '.byline', '[rel="author"]']
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem and author_elem.get_text(strip=True):
                    article_data["author"] = author_elem.get_text(strip=True)
                    break
            
            return {
                "success": len(article_data) > 2,  # More than just url and type
                "data": [article_data],
                "metadata": {"extraction_method": "article_extraction", "items_count": 1}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Article extraction failed: {str(e)}", "data": []}
    
    async def _process_html_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process general HTML extraction as fallback."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch HTML content", "data": []}
            
            # Use existing content extraction logic
            return await self._extract_content_fallback_async(url, html, options)
            
        except Exception as e:
            return {"success": False, "error": f"HTML extraction failed: {str(e)}", "data": []}
    
    # ============================================================================
    # PHASE 5: ERROR HANDLING & DEBUGGING ENHANCEMENTS
    # ============================================================================
    
    def _validate_extraction_result(self, result: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate extraction results and return issues found.
        
        Based on gameplan Phase 5 specifications for comprehensive result validation.
        """
        issues = []
        
        try:
            # Check basic structure
            if not isinstance(result, dict):
                issues.append("Result is not a dictionary")
                return False, issues
            
            # Check required fields
            required_fields = ['success', 'data']
            for field in required_fields:
                if field not in result:
                    issues.append(f"Missing required field: {field}")
            
            # Check success flag consistency
            if result.get('success') == True:
                if not result.get('data'):
                    issues.append("Success=True but no data provided")
                elif isinstance(result['data'], list) and len(result['data']) == 0:
                    issues.append("Success=True but data list is empty")
            
            # Check data quality
            if result.get('data'):
                data = result['data']
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if not isinstance(item, dict):
                            issues.append(f"Data item {i} is not a dictionary")
                        elif len(item) == 0:
                            issues.append(f"Data item {i} is empty")
                        else:
                            # Check for meaningful content
                            has_content = any(
                                key in item and item[key] and str(item[key]).strip()
                                for key in ['title', 'content', 'text', 'description', 'name', 'url']
                            )
                            if not has_content:
                                issues.append(f"Data item {i} lacks meaningful content")
            
            # Check metadata if present
            if 'metadata' in result:
                metadata = result['metadata']
                if not isinstance(metadata, dict):
                    issues.append("Metadata is not a dictionary")
                elif 'extraction_method' not in metadata:
                    issues.append("Metadata missing extraction_method")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            return False, issues
    
    def _log_extraction_attempt(self, url: str, method: str, options: Dict[str, Any] = None):
        """Log extraction attempt with detailed context."""
        self.logger.info(
            f"Starting {method} extraction for {url}",
            extra={
                "component": "AdaptiveScraper",
                "url": url,
                "method": method,
                "options": options or {},
                "timestamp": time.time()
            }
        )
    
    def _log_extraction_result(self, url: str, success: bool, result_size: int = 0, 
                             method: str = None, error: str = None, 
                             execution_time: float = None):
        """Log extraction result with comprehensive details."""
        if success:
            self.logger.info(
                f"Extraction successful for {url}, {result_size} items extracted via {method}",
                extra={
                    "component": "AdaptiveScraper",
                    "url": url,
                    "success": True,
                    "result_size": result_size,
                    "method": method,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
            )
        else:
            self.logger.error(
                f"Extraction failed for {url} via {method}: {error}",
                extra={
                    "component": "AdaptiveScraper",
                    "url": url,
                    "success": False,
                    "method": method,
                    "error": error,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
            )
    
    def _log_fallback_trigger(self, url: str, primary_method: str, 
                            fallback_method: str, reason: str):
        """Log fallback strategy trigger with context."""
        self.logger.warning(
            f"Triggering fallback from {primary_method} to {fallback_method} for {url}: {reason}",
            extra={
                "component": "AdaptiveScraper",
                "url": url,
                "primary_method": primary_method,
                "fallback_method": fallback_method,
                "reason": reason,
                "timestamp": time.time()
            }
        )
    
    async def _execute_monitored_pipeline(self, pipeline_type: str, url: str, 
                                        html: str, options: Dict[str, Any], 
                                        execution_id: str) -> Dict[str, Any]:
        """
        Execute pipeline with comprehensive monitoring.
        
        Based on gameplan Phase 3 specifications for pipeline execution monitoring.
        """
        start_time = time.time()
        
        try:
            self._log_extraction_attempt(url, pipeline_type, options)
            
            # Execute the actual pipeline
            if pipeline_type == "sitemap_extraction":
                result = await self._process_sitemap_extraction([url], options or {})
            elif pipeline_type == "feed_extraction":
                result = await self._process_feed_extraction(url, options or {})
            elif pipeline_type == "api_extraction":
                result = await self._process_api_extraction(url, options or {})
            elif pipeline_type == "product_extraction":
                result = await self._process_product_extraction(url, html, options or {})
            elif pipeline_type == "listing_extraction":
                result = await self._process_listing_extraction(url, html, options or {})
            elif pipeline_type == "article_extraction":
                result = await self._process_article_extraction(url, html, options or {})
            else:
                # Default HTML extraction
                result = await self._process_html_extraction(url, html, options or {})
            
            execution_time = time.time() - start_time
            result_size = len(result.get('data', [])) if isinstance(result.get('data'), list) else (1 if result.get('data') else 0)
            
            self._log_extraction_result(
                url, result.get('success', False), result_size, 
                pipeline_type, result.get('error'), execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Pipeline execution error: {str(e)}"
            
            self._log_extraction_result(
                url, False, 0, pipeline_type, error_msg, execution_time
            )
            
            return {
                "success": False,
                "error": error_msg,
                "data": [],
                "metadata": {
                    "pipeline_type": pipeline_type,
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "error_details": traceback.format_exc()
                }
            }
    
    async def _execute_fallback_pipeline(self, url: str, html: str, 
                                       options: Dict[str, Any], 
                                       execution_id: str) -> Dict[str, Any]:
        """Execute fallback pipeline when primary extraction fails."""
        try:
            self._log_fallback_trigger(url, "primary_pipeline", "fallback_pipeline", 
                                     "Primary pipeline validation failed")
            
            # Try fallback extraction methods
            fallback_result = await self._extract_content_fallback_async(url, html, options)
            
            # Validate fallback result
            is_valid, issues = self._validate_extraction_result(fallback_result)
            if not is_valid:
                self.logger.warning(f"Fallback extraction also produced invalid results: {issues}")
                
                # Final fallback - basic URL info
                return {
                    "success": True,
                    "data": [{
                        "url": url,
                        "title": "Content extraction failed",
                        "content": "Unable to extract meaningful content",
                        "extraction_method": "final_fallback"
                    }],
                    "metadata": {
                        "pipeline_type": "final_fallback",
                        "execution_id": execution_id,
                        "validation_issues": issues
                    }
                }
            
            return fallback_result
            
        except Exception as e:
            self.logger.error(f"Fallback pipeline failed: {e}")
            return {
                "success": False,
                "error": f"All pipelines failed: {str(e)}",
                "data": [],
                "metadata": {
                    "pipeline_type": "failed_fallback",
                    "execution_id": execution_id,
                    "error_details": traceback.format_exc()
                }
            }
    
    # Helper methods for specific pipeline types referenced in _execute_monitored_pipeline
    async def _process_feed_extraction(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process RSS/Atom feed extraction."""
        try:
            # Use existing feed extraction logic or implement basic version
            html_content = await self._fetch_html_content(url, options)
            if not html_content:
                return {"success": False, "error": "Failed to fetch feed content", "data": []}
            
            # Basic feed parsing
            soup = BeautifulSoup(html_content, 'xml')
            items = []
            
            # RSS format
            for item in soup.find_all('item')[:10]:  # Limit to 10 items
                title = item.find('title')
                link = item.find('link')
                description = item.find('description')
                
                items.append({
                    "title": title.text if title else "No title",
                    "url": link.text if link else url,
                    "content": description.text if description else "No description",
                    "type": "feed_item"
                })
            
            # Atom format fallback
            if not items:
                for entry in soup.find_all('entry')[:10]:
                    title = entry.find('title')
                    link = entry.find('link')
                    summary = entry.find('summary')
                    
                    items.append({
                        "title": title.text if title else "No title",
                        "url": link.get('href') if link else url,
                        "content": summary.text if summary else "No summary",
                        "type": "feed_item"
                    })
            
            return {
                "success": len(items) > 0,
                "data": items,
                "metadata": {"extraction_method": "feed_extraction", "items_count": len(items)}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Feed extraction failed: {str(e)}", "data": []}
    
    async def _process_api_extraction(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process API endpoint extraction."""
        try:
            # Fetch JSON data
            html_content = await self._fetch_html_content(url, options)
            if not html_content:
                return {"success": False, "error": "Failed to fetch API content", "data": []}
            
            # Try to parse as JSON
            try:
                data = json.loads(html_content)
                
                # Handle different JSON structures
                items = []
                if isinstance(data, list):
                    items = data[:20]  # Limit to 20 items
                elif isinstance(data, dict):
                    # Look for common list fields
                    for key in ['data', 'items', 'results', 'records']:
                        if key in data and isinstance(data[key], list):
                            items = data[key][:20]
                            break
                    
                    # If no list found, treat as single item
                    if not items:
                        items = [data]
                
                return {
                    "success": len(items) > 0,
                    "data": items,
                    "metadata": {"extraction_method": "api_extraction", "items_count": len(items)}
                }
                
            except json.JSONDecodeError:
                return {"success": False, "error": "Invalid JSON response", "data": []}
            
        except Exception as e:
            return {"success": False, "error": f"API extraction failed: {str(e)}", "data": []}
    
    async def _process_product_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process product page extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch product content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract product information
            product_data = {
                "url": url,
                "type": "product"
            }
            
            # Try to find title
            title_selectors = ['h1', '.product-title', '.title', '[data-testid*="title"]']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text(strip=True):
                    product_data["title"] = title_elem.get_text(strip=True)
                    break
            
            # Try to find price
            price_selectors = ['.price', '.cost', '[data-testid*="price"]', '.amount']
            for selector in price_selectors:
                price_elem = soup.select_one(selector)
                if price_elem and price_elem.get_text(strip=True):
                    product_data["price"] = price_elem.get_text(strip=True)
                    break
            
            # Try to find description
            desc_selectors = ['.description', '.product-description', '.details']
            for selector in desc_selectors:
                desc_elem = soup.select_one(selector)
                if desc_elem and desc_elem.get_text(strip=True):
                    product_data["description"] = desc_elem.get_text(strip=True)[:500]
                    break
            
            return {
                "success": len(product_data) > 2,  # More than just url and type
                "data": [product_data],
                "metadata": {"extraction_method": "product_extraction", "items_count": 1}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Product extraction failed: {str(e)}", "data": []}
    
    async def _process_listing_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process listing/directory page extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch listing content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find repeating patterns that might be listings
            items = []
            
            # Common listing selectors
            listing_selectors = [
                '.item', '.listing', '.result', '.card', '.entry',
                '[class*="item"]', '[class*="listing"]', '[class*="result"]'
            ]
            
            for selector in listing_selectors:
                elements = soup.select(selector)
                if len(elements) > 1:  # Must have multiple items to be a listing
                    for elem in elements[:20]:  # Limit to 20 items
                        item_data = {"type": "listing_item", "url": url}
                        
                        # Extract title
                        title_elem = elem.find(['h1', 'h2', 'h3', 'h4', 'a'])
                        if title_elem:
                            item_data["title"] = title_elem.get_text(strip=True)
                        
                        # Extract link if available
                        link_elem = elem.find('a')
                        if link_elem and link_elem.get('href'):
                            item_data["link"] = urljoin(url, link_elem['href'])
                        
                        # Extract content
                        content = elem.get_text(strip=True)
                        if content:
                            item_data["content"] = content[:300]
                        
                        if len(item_data) > 2:  # More than just type and url
                            items.append(item_data)
                    
                    if items:
                        break  # Found listings, no need to try other selectors
            
            return {
                "success": len(items) > 0,
                "data": items,
                "metadata": {"extraction_method": "listing_extraction", "items_count": len(items)}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Listing extraction failed: {str(e)}", "data": []}
    
    async def _process_article_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process article/blog post extraction."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch article content", "data": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            article_data = {
                "url": url,
                "type": "article"
            }
            
            # Extract title
            title_selectors = ['h1', '.title', '.headline', 'header h1', 'article h1']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text(strip=True):
                    article_data["title"] = title_elem.get_text(strip=True)
                    break
            
            # Extract article content
            content_selectors = ['article', '.content', '.post-content', '.entry-content', 'main']
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Clean up content
                    for script in content_elem(['script', 'style']):
                        script.decompose()
                    content_text = content_elem.get_text(strip=True)
                    if len(content_text) > 100:  # Minimum content length
                        article_data["content"] = content_text[:2000]  # Limit content length
                        break
            
            # Extract author if available
            author_selectors = ['.author', '.byline', '[rel="author"]']
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem and author_elem.get_text(strip=True):
                    article_data["author"] = author_elem.get_text(strip=True)
                    break
            
            return {
                "success": len(article_data) > 2,  # More than just url and type
                "data": [article_data],
                "metadata": {"extraction_method": "article_extraction", "items_count": 1}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Article extraction failed: {str(e)}", "data": []}
    
    async def _process_html_extraction(self, url: str, html: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process general HTML extraction as fallback."""
        try:
            if not html:
                html = await self._fetch_html_content(url, options)
                if not html:
                    return {"success": False, "error": "Failed to fetch HTML content", "data": []}
            
            # Use existing content extraction logic
            return await self._extract_content_fallback_async(url, html, options)
            
        except Exception as e:
            return {"success": False, "error": f"HTML extraction failed: {str(e)}", "data": []}
    
    # ============================================================================
    # PHASE 5: ERROR HANDLING & DEBUGGING ENHANCEMENTS
    # ============================================================================
    
    def _validate_extraction_result(self, result: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate extraction results and return issues found.
        
        Based on gameplan Phase 5 specifications for comprehensive result validation.
        """
        issues = []
        
        try:
            # Check basic structure
            if not isinstance(result, dict):
                issues.append("Result is not a dictionary")
                return False, issues
            
            # Check required fields
            required_fields = ['success', 'data']
            for field in required_fields:
                if field not in result:
                    issues.append(f"Missing required field: {field}")
            
            # Check success flag consistency
            if result.get('success') == True:
                if not result.get('data'):
                    issues.append("Success=True but no data provided")
                elif isinstance(result['data'], list) and len(result['data']) == 0:
                    issues.append("Success=True but data list is empty")
            
            # Check data quality
            if result.get('data'):
                data = result['data']
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if not isinstance(item, dict):
                            issues.append(f"Data item {i} is not a dictionary")
                        elif len(item) == 0:
                            issues.append(f"Data item {i} is empty")
                        else:
                            # Check for meaningful content
                            has_content = any(
                                key in item and item[key] and str(item[key]).strip()
                                for key in ['title', 'content', 'text', 'description', 'name', 'url']
                            )
                            if not has_content:
                                issues.append(f"Data item {i} lacks meaningful content")
            
            # Check metadata if present
            if 'metadata' in result:
                metadata = result['metadata']
                if not isinstance(metadata, dict):
                    issues.append("Metadata is not a dictionary")
                elif 'extraction_method' not in metadata:
                    issues.append("Metadata missing extraction_method")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            return False, issues
    
    def _log_extraction_attempt(self, url: str, method: str, options: Dict[str, Any] = None):
        """Log extraction attempt with detailed context."""
        self.logger.info(
            f"Starting {method} extraction for {url}",
            extra={
                "component": "AdaptiveScraper",
                "url": url,
                "method": method,
                "options": options or {},
                "timestamp": time.time()
            }
        )
    
    def _log_extraction_result(self, url: str, success: bool, result_size: int = 0, 
                             method: str = None, error: str = None, 
                             execution_time: float = None):
        """Log extraction result with comprehensive details."""
        if success:
            self.logger.info(
                f"Extraction successful for {url}, {result_size} items extracted via {method}",
                extra={
                    "component": "AdaptiveScraper",
                    "url": url,
                    "success": True,
                    "result_size": result_size,
                    "method": method,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
            )
        else:
            self.logger.error(
                f"Extraction failed for {url} via {method}: {error}",
                extra={
                    "component": "AdaptiveScraper",
                    "url": url,
                    "success": False,
                    "method": method,
                    "error": error,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
            )
    
    def _log_fallback_trigger(self, url: str, primary_method: str, 
                            fallback_method: str, reason: str):
        """Log fallback strategy trigger with context."""
        self.logger.warning(
            f"Triggering fallback from {primary_method} to {fallback_method} for {url}: {reason}",
            extra={
                "component": "AdaptiveScraper",
                "url": url,
                "primary_method": primary_method,
                "fallback_method": fallback_method,
                "reason": reason,
                "timestamp": time.time()
            }
        )
    
    async def _execute_monitored_pipeline(self, pipeline_type: str, url: str, 
                                        html: str, options: Dict[str, Any], 
                                        execution_id: str) -> Dict[str, Any]:
        """
        Execute pipeline with comprehensive monitoring.
        
        Based on gameplan Phase 3 specifications for pipeline execution monitoring.
        """
        start_time = time.time()
        
        try:
            self._log_extraction_attempt(url, pipeline_type, options)
            
            # Execute the actual pipeline
            if pipeline_type == "sitemap_extraction":
                result = await self._process_sitemap_extraction([url], options or {})
            elif pipeline_type == "feed_extraction":
                result = await self._process_feed_extraction(url, options or {})
            elif pipeline_type == "api_extraction":
                result = await self._process_api_extraction(url, options or {})
            elif pipeline_type == "product_extraction":
                result = await self._process_product_extraction(url, html, options or {})
            elif pipeline_type == "listing_extraction":
                result = await self._process_listing_extraction(url, html, options or {})
            elif pipeline_type == "article_extraction":
                result = await self._process_article_extraction(url, html, options or {})
            else:
                # Default HTML extraction
                result = await self._process_html_extraction(url, html, options or {})
            
            execution_time = time.time() - start_time
            result_size = len(result.get('data', [])) if isinstance(result.get('data'), list) else (1 if result.get('data') else 0)
            
            self._log_extraction_result(
                url, result.get('success', False), result_size, 
                pipeline_type, result.get('error'), execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Pipeline execution error: {str(e)}"
            
            self._log_extraction_result(
                url, False, 0, pipeline_type, error_msg, execution_time            )
            
            return {
                "success": False,
                "error": error_msg,
                "data": [],
                "metadata": {
                    "pipeline_type": pipeline_type,
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "error_details": traceback.format_exc()
                }
            }

    async def _process_sitemap_extraction(self, sitemap_urls: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process sitemap extraction by discovering page URLs and extracting content from each.
        
        This method implements the enhanced sitemap processing from the gameplan,
        systematically extracting and processing URLs from sitemaps for content extraction.
        
        Args:
            sitemap_urls: List of sitemap URLs to process
            options: Extraction options including limits and configuration
            
        Returns:
            Dictionary with success status and extracted data
        """
        all_extracted_results = []
        processed_urls_count = 0
        max_total_urls = options.get('max_total_urls_from_sitemaps', 50)

        for sitemap_url in sitemap_urls:
            if processed_urls_count >= max_total_urls:
                break
                
            try:
                # Discover page URLs from sitemap
                discovered_page_urls = await self.site_discovery.extract_urls_from_sitemap(
                    sitemap_url,
                    max_urls_per_sitemap=max_total_urls - processed_urls_count
                )
                logger.info(f"Discovered {len(discovered_page_urls)} page URLs from sitemap: {sitemap_url}")

                # Process each discovered URL
                for page_url in discovered_page_urls:
                    if processed_urls_count >= max_total_urls:
                        break
                        
                    # Mark as sitemap-discovered URL
                    page_options = options.copy()
                    page_options['is_sitemap_discovered_url'] = True
                    
                    # Extract content from the page URL
                    result = await self._extract_from_final_url(page_url, page_options)
                    
                    if result.get('success') and result.get('data'):
                        # Handle both single items and lists
                        data = result['data']
                        if isinstance(data, list):
                            all_extracted_results.extend(data)
                        else:
                            all_extracted_results.append(data)
                            
                    processed_urls_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing sitemap {sitemap_url}: {e}")
                continue
       
        return {
            "success": len(all_extracted_results) > 0, 
            "data": all_extracted_results,
            "processed_urls": processed_urls_count,
            "total_results": len(all_extracted_results)
        }

    async def _execute_crawl4ai_strategy(self, url: str, html_content: str, pydantic_schema: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Crawl4AI extraction strategy with LLM configuration.
        
        This method implements the Crawl4AI integration from the gameplan,
        using LLM-powered extraction with Pydantic schemas.
        
        Args:
            url: The URL to extract content from
            html_content: Pre-fetched HTML content (can be None)
            pydantic_schema: Pydantic schema for structured extraction
            options: Extraction options and configuration
            
        Returns:
            Dictionary with success status and extracted data
        """
        if not pydantic_schema:
            logger.warning(f"No Pydantic schema provided for Crawl4AI extraction for {url}")
            return {"success": False, "error": "Missing schema for Crawl4AI", "data": []}

        try:
            # Import Crawl4AI components
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig
            from crawl4ai.extraction_strategy import LLMExtractionStrategy
            
            # Configure LLM for extraction
            llm_config = LLMConfig(
                provider="openai/gpt-4o",
                extraction_schema=pydantic_schema
            )
            
            # Create crawler with LLM configuration
            crawler = AsyncWebCrawler(
                llm_config=llm_config,
                run_config=CrawlerRunConfig(max_retries=2)
            )
            
            # Run extraction
            result = await crawler.run(url=url)
            
            if result.status_code == 200 and result.extracted_data:
                # Process extracted data
                data_list = result.extracted_data if isinstance(result.extracted_data, list) else [result.extracted_data]
                processed_data = []
                
                for item in data_list:
                    if item:
                        # Convert Pydantic models to dictionaries
                        if hasattr(item, 'model_dump'):
                            processed_data.append(item.model_dump())
                        elif hasattr(item, 'dict'):
                            processed_data.append(item.dict())
                        else:
                            processed_data.append(item)
                
                logger.info(f"Crawl4AI extraction successful for {url}, extracted {len(processed_data)} items")
                return {"success": True, "data": processed_data}
            else:
                error_msg = result.error if hasattr(result, 'error') else f"Crawl4AI extraction failed. Status: {result.status_code}"
                logger.warning(f"Crawl4AI extraction failed for {url}. {error_msg}")
                return {"success": False, "error": error_msg, "data": []}
                
        except ImportError as e:
            logger.error(f"Crawl4AI not available: {e}")
            return {"success": False, "error": "Crawl4AI package not available", "data": []}
        except Exception as e:
            logger.error(f"Error executing Crawl4AI strategy for {url}: {e}")
            return {"success": False, "error": str(e), "data": []}



# Global instance for backward compatibility
_global_adaptive_scraper = None

def get_adaptive_scraper() -> 'AdaptiveScraper':
    """Get the global adaptive scraper instance."""
    global _global_adaptive_scraper
    if _global_adaptive_scraper is None:
        _global_adaptive_scraper = AdaptiveScraper()
    return _global_adaptive_scraper
