"""
Multi-Strategy Extraction System

This module implements a system for combining multiple extraction strategies 
and search engines to maximize data extraction success and quality. It prioritizes 
and merges results from different strategies based on confidence scores and relevance.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Set
import asyncio
import logging
from collections import defaultdict
import json

from strategies.base_strategy import BaseStrategy, SearchEngineInterface, SearchCapabilityType, get_registered_search_engines  
from strategies.core.composite_strategy import CompositeStrategy
from strategies.core.strategy_context import StrategyContext
from extraction.schema_extraction import SchemaExtractor, ExtractionSchema
from strategies.dfs_strategy import DFSStrategy
from strategies.bfs_strategy import BFSStrategy
from strategies.best_first import BestFirstStrategy
from strategies.ai_guided_strategy import AIGuidedStrategy
from strategies.core.strategy_types import StrategyType, StrategyCapability, strategy_metadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiStrategy")

@strategy_metadata(
    strategy_type=StrategyType.SPECIAL_PURPOSE,
    capabilities={
        StrategyCapability.LINK_EXTRACTION,
        StrategyCapability.DYNAMIC_CONTENT,
        StrategyCapability.API_INTERACTION,
        StrategyCapability.FORM_INTERACTION,
        StrategyCapability.ERROR_HANDLING
    },
    description="Strategy that combines multiple specialized strategies for comprehensive data extraction"
)
class MultiStrategy(CompositeStrategy):
    """
    A strategy that combines multiple extraction strategies to improve overall
    extraction results. It runs multiple strategies in parallel and consolidates
    their results based on confidence scores.
    """
    
    def __init__(self, 
                 context: StrategyContext,
                 search_engines: List[SearchEngineInterface] = None,
                 fallback_threshold: float = 0.4,
                 confidence_threshold: float = 0.7,
                 use_voting: bool = True,
                 max_depth: int = 2, 
                 max_pages: int = 100,
                 include_external: bool = False,
                 user_prompt: str = "",
                 filter_chain: Optional[Any] = None):
        """
        Initialize the multi-strategy extraction system.
        
        Args:
            context: The strategy context containing shared services and configuration
            search_engines: List of search engine instances to use
            fallback_threshold: Threshold below which to try fallback strategies
            confidence_threshold: Threshold for accepting extraction results
            use_voting: Whether to use voting to resolve conflicts
            max_depth: Maximum crawling depth
            max_pages: Maximum number of pages to crawl
            include_external: Whether to include external links
            user_prompt: The user's original request/prompt
            filter_chain: Filter chain to apply to URLs
        """
        super().__init__(context)
        
        # Initialize MultiStrategy-specific attributes
        self.search_engines = search_engines or []
        self.fallback_threshold = fallback_threshold
        self.confidence_threshold = confidence_threshold
        self.use_voting = use_voting
        self.schema_extractor = SchemaExtractor(use_pattern_analyzer=True)
        
        # Set inherited BaseStrategy attributes
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.include_external = include_external
        self.user_prompt = user_prompt
        self.filter_chain = filter_chain
        
        # Track individual strategy results
        self.strategy_results = defaultdict(list)
        self.extraction_stats = {
            "total_attempts": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "fallbacks_triggered": 0,
            "strategy_usage": defaultdict(int),
            "strategy_success": defaultdict(int),
            "search_engine_usage": defaultdict(int),
            "search_engine_success": defaultdict(int),
            "avg_confidence": 0.0
        }
        
        # Engine performance tracking
        self.engine_performance = {}
        
        # Initialize default strategies if context has strategy factory
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default child strategies if none are configured."""
        if len(self._child_strategies) == 0 and self.context and hasattr(self.context, 'strategy_factory'):
            try:
                # Add common strategies
                default_strategies = ['dfs_strategy', 'bfs_strategy', 'best_first_strategy']
                for strategy_name in default_strategies:
                    try:
                        self.add_strategy_by_name(strategy_name)
                    except Exception as e:
                        self.logger.debug(f"Could not add strategy {strategy_name}: {e}")
            except Exception as e:
                self.logger.debug(f"Could not initialize default strategies: {e}")
    
    @property
    def strategies(self) -> List[BaseStrategy]:
        """Get list of child strategies for backward compatibility."""
        return list(self._child_strategies.values())
    
    @property
    def name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            str: Strategy name
        """
        return "multi_strategy"
    
    def can_handle(self, url: str, **kwargs) -> bool:
        """
        Check if the multi-strategy can handle the given URL.
        MultiStrategy can handle most URLs as long as at least one child strategy can handle it.
        
        Args:
            url: The URL to check
            **kwargs: Additional arguments
            
        Returns:
            bool: True if the multi-strategy can handle the URL, False otherwise
        """
        try:
            # Basic URL validation
            if not url or not isinstance(url, str):
                return False
            
            # If we have child strategies, check if any can handle the URL
            if self.strategies:
                for strategy in self.strategies:
                    if hasattr(strategy, 'can_handle') and strategy.can_handle(url, **kwargs):
                        return True
                # If no child strategies can handle it, return False
                return False
            
            # If no child strategies are configured yet, MultiStrategy can still
            # potentially handle most HTTP/HTTPS URLs as a fallback
            url_lower = url.lower()
            if url_lower.startswith(('http://', 'https://')):
                # Exclude problematic file types
                excluded_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.exe', '.dmg'}
                if any(url_lower.endswith(ext) for ext in excluded_extensions):
                    return False
                return True
            
            return False
            
        except Exception:
            return False
    
    async def select_search_engine(self, url: str, query: str, site_characteristics: Dict[str, Any] = None) -> Tuple[SearchEngineInterface, float]:
        """
        Select the best search engine for the given URL and query based on a decision tree.
        
        Args:
            url: The URL to search on
            query: The search query
            site_characteristics: Optional characteristics of the site
            
        Returns:
            Tuple of (selected_engine, confidence)
        """
        logger.info(f"Selecting search engine for URL: {url}")
        
        if not self.search_engines:
            # Load all registered search engines if none provided
            registered_engines = get_registered_search_engines()
            for engine_class in registered_engines.values():
                self.search_engines.append(engine_class())
        
        # Check if we have site characteristics, if not, try to determine them
        if not site_characteristics:
            site_characteristics = await self._analyze_site_characteristics(url)
        
        # Convert capabilities needed from site characteristics
        required_capabilities = self._determine_required_capabilities(site_characteristics)
        
        # Filter engines that can handle this URL
        candidate_engines = []
        for engine in self.search_engines:
            can_handle, confidence = await engine.can_handle(url)
            if can_handle:
                # Check if engine supports all required capabilities
                capability_score = self._calculate_capability_match(engine, required_capabilities)
                
                # Consider historical performance if available
                performance_score = self.engine_performance.get(engine.name, {}).get('success_rate', 0.5)
                
                # Calculate overall score (weighted average)
                overall_score = (confidence * 0.4) + (capability_score * 0.4) + (performance_score * 0.2)
                
                candidate_engines.append((engine, overall_score))
        
        # If no engines can handle this URL, return None
        if not candidate_engines:
            logger.warning(f"No search engines can handle URL: {url}")
            return None, 0.0
        
        # Sort by score (highest first)
        candidate_engines.sort(key=lambda x: x[1], reverse=True)
        best_engine, best_score = candidate_engines[0]
        
        logger.info(f"Selected search engine '{best_engine.name}' with score {best_score:.2f}")
        return best_engine, best_score
    
    async def _analyze_site_characteristics(self, url: str) -> Dict[str, Any]:
        """
        Analyze a site to determine its characteristics.
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary of site characteristics
        """
        # Try to fetch the page to analyze it
        try:
            import httpx
            from bs4 import BeautifulSoup
            
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                response = await client.get(url)
                html = response.text
                
                # Parse HTML
                soup = BeautifulSoup(html, 'html.parser')
                
                characteristics = {
                    "has_search_form": bool(soup.find("form", action=True)),
                    "has_input_fields": bool(soup.find("input")),
                    "has_ajax": "XMLHttpRequest" in html or "fetch(" in html or "ajax" in html.lower(),
                    "has_autocomplete": "autocomplete" in html.lower() or "typeahead" in html.lower(),
                    "has_infinite_scroll": "infinite" in html.lower() and "scroll" in html.lower(),
                    "has_pagination": bool(soup.find_all("a", href=True, text=lambda t: t and t.isdigit())),
                    "form_complexity": "simple",  # Default
                    "javascript_required": "onclick=" in html or "addEventListener" in html,
                    "html": html  # Include HTML for more detailed analysis if needed
                }
                
                # Determine form complexity
                forms = soup.find_all("form")
                if forms:
                    complex_form_indicators = 0
                    for form in forms:
                        # Count inputs
                        inputs = form.find_all("input")
                        if len(inputs) > 5:
                            complex_form_indicators += 1
                        
                        # Check for select dropdowns
                        selects = form.find_all("select")
                        if selects:
                            complex_form_indicators += 1
                        
                        # Check for hidden fields
                        hidden_fields = form.find_all("input", type="hidden")
                        if len(hidden_fields) > 2:
                            complex_form_indicators += 1
                    
                    if complex_form_indicators >= 2:
                        characteristics["form_complexity"] = "complex"
                
                return characteristics
        except Exception as e:
            logger.error(f"Error analyzing site characteristics: {str(e)}")
            
            # Return basic characteristics on error
            return {
                "has_search_form": False,
                "has_input_fields": False,
                "has_ajax": False,
                "has_autocomplete": False,
                "has_infinite_scroll": False,
                "has_pagination": False,
                "form_complexity": "unknown",
                "javascript_required": False
            }
    
    def _determine_required_capabilities(self, site_characteristics: Dict[str, Any]) -> List[SearchCapabilityType]:
        """
        Determine the capabilities required for a site based on its characteristics.
        
        Args:
            site_characteristics: Dictionary of site characteristics
            
        Returns:
            List of required SearchCapabilityType values
        """
        required_capabilities = []
        
        # Map site characteristics to required capabilities
        if site_characteristics.get("has_search_form", False):
            required_capabilities.append(SearchCapabilityType.FORM_BASED)
            
            # If complex form, add multi-step capability
            if site_characteristics.get("form_complexity") == "complex":
                required_capabilities.append(SearchCapabilityType.MULTI_STEP)
        
        if site_characteristics.get("has_ajax", False):
            required_capabilities.append(SearchCapabilityType.AJAX_HANDLING)
        
        if site_characteristics.get("has_autocomplete", False):
            required_capabilities.append(SearchCapabilityType.AUTOCOMPLETE)
        
        if site_characteristics.get("has_infinite_scroll", False):
            required_capabilities.append(SearchCapabilityType.INFINITE_SCROLL)
        
        if site_characteristics.get("javascript_required", False):
            required_capabilities.append(SearchCapabilityType.DOM_MANIPULATION)
        
        # Default to URL parameter if no other capabilities detected
        if not required_capabilities:
            required_capabilities.append(SearchCapabilityType.URL_PARAMETER)
        
        return required_capabilities
    
    def _calculate_capability_match(self, engine: SearchEngineInterface, required_capabilities: List[SearchCapabilityType]) -> float:
        """
        Calculate how well an engine's capabilities match the required capabilities.
        
        Args:
            engine: The search engine to evaluate
            required_capabilities: List of required capability types
            
        Returns:
            Match score from 0.0 to 1.0
        """
        if not required_capabilities:
            return 1.0  # Perfect match if no capabilities required
        
        # Extract engine capabilities
        engine_capabilities = engine.capabilities
        engine_capability_types = [cap.capability_type for cap in engine_capabilities]
        
        # Count matching capabilities
        matches = 0
        total_required = len(required_capabilities)
        
        for required_cap in required_capabilities:
            if required_cap in engine_capability_types:
                # Find the capability object to get its confidence
                for cap in engine_capabilities:
                    if cap.capability_type == required_cap:
                        matches += cap.confidence
                        break
        
        # Calculate match score
        if total_required > 0:
            return matches / total_required
        return 0.0
    
    def update_engine_performance(self, engine_name: str, success: bool) -> None:
        """
        Update the performance metrics for a search engine.
        
        Args:
            engine_name: Name of the engine
            success: Whether the search was successful
        """
        if engine_name not in self.engine_performance:
            self.engine_performance[engine_name] = {
                'total_searches': 0,
                'successful_searches': 0,
                'success_rate': 0.0
            }
        
        perf = self.engine_performance[engine_name]
        perf['total_searches'] += 1
        if success:
            perf['successful_searches'] += 1
        
        # Update success rate
        if perf['total_searches'] > 0:
            perf['success_rate'] = perf['successful_searches'] / perf['total_searches']
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute the multi-strategy for the given URL.
        
        Args:
            url: The URL to process
            **kwargs: Additional parameters including crawler, extraction_config
            
        Returns:
            Dictionary containing the results
        """
        import asyncio
        
        # Extract parameters from kwargs
        crawler = kwargs.get('crawler')
        extraction_config = kwargs.get('extraction_config')
        
        # If no crawler provided, create a default one
        if not crawler:
            logger.info("No crawler provided, creating default AsyncWebCrawler")
            try:
                from crawl4ai import AsyncWebCrawler
                # Create a basic crawler configuration
                crawler = AsyncWebCrawler()
            except ImportError:
                logger.error("Could not import AsyncWebCrawler - crawler functionality unavailable")
                return {
                    "success": False,
                    "error": "Crawler unavailable - AsyncWebCrawler could not be imported",
                    "url": url,
                    "results": []
                }
        
        # Run the async execute method
        try:
            if asyncio.get_event_loop().is_running():
                # If we're already in an async context, we need to handle this differently
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._execute_async(crawler, url, extraction_config))
                    return future.result()
            else:
                return asyncio.run(self._execute_async(crawler, url, extraction_config))
        except Exception as e:
            logger.error(f"Error executing multi-strategy for {url}: {e}")
            return None
    
    async def _execute_async(self, crawler, start_url, extraction_config=None):
        """
        Execute the multi-strategy crawling and extraction.
        
        Args:
            crawler: The AsyncWebCrawler instance
            start_url: The starting URL
            extraction_config: Extraction configuration
            
        Returns:
            Dictionary containing the consolidated results
        """
        logger.info(f"Starting multi-strategy extraction from {start_url}")
        
        # Initialize the queue with the start URL
        queue = [{
            'url': start_url,
            'depth': 0,
            'score': 1.0
        }]
        
        # Track visited URLs and extraction results
        visited_urls = set()
        results = []
        
        # Process queue until empty or max pages reached
        while queue and len(visited_urls) < self.max_pages:
            # Sort the queue by score (highest first)
            queue.sort(key=lambda x: x['score'], reverse=True)
            
            # Get the next URL to process
            current = queue.pop(0)
            url = current['url']
            depth = current['depth']
            
            # Skip if already visited
            if url in visited_urls:
                continue
            
            visited_urls.add(url)
            logger.info(f"Processing URL: {url} (Depth: {depth})")
            
            try:
                # Fetch the page content
                fetch_result = await crawler.arun(url)
                html = fetch_result.html if fetch_result.success else ""
                
                if not html:
                    logger.warning(f"Empty HTML content for URL: {url}")
                    continue
                
                # Extract data using multiple strategies
                extraction_result, confidence, strategy_used = await self._extract_with_multi_strategy(
                    url=url,
                    html=html,
                    config=extraction_config
                )
                
                # Update extraction statistics
                self.extraction_stats["total_attempts"] += 1
                
                if extraction_result and confidence >= self.confidence_threshold:
                    self.extraction_stats["successful_extractions"] += 1
                    self.extraction_stats["strategy_success"][strategy_used] += 1
                    
                    # Add metadata to the result
                    result_with_metadata = {
                        "data": extraction_result,
                        "source_url": url,
                        "depth": depth,
                        "score": confidence,
                        "strategy": strategy_used
                    }
                    
                    results.append(result_with_metadata)
                else:
                    self.extraction_stats["failed_extractions"] += 1
                
                # Calculate running average confidence
                total_extractions = self.extraction_stats["successful_extractions"]
                if total_extractions > 0:
                    self.extraction_stats["avg_confidence"] = (
                        (self.extraction_stats["avg_confidence"] * (total_extractions - 1) + confidence) 
                        / total_extractions
                    )
                
                # Don't go deeper if max depth reached
                if depth >= self.max_depth:
                    continue
                
                # Get next URLs from all strategies
                for strategy in self.strategies:
                    next_urls = await strategy.get_next_urls(
                        url=url,
                        html=html,
                        depth=depth,
                        visited=visited_urls,
                        extraction_result=(extraction_result, confidence)
                    )
                    
                    # Add new URLs to the queue
                    for next_url_info in next_urls:
                        if next_url_info['url'] not in visited_urls:
                            # Check if URL already in queue
                            existing = next((item for item in queue if item['url'] == next_url_info['url']), None)
                            
                            if existing:
                                # Update score if higher
                                if next_url_info['score'] > existing['score']:
                                    existing['score'] = next_url_info['score']
                            else:
                                queue.append(next_url_info)
            
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
        
        logger.info(f"Multi-strategy extraction completed. Visited {len(visited_urls)} URLs")
        
        return {
            "results": results,
            "stats": dict(self.extraction_stats),
            "engine_performance": self.engine_performance,
            "visited_urls": list(visited_urls)
        }
    
    async def search(self, query: str, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a search operation using the most appropriate search engine.
        
        Args:
            query: The search query string
            url: The target URL to search on
            params: Additional parameters for the search
            
        Returns:
            Dictionary containing search results and metadata
        """
        # Select the best search engine for this URL and query
        engine, confidence = await self.select_search_engine(url, query)
        
        if not engine:
            return {
                "success": False,
                "error": "No suitable search engine found for this URL",
                "results": []
            }
        
        # Track usage
        self.extraction_stats["search_engine_usage"][engine.name] += 1
        
        try:
            # Execute the search with the selected engine
            logger.info(f"Executing search with engine: {engine.name}")
            
            search_result = await engine.search(query, url, params)
            
            # Check for success
            if search_result.get("success", False):
                self.extraction_stats["search_engine_success"][engine.name] += 1
                self.update_engine_performance(engine.name, True)
            else:
                self.update_engine_performance(engine.name, False)
                
                # Try fallback engines if primary failed
                if confidence < self.fallback_threshold:
                    logger.info("Primary search engine failed, trying fallbacks")
                    self.extraction_stats["fallbacks_triggered"] += 1
                    
                    for fallback_engine in self.search_engines:
                        # Skip the one we just tried
                        if fallback_engine.name == engine.name:
                            continue
                        
                        # Check if this engine can handle the URL
                        can_handle, _ = await fallback_engine.can_handle(url)
                        if can_handle:
                            logger.info(f"Trying fallback engine: {fallback_engine.name}")
                            
                            # Track usage
                            self.extraction_stats["search_engine_usage"][fallback_engine.name] += 1
                            
                            fallback_result = await fallback_engine.search(query, url, params)
                            
                            # If fallback succeeded, use its results
                            if fallback_result.get("success", False):
                                self.extraction_stats["search_engine_success"][fallback_engine.name] += 1
                                self.update_engine_performance(fallback_engine.name, True)
                                return fallback_result
                            else:
                                self.update_engine_performance(fallback_engine.name, False)
            
            return search_result
            
        except Exception as e:
            logger.error(f"Search error with engine {engine.name}: {str(e)}")
            self.update_engine_performance(engine.name, False)
            
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    async def _extract_with_multi_strategy(self, url: str, html: str, config=None) -> Tuple[Dict[str, Any], float, str]:
        """
        Extract data using multiple strategies in parallel and combine results.
        
        Args:
            url: The URL being processed
            html: The HTML content to extract from
            config: Optional extraction configuration
            
        Returns:
            Tuple of (combined_result, confidence_score, strategy_used)
        """
        logger.info(f"Extracting data from {url} using multiple strategies")
        
        strategy_results = []
        tasks = []
        
        # Create extraction tasks for each strategy
        for strategy in self.strategies:
            tasks.append(self._extract_with_strategy(strategy, url, html, config))
        
        # Run all extraction strategies in parallel
        if tasks:
            strategy_results = await asyncio.gather(*tasks)
            
            # Filter out failed extractions
            strategy_results = [
                (data, confidence, strategy_name) 
                for data, confidence, strategy_name in strategy_results 
                if data and confidence > 0
            ]
        
        # Return early if no successful extractions
        if not strategy_results:
            logger.warning(f"No successful extractions for URL: {url}")
            return {}, 0.0, "none"
        
        # Get the best single strategy result
        best_result = max(strategy_results, key=lambda x: x[1])
        best_data, best_confidence, best_strategy = best_result
        
        # If only one strategy succeeded, use that result
        if len(strategy_results) == 1:
            return best_data, best_confidence, best_strategy
        
        # If high confidence with best strategy, use that
        if best_confidence >= self.confidence_threshold:
            return best_data, best_confidence, best_strategy
        
        # Otherwise, combine results with voting
        if self.use_voting:
            combined_data, combined_confidence, strategy_name = self._combine_results_with_voting(strategy_results)
            return combined_data, combined_confidence, strategy_name
        
        # If not using voting, fall back to the best result
        return best_data, best_confidence, best_strategy
    
    async def _extract_with_strategy(self, strategy: BaseStrategy, url: str, html: str, config=None) -> Tuple[Dict[str, Any], float, str]:
        """
        Extract data with a single strategy.
        
        Args:
            strategy: The strategy to use
            url: The URL being processed
            html: The HTML content to extract from
            config: Optional extraction configuration
            
        Returns:
            Tuple of (extraction_result, confidence_score, strategy_name)
        """
        strategy_name = strategy.name
        self.extraction_stats["strategy_usage"][strategy_name] += 1
        
        try:
            logger.debug(f"Extracting with strategy: {strategy_name}")
            
            # Extract using the strategy
            result = await strategy.extract(url, html, config)
            
            # Get data and confidence from result
            data = result.get('data', {})
            confidence = result.get('confidence', 0.0)
            
            # Ensure the result has source_url field - this is critical for validation
            if data and isinstance(data, dict) and "source_url" not in data:
                data["source_url"] = url
            
            logger.debug(f"Strategy {strategy_name} extraction result: {len(data)} items, confidence: {confidence:.2f}")
            
            return data, confidence, strategy_name
        
        except Exception as e:
            logger.error(f"Error in strategy {strategy_name}: {str(e)}")
            return {}, 0.0, strategy_name
    
    def _combine_results_with_voting(self, strategy_results: List[Tuple[Dict[str, Any], float, str]]) -> Tuple[Dict[str, Any], float, str]:
        """
        Combine results from multiple strategies using weighted voting.
        
        This method implements a weighted voting system where:
        1. Each strategy contributes to the final result based on its confidence score
        2. For conflicting values, higher confidence scores get more voting weight
        3. The final confidence score is calculated based on agreement between strategies
        
        Args:
            strategy_results: List of (data, confidence, strategy_name) tuples
            
        Returns:
            Tuple of (combined_data, confidence_score, strategy_name)
        """
        logger.info(f"Combining results from {len(strategy_results)} strategies using weighted voting")
        
        if not strategy_results:
            return {}, 0.0, "none"
        
        # Initialize voting structures
        field_votes = defaultdict(lambda: defaultdict(float))
        field_confidences = defaultdict(float)
        total_confidence = sum(confidence for _, confidence, _ in strategy_results)
        
        # Collect votes for each field value
        for data, confidence, _ in strategy_results:
            # Skip empty results
            if not data:
                continue
                
            # Calculate weight based on confidence
            weight = confidence / total_confidence if total_confidence > 0 else 0
            
            # Cast votes for each field in this result
            for field, value in data.items():
                # Convert value to string for comparison
                value_str = str(value)
                
                # Add weighted vote for this value
                field_votes[field][value_str] += weight
                
                # Track maximum confidence for each field
                field_confidences[field] = max(field_confidences[field], confidence)
        
        # Determine winning values for each field
        combined_data = {}
        field_scores = {}
        
        for field, votes in field_votes.items():
            # Get the value with the highest vote count
            winning_value_str, vote_score = max(votes.items(), key=lambda x: x[1])
            
            # Convert back to original type if possible
            for data, _, _ in strategy_results:
                if field in data and str(data[field]) == winning_value_str:
                    combined_data[field] = data[field]
                    break
            else:
                # If not found in original data, use the string version
                combined_data[field] = winning_value_str
            
            # Store the vote score (agreement level) for this field
            field_scores[field] = vote_score
        
        # Calculate overall confidence based on agreement and original confidences
        if field_scores:
            # Average of field agreement scores weighted by field confidences
            weighted_sum = sum(score * field_confidences[field] for field, score in field_scores.items())
            total_field_confidence = sum(field_confidences.values())
            
            combined_confidence = weighted_sum / total_field_confidence if total_field_confidence > 0 else 0
        else:
            combined_confidence = 0.0
        
        return combined_data, combined_confidence, "combined"

    async def search_parallel(self, queries: List[str], url: str, 
                             params: Optional[Dict[str, Any]] = None, 
                             max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Execute multiple search operations in parallel with concurrency control.
        
        This implements efficient parallel search execution with the following features:
        - Automatic concurrency limiting to prevent server overload
        - Separation of fast and complex search engines for optimal performance
        - Intelligent error handling and fallback mechanisms
        - Consolidated results from multiple search engines
        
        Args:
            queries: List of search query strings to execute
            url: The target URL to search on
            params: Additional parameters for the search
            max_concurrent: Maximum number of concurrent search operations
            
        Returns:
            List of dictionaries containing search results for each query
        """
        logger.info(f"Executing {len(queries)} search queries in parallel (max_concurrent={max_concurrent})")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _execute_single_search(query: str) -> Dict[str, Any]:
            """Execute a single search with semaphore-based concurrency control"""
            async with semaphore:
                logger.debug(f"Executing search for query: {query}")
                try:
                    result = await self.search(query, url, params)
                    return result
                except Exception as e:
                    logger.error(f"Error in parallel search for query '{query}': {str(e)}")
                    return {
                        "success": False,
                        "error": str(e),
                        "query": query,
                        "results": []
                    }
        
        # Create tasks for all queries
        tasks = [_execute_single_search(query) for query in queries]
        
        # Execute all searches in parallel with concurrency control
        results = await asyncio.gather(*tasks)
        
        logger.info(f"Completed {len(queries)} parallel searches")
        return results

    async def search_parallel_with_engines(self, query: str, url: str, 
                                        params: Optional[Dict[str, Any]] = None,
                                        engines: Optional[List[str]] = None,
                                        consolidate: bool = True) -> Dict[str, Any]:
        """
        Execute a search using multiple engines in parallel and optionally consolidate results.
        
        This method implements parallel execution across multiple search engines,
        allowing different search strategies to be tried simultaneously for better
        results and faster overall response time.
        
        Args:
            query: The search query string
            url: The target URL to search on
            params: Additional parameters for the search
            engines: List of engine names to use (if None, use all available engines)
            consolidate: Whether to consolidate results from all engines
            
        Returns:
            Dictionary containing search results and metadata
        """
        logger.info(f"Executing parallel search across multiple engines for query: {query}")
        
        # If no specific engines requested, use all compatible engines
        if not engines:
            compatible_engines = []
            for engine in self.search_engines:
                can_handle, confidence = await engine.can_handle(url)
                if can_handle:
                    compatible_engines.append(engine)
        else:
            # Filter the requested engines that are available
            compatible_engines = [
                engine for engine in self.search_engines
                if engine.name in engines
            ]
        
        if not compatible_engines:
            return {
                "success": False,
                "error": "No compatible search engines found",
                "results": []
            }
        
        logger.info(f"Using {len(compatible_engines)} compatible engines for parallel search")
        
        # Execute searches with all compatible engines in parallel
        async def _execute_engine_search(engine):
            try:
                # Track usage
                self.extraction_stats["search_engine_usage"][engine.name] += 1
                
                # Execute search
                result = await engine.search(query, url, params)
                
                # Track success/failure
                success = result.get("success", False)
                self.update_engine_performance(engine.name, success)
                if success:
                    self.extraction_stats["search_engine_success"][engine.name()] += 1
                
                return {
                    "engine": engine.name(),
                    "result": result,
                    "success": success
                }
            except Exception as e:
                logger.error(f"Error in engine {engine.name()}: {str(e)}")
                self.update_engine_performance(engine.name(), False)
                return {
                    "engine": engine.name(),
                    "result": {
                        "success": False,
                        "error": str(e),
                        "results": []
                    },
                    "success": False
                }
        
        # Create tasks for all engines
        tasks = [_execute_engine_search(engine) for engine in compatible_engines]
        
        # Execute all engine searches in parallel
        engine_results = await asyncio.gather(*tasks)
        
        # Filter successful results
        successful_results = [r for r in engine_results if r["success"]]
        
        if not successful_results:
            logger.warning(f"No successful results from any engine for query: {query}")
            return {
                "success": False,
                "error": "All search engines failed",
                "engine_results": engine_results,
                "results": []
            }
        
        # If not consolidating, return the result from the best engine
        if not consolidate:
            # Sort by engine performance and pick the best
            successful_results.sort(
                key=lambda r: self.engine_performance.get(r["engine"], {}).get("success_rate", 0),
                reverse=True
            )
            
            best_result = successful_results[0]["result"]
            best_result["engine"] = successful_results[0]["engine"]
            
            logger.info(f"Returning results from best engine: {best_result['engine']}")
            return best_result
        
        # Consolidate results from all successful engines
        all_results = []
        seen_urls = set()
        engine_counts = {}
        
        # Process each successful engine result
        for engine_result in successful_results:
            engine_name = engine_result["engine"]
            result_data = engine_result["result"]
            
            # Track the number of results from each engine
            result_count = len(result_data.get("results", []))
            engine_counts[engine_name] = result_count
            
            # Add unique results to the consolidated list
            for item in result_data.get("results", []):
                # Extract URL for deduplication
                item_url = None
                if isinstance(item, dict):
                    item_url = item.get("url") or item.get("source_url")
                
                # Skip if we've seen this URL before
                if item_url and item_url in seen_urls:
                    continue
                    
                # Add to consolidated results
                if item_url:
                    seen_urls.add(item_url)
                
                # Add engine attribution
                if isinstance(item, dict):
                    item["engine"] = engine_name
                    
                all_results.append(item)
        
        logger.info(f"Consolidated {len(all_results)} unique results from {len(successful_results)} engines")
        
        return {
            "success": True,
            "results": all_results,
            "engine_counts": engine_counts,
            "total_engines_used": len(engine_results),
            "successful_engines": len(successful_results)
        }

    async def search_with_query_expansion(self, base_query: str, url: str,
                                       params: Optional[Dict[str, Any]] = None,
                                       max_variants: int = 3,
                                       max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Execute a search with automatic query expansion for better coverage.
        
        This method generates variations of the base query and executes them in parallel,
        then consolidates the results for more comprehensive search coverage.
        
        Args:
            base_query: The base search query to expand
            url: The target URL to search on
            params: Additional parameters for the search
            max_variants: Maximum number of query variants to generate
            max_concurrent: Maximum number of concurrent search operations
            
        Returns:
            Dictionary containing consolidated search results
        """
        logger.info(f"Executing search with query expansion for: {base_query}")
        
        # Generate query variants
        variants = [base_query]  # Always include the original query
        
        try:
            # Try to use advanced query expansion if available
            from components.search_term_generator import SearchTermGenerator
            term_generator = SearchTermGenerator()
            expanded_terms = await term_generator.generate_query_variations(
                base_query, 
                max_variants=max_variants
            )
            
            if expanded_terms and isinstance(expanded_terms, list):
                # Add unique variants
                for term in expanded_terms:
                    if term not in variants:
                        variants.append(term)
        except Exception as e:
            logger.warning(f"Error generating query variants: {str(e)}")
            
            # Fallback to simple expansion if advanced fails
            variants.extend([
                f"{base_query} {suffix}" for suffix in [
                    "review", "best", "information", "guide", "how to"
                ]
            ][:max_variants])
        
        # Limit the number of variants
        variants = variants[:max_variants + 1]  # +1 for the original query
        
        logger.info(f"Generated {len(variants)} query variants: {variants}")
        
        # Execute searches for all variants in parallel
        all_results = await self.search_parallel(
            queries=variants,
            url=url,
            params=params,
            max_concurrent=max_concurrent
        )
        
        # Consolidate results
        consolidated_results = []
        seen_urls = set()
        
        # Process results from each variant query
        for variant_result in all_results:
            if variant_result.get("success", False):
                # Extract results from this variant
                results = variant_result.get("results", [])
                
                # Add unique results to the consolidated list
                for item in results:
                    # Extract URL for deduplication
                    item_url = None
                    if isinstance(item, dict):
                        item_url = item.get("url") or item.get("source_url")
                    
                    # Skip if we've seen this URL before
                    if item_url and item_url in seen_urls:
                        continue
                        
                    # Add to consolidated results
                    if item_url:
                        seen_urls.add(item_url)
                        
                    consolidated_results.append(item)
        
        logger.info(f"Consolidated {len(consolidated_results)} unique results from {len(variants)} query variants")
        
        return {
            "success": len(consolidated_results) > 0,
            "results": consolidated_results,
            "query_variants": variants,
            "total_results_per_variant": {
                query: len(result.get("results", [])) 
                for query, result in zip(variants, all_results)
            }
        }

    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Crawl from a starting URL using this strategy.
        
        Args:
            start_url: The URL to start crawling from
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Optional dictionary with crawl results, or None if crawl failed
        """
        # For MultiStrategy, crawl is the same as execute
        return self.execute(start_url, **kwargs)

    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Extract data from HTML content using this strategy.
        
        Args:
            html_content: The HTML content to extract data from
            url: The URL the content was fetched from
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Optional dictionary with extracted data, or None if extraction failed
        """
        import asyncio
        
        # Extract data using multiple strategies
        try:
            if asyncio.get_event_loop().is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._extract_with_multi_strategy(url, html_content, kwargs.get('extraction_config')))
                    result, confidence, strategy_used = future.result()
            else:
                result, confidence, strategy_used = asyncio.run(self._extract_with_multi_strategy(url, html_content, kwargs.get('extraction_config')))
            
            return {
                'data': result,
                'confidence': confidence,
                'strategy': strategy_used
            }
        except Exception as e:
            logger.error(f"Error extracting data for {url}: {e}")
            return None

    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get all results collected by this strategy.
        
        Returns:
            List of dictionaries containing the collected results
        """
        return self.results


def create_multi_strategy(strategy_types: List[str], 
                         context: StrategyContext = None,
                         max_depth: int = 3, 
                         max_pages: int = 100,
                         fallback_threshold: float = 0.4,
                         confidence_threshold: float = 0.7,
                         use_voting: bool = True,
                         include_external: bool = False,
                         user_prompt: str = "",
                         filter_chain: Optional[Any] = None) -> MultiStrategy:
    """
    Factory function to create a multi-strategy with specified strategy types.
    
    Args:
        strategy_types: List of strategy type strings (e.g., ["dfs", "bfs", "best-first", "ai-guided"])
        context: The strategy context (optional - will create a basic one if not provided)
        max_depth: Maximum crawling depth
        max_pages: Maximum number of pages to crawl
        fallback_threshold: Threshold below which to try fallback strategies
        confidence_threshold: Threshold for accepting extraction results
        use_voting: Whether to use voting to resolve conflicts
        include_external: Whether to include external links
        user_prompt: The user's original request/prompt
        filter_chain: Filter chain to apply to URLs
        
    Returns:
        Configured MultiStrategy instance
    """
    # Create a basic context if none provided
    if not context:
        context = StrategyContext()
    
    # Create the multi-strategy instance
    multi_strategy = MultiStrategy(
        context=context,
        fallback_threshold=fallback_threshold,
        confidence_threshold=confidence_threshold,
        use_voting=use_voting,
        max_depth=max_depth,
        max_pages=max_pages,
        include_external=include_external,
        user_prompt=user_prompt,
        filter_chain=filter_chain
    )
    
    # Add strategies based on requested types
    for strategy_type in strategy_types:
        try:
            if strategy_type.lower() == "dfs":
                strategy = DFSStrategy(
                    context=context,
                    max_depth=max_depth,
                    max_pages=max_pages,
                    include_external=include_external,
                    user_prompt=user_prompt,
                    filter_chain=filter_chain
                )
                multi_strategy.add_strategy(strategy)
            elif strategy_type.lower() == "bfs":
                strategy = BFSStrategy(
                    context=context,
                    max_depth=max_depth,
                    max_pages=max_pages,
                    include_external=include_external,
                    user_prompt=user_prompt,
                    filter_chain=filter_chain
                )
                multi_strategy.add_strategy(strategy)
            elif strategy_type.lower() in ["best-first", "bestfirst"]:
                strategy = BestFirstStrategy(
                    context=context,
                    max_depth=max_depth,
                    max_pages=max_pages,
                    include_external=include_external,
                    user_prompt=user_prompt,
                    filter_chain=filter_chain
                )
                multi_strategy.add_strategy(strategy)
            elif strategy_type.lower() in ["ai-guided", "aiguided"]:
                strategy = AIGuidedStrategy(
                    context=context,
                    max_depth=max_depth,
                    max_pages=max_pages,
                    include_external=include_external,
                    user_prompt=user_prompt,
                    filter_chain=filter_chain
                )
                multi_strategy.add_strategy(strategy)
        except Exception as e:
            logger.warning(f"Could not create strategy {strategy_type}: {e}")
    
    return multi_strategy