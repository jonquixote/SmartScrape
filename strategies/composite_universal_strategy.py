"""
Composite Universal Strategy for SmartScrape

This module implements a composite strategy that combines multiple enhanced strategies
including the UniversalCrawl4AIStrategy with other specialized strategies for
comprehensive, intelligent web scraping.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass

from strategies.core.composite_strategy import CompositeStrategy
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_types import StrategyType, StrategyCapability, strategy_metadata
from strategies.base_strategy import BaseStrategy

# Import the Universal Hunter for intelligent content hunting
from intelligence.universal_hunter import UniversalHunter


@dataclass
class UniversalExtractionPlan:
    """Plan for universal extraction combining multiple strategies"""
    primary_strategy: str
    fallback_strategies: List[str]
    intent_analysis: Dict[str, Any]
    schema_definition: Optional[Any]
    progressive_collection: bool
    ai_consolidation: bool
    cache_enabled: bool


@strategy_metadata(
    strategy_type=StrategyType.SPECIAL_PURPOSE,
    capabilities={
        StrategyCapability.AI_ASSISTED,
        StrategyCapability.PROGRESSIVE_CRAWLING,
        StrategyCapability.SEMANTIC_SEARCH,
        StrategyCapability.INTENT_ANALYSIS,
        StrategyCapability.AI_SCHEMA_GENERATION,
        StrategyCapability.INTELLIGENT_URL_GENERATION,
        StrategyCapability.AI_PATHFINDING,
        StrategyCapability.EARLY_RELEVANCE_TERMINATION,
        StrategyCapability.MEMORY_ADAPTIVE,
        StrategyCapability.CIRCUIT_BREAKER,
        StrategyCapability.CONSOLIDATED_AI_PROCESSING,
        StrategyCapability.ERROR_HANDLING,
        StrategyCapability.RETRY_MECHANISM,
        StrategyCapability.CONTENT_NORMALIZATION,
        StrategyCapability.DATA_VALIDATION
    },
    description="Composite strategy that intelligently combines UniversalCrawl4AIStrategy with specialized strategies for optimal results"
)
class CompositeUniversalStrategy(CompositeStrategy):
    """
    A composite strategy that leverages the enhanced capabilities of UniversalCrawl4AIStrategy
    and other new components for intelligent, adaptive web scraping.
    
    Key Features:
    - Intelligent strategy selection based on intent analysis
    - Progressive data collection with fallback strategies
    - AI-driven schema generation and validation
    - Consolidated processing across multiple strategies
    - Adaptive strategy switching based on results
    """
    
    def __init__(self, context: Optional[StrategyContext] = None):
        """Initialize the Composite Universal Strategy"""
        super().__init__(context)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Universal Hunter for intelligent content hunting
        self.universal_hunter = UniversalHunter()
        
        # Strategy priorities and configurations
        self.strategy_priorities = {
            'universal_hunter': 0,        # Highest priority for intelligent hunting
            'universal_crawl4ai': 1,      # High priority for complex sites
            'dom_strategy': 2,            # Good for structured sites
            'api_strategy': 3,            # For API-accessible content
            'form_search_engine': 4,      # For form-based searches
            'url_param_strategy': 5       # For URL parameter searches
        }
        
        # Performance tracking
        self.strategy_performance = {}
        self.extraction_history = []
        
        # Configuration from context
        self.progressive_collection_enabled = getattr(context.config if context else None, 'PROGRESSIVE_DATA_COLLECTION', True)
        self.ai_schema_generation_enabled = getattr(context.config if context else None, 'AI_SCHEMA_GENERATION_ENABLED', True)
        self.semantic_search_enabled = getattr(context.config if context else None, 'SEMANTIC_SEARCH_ENABLED', True)
        self.intelligent_hunting_enabled = getattr(context.config if context else None, 'INTELLIGENT_HUNTING_ENABLED', True)
    
    def supports_enhanced_context(self) -> bool:
        """Indicate that this strategy supports enhanced context with intent analysis and schema"""
        return True
    
    async def search(self, query: str, url: str = None, context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Search method that accepts enhanced context and delegates to execute method.
        
        Args:
            query: Search query string
            url: Target URL (optional, can be generated from context)
            context: Enhanced context with intent analysis, schema, etc.
            **kwargs: Additional parameters
            
        Returns:
            Search results in standardized format
        """
        # Merge context into kwargs for execute method
        if context:
            kwargs.update({
                'user_prompt': query,
                'intent_analysis': context.get('intent_analysis', {}),
                'pydantic_schema': context.get('pydantic_schema'),
                'site_analysis': context.get('site_analysis', {}),
                'required_capabilities': context.get('required_capabilities', set())
            })
        else:
            kwargs['user_prompt'] = query
            
        # Use provided URL or try to determine from context
        target_url = url
        if not target_url and context and context.get('intent_analysis'):
            # Try to get URL from intent analysis or site analysis
            intent_data = context['intent_analysis']
            if 'target_urls' in intent_data and intent_data['target_urls']:
                target_url = intent_data['target_urls'][0]
        
        if not target_url:
            return {
                "success": False,
                "error": "No target URL provided or could be determined from context",
                "results": []
            }
            
        # Execute the composite strategy
        result = await self.execute(target_url, **kwargs)
        
        if result:
            return {
                "success": True,
                "results": result.get('items', result.get('results', [])),
                "metadata": {
                    "strategy": "composite_universal",
                    "strategies_used": result.get('strategies_used', []),
                    "processing_time": result.get('processing_time', 0.0),
                    "validation_status": result.get('validation_status', 'not_validated')
                }
            }
        else:
            return {
                "success": False,
                "error": "Composite strategy execution failed",
                "results": []
            }
    
    async def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute the composite universal strategy with intelligent strategy selection and coordination.
        
        Args:
            url: Target URL for extraction
            **kwargs: Additional parameters including user_prompt, intent_analysis, etc.
            
        Returns:
            Unified extraction results from optimal strategy combination
        """
        try:
            self.logger.info(f"Starting composite universal extraction for URL: {url}")
            
            # Step 1: Create extraction plan
            extraction_plan = await self._create_extraction_plan(url, kwargs)
            
            # Step 2: Execute primary strategy with enhanced capabilities
            primary_result = await self._execute_primary_strategy(url, extraction_plan, kwargs)
            
            # Step 3: Evaluate results and apply fallbacks if needed
            if not self._is_result_satisfactory(primary_result, extraction_plan):
                fallback_result = await self._execute_fallback_strategies(url, extraction_plan, kwargs)
                if fallback_result:
                    primary_result = self._merge_results(primary_result, fallback_result)
            
            # Step 4: Apply post-processing and validation
            final_result = await self._post_process_results(primary_result, extraction_plan)
            
            # Step 5: Update performance metrics
            self._update_performance_metrics(extraction_plan, final_result)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in composite universal strategy execution: {str(e)}")
            return await self._handle_execution_error(url, kwargs, e)
    
    async def _create_extraction_plan(self, url: str, kwargs: Dict[str, Any]) -> UniversalExtractionPlan:
        """
        Create a comprehensive extraction plan based on URL analysis and intent.
        
        Args:
            url: Target URL
            kwargs: Additional parameters
            
        Returns:
            UniversalExtractionPlan with strategy selection and configuration
        """
        try:
            # Analyze intent if available
            user_prompt = kwargs.get('user_prompt', '')
            intent_analysis = kwargs.get('intent_analysis', {})
            
            # Get intent analyzer from context if available
            if self.context and hasattr(self.context, 'service_registry'):
                try:
                    intent_analyzer = self.context.service_registry.get_service('intent_analyzer')
                    if intent_analyzer and user_prompt:
                        intent_analysis = intent_analyzer.analyze_intent(user_prompt)
                except Exception as e:
                    self.logger.warning(f"Could not get intent analysis: {e}")
            
            # Determine primary strategy based on URL and intent
            primary_strategy = self._select_primary_strategy(url, intent_analysis)
            
            # Determine fallback strategies
            fallback_strategies = self._select_fallback_strategies(primary_strategy, intent_analysis)
            
            # Create schema definition if AI schema generation is enabled
            schema_definition = None
            if self.ai_schema_generation_enabled and intent_analysis:
                schema_definition = await self._generate_schema_from_intent(intent_analysis)
            
            return UniversalExtractionPlan(
                primary_strategy=primary_strategy,
                fallback_strategies=fallback_strategies,
                intent_analysis=intent_analysis,
                schema_definition=schema_definition,
                progressive_collection=self.progressive_collection_enabled,
                ai_consolidation=primary_strategy == 'universal_crawl4ai',
                cache_enabled=getattr(self.context.config if self.context else None, 'REDIS_CACHE_ENABLED', False)
            )
            
        except Exception as e:
            self.logger.error(f"Error creating extraction plan: {str(e)}")
            # Return default plan
            return UniversalExtractionPlan(
                primary_strategy='universal_crawl4ai',
                fallback_strategies=['dom_strategy'],
                intent_analysis={},
                schema_definition=None,
                progressive_collection=True,
                ai_consolidation=True,
                cache_enabled=False
            )
    
    def _select_primary_strategy(self, url: str, intent_analysis: Dict[str, Any]) -> str:
        """
        Select the primary strategy based on URL characteristics and intent analysis.
        
        Args:
            url: Target URL
            intent_analysis: Results from intent analysis
            
        Returns:
            Name of the selected primary strategy
        """
        # Use Universal Hunter for content-type specific queries
        if self.intelligent_hunting_enabled and intent_analysis:
            content_type = intent_analysis.get('content_type', '').upper()
            if content_type in ['NEWS_ARTICLES', 'PRODUCT_INFORMATION', 'JOB_LISTINGS', 'REVIEWS_RATINGS']:
                self.logger.info(f"🎯 Selecting Universal Hunter for content type: {content_type}")
                return 'universal_hunter'
            
            # Also use Universal Hunter for temporal queries (latest, recent, etc.)
            temporal_context = intent_analysis.get('temporal_context', {})
            if temporal_context.get('is_temporal') and 'latest' in str(temporal_context.get('temporal_indicators', [])):
                self.logger.info("🎯 Selecting Universal Hunter for temporal query")
                return 'universal_hunter'
        
        # Default to universal_crawl4ai for complex extractions
        if intent_analysis.get('complexity', 'medium') in ['high', 'very_high']:
            return 'universal_crawl4ai'
        
        # Check for API patterns
        if any(api_indicator in url.lower() for api_indicator in ['/api/', 'api.', '.json', '.xml']):
            return 'api_strategy'
        
        # Check for form-based patterns
        if intent_analysis.get('requires_interaction', False):
            return 'form_search_engine'
        
        # Check for parameter-based searches
        if intent_analysis.get('search_type') == 'parameter_based':
            return 'url_param_strategy'
        
        # For semantic search requirements, prefer universal_crawl4ai
        if self.semantic_search_enabled and intent_analysis.get('requires_semantic_analysis', False):
            return 'universal_crawl4ai'
        
        # Default for structured content
        return 'dom_strategy'
    
    def _select_fallback_strategies(self, primary_strategy: str, intent_analysis: Dict[str, Any]) -> List[str]:
        """
        Select appropriate fallback strategies based on primary strategy and context.
        
        Args:
            primary_strategy: The selected primary strategy
            intent_analysis: Results from intent analysis
            
        Returns:
            List of fallback strategy names
        """
        fallback_strategies = []
        
        # Add Universal Hunter as fallback for content-type queries if not primary
        if primary_strategy != 'universal_hunter' and self.intelligent_hunting_enabled:
            content_type = intent_analysis.get('content_type', '').upper()
            if content_type in ['NEWS_ARTICLES', 'PRODUCT_INFORMATION', 'JOB_LISTINGS']:
                fallback_strategies.append('universal_hunter')
        
        # Always include universal_crawl4ai as fallback if not primary
        if primary_strategy != 'universal_crawl4ai':
            fallback_strategies.append('universal_crawl4ai')
        
        # Add DOM strategy as reliable fallback
        if primary_strategy != 'dom_strategy':
            fallback_strategies.append('dom_strategy')
        
        # Add API strategy if applicable
        if primary_strategy != 'api_strategy' and intent_analysis.get('api_accessible', False):
            fallback_strategies.append('api_strategy')
        
        return fallback_strategies[:2]  # Limit to 2 fallback strategies
    
    async def _execute_primary_strategy(self, url: str, plan: UniversalExtractionPlan, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute the primary strategy with enhanced parameters.
        
        Args:
            url: Target URL
            plan: Extraction plan
            kwargs: Additional parameters
            
        Returns:
            Results from primary strategy execution
        """
        try:
            primary_strategy_name = plan.primary_strategy
            
            # Handle Universal Hunter strategy
            if primary_strategy_name == 'universal_hunter':
                return await self._execute_universal_hunter(url, plan, kwargs)
            
            # Get the strategy instance
            if self.context and hasattr(self.context, 'strategy_factory'):
                strategy = self.context.strategy_factory.get_strategy(primary_strategy_name)
            else:
                # Fallback: try to get from child strategies
                strategy = self._child_strategies.get(primary_strategy_name)
            
            if not strategy:
                self.logger.warning(f"Primary strategy '{primary_strategy_name}' not available")
                return None
            
            # Enhance kwargs with plan information
            enhanced_kwargs = {
                **kwargs,
                'intent_analysis': plan.intent_analysis,
                'schema_definition': plan.schema_definition,
                'progressive_collection': plan.progressive_collection,
                'ai_consolidation': plan.ai_consolidation
            }
            
            self.logger.info(f"Executing primary strategy: {primary_strategy_name}")
            
            # Execute the strategy
            if hasattr(strategy, 'execute') and asyncio.iscoroutinefunction(strategy.execute):
                result = await strategy.execute(url, **enhanced_kwargs)
            else:
                result = strategy.execute(url, **enhanced_kwargs)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing primary strategy: {str(e)}")
            return None
    
    async def _execute_fallback_strategies(self, url: str, plan: UniversalExtractionPlan, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute fallback strategies if primary strategy results are unsatisfactory.
        
        Args:
            url: Target URL
            plan: Extraction plan
            kwargs: Additional parameters
            
        Returns:
            Results from successful fallback strategy
        """
        for fallback_name in plan.fallback_strategies:
            try:
                self.logger.info(f"Trying fallback strategy: {fallback_name}")
                
                # Get the strategy instance
                if self.context and hasattr(self.context, 'strategy_factory'):
                    strategy = self.context.strategy_factory.get_strategy(fallback_name)
                else:
                    strategy = self._child_strategies.get(fallback_name)
                
                if not strategy:
                    self.logger.warning(f"Fallback strategy '{fallback_name}' not available")
                    continue
                
                # Execute the fallback strategy
                if hasattr(strategy, 'execute') and asyncio.iscoroutinefunction(strategy.execute):
                    result = await strategy.execute(url, **kwargs)
                else:
                    result = strategy.execute(url, **kwargs)
                
                if result and self._is_result_satisfactory(result, plan):
                    self.logger.info(f"Fallback strategy '{fallback_name}' succeeded")
                    return result
                    
            except Exception as e:
                self.logger.warning(f"Fallback strategy '{fallback_name}' failed: {str(e)}")
                continue
        
        return None
    
    def _is_result_satisfactory(self, result: Optional[Dict[str, Any]], plan: UniversalExtractionPlan) -> bool:
        """
        Evaluate if the extraction result meets quality criteria.
        
        Args:
            result: Extraction result to evaluate
            plan: Extraction plan with criteria
            
        Returns:
            True if result is satisfactory, False otherwise
        """
        if not result:
            return False
        
        # Check for minimum data requirements
        extracted_items = result.get('extracted_items', [])
        if not extracted_items:
            return False
        
        # Check against schema if available
        if plan.schema_definition:
            try:
                # Validate items against schema
                valid_items = 0
                for item in extracted_items:
                    if self._validate_against_schema(item, plan.schema_definition):
                        valid_items += 1
                
                # Require at least 50% of items to be valid
                if valid_items / len(extracted_items) < 0.5:
                    return False
            except Exception as e:
                self.logger.warning(f"Schema validation error: {str(e)}")
        
        # Check for minimum content quality
        total_content_length = sum(len(str(item.get('content', ''))) for item in extracted_items)
        if total_content_length < 100:  # Minimum content threshold
            return False
        
        return True
    
    def _validate_against_schema(self, item: Dict[str, Any], schema_definition: Any) -> bool:
        """
        Validate an extracted item against the schema definition.
        
        Args:
            item: Extracted item to validate
            schema_definition: Schema to validate against
            
        Returns:
            True if item is valid, False otherwise
        """
        try:
            if hasattr(schema_definition, 'validate'):
                schema_definition.validate(item)
                return True
            elif hasattr(schema_definition, '__call__'):
                schema_definition(**item)
                return True
        except Exception:
            pass
        
        return False
    
    def _merge_results(self, primary_result: Dict[str, Any], fallback_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge results from primary and fallback strategies.
        
        Args:
            primary_result: Results from primary strategy
            fallback_result: Results from fallback strategy
            
        Returns:
            Merged results
        """
        try:
            merged_result = primary_result.copy()
            
            # Merge extracted items
            primary_items = primary_result.get('extracted_items', [])
            fallback_items = fallback_result.get('extracted_items', [])
            
            # Add unique items from fallback
            existing_content = {str(item.get('content', '')): item for item in primary_items}
            
            for item in fallback_items:
                content_key = str(item.get('content', ''))
                if content_key not in existing_content:
                    primary_items.append(item)
            
            merged_result['extracted_items'] = primary_items
            
            # Merge metadata
            merged_result['strategies_used'] = [
                primary_result.get('strategy_name', 'unknown'),
                fallback_result.get('strategy_name', 'unknown')
            ]
            
            return merged_result
            
        except Exception as e:
            self.logger.error(f"Error merging results: {str(e)}")
            return primary_result
    
    async def _post_process_results(self, result: Optional[Dict[str, Any]], plan: UniversalExtractionPlan) -> Optional[Dict[str, Any]]:
        """
        Apply post-processing and validation to extraction results.
        
        Args:
            result: Raw extraction results
            plan: Extraction plan
            
        Returns:
            Post-processed results
        """
        if not result:
            return result
        
        try:
            # Apply schema validation if available
            if plan.schema_definition and result.get('extracted_items'):
                validated_items = []
                for item in result['extracted_items']:
                    if self._validate_against_schema(item, plan.schema_definition):
                        validated_items.append(item)
                    else:
                        self.logger.debug(f"Item failed schema validation: {item}")
                
                result['extracted_items'] = validated_items
                result['validation_applied'] = True
            
            # Add plan metadata to results
            result['extraction_plan'] = {
                'primary_strategy': plan.primary_strategy,
                'fallback_strategies': plan.fallback_strategies,
                'progressive_collection': plan.progressive_collection,
                'ai_consolidation': plan.ai_consolidation
            }
            
            # Add quality metrics
            if result.get('extracted_items'):
                result['quality_metrics'] = {
                    'item_count': len(result['extracted_items']),
                    'total_content_length': sum(len(str(item.get('content', ''))) for item in result['extracted_items']),
                    'validation_passed': result.get('validation_applied', False)
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in post-processing: {str(e)}")
            return result
    
    def _update_performance_metrics(self, plan: UniversalExtractionPlan, result: Optional[Dict[str, Any]]):
        """
        Update performance metrics for strategy selection optimization.
        
        Args:
            plan: Extraction plan that was executed
            result: Final results
        """
        try:
            strategy_name = plan.primary_strategy
            success = result is not None and bool(result.get('extracted_items'))
            
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = {
                    'success_count': 0,
                    'total_attempts': 0,
                    'average_items': 0,
                    'last_success': None
                }
            
            metrics = self.strategy_performance[strategy_name]
            metrics['total_attempts'] += 1
            
            if success:
                metrics['success_count'] += 1
                metrics['last_success'] = asyncio.get_event_loop().time()
                
                # Update average items
                item_count = len(result.get('extracted_items', []))
                metrics['average_items'] = (metrics['average_items'] + item_count) / 2
            
        except Exception as e:
            self.logger.warning(f"Error updating performance metrics: {str(e)}")
    
    async def _handle_execution_error(self, url: str, kwargs: Dict[str, Any], error: Exception) -> Optional[Dict[str, Any]]:
        """
        Handle execution errors with fallback to simple strategy.
        
        Args:
            url: Target URL
            kwargs: Original parameters
            error: Exception that occurred
            
        Returns:
            Fallback results or None
        """
        self.logger.error(f"Composite strategy failed, attempting simple fallback: {str(error)}")
        
        try:
            # Try simple DOM strategy as last resort
            if self.context and hasattr(self.context, 'strategy_factory'):
                dom_strategy = self.context.strategy_factory.get_strategy('dom_strategy')
                if dom_strategy:
                    return dom_strategy.execute(url, **kwargs)
        except Exception as fallback_error:
            self.logger.error(f"Fallback strategy also failed: {str(fallback_error)}")
        
        return None
    
    async def _generate_schema_from_intent(self, intent_analysis: Dict[str, Any]) -> Optional[Any]:
        """
        Generate schema definition from intent analysis.
        
        Args:
            intent_analysis: Results from intent analysis
            
        Returns:
            Schema definition or None
        """
        try:
            # Get schema generator from context if available
            if self.context and hasattr(self.context, 'service_registry'):
                schema_generator = self.context.service_registry.get_service('schema_generator')
                if schema_generator:
                    return schema_generator.generate_schema_from_intent(intent_analysis)
        except Exception as e:
            self.logger.warning(f"Could not generate schema from intent: {str(e)}")
        
        return None
    
    @property
    def name(self) -> str:
        """Get the strategy name"""
        return "composite_universal"
    
    def capabilities(self) -> List:
        """
        Get the capabilities of this search engine.
        
        Returns:
            List of SearchEngineCapability objects
        """
        from strategies.base_strategy import SearchEngineCapability, SearchCapabilityType
        return [
            SearchEngineCapability(SearchCapabilityType.DOM_MANIPULATION, confidence=0.9, requires_browser=True),
            SearchEngineCapability(SearchCapabilityType.AJAX_HANDLING, confidence=0.9, requires_browser=True),
            SearchEngineCapability(SearchCapabilityType.INFINITE_SCROLL, confidence=0.8, requires_browser=True),
            SearchEngineCapability(SearchCapabilityType.FACETED_SEARCH, confidence=0.8, requires_browser=True),
            SearchEngineCapability(SearchCapabilityType.MULTI_STEP, confidence=0.9, requires_browser=True),
            SearchEngineCapability(SearchCapabilityType.FORM_BASED, confidence=0.7, requires_browser=True),
            SearchEngineCapability(SearchCapabilityType.API_BASED, confidence=0.6, requires_browser=False)
        ]

    async def can_handle(self, url: str, html: Optional[str] = None) -> Tuple[bool, float]:
        """
        Determine if this strategy can handle the given URL
        
        Returns:
            Tuple of (can_handle, confidence)
        """
        # Composite strategy can handle any URL with high confidence
        confidence = 0.9
        
        # Higher confidence for complex sites
        if html:
            if any(indicator in html.lower() for indicator in ['javascript', 'react', 'angular', 'vue', 'api']):
                confidence = 0.95
        
        return True, confidence
    
    def get_required_parameters(self) -> Dict[str, Any]:
        """Get required parameters for this strategy"""
        return {
            'url': {
                'type': 'string',
                'required': True,
                'description': 'Starting URL for crawling'
            },
            'user_prompt': {
                'type': 'string',
                'required': False,
                'description': 'User query for semantic analysis'
            },
            'intent_analysis': {
                'type': 'dict',
                'required': False,
                'description': 'Intent analysis results'
            },
            'max_strategies': {
                'type': 'int',
                'required': False,
                'default': 3,
                'description': 'Maximum number of strategies to use'
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
        try:
            # Use execute method for actual crawling logic
            return self.execute(start_url, **kwargs)
        except Exception as e:
            self.logger.error(f"Crawl failed for {start_url}: {str(e)}")
            return None
    
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
        try:
            # Use child strategies for extraction
            results = []
            for strategy in self.strategies:
                try:
                    if hasattr(strategy, 'extract'):
                        result = strategy.extract(html_content, url, **kwargs)
                        if result:
                            results.append(result)
                except Exception as e:
                    self.logger.warning(f"Strategy {getattr(strategy, 'name', 'unknown')} extraction failed: {str(e)}")
            
            if results:
                # Combine results from all strategies
                combined_result = {
                    'url': url,
                    'timestamp': time.time(),
                    'strategy': self.name,
                    'sub_results': results,
                    'content': results[0].get('content', '') if results else ''
                }
                return combined_result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Extraction failed for {url}: {str(e)}")
            return None
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get all results collected by this strategy.
        
        Returns:
            List of dictionaries containing the collected results
        """
        # Combine results from all child strategies
        all_results = []
        for strategy in self.strategies:
            try:
                if hasattr(strategy, 'get_results'):
                    strategy_results = strategy.get_results()
                    if strategy_results:
                        all_results.extend(strategy_results)
            except Exception as e:
                self.logger.warning(f"Failed to get results from strategy {getattr(strategy, 'name', 'unknown')}: {str(e)}")
        
        # Also return any results stored in the base class
        base_results = getattr(self, 'results', [])
        if base_results:
            all_results.extend(base_results)
            
        return all_results

# ===== Intent-Driven Strategy Selection Methods =====
    
    def select_strategy_by_intent(self, intent_analysis: Dict[str, Any], 
                                 site_analysis: Dict[str, Any] = None) -> List[str]:
        """
        Select optimal strategies based on intent analysis and site characteristics.
        
        Args:
            intent_analysis: Analysis of user intent and content type
            site_analysis: Analysis of target site characteristics
            
        Returns:
            Ordered list of strategy names to try (primary first)
        """
        content_type = intent_analysis.get('content_type', 'GENERAL')
        temporal_context = intent_analysis.get('temporal_context', {})
        is_temporal = temporal_context.get('is_temporal', False)
        specificity = intent_analysis.get('specificity_level', 'moderate')
        
        self.logger.info(f"🧠 Selecting strategies for content type: {content_type}, temporal: {is_temporal}, specificity: {specificity}")
        
        # Content-type specific strategy selection
        strategy_selection = []
        
        if content_type in ['NEWS_ARTICLES', 'PRODUCT_INFORMATION', 'JOB_LISTINGS', 'CONTACT_INFORMATION']:
            # Use Universal Hunter for specialized content types
            strategy_selection.append('universal_hunter')
            self.logger.info(f"📰 Selected Universal Hunter for {content_type}")
        
        # Add site-specific strategies based on site analysis
        if site_analysis:
            site_type = site_analysis.get('site_type', 'unknown')
            cms_platform = site_analysis.get('cms_platform', 'unknown')
            
            if site_type in ['e-commerce', 'marketplace']:
                strategy_selection.extend(['universal_crawl4ai', 'dom_strategy'])
                self.logger.info("🛍️ Added e-commerce optimized strategies")
            elif site_type in ['news', 'blog', 'media']:
                strategy_selection.extend(['universal_hunter', 'dom_strategy'])
                self.logger.info("📰 Added news/media optimized strategies")
            elif cms_platform in ['wordpress', 'drupal', 'joomla']:
                strategy_selection.extend(['dom_strategy', 'universal_crawl4ai'])
                self.logger.info(f"📝 Added CMS-optimized strategies for {cms_platform}")
        
        # Add universal strategies as fallbacks
        fallback_strategies = ['universal_crawl4ai', 'dom_strategy', 'api_strategy']
        for strategy in fallback_strategies:
            if strategy not in strategy_selection:
                strategy_selection.append(strategy)
        
        # Reorder based on performance history
        strategy_selection = self._reorder_by_performance(strategy_selection, intent_analysis)
        
        self.logger.info(f"🎯 Final strategy selection: {strategy_selection[:3]}...")
        return strategy_selection
    
    def map_domain_to_strategies(self, url: str, site_analysis: Dict[str, Any] = None) -> List[str]:
        """
        Map specific domains to optimal strategies based on known patterns.
        
        Args:
            url: Target URL
            site_analysis: Optional site analysis results
            
        Returns:
            Ordered list of optimal strategies for this domain
        """
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        
        # Domain-specific strategy mappings
        domain_strategies = {
            # E-commerce sites
            'amazon.com': ['universal_hunter', 'dom_strategy'],
            'ebay.com': ['universal_hunter', 'api_strategy', 'dom_strategy'],
            'walmart.com': ['dom_strategy', 'universal_crawl4ai'],
            'target.com': ['dom_strategy', 'universal_crawl4ai'],
            'bestbuy.com': ['dom_strategy', 'universal_crawl4ai'],
            
            # News sites
            'reuters.com': ['universal_hunter', 'dom_strategy'],
            'bbc.com': ['universal_hunter', 'dom_strategy'],
            'cnn.com': ['universal_hunter', 'dom_strategy'],
            'techcrunch.com': ['universal_hunter', 'dom_strategy'],
            'theverge.com': ['universal_hunter', 'dom_strategy'],
            
            # Job sites
            'linkedin.com': ['universal_hunter', 'universal_crawl4ai'],
            'indeed.com': ['universal_hunter', 'form_search_engine'],
            'glassdoor.com': ['universal_hunter', 'dom_strategy'],
            
            # Social media / dynamic sites
            'twitter.com': ['universal_crawl4ai', 'api_strategy'],
            'facebook.com': ['universal_crawl4ai'],
            'instagram.com': ['api_strategy', 'universal_crawl4ai'],
            
            # Generic platforms
            'wordpress.': ['dom_strategy', 'universal_crawl4ai'],
            'shopify.': ['dom_strategy', 'universal_hunter'],
            'github.com': ['api_strategy', 'dom_strategy']
        }
        
        # Check for exact domain matches
        for domain_pattern, strategies in domain_strategies.items():
            if domain_pattern in domain:
                self.logger.info(f"🏷️ Found domain-specific strategies for {domain}: {strategies}")
                return strategies
        
        # Use site analysis for generic strategy selection
        if site_analysis:
            return self._analyze_site_for_strategies(site_analysis)
        
        # Default strategy order
        return ['universal_crawl4ai', 'dom_strategy', 'universal_hunter']
    
    def create_fallback_chain(self, primary_strategies: List[str], 
                             intent_analysis: Dict[str, Any]) -> List[str]:
        """
        Create an intelligent fallback chain for when primary strategies fail.
        
        Args:
            primary_strategies: List of primary strategies
            intent_analysis: Intent analysis results
            
        Returns:
            Complete strategy chain with intelligent fallbacks
        """
        content_type = intent_analysis.get('content_type', 'GENERAL')
        
        # All available strategies
        all_strategies = [
            'universal_hunter', 'universal_crawl4ai', 'dom_strategy',
            'api_strategy', 'form_search_engine', 'url_param_strategy'
        ]
        
        # Create fallback chain
        fallback_chain = primary_strategies.copy()
        
        # Add content-type specific fallbacks
        if content_type == 'NEWS_ARTICLES':
            fallback_order = ['universal_hunter', 'dom_strategy', 'universal_crawl4ai']
        elif content_type == 'PRODUCT_INFORMATION':
            fallback_order = ['universal_hunter', 'dom_strategy', 'universal_crawl4ai', 'api_strategy']
        elif content_type == 'JOB_LISTINGS':
            fallback_order = ['universal_hunter', 'form_search_engine', 'dom_strategy']
        elif content_type == 'CONTACT_INFORMATION':
            fallback_order = ['universal_hunter', 'dom_strategy', 'universal_crawl4ai']
        else:
            fallback_order = ['universal_crawl4ai', 'dom_strategy', 'universal_hunter', 'api_strategy']
        
        # Add missing strategies from fallback order
        for strategy in fallback_order:
            if strategy not in fallback_chain and strategy in all_strategies:
                fallback_chain.append(strategy)
        
        # Ensure we have at least 3 strategies in the chain
        while len(fallback_chain) < 3:
            for strategy in all_strategies:
                if strategy not in fallback_chain:
                    fallback_chain.append(strategy)
                    break
        
        self.logger.info(f"🔗 Created fallback chain: {fallback_chain[:5]}...")
        return fallback_chain
    
    def track_strategy_performance(self, strategy_name: str, success: bool, 
                                  execution_time: float, quality_score: float = 0.0):
        """
        Track strategy performance for future optimization.
        
        Args:
            strategy_name: Name of the strategy
            success: Whether the strategy succeeded
            execution_time: Time taken to execute
            quality_score: Quality score of results (0.0-1.0)
        """
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                'total_executions': 0,
                'successes': 0,
                'avg_execution_time': 0.0,
                'avg_quality_score': 0.0,
                'success_rate': 0.0
            }
        
        perf = self.strategy_performance[strategy_name]
        perf['total_executions'] += 1
        
        if success:
            perf['successes'] += 1
        
        # Update averages
        perf['success_rate'] = perf['successes'] / perf['total_executions']
        perf['avg_execution_time'] = (
            (perf['avg_execution_time'] * (perf['total_executions'] - 1) + execution_time) /
            perf['total_executions']
        )
        perf['avg_quality_score'] = (
            (perf['avg_quality_score'] * (perf['total_executions'] - 1) + quality_score) /
            perf['total_executions']
        )
        
        self.logger.debug(f"📊 Updated performance for {strategy_name}: "
                         f"success_rate={perf['success_rate']:.2f}, "
                         f"avg_time={perf['avg_execution_time']:.2f}s")
    
    def _reorder_by_performance(self, strategies: List[str], 
                               intent_analysis: Dict[str, Any]) -> List[str]:
        """Reorder strategies based on historical performance."""
        content_type = intent_analysis.get('content_type', 'GENERAL')
        
        # Calculate performance scores for each strategy
        strategy_scores = {}
        for strategy in strategies:
            if strategy in self.strategy_performance:
                perf = self.strategy_performance[strategy]
                # Weight success rate more heavily, but consider speed and quality
                score = (
                    perf['success_rate'] * 0.6 +
                    (1.0 / max(1.0, perf['avg_execution_time'])) * 0.2 +
                    perf['avg_quality_score'] * 0.2
                )
                strategy_scores[strategy] = score
            else:
                # Default score for strategies without history
                strategy_scores[strategy] = 0.5
        
        # Sort strategies by performance score (descending)
        reordered = sorted(strategies, key=lambda s: strategy_scores.get(s, 0.0), reverse=True)
        
        if reordered != strategies:
            self.logger.info(f"📈 Reordered strategies based on performance: {reordered[:3]}...")
        
        return reordered
    
    def _analyze_site_for_strategies(self, site_analysis: Dict[str, Any]) -> List[str]:
        """Analyze site characteristics to determine optimal strategies."""
        strategies = []
        
        # Check for API availability
        if site_analysis.get('has_api', False):
            strategies.append('api_strategy')
        
        # Check for complex JavaScript
        if site_analysis.get('uses_heavy_js', False):
            strategies.append('universal_crawl4ai')
        
        # Check for forms
        if site_analysis.get('has_search_forms', False):
            strategies.append('form_search_engine')
        
        # Default strategies
        if not strategies:
            strategies = ['dom_strategy', 'universal_crawl4ai']
        
        return strategies
    
    # ===== Enhanced Strategy Execution with Intent-Driven Selection =====
    
    async def execute_with_intent_selection(self, url: str, user_prompt: str,
                                          intent_analysis: Dict[str, Any] = None,
                                          site_analysis: Dict[str, Any] = None,
                                          **kwargs) -> Dict[str, Any]:
        """
        Execute extraction with intelligent strategy selection based on intent.
        
        Args:
            url: Target URL
            user_prompt: User's query/prompt
            intent_analysis: Analysis of user intent
            site_analysis: Analysis of target site
            **kwargs: Additional parameters
            
        Returns:
            Extraction results with metadata about strategy selection
        """
        start_time = time.time()
        
        # Determine optimal strategy selection
        if intent_analysis:
            selected_strategies = self.select_strategy_by_intent(intent_analysis, site_analysis)
        else:
            selected_strategies = self.map_domain_to_strategies(url, site_analysis)
        
        # Create fallback chain
        if intent_analysis:
            strategy_chain = self.create_fallback_chain(selected_strategies, intent_analysis)
        else:
            strategy_chain = selected_strategies
        
        self.logger.info(f"🎯 Executing with strategy chain: {strategy_chain[:3]}...")
        
        # Execute strategies in order until one succeeds
        last_error = None
        strategy_attempts = []
        
        for i, strategy_name in enumerate(strategy_chain[:5]):  # Limit to 5 attempts
            strategy_start_time = time.time()
            
            try:
                self.logger.info(f"🔄 Attempting strategy {i+1}/{min(5, len(strategy_chain))}: {strategy_name}")
                
                # Execute the strategy
                if strategy_name == 'universal_hunter':
                    result = await self._execute_universal_hunter(url, user_prompt, intent_analysis, **kwargs)
                else:
                    result = await self._execute_traditional_strategy(strategy_name, url, user_prompt, **kwargs)
                
                strategy_execution_time = time.time() - strategy_start_time
                
                if result and result.get('success', False) and result.get('items'):
                    # Success! Track performance and return result
                    quality_score = self._calculate_result_quality_score(result, intent_analysis)
                    self.track_strategy_performance(strategy_name, True, strategy_execution_time, quality_score)
                    
                    strategy_attempts.append({
                        'strategy': strategy_name,
                        'success': True,
                        'execution_time': strategy_execution_time,
                        'quality_score': quality_score
                    })
                    
                    result['metadata'] = result.get('metadata', {})
                    result['metadata'].update({
                        'successful_strategy': strategy_name,
                        'strategy_attempts': strategy_attempts,
                        'total_execution_time': time.time() - start_time,
                        'strategy_selection_reason': 'intent_driven'
                    })
                    
                    self.logger.info(f"✅ Strategy {strategy_name} succeeded with {len(result.get('items', []))} results")
                    return result
                else:
                    # Strategy failed or returned no results
                    self.track_strategy_performance(strategy_name, False, strategy_execution_time, 0.0)
                    strategy_attempts.append({
                        'strategy': strategy_name,
                        'success': False,
                        'execution_time': strategy_execution_time,
                        'error': 'No results returned'
                    })
                    self.logger.warning(f"❌ Strategy {strategy_name} failed or returned no results")
                    
            except Exception as e:
                strategy_execution_time = time.time() - strategy_start_time
                self.track_strategy_performance(strategy_name, False, strategy_execution_time, 0.0)
                
                strategy_attempts.append({
                    'strategy': strategy_name,
                    'success': False,
                    'execution_time': strategy_execution_time,
                    'error': str(e)
                })
                
                last_error = e
                self.logger.error(f"❌ Strategy {strategy_name} failed with error: {e}")
                continue
        
        # All strategies failed
        self.logger.error(f"💥 All {len(strategy_attempts)} strategies failed")
        return {
            'success': False,
            'error': f'All strategies failed. Last error: {last_error}',
            'items': [],
            'metadata': {
                'strategy_attempts': strategy_attempts,
                'total_execution_time': time.time() - start_time,
                'strategy_selection_reason': 'intent_driven'
            }
        }
    
    async def _execute_universal_hunter(self, url: str, user_prompt: str,
                                       intent_analysis: Dict[str, Any] = None,
                                       **kwargs) -> Dict[str, Any]:
        """Execute the Universal Hunter strategy."""
        try:
            # Use Universal Hunter for intelligent content hunting
            targets = await self.universal_hunter.hunt_intelligently(
                query=user_prompt,
                urls=[url],
                max_targets=kwargs.get('max_results', 5),
                direct_urls=True
            )
            
            if targets:
                # Convert targets to standard format
                items = []
                for target in targets:
                    item = {
                        'title': target.title,
                        'url': target.url,
                        'content': target.content,
                        'relevance_score': target.relevance_score,
                        'quality_score': target.quality_score,
                        'content_type': target.content_type,
                        'preview': target.preview
                    }
                    items.append(item)
                
                return {
                    'success': True,
                    'items': items,
                    'extraction_type': 'universal_hunter',
                    'metadata': {
                        'total_targets': len(targets),
                        'pages_analyzed': self.universal_hunter.pages_analyzed,
                        'navigation_hops': self.universal_hunter.navigation_hops
                    }
                }
            else:
                return {'success': False, 'error': 'No targets found', 'items': []}
                
        except Exception as e:
            self.logger.error(f"Universal Hunter execution failed: {e}")
            return {'success': False, 'error': str(e), 'items': []}
    
    async def _execute_traditional_strategy(self, strategy_name: str, url: str,
                                          user_prompt: str, **kwargs) -> Dict[str, Any]:
        """Execute a traditional strategy (fallback to parent implementation)."""
        # This would call the parent class's strategy execution
        # For now, return a placeholder
        return {
            'success': False,
            'error': f'Strategy {strategy_name} not yet integrated',
            'items': []
        }
    
    def _calculate_result_quality_score(self, result: Dict[str, Any],
                                       intent_analysis: Dict[str, Any] = None) -> float:
        """Calculate quality score for extraction results."""
        if not result.get('items'):
            return 0.0
        
        items = result['items']
        scores = []
        
        for item in items:
            item_score = 0.0
            
            # Content completeness
            if item.get('title'):
                item_score += 0.3
            if item.get('content'):
                item_score += 0.4
            if item.get('url'):
                item_score += 0.1
            
            # Use individual quality scores if available
            if 'quality_score' in item:
                item_score = max(item_score, item['quality_score'])
            if 'relevance_score' in item:
                item_score = (item_score + item['relevance_score']) / 2
            
            scores.append(item_score)
        
        # Return average quality score
        return sum(scores) / len(scores) if scores else 0.0
