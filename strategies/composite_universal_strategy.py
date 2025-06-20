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
        
        # Strategy priorities and configurations
        self.strategy_priorities = {
            'universal_crawl4ai': 1,  # Highest priority for complex sites
            'dom_strategy': 2,        # Good for structured sites
            'api_strategy': 3,        # For API-accessible content
            'form_search_engine': 4,       # For form-based searches
            'url_param_strategy': 5   # For URL parameter searches
        }
        
        # Performance tracking
        self.strategy_performance = {}
        self.extraction_history = []
        
        # Configuration from context
        self.progressive_collection_enabled = getattr(context.config if context else None, 'PROGRESSIVE_DATA_COLLECTION', True)
        self.ai_schema_generation_enabled = getattr(context.config if context else None, 'AI_SCHEMA_GENERATION_ENABLED', True)
        self.semantic_search_enabled = getattr(context.config if context else None, 'SEMANTIC_SEARCH_ENABLED', True)
    
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
