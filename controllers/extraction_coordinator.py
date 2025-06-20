"""
Extraction Coordinator for SmartScrape

This module provides the central coordination layer for extraction operations,
implementing multi-page data aggregation, AI-driven schema generation,
unified output processing, and comprehensive validation.

The coordinator manages the entire extraction workflow from intent analysis
to final output validation, ensuring coherent and unified results.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict
import hashlib

# Import core components
from config import (
    AI_SCHEMA_GENERATION_ENABLED, PYDANTIC_VALIDATION_ENABLED,
    REDIS_CACHE_ENABLED, REDIS_CONFIG, CACHE_TTL, get_config, USE_DUCKDUCKGO_BY_DEFAULT,
    PREFER_SEARCH_OVER_AI_URLS, FORCE_DUCKDUCKGO_ONLY
)

# Import performance monitoring
from utils.memory_monitor import memory_monitor, monitor_memory_during_extraction
from core.performance_optimizer import PerformanceOptimizer

# Import error handling and metrics
from utils.error_handler import error_handler, ErrorContext, ErrorType
from monitoring.metrics_collector import metrics_collector, ExtractionMetrics

# Import enhanced components
from components.universal_intent_analyzer import UniversalIntentAnalyzer
from components.intelligent_url_generator import IntelligentURLGenerator
from components.ai_schema_generator import AISchemaGenerator
# Import get_adaptive_scraper conditionally to avoid circular imports

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExtractionCoordinator")

# Import DuckDuckGo URL generator as alternative
try:
    from components.duckduckgo_url_generator import DuckDuckGoURLGenerator
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False
    logger.warning("DuckDuckGo URL generator not available")

# Optional Redis import for caching
try:
    import redis
    from redis.exceptions import ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, caching will be disabled")


class ExtractionCoordinator:
    """
    Coordinates extraction operations with AI-driven schema generation,
    multi-page data aggregation, and unified output processing.
    """

    # Fallback chain for extraction strategies
    FALLBACK_CHAIN = [
        'universal_crawl4ai',
        'trafilatura', 
        'playwright'
    ]

    def __init__(self, cache_ttl: int = 3600, use_duckduckgo: bool = None):
        """
        Initialize the Extraction Coordinator.
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
            use_duckduckgo: Whether to use DuckDuckGo URL generator. If None, uses config default (True)
        """
        self.config = get_config()
        self.cache_ttl = cache_ttl
        
        # Force DuckDuckGo usage - permanently disable AI-based URL generation
        self.use_duckduckgo = True
        
        # Initialize core components
        self.intent_analyzer = UniversalIntentAnalyzer()
        
        # Always use DuckDuckGo URL generator - AI-based generation is permanently disabled
        if DUCKDUCKGO_AVAILABLE:
            self.url_generator = DuckDuckGoURLGenerator(self.intent_analyzer)
            logger.info("Using DuckDuckGo URL generator (AI-based URL generation permanently disabled)")
        else:
            logger.error("DuckDuckGo generator not available! System requires DuckDuckGo for URL generation.")
            raise ImportError("DuckDuckGo URL generator is required but not available")
        
        # Store adaptive_scraper reference (will be set externally to avoid circular import)
        self.adaptive_scraper = None
        
        # Initialize AI schema generator if enabled
        if self.config.get('AI_SCHEMA_GENERATION_ENABLED', AI_SCHEMA_GENERATION_ENABLED):
            self.schema_generator = AISchemaGenerator(self.intent_analyzer)
        else:
            self.schema_generator = None
            
        # Initialize Redis cache if enabled and available
        self.redis_client = None
        if (self.config.get('REDIS_CACHE_ENABLED', REDIS_CACHE_ENABLED) and 
            REDIS_AVAILABLE):
            try:
                # Use enhanced Redis configuration
                redis_config = self.config.get('REDIS_CONFIG', REDIS_CONFIG)
                self.redis_client = redis.Redis(**redis_config)
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache initialized successfully with enhanced configuration")
            except (RedisConnectionError, Exception) as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                self.redis_client = None
                
        # Initialize database manager
        self.db_manager = None
        try:
            from utils.database_manager import db_manager
            self.db_manager = db_manager
            if self.db_manager.enabled:
                logger.info("Database manager integrated successfully")
        except Exception as e:
            logger.warning(f"Database manager initialization failed: {e}")

        # Initialize performance optimizer
        try:
            self.performance_optimizer = PerformanceOptimizer()
            logger.info("Performance optimizer initialized successfully")
        except Exception as e:
            logger.warning(f"Performance optimizer initialization failed: {e}")
            self.performance_optimizer = None

        # Performance tracking
        self.extraction_metrics = {}
        self.session_cache = {}  # In-memory fallback cache
        
        logger.info("ExtractionCoordinator initialized successfully")

    def switch_url_generator(self, use_duckduckgo: bool = True):
        """
        Switch URL generator (AI-based generation permanently disabled).
        
        Args:
            use_duckduckgo: Ignored - always uses DuckDuckGo (AI generation disabled)
        """
        # AI-based URL generation is permanently disabled
        if DUCKDUCKGO_AVAILABLE:
            self.url_generator = DuckDuckGoURLGenerator(self.intent_analyzer)
            self.use_duckduckgo = True
            logger.info("URL generator confirmed as DuckDuckGo (AI generation permanently disabled)")
        else:
            logger.error("DuckDuckGo generator not available but required!")
            raise RuntimeError("DuckDuckGo URL generator is required but not available")

    def get_url_generator_info(self) -> Dict[str, Any]:
        """
        Get information about the currently active URL generator.
        
        Returns:
            Dict with generator type and capabilities
        """
        return {
            "generator_type": "DuckDuckGo" if self.use_duckduckgo else "AI-based",
            "class_name": self.url_generator.__class__.__name__,
            "duckduckgo_available": DUCKDUCKGO_AVAILABLE,
            "current_generator": self.use_duckduckgo
        }

    async def coordinate_extraction(self, query: str, options: Dict = None) -> Dict:
        """
        DISABLED - This method was causing recursive DuckDuckGo calls and fake URL generation.
        Returns empty results to prevent the recursion loops.
        """
        logger.warning("🚫 ExtractionCoordinator.coordinate_extraction() DISABLED to prevent recursion")
        logger.warning("🚫 This method was the root cause of fake DuckDuckGo URLs and infinite loops")
        logger.info(f"Query '{query}' should use SimpleScraper for non-recursive scraping")
        
        # Return empty results in expected format to avoid breaking callers
        return {
            'success': False,
            'status': 'disabled_to_prevent_recursion',
            'message': 'ExtractionCoordinator disabled - use SimpleScraper instead',
            'data': {},
            'metadata': {
                'extraction_coordinator_disabled': True,
                'query': query,
                'timestamp': time.time()
            },
            'operation_id': str(uuid.uuid4())
        }

    async def analyze_and_plan(self, query: str, options: Dict = None) -> Dict:
        """
        DISABLED to prevent recursive DuckDuckGo calls that generate fake URLs.
        This method was the source of recursive URL generation loops.
        """
        logger.warning("🚫 ExtractionCoordinator.analyze_and_plan() DISABLED to prevent recursion")
        logger.warning("🚫 This method was generating fake DuckDuckGo URLs and causing infinite loops")
        logger.info(f"Query '{query}' should use SimpleScraper instead")
        
        # Return minimal plan to avoid breaking callers
        return {
            'status': 'disabled_to_prevent_recursion',
            'message': 'ExtractionCoordinator disabled - use SimpleScraper for non-recursive scraping',
            'query': query,
            'url_candidates': [],  # Empty to prevent further processing
            'extraction_plan': {},
            'disable_recursive_url_generation': True
        }
        options = options or {}
        
        logger.info(f"Analyzing and planning extraction for query: {query}")
        
        # Step 1: Analyze user intent with enhanced semantic capabilities
        intent_analysis = self.intent_analyzer.analyze_intent(query)
        
        # Step 2: URL Generation - check if URLs are already provided to prevent recursion
        if 'target_urls' in options and options['target_urls']:
            # Use provided URLs - NO additional DuckDuckGo calls
            url_candidates = options['target_urls']
            logger.info(f"Using provided target URLs: {len(url_candidates)} URLs")
        elif options.get('disable_recursive_url_generation', False):
            # DISABLE recursive URL generation - use empty list to prevent DuckDuckGo calls
            url_candidates = []
            logger.info("Recursive URL generation disabled - using empty URL list")
        else:
            # Only call DuckDuckGo ONCE here if no URLs provided
            url_candidates = self.url_generator.generate_urls(
                query,
                intent_analysis=intent_analysis, 
                max_urls=options.get('max_urls', 10)
            )
            logger.info(f"Generated {len(url_candidates)} URLs from URL generator")
        
        # Step 3: Create extraction plan
        extraction_plan = {
            "query": query,
            "intent_analysis": intent_analysis,
            "url_candidates": url_candidates,
            "extraction_strategy": options.get('strategy', 'auto'),
            "max_pages": options.get('max_pages', 50),
            "progressive_collection": True,
            "consolidation_enabled": True,
            "validation_enabled": PYDANTIC_VALIDATION_ENABLED,
            "created_at": time.time(),
            "disable_recursive_url_generation": True  # PREVENT recursion
        }
        
        # Step 4: Generate AI schema if enabled
        if (self.schema_generator and 
            self.config.get('AI_SCHEMA_GENERATION_ENABLED', AI_SCHEMA_GENERATION_ENABLED)):
            try:
                pydantic_schema = self.schema_generator.generate_schema_from_intent(intent_analysis)
                extraction_plan['pydantic_schema'] = pydantic_schema
                logger.info("AI-generated schema included in extraction plan")
            except Exception as e:
                logger.warning(f"Failed to generate AI schema: {e}")
                extraction_plan['pydantic_schema'] = None
        else:
            extraction_plan['pydantic_schema'] = None
            
        # Step 5: Add progressive collection configuration
        extraction_plan['collection_config'] = {
            "early_relevance_termination": True,
            "memory_adaptive": True,
            "circuit_breaker_enabled": True,
            "consolidated_ai_processing": True
        }
        
        logger.info(f"Extraction plan created with {len(url_candidates)} URL candidates")
        return extraction_plan

    async def _execute_extraction_plan(self, plan: Dict) -> Dict:
        """
        DISABLED to prevent recursive extraction and fake URL processing.
        """
        logger.warning("🚫 ExtractionCoordinator._execute_extraction_plan() DISABLED")
        logger.warning("🚫 This method was creating recursive AdaptiveScraper instances")
        
        # Return empty results to prevent further processing
        return {
            'status': 'disabled_to_prevent_recursion',
            'message': 'Extraction plan execution disabled',
            'results': [],
            'total_extracted': 0,
            'collection_metadata': {
                'collection_time': time.time(),
                'plan_id': plan.get('operation_id', 'disabled'),
                'pages_attempted': 0,
                'pages_collected': 0
            }
        }
        """
        Execute the extraction plan using the adaptive scraper and progressive collection.
        
        Args:
            plan: Extraction plan from analyze_and_plan()
            
        Returns:
            Dict: Raw aggregated results from multiple pages
        """
        logger.info(f"Executing extraction plan for operation {plan.get('operation_id')}")
        
        # Prepare extraction context for adaptive scraper
        extraction_context = {
            "query": plan["query"],
            "intent_analysis": plan["intent_analysis"],
            "url_candidates": plan["url_candidates"],
            "schema": plan.get("pydantic_schema"),
            "progressive_collection": plan["progressive_collection"],
            "max_pages": plan["max_pages"],
            "strategy": plan.get("extraction_strategy", "auto")
        }
        
        # Execute using adaptive scraper with enhanced strategy selection
        if self.adaptive_scraper is None:
            # Lazy initialization if not available during constructor
            try:
                from controllers.adaptive_scraper import get_adaptive_scraper
                self.adaptive_scraper = get_adaptive_scraper()
            except ImportError as e:
                raise RuntimeError(f"AdaptiveScraper not available for extraction: {e}")
                
        raw_results = await self.adaptive_scraper.execute_search_pipeline(
            query=plan["query"],
            url=None,  # Let URL generator handle URL selection
            options={
                **extraction_context,
                "use_universal_crawl4ai_strategy": True,
                "enable_progressive_collection": True,
                "enable_consolidated_ai_processing": True
            }
        )
        
        # Add collection metadata
        raw_results["collection_metadata"] = {
            "collection_time": time.time(),
            "plan_id": plan.get("operation_id"),
            "intent_type": plan["intent_analysis"].get("primary_intent"),
            "pages_attempted": len(plan["url_candidates"]),
            "pages_collected": len(raw_results.get("results", []))
        }
        
        return raw_results

    async def _consolidate_multi_page_data(self, raw_results: Dict, plan: Dict) -> Dict:
        """
        Consolidate and process data aggregated from multiple pages.
        
        Args:
            raw_results: Raw results from extraction execution
            plan: Original extraction plan
            
        Returns:
            Dict: Consolidated results ready for validation
        """
        logger.info("Consolidating multi-page data")
        
        if not raw_results.get("results"):
            return {"consolidated_items": [], "consolidation_metadata": {"items_processed": 0}}
        
        # Extract items from all pages
        all_items = []
        pages_data = raw_results.get("results", [])
        
        for page_data in pages_data:
            if isinstance(page_data, dict) and "items" in page_data:
                all_items.extend(page_data["items"])
            elif isinstance(page_data, list):
                all_items.extend(page_data)
            elif isinstance(page_data, dict):
                all_items.append(page_data)
        
        # Phase 1: De-duplicate similar items across pages
        unique_items = await self._deduplicate_items(all_items)
        
        # Phase 2: Merge complementary information about the same entities
        merged_entities = await self._merge_entity_information(unique_items)
        
        # Phase 3: Apply relevance scoring and filtering
        scored_entities = await self._apply_relevance_scoring(merged_entities, plan)
        
        # Phase 4: Sort by relevance and apply limits
        final_items = await self._rank_and_limit_results(scored_entities, plan)
        
        consolidation_metadata = {
            "items_processed": len(all_items),
            "items_after_deduplication": len(unique_items),
            "items_after_merging": len(merged_entities),
            "final_items": len(final_items),
            "consolidation_time": time.time()
        }
        
        logger.info(f"Consolidation complete: {len(all_items)} → {len(final_items)} items")
        
        return {
            "consolidated_items": final_items,
            "consolidation_metadata": consolidation_metadata
        }

    async def _validate_against_schema(self, consolidated_results: Dict, plan: Dict) -> Dict:
        """
        Validate consolidated results against the generated Pydantic schema.
        
        Args:
            consolidated_results: Results from consolidation phase
            plan: Original extraction plan with schema
            
        Returns:
            Dict: Results with validation status and errors
        """
        if not plan.get("pydantic_schema") or not PYDANTIC_VALIDATION_ENABLED:
            logger.info("Schema validation skipped (no schema or validation disabled)")
            return {
                **consolidated_results,
                "validation_status": "skipped",
                "validation_errors": []
            }
        
        logger.info("Validating consolidated results against generated schema")
        
        items = consolidated_results.get("consolidated_items", [])
        validated_items = []
        validation_errors = []
        
        try:
            # Use schema generator for validation if available
            if self.schema_generator and hasattr(self.schema_generator, 'validate_batch'):
                validation_result = self.schema_generator.validate_batch(
                    items, 
                    plan["pydantic_schema"]
                )
                
                validated_items = validation_result.get("valid_items", [])
                validation_errors = validation_result.get("errors", [])
                
                logger.info(f"Schema validation: {len(validated_items)}/{len(items)} items valid")
            else:
                # Simple validation fallback
                validated_items = items
                validation_errors = []
                
        except Exception as e:
            logger.error(f"Error during schema validation: {e}")
            validated_items = items  # Keep original items on validation error
            validation_errors = [{"error": str(e), "type": "validation_exception"}]
        
        return {
            **consolidated_results,
            "consolidated_items": validated_items,
            "validation_status": "completed" if not validation_errors else "completed_with_errors",
            "validation_errors": validation_errors,
            "validation_summary": {
                "total_items": len(items),
                "valid_items": len(validated_items),
                "error_count": len(validation_errors)
            }
        }

    async def _create_unified_output(self, validated_results: Dict, plan: Dict) -> Dict:
        """
        Transform validated multi-page data into a cohesive, unified result.
        
        Args:
            validated_results: Results from validation phase
            plan: Original extraction plan
            
        Returns:
            Dict: Final unified output structure
        """
        logger.info("Creating unified output structure")
        
        items = validated_results.get("consolidated_items", [])
        
        # Apply final structuring based on intent type
        intent_type = plan["intent_analysis"].get("primary_intent", "general")
        
        # Structure output based on schema if available
        if plan.get("pydantic_schema"):
            structured_items = await self._apply_schema_structure(items, plan["pydantic_schema"])
        else:
            structured_items = items
        
        # Create comprehensive unified output
        unified_output = {
            "data": structured_items,
            "metadata": {
                "query": plan["query"],
                "intent_type": intent_type,
                "total_items": len(structured_items),
                "data_quality_score": await self._calculate_data_quality_score(structured_items),
                "coverage_analysis": await self._analyze_field_coverage(structured_items),
                "extraction_summary": {
                    "pages_processed": validated_results.get("consolidation_metadata", {}).get("items_processed", 0),
                    "deduplication_ratio": self._calculate_deduplication_ratio(validated_results),
                    "validation_status": validated_results.get("validation_status", "unknown")
                }
            },
            "quality_indicators": {
                "completeness": await self._assess_completeness(structured_items, plan),
                "consistency": await self._assess_consistency(structured_items),
                "relevance": await self._assess_relevance(structured_items, plan)
            }
        }
        
        logger.info(f"Unified output created with {len(structured_items)} items")
        return unified_output

    async def _deduplicate_items(self, items: List[Dict]) -> List[Dict]:
        """Remove duplicate items using multiple similarity metrics."""
        if not items:
            return []
        
        unique_items = []
        seen_hashes = set()
        
        for item in items:
            # Create content hash for exact duplicate detection
            content_str = json.dumps(item, sort_keys=True, default=str)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_items.append(item)
        
        # TODO: Add semantic similarity-based deduplication using spaCy
        
        logger.info(f"Deduplication: {len(items)} → {len(unique_items)} items")
        return unique_items

    async def _merge_entity_information(self, items: List[Dict]) -> List[Dict]:
        """Merge complementary information about the same entities."""
        if not items:
            return []
        
        # Simple merging strategy - can be enhanced with entity matching
        merged = []
        
        # Group by potential entity identifiers
        entity_groups = defaultdict(list)
        
        for item in items:
            # Use title, name, or url as entity identifier
            entity_key = (
                item.get("title", "") or 
                item.get("name", "") or 
                item.get("url", "") or 
                str(hash(str(item)))
            )
            entity_groups[entity_key].append(item)
        
        # Merge items in each group
        for entity_key, group_items in entity_groups.items():
            if len(group_items) == 1:
                merged.append(group_items[0])
            else:
                merged_item = await self._merge_items(group_items)
                merged.append(merged_item)
        
        logger.info(f"Entity merging: {len(items)} → {len(merged)} items")
        return merged

    async def _merge_items(self, items: List[Dict]) -> Dict:
        """Merge multiple items representing the same entity."""
        if not items:
            return {}
        
        merged = {}
        
        # Combine all fields, preferring non-empty values
        for item in items:
            for key, value in item.items():
                if key not in merged or not merged[key]:
                    merged[key] = value
                elif isinstance(merged[key], list) and isinstance(value, list):
                    # Combine lists and remove duplicates
                    merged[key] = list(set(merged[key] + value))
        
        # Add merge metadata
        merged["_merge_metadata"] = {
            "merged_from": len(items),
            "merge_time": time.time()
        }
        
        return merged

    async def _apply_relevance_scoring(self, items: List[Dict], plan: Dict) -> List[Dict]:
        """Apply relevance scoring to items based on query intent."""
        if not items:
            return []
        
        query = plan["query"]
        intent_analysis = plan["intent_analysis"]
        
        scored_items = []
        
        for item in items:
            # Calculate relevance score (simplified - can be enhanced with semantic similarity)
            relevance_score = await self._calculate_item_relevance(item, query, intent_analysis)
            
            item["_relevance_score"] = relevance_score
            scored_items.append(item)
        
        return scored_items

    async def _calculate_item_relevance(self, item: Dict, query: str, intent_analysis: Dict) -> float:
        """Calculate relevance score for a single item."""
        # Simple keyword-based scoring (can be enhanced with semantic similarity)
        query_terms = query.lower().split()
        
        # Extract searchable text from item
        searchable_text = " ".join([
            str(item.get("title", "")),
            str(item.get("description", "")),
            str(item.get("content", "")),
        ]).lower()
        
        # Calculate term overlap score
        matching_terms = sum(1 for term in query_terms if term in searchable_text)
        base_score = matching_terms / len(query_terms) if query_terms else 0
        
        # Boost score based on intent-specific fields
        intent_type = intent_analysis.get("primary_intent", "")
        if intent_type == "real_estate" and any(field in item for field in ["price", "bedrooms", "address"]):
            base_score *= 1.2
        elif intent_type == "e_commerce" and any(field in item for field in ["price", "rating", "product"]):
            base_score *= 1.2
        
        return min(1.0, base_score)

    async def _rank_and_limit_results(self, items: List[Dict], plan: Dict) -> List[Dict]:
        """Rank items by relevance and apply result limits."""
        if not items:
            return []
        
        # Sort by relevance score (descending)
        sorted_items = sorted(
            items, 
            key=lambda x: x.get("_relevance_score", 0), 
            reverse=True
        )
        
        # Apply limit
        max_results = plan.get("max_results", 100)
        limited_items = sorted_items[:max_results]
        
        logger.info(f"Ranking and limiting: {len(sorted_items)} → {len(limited_items)} items")
        return limited_items

    async def _apply_schema_structure(self, items: List[Dict], schema) -> List[Dict]:
        """Apply schema structure to items."""
        # Simplified schema application - enhance with actual Pydantic validation
        if not schema or not items:
            return items
        
        structured_items = []
        
        for item in items:
            # Apply basic field mapping based on schema
            structured_item = {}
            
            # Copy known fields that match schema
            for key, value in item.items():
                if not key.startswith("_"):  # Skip internal metadata
                    structured_item[key] = value
            
            structured_items.append(structured_item)
        
        return structured_items

    async def _calculate_data_quality_score(self, items: List[Dict]) -> float:
        """Calculate overall data quality score."""
        if not items:
            return 0.0
        
        # Simple quality metrics
        total_fields = sum(len(item) for item in items)
        non_empty_fields = sum(
            len([v for v in item.values() if v and str(v).strip()]) 
            for item in items
        )
        
        completeness = non_empty_fields / total_fields if total_fields > 0 else 0
        return min(1.0, completeness)

    async def _analyze_field_coverage(self, items: List[Dict]) -> Dict:
        """Analyze field coverage across all items."""
        if not items:
            return {}
        
        field_counts = defaultdict(int)
        total_items = len(items)
        
        for item in items:
            for field in item.keys():
                if item[field] and str(item[field]).strip():
                    field_counts[field] += 1
        
        coverage = {
            field: count / total_items 
            for field, count in field_counts.items()
        }
        
        return coverage

    def _calculate_deduplication_ratio(self, validated_results: Dict) -> float:
        """Calculate the deduplication ratio from consolidation metadata."""
        metadata = validated_results.get("consolidation_metadata", {})
        original = metadata.get("items_processed", 0)
        final = metadata.get("final_items", 0)
        
        if original == 0:
            return 0.0
        
        return 1.0 - (final / original)

    async def _assess_completeness(self, items: List[Dict], plan: Dict) -> float:
        """Assess data completeness based on expected fields."""
        # Simplified completeness assessment
        return await self._calculate_data_quality_score(items)

    async def _assess_consistency(self, items: List[Dict]) -> float:
        """Assess data consistency across items."""
        if not items:
            return 1.0
        
        # Check field name consistency
        all_fields = set()
        for item in items:
            all_fields.update(item.keys())
        
        field_presence = defaultdict(int)
        for item in items:
            for field in all_fields:
                if field in item:
                    field_presence[field] += 1
        
        # Calculate consistency score based on field presence uniformity
        if not all_fields:
            return 1.0
        
        consistency_scores = [
            count / len(items) for count in field_presence.values()
        ]
        
        return sum(consistency_scores) / len(consistency_scores)

    async def _assess_relevance(self, items: List[Dict], plan: Dict) -> float:
        """Assess overall relevance of results to query."""
        if not items:
            return 0.0
        
        relevance_scores = [
            item.get("_relevance_score", 0) for item in items
        ]
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

    async def _cache_results(self, query: str, results: Dict, options: Dict):
        """Cache extraction results for future use."""
        if not self.redis_client:
            # Use in-memory cache as fallback
            cache_key = hashlib.md5(f"{query}:{json.dumps(options, sort_keys=True)}".encode()).hexdigest()
            self.session_cache[cache_key] = {
                "results": results,
                "timestamp": time.time()
            }
            return
        
        try:
            cache_key = f"extraction:{hashlib.md5(query.encode()).hexdigest()}"
            cached_data = {
                "query": query,
                "results": results,
                "options": options,
                "timestamp": time.time()
            }
            
            self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(cached_data, default=str)
            )
            
            logger.info(f"Results cached with key: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")

    async def _record_extraction_metrics(self, operation_id: str, execution_time: float, results: Dict):
        """Record extraction metrics for monitoring and optimization."""
        metrics = {
            "operation_id": operation_id,
            "execution_time": execution_time,
            "items_extracted": len(results.get("data", [])),
            "data_quality_score": results.get("metadata", {}).get("data_quality_score", 0),
            "timestamp": time.time()
        }
        
        self.extraction_metrics[operation_id] = metrics
        
        # Keep only recent metrics (last 1000 operations)
        if len(self.extraction_metrics) > 1000:
            oldest_keys = sorted(self.extraction_metrics.keys())[:100]
            for key in oldest_keys:
                del self.extraction_metrics[key]

    async def _save_result_to_database(self, result: Dict, response_time: float = None):
        """Save extraction result to database"""
        if not self.db_manager or not self.db_manager.enabled:
            return
            
        try:
            # Enhance result with additional metadata
            enhanced_result = result.copy()
            enhanced_result['response_time'] = response_time
            enhanced_result['quality_score'] = self._assess_content_quality(result)
            
            # Save to database
            result_id = await self.db_manager.save_extraction_result(enhanced_result)
            if result_id:
                logger.debug(f"Saved extraction result to database with ID {result_id}")
                
        except Exception as e:
            logger.warning(f"Failed to save result to database: {e}")
    
    async def _update_database_performance(self, url: str, strategy: str, success: bool, 
                                         response_time: float, quality_score: float = None):
        """Update strategy performance in database"""
        if not self.db_manager or not self.db_manager.enabled:
            return
            
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower();
            
            await self.db_manager.update_strategy_performance(
                domain, strategy, success, response_time, quality_score
            )
            
        except Exception as e:
            logger.warning(f"Failed to update database performance: {e}")

    async def get_cached_results(self, query: str, options: Dict = None) -> Optional[Dict]:
        """Retrieve cached results if available."""
        options = options or {}
        
        if self.redis_client:
            try:
                cache_key = f"extraction:{hashlib.md5(query.encode()).hexdigest()}"
                cached_data = self.redis_client.get(cache_key)
                
                if cached_data:
                    return json.loads(cached_data)
                    
            except Exception as e:
                logger.warning(f"Failed to retrieve cached results: {e}")
        
        # Check in-memory cache
        cache_key = hashlib.md5(f"{query}:{json.dumps(options, sort_keys=True)}".encode()).hexdigest()
        cached_entry = self.session_cache.get(cache_key)
        
        if cached_entry:
            # Check if cache is still valid (1 hour default)
            if time.time() - cached_entry["timestamp"] < self.cache_ttl:
                return cached_entry["results"]
            else:
                del self.session_cache[cache_key]
        
        return None

    async def get_cached_content(self, url: str, strategy_name: str) -> Optional[Dict]:
        """Get cached extraction result"""
        if not self.redis_client:
            return None
            
        cache_key = f"extraction:{strategy_name}:{hashlib.md5(url.encode()).hexdigest()}"
        try:
            cached = await asyncio.to_thread(self.redis_client.get, cache_key)
            if cached:
                result = json.loads(cached)
                logger.info(f"Cache hit for {cache_key}")
                return result
        except Exception as e:
            logger.warning(f"Cache retrieval failed for {cache_key}: {e}")
        return None

    async def cache_content(self, url: str, strategy_name: str, result: Dict, ttl: int = None):
        """Cache extraction result"""
        if not self.redis_client:
            return
            
        cache_key = f"extraction:{strategy_name}:{hashlib.md5(url.encode()).hexdigest()}"
        if ttl is None:
            # Use appropriate TTL based on content type
            cache_ttl_config = self.config.get('CACHE_TTL', CACHE_TTL)
            ttl = cache_ttl_config.get('content', 3600)
            
        try:
            await asyncio.to_thread(
                self.redis_client.setex,
                cache_key, 
                ttl, 
                json.dumps(result, default=str)
            )
            logger.info(f"Cached content for {cache_key} with TTL {ttl}s")
        except Exception as e:
            logger.warning(f"Cache storage failed for {cache_key}: {e}")

    def get_extraction_metrics(self) -> Dict:
        """Get current extraction performance metrics."""
        if not self.extraction_metrics:
            return {}
        
        metrics_values = list(self.extraction_metrics.values())
        
        return {
            "total_operations": len(metrics_values),
            "average_execution_time": sum(m["execution_time"] for m in metrics_values) / len(metrics_values),
            "average_items_per_operation": sum(m["items_extracted"] for m in metrics_values) / len(metrics_values),
            "average_quality_score": sum(m["data_quality_score"] for m in metrics_values) / len(metrics_values),
            "recent_operations": metrics_values[-10:]  # Last 10 operations
        }

    async def shutdown(self):
        """Shutdown coordinator and cleanup resources."""
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
        
        self.session_cache.clear()
        logger.info("ExtractionCoordinator shutdown completed")

    def set_adaptive_scraper(self, adaptive_scraper):
        """
        Set the adaptive scraper reference to avoid circular imports.
        
        Args:
            adaptive_scraper: The AdaptiveScraper instance
        """
        self.adaptive_scraper = adaptive_scraper
        logger.info("AdaptiveScraper reference set in ExtractionCoordinator")

    @monitor_memory_during_extraction
    async def extract_with_fallbacks(self, url: str, **kwargs) -> Dict:
        """Extract content using fallback chain with memory monitoring, error handling, and metrics"""
        start_time = time.time()
        last_error = None
        attempt_number = 0
        
        for strategy_name in self.FALLBACK_CHAIN:
            attempt_number += 1
            strategy_start_time = time.time()
            
            try:
                logger.info(f"Attempting extraction with {strategy_name} for {url} (attempt {attempt_number})")
                
                # Check cache first
                cached = await self.get_cached_content(url, strategy_name)
                if cached and cached.get('success'):
                    logger.info(f"Cache hit for {strategy_name} strategy")
                    
                    # Record metrics for cache hit
                    metrics = ExtractionMetrics(
                        url=url,
                        strategy=strategy_name,
                        success=True,
                        response_time=time.time() - start_time,
                        content_length=len(cached.get('content', '')),
                        cache_hit=True,
                        attempt_number=attempt_number
                    )
                    metrics_collector.record_extraction(metrics)
                    
                    return cached
                
                # Try extraction with current strategy
                result = await self._extract_with_strategy(url, strategy_name, **kwargs)
                strategy_response_time = time.time() - strategy_start_time
                
                if result and result.get('success'):
                    # Cache successful result
                    await self.cache_content(url, strategy_name, result)
                    logger.info(f"Successful extraction with {strategy_name}")
                    
                    # Record successful metrics
                    metrics = ExtractionMetrics(
                        url=url,
                        strategy=strategy_name,
                        success=True,
                        response_time=strategy_response_time,
                        content_length=len(result.get('content', '')),
                        cache_hit=False,
                        attempt_number=attempt_number
                    )
                    metrics_collector.record_extraction(metrics)
                    
                    return result
                else:
                    error_msg = result.get('error', f'{strategy_name} extraction failed')
                    logger.warning(f"Strategy {strategy_name} failed: {error_msg}")
                    last_error = error_msg
                    
                    # Record failed metrics
                    metrics = ExtractionMetrics(
                        url=url,
                        strategy=strategy_name,
                        success=False,
                        response_time=strategy_response_time,
                        content_length=0,
                        error_message=error_msg,
                        cache_hit=False,
                        attempt_number=attempt_number
                    )
                    metrics_collector.record_extraction(metrics)
                    
            except Exception as e:
                strategy_response_time = time.time() - strategy_start_time
                logger.error(f"Strategy {strategy_name} threw exception: {e}")
                
                # Classify and handle error
                error_context = ErrorContext(
                    url=url,
                    strategy=strategy_name,
                    timestamp=time.time(),
                    attempt_number=attempt_number
                )
                
                error_type = error_handler.classify_error(e, error_context)
                error_handler.record_error(e, error_context, error_type)
                
                # Get recovery strategy
                recovery_strategy = error_handler.get_recovery_strategy(error_type, error_context)
                
                # Record failed metrics with error type
                metrics = ExtractionMetrics(
                    url=url,
                    strategy=strategy_name,
                    success=False,
                    response_time=strategy_response_time,
                    content_length=0,
                    error_type=error_type.value,
                    error_message=str(e),
                    cache_hit=False,
                    attempt_number=attempt_number
                )
                metrics_collector.record_extraction(metrics)
                
                last_error = str(e)
                
                # Apply recovery strategy
                if recovery_strategy.retry and attempt_number < len(self.FALLBACK_CHAIN):
                    if recovery_strategy.backoff_type == "exponential":
                        delay = recovery_strategy.delay * (2 ** (attempt_number - 1))
                    elif recovery_strategy.backoff_type == "linear":
                        delay = recovery_strategy.delay * attempt_number
                    else:  # fixed
                        delay = recovery_strategy.delay
                    
                    logger.info(f"Applying recovery delay: {delay}s")
                    await asyncio.sleep(delay)
                
                continue
        
        # All strategies failed - record final failure
        total_response_time = time.time() - start_time
        final_error = f'All fallback strategies failed. Last error: {last_error}'
        
        # Record final failure metrics
        metrics = ExtractionMetrics(
            url=url,
            strategy="fallback_chain",
            success=False,
            response_time=total_response_time,
            content_length=0,
            error_message=final_error,
            cache_hit=False,
            attempt_number=attempt_number
        )
        metrics_collector.record_extraction(metrics)
        
        return {
            'success': False,
            'error': final_error,
            'url': url,
            'strategies_attempted': self.FALLBACK_CHAIN,
            'total_attempts': attempt_number,
            'total_response_time': total_response_time
        }
    
    async def _extract_with_strategy(self, url: str, strategy_name: str, **kwargs) -> Dict:
        """Extract content using a specific strategy"""
        try:
            if strategy_name == 'trafilatura':
                from strategies.trafilatura_strategy import TrafilaturaStrategy
                strategy = TrafilaturaStrategy()
                return await strategy.extract(url, **kwargs)
                
            elif strategy_name == 'playwright':
                from strategies.playwright_strategy import PlaywrightStrategy
                strategy = PlaywrightStrategy()
                return await strategy.extract(url, **kwargs)
                
            elif strategy_name == 'universal_crawl4ai':
                # Use crawl4ai directly
                from crawl4ai import AsyncWebCrawler
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url=url)
                    return {
                        'success': result.success,
                        'content': result.markdown or result.cleaned_html or '',
                        'url': url,
                        'strategy': strategy_name,
                        'metadata': {
                            'title': result.metadata.get('title', '') if result.metadata else '',
                            'description': result.metadata.get('description', '') if result.metadata else ''
                        },
                        'word_count': len((result.markdown or result.cleaned_html or '').split()),
                        'extraction_method': 'crawl4ai'
                    }
                
            else:
                return {
                    'success': False,
                    'error': f'Unknown strategy: {strategy_name}',
                    'url': url
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Strategy {strategy_name} failed: {str(e)}',
                'url': url
            }

    @monitor_memory_during_extraction
    async def extract_with_intelligent_selection(self, url: str, **kwargs) -> Dict:
        """Extract content using intelligent strategy selection with memory monitoring"""
        from components.strategy_selector import AdaptiveStrategySelector
        from components.domain_intelligence import DomainIntelligence
        
        start_time = time.time()
        
        # Check memory before extraction
        memory_summary = memory_monitor.get_memory_summary()
        logger.info(f"Memory status before extraction: {memory_summary['status']} ({memory_summary['current_usage_mb']:.1f}MB)")
        
        try:
            # Initialize components
            strategy_selector = AdaptiveStrategySelector()
            domain_intel = DomainIntelligence()
            if not domain_intel._initialized:
                domain_intel.initialize()
            
            # Analyze domain first (if not provided)
            domain_info = kwargs.get('domain_info')
            if not domain_info:
                # Fetch HTML for analysis
                try:
                    html = await self._fetch_html_for_analysis(url)
                    if html:
                        domain_info = await domain_intel.detect_javascript_dependency(url, html)
                        # Cache the domain analysis
                        strategy_selector.cache_domain_analysis(
                            strategy_selector._extract_domain(url), 
                            domain_info
                        )
                except Exception as e:
                    logger.warning(f"Domain analysis failed for {url}: {e}")
                    domain_info = {}
            
            # Select optimal strategy
            selected_strategy = await strategy_selector.select_optimal_strategy(
                url, domain_info, kwargs.get('content_hints')
            )
            
            logger.info(f"Intelligent selection chose '{selected_strategy}' for {url}")
            
            # Try the selected strategy first
            result = await self._extract_with_strategy(url, selected_strategy, **kwargs)
            
            extraction_time = time.time() - start_time
            success = result and result.get('success', False)
            
            # Update performance metrics
            content_quality = self._assess_content_quality(result) if success else 0.0
            await strategy_selector.update_performance(
                url, selected_strategy, success, extraction_time, content_quality
            )
            
            # Save to database
            await self._save_result_to_database(result, extraction_time)
            await self._update_database_performance(
                url, selected_strategy, success, extraction_time, content_quality
            )
            
            if success:
                # Cache successful result
                await self.cache_content(url, selected_strategy, result)
                logger.info(f"Successful intelligent extraction with {selected_strategy}")
                return result
            else:
                logger.warning(f"Selected strategy {selected_strategy} failed, trying fallbacks")
                # Fall back to the fallback chain, excluding the already tried strategy
                fallback_chain = [s for s in self.FALLBACK_CHAIN if s != selected_strategy]
                return await self._try_fallback_strategies(url, fallback_chain, strategy_selector, **kwargs)
        
        except Exception as e:
            logger.error(f"Intelligent extraction failed for {url}: {e}")
            # Fall back to standard fallback chain
            return await self.extract_with_fallbacks(url, **kwargs)
    
    async def _try_fallback_strategies(self, url: str, strategies: List[str], 
                                     strategy_selector, **kwargs) -> Dict:
        """Try fallback strategies and update performance"""
        
        for strategy_name in strategies:
            try:
                start_time = time.time()
                logger.info(f"Trying fallback strategy {strategy_name} for {url}")
                
                result = await self._extract_with_strategy(url, strategy_name, **kwargs)
                extraction_time = time.time() - start_time
                success = result and result.get('success', False)
                
                # Update performance metrics
                content_quality = self._assess_content_quality(result) if success else 0.0
                await strategy_selector.update_performance(
                    url, strategy_name, success, extraction_time, content_quality
                )
                
                # Save to database
                await self._save_result_to_database(result, extraction_time)
                await self._update_database_performance(
                    url, strategy_name, success, extraction_time, content_quality
                )
                
                if success:
                    await self.cache_content(url, strategy_name, result)
                    logger.info(f"Successful fallback extraction with {strategy_name}")
                    return result
                
            except Exception as e:
                logger.error(f"Fallback strategy {strategy_name} failed: {e}")
                continue
        
        # All strategies failed
        return {
            'success': False,
            'error': 'All extraction strategies failed',
            'url': url,
            'strategies_attempted': strategies
        }
    
    async def _fetch_html_for_analysis(self, url: str) -> Optional[str]:
        """Fetch HTML content for domain analysis"""
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return await response.text()
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch HTML for analysis: {e}")
            return None
    
    def _assess_content_quality(self, result: Dict) -> float:
        """Assess the quality of extracted content"""
        if not result or not result.get('success'):
            return 0.0
        
        content = result.get('content', '')
        if not content:
            return 0.0
        
        quality_score = 0.0
        
        # Length-based scoring
        content_length = len(content)
        if content_length > 500:
            quality_score += 0.3
        elif content_length > 100:
            quality_score += 0.1
        
        # Word count scoring
        word_count = len(content.split())
        if word_count > 100:
            quality_score += 0.3
        elif word_count > 20:
            quality_score += 0.1
        
        # Metadata scoring
        metadata = result.get('metadata', {})
        if metadata.get('title'):
            quality_score += 0.2
        if metadata.get('description'):
            quality_score += 0.1
        
        # Structure scoring
        if result.get('html_content'):
            quality_score += 0.1
        
        return min(quality_score, 1.0)


# Global instance for singleton pattern
_extraction_coordinator_instance = None

def get_extraction_coordinator() -> ExtractionCoordinator:
    """
    Get or create singleton instance of ExtractionCoordinator.
    
    Returns:
        ExtractionCoordinator: The singleton instance
    """
    global _extraction_coordinator_instance
    
    if _extraction_coordinator_instance is None:
        _extraction_coordinator_instance = ExtractionCoordinator()
    
    return _extraction_coordinator_instance
