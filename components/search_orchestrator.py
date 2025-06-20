"""
Search Orchestrator Module

This module provides a centralized service for search operations across
different types of websites, intelligently selecting the most appropriate
search engine strategy for a given search task.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import json
from urllib.parse import urlparse

from strategies.base_strategy import (
    SearchEngineInterface,
    get_registered_search_engines,
    get_search_engine
)
from strategies.multi_strategy import MultiStrategy
from core.service_interface import BaseService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SearchOrchestrator")

class SearchOrchestrator(BaseService):
    """
    Orchestrates search operations across different types of websites,
    automatically selecting the most appropriate search engine for each task.
    """
    
    def __init__(self, 
                 default_engines: Optional[List[str]] = None,
                 cache_results: bool = True,
                 use_fallbacks: bool = True,
                 fallback_threshold: float = 0.4,
                 performance_tracking: bool = True):
        """
        Initialize the search orchestrator.
        
        Args:
            default_engines: List of default engine names to try (in order of preference)
            cache_results: Whether to cache search results
            use_fallbacks: Whether to use fallback engines if primary fails
            fallback_threshold: Confidence threshold below which to use fallbacks
            performance_tracking: Whether to track engine performance
        """
        self.default_engines = default_engines or ["form-search-engine", "url-param-search-engine"]
        self.cache_results = cache_results
        self.use_fallbacks = use_fallbacks
        self.fallback_threshold = fallback_threshold
        self.performance_tracking = performance_tracking
        
        # Cache for search results (url+query -> results)
        self._results_cache = {}
        
        # Performance tracking for engines (engine_name -> stats)
        self._performance_stats = {}
        
        # Flag to track initialization status
        self._initialized = False
        
        # These will be initialized in initialize()
        self._registered_engines = {}
        self._multi_strategy = None

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the service with the given configuration."""
        if self._initialized:
            return
            
        # Apply config if provided
        if config:
            self.default_engines = config.get('default_engines', self.default_engines)
            self.cache_results = config.get('cache_results', self.cache_results)
            self.use_fallbacks = config.get('use_fallbacks', self.use_fallbacks)
            self.fallback_threshold = config.get('fallback_threshold', self.fallback_threshold)
            self.performance_tracking = config.get('performance_tracking', self.performance_tracking)
            
        # Load all available search engines
        self._registered_engines = get_registered_search_engines()
        
        # Create the multi-strategy for engine selection
        self._multi_strategy = MultiStrategy(
            fallback_threshold=self.fallback_threshold,
            search_engines=[get_search_engine(engine) for engine in self.default_engines if get_search_engine(engine)]
        )
        
        self._initialized = True
        logger.info("SearchOrchestrator initialized")
    
    def shutdown(self) -> None:
        """Perform cleanup and shutdown operations."""
        self.clear_cache()
        self.clear_performance_stats()
        self._initialized = False
        logger.info("SearchOrchestrator shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "search_orchestrator"
    
    async def search(self, 
                    query: str, 
                    url: str, 
                    engine_name: Optional[str] = None,
                    params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a search operation using the most appropriate search engine.
        
        Args:
            query: The search query string
            url: The target URL to search on
            engine_name: Optional specific engine to use
            params: Additional parameters for the search
            
        Returns:
            Dictionary containing search results and metadata
        """
        # Check cache first if enabled
        cache_key = f"{url}::{query}::{json.dumps(params or {})}"
        if self.cache_results and cache_key in self._results_cache:
            logger.info(f"Cache hit for search: '{query}' on {url}")
            return self._results_cache[cache_key]
        
        # If engine specified, use it directly
        if engine_name:
            return await self._search_with_engine(engine_name, query, url, params)
        
        # Otherwise, use the multi-strategy to select best engine
        results = await self._multi_strategy.search(query, url, params)
        
        # Cache the results if successful and caching is enabled
        if self.cache_results and results.get("success", False):
            self._results_cache[cache_key] = results
        
        return results
    
    async def _search_with_engine(self, 
                                 engine_name: str, 
                                 query: str, 
                                 url: str, 
                                 params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a search with a specific engine.
        
        Args:
            engine_name: Name of the engine to use
            query: The search query string
            url: The target URL to search on
            params: Additional parameters for the search
            
        Returns:
            Dictionary containing search results and metadata
        """
        engine = get_search_engine(engine_name)
        if not engine:
            return {
                "success": False,
                "error": f"Unknown search engine: {engine_name}",
                "results": []
            }
        
        # Check if engine can handle this URL
        can_handle, confidence = await engine.can_handle(url)
        if not can_handle:
            return {
                "success": False,
                "error": f"Engine {engine_name} cannot handle URL: {url}",
                "results": []
            }
        
        # Track performance if enabled
        if self.performance_tracking:
            self._update_performance_stats(engine_name, "attempts", 1)
        
        try:
            # Execute the search
            results = await engine.search(query, url, params)
            
            # Track success/failure if enabled
            if self.performance_tracking:
                if results.get("success", False):
                    self._update_performance_stats(engine_name, "successes", 1)
                else:
                    self._update_performance_stats(engine_name, "failures", 1)
            
            # If failed and fallbacks enabled, try using other engines
            if not results.get("success", False) and self.use_fallbacks and confidence < self.fallback_threshold:
                return await self._try_fallbacks(engine_name, query, url, params)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in engine {engine_name}: {str(e)}")
            
            if self.performance_tracking:
                self._update_performance_stats(engine_name, "errors", 1)
            
            # Try fallbacks if enabled
            if self.use_fallbacks:
                return await self._try_fallbacks(engine_name, query, url, params)
            
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    async def _try_fallbacks(self, 
                            failed_engine: str, 
                            query: str, 
                            url: str, 
                            params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Try fallback engines when the primary engine fails.
        
        Args:
            failed_engine: Name of the engine that failed
            query: The search query string
            url: The target URL to search on
            params: Additional parameters for the search
            
        Returns:
            Dictionary containing search results and metadata
        """
        logger.info(f"Trying fallback engines for '{query}' on {url}")
        
        # Try each engine in default_engines list (except the one that failed)
        for engine_name in self.default_engines:
            if engine_name == failed_engine:
                continue
            
            engine = get_search_engine(engine_name)
            if not engine:
                continue
            
            # Check if engine can handle this URL
            can_handle, _ = await engine.can_handle(url)
            if not can_handle:
                continue
            
            logger.info(f"Trying fallback engine: {engine_name}")
            
            if self.performance_tracking:
                self._update_performance_stats(engine_name, "fallback_attempts", 1)
            
            try:
                # Execute the search with the fallback engine
                results = await engine.search(query, url, params)
                
                # Track success/failure
                if self.performance_tracking:
                    if results.get("success", False):
                        self._update_performance_stats(engine_name, "fallback_successes", 1)
                    else:
                        self._update_performance_stats(engine_name, "fallback_failures", 1)
                
                # If successful, return results
                if results.get("success", False):
                    return results
                
            except Exception as e:
                logger.error(f"Error in fallback engine {engine_name}: {str(e)}")
                if self.performance_tracking:
                    self._update_performance_stats(engine_name, "fallback_errors", 1)
        
        # If all fallbacks failed, return generic error
        return {
            "success": False,
            "error": "All search engines failed to complete the search",
            "results": []
        }
    
    def _update_performance_stats(self, engine_name: str, stat_name: str, increment: int = 1) -> None:
        """
        Update performance statistics for an engine.
        
        Args:
            engine_name: Name of the engine
            stat_name: Name of the statistic to update
            increment: Amount to increment the statistic by
        """
        if engine_name not in self._performance_stats:
            self._performance_stats[engine_name] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "errors": 0,
                "fallback_attempts": 0,
                "fallback_successes": 0,
                "fallback_failures": 0,
                "fallback_errors": 0
            }
        
        self._performance_stats[engine_name][stat_name] += increment
    
    def get_performance_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get performance statistics for all engines.
        
        Returns:
            Dictionary of engine name to statistics
        """
        return self._performance_stats
    
    def clear_cache(self) -> None:
        """Clear the search results cache."""
        self._results_cache.clear()
    
    def clear_performance_stats(self) -> None:
        """Clear the performance statistics."""
        self._performance_stats.clear()
    
    def get_available_engines(self) -> List[str]:
        """
        Get the names of all available search engines.
        
        Returns:
            List of engine names
        """
        return list(self._registered_engines.keys())
    
    async def analyze_site(self, url: str) -> Dict[str, Any]:
        """
        Analyze a site to determine its search capabilities and structure.
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary of site characteristics and recommended engines
        """
        # Get site characteristics using multi-strategy's analyzer
        characteristics = await self._multi_strategy._analyze_site_characteristics(url)
        
        # Determine potential engines and their confidence
        engine_scores = []
        
        for engine_name, engine_class in self._registered_engines.items():
            engine = engine_class()
            can_handle, confidence = await engine.can_handle(url, characteristics.get("html"))
            
            if can_handle:
                engine_scores.append({
                    "engine": engine_name,
                    "confidence": confidence,
                    "capabilities": [cap.to_dict() for cap in engine.capabilities]
                })
        
        # Sort by confidence (highest first)
        engine_scores.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Remove HTML from characteristics to reduce response size
        if "html" in characteristics:
            del characteristics["html"]
        
        return {
            "site_url": url,
            "characteristics": characteristics,
            "recommended_engines": engine_scores,
            "parsed_domain": urlparse(url).netloc
        }
    
    async def batch_search(self, 
                          queries: List[str], 
                          url: str, 
                          engine_name: Optional[str] = None,
                          params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute multiple search queries in parallel.
        
        Args:
            queries: List of search query strings
            url: The target URL to search on
            engine_name: Optional specific engine to use
            params: Additional parameters for the search
            
        Returns:
            List of search result dictionaries
        """
        # Create search tasks for all queries
        tasks = []
        for query in queries:
            tasks.append(self.search(query, url, engine_name, params))
        
        # Run all searches in parallel
        if tasks:
            return await asyncio.gather(*tasks)
        
        return []


# Singleton instance for global access
_orchestrator_instance = None

def get_search_orchestrator() -> SearchOrchestrator:
    """
    Get the global search orchestrator instance.
    
    Returns:
        SearchOrchestrator instance
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = SearchOrchestrator()
    return _orchestrator_instance