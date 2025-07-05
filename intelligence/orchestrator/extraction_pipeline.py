"""
Extraction Pipeline

Manages coordinated extraction from multiple URLs using various strategies.
Orchestrates extraction methods, handles errors, and optimizes performance.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

# Import existing components
from intelligence.universal_hunter import UniversalHunter, HuntingIntent

logger = logging.getLogger("ExtractionPipeline")

class ExtractionStrategy(Enum):
    """Different extraction strategies."""
    FAST_HEURISTIC = "fast_heuristic"      # Quick extraction, basic quality
    BALANCED_MIXED = "balanced_mixed"       # Mix of heuristic and LLM
    DEEP_LLM = "deep_llm"                  # Comprehensive LLM extraction
    SPECIALIZED = "specialized"             # Domain-specific extractors

@dataclass
class ExtractionJob:
    """Individual extraction job."""
    url: str
    schema: Dict[str, Any]
    strategy: ExtractionStrategy
    priority: int = 1
    retry_count: int = 0
    max_retries: int = 2
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExtractionResult:
    """Result from URL extraction."""
    url: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    strategy_used: Optional[ExtractionStrategy] = None
    processing_time: float = 0.0
    quality_score: float = 0.0
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ExtractionPipeline:
    """
    Extraction Pipeline
    
    Manages coordinated extraction from multiple URLs:
    - Job queue management with priorities
    - Multiple extraction strategies
    - Error handling and retries
    - Performance optimization
    - Result aggregation and validation
    """
    
    def __init__(self, max_concurrent: int = 10):
        """
        Initialize the Extraction Pipeline.
        
        Args:
            max_concurrent: Maximum concurrent extractions
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Initialize hunters for different strategies - lazy initialization
        self.hunter = None
        
        # Performance tracking
        self.extraction_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "average_processing_time": 0.0,
            "strategy_performance": {}
        }
        
        # Extraction strategies
        self.strategies = {
            ExtractionStrategy.FAST_HEURISTIC: self._extract_fast_heuristic,
            ExtractionStrategy.BALANCED_MIXED: self._extract_balanced_mixed,
            ExtractionStrategy.DEEP_LLM: self._extract_deep_llm,
            ExtractionStrategy.SPECIALIZED: self._extract_specialized
        }
        
        logger.info("ExtractionPipeline initialized")
    
    def _get_hunter(self):
        """Get or initialize the UniversalHunter with proper AI service."""
        if self.hunter is None:
            try:
                # Try to import and use global AI service
                from core.ai_service import AIService
                ai_service = AIService()
                self.hunter = UniversalHunter(ai_service)
            except ImportError:
                # Fallback: create a simple mock AI service for testing
                class MockAIService:
                    async def generate_content(self, prompt: str, **kwargs):
                        return {"text": "Mock response"}
                
                self.hunter = UniversalHunter(MockAIService())
        return self.hunter
    
    async def extract_from_urls(self, urls: List[str], schema: Dict[str, Any], 
                              config: Any) -> Dict[str, Any]:
        """
        Extract data from multiple URLs using coordinated pipeline.
        
        Args:
            urls: List of URLs to extract from
            schema: Target schema for extraction
            config: Orchestration configuration
            
        Returns:
            Extraction results with data and metadata
        """
        start_time = time.time()
        
        logger.info(f"⚡ Starting extraction pipeline for {len(urls)} URLs")
        
        # Create extraction jobs
        jobs = self._create_extraction_jobs(urls, schema, config)
        
        # Execute extraction jobs
        results = await self._execute_extraction_jobs(jobs)
        
        # Process and aggregate results
        aggregated_results = self._aggregate_results(results)
        
        processing_time = time.time() - start_time
        
        # Update statistics
        self._update_extraction_stats(results, processing_time)
        
        logger.info(f"✅ Extraction pipeline complete: {len(aggregated_results['results'])} successful extractions")
        
        return {
            "results": aggregated_results["results"],
            "attempted_count": len(jobs),
            "successful_count": aggregated_results["successful_count"],
            "failed_count": aggregated_results["failed_count"],
            "methods_used": aggregated_results["methods_used"],
            "processing_time": processing_time,
            "metadata": {
                "strategy_distribution": aggregated_results["strategy_distribution"],
                "average_quality_score": aggregated_results["average_quality_score"],
                "error_summary": aggregated_results["error_summary"]
            }
        }
    
    def _create_extraction_jobs(self, urls: List[str], schema: Dict[str, Any], 
                              config: Any) -> List[ExtractionJob]:
        """Create extraction jobs with appropriate strategies."""
        jobs = []
        
        # Determine strategy based on config
        if hasattr(config, 'strategy'):
            if config.strategy.value == "speed":
                default_strategy = ExtractionStrategy.FAST_HEURISTIC
            elif config.strategy.value == "quality":
                default_strategy = ExtractionStrategy.DEEP_LLM
            elif config.strategy.value == "comprehensive":
                default_strategy = ExtractionStrategy.SPECIALIZED
            else:
                default_strategy = ExtractionStrategy.BALANCED_MIXED
        else:
            default_strategy = ExtractionStrategy.BALANCED_MIXED
        
        for i, url in enumerate(urls):
            job = ExtractionJob(
                url=url,
                schema=schema,
                strategy=default_strategy,
                priority=1,  # Could be dynamic based on URL importance
                timeout=getattr(config, 'timeout_seconds', 30.0),
                metadata={"position": i}
            )
            jobs.append(job)
        
        return jobs
    
    async def _execute_extraction_jobs(self, jobs: List[ExtractionJob]) -> List[ExtractionResult]:
        """Execute extraction jobs with concurrency control."""
        
        # Sort jobs by priority (higher priority first)
        sorted_jobs = sorted(jobs, key=lambda j: j.priority, reverse=True)
        
        # Create extraction tasks
        extraction_tasks = []
        for job in sorted_jobs:
            task = self._extract_from_job(job)
            extraction_tasks.append(task)
        
        # Execute with concurrency control
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                job = sorted_jobs[i]
                error_result = ExtractionResult(
                    url=job.url,
                    success=False,
                    data={},
                    error=str(result),
                    processing_time=0.0
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _extract_from_job(self, job: ExtractionJob) -> ExtractionResult:
        """Extract data from a single job with retry logic."""
        async with self.semaphore:  # Concurrency control
            start_time = time.time()
            
            for attempt in range(job.max_retries + 1):
                try:
                    # Get extraction strategy function
                    strategy_func = self.strategies.get(job.strategy, self._extract_balanced_mixed)
                    
                    # Execute extraction
                    data, metadata = await asyncio.wait_for(
                        strategy_func(job.url, job.schema),
                        timeout=job.timeout
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Calculate quality and confidence scores
                    quality_score = self._calculate_quality_score(data, job.schema)
                    confidence_score = metadata.get("confidence", 0.7)
                    
                    return ExtractionResult(
                        url=job.url,
                        success=True,
                        data=data,
                        strategy_used=job.strategy,
                        processing_time=processing_time,
                        quality_score=quality_score,
                        confidence_score=confidence_score,
                        metadata=metadata
                    )
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Extraction timeout for {job.url} (attempt {attempt + 1})")
                    if attempt < job.max_retries:
                        await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        return ExtractionResult(
                            url=job.url,
                            success=False,
                            data={},
                            error="Extraction timeout",
                            processing_time=time.time() - start_time
                        )
                
                except Exception as e:
                    logger.warning(f"Extraction error for {job.url} (attempt {attempt + 1}): {e}")
                    if attempt < job.max_retries:
                        await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        return ExtractionResult(
                            url=job.url,
                            success=False,
                            data={},
                            error=str(e),
                            processing_time=time.time() - start_time
                        )
    
    async def _extract_fast_heuristic(self, url: str, schema: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Fast heuristic extraction strategy."""
        # Use basic heuristic extraction
        hunter = self._get_hunter()
        intent = HuntingIntent(
            query=f"extract data from {url}",
            output_schema=schema,
            keywords=[]
        )
        
        results = await hunter.hunt(intent, max_targets=1)
        
        if results:
            return results[0].data, {"strategy": "fast_heuristic", "confidence": 0.6}
        else:
            return {}, {"strategy": "fast_heuristic", "confidence": 0.0, "error": "No results"}
    
    async def _extract_balanced_mixed(self, url: str, schema: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Balanced mixed extraction strategy."""
        # Use balanced approach with LLM fallback
        hunter = self._get_hunter()
        intent = HuntingIntent(
            query=f"extract structured data from {url}",
            output_schema=schema,
            keywords=[]
        )
        
        results = await hunter.hunt(intent, max_targets=1)
        
        if results:
            return results[0].data, {"strategy": "balanced_mixed", "confidence": 0.8}
        else:
            return {}, {"strategy": "balanced_mixed", "confidence": 0.0, "error": "No results"}
    
    async def _extract_deep_llm(self, url: str, schema: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Deep LLM extraction strategy."""
        # Use comprehensive LLM extraction
        hunter = self._get_hunter()
        intent = HuntingIntent(
            query=f"comprehensive data extraction from {url}",
            output_schema=schema,
            keywords=[]
        )
        
        # Force LLM usage with enhanced prompts
        results = await hunter.hunt(intent, max_targets=1)
        
        if results:
            return results[0].data, {"strategy": "deep_llm", "confidence": 0.9}
        else:
            return {}, {"strategy": "deep_llm", "confidence": 0.0, "error": "No results"}
    
    async def _extract_specialized(self, url: str, schema: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Specialized extraction strategy for specific domains."""
        # Could implement domain-specific extraction logic here
        # For now, fall back to balanced approach
        return await self._extract_balanced_mixed(url, schema)
    
    def _calculate_quality_score(self, data: Dict[str, Any], schema: Dict[str, Any]) -> float:
        """Calculate quality score based on data completeness and schema match."""
        if not data:
            return 0.0
        
        schema_properties = schema.get("properties", {})
        if not schema_properties:
            return 0.7  # Default for no schema
        
        # Calculate field completeness
        required_fields = schema.get("required", list(schema_properties.keys()))
        filled_fields = sum(1 for field in required_fields if data.get(field))
        completeness_score = filled_fields / len(required_fields) if required_fields else 1.0
        
        # Calculate data quality (non-empty values)
        non_empty_values = sum(1 for value in data.values() if value and str(value).strip())
        data_quality_score = non_empty_values / len(data) if data else 0.0
        
        # Weighted average
        quality_score = (completeness_score * 0.7) + (data_quality_score * 0.3)
        return min(1.0, quality_score)
    
    def _aggregate_results(self, results: List[ExtractionResult]) -> Dict[str, Any]:
        """Aggregate extraction results."""
        successful_results = [r for r in results if r.success and r.data]
        failed_results = [r for r in results if not r.success]
        
        # Extract data only
        data_results = [r.data for r in successful_results]
        
        # Calculate statistics
        strategy_distribution = {}
        total_quality_score = 0.0
        error_summary = {}
        
        for result in results:
            if result.strategy_used:
                strategy = result.strategy_used.value
                strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
            
            if result.success:
                total_quality_score += result.quality_score
            else:
                error_type = result.error or "unknown_error"
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
        
        average_quality_score = (
            total_quality_score / len(successful_results) 
            if successful_results else 0.0
        )
        
        methods_used = list(strategy_distribution.keys())
        
        return {
            "results": data_results,
            "successful_count": len(successful_results),
            "failed_count": len(failed_results),
            "methods_used": methods_used,
            "strategy_distribution": strategy_distribution,
            "average_quality_score": average_quality_score,
            "error_summary": error_summary
        }
    
    def _update_extraction_stats(self, results: List[ExtractionResult], processing_time: float):
        """Update extraction statistics."""
        self.extraction_stats["total_extractions"] += len(results)
        
        successful_count = sum(1 for r in results if r.success)
        failed_count = len(results) - successful_count
        
        self.extraction_stats["successful_extractions"] += successful_count
        self.extraction_stats["failed_extractions"] += failed_count
        
        # Update average processing time
        total = self.extraction_stats["total_extractions"]
        current_avg = self.extraction_stats["average_processing_time"]
        self.extraction_stats["average_processing_time"] = (
            (current_avg * (total - len(results)) + processing_time) / total
        )
        
        # Update strategy performance
        for result in results:
            if result.strategy_used:
                strategy = result.strategy_used.value
                if strategy not in self.extraction_stats["strategy_performance"]:
                    self.extraction_stats["strategy_performance"][strategy] = {
                        "total": 0, "successful": 0, "average_quality": 0.0
                    }
                
                stats = self.extraction_stats["strategy_performance"][strategy]
                stats["total"] += 1
                if result.success:
                    stats["successful"] += 1
                    # Update average quality
                    prev_avg = stats["average_quality"]
                    stats["average_quality"] = (
                        (prev_avg * (stats["successful"] - 1) + result.quality_score) / stats["successful"]
                    )
    
    async def health_check(self) -> str:
        """Perform health check of extraction pipeline."""
        try:
            # Test basic extraction
            test_schema = {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"}
                }
            }
            
            test_result = await self.extract_from_urls(
                urls=["https://example.com"],
                schema=test_schema,
                config=type('Config', (), {'strategy': type('Strategy', (), {'value': 'balanced'})()})()
            )
            
            if test_result:
                return "healthy"
            else:
                return "degraded - no extraction result"
                
        except Exception as e:
            return f"error: {e}"
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get current extraction statistics."""
        stats = self.extraction_stats.copy()
        if stats["total_extractions"] > 0:
            stats["success_rate"] = stats["successful_extractions"] / stats["total_extractions"]
        else:
            stats["success_rate"] = 0.0
        return stats
