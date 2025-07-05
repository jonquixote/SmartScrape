"""
Universal Scraping Orchestrator

The central coordinator that manages the entire scraping workflow from query to results.
This orchestrator combines intent analysis, schema generation, discovery, extraction, 
quality control, and result formatting into a seamless pipeline.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

# Import core components
from intelligence.universal_hunter import UniversalHunter, HuntingIntent
from components.universal_intent_analyzer import UniversalIntentAnalyzer
from components.ai_schema_generator import AISchemaGenerator

# Import orchestrator components
from .discovery_coordinator import DiscoveryCoordinator
from .extraction_pipeline import ExtractionPipeline
from .quality_controller import QualityController

# Configure logging
logger = logging.getLogger("UniversalOrchestrator")

class OrchestrationStrategy(Enum):
    """Different orchestration strategies for different use cases."""
    SPEED_OPTIMIZED = "speed"      # Fast results, basic quality
    QUALITY_OPTIMIZED = "quality"  # Best quality, slower
    BALANCED = "balanced"          # Balance of speed and quality
    COMPREHENSIVE = "comprehensive" # Deep extraction, all sources

@dataclass
class OrchestrationConfig:
    """Configuration for orchestration behavior."""
    strategy: OrchestrationStrategy = OrchestrationStrategy.BALANCED
    max_concurrent_extractions: int = 10
    max_urls_per_source: int = 20
    timeout_seconds: int = 300
    enable_deep_extraction: bool = False
    enable_content_caching: bool = True
    quality_threshold: float = 0.7
    relevance_threshold: float = 0.6
    deduplication_threshold: float = 0.8
    
    # Plugin configurations
    discovery_plugins: List[str] = field(default_factory=list)
    extraction_plugins: List[str] = field(default_factory=list)
    quality_plugins: List[str] = field(default_factory=list)

@dataclass
class OrchestrationRequest:
    """Complete request specification for orchestrated scraping."""
    query: str
    schema_hint: Optional[Dict[str, Any]] = None
    output_format: str = "json"
    config: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    context: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class OrchestrationResult:
    """Complete orchestration result with metadata."""
    request_id: str
    query: str
    results: List[Dict[str, Any]]
    schema: Dict[str, Any]
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    quality_score: float
    confidence_score: float
    status: str = "completed"
    error_message: Optional[str] = None

class UniversalOrchestrator:
    """
    Universal Scraping Orchestrator
    
    The central brain that coordinates all aspects of intelligent scraping:
    1. Intent analysis and query understanding
    2. Dynamic schema generation
    3. Multi-source URL discovery
    4. Coordinated extraction pipeline
    5. Quality control and validation
    6. Result formatting and delivery
    """
    
    def __init__(self,
                 intent_analyzer: Optional[UniversalIntentAnalyzer] = None,
                 schema_generator: Optional[AISchemaGenerator] = None,
                 hunter: Optional[UniversalHunter] = None):
        """
        Initialize the Universal Orchestrator.
        
        Args:
            intent_analyzer: Intent analysis component
            schema_generator: Schema generation component  
            hunter: Core hunting component
        """
        self.intent_analyzer = intent_analyzer or UniversalIntentAnalyzer()
        self.schema_generator = schema_generator or AISchemaGenerator()
        
        # Initialize hunter with lazy loading
        self._hunter = hunter
        
        # Initialize orchestrator components
        self.discovery_coordinator = DiscoveryCoordinator()
        self.extraction_pipeline = ExtractionPipeline()
        self.quality_controller = QualityController()
        
        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0.0,
            "cache_hit_rate": 0.0
        }
        
        # Plugin registry
        self.plugins = {
            "discovery": {},
            "extraction": {},
            "quality": {}
        }
        
        logger.info("UniversalOrchestrator initialized")
    
    def _get_hunter(self):
        """Get or initialize the UniversalHunter with proper AI service."""
        if self._hunter is None:
            try:
                # Try to import and use global AI service
                from core.ai_service import AIService
                ai_service = AIService()
                self._hunter = UniversalHunter(ai_service)
            except ImportError:
                # Fallback: create a simple mock AI service for testing
                class MockAIService:
                    async def generate_content(self, prompt: str, **kwargs):
                        return {"text": "Mock response"}
                
                self._hunter = UniversalHunter(MockAIService())
        return self._hunter
    
    async def orchestrate(self, request: OrchestrationRequest) -> OrchestrationResult:
        """
        Main orchestration method that coordinates the entire scraping workflow.
        
        Args:
            request: Complete orchestration request
            
        Returns:
            Complete orchestration result with data and metadata
        """
        start_time = time.time()
        request_id = request.request_id
        
        logger.info(f"🎭 Starting orchestration for request {request_id}")
        logger.info(f"Query: {request.query}")
        logger.info(f"Strategy: {request.config.strategy.value}")
        
        try:
            # Phase 1: Intent Analysis
            logger.info(f"📊 Phase 1: Analyzing intent - {request_id}")
            intent_result = await self._analyze_intent(request)
            
            # Phase 2: Schema Generation  
            logger.info(f"🏗️ Phase 2: Generating schema - {request_id}")
            schema_result = await self._generate_schema(request, intent_result)
            
            # Phase 3: Discovery Coordination
            logger.info(f"🔍 Phase 3: Coordinating discovery - {request_id}")
            discovery_result = await self._coordinate_discovery(request, intent_result)
            
            # Phase 4: Extraction Pipeline
            logger.info(f"⚡ Phase 4: Running extraction pipeline - {request_id}")
            extraction_result = await self._run_extraction_pipeline(
                request, schema_result, discovery_result
            )
            
            # Phase 5: Quality Control
            logger.info(f"🎯 Phase 5: Quality control - {request_id}")
            quality_result = await self._apply_quality_control(
                request, extraction_result
            )
            
            # Phase 6: Result Assembly
            logger.info(f"📦 Phase 6: Assembling results - {request_id}")
            final_result = await self._assemble_results(
                request, intent_result, schema_result, 
                discovery_result, extraction_result, quality_result
            )
            
            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time, True)
            
            logger.info(f"✅ Orchestration complete - {request_id}")
            logger.info(f"Results: {len(final_result.results)} items")
            logger.info(f"Quality score: {final_result.quality_score:.2f}")
            logger.info(f"Execution time: {execution_time:.2f}s")
            
            return final_result
            
        except Exception as e:
            # Update performance stats for failure
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time, False)
            
            logger.error(f"❌ Orchestration failed - {request_id}: {e}")
            
            # Return error result
            return OrchestrationResult(
                request_id=request_id,
                query=request.query,
                results=[],
                schema={},
                metadata={"error": str(e)},
                performance_metrics={"execution_time": execution_time},
                quality_score=0.0,
                confidence_score=0.0,
                status="error",
                error_message=str(e)
            )
    
    async def _analyze_intent(self, request: OrchestrationRequest) -> Dict[str, Any]:
        """Phase 1: Analyze user intent and generate structured understanding."""
        intent_analysis = self.intent_analyzer.analyze_intent(request.query)
        
        # Enhance with context if provided
        if request.context:
            intent_analysis["context"] = request.context
        
        logger.info(f"Intent analysis: {intent_analysis.get('content_type', 'unknown')} content")
        return intent_analysis
    
    async def _generate_schema(self, request: OrchestrationRequest, 
                             intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Generate appropriate output schema."""
        # Use provided schema hint or generate from intent
        if request.schema_hint:
            schema = request.schema_hint
        else:
            # Generate schema based on intent and query
            sample_data = {
                "query": request.query,
                "content_type": intent_result.get("content_type", "general"),
                "entity_type": intent_result.get("entity_type", "unknown"),
                "expected_content": intent_result.get("sample_data")
            }
            
            schema = await self.schema_generator.generate_schema(
                sample_data=sample_data,
                schema_name=f"orchestrated_schema_{request.request_id[:8]}",
                strict_mode=request.config.strategy == OrchestrationStrategy.QUALITY_OPTIMIZED
            )
        
        logger.info(f"Generated schema with {len(schema.get('properties', {}))} fields")
        return schema
    
    async def _coordinate_discovery(self, request: OrchestrationRequest,
                                  intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Coordinate multi-source URL discovery."""
        discovery_result = await self.discovery_coordinator.discover_urls(
            query=request.query,
            intent=intent_result,
            config=request.config
        )
        
        logger.info(f"Discovered {len(discovery_result.get('urls', []))} URLs from {len(discovery_result.get('sources', []))} sources")
        return discovery_result
    
    async def _run_extraction_pipeline(self, request: OrchestrationRequest,
                                     schema: Dict[str, Any],
                                     discovery_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Run coordinated extraction pipeline."""
        extraction_result = await self.extraction_pipeline.extract_from_urls(
            urls=discovery_result.get("urls", []),
            schema=schema,
            config=request.config
        )
        
        logger.info(f"Extracted {len(extraction_result.get('results', []))} results")
        return extraction_result
    
    async def _apply_quality_control(self, request: OrchestrationRequest,
                                   extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 5: Apply quality control and validation."""
        quality_result = await self.quality_controller.validate_and_rank(
            results=extraction_result.get("results", []),
            config=request.config
        )
        
        logger.info(f"Quality control: {len(quality_result.get('validated_results', []))} items passed")
        return quality_result
    
    async def _assemble_results(self, request: OrchestrationRequest,
                              intent_result: Dict[str, Any],
                              schema_result: Dict[str, Any],
                              discovery_result: Dict[str, Any],
                              extraction_result: Dict[str, Any],
                              quality_result: Dict[str, Any]) -> OrchestrationResult:
        """Phase 6: Assemble final orchestration result."""
        
        # Calculate overall quality and confidence scores
        quality_score = quality_result.get("overall_quality_score", 0.0)
        confidence_score = quality_result.get("overall_confidence_score", 0.0)
        
        # Compile performance metrics
        performance_metrics = {
            "intent_analysis_time": intent_result.get("processing_time", 0),
            "schema_generation_time": schema_result.get("processing_time", 0),
            "discovery_time": discovery_result.get("processing_time", 0),
            "extraction_time": extraction_result.get("processing_time", 0),
            "quality_control_time": quality_result.get("processing_time", 0),
            "total_urls_discovered": len(discovery_result.get("urls", [])),
            "total_extractions_attempted": extraction_result.get("attempted_count", 0),
            "successful_extractions": len(extraction_result.get("results", [])),
            "quality_filtered_count": quality_result.get("filtered_count", 0)
        }
        
        # Compile metadata
        metadata = {
            "intent": intent_result,
            "discovery_sources": discovery_result.get("sources", []),
            "extraction_methods": extraction_result.get("methods_used", []),
            "quality_metrics": quality_result.get("quality_metrics", {}),
            "orchestration_strategy": request.config.strategy.value,
            "processing_timestamp": time.time()
        }
        
        return OrchestrationResult(
            request_id=request.request_id,
            query=request.query,
            results=quality_result.get("validated_results", []),
            schema=schema_result,
            metadata=metadata,
            performance_metrics=performance_metrics,
            quality_score=quality_score,
            confidence_score=confidence_score,
            status="completed"
        )
    
    def _update_performance_stats(self, execution_time: float, success: bool):
        """Update internal performance statistics."""
        self.performance_stats["total_requests"] += 1
        if success:
            self.performance_stats["successful_requests"] += 1
        
        # Update average response time
        total = self.performance_stats["total_requests"]
        current_avg = self.performance_stats["average_response_time"]
        self.performance_stats["average_response_time"] = (
            (current_avg * (total - 1) + execution_time) / total
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_stats.copy()
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
        else:
            stats["success_rate"] = 0.0
        return stats
    
    def register_plugin(self, plugin_type: str, name: str, plugin: Any):
        """Register a plugin for orchestration enhancement."""
        if plugin_type in self.plugins:
            self.plugins[plugin_type][name] = plugin
            logger.info(f"Registered {plugin_type} plugin: {name}")
        else:
            logger.warning(f"Unknown plugin type: {plugin_type}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of all orchestrator components."""
        health_status = {
            "orchestrator": "healthy",
            "components": {},
            "performance": self.get_performance_stats(),
            "timestamp": time.time()
        }
        
        try:
            # Check intent analyzer
            test_intent = self.intent_analyzer.analyze_intent("test query")
            health_status["components"]["intent_analyzer"] = "healthy"
        except Exception as e:
            health_status["components"]["intent_analyzer"] = f"error: {e}"
        
        try:
            # Check schema generator
            test_schema = await self.schema_generator.generate_schema(
                query="test query", schema_name="health_check"
            )
            health_status["components"]["schema_generator"] = "healthy"
        except Exception as e:
            health_status["components"]["schema_generator"] = f"error: {e}"
        
        # Check sub-components
        health_status["components"]["discovery_coordinator"] = await self.discovery_coordinator.health_check()
        health_status["components"]["extraction_pipeline"] = await self.extraction_pipeline.health_check()
        health_status["components"]["quality_controller"] = await self.quality_controller.health_check()
        
        return health_status
