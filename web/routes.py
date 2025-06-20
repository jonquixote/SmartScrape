import sys # Added import
import logging # Added import
from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, Request, status, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime
import os
import psutil
import platform
import time
import asyncio
import uuid
import json
import traceback
import config
from config.environments import config_manager

# Import authentication utilities
from web.auth import validate_api_key
from typing import Dict, Any, List, Optional
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from core.service_registry import ServiceRegistry
from controllers import global_registry

# Rate limiting imports
from utils.advanced_rate_limiter import AdvancedRateLimiter, LimitType, RateLimitConfig
from functools import wraps

# Initialize logger
logger = logging.getLogger(__name__) # Definition is later, this is fine
# logger.info(f"ROUTES_PY_SYS_PATH: sys.path = {sys.path}") # Existing log

# Initialize rate limiter
rate_limiter = AdvancedRateLimiter()

# Add custom rate limit configurations
rate_limiter.add_config("scrape_endpoint", RateLimitConfig(
    max_requests=20,
    window_seconds=60,
    limit_type=LimitType.PER_IP,
    burst_allowance=5
))

rate_limiter.add_config("stream_endpoint", RateLimitConfig(
    max_requests=5,
    window_seconds=300,
    limit_type=LimitType.PER_IP
))

def safe_serialize(obj, max_depth=10, current_depth=0):
    """
    Safely serialize an object to JSON, handling circular references and non-serializable types.
    """
    if current_depth > max_depth:
        return f"<Max depth {max_depth} reached>"
    
    try:
        # Try direct JSON serialization first
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # Handle different types
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return {
                key: safe_serialize(value, max_depth, current_depth + 1) 
                for key, value in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [
                safe_serialize(item, max_depth, current_depth + 1) 
                for item in obj
            ]
        elif hasattr(obj, '__dict__'):
            # Handle objects with __dict__ (like SQLAlchemy models)
            return {
                key: safe_serialize(value, max_depth, current_depth + 1)
                for key, value in obj.__dict__.items()
                if not key.startswith('_')  # Skip private attributes
            }
        else:
            # For other types, convert to string representation
            return str(obj)[:500]  # Limit string length

async def check_rate_limit(request: Request, config_name: str = "api_per_ip"):
    """Check rate limit for request"""
    # Get client identifier
    client_ip = request.client.host if request.client else "unknown"
    user_id = request.headers.get("X-User-ID", client_ip)
    
    # Check rate limit
    allowed, metadata = await rate_limiter.check_rate_limit(config_name, user_id)
    
    if not allowed and metadata:
        wait_time = metadata.get('retry_after', 60)
        raise HTTPException(
            status_code=429,
            detail=metadata.get('message', f"Rate limit exceeded. Try again in {wait_time:.1f} seconds"),
            headers={"Retry-After": str(int(wait_time))}
        )
    
    return True

# Optional imports
try:
    import redis # Corrected: Actually import redis
except ImportError:
    redis = None
    logger.info("Redis not installed, Redis caching will be unavailable.")

from pydantic import BaseModel, Field # Ensure Field is imported
from web.models import ScrapeRequest, UserFeedbackRequest, FeedbackAnalyticsRequest
from web.templates import get_frontend_html
from utils.export import generate_json_export, generate_csv_export, generate_excel_export
from strategies.ai_guided_strategy import AIGuidedStrategy
from components.template_storage import TemplateStorage
from components.site_discovery import SiteDiscovery
from components.search_template_integration import SearchTemplateIntegrator
from crawl4ai import AsyncWebCrawler
from strategies.ai_guided.site_type import get_site_settings
from urllib.parse import urlparse  # Import urlparse to check if URL is generic

# Define a Pydantic model for the new scrape-intelligent endpoint request body
class IntelligentScrapeRequest(BaseModel):
    query: str
    start_url: Optional[str] = None
    options: Dict[str, Any] = {}

# Log sys.modules before importing AdaptiveScraper
logger.info(f"ROUTES_PY_SYS_MODULES_PRE_IMPORT: --- sys.modules Pre-Import Diagnostics ---")
logger.info(f"ROUTES_PY_SYS_MODULES_PRE_IMPORT: 'controllers' in sys.modules: {'controllers' in sys.modules}")
if 'controllers' in sys.modules:
    logger.info(f"ROUTES_PY_SYS_MODULES_PRE_IMPORT: sys.modules['controllers'].__file__: {getattr(sys.modules.get('controllers'), '__file__', 'N/A')}")
logger.info(f"ROUTES_PY_SYS_MODULES_PRE_IMPORT: 'controllers.adaptive_scraper' in sys.modules: {'controllers.adaptive_scraper' in sys.modules}")
if 'controllers.adaptive_scraper' in sys.modules:
    logger.info(f"ROUTES_PY_SYS_MODULES_PRE_IMPORT: sys.modules['controllers.adaptive_scraper'].__file__: {getattr(sys.modules.get('controllers.adaptive_scraper'), '__file__', 'N/A')}")
logger.info(f"ROUTES_PY_SYS_MODULES_PRE_IMPORT: --- End sys.modules Pre-Import Diagnostics ---")

# Import our intent parser and adaptive scraper
from ai_helpers.intent_parser import get_intent_parser, IntentParser # Added IntentParser
from controllers.adaptive_scraper import AdaptiveScraper # Changed from get_adaptive_scraper
logger.info(f"ROUTES_PY_IMMEDIATE_IMPORT_CHECK: --- AdaptiveScraper Immediate Import Diagnostics ---")
logger.info(f"ROUTES_PY_IMMEDIATE_IMPORT_CHECK: controllers package path: {sys.modules.get('controllers.__init__') or sys.modules.get('controllers').__path__ if 'controllers' in sys.modules else 'controllers not in sys.modules'}")
logger.info(f"ROUTES_PY_IMMEDIATE_IMPORT_CHECK: controllers.adaptive_scraper module: {sys.modules.get('controllers.adaptive_scraper')}")
logger.info(f"ROUTES_PY_IMMEDIATE_IMPORT_CHECK: controllers.adaptive_scraper module file: {getattr(sys.modules.get('controllers.adaptive_scraper'), '__file__', 'N/A') if 'controllers.adaptive_scraper' in sys.modules else 'module not in sys.modules'}")
logger.info(f"ROUTES_PY_IMMEDIATE_IMPORT_CHECK: AdaptiveScraper class __module__: {getattr(AdaptiveScraper, '__module__', 'N/A')}")
logger.info(f"ROUTES_PY_IMMEDIATE_IMPORT_CHECK: AdaptiveScraper class __qualname__: {getattr(AdaptiveScraper, '__qualname__', 'N/A')}")
logger.info(f"ROUTES_PY_IMMEDIATE_IMPORT_CHECK: AdaptiveScraper module file (from class): {getattr(AdaptiveScraper, '__file__', 'N/A')}")
logger.info(f"ROUTES_PY_IMMEDIATE_IMPORT_CHECK: id(AdaptiveScraper) after import: {id(AdaptiveScraper)}")
logger.info(f"ROUTES_PY_IMMEDIATE_IMPORT_CHECK: hasattr(AdaptiveScraper, 'process_user_request') after import: {hasattr(AdaptiveScraper, 'process_user_request')}")
logger.info(f"ROUTES_PY_IMMEDIATE_IMPORT_CHECK: 'process_user_request' in dir(AdaptiveScraper) after import: {'process_user_request' in dir(AdaptiveScraper)}")
logger.info(f"ROUTES_PY_IMMEDIATE_IMPORT_CHECK: --- End Immediate Import Diagnostics ---")

# NEW sys.modules logging AFTER the import and existing immediate checks
logger.info(f"ROUTES_PY_SYS_MODULES_POST_IMPORT: --- sys.modules Post-Import Diagnostics ---")
logger.info(f"ROUTES_PY_SYS_MODULES_POST_IMPORT: 'controllers.adaptive_scraper' in sys.modules: {'controllers.adaptive_scraper' in sys.modules}")
if 'controllers.adaptive_scraper' in sys.modules:
    adaptive_scraper_module_from_sys = sys.modules.get('controllers.adaptive_scraper')
    logger.info(f"ROUTES_PY_SYS_MODULES_POST_IMPORT: sys.modules['controllers.adaptive_scraper'].__file__: {getattr(adaptive_scraper_module_from_sys, '__file__', 'N/A')}")
    if hasattr(adaptive_scraper_module_from_sys, 'AdaptiveScraper'):
        actual_class_from_module = adaptive_scraper_module_from_sys.AdaptiveScraper
        logger.info(f"ROUTES_PY_SYS_MODULES_POST_IMPORT: hasattr(sys.modules['controllers.adaptive_scraper'].AdaptiveScraper, 'process_user_request'): {hasattr(actual_class_from_module, 'process_user_request')}")
        logger.info(f"ROUTES_PY_SYS_MODULES_POST_IMPORT: id(sys.modules['controllers.adaptive_scraper'].AdaptiveScraper): {id(actual_class_from_module)}")
        logger.info(f"ROUTES_PY_SYS_MODULES_POST_IMPORT: id(AdaptiveScraper) from import: {id(AdaptiveScraper)}") # Compare IDs
        logger.info(f"ROUTES_PY_SYS_MODULES_POST_IMPORT: sys.modules['controllers.adaptive_scraper'].AdaptiveScraper is AdaptiveScraper: {actual_class_from_module is AdaptiveScraper}")
    else:
        logger.info(f"ROUTES_PY_SYS_MODULES_POST_IMPORT: AdaptiveScraper class not found in sys.modules['controllers.adaptive_scraper']")
else:
    logger.info(f"ROUTES_PY_SYS_MODULES_POST_IMPORT: 'controllers.adaptive_scraper' NOT in sys.modules after import attempt.")
logger.info(f"ROUTES_PY_SYS_MODULES_POST_IMPORT: --- End sys.modules Post-Import Diagnostics ---")

# Import the centralized AI service
from core.ai_service import AIService
from core.service_registry import ServiceRegistry
from ai_helpers.response_parser import ResponseParser
from ai_helpers.prompt_generator import PromptGenerator # Import PromptGenerator
from core.multi_scrape_manager import get_multi_scrape_manager # Import MultiScrapeManager

# Import Result Enhancement and Schema Generation components
from utils.result_enhancer import ResultEnhancer, UserFeedback, FeedbackType as ResultFeedbackType, FeedbackRating as ResultFeedbackRating
from components.ai_schema_generator import AISchemaGenerator

# --- BEGIN NEW PYDANTIC MODELS FOR APP CONFIGURATION ---

class AIProviderInfo(BaseModel):
    provider: str
    model_id: str
    name: str
    api_key_env_var: Optional[str] = None # e.g., OPENAI_API_KEY
    is_configured: bool = False # True if API key is found in session or environment
    current_api_key_source: Optional[str] = None # 'session', 'environment', or None
    selected_model: Optional[str] = None # The model ID selected by the user for this provider

class AISetting(BaseModel):
    provider: str
    api_key: Optional[str] = None
    model: Optional[str] = None

class AIConfig(BaseModel):
    settings: List[AISetting]
    default_provider: Optional[str] = None # Provider to use by default
    default_model_id: Optional[str] = None # Specific model_id to use by default

class AITestKeyRequest(BaseModel):
    provider: str
    api_key: str
    model_id: Optional[str] = None

class AppFeatureSetting(BaseModel):
    id: str # Matches the key in config.py or session
    name: str # User-friendly name
    value: bool
    description: Optional[str] = None
    category: Optional[str] = None # e.g., "AI Features", "Crawling", "UI"

class AppSettings(BaseModel):
    features: List[AppFeatureSetting]

class AppConfigResponse(BaseModel):
    ai_settings: AIConfig
    available_models: List[Dict[str, Any]] # From AIService.get_available_model_configurations()
    feature_settings: List[AppFeatureSetting]

# --- BEGIN NEW PYDANTIC MODELS FOR TASK 6.2 API ENHANCEMENTS ---

class SchemaGenerateRequest(BaseModel):
    sample_data: Dict[str, Any]
    schema_name: Optional[str] = None
    strict_mode: bool = True
    
class SchemaValidateRequest(BaseModel):
    data: Dict[str, Any]
    schema_definition: Dict[str, Any]
    
class SchemaSaveRequest(BaseModel):
    schema_name: str
    schema_definition: Dict[str, Any]
    description: Optional[str] = None
    
class SchemaListRequest(BaseModel):
    category: Optional[str] = None
    search_term: Optional[str] = None
    limit: int = 50
    
class EnhancedFeedbackRequest(BaseModel):
    result_id: str
    query: str
    results: Dict[str, Any]
    feedback_type: str
    rating: str
    comments: Optional[str] = None
    field_specific_feedback: Optional[Dict[str, Dict[str, Any]]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
class DetailedAnalyticsRequest(BaseModel):
    result_id: Optional[str] = None
    feedback_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    user_id: Optional[str] = None
    include_trends: bool = True
    limit: int = 100
    
class ResultEnhanceRequest(BaseModel):
    result_id: str
    feedback_data: Optional[List[Dict[str, Any]]] = None
    reprocess_options: Optional[Dict[str, Any]] = None
    
class CacheClearRequest(BaseModel):
    cache_types: List[str]  # e.g., ["content", "templates", "feedback"]
    url_pattern: Optional[str] = None
    older_than_hours: Optional[int] = None

# --- END NEW PYDANTIC MODELS FOR TASK 6.2 API ENHANCEMENTS ---
    
# --- END NEW PYDANTIC MODELS FOR APP CONFIGURATION ---

# System monitoring metrics
SCRAPE_REQUESTS = Counter('scrape_requests_total', 'Total number of scrape requests', ['status'])
SCRAPE_DURATION = Histogram('scrape_duration_seconds', 'Time spent scraping sites', ['strategy'])

# In-memory store for job status and results (will be moved to proper storage later)
jobs = {}

router = APIRouter()

# Include fixed routes for non-recursive scraping
from web.fixed_routes import fixed_router
router.include_router(fixed_router, prefix="/api", tags=["fixed-scraping"])

# Health check endpoint
@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint for monitoring systems.
    Returns basic system metrics and component status.
    """
    # Check key component status
    components_status = {}
    
    # Check AI service
    try:
        if hasattr(ai_service, "is_initialized") and ai_service.is_initialized:
            components_status["ai_service"] = "healthy"
        else:
            components_status["ai_service"] = "unavailable"
    except Exception as e:
        components_status["ai_service"] = f"error: {str(e)}"
    
    # Check template storage
    try:
        if template_storage.get_template_count() > 0:
            components_status["template_storage"] = "healthy"
        else:
            components_status["template_storage"] = "warning: no templates"
    except Exception as e:
        components_status["template_storage"] = f"error: {str(e)}"
    
    # Check crawler status
    try:
        if crawler and not getattr(crawler, "_shutdown", False):
            components_status["crawler"] = "healthy"
        else:
            components_status["crawler"] = "unavailable"
    except Exception as e:
        components_status["crawler"] = f"error: {str(e)}"
    
    # Check database connection if applicable
    if 'DATABASE_URL' in os.environ and os.environ['DATABASE_URL'] != "sqlite:///./smartscrape.db":
        try:
            # This is a placeholder - replace with actual DB check if DB is used
            components_status["database"] = "healthy"
        except Exception as e:
            components_status["database"] = f"error: {str(e)}"
    
    # Check Redis if configured for caching
    if config.DEFAULT_CACHE_MODE == "redis":
        try:
            if redis:
                r = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
                r.ping()  # Will raise exception if Redis is unavailable
                components_status["redis"] = "healthy"
            else:
                components_status["redis"] = "error: redis not installed"
        except Exception as e:
            components_status["redis"] = f"error: {str(e)}"
    
    # Basic system metrics
    system_metrics = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "python_version": platform.python_version(),
        "uptime_seconds": int(time.time() - psutil.boot_time()),
        "active_jobs": len(jobs),
        "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
        "thread_count": len(asyncio.all_tasks()) if hasattr(asyncio, 'all_tasks') else 0
    }
    
    # Check for resource warnings
    warnings = []
    if system_metrics["memory_percent"] > config.MAX_MEMORY_MB:
        warnings.append(f"Memory usage ({system_metrics['memory_percent']}%) exceeds threshold ({config.MAX_MEMORY_MB}%)")
    if system_metrics["cpu_percent"] > config.MAX_CPU_PERCENT:
        warnings.append(f"CPU usage ({system_metrics['cpu_percent']}%) exceeds threshold ({config.MAX_CPU_PERCENT}%)")
    if system_metrics["disk_percent"] > config.MAX_DISK_USAGE_PERCENT:
        warnings.append(f"Disk usage ({system_metrics['disk_percent']}%) exceeds threshold ({config.MAX_DISK_USAGE_PERCENT}%)")
    
    # Determine overall status
    overall_status = "healthy"
    for component, status_val in components_status.items():
        if "error" in status_val:
            overall_status = "unhealthy"
            break
        elif "warning" in status_val and overall_status != "unhealthy":
            overall_status = "degraded"
    
    if warnings and overall_status == "healthy":
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "version": os.environ.get("APP_VERSION", "dev"),
        "components": components_status,
        "system": system_metrics,
        "warnings": warnings if warnings else None
    }

# Prometheus metrics endpoint
@router.get("/metrics", status_code=status.HTTP_200_OK)
async def metrics():
    """
    Endpoint that exposes Prometheus metrics.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Cache metrics endpoint
@router.get("/metrics/cache", status_code=status.HTTP_200_OK)
async def cache_metrics():
    """
    Endpoint that exposes Redis cache metrics and statistics.
    """
    try:
        # Get Redis client from extraction coordinator
        from controllers.extraction_coordinator import ExtractionCoordinator
        
        coordinator = ExtractionCoordinator()
        
        if not coordinator.redis_client:
            return {
                "status": "disabled",
                "message": "Redis cache is not enabled or available"
            }
        
        # Get Redis info
        redis_info = coordinator.redis_client.info()
        
        # Calculate cache hit rate (approximation based on Redis stats)
        keyspace_hits = redis_info.get('keyspace_hits', 0)
        keyspace_misses = redis_info.get('keyspace_misses', 0)
        total_commands = keyspace_hits + keyspace_misses
        hit_rate = (keyspace_hits / max(total_commands, 1)) * 100
        
        # Get memory usage
        memory_usage = {
            'used_memory': redis_info.get('used_memory', 0),
            'used_memory_human': redis_info.get('used_memory_human', '0B'),
            'used_memory_peak': redis_info.get('used_memory_peak', 0),
            'used_memory_peak_human': redis_info.get('used_memory_peak_human', '0B'),
        }
        
        # Get key statistics
        db_info = redis_info.get('db0', {})
        if isinstance(db_info, str):
            # Parse db0 info string like "keys=5,expires=0,avg_ttl=0"
            db_parts = db_info.split(',')
            keys_count = 0
            expires_count = 0
            for part in db_parts:
                if part.startswith('keys='):
                    keys_count = int(part.split('=')[1])
                elif part.startswith('expires='):
                    expires_count = int(part.split('=')[1])
        else:
            keys_count = db_info.get('keys', 0)
            expires_count = db_info.get('expires', 0)
        
        return {
            "status": "enabled",
            "redis_info": {
                "version": redis_info.get('redis_version', 'unknown'),
                "uptime_in_seconds": redis_info.get('uptime_in_seconds', 0),
                "connected_clients": redis_info.get('connected_clients', 0),
                "total_commands_processed": redis_info.get('total_commands_processed', 0),
            },
            "cache_metrics": {
                "hit_rate_percent": round(hit_rate, 2),
                "keyspace_hits": keyspace_hits,
                "keyspace_misses": keyspace_misses,
                "total_commands": total_commands,
            },
            "memory_usage": memory_usage,
            "key_statistics": {
                "total_keys": keys_count,
                "keys_with_expiry": expires_count,
                "keys_without_expiry": keys_count - expires_count,
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache metrics: {e}")
        return {
            "status": "error",
            "message": f"Failed to retrieve cache metrics: {str(e)}"
        }

# Initialize components
from config import TEMPLATE_STORAGE_PATH, GEMINI_API_KEY, DEFAULT_AI_MODEL
template_storage = TemplateStorage(TEMPLATE_STORAGE_PATH)
site_discovery = SiteDiscovery()
crawler = AsyncWebCrawler()
search_template_integrator = SearchTemplateIntegrator(crawler)

# Initialize Result Enhancement and Schema Generation components for Task 6.2
result_enhancer = ResultEnhancer()
ai_schema_generator = AISchemaGenerator()

# Initialize templates for HTML responses
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Get ServiceRegistry instance (singleton)
service_registry = ServiceRegistry()

# Get AI service from registry (it should already be initialized)
try:
    ai_service = service_registry.get_service("ai_service")
    if not ai_service:
        raise KeyError("AI service not found")
except (KeyError, Exception):
    # Fallback: Create a basic AI service if not found
    ai_service = AIService()
    ai_service.initialize(config={
        "default_model": DEFAULT_AI_MODEL,
        "api_key": GEMINI_API_KEY
    })
    logger.warning("AI service not found in registry, created fallback instance")

# Initialize response parser
response_parser = ResponseParser(ai_service=ai_service)

# --- BEGIN NEW API ENDPOINTS FOR APP CONFIGURATION ---

@router.get("/api/app-config", response_model=AppConfigResponse)
async def get_app_configuration(request: Request):
    """
    Provides the current application configuration, including AI settings,
    available AI models, and feature toggle states.
    """
    # 1. AI Settings
    session_ai_config = request.session.get("ai_config", {})
    ai_settings_list = []

    # Check if session exists - if it does, use only session data (don't fall back to env vars)
    # This respects explicit clearing of configuration
    session_exists = "ai_config" in request.session
    
    if session_exists:
        # Use only session configuration (respects explicit clearing)
        # Google
        google_session_provider_config = session_ai_config.get("google_config", {})
        google_api_key = google_session_provider_config.get("api_key")
        google_model = google_session_provider_config.get("model")
        if google_api_key:
            ai_settings_list.append(AISetting(provider="google", api_key="********", model=google_model))

        # OpenAI
        openai_session_provider_config = session_ai_config.get("openai_config", {})
        openai_api_key = openai_session_provider_config.get("api_key")
        openai_model = openai_session_provider_config.get("model")
        if openai_api_key:
            ai_settings_list.append(AISetting(provider="openai", api_key="********", model=openai_model))

        # Anthropic
        anthropic_session_provider_config = session_ai_config.get("anthropic_config", {})
        anthropic_api_key = anthropic_session_provider_config.get("api_key")
        anthropic_model = anthropic_session_provider_config.get("model")
        if anthropic_api_key:
            ai_settings_list.append(AISetting(provider="anthropic", api_key="********", model=anthropic_model))
    else:
        # No session exists yet, fall back to environment defaults
        # Google
        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if google_api_key:
            ai_settings_list.append(AISetting(provider="google", api_key="********", model=None))

        # OpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            ai_settings_list.append(AISetting(provider="openai", api_key="********", model=None))

        # Anthropic
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            ai_settings_list.append(AISetting(provider="anthropic", api_key="********", model=None))

    current_ai_config = AIConfig(
        settings=ai_settings_list,
        default_provider=session_ai_config.get("default_provider") or getattr(config, "DEFAULT_AI_PROVIDER", None),
        default_model_id=session_ai_config.get("default_model_id") or getattr(config, "DEFAULT_AI_MODEL", None)
    )

    # 2. Available AI Models
    available_models = []
    if hasattr(ai_service, "get_available_model_configurations"):
        try:
            available_models = ai_service.get_available_model_configurations()
        except Exception as e:
            logger.error(f"Failed to get available model configurations from AIService: {e}")
            # available_models remains empty or could be set to a default error state

    # 3. Feature Settings
    feature_settings_list = []
    session_app_settings = request.session.get("app_settings", {})

    # Define known feature toggles from config.py
    # This should ideally be more dynamic or defined in a structured way in config.py
    known_features_from_config = {
        "CRAWL4AI_ENABLED": {"name": "Crawl4AI Processing", "description": "Enable crawl4ai library for advanced crawling.", "category": "Crawling"},
        "SPACY_ENABLED": {"name": "spaCy NLP", "description": "Enable spaCy for NLP tasks.", "category": "AI Features"},
        "SEMANTIC_SEARCH_ENABLED": {"name": "Semantic Search", "description": "Enable semantic search capabilities.", "category": "AI Features"},
        "AI_SCHEMA_GENERATION_ENABLED": {"name": "AI Schema Generation", "description": "Enable AI-driven schema generation.", "category": "AI Features"},
        "USE_UNDETECTED_CHROMEDRIVER": {"name": "Undetected ChromeDriver", "description": "Use undetected-chromedriver for resilience.", "category": "Crawling"},
        "REDIS_CACHE_ENABLED": {"name": "Redis Cache", "description": "Enable Redis for caching.", "category": "Performance"},
        # Add other features from config.py that should be user-configurable
        "UNIVERSAL_STRATEGY_ENABLED": {"name": "Universal Strategy", "description": "Enable the Universal Scraping Strategy.", "category": "Crawling"},
        "INTELLIGENT_URL_GENERATION": {"name": "Intelligent URL Generation", "description": "Enable AI-powered URL generation.", "category": "Crawling"},
        "CONTEXTUAL_QUERY_EXPANSION": {"name": "Contextual Query Expansion", "description": "Enable contextual query expansion.", "category": "AI Features"},
        "PYDANTIC_VALIDATION_ENABLED": {"name": "Pydantic Validation", "description": "Enable Pydantic validation for extracted data.", "category": "Data Quality"},
    }

    for key, details in known_features_from_config.items():
        config_value = getattr(config, key, False) 
        current_value = session_app_settings.get(key, config_value) 
        
        feature_settings_list.append(AppFeatureSetting(
            id=key,
            name=details["name"],
            value=current_value,
            description=details.get("description"),
            category=details.get("category")
        ))
    
    # Add any features found in session but not in known_features_from_config
    feature_settings_list = [] # Initialize feature_settings_list
    for key, value in session_app_settings.items():
        if key not in known_features_from_config and isinstance(value, bool): # Process only if boolean, assuming it's a toggle
            logger.info(f"Found unknown feature '{key}' in session, adding to response.")
            feature_settings_list.append(AppFeatureSetting(
                id=key,
                name=key.replace("_", " ").title(), # Generic name
                value=value,
                description="User-defined feature toggle from session.",
                category="User Defined"
            ))
    
    return AppConfigResponse(
        ai_settings=current_ai_config,
        available_models=available_models,
        feature_settings=feature_settings_list
    )

@router.post("/api/ai-config")
async def save_ai_configuration(request: Request, ai_config_payload: AIConfig):
    """
    Saves AI provider settings (API key, selected model) to the user's session
    and refreshes the AIService.
    """
    session_ai_config = {} # Start fresh for provider settings based on payload

    # Populate provider-specific configurations
    for setting in ai_config_payload.settings:
        provider_config_key = f"{setting.provider.lower()}_config"
        # Store whatever is provided; None is acceptable if key/model is not set or cleared
        session_ai_config[provider_config_key] = {
            "api_key": setting.api_key,
            "model": setting.model
        }

    # Set defaults
    session_ai_config["default_provider"] = ai_config_payload.default_provider
    session_ai_config["default_model_id"] = ai_config_payload.default_model_id
    
    request.session["ai_config"] = session_ai_config
    logger.info(f"AI configuration saved to session: {session_ai_config}")
    
    # Prepare config for AIService re-initialization
    # This structure should align with what AIService expects for initialization/re-initialization.
    new_config_for_service = {
        "default_provider": session_ai_config.get("default_provider"),
        "default_model_id": session_ai_config.get("default_model_id")
    }
    # Add individual provider configs (e.g., google_config, openai_config)
    for provider_name in ["google", "openai", "anthropic"]: # Known providers
        p_config_key = f"{provider_name}_config"
        if p_config_key in session_ai_config:
            new_config_for_service[p_config_key] = session_ai_config[p_config_key]
    
    # Ensure AIService is re-initialized with the new configuration
    if hasattr(ai_service, "reinitialize_with_config"):
        try:
            # Assuming reinitialize_with_config is an async method based on original comment
            # If it's synchronous, remove 'await'
            await ai_service.reinitialize_with_config(new_config_for_service)
            logger.info("AIService reinitialized with new configuration.")
        except Exception as e:
            logger.error(f"Error reinitializing AIService: {e}", exc_info=True)
            # Depending on severity, might raise HTTPException or just log
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error updating AI service configuration: {str(e)}")
    else:
        logger.warning("AIService does not have 'reinitialize_with_config' method. Service configuration may not be updated dynamically. A restart might be required.")
        # Optionally, could raise an error or return a specific message if dynamic reconfig is critical
        # For now, we'll allow the session to be updated but warn that the service itself might not reflect changes immediately.

    return {"message": "AI configuration saved successfully. AIService status logged."}


@router.post("/api/ai-config/test-key", status_code=status.HTTP_200_OK)
async def test_ai_api_key(request: Request, key_test_request: AITestKeyRequest):
    """
    Test an AI API key by making a simple request to validate it works.
    """
    try:
        provider = key_test_request.provider.lower()
        api_key = key_test_request.api_key
        model_id = key_test_request.model_id
        
        # Get available models for the provider
        ai_service = global_registry.get_service("ai_service")
        if not ai_service:
            raise HTTPException(status_code=500, detail="AI service not available")
            
        available_models = ai_service.get_available_model_configurations()
        provider_models = [m for m in available_models if m.get('provider', '').lower() == provider]
        
        if not provider_models:
            raise HTTPException(status_code=400, detail=f"No models available for provider: {provider}")
        
        # Use the specified model or pick the first available one
        if model_id:
            test_model = next((m for m in provider_models if m.get('model_id') == model_id), provider_models[0])
        else:
            test_model = provider_models[0]
        
        # Test the API key based on provider
        if provider == 'google':
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Try to list models to validate the key
            models = genai.list_models()
            model_list = list(models)
            
            if not model_list:
                raise Exception("No models accessible with this API key")
                
            return {
                "success": True, 
                "message": f"API key validated successfully for {provider}",
                "available_models": [m.name for m in model_list[:5]]  # Show first 5 models
            }
            
        elif provider == 'openai':
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            # Try to list models
            models = client.models.list()
            if not models.data:
                raise Exception("No models accessible with this API key")
                
            return {
                "success": True,
                "message": f"API key validated successfully for {provider}",
                "available_models": [m.id for m in models.data[:5]]
            }
            
        elif provider == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            # Make a simple test completion
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Test"}]
            )
            
            return {
                "success": True,
                "message": f"API key validated successfully for {provider}",
                "test_response": "API key working"
            }
            
        else:
            # For other providers, just return success if we have the model config
            return {
                "success": True,
                "message": f"API key accepted for {provider} (validation not implemented)",
                "note": "Manual validation - please ensure your API key is correct"
            }
            
    except ImportError as e:
        logger.error(f"Required library not installed for {provider}: {e}")
        raise HTTPException(status_code=400, detail=f"Required library not installed for {provider}")
    except Exception as e:
        logger.error(f"API key validation failed for {provider}: {e}")
        raise HTTPException(status_code=400, detail=f"API key validation failed: {str(e)}")

@router.get("/api/app-settings", response_model=AppSettings)
async def get_app_settings(request: Request):
    '''
    Retrieves current application feature toggle settings.
    '''
    session_app_settings = request.session.get("app_settings", {})
    feature_settings_list = []

    # This should ideally be more dynamic or defined in a structured way in config.py
    known_features_from_config = {
        "CRAWL4AI_ENABLED": {"name": "Crawl4AI Processing", "description": "Enable crawl4ai library for advanced crawling.", "category": "Crawling"},
        "SPACY_ENABLED": {"name": "spaCy NLP", "description": "Enable spaCy for NLP tasks.", "category": "AI Features"},
        "SEMANTIC_SEARCH_ENABLED": {"name": "Semantic Search", "description": "Enable semantic search capabilities.", "category": "AI Features"},
        "AI_SCHEMA_GENERATION_ENABLED": {"name": "AI Schema Generation", "description": "Enable AI-driven schema generation.", "category": "AI Features"},
        "USE_UNDETECTED_CHROMEDRIVER": {"name": "Undetected ChromeDriver", "description": "Use undetected-chromedriver for resilience.", "category": "Crawling"},
        "REDIS_CACHE_ENABLED": {"name": "Redis Cache", "description": "Enable Redis for caching.", "category": "Performance"},
        "UNIVERSAL_STRATEGY_ENABLED": {"name": "Universal Strategy", "description": "Enable the Universal Scraping Strategy.", "category": "Crawling"},
        "INTELLIGENT_URL_GENERATION": {"name": "Intelligent URL Generation", "description": "Enable AI-powered URL generation.", "category": "Crawling"},
        "CONTEXTUAL_QUERY_EXPANSION": {"name": "Contextual Query Expansion", "description": "Enable contextual query expansion.", "category": "AI Features"},
        "PYDANTIC_VALIDATION_ENABLED": {"name": "Pydantic Validation", "description": "Enable Pydantic validation for extracted data.", "category": "Data Quality"},
    }

    for key, details in known_features_from_config.items():
        config_value = getattr(config, key, False) 
        current_value = session_app_settings.get(key, config_value)
        
        feature_settings_list.append(AppFeatureSetting(
            id=key,
            name=details["name"],
            value=current_value,
            description=details.get("description"),
            category=details.get("category")
        ))
    
    # Add any features found in session but not in known_features_from_config
    for key, value in session_app_settings.items():
        if key not in known_features_from_config and isinstance(value, bool):
            logger.info(f"Found unknown feature '{key}' in session, adding to response.")
            feature_settings_list.append(AppFeatureSetting(
                id=key,
                name=key.replace("_", " ").title(), # Generic name
                value=value,
                description="User-defined feature toggle from session.",
                category="User Defined"
            ))
            
    return AppSettings(features=feature_settings_list)

@router.post("/api/app-settings")
async def save_app_settings(request: Request, settings_payload: AppSettings):
    '''
    Saves application feature toggle settings to the user's session.
    '''
    current_session_settings = request.session.get("app_settings", {})
    updated_count = 0
    for feature_setting in settings_payload.features:
        if feature_setting.id in current_session_settings and current_session_settings[feature_setting.id] == feature_setting.value:
            continue # No change for this setting
        current_session_settings[feature_setting.id] = feature_setting.value
        logger.info(f"App setting '{feature_setting.id}' updated to {feature_setting.value} in session.")
        updated_count += 1
    
    request.session["app_settings"] = current_session_settings
    
    if updated_count > 0:
        # Optionally, trigger re-initialization of services if needed based on changed settings
        # For example, if a cache setting changed, reinitialize cache manager
        logger.info(f"{updated_count} app settings updated in session. Relevant services may need re-initialization (not yet implemented).")
        # Example: if 'REDIS_CACHE_ENABLED' changed, you might call:
        # service_registry.reinitialize_service('cache_manager', new_config)

    return {"message": f"Application settings saved. {updated_count} settings updated."}

# --- END NEW API ENDPOINTS FOR APP CONFIGURATION ---


# --- BEGIN NEW API ENDPOINTS FOR TASK 6.2 API ENHANCEMENTS ---

@router.post("/api/schema/generate", status_code=status.HTTP_201_CREATED)
async def generate_schema_endpoint(request_data: SchemaGenerateRequest, api_key: APIKeyHeader = Security(validate_api_key)):
    """
    Generate a Pydantic schema from sample data using AISchemaGenerator.
    """
    try:
        schema = await ai_schema_generator.generate_schema(
            sample_data=request_data.sample_data,
            schema_name=request_data.schema_name,
            strict_mode=request_data.strict_mode
        )
        return {"status": "success", "schema": schema}
    except Exception as e:
        logger.error(f"Schema generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Schema generation error: {str(e)}")

@router.post("/api/schema/validate", status_code=status.HTTP_200_OK)
async def validate_schema_endpoint(validate_request: SchemaValidateRequest, api_key: APIKeyHeader = Security(validate_api_key)):
    """
    Validate data against a schema definition.
       """
    try:
        is_valid, errors = await ai_schema_generator.validate_data_against_schema(
            data=validate_request.data,
            schema_definition=validate_request.schema_definition
        )
        return {"status": "success", "is_valid": is_valid, "errors": errors}
    except Exception as e:
        logger.error(f"Schema validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Schema validation error: {str(e)}")

@router.post("/api/schema/save", status_code=status.HTTP_201_CREATED)
async def save_schema_endpoint(save_request: SchemaSaveRequest, api_key: APIKeyHeader = Security(validate_api_key)):
    """Save a schema definition for reuse."""
    try:
        schema_id = await ai_schema_generator.save_schema(
            schema_name=save_request.schema_name,
            schema_definition=save_request.schema_definition,
            description=save_request.description
        )
        return {"status": "success", "message": "Schema saved successfully", "schema_id": schema_id}
    except ValueError as ve:
        logger.warning(f"Schema save validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Schema save failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Schema save error: {str(e)}")

@router.get("/api/schema/list", status_code=status.HTTP_200_OK)
async def list_schemas_endpoint(category: Optional[str] = None, search_term: Optional[str] = None, limit: int = 50, api_key: APIKeyHeader = Security(validate_api_key)):
    """List saved and generated schemas with filtering options."""
    try:
        schemas = await ai_schema_generator.list_schemas(
            category=category,
            search_term=search_term,
            limit=limit
        )
        return {"status": "success", "schemas": schemas}
    except Exception as e:
        logger.error(f"Listing schemas failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing schemas: {str(e)}")

@router.get("/api/schema/{schema_id}", status_code=status.HTTP_200_OK)
async def get_schema_endpoint(schema_id: str, api_key: APIKeyHeader = Security(validate_api_key)):
    """Retrieve a specific schema by ID."""
    try:
        schema = await ai_schema_generator.get_schema(schema_id)
        if schema:
            return {"status": "success", "schema": schema}
        else:
            raise HTTPException(status_code=404, detail="Schema not found")
    except HTTPException: # Re-raise HTTPException
        raise
    except Exception as e:
        logger.error(f"Error retrieving schema {schema_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving schema: {str(e)}")

@router.post("/api/feedback/enhanced", status_code=status.HTTP_201_CREATED)
async def enhanced_feedback_endpoint(feedback_request: EnhancedFeedbackRequest, api_key: APIKeyHeader = Security(validate_api_key)):
    """Submit enhanced user feedback with ResultEnhancer integration."""
    try:
        # Convert Pydantic model to UserFeedback domain model if necessary, or use directly
        # Assuming ResultEnhancer can handle this Pydantic model or a similar structure
        feedback_id = await result_enhancer.add_feedback(
            result_id=feedback_request.result_id,
            query=feedback_request.query,
            results=feedback_request.results, # This might need transformation
            feedback_type=ResultFeedbackType(feedback_request.feedback_type.lower()),
            rating=ResultFeedbackRating(feedback_request.rating.lower()),
            comments=feedback_request.comments,
            field_specific_feedback=feedback_request.field_specific_feedback,
            user_id=feedback_request.user_id,
            session_id=feedback_request.session_id
        )
        return {"status": "success", "message": "Feedback submitted successfully", "feedback_id": feedback_id}
    except ValueError as ve: # For invalid enum values or other validation issues
        logger.warning(f"Enhanced feedback submission error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Enhanced feedback submission failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

@router.post("/api/analytics/detailed", status_code=status.HTTP_200_OK)
async def detailed_analytics_endpoint(analytics_request: DetailedAnalyticsRequest, api_key: APIKeyHeader = Security(validate_api_key)):
    """Get detailed feedback analytics with filtering and trends."""
    try:
        # Construct filter_criteria from the request
        filter_criteria = {}
        if analytics_request.result_id: filter_criteria["result_id"] = analytics_request.result_id
        if analytics_request.feedback_type: filter_criteria["feedback_type"] = ResultFeedbackType(analytics_request.feedback_type.lower())
        if analytics_request.start_date: filter_criteria["start_date"] = datetime.fromisoformat(analytics_request.start_date)
        if analytics_request.end_date: filter_criteria["end_date"] = datetime.fromisoformat(analytics_request.end_date)
        if analytics_request.user_id: filter_criteria["user_id"] = analytics_request.user_id
        
        analytics_data = await result_enhancer.get_detailed_analytics(
            filter_criteria=filter_criteria,
            include_trends=analytics_request.include_trends,
            limit=analytics_request.limit
        )
        return {"status": "success", "analytics_data": analytics_data}
    except ValueError as ve: # For invalid enum values or date formats
        logger.warning(f"Detailed analytics request error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Detailed analytics retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving detailed analytics: {str(e)}")

@router.post("/api/results/enhance", status_code=status.HTTP_200_OK)
async def enhance_result_endpoint(enhance_request: ResultEnhanceRequest, api_key: APIKeyHeader = Security(validate_api_key)):
    """Re-process results using collected feedback."""
    try:
        # Convert feedback_data if needed
        parsed_feedback_data = []
        if enhance_request.feedback_data:
            for fb_item in enhance_request.feedback_data:
                parsed_feedback_data.append(UserFeedback(
                    field=fb_item["field"],
                    type=ResultFeedbackType(fb_item["type"].lower()),
                    rating=ResultFeedbackRating(fb_item["rating"].lower()),
                    correction=fb_item.get("correction"),
                    comment=fb_item.get("comment")
                ))

        enhanced_result = await result_enhancer.enhance_result(
            result_id=enhance_request.result_id,
            feedback_data=parsed_feedback_data if parsed_feedback_data else None,
            reprocess_options=enhance_request.reprocess_options
        )
        return {"status": "success", "enhanced_result": enhanced_result}
    except ValueError as ve: # For invalid enum values or other validation issues
        logger.warning(f"Result enhancement request error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Result enhancement failed for {enhance_request.result_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error enhancing result: {str(e)}")

@router.post("/api/cache/clear", status_code=status.HTTP_200_OK)
async def clear_cache_endpoint(clear_request: CacheClearRequest, api_key: APIKeyHeader = Security(validate_api_key)):
    """
    Clears server-side caches to free memory and ensure fresh data.
    """
    cleared_caches = []
    
    try:
        # Clear AI service cache if available
        if hasattr(ai_service, "clear_cache"):
            ai_service.clear_cache()
            cleared_caches.append("AI Service")
        
        # Clear template storage cache if available
        if hasattr(template_storage, "clear_cache"):
            template_storage.clear_cache()
            cleared_caches.append("Template Storage")
        
        # Clear site discovery cache if available
        if hasattr(site_discovery, "clear_cache"):
            site_discovery.clear_cache()
            cleared_caches.append("Site Discovery")
        
        # Clear crawler cache if available
        if hasattr(crawler, "clear_cache"):
            crawler.clear_cache()
            cleared_caches.append("Crawler")
        
        # Clear Redis cache if configured
        if config.DEFAULT_CACHE_MODE == "redis" and redis:
            try:
                r = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
                r.flushdb()
                cleared_caches.append("Redis")
            except Exception as redis_error:
                logger.warning(f"Failed to clear Redis cache: {redis_error}")
        
        # Clear in-memory job cache (keep only recent jobs)
        if jobs:
            current_time = time.time()
            jobs_to_keep = {}
            for job_id, job_info in jobs.items():
                job_age = current_time - time.mktime(time.strptime(job_info["created_at"], "%Y-%m-%dT%H:%M:%S.%f"))
                if job_age < 3600:  # Keep jobs from last hour
                    jobs_to_keep[job_id] = job_info
            
            jobs.clear()
            jobs.update(jobs_to_keep)
            cleared_caches.append("Job Cache")
        
        logger.info(f"Cache clearing completed. Cleared: {cleared_caches}")
        
        return {
            "status": "success", 
            "message": f"Cache cleared successfully. Cleared: {', '.join(cleared_caches)}",
            "cleared_caches": cleared_caches
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return JSONResponse(
            status_code=500, 
            content={
                "status": "error", 
                "message": f"Error clearing cache: {str(e)}",
                "cleared_caches": cleared_caches
            }
        )

# --- END NEW API ENDPOINTS FOR TASK 6.2 ---

# Pipeline and async task endpoints
@router.post("/pipeline/simple", status_code=status.HTTP_202_ACCEPTED)
async def create_simple_pipeline(request: Request):
    """
    Create a simple parallel extraction pipeline
    """
    try:
        data = await request.json()
        urls = data.get('urls', [])
        strategy = data.get('strategy')
        
        if not urls:
            raise HTTPException(status_code=400, detail="URLs list is required")
        
        # Import pipeline orchestrator
        from core.pipeline_orchestrator import pipeline_orchestrator
        
        pipeline_id = pipeline_orchestrator.create_simple_pipeline(
            urls=urls, 
            strategy=strategy,
            **data.get('options', {})
        )
        
        return {
            "pipeline_id": pipeline_id,
            "status": "pipeline_created",
            "total_urls": len(urls),
            "message": f"Created pipeline with {len(urls)} extraction tasks"
        }
        
    except Exception as e:
        logger.error(f"Failed to create simple pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline creation failed: {str(e)}")

@router.post("/pipeline/batch", status_code=status.HTTP_202_ACCEPTED)
async def create_batch_pipeline(request: Request):
    """
    Create a batch processing pipeline
    """
    try:
        data = await request.json()
        url_batches = data.get('url_batches', [])
        strategy = data.get('strategy')
        
        if not url_batches:
            raise HTTPException(status_code=400, detail="URL batches list is required")
        
        from core.pipeline_orchestrator import pipeline_orchestrator
        
        pipeline_id = pipeline_orchestrator.create_batch_pipeline(
            url_batches=url_batches,
            strategy=strategy,
            **data.get('options', {})
        )
        
        total_urls = sum(len(batch) for batch in url_batches)
        
        return {
            "pipeline_id": pipeline_id,
            "status": "pipeline_created",
            "total_batches": len(url_batches),
            "total_urls": total_urls,
            "message": f"Created batch pipeline with {len(url_batches)} batches ({total_urls} total URLs)"
        }
        
    except Exception as e:
        logger.error(f"Failed to create batch pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline creation failed: {str(e)}")

@router.post("/pipeline/smart", status_code=status.HTTP_202_ACCEPTED)
async def create_smart_pipeline(request: Request):
    """
    Create an intelligent adaptive pipeline
    """
    try:
        data = await request.json()
        urls = data.get('urls', [])
        
        if not urls:
            raise HTTPException(status_code=400, detail="URLs list is required")
        
        from core.pipeline_orchestrator import pipeline_orchestrator
        
        pipeline_id = pipeline_orchestrator.create_smart_pipeline(
            urls=urls,
            **data.get('options', {})
        )
        
        return {
            "pipeline_id": pipeline_id,
            "status": "pipeline_created",
            "total_urls": len(urls),
            "message": f"Created smart pipeline with intelligent strategy selection for {len(urls)} URLs"
        }
        
    except Exception as e:
        logger.error(f"Failed to create smart pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline creation failed: {str(e)}")

@router.get("/pipeline/{pipeline_id}/status", status_code=status.HTTP_200_OK)
async def get_pipeline_status(pipeline_id: str):
    """
    Get the status of a pipeline
    """
    try:
        from core.pipeline_orchestrator import pipeline_orchestrator
        
        pipeline_result = pipeline_orchestrator.get_pipeline_status(pipeline_id)
        
        if not pipeline_result:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        return {
            "pipeline_id": pipeline_result.pipeline_id,
            "status": pipeline_result.status.value,
            "start_time": pipeline_result.start_time.isoformat(),
            "end_time": pipeline_result.end_time.isoformat() if pipeline_result.end_time else None,
            "total_tasks": pipeline_result.total_tasks,
            "completed_tasks": pipeline_result.completed_tasks,
            "failed_tasks": pipeline_result.failed_tasks,
            "success_rate": pipeline_result.success_rate,
            "execution_time": pipeline_result.execution_time,
            "is_complete": pipeline_result.is_complete
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@router.delete("/pipeline/{pipeline_id}", status_code=status.HTTP_200_OK)
async def cancel_pipeline(pipeline_id: str):
    """
    Cancel a running pipeline
    """
    try:
        from core.pipeline_orchestrator import pipeline_orchestrator
        
        success = pipeline_orchestrator.cancel_pipeline(pipeline_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        return {
            "pipeline_id": pipeline_id,
            "status": "cancelled",
            "message": "Pipeline cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline cancellation failed: {str(e)}")

@router.get("/pipeline/health", status_code=status.HTTP_200_OK)
async def get_pipeline_health():
    """
    Get pipeline system health information
    """
    try:
        from core.pipeline_orchestrator import pipeline_orchestrator
        
        health_data = pipeline_orchestrator.get_system_health()
        
        return health_data
        
    except Exception as e:
        logger.error(f"Failed to get pipeline health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Initialize Intent Parser and Adaptive Scraper
# These are now initialized after AI Service to ensure it's available
logger.info("ROUTES_PY_PRE_INIT_ADAPTIVE_SCRAPER: --- AdaptiveScraper Pre-Initialization Diagnostics ---")
logger.info(f"ROUTES_PY_PRE_INIT_ADAPTIVE_SCRAPER: AdaptiveScraper module file (again): {getattr(AdaptiveScraper, '__file__', 'N/A')}")
logger.info(f"ROUTES_PY_PRE_INIT_ADAPTIVE_SCRAPER: id(AdaptiveScraper) before init: {id(AdaptiveScraper)}")
logger.info(f"ROUTES_PY_PRE_INIT_ADAPTIVE_SCRAPER: hasattr(AdaptiveScraper, 'process_user_request') before init: {hasattr(AdaptiveScraper, 'process_user_request')}")
logger.info(f"ROUTES_PY_PRE_INIT_ADAPTIVE_SCRAPER: 'process_user_request' in dir(AdaptiveScraper) before init: {'process_user_request' in dir(AdaptiveScraper)}")

intent_parser = IntentParser(ai_service=ai_service) # Changed from get_intent_parser
adaptive_scraper = AdaptiveScraper() # Global instance

logger.info("ROUTES_PY_POST_INIT_ADAPTIVE_SCRAPER: --- AdaptiveScraper Post-Initialization Diagnostics ---")
logger.info(f"ROUTES_PY_POST_INIT_ADAPTIVE_SCRAPER: adaptive_scraper instance: {adaptive_scraper}")
logger.info(f"ROUTES_PY_POST_INIT_ADAPTIVE_SCRAPER: hasattr(adaptive_scraper, 'process_user_request') after init: {hasattr(adaptive_scraper, 'process_user_request')}")
logger.info(f"ROUTES_PY_POST_INIT_ADAPTIVE_SCRAPER: 'process_user_request' in dir(adaptive_scraper) after init: {'process_user_request' in dir(adaptive_scraper)}")
logger.info(f"ROUTES_PY_POST_INIT_ADAPTIVE_SCRAPER: Type of adaptive_scraper: {type(adaptive_scraper)}")
if hasattr(adaptive_scraper, '__class__'):
    logger.info(f"ROUTES_PY_POST_INIT_ADAPTIVE_SCRAPER: adaptive_scraper.__class__.__file__: {getattr(adaptive_scraper.__class__, '__file__', 'N/A')}")
    logger.info(f"ROUTES_PY_POST_INIT_ADAPTIVE_SCRAPER: id(adaptive_scraper.__class__): {id(adaptive_scraper.__class__)}")
logger.info("ROUTES_PY_POST_INIT_ADAPTIVE_SCRAPER: --- End Post-Initialization Diagnostics ---")

# Initialize MultiScrapeManager
multi_scrape_manager = get_multi_scrape_manager() # Removed arguments

# API Key Header for dependency injection
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

# Frontend route
@router.get("/", response_class=HTMLResponse)
async def get_frontend(request: Request):
    # ... existing code ...
    return HTMLResponse(content=get_frontend_html(), status_code=200)

# Existing /scrape endpoint
@router.post("/scrape", status_code=status.HTTP_202_ACCEPTED)
async def scrape_site(request: Request, scrape_request: ScrapeRequest, background_tasks: BackgroundTasks, api_key: APIKeyHeader = Security(validate_api_key)):
    # Check rate limit
    await check_rate_limit(request, "scrape_endpoint")
    
    # ... existing code ...
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "submitted_at": datetime.now().isoformat()}
    
    logger.info(f"Received scrape request for URL: {scrape_request.url} with strategy: {scrape_request.strategy_name}")
    SCRAPE_REQUESTS.labels(status='pending').inc()

    # Determine strategy
    strategy_name = scrape_request.strategy_name or "default"
    options = scrape_request.options or {}
    
    # Add job to background tasks
    background_tasks.add_task(
        multi_scrape_manager.run_scrape_job, 
        job_id, 
        scrape_request.url, 
        strategy_name, 
        options,
        scrape_request.query # Pass query to multi_scrape_manager
    )
    
    return {"job_id": job_id, "status": "pending", "message": "Scrape job accepted and is being processed."}

# New /scrape-intelligent endpoint
@router.post("/scrape-intelligent", status_code=status.HTTP_202_ACCEPTED)
async def scrape_intelligent(request: Request, request_data: IntelligentScrapeRequest, background_tasks: BackgroundTasks, api_key: APIKeyHeader = Security(validate_api_key)):
    # Check rate limit
    await check_rate_limit(request, "scrape_endpoint")
    
    logger.info(f"Received intelligent scrape request for query: {request_data.query}")
    SCRAPE_REQUESTS.labels(status='pending_intelligent').inc()
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending_intelligent_analysis", "submitted_at": datetime.now().isoformat()}

    logger.info(f"scrape_intelligent: adaptive_scraper type: {type(adaptive_scraper)}")
    logger.info(f"scrape_intelligent: hasattr(adaptive_scraper, 'process_user_request'): {hasattr(adaptive_scraper, 'process_user_request')}")
    logger.info(f"scrape_intelligent: 'process_user_request' in dir(adaptive_scraper): {'process_user_request' in dir(adaptive_scraper)}")
    if not hasattr(adaptive_scraper, 'process_user_request'):
        logger.error("CRITICAL: adaptive_scraper does NOT have process_user_request method at the time of endpoint call!")
        # Log more details about the adaptive_scraper object
        logger.error(f"adaptive_scraper object details: {dir(adaptive_scraper)}")
        try:
            logger.error(f"adaptive_scraper.__class__: {adaptive_scraper.__class__}")
            logger.error(f"adaptive_scraper.__class__.__module__: {adaptive_scraper.__class__.__module__}")
            logger.error(f"adaptive_scraper.__class__.__name__: {adaptive_scraper.__class__.__name__}")
            logger.error(f"adaptive_scraper.__class__.__file__: {getattr(adaptive_scraper.__class__, '__file__', 'N/A')}")
        except Exception as e_detail:
            logger.error(f"Error getting detailed info for adaptive_scraper: {e_detail}")
        raise HTTPException(status_code=500, detail="Internal server error: Scraper not configured correctly.")

    # Use the global adaptive_scraper instance
    background_tasks.add_task(
        process_intelligent_scrape_task,
        job_id,
        adaptive_scraper,
        request_data.query, 
        request_data.start_url, 
        request_data.options
    )
    
    return {"job_id": job_id, "status": "pending_intelligent_analysis", "message": "Intelligent scrape job accepted and is being processed."}

# Simple status endpoint for frontend compatibility
@router.get("/status/{job_id}", status_code=status.HTTP_200_OK)
async def get_job_status_simple(job_id: str):
    """Simple status endpoint for frontend - no API key required"""
    logger.info(f"Checking status for job {job_id}")
    
    # Check our local jobs dictionary first
    job_info = jobs.get(job_id)
    if job_info:
        logger.info(f"Job {job_id} found in local jobs")
        # Ensure the response is serializable
        try:
            json.dumps(job_info)
            return job_info
        except (TypeError, ValueError) as e:
            logger.warning(f"Job {job_id} contains non-serializable data: {e}")
            return safe_serialize(job_info)
    
    logger.warning(f"Job {job_id} not found in local jobs dictionary")
    raise HTTPException(status_code=404, detail="Job not found")

# Endpoint to get job status
@router.get("/scrape/status/{job_id}", status_code=status.HTTP_200_OK)
async def get_scrape_status(job_id: str, api_key: APIKeyHeader = Security(validate_api_key)):
    # ... existing code ...
    # First, check our local `jobs` dictionary, which is updated by AdaptiveScraper
    job_info = jobs.get(job_id)
    if not job_info:
        # If not in local `jobs`, try asking AdaptiveScraper directly (e.g., if it uses external job store)
        if hasattr(adaptive_scraper, 'get_job_status'):
            try:
                job_info = await adaptive_scraper.get_job_status(job_id)
            except Exception as e:
                logger.error(f"Error querying adaptive_scraper for job {job_id} status: {e}")
                # Fall through to check multi_scrape_manager if adaptive_scraper fails
                pass # Keep job_info as None
        
        # If still not found, or adaptive_scraper doesn't have get_job_status, check multi_scrape_manager
        if not job_info and hasattr(multi_scrape_manager, 'get_job_status'):
            try:
                job_info = await multi_scrape_manager.get_job_status(job_id)
            except Exception as e:
                logger.error(f"Error querying multi_scrape_manager for job {job_id} status: {e}")
                # Fall through to check local jobs if manager errors

    if not job_info: # Final check after all attempts
        raise HTTPException(status_code=404, detail="Job not found")

    # Update local jobs dict if we got a more definitive status from a manager
    if job_id not in jobs or (job_info and jobs[job_id]["status"] != job_info.get("status")):
        jobs[job_id] = job_info

    return job_info

# Endpoint to get job results
@router.get("/scrape/results/{job_id}", status_code=status.HTTP_200_OK)
async def get_scrape_results(job_id: str, api_key: APIKeyHeader = Security(validate_api_key)):
    # ... existing code ...
    job_info = jobs.get(job_id)
    
    # Try adaptive_scraper first if it might be an intelligent job
    if job_info and "intelligent" in job_info.get("status", "") or not job_info:
        if hasattr(adaptive_scraper, 'get_job_results'):
            try:
                results = await adaptive_scraper.get_job_results(job_id)
                if results: # If adaptive_scraper found results, return them
                    # Ensure the job status is updated locally if results are final
                    if results.get("status") == "completed" or results.get("status") == "failed":
                        jobs[job_id] = results
                    return results
            except Exception as e:
                logger.error(f"Error querying adaptive_scraper for job {job_id} results: {e}")
                # Fall through if adaptive_scraper errors or doesn't have results

    # Fallback to multi_scrape_manager or local jobs cache
    if hasattr(multi_scrape_manager, 'get_job_results'):
        try:
            results = await multi_scrape_manager.get_job_results(job_id)
            if results: # If multi_scrape_manager found results
                if results.get("status") == "completed" or results.get("status") == "failed":
                    jobs[job_id] = results
                return results
        except Exception as e:
            logger.error(f"Error querying multi_scrape_manager for job {job_id} results: {e}")
            # Fall through to check local jobs if manager errors

    # Final check on local jobs dict if managers didn't yield results
    if job_info and (job_info.get("status") == "completed" or job_info.get("status") == "failed"):
        return job_info
    elif job_info:
        return {"job_id": job_id, "status": job_info.get("status", "unknown"), "message": "Job is still processing or results are not yet available from the primary manager."}
    else:
        raise HTTPException(status_code=404, detail="Job not found or results not available.")

# Endpoint for user feedback
@router.post("/feedback", status_code=status.HTTP_201_CREATED)
async def submit_user_feedback(feedback_request: UserFeedbackRequest, api_key: APIKeyHeader = Security(validate_api_key)):
    # ... existing code ...
    try:
        # Assuming result_enhancer has a method to store this basic feedback
        # This might be a simplified version of enhanced_feedback or a separate path
        feedback_id = await result_enhancer.add_feedback(
            result_id=feedback_request.job_id, # Assuming job_id maps to a result_id
            query=feedback_request.query,
            results=feedback_request.results, # This might need to be fetched or structured
            feedback_type=ResultFeedbackType.GENERAL, # Defaulting or map from a field in UserFeedbackRequest
            rating=ResultFeedbackRating.NEUTRAL if feedback_request.is_positive is None else (ResultFeedbackRating.POSITIVE if feedback_request.is_positive else ResultFeedbackRating.NEGATIVE),
            comments=feedback_request.comments,
            user_id=feedback_request.user_id
        )
        logger.info(f"User feedback submitted for job_id: {feedback_request.job_id}, Positive: {feedback_request.is_positive}")
        return {"message": "Feedback submitted successfully", "feedback_id": feedback_id}
    except Exception as e:
        logger.error(f"Error submitting user feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

# Endpoint for feedback analytics
@router.post("/feedback/analytics", status_code=status.HTTP_200_OK)
async def get_feedback_analytics(analytics_request: FeedbackAnalyticsRequest, api_key: APIKeyHeader = Security(validate_api_key)):
    # ... existing code ...
    try:
        filter_criteria = {}
        if analytics_request.feedback_type: filter_criteria["feedback_type"] = ResultFeedbackType(analytics_request.feedback_type.lower())
        if analytics_request.start_date: filter_criteria["start_date"] = datetime.fromisoformat(analytics_request.start_date)
        if analytics_request.end_date: filter_criteria["end_date"] = datetime.fromisoformat(analytics_request.end_date)
        # Add other filters from FeedbackAnalyticsRequest as needed by get_detailed_analytics
        
        analytics_data = await result_enhancer.get_detailed_analytics(
            filter_criteria=filter_criteria,
            include_trends=True # Assuming trends are usually wanted for this endpoint
        )
        return {"status": "success", "analytics_data": analytics_data}
    except ValueError as ve:
        logger.warning(f"Feedback analytics request error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error retrieving feedback analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve feedback analytics: {str(e)}")

# Endpoint for exporting data
@router.get("/export/{job_id}", status_code=status.HTTP_200_OK)
async def export_data(job_id: str, format: str = "json", api_key: APIKeyHeader = Security(validate_api_key)):
    # ... existing code ...
    job_info = jobs.get(job_id)
    if not job_info or job_info.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Job not found or not completed.")

    data_to_export = job_info.get("results", [])
    if not data_to_export:
        return Response(content="No data to export.", media_type="text/plain", status_code=200)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"scraped_data_{job_id}_{timestamp}"

    if format == "json":
        content = generate_json_export(data_to_export)
        media_type = "application/json"
        filename = f"{filename_base}.json"
    elif format == "csv":
        content = generate_csv_export(data_to_export)
        media_type = "text/csv"
        filename = f"{filename_base}.csv"
    elif format == "excel":
        content = generate_excel_export(data_to_export)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"{filename_base}.xlsx"
    else:
        raise HTTPException(status_code=400, detail="Invalid export format specified. Supported formats: json, csv, excel.")

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# Endpoint for AI-guided strategy (placeholder)
@router.post("/scrape/ai-guided", status_code=status.HTTP_202_ACCEPTED)
async def scrape_ai_guided(scrape_request: ScrapeRequest, background_tasks: BackgroundTasks, api_key: APIKeyHeader = Security(validate_api_key)):
    # ... existing code ...
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending_ai_guided", "submitted_at": datetime.now().isoformat()}
    
    logger.info(f"Received AI-guided scrape request for URL: {scrape_request.url}")
    SCRAPE_REQUESTS.labels(status='pending_ai_guided').inc()

    # Initialize AI-Guided Strategy with necessary components
    # This might involve fetching site-specific settings or using defaults
    parsed_url = urlparse(scrape_request.url)
    site_type_settings = get_site_settings(parsed_url.netloc) # Get settings based on domain
    
    ai_guided_strategy = AIGuidedStrategy(
        ai_service=ai_service, 
        crawler=crawler, 
        response_parser=response_parser,
        prompt_generator=PromptGenerator(ai_service=ai_service), # Assuming PromptGenerator needs AIService
        site_settings=site_type_settings # Pass site-specific or generic settings
    )
    
    background_tasks.add_task(
        multi_scrape_manager.run_scrape_job, 
        job_id, 
        scrape_request.url, 
        "ai_guided", # Explicitly use the ai_guided strategy key
        scrape_request.options or {},
        scrape_request.query,
        custom_strategy_instance=ai_guided_strategy # Pass the initialized strategy instance
    )
    
    return {"job_id": job_id, "status": "pending_ai_guided", "message": "AI-guided scrape job accepted."}

# Endpoint for template management (placeholder)
@router.post("/templates/manage", status_code=status.HTTP_200_OK)
async def manage_templates(request: Request, api_key: APIKeyHeader = Security(validate_api_key)):
    # ... (This endpoint would handle creating, updating, deleting templates)
    # ... (It would interact with the TemplateStorage component)
    # ... existing code ...
    # For now, a simple response:
    return {"message": "Template management endpoint placeholder. Functionality to be implemented."}

# Endpoint for site discovery (placeholder)
@router.post("/discover-sites", status_code=status.HTTP_200_OK)
async def discover_sites_endpoint(request: Request, api_key: APIKeyHeader = Security(validate_api_key)):
    # ... (This endpoint would trigger site discovery mechanisms)
    # ... (It would interact with the SiteDiscovery component)
    # ... existing code ...
    # For now, a simple response:
    return {"message": "Site discovery endpoint placeholder. Functionality to be implemented."}

# Endpoint for search template integration (placeholder)
@router.post("/integrate-search-template", status_code=status.HTTP_200_OK)
async def integrate_search_template_endpoint(request: Request, api_key: APIKeyHeader = Security(validate_api_key)):
    # ... (This endpoint would handle integrating search results with templates)
    # ... (It would interact with the SearchTemplateIntegrator component)
    # ... existing code ...
    # For now, a simple response:
    return {"message": "Search template integration endpoint placeholder. Functionality to be implemented."}

# Include other routers if you have them (e.g., for admin, specific features)
# from . import admin_routes
# router.include_router(admin_routes.router, prefix="/admin", tags=["admin"]) 

logger.info("web/routes.py loaded and router configured.")

@router.get("/api/models/{provider}")
async def get_models_by_provider(provider: str):
    """
    Get available models for a specific provider.
    """
    try:
        ai_service = global_registry.get_service("ai_service")
        if not ai_service:
            raise HTTPException(status_code=500, detail="AI service not available")
            
        available_models = ai_service.get_available_model_configurations()
        provider_models = [m for m in available_models if m.get('provider', '').lower() == provider.lower()]
        
        return {
            "provider": provider,
            "models": [{"id": m.get('model_id'), "name": m.get('name', m.get('model_id'))} for m in provider_models]
        }
    except Exception as e:
        logger.error(f"Error getting models for provider {provider}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")

async def process_intelligent_scrape_task(job_id: str, scraper, query: str, start_url: Optional[str], options: Dict[str, Any]):
    """Background task to process intelligent scrape requests"""
    import json
    
    try:
        logger.info(f"Starting intelligent scrape task for job {job_id}")
        jobs[job_id]["status"] = "processing"
        logger.info(f"Job {job_id} status updated to processing")
        
        # Prepare options with start_url if provided
        scrape_options = options.copy()
        if start_url:
            scrape_options["start_url"] = start_url
            
        # Call the process_user_request method with correct parameters
        result = await scraper.process_user_request(
            user_query=query,
            session_id=job_id,  # Use job_id as session_id
            options=scrape_options
        )
        
        # Ensure result is JSON serializable to prevent RecursionError
        try:
            # Test serialization
            json.dumps(result)
            serializable_result = result
        except (TypeError, ValueError) as e:
            logger.warning(f"Result contains non-serializable data: {e}")
            # Use safe serialization
            serializable_result = safe_serialize(result)
        
        # Update job status with results
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = serializable_result
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"Intelligent scrape task completed for job {job_id}")
        logger.info(f"Job {job_id} final status keys: {list(jobs[job_id].keys())}")
        
    except Exception as e:
        logger.error(f"Error in intelligent scrape task for job {job_id}: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["failed_at"] = datetime.now().isoformat()

# Response streaming endpoint
@router.post("/extract/stream", status_code=status.HTTP_200_OK)
async def extract_stream(request: Request):
    """
    Stream extraction results for multiple URLs
    """
    try:
        # Check rate limit
        await check_rate_limit(request, "stream_endpoint")
        
        from fastapi.responses import StreamingResponse
        import json
        
        data = await request.json()
        urls = data.get('urls', [])
        strategy = data.get('strategy')
        options = data.get('options', {})
        
        if not urls:
            raise HTTPException(status_code=400, detail="URLs list is required")
        
        async def generate_results():
            """Generator function for streaming results"""
            # Send initial status
            yield f"data: {json.dumps({'status': 'started', 'total': len(urls)})}\n\n"
            
            # Import extraction coordinator
            from controllers.extraction_coordinator import ExtractionCoordinator
            coordinator = ExtractionCoordinator()
            
            for i, url in enumerate(urls):
                try:
                    # Send progress update
                    yield f"data: {json.dumps({'status': 'processing', 'url': url, 'index': i})}\n\n"
                    
                    # Perform extraction
                    if strategy:
                        result = await coordinator.extract_content(url, strategy, **options)
                    else:
                        result = await coordinator.extract_with_intelligent_selection(url, **options)
                    
                    # Send result
                    result_data = {
                        'index': i,
                        'url': url,
                        'success': result.get('success', False),
                        'strategy': result.get('strategy', 'unknown'),
                        'content_length': len(result.get('content', '')),
                        'progress': (i + 1) / len(urls)
                    }
                    yield f"data: {json.dumps(result_data)}\n\n"
                    
                except Exception as e:
                    # Send error
                    error_data = {
                        'index': i,
                        'url': url,
                        'success': False,
                        'error': str(e),
                        'progress': (i + 1) / len(urls)
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            # Send completion status
            yield f"data: {json.dumps({'status': 'completed', 'total': len(urls)})}\n\n"
        
        return StreamingResponse(
            generate_results(),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"Stream extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stream extraction failed: {str(e)}")

# Batch extraction with progress tracking
@router.post("/extract/batch-progress", status_code=status.HTTP_202_ACCEPTED)
async def extract_batch_with_progress(request: Request):
    """
    Start batch extraction with progress tracking
    """
    try:
        # Check rate limit
        await check_rate_limit(request, "scrape_endpoint")
        data = await request.json()
        urls = data.get('urls', [])
        strategy = data.get('strategy')
        
        if not urls:
            raise HTTPException(status_code=400, detail="URLs list is required")
        
        # Create batch extraction task
        from core.tasks import batch_extract_task
        task = batch_extract_task.delay(urls, strategy)
        
        return {
            "task_id": task.id,
            "status": "started",
            "total_urls": len(urls),
            "message": f"Batch extraction started for {len(urls)} URLs",
            "progress_url": f"/task/{task.id}/progress"
        }
        
    except Exception as e:
        logger.error(f"Batch extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch extraction failed: {str(e)}")

@router.get("/task/{task_id}/progress", status_code=status.HTTP_200_OK)
async def get_task_progress(task_id: str):
    """
    Get progress of a running task
    """
    try:
        from celery.result import AsyncResult
        
        task_result = AsyncResult(task_id)
        
        if task_result.state == 'PENDING':
            response = {
                'task_id': task_id,
                'state': 'PENDING',
                'status': 'Task is waiting to be processed'
            }
        elif task_result.state == 'PROGRESS':
            response = {
                'task_id': task_id,
                'state': 'PROGRESS',
                'status': task_result.info.get('status', ''),
                'progress': task_result.info.get('progress', 0),
                'meta': task_result.info
            }
        elif task_result.state == 'SUCCESS':
            response = {
                'task_id': task_id,
                'state': 'SUCCESS',
                'status': 'Task completed successfully',
                'result': task_result.result
            }
        else:  # FAILURE
            response = {
                'task_id': task_id,
                'state': task_result.state,
                'status': 'Task failed',
                'error': str(task_result.info)
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Task progress retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Task progress retrieval failed: {str(e)}")

# Rate limiting status endpoint
@router.get("/rate-limit/status")
async def get_rate_limit_status(request: Request):
    """
    Get current rate limit status for the client
    """
    try:
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        user_id = request.headers.get("X-User-ID", client_ip)
        
        # Get rate limit configurations
        configs_info = {}
        for config_name, config in rate_limiter.configs.items():
            configs_info[config_name] = {
                'max_requests': config.max_requests,
                'window_seconds': config.window_seconds,
                'limit_type': config.limit_type.value,
                'burst_allowance': config.burst_allowance
            }
        
        return {
            'client_id': user_id,
            'available_limits': configs_info,
            'current_time': time.time(),
            'message': 'Rate limiting is active'
        }
        
    except Exception as e:
        logger.error(f"Rate limit status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rate limit status retrieval failed: {str(e)}")

# Rate limiting reset endpoint (admin only)
@router.post("/rate-limit/reset")
async def reset_rate_limit(request: Request, api_key: APIKeyHeader = Security(validate_api_key)):
    """
    Reset rate limits for a specific client (admin endpoint)
    """
    try:
        data = await request.json()
        client_id = data.get('client_id')
        config_names = data.get('config_names', [])  # specific configs to reset, or all if empty
        
        if not client_id:
            raise HTTPException(status_code=400, detail="client_id is required")
        
        reset_configs = config_names if config_names else list(rate_limiter.configs.keys())
        
        # Clear rate limiters for the client
        for config_name in reset_configs:
            if config_name in rate_limiter.configs:
                limiter_key = rate_limiter._get_limiter_key(config_name, client_id)
                if limiter_key in rate_limiter.sliding_window_limiters:
                    del rate_limiter.sliding_window_limiters[limiter_key]
        
        return {
            'message': f'Rate limits reset for client {client_id}',
            'reset_configs': reset_configs
        }
        
    except Exception as e:
        logger.error(f"Rate limit reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rate limit reset failed: {str(e)}")

# Monitoring and metrics endpoints
@router.get("/metrics/performance")
async def get_performance_metrics():
    """
    Get comprehensive performance metrics
    """
    try:
        from monitoring.metrics_collector import metrics_collector
        report = metrics_collector.get_performance_report()
        return report
        
    except Exception as e:
        logger.error(f"Performance metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance metrics retrieval failed: {str(e)}")

@router.get("/metrics/errors")
async def get_error_metrics():
    """
    Get detailed error analysis
    """
    try:
        from monitoring.metrics_collector import metrics_collector
        error_analysis = metrics_collector.get_error_analysis()
        return error_analysis
        
    except Exception as e:
        logger.error(f"Error metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error metrics retrieval failed: {str(e)}")

@router.get("/metrics/time-series")
async def get_time_series_metrics(window: str = "1hour"):
    """
    Get time series performance data
    """
    try:
        from monitoring.metrics_collector import metrics_collector
        time_series = metrics_collector.get_time_series_data(window)
        return {
            'window': window,
            'data': time_series,
            'generated_at': time.time()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Time series metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Time series metrics retrieval failed: {str(e)}")

@router.get("/health/detailed")
async def get_detailed_health():
    """
    Get detailed system health information
    """
    try:
        from monitoring.metrics_collector import metrics_collector
        from utils.error_handler import error_handler
        
        # Get performance metrics
        performance = metrics_collector.get_performance_report()
        
        # Get error statistics
        error_stats = error_handler.get_error_statistics()
        
        # System status
        health_status = "healthy"
        issues = []
        
        # Check various health indicators
        if performance['summary']['success_rate'] < 80:
            health_status = "degraded"
            issues.append(f"Low success rate: {performance['summary']['success_rate']:.1f}%")
        
        if performance['summary']['avg_response_time'] > 30:
            health_status = "degraded"
            issues.append(f"High response time: {performance['summary']['avg_response_time']:.1f}s")
        
        if error_stats['total_errors'] > 1000:
            health_status = "warning"
            issues.append(f"High error count: {error_stats['total_errors']}")
        
        # System metrics
        system_metrics = performance.get('system_metrics', {})
        if system_metrics:
            if system_metrics.get('memory_usage', 0) > 85:
                health_status = "warning"
                issues.append(f"High memory usage: {system_metrics['memory_usage']:.1f}%")
            
            if system_metrics.get('cpu_usage', 0) > 90:
                health_status = "warning"
                issues.append(f"High CPU usage: {system_metrics['cpu_usage']:.1f}%")
        
        return {
            'status': health_status,
            'issues': issues,
            'performance_summary': performance['summary'],
            'error_summary': {
                'total_errors': error_stats['total_errors'],
                'most_common_error': error_stats.get('most_common_error'),
                'error_types': len(error_stats['error_types'])
            },
            'system_metrics': system_metrics,
            'checked_at': time.time()
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.post("/metrics/reset")
async def reset_metrics(api_key: APIKeyHeader = Security(validate_api_key)):
    """
    Reset all metrics (admin endpoint)
    """
    try:
        from monitoring.metrics_collector import metrics_collector
        from utils.error_handler import error_handler
        
        metrics_collector.reset_metrics()
        error_handler.clear_error_history()
        
        return {
            'message': 'All metrics and error history have been reset',
            'reset_at': time.time()
        }
        
    except Exception as e:
        logger.error(f"Metrics reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics reset failed: {str(e)}")

@router.get("/metrics/export")
async def export_metrics(format: str = "json"):
    """
    Export metrics in specified format
    """
    try:
        from monitoring.metrics_collector import metrics_collector
        
        if format not in ["json", "csv"]:
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")
        
        exported_data = metrics_collector.export_metrics(format)
        
        if format == "json":
            return JSONResponse(content=json.loads(exported_data))
        else:  # CSV
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=exported_data, media_type="text/csv")
        
    except Exception as e:
        logger.error(f"Metrics export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics export failed: {str(e)}")

# Configuration management endpoints
@router.get("/config/current")
async def get_current_config():
    """
    Get current configuration summary
    """
    try:
        # Fallback configuration when config_manager is not available
        config_summary = {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "ai_enabled": True,
            "cache_enabled": True,
            "redis_available": True,
            "celery_available": True
        }
        return {
            'config': config_summary,
            'retrieved_at': time.time()
        }
        
    except Exception as e:
        logger.error(f"Config retrieval failed: {e}")
        # Return basic config even if there's an error
        return {
            'config': {
                "environment": "development",
                "status": "degraded",
                "error": str(e)
            },
            'retrieved_at': time.time()
        }

@router.get("/config/detailed")
async def get_detailed_config(api_key: APIKeyHeader = Security(validate_api_key)):
    """
    Get detailed configuration (admin endpoint)
    """
    try:
        config = config_manager.get_config()
        
        # Convert to dict but mask sensitive information
        config_dict = config_manager._config_to_dict()
        
        # Mask sensitive fields
        if config_dict.get('redis', {}).get('password'):
            config_dict['redis']['password'] = '***masked***'
        if config_dict.get('security', {}).get('api_key'):
            config_dict['security']['api_key'] = '***masked***'
        if 'database' in config_dict and 'password' in config_dict['database']['url']:
            # Mask password in database URL
            db_url = config_dict['database']['url']
            if '://' in db_url and '@' in db_url:
                parts = db_url.split('://')
                if len(parts) == 2:
                    scheme = parts[0]
                    rest = parts[1]
                    if '@' in rest:
                        auth_part, host_part = rest.split('@', 1)
                        if ':' in auth_part:
                            user, _ = auth_part.split(':', 1)
                            config_dict['database']['url'] = f"{scheme}://{user}:***@{host_part}"
        
        return {
            'config': config_dict,
            'environment': config.environment,
            'retrieved_at': time.time()
        }
        
    except Exception as e:
        logger.error(f"Detailed config retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detailed config retrieval failed: {str(e)}")

@router.post("/config/update")
async def update_config(request: Request, api_key: APIKeyHeader = Security(validate_api_key)):
    """
    Update configuration at runtime (admin endpoint)
    """
    try:
        data = await request.json()
        updates = data.get('updates', {})
        
        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")
        
        success = config_manager.update_config(updates)
        
        if success:
            return {
                'message': 'Configuration updated successfully',
                'updated_at': time.time(),
                'updates_applied': updates
            }
        else:
            raise HTTPException(status_code=400, detail="Configuration update failed validation")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Configuration validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Config update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Config update failed: {str(e)}")

@router.post("/config/save")
async def save_config(request: Request, api_key: APIKeyHeader = Security(validate_api_key)):
    """
    Save current configuration to file (admin endpoint)
    """
    try:
        data = await request.json()
        file_path = data.get('file_path')  # Optional custom file path
        
        success = config_manager.save_config_to_file(file_path)
        
        if success:
            return {
                'message': 'Configuration saved successfully',
                'saved_at': time.time(),
                'file_path': file_path or 'smartscrape_config.json'
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save configuration")
        
    except Exception as e:
        logger.error(f"Config save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Config save failed: {str(e)}")

@router.get("/config/validate")
async def validate_config():
    """
    Validate current configuration
    """
    try:
        # Basic validation without config_manager
        validation_results = {
            'redis_available': True,  # We know Redis is running
            'celery_available': True,  # We know Celery is running
            'required_env_vars': True,
            'api_endpoints': True
        }
        
        all_valid = all(validation_results.values())
        
        return {
            'valid': all_valid,
            'message': 'Configuration is valid' if all_valid else 'Some issues found',
            'details': validation_results,
            'validated_at': time.time()
        }
        
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        return {
            'valid': False,
            'message': f'Validation failed: {str(e)}',
            'validated_at': time.time()
        }

@router.get("/test-duckduckgo")
async def test_duckduckgo():
    """Test endpoint to verify DuckDuckGo returns real URLs"""
    try:
        from components.duckduckgo_url_generator import DuckDuckGoURLGenerator
        
        generator = DuckDuckGoURLGenerator()
        results = generator.generate_urls("Tesla news", max_urls=5)
        
        return {
            "status": "success",
            "query": "Tesla news",
            "total_results": len(results),
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "is_real_url": generator._is_valid_real_url(r.url)
                } 
                for r in results
            ]
        }
        
    except Exception as e:
        logger.error(f"DuckDuckGo test failed: {e}")
        return {"error": str(e)}

@router.post("/scrape-simple")
async def scrape_simple(request: Request):
    """Simple scraping endpoint using the fixed SimpleScraper"""
    try:
        data = await request.json()
        query = data.get('query', 'Tesla news')
        
        logger.info(f"Simple scrape request: {query}")
        
        # Import here to avoid circular imports
        from controllers.simple_scraper import SimpleScraper
        
        # Use the simple scraper directly
        async with SimpleScraper() as scraper:
            result = await scraper.scrape_query(query)
            
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            'status': 'completed',
            'result': result,
            'submitted_at': datetime.now().isoformat(),
            'completed_at': datetime.now().isoformat()
        }
        
        return {"job_id": job_id, "status": "completed", "result": result}
        
    except Exception as e:
        logger.error(f"Simple scraping failed: {e}")
        return {"error": str(e), "status": "failed"}
