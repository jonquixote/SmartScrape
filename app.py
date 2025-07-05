# app.py - Intelligent Web Scraper Application
# Combining crawl4ai, Google's Generative AI, and Beautiful Soup

# Standard library imports
import asyncio
import os
from datetime import datetime
from typing import Dict, Any

# FastAPI for the web framework
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

# Import our configuration
import config

# Import logging utilities
from utils.logging import log_request_middleware, get_logger

# Setup logger
logger = get_logger("app")

# Import our components
from web.routes import router as api_router
from strategies.base_strategy import BaseStrategy
from components.site_discovery import SiteDiscovery
from components.search_automation import SearchFormDetector
from components.domain_intelligence import DomainIntelligence
from components.pagination_handler import PaginationHandler
from components.template_storage import TemplateStorage
from components.search_template_integration import SearchTemplateIntegrator
from crawl4ai import AsyncWebCrawler

# Import the new UniversalHunter
from intelligence.universal_hunter import UniversalHunter, HuntingIntent

# Import our AI helpers
from ai_helpers.intent_parser import get_intent_parser
from ai_helpers.prompt_generator import PromptGenerator

# Import the centralized AI service
from core.ai_service import AIService
from core.service_registry import ServiceRegistry

# Initialize the centralized AI service
ai_service = AIService()

# Build AI service configuration with proper model definitions
ai_config = {
    "models": [],
    "default_model": "default",
    "cache": {
        "backend": "memory",
        "default_ttl": config.AI_CACHE_TTL,
        "enabled": config.AI_CACHE_ENABLED
    },
    "content_processor": {},
    "rule_engine": {},
    "batch_processor": {}
}

# Add available models based on environment variables
model_added = False

# Google/Gemini models
if config.GEMINI_API_KEY:
    # Use the actual model ID as the name so ModelSelector can find it
    model_name = config.DEFAULT_AI_MODEL  # This will be "gemini-2.0-flash-lite"
    ai_config["models"].append({
        "name": model_name,
        "type": "google",
        "model_id": model_name,
        "api_key": config.GEMINI_API_KEY
    })
    if not model_added:
        ai_config["default_model"] = model_name
        model_added = True

# OpenAI models (if API key is available)
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    model_name = "gpt-3.5-turbo"  # Use specific model name
    ai_config["models"].append({
        "name": model_name,
        "type": "openai", 
        "model_id": model_name,
        "api_key": openai_api_key
    })
    if not model_added:
        ai_config["default_model"] = model_name
        model_added = True

# Anthropic models (if API key is available)
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if anthropic_api_key:
    model_name = "claude-3-sonnet-20240229"  # Use specific model name
    ai_config["models"].append({
        "name": model_name,
        "type": "anthropic",
        "model_id": model_name,
        "api_key": anthropic_api_key
    })
    if not model_added:
        ai_config["default_model"] = model_name
        model_added = True

# If no real models are available, add a mock model for development
if not model_added:
    ai_config["models"].append({
        "name": "mock",
        "type": "mock",
        "model_id": "mock-model"
    })
    ai_config["default_model"] = "mock"

ai_service.initialize(config=ai_config)

# Register the AI service in the service registry
service_registry = ServiceRegistry()
service_registry.register_service("ai_service", ai_service)

# Initialize intent parser with the AI service
intent_parser = get_intent_parser(use_ai=config.USE_AI)

# Initialize prompt generator with the AI service
prompt_generator = PromptGenerator(ai_service=ai_service)

# Initialize web crawler lazily (will be created when needed)
crawler = None

def get_crawler():
    """Get or create the AsyncWebCrawler instance."""
    global crawler
    if crawler is None:
        crawler = AsyncWebCrawler()
    return crawler

def get_search_template_integrator():
    """Get or create the SearchTemplateIntegrator instance."""
    global search_template_integrator
    if search_template_integrator is None:
        search_template_integrator = SearchTemplateIntegrator(get_crawler())
    return search_template_integrator

# Initialize our intelligent scraping components
site_discovery = SiteDiscovery()
search_detector = SearchFormDetector()
domain_intelligence = DomainIntelligence()
pagination_handler = PaginationHandler()
template_storage = TemplateStorage(config.TEMPLATE_STORAGE_PATH)
# Initialize search template integrator without crawler initially (will be set later)
search_template_integrator = None

# Initialize the UniversalHunter with the AI service
universal_hunter = UniversalHunter(ai_service_client=ai_service)

# Lifespan event handler for initializing and cleaning up services
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    try:
        logger.info("Initializing services on startup...")
        
        # Initialize model discovery service
        from core.model_discovery_service import get_model_discovery_service
        discovery_service = get_model_discovery_service()
        
        # Start background model discovery
        asyncio.create_task(discovery_service.start_background_discovery())
        logger.info("Model discovery service started")
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup initialization: {e}")
    
    yield  # Application runs here
    
    # Shutdown
    try:
        logger.info("Shutting down services...")
        
        # Shutdown model discovery service
        from core.model_discovery_service import get_model_discovery_service
        discovery_service = get_model_discovery_service()
        await discovery_service.shutdown()
        
        logger.info("All services shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Initialize FastAPI app with metadata and lifespan handler
app = FastAPI(
    title=f"{config.APP_NAME}",
    description=f"{config.APP_DESCRIPTION}",
    version=os.getenv("APP_VERSION", "dev"),
    docs_url=None if config.ENVIRONMENT == "production" else "/docs",
    redoc_url=None if config.ENVIRONMENT == "production" else "/redoc",
    openapi_url=None if config.ENVIRONMENT == "production" else "/openapi.json",
    lifespan=lifespan
)

# Configure security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    
    # Add security headers in production
    if config.ENVIRONMENT == "production":
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
    return response

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add session middleware for API key storage
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "your-secret-key-change-in-production")
)

# Add GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add request logging middleware
app.add_middleware(BaseHTTPMiddleware, dispatch=log_request_middleware)

# Set up rate limiting
if config.RATE_LIMIT_REQUESTS_PER_MINUTE > 0:
    # Use our custom implementation directly
    from web.rate_limiting import RateLimitMiddleware
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=config.RATE_LIMIT_REQUESTS_PER_MINUTE,
        exclude_paths=["/health", "/metrics"]
    )
    logger.info(f"Rate limiting enabled with limit of {config.RATE_LIMIT_REQUESTS_PER_MINUTE} requests per minute")

# Mount static files directory
from fastapi.staticfiles import StaticFiles
import os
static_folder = os.path.join(os.path.dirname(__file__), "web", "static")
app.mount("/static", StaticFiles(directory=static_folder), name="static")
logger.info(f"Static files mounted from {static_folder}")

# Include our API routes with prefix
app.include_router(api_router, prefix="/api")

# Include the same router without prefix to serve frontend at root
app.include_router(api_router)

# New function to parse user intent and prepare for scraping
async def prepare_scraping_job(user_query: str, output_schema: Dict[str, Any]):
    """
    Uses the UniversalHunter to perform a scraping job.

    Args:
        user_query: The user's natural language query.
        output_schema: The desired JSON schema for the output.

    Returns:
        A list of HuntingResult objects.
    """
    # Parse the user's intent to extract keywords and entities
    parsed_intent = await intent_parser.parse_query(user_query)
    
    # Create a HuntingIntent
    intent = HuntingIntent(
        query=user_query,
        output_schema=output_schema,
        keywords=parsed_intent.get("keywords", []),
        entities=parsed_intent.get("entities", [])
    )

    # Run the hunt
    results = await universal_hunter.hunt(intent)

    return results

# Main function to run the application
if __name__ == "__main__":
    import argparse
    import uvicorn
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=f"Start {config.APP_NAME} server")
    parser.add_argument("--host", default=config.HOST, help=f"Host to bind to (default: {config.HOST})")
    parser.add_argument("--port", type=int, default=config.PORT, help=f"Port to bind to (default: {config.PORT})")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Log level")
    
    args = parser.parse_args()
    
    print(f"Starting {config.APP_NAME} v{config.APP_VERSION}...")
    print(f"AI Service initialized with model: {config.DEFAULT_AI_MODEL}")
    print(f"AI Intent Parsing: {'Enabled' if config.USE_AI else 'Disabled'}")
    print(f"Server running at http://{args.host}:{args.port}")
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port, 
        reload=args.reload,
        log_level=args.log_level
    )
