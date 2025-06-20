# web/app.py - FastAPI App Factory for SmartScrape
"""
FastAPI application factory for SmartScrape.
Provides a clean entry point for the web API and supports different configurations.
"""

import os
import logging
from datetime import datetime
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

# Import configuration
import config

# Import logging utilities
from utils.logging import log_request_middleware, get_logger

# Import routes
from web.routes import router as api_router

# Import core services
from core.ai_service import AIService
from core.service_registry import ServiceRegistry

# Setup logger
logger = get_logger("web.app")

def create_app(debug: bool = False) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        debug: Enable debug mode with additional logging and features
        
    Returns:
        Configured FastAPI application instance
    """
    
    # Create FastAPI app with metadata
    app = FastAPI(
        title=config.APP_NAME,
        description="Intelligent Web Scraping with AI-powered content extraction",
        version=config.APP_VERSION,
        debug=debug,
        docs_url="/docs" if debug else None,
        redoc_url="/redoc" if debug else None
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add session middleware
    app.add_middleware(
        SessionMiddleware,
        secret_key=config.SECRET_KEY if hasattr(config, 'SECRET_KEY') else "smartscrape-dev-key"
    )
    
    # Add request logging middleware
    app.add_middleware(BaseHTTPMiddleware, dispatch=log_request_middleware)
    
    # Initialize core services
    initialize_services(app)
    
    # Include API routes
    app.include_router(api_router, prefix="/api")
    
    # Add startup and shutdown events
    setup_lifecycle_events(app)
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": config.APP_VERSION,
            "services": {
                "ai_service": "initialized" if hasattr(app.state, 'ai_service') else "not_initialized",
                "service_registry": "initialized" if hasattr(app.state, 'service_registry') else "not_initialized"
            }
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "name": config.APP_NAME,
            "version": config.APP_VERSION,
            "description": "Intelligent Web Scraping API",
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "api": "/api"
            }
        }
    
    logger.info(f"FastAPI app created - {config.APP_NAME} v{config.APP_VERSION}")
    return app

def initialize_services(app: FastAPI):
    """Initialize core services and attach to app state"""
    try:
        # Initialize AI service
        ai_service = AIService()
        app.state.ai_service = ai_service
        
        # Initialize service registry
        service_registry = ServiceRegistry()
        app.state.service_registry = service_registry
        
        logger.info("Core services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

def setup_lifecycle_events(app: FastAPI):
    """Setup application lifecycle events"""
    
    @app.on_event("startup")
    async def startup_event():
        """Application startup tasks"""
        logger.info(f"Starting {config.APP_NAME} v{config.APP_VERSION}")
        logger.info(f"AI Service model: {config.DEFAULT_AI_MODEL}")
        logger.info(f"AI Intent Parsing: {'Enabled' if config.USE_AI else 'Disabled'}")
        
        # Additional startup tasks can be added here
        # - Database connections
        # - Cache initialization
        # - Background task setup
        
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown tasks"""
        logger.info(f"Shutting down {config.APP_NAME}")
        
        # Cleanup tasks
        # - Close database connections
        # - Stop background tasks
        # - Clear caches

# Create the default app instance
app = create_app(debug=config.DEBUG if hasattr(config, 'DEBUG') else False)

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=f"Start {config.APP_NAME} server")
    parser.add_argument("--host", default=config.HOST, help=f"Host to bind to (default: {config.HOST})")
    parser.add_argument("--port", type=int, default=config.PORT, help=f"Port to bind to (default: {config.PORT})")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Log level")
    
    args = parser.parse_args()
    
    # Create app with debug mode if specified
    if args.debug:
        app = create_app(debug=True)
    
    print(f"Starting {config.APP_NAME} v{config.APP_VERSION}...")
    print(f"Server running at http://{args.host}:{args.port}")
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port, 
        reload=args.reload,
        log_level=args.log_level
    )
