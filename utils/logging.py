"""
Structured logging configuration for SmartScrape.
Uses structlog for structured JSON logs in production and more readable logs in development.
"""

import logging
import sys
import os
from datetime import datetime
import structlog
from typing import Any, Dict, Optional

import config

# Configure standard logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(message)s",
    stream=sys.stdout,
)

# Determine if we should use JSON format (in production) or pretty console output (in development)
USE_JSON_LOGS = config.LOG_FORMAT.lower() in ("json", "structured") or config.ENVIRONMENT == "production"

# Configure structlog processors based on environment
pre_chain = [
    # Add the log level and a timestamp to the event_dict if the log entry is not from structlog
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso"),
]

# Set up structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        # Removed format_exc_info processor - it's not needed with ConsoleRenderer
        # and causes warnings with pretty exception formatting
        # Add trace context if Sentry is enabled
        *([structlog.processors.CallsiteParameterAdder(
            parameters={
                "file": structlog.processors.CallsiteParameter.PATHNAME,
                "func": structlog.processors.CallsiteParameter.FUNC_NAME,
                "line": structlog.processors.CallsiteParameter.LINENO,
            }
        )] if config.SENTRY_DSN else []),
        structlog.processors.UnicodeDecoder(),
        # Use JSON in production, pretty console output in development
        structlog.processors.JSONRenderer() if USE_JSON_LOGS else structlog.dev.ConsoleRenderer(colors=True),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Create a request ID context var
import contextvars
request_id_contextvar = contextvars.ContextVar("request_id", default=None)

def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return request_id_contextvar.get()

def set_request_id(request_id: str) -> None:
    """Set the current request ID in context."""
    request_id_contextvar.set(request_id)

def clear_request_id() -> None:
    """Clear the request ID from context."""
    request_id_contextvar.set(None)

# API logging middleware
async def log_request_middleware(request, call_next):
    """Middleware to log requests and responses with structured logs."""
    # Generate request ID if not provided
    request_id = request.headers.get("X-Request-ID") or f"req-{datetime.now().timestamp()}"
    set_request_id(request_id)
    
    # Get the logger
    log = structlog.get_logger("api")
    
    # Start timer
    start_time = datetime.now()
    
    # Log request
    log.info(
        "Request received",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("User-Agent", "unknown"),
    )
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log response
        log.info(
            "Response sent",
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
    except Exception as e:
        # Log exception
        log.exception(
            "Request failed",
            request_id=request_id,
            error=str(e),
            exc_info=True,
        )
        raise
    finally:
        # Clear request ID
        clear_request_id()

# Get a configured logger
def get_logger(name: str = None) -> structlog.typing.FilteringBoundLogger:
    """Get a pre-configured structlog logger."""
    return structlog.get_logger(name)
