"""
Configuration settings for the SmartScrape application.
This file contains all configurable parameters and settings.
"""

import os
import secrets
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_env_bool(key: str, default: bool = False) -> bool:
    """Convert environment variable to boolean."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

def get_env_int(key: str, default: int) -> int:
    """Convert environment variable to integer."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

def get_env_float(key: str, default: float) -> float:
    """Convert environment variable to float."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default

# API Keys & Authentication
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Environment Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = get_env_bool("DEBUG", ENVIRONMENT == "development")

# Validate required API keys only in production
if not GEMINI_API_KEY and ENVIRONMENT == "production":
    raise ValueError("GEMINI_API_KEY environment variable is required in production mode")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32) if DEBUG else None)

if not SECRET_KEY and ENVIRONMENT == "production":
    raise ValueError("SECRET_KEY environment variable is required in production")

# AI Settings
USE_AI = get_env_bool("USE_AI", True)
DEFAULT_AI_MODEL = os.getenv("DEFAULT_AI_MODEL", "gemini-2.0-flash-lite")
AI_CACHE_ENABLED = get_env_bool("AI_CACHE_ENABLED", True)
AI_CACHE_TTL = get_env_int("AI_CACHE_TTL", 3600)

# Enhanced crawl4ai settings
CRAWL4AI_ENABLED = get_env_bool("CRAWL4AI_ENABLED", True)
CRAWL4AI_MAX_PAGES = get_env_int("CRAWL4AI_MAX_PAGES", 50)
CRAWL4AI_DEEP_CRAWL = get_env_bool("CRAWL4AI_DEEP_CRAWL", True)
CRAWL4AI_MEMORY_THRESHOLD = get_env_float("CRAWL4AI_MEMORY_THRESHOLD", 80.0)
CRAWL4AI_AI_PATHFINDING = get_env_bool("CRAWL4AI_AI_PATHFINDING", True)

# spaCy settings
SPACY_ENABLED = get_env_bool("SPACY_ENABLED", True)
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_md")
SPACY_MODEL_NAME = SPACY_MODEL  # Alias for backward compatibility
SPACY_INTENT_ANALYSIS = get_env_bool("SPACY_INTENT_ANALYSIS", True)
SPACY_POST_EXTRACTION_FILTERING = get_env_bool("SPACY_POST_EXTRACTION_FILTERING", True)

# Semantic Search & Intent Analysis
SEMANTIC_SEARCH_ENABLED = get_env_bool("SEMANTIC_SEARCH_ENABLED", True)
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
CONTEXTUAL_QUERY_EXPANSION = get_env_bool("CONTEXTUAL_QUERY_EXPANSION", True)
SEMANTIC_SIMILARITY_THRESHOLD = get_env_float("SEMANTIC_SIMILARITY_THRESHOLD", 0.75)

# AI Schema Generation
AI_SCHEMA_GENERATION_ENABLED = get_env_bool("AI_SCHEMA_GENERATION_ENABLED", True)
PYDANTIC_VALIDATION_ENABLED = get_env_bool("PYDANTIC_VALIDATION_ENABLED", True)
SCHEMA_VALIDATION_STRICT = get_env_bool("SCHEMA_VALIDATION_STRICT", False)
ADAPTIVE_SCHEMA_REFINEMENT = get_env_bool("ADAPTIVE_SCHEMA_REFINEMENT", True)

# Progressive Data Collection
PROGRESSIVE_DATA_COLLECTION = get_env_bool("PROGRESSIVE_DATA_COLLECTION", True)
DATA_CONSISTENCY_CHECKS = get_env_bool("DATA_CONSISTENCY_CHECKS", True)
CROSS_REFERENCE_VALIDATION = get_env_bool("CROSS_REFERENCE_VALIDATION", True)

# Resilience & Error Handling
USE_UNDETECTED_CHROMEDRIVER = get_env_bool("USE_UNDETECTED_CHROMEDRIVER", True)
CIRCUIT_BREAKER_ENABLED = get_env_bool("CIRCUIT_BREAKER_ENABLED", True)
CIRCUIT_BREAKER_FAILURE_THRESHOLD = get_env_int("CIRCUIT_BREAKER_FAILURE_THRESHOLD", 5)
CIRCUIT_BREAKER_TIMEOUT = get_env_int("CIRCUIT_BREAKER_TIMEOUT", 60)
RETRY_STRATEGIES_ENABLED = get_env_bool("RETRY_STRATEGIES_ENABLED", True)
MAX_RETRY_ATTEMPTS = get_env_int("MAX_RETRY_ATTEMPTS", 3)
EXPONENTIAL_BACKOFF = get_env_bool("EXPONENTIAL_BACKOFF", True)

# Crawling & Extraction Settings
DEFAULT_MAX_PAGES = get_env_int("DEFAULT_MAX_PAGES", 100)
DEFAULT_MAX_DEPTH = get_env_int("DEFAULT_MAX_DEPTH", 2)
SEARCH_DEPTH = get_env_int("SEARCH_DEPTH", 3)  # Maximum depth for search operations
DEFAULT_TIMEOUT_SECONDS = get_env_int("DEFAULT_TIMEOUT_SECONDS", 300)
DEFAULT_MAX_CONCURRENT_REQUESTS = get_env_int("DEFAULT_MAX_CONCURRENT_REQUESTS", 5)
DEFAULT_MIN_DELAY = get_env_float("DEFAULT_MIN_DELAY", 0.5)
DEFAULT_MAX_DELAY = get_env_float("DEFAULT_MAX_DELAY", 2.0)

# Browser Settings
DEFAULT_USE_BROWSER = get_env_bool("DEFAULT_USE_BROWSER", True)
DEFAULT_DISABLE_IMAGES = get_env_bool("DEFAULT_DISABLE_IMAGES", True)
DEFAULT_DISABLE_CSS = get_env_bool("DEFAULT_DISABLE_CSS", True)
DEFAULT_DISABLE_JAVASCRIPT = get_env_bool("DEFAULT_DISABLE_JAVASCRIPT", False)

# Cache Settings
DEFAULT_CACHE_MODE = os.getenv("DEFAULT_CACHE_MODE", "memory")
CACHE_DIRECTORY = os.getenv("CACHE_DIRECTORY", "./cache")

# Caching (Redis)
REDIS_CACHE_ENABLED = get_env_bool("REDIS_CACHE_ENABLED", True)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = get_env_int("REDIS_PORT", 6379)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_DB = get_env_int("REDIS_DB", 0)

# Enhanced Redis Configuration
REDIS_CONFIG = {
    'host': REDIS_HOST,
    'port': REDIS_PORT,
    'db': REDIS_DB,
    'password': REDIS_PASSWORD,
    'decode_responses': True,
    'socket_connect_timeout': get_env_int("REDIS_CONNECT_TIMEOUT", 5),
    'socket_timeout': get_env_int("REDIS_SOCKET_TIMEOUT", 5),
    'retry_on_timeout': get_env_bool("REDIS_RETRY_ON_TIMEOUT", True),
    'health_check_interval': get_env_int("REDIS_HEALTH_CHECK_INTERVAL", 30)
}

# Cache TTL Settings
CACHE_TTL = {
    'content': get_env_int("CACHE_TTL_CONTENT", 3600),  # 1 hour
    'metadata': get_env_int("CACHE_TTL_METADATA", 7200),  # 2 hours
    'schema': get_env_int("CACHE_TTL_SCHEMA", 86400),   # 24 hours
    'embeddings': get_env_int("CACHE_TTL_EMBEDDINGS", 604800),  # 1 week
    'default': get_env_int("CACHE_TTL_DEFAULT", 3600)
}

# Legacy TTL settings for backward compatibility
CACHE_TTL_DEFAULT = CACHE_TTL['default']
CACHE_TTL_SCHEMA = CACHE_TTL['schema']
CACHE_TTL_EMBEDDINGS = CACHE_TTL['embeddings']

# Unified Output System
UNIFIED_OUTPUT_ENABLED = get_env_bool("UNIFIED_OUTPUT_ENABLED", True)
OUTPUT_FORMAT_STANDARDIZATION = get_env_bool("OUTPUT_FORMAT_STANDARDIZATION", True)
METADATA_PRESERVATION = get_env_bool("METADATA_PRESERVATION", True)
DATA_PROVENANCE_TRACKING = get_env_bool("DATA_PROVENANCE_TRACKING", True)
CONFIDENCE_SCORING = get_env_bool("CONFIDENCE_SCORING", True)

# Integration controls
UNIVERSAL_STRATEGY_ENABLED = get_env_bool("UNIVERSAL_STRATEGY_ENABLED", False)
# Permanently disable AI-based URL generation in favor of DuckDuckGo
INTELLIGENT_URL_GENERATION = get_env_bool("INTELLIGENT_URL_GENERATION", False)

# URL Generation Settings - Permanently switch to DuckDuckGo for all URL generation
USE_DUCKDUCKGO_BY_DEFAULT = get_env_bool("USE_DUCKDUCKGO_BY_DEFAULT", True)
PREFER_SEARCH_OVER_AI_URLS = get_env_bool("PREFER_SEARCH_OVER_AI_URLS", True)
# Force DuckDuckGo to be the only URL generation method
FORCE_DUCKDUCKGO_ONLY = get_env_bool("FORCE_DUCKDUCKGO_ONLY", True)

# User Feedback Loop
USER_FEEDBACK_ENABLED = get_env_bool("USER_FEEDBACK_ENABLED", False)

# Server Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = get_env_int("PORT", 5000)

# Security Settings
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else ["*"]
RATE_LIMIT_REQUESTS_PER_MINUTE = get_env_int("RATE_LIMIT_REQUESTS_PER_MINUTE", 100)
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")

# Database Settings
DATABASE_ENABLED = get_env_bool("DATABASE_ENABLED", True)  # Enable by default for development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///smartscrape.db")  # Default to SQLite
DATABASE_CONFIG = {
    'pool_size': get_env_int("DB_POOL_SIZE", 5),
    'max_overflow': get_env_int("DB_MAX_OVERFLOW", 10),
    'pool_timeout': get_env_int("DB_POOL_TIMEOUT", 30),
    'pool_recycle': get_env_int("DB_POOL_RECYCLE", 3600),
    'echo': get_env_bool("DB_ECHO", False)  # Set to True for SQL query logging
}

# Monitoring & Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")
SENTRY_DSN = os.getenv("SENTRY_DSN")
METRICS_ENABLED = get_env_bool("METRICS_ENABLED", True)

# Proxy Settings
PROXY_ENABLED = get_env_bool("PROXY_ENABLED", False)
PROXY_ROTATION_ENABLED = get_env_bool("PROXY_ROTATION_ENABLED", False)
PROXY_LIST = os.getenv("PROXY_LIST", "").split(",") if os.getenv("PROXY_LIST") else []

# Resource Limits
MAX_MEMORY_MB = get_env_int("MAX_MEMORY_MB", 1024)
MAX_CPU_PERCENT = get_env_int("MAX_CPU_PERCENT", 80)
MAX_DISK_USAGE_PERCENT = get_env_int("MAX_DISK_USAGE_PERCENT", 85)

# Backup Settings
BACKUP_ENABLED = get_env_bool("BACKUP_ENABLED", False)
BACKUP_SCHEDULE = os.getenv("BACKUP_SCHEDULE", "0 2 * * *")
BACKUP_RETENTION_DAYS = get_env_int("BACKUP_RETENTION_DAYS", 30)

# File Paths
TEMPLATE_STORAGE_PATH = "./extraction_strategies"
STATIC_FILES_PATH = "./static"
TEMPLATES_PATH = "./web/templates"

# Supported Export Formats
EXPORT_FORMATS = ["json", "csv", "excel"]

# Content Filter Settings
DEFAULT_EXTRACT_HEADERS = True
DEFAULT_EXTRACT_LISTS = True
DEFAULT_EXTRACT_TABLES = True
DEFAULT_EXTRACT_LINKS = False
DEFAULT_EXTRACT_IMAGES = False

# Application settings
APP_NAME = "SmartScrape"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "Intelligent Web Scraper with AI-Guided Crawling"

# Server settings
DEBUG = True

# Crawler settings
DEFAULT_CONCURRENT_REQUESTS = 5

# Extraction settings
DEFAULT_EXTRACTION_STRATEGY = "ai-guided"  # "ai-guided", "best-first", "bfs", or "dfs"

# Define crawler strategies with their display names
CRAWLER_STRATEGIES = {
    "ai-guided": "AI-Guided Search (Intelligent)",
    "best-first": "Best-First Search (Relevance-based)",
    "bfs": "Breadth-First Search (Level by level)",
    "dfs": "Depth-First Search (Path by path)"
}

# Define extraction methods
EXTRACTION_METHODS = {
    "raw": "Raw Content Extraction",
    "structured": "Structured Data Extraction",
    "css": "CSS Selector-Based Extraction",
    "hybrid": "Hybrid Extraction (AI + CSS)"
}

# Default browser configuration
DEFAULT_BROWSER_CONFIG = {
    "headless": True,
    "disable_images": True,
    "disable_css": True,
    "disable_javascript": False
}

# Create a unified configuration dictionary
def get_config() -> Dict[str, Any]:
    """Get the full application configuration."""
    return {
        "app": {
            "name": APP_NAME,
            "version": APP_VERSION,
            "description": APP_DESCRIPTION
        },
        "server": {
            "host": HOST,
            "port": PORT,
            "debug": DEBUG
        },
        "crawler": {
            "max_depth": DEFAULT_MAX_DEPTH,
            "max_pages": DEFAULT_MAX_PAGES,
            "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
            "min_delay": DEFAULT_MIN_DELAY,
            "max_delay": DEFAULT_MAX_DELAY,
            "concurrent_requests": DEFAULT_CONCURRENT_REQUESTS,
            "cache_mode": DEFAULT_CACHE_MODE,
            "cache_directory": CACHE_DIRECTORY
        },
        "extraction": {
            "default_strategy": DEFAULT_EXTRACTION_STRATEGY,
            "strategies": CRAWLER_STRATEGIES,
            "methods": EXTRACTION_METHODS
        },
        "browser": DEFAULT_BROWSER_CONFIG,
        "ai": {
            "enabled": USE_AI,
            "gemini_api_key": GEMINI_API_KEY,
            "default_model": DEFAULT_AI_MODEL,
            "cache_enabled": AI_CACHE_ENABLED,
            "cache_ttl": AI_CACHE_TTL
        }
    }

class Config:
    """Configuration class for test compatibility and structured access to settings."""
    
    # AI Settings
    USE_AI = USE_AI
    DEFAULT_AI_MODEL = DEFAULT_AI_MODEL
    AI_CACHE_ENABLED = AI_CACHE_ENABLED
    AI_CACHE_TTL = AI_CACHE_TTL
    GEMINI_API_KEY = GEMINI_API_KEY
    
    # spaCy settings
    SPACY_ENABLED = SPACY_ENABLED
    SPACY_MODEL = SPACY_MODEL
    SPACY_MODEL_NAME = SPACY_MODEL_NAME
    SPACY_INTENT_ANALYSIS = SPACY_INTENT_ANALYSIS
    
    # Semantic Search
    SEMANTIC_SEARCH_ENABLED = SEMANTIC_SEARCH_ENABLED
    SENTENCE_TRANSFORMER_MODEL = SENTENCE_TRANSFORMER_MODEL
    SEMANTIC_SIMILARITY_THRESHOLD = SEMANTIC_SIMILARITY_THRESHOLD
    
    # Crawling settings
    DEFAULT_MAX_PAGES = DEFAULT_MAX_PAGES
    DEFAULT_MAX_DEPTH = DEFAULT_MAX_DEPTH
    DEFAULT_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_SECONDS
    
    # Browser settings
    USE_UNDETECTED_CHROMEDRIVER = USE_UNDETECTED_CHROMEDRIVER
    
    @classmethod
    def get_dict(cls) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return get_config()