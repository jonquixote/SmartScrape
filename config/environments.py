"""
Environment-Specific Configuration for SmartScrape

This module provides configuration management with environment-specific settings,
validation, and runtime configuration updates.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RedisConfig:
    """Redis configuration settings"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    decode_responses: bool = True
    socket_connect_timeout: int = 5
    socket_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    max_connections: int = 10

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = "sqlite:///smartscrape.db"
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600

@dataclass
class PerformanceConfig:
    """Performance-related configuration"""
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    connection_pool_size: int = 100
    max_memory_mb: int = 1024
    cleanup_threshold: float = 0.8

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    burst_allowance: int = 10
    sliding_window_size: int = 60

@dataclass
class CacheConfig:
    """Caching configuration"""
    enabled: bool = True
    default_ttl: int = 3600
    max_memory_mb: int = 256
    compression: bool = True
    cache_ttl: Dict[str, int] = field(default_factory=lambda: {
        'content': 3600,
        'metadata': 7200,
        'schema': 86400
    })

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    log_to_file: bool = True
    log_file: str = "smartscrape.log"

@dataclass
class SecurityConfig:
    """Security configuration"""
    api_key_required: bool = True
    api_key: Optional[str] = None
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limiting_enabled: bool = True
    content_filtering_enabled: bool = True

@dataclass
class EnvironmentConfig:
    """Complete environment configuration"""
    environment: str = "development"
    debug: bool = False
    redis: RedisConfig = field(default_factory=RedisConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Additional environment-specific settings
    extra_settings: Dict[str, Any] = field(default_factory=dict)

class DevelopmentConfig(EnvironmentConfig):
    """Development environment configuration"""
    def __init__(self):
        super().__init__()
        self.environment = "development"
        self.debug = True
        
        # Development-specific settings
        self.performance.max_concurrent_requests = 5
        self.performance.request_timeout = 60
        self.performance.max_memory_mb = 512
        
        self.rate_limit.requests_per_minute = 30
        self.rate_limit.requests_per_hour = 1800
        
        self.security.api_key_required = False
        self.security.rate_limiting_enabled = False
        
        self.logging.level = "DEBUG"
        self.logging.log_to_file = False
        
        self.cache.default_ttl = 300  # 5 minutes for development

class ProductionConfig(EnvironmentConfig):
    """Production environment configuration"""
    def __init__(self):
        super().__init__()
        self.environment = "production"
        self.debug = False
        
        # Production-specific settings
        self.redis.host = os.getenv("REDIS_HOST", "redis")
        self.redis.password = os.getenv("REDIS_PASSWORD")
        
        self.database.url = os.getenv("DATABASE_URL", "postgresql://user:pass@db/smartscrape")
        self.database.pool_size = 20
        self.database.max_overflow = 50
        
        self.performance.max_concurrent_requests = 50
        self.performance.request_timeout = 15
        self.performance.max_memory_mb = 2048
        
        self.rate_limit.requests_per_minute = 100
        self.rate_limit.requests_per_hour = 6000
        
        self.security.api_key_required = True
        self.security.api_key = os.getenv("SMARTSCRAPE_API_KEY")
        self.security.allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
        
        self.logging.level = "INFO"
        self.logging.log_to_file = True
        
        self.cache.default_ttl = 3600
        self.cache.max_memory_mb = 1024

class TestingConfig(EnvironmentConfig):
    """Testing environment configuration"""
    def __init__(self):
        super().__init__()
        self.environment = "testing"
        self.debug = True
        
        # Testing-specific settings
        self.redis.db = 1  # Different DB for testing
        self.database.url = "sqlite:///:memory:"
        
        self.performance.max_concurrent_requests = 2
        self.performance.request_timeout = 5
        self.performance.max_memory_mb = 256
        
        self.rate_limit.requests_per_minute = 10
        self.rate_limit.requests_per_hour = 600
        
        self.security.api_key_required = False
        self.security.rate_limiting_enabled = False
        
        self.logging.level = "DEBUG"
        self.logging.log_to_file = False
        
        self.cache.enabled = False  # Disable caching for testing

class StagingConfig(EnvironmentConfig):
    """Staging environment configuration"""
    def __init__(self):
        super().__init__()
        self.environment = "staging"
        self.debug = False
        
        # Staging-specific settings (similar to prod but more lenient)
        self.redis.host = os.getenv("REDIS_HOST", "redis-staging")
        self.database.url = os.getenv("DATABASE_URL", "postgresql://user:pass@db-staging/smartscrape")
        
        self.performance.max_concurrent_requests = 25
        self.performance.request_timeout = 30
        self.performance.max_memory_mb = 1024
        
        self.rate_limit.requests_per_minute = 80
        self.rate_limit.requests_per_hour = 4800
        
        self.security.api_key_required = True
        self.security.api_key = os.getenv("SMARTSCRAPE_API_KEY", "staging-key")
        
        self.logging.level = "INFO"
        self.logging.log_to_file = True

class ConfigurationManager:
    """Manages configuration loading, validation, and runtime updates"""
    
    def __init__(self):
        self.config: Optional[EnvironmentConfig] = None
        self.config_file_path: Optional[Path] = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration based on environment"""
        env = os.getenv("SMARTSCRAPE_ENV", "development").lower()
        
        # Load base config based on environment
        if env == "production":
            self.config = ProductionConfig()
        elif env == "testing":
            self.config = TestingConfig()
        elif env == "staging":
            self.config = StagingConfig()
        else:
            self.config = DevelopmentConfig()
        
        # Load additional config from file if specified
        config_file = os.getenv("SMARTSCRAPE_CONFIG_FILE")
        if config_file:
            self._load_config_from_file(config_file)
        
        # Override with environment variables
        self._load_env_overrides()
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Configuration loaded for environment: {self.config.environment}")
    
    def _load_config_from_file(self, config_file: str):
        """Load additional configuration from JSON file"""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                
                # Merge file config with base config
                self._merge_config(file_config)
                self.config_file_path = config_path
                logger.info(f"Configuration loaded from file: {config_file}")
            else:
                logger.warning(f"Configuration file not found: {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration file {config_file}: {e}")
    
    def _merge_config(self, file_config: Dict[str, Any]):
        """Merge file configuration with base configuration"""
        for section, values in file_config.items():
            if hasattr(self.config, section) and isinstance(values, dict):
                section_obj = getattr(self.config, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                    else:
                        logger.warning(f"Unknown config key: {section}.{key}")
            elif section == "extra_settings":
                self.config.extra_settings.update(values)
            else:
                logger.warning(f"Unknown config section: {section}")
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        # Redis overrides
        if os.getenv("REDIS_HOST"):
            self.config.redis.host = os.getenv("REDIS_HOST")
        if os.getenv("REDIS_PORT"):
            self.config.redis.port = int(os.getenv("REDIS_PORT"))
        if os.getenv("REDIS_PASSWORD"):
            self.config.redis.password = os.getenv("REDIS_PASSWORD")
        
        # Database overrides
        if os.getenv("DATABASE_URL"):
            self.config.database.url = os.getenv("DATABASE_URL")
        
        # Performance overrides
        if os.getenv("MAX_CONCURRENT_REQUESTS"):
            self.config.performance.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS"))
        if os.getenv("REQUEST_TIMEOUT"):
            self.config.performance.request_timeout = int(os.getenv("REQUEST_TIMEOUT"))
        
        # Security overrides
        if os.getenv("SMARTSCRAPE_API_KEY"):
            self.config.security.api_key = os.getenv("SMARTSCRAPE_API_KEY")
        
        # Logging overrides
        if os.getenv("LOG_LEVEL"):
            self.config.logging.level = os.getenv("LOG_LEVEL")
    
    def _validate_config(self):
        """Validate configuration settings"""
        errors = []
        
        # Validate Redis config
        if self.config.redis.port < 1 or self.config.redis.port > 65535:
            errors.append("Redis port must be between 1 and 65535")
        
        # Validate performance config
        if self.config.performance.max_concurrent_requests < 1:
            errors.append("max_concurrent_requests must be at least 1")
        
        if self.config.performance.request_timeout < 1:
            errors.append("request_timeout must be at least 1 second")
        
        # Validate rate limiting
        if self.config.rate_limit.requests_per_minute < 1:
            errors.append("requests_per_minute must be at least 1")
        
        # Validate security
        if self.config.security.api_key_required and not self.config.security.api_key:
            if self.config.environment == "production":
                errors.append("API key is required in production environment")
        
        # Validate logging
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.config.logging.level.upper() not in valid_log_levels:
            errors.append(f"Invalid log level: {self.config.logging.level}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def get_config(self) -> EnvironmentConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration at runtime"""
        try:
            # Create a backup of current config
            backup_config = self.config
            
            # Apply updates
            self._merge_config(updates)
            
            # Validate updated config
            self._validate_config()
            
            logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            # Restore backup config
            self.config = backup_config
            logger.error(f"Configuration update failed: {e}")
            return False
    
    def save_config_to_file(self, file_path: str = None) -> bool:
        """Save current configuration to file"""
        try:
            if file_path is None:
                file_path = self.config_file_path or "smartscrape_config.json"
            
            config_dict = self._config_to_dict()
            
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "redis": {
                "host": self.config.redis.host,
                "port": self.config.redis.port,
                "db": self.config.redis.db,
                "password": self.config.redis.password,
                "decode_responses": self.config.redis.decode_responses,
                "socket_connect_timeout": self.config.redis.socket_connect_timeout,
                "socket_timeout": self.config.redis.socket_timeout,
                "max_connections": self.config.redis.max_connections
            },
            "database": {
                "url": self.config.database.url,
                "pool_size": self.config.database.pool_size,
                "max_overflow": self.config.database.max_overflow,
                "pool_timeout": self.config.database.pool_timeout
            },
            "performance": {
                "max_concurrent_requests": self.config.performance.max_concurrent_requests,
                "request_timeout": self.config.performance.request_timeout,
                "max_retries": self.config.performance.max_retries,
                "max_memory_mb": self.config.performance.max_memory_mb
            },
            "rate_limit": {
                "requests_per_minute": self.config.rate_limit.requests_per_minute,
                "requests_per_hour": self.config.rate_limit.requests_per_hour,
                "burst_allowance": self.config.rate_limit.burst_allowance
            },
            "cache": {
                "enabled": self.config.cache.enabled,
                "default_ttl": self.config.cache.default_ttl,
                "max_memory_mb": self.config.cache.max_memory_mb,
                "cache_ttl": self.config.cache.cache_ttl
            },
            "logging": {
                "level": self.config.logging.level,
                "format": self.config.logging.format,
                "log_to_file": self.config.logging.log_to_file,
                "log_file": self.config.logging.log_file
            },
            "security": {
                "api_key_required": self.config.security.api_key_required,
                "allowed_origins": self.config.security.allowed_origins,
                "rate_limiting_enabled": self.config.security.rate_limiting_enabled
            },
            "extra_settings": self.config.extra_settings
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for monitoring"""
        return {
            "environment": self.config.environment,
            "debug": self.config.debug,
            "redis_host": self.config.redis.host,
            "redis_port": self.config.redis.port,
            "max_concurrent_requests": self.config.performance.max_concurrent_requests,
            "request_timeout": self.config.performance.request_timeout,
            "cache_enabled": self.config.cache.enabled,
            "api_key_required": self.config.security.api_key_required,
            "log_level": self.config.logging.level
        }

# Global configuration manager instance
config_manager = ConfigurationManager()
