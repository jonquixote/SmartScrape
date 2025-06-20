# SmartScrape Configuration Guide

## Overview

SmartScrape offers comprehensive configuration options to customize scraping behavior, resource management, AI services, and advanced features. This guide covers all configuration aspects including the new features from Phases 1-7.

## Core Configuration

### Basic Configuration (`config.py`)

```python
# Basic application settings
DEBUG = True
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Database configuration
DATABASE_URL = "sqlite:///smartscrape.db"
REDIS_URL = "redis://localhost:6379/0"

# AI Service configuration
GOOGLE_API_KEY = "your-google-api-key"
OPENAI_API_KEY = "your-openai-api-key"
DEFAULT_AI_MODEL = "gemini-pro"
```

### Resource Management Configuration

```python
# Session Management
SESSION_CONFIG = {
    "pool_size": 10,
    "max_retries": 3,
    "timeout": 30,
    "headers": {
        "User-Agent": "SmartScrape/1.0"
    }
}

# Rate Limiting
RATE_LIMIT_CONFIG = {
    "requests_per_second": 5,
    "requests_per_minute": 100,
    "burst_size": 10,
    "adaptive": True
}

# Proxy Management
PROXY_CONFIG = {
    "enabled": True,
    "rotation_strategy": "round_robin",
    "health_check_interval": 300,
    "max_failures": 3,
    "proxy_sources": [
        "proxy_provider_1",
        "proxy_provider_2"
    ]
}
```

## Advanced Features Configuration

### Semantic Intent Analysis

```python
# UniversalIntentAnalyzer configuration
INTENT_ANALYSIS_CONFIG = {
    "spacy_model": "en_core_web_sm",
    "enable_query_expansion": True,
    "enable_entity_recognition": True,
    "similarity_threshold": 0.7,
    "max_expanded_terms": 10,
    "intent_categories": [
        "e_commerce",
        "news",
        "research",
        "social_media",
        "business",
        "entertainment"
    ]
}

# Semantic search configuration
SEMANTIC_SEARCH_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "max_results": 50,
    "min_relevance_score": 0.6,
    "enable_cross_domain_search": True
}
```

### AI Schema Generation

```python
# AISchemaGenerator configuration
SCHEMA_GENERATION_CONFIG = {
    "enable_content_aware_generation": True,
    "use_pydantic_validation": True,
    "max_schema_depth": 5,
    "enable_hierarchical_schemas": True,
    "schema_evolution_enabled": True,
    "merge_strategy": "intelligent",
    "validation_strictness": "medium"
}

# Domain-specific schema templates
DOMAIN_SCHEMAS = {
    "e_commerce": {
        "product_fields": ["name", "price", "description", "rating"],
        "required_fields": ["name", "price"],
        "validation_rules": {
            "price": {"type": "decimal", "min": 0}
        }
    },
    "news": {
        "article_fields": ["title", "content", "author", "publish_date"],
        "required_fields": ["title", "content"],
        "validation_rules": {
            "publish_date": {"type": "datetime"}
        }
    }
}
```

### Resilience and Proxy Management

```python
# Advanced resilience configuration
RESILIENCE_CONFIG = {
    "circuit_breaker": {
        "failure_threshold": 5,
        "recovery_timeout": 60,
        "half_open_max_calls": 3
    },
    "anti_detection": {
        "randomize_user_agents": True,
        "randomize_headers": True,
        "simulate_human_behavior": True,
        "request_delays": {
            "min": 1,
            "max": 5,
            "distribution": "normal"
        }
    },
    "captcha_handling": {
        "detection_enabled": True,
        "solving_service": "2captcha",
        "max_solving_attempts": 3,
        "timeout": 120
    }
}

# Session rotation configuration
SESSION_ROTATION_CONFIG = {
    "rotation_interval": 300,  # seconds
    "max_requests_per_session": 100,
    "enable_cookie_persistence": True,
    "fingerprint_randomization": True
}
```

### Caching Strategy

```python
# Multi-tier caching configuration
CACHING_CONFIG = {
    "memory_cache": {
        "enabled": True,
        "max_size": "100MB",
        "ttl": 3600,  # seconds
        "eviction_policy": "LRU"
    },
    "redis_cache": {
        "enabled": True,
        "url": "redis://localhost:6379/1",
        "ttl": 86400,  # 24 hours
        "compression": True
    },
    "persistent_cache": {
        "enabled": True,
        "storage_path": "./cache",
        "max_size": "1GB",
        "cleanup_interval": 86400
    }
}

# Content-aware caching strategies
CONTENT_CACHING_STRATEGIES = {
    "html_content": {
        "ttl": 3600,
        "compression": True,
        "invalidation_triggers": ["url_change", "content_hash_change"]
    },
    "ai_responses": {
        "ttl": 86400,
        "cache_key_includes": ["prompt_hash", "model_version"],
        "compression": True
    },
    "extracted_data": {
        "ttl": 7200,
        "validation_on_retrieval": True,
        "schema_version_tracking": True
    }
}
```

### User Feedback Integration

```python
# Feedback collection configuration
FEEDBACK_CONFIG = {
    "collection_modes": {
        "explicit": {
            "enabled": True,
            "rating_scale": "1-5",
            "comment_length_limit": 500
        },
        "implicit": {
            "enabled": True,
            "track_user_interactions": True,
            "session_analytics": True
        },
        "comparative": {
            "enabled": True,
            "a_b_testing": True,
            "result_comparison_ui": True
        }
    },
    "analytics": {
        "sentiment_analysis": True,
        "trend_analysis": True,
        "quality_correlation": True
    },
    "personalization": {
        "enabled": True,
        "user_profiling": True,
        "adaptive_parameters": True,
        "preference_learning": True
    }
}

# Quality assessment configuration
QUALITY_ASSESSMENT_CONFIG = {
    "metrics": ["completeness", "accuracy", "relevance", "freshness"],
    "thresholds": {
        "completeness": 0.8,
        "accuracy": 0.9,
        "relevance": 0.7,
        "freshness": 86400  # seconds
    },
    "improvement_triggers": {
        "low_quality_threshold": 0.6,
        "user_complaint_threshold": 3
    }
}
```

### Progressive Collection and Consolidated AI Processing

```python
# Progressive collection configuration
PROGRESSIVE_COLLECTION_CONFIG = {
    "collection_phase": {
        "lightweight_extraction": True,
        "batch_size": 50,
        "parallel_workers": 5,
        "metadata_only": False,
        "content_sampling": {
            "enabled": True,
            "sample_percentage": 20
        }
    },
    "consolidation_phase": {
        "ai_processing_batch_size": 10,
        "enable_cross_page_analysis": True,
        "deduplication_strategy": "semantic",
        "quality_enhancement": True,
        "token_optimization": True
    }
}

# Deduplication configuration
DEDUPLICATION_CONFIG = {
    "strategies": ["exact_match", "fuzzy_match", "semantic_similarity"],
    "similarity_thresholds": {
        "fuzzy_match": 0.9,
        "semantic_similarity": 0.85
    },
    "cross_page_analysis": {
        "enabled": True,
        "max_pages_to_compare": 100,
        "clustering_algorithm": "hierarchical"
    }
}
```

## Pipeline Configuration

### Custom Pipeline Definitions

```python
# Pipeline configuration for different use cases
PIPELINE_CONFIGS = {
    "basic_extraction": {
        "stages": [
            "url_validation",
            "content_fetch",
            "html_parse",
            "data_extract",
            "quality_check"
        ],
        "parallel_execution": True,
        "error_handling": "continue_on_error"
    },
    "ai_enhanced_extraction": {
        "stages": [
            "intent_analysis",
            "schema_generation",
            "progressive_collection",
            "ai_consolidation",
            "quality_assessment",
            "feedback_integration"
        ],
        "parallel_execution": False,
        "error_handling": "stop_on_critical_error"
    }
}
```

## Monitoring and Observability

```python
# Monitoring configuration
MONITORING_CONFIG = {
    "metrics": {
        "enabled": True,
        "collection_interval": 60,
        "retention_period": 2592000,  # 30 days
        "export_format": "prometheus"
    },
    "logging": {
        "level": "INFO",
        "structured_logging": True,
        "log_rotation": {
            "max_size": "100MB",
            "backup_count": 5
        }
    },
    "alerting": {
        "enabled": True,
        "channels": ["email", "slack"],
        "thresholds": {
            "error_rate": 0.05,
            "response_time": 30,
            "success_rate": 0.95
        }
    }
}

# Performance metrics configuration
PERFORMANCE_METRICS = {
    "track_extraction_time": True,
    "track_ai_processing_time": True,
    "track_cache_hit_ratio": True,
    "track_success_rates": True,
    "track_resource_utilization": True
}
```

## Security Configuration

```python
# Security settings
SECURITY_CONFIG = {
    "api_keys": {
        "encryption": True,
        "rotation_interval": 2592000,  # 30 days
        "secure_storage": True
    },
    "data_privacy": {
        "anonymize_user_data": True,
        "data_retention_period": 7776000,  # 90 days
        "gdpr_compliance": True
    },
    "request_security": {
        "verify_ssl": True,
        "timeout_settings": {
            "connect": 10,
            "read": 30,
            "total": 60
        }
    }
}
```

## Environment-Specific Configuration

### Development Configuration

```python
# development.py
from config import *

DEBUG = True
LOG_LEVEL = "DEBUG"
RATE_LIMIT_CONFIG["requests_per_second"] = 10
CACHING_CONFIG["memory_cache"]["ttl"] = 300
```

### Production Configuration

```python
# production.py
from config import *

DEBUG = False
LOG_LEVEL = "WARNING"
RATE_LIMIT_CONFIG["requests_per_second"] = 2
CACHING_CONFIG["redis_cache"]["enabled"] = True
MONITORING_CONFIG["alerting"]["enabled"] = True
```

### Testing Configuration

```python
# testing.py
from config import *

DATABASE_URL = "sqlite:///:memory:"
CACHING_CONFIG["memory_cache"]["enabled"] = False
FEEDBACK_CONFIG["collection_modes"]["implicit"]["enabled"] = False
```

## Configuration Validation

SmartScrape includes built-in configuration validation to ensure all settings are valid:

```python
# Configuration validation rules
VALIDATION_RULES = {
    "required_fields": [
        "GOOGLE_API_KEY",
        "DATABASE_URL"
    ],
    "type_validation": {
        "RATE_LIMIT_CONFIG.requests_per_second": "int",
        "CACHING_CONFIG.memory_cache.ttl": "int",
        "RESILIENCE_CONFIG.circuit_breaker.failure_threshold": "int"
    },
    "range_validation": {
        "RATE_LIMIT_CONFIG.requests_per_second": {"min": 1, "max": 100},
        "SEMANTIC_SEARCH_CONFIG.similarity_threshold": {"min": 0.0, "max": 1.0}
    }
}
```

## Configuration Best Practices

### Performance Optimization

1. **Caching Strategy**: Use appropriate TTL values based on content volatility
2. **Rate Limiting**: Balance speed with respect for target servers
3. **Parallel Processing**: Configure worker counts based on available resources
4. **Memory Management**: Set appropriate cache size limits

### Reliability

1. **Circuit Breakers**: Configure appropriate failure thresholds
2. **Retry Logic**: Use exponential backoff for transient failures
3. **Health Checks**: Enable monitoring for all critical components
4. **Fallback Strategies**: Configure multiple extraction strategies

### Security

1. **API Key Management**: Use environment variables for sensitive data
2. **Request Validation**: Enable SSL verification and timeouts
3. **Data Privacy**: Configure appropriate retention periods
4. **Access Control**: Implement proper authentication and authorization

### Scalability

1. **Resource Pools**: Configure appropriate connection pool sizes
2. **Load Balancing**: Use multiple proxy sources and AI models
3. **Horizontal Scaling**: Configure Redis for shared caching
4. **Monitoring**: Track resource utilization and performance metrics

## Troubleshooting Common Configuration Issues

### Cache Configuration Issues

```python
# Common cache misconfigurations and fixes
CACHE_TROUBLESHOOTING = {
    "redis_connection_errors": {
        "symptoms": ["Connection refused", "Timeout errors"],
        "solutions": [
            "Check Redis server status",
            "Verify REDIS_URL configuration",
            "Check network connectivity"
        ]
    },
    "memory_cache_overflow": {
        "symptoms": ["Out of memory errors", "Performance degradation"],
        "solutions": [
            "Reduce max_size setting",
            "Implement more aggressive TTL",
            "Enable compression"
        ]
    }
}
```

### AI Service Configuration Issues

```python
# AI service troubleshooting
AI_TROUBLESHOOTING = {
    "api_key_errors": {
        "symptoms": ["Authentication failed", "Invalid API key"],
        "solutions": [
            "Verify API key is correct",
            "Check API key permissions",
            "Ensure API key is not expired"
        ]
    },
    "model_selection_errors": {
        "symptoms": ["Model not found", "Unsupported model"],
        "solutions": [
            "Check model availability",
            "Verify model name spelling",
            "Use fallback model configuration"
        ]
    }
}
```

This configuration guide provides comprehensive coverage of all SmartScrape configuration options, including the advanced features implemented in Phases 1-7. Users can customize the application behavior to meet their specific requirements while following best practices for performance, reliability, and security.
