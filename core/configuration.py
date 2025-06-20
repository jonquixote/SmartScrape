"""
Configuration management for resource services and error handling components.
Provides standardized configuration schema, validation, and defaults.
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List, Set
import jsonschema

# Import main app config
from config import get_config

logger = logging.getLogger(__name__)

# Default User-Agent strings
DEFAULT_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
]

# JSON Schema for session manager configuration
SESSION_MANAGER_SCHEMA = {
    "type": "object",
    "properties": {
        "user_agents": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of user agent strings to rotate"
        },
        "timeouts": {
            "type": "object",
            "properties": {
                "connect": {"type": "number", "minimum": 0},
                "read": {"type": "number", "minimum": 0},
                "total": {"type": "number", "minimum": 0}
            }
        },
        "max_sessions": {"type": "integer", "minimum": 1},
        "session_ttl": {"type": "integer", "minimum": 0},
        "browser_settings": {
            "type": "object",
            "properties": {
                "headless": {"type": "boolean"},
                "disable_images": {"type": "boolean"},
                "disable_css": {"type": "boolean"},
                "disable_javascript": {"type": "boolean"},
                "max_instances": {"type": "integer", "minimum": 1}
            }
        }
    }
}

# JSON Schema for rate limiter configuration
RATE_LIMITER_SCHEMA = {
    "type": "object",
    "properties": {
        "default_limits": {
            "type": "object",
            "properties": {
                "requests_per_minute": {"type": "integer", "minimum": 1},
                "requests_per_hour": {"type": "integer", "minimum": 1},
                "concurrent_requests": {"type": "integer", "minimum": 1}
            }
        },
        "domain_limits": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "requests_per_minute": {"type": "integer", "minimum": 1},
                    "requests_per_hour": {"type": "integer", "minimum": 1},
                    "concurrent_requests": {"type": "integer", "minimum": 1}
                }
            }
        },
        "backoff_factor": {"type": "number", "minimum": 1.0},
        "jitter_factor": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    }
}

# JSON Schema for proxy manager configuration
PROXY_MANAGER_SCHEMA = {
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean"},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["file", "api", "static"]},
                    "url": {"type": "string"},
                    "file_path": {"type": "string"},
                    "api_key": {"type": "string"},
                    "proxies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "protocol": {"type": "string", "enum": ["http", "https", "socks4", "socks5"]},
                                "username": {"type": "string"},
                                "password": {"type": "string"},
                                "tags": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["url"]
                        }
                    }
                }
            }
        },
        "rotation_strategy": {"type": "string", "enum": ["round_robin", "random", "performance"]},
        "health_check_interval": {"type": "integer", "minimum": 0},
        "max_failures": {"type": "integer", "minimum": 1},
        "retry_delay": {"type": "integer", "minimum": 0}
    }
}

# JSON Schema for error classifier configuration
ERROR_CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "captcha_patterns": {
            "type": "array",
            "items": {"type": "string"}
        },
        "access_denied_patterns": {
            "type": "array",
            "items": {"type": "string"}
        },
        "retry_status_codes": {
            "type": "array",
            "items": {"type": "integer"}
        },
        "fatal_status_codes": {
            "type": "array",
            "items": {"type": "integer"}
        },
        "error_patterns": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "patterns": {"type": "array", "items": {"type": "string"}},
                    "severity": {"type": "string", "enum": ["transient", "persistent", "fatal"]},
                    "category": {"type": "string"}
                }
            }
        }
    }
}

# JSON Schema for retry manager configuration
RETRY_MANAGER_SCHEMA = {
    "type": "object",
    "properties": {
        "max_attempts": {"type": "integer", "minimum": 1},
        "backoff_base": {"type": "number", "minimum": 1.0},
        "jitter": {"type": "boolean"},
        "jitter_factor": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "max_backoff": {"type": "number", "minimum": 0},
        "retry_for": {
            "type": "array",
            "items": {"type": "string"}
        },
        "retry_if_status": {
            "type": "array",
            "items": {"type": "integer"}
        }
    }
}

# JSON Schema for circuit breaker configuration
CIRCUIT_BREAKER_SCHEMA = {
    "type": "object",
    "properties": {
        "default_settings": {
            "type": "object",
            "properties": {
                "failure_threshold": {"type": "integer", "minimum": 1},
                "reset_timeout": {"type": "integer", "minimum": 1},
                "half_open_max": {"type": "integer", "minimum": 1}
            }
        },
        "circuit_breakers": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "failure_threshold": {"type": "integer", "minimum": 1},
                    "reset_timeout": {"type": "integer", "minimum": 1},
                    "half_open_max": {"type": "integer", "minimum": 1}
                }
            }
        }
    }
}

# Combine all schemas into a unified resource services schema
RESOURCE_SERVICES_SCHEMA = {
    "type": "object",
    "properties": {
        "session_manager": SESSION_MANAGER_SCHEMA,
        "rate_limiter": RATE_LIMITER_SCHEMA,
        "proxy_manager": PROXY_MANAGER_SCHEMA,
        "error_classifier": ERROR_CLASSIFIER_SCHEMA,
        "retry_manager": RETRY_MANAGER_SCHEMA,
        "circuit_breaker": CIRCUIT_BREAKER_SCHEMA
    }
}

# Default configurations for resource management services
DEFAULT_RESOURCE_CONFIG = {
    "session_manager": {
        "user_agents": DEFAULT_USER_AGENTS,
        "timeouts": {
            "connect": 10,
            "read": 30,
            "total": 60
        },
        "max_sessions": 50,
        "session_ttl": 3600,
        "browser_settings": {
            "headless": True,
            "disable_images": True,
            "disable_css": True,
            "disable_javascript": False,
            "max_instances": 3
        }
    },
    "rate_limiter": {
        "default_limits": {
            "requests_per_minute": 60,
            "requests_per_hour": 600,
            "concurrent_requests": 5
        },
        "domain_limits": {
            # Example domain-specific limits
            "google.com": {
                "requests_per_minute": 5,
                "requests_per_hour": 100,
                "concurrent_requests": 2
            }
        },
        "backoff_factor": 2.0,
        "jitter_factor": 0.1
    },
    "proxy_manager": {
        "enabled": False,  # Disabled by default
        "sources": [],
        "rotation_strategy": "round_robin",
        "health_check_interval": 300,  # 5 minutes
        "max_failures": 3,
        "retry_delay": 600  # 10 minutes
    },
    "error_classifier": {
        "captcha_patterns": [
            r'captcha',
            r'robot check',
            r'human verification',
            r'are you a robot',
            r'prove you\'re human'
        ],
        "access_denied_patterns": [
            r'access denied',
            r'permission denied',
            r'403 forbidden',
            r'not authorized',
            r'blocked'
        ],
        "retry_status_codes": [429, 503, 502, 500],
        "fatal_status_codes": [401, 403, 404],
        "error_patterns": {
            "authentication": {
                "patterns": [r"login required", r"please sign in", r"authentication required"],
                "severity": "persistent",
                "category": "authentication"
            }
        }
    },
    "retry_manager": {
        "max_attempts": 3,
        "backoff_base": 2.0,
        "jitter": True,
        "jitter_factor": 0.1,
        "max_backoff": 60,  # 1 minute
        "retry_for": ["network", "rate_limit", "http"],
        "retry_if_status": [429, 500, 502, 503, 504]
    },
    "circuit_breaker": {
        "default_settings": {
            "failure_threshold": 5,
            "reset_timeout": 60,  # 1 minute
            "half_open_max": 1
        },
        "circuit_breakers": {
            # Domain-specific circuit breaker settings example
            "api.example.com": {
                "failure_threshold": 3,
                "reset_timeout": 300,  # 5 minutes
                "half_open_max": 2
            }
        }
    }
}

def validate_resource_configuration(config: Dict[str, Any]) -> List[str]:
    """
    Validate the resource services configuration against the schema.
    
    Args:
        config: The configuration dictionary to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    validator = jsonschema.Draft7Validator(RESOURCE_SERVICES_SCHEMA)
    for error in validator.iter_errors(config):
        path = '.'.join([str(p) for p in error.path])
        errors.append(f"{path}: {error.message}")
    
    return errors

def sanitize_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure configuration has all required fields with valid values.
    
    Args:
        config: The configuration dictionary to sanitize
        
    Returns:
        Sanitized configuration dictionary with defaults applied
    """
    # Start with default configuration
    sanitized = DEFAULT_RESOURCE_CONFIG.copy()
    
    # Update with provided configuration (deep merge)
    _deep_update(sanitized, config)
    
    return sanitized

def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """
    Deep update target dict with source (nested dictionaries are updated rather than replaced).
    
    Args:
        target: The dictionary to update
        source: The dictionary with updates
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        elif key in target and isinstance(target[key], list) and isinstance(value, list):
            # For lists, we typically replace rather than extend, but this can be customized
            target[key] = value
        else:
            target[key] = value

def get_resource_config() -> Dict[str, Any]:
    """
    Get the resource management and error handling configuration with defaults applied.
    
    Returns:
        Complete resource services configuration dictionary
    """
    # Get the main application configuration
    app_config = get_config()
    
    # Extract resource service configuration or use empty dict if not present
    resource_config = app_config.get('resource_services', {})
    
    # Apply defaults and validate
    sanitized_config = sanitize_configuration(resource_config)
    errors = validate_resource_configuration(sanitized_config)
    
    if errors:
        logger.warning(f"Resource configuration validation errors: {errors}")
    
    return sanitized_config

def update_app_config_with_resources() -> Dict[str, Any]:
    """
    Update the main application config with resource services configuration.
    
    Returns:
        Updated application configuration with resource services
    """
    app_config = get_config()
    
    # Add or update resource services configuration
    app_config['resource_services'] = get_resource_config()
    
    return app_config