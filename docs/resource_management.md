# Resource Management Architecture

## Overview

The SmartScrape Resource Management architecture provides a robust foundation for handling web scraping resources efficiently and reliably. This document outlines the key components, their responsibilities, and how they work together.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  SessionManager │     │   RateLimiter   │     │   ProxyManager  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────┬───────────┴───────────┬───────────┘
                     │                       │
            ┌────────▼────────┐     ┌───────▼────────┐
            │ ServiceRegistry │     │ StrategyContext│
            └────────┬────────┘     └───────┬────────┘
                     │                      │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼──────────┐
                     │   Scraping Engines  │
                     │  (Strategies/Stages)│
                     └─────────────────────┘
```

### Component Responsibilities

1. **SessionManager**: Central hub for HTTP and browser session management
   - Creates and pools HTTP sessions
   - Manages browser contexts and pages
   - Handles cookies and state
   - Rotates user agents
   - Tracks session health

2. **RateLimiter**: Controls request frequency to avoid blocking
   - Enforces domain-specific rate limits
   - Implements adaptive rate limiting
   - Applies backoff strategies when needed
   - Controls concurrency levels

3. **ProxyManager**: Manages proxy servers for distribution and anonymity
   - Maintains proxy pools and rotates IPs
   - Monitors proxy health and performance
   - Tags proxies for specific purposes
   - Handles proxy authentication

4. **ServiceRegistry**: Central registry for all services
   - Provides dependency injection and service location
   - Manages service lifecycles
   - Resolves dependencies between services

5. **StrategyContext**: Provides services to strategies
   - Acts as facade to underlying services
   - Maintains configuration and state
   - Provides consistent interface to strategies

### Extension Points

The resource management architecture provides several extension points:

1. **Custom Session Handlers**: Extend `SessionManager` to support different session types or protocols
2. **Custom Rate Limiting Algorithms**: Add new rate limiting strategies to `RateLimiter`
3. **Proxy Providers**: Implement custom proxy source integrations
4. **Metrics and Monitoring**: Add custom metrics collection for resource usage

## Detailed Component Documentation

### SessionManager

The `SessionManager` handles all HTTP and browser sessions, providing a unified interface for making requests while handling cookies, headers, and state management.

#### Key Features

- **HTTP Session Pooling**: Reuse sessions for better performance and cookie persistence
- **Browser Automation**: Manage Playwright/Selenium browser instances
- **User Agent Rotation**: Automatically rotate user agents to avoid detection
- **Cookie Management**: Persist and manage cookies across requests
- **Authentication Handling**: Support for various authentication mechanisms

#### Usage Example

```python
# Get a session for a specific domain
session = session_manager.get_session("example.com")

# Make a request using the managed session
response = session.get("https://example.com/page")

# Use a browser session
browser_page = session_manager.get_browser_page("example.com")
await browser_page.goto("https://example.com/page")
```

### RateLimiter

The `RateLimiter` ensures requests are properly spaced to avoid triggering rate limiting mechanisms on target websites.

#### Key Features

- **Domain-Specific Limits**: Different limits for different domains
- **Adaptive Rate Limiting**: Automatically adjust based on responses
- **Concurrency Control**: Limit parallel requests to a domain
- **Backoff Strategies**: Implement exponential backoff and jitter

#### Usage Example

```python
# Wait if needed before making a request
rate_limiter.wait_if_needed("example.com")

# Report outcomes to adjust limits
rate_limiter.report_success("example.com")
rate_limiter.report_rate_limited("example.com")
```

### ProxyManager

The `ProxyManager` handles proxy selection, rotation, and health monitoring.

#### Key Features

- **Proxy Rotation**: Cycle through available proxies
- **Health Checking**: Monitor and exclude failing proxies
- **Targeted Selection**: Select proxies based on requirements
- **Geolocation Support**: Choose proxies from specific regions

#### Usage Example

```python
# Get a proxy for a specific domain
proxy = proxy_manager.get_proxy("example.com")

# Use in requests
response = session.get("https://example.com", proxies=proxy.as_dict())

# Report proxy success/failure
proxy_manager.report_success(proxy)
proxy_manager.report_failure(proxy, error="Connection timeout")
```

## Configuration Guide

### Sample Configurations

#### SessionManager Configuration

```python
session_manager_config = {
    "user_agents": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15"
    ],
    "timeout": 30,
    "max_retries": 3,
    "verify_ssl": True,
    "browser": {
        "headless": True,
        "use_stealth": True,
        "max_pages": 5
    }
}
```

#### RateLimiter Configuration

```python
rate_limiter_config = {
    "default_limits": {
        "requests_per_minute": 60,
        "requests_per_hour": 600,
        "concurrent_requests": 5
    },
    "domain_limits": {
        "example.com": {
            "requests_per_minute": 30,
            "requests_per_hour": 300,
            "concurrent_requests": 3
        }
    }
}
```

#### ProxyManager Configuration

```python
proxy_manager_config = {
    "sources": [
        {
            "type": "static",
            "proxies": [
                {"url": "http://user:pass@proxy1.example.com:8080", "tags": ["residential"]},
                {"url": "http://user:pass@proxy2.example.com:8080", "tags": ["datacenter"]}
            ]
        },
        {
            "type": "api",
            "url": "https://proxy-provider.com/api",
            "api_key": "YOUR_API_KEY",
            "refresh_interval": 3600
        }
    ],
    "rotation_strategy": "round_robin",
    "health_check": {
        "enabled": True,
        "test_url": "https://httpbin.org/ip",
        "timeout": 5,
        "interval": 300
    }
}
```

### Optimization Recommendations

1. **Session Reuse**: Always reuse sessions when making multiple requests to the same domain
2. **Connection Pooling**: Configure proper connection pooling for optimal performance
3. **Proxy Selection**: Use dedicated proxies for sensitive operations, rotate for general scraping
4. **Rate Limit Tuning**: Start conservative and gradually increase rate limits based on success

### Scaling Considerations

1. **Distributed Rate Limiting**: Use Redis or other distributed storage for rate limiting across multiple instances
2. **Proxy Sharing**: Implement a centralized proxy management service for larger deployments
3. **Session Serialization**: Consider serializing and sharing session state for distributed workloads
4. **Monitoring**: Implement comprehensive monitoring to detect and respond to resource constraints

## Integration Examples

### Complete Request Flow Example

```python
from core.service_registry import ServiceRegistry
from strategies.core.strategy_context import StrategyContext

# Get services from registry
registry = ServiceRegistry()
session_manager = registry.get_service("session_manager")
rate_limiter = registry.get_service("rate_limiter")
proxy_manager = registry.get_service("proxy_manager")

# Target domain
domain = "example.com"

# Get a proxy
proxy = proxy_manager.get_proxy(domain)

# Respect rate limits
rate_limiter.wait_if_needed(domain)

try:
    # Get a session with the proxy
    session = session_manager.get_session(domain)
    
    # Configure session with proxy if available
    if proxy:
        session.proxies = proxy.as_dict()
    
    # Make the request
    response = session.get(f"https://{domain}/page")
    
    # Report success
    rate_limiter.report_success(domain)
    if proxy:
        proxy_manager.report_success(proxy)
    
    # Process response
    # ...

except Exception as e:
    # Report failures
    if proxy:
        proxy_manager.report_failure(proxy, error=str(e))
    
    # If rate limited, adjust limits
    if hasattr(e, 'response') and e.response.status_code == 429:
        rate_limiter.report_rate_limited(domain)
    
    # Handle the error
    # ...
```

### Using Strategy Context Integration

```python
class ExampleStrategy:
    def execute(self, context: StrategyContext, url: str):
        # The context provides access to all resource management services
        domain = extract_domain(url)
        
        # Rate limiting handled automatically through context
        context.rate_limiter.wait_if_needed(domain)
        
        # Get a session with proxy through context
        session = context.get_session_for_url(url)
        
        try:
            # Make request
            response = session.get(url)
            # Process response
            # ...
            
        except Exception as e:
            # Handle errors
            # ...
```

### Health Monitoring Setup

```python
from prometheus_client import start_http_server, Counter, Gauge

# Define metrics
session_count = Gauge('smartscrape_active_sessions', 'Number of active sessions', ['domain'])
request_count = Counter('smartscrape_requests_total', 'Total requests made', ['domain', 'status'])
proxy_health = Gauge('smartscrape_proxy_health', 'Proxy health score (0-100)', ['proxy_id'])
rate_limit_wait = Counter('smartscrape_rate_limit_wait_seconds', 'Time spent waiting due to rate limits', ['domain'])

# Start metrics server
start_http_server(8000)

# Hook into services
class MonitoredSessionManager(SessionManager):
    def get_session(self, domain, force_new=False):
        session = super().get_session(domain, force_new)
        session_count.labels(domain=domain).inc()
        return session
        
    def _track_request(self, domain, status):
        request_count.labels(domain=domain, status=status).inc()
```

## Troubleshooting Tips

### Common Issues and Solutions

1. **Excessive Rate Limiting**
   - **Symptoms**: Many 429 responses, increased blocking
   - **Solutions**: Reduce request frequency, increase delay between requests, rotate IPs more frequently

2. **Proxy Failures**
   - **Symptoms**: Connection errors, timeouts
   - **Solutions**: Implement more aggressive health checks, use higher quality proxies, implement automatic proxy rotation

3. **Session Management Issues**
   - **Symptoms**: Authentication failures, cookie-related errors
   - **Solutions**: Ensure proper session reuse, verify cookie handling, check for cookie expiration

4. **Memory Leaks**
   - **Symptoms**: Increasing memory usage over time
   - **Solutions**: Properly close unused sessions, implement session cleanup, limit maximum pool size

### Diagnostic Procedures

1. **Enable Debug Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   logging.getLogger('core.session_manager').setLevel(logging.DEBUG)
   logging.getLogger('core.rate_limiter').setLevel(logging.DEBUG)
   logging.getLogger('core.proxy_manager').setLevel(logging.DEBUG)
   ```

2. **Monitor Resource Usage**
   - Track active sessions, request rates, and proxy utilization
   - Use the integrated Prometheus metrics for visualization

3. **Test Individual Components**
   - Isolate session management, rate limiting, and proxy management for testing
   - Use the provided test utilities in the testing package