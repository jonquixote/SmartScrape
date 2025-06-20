# Error Handling Architecture

## Overview

SmartScrape's error handling architecture provides a comprehensive framework for managing, classifying, and recovering from errors in web scraping operations. This document outlines the key components, their responsibilities, and how they work together to create resilient scraping workflows.

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  ErrorClassifier  │     │   RetryManager    │     │   CircuitBreaker  │
└─────────┬─────────┘     └─────────┬─────────┘     └─────────┬─────────┘
          │                         │                         │
          └─────────────┬───────────┴─────────────┬───────────┘
                        │                         │
               ┌────────▼────────┐      ┌─────────▼─────────┐
               │ ServiceRegistry │      │  StrategyContext  │
               └────────┬────────┘      └─────────┬─────────┘
                        │                         │
                        └─────────────┬───────────┘
                                      │
                  ┌──────────────────────────────────────┐
                  │                                      │
        ┌─────────▼─────────┐              ┌─────────────▼───────┐
        │  Fallback System  │              │  Scraping Engines   │
        │ (Strategy/Extract)│              │ (Strategies/Stages) │
        └───────────────────┘              └─────────────────────┘
```

### Component Responsibilities

1. **ErrorClassifier**: Categorizes errors for appropriate handling
   - Identifies error types (network, HTTP, content, parsing)
   - Determines error severity
   - Suggests remediation actions
   - Tracks error patterns

2. **RetryManager**: Implements configurable retry policies
   - Determines when to retry operations
   - Implements backoff strategies
   - Limits retry attempts
   - Tracks retry statistics

3. **CircuitBreaker**: Prevents repeated failures and system overload
   - Detects failure patterns
   - Manages circuit states (open, closed, half-open)
   - Blocks requests to failing services
   - Allows graceful recovery

4. **Fallback Frameworks**: Provides alternative execution paths
   - Strategy fallbacks for navigation/discovery
   - Extraction fallbacks for data parsing
   - Graceful degradation paths
   - Recovery mechanisms

### Extension Points

The error handling architecture provides several extension points:

1. **Custom Error Categories**: Extend `ErrorClassifier` with domain-specific error types
2. **Custom Retry Strategies**: Add specialized retry policies to `RetryManager` 
3. **Circuit Breaker Policies**: Customize circuit breaker behavior for different services
4. **Fallback Implementations**: Add domain-specific fallback mechanisms

## Detailed Component Documentation

### ErrorClassifier

The `ErrorClassifier` analyzes exceptions and identifies their type, severity, and potential remediation strategies.

#### Key Features

- **Error Categorization**: Classify by type (network, HTTP, content, parsing)
- **Severity Assessment**: Determine if transient, persistent, or fatal
- **Remediation Suggestions**: Recommend actions (retry, rotate proxy, etc.)
- **Pattern Recognition**: Detect common error signatures

#### Usage Example

```python
# Classify an exception
try:
    response = session.get(url)
    # Process response
except Exception as e:
    classification = error_classifier.classify_exception(e, {
        'url': url,
        'domain': domain
    })
    
    # Take action based on classification
    if classification['is_retryable']:
        # Retry the operation
        pass
    elif classification['category'] == ErrorCategory.CAPTCHA:
        # Handle CAPTCHA challenge
        pass
```

### RetryManager

The `RetryManager` implements configurable retry policies with exponential backoff, jitter, and conditional retry logic.

#### Key Features

- **Configurable Policies**: Customize retry counts, delays, and conditions
- **Exponential Backoff**: Increase delays between retry attempts
- **Jitter**: Add randomness to prevent thundering herd problems
- **Decorator Support**: Apply retry logic with `@retry` decorator

#### Usage Example

```python
# Basic retry decorator usage
from core.retry_manager import retry

@retry(max_attempts=3, backoff_factor=2)
def fetch_url(url):
    return requests.get(url)

# Conditional retry with error classification
@retry(retry_on=lambda e: error_classifier.classify_exception(e)['is_retryable'])
def fetch_with_conditional_retry(url):
    return requests.get(url)
```

### CircuitBreaker

The `CircuitBreaker` prevents cascade failures by temporarily blocking calls to failing services.

#### Key Features

- **State Management**: Track closed, open, and half-open states
- **Failure Detection**: Monitor failure rates and patterns
- **Automatic Recovery**: Test services after timeout period
- **Configurable Thresholds**: Customize failure limits and timeouts

#### Usage Example

```python
# Basic circuit breaker usage
from core.circuit_breaker import circuit_breaker

@circuit_breaker('example_com')
def call_service():
    return requests.get('https://example.com/api')

# Manual circuit breaker usage
def call_with_circuit_breaking():
    circuit = circuit_breaker_manager.get_circuit_breaker('example_com')
    
    if not circuit.allow_request():
        # Circuit is open, use fallback
        return get_cached_data()
    
    try:
        result = requests.get('https://example.com/api')
        circuit.record_success()
        return result
    except Exception as e:
        circuit.record_failure()
        raise
```

### Fallback Frameworks

The fallback frameworks provide mechanisms for graceful degradation when primary strategies fail.

#### Strategy Fallbacks

Strategy fallbacks define alternative navigation and discovery paths when primary approaches fail.

```python
# Define fallbacks in strategy implementations
class MultiStageStrategy:
    def execute(self, context, url):
        try:
            # Try AI-guided approach first
            return self.ai_guided_strategy.execute(context, url)
        except Exception as e:
            classification = context.error_classifier.classify_exception(e)
            
            # If circuit is open or persistent error, try DOM-based approach
            if (classification['severity'] == ErrorSeverity.PERSISTENT or
                not context.circuit_breaker_manager.get_circuit_breaker('ai_service').allow_request()):
                return self.dom_based_strategy.execute(context, url)
```

#### Extraction Fallbacks

Extraction fallbacks provide alternative data extraction methods when primary extractors fail.

```python
# Define extraction fallbacks
class ResilientExtractor:
    def extract(self, html, context):
        # Try in order of preference
        extractors = [
            self.schema_extractor,
            self.ai_extractor,
            self.css_extractor,
            self.regex_extractor
        ]
        
        last_error = None
        for extractor in extractors:
            try:
                return extractor.extract(html, context)
            except Exception as e:
                last_error = e
                # Log the error and continue to next extractor
                logging.warning(f"Extractor {extractor.__class__.__name__} failed: {str(e)}")
        
        # All extractors failed
        raise ExtractionError("All extraction methods failed") from last_error
```

## Implementing Robust Error Handling

### Strategy for Categorizing Errors

1. **Identify Error Types**:
   - Network errors: Connection issues, timeouts, DNS failures
   - HTTP errors: Status codes (4xx, 5xx), redirects
   - Content errors: Missing content, unexpected format, CAPTCHA
   - Parsing errors: Selector mismatches, extraction failures
   - Resource errors: Rate limiting, IP blocks, authentication failures

2. **Determine Severity**:
   - **Transient**: Temporary issues that might resolve with retry (e.g., network glitch)
   - **Persistent**: Issues that require changing approach (e.g., IP blocked)
   - **Fatal**: Issues that cannot be resolved automatically (e.g., invalid URL)

3. **Error Classification Process**:
   - Analyze exception type and message
   - Check HTTP status codes and headers
   - Examine response content for error indicators
   - Consider context (domain, previous errors)
   - Tag with metadata for tracking and analysis

### Guidelines for Retry Policies

1. **When to Retry**:
   - Network connectivity issues
   - Temporary server errors (5xx)
   - Rate limiting (with appropriate backoff)
   - Partial or incomplete responses

2. **When Not to Retry**:
   - Authentication failures (unless credentials can be refreshed)
   - Permission errors (403, unless using different proxy)
   - Not found errors (404)
   - Malformed requests (400)

3. **Implementing Retry Logic**:
   - Start with conservative retry counts (3-5 attempts)
   - Use exponential backoff (e.g., 1s, 2s, 4s, 8s)
   - Add jitter to prevent synchronized retries
   - Set maximum backoff time (e.g., 60s)
   - Track and log retry attempts

### When to Use Circuit Breakers

1. **Ideal Use Cases**:
   - External API dependencies
   - Domains with inconsistent availability
   - Services with potential for cascade failures
   - Rate-limited resources

2. **Configuration Guidelines**:
   - Set failure threshold based on service reliability (3-10 failures)
   - Configure reset timeout based on expected recovery time
   - Use half-open state to test recovery without full load
   - Implement per-domain or per-service circuits

3. **Circuit Breaker States**:
   - **Closed**: Normal operation, requests allowed
   - **Open**: Failure threshold exceeded, requests blocked
   - **Half-Open**: Testing if service has recovered

### Implementing Graceful Degradation

1. **Degradation Strategies**:
   - Reduce complexity (simpler parsing, fewer features)
   - Fallback to different data sources
   - Return partial results when complete data unavailable
   - Use cached data when live data inaccessible

2. **User Experience Considerations**:
   - Communicate limitations clearly
   - Indicate when using fallback or cached data
   - Provide options to retry or adjust parameters
   - Estimate quality/completeness of degraded results

3. **Implementation Approach**:
   - Define quality tiers for results
   - Implement progressive enhancement pattern
   - Create feature toggles for functionality
   - Design components with fallback awareness

### Creating Effective Fallbacks

1. **Fallback Chain Design**:
   - Define clear priorities and sequences
   - Consider resource usage and performance
   - Test fallback transitions extensively
   - Implement timeouts for fallback methods

2. **Data Quality Management**:
   - Define minimum viable data requirements
   - Validate results from fallback methods
   - Tag results with source/method information
   - Implement confidence scores for results

3. **Recovery Strategies**:
   - Periodically test primary methods
   - Implement progressive recovery
   - Use background refresh for cached data
   - Track success rates to adjust strategies

## Examples for Common Scenarios

### Handling Network Errors

```python
@retry(
    max_attempts=5,
    retry_on=[requests.exceptions.ConnectionError, requests.exceptions.Timeout],
    backoff_factor=2,
    max_backoff=60
)
def fetch_with_network_resilience(url, session=None):
    """Fetch a URL with automatic retry for network errors."""
    session = session or requests.Session()
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        # Log detailed error information
        logging.error(f"Request failed for {url}: {str(e)}")
        raise
```

### Dealing with Rate Limiting

```python
def fetch_with_rate_limit_handling(url, context):
    """Fetch a URL with automatic rate limit handling."""
    domain = extract_domain(url)
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Wait according to rate limits
            context.rate_limiter.wait_if_needed(domain)
            
            # Get session and make request
            session = context.session_manager.get_session(domain)
            response = session.get(url)
            response.raise_for_status()
            
            # Report success and return
            context.rate_limiter.report_success(domain)
            return response
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limited - adjust limits and retry
                context.rate_limiter.report_rate_limited(domain)
                
                # Calculate backoff (exponential with jitter)
                backoff = min(60, (2 ** retry_count) + random.uniform(0, 1))
                logging.warning(f"Rate limited for {domain}, backing off for {backoff:.2f}s")
                time.sleep(backoff)
                
                retry_count += 1
            else:
                # Other HTTP error
                raise
        
        except Exception:
            # Other exceptions
            raise
    
    # Max retries exceeded
    raise RateLimitExceededError(f"Rate limit retries exceeded for {url}")
```

### Managing CAPTCHAs and Blocks

```python
def fetch_with_captcha_handling(url, context):
    """Fetch a URL with CAPTCHA detection and handling."""
    domain = extract_domain(url)
    captcha_circuit = context.circuit_breaker_manager.get_circuit_breaker(f"{domain}_captcha")
    
    # Check if we're likely to hit a CAPTCHA based on history
    if not captcha_circuit.allow_request():
        logging.warning(f"CAPTCHA circuit open for {domain}, using alternative approach")
        return fetch_with_browser(url, context)
    
    try:
        # Try normal request first
        session = context.session_manager.get_session(domain)
        response = session.get(url)
        
        # Check for CAPTCHA in response
        if context.error_classifier.check_for_captcha(response):
            # Record CAPTCHA encounter
            captcha_circuit.record_failure()
            
            # Try browser approach with stealth
            logging.info(f"CAPTCHA detected for {domain}, switching to browser approach")
            return fetch_with_browser(url, context)
        
        # No CAPTCHA found, record success
        captcha_circuit.record_success()
        return response
        
    except Exception as e:
        # Check if exception indicates CAPTCHA/blocking
        classification = context.error_classifier.classify_exception(e)
        if classification['category'] == ErrorCategory.CAPTCHA:
            captcha_circuit.record_failure()
            logging.warning(f"CAPTCHA/block detected via exception for {domain}")
            return fetch_with_browser(url, context)
        raise
```

### Recovering from Service Failures

```python
def fetch_with_service_resilience(url, context):
    """Fetch content with comprehensive resilience strategies."""
    domain = extract_domain(url)
    circuit = context.circuit_breaker_manager.get_circuit_breaker(domain)
    
    # Check circuit state
    if not circuit.allow_request():
        # Circuit open, use cached data if available
        cached_data = context.cache_manager.get(url)
        if cached_data:
            logging.info(f"Circuit open for {domain}, using cached data")
            return {
                'data': cached_data,
                'source': 'cache',
                'timestamp': context.cache_manager.get_timestamp(url)
            }
        else:
            # No cache, try alternative source
            alt_data = fetch_from_alternative_source(url, context)
            if alt_data:
                return {
                    'data': alt_data,
                    'source': 'alternative',
                    'timestamp': time.time()
                }
            else:
                raise ServiceUnavailableError(f"Service {domain} unavailable and no fallbacks succeeded")
    
    # Circuit closed or half-open, attempt request
    try:
        # Apply rate limiting
        context.rate_limiter.wait_if_needed(domain)
        
        # Get session and make request
        session = context.session_manager.get_session(domain)
        response = session.get(url)
        response.raise_for_status()
        
        # Process response
        data = process_response(response)
        
        # Cache successful response
        context.cache_manager.set(url, data)
        
        # Record success for circuit breaker and rate limiter
        circuit.record_success()
        context.rate_limiter.report_success(domain)
        
        return {
            'data': data,
            'source': 'live',
            'timestamp': time.time()
        }
        
    except Exception as e:
        # Record failure for circuit breaker
        circuit.record_failure()
        
        # Classify error
        classification = context.error_classifier.classify_exception(e, {
            'url': url,
            'domain': domain
        })
        
        # Log detailed error
        logging.error(
            f"Service failure for {domain}: {str(e)}\n"
            f"Category: {classification['category']}, "
            f"Severity: {classification['severity']}"
        )
        
        # Try fallbacks
        return fetch_with_fallbacks(url, context, classification)
```

## Troubleshooting Guide

### Common Error Patterns and Solutions

1. **Connection Timeouts**
   - **Possible Causes**: Network issues, server overload, proxy problems
   - **Solutions**: Increase timeout value, retry with backoff, check proxy health

2. **HTTP 403 Forbidden**
   - **Possible Causes**: IP blocking, missing authentication, bot detection
   - **Solutions**: Rotate IP, add/verify credentials, enhance browser fingerprinting

3. **HTTP 429 Too Many Requests**
   - **Possible Causes**: Rate limit exceeded, aggressive scraping
   - **Solutions**: Implement stricter rate limits, add delays, distribute requests

4. **Parse Failures**
   - **Possible Causes**: Site structure changes, conditional content
   - **Solutions**: Implement multiple selector strategies, use AI-guided extraction

### Analyzing Error Logs

1. **Error Pattern Analysis**
   ```python
   import pandas as pd
   from collections import Counter
   
   # Load error logs into DataFrame
   errors_df = pd.read_csv('error_logs.csv')
   
   # Analyze by domain
   domain_errors = errors_df.groupby('domain')['error_category'].apply(list)
   domain_error_counts = {domain: Counter(errors) for domain, errors in domain_errors.items()}
   
   # Find domains with high error rates
   for domain, counts in domain_error_counts.items():
       total = sum(counts.values())
       print(f"{domain}: {total} errors")
       for category, count in counts.most_common():
           print(f"  {category}: {count} ({count/total:.1%})")
   ```

2. **Temporal Analysis**
   ```python
   # Convert timestamp to datetime
   errors_df['timestamp'] = pd.to_datetime(errors_df['timestamp'])
   
   # Analyze errors by hour
   errors_by_hour = errors_df.groupby([errors_df['timestamp'].dt.date, 
                                       errors_df['timestamp'].dt.hour]).size()
   
   # Plot error frequency
   errors_by_hour.unstack().plot(kind='heatmap')
   ```

### Developing Custom Error Handlers

1. **Creating Domain-Specific Error Classifiers**
   ```python
   class AmazonErrorClassifier(ErrorClassifier):
       def __init__(self):
           super().__init__()
           # Amazon-specific error patterns
           self._robot_check_patterns = [
               re.compile(r'robot check', re.IGNORECASE),
               re.compile(r'verify you\'re a human', re.IGNORECASE)
           ]
       
       def classify_exception(self, exception, metadata=None):
           # Get base classification
           classification = super().classify_exception(exception, metadata)
           
           # Add Amazon-specific logic
           if hasattr(exception, 'response') and exception.response is not None:
               content = exception.response.text.lower()
               
               # Check for Amazon's robot verification page
               if any(pattern.search(content) for pattern in self._robot_check_patterns):
                   classification['category'] = ErrorCategory.CAPTCHA
                   classification['severity'] = ErrorSeverity.PERSISTENT
                   classification['suggested_actions'] = ['rotate_proxy', 'use_browser_stealth']
           
           return classification
   ```

2. **Specialized Retry Strategies**
   ```python
   def amazon_retry_strategy(retry_state):
       """Custom retry strategy for Amazon."""
       # Get exception if any
       exc = retry_state.outcome.exception()
       
       # Determine if we should retry
       if exc is None:
           # Success, no retry needed
           return False
           
       # Check exception type
       if isinstance(exc, requests.exceptions.HTTPError):
           # Check status code
           if exc.response.status_code == 503:
               # Service unavailable, always retry
               return True
           elif exc.response.status_code == 429:
               # Rate limited, retry with longer backoff
               retry_state.next_action.sleep = min(60, (2 ** retry_state.attempt_number) + random.uniform(0, 1))
               return True
       
       # Use error classifier for other exceptions
       classification = error_classifier.classify_exception(exc)
       return classification['is_retryable']
   ```

## Performance Considerations

### Error Handling Overhead

1. **Classification Performance**
   - Cache common error patterns
   - Use fast regex patterns for content analysis
   - Consider lightweight classifiers for high-frequency operations

2. **Retry Impact**
   - Monitor cumulative retry time
   - Implement retry budgets to limit total retry time
   - Consider concurrent retries for independent operations

3. **Circuit Breaker Efficiency**
   - Use in-memory state for high-performance scenarios
   - Implement circuit breaker aggregation for related domains
   - Monitor circuit state changes for performance impacts

### Monitoring and Alerting

1. **Key Metrics to Track**
   - Error rates by domain and category
   - Retry counts and success rates
   - Circuit breaker state changes
   - Fallback usage frequency

2. **Alert Configuration**
   - Alert on unexpected error patterns
   - Monitor for circuits remaining open
   - Track retry exhaustion patterns
   - Alert on fallback chain exhaustion