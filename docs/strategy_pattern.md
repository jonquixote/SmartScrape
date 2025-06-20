# Strategy Pattern Documentation

This document provides detailed documentation for the SmartScrape strategy pattern implementation, explaining key concepts, components, and usage patterns.

## Overview

The strategy pattern is a behavioral design pattern that enables selecting an algorithm at runtime. In SmartScrape, the strategy pattern is implemented to allow flexible selection and composition of scraping strategies.

## Core Components

### 1. Strategy Interface (`BaseStrategy`)

The base interface for all strategies defines the contract that all concrete strategies must implement:

```python
class BaseStrategy(ABC, StrategyErrorHandlingMixin):
    def __init__(self, context: Optional['StrategyContext'] = None):
        # Initialize with a context object
        
    @abstractmethod
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute the strategy for the given URL."""
        
    @abstractmethod
    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Main entry point for crawling using this strategy."""
        
    @abstractmethod
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Data extraction interface for this strategy."""
        
    @abstractmethod
    def get_results(self) -> List[Dict[str, Any]]:
        """Standardized method to access results collected by the strategy."""
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the strategy."""
```

### 2. Strategy Types and Metadata

Strategies are classified by type and capabilities, making it easier to select the right strategy:

```python
class StrategyType(Enum):
    TRAVERSAL = "traversal"            # Strategies that navigate through websites
    INTERACTION = "interaction"        # Strategies that interact with pages
    EXTRACTION = "extraction"          # Strategies focused on data extraction
    SPECIAL_PURPOSE = "special_purpose" # Specialized strategies

class StrategyCapability(Enum):
    JAVASCRIPT_EXECUTION = "javascript_execution"
    FORM_INTERACTION = "form_interaction"
    API_INTERACTION = "api_interaction"
    # ...more capabilities...
```

Strategy metadata is attached using a decorator:

```python
@strategy_metadata(
    strategy_type=StrategyType.TRAVERSAL,
    capabilities={StrategyCapability.ROBOTS_TXT_ADHERENCE, StrategyCapability.RATE_LIMITING},
    description="Breadth-First Search traversal strategy."
)
class BFSStrategy(BaseStrategyV2):
    # Implementation...
```

### 3. Strategy Context

The context provides shared services and configuration to strategies:

```python
class StrategyContext:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.service_registry = ServiceRegistry()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_service(self, service_name: str):
        """Get a service from the registry."""
        return self.service_registry.get_service(service_name)
```

### 4. Strategy Factory

The factory creates strategy instances and provides discovery mechanisms:

```python
class StrategyFactory:
    def __init__(self, context: StrategyContext):
        self.context = context
        self._strategy_classes = {}
        self._strategy_metadata = {}
        
    def register_strategy(self, strategy_class: Type[BaseStrategy]) -> None:
        """Register a strategy class and its metadata with the factory."""
        
    def get_strategy(self, strategy_name: str) -> BaseStrategy:
        """Get a strategy instance by name."""
        
    def get_strategies_by_capability(self, required_capabilities: Set[StrategyCapability]) -> List[Type[BaseStrategy]]:
        """Get strategy classes that have all required capabilities."""
```

### 5. Error Handling Framework

The error handling framework provides standardized error reporting and handling:

```python
class StrategyErrorSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"

class StrategyErrorCategory(Enum):
    NETWORK = "network"
    HTML_PARSING = "html_parsing"
    # ...more categories...

class StrategyErrorHandlingMixin:
    def handle_error(self, message: str, category: StrategyErrorCategory, 
                    severity: StrategyErrorSeverity, exception: Optional[Exception] = None):
        """Record an error with the strategy."""
        
    def get_errors(self, category: Optional[StrategyErrorCategory] = None, 
                  severity: Optional[StrategyErrorSeverity] = None) -> List[StrategyError]:
        """Get errors, optionally filtered by category and/or severity."""
```

### 6. Composite Strategies

Composite strategies allow combining multiple strategies in useful ways:

```python
class SequentialStrategy(StrategyComposite):
    """Execute multiple strategies in sequence."""
    
class FallbackStrategy(StrategyComposite):
    """Try strategies in order until one succeeds."""
    
class PipelineStrategy(StrategyComposite):
    """Chain strategies where output of one feeds into the next."""
```

### 7. Base Implementation (`BaseStrategyV2`)

`BaseStrategyV2` provides a concrete implementation of common functionality:

```python
class BaseStrategyV2(WebScrapingStrategy):
    """Base strategy with common web scraping utilities."""
    
    def _fetch_url(self, url: str, **kwargs) -> Optional[str]:
        """Fetch content from a URL."""
        
    def _parse_html(self, html_content: str) -> Any:
        """Parse HTML content."""
        
    def _clean_and_extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Clean HTML and extract data."""
```

## Usage Examples

### 1. Basic Strategy Usage

```python
# Create context and factory
context = StrategyContext({"max_depth": 3})
factory = StrategyFactory(context)

# Register strategies
factory.register_strategy(BFSStrategy)

# Get a strategy instance
bfs = factory.get_strategy("bfs_strategy")

# Execute the strategy
result = bfs.execute("https://example.com")
print(f"Execution result: {result}")
print(f"Collected results: {bfs.get_results()}")
print(f"Errors: {bfs.get_errors()}")
```

### 2. Adaptive Strategy Selection

```python
# Select strategy based on capabilities
required_capabilities = {
    StrategyCapability.PAGINATION_HANDLING,
    StrategyCapability.CONTENT_NORMALIZATION
}

strategy_classes = factory.get_strategies_by_capability(required_capabilities)
if strategy_classes:
    strategy = factory.get_strategy(strategy_classes[0](context).name)
    result = strategy.execute("https://example.com/page/1")
```

### 3. Composite Strategy Examples

```python
# Sequential strategy - execute strategies in sequence
sequential = SequentialStrategy(context)
sequential.add_strategy(factory.get_strategy("sitemap_strategy"))
sequential.add_strategy(factory.get_strategy("bfs_strategy"))
result = sequential.execute("https://example.com")

# Fallback strategy - try strategies until one succeeds
fallback = FallbackStrategy(context)
fallback.add_strategy(factory.get_strategy("api_strategy"))
fallback.add_strategy(factory.get_strategy("dom_strategy"))
fallback.add_strategy(factory.get_strategy("html_strategy"))
result = fallback.execute("https://example.com/products")

# Pipeline strategy - chain strategies where output feeds into the next
pipeline = PipelineStrategy(context)
pipeline.add_strategy(factory.get_strategy("discovery_strategy"))
pipeline.add_strategy(factory.get_strategy("extraction_strategy"))
pipeline.add_strategy(factory.get_strategy("validation_strategy"))
result = pipeline.execute("https://example.com")
```

### 4. Using the Adaptive Scraper

```python
from controllers.adaptive_scraper import AdaptiveScraper

scraper = AdaptiveScraper({"max_depth": 3})

# Use a specific strategy
result1 = scraper.scrape("https://example.com", strategy_name="bfs_strategy")

# Let the scraper choose based on capabilities
result2 = scraper.scrape("https://example.com", 
                        required_capabilities={StrategyCapability.AI_ASSISTED})

# Pass additional parameters to the strategy
result3 = scraper.scrape("https://example.com", 
                        strategy_name="pagination_strategy",
                        max_pages=5)
```

## Best Practices

### 1. Strategy Design

* **Single Responsibility**: Each strategy should focus on a specific scraping technique.
* **Clear Capabilities**: Accurately specify the capabilities your strategy provides.
* **Proper Error Handling**: Use the error handling framework to report issues.
* **Service Independence**: Rely on services provided by the context rather than creating your own.

### 2. Strategy Testing

* **Isolated Tests**: Test each strategy in isolation with mock services.
* **Capability Verification**: Verify that the strategy delivers on its advertised capabilities.
* **Error Case Testing**: Test that errors are properly handled and reported.
* **Integration Testing**: Test strategies within composite structures.

### 3. Composing Strategies

* **Sequential for Process Steps**: Use `SequentialStrategy` for distinct process steps.
* **Fallback for Alternatives**: Use `FallbackStrategy` when you have multiple approaches.
* **Pipeline for Data Flow**: Use `PipelineStrategy` when data passes through transformations.

### 4. Error Handling Practices

* **Use Appropriate Categories**: Select the most specific error category available.
* **Proper Severity Levels**: Use severity levels consistently:
  * `INFO`: Informational, non-critical issues
  * `WARNING`: Issues that don't prevent execution but may affect quality
  * `ERROR`: Issues that prevent a specific operation
  * `FATAL`: Critical issues that should terminate execution

## Strategy Implementation Checklist

- [ ] Does the strategy implement all required abstract methods?
- [ ] Is the strategy decorated with `@strategy_metadata`?
- [ ] Does the strategy use the context for accessing services?
- [ ] Does the strategy properly handle and report errors?
- [ ] Is the strategy's `name` property unique and descriptive?
- [ ] Does the strategy store results in the standard `_results` list?
- [ ] Are type hints used consistently throughout the implementation?
- [ ] Is the strategy properly tested, including error cases?

## Advanced Topics

### 1. Creating Custom Composite Strategies

You can create custom composite strategies by extending `StrategyComposite`:

```python
class ParallelStrategy(StrategyComposite):
    """Execute multiple strategies in parallel."""
    
    @property
    def name(self) -> str:
        return "parallel_strategy"
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        # Implementation using threading or asyncio...
```

### 2. Strategy Lifecycle Management

Strategies support lifecycle methods:

```python
def initialize(self) -> None:
    """Initialize resources."""
    
def shutdown(self) -> None:
    """Clean up resources."""
    
def pause(self) -> None:
    """Pause execution."""
    
def resume(self) -> None:
    """Resume execution."""
```

### 3. Strategy Execution Utilities

The framework provides decorators for common execution patterns:

```python
from strategies.core.strategy_utils import retry_on_failure, with_timeout, measure_performance

class MyStrategy(BaseStrategyV2):
    @retry_on_failure(max_attempts=3, delay_seconds=1.0)
    @with_timeout(timeout_seconds=30.0)
    @measure_performance
    def execute(self, url: str, **kwargs):
        # Implementation...
```

## Troubleshooting

### Common Issues

1. **Strategy not found**: Ensure the strategy is registered with the factory.
2. **Service not available**: Check if required services are properly registered.
3. **Type errors**: Ensure all required methods return the correct types.
4. **Missing metadata**: Check if the strategy is decorated with `@strategy_metadata`.

### Debugging Strategies

1. Enable debug logging to see detailed execution flow:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Use the strategy's error collection to check for issues:
   ```python
   errors = strategy.get_errors()
   for error in errors:
       print(f"{error.severity.value.upper()}: {error.message}")
       if error.exception:
           traceback.print_exception(type(error.exception), error.exception, error.traceback)
   ```

## References

- [Strategy Pattern (Wikipedia)](https://en.wikipedia.org/wiki/Strategy_pattern)
- [Python ABC Module](https://docs.python.org/3/library/abc.html)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Composite Pattern (Refactoring Guru)](https://refactoring.guru/design-patterns/composite)

## Appendix

### Strategy Capability Reference

This table lists all available strategy capabilities and their intended usage:

| Capability | Description | Typical Strategy Types |
|------------|-------------|------------------------|
| `JAVASCRIPT_EXECUTION` | Execute JavaScript on web pages | INTERACTION |
| `FORM_INTERACTION` | Interact with forms (fill, submit) | INTERACTION |
| `API_INTERACTION` | Interact with APIs | SPECIAL_PURPOSE |
| `AI_ASSISTED` | Uses AI models for extraction/decision | Any |
| `PAGINATION_HANDLING` | Navigate through paginated content | TRAVERSAL |
| `LOGIN_HANDLING` | Handle login forms and authentication | INTERACTION |
| `CAPTCHA_SOLVING` | Solve or bypass CAPTCHAs | INTERACTION |
| `DYNAMIC_CONTENT` | Handle dynamically loaded content | INTERACTION |
| `SITEMAP_DISCOVERY` | Discover URLs via sitemaps | TRAVERSAL |
| `ROBOTS_TXT_ADHERENCE` | Respects robots.txt rules | TRAVERSAL |
| `RATE_LIMITING` | Implements rate limiting | Any |
| `PROXY_SUPPORT` | Can use proxy servers | Any |
| `ERROR_HANDLING` | Advanced error handling | Any |
| `SCHEMA_EXTRACTION` | Extract structured data schemas | EXTRACTION |
| `CONTENT_NORMALIZATION` | Normalize extracted content | EXTRACTION |
| `DATA_VALIDATION` | Validate extracted data | EXTRACTION |
| `SITE_SPECIFIC` | Tailored for specific websites | SPECIAL_PURPOSE |

### Strategy Type Reference

| Strategy Type | Primary Focus | Common Capabilities |
|---------------|---------------|---------------------|
| `TRAVERSAL` | Discovering and navigating content | ROBOTS_TXT_ADHERENCE, SITEMAP_DISCOVERY, PAGINATION_HANDLING |
| `INTERACTION` | Interacting with web pages | JAVASCRIPT_EXECUTION, FORM_INTERACTION, LOGIN_HANDLING |
| `EXTRACTION` | Extracting data from content | SCHEMA_EXTRACTION, CONTENT_NORMALIZATION, DATA_VALIDATION |
| `SPECIAL_PURPOSE` | Specialized functionality | API_INTERACTION, SITE_SPECIFIC |