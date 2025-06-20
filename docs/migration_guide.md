# Migration Guide: Updating Strategies to the New Pattern

This guide outlines the steps needed to migrate existing scraping strategies to the new Strategy Pattern implementation, ensuring a smooth transition while maintaining backward compatibility.

## Why Migrate

The new Strategy Pattern provides several advantages:

- **Standardized Interfaces**: All strategies now implement the same interfaces, making it easier to understand and use different strategies.
- **Composition**: Strategies can be composed together using SequentialStrategy, FallbackStrategy, and PipelineStrategy.
- **Type Safety**: Using type hints throughout the codebase makes it more maintainable and less error-prone.
- **Metadata & Discovery**: Strategies can specify capabilities, enabling adaptive selection based on requirements.
- **Robust Error Handling**: Standardized error handling framework for all strategies.
- **Testability**: Clean separation of concerns makes testing easier and more reliable.

## Overview of Changes

1. Strategy classes now inherit from `BaseStrategyV2` (web scraping) or `BaseStrategy` (non-web).
2. Strategies use the `@strategy_metadata` decorator to declare capabilities.
3. Strategies accept a `StrategyContext` object for accessing services, rather than creating them directly.
4. Implementation of standardized methods like `execute()`, `crawl()`, `extract()`, and `get_results()`.
5. Error handling now uses the `StrategyErrorHandler` framework.

## Migration Checklist

- [ ] Update imports
- [ ] Add strategy metadata decorator
- [ ] Change base class
- [ ] Update constructor to use StrategyContext
- [ ] Implement required abstract methods
- [ ] Modify internal logic to use context services
- [ ] Update error handling
- [ ] Test the migrated strategy

## Step-by-Step Migration Guide

### 1. Update Imports

```python
# Before
from strategies.base_strategy import BaseStrategy
from core.url_service import URLService
from core.html_service import HTMLService

# After
from strategies.base_strategy_v2 import BaseStrategyV2
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_types import StrategyType, StrategyCapability, strategy_metadata
from strategies.core.strategy_error_handler import StrategyErrorCategory, StrategyErrorSeverity
```

### 2. Add Strategy Metadata Decorator

Add the `@strategy_metadata` decorator to specify the strategy's type, capabilities, and description:

```python
# Add this before your class definition
@strategy_metadata(
    strategy_type=StrategyType.TRAVERSAL,  # Choose the appropriate type
    capabilities={
        StrategyCapability.ROBOTS_TXT_ADHERENCE,
        StrategyCapability.RATE_LIMITING
    },
    description="Breadth-First Search traversal strategy."
)
```

Choose from the available strategy types:
- `StrategyType.TRAVERSAL`: For strategies that navigate through websites (BFS, DFS)
- `StrategyType.INTERACTION`: For strategies that interact with pages (forms, JavaScript)
- `StrategyType.EXTRACTION`: For strategies focused on data extraction
- `StrategyType.SPECIAL_PURPOSE`: For specialized strategies

Select appropriate capabilities for your strategy from `StrategyCapability` enum:
- `JAVASCRIPT_EXECUTION`, `FORM_INTERACTION`, `API_INTERACTION`, `AI_ASSISTED`
- `PAGINATION_HANDLING`, `LOGIN_HANDLING`, `CAPTCHA_SOLVING`
- `DYNAMIC_CONTENT`, `SITEMAP_DISCOVERY`, `ROBOTS_TXT_ADHERENCE`
- `RATE_LIMITING`, `PROXY_SUPPORT`, `ERROR_HANDLING`
- `SCHEMA_EXTRACTION`, `CONTENT_NORMALIZATION`, `DATA_VALIDATION`, etc.

### 3. Change Base Class

Change the class to inherit from `BaseStrategyV2` (for web scraping strategies) or directly from `BaseStrategy` (for other types of strategies):

```python
# Before
class BFSStrategy(BaseStrategy):

# After
class BFSStrategy(BaseStrategyV2):
```

### 4. Update Constructor

Update the constructor to accept a `StrategyContext` object:

```python
# Before
def __init__(self, config=None):
    self.config = config or {}
    self.url_service = URLService()
    self.html_service = HTMLService()
    self.results = []

# After
def __init__(self, context: StrategyContext):
    super().__init__(context)
    # context.config and self._results are available from the base class
    # url_service and html_service are available via properties from BaseStrategyV2
```

### 5. Implement Required Methods

Ensure all required abstract methods are implemented:

```python
@property
def name(self) -> str:
    """Return the strategy name."""
    return "bfs_strategy"  # Use a unique, descriptive name

def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Main entry point for the strategy."""
    # Implementation here
    return result

def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Crawl starting from a URL."""
    # Implementation here (can defer to execute if appropriate)
    return result

def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Extract data from HTML content."""
    # Implementation here
    return extracted_data

def get_results(self) -> List[Dict[str, Any]]:
    """Return all collected results."""
    return self._results  # Use the base class's _results list
```

### 6. Use Context Services

Replace direct service instantiation with context-provided services:

```python
# Before
html_content = requests.get(url).text
soup = BeautifulSoup(html_content, 'html.parser')

# After
html_content = self._fetch_url(url)  # Use utility method from BaseStrategyV2
soup = self._parse_html(html_content)  # Use utility method from BaseStrategyV2
```

The `BaseStrategyV2` class provides common utility methods:
- `_fetch_url(url, **kwargs)`: Fetch content with proper error handling
- `_parse_html(html_content)`: Parse HTML using the HTML service
- `_clean_and_extract(html_content, url, **kwargs)`: Combined cleaning and extraction

Access core services using properties or context:
```python
# Direct access to commonly used services (available in BaseStrategyV2)
normalized_url = self.url_service.normalize_url(url)
is_allowed = self.url_service.is_allowed(url)

# Access other services via context
cache_service = self.context.get_service("cache_service")
```

### 7. Update Error Handling

Replace custom error handling with the standardized error handling framework:

```python
# Before
try:
    response = requests.get(url)
    response.raise_for_status()
except requests.RequestException as e:
    logger.error(f"Error fetching {url}: {str(e)}")
    return None

# After
try:
    response = requests.get(url)
    response.raise_for_status()
except requests.RequestException as e:
    self.handle_error(
        message=f"Error fetching {url}",
        exception=e,
        category=StrategyErrorCategory.NETWORK,
        severity=StrategyErrorSeverity.ERROR
    )
    return None
```

Use the appropriate error categories and severities:
- Categories: `NETWORK`, `HTML_PARSING`, `RATE_LIMIT`, `AUTHENTICATION`, etc.
- Severities: `INFO`, `WARNING`, `ERROR`, `FATAL`

Use `self.get_errors()` to retrieve errors, optionally filtered by category or severity.

### 8. Update Result Storage

Use the standardized `_results` list from the base class:

```python
# Before
self.results.append(data)

# After
self._results.append(data)
```

### 9. Test the Migrated Strategy

After migrating, test the strategy:

```python
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_factory import StrategyFactory

# Create context and factory
context = StrategyContext({})
factory = StrategyFactory(context)

# Register and get your strategy
factory.register_strategy(BFSStrategy)
strategy = factory.get_strategy("bfs_strategy")

# Execute and check results
result = strategy.execute("https://example.com")
print(f"Result: {result}")
print(f"Collected results: {strategy.get_results()}")
print(f"Errors: {strategy.get_errors()}")
```

## Advanced Migration: Using Composite Strategies

Once your individual strategies are migrated, you can take advantage of composite strategies:

### Sequential Strategy

```python
# Execute multiple strategies in sequence
sequential = SequentialStrategy(context)
sequential.add_strategy(factory.get_strategy("sitemap_strategy"))
sequential.add_strategy(factory.get_strategy("bfs_strategy"))
result = sequential.execute("https://example.com")
```

### Fallback Strategy

```python
# Try strategies in order until one succeeds
fallback = FallbackStrategy(context)
fallback.add_strategy(factory.get_strategy("api_strategy"))
fallback.add_strategy(factory.get_strategy("dom_strategy"))
fallback.add_strategy(factory.get_strategy("html_strategy"))
result = fallback.execute("https://example.com/products")
```

### Pipeline Strategy

```python
# Chain strategies where output of one feeds into the next
pipeline = PipelineStrategy(context)
pipeline.add_strategy(factory.get_strategy("discovery_strategy"))
pipeline.add_strategy(factory.get_strategy("extraction_strategy"))
pipeline.add_strategy(factory.get_strategy("validation_strategy"))
result = pipeline.execute("https://example.com")
```

## Backward Compatibility Layer

For a smoother transition, you can create a backward compatibility layer:

```python
def create_legacy_compatible_strategy(strategy_class, config=None):
    """Create a strategy using the new pattern but compatible with old code."""
    context = StrategyContext(config)
    factory = StrategyFactory(context)
    factory.register_strategy(strategy_class)
    
    strategy = factory.get_strategy(strategy_class(context).name)
    return strategy

# Usage with old code
strategy = create_legacy_compatible_strategy(BFSStrategy, config={"max_depth": 3})
```

## Common Issues and Solutions

### Issue 1: AttributeError when accessing services

**Problem**: `AttributeError: 'NoneType' object has no attribute 'method_name'`

**Solution**: Ensure the service is properly registered in the service registry and accessible via the context.

```python
# Check if service is available before using
url_service = self.context.get_service("url_service")
if url_service is not None:
    url_service.normalize_url(url)
else:
    # Fallback logic
```

### Issue 2: Strategy execute returns None unexpectedly

**Problem**: Strategy's `execute()` method returns `None` when a result was expected.

**Solution**: Check error handling in the strategy and ensure all failure paths are properly handled.

```python
# Add appropriate error handling and default returns
def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
    try:
        # Implementation
        if not result:
            self.handle_error(
                message="No result found",
                category=StrategyErrorCategory.CONTENT_EXTRACTION,
                severity=StrategyErrorSeverity.WARNING
            )
            return {"status": "no_data", "url": url}
        return result
    except Exception as e:
        self.handle_error(
            message="Unhandled exception in execute",
            exception=e,
            category=StrategyErrorCategory.UNKNOWN,
            severity=StrategyErrorSeverity.ERROR
        )
        return {"status": "error", "url": url}
```

### Issue 3: Strategy metadata not found

**Problem**: `ValueError: Strategy class MyStrategy is missing valid StrategyMetadata.`

**Solution**: Ensure the `@strategy_metadata` decorator is properly applied to the strategy class.

```python
# Make sure the decorator is directly above the class definition
@strategy_metadata(
    strategy_type=StrategyType.TRAVERSAL,
    capabilities={StrategyCapability.ROBOTS_TXT_ADHERENCE},
    description="My strategy description."
)
class MyStrategy(BaseStrategyV2):
    # Class implementation
```

## Complete Example: Migrated BFS Strategy

Here's a complete example of a BFS strategy migrated to the new pattern:

```python
from typing import Dict, Any, Optional, List, Set
import logging
from urllib.parse import urlparse, urljoin

from strategies.base_strategy_v2 import BaseStrategyV2
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_types import StrategyType, StrategyCapability, strategy_metadata
from strategies.core.strategy_error_handler import StrategyErrorCategory, StrategyErrorSeverity

logger = logging.getLogger(__name__)

@strategy_metadata(
    strategy_type=StrategyType.TRAVERSAL,
    capabilities={
        StrategyCapability.ROBOTS_TXT_ADHERENCE,
        StrategyCapability.RATE_LIMITING
    },
    description="Breadth-First Search traversal strategy for web scraping."
)
class BFSStrategy(BaseStrategyV2):
    """
    A Breadth-First Search strategy for traversing websites.
    Visits all links at the current depth before moving to the next depth level.
    """
    
    @property
    def name(self) -> str:
        return "bfs_strategy"
    
    def __init__(self, context: StrategyContext):
        super().__init__(context)
        self.visited_urls: Set[str] = set()
        self.max_urls_to_visit = context.config.get("max_urls_to_visit", 100)
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute the BFS strategy starting from the given URL."""
        max_depth = kwargs.get("max_depth", 2)
        
        # Reset state for new execution
        self.visited_urls = set()
        
        # Start crawling
        result = self.crawl(url, max_depth=max_depth, **kwargs)
        
        return {
            "start_url": url,
            "max_depth": max_depth,
            "urls_visited": len(self.visited_urls),
            "results_count": len(self._results)
        }
    
    def crawl(self, start_url: str, max_depth: int = 2, **kwargs) -> Optional[Dict[str, Any]]:
        """Perform BFS crawling from the start URL up to max_depth."""
        queue = [(self.url_service.normalize_url(start_url), 0)]  # (url, depth)
        
        while queue and len(self.visited_urls) < self.max_urls_to_visit:
            current_url, depth = queue.pop(0)
            
            if current_url in self.visited_urls or depth > max_depth:
                continue
            
            logger.info(f"Visiting {current_url} (depth {depth})")
            
            # Mark as visited
            self.visited_urls.add(current_url)
            
            # Check robots.txt
            if not self.url_service.is_allowed(current_url):
                logger.info(f"Skipping {current_url} (disallowed by robots.txt)")
                continue
            
            # Fetch content
            try:
                content = self._fetch_url(current_url)
                if not content:
                    continue
                
                # Extract data and store in results
                data = self._extract_page_data(content, current_url, depth)
                if data:
                    self._results.append(data)
                
                # If not at max depth, extract links and add to queue
                if depth < max_depth:
                    links = self._extract_links(content, current_url)
                    for link in links:
                        if link not in self.visited_urls:
                            queue.append((link, depth + 1))
            
            except Exception as e:
                self.handle_error(
                    message=f"Error processing {current_url}",
                    exception=e,
                    category=StrategyErrorCategory.UNKNOWN,
                    severity=StrategyErrorSeverity.ERROR
                )
        
        return {
            "urls_queued": len(queue),
            "urls_visited": len(self.visited_urls)
        }
    
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Extract data from HTML content."""
        return self._extract_page_data(html_content, url, 0)
    
    def _extract_page_data(self, html_content: str, url: str, depth: int) -> Optional[Dict[str, Any]]:
        """Extract data from a page."""
        try:
            # Use the HTML service to parse content
            soup = self._parse_html(html_content)
            if not soup:
                return None
            
            # Extract basic page data
            title = soup.title.text.strip() if soup.title else "No title"
            
            # Extract meta description
            description = ""
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.has_attr("content"):
                description = meta_desc["content"]
            
            return {
                "url": url,
                "depth": depth,
                "title": title,
                "description": description,
                "content_length": len(html_content)
            }
        
        except Exception as e:
            self.handle_error(
                message=f"Error extracting data from {url}",
                exception=e,
                category=StrategyErrorCategory.CONTENT_EXTRACTION,
                severity=StrategyErrorSeverity.WARNING
            )
            return None
    
    def _extract_links(self, html_content: str, base_url: str) -> List[str]:
        """Extract and normalize links from HTML content."""
        try:
            soup = self._parse_html(html_content)
            if not soup:
                return []
            
            links = []
            base_domain = urlparse(base_url).netloc
            
            # Find all anchor tags
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()
                
                # Skip empty links and anchors
                if not href or href.startswith("#"):
                    continue
                
                # Normalize the URL
                try:
                    full_url = urljoin(base_url, href)
                    normalized_url = self.url_service.normalize_url(full_url)
                    
                    # Optional: filter to stay on the same domain
                    if urlparse(normalized_url).netloc == base_domain:
                        links.append(normalized_url)
                except Exception as e:
                    self.handle_error(
                        message=f"Error normalizing link {href}",
                        exception=e,
                        category=StrategyErrorCategory.URL_PROCESSING,
                        severity=StrategyErrorSeverity.INFO
                    )
            
            return links
        
        except Exception as e:
            self.handle_error(
                message=f"Error extracting links from {base_url}",
                exception=e,
                category=StrategyErrorCategory.CONTENT_EXTRACTION,
                severity=StrategyErrorSeverity.WARNING
            )
            return []
```

## Troubleshooting

If you encounter issues during migration, check the documentation in `docs/strategy_pattern.md` for detailed troubleshooting guidance.

For specific issues not covered in this guide, please open an issue in the repository with the tag `migration-help`.

## Timeline for Migration

We recommend the following timeline for migrating existing strategies:

1. **Week 1-2**: Migrate simple, standalone strategies
2. **Week 3-4**: Migrate complex strategies and test thoroughly
3. **Week 5-6**: Update controllers and user-facing code to use the new pattern
4. **Week 7-8**: Deprecate old interfaces and complete transition

During the transition period, both old and new interfaces will be supported to ensure a smooth migration.