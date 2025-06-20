# SmartScrape Core Components

This document provides an overview of the core components implemented in Batch 1 of SmartScrape development.

## 1. Service Interface and Registry

The service interface and registry provide a solid foundation for all SmartScrape services.

### BaseService Abstract Class

The `BaseService` abstract class in `core/service_interface.py` defines the contract that all services must implement:

```python
class BaseService(ABC):
    """Base interface for all SmartScrape services."""
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the service with the given configuration."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Perform cleanup and shutdown operations."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the service."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if the service has been initialized."""
        return getattr(self, '_initialized', False)
```

### ServiceRegistry

The `ServiceRegistry` in `core/service_registry.py` implements a thread-safe singleton for managing services:

- **Dependency Management**: Handles service dependencies and prevents circular references
- **Lifecycle Control**: Manages initialization and shutdown sequences
- **Service Resolution**: Provides services on-demand with dependency injection

## 2. URL Service

The `URLService` in `core/url_service.py` provides all URL-related functionality:

### URL Normalization

```python
def normalize_url(self, url: str, base_url: Optional[str] = None) -> str:
    """Normalize a URL to a canonical form."""
    # Handles relative URLs, scheme defaults, tracking parameter removal, etc.
```

Features:
- Resolves relative URLs against a base URL
- Adds default schemes when missing
- Normalizes hostnames (lowercase)
- Removes default ports (80 for HTTP, 443 for HTTPS)
- Removes tracking parameters (UTM, fbclid, etc.)
- Sorts query parameters for consistent results

### Robots.txt Handling

The service includes a `RobotsTxtChecker` class to ensure compliance with robots.txt directives:

- Checks if URLs are allowed for crawling
- Retrieves crawl delay for respectful crawling
- Discovers sitemaps listed in robots.txt
- Implements caching for performance

### URL Queue Management

The `URLQueue` class provides thread-safe queue operations:

- Tracks visited and in-progress URLs
- Supports priority-based queuing
- Provides queue statistics and management functions

### URL Classification

```python
def classify_url(self, url: str) -> Dict[str, Any]:
    """Classify a URL by type, section, etc."""
```

Classification includes:
- Resource vs. navigation URLs
- Path depth analysis
- URL parameter examination
- Path type identification (product, category, search, etc.)

## 3. HTML Service

The `HTMLService` in `core/html_service.py` handles all HTML processing:

### HTML Cleaning

```python
def clean_html(self, html: str, remove_js: bool = True, remove_css: bool = True, 
             remove_comments: bool = True) -> str:
    """Clean HTML by removing unwanted elements and normalizing structure."""
```

Features:
- Removes scripts, styles, and comments
- Eliminates hidden elements
- Normalizes whitespace
- Preserves important content structures

### Selector Generation

```python
def generate_selector(self, element: Union[Tag, str], html: Optional[str] = None, 
                    method: str = 'css', optimized: bool = True) -> str:
    """Generate a CSS or XPath selector for an element."""
```

Features:
- Generates both CSS and XPath selectors
- Creates optimized selectors when possible
- Implements caching for performance
- Supports element lookup by existing selectors

### Content Extraction

```python
def extract_main_content(self, html: str) -> str:
    """Extract the main content area from an HTML document."""
```

The service also provides:
- Table extraction with headers and data rows
- Link extraction with internal/external classification
- Element comparison for structural similarity

## 4. Strategy Pattern Implementation

The strategy pattern is implemented in `strategies/base_strategy.py`:

### BaseStrategy Abstract Class

```python
class BaseStrategy(ABC):
    """
    Abstract base class that defines the interface for all crawling strategies.
    All concrete crawling strategies should inherit from this class.
    """
    
    def __init__(self, 
                 max_depth: int = 2, 
                 max_pages: int = 100,
                 include_external: bool = False,
                 user_prompt: str = "",
                 filter_chain: Optional[Any] = None):
        """Initialize the base crawling strategy."""
```

The base strategy:
- Defines common parameters and behavior
- Integrates with the service registry to access URL and HTML services
- Implements URL filtering and tracking
- Provides result collection capabilities

### Search Engine Interface

The framework also includes a robust search engine interface:

```python
class SearchEngineInterface(ABC):
    """
    Interface for search engine strategies that can process search requests.
    This is an extension to the BaseStrategy for specialized search operations.
    """
    
    @property
    @abstractmethod
    def capabilities(self) -> List[SearchEngineCapability]:
        """Get the capabilities of this search engine."""
        pass
```

Features:
- `SearchCapabilityType` enum for capability classification
- `SearchEngineCapability` class to describe capabilities
- Registry system for search engines

## 5. Integration and Testing

The components are extensively tested through integration tests in `tests/integration/test_core_services.py`, including:

- **Service Registry Management**: Verifies service registry operations
- **URL and HTML Service Integration**: Tests URL and HTML service cooperation
- **Dependency Resolution**: Confirms proper dependency injection
- **Service Integration Workflow**: Tests a complete workflow
- **Strategy Classes Using Services**: Verifies strategies use services correctly
- **End-to-End Core Flow**: Complete data flow through services
- **Comprehensive Service Features**: Detailed feature verification

## Design Principles

The core architecture follows several key design principles:

1. **Separation of Concerns**: Each service has a specific, well-defined responsibility
2. **Dependency Injection**: Services get dependencies via the service registry
3. **Thread Safety**: Critical operations are protected with locks
4. **Comprehensive Error Handling**: Services gracefully handle exceptional conditions
5. **Extensibility**: The architecture makes it easy to add new services or capabilities

## Conclusion

The Batch 1 core components provide a solid foundation for the SmartScrape system with a service-oriented architecture that ensures modular development and easy extension. The service registry serves as the central coordination point, enabling flexible service composition and dependency management.