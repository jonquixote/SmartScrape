"""
Base strategy module defining the abstract class for all crawling strategies and search engines.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from urllib.parse import urlparse
from enum import Enum, auto
import logging
import time
import requests
from core.service_registry import ServiceRegistry
from strategies.mixins.resource_management_mixin import ResourceManagementMixin
from strategies.mixins.error_handling_mixin import ErrorHandlingMixin, CircuitOpenError

logger = logging.getLogger(__name__)

class StrategyCapability(Enum):
    """Capabilities that a strategy can provide"""
    FORM_INTERACTION = auto()    # Can interact with forms
    DOM_NAVIGATION = auto()      # Can navigate the DOM
    URL_PARAMETER = auto()       # Can handle URL parameters
    API_INTEGRATION = auto()     # Can interact with APIs
    JAVASCRIPT_EXECUTION = auto()# Can execute JavaScript
    CAPTCHA_HANDLING = auto()    # Can handle CAPTCHAs
    LOGIN_HANDLING = auto()      # Can handle login forms
    FACETED_SEARCH = auto()      # Can handle faceted search interfaces
    INFINITE_SCROLL = auto()     # Can handle infinite scroll
    PAGINATION = auto()          # Can handle pagination

class StrategyMetadata:
    """Metadata describing a strategy"""
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 capabilities: List[StrategyCapability],
                 parameters: Dict[str, Dict[str, Any]] = None):
        """
        Initialize strategy metadata.
        
        Args:
            name: Strategy name
            description: Strategy description
            capabilities: List of capabilities the strategy provides
            parameters: Dictionary of parameters the strategy accepts
        """
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.parameters = parameters or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": [c.name for c in self.capabilities],
            "parameters": self.parameters
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyMetadata':
        """Create metadata from dictionary"""
        capabilities = [StrategyCapability[c] for c in data.get("capabilities", [])]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            capabilities=capabilities,
            parameters=data.get("parameters", {})
        )

class SearchCapabilityType(Enum):
    """Enumeration of different search engine capability types"""
    FORM_BASED = auto()          # Can interact with HTML forms 
    URL_PARAMETER = auto()       # Can construct URL parameters for search
    DOM_MANIPULATION = auto()    # Can manipulate DOM elements directly
    API_BASED = auto()           # Can interact with APIs
    AJAX_HANDLING = auto()       # Can handle AJAX requests and responses
    AUTOCOMPLETE = auto()        # Can handle autocomplete/typeahead interfaces
    MULTI_STEP = auto()          # Can handle multi-step search processes
    CAPTCHA_HANDLING = auto()    # Can handle captchas
    INFINITE_SCROLL = auto()     # Can handle infinite scroll
    FACETED_SEARCH = auto()      # Can handle faceted search interfaces
    
    
class SearchEngineCapability:
    """Describes a specific capability of a search engine strategy"""
    
    def __init__(self, 
                 capability_type: SearchCapabilityType,
                 confidence: float = 1.0,
                 requires_browser: bool = False,
                 description: str = ""):
        """
        Initialize a search engine capability.
        
        Args:
            capability_type: Type of search capability
            confidence: Confidence level for this capability (0.0-1.0)
            requires_browser: Whether this capability requires a browser
            description: Human-readable description of the capability
        """
        self.capability_type = capability_type
        self.confidence = confidence
        self.requires_browser = requires_browser
        self.description = description
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary for serialization"""
        return {
            "type": self.capability_type.name,
            "confidence": self.confidence,
            "requires_browser": self.requires_browser,
            "description": self.description
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchEngineCapability':
        """Create capability from dictionary"""
        return cls(
            capability_type=SearchCapabilityType[data["type"]],
            confidence=data.get("confidence", 1.0),
            requires_browser=data.get("requires_browser", False),
            description=data.get("description", "")
        )


class SearchEngineInterface(ABC):
    """
    Interface for search engine strategies that can process search requests.
    This is an extension to the BaseStrategy for specialized search operations.
    """
    
    @property
    @abstractmethod
    def capabilities(self) -> List[SearchEngineCapability]:
        """
        Get the capabilities of this search engine.
        
        Returns:
            List of SearchEngineCapability objects
        """
        pass
    
    @abstractmethod
    async def search(self, 
                    query: str, 
                    url: str, 
                    params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a search operation using this engine.
        
        Args:
            query: The search query string
            url: The target URL to search on
            params: Additional parameters for the search
            
        Returns:
            Dictionary containing search results and metadata
        """
        pass
    
    @abstractmethod
    async def can_handle(self, url: str, html: Optional[str] = None) -> Tuple[bool, float]:
        """
        Determine if this search engine can handle the given URL/page.
        
        Args:
            url: The URL to check
            html: Optional HTML content of the page
            
        Returns:
            Tuple of (can_handle, confidence)
        """
        pass
    
    @abstractmethod
    def get_required_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters required by this search engine.
        
        Returns:
            Dictionary of parameter name to parameter specification
        """
        pass
        

class BaseStrategy(ABC, ResourceManagementMixin, ErrorHandlingMixin):
    """
    Abstract base class that defines the interface for all crawling strategies.
    All concrete crawling strategies should inherit from this class.
    """
    
    def __init__(self, 
                 context=None,
                 max_depth: int = 2, 
                 max_pages: int = 100,
                 include_external: bool = False,
                 user_prompt: str = "",
                 filter_chain: Optional[Any] = None):
        """
        Initialize the base crawling strategy.
        
        Args:
            context: The strategy context containing shared services and configuration
            max_depth (int): Maximum crawling depth
            max_pages (int): Maximum number of pages to crawl
            include_external (bool): Whether to include external links
            user_prompt (str): The user's original request/prompt
            filter_chain: Filter chain to apply to URLs
        """
        self.context = context
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.include_external = include_external
        self.user_prompt = user_prompt
        self.filter_chain = filter_chain
        self.visited_urls: Set[str] = set()
        self.url_scores: Dict[str, float] = {}
        self.results = []
        
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Get services from registry
        registry = ServiceRegistry()
        
        # Try to get services from the registry or fallback to global_registry if available
        try:
            self.url_service = registry.get_service("url_service")
        except KeyError:
            # Try to get from global registry in controllers module
            try:
                from controllers import global_registry
                self.url_service = global_registry.get_service("url_service")
            except (ImportError, KeyError):
                raise KeyError("Service url_service not registered in any registry")
        
        try:
            self.html_service = registry.get_service("html_service")
        except KeyError:
            # Try to get from global registry
            try:
                from controllers import global_registry
                self.html_service = global_registry.get_service("html_service")
            except (ImportError, KeyError):
                raise KeyError("Service html_service not registered in any registry")
        
        # Get session_manager if needed by strategies
        try:
            self.session_manager = registry.get_service("session_manager")
        except KeyError:
            # Try to get from global registry
            try:
                from controllers import global_registry
                self.session_manager = global_registry.get_service("session_manager")
            except (ImportError, KeyError):
                # This is optional, so don't raise an error
                self.session_manager = None
        
        # Initialize mixins
        self._initialize_resource_management()
        self._initialize_error_handling()
        
    def initialize(self):
        """
        Initialize the strategy. This method can be overridden by subclasses
        to perform additional initialization after construction.
        """
        pass
        
    @abstractmethod
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute the crawling strategy.
        
        Args:
            url: The URL to crawl
            **kwargs: Additional arguments including crawler, extraction_config, etc.
            
        Returns:
            Dictionary containing the results
        """
        pass
    
    async def execute_async(self, crawler, start_url, extraction_config=None):
        """
        Execute the crawling strategy asynchronously.
        This is an optional method that strategies can override if they support async execution.
        
        Args:
            crawler: The AsyncWebCrawler instance
            start_url: The starting URL
            extraction_config: Optional extraction configuration
            
        Returns:
            Dictionary containing the results
        """
        # Default implementation: call the synchronous execute method
        return self.execute(start_url, crawler=crawler, extraction_config=extraction_config)
    
    @abstractmethod
    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Crawl from a starting URL using this strategy.
        
        Args:
            start_url: The URL to start crawling from
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Optional dictionary with crawl results, or None if crawl failed
        """
        pass

    @abstractmethod
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Extract data from HTML content using this strategy.
        
        Args:
            html_content: The HTML content to extract data from
            url: The URL the content was fetched from
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Optional dictionary with extracted data, or None if extraction failed
        """
        pass

    @abstractmethod
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get all results collected by this strategy.
        
        Returns:
            List of dictionaries containing the collected results
        """
        pass
    
    def should_visit(self, url: str, base_url: str) -> bool:
        """
        Determine whether a URL should be visited based on filters.
        
        Args:
            url (str): URL to check
            base_url (str): The base URL of the crawl
            
        Returns:
            bool: True if the URL should be visited, False otherwise
        """
        # Skip already visited URLs
        if url in self.visited_urls:
            return False
            
        # Apply domain filtering if not including external links
        if not self.include_external:
            base_domain = urlparse(base_url).netloc
            url_domain = urlparse(url).netloc
            
            # Skip URLs from different domains
            if url_domain != base_domain:
                return False
        
        # Apply filter chain if available
        if self.filter_chain:
            return self.filter_chain.should_keep(url)
            
        return True
    
    def add_visited(self, url: str) -> None:
        """
        Mark a URL as visited.
        
        Args:
            url (str): URL that has been visited
        """
        self.visited_urls.add(url)
        
    def get_visited_count(self) -> int:
        """
        Get the count of visited URLs.
        
        Returns:
            int: Count of visited URLs
        """
        return len(self.visited_urls)
    
    def add_result(self, url: str, content: Any, depth: int, relevance: float = 1.0, explanation: str = None) -> None:
        """
        Add a result to the collection.
        
        Args:
            url (str): URL of the page
            content: Extracted content
            depth (int): Crawl depth
            relevance (float): Relevance score (0.0-1.0)
            explanation (str, optional): Explanation of the relevance score
        """
        result = {
            "source_url": url,
            "depth": depth,
            "relevance": relevance,
            "data": content
        }
        
        if explanation:
            result["relevance_explanation"] = explanation
            
        self.results.append(result)
    
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            str: Strategy name
        """
        pass
        
    def _get_session(self, domain: str, force_new: bool = False) -> requests.Session:
        """
        Get an HTTP session for the specified domain.
        
        Args:
            domain: The domain for which to get a session
            force_new: Whether to force creation of a new session
            
        Returns:
            An HTTP session object
        """
        # This is now implemented in ResourceManagementMixin
        return super()._get_session(domain, force_new)
    
    def _classify_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify an error using the ErrorClassifier service.
        
        Args:
            error: The exception to classify
            context: Context information about the error (URL, domain, etc.)
            
        Returns:
            A dictionary with error classification details
        """
        # This is now implemented in ErrorHandlingMixin
        return super()._classify_error(error, context)
    
    def _should_retry(self, operation: str, attempt: int, error: Dict[str, Any]) -> bool:
        """
        Determine if an operation should be retried based on the error.
        
        Args:
            operation: The name of the operation (e.g., 'fetch', 'parse')
            attempt: The current attempt number (1 for first attempt)
            error: The classified error dictionary
            
        Returns:
            True if the operation should be retried, False otherwise
        """
        # This is now implemented in ErrorHandlingMixin
        return super()._should_retry(operation, attempt, error)
    
    def _get_proxy(self, domain: str) -> Optional[Dict[str, str]]:
        """
        Get a proxy for the specified domain.
        
        Args:
            domain: The domain for which to get a proxy
            
        Returns:
            A proxy configuration dictionary or None
        """
        # This is now implemented in ResourceManagementMixin
        return super()._get_proxy(domain)
    
    def _handle_rate_limiting(self, domain: str) -> None:
        """
        Handle rate limiting for the specified domain.
        
        Args:
            domain: The domain for which to handle rate limiting
        """
        # This is now implemented in ResourceManagementMixin
        return super()._handle_rate_limiting(domain)
    
    def _make_request(self, url: str, method: str = 'GET', **kwargs) -> requests.Response:
        """
        Make an HTTP request with robust error handling and resource management.
        
        This method handles:
        - Session management
        - Proxy selection
        - Rate limiting
        - Circuit breaker protection
        - Retries with backoff
        - Error classification and handling
        - Performance metrics
        
        Args:
            url: The URL to request
            method: The HTTP method to use ('GET', 'POST', etc.)
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            requests.Response: The response object
            
        Raises:
            requests.exceptions.RequestException: If the request fails after retries
            CircuitOpenError: If the circuit breaker is open for the domain
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        max_retries = kwargs.pop('max_retries', 3)
        
        # Handle rate limiting
        self._handle_rate_limiting(domain)
        
        # Check circuit breaker
        if not self._check_circuit_breaker(domain):
            logger.warning(f"Circuit breaker open for {domain}, skipping request to {url}")
            raise CircuitOpenError(f"Circuit breaker open for {domain}")
        
        # Get session and proxy
        session = self._get_session(domain)
        proxy = self._get_proxy(domain)
        if proxy:
            session.proxies.update(proxy)
            
        # Record metrics
        start_time = time.time()
        
        # Define the actual request function
        def do_request():
            return session.request(method, url, **kwargs)
        
        try:
            # Execute with error handling (includes retries)
            response = self.execute_with_error_handling(
                do_request,
                operation=f"{method.lower()}_request",
                domain=domain,
                max_retries=max_retries,
                url=url
            )
            
            # Raise for HTTP errors
            response.raise_for_status()
            
            # Record success metrics
            self._record_request_metrics(url, start_time, True)
            self._record_success(domain)
            
            return response
            
        except Exception as e:
            # Record failure metrics
            self._record_request_metrics(url, start_time, False)
            
            # Re-raise the exception
            logger.error(f"Request to {url} failed: {str(e)}")
            raise
    
    def execute_with_error_handling(self, 
                              func, 
                              operation: str, 
                              domain: str,
                              max_retries: int = 3,
                              **error_context) -> Any:
        """
        Execute a function with error handling, retry logic, and backoff.
        
        Args:
            func: Function to execute
            operation: Name of the operation (for logging and tracking)
            domain: Domain associated with the operation
            max_retries: Maximum number of retry attempts
            **error_context: Additional context for error classification
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: The last exception that occurred if all retries fail
        """
        last_exception = None
        
        for attempt in range(1, max_retries + 1):
            try:
                return func()
                
            except Exception as e:
                last_exception = e
                
                # Add attempt info to error context
                context = {
                    'operation': operation,
                    'domain': domain,
                    'attempt': attempt,
                    **error_context
                }
                
                # Classify the error
                error_info = self._classify_error(e, context)
                
                # Log the error
                if attempt < max_retries:
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} for {operation} failed: {str(e)} - "
                        f"{error_info.get('category', 'unknown')}"
                    )
                else:
                    logger.error(
                        f"All {max_retries} attempts for {operation} failed: {str(e)} - "
                        f"{error_info.get('category', 'unknown')}"
                    )
                
                # Check if we should retry
                if attempt < max_retries and self._should_retry(operation, attempt, error_info):
                    # Calculate backoff time
                    backoff_time = self._get_backoff_time(attempt)
                    
                    logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
                    
                    # For some errors, we might want to change our approach
                    for action in error_info.get('suggested_actions', []):
                        if action == 'rotate_proxy':
                            # Get a new proxy for the next attempt
                            new_proxy = self._get_proxy(domain)
                            if new_proxy:
                                session = self._get_session(domain)
                                session.proxies.update(new_proxy)
                                logger.info(f"Rotated proxy for {domain}")
                        
                        elif action == 'rotate_user_agent':
                            # Rotate the user agent
                            self._rotate_user_agent(domain)
                            logger.info(f"Rotated user agent for {domain}")
                else:
                    # Don't retry
                    break
        
        # If we reach here, all retries failed
        if last_exception:
            raise last_exception
    
    # Backward compatibility for the BaseCrawlStrategy interface
    async def get_next_urls(self, current_url=None, links=None, current_depth=None, url=None, html=None, depth=None, visited=None, extraction_result=None):
        """
        Get the next URLs to crawl based on the strategy.
        
        This method supports two different parameter sets for compatibility:
        1. Old: (current_url, links, current_depth)
        2. New: (url, html, depth, visited, extraction_result)
        
        Args:
            current_url (str, optional): The current URL being crawled (old style)
            links (List[str], optional): List of links found on the current page (old style)
            current_depth (int, optional): Current crawling depth (old style)
            url (str, optional): The URL being processed (new style)
            html (str, optional): HTML content of the page (new style)
            depth (int, optional): Current depth level (new style)
            visited (Set[str], optional): Set of visited URLs (new style)
            extraction_result (tuple, optional): Tuple of (data, confidence) from extraction (new style)
            
        Returns:
            List[Dict[str, Any]]: List of URLs with metadata for crawling
        """
        # Handle new-style parameters
        if url is not None:
            # Extract links from HTML if provided
            if html:
                # Use the HTML service to extract links instead of direct BeautifulSoup implementation
                links_list = []
                try:
                    extracted_links = self.html_service.extract_links(html, base_url=url)
                    links_list = [link['url'] for link in extracted_links 
                                 if link['url'].startswith(('http://', 'https://'))]
                except Exception as e:
                    error_context = {'url': url, 'operation': 'extract_links'}
                    classified_error = self._classify_error(e, error_context)
                    logger.error(f"Error extracting links from {url}: {str(e)} - {classified_error.get('category', 'unknown')}")
                    links_list = []
            else:
                links_list = []
            
            # Generate URLs with default scoring
            next_urls = []
            for link in links_list:
                if visited and link in visited:
                    continue
                
                # Normalize the URL using URL service
                try:
                    normalized_link = self.url_service.normalize_url(link)
                    next_urls.append({
                        'url': normalized_link,
                        'depth': (depth or 0) + 1,
                        'score': 1.0
                    })
                except Exception as e:
                    error_context = {'url': link, 'operation': 'normalize_url'}
                    classified_error = self._classify_error(e, error_context)
                    logger.warning(f"Error normalizing URL {link}: {str(e)} - {classified_error.get('category', 'unknown')}")
            
            return next_urls
            
        # Handle old-style parameters (backward compatibility)
        elif current_url is not None and links is not None:
            next_urls = []
            for link in links:
                try:
                    # Normalize the URL using URL service
                    normalized_link = self.url_service.normalize_url(link, base_url=current_url)
                    
                    if self.should_visit(normalized_link, current_url):
                        next_urls.append({
                            "url": normalized_link,
                            "depth": (current_depth or 0) + 1,
                            "score": 1.0
                        })
                except Exception as e:
                    error_context = {'url': link, 'base_url': current_url, 'operation': 'normalize_url'}
                    classified_error = self._classify_error(e, error_context)
                    logger.warning(f"Error normalizing URL {link}: {str(e)} - {classified_error.get('category', 'unknown')}")
            return next_urls
            
        # Default empty response
        return []


# Search engine registry - stores all registered search engines
_search_engine_registry = {}

def register_search_engine(engine_class):
    """
    Decorator to register a search engine class in the registry.
    
    Args:
        engine_class: The search engine class to register
        
    Returns:
        The original class (unchanged)
    """
    engine_instance = engine_class()
    engine_name = engine_instance.name
    _search_engine_registry[engine_name] = engine_class
    return engine_class

def get_registered_search_engines() -> Dict[str, Any]:
    """
    Get all registered search engines.
    
    Returns:
        Dictionary of engine name to engine class
    """
    return _search_engine_registry

def get_search_engine(engine_name: str, **kwargs) -> Optional[SearchEngineInterface]:
    """
    Get a search engine instance by name.
    
    Args:
        engine_name: Name of the search engine
        **kwargs: Additional arguments for the engine
        
    Returns:
        Search engine instance or None if not found
    """
    if engine_name in _search_engine_registry:
        return _search_engine_registry[engine_name](**kwargs)
    return None

def get_crawl_strategy(strategy_type: str, **kwargs) -> BaseStrategy:
    """
    Factory function to create a crawling strategy based on the strategy type.
    
    Args:
        strategy_type: Type of strategy to create (ai-guided, best-first, bfs, dfs)
        **kwargs: Additional arguments for the strategy
        
    Returns:
        A crawling strategy instance
    """
    from strategies.ai_guided_strategy import AIGuidedStrategy
    from strategies.best_first import BestFirstStrategy
    from strategies.bfs_strategy import BFSStrategy
    from strategies.dfs_strategy import DFSStrategy
    
    strategy_map = {
        "ai-guided": AIGuidedStrategy,
        "best-first": BestFirstStrategy,
        "bfs": BFSStrategy,
        "dfs": DFSStrategy
    }
    
    if strategy_type not in strategy_map:
        raise ValueError(f"Unknown strategy type: {strategy_type}. Available strategies: {list(strategy_map.keys())}")
    
    strategy_class = strategy_map[strategy_type]
    return strategy_class(**kwargs)

class BaseExtractionStrategy(ABC):
    """Abstract base class for extraction strategies"""
    
    def __init__(self):
        self.name = "base"
        self.description = "Base extraction strategy"
        self.priority = 100  # Higher number = lower priority
        
    @abstractmethod
    async def extract(self, url: str, html: str = None, **kwargs) -> Dict[str, Any]:
        """
        Extract content from a URL
        
        Args:
            url: The URL to extract content from
            html: Optional pre-fetched HTML content
            **kwargs: Additional parameters
            
        Returns:
            Dict containing extracted content and metadata
        """
        pass
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for this strategy"""
        return {}