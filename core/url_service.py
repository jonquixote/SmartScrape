import logging
import re
import threading
import time
from collections import deque, defaultdict
from typing import Dict, Set, List, Optional, Any, Tuple, Callable
from urllib.parse import urlparse, urljoin, parse_qs, urlencode, urlunparse
import urllib.robotparser

from core.service_interface import BaseService

logger = logging.getLogger(__name__)

class URLQueue:
    """Thread-safe URL queue with visited URL tracking."""
    
    def __init__(self):
        self._queue = deque()
        self._visited = set()
        self._in_progress = set()
        self._lock = threading.RLock()
        
    def add(self, url: str, priority: int = 0) -> bool:
        """Add a URL to the queue if not already visited."""
        with self._lock:
            if url in self._visited or url in self._in_progress:
                return False
            
            self._queue.append((url, priority))
            self._in_progress.add(url)
            return True
    
    def get(self) -> Optional[str]:
        """Get the next URL from the queue."""
        with self._lock:
            if not self._queue:
                return None
            
            url, _ = self._queue.popleft()
            return url
    
    def complete(self, url: str) -> None:
        """Mark a URL as completed (visited)."""
        with self._lock:
            if url in self._in_progress:
                self._in_progress.remove(url)
                self._visited.add(url)
    
    def is_visited(self, url: str) -> bool:
        """Check if a URL has been visited."""
        with self._lock:
            return url in self._visited
    
    def is_in_progress(self, url: str) -> bool:
        """Check if a URL is currently being processed."""
        with self._lock:
            return url in self._in_progress
    
    def clear(self) -> None:
        """Clear the queue and visited/in-progress sets."""
        with self._lock:
            self._queue.clear()
            self._visited.clear()
            self._in_progress.clear()
    
    @property
    def size(self) -> int:
        """Return the number of URLs in the queue."""
        with self._lock:
            return len(self._queue)
    
    @property
    def visited_count(self) -> int:
        """Return the number of visited URLs."""
        with self._lock:
            return len(self._visited)


class RobotsTxtChecker:
    """Handles robotstxt parsing and checking."""
    
    def __init__(self, user_agent: str = "SmartScrape"):
        self._parsers = {}
        self._cache = {}
        self._user_agent = user_agent
        self._lock = threading.RLock()
    
    def is_allowed(self, url: str) -> bool:
        """Check if the URL is allowed by robots.txt."""
        parsed_url = urlparse(url)
        
        # URLs without hostname are always allowed
        if not parsed_url.netloc:
            return True
        
        robot_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        path = parsed_url.path or "/"
        
        with self._lock:
            if robot_url in self._cache:
                # Use cached result if available and not expired
                timestamp, result = self._cache[robot_url]
                if time.time() - timestamp < 3600:  # Cache for 1 hour
                    return result
            
            try:
                if robot_url not in self._parsers:
                    parser = urllib.robotparser.RobotFileParser()
                    parser.set_url(robot_url)
                    parser.read()
                    self._parsers[robot_url] = parser
                
                result = self._parsers[robot_url].can_fetch(self._user_agent, path)
                self._cache[robot_url] = (time.time(), result)
                return result
            except Exception as e:
                logger.warning(f"Error checking robots.txt for {url}: {str(e)}")
                # If we can't check, assume it's allowed
                return True
    
    def get_crawl_delay(self, url: str) -> Optional[float]:
        """Get the crawl delay for a URL's domain if specified in robots.txt."""
        parsed_url = urlparse(url)
        
        # URLs without hostname have no delay
        if not parsed_url.netloc:
            return None
        
        robot_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        
        with self._lock:
            try:
                if robot_url not in self._parsers:
                    parser = urllib.robotparser.RobotFileParser()
                    parser.set_url(robot_url)
                    parser.read()
                    self._parsers[robot_url] = parser
                
                return self._parsers[robot_url].crawl_delay(self._user_agent)
            except Exception as e:
                logger.warning(f"Error getting crawl delay for {url}: {str(e)}")
                return None
    
    def get_sitemaps(self, url: str) -> List[str]:
        """Get sitemaps listed in the robots.txt for a URL's domain."""
        parsed_url = urlparse(url)
        
        # URLs without hostname have no sitemaps
        if not parsed_url.netloc:
            return []
        
        robot_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        
        with self._lock:
            try:
                if robot_url not in self._parsers:
                    parser = urllib.robotparser.RobotFileParser()
                    parser.set_url(robot_url)
                    parser.read()
                    self._parsers[robot_url] = parser
                
                return parser.site_maps() or []
            except Exception as e:
                logger.warning(f"Error getting sitemaps for {url}: {str(e)}")
                return []


class URLService(BaseService):
    """Service for URL operations including normalization, robots.txt, and queue management."""
    
    def __init__(self):
        self._initialized = False
        self._config = None
        self._robots_checker = None
        self._queues = {}
        self._tracking_parameters = {
            # Common tracking parameters to remove
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'fbclid', 'gclid', 'msclkid', 'zanpid', 'dclid', '_hsenc', '_hsmi',
            'hsa_cam', 'hsa_grp', 'hsa_mt', 'hsa_src', 'hsa_ad', 'hsa_acc',
            'hsa_net', 'hsa_kw', 'hsa_tgt', 'hsa_ver',
        }
        self._url_classifier = None
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the URL service."""
        if self._initialized:
            return
        
        self._config = config or {}
        user_agent = self._config.get('user_agent', 'SmartScrape')
        self._robots_checker = RobotsTxtChecker(user_agent)
        
        # Additional tracking parameters from config
        if 'tracking_parameters' in self._config:
            self._tracking_parameters.update(self._config['tracking_parameters'])
        
        self._initialized = True
        logger.info("URL service initialized")
    
    def shutdown(self) -> None:
        """Shutdown the URL service."""
        if not self._initialized:
            return
        
        # Clear all queues
        for queue in self._queues.values():
            queue.clear()
        
        self._queues.clear()
        self._initialized = False
        logger.info("URL service shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "url_service"
    
    def normalize_url(self, url: str, base_url: Optional[str] = None) -> str:
        """Normalize a URL to a canonical form."""
        if not url:
            return ""
        
        # Handle relative URLs
        if base_url and not (url.startswith(('http://', 'https://', 'mailto:', 'tel:', '//'))):
            url = urljoin(base_url, url)
        
        # Parse the URL
        parsed = urlparse(url)
        
        # If no scheme is provided, default to http
        if not parsed.scheme:
            # For URLs without scheme but with netloc (e.g. example.com)
            if not parsed.netloc and parsed.path and '/' not in parsed.path.strip('/'):
                # This is likely a domain-only URL (e.g. "example.com")
                netloc = parsed.path
                path = "/"
                url = f"http://{netloc}{path}"
                parsed = urlparse(url)
            else:
                # Just add http:// to the original URL
                url = f"http://{url}"
                parsed = urlparse(url)
        
        scheme = parsed.scheme
        
        # Normalize hostname (lowercase)
        netloc = parsed.netloc.lower()
        
        # Remove default ports
        if netloc.endswith(':80') and scheme == 'http':
            netloc = netloc[:-3]
        elif netloc.endswith(':443') and scheme == 'https':
            netloc = netloc[:-4]
        
        # Normalize path (handle /./ and /../)
        path = parsed.path
        if not path:
            path = "/"
        
        # Parse query parameters and remove tracking params
        query_params = parse_qs(parsed.query)
        
        # Remove tracking parameters
        for param in list(query_params.keys()):
            if param.lower() in self._tracking_parameters:
                del query_params[param]
        
        # Rebuild query string, sorted for consistency
        query = urlencode(query_params, doseq=True) if query_params else ""
        
        # Remove fragment unless specified to keep it
        fragment = ""
        if self._config.get('keep_fragments', False):
            fragment = parsed.fragment
        
        # Reconstruct the URL
        normalized = urlunparse((scheme, netloc, path, parsed.params, query, fragment))
        
        return normalized
    
    def is_allowed(self, url: str) -> bool:
        """Check if crawling this URL is allowed by robots.txt."""
        return self._robots_checker.is_allowed(url)
    
    def get_crawl_delay(self, url: str) -> Optional[float]:
        """Get the crawl delay for a domain if specified in robots.txt."""
        return self._robots_checker.get_crawl_delay(url)
    
    def get_sitemaps(self, url: str) -> List[str]:
        """Get sitemaps listed in the robots.txt for a domain."""
        return self._robots_checker.get_sitemaps(url)
    
    def get_queue(self, name: str = "default") -> URLQueue:
        """Get a URL queue by name, creating it if it doesn't exist."""
        if name not in self._queues:
            self._queues[name] = URLQueue()
        return self._queues[name]
    
    def classify_url(self, url: str) -> Dict[str, Any]:
        """Classify a URL by type, section, etc."""
        parsed = urlparse(url)
        result = {
            'is_resource': False,
            'is_navigation': False,
            'depth': len(parsed.path.strip('/').split('/')),
            'has_parameters': bool(parsed.query),
            'parameter_count': len(parse_qs(parsed.query)),
            'path_type': 'unknown',
        }
        
        # Check if it's a resource URL
        resource_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.pdf', '.zip', '.doc', '.xls'}
        path_lower = parsed.path.lower()
        result['is_resource'] = any(path_lower.endswith(ext) for ext in resource_extensions)
        
        # Check if it's a likely navigation URL
        nav_patterns = [
            r'/category/', r'/categories/', r'/departments/', r'/catalog/',
            r'/index\.', r'/search', r'/browse', r'/products/', r'/collection/'
        ]
        result['is_navigation'] = any(re.search(pattern, parsed.path, re.IGNORECASE) for pattern in nav_patterns)
        
        # Identify the URL path type
        if re.search(r'/product[s]?/', path_lower) or re.search(r'/item[s]?/', path_lower):
            result['path_type'] = 'product'
        elif re.search(r'/category/', path_lower) or re.search(r'/categories/', path_lower):
            result['path_type'] = 'category'
        elif re.search(r'/cart/', path_lower) or re.search(r'/checkout/', path_lower):
            result['path_type'] = 'cart'
        elif re.search(r'/search/', path_lower) or 'q=' in parsed.query or 'query=' in parsed.query:
            result['path_type'] = 'search'
        elif re.search(r'/blog/', path_lower) or re.search(r'/news/', path_lower) or re.search(r'/article[s]?/', path_lower):
            result['path_type'] = 'content'
        elif re.search(r'/account/', path_lower) or re.search(r'/login/', path_lower) or re.search(r'/register/', path_lower):
            result['path_type'] = 'account'
        
        return result