import logging
import time
import threading
import json
import os
import random
import re
import socket
import requests
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from urllib.parse import urlparse
import ipaddress

from core.service_interface import BaseService

logger = logging.getLogger(__name__)

class ProxyType(Enum):
    """Types of proxies supported by the system."""
    HTTP = "http"
    HTTPS = "https"
    SOCKS4 = "socks4"
    SOCKS5 = "socks5"


class ProxyAnonymity(Enum):
    """Anonymity levels of proxies."""
    TRANSPARENT = "transparent"    # Remote server knows your IP and knows it's a proxy
    ANONYMOUS = "anonymous"        # Remote server knows it's a proxy but doesn't know your IP
    ELITE = "elite"                # Remote server doesn't know your IP or that it's a proxy
    UNKNOWN = "unknown"            # Anonymity level not verified


class ProxyStatus(Enum):
    """Status of a proxy in the system."""
    ACTIVE = "active"              # Proxy is available for use
    TESTING = "testing"            # Proxy is currently being tested
    BLACKLISTED = "blacklisted"    # Proxy is temporarily not available
    REMOVED = "removed"            # Proxy is permanently removed


class Proxy:
    """Represents a proxy with its configuration and metadata."""
    
    def __init__(self, address: str, port: int, 
                proxy_type: Union[ProxyType, str] = ProxyType.HTTP,
                username: Optional[str] = None, 
                password: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a proxy instance.
        
        Args:
            address: IP address or hostname of the proxy
            port: Port number
            proxy_type: Type of proxy (HTTP, HTTPS, SOCKS4, SOCKS5)
            username: Optional username for authentication
            password: Optional password for authentication
            metadata: Additional metadata about the proxy
        """
        self.address = address
        self.port = port
        
        # Convert string to enum if needed
        if isinstance(proxy_type, str):
            try:
                self.proxy_type = ProxyType(proxy_type.lower())
            except ValueError:
                self.proxy_type = ProxyType.HTTP
        else:
            self.proxy_type = proxy_type
            
        self.username = username
        self.password = password
        self.metadata = metadata or {}
        
        # Status tracking
        self.status = ProxyStatus.ACTIVE
        self.anonymity = ProxyAnonymity.UNKNOWN
        self.blacklisted_until = None
        self.blacklist_reason = None
        
        # Performance metrics
        self.success_count = 0
        self.failure_count = 0
        self.total_response_time = 0
        self.last_used = None
        self.last_checked = None
        self.last_success = None
        self.consecutive_failures = 0
        self.error_types = {}  # Error type -> count
        
        # Geographic information
        self.country = self.metadata.get('country')
        self.city = self.metadata.get('city')
        self.isp = self.metadata.get('isp')
        
    @property
    def url(self) -> str:
        """Get the proxy URL in the format 'type://address:port'."""
        return f"{self.proxy_type.value}://{self.address}:{self.port}"
    
    @property
    def auth_url(self) -> str:
        """Get the proxy URL with authentication if available."""
        if self.username and self.password:
            return f"{self.proxy_type.value}://{self.username}:{self.password}@{self.address}:{self.port}"
        return self.url
    
    @property
    def is_active(self) -> bool:
        """Check if the proxy is active and not blacklisted."""
        if self.status != ProxyStatus.ACTIVE:
            return False
        if self.blacklisted_until and datetime.now() < self.blacklisted_until:
            return False
        return True
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the proxy."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    @property
    def average_response_time(self) -> float:
        """Calculate the average response time of the proxy."""
        if self.success_count == 0:
            return float('inf')
        return self.total_response_time / self.success_count
    
    @property
    def last_used_delta(self) -> float:
        """Get the time in seconds since the proxy was last used."""
        if not self.last_used:
            return float('inf')
        return (datetime.now() - self.last_used).total_seconds()
    
    def get_dict_for_requests(self) -> Dict[str, str]:
        """Get the proxy configuration in the format expected by the requests library."""
        proxy_dict = {}
        
        # Format depends on proxy type
        if self.proxy_type in (ProxyType.HTTP, ProxyType.HTTPS):
            protocol = "http" if self.proxy_type == ProxyType.HTTP else "https"
            if self.username and self.password:
                proxy_dict[protocol] = f"{protocol}://{self.username}:{self.password}@{self.address}:{self.port}"
            else:
                proxy_dict[protocol] = f"{protocol}://{self.address}:{self.port}"
        else:  # SOCKS proxies
            # Both HTTP and HTTPS traffic go through the SOCKS proxy
            socks_url = f"{self.proxy_type.value}://{self.address}:{self.port}"
            if self.username and self.password:
                socks_url = f"{self.proxy_type.value}://{self.username}:{self.password}@{self.address}:{self.port}"
            proxy_dict["http"] = socks_url
            proxy_dict["https"] = socks_url
            
        return proxy_dict
    
    def record_success(self, response_time: float = 0.0) -> None:
        """Record a successful use of the proxy."""
        self.success_count += 1
        self.last_used = datetime.now()
        self.last_success = self.last_used
        self.total_response_time += response_time
        self.consecutive_failures = 0
    
    def record_failure(self, error_type: str = None) -> None:
        """Record a failed use of the proxy."""
        self.failure_count += 1
        self.last_used = datetime.now()
        self.consecutive_failures += 1
        
        if error_type:
            self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
    
    def blacklist(self, reason: str, duration: Optional[timedelta] = None) -> None:
        """Blacklist the proxy for a specified duration."""
        self.status = ProxyStatus.BLACKLISTED
        self.blacklist_reason = reason
        
        if duration:
            self.blacklisted_until = datetime.now() + duration
        else:
            # Default blacklist for 1 hour
            self.blacklisted_until = datetime.now() + timedelta(hours=1)
    
    def unblacklist(self) -> None:
        """Remove the proxy from blacklist."""
        if self.status == ProxyStatus.BLACKLISTED:
            self.status = ProxyStatus.ACTIVE
            self.blacklisted_until = None
            self.blacklist_reason = None
    
    def __eq__(self, other: object) -> bool:
        """Compare two proxies for equality."""
        if not isinstance(other, Proxy):
            return False
        return (self.address == other.address and 
                self.port == other.port and 
                self.proxy_type == other.proxy_type)
    
    def __hash__(self) -> int:
        """Generate a hash for the proxy."""
        return hash((self.address, self.port, self.proxy_type))
    
    def __str__(self) -> str:
        """Get a string representation of the proxy."""
        return f"Proxy({self.url}, active={self.is_active}, success_rate={self.success_rate:.2f})"


class ProxyProvider:
    """Base class for proxy providers."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the proxy provider."""
        self.name = name
        self.config = config or {}
        
    def get_proxies(self) -> List[Proxy]:
        """Retrieve proxies from the provider."""
        raise NotImplementedError("Proxy providers must implement get_proxies()")
    
    def validate_proxy_format(self, proxy_str: str) -> bool:
        """Validate that a proxy string has the correct format."""
        # Look for patterns like:
        # 192.168.1.1:8080
        # http://192.168.1.1:8080
        # username:password@192.168.1.1:8080
        # http://username:password@192.168.1.1:8080
        
        # Strip protocol if present
        if "://" in proxy_str:
            proxy_str = proxy_str.split("://", 1)[1]
            
        # Strip authentication if present
        if "@" in proxy_str:
            proxy_str = proxy_str.split("@", 1)[1]
            
        # Check format of IP:PORT
        try:
            ip, port = proxy_str.rsplit(":", 1)
            port = int(port)
            if not (1 <= port <= 65535):
                return False
                
            # Simple IP validation
            if not re.match(r'^[\w.-]+$', ip):
                return False
                
            return True
        except (ValueError, IndexError):
            return False
    
    def parse_proxy_string(self, proxy_str: str) -> Optional[Proxy]:
        """Parse a proxy string into a Proxy object."""
        if not self.validate_proxy_format(proxy_str):
            logger.warning(f"Invalid proxy format: {proxy_str}")
            return None
            
        try:
            proxy_type = ProxyType.HTTP  # Default
            username = None
            password = None
            
            # Extract protocol if present
            if "://" in proxy_str:
                protocol, proxy_str = proxy_str.split("://", 1)
                try:
                    proxy_type = ProxyType(protocol.lower())
                except ValueError:
                    proxy_type = ProxyType.HTTP
            
            # Extract authentication if present
            if "@" in proxy_str:
                auth, proxy_str = proxy_str.split("@", 1)
                if ":" in auth:
                    username, password = auth.split(":", 1)
            
            # Extract address and port
            address, port_str = proxy_str.rsplit(":", 1)
            port = int(port_str)
            
            return Proxy(
                address=address,
                port=port,
                proxy_type=proxy_type,
                username=username,
                password=password
            )
        except Exception as e:
            logger.warning(f"Error parsing proxy string {proxy_str}: {str(e)}")
            return None


class StaticProxyProvider(ProxyProvider):
    """Provider that returns a static list of proxies."""
    
    def get_proxies(self) -> List[Proxy]:
        """Return proxies from static configuration."""
        proxies = []
        
        # Process proxy list from config
        proxy_list = self.config.get('proxies', [])
        for proxy_data in proxy_list:
            # Handle string format (e.g., "http://1.2.3.4:8080")
            if isinstance(proxy_data, str):
                proxy = self.parse_proxy_string(proxy_data)
                if proxy:
                    proxies.append(proxy)
                    
            # Handle dictionary format
            elif isinstance(proxy_data, dict):
                try:
                    proxy = Proxy(
                        address=proxy_data.get('address'),
                        port=proxy_data.get('port'),
                        proxy_type=proxy_data.get('type', ProxyType.HTTP),
                        username=proxy_data.get('username'),
                        password=proxy_data.get('password'),
                        metadata=proxy_data.get('metadata', {})
                    )
                    proxies.append(proxy)
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Error creating proxy from {proxy_data}: {str(e)}")
        
        return proxies


class FileProxyProvider(ProxyProvider):
    """Provider that loads proxies from a file."""
    
    def get_proxies(self) -> List[Proxy]:
        """Load proxies from a file."""
        proxies = []
        
        # Get file path from config
        file_path = self.config.get('file_path')
        if not file_path:
            logger.warning("No file path specified for FileProxyProvider")
            return proxies
            
        if not os.path.exists(file_path):
            logger.warning(f"Proxy file does not exist: {file_path}")
            return proxies
            
        try:
            with open(file_path, 'r') as f:
                # Detect file format based on extension
                if file_path.endswith('.json'):
                    # JSON format
                    proxy_data = json.load(f)
                    if isinstance(proxy_data, list):
                        for item in proxy_data:
                            if isinstance(item, str):
                                proxy = self.parse_proxy_string(item)
                                if proxy:
                                    proxies.append(proxy)
                            elif isinstance(item, dict):
                                try:
                                    proxy = Proxy(
                                        address=item.get('address'),
                                        port=item.get('port'),
                                        proxy_type=item.get('type', ProxyType.HTTP),
                                        username=item.get('username'),
                                        password=item.get('password'),
                                        metadata=item.get('metadata', {})
                                    )
                                    proxies.append(proxy)
                                except Exception as e:
                                    logger.warning(f"Error creating proxy from {item}: {str(e)}")
                else:
                    # Text format (one proxy per line)
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            proxy = self.parse_proxy_string(line)
                            if proxy:
                                proxies.append(proxy)
        except Exception as e:
            logger.error(f"Error loading proxies from file {file_path}: {str(e)}")
            
        return proxies


class APIProxyProvider(ProxyProvider):
    """Provider that fetches proxies from an API."""
    
    def get_proxies(self) -> List[Proxy]:
        """Fetch proxies from an API endpoint."""
        proxies = []
        
        # Get API configuration
        api_url = self.config.get('api_url')
        if not api_url:
            logger.warning("No API URL specified for APIProxyProvider")
            return proxies
            
        headers = self.config.get('headers', {})
        params = self.config.get('params', {})
        auth = self.config.get('auth')
        
        # Authentication handling
        auth_tuple = None
        if auth and 'username' in auth and 'password' in auth:
            auth_tuple = (auth.get('username'), auth.get('password'))
        
        try:
            # Fetch proxies from API
            response = requests.get(
                api_url,
                headers=headers,
                params=params,
                auth=auth_tuple,
                timeout=self.config.get('timeout', 30)
            )
            
            if response.status_code != 200:
                logger.warning(f"API returned status code {response.status_code}: {response.text}")
                return proxies
                
            try:
                data = response.json()
            except ValueError:
                # If not JSON, try processing as text
                data = response.text.splitlines()
                
            # Process the response based on format specified in config
            format_type = self.config.get('format', 'json')
            
            if format_type == 'json':
                # Handle JSON response
                json_path = self.config.get('json_path', [])
                
                # Extract the proxy list using the JSON path
                current_data = data
                for key in json_path:
                    if key in current_data:
                        current_data = current_data[key]
                    else:
                        logger.warning(f"JSON path key '{key}' not found in API response")
                        current_data = []
                        break
                
                # Ensure we have a list at this point
                if not isinstance(current_data, list):
                    logger.warning(f"Expected list at JSON path, got {type(current_data)}")
                    return proxies
                    
                # Process each proxy in the list
                mapping = self.config.get('mapping', {})
                for item in current_data:
                    if isinstance(item, str):
                        # String format
                        proxy = self.parse_proxy_string(item)
                        if proxy:
                            proxies.append(proxy)
                    elif isinstance(item, dict):
                        # Dictionary format with mapping
                        try:
                            # Apply mapping to extract fields
                            address = item.get(mapping.get('address', 'address'))
                            port = int(item.get(mapping.get('port', 'port')))
                            proxy_type = item.get(mapping.get('type', 'type'), ProxyType.HTTP)
                            username = item.get(mapping.get('username', 'username'))
                            password = item.get(mapping.get('password', 'password'))
                            
                            # Extract metadata based on mapping
                            metadata = {}
                            meta_mapping = mapping.get('metadata', {})
                            for meta_key, source_key in meta_mapping.items():
                                if source_key in item:
                                    metadata[meta_key] = item[source_key]
                            
                            proxy = Proxy(
                                address=address,
                                port=port,
                                proxy_type=proxy_type,
                                username=username,
                                password=password,
                                metadata=metadata
                            )
                            proxies.append(proxy)
                        except (KeyError, TypeError, ValueError) as e:
                            logger.warning(f"Error creating proxy from API data: {str(e)}")
            else:
                # Handle text response (one proxy per line)
                for line in data:
                    if isinstance(line, str) and line.strip() and not line.strip().startswith('#'):
                        proxy = self.parse_proxy_string(line.strip())
                        if proxy:
                            proxies.append(proxy)
                
        except Exception as e:
            logger.error(f"Error fetching proxies from API {api_url}: {str(e)}")
            
        return proxies


class DatabaseProxyProvider(ProxyProvider):
    """Provider that loads proxies from a database."""
    
    def get_proxies(self) -> List[Proxy]:
        """Load proxies from a database."""
        # This is a placeholder - implementation would depend on the database system used
        logger.warning("DatabaseProxyProvider is not implemented")
        return []


class ProxyRotationStrategy:
    """Base class for proxy rotation strategies."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the rotation strategy."""
        self.name = name
        self.config = config or {}
        
    def select_proxy(self, proxies: List[Proxy], context: Dict[str, Any] = None) -> Optional[Proxy]:
        """Select a proxy from the list based on the strategy."""
        raise NotImplementedError("Proxy rotation strategies must implement select_proxy()")


class RoundRobinStrategy(ProxyRotationStrategy):
    """Strategy that selects proxies in a round-robin fashion."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.current_index = 0
        self.lock = threading.RLock()
        
    def select_proxy(self, proxies: List[Proxy], context: Dict[str, Any] = None) -> Optional[Proxy]:
        """Select the next proxy in the list."""
        if not proxies:
            return None
            
        with self.lock:
            # Filter to only active proxies
            active_proxies = [p for p in proxies if p.is_active]
            if not active_proxies:
                return None
                
            # Select the next proxy
            self.current_index = (self.current_index + 1) % len(active_proxies)
            return active_proxies[self.current_index - 1]


class RandomStrategy(ProxyRotationStrategy):
    """Strategy that selects proxies randomly."""
    
    def select_proxy(self, proxies: List[Proxy], context: Dict[str, Any] = None) -> Optional[Proxy]:
        """Select a random proxy from the list."""
        # Filter to only active proxies
        active_proxies = [p for p in proxies if p.is_active]
        if not active_proxies:
            return None
            
        return random.choice(active_proxies)


class WeightedStrategy(ProxyRotationStrategy):
    """Strategy that selects proxies based on their success rate."""
    
    def select_proxy(self, proxies: List[Proxy], context: Dict[str, Any] = None) -> Optional[Proxy]:
        """Select a proxy weighted by success rate."""
        # Filter to only active proxies
        active_proxies = [p for p in proxies if p.is_active]
        if not active_proxies:
            return None
            
        # Calculate weights based on success rate and response time
        weights = []
        for proxy in active_proxies:
            # Use success rate as the base weight
            weight = max(0.1, proxy.success_rate)
            
            # Penalize high response times
            if proxy.average_response_time < float('inf'):
                # Convert to seconds if in milliseconds
                avg_time = proxy.average_response_time
                if avg_time > 10:  # Assuming values > 10 are in milliseconds
                    avg_time /= 1000
                
                # Response time penalty factor (1.0 for 0s, 0.5 for 2s, etc.)
                time_factor = max(0.1, 1.0 - (avg_time / 10.0))
                weight *= time_factor
            
            weights.append(weight)
            
        # If all weights are zero, use equal weights
        if sum(weights) <= 0:
            return random.choice(active_proxies)
            
        # Select a proxy based on weights
        return random.choices(active_proxies, weights=weights, k=1)[0]


class LeastUsedStrategy(ProxyRotationStrategy):
    """Strategy that selects the least recently used proxy."""
    
    def select_proxy(self, proxies: List[Proxy], context: Dict[str, Any] = None) -> Optional[Proxy]:
        """Select the proxy that was used least recently."""
        # Filter to only active proxies
        active_proxies = [p for p in proxies if p.is_active]
        if not active_proxies:
            return None
            
        # Find the proxy with the oldest last_used timestamp
        return max(active_proxies, key=lambda p: p.last_used_delta if p.last_used else float('inf'))


class GeoMatchStrategy(ProxyRotationStrategy):
    """Strategy that selects proxies based on geographic criteria."""
    
    def select_proxy(self, proxies: List[Proxy], context: Dict[str, Any] = None) -> Optional[Proxy]:
        """Select a proxy matching geographic criteria."""
        # Filter to only active proxies
        active_proxies = [p for p in proxies if p.is_active]
        if not active_proxies:
            return None
            
        context = context or {}
        target_country = context.get('country')
        
        if target_country:
            # Try to find proxies matching the target country
            matching_proxies = [p for p in active_proxies if p.country and p.country.lower() == target_country.lower()]
            
            if matching_proxies:
                # If we found matching proxies, select one randomly
                return random.choice(matching_proxies)
        
        # Fallback to random selection
        return random.choice(active_proxies)


class AdaptiveStrategy(ProxyRotationStrategy):
    """Strategy that adapts proxy selection based on domain-specific performance."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.domain_performance = {}  # domain -> {proxy_url -> success_rate}
        self.lock = threading.RLock()
        
    def select_proxy(self, proxies: List[Proxy], context: Dict[str, Any] = None) -> Optional[Proxy]:
        """Select a proxy based on its performance for the target domain."""
        # Filter to only active proxies
        active_proxies = [p for p in proxies if p.is_active]
        if not active_proxies:
            return None
            
        context = context or {}
        domain = context.get('domain')
        
        if not domain:
            # If no domain specified, fall back to random selection
            return random.choice(active_proxies)
            
        with self.lock:
            # Get domain-specific performance data
            domain_data = self.domain_performance.get(domain, {})
            
            if not domain_data:
                # No data for this domain yet, use random proxy
                selected = random.choice(active_proxies)
                # Initialize domain data for this proxy
                self.domain_performance.setdefault(domain, {})[selected.url] = 0.5  # Neutral initial value
                return selected
                
            # Calculate selection weights based on domain-specific performance
            weights = []
            for proxy in active_proxies:
                # Get domain-specific success rate, defaulting to overall success rate
                weight = domain_data.get(proxy.url, proxy.success_rate)
                # Ensure minimum weight for exploration
                weight = max(0.1, weight)
                weights.append(weight)
                
            # Select a proxy based on weights
            return random.choices(active_proxies, weights=weights, k=1)[0]
    
    def update_performance(self, proxy: Proxy, domain: str, success: bool) -> None:
        """Update the performance data for a proxy and domain."""
        with self.lock:
            # Initialize domain data if needed
            if domain not in self.domain_performance:
                self.domain_performance[domain] = {}
                
            # Get current success rate
            current_rate = self.domain_performance[domain].get(proxy.url, 0.5)
            
            # Update with exponential moving average
            alpha = 0.2  # Learning rate
            new_rate = current_rate + alpha * (1.0 if success else 0.0 - current_rate)
            
            # Store updated rate
            self.domain_performance[domain][proxy.url] = new_rate


class ProxyIterator:
    """
    Iterator for cycling through proxies based on a specific strategy.
    This allows users to systematically iterate through proxies when needed.
    """
    
    def __init__(self, proxy_manager, strategy: Optional[str] = None,
                 domain: Optional[str] = None, proxy_type: Optional[str] = None,
                 country: Optional[str] = None, max_attempts: int = None):
        """
        Initialize a proxy iterator.
        
        Args:
            proxy_manager: Reference to the ProxyManager
            strategy: Strategy to use for proxy selection
            domain: Target domain for the requests
            proxy_type: Type of proxy needed
            country: Preferred country for the proxy
            max_attempts: Maximum number of proxies to try (None for unlimited)
        """
        self.proxy_manager = proxy_manager
        self.strategy = strategy
        self.domain = domain
        self.proxy_type = proxy_type
        self.country = country
        self.max_attempts = max_attempts
        self.attempts = 0
        self.used_proxies = set()
    
    def __iter__(self):
        """Return self as an iterator."""
        return self
    
    def __next__(self):
        """Get the next proxy in the iteration."""
        if self.max_attempts is not None and self.attempts >= self.max_attempts:
            raise StopIteration
            
        # Get a proxy using the specified criteria
        proxy = self.proxy_manager.get_proxy(
            domain=self.domain,
            proxy_type=self.proxy_type,
            country=self.country,
            strategy=self.strategy
        )
        
        # If no proxy is available, stop iteration
        if not proxy:
            raise StopIteration
            
        # Keep track of which proxies we've seen to avoid infinite loops
        proxy_key = (proxy.address, proxy.port, proxy.proxy_type)
        
        # If we've seen all available proxies, stop iteration
        if proxy_key in self.used_proxies:
            # Check if we've used all available proxies
            with self.proxy_manager._lock:
                active_proxies = [p for p in self.proxy_manager._proxies 
                                 if p.is_active and 
                                 (self.proxy_type is None or p.proxy_type.value == self.proxy_type) and
                                 (self.country is None or (p.country and p.country.lower() == self.country.lower()))]
                
                if len(self.used_proxies) >= len(active_proxies):
                    raise StopIteration
        
        # Add to used proxies
        self.used_proxies.add(proxy_key)
        self.attempts += 1
        
        return proxy
    
    def register_result(self, proxy: Proxy, success: bool, response: Any = None, 
                        error_type: Optional[str] = None) -> None:
        """
        Register the result of using a proxy.
        
        Args:
            proxy: The proxy that was used
            success: Whether the request was successful
            response: Optional response object
            error_type: Type of error if request failed
        """
        if proxy:
            self.proxy_manager.register_proxy_result(
                proxy=proxy,
                success=success,
                response=response,
                domain=self.domain,
                error_type=error_type
            )
            
            # If the proxy failed, remove it from our usable set to avoid reusing it
            if not success:
                proxy_key = (proxy.address, proxy.port, proxy.proxy_type)
                if proxy_key in self.used_proxies:
                    self.used_proxies.remove(proxy_key)


class ProxyManager(BaseService):
    """
    Manage and rotate proxies for web scraping.
    
    Features:
    - Load proxies from multiple sources
    - Verify and monitor proxy health
    - Select best proxy based on various strategies
    - Track proxy performance and apply auto-blacklisting
    - Rotate user agents to improve stealth
    """
    
    def __init__(self):
        """Initialize the proxy manager."""
        self._initialized = False
        self._config = None
        self._proxies = []
        self._providers = {}
        self._strategies = {}
        self._current_strategy = None
        self._lock = threading.RLock()
        self._blacklist_threshold = 5  # Consecutive failures before blacklisting
        self._check_thread = None
        self._shutdown_event = threading.Event()
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the proxy manager with configuration.
        
        Args:
            config: Configuration dictionary with proxy sources and settings
        """
        if self._initialized:
            return
            
        self._config = config or {}
        
        # Load configuration settings
        self._blacklist_threshold = self._config.get('max_failures', 5)
        self._check_interval = self._config.get('check_interval', 300)  # 5 minutes
        
        # Create proxy providers
        self._setup_providers()
        
        # Create rotation strategies
        self._setup_strategies()
        
        # Set default strategy
        default_strategy = self._config.get('default_strategy', 'round_robin')
        self._current_strategy = self._strategies.get(default_strategy)
        if not self._current_strategy:
            self._current_strategy = RoundRobinStrategy('round_robin')
            self._strategies['round_robin'] = self._current_strategy
        
        # Load initial proxies
        self.refresh_proxies()
        
        # Start health check thread if enabled
        if self._config.get('enable_health_checks', True):
            self._start_health_check_thread()
        
        self._initialized = True
        logger.info(f"Proxy manager initialized with {len(self._proxies)} proxies")
        
    def shutdown(self) -> None:
        """Shutdown the proxy manager and clean up resources."""
        if not self._initialized:
            return
            
        # Stop health check thread
        if self._check_thread:
            self._shutdown_event.set()
            self._check_thread.join(timeout=5.0)
            self._check_thread = None
            
        # Clear proxy list
        with self._lock:
            self._proxies.clear()
            self._providers.clear()
            self._strategies.clear()
        
        self._initialized = False
        logger.info("Proxy manager shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "proxy_manager"
    
    def _setup_providers(self) -> None:
        """Initialize proxy providers from configuration."""
        provider_configs = self._config.get('providers', {})
        
        for name, config in provider_configs.items():
            provider_type = config.get('type', 'static')
            
            try:
                if provider_type == 'static':
                    self._providers[name] = StaticProxyProvider(name, config)
                elif provider_type == 'file':
                    self._providers[name] = FileProxyProvider(name, config)
                elif provider_type == 'api':
                    self._providers[name] = APIProxyProvider(name, config)
                elif provider_type == 'database':
                    self._providers[name] = DatabaseProxyProvider(name, config)
                else:
                    logger.warning(f"Unknown provider type: {provider_type}")
            except Exception as e:
                logger.error(f"Error creating provider {name}: {str(e)}")
    
    def _setup_strategies(self) -> None:
        """Initialize proxy rotation strategies from configuration."""
        strategy_configs = self._config.get('strategies', {})
        
        # Create default strategies
        self._strategies['round_robin'] = RoundRobinStrategy('round_robin')
        self._strategies['random'] = RandomStrategy('random')
        self._strategies['weighted'] = WeightedStrategy('weighted')
        self._strategies['least_used'] = LeastUsedStrategy('least_used')
        self._strategies['geo_match'] = GeoMatchStrategy('geo_match')
        self._strategies['adaptive'] = AdaptiveStrategy('adaptive')
        
        # Configure strategies from config
        for name, config in strategy_configs.items():
            strategy_type = config.get('type', 'round_robin')
            
            try:
                if strategy_type == 'round_robin':
                    self._strategies[name] = RoundRobinStrategy(name, config)
                elif strategy_type == 'random':
                    self._strategies[name] = RandomStrategy(name, config)
                elif strategy_type == 'weighted':
                    self._strategies[name] = WeightedStrategy(name, config)
                elif strategy_type == 'least_used':
                    self._strategies[name] = LeastUsedStrategy(name, config)
                elif strategy_type == 'geo_match':
                    self._strategies[name] = GeoMatchStrategy(name, config)
                elif strategy_type == 'adaptive':
                    self._strategies[name] = AdaptiveStrategy(name, config)
                else:
                    logger.warning(f"Unknown strategy type: {strategy_type}")
            except Exception as e:
                logger.error(f"Error creating strategy {name}: {str(e)}")
    
    def refresh_proxies(self) -> None:
        """Reload proxies from all providers."""
        with self._lock:
            # Save existing proxies' performance data
            existing_data = {(p.address, p.port, p.proxy_type): p for p in self._proxies}
            
            # Clear current proxy list
            self._proxies.clear()
            
            # Load proxies from all providers
            for name, provider in self._providers.items():
                try:
                    new_proxies = provider.get_proxies()
                    
                    for proxy in new_proxies:
                        # Check if we already have performance data for this proxy
                        key = (proxy.address, proxy.port, proxy.proxy_type)
                        if key in existing_data:
                            # Transfer performance data
                            existing = existing_data[key]
                            proxy.success_count = existing.success_count
                            proxy.failure_count = existing.failure_count
                            proxy.total_response_time = existing.total_response_time
                            proxy.last_used = existing.last_used
                            proxy.last_checked = existing.last_checked
                            proxy.last_success = existing.last_success
                            proxy.consecutive_failures = existing.consecutive_failures
                            proxy.error_types = existing.error_types.copy()
                            
                            # Preserve status unless the existing proxy was blacklisted due to health issues
                            if existing.status == ProxyStatus.BLACKLISTED and existing.blacklist_reason == "health_check_failure":
                                proxy.status = existing.status
                                proxy.blacklisted_until = existing.blacklisted_until
                                proxy.blacklist_reason = existing.blacklist_reason
                        
                        # Add to the proxy list if not already present
                        if proxy not in self._proxies:
                            self._proxies.append(proxy)
                except Exception as e:
                    logger.error(f"Error loading proxies from provider {name}: {str(e)}")
            
            # Log appropriate message based on whether providers were configured
            if self._providers:
                logger.info(f"Refreshed proxies from all providers, now have {len(self._proxies)} proxies")
            else:
                logger.debug("No proxy providers configured - running without proxies")
    
    def _start_health_check_thread(self) -> None:
        """Start a background thread to periodically check proxy health."""
        if self._check_thread:
            return
            
        def health_check_worker():
            """Worker function for the health check thread."""
            logger.info("Proxy health check thread started")
            
            while not self._shutdown_event.is_set():
                try:
                    # Perform health checks
                    self._check_all_proxies()
                    
                    # Wait for the next check interval or until shutdown
                    self._shutdown_event.wait(self._check_interval)
                except Exception as e:
                    logger.error(f"Error in proxy health check thread: {str(e)}")
                    # Wait a bit before retrying to avoid busy loop on error
                    self._shutdown_event.wait(60)
            
            logger.info("Proxy health check thread stopped")
        
        # Start the thread
        self._check_thread = threading.Thread(
            target=health_check_worker,
            name="ProxyHealthCheck",
            daemon=True
        )
        self._check_thread.start()
    
    def _check_all_proxies(self) -> None:
        """Perform health checks on all proxies."""
        logger.debug("Starting health check for all proxies")
        
        with self._lock:
            # Get a copy of the proxy list to avoid modification during iteration
            proxies = self._proxies.copy()
            
        # Check proxies in parallel if enabled
        max_threads = self._config.get('max_health_check_threads', 5)
        if max_threads > 1:
            # Create a thread pool
            threads = []
            checked_count = 0
            
            def check_proxy_thread(proxy):
                """Thread worker for checking a proxy."""
                result = self.check_proxy_health(proxy)
                if not result:
                    # Blacklist the proxy if health check failed
                    self.blacklist_proxy(proxy, "health_check_failure", timedelta(hours=1))
            
            # Start threads up to max_threads
            for proxy in proxies:
                # Skip proxies that were recently checked or are blacklisted
                if proxy.status == ProxyStatus.BLACKLISTED or proxy.status == ProxyStatus.REMOVED:
                    continue
                    
                if proxy.last_checked and (datetime.now() - proxy.last_checked).total_seconds() < self._check_interval:
                    continue
                    
                # Set proxy to testing state
                proxy.status = ProxyStatus.TESTING
                
                # Start a thread for this proxy
                thread = threading.Thread(
                    target=check_proxy_thread,
                    args=(proxy,),
                    daemon=True
                )
                thread.start()
                threads.append(thread)
                checked_count += 1
                
                # Wait for threads to finish if we've hit the limit
                if len(threads) >= max_threads:
                    for t in threads:
                        t.join()
                    threads = []
            
            # Wait for any remaining threads
            for t in threads:
                t.join()
                
            logger.info(f"Completed health checks for {checked_count} proxies")
        else:
            # Check proxies sequentially
            checked_count = 0
            for proxy in proxies:
                # Skip proxies that were recently checked or are blacklisted
                if proxy.status == ProxyStatus.BLACKLISTED or proxy.status == ProxyStatus.REMOVED:
                    continue
                    
                if proxy.last_checked and (datetime.now() - proxy.last_checked).total_seconds() < self._check_interval:
                    continue
                    
                # Set proxy to testing state
                proxy.status = ProxyStatus.TESTING
                
                # Check the proxy
                result = self.check_proxy_health(proxy)
                if not result:
                    # Blacklist the proxy if health check failed
                    self.blacklist_proxy(proxy, "health_check_failure", timedelta(hours=1))
                
                checked_count += 1
                
            logger.info(f"Completed health checks for {checked_count} proxies")
    
    def get_proxy(self, domain: Optional[str] = None, proxy_type: Optional[str] = None, 
                 country: Optional[str] = None, strategy: Optional[str] = None) -> Optional[Proxy]:
        """
        Get an appropriate proxy based on the request context.
        
        Args:
            domain: Target domain for the request
            proxy_type: Type of proxy needed (HTTP, HTTPS, SOCKS4, SOCKS5)
            country: Preferred country for the proxy
            strategy: Proxy selection strategy to use
            
        Returns:
            A Proxy object or None if no suitable proxy is available
        """
        if not self._initialized:
            raise RuntimeError("ProxyManager is not initialized")
            
        with self._lock:
            # Filter proxies based on criteria
            candidates = self._filter_proxies(proxy_type, country)
            
            if not candidates:
                logger.warning(f"No suitable proxies found for {domain} (type={proxy_type}, country={country})")
                return None
                
            # Select proxy using the specified strategy
            context = {
                'domain': domain,
                'country': country
            }
            
            if strategy and strategy in self._strategies:
                selected = self._strategies[strategy].select_proxy(candidates, context)
            else:
                selected = self._current_strategy.select_proxy(candidates, context)
                
            if selected:
                # Update last_used timestamp
                selected.last_used = datetime.now()
                
            return selected
    
    def _filter_proxies(self, proxy_type: Optional[str] = None, 
                       country: Optional[str] = None) -> List[Proxy]:
        """
        Filter proxies based on criteria.
        
        Args:
            proxy_type: Type of proxy (HTTP, HTTPS, SOCKS4, SOCKS5)
            country: Country code
            
        Returns:
            List of matching proxies
        """
        # Start with all active proxies
        result = [p for p in self._proxies if p.is_active]
        
        # Filter by proxy type if specified
        if proxy_type:
            try:
                proxy_type_enum = ProxyType(proxy_type.lower())
                result = [p for p in result if p.proxy_type == proxy_type_enum]
            except ValueError:
                # Invalid proxy type, keep all proxies
                pass
        
        # Filter by country if specified
        if country:
            country_candidates = [p for p in result if p.country and p.country.lower() == country.lower()]
            # Only use country filter if it doesn't eliminate all proxies
            if country_candidates:
                result = country_candidates
        
        return result
    
    def register_proxy_result(self, proxy: Proxy, success: bool, response: Any = None, 
                             domain: Optional[str] = None, error_type: Optional[str] = None) -> None:
        """
        Register the result of using a proxy.
        
        Args:
            proxy: The proxy that was used
            success: Whether the request was successful
            response: Optional response object
            domain: Target domain for the request
            error_type: Type of error if request failed
        """
        if not proxy:
            return
            
        with self._lock:
            # Find the proxy in our list
            for p in self._proxies:
                if p == proxy:
                    # Calculate response time if available
                    response_time = 0.0
                    if hasattr(response, 'elapsed'):
                        response_time = response.elapsed.total_seconds()
                    
                    if success:
                        p.record_success(response_time)
                        
                        # Update adaptive strategy if domain is provided
                        if domain and 'adaptive' in self._strategies:
                            strategy = self._strategies['adaptive']
                            if isinstance(strategy, AdaptiveStrategy):
                                strategy.update_performance(p, domain, True)
                    else:
                        p.record_failure(error_type)
                        
                        # Update adaptive strategy if domain is provided
                        if domain and 'adaptive' in self._strategies:
                            strategy = self._strategies['adaptive']
                            if isinstance(strategy, AdaptiveStrategy):
                                strategy.update_performance(p, domain, False)
                        
                        # Check if proxy should be blacklisted
                        if p.consecutive_failures >= self._blacklist_threshold:
                            reason = f"Exceeded failure threshold ({p.consecutive_failures} consecutive failures)"
                            self.blacklist_proxy(p, reason)
                    
                    return
            
            # If we get here, the proxy wasn't found in our list
            logger.warning(f"Proxy {proxy} not found in manager")
    
    def blacklist_proxy(self, proxy: Proxy, reason: str, duration: Optional[timedelta] = None) -> None:
        """
        Temporarily blacklist a proxy.
        
        Args:
            proxy: The proxy to blacklist
            reason: Reason for blacklisting
            duration: How long to blacklist the proxy (default: 1 hour)
        """
        if not proxy:
            return
            
        with self._lock:
            # Find the proxy in our list
            for p in self._proxies:
                if p == proxy:
                    p.blacklist(reason, duration)
                    logger.info(f"Blacklisted proxy {p.url} for {reason}")
                    return
            
            # If we get here, the proxy wasn't found in our list
            logger.warning(f"Proxy {proxy} not found in manager")
    
    def remove_proxy(self, proxy: Proxy) -> None:
        """
        Permanently remove a proxy from the pool.
        
        Args:
            proxy: The proxy to remove
        """
        if not proxy:
            return
            
        with self._lock:
            # Find the proxy in our list
            for i, p in enumerate(self._proxies):
                if p == proxy:
                    p.status = ProxyStatus.REMOVED
                    logger.info(f"Removed proxy {p.url} from the pool")
                    return
            
            # If we get here, the proxy wasn't found in our list
            logger.warning(f"Proxy {proxy} not found in manager")
    
    def add_proxy(self, proxy: Union[Proxy, str], metadata: Dict[str, Any] = None) -> Optional[Proxy]:
        """
        Add a new proxy to the pool.
        
        Args:
            proxy: The proxy to add (Proxy object or string)
            metadata: Additional metadata for the proxy
            
        Returns:
            The added Proxy object or None if invalid
        """
        if isinstance(proxy, str):
            # Parse the proxy string
            provider = StaticProxyProvider("temp", {})
            proxy_obj = provider.parse_proxy_string(proxy)
            if not proxy_obj:
                logger.warning(f"Failed to parse proxy string: {proxy}")
                return None
            
            # Add metadata if provided
            if metadata:
                proxy_obj.metadata.update(metadata)
        else:
            proxy_obj = proxy
        
        with self._lock:
            # Check if proxy already exists
            for p in self._proxies:
                if p == proxy_obj:
                    logger.info(f"Proxy {proxy_obj.url} already exists, updating metadata")
                    # Update metadata
                    if metadata:
                        p.metadata.update(metadata)
                    return p
            
            # Add the new proxy
            self._proxies.append(proxy_obj)
            logger.info(f"Added new proxy {proxy_obj.url} to the pool")
            
            return proxy_obj
    
    def check_proxy_health(self, proxy: Proxy) -> bool:
        """
        Test the health of a proxy.
        
        Args:
            proxy: The proxy to check
            
        Returns:
            True if the proxy is healthy, False otherwise
        """
        if not proxy:
            return False
            
        # Mark the proxy as being tested
        proxy.status = ProxyStatus.TESTING
        proxy.last_checked = datetime.now()
        
        try:
            # Choose a test URL from the configuration
            test_urls = self._config.get('health_check_urls', [
                'http://httpbin.org/ip',
                'https://httpbin.org/ip'
            ])
            
            # Try each test URL
            for test_url in test_urls:
                # Select URL protocol based on proxy type
                if proxy.proxy_type in (ProxyType.HTTP, ProxyType.SOCKS4, ProxyType.SOCKS5) and test_url.startswith('http:'):
                    url = test_url
                elif proxy.proxy_type in (ProxyType.HTTPS, ProxyType.SOCKS4, ProxyType.SOCKS5) and test_url.startswith('https:'):
                    url = test_url
                else:
                    continue
                
                # Set up the request
                proxies = proxy.get_dict_for_requests()
                timeout = self._config.get('health_check_timeout', 10)
                
                # Make the request
                start_time = time.time()
                response = requests.get(
                    url,
                    proxies=proxies,
                    timeout=timeout,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                )
                response_time = time.time() - start_time
                
                # Check response
                if response.status_code == 200:
                    # Success - update proxy stats
                    proxy.status = ProxyStatus.ACTIVE
                    proxy.record_success(response_time)
                    
                    # Check anonymity level if using httpbin
                    if 'httpbin.org/ip' in url:
                        try:
                            data = response.json()
                            origin = data.get('origin', '')
                            
                            # If we see the proxy IP, it's transparent or anonymous
                            if origin and ',' not in origin:  # No commas means a single IP
                                # Assuming this is the proxy IP, would require external verification
                                # In a real implementation, compare to the actual proxy address
                                proxy.anonymity = ProxyAnonymity.ELITE
                            elif ',' in origin:  # Multiple IPs, comma-separated
                                proxy.anonymity = ProxyAnonymity.ANONYMOUS
                            else:
                                proxy.anonymity = ProxyAnonymity.UNKNOWN
                        except Exception as e:
                            logger.warning(f"Error parsing httpbin response for {proxy.url}: {str(e)}")
                    
                    logger.debug(f"Proxy {proxy.url} health check passed, response time: {response_time:.2f}s")
                    return True
        except Exception as e:
            # Request failed
            logger.debug(f"Proxy {proxy.url} health check failed: {str(e)}")
            proxy.record_failure(str(type(e).__name__))
        
        # If we get here, all checks failed
        proxy.status = ProxyStatus.ACTIVE  # Reset status even on failure
        return False
    
    def verify_proxy_anonymity(self, proxy: Proxy) -> ProxyAnonymity:
        """
        Check the anonymity level of a proxy.
        
        Args:
            proxy: The proxy to check
            
        Returns:
            The anonymity level of the proxy
        """
        if not proxy:
            return ProxyAnonymity.UNKNOWN
            
        try:
            # Use a service that returns client info
            url = 'https://httpbin.org/headers'
            proxies = proxy.get_dict_for_requests()
            timeout = self._config.get('anonymity_check_timeout', 10)
            
            # Make the request
            response = requests.get(
                url,
                proxies=proxies,
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    headers = data.get('headers', {})
                    
                    # Check for headers that might reveal proxy usage
                    proxy_headers = [
                        'Via',
                        'X-Forwarded-For',
                        'Forwarded',
                        'X-Real-IP',
                        'Proxy-Connection'
                    ]
                    
                    # Count how many proxy-related headers we find
                    proxy_header_count = sum(1 for h in proxy_headers if h.lower() in {k.lower() for k in headers.keys()})
                    
                    if proxy_header_count == 0:
                        anonymity = ProxyAnonymity.ELITE
                    elif 'X-Forwarded-For'.lower() in {k.lower() for k in headers.keys()}:
                        anonymity = ProxyAnonymity.TRANSPARENT
                    else:
                        anonymity = ProxyAnonymity.ANONYMOUS
                        
                    # Update proxy anonymity level
                    proxy.anonymity = anonymity
                    return anonymity
                    
                except Exception as e:
                    logger.warning(f"Error parsing anonymity check response for {proxy.url}: {str(e)}")
        except Exception as e:
            logger.warning(f"Anonymity check failed for {proxy.url}: {str(e)}")
            
        return ProxyAnonymity.UNKNOWN
    
    def check_proxy_location(self, proxy: Proxy) -> Dict[str, Any]:
        """
        Check the geographic location of a proxy.
        
        Args:
            proxy: The proxy to check
            
        Returns:
            Dictionary with location information
        """
        if not proxy:
            return {}
            
        try:
            # Use a geolocation service
            url = 'https://ipinfo.io/json'
            proxies = proxy.get_dict_for_requests()
            timeout = self._config.get('location_check_timeout', 10)
            
            # Make the request
            response = requests.get(
                url,
                proxies=proxies,
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Extract location data
                    location = {
                        'ip': data.get('ip'),
                        'country': data.get('country'),
                        'region': data.get('region'),
                        'city': data.get('city'),
                        'loc': data.get('loc'),
                        'org': data.get('org')
                    }
                    
                    # Update proxy metadata
                    proxy.country = data.get('country')
                    proxy.city = data.get('city')
                    proxy.isp = data.get('org')
                    
                    # Update proxy metadata
                    proxy.metadata.update({
                        'location': location
                    })
                    
                    return location
                    
                except Exception as e:
                    logger.warning(f"Error parsing location check response for {proxy.url}: {str(e)}")
        except Exception as e:
            logger.warning(f"Location check failed for {proxy.url}: {str(e)}")
            
        return {}
    
    def measure_proxy_latency(self, proxy: Proxy, url: Optional[str] = None) -> float:
        """
        Measure the latency of a proxy.
        
        Args:
            proxy: The proxy to check
            url: Optional URL to test against
            
        Returns:
            Latency in seconds, or float('inf') if check failed
        """
        if not proxy:
            return float('inf')
            
        try:
            # Use a test URL if not provided
            if not url:
                url = 'http://httpbin.org/get'
                
            # Set up the request
            proxies = proxy.get_dict_for_requests()
            timeout = self._config.get('latency_check_timeout', 10)
            
            # Make the request and measure time
            start_time = time.time()
            response = requests.get(
                url,
                proxies=proxies,
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            )
            latency = time.time() - start_time
            
            if response.status_code < 400:
                return latency
        except Exception as e:
            logger.debug(f"Latency check failed for {proxy.url}: {str(e)}")
            
        return float('inf')
    
    def get_proxy_performance(self, proxy: Proxy) -> Dict[str, Any]:
        """
        Get performance statistics for a proxy.
        
        Args:
            proxy: The proxy to check
            
        Returns:
            Dictionary with performance statistics
        """
        return {
            'address': proxy.address,
            'port': proxy.port,
            'type': proxy.proxy_type.value,
            'success_count': proxy.success_count,
            'failure_count': proxy.failure_count,
            'success_rate': proxy.success_rate,
            'average_response_time': proxy.average_response_time,
            'last_used': proxy.last_used.isoformat() if proxy.last_used else None,
            'last_checked': proxy.last_checked.isoformat() if proxy.last_checked else None,
            'last_success': proxy.last_success.isoformat() if proxy.last_success else None,
            'consecutive_failures': proxy.consecutive_failures,
            'country': proxy.country,
            'isp': proxy.isp
        }
    
    def get_top_proxies(self, count: int = 5) -> List[Proxy]:
        """
        Get the top N proxies based on success rate and response time.
        
        Args:
            count: Number of top proxies to return
            
        Returns:
            List of top proxies
        """
        with self._lock:
            # Filter to only active proxies
            candidates = [p for p in self._proxies if p.is_active]
            
            # Sort by combined score of success rate and response time
            def proxy_score(p):
                success_score = p.success_rate
                time_score = 1.0 / (1.0 + min(10.0, p.average_response_time))  # Cap at 10 seconds
                return success_score * 0.7 + time_score * 0.3
            
            candidates.sort(key=proxy_score, reverse=True)
            
            # Return the top N
            return candidates[:count]
    
    def get_proxy_usage_stats(self) -> Dict[str, Any]:
        """
        Get overall proxy usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        with self._lock:
            active_count = sum(1 for p in self._proxies if p.status == ProxyStatus.ACTIVE)
            blacklisted_count = sum(1 for p in self._proxies if p.status == ProxyStatus.BLACKLISTED)
            testing_count = sum(1 for p in self._proxies if p.status == ProxyStatus.TESTING)
            
            total_requests = sum(p.success_count + p.failure_count for p in self._proxies)
            successful_requests = sum(p.success_count for p in self._proxies)
            success_rate = successful_requests / total_requests if total_requests > 0 else 0
            
            # Group proxies by country
            countries = {}
            for p in self._proxies:
                if p.country:
                    countries[p.country] = countries.get(p.country, 0) + 1
                    
            # Group proxies by type
            types = {}
            for p in self._proxies:
                proxy_type = p.proxy_type.value
                types[proxy_type] = types.get(proxy_type, 0) + 1
                
            return {
                'total_proxies': len(self._proxies),
                'active_proxies': active_count,
                'blacklisted_proxies': blacklisted_count,
                'testing_proxies': testing_count,
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'overall_success_rate': success_rate,
                'countries': countries,
                'types': types
            }
    
    def export_proxy_metrics(self, include_blacklisted: bool = False) -> List[Dict[str, Any]]:
        """
        Export metrics for all proxies.
        
        Args:
            include_blacklisted: Whether to include blacklisted proxies
            
        Returns:
            List of proxy metrics
        """
        metrics = []
        
        with self._lock:
            for proxy in self._proxies:
                # Skip blacklisted proxies if not included
                if not include_blacklisted and proxy.status == ProxyStatus.BLACKLISTED:
                    continue
                    
                metrics.append(self.get_proxy_performance(proxy))
                
        return metrics
    
    def unblacklist_all(self) -> int:
        """
        Remove all proxies from the blacklist.
        
        Returns:
            Number of proxies unblacklisted
        """
        count = 0
        
        with self._lock:
            for proxy in self._proxies:
                if proxy.status == ProxyStatus.BLACKLISTED:
                    proxy.unblacklist()
                    count += 1
        
        logger.info(f"Unblacklisted {count} proxies")
        return count
    
    def set_strategy(self, strategy_name: str) -> bool:
        """
        Set the active proxy rotation strategy.
        
        Args:
            strategy_name: Name of the strategy to use
            
        Returns:
            True if the strategy was set, False if not found
        """
        with self._lock:
            if strategy_name in self._strategies:
                self._current_strategy = self._strategies[strategy_name]
                logger.info(f"Set proxy rotation strategy to {strategy_name}")
                return True
            else:
                logger.warning(f"Strategy {strategy_name} not found")
                return False

    def get_proxy_iterator(self, strategy: Optional[str] = None,
                           domain: Optional[str] = None, proxy_type: Optional[str] = None,
                           country: Optional[str] = None, max_attempts: int = None) -> ProxyIterator:
        """
        Get an iterator that allows for systematic proxy iteration.
        
        Args:
            strategy: Proxy rotation strategy to use
            domain: Target domain for the requests
            proxy_type: Type of proxy needed
            country: Preferred country for the proxy
            max_attempts: Maximum number of proxies to try
            
        Returns:
            A ProxyIterator instance
        """
        return ProxyIterator(
            proxy_manager=self,
            strategy=strategy,
            domain=domain,
            proxy_type=proxy_type,
            country=country,
            max_attempts=max_attempts
        )

    def get_all_proxies(self) -> List[Proxy]:
        """
        Get all proxies managed by this proxy manager.
        
        Returns:
            List of all proxies (active, blacklisted, and testing)
        """
        with self._lock:
            return self._proxies.copy()
    
    def get_proxy_usage_stats(self) -> Dict[str, Any]:
        """
        Get overall proxy usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        with self._lock:
            active_count = sum(1 for p in self._proxies if p.status == ProxyStatus.ACTIVE)
            blacklisted_count = sum(1 for p in self._proxies if p.status == ProxyStatus.BLACKLISTED)
            testing_count = sum(1 for p in self._proxies if p.status == ProxyStatus.TESTING)
            
            total_requests = sum(p.success_count + p.failure_count for p in self._proxies)
            successful_requests = sum(p.success_count for p in self._proxies)
            success_rate = successful_requests / total_requests if total_requests > 0 else 0
            
            # Group proxies by country
            countries = {}
            for p in self._proxies:
                if p.country:
                    countries[p.country] = countries.get(p.country, 0) + 1
                    
            # Group proxies by type
            types = {}
            for p in self._proxies:
                proxy_type = p.proxy_type.value
                types[proxy_type] = types.get(proxy_type, 0) + 1
                
            return {
                'total_proxies': len(self._proxies),
                'active_proxies': active_count,
                'blacklisted_proxies': blacklisted_count,
                'testing_proxies': testing_count,
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'overall_success_rate': success_rate,
                'countries': countries,
                'types': types
            }
    
    def export_proxy_metrics(self, include_blacklisted: bool = False) -> List[Dict[str, Any]]:
        """
        Export metrics for all proxies.
        
        Args:
            include_blacklisted: Whether to include blacklisted proxies
            
        Returns:
            List of proxy metrics
        """
        metrics = []
        
        with self._lock:
            for proxy in self._proxies:
                # Skip blacklisted proxies if not included
                if not include_blacklisted and proxy.status == ProxyStatus.BLACKLISTED:
                    continue
                    
                metrics.append(self.get_proxy_performance(proxy))
                
        return metrics
    
    def unblacklist_all(self) -> int:
        """
        Remove all proxies from the blacklist.
        
        Returns:
            Number of proxies unblacklisted
        """
        count = 0
        
        with self._lock:
            for proxy in self._proxies:
                if proxy.status == ProxyStatus.BLACKLISTED:
                    proxy.unblacklist()
                    count += 1
        
        logger.info(f"Unblacklisted {count} proxies")
        return count
    
    def set_strategy(self, strategy_name: str) -> bool:
        """
        Set the active proxy rotation strategy.
        
        Args:
            strategy_name: Name of the strategy to use
            
        Returns:
            True if the strategy was set, False if not found
        """
        with self._lock:
            if strategy_name in self._strategies:
                self._current_strategy = self._strategies[strategy_name]
                logger.info(f"Set proxy rotation strategy to {strategy_name}")
                return True
            else:
                logger.warning(f"Strategy {strategy_name} not found")
                return False

    def get_proxy_iterator(self, strategy: Optional[str] = None,
                           domain: Optional[str] = None, proxy_type: Optional[str] = None,
                           country: Optional[str] = None, max_attempts: int = None) -> ProxyIterator:
        """
        Get an iterator that allows for systematic proxy iteration.
        
        Args:
            strategy: Proxy rotation strategy to use
            domain: Target domain for the requests
            proxy_type: Type of proxy needed
            country: Preferred country for the proxy
            max_attempts: Maximum number of proxies to try
            
        Returns:
            A ProxyIterator instance
        """
        return ProxyIterator(
            proxy_manager=self,
            strategy=strategy,
            domain=domain,
            proxy_type=proxy_type,
            country=country,
            max_attempts=max_attempts
        )


class ProxyIterator:
    """
    Iterator for cycling through proxies based on a specific strategy.
    This allows users to systematically iterate through proxies when needed.
    """
    
    def __init__(self, proxy_manager, strategy: Optional[str] = None,
                 domain: Optional[str] = None, proxy_type: Optional[str] = None,
                 country: Optional[str] = None, max_attempts: int = None):
        """
        Initialize a proxy iterator.
        
        Args:
            proxy_manager: Reference to the ProxyManager
            strategy: Strategy to use for proxy selection
            domain: Target domain for the requests
            proxy_type: Type of proxy needed
            country: Preferred country for the proxy
            max_attempts: Maximum number of proxies to try (None for unlimited)
        """
        self.proxy_manager = proxy_manager
        self.strategy = strategy
        self.domain = domain
        self.proxy_type = proxy_type
        self.country = country
        self.max_attempts = max_attempts
        self.attempts = 0
        self.used_proxies = set()
    
    def __iter__(self):
        """Return self as an iterator."""
        return self
    
    def __next__(self):
        """Get the next proxy in the iteration."""
        if self.max_attempts is not None and self.attempts >= self.max_attempts:
            raise StopIteration
            
        # Get a proxy using the specified criteria
        proxy = self.proxy_manager.get_proxy(
            domain=self.domain,
            proxy_type=self.proxy_type,
            country=self.country,
            strategy=self.strategy
        )
        
        # If no proxy is available, stop iteration
        if not proxy:
            raise StopIteration
            
        # Keep track of which proxies we've seen to avoid infinite loops
        proxy_key = (proxy.address, proxy.port, proxy.proxy_type)
        
        # If we've seen all available proxies, stop iteration
        if proxy_key in self.used_proxies:
            # Check if we've used all available proxies
            with self.proxy_manager._lock:
                active_proxies = [p for p in self.proxy_manager._proxies 
                                 if p.is_active and 
                                 (self.proxy_type is None or p.proxy_type.value == self.proxy_type) and
                                 (self.country is None or (p.country and p.country.lower() == self.country.lower()))]
                
                if len(self.used_proxies) >= len(active_proxies):
                    raise StopIteration
        
        # Add to used proxies
        self.used_proxies.add(proxy_key)
        self.attempts += 1
        
        return proxy
    
    def register_result(self, proxy: Proxy, success: bool, response: Any = None, 
                        error_type: Optional[str] = None) -> None:
        """
        Register the result of using a proxy.
        
        Args:
            proxy: The proxy that was used
            success: Whether the request was successful
            response: Optional response object
            error_type: Type of error if request failed
        """
        if proxy:
            self.proxy_manager.register_proxy_result(
                proxy=proxy,
                success=success,
                response=response,
                domain=self.domain,
                error_type=error_type
            )
            
            # If the proxy failed, remove it from our usable set to avoid reusing it
            if not success:
                proxy_key = (proxy.address, proxy.port, proxy.proxy_type)
                if proxy_key in self.used_proxies:
                    self.used_proxies.remove(proxy_key)