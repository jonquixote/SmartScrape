"""
Resource Management Mixin for strategies.

This mixin provides resource management capabilities to strategies, including
session management, rate limiting, and proxy management.
"""

import logging
import time
import random
import threading
import requests
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class ResourceManagementMixin:
    """
    Mixin that provides resource management capabilities to strategies.
    
    This mixin is designed to be used with BaseStrategy and its subclasses.
    It provides:
    - Session management (pooling, reuse, rotation)
    - Rate limiting (adaptive, per-domain)
    - Proxy management (rotation, health checks)
    - Resource usage metrics
    """
    
    def _initialize_resource_management(self):
        """Initialize resource management state."""
        # Session management
        self._sessions = {}
        self._session_lock = threading.RLock()
        
        # Rate limiting
        self._rate_limits = {}
        self._last_request_time = {}
        self._rate_limit_lock = threading.RLock()
        
        # Proxy management
        self._proxies = []
        self._proxy_health = {}
        self._proxy_lock = threading.RLock()
        
        # Metrics
        self._request_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_request_time': 0,
            'per_domain_metrics': {}
        }
        self._metrics_lock = threading.RLock()
        
        # Try to load proxies from the proxy manager service if available
        self._load_proxies_from_service()
    
    def _load_proxies_from_service(self):
        """Load proxies from the proxy manager service if available."""
        try:
            from core.service_registry import ServiceRegistry
            registry = ServiceRegistry()
            proxy_manager = registry.get_service("proxy_manager")
            
            if proxy_manager:
                self._proxies = proxy_manager.get_all_proxies()
                logger.info(f"Loaded {len(self._proxies)} proxies from proxy manager service")
        except KeyError:
            # Proxy manager not registered yet, this is normal during startup
            logger.debug("Proxy manager not registered yet, will use empty proxy list")
        except Exception as e:
            logger.warning(f"Error loading proxies from service: {str(e)}")
    
    def _get_session(self, domain: str, force_new: bool = False) -> requests.Session:
        """
        Get a session for the specified domain.
        
        Args:
            domain: The domain name
            force_new: Whether to force creation of a new session
            
        Returns:
            requests.Session object
        """
        with self._session_lock:
            if domain in self._sessions and not force_new:
                return self._sessions[domain]
            
            # Create a new session
            session = requests.Session()
            
            # Set default headers
            session.headers.update({
                'User-Agent': self._get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'DNT': '1',
            })
            
            # Store the session
            self._sessions[domain] = session
            return session
    
    def _get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36 Edg/92.0.902.67',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36 OPR/78.0.4093.147'
        ]
        return random.choice(user_agents)
    
    def _rotate_user_agent(self, domain: str) -> None:
        """
        Rotate the user agent for a domain's session.
        
        Args:
            domain: The domain to rotate user agent for
        """
        with self._session_lock:
            if domain in self._sessions:
                self._sessions[domain].headers.update({
                    'User-Agent': self._get_random_user_agent()
                })
    
    def _handle_rate_limiting(self, domain: str) -> None:
        """
        Handle rate limiting for the specified domain.
        
        This will pause execution if necessary to comply with rate limits.
        
        Args:
            domain: The domain to handle rate limiting for
        """
        with self._rate_limit_lock:
            # Get domain-specific rate limit or use default
            requests_per_minute = self._rate_limits.get(domain, {}).get('requests_per_minute', 60)
            
            # Calculate minimum interval between requests
            min_interval = 60.0 / requests_per_minute
            
            # Check if we need to wait
            current_time = time.time()
            last_time = self._last_request_time.get(domain, 0)
            time_since_last = current_time - last_time
            
            if time_since_last < min_interval:
                # Need to wait
                wait_time = min_interval - time_since_last
                
                # Add jitter to prevent thundering herd
                jitter = random.uniform(0, 0.1 * wait_time)
                wait_time += jitter
                
                logger.debug(f"Rate limit: Waiting {wait_time:.2f}s for {domain}")
                
                # Release lock during wait
                self._rate_limit_lock.release()
                try:
                    time.sleep(wait_time)
                finally:
                    self._rate_limit_lock.acquire()
            
            # Update last request time
            self._last_request_time[domain] = time.time()
    
    def _update_rate_limit(self, domain: str, status_code: int) -> None:
        """
        Update rate limits based on server response.
        
        Args:
            domain: The domain to update rate limit for
            status_code: HTTP status code from the response
        """
        with self._rate_limit_lock:
            # Get current rate limit or use default
            current_limit = self._rate_limits.get(domain, {}).get('requests_per_minute', 60)
            
            # Adjust based on response
            if status_code == 429:  # Too Many Requests
                # Reduce rate limit by 50%
                new_limit = max(1, int(current_limit * 0.5))
                logger.warning(f"Rate limit hit for {domain}. Reducing from {current_limit} to {new_limit} requests per minute")
            elif status_code == 503:  # Service Unavailable
                # Reduce rate limit by 25%
                new_limit = max(1, int(current_limit * 0.75))
                logger.warning(f"Service unavailable for {domain}. Reducing from {current_limit} to {new_limit} requests per minute")
            elif status_code >= 200 and status_code < 300:
                # Success - gradually increase rate limit up to a maximum
                if current_limit < 60:
                    new_limit = min(60, current_limit + 1)
                    logger.debug(f"Increasing rate limit for {domain} to {new_limit} requests per minute")
                else:
                    return  # No change needed
            else:
                return  # No change for other status codes
            
            # Store updated rate limit
            self._rate_limits[domain] = {
                'requests_per_minute': new_limit,
                'updated_at': time.time()
            }
    
    def _get_proxy(self, domain: str) -> Optional[Dict[str, str]]:
        """
        Get a proxy for the specified domain.
        
        Args:
            domain: The domain to get a proxy for
            
        Returns:
            Proxy dictionary or None if no proxy available
        """
        with self._proxy_lock:
            if not self._proxies:
                return None
            
            # Try to find a healthy proxy for this domain
            domain_key = domain.replace('.', '_')
            healthy_proxies = [p for p in self._proxies if self._proxy_health.get(f"{domain_key}_{p.get('id')}", {}).get('status', 'unknown') != 'failed']
            
            if not healthy_proxies:
                # No healthy proxies, try any proxy
                proxy = random.choice(self._proxies)
            else:
                proxy = random.choice(healthy_proxies)
            
            # Convert to requests format
            proxy_url = proxy.get('url')
            if not proxy_url:
                return None
            
            return {
                'http': proxy_url,
                'https': proxy_url
            }
    
    def _update_proxy_health(self, domain: str, proxy_id: str, success: bool) -> None:
        """
        Update proxy health status based on request result.
        
        Args:
            domain: The domain the proxy was used for
            proxy_id: The ID of the proxy
            success: Whether the request succeeded
        """
        with self._proxy_lock:
            domain_key = domain.replace('.', '_')
            key = f"{domain_key}_{proxy_id}"
            
            health = self._proxy_health.get(key, {
                'successes': 0,
                'failures': 0,
                'status': 'unknown'
            })
            
            if success:
                health['successes'] = health.get('successes', 0) + 1
                # Reset failures after 3 consecutive successes
                if health.get('successes', 0) >= 3:
                    health['failures'] = 0
                    health['status'] = 'healthy'
            else:
                health['failures'] = health.get('failures', 0) + 1
                # Mark as failed after 3 consecutive failures
                if health.get('failures', 0) >= 3:
                    health['status'] = 'failed'
            
            self._proxy_health[key] = health
    
    def _record_request_metrics(self, url: str, start_time: float, success: bool) -> None:
        """
        Record metrics for a request.
        
        Args:
            url: The URL that was requested
            start_time: The start time of the request
            success: Whether the request succeeded
        """
        with self._metrics_lock:
            # Calculate duration
            duration = time.time() - start_time
            
            # Update global metrics
            self._request_metrics['total_requests'] += 1
            self._request_metrics['total_request_time'] += duration
            
            if success:
                self._request_metrics['successful_requests'] += 1
            else:
                self._request_metrics['failed_requests'] += 1
            
            # Update domain-specific metrics
            domain = urlparse(url).netloc
            domain_metrics = self._request_metrics['per_domain_metrics'].get(domain, {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_request_time': 0
            })
            
            domain_metrics['total_requests'] += 1
            domain_metrics['total_request_time'] += duration
            
            if success:
                domain_metrics['successful_requests'] += 1
            else:
                domain_metrics['failed_requests'] += 1
            
            self._request_metrics['per_domain_metrics'][domain] = domain_metrics
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """
        Get statistics about resource usage.
        
        Returns:
            Dictionary with resource usage statistics
        """
        with self._metrics_lock, self._session_lock, self._proxy_lock, self._rate_limit_lock:
            stats = {
                'sessions': {
                    'count': len(self._sessions),
                    'domains': list(self._sessions.keys())
                },
                'rate_limits': {
                    'count': len(self._rate_limits),
                    'domains': {d: v.get('requests_per_minute', 0) for d, v in self._rate_limits.items()}
                },
                'proxies': {
                    'count': len(self._proxies),
                    'healthy_count': len([p for p in self._proxy_health.values() if p.get('status') == 'healthy']),
                    'failed_count': len([p for p in self._proxy_health.values() if p.get('status') == 'failed'])
                },
                'requests': {
                    'total': self._request_metrics['total_requests'],
                    'successful': self._request_metrics['successful_requests'],
                    'failed': self._request_metrics['failed_requests']
                }
            }
            
            # Add average request time if there are any requests
            if self._request_metrics['total_requests'] > 0:
                stats['requests']['avg_time'] = self._request_metrics['total_request_time'] / self._request_metrics['total_requests']
            
            return stats
    
    def cleanup_resources(self) -> None:
        """Clean up all resources held by the mixin."""
        with self._session_lock:
            for session in self._sessions.values():
                try:
                    session.close()
                except:
                    pass
            self._sessions.clear()