"""
HTTP Utilities Module

This module provides HTTP utilities for making requests, managing sessions,
and handling cookies. It includes advanced features like:
- Persistent session management
- Cookie handling with jar support
- Request fingerprint randomization
- Exponential backoff retry logic
- Rate limiting
- Proxy rotation
- Anti-bot detection mechanisms
"""

import asyncio
import os
import json
import time
import random
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, urljoin, parse_qs, urlencode, urlunparse
import hashlib

import httpx
from fake_useragent import UserAgent
import aiohttp
import aiofiles
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils.retry_utils import retry_on_network_errors, retry_on_http_errors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HTTPUtils")

# Add imports for captcha detection
import re
import base64
from PIL import Image
import io

@retry_on_network_errors(max_attempts=3)
async def fetch_html(url: str, headers: Optional[Dict[str, str]] = None, 
                    timeout: int = 30, verify: bool = True) -> str:
    """
    Fetch HTML content from a URL.
    
    Args:
        url: URL to fetch HTML from
        headers: Optional HTTP headers
        timeout: Request timeout in seconds
        verify: Whether to verify SSL certificates
        
    Returns:
        HTML content as string
    """
    if not headers:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
    
    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response.text
    except Exception as e:
        logger.error(f"Error fetching HTML from {url}: {str(e)}")
        raise

def clean_url(url: str, remove_tracking: bool = True, default_scheme: str = "http") -> str:
    """
    Clean and normalize a URL.
    
    Args:
        url: URL to clean
        remove_tracking: Whether to remove tracking parameters
        default_scheme: Default scheme to add if none present
        
    Returns:
        Cleaned URL string
    """
    # Handle empty URLs
    if not url:
        return ""
    
    # Add default scheme if none present
    if not url.startswith(('http://', 'https://')):
        url = f"{default_scheme}://{url}"
    
    # Parse the URL
    parsed = urlparse(url)
    
    # Normalize hostname (lowercase)
    netloc = parsed.netloc.lower()
    
    # Remove default ports
    if netloc.endswith(':80') and parsed.scheme == 'http':
        netloc = netloc[:-3]
    elif netloc.endswith(':443') and parsed.scheme == 'https':
        netloc = netloc[:-4]
    
    # Normalize path
    path = parsed.path or "/"
    
    # Process query parameters
    query = parsed.query
    
    if remove_tracking:
        # Common tracking parameters to remove
        tracking_params = {
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'fbclid', 'gclid', 'msclkid', 'zanpid', 'dclid', '_hsenc', '_hsmi',
            'hsa_cam', 'hsa_grp', 'hsa_mt', 'hsa_src', 'hsa_ad', 'hsa_acc',
            'hsa_net', 'hsa_kw', 'hsa_tgt', 'hsa_ver', 
        }
        
        # Parse query parameters and remove tracking params
        if query:
            params = parse_qs(query)
            for param in list(params.keys()):
                if param.lower() in tracking_params:
                    del params[param]
            
            # Rebuild query string
            query = urlencode(params, doseq=True) if params else ""
    
    # Reconstruct the URL
    clean = urlunparse((parsed.scheme, netloc, path, parsed.params, query, ""))
    
    return clean

class ProxyManager:
    """
    Manages a pool of proxies with health checking and rotation.
    
    This class provides advanced proxy management, including:
    - Proxy pool management with health checking
    - Smart proxy rotation based on performance metrics
    - Automatic proxy testing and validation
    - Per-domain proxy selection
    - Proxy authentication support
    """
    
    def __init__(self, 
                 proxies: Optional[List[str]] = None,
                 proxy_file: Optional[str] = None,
                 test_url: str = "https://httpbin.org/ip",
                 max_failures: int = 3,
                 health_check_interval: int = 600):
        """
        Initialize the proxy manager.
        
        Args:
            proxies: List of proxy URLs
            proxy_file: Path to file containing proxy URLs (one per line)
            test_url: URL to use for proxy testing
            max_failures: Maximum consecutive failures before marking proxy as bad
            health_check_interval: Interval in seconds between health checks
        """
        self.proxies = []
        self.test_url = test_url
        self.max_failures = max_failures
        self.health_check_interval = health_check_interval
        
        # Proxy metrics tracking
        self.proxy_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Domain-specific proxy mapping
        self.domain_proxies: Dict[str, List[str]] = {}
        
        # Current proxy index per domain
        self.domain_proxy_index: Dict[str, int] = {}
        
        # Fallback proxy for when all are failing
        self.fallback_proxy = None
        
        # Load proxies from list or file
        if proxies:
            self.add_proxies(proxies)
        
        if proxy_file:
            self.load_proxies_from_file(proxy_file)
        
        logger.info(f"Proxy manager initialized with {len(self.proxies)} proxies")
    
    def add_proxies(self, proxies: List[str]) -> None:
        """
        Add proxies to the pool.
        
        Args:
            proxies: List of proxy URLs to add
        """
        for proxy in proxies:
            if proxy not in self.proxies:
                self.proxies.append(proxy)
                
                # Initialize metrics for new proxy
                if proxy not in self.proxy_metrics:
                    self.proxy_metrics[proxy] = {
                        "success_count": 0,
                        "failure_count": 0,
                        "consecutive_failures": 0,
                        "response_times": [],
                        "last_used": 0,
                        "last_checked": 0,
                        "available": True,
                        "avg_response_time": 0,
                        "success_rate": 0
                    }
        
        logger.info(f"Added {len(proxies)} proxies, total: {len(self.proxies)}")
    
    def load_proxies_from_file(self, file_path: str) -> None:
        """
        Load proxies from a file.
        
        Args:
            file_path: Path to file containing proxies (one per line)
        """
        try:
            with open(file_path, 'r') as f:
                proxies = [line.strip() for line in f if line.strip()]
                self.add_proxies(proxies)
                logger.info(f"Loaded {len(proxies)} proxies from {file_path}")
        except Exception as e:
            logger.error(f"Error loading proxies from {file_path}: {str(e)}")
    
    def save_proxies_to_file(self, file_path: str) -> None:
        """
        Save current proxies to a file.
        
        Args:
            file_path: Path to save proxies to
        """
        try:
            with open(file_path, 'w') as f:
                for proxy in self.proxies:
                    f.write(f"{proxy}\n")
            logger.info(f"Saved {len(self.proxies)} proxies to {file_path}")
        except Exception as e:
            logger.error(f"Error saving proxies to {file_path}: {str(e)}")
    
    def get_proxy(self, domain: Optional[str] = None) -> Optional[str]:
        """
        Get the next proxy based on domain and performance.
        
        Args:
            domain: Optional domain for domain-specific proxy selection
            
        Returns:
            Proxy URL or None if no proxies are available
        """
        if not self.proxies:
            return None
        
        # Get domain-specific proxies if available
        available_proxies = self.domain_proxies.get(domain, self.proxies) if domain else self.proxies
        
        # Filter out unavailable proxies
        available_proxies = [p for p in available_proxies if self.proxy_metrics.get(p, {}).get("available", True)]
        
        if not available_proxies:
            # If no available proxies, try using fallback
            if self.fallback_proxy:
                return self.fallback_proxy
            
            # If no fallback, reset all proxies and try again
            logger.warning("No available proxies, resetting all proxy statuses")
            for proxy in self.proxies:
                self.proxy_metrics[proxy]["available"] = True
                self.proxy_metrics[proxy]["consecutive_failures"] = 0
            
            return self.get_proxy(domain)
        
        # Use domain-specific rotation index
        if domain:
            if domain not in self.domain_proxy_index:
                self.domain_proxy_index[domain] = 0
            
            # Get proxy using current index
            index = self.domain_proxy_index[domain]
            proxy = available_proxies[index % len(available_proxies)]
            
            # Update index for next time
            self.domain_proxy_index[domain] = (index + 1) % len(available_proxies)
        else:
            # Select proxy based on performance metrics
            if random.random() < 0.8:
                # 80% of the time, use the best performing proxy
                proxy = self._get_best_proxy(available_proxies)
            else:
                # 20% of the time, use a random proxy to explore performance
                proxy = random.choice(available_proxies)
        
        # Update last used time
        self.proxy_metrics[proxy]["last_used"] = time.time()
        
        return proxy
    
    def _get_best_proxy(self, proxies: List[str]) -> str:
        """
        Get the best performing proxy from a list.
        
        Args:
            proxies: List of proxy URLs to choose from
            
        Returns:
            Best proxy URL
        """
        if not proxies:
            return None
        
        # Create a list of (proxy, score) tuples
        scored_proxies = []
        
        for proxy in proxies:
            metrics = self.proxy_metrics.get(proxy, {})
            
            # Calculate score based on success rate and response time
            success_rate = metrics.get("success_rate", 0)
            avg_response_time = metrics.get("avg_response_time", 1)
            
            # Avoid division by zero
            if avg_response_time == 0:
                avg_response_time = 1
            
            # Score formula: success_rate / response_time
            # Higher success rate and lower response time = better score
            score = (success_rate + 0.1) / avg_response_time
            
            scored_proxies.append((proxy, score))
        
        # Sort by score in descending order
        scored_proxies.sort(key=lambda x: x[1], reverse=True)
        
        # Return the proxy with the highest score
        return scored_proxies[0][0]
    
    def record_success(self, proxy: str, response_time: float) -> None:
        """
        Record a successful proxy request.
        
        Args:
            proxy: Proxy URL
            response_time: Response time in seconds
        """
        if proxy not in self.proxy_metrics:
            self.proxy_metrics[proxy] = {
                "success_count": 0,
                "failure_count": 0,
                "consecutive_failures": 0,
                "response_times": [],
                "last_used": 0,
                "last_checked": 0,
                "available": True,
                "avg_response_time": 0,
                "success_rate": 0
            }
        
        metrics = self.proxy_metrics[proxy]
        
        # Update counts
        metrics["success_count"] += 1
        metrics["consecutive_failures"] = 0
        
        # Track response time (keep last 10)
        metrics["response_times"].append(response_time)
        if len(metrics["response_times"]) > 10:
            metrics["response_times"].pop(0)
        
        # Recalculate average response time
        metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
        
        # Recalculate success rate
        total = metrics["success_count"] + metrics["failure_count"]
        if total > 0:
            metrics["success_rate"] = metrics["success_count"] / total
        
        # Ensure proxy is marked as available
        metrics["available"] = True
    
    def record_failure(self, proxy: str) -> None:
        """
        Record a failed proxy request.
        
        Args:
            proxy: Proxy URL
        """
        if proxy not in self.proxy_metrics:
            self.proxy_metrics[proxy] = {
                "success_count": 0,
                "failure_count": 0,
                "consecutive_failures": 0,
                "response_times": [],
                "last_used": 0,
                "last_checked": 0,
                "available": True,
                "avg_response_time": 0,
                "success_rate": 0
            }
        
        metrics = self.proxy_metrics[proxy]
        
        # Update counts
        metrics["failure_count"] += 1
        metrics["consecutive_failures"] += 1
        
        # Recalculate success rate
        total = metrics["success_count"] + metrics["failure_count"]
        if total > 0:
            metrics["success_rate"] = metrics["success_count"] / total
        
        # Check if proxy should be marked unavailable
        if metrics["consecutive_failures"] >= self.max_failures:
            metrics["available"] = False
            logger.warning(f"Proxy {proxy} marked as unavailable after {metrics['consecutive_failures']} consecutive failures")
    
    async def test_proxy(self, proxy: str) -> Tuple[bool, float]:
        """
        Test a proxy by making a request through it.
        
        Args:
            proxy: Proxy URL to test
            
        Returns:
            Tuple of (success, response_time)
        """
        try:
            start_time = time.time()
            
            # Create client with proxy
            async with httpx.AsyncClient(proxies={"all://": proxy}, timeout=10.0) as client:
                response = await client.get(self.test_url)
                
                response_time = time.time() - start_time
                
                # Check if response is valid
                if response.status_code == 200:
                    logger.debug(f"Proxy {proxy} test successful ({response_time:.2f}s)")
                    return True, response_time
                else:
                    logger.debug(f"Proxy {proxy} test failed with status code {response.status_code}")
                    return False, response_time
        
        except Exception as e:
            logger.debug(f"Proxy {proxy} test failed with error: {str(e)}")
            return False, 999.0
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Run a health check on all proxies.
        
        Returns:
            Health check results
        """
        results = {
            "total": len(self.proxies),
            "available": 0,
            "unavailable": 0,
            "avg_response_time": 0,
            "tested": 0
        }
        
        response_times = []
        
        for proxy in self.proxies:
            # Only test proxies that haven't been checked recently
            metrics = self.proxy_metrics.get(proxy, {})
            last_checked = metrics.get("last_checked", 0)
            
            if time.time() - last_checked < self.health_check_interval:
                # Skip recent checks
                continue
            
            # Test proxy
            success, response_time = await self.test_proxy(proxy)
            
            # Update metrics
            if success:
                self.record_success(proxy, response_time)
                response_times.append(response_time)
            else:
                self.record_failure(proxy)
            
            # Update last checked time
            self.proxy_metrics[proxy]["last_checked"] = time.time()
            results["tested"] += 1
        
        # Count available proxies
        for proxy in self.proxies:
            metrics = self.proxy_metrics.get(proxy, {})
            if metrics.get("available", True):
                results["available"] += 1
            else:
                results["unavailable"] += 1
        
        # Calculate average response time
        if response_times:
            results["avg_response_time"] = sum(response_times) / len(response_times)
        
        logger.info(f"Proxy health check completed: {results['available']}/{results['total']} available")
        
        return results
    
    def associate_domain_with_proxies(self, domain: str, proxies: List[str]) -> None:
        """
        Associate a domain with specific proxies.
        
        Args:
            domain: Domain to associate
            proxies: List of proxy URLs to use for the domain
        """
        # Ensure all proxies exist in the main list
        self.add_proxies(proxies)
        
        # Associate domain with these proxies
        self.domain_proxies[domain] = proxies.copy()
        logger.info(f"Associated domain {domain} with {len(proxies)} proxies")
    
    def set_fallback_proxy(self, proxy: str) -> None:
        """
        Set a fallback proxy to use when all others fail.
        
        Args:
            proxy: Proxy URL to use as fallback
        """
        if proxy not in self.proxies:
            self.add_proxies([proxy])
        
        self.fallback_proxy = proxy
        logger.info(f"Set fallback proxy to {proxy}")
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get proxy performance metrics.
        
        Returns:
            Dictionary of proxy metrics
        """
        return self.proxy_metrics
    
    def get_best_proxies(self, count: int = 5) -> List[str]:
        """
        Get the best performing proxies.
        
        Args:
            count: Number of proxies to return
            
        Returns:
            List of best proxy URLs
        """
        # Create a list of (proxy, score) tuples
        scored_proxies = []
        
        for proxy in self.proxies:
            metrics = self.proxy_metrics.get(proxy, {})
            
            # Only include available proxies
            if not metrics.get("available", True):
                continue
            
            # Calculate score based on success rate and response time
            success_rate = metrics.get("success_rate", 0)
            avg_response_time = metrics.get("avg_response_time", 1)
            
            # Avoid division by zero
            if avg_response_time == 0:
                avg_response_time = 1
            
            # Score formula: success_rate / response_time
            score = (success_rate + 0.1) / avg_response_time
            
            scored_proxies.append((proxy, score))
        
        # Sort by score in descending order
        scored_proxies.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top N proxies
        return [p[0] for p in scored_proxies[:count]]

class ProxyRotationManager:
    """
    Manages a pool of proxies and rotates them to distribute requests.
    
    Features:
    - Smart proxy rotation using various strategies
    - Proxy health monitoring and auto-removal of dead proxies
    - Support for authenticated proxies
    - Geolocation-based proxy selection
    - Circuit breaking for failed proxies
    - Rotation patterns to avoid detection
    """
    
    def __init__(self, 
                 proxy_list: Optional[List[str]] = None,
                 proxy_file: Optional[str] = None,
                 proxy_type: str = "http",  # http, https, socks4, socks5
                 test_url: str = "https://httpbin.org/ip",
                 rotation_strategy: str = "round_robin",  # round_robin, random, performance, sticky_session
                 max_failures: int = 3,
                 auto_refresh: bool = False,
                 refresh_interval: int = 3600,  # seconds
                 timeout: int = 10,
                 proxy_api_url: Optional[str] = None,
                 proxy_api_key: Optional[str] = None):
        """
        Initialize the proxy rotation manager.
        
        Args:
            proxy_list: List of proxy URLs
            proxy_file: Path to file containing proxy URLs (one per line)
            proxy_type: Type of proxy (http, https, socks4, socks5)
            test_url: URL to test proxy connectivity
            rotation_strategy: Strategy for rotating proxies
            max_failures: Max consecutive failures before removing a proxy
            auto_refresh: Whether to automatically refresh proxy list
            refresh_interval: Interval for proxy list refresh (in seconds)
            timeout: Connection timeout for proxy testing
            proxy_api_url: URL for proxy API service (if using)
            proxy_api_key: API key for proxy service (if using)
        """
        self.proxies = []  # List of available proxies
        self.dead_proxies = []  # List of non-working proxies
        self.proxy_type = proxy_type
        self.test_url = test_url
        self.rotation_strategy = rotation_strategy
        self.max_failures = max_failures
        self.auto_refresh = auto_refresh
        self.refresh_interval = refresh_interval
        self.timeout = timeout
        self.proxy_api_url = proxy_api_url
        self.proxy_api_key = proxy_api_key
        
        # Dictionary to track proxy performance and failure counts
        self.proxy_stats = {}
        
        # Set up proxy request session
        self.session = requests.Session()
        
        # Used for round-robin rotation
        self.current_index = 0
        
        # Last refresh timestamp
        self.last_refresh = time.time()
        
        # Set up proxies from provided sources
        self._setup_proxies(proxy_list, proxy_file)
        
        # Test all proxies on initialization
        if self.proxies:
            self._test_all_proxies()
        
        logger.info(f"Proxy rotation manager initialized with {len(self.proxies)} proxies")
        
        # Start background refresh task if auto_refresh is enabled
        if self.auto_refresh:
            self._start_refresh_task()
    
    def _setup_proxies(self, proxy_list: Optional[List[str]], proxy_file: Optional[str]):
        """
        Set up proxies from the provided sources.
        
        Args:
            proxy_list: List of proxy URLs
            proxy_file: Path to file containing proxy URLs
        """
        # Add proxies from list if provided
        if proxy_list:
            self.add_proxies(proxy_list)
        
        # Add proxies from file if provided
        if proxy_file and os.path.exists(proxy_file):
            try:
                with open(proxy_file, 'r') as f:
                    file_proxies = [line.strip() for line in f if line.strip()]
                self.add_proxies(file_proxies)
            except Exception as e:
                logger.error(f"Error loading proxies from file: {str(e)}")
        
        # Add proxies from API if provided
        if self.proxy_api_url and self.proxy_api_key:
            self._fetch_proxies_from_api()
    
    def _format_proxy_url(self, proxy: str) -> str:
        """
        Format proxy URL to ensure it includes correct protocol.
        
        Args:
            proxy: Proxy URL string
            
        Returns:
            Formatted proxy URL
        """
        # Check if proxy already has protocol
        if proxy.startswith(('http://', 'https://', 'socks4://', 'socks5://')):
            return proxy
        
        # Add protocol based on proxy_type
        if self.proxy_type == "http":
            return f"http://{proxy}"
        elif self.proxy_type == "https":
            return f"https://{proxy}"
        elif self.proxy_type == "socks4":
            return f"socks4://{proxy}"
        elif self.proxy_type == "socks5":
            return f"socks5://{proxy}"
        else:
            return f"http://{proxy}"
    
    def add_proxies(self, proxies: List[str]):
        """
        Add proxies to the rotation pool.
        
        Args:
            proxies: List of proxy URLs to add
        """
        for proxy in proxies:
            formatted_proxy = self._format_proxy_url(proxy)
            if formatted_proxy not in self.proxies and formatted_proxy not in self.dead_proxies:
                self.proxies.append(formatted_proxy)
                self.proxy_stats[formatted_proxy] = {
                    "success_count": 0,
                    "failure_count": 0,
                    "response_times": [],
                    "last_used": None,
                    "last_success": None
                }
    
    def _test_proxy(self, proxy: str) -> bool:
        """
        Test if a proxy is working.
        
        Args:
            proxy: Proxy URL to test
            
        Returns:
            True if proxy is working, False otherwise
        """
        try:
            # Create proxies dict for requests
            proxies = {
                "http": proxy,
                "https": proxy
            }
            
            # Record start time for performance tracking
            start_time = time.time()
            
            # Make request to test URL
            response = self.session.get(
                self.test_url,
                proxies=proxies,
                timeout=self.timeout
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update proxy stats
            if response.status_code == 200:
                self.proxy_stats[proxy]["success_count"] += 1
                self.proxy_stats[proxy]["failure_count"] = 0
                self.proxy_stats[proxy]["last_success"] = time.time()
                self.proxy_stats[proxy]["response_times"].append(response_time)
                
                # Keep only the last 10 response times
                if len(self.proxy_stats[proxy]["response_times"]) > 10:
                    self.proxy_stats[proxy]["response_times"].pop(0)
                
                return True
            else:
                self.proxy_stats[proxy]["failure_count"] += 1
                return False
                
        except Exception as e:
            # Mark as failure
            self.proxy_stats[proxy]["failure_count"] += 1
            logger.debug(f"Proxy test failed for {proxy}: {str(e)}")
            return False
    
    def _test_all_proxies(self):
        """Test all proxies and remove non-working ones."""
        working_proxies = []
        for proxy in self.proxies:
            if self._test_proxy(proxy):
                working_proxies.append(proxy)
            else:
                self.dead_proxies.append(proxy)
                logger.debug(f"Removed non-working proxy: {proxy}")
        
        self.proxies = working_proxies
        logger.info(f"Proxy testing complete. {len(self.proxies)} working, {len(self.dead_proxies)} dead")
    
    def _fetch_proxies_from_api(self):
        """Fetch proxies from API service if configured."""
        if not self.proxy_api_url:
            return
        
        try:
            headers = {}
            if self.proxy_api_key:
                headers["Authorization"] = f"Bearer {self.proxy_api_key}"
            
            response = requests.get(
                self.proxy_api_url,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                # Parse response based on common proxy API formats
                try:
                    data = response.json()
                    
                    # Handle different API response formats
                    if isinstance(data, list):
                        # Simple list of proxies
                        self.add_proxies(data)
                    elif isinstance(data, dict) and "data" in data:
                        # Format: {"data": [...proxies...]}
                        if isinstance(data["data"], list):
                            self.add_proxies(data["data"])
                    elif isinstance(data, dict) and "proxies" in data:
                        # Format: {"proxies": [...proxies...]}
                        if isinstance(data["proxies"], list):
                            self.add_proxies(data["proxies"])
                    
                    logger.info(f"Fetched {len(self.proxies)} proxies from API")
                except Exception as e:
                    logger.error(f"Error parsing proxy API response: {str(e)}")
            else:
                logger.error(f"Proxy API returned status code {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching proxies from API: {str(e)}")
    
    def _start_refresh_task(self):
        """Start background thread for proxy list refreshing."""
        def refresh_task():
            while self.auto_refresh:
                # Sleep until next refresh
                time.sleep(10)  # Check every 10 seconds
                
                # Check if it's time to refresh
                if time.time() - self.last_refresh >= self.refresh_interval:
                    logger.info("Auto-refreshing proxy list")
                    self._fetch_proxies_from_api()
                    self._test_all_proxies()
                    self.last_refresh = time.time()
        
        # Start thread
        thread = threading.Thread(target=refresh_task, daemon=True)
        thread.start()
    
    def get_proxy(self, url: Optional[str] = None) -> Optional[str]:
        """
        Get a proxy based on the rotation strategy.
        
        Args:
            url: Optional URL for context-aware proxy selection
            
        Returns:
            Proxy URL or None if no proxies available
        """
        # Return None if no proxies available
        if not self.proxies:
            return None
        
        # Retest dead proxies occasionally
        self._maybe_retest_dead_proxies()
        
        # Select proxy based on strategy
        if self.rotation_strategy == "round_robin":
            return self._get_round_robin_proxy()
        elif self.rotation_strategy == "random":
            return self._get_random_proxy()
        elif self.rotation_strategy == "performance":
            return self._get_performance_based_proxy()
        elif self.rotation_strategy == "sticky_session":
            return self._get_sticky_session_proxy(url)
        else:
            # Default to round robin
            return self._get_round_robin_proxy()
    
    def _get_round_robin_proxy(self) -> str:
        """
        Get proxy using round-robin strategy.
        
        Returns:
            Proxy URL
        """
        # Get proxy and update index
        proxy = self.proxies[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.proxies)
        
        # Update stats
        self.proxy_stats[proxy]["last_used"] = time.time()
        
        return proxy
    
    def _get_random_proxy(self) -> str:
        """
        Get random proxy.
        
        Returns:
            Random proxy URL
        """
        proxy = random.choice(self.proxies)
        
        # Update stats
        self.proxy_stats[proxy]["last_used"] = time.time()
        
        return proxy
    
    def _get_performance_based_proxy(self) -> str:
        """
        Get proxy based on performance metrics.
        
        Returns:
            Best performing proxy URL
        """
        # Calculate average response time for each proxy
        proxy_scores = []
        
        for proxy in self.proxies:
            stats = self.proxy_stats[proxy]
            
            # Skip proxies with no successful requests
            if stats["success_count"] == 0:
                continue
            
            # Calculate average response time
            avg_response_time = 0
            if stats["response_times"]:
                avg_response_time = sum(stats["response_times"]) / len(stats["response_times"])
            
            # Calculate score (lower is better)
            # Score is based on:
            # - Average response time (lower is better)
            # - Success/failure ratio (higher is better)
            # - Time since last use (higher is better, to distribute load)
            
            success_ratio = stats["success_count"] / max(1, stats["success_count"] + stats["failure_count"])
            time_since_last_use = time.time() - (stats["last_used"] or 0) if stats["last_used"] else 1000
            
            # Normalize time since last use (0-1 scale, higher is better)
            normalized_time = min(time_since_last_use / 3600, 1.0)  # Cap at 1 hour
            
            # Calculate final score (lower is better)
            score = avg_response_time * 0.6 - success_ratio * 0.3 - normalized_time * 0.1
            
            proxy_scores.append((proxy, score))
        
        # If no proxy has been used successfully, fall back to random
        if not proxy_scores:
            return self._get_random_proxy()
        
        # Sort by score (lower is better)
        proxy_scores.sort(key=lambda x: x[1])
        
        # Get best proxy
        best_proxy = proxy_scores[0][0]
        
        # Update stats
        self.proxy_stats[best_proxy]["last_used"] = time.time()
        
        return best_proxy
    
    def _get_sticky_session_proxy(self, url: Optional[str]) -> str:
        """
        Get proxy using sticky session strategy (same proxy for same domain).
        
        Args:
            url: URL to get proxy for
            
        Returns:
            Proxy URL
        """
        if not url:
            return self._get_random_proxy()
        
        # Extract domain from URL
        try:
            domain = urlparse(url).netloc
        except:
            domain = None
        
        if not domain:
            return self._get_random_proxy()
        
        # Use domain hash to consistently select the same proxy
        hash_value = int(hashlib.md5(domain.encode()).hexdigest(), 16)
        proxy_index = hash_value % len(self.proxies)
        
        proxy = self.proxies[proxy_index]
        
        # Update stats
        self.proxy_stats[proxy]["last_used"] = time.time()
        
        return proxy
    
    def _maybe_retest_dead_proxies(self):
        """Occasionally retest dead proxies to see if they're working again."""
        # Only check dead proxies 5% of the time to reduce overhead
        if random.random() < 0.05 and self.dead_proxies:
            # Select a random dead proxy to test
            proxy_to_test = random.choice(self.dead_proxies)
            
            if self._test_proxy(proxy_to_test):
                # Proxy is working again, move it back to active list
                self.dead_proxies.remove(proxy_to_test)
                self.proxies.append(proxy_to_test)
                logger.info(f"Reactivated previously dead proxy: {proxy_to_test}")
    
    def report_success(self, proxy: str):
        """
        Report successful use of a proxy.
        
        Args:
            proxy: Proxy URL that was successful
        """
        if proxy in self.proxy_stats:
            self.proxy_stats[proxy]["success_count"] += 1
            self.proxy_stats[proxy]["failure_count"] = 0
            self.proxy_stats[proxy]["last_success"] = time.time()
    
    def report_failure(self, proxy: str):
        """
        Report failed use of a proxy.
        
        Args:
            proxy: Proxy URL that failed
        """
        if proxy in self.proxy_stats:
            self.proxy_stats[proxy]["failure_count"] += 1
            
            # Check if proxy should be removed
            if self.proxy_stats[proxy]["failure_count"] >= self.max_failures:
                if proxy in self.proxies:
                    self.proxies.remove(proxy)
                    self.dead_proxies.append(proxy)
                    logger.info(f"Removed failing proxy: {proxy}")
    
    def get_proxy_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all proxies.
        
        Returns:
            Dictionary of proxy statistics
        """
        return self.proxy_stats
    
    def get_proxy_count(self) -> Dict[str, int]:
        """
        Get count of active and dead proxies.
        
        Returns:
            Dictionary with counts
        """
        return {
            "active": len(self.proxies),
            "dead": len(self.dead_proxies)
        }
    
    def clear_all_proxies(self):
        """Remove all proxies from both active and dead lists."""
        self.proxies = []
        self.dead_proxies = []
        self.proxy_stats = {}
        logger.info("Cleared all proxies")

class CaptchaDetector:
    """
    Detects and helps handle CAPTCHA challenges in responses.
    
    This class provides CAPTCHA detection and handling capabilities:
    - Detecting common CAPTCHA implementations
    - Extracting CAPTCHA images
    - Identifying CAPTCHA redirect patterns
    - Providing strategies for CAPTCHA handling
    """
    
    def __init__(self):
        """Initialize the CAPTCHA detector."""
        # Common CAPTCHA keywords
        self.captcha_keywords = [
            "captcha", "recaptcha", "hcaptcha", "solve captcha", "verification", 
            "verify you're not a robot", "are you a robot", "prove you are human",
            "security check", "human verification", "robot check", "bot check",
            "automated access", "suspicious activity", "verify your identity"
        ]
        
        # Common CAPTCHA providers
        self.captcha_providers = [
            "recaptcha", "hcaptcha", "arkose", "solvemedia", "funcaptcha",
            "kasada", "cloudflare", "imperva", "akamai"
        ]
        
        # Regex patterns for detecting CAPTCHAs
        self.captcha_patterns = [
            r'(?i)<div[^>]*class=["\']recaptcha|g-recaptcha["\']',
            r'(?i)<iframe[^>]*recaptcha|captcha|verification',
            r'(?i)captcha.*?required',
            r'(?i)verify.*?human',
            r'(?i)human.*?check',
            r'(?i)security.*?check',
            r'(?i)suspicious.*?activity',
        ]
        
        # Compiled regex patterns
        self.compiled_patterns = [re.compile(pattern) for pattern in self.captcha_patterns]
        
        # CAPTCHA image detection patterns
        self.image_patterns = [
            r'(?i)<img[^>]*captcha[^>]*>',
            r'(?i)src=["\'][^"\']*captcha[^"\']*["\']'
        ]
        
        # Compiled image patterns
        self.compiled_image_patterns = [re.compile(pattern) for pattern in self.image_patterns]
        
        logger.info("CAPTCHA detector initialized")
    
    def detect_captcha(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect if a response contains a CAPTCHA challenge.
        
        Args:
            response: Response dictionary
            
        Returns:
            Detection results
        """
        result = {
            "has_captcha": False,
            "confidence": 0,
            "provider": None,
            "type": None,
            "image_url": None,
            "form_data": None,
            "evidence": []
        }
        
        # Get response content
        content = response.get("content", "")
        
        # Check status code (common CAPTCHA status codes)
        status_code = response.get("status_code", 200)
        if status_code in [403, 429, 503]:
            result["confidence"] += 10
            result["evidence"].append(f"Suspicious status code: {status_code}")
        
        # Check for CAPTCHA keywords in content
        for keyword in self.captcha_keywords:
            if keyword.lower() in content.lower():
                result["confidence"] += 15
                result["evidence"].append(f"CAPTCHA keyword found: {keyword}")
        
        # Check for CAPTCHA providers
        for provider in self.captcha_providers:
            if provider.lower() in content.lower():
                result["confidence"] += 20
                result["provider"] = provider
                result["evidence"].append(f"CAPTCHA provider found: {provider}")
        
        # Check for regex patterns
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(content):
                result["confidence"] += 25
                result["evidence"].append(f"CAPTCHA pattern {i+1} matched")
        
        # Look for CAPTCHA images
        for i, pattern in enumerate(self.compiled_image_patterns):
            match = pattern.search(content)
            if match:
                result["confidence"] += 20
                result["type"] = "image"
                
                # Try to extract image URL
                img_match = re.search(r'src=["\']([^"\']*captcha[^"\']*)["\']', content)
                if img_match:
                    result["image_url"] = img_match.group(1)
                
                result["evidence"].append(f"CAPTCHA image pattern {i+1} matched")
        
        # Check for ReCAPTCHA specific patterns
        if "g-recaptcha" in content or "google.com/recaptcha" in content:
            result["confidence"] += 30
            result["provider"] = "recaptcha"
            result["type"] = "recaptcha"
            
            # Try to extract site key
            site_key_match = re.search(r'data-sitekey=["\']([^"\']*)["\']', content)
            if site_key_match:
                result["site_key"] = site_key_match.group(1)
                result["evidence"].append(f"ReCAPTCHA site key found: {result['site_key']}")
            
            result["evidence"].append("ReCAPTCHA specific patterns found")
        
        # Check for hCaptcha specific patterns
        if "hcaptcha.com" in content or "h-captcha" in content:
            result["confidence"] += 30
            result["provider"] = "hcaptcha"
            result["type"] = "hcaptcha"
            
            # Try to extract site key
            site_key_match = re.search(r'data-sitekey=["\']([^"\']*)["\']', content)
            if site_key_match:
                result["site_key"] = site_key_match.group(1)
                result["evidence"].append(f"hCaptcha site key found: {result['site_key']}")
            
            result["evidence"].append("hCaptcha specific patterns found")
        
        # Check for Cloudflare CAPTCHA/challenge
        if "cloudflare" in content and (
            "challenge-form" in content or "cf_captcha_kind" in content or "jschl-answer" in content
        ):
            result["confidence"] += 40
            result["provider"] = "cloudflare"
            result["type"] = "challenge"
            result["evidence"].append("Cloudflare challenge detected")
        
        # Set has_captcha based on confidence
        if result["confidence"] >= 30:
            result["has_captcha"] = True
        
        # Try to extract form data if CAPTCHA detected
        if result["has_captcha"]:
            form_data = self._extract_form_data(content)
            if form_data:
                result["form_data"] = form_data
        
        return result
    
    def _extract_form_data(self, content: str) -> Optional[Dict[str, str]]:
        """
        Extract form data from CAPTCHA page.
        
        Args:
            content: HTML content
            
        Returns:
            Dictionary of form fields or None
        """
        try:
            # Look for forms
            form_match = re.search(r'<form[^>]*>(.*?)</form>', content, re.DOTALL)
            if not form_match:
                return None
            
            form_content = form_match.group(1)
            
            # Extract input fields
            input_fields = re.findall(r'<input[^>]*>', form_content)
            
            # Extract name and value from each input
            form_data = {}
            for field in input_fields:
                name_match = re.search(r'name=["\']([^"\']*)["\']', field)
                value_match = re.search(r'value=["\']([^"\']*)["\']', field)
                
                if name_match:
                    name = name_match.group(1)
                    value = value_match.group(1) if value_match else ""
                    form_data[name] = value
            
            return form_data
        
        except Exception as e:
            logger.warning(f"Error extracting form data: {str(e)}")
            return None
    
    async def extract_captcha_image(self, image_url: str, response: Dict[str, Any]) -> Optional[bytes]:
        """
        Extract and decode CAPTCHA image.
        
        Args:
            image_url: URL of CAPTCHA image
            response: Original response dictionary
            
        Returns:
            Image bytes or None
        """
        try:
            # Handle relative URLs
            if image_url.startswith('/'):
                parsed_url = urlparse(response.get("url", ""))
                base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                image_url = f"{base_url}{image_url}"
            
            # Handle data URLs (base64 encoded images)
            if image_url.startswith('data:image'):
                data_match = re.match(r'data:image/[^;]+;base64,(.+)', image_url)
                if data_match:
                    # Decode base64 image
                    return base64.b64decode(data_match.group(1))
            
            # Make request to get image
            async with httpx.AsyncClient() as client:
                img_response = await client.get(image_url)
                
                if img_response.status_code == 200:
                    return img_response.content
            
            return None
        
        except Exception as e:
            logger.warning(f"Error extracting CAPTCHA image: {str(e)}")
            return None
    
    def get_captcha_handling_strategy(self, detection_result: Dict[str, Any]) -> str:
        """
        Get a strategy for handling the detected CAPTCHA.
        
        Args:
            detection_result: CAPTCHA detection results
            
        Returns:
            Strategy recommendation string
        """
        provider = detection_result.get("provider")
        captcha_type = detection_result.get("type")
        
        if not detection_result.get("has_captcha"):
            return "No CAPTCHA detected, proceed normally"
        
        # Provider-specific strategies
        if provider == "recaptcha":
            return "ReCAPTCHA detected. Consider using 2captcha/anti-captcha service or switching IP/user-agent"
        
        elif provider == "hcaptcha":
            return "hCaptcha detected. Consider using 2captcha/anti-captcha service or switching IP/user-agent"
        
        elif provider == "cloudflare":
            return "Cloudflare challenge detected. Wait and retry with a different IP or try using a specialized Cloudflare bypass library"
        
        # Type-specific strategies
        if captcha_type == "image":
            return "Image CAPTCHA detected. Consider using OCR service or manual solver"
        
        # Generic strategy
        return "CAPTCHA detected. Consider rotating IP, waiting before retry, or using a CAPTCHA solving service"

class SessionManager:
    """
    Manages HTTP sessions with persistent cookies and fingerprinting.
    
    This class provides session management for HTTP requests, including:
    - Persistent cookie storage
    - Session fingerprinting
    - Domain-specific session isolation
    - Automatic request signature randomization
    """
    
    def __init__(self, 
                 cookie_dir: str = ".cookies", 
                 max_sessions: int = 10,
                 user_agent_rotation: bool = True,
                 fingerprint_variation: bool = True):
        """
        Initialize the session manager.
        
        Args:
            cookie_dir: Directory to store cookies
            max_sessions: Maximum number of concurrent sessions
            user_agent_rotation: Whether to rotate user agents
            fingerprint_variation: Whether to vary fingerprints
        """
        self.cookie_dir = cookie_dir
        self.max_sessions = max_sessions
        self.user_agent_rotation = user_agent_rotation
        self.fingerprint_variation = fingerprint_variation
        
        # Create cookie directory if it doesn't exist
        os.makedirs(cookie_dir, exist_ok=True)
        
        # Initialize sessions
        self.sessions: Dict[str, Any] = {}
        self.session_cookies: Dict[str, Dict[str, Any]] = {}
        self.session_fingerprints: Dict[str, Dict[str, Any]] = {}
        
        # Initialize user agent generator
        try:
            self.ua = UserAgent()
        except Exception as e:
            logger.warning(f"Could not initialize UserAgent: {str(e)}. Using fallback.")
            self.ua = None
        
        # Initialize cookie persistence
        self.cookie_expiry: Dict[str, datetime] = {}
        
        # Create domain-specific session attributes
        self.domain_headers: Dict[str, Dict[str, str]] = {}
        
        # Create shared semaphore for rate limiting
        self.semaphore = asyncio.Semaphore(max_sessions)
        
        logger.info(f"Session manager initialized with cookie directory: {cookie_dir}")
    
    def _get_session_key(self, url: str, session_id: Optional[str] = None) -> str:
        """
        Get a unique session key for a URL.
        
        Args:
            url: URL to get session key for
            session_id: Optional explicit session ID
            
        Returns:
            Session key string
        """
        if session_id:
            return session_id
        
        # Extract domain as the default session key
        domain = urlparse(url).netloc
        return domain
    
    async def get_session(self, url: str, session_id: Optional[str] = None) -> httpx.AsyncClient:
        """
        Get or create a session for the given URL.
        
        Args:
            url: URL to get session for
            session_id: Optional explicit session ID
            
        Returns:
            httpx.AsyncClient session
        """
        session_key = self._get_session_key(url, session_id)
        
        # Check if session exists and is valid
        if session_key in self.sessions:
            return self.sessions[session_key]
        
        # Create new session with cookies
        logger.debug(f"Creating new session for {session_key}")
        
        # Load cookies if they exist
        cookies = await self._load_cookies(session_key)
        
        # Generate headers
        headers = await self._generate_headers(url, session_key)
        
        # Create new httpx client
        timeout = httpx.Timeout(30.0, connect=10.0)
        session = httpx.AsyncClient(
            cookies=cookies,
            headers=headers,
            timeout=timeout,
            follow_redirects=True
        )
        
        # Store session
        self.sessions[session_key] = session
        
        return session
    
    async def get_aiohttp_session(self, url: str, session_id: Optional[str] = None) -> aiohttp.ClientSession:
        """
        Get or create an aiohttp session for the given URL.
        
        Args:
            url: URL to get session for
            session_id: Optional explicit session ID
            
        Returns:
            aiohttp.ClientSession session
        """
        session_key = self._get_session_key(url, session_id)
        
        # Check if session exists
        if session_key in self.sessions and isinstance(self.sessions[session_key], aiohttp.ClientSession):
            return self.sessions[session_key]
        
        # Create new session
        logger.debug(f"Creating new aiohttp session for {session_key}")
        
        # Load cookies if they exist
        cookies = await self._load_cookies(session_key)
        
        # Convert cookies to aiohttp format if needed
        if cookies and not isinstance(cookies, aiohttp.CookieJar):
            cookie_jar = aiohttp.CookieJar()
            for name, value in cookies.items():
                cookie_jar.update_cookies({name: value})
            cookies = cookie_jar
        
        # Generate headers
        headers = await self._generate_headers(url, session_key)
        
        # Create new aiohttp session
        session = aiohttp.ClientSession(
            headers=headers,
            cookies=cookies
        )
        
        # Store session
        self.sessions[session_key] = session
        
        return session
    
    async def _generate_headers(self, url: str, session_key: str) -> Dict[str, str]:
        """
        Generate headers for a session.
        
        Args:
            url: URL to generate headers for
            session_key: Session key
            
        Returns:
            Dictionary of headers
        """
        # Check if we have cached headers for this domain
        domain = urlparse(url).netloc
        if domain in self.domain_headers:
            return self.domain_headers[domain].copy()
        
        # Generate new headers
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
        }
        
        # Add User-Agent
        if self.user_agent_rotation and self.ua:
            try:
                headers["User-Agent"] = self.ua.random
            except Exception:
                headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        else:
            headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        
        # Add fingerprint variation
        if self.fingerprint_variation:
            if random.random() < 0.5:
                headers["Sec-Fetch-User"] = "?1"
            if random.random() < 0.3:
                headers["DNT"] = "1"
        
        # Store headers for this domain
        self.domain_headers[domain] = headers.copy()
        
        return headers
    
    async def _load_cookies(self, session_key: str) -> Dict[str, str]:
        """
        Load cookies for a session from disk.
        
        Args:
            session_key: Session key
            
        Returns:
            Dictionary of cookies
        """
        cookie_file = os.path.join(self.cookie_dir, f"{session_key}.json")
        
        try:
            if os.path.exists(cookie_file):
                async with aiofiles.open(cookie_file, 'r') as f:
                    cookie_data = await f.read()
                    cookies = json.loads(cookie_data)
                    logger.debug(f"Loaded cookies for {session_key}")
                    return cookies
        except Exception as e:
            logger.warning(f"Error loading cookies for {session_key}: {str(e)}")
        
        return {}
    
    async def save_cookies(self, url: str, session_id: Optional[str] = None) -> None:
        """
        Save cookies for a session to disk.
        
        Args:
            url: URL to save cookies for
            session_id: Optional explicit session ID
        """
        session_key = self._get_session_key(url, session_id)
        
        if session_key not in self.sessions:
            logger.warning(f"No session found for {session_key}")
            return
        
        session = self.sessions[session_key]
        cookie_file = os.path.join(self.cookie_dir, f"{session_key}.json")
        
        try:
            # Extract cookies based on session type
            if isinstance(session, httpx.AsyncClient):
                cookies = dict(session.cookies)
            elif isinstance(session, aiohttp.ClientSession):
                cookies = {}
                for cookie in session.cookie_jar:
                    cookies[cookie.key] = cookie.value
            else:
                logger.warning(f"Unknown session type for {session_key}")
                return
            
            # Save cookies to file
            async with aiofiles.open(cookie_file, 'w') as f:
                await f.write(json.dumps(cookies))
                
            logger.debug(f"Saved cookies for {session_key}")
            
        except Exception as e:
            logger.error(f"Error saving cookies for {session_key}: {str(e)}")
    
    async def close_session(self, url: str, session_id: Optional[str] = None) -> None:
        """
        Close a session and save its cookies.
        
        Args:
            url: URL to close session for
            session_id: Optional explicit session ID
        """
        session_key = self._get_session_key(url, session_id)
        
        if session_key not in self.sessions:
            return
        
        # Save cookies before closing
        await self.save_cookies(url, session_id)
        
        # Close session
        session = self.sessions[session_key]
        
        try:
            await session.aclose()
            logger.debug(f"Closed session for {session_key}")
        except Exception as e:
            logger.warning(f"Error closing session for {session_key}: {str(e)}")
        
        # Remove from sessions
        del self.sessions[session_key]
    
    async def close_all_sessions(self) -> None:
        """Close all sessions and save their cookies."""
        for session_key, session in list(self.sessions.items()):
            try:
                # Try to save cookies
                cookie_file = os.path.join(self.cookie_dir, f"{session_key}.json")
                
                # Extract cookies based on session type
                if isinstance(session, httpx.AsyncClient):
                    cookies = dict(session.cookies)
                elif isinstance(session, aiohttp.ClientSession):
                    cookies = {}
                    for cookie in session.cookie_jar:
                        cookies[cookie.key] = cookie.value
                else:
                    cookies = {}
                
                # Save cookies to file if we have any
                if cookies:
                    async with aiofiles.open(cookie_file, 'w') as f:
                        await f.write(json.dumps(cookies))
                
                # Close session
                await session.aclose()
                logger.debug(f"Closed session for {session_key}")
            except Exception as e:
                logger.warning(f"Error closing session for {session_key}: {str(e)}")
        
        # Clear sessions
        self.sessions.clear()
        logger.info("Closed all sessions")

class RequestManager:
    """
    Manages HTTP requests with advanced features.
    
    This class provides advanced HTTP request capabilities, including:
    - Rate limiting
    - Retry with exponential backoff
    - Cookie management
    - Request fingerprinting
    - Response caching
    """
    
    def __init__(self, 
                 session_manager: Optional[SessionManager] = None,
                 rate_limit: float = 1.0,
                 cache_dir: Optional[str] = ".cache",
                 cache_time: int = 3600,
                 proxy_list: Optional[List[str]] = None):
        """
        Initialize the request manager.
        
        Args:
            session_manager: SessionManager instance to use
            rate_limit: Requests per second limit
            cache_dir: Directory to cache responses, None to disable
            cache_time: Cache expiry time in seconds
            proxy_list: List of proxy URLs to use
        """
        self.session_manager = session_manager or SessionManager()
        self.rate_limit = rate_limit
        self.cache_dir = cache_dir
        self.cache_time = cache_time
        self.proxy_list = proxy_list or []
        
        # Create cache directory if needed
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Create rate limiting mechanisms
        self.last_request_time: Dict[str, float] = {}
        self.domain_rate_limits: Dict[str, float] = {}
        
        # Create proxy rotation mechanism
        self.current_proxy_index = 0
        self.proxy_performance: Dict[str, Dict[str, Union[int, float]]] = {}
        
        logger.info("Request manager initialized")
    
    def _get_domain(self, url: str) -> str:
        """
        Extract domain from URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain string
        """
        return urlparse(url).netloc
    
    def _get_cache_key(self, url: str, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a cache key for a request.
        
        Args:
            url: Request URL
            params: Request parameters
            data: Request data
            
        Returns:
            Cache key string
        """
        # Create key from URL and parameters
        key_parts = [url]
        
        if params:
            key_parts.append(json.dumps(params, sort_keys=True))
        
        if data:
            key_parts.append(json.dumps(data, sort_keys=True))
        
        # Generate hash of the key parts
        key = hashlib.md5(''.join(key_parts).encode()).hexdigest()
        return key
    
    async def _check_cache(self, url: str, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Check if a response is cached.
        
        Args:
            url: Request URL
            params: Request parameters
            data: Request data
            
        Returns:
            Cached response or None
        """
        if not self.cache_dir:
            return None
        
        # Get cache key
        cache_key = self._get_cache_key(url, params, data)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        # Check if cache file exists
        if not os.path.exists(cache_file):
            return None
        
        try:
            # Check if cache is expired
            cache_time = os.path.getmtime(cache_file)
            if time.time() - cache_time > self.cache_time:
                return None
            
            # Load cache file
            async with aiofiles.open(cache_file, 'r') as f:
                cache_data = await f.read()
                response = json.loads(cache_data)
                
                # Add cache metadata
                response['_cache'] = {
                    'hit': True,
                    'time': cache_time,
                    'age': time.time() - cache_time
                }
                
                return response
        except Exception as e:
            logger.warning(f"Error reading cache for {url}: {str(e)}")
            return None
    
    async def _write_cache(self, url: str, response: Dict[str, Any], params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Write a response to cache.
        
        Args:
            url: Request URL
            response: Response to cache
            params: Request parameters
            data: Request data
        """
        if not self.cache_dir:
            return
        
        # Get cache key
        cache_key = self._get_cache_key(url, params, data)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            # Create a copy of the response to cache
            cache_response = response.copy()
            
            # Remove cache metadata if present
            if '_cache' in cache_response:
                del cache_response['_cache']
            
            # Write to cache file
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(cache_response))
                
        except Exception as e:
            logger.warning(f"Error writing cache for {url}: {str(e)}")
    
    def _get_next_proxy(self) -> Optional[str]:
        """
        Get the next proxy from the rotation.
        
        Returns:
            Proxy URL or None
        """
        if not self.proxy_list:
            return None
        
        # Rotate to next proxy
        proxy = self.proxy_list[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_list)
        
        return proxy
    
    async def _wait_for_rate_limit(self, domain: str) -> None:
        """
        Wait for rate limit for a domain.
        
        Args:
            domain: Domain to wait for
        """
        # Get domain-specific rate limit
        rate_limit = self.domain_rate_limits.get(domain, self.rate_limit)
        
        # Calculate time to wait
        if domain in self.last_request_time:
            elapsed = time.time() - self.last_request_time[domain]
            wait_time = max(0, 1 / rate_limit - elapsed)
            
            if wait_time > 0:
                logger.debug(f"Rate limiting {domain}, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Update last request time
        self.last_request_time[domain] = time.time()
    
    @retry_on_network_errors
    @retry_on_http_errors
    async def get(self, 
                 url: str, 
                 params: Optional[Dict[str, Any]] = None, 
                 session_id: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None,
                 use_cache: bool = True,
                 use_proxy: bool = False,
                 verify: bool = True) -> Dict[str, Any]:
        """
        Make a GET request with advanced features.
        
        Args:
            url: URL to request
            params: Query parameters
            session_id: Optional session ID
            headers: Additional headers
            use_cache: Whether to use cache
            use_proxy: Whether to use proxy
            verify: Whether to verify SSL
            
        Returns:
            Response dictionary with content and metadata
        """
        domain = self._get_domain(url)
        
        # Check cache first
        if use_cache:
            cached = await self._check_cache(url, params)
            if cached:
                logger.debug(f"Cache hit for {url}")
                return cached
        
        # Wait for rate limit
        await self._wait_for_rate_limit(domain)
        
        # Get session
        session = await self.session_manager.get_session(url, session_id)
        
        # Prepare request options
        options = {
            "params": params,
            "verify": verify
        }
        
        # Merge headers if provided
        if headers:
            options["headers"] = headers
        
        # Add proxy if requested
        if use_proxy and self.proxy_list:
            proxy = self._get_next_proxy()
            if proxy:
                options["proxy"] = proxy
        
        # Make request
        try:
            start_time = time.time()
            response = await session.get(url, **options)
            response_time = time.time() - start_time
            
            # Save cookies
            await self.session_manager.save_cookies(url, session_id)
            
            # Create response dict
            content = await response.aread()
            result = {
                "status_code": response.status_code,
                "content": content.decode("utf-8", errors="replace"),
                "headers": dict(response.headers),
                "url": str(response.url),
                "response_time": response_time
            }
            
            # Cache successful responses
            if use_cache and 200 <= response.status_code < 300:
                await self._write_cache(url, result, params)
            
            return result
        
        except Exception as e:
            logger.error(f"Error making GET request to {url}: {str(e)}")
            raise
    
    @retry_on_network_errors
    @retry_on_http_errors
    async def post(self, 
                  url: str, 
                  data: Optional[Dict[str, Any]] = None,
                  json_data: Optional[Dict[str, Any]] = None,
                  session_id: Optional[str] = None,
                  headers: Optional[Dict[str, str]] = None,
                  use_cache: bool = False,
                  use_proxy: bool = False,
                  verify: bool = True) -> Dict[str, Any]:
        """
        Make a POST request with advanced features.
        
        Args:
            url: URL to request
            data: Form data
            json_data: JSON data
            session_id: Optional session ID
            headers: Additional headers
            use_cache: Whether to use cache
            use_proxy: Whether to use proxy
            verify: Whether to verify SSL
            
        Returns:
            Response dictionary with content and metadata
        """
        domain = self._get_domain(url)
        
        # Check cache first (if enabled for POST)
        if use_cache:
            cached = await self._check_cache(url, data=data or json_data)
            if cached:
                logger.debug(f"Cache hit for POST {url}")
                return cached
        
        # Wait for rate limit
        await self._wait_for_rate_limit(domain)
        
        # Get session
        session = await self.session_manager.get_session(url, session_id)
        
        # Prepare request options
        options = {
            "verify": verify
        }
        
        # Add data or json
        if data:
            options["data"] = data
        if json_data:
            options["json"] = json_data
        
        # Merge headers if provided
        if headers:
            options["headers"] = headers
        
        # Add proxy if requested
        if use_proxy and self.proxy_list:
            proxy = self._get_next_proxy()
            if proxy:
                options["proxy"] = proxy
        
        # Make request
        try:
            start_time = time.time()
            response = await session.post(url, **options)
            response_time = time.time() - start_time
            
            # Save cookies
            await self.session_manager.save_cookies(url, session_id)
            
            # Create response dict
            content = await response.aread()
            result = {
                "status_code": response.status_code,
                "content": content.decode("utf-8", errors="replace"),
                "headers": dict(response.headers),
                "url": str(response.url),
                "response_time": response_time
            }
            
            # Cache successful responses if enabled
            if use_cache and 200 <= response.status_code < 300:
                await self._write_cache(url, result, data=data or json_data)
            
            return result
        
        except Exception as e:
            logger.error(f"Error making POST request to {url}: {str(e)}")
            raise

class CookieJar:
    """
    Standalone cookie jar for managing cookies across domains.
    
    This class provides cookie management capabilities including:
    - Cookie storage and retrieval
    - Domain mapping
    - Automatic expiry
    - Cookie filtering
    """
    
    def __init__(self, cookie_dir: str = ".cookies", auto_save: bool = True):
        """
        Initialize the cookie jar.
        
        Args:
            cookie_dir: Directory to store cookies
            auto_save: Whether to automatically save cookies after changes
        """
        self.cookie_dir = cookie_dir
        self.auto_save = auto_save
        
        # Create cookie directory if needed
        os.makedirs(cookie_dir, exist_ok=True)
        
        # Initialize cookie store
        self.cookies: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Load cookies from disk
        self._load_cookies()
        
        logger.info(f"Cookie jar initialized with directory: {cookie_dir}")
    
    def _load_cookies(self) -> None:
        """Load all cookies from disk."""
        try:
            cookie_files = [f for f in os.listdir(self.cookie_dir) if f.endswith('.json')]
            
            for cookie_file in cookie_files:
                domain = cookie_file.replace('.json', '')
                file_path = os.path.join(self.cookie_dir, cookie_file)
                
                with open(file_path, 'r') as f:
                    domain_cookies = json.load(f)
                    self.cookies[domain] = domain_cookies
            
            logger.debug(f"Loaded cookies for {len(self.cookies)} domains")
        
        except Exception as e:
            logger.warning(f"Error loading cookies: {str(e)}")
    
    def save_cookies(self, domain: Optional[str] = None) -> None:
        """
        Save cookies to disk.
        
        Args:
            domain: Domain to save cookies for, or None for all
        """
        try:
            domains_to_save = [domain] if domain else list(self.cookies.keys())
            
            for d in domains_to_save:
                if d in self.cookies:
                    file_path = os.path.join(self.cookie_dir, f"{d}.json")
                    
                    with open(file_path, 'w') as f:
                        json.dump(self.cookies[d], f)
            
            logger.debug(f"Saved cookies for {len(domains_to_save)} domains")
        
        except Exception as e:
            logger.warning(f"Error saving cookies: {str(e)}")
    
    def get_domain_cookies(self, domain: str) -> Dict[str, Dict[str, Any]]:
        """
        Get cookies for a domain.
        
        Args:
            domain: Domain to get cookies for
            
        Returns:
            Domain cookies
        """
        # Clean expired cookies first
        self._clean_expired_cookies(domain)
        
        return self.cookies.get(domain, {})
    
    def set_cookie(self, domain: str, name: str, value: str, **kwargs) -> None:
        """
        Set a cookie.
        
        Args:
            domain: Cookie domain
            name: Cookie name
            value: Cookie value
            **kwargs: Additional cookie attributes
        """
        # Initialize domain if needed
        if domain not in self.cookies:
            self.cookies[domain] = {}
        
        # Set cookie with attributes
        self.cookies[domain][name] = {
            "value": value,
            "expires": kwargs.get("expires"),
            "path": kwargs.get("path", "/"),
            "secure": kwargs.get("secure", False),
            "httpOnly": kwargs.get("httpOnly", False),
            "created": int(time.time())
        }
        
        # Save cookies if auto_save enabled
        if self.auto_save:
            self.save_cookies(domain)
    
    def delete_cookie(self, domain: str, name: str) -> None:
        """
        Delete a cookie.
        
        Args:
            domain: Cookie domain
            name: Cookie name
        """
        if domain in self.cookies and name in self.cookies[domain]:
            del self.cookies[domain][name]
            
            # Save cookies if auto_save enabled
            if self.auto_save:
                self.save_cookies(domain)
    
    def clear_domain_cookies(self, domain: str) -> None:
        """
        Clear all cookies for a domain.
        
        Args:
            domain: Domain to clear cookies for
        """
        if domain in self.cookies:
            self.cookies[domain] = {}
            
            # Save cookies if auto_save enabled
            if self.auto_save:
                self.save_cookies(domain)
    
    def clear_all_cookies(self) -> None:
        """Clear all cookies."""
        self.cookies = {}
        
        # Save cookies if auto_save enabled
        if self.auto_save:
            for domain in list(self.cookies.keys()):
                self.save_cookies(domain)
    
    def _clean_expired_cookies(self, domain: Optional[str] = None) -> None:
        """
        Clean expired cookies.
        
        Args:
            domain: Domain to clean cookies for, or None for all
        """
        now = int(time.time())
        domains_to_clean = [domain] if domain else list(self.cookies.keys())
        
        for d in domains_to_clean:
            if d in self.cookies:
                # Find expired cookies
                expired = []
                for name, cookie in self.cookies[d].items():
                    expires = cookie.get("expires")
                    if expires and int(expires) < now:
                        expired.append(name)
                
                # Remove expired cookies
                for name in expired:
                    del self.cookies[d][name]
                
                if expired and self.auto_save:
                    self.save_cookies(d)

class CircuitBreaker:
    """
    Circuit breaker for domain-level failure handling.
    
    This class implements the circuit breaker pattern to prevent
    repeated requests to failing domains.
    """
    
    def __init__(self, 
                 threshold: int = 5, 
                 reset_timeout: int = 300,
                 half_open_timeout: int = 60):
        """
        Initialize the circuit breaker.
        
        Args:
            threshold: Number of failures before opening circuit
            reset_timeout: Time in seconds to reset circuit
            half_open_timeout: Time in seconds for half-open state
        """
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        
        # Circuit state storage
        self.circuits: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Circuit breaker initialized with threshold: {threshold}")
    
    def get_circuit_state(self, domain: str) -> Dict[str, Any]:
        """
        Get the circuit state for a domain.
        
        Args:
            domain: Domain to get state for
            
        Returns:
            Circuit state dictionary
        """
        # Initialize circuit if needed
        if domain not in self.circuits:
            self.circuits[domain] = {
                "state": "closed",
                "failures": 0,
                "last_failure": None,
                "last_success": None,
                "opened_at": None
            }
        
        return self.circuits[domain]
    
    def record_failure(self, domain: str) -> None:
        """
        Record a failure for a domain.
        
        Args:
            domain: Domain to record failure for
        """
        circuit = self.get_circuit_state(domain)
        now = time.time()
        
        # Record failure
        circuit["failures"] += 1
        circuit["last_failure"] = now
        
        # Check if we need to open the circuit
        if circuit["state"] == "closed" and circuit["failures"] >= self.threshold:
            logger.warning(f"Opening circuit for domain: {domain} after {circuit['failures']} failures")
            circuit["state"] = "open"
            circuit["opened_at"] = now
    
    def record_success(self, domain: str) -> None:
        """
        Record a success for a domain.
        
        Args:
            domain: Domain to record success for
        """
        circuit = self.get_circuit_state(domain)
        now = time.time()
        
        # Record success
        circuit["last_success"] = now
        
        # Reset the circuit if it was half-open
        if circuit["state"] == "half-open":
            logger.info(f"Closing circuit for domain: {domain} after successful request")
            circuit["state"] = "closed"
            circuit["failures"] = 0
            circuit["opened_at"] = None
    
    def is_allowed(self, domain: str) -> bool:
        """
        Check if a request is allowed for a domain.
        
        Args:
            domain: Domain to check
            
        Returns:
            Whether the request is allowed
        """
        circuit = self.get_circuit_state(domain)
        now = time.time()
        
        # Check circuit state
        if circuit["state"] == "open":
            # Check if we should transition to half-open
            if circuit["opened_at"] and now - circuit["opened_at"] > self.reset_timeout:
                logger.info(f"Moving circuit for domain: {domain} to half-open state")
                circuit["state"] = "half-open"
                return True
            
            return False
        
        elif circuit["state"] == "half-open":
            # Only allow one request during half-open state
            if circuit["last_success"] and now - circuit["last_success"] < self.half_open_timeout:
                return False
            
            return True
        
        # Always allow if closed
        return True
    
    def reset(self, domain: str) -> None:
        """
        Reset the circuit for a domain.
        
        Args:
            domain: Domain to reset
        """
        if domain in self.circuits:
            logger.info(f"Resetting circuit for domain: {domain}")
            self.circuits[domain] = {
                "state": "closed",
                "failures": 0,
                "last_failure": None,
                "last_success": None,
                "opened_at": None
            }

class RateLimiter:
    """
    Rate limiter for domain-level request throttling.
    
    This class provides rate limiting functionality to prevent
    overloading servers and triggering anti-bot mechanisms.
    """
    
    def __init__(self, 
                 default_rate: float = 1.0,
                 per_domain_rates: Optional[Dict[str, float]] = None):
        """
        Initialize the rate limiter.
        
        Args:
            default_rate: Default requests per second
            per_domain_rates: Domain-specific rates
        """
        self.default_rate = default_rate
        self.per_domain_rates = per_domain_rates or {}
        
        # Last request tracking
        self.last_request: Dict[str, float] = {}
        
        logger.info(f"Rate limiter initialized with default rate: {default_rate} req/s")
    
    def set_domain_rate(self, domain: str, rate: float) -> None:
        """
        Set rate limit for a domain.
        
        Args:
            domain: Domain to set rate for
            rate: Requests per second
        """
        self.per_domain_rates[domain] = rate
        logger.debug(f"Set rate limit for {domain} to {rate} req/s")
    
    async def wait(self, domain: str) -> None:
        """
        Wait for rate limit for a domain.
        
        Args:
            domain: Domain to wait for
        """
        # Get domain-specific rate
        rate = self.per_domain_rates.get(domain, self.default_rate)
        delay = 1.0 / rate
        
        # Check if we need to wait
        now = time.time()
        last = self.last_request.get(domain, 0)
        
        if now - last < delay:
            wait_time = delay - (now - last)
            
            # Add a small random variation to avoid patterns
            variation = random.uniform(-0.1, 0.1) * wait_time
            wait_time = max(0, wait_time + variation)
            
            if wait_time > 0:
                logger.debug(f"Rate limiting {domain}, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Update last request time
        self.last_request[domain] = time.time()

class AdvancedRateLimiter:
    """
    Advanced rate limiter with domain-specific controls.
    
    This class provides sophisticated rate limiting features including:
    - Per-domain rate limits with automatic adjustment
    - Concurrent request limiting
    - Adaptive rate limiting based on server response
    - Traffic shaping for more natural request patterns
    """
    
    def __init__(self, 
                 default_rate: float = 1.0,
                 max_rate: float = 10.0,
                 min_rate: float = 0.1,
                 per_domain_rates: Optional[Dict[str, float]] = None,
                 max_concurrent: int = 5,
                 use_dynamic_adjustment: bool = True):
        """
        Initialize the advanced rate limiter.
        
        Args:
            default_rate: Default requests per second rate
            max_rate: Maximum allowed rate in requests per second
            min_rate: Minimum allowed rate in requests per second
            per_domain_rates: Domain-specific rates dictionary
            max_concurrent: Maximum concurrent requests per domain
            use_dynamic_adjustment: Whether to use dynamic rate adjustment
        """
        self.default_rate = default_rate
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.per_domain_rates = per_domain_rates or {}
        self.max_concurrent = max_concurrent
        self.use_dynamic_adjustment = use_dynamic_adjustment
        
        # Track request timing
        self.last_request: Dict[str, float] = {}
        self.request_history: Dict[str, List[float]] = {}
        self.response_times: Dict[str, List[float]] = {}
        
        # Create concurrency control
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # Track rate adjustment factors
        self.adjustment_factors: Dict[str, float] = {}
        
        # Request success/failure tracking
        self.success_rates: Dict[str, float] = {}
        self.request_counters: Dict[str, Dict[str, int]] = {}
        
        logger.info(f"Advanced rate limiter initialized with default rate: {default_rate} req/s, max concurrent: {max_concurrent}")
    
    def get_semaphore(self, domain: str) -> asyncio.Semaphore:
        """
        Get or create a semaphore for a domain.
        
        Args:
            domain: Domain to get semaphore for
            
        Returns:
            Domain-specific semaphore
        """
        if domain not in self.semaphores:
            self.semaphores[domain] = asyncio.Semaphore(self.max_concurrent)
        
        return self.semaphores[domain]
    
    def get_current_rate(self, domain: str) -> float:
        """
        Get the current rate limit for a domain.
        
        Args:
            domain: Domain to get rate for
            
        Returns:
            Current rate limit in requests per second
        """
        # Get base rate from domain-specific settings or default
        base_rate = self.per_domain_rates.get(domain, self.default_rate)
        
        # Apply adjustment factor if dynamic adjustment enabled
        if self.use_dynamic_adjustment and domain in self.adjustment_factors:
            adjusted_rate = base_rate * self.adjustment_factors[domain]
            
            # Ensure rate is within allowed range
            return max(self.min_rate, min(self.max_rate, adjusted_rate))
        
        return base_rate
    
    def set_domain_rate(self, domain: str, rate: float) -> None:
        """
        Set rate limit for a domain.
        
        Args:
            domain: Domain to set rate for
            rate: Requests per second
        """
        # Ensure rate is within allowed range
        rate = max(self.min_rate, min(self.max_rate, rate))
        
        self.per_domain_rates[domain] = rate
        logger.debug(f"Set rate limit for {domain} to {rate} req/s")
    
    def update_adjustment_factor(self, domain: str, success: bool, response_time: Optional[float] = None) -> None:
        """
        Update the rate adjustment factor based on request success and response time.
        
        Args:
            domain: Domain to update
            success: Whether the request was successful
            response_time: Optional response time in seconds
        """
        if not self.use_dynamic_adjustment:
            return
        
        # Initialize adjustment factor if needed
        if domain not in self.adjustment_factors:
            self.adjustment_factors[domain] = 1.0
        
        # Initialize counters if needed
        if domain not in self.request_counters:
            self.request_counters[domain] = {
                "success": 0,
                "failure": 0,
                "total": 0
            }
        
        # Update request counters
        self.request_counters[domain]["total"] += 1
        if success:
            self.request_counters[domain]["success"] += 1
        else:
            self.request_counters[domain]["failure"] += 1
        
        # Calculate success rate
        total = self.request_counters[domain]["total"]
        if total > 0:
            self.success_rates[domain] = self.request_counters[domain]["success"] / total
        
        # Track response time
        if response_time is not None:
            if domain not in self.response_times:
                self.response_times[domain] = []
            
            # Keep last 10 response times
            self.response_times[domain].append(response_time)
            if len(self.response_times[domain]) > 10:
                self.response_times[domain].pop(0)
        
        # Adjust factor based on success/failure
        factor = self.adjustment_factors[domain]
        
        if success:
            # Successful request - potentially increase rate if consistently successful
            if self.success_rates.get(domain, 0) > 0.95 and total >= 10:
                # Gradually increase rate (more conservative)
                factor = min(factor * 1.05, 2.0)
        else:
            # Failed request - reduce rate significantly
            factor = max(factor * 0.5, 0.1)
        
        # Adjust based on response time trend
        if domain in self.response_times and len(self.response_times[domain]) >= 5:
            avg_response_time = sum(self.response_times[domain]) / len(self.response_times[domain])
            
            # If response times are increasing, slow down
            if response_time and response_time > avg_response_time * 1.5:
                factor = max(factor * 0.9, 0.1)
        
        # Update adjustment factor
        self.adjustment_factors[domain] = factor
        
        logger.debug(f"Updated rate adjustment factor for {domain} to {factor:.2f}")
    
    async def wait(self, domain: str) -> None:
        """
        Wait for rate limit for a domain with traffic shaping.
        
        Args:
            domain: Domain to wait for
        """
        # Get current rate
        rate = self.get_current_rate(domain)
        delay = 1.0 / rate
        
        # Check if we need to wait
        now = time.time()
        last = self.last_request.get(domain, 0)
        
        if now - last < delay:
            wait_time = delay - (now - last)
            
            # Add traffic shaping - random variation to avoid patterns
            variation = random.uniform(-0.1, 0.1) * wait_time
            wait_time = max(0, wait_time + variation)
            
            if wait_time > 0:
                logger.debug(f"Rate limiting {domain}, waiting {wait_time:.2f}s (rate: {rate:.2f} req/s)")
                await asyncio.sleep(wait_time)
        
        # Update last request time
        self.last_request[domain] = time.time()
        
        # Update request history
        if domain not in self.request_history:
            self.request_history[domain] = []
        
        self.request_history[domain].append(now)
        
        # Keep last 100 requests in history
        if len(self.request_history[domain]) > 100:
            self.request_history[domain].pop(0)
    
    async def execute_with_rate_limit(self, domain: str, func, *args, **kwargs):
        """
        Execute a function with domain-specific rate limiting.
        
        Args:
            domain: Domain for rate limiting
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function
        """
        # Wait for rate limit
        await self.wait(domain)
        
        # Get semaphore for concurrent request limiting
        semaphore = self.get_semaphore(domain)
        
        # Execute with concurrency control
        start_time = time.time()
        success = False
        
        try:
            async with semaphore:
                result = await func(*args, **kwargs)
                success = True
                return result
        except Exception as e:
            logger.error(f"Error in rate-limited function for {domain}: {str(e)}")
            raise
        finally:
            # Update adjustment factor
            response_time = time.time() - start_time
            self.update_adjustment_factor(domain, success, response_time)

class RequestFingerprinter:
    """
    Request fingerprint randomization to avoid detection.
    
    This class provides methods to generate randomized browser fingerprints
    including user-agents, accept headers, and other browser characteristics
    to avoid being detected as a bot.
    """
    
    def __init__(self, use_fake_useragent: bool = True):
        """
        Initialize the request fingerprinter.
        
        Args:
            use_fake_useragent: Whether to use fake-useragent library for user agent generation
        """
        self.use_fake_useragent = use_fake_useragent
        self._user_agents = []
        self._browser_profiles = []
        
        # Initialize user agents
        self._initialize_user_agents()
        
        # Initialize browser profiles
        self._initialize_browser_profiles()
        
        logger.info("Request fingerprinter initialized")
    
    def _initialize_user_agents(self) -> None:
        """Initialize the list of user agents."""
        try:
            if self.use_fake_useragent:
                # Try to use fake-useragent library for realistic user agents
                from fake_useragent import UserAgent
                ua = UserAgent()
                
                # Add a variety of user agents
                self._user_agents = [
                    ua.chrome,
                    ua.firefox,
                    ua.safari,
                    ua.edge,
                    ua.random
                ]
                
                # Add more random variants
                for _ in range(5):
                    self._user_agents.append(ua.random)
                
                logger.info("Using fake-useragent for user agent generation")
            else:
                # Fallback to a predefined list of common user agents
                self._user_agents = [
                    # Chrome
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                    # Firefox
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
                    # Safari
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
                    # Edge
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
                    # Mobile
                    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
                    "Mozilla/5.0 (Android 11; Mobile; rv:68.0) Gecko/68.0 Firefox/89.0"
                ]
                
                logger.info("Using predefined list of user agents")
        except Exception as e:
            # Fallback if fake-useragent fails
            logger.warning(f"Error initializing fake-useragent: {str(e)}. Using predefined user agents.")
            self._user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
            ]
    
    def _initialize_browser_profiles(self) -> None:
        """Initialize a list of browser profiles with consistent fingerprints."""
        # Create profiles for different browser types
        browsers = ["chrome", "firefox", "safari", "edge"]
        os_list = ["Windows", "MacOS", "Linux", "iOS", "Android"]
        
        for browser in browsers:
            for os_name in os_list:
                # Skip incompatible combinations
                if browser == "safari" and os_name not in ["MacOS", "iOS"]:
                    continue
                if browser == "edge" and os_name not in ["Windows", "MacOS"]:
                    continue
                
                # Create consistent profile
                profile = self._create_browser_profile(browser, os_name)
                self._browser_profiles.append(profile)
        
        # Add some random variations
        for _ in range(5):
            browser = random.choice(browsers)
            os_name = random.choice(os_list)
            
            # Skip incompatible combinations
            if browser == "safari" and os_name not in ["MacOS", "iOS"]:
                os_name = "MacOS"
            if browser == "edge" and os_name not in ["Windows", "MacOS"]:
                os_name = "Windows"
            
            profile = self._create_browser_profile(browser, os_name)
            self._browser_profiles.append(profile)
    
    def _create_browser_profile(self, browser: str, os_name: str) -> Dict[str, Any]:
        """
        Create a consistent browser profile.
        
        Args:
            browser: Browser name
            os_name: Operating system name
            
        Returns:
            Browser profile dictionary
        """
        # Generate random browser version based on browser type
        version = {
            "chrome": f"{random.randint(70, 116)}.0.{random.randint(1000, 9999)}.{random.randint(10, 250)}",
            "firefox": f"{random.randint(60, 116)}.0",
            "safari": f"{random.randint(12, 17)}.{random.randint(0, 3)}",
            "edge": f"{random.randint(90, 116)}.0.{random.randint(100, 999)}.{random.randint(10, 100)}"
        }[browser]
        
        # Generate OS version based on OS type
        os_version = {
            "Windows": f"Windows NT {random.choice(['10.0', '11.0'])}",
            "MacOS": f"Macintosh; Intel Mac OS X 10_{random.randint(13, 15)}_{random.randint(0, 7)}",
            "Linux": f"X11; Linux x86_64",
            "iOS": f"iPhone; CPU iPhone OS {random.randint(13, 17)}_{random.randint(0, 6)} like Mac OS X",
            "Android": f"Android {random.randint(9, 14)}; Mobile"
        }[os_name]
        
        # Generate consistent accept headers
        accept = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
        accept_language = f"en-US,en;q=0.{random.randint(5, 9)}"
        accept_encoding = "gzip, deflate, br"
        
        # Create WebGL fingerprint
        webgl_vendor = random.choice([
            "Google Inc. (NVIDIA)",
            "Intel Inc.",
            "NVIDIA Corporation",
            "ATI Technologies Inc.",
            "Apple Inc."
        ])
        
        webgl_renderer = random.choice([
            "ANGLE (NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0)",
            "ANGLE (Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0)",
            "ANGLE (AMD Radeon RX 6800 XT Direct3D11 vs_5_0 ps_5_0)",
            "Apple M1",
            "Apple M2",
            "Mali-G78"
        ])
        
        # Build full profile
        profile = {
            "browser": browser,
            "browser_version": version,
            "os": os_name,
            "os_version": os_version,
            "user_agent": self._create_user_agent(browser, version, os_version),
            "accept": accept,
            "accept_language": accept_language,
            "accept_encoding": accept_encoding,
            "sec_fetch_dest": "document",
            "sec_fetch_mode": "navigate",
            "sec_fetch_site": "none",
            "sec_fetch_user": "?1",
            "webgl": {
                "vendor": webgl_vendor,
                "renderer": webgl_renderer
            },
            "screen": {
                "width": random.choice([1366, 1440, 1536, 1920, 2560, 3440, 3840]),
                "height": random.choice([768, 900, 1080, 1440, 1600, 2160]),
                "color_depth": 24
            },
            "timezone": random.choice([-480, -420, -360, -300, -240, -180, -120, -60, 0, 60, 120, 180, 240, 300, 360, 420, 480]),
            "do_not_track": random.choice([None, "1", "0"])
        }
        
        return profile
    
    def _create_user_agent(self, browser: str, version: str, os_version: str) -> str:
        """
        Create a consistent user agent string.
        
        Args:
            browser: Browser name
            version: Browser version
            os_version: Operating system version
            
        Returns:
            User agent string
        """
        if browser == "chrome":
            return f"Mozilla/5.0 ({os_version}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36"
        elif browser == "firefox":
            return f"Mozilla/5.0 ({os_version}; rv:{version.split('.')[0]}.0) Gecko/20100101 Firefox/{version}"
        elif browser == "safari":
            return f"Mozilla/5.0 ({os_version}) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{version} Safari/605.1.15"
        elif browser == "edge":
            return f"Mozilla/5.0 ({os_version}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36 Edg/{version}"
        else:
            return random.choice(self._user_agents)
    
    def get_random_user_agent(self) -> str:
        """
        Get a random user agent string.
        
        Returns:
            Random user agent string
        """
        return random.choice(self._user_agents)
    
    def get_random_profile(self) -> Dict[str, Any]:
        """
        Get a random browser profile.
        
        Returns:
            Random browser profile dictionary
        """
        return random.choice(self._browser_profiles)
    
    def get_domain_consistent_profile(self, domain: str) -> Dict[str, Any]:
        """
        Get a browser profile that remains consistent for a specific domain.
        
        This helps avoid detection by making sure the same browser profile
        is used for all requests to a specific domain.
        
        Args:
            domain: The domain to get consistent profile for
            
        Returns:
            Browser profile dictionary
        """
        # Use domain as seed for random selection to ensure consistency
        domain_hash = sum(ord(c) for c in domain)
        random.seed(domain_hash)
        
        # Select profile using seeded random
        profile = random.choice(self._browser_profiles)
        
        # Reset random seed
        random.seed()
        
        return profile
    
    def apply_profile_to_headers(self, profile: Dict[str, Any]) -> Dict[str, str]:
        """
        Apply a browser profile to HTTP headers.
        
        Args:
            profile: Browser profile dictionary
            
        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            "User-Agent": profile["user_agent"],
            "Accept": profile["accept"],
            "Accept-Language": profile["accept_language"],
            "Accept-Encoding": profile["accept_encoding"],
            "Sec-Fetch-Dest": profile["sec_fetch_dest"],
            "Sec-Fetch-Mode": profile["sec_fetch_mode"],
            "Sec-Fetch-Site": profile["sec_fetch_site"],
            "Sec-Fetch-User": profile["sec_fetch_user"]
        }
        
        # Add DNT header if present in profile
        if profile["do_not_track"] is not None:
            headers["DNT"] = profile["do_not_track"]
        
        return headers
    
    def get_headers_for_domain(self, domain: str) -> Dict[str, str]:
        """
        Get consistent headers for a specific domain.
        
        Args:
            domain: The domain to get headers for
            
        Returns:
            Dictionary of HTTP headers
        """
        profile = self.get_domain_consistent_profile(domain)
        return self.apply_profile_to_headers(profile)
    
    def get_random_headers(self) -> Dict[str, str]:
        """
        Get random headers for a request.
        
        Returns:
            Dictionary of HTTP headers
        """
        profile = self.get_random_profile()
        return self.apply_profile_to_headers(profile)

class RequestFingerprintRandomizer:
    """
    Randomizes HTTP request fingerprints to prevent detection.
    
    This class provides advanced request fingerprinting capabilities:
    - Dynamic user agent rotation
    - HTTP header variation
    - Accept header randomization
    - Connection behavior simulation
    - Browser-specific header patterns
    - Device-specific fingerprint patterns
    """
    
    def __init__(self, 
                 custom_user_agents: Optional[List[str]] = None,
                 custom_headers: Optional[Dict[str, List[str]]] = None,
                 fingerprint_variation: str = "medium"):
        """
        Initialize the fingerprint randomizer.
        
        Args:
            custom_user_agents: Optional list of custom user agents
            custom_headers: Optional dictionary of custom headers and their possible values
            fingerprint_variation: Level of variation ("low", "medium", "high")
        """
        # Initialize fake user agent generator
        self.ua_generator = UserAgent(fallback="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
        
        # Store custom user agents if provided
        self.custom_user_agents = custom_user_agents or []
        
        # Custom headers dictionary
        self.custom_headers = custom_headers or {}
        
        # Set variation level
        self.variation_level = fingerprint_variation
        
        # Initialize browser profiles for more realistic fingerprints
        self._init_browser_profiles()
        
        # Track history of generated fingerprints to avoid repetition
        self.fingerprint_history = []
        self.max_history = 100
        
        logger.info(f"Request fingerprint randomizer initialized with {fingerprint_variation} variation")
    
    def _init_browser_profiles(self):
        """Initialize realistic browser profiles for fingerprinting."""
        # Chrome profiles
        self.chrome_profiles = [
            {
                "name": "Chrome Windows",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36",
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
                "accept_language": "en-US,en;q=0.9",
                "accept_encoding": "gzip, deflate, br",
                "sec_ch_ua": '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
                "sec_ch_ua_mobile": "?0",
                "sec_ch_ua_platform": '"Windows"',
                "sec_fetch_dest": "document",
                "sec_fetch_mode": "navigate",
                "sec_fetch_site": "none",
                "sec_fetch_user": "?1"
            },
            {
                "name": "Chrome macOS",
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36",
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
                "accept_language": "en-US,en;q=0.9",
                "accept_encoding": "gzip, deflate, br",
                "sec_ch_ua": '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
                "sec_ch_ua_mobile": "?0",
                "sec_ch_ua_platform": '"macOS"',
                "sec_fetch_dest": "document",
                "sec_fetch_mode": "navigate",
                "sec_fetch_site": "none",
                "sec_fetch_user": "?1"
            },
            {
                "name": "Chrome Android",
                "user_agent": "Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.58 Mobile Safari/537.36",
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
                "accept_language": "en-US,en;q=0.9",
                "accept_encoding": "gzip, deflate, br",
                "sec_ch_ua": '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
                "sec_ch_ua_mobile": "?1",
                "sec_ch_ua_platform": '"Android"',
                "sec_fetch_dest": "document",
                "sec_fetch_mode": "navigate",
                "sec_fetch_site": "none",
                "sec_fetch_user": "?1"
            }
        ]
        
        # Firefox profiles
        self.firefox_profiles = [
            {
                "name": "Firefox Windows",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "accept_language": "en-US,en;q=0.5",
                "accept_encoding": "gzip, deflate, br",
                "sec_fetch_dest": "document",
                "sec_fetch_mode": "navigate",
                "sec_fetch_site": "none",
                "sec_fetch_user": "?1",
                "te": "trailers"
            },
            {
                "name": "Firefox macOS",
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:98.0) Gecko/20100101 Firefox/98.0",
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "accept_language": "en-US,en;q=0.5",
                "accept_encoding": "gzip, deflate, br",
                "sec_fetch_dest": "document",
                "sec_fetch_mode": "navigate",
                "sec_fetch_site": "none",
                "sec_fetch_user": "?1",
                "te": "trailers"
            }
        ]
        
        # Safari profiles
        self.safari_profiles = [
            {
                "name": "Safari macOS",
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Safari/605.1.15",
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "accept_language": "en-US,en;q=0.9",
                "accept_encoding": "gzip, deflate, br"
            },
            {
                "name": "Safari iOS",
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Mobile/15E148 Safari/604.1",
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "accept_language": "en-US,en;q=0.9",
                "accept_encoding": "gzip, deflate, br"
            }
        ]
        
        # Edge profiles
        self.edge_profiles = [
            {
                "name": "Edge Windows",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.74 Safari/537.36 Edg/99.0.1150.46",
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
                "accept_language": "en-US,en;q=0.9",
                "accept_encoding": "gzip, deflate, br",
                "sec_ch_ua": '" Not A;Brand";v="99", "Chromium";v="99", "Microsoft Edge";v="99"',
                "sec_ch_ua_mobile": "?0",
                "sec_ch_ua_platform": '"Windows"',
                "sec_fetch_dest": "document",
                "sec_fetch_mode": "navigate",
                "sec_fetch_site": "none",
                "sec_fetch_user": "?1"
            }
        ]
        
        # Combine all profiles for easy access
        self.all_profiles = (
            self.chrome_profiles + 
            self.firefox_profiles + 
            self.safari_profiles + 
            self.edge_profiles
        )
    
    def get_random_user_agent(self, browser_type: Optional[str] = None) -> str:
        """
        Get a random user agent string.
        
        Args:
            browser_type: Optional browser type (chrome, firefox, safari, edge, random)
            
        Returns:
            User agent string
        """
        # If custom user agents are provided, use them
        if self.custom_user_agents and random.random() < 0.7:
            return random.choice(self.custom_user_agents)
        
        # Use fake-useragent library with specific browser type
        try:
            if not browser_type or browser_type == "random":
                return self.ua_generator.random
            elif browser_type == "chrome":
                return self.ua_generator.chrome
            elif browser_type == "firefox":
                return self.ua_generator.firefox
            elif browser_type == "safari":
                return self.ua_generator.safari
            elif browser_type == "edge":
                return random.choice([p["user_agent"] for p in self.edge_profiles])
            else:
                return self.ua_generator.random
        except:
            # Fallback to profiles if fake-useragent fails
            if browser_type == "chrome":
                return random.choice([p["user_agent"] for p in self.chrome_profiles])
            elif browser_type == "firefox":
                return random.choice([p["user_agent"] for p in self.firefox_profiles])
            elif browser_type == "safari":
                return random.choice([p["user_agent"] for p in self.safari_profiles])
            elif browser_type == "edge":
                return random.choice([p["user_agent"] for p in self.edge_profiles])
            else:
                return random.choice([p["user_agent"] for p in self.all_profiles])
    
    def generate_headers(self, url: str, browser_family: Optional[str] = None) -> Dict[str, str]:
        """
        Generate randomized headers for a request.
        
        Args:
            url: The URL being requested
            browser_family: Optional browser family to use (chrome, firefox, safari, edge)
            
        Returns:
            Dictionary of HTTP headers
        """
        # Parse the URL
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc
        
        # Determine browser family if not specified
        if not browser_family:
            # Weighted random choice
            browser_weights = {"chrome": 0.65, "firefox": 0.2, "safari": 0.1, "edge": 0.05}
            options = list(browser_weights.keys())
            weights = list(browser_weights.values())
            browser_family = random.choices(options, weights=weights, k=1)[0]
        
        # Select a random profile based on browser family
        if browser_family == "chrome":
            profile = random.choice(self.chrome_profiles)
        elif browser_family == "firefox":
            profile = random.choice(self.firefox_profiles)
        elif browser_family == "safari":
            profile = random.choice(self.safari_profiles)
        elif browser_family == "edge":
            profile = random.choice(self.edge_profiles)
        else:
            profile = random.choice(self.all_profiles)
        
        # Start with a base set of headers from the profile
        headers = {
            "User-Agent": profile["user_agent"],
            "Accept": profile["accept"],
            "Accept-Language": profile["accept_language"],
            "Accept-Encoding": profile["accept_encoding"],
            "Connection": "keep-alive",
            "DNT": "1" if random.random() < 0.5 else "0",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": random.choice(["max-age=0", "no-cache", "no-store"]),
            "TE": "Trailers" if random.random() < 0.3 else None,
            "Pragma": "no-cache" if random.random() < 0.5 else None
        }
        
        # Add browser-specific headers
        if browser_family in ["chrome", "edge"]:
            # Add Chrome/Edge specific headers
            if "sec_ch_ua" in profile:
                headers["sec-ch-ua"] = profile["sec_ch_ua"]
            if "sec_ch_ua_mobile" in profile:
                headers["sec-ch-ua-mobile"] = profile["sec_ch_ua_mobile"]
            if "sec_ch_ua_platform" in profile:
                headers["sec-ch-ua-platform"] = profile["sec_ch_ua_platform"]
            
            headers["sec-fetch-dest"] = profile.get("sec_fetch_dest", "document")
            headers["sec-fetch-mode"] = profile.get("sec_fetch_mode", "navigate")
            headers["sec-fetch-site"] = profile.get("sec_fetch_site", "none")
            
            # Randomize sec-fetch-user for some requests
            if random.random() < 0.7:
                headers["sec-fetch-user"] = "?1"
        
        # Add referer header sometimes
        if random.random() < 0.3:
            if random.random() < 0.7:
                # Use major search engine as referer
                search_engines = [
                    f"https://www.google.com/search?q={hostname}",
                    f"https://www.bing.com/search?q={hostname}",
                    f"https://search.yahoo.com/search?p={hostname}",
                    f"https://duckduckgo.com/?q={hostname}"
                ]
                headers["Referer"] = random.choice(search_engines)
            else:
                # Use same domain or https://www.domain.com as referer
                scheme = parsed_url.scheme
                headers["Referer"] = f"{scheme}://{hostname}/"
        
        # If "high" variation, add some random custom headers
        if self.variation_level == "high":
            # Random client hints
            if random.random() < 0.3:
                headers["device-memory"] = random.choice(["0.5", "1", "2", "4", "8"])
            
            if random.random() < 0.3:
                headers["viewport-width"] = str(random.choice([1280, 1366, 1440, 1920, 2560]))
            
            # Random content preference
            if random.random() < 0.2:
                headers["prefer"] = random.choice(["safe", "fast", "return=minimal", "wait=60"])
        
        # Remove None values
        headers = {k: v for k, v in headers.items() if v is not None}
        
        # Add any custom headers from initialization
        for header_name, possible_values in self.custom_headers.items():
            if possible_values and random.random() < 0.7:
                headers[header_name] = random.choice(possible_values)
        
        # Check history to avoid recent fingerprints
        fingerprint_hash = self._calculate_fingerprint_hash(headers)
        
        # If we have this exact fingerprint in recent history and a non-low variation,
        # recursively try again with a different browser family
        if (fingerprint_hash in self.fingerprint_history and
            self.variation_level != "low" and
            len(self.fingerprint_history) > 5):
            
            # Try again with a different browser family
            new_family = random.choice([f for f in ["chrome", "firefox", "safari", "edge"] 
                                      if f != browser_family])
            return self.generate_headers(url, new_family)
        
        # Add to history
        self.fingerprint_history.append(fingerprint_hash)
        
        # Keep history size limited
        if len(self.fingerprint_history) > self.max_history:
            self.fingerprint_history.pop(0)
        
        return headers
    
    def _calculate_fingerprint_hash(self, headers: Dict[str, str]) -> str:
        """
        Calculate a hash for the headers fingerprint.
        
        Args:
            headers: Headers dictionary
            
        Returns:
            Hash string
        """
        serialized = json.dumps(headers, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()
    
    def randomize_request_data(self, url: str, headers: Dict[str, str] = None, 
                             params: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Randomize all request components.
        
        Args:
            url: The URL for the request
            headers: Optional existing headers to enhance
            params: Optional existing query parameters
            
        Returns:
            Dictionary with randomized components
        """
        result = {"url": url}
        
        # Generate headers if none provided
        if not headers:
            result["headers"] = self.generate_headers(url)
        else:
            # Enhance provided headers
            enhanced_headers = headers.copy()
            
            # Only add User-Agent if not present
            if "User-Agent" not in enhanced_headers:
                enhanced_headers["User-Agent"] = self.get_random_user_agent()
            
            # Add some randomized headers if variation is medium or high
            if self.variation_level in ["medium", "high"]:
                random_headers = self.generate_headers(url)
                
                # Add random headers that don't conflict with existing ones
                for key, value in random_headers.items():
                    if key not in enhanced_headers and random.random() < 0.5:
                        enhanced_headers[key] = value
            
            result["headers"] = enhanced_headers
        
        # Randomize query parameters if provided
        if params:
            result["params"] = self._randomize_params(params)
        
        return result
    
    def _randomize_params(self, params: Dict[str, str]) -> Dict[str, str]:
        """
        Randomize the order of query parameters.
        
        Args:
            params: Query parameters dictionary
            
        Returns:
            Randomized parameters
        """
        # Convert to list of items
        items = list(params.items())
        
        # Shuffle the items
        random.shuffle(items)
        
        # Return as dict
        return dict(items)
    
    def get_browser_properties(self, browser_family: str = "chrome") -> Dict[str, Any]:
        """
        Get realistic browser properties for fingerprinting.
        
        Args:
            browser_family: Browser family (chrome, firefox, safari, edge)
            
        Returns:
            Dictionary of browser properties
        """
        # Select profiles for the browser family
        if browser_family == "chrome":
            profiles = self.chrome_profiles
        elif browser_family == "firefox":
            profiles = self.firefox_profiles
        elif browser_family == "safari":
            profiles = self.safari_profiles
        elif browser_family == "edge":
            profiles = self.edge_profiles
        else:
            profiles = self.all_profiles
        
        # Select a random profile
        profile = random.choice(profiles)
        
        # Extract platform from user agent
        platform = "Windows"
        if "Mac" in profile["user_agent"]:
            platform = "macOS"
        elif "Linux" in profile["user_agent"] or "Android" in profile["user_agent"]:
            platform = "Linux/Android"
        elif "iPhone" in profile["user_agent"] or "iPad" in profile["user_agent"]:
            platform = "iOS"
        
        # Create browser properties dictionary
        properties = {
            "name": profile["name"],
            "userAgent": profile["user_agent"],
            "platform": platform,
            "vendor": "Google Inc." if browser_family in ["chrome", "edge"] else "",
            "accept": profile["accept"],
            "acceptLanguage": profile["accept_language"],
            "headers": {k: v for k, v in profile.items() if k != "name"}
        }
        
        # Add screen properties
        if platform == "Windows" or platform == "macOS":
            properties["screen"] = {
                "width": random.choice([1280, 1366, 1440, 1536, 1920, 2560, 3440]),
                "height": random.choice([720, 768, 900, 1024, 1080, 1440, 2160]),
                "colorDepth": 24
            }
        elif platform == "Linux/Android":
            properties["screen"] = {
                "width": random.choice([360, 375, 390, 412, 414]),
                "height": random.choice([640, 667, 736, 812, 844, 896]),
                "colorDepth": 24
            }
        elif platform == "iOS":
            properties["screen"] = {
                "width": random.choice([320, 375, 390, 414, 428]),
                "height": random.choice([568, 667, 736, 812, 844, 926]),
                "colorDepth": 24
            }
        
        return properties

# Create a global instance for convenience
REQUEST_FINGERPRINTER = RequestFingerprinter()

# Create singleton instances for global use
SESSION_MANAGER = SessionManager()
REQUEST_MANAGER = RequestManager(session_manager=SESSION_MANAGER)
RATE_LIMITER = RateLimiter()
CIRCUIT_BREAKER = CircuitBreaker()
COOKIE_JAR = CookieJar()
ADVANCED_RATE_LIMITER = AdvancedRateLimiter()

async def get(url: str, **kwargs) -> Dict[str, Any]:
    """
    Global helper function for GET requests.
    
    Args:
        url: URL to request
        **kwargs: Additional parameters
        
    Returns:
        Response dictionary
    """
    return await REQUEST_MANAGER.get(url, **kwargs)

async def post(url: str, **kwargs) -> Dict[str, Any]:
    """
    Global helper function for POST requests.
    
    Args:
        url: URL to request
        **kwargs: Additional parameters
        
    Returns:
        Response dictionary
    """
    return await REQUEST_MANAGER.post(url, **kwargs)

async def close_all_sessions() -> None:
    """Global helper function to close all sessions."""
    await SESSION_MANAGER.close_all_sessions()

async def execute_with_rate_limit(domain: str, func, *args, **kwargs):
    """
    Global helper function to execute with rate limiting.
    
    Args:
        domain: Domain for rate limiting
        func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the function
    """
    return await ADVANCED_RATE_LIMITER.execute_with_rate_limit(domain, func, *args, **kwargs)

# Ensure sessions are closed on exit
import atexit
import asyncio

def _close_sessions_sync():
    """Close sessions synchronously on exit."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(close_all_sessions())
        else:
            loop.run_until_complete(close_all_sessions())
    except Exception:
        pass

atexit.register(_close_sessions_sync)