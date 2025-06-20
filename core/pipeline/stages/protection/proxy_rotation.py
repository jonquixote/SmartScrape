"""
Proxy Rotation Stage Module.

This module provides a ProxyRotationStage that integrates with the ProxyManager service
to manage and rotate proxies throughout pipeline executions.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from core.pipeline.stages.base_stage import BaseStage
from core.pipeline.context import PipelineContext
from core.service_registry import ServiceRegistry
from core.proxy_manager import ProxyManager


class ProxyRotationStage(BaseStage):
    """
    Proxy Rotation Stage for managing proxy usage in pipelines.
    
    This stage handles proxy selection, rotation, and management for HTTP requests,
    helping to distribute traffic across different IPs and avoid IP blocks.
    
    Features:
    - Automatic proxy selection based on target domain
    - Intelligent proxy rotation on failures
    - Geographic location targeting for proxies
    - Health monitoring of proxies with automatic fallback
    - Configurable proxy selection strategies
    - Dynamic proxy pool management
    
    Configuration:
    - rotation_strategy: Strategy for rotating proxies ("round_robin", "random", "sticky")
    - sticky_domain: Whether to use the same proxy for the same domain
    - rotation_interval: Time in seconds before forcing a proxy rotation
    - health_check_enabled: Whether to check proxy health before using
    - fallback_to_direct: Whether to fall back to direct connection if all proxies fail
    - max_failures: Maximum failures before blacklisting a proxy
    - location_preference: List of preferred geographic locations for proxies
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the proxy rotation stage.
        
        Args:
            name: Optional name for the stage (defaults to class name)
            config: Optional configuration dictionary
        """
        super().__init__(name, config)
        
        # Configuration
        self.rotation_strategy = self.config.get("rotation_strategy", "round_robin")
        self.sticky_domain = self.config.get("sticky_domain", True)
        self.rotation_interval = self.config.get("rotation_interval", 600)  # 10 minutes
        self.health_check_enabled = self.config.get("health_check_enabled", True)
        self.fallback_to_direct = self.config.get("fallback_to_direct", True)
        self.max_failures = self.config.get("max_failures", 3)
        self.location_preference = self.config.get("location_preference", [])
        
        # Internal state
        self._proxy_manager = None
        self._domain_proxies = {}  # Maps domains to their current proxy
        self._proxy_last_used = {}  # Maps proxy URLs to last used timestamp
        self._proxy_failures = {}  # Maps proxy URLs to failure counts
        self._proxy_successes = {}  # Maps proxy URLs to success counts
        
        # Metrics
        self._proxy_rotations = 0
        self._proxy_failures_total = 0
        self._proxy_successes_total = 0
        self._direct_fallbacks = 0
    
    async def _process(self, context: PipelineContext) -> bool:
        """
        Select and apply proxy configuration to the pipeline context.
        
        This method extracts target information from the context,
        selects an appropriate proxy, and enriches the context with 
        proxy configuration.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        # Extract target information
        urls = self._extract_urls(context)
        domains = set(urlparse(url).netloc for url in urls)
        
        if not domains:
            # Check for explicit domain in context
            domain = context.get("domain")
            if domain:
                domains.add(domain)
        
        if not domains:
            # No domains found, skip proxy selection
            self._logger.debug("No domains found for proxy selection, skipping")
            return True
        
        # Get proxy manager service
        proxy_manager = self._get_proxy_manager()
        if not proxy_manager and not context.get("proxy"):
            if not self.fallback_to_direct:
                self._logger.warning("No proxy manager available and fallback to direct is disabled")
                context.add_error(self.name, "No proxy manager available")
                return False
            else:
                # Set direct connection in context
                context.set("proxy", None)
                context.set("proxy_info", {"type": "direct", "reason": "no_proxy_manager"})
                return True
        
        # Apply proxy configuration
        proxy_applied = False
        
        try:
            # If we have an existing proxy in context, check if we need to rotate
            current_proxy = context.get("proxy")
            
            if current_proxy and not self._should_rotate_proxy(current_proxy):
                # Keep using the existing proxy
                self._logger.debug(f"Continuing to use proxy: {self._sanitize_proxy(current_proxy)}")
                return True
            
            # Select proxy for each domain and set in context
            domain_proxies = {}
            
            for domain in domains:
                proxy_url = await self._select_proxy(domain, context)
                if proxy_url:
                    domain_proxies[domain] = proxy_url
                    # Track usage
                    self._proxy_last_used[proxy_url] = time.time()
                    
                    # If sticky domain is disabled, use the first proxy for all domains
                    if not self.sticky_domain:
                        break
            
            # Set proxy information in context
            if domain_proxies:
                # If there's only one domain or sticky is disabled, use that proxy
                if len(domain_proxies) == 1 or not self.sticky_domain:
                    proxy_url = next(iter(domain_proxies.values()))
                    context.set("proxy", proxy_url)
                    context.set("proxy_info", {
                        "type": "single",
                        "url": self._sanitize_proxy(proxy_url),
                        "strategy": self.rotation_strategy
                    })
                else:
                    # Store multiple proxies for different domains
                    context.set("domain_proxies", domain_proxies)
                    context.set("proxy_info", {
                        "type": "per_domain",
                        "domains": {d: self._sanitize_proxy(p) for d, p in domain_proxies.items()},
                        "strategy": self.rotation_strategy
                    })
                
                proxy_applied = True
                
            elif self.fallback_to_direct:
                # No proxies available, fall back to direct
                context.set("proxy", None)
                context.set("proxy_info", {"type": "direct", "reason": "no_proxies_available"})
                self._direct_fallbacks += 1
                
                # Register metrics
                self._register_metrics({
                    "direct_fallbacks": self._direct_fallbacks
                })
                
                return True
            else:
                # No proxies available and fallback is disabled
                self._logger.warning("No proxies available and fallback to direct is disabled")
                context.add_error(self.name, "No proxies available")
                return False
            
            # Register metrics if proxy was applied
            if proxy_applied:
                self._proxy_rotations += 1
                self._register_metrics({
                    "proxy_rotations": self._proxy_rotations,
                    "proxy_failures": self._proxy_failures_total,
                    "proxy_successes": self._proxy_successes_total,
                    "direct_fallbacks": self._direct_fallbacks
                })
            
            return True
            
        except Exception as e:
            self._logger.error(f"Error selecting proxy: {str(e)}")
            
            if self.fallback_to_direct:
                # Set direct connection in context on error
                context.set("proxy", None)
                context.set("proxy_info", {"type": "direct", "reason": "error", "error": str(e)})
                self._direct_fallbacks += 1
                
                # Register metrics
                self._register_metrics({
                    "direct_fallbacks": self._direct_fallbacks
                })
                
                return True
            else:
                # Log the error
                context.add_error(self.name, f"Error selecting proxy: {str(e)}")
                return False
    
    def _extract_urls(self, context: PipelineContext) -> List[str]:
        """
        Extract URLs from the context to determine domains for proxy selection.
        
        Args:
            context: The pipeline context
            
        Returns:
            List[str]: List of URLs found in the context
        """
        urls = []
        
        # Check direct URL field
        if context.contains("url"):
            url = context.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                urls.append(url)
        
        # Check request object if present
        request = context.get("request")
        if request:
            # Check request source
            if hasattr(request, "source") and isinstance(request.source, str):
                if request.source.startswith(("http://", "https://")):
                    urls.append(request.source)
            
            # Check request URLs list if present
            if hasattr(request, "urls") and isinstance(request.urls, list):
                for url in request.urls:
                    if isinstance(url, str) and url.startswith(("http://", "https://")):
                        urls.append(url)
        
        # Check for URLs list in context
        if context.contains("urls"):
            url_list = context.get("urls")
            if isinstance(url_list, list):
                for url in url_list:
                    if isinstance(url, str) and url.startswith(("http://", "https://")):
                        urls.append(url)
        
        return urls
    
    async def _select_proxy(self, domain: str, context: PipelineContext) -> Optional[str]:
        """
        Select a proxy for the specified domain.
        
        This method applies the configured selection strategy to choose 
        an appropriate proxy for the domain.
        
        Args:
            domain: The domain to select a proxy for
            context: The pipeline context with additional information
            
        Returns:
            Optional[str]: The selected proxy URL or None if none available
        """
        proxy_manager = self._get_proxy_manager()
        if not proxy_manager:
            return None
            
        # Check if we have a sticky proxy for this domain already
        if self.sticky_domain and domain in self._domain_proxies:
            proxy_url = self._domain_proxies[domain]
            
            # Check if the proxy is still valid
            if not self._is_proxy_blacklisted(proxy_url):
                # Check if we should rotate based on time interval
                if not self._should_rotate_by_interval(proxy_url):
                    return proxy_url
        
        # Apply selection strategy
        try:
            # Get custom tags from context that might influence proxy selection
            tags = context.get("proxy_tags", [])
            
            # Add location preference if configured
            if self.location_preference:
                if isinstance(self.location_preference, list):
                    tags.extend([f"location:{loc}" for loc in self.location_preference])
                elif isinstance(self.location_preference, str):
                    tags.append(f"location:{self.location_preference}")
            
            if self.rotation_strategy == "random":
                proxy_url = await proxy_manager.get_random_proxy(domain, tags)
            elif self.rotation_strategy == "round_robin":
                proxy_url = await proxy_manager.get_next_proxy(domain, tags)
            elif self.rotation_strategy == "sticky":
                # Try to get the "best" proxy for this domain
                proxy_url = await proxy_manager.get_best_proxy(domain, tags)
            else:
                # Default to round-robin
                proxy_url = await proxy_manager.get_next_proxy(domain, tags)
            
            # Store the selected proxy for this domain
            if proxy_url:
                self._domain_proxies[domain] = proxy_url
            
            return proxy_url
            
        except Exception as e:
            self._logger.warning(f"Error selecting proxy for {domain}: {str(e)}")
            return None
    
    def _should_rotate_proxy(self, current_proxy: str) -> bool:
        """
        Determine if the current proxy should be rotated.
        
        Args:
            current_proxy: The current proxy URL
            
        Returns:
            bool: True if proxy should be rotated, False otherwise
        """
        # Check if proxy is blacklisted
        if self._is_proxy_blacklisted(current_proxy):
            return True
            
        # Check if we should rotate based on time interval
        if self._should_rotate_by_interval(current_proxy):
            return True
            
        return False
    
    def _should_rotate_by_interval(self, proxy_url: str) -> bool:
        """
        Check if proxy should be rotated based on time interval.
        
        Args:
            proxy_url: The proxy URL to check
            
        Returns:
            bool: True if proxy should be rotated, False otherwise
        """
        if self.rotation_interval <= 0:
            return False
            
        last_used = self._proxy_last_used.get(proxy_url, 0)
        time_since_last = time.time() - last_used
        
        return time_since_last > self.rotation_interval
    
    def _is_proxy_blacklisted(self, proxy_url: str) -> bool:
        """
        Check if a proxy is blacklisted due to failures.
        
        Args:
            proxy_url: The proxy URL to check
            
        Returns:
            bool: True if proxy is blacklisted, False otherwise
        """
        failure_count = self._proxy_failures.get(proxy_url, 0)
        return failure_count >= self.max_failures
    
    def _get_proxy_manager(self) -> Optional[ProxyManager]:
        """
        Get the ProxyManager service if available.
        
        Returns:
            Optional[ProxyManager]: The proxy manager service or None
        """
        if self._proxy_manager is not None:
            return self._proxy_manager
            
        try:
            service = self._get_service("proxy_manager")
            if isinstance(service, ProxyManager):
                self._proxy_manager = service
                return service
        except Exception as e:
            self._logger.warning(f"Could not get ProxyManager service: {str(e)}")
            
        return None
    
    def _sanitize_proxy(self, proxy_url: str) -> str:
        """
        Sanitize a proxy URL for logging (hide authentication).
        
        Args:
            proxy_url: The proxy URL to sanitize
            
        Returns:
            str: Sanitized proxy URL
        """
        if not proxy_url:
            return "None"
            
        try:
            # Simple sanitization - replace username:password with ***
            if "@" in proxy_url:
                parts = proxy_url.split("@")
                if len(parts) > 1 and "://" in parts[0]:
                    protocol = parts[0].split("://")[0]
                    return f"{protocol}://***@{parts[1]}"
            
            return proxy_url
        except Exception:
            return "[sanitization-error]"
    
    def report_proxy_success(self, proxy_url: str) -> None:
        """
        Report successful usage of a proxy.
        
        This method should be called when a request through a proxy succeeds,
        helping to track proxy performance.
        
        Args:
            proxy_url: The proxy URL that was successful
        """
        # Update success count
        success_count = self._proxy_successes.get(proxy_url, 0)
        success_count += 1
        self._proxy_successes[proxy_url] = success_count
        self._proxy_successes_total += 1
        
        # Reset failure count on success
        self._proxy_failures[proxy_url] = 0
        
        # Forward to proxy manager if available
        proxy_manager = self._get_proxy_manager()
        if proxy_manager:
            proxy_manager.report_success(proxy_url)
    
    def report_proxy_failure(self, proxy_url: str, error: Optional[str] = None) -> None:
        """
        Report a proxy failure.
        
        This method should be called when a request through a proxy fails,
        allowing for proxy rotation and blacklisting.
        
        Args:
            proxy_url: The proxy URL that failed
            error: Optional error message
        """
        # Update failure count
        failure_count = self._proxy_failures.get(proxy_url, 0)
        failure_count += 1
        self._proxy_failures[proxy_url] = failure_count
        self._proxy_failures_total += 1
        
        # Log the failure
        self._logger.warning(
            f"Proxy failure for {self._sanitize_proxy(proxy_url)} "
            f"(failure {failure_count}/{self.max_failures}): {error or 'unknown error'}"
        )
        
        # Forward to proxy manager if available
        proxy_manager = self._get_proxy_manager()
        if proxy_manager:
            proxy_manager.report_failure(proxy_url, error)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics about proxy usage.
        
        Returns:
            Dict[str, Any]: Dictionary of proxy metrics
        """
        metrics = super().get_metrics()
        
        # Add proxy specific metrics
        proxy_metrics = {
            "proxy_rotations": self._proxy_rotations,
            "proxy_failures": self._proxy_failures_total,
            "proxy_successes": self._proxy_successes_total,
            "direct_fallbacks": self._direct_fallbacks,
            "blacklisted_proxies": sum(1 for count in self._proxy_failures.values() if count >= self.max_failures),
            "domain_proxy_mapping": len(self._domain_proxies)
        }
        
        metrics.update(proxy_metrics)
        return metrics
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for proxy rotation stage configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing JSON schema
        """
        schema = super().get_config_schema()
        
        # Add proxy rotation specific properties
        proxy_properties = {
            "rotation_strategy": {"type": "string", "enum": ["round_robin", "random", "sticky"]},
            "sticky_domain": {"type": "boolean"},
            "rotation_interval": {"type": "number", "minimum": 0},
            "health_check_enabled": {"type": "boolean"},
            "fallback_to_direct": {"type": "boolean"},
            "max_failures": {"type": "integer", "minimum": 0},
            "location_preference": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}}
                ]
            }
        }
        
        schema["properties"].update(proxy_properties)
        return schema