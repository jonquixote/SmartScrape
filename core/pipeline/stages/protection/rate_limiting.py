"""
Rate Limiting Stage Module.

This module provides a RateLimitingStage that integrates with the RateLimiter service
to enforce rate limiting policies across pipeline executions.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from core.pipeline.stages.base_stage import BaseStage
from core.pipeline.context import PipelineContext
from core.service_registry import ServiceRegistry
from core.rate_limiter import RateLimiter


class RateLimitingStage(BaseStage):
    """
    Rate Limiting Stage for enforcing request rate policies.
    
    This stage ensures that requests to specific domains respect rate limits,
    helping to avoid being blocked and ensuring responsible scraping behavior.
    
    Features:
    - Domain-specific rate limiting
    - Global rate limiting across all domains
    - Dynamic rate adjustment based on service responses
    - Automatic backoff on rate limit detection
    - Integration with centralized RateLimiter service
    - Fallback to local rate limiting when service unavailable
    
    Configuration:
    - mode: Rate limiting mode ("domain", "global", or "both")
    - global_rate: Global rate limit (requests per second)
    - domain_rates: Dictionary of domain-specific rate limits
    - default_domain_rate: Default rate limit for domains not in domain_rates
    - backoff_factor: Factor to multiply delay by after rate limit detection
    - max_delay: Maximum delay time in seconds
    - respect_retry_after: Whether to respect Retry-After headers
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the rate limiting stage.
        
        Args:
            name: Optional name for the stage (defaults to class name)
            config: Optional configuration dictionary
        """
        super().__init__(name, config)
        
        # Configuration
        self.mode = self.config.get("mode", "domain")  # "domain", "global", or "both"
        self.global_rate = self.config.get("global_rate", 1.0)  # requests per second
        self.domain_rates = self.config.get("domain_rates", {})  # domain-specific rates
        self.default_domain_rate = self.config.get("default_domain_rate", 0.5)  # rps
        self.backoff_factor = self.config.get("backoff_factor", 2.0)
        self.max_delay = self.config.get("max_delay", 300.0)  # 5 minutes max delay
        self.respect_retry_after = self.config.get("respect_retry_after", True)
        
        # Internal state
        self._last_global_request = 0.0
        self._domain_last_requests = {}
        self._domain_backoffs = {}
        self._rate_limiter = None
        
        # Metrics
        self._wait_count = 0
        self._total_wait_time = 0.0
        self._domain_stats = {}
    
    async def _process(self, context: PipelineContext) -> bool:
        """
        Apply rate limiting to the pipeline flow.
        
        This method implements the core rate limiting logic, waiting as needed
        to ensure rate limits are respected.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        # Extract URLs from context to determine domains
        urls = self._extract_urls(context)
        domains = set(urlparse(url).netloc for url in urls)
        
        if not domains:
            # No domains found, check for explicit domain in context
            domain = context.get("domain")
            if domain:
                domains.add(domain)
                
        if not domains:
            # Still no domains found, skip rate limiting
            self._logger.debug("No domains found for rate limiting, skipping")
            return True
            
        # Apply rate limiting
        delay_applied = False
        total_delay = 0.0
        
        # Get rate limiter service if available
        rate_limiter = self._get_rate_limiter()
        
        if rate_limiter:
            # Use service-based rate limiting
            for domain in domains:
                wait_time = rate_limiter.get_wait_time(domain)
                if wait_time > 0:
                    self._logger.debug(f"Rate limiting {domain} for {wait_time:.2f}s")
                    delay_applied = True
                    total_delay += wait_time
                    self._wait_count += 1
                    self._total_wait_time += wait_time
                    
                    # Update domain stats
                    self._update_domain_stats(domain, wait_time)
                    
                    # Wait for the required time
                    await asyncio.sleep(wait_time)
                    
                    # Notify the rate limiter that we've waited
                    rate_limiter.register_wait(domain, wait_time)
        else:
            # Use local rate limiting
            # Apply global rate limiting if configured
            if self.mode in ("global", "both"):
                delay = self._calculate_global_delay()
                if delay > 0:
                    self._logger.debug(f"Global rate limiting for {delay:.2f}s")
                    delay_applied = True
                    total_delay += delay
                    self._wait_count += 1
                    self._total_wait_time += delay
                    await asyncio.sleep(delay)
            
            # Apply domain-specific rate limiting if configured
            if self.mode in ("domain", "both"):
                for domain in domains:
                    delay = self._calculate_domain_delay(domain)
                    if delay > 0:
                        self._logger.debug(f"Domain rate limiting for {domain}: {delay:.2f}s")
                        delay_applied = True
                        total_delay += delay
                        self._wait_count += 1
                        self._total_wait_time += delay
                        
                        # Update domain stats
                        self._update_domain_stats(domain, delay)
                        
                        # Wait for the required time
                        await asyncio.sleep(delay)
        
        # Update context with rate limiting information
        context.set("rate_limiting_applied", delay_applied)
        context.set("rate_limiting_delay", total_delay)
        
        # Register metrics
        self._register_metrics({
            "wait_count": self._wait_count,
            "total_wait_time": self._total_wait_time,
            "average_wait_time": self._total_wait_time / self._wait_count if self._wait_count > 0 else 0
        })
        
        return True
    
    def _extract_urls(self, context: PipelineContext) -> List[str]:
        """
        Extract URLs from the context to determine domains for rate limiting.
        
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
    
    def _calculate_global_delay(self) -> float:
        """
        Calculate delay needed for global rate limiting.
        
        Returns:
            float: Delay in seconds (0 if no delay needed)
        """
        if self.global_rate <= 0:
            return 0.0
            
        current_time = time.time()
        time_since_last = current_time - self._last_global_request
        min_interval = 1.0 / self.global_rate
        
        if time_since_last < min_interval:
            delay = min_interval - time_since_last
            # Update last request time (after the delay will complete)
            self._last_global_request = current_time + delay
            return delay
        else:
            # Update last request time
            self._last_global_request = current_time
            return 0.0
    
    def _calculate_domain_delay(self, domain: str) -> float:
        """
        Calculate delay needed for domain-specific rate limiting.
        
        Args:
            domain: The domain to calculate delay for
            
        Returns:
            float: Delay in seconds (0 if no delay needed)
        """
        # Get domain-specific rate or default
        rate = self.domain_rates.get(domain, self.default_domain_rate)
        
        # Check if we have a backoff for this domain
        backoff_multiplier = self._domain_backoffs.get(domain, 1.0)
        effective_rate = rate / backoff_multiplier
        
        if effective_rate <= 0:
            return 0.0
            
        current_time = time.time()
        last_request = self._domain_last_requests.get(domain, 0.0)
        time_since_last = current_time - last_request
        min_interval = 1.0 / effective_rate
        
        if time_since_last < min_interval:
            delay = min_interval - time_since_last
            # Update last request time (after the delay will complete)
            self._domain_last_requests[domain] = current_time + delay
            return delay
        else:
            # Update last request time
            self._domain_last_requests[domain] = current_time
            return 0.0
    
    def _get_rate_limiter(self) -> Optional[RateLimiter]:
        """
        Get the RateLimiter service if available.
        
        Returns:
            Optional[RateLimiter]: The rate limiter service or None
        """
        if self._rate_limiter is not None:
            return self._rate_limiter
            
        try:
            service = self._get_service("rate_limiter")
            if isinstance(service, RateLimiter):
                self._rate_limiter = service
                return service
        except Exception as e:
            self._logger.warning(f"Could not get RateLimiter service: {str(e)}")
            
        return None
    
    def _update_domain_stats(self, domain: str, delay: float) -> None:
        """
        Update statistics for a domain.
        
        Args:
            domain: The domain being rate limited
            delay: The delay applied
        """
        if domain not in self._domain_stats:
            self._domain_stats[domain] = {
                "wait_count": 0,
                "total_wait_time": 0.0,
                "last_wait_time": 0.0
            }
            
        stats = self._domain_stats[domain]
        stats["wait_count"] += 1
        stats["total_wait_time"] += delay
        stats["last_wait_time"] = delay
        stats["average_wait_time"] = stats["total_wait_time"] / stats["wait_count"]
    
    def report_rate_limited(self, domain: str, retry_after: Optional[float] = None) -> None:
        """
        Report that a domain has rate limited a request.
        
        This method should be called when a rate limit response (HTTP 429)
        is received, allowing the stage to adjust its rate limiting strategy.
        
        Args:
            domain: The domain that triggered the rate limit
            retry_after: Optional Retry-After value in seconds
        """
        # Get current backoff or initialize
        current_backoff = self._domain_backoffs.get(domain, 1.0)
        
        # If we respect retry_after and it was provided, use that
        if self.respect_retry_after and retry_after is not None:
            # Apply retry_after but respect max_delay
            self._domain_backoffs[domain] = min(current_backoff * self.backoff_factor, self.max_delay)
            self._logger.info(f"Rate limited by {domain}, backing off with Retry-After: {retry_after}s")
            
            # Forward to rate limiter service if available
            rate_limiter = self._get_rate_limiter()
            if rate_limiter:
                rate_limiter.report_rate_limited(domain, retry_after)
        else:
            # Increase backoff factor
            new_backoff = min(current_backoff * self.backoff_factor, self.max_delay)
            self._domain_backoffs[domain] = new_backoff
            self._logger.info(f"Rate limited by {domain}, increasing backoff factor to {new_backoff:.2f}")
            
            # Forward to rate limiter service if available
            rate_limiter = self._get_rate_limiter()
            if rate_limiter:
                rate_limiter.report_rate_limited(domain)
    
    def decrease_backoff(self, domain: str) -> None:
        """
        Decrease the backoff for a domain after successful requests.
        
        This method should be called periodically after successful requests
        to gradually return to normal rate limits.
        
        Args:
            domain: The domain to adjust backoff for
        """
        current_backoff = self._domain_backoffs.get(domain, 1.0)
        
        # Don't reduce below 1.0
        if current_backoff > 1.0:
            new_backoff = max(current_backoff / self.backoff_factor, 1.0)
            self._domain_backoffs[domain] = new_backoff
            self._logger.debug(f"Decreasing backoff for {domain} to {new_backoff:.2f}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics about rate limiting activity.
        
        Returns:
            Dict[str, Any]: Dictionary of rate limiting metrics
        """
        metrics = super().get_metrics()
        
        # Add rate limiting specific metrics
        rate_metrics = {
            "wait_count": self._wait_count,
            "total_wait_time": self._total_wait_time,
            "average_wait_time": self._total_wait_time / self._wait_count if self._wait_count > 0 else 0,
            "domain_stats": self._domain_stats,
            "current_backoffs": self._domain_backoffs
        }
        
        metrics.update(rate_metrics)
        return metrics
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for rate limiting stage configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing JSON schema
        """
        schema = super().get_config_schema()
        
        # Add rate limiting specific properties
        rate_properties = {
            "mode": {"type": "string", "enum": ["domain", "global", "both"]},
            "global_rate": {"type": "number", "minimum": 0},
            "domain_rates": {
                "type": "object",
                "additionalProperties": {"type": "number", "minimum": 0}
            },
            "default_domain_rate": {"type": "number", "minimum": 0},
            "backoff_factor": {"type": "number", "minimum": 1},
            "max_delay": {"type": "number", "minimum": 0},
            "respect_retry_after": {"type": "boolean"}
        }
        
        schema["properties"].update(rate_properties)
        return schema