"""
Request Throttling Module

This module provides functionality for controlling request rates and
implementing adaptive throttling to avoid detection and IP blocking.

Features:
1. Rate limiting based on domain
2. Adaptive throttling based on response patterns
3. Automatic backoff on detection signals
4. IP rotation recommendations
"""

import time
import random
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from collections import defaultdict
from datetime import datetime, timedelta
import math

logger = logging.getLogger("RequestThrottling")

class RequestThrottler:
    """
    Request throttler for controlling request rates and implementing adaptive throttling.
    
    This class helps avoid detection by websites by controlling the rate of requests
    and implementing adaptive behavior based on response patterns.
    """
    
    def __init__(self, 
                 default_min_delay: float = 1.0,
                 default_max_delay: float = 3.0,
                 jitter_factor: float = 0.25,
                 adaptive_enabled: bool = True,
                 detection_sensitivity: float = 0.7):
        """
        Initialize the request throttler.
        
        Args:
            default_min_delay: Minimum delay between requests in seconds
            default_max_delay: Maximum delay between requests in seconds
            jitter_factor: Random jitter factor to add to delays
            adaptive_enabled: Whether to enable adaptive throttling
            detection_sensitivity: Sensitivity for detection signals (0.0-1.0)
        """
        self.default_min_delay = default_min_delay
        self.default_max_delay = default_max_delay
        self.jitter_factor = jitter_factor
        self.adaptive_enabled = adaptive_enabled
        self.detection_sensitivity = detection_sensitivity
        
        # Domain-specific settings
        self.domain_settings = {}
        
        # Last request timestamps by domain
        self.last_request_time = defaultdict(float)
        
        # Domain throttling state
        self.domain_state = {}
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Initialize metrics for each domain
        self.domain_metrics = defaultdict(lambda: {
            "request_count": 0,
            "error_count": 0,
            "captcha_count": 0,
            "block_count": 0,
            "response_times": [],
            "backoff_events": [],
            "current_delay": None
        })
        
        logger.info(f"Request throttler initialized with default delay range: {default_min_delay}-{default_max_delay}s")
    
    def set_domain_settings(self, 
                           domain: str, 
                           min_delay: float,
                           max_delay: float,
                           max_requests_per_minute: Optional[int] = None,
                           max_requests_per_hour: Optional[int] = None,
                           respect_robots_txt: bool = True) -> None:
        """
        Set throttling settings for a specific domain.
        
        Args:
            domain: The domain to set settings for
            min_delay: Minimum delay between requests in seconds
            max_delay: Maximum delay between requests in seconds
            max_requests_per_minute: Maximum requests allowed per minute
            max_requests_per_hour: Maximum requests allowed per hour
            respect_robots_txt: Whether to respect robots.txt rules
        """
        with self.lock:
            self.domain_settings[domain] = {
                "min_delay": min_delay,
                "max_delay": max_delay,
                "max_requests_per_minute": max_requests_per_minute,
                "max_requests_per_hour": max_requests_per_hour,
                "respect_robots_txt": respect_robots_txt
            }
            
            # Initialize domain state if not already set
            if domain not in self.domain_state:
                self.domain_state[domain] = {
                    "current_min_delay": min_delay,
                    "current_max_delay": max_delay,
                    "backoff_level": 0,
                    "detection_signals": 0,
                    "request_times": [],
                    "backoff_until": None,
                    "last_error_time": None,
                    "consecutive_errors": 0
                }
            
            logger.info(f"Set throttling settings for {domain}: delay={min_delay}-{max_delay}s, "
                       f"max/min={max_requests_per_minute}, max/hour={max_requests_per_hour}")
    
    def get_domain_settings(self, domain: str) -> Dict[str, Any]:
        """
        Get current throttling settings for a domain.
        
        Args:
            domain: The domain to get settings for
            
        Returns:
            Dictionary with domain throttling settings
        """
        with self.lock:
            # Return domain-specific settings if available, otherwise default settings
            if domain in self.domain_settings:
                return self.domain_settings[domain]
            else:
                return {
                    "min_delay": self.default_min_delay,
                    "max_delay": self.default_max_delay,
                    "max_requests_per_minute": None,
                    "max_requests_per_hour": None,
                    "respect_robots_txt": True
                }
    
    def get_delay(self, domain: str) -> float:
        """
        Get the current delay to use for a domain.
        
        Args:
            domain: The domain to get delay for
            
        Returns:
            Delay in seconds to wait before making the next request
        """
        with self.lock:
            # Get domain state or initialize with defaults
            if domain not in self.domain_state:
                settings = self.get_domain_settings(domain)
                self.domain_state[domain] = {
                    "current_min_delay": settings["min_delay"],
                    "current_max_delay": settings["max_delay"],
                    "backoff_level": 0,
                    "detection_signals": 0,
                    "request_times": [],
                    "backoff_until": None,
                    "last_error_time": None,
                    "consecutive_errors": 0
                }
            
            domain_state = self.domain_state[domain]
            
            # Check if we're in backoff mode
            now = datetime.now()
            if domain_state["backoff_until"] and now < domain_state["backoff_until"]:
                # Calculate remaining backoff time
                remaining = (domain_state["backoff_until"] - now).total_seconds()
                logger.debug(f"Domain {domain} in backoff mode. {remaining:.1f}s remaining.")
                
                # Use max delay during backoff period
                delay = domain_state["current_max_delay"]
            else:
                # Normal operation - use randomized delay between min and max
                min_delay = domain_state["current_min_delay"]
                max_delay = domain_state["current_max_delay"]
                
                # Base delay is random between min and max
                base_delay = random.uniform(min_delay, max_delay)
                
                # Add random jitter
                jitter = random.uniform(0, self.jitter_factor * base_delay)
                delay = base_delay + jitter
            
            # Apply humanized patterns
            delay = self._apply_humanized_pattern(delay, domain)
            
            # Check rate limits
            delay = max(delay, self._check_rate_limits(domain))
            
            return delay
    
    def _apply_humanized_pattern(self, delay: float, domain: str) -> float:
        """
        Apply humanized patterns to delay.
        
        This adds subtle human-like patterns to request timing.
        
        Args:
            delay: Base delay in seconds
            domain: The domain being accessed
            
        Returns:
            Modified delay in seconds
        """
        # Occasionally add a longer pause (simulating user distraction)
        if random.random() < 0.05:  # 5% chance
            delay += random.uniform(2.0, 8.0)
            logger.debug(f"Adding human-like pause for {domain}: +{delay:.1f}s")
        
        # Occasionally add a very short delay (quick browsing)
        if random.random() < 0.1:  # 10% chance
            delay = delay * 0.7
            logger.debug(f"Adding quick browsing pattern for {domain}: {delay:.1f}s")
        
        # Batch requests together occasionally (page loads multiple resources)
        domain_state = self.domain_state.get(domain, {})
        request_times = domain_state.get("request_times", [])
        
        if request_times and len(request_times) >= 3:
            # Check if we're in a burst pattern (multiple recent requests)
            recent_times = [t for t in request_times if time.time() - t < 5.0]
            if len(recent_times) >= 2:
                # In a burst, use shorter delays
                delay = delay * 0.5
                logger.debug(f"In burst pattern for {domain}, reducing delay: {delay:.1f}s")
        
        return delay
    
    def _check_rate_limits(self, domain: str) -> float:
        """
        Check if we're exceeding rate limits and calculate required delay.
        
        Args:
            domain: The domain to check
            
        Returns:
            Additional delay needed in seconds, or 0 if no additional delay needed
        """
        settings = self.get_domain_settings(domain)
        domain_state = self.domain_state.get(domain, {})
        request_times = domain_state.get("request_times", [])
        
        # Remove old request times
        now = time.time()
        recent_times = [t for t in request_times if now - t < 3600]  # Keep last hour
        
        # Check hourly limit
        hourly_limit = settings.get("max_requests_per_hour")
        if hourly_limit and len(recent_times) >= hourly_limit:
            # Calculate time until oldest request is one hour old
            oldest = min(recent_times)
            time_until_slot_opens = (oldest + 3600) - now
            logger.warning(f"Hourly rate limit reached for {domain}. Adding {time_until_slot_opens:.1f}s delay.")
            return max(0, time_until_slot_opens)
        
        # Check minute limit
        minute_limit = settings.get("max_requests_per_minute")
        if minute_limit:
            last_minute = [t for t in recent_times if now - t < 60]
            if len(last_minute) >= minute_limit:
                # Calculate time until oldest request is one minute old
                oldest = min(last_minute)
                time_until_slot_opens = (oldest + 60) - now
                logger.warning(f"Minute rate limit reached for {domain}. Adding {time_until_slot_opens:.1f}s delay.")
                return max(0, time_until_slot_opens)
        
        return 0
    
    def wait(self, domain: str) -> float:
        """
        Wait the appropriate amount of time before making a request to the domain.
        
        Args:
            domain: The domain to wait for
            
        Returns:
            The actual time waited in seconds
        """
        # Get delay time
        delay = self.get_delay(domain)
        
        # Calculate time since last request
        now = time.time()
        last_request = self.last_request_time.get(domain, 0)
        time_since_last = now - last_request
        
        # Only wait if needed
        if time_since_last < delay:
            wait_time = delay - time_since_last
            logger.debug(f"Waiting {wait_time:.2f}s before requesting {domain}")
            time.sleep(wait_time)
            actual_waited = wait_time
        else:
            actual_waited = 0
            logger.debug(f"No need to wait for {domain}, {time_since_last:.2f}s since last request")
        
        # Update last request time
        with self.lock:
            self.last_request_time[domain] = time.time()
            
            # Update request times list
            if domain in self.domain_state:
                self.domain_state[domain]["request_times"].append(time.time())
                # Keep only last 1000 request times to avoid memory issues
                if len(self.domain_state[domain]["request_times"]) > 1000:
                    self.domain_state[domain]["request_times"] = self.domain_state[domain]["request_times"][-1000:]
            
            # Update metrics
            self.domain_metrics[domain]["request_count"] += 1
            self.domain_metrics[domain]["current_delay"] = delay
        
        return actual_waited
    
    def record_response(self, 
                       domain: str, 
                       status_code: int, 
                       response_time: float,
                       response_text: Optional[str] = None,
                       headers: Optional[Dict[str, str]] = None) -> None:
        """
        Record a response to update adaptive throttling.
        
        Args:
            domain: The domain that was requested
            status_code: HTTP status code received
            response_time: Time taken for the request in seconds
            response_text: Optional response text for detection analysis
            headers: Optional response headers for detection analysis
        """
        if not self.adaptive_enabled:
            return
        
        with self.lock:
            # Initialize domain state if not exists
            if domain not in self.domain_state:
                settings = self.get_domain_settings(domain)
                self.domain_state[domain] = {
                    "current_min_delay": settings["min_delay"],
                    "current_max_delay": settings["max_delay"],
                    "backoff_level": 0,
                    "detection_signals": 0,
                    "request_times": [],
                    "backoff_until": None,
                    "last_error_time": None,
                    "consecutive_errors": 0
                }
            
            # Update metrics
            self.domain_metrics[domain]["response_times"].append(response_time)
            if len(self.domain_metrics[domain]["response_times"]) > 100:
                self.domain_metrics[domain]["response_times"] = self.domain_metrics[domain]["response_times"][-100:]
            
            # Check for error status codes
            if status_code >= 400:
                self._handle_error_response(domain, status_code, response_text, headers)
                return
            
            # Reset consecutive errors on success
            self.domain_state[domain]["consecutive_errors"] = 0
            
            # Check for detection signals in successful responses
            detection_signals = self._check_detection_signals(domain, status_code, response_text, headers)
            
            if detection_signals > 0:
                self._apply_backoff(domain, detection_signals)
            else:
                # Gradually reduce backoff if no detection signals
                self._reduce_backoff(domain)
    
    def _handle_error_response(self, 
                              domain: str, 
                              status_code: int,
                              response_text: Optional[str] = None,
                              headers: Optional[Dict[str, str]] = None) -> None:
        """
        Handle error responses.
        
        Args:
            domain: The domain that was requested
            status_code: HTTP status code received
            response_text: Optional response text
            headers: Optional response headers
        """
        # Update metrics
        self.domain_metrics[domain]["error_count"] += 1
        
        # Update domain state
        domain_state = self.domain_state[domain]
        domain_state["last_error_time"] = datetime.now()
        domain_state["consecutive_errors"] += 1
        
        # Log the error
        logger.warning(f"Error response from {domain}: HTTP {status_code}, "
                      f"consecutive errors: {domain_state['consecutive_errors']}")
        
        # Apply backoff based on status code and consecutive errors
        backoff_signals = 0
        
        # Severe error (blocked, rate limited, etc.)
        if status_code in (403, 429, 503):
            backoff_signals = 0.8
            logger.warning(f"Received blocking status code {status_code} from {domain}")
            
            if status_code == 429:
                logger.error(f"Rate limited by {domain}, applying strong backoff")
                backoff_signals = 1.0
                
                # Record block event
                self.domain_metrics[domain]["block_count"] += 1
        
        # Client errors
        elif 400 <= status_code < 500:
            backoff_signals = 0.3
            
            # Check for CAPTCHA or anti-bot measures in the response
            if response_text and self._contains_captcha_signals(response_text):
                logger.error(f"CAPTCHA detected in response from {domain}")
                backoff_signals = 0.9
                
                # Record captcha event
                self.domain_metrics[domain]["captcha_count"] += 1
        
        # Server errors
        elif 500 <= status_code < 600:
            backoff_signals = 0.5
        
        # Increase backoff signals based on consecutive errors
        if domain_state["consecutive_errors"] > 1:
            # Exponentially increase backoff with consecutive errors
            backoff_signals *= (1.0 + 0.5 * math.log(domain_state["consecutive_errors"]))
        
        # Apply backoff
        if backoff_signals > 0:
            self._apply_backoff(domain, backoff_signals)
    
    def _contains_captcha_signals(self, response_text: str) -> bool:
        """
        Check if the response contains signals indicating a CAPTCHA or anti-bot page.
        
        Args:
            response_text: The response text to check
            
        Returns:
            True if CAPTCHA signals detected, False otherwise
        """
        if not response_text:
            return False
        
        # Common CAPTCHA and anti-bot terms
        captcha_signals = [
            "captcha", "recaptcha", "security check", "bot check", "are you human",
            "verify you are human", "automated request", "suspicious activity",
            "unusual traffic", "too many requests", "access denied",
            "blocked", "security challenge", "verify your identity"
        ]
        
        lower_text = response_text.lower()
        
        # Check for CAPTCHA signals
        for signal in captcha_signals:
            if signal in lower_text:
                logger.warning(f"CAPTCHA signal detected: '{signal}'")
                return True
        
        # Check for CAPTCHA libraries
        captcha_libraries = [
            "grecaptcha", "recaptcha/api.js", "hcaptcha", "captcha.js",
            "funcaptcha", "arkoselabs", "cloudflare-static", "challenge-platform"
        ]
        
        for lib in captcha_libraries:
            if lib in lower_text:
                logger.warning(f"CAPTCHA library detected: '{lib}'")
                return True
        
        return False
    
    def _check_detection_signals(self,
                                domain: str,
                                status_code: int,
                                response_text: Optional[str] = None,
                                headers: Optional[Dict[str, str]] = None) -> float:
        """
        Check for signals that our scraping might be detected.
        
        Args:
            domain: The domain that was requested
            status_code: HTTP status code received
            response_text: Optional response text
            headers: Optional response headers
            
        Returns:
            Detection signal strength (0.0-1.0)
        """
        signals = 0.0
        
        # Check for CAPTCHA in the response
        if response_text and self._contains_captcha_signals(response_text):
            signals += 0.9
            logger.warning(f"CAPTCHA detected in response from {domain}")
            
            # Record captcha event
            self.domain_metrics[domain]["captcha_count"] += 1
        
        # Check headers for detection signals
        if headers:
            # Check for rate limiting headers
            rate_limit_headers = ["x-rate-limit", "retry-after", "x-ratelimit-remaining",
                                 "x-ratelimit-reset", "ratelimit-limit"]
            
            for header in rate_limit_headers:
                if any(h.lower() == header for h in headers):
                    signals += 0.7
                    logger.warning(f"Rate limit header detected: {header}")
                    break
            
            # Check for unusual server behavior
            if "cf-ray" in headers and status_code in (403, 503):
                # Cloudflare is likely blocking us
                signals += 0.8
                logger.warning("Cloudflare protection detected")
            
            # Check for browser verification headers
            if "x-frame-options" in headers and "x-content-type-options" in headers and status_code == 403:
                signals += 0.6
                logger.warning("Possible anti-bot measures detected in headers")
        
        # Consider response time anomalies
        response_times = self.domain_metrics[domain]["response_times"]
        if len(response_times) >= 5:
            avg_time = sum(response_times[-5:]) / 5
            if response_times[-1] > avg_time * 3:
                # Unusually slow response could indicate processing by anti-bot systems
                signals += 0.4
                logger.warning(f"Unusually slow response time: {response_times[-1]:.2f}s vs avg {avg_time:.2f}s")
        
        # Apply detection sensitivity
        signals *= self.detection_sensitivity
        
        return signals
    
    def _apply_backoff(self, domain: str, signal_strength: float) -> None:
        """
        Apply backoff strategy based on detection signals.
        
        Args:
            domain: The domain to apply backoff for
            signal_strength: Strength of detection signal (0.0-1.0)
        """
        with self.lock:
            # Get current domain state
            domain_state = self.domain_state[domain]
            settings = self.get_domain_settings(domain)
            
            # Increase backoff level
            old_backoff = domain_state["backoff_level"]
            domain_state["backoff_level"] += signal_strength
            
            # Cap backoff level
            domain_state["backoff_level"] = min(domain_state["backoff_level"], 5.0)
            
            # Calculate backoff duration based on level
            backoff_minutes = 0
            if domain_state["backoff_level"] < 1.0:
                backoff_minutes = 1
            elif domain_state["backoff_level"] < 2.0:
                backoff_minutes = 5
            elif domain_state["backoff_level"] < 3.0:
                backoff_minutes = 15
            elif domain_state["backoff_level"] < 4.0:
                backoff_minutes = 30
            else:
                backoff_minutes = 60
            
            # Apply random jitter to backoff time (±20%)
            jitter = random.uniform(0.8, 1.2)
            backoff_minutes *= jitter
            
            # Set backoff until time
            now = datetime.now()
            domain_state["backoff_until"] = now + timedelta(minutes=backoff_minutes)
            
            # Increase delays based on backoff level
            level_factor = 1.0 + domain_state["backoff_level"] * 0.5
            domain_state["current_min_delay"] = settings["min_delay"] * level_factor
            domain_state["current_max_delay"] = settings["max_delay"] * level_factor
            
            # Cap maximum delay
            domain_state["current_max_delay"] = min(domain_state["current_max_delay"], 30.0)
            
            logger.warning(f"Applied backoff for {domain}: level {old_backoff:.1f}->{domain_state['backoff_level']:.1f}, "
                          f"delay {domain_state['current_min_delay']:.1f}-{domain_state['current_max_delay']:.1f}s, "
                          f"duration {backoff_minutes:.1f} minutes")
            
            # Record backoff event
            self.domain_metrics[domain]["backoff_events"].append({
                "timestamp": now.isoformat(),
                "level": domain_state["backoff_level"],
                "duration_minutes": backoff_minutes,
                "signal_strength": signal_strength
            })
    
    def _reduce_backoff(self, domain: str) -> None:
        """
        Gradually reduce backoff level after successful requests.
        
        Args:
            domain: The domain to reduce backoff for
        """
        with self.lock:
            # Get current domain state
            domain_state = self.domain_state[domain]
            settings = self.get_domain_settings(domain)
            
            # Only reduce if we have a backoff level
            if domain_state["backoff_level"] > 0:
                # Gradually reduce backoff level
                domain_state["backoff_level"] *= 0.95
                
                # Reset backoff if it's very low
                if domain_state["backoff_level"] < 0.1:
                    domain_state["backoff_level"] = 0
                    domain_state["backoff_until"] = None
                    domain_state["current_min_delay"] = settings["min_delay"]
                    domain_state["current_max_delay"] = settings["max_delay"]
                    logger.info(f"Backoff fully reset for {domain}")
                else:
                    # Update delays based on reduced backoff level
                    level_factor = 1.0 + domain_state["backoff_level"] * 0.5
                    domain_state["current_min_delay"] = settings["min_delay"] * level_factor
                    domain_state["current_max_delay"] = settings["max_delay"] * level_factor
                    
                    logger.debug(f"Reduced backoff level for {domain} to {domain_state['backoff_level']:.2f}")
    
    def get_domain_status(self, domain: str) -> Dict[str, Any]:
        """
        Get current status for a domain.
        
        Args:
            domain: The domain to get status for
            
        Returns:
            Dictionary with domain status
        """
        with self.lock:
            if domain not in self.domain_state:
                return {
                    "status": "normal",
                    "backoff_level": 0,
                    "current_delay_range": [self.default_min_delay, self.default_max_delay],
                    "request_count": 0,
                    "error_count": 0,
                    "in_backoff": False,
                    "backoff_remaining": 0
                }
            
            domain_state = self.domain_state[domain]
            metrics = self.domain_metrics[domain]
            
            # Determine current status
            status = "normal"
            backoff_remaining = 0
            
            if domain_state["backoff_until"]:
                now = datetime.now()
                if now < domain_state["backoff_until"]:
                    status = "backoff"
                    backoff_remaining = (domain_state["backoff_until"] - now).total_seconds()
            
            if domain_state["consecutive_errors"] >= 3:
                status = "errors"
            
            if metrics["captcha_count"] > 0:
                status = "captcha_detected"
            
            if metrics["block_count"] > 0:
                status = "blocked"
            
            return {
                "status": status,
                "backoff_level": domain_state["backoff_level"],
                "current_delay_range": [
                    domain_state["current_min_delay"],
                    domain_state["current_max_delay"]
                ],
                "request_count": metrics["request_count"],
                "error_count": metrics["error_count"],
                "captcha_count": metrics["captcha_count"],
                "block_count": metrics["block_count"],
                "consecutive_errors": domain_state["consecutive_errors"],
                "in_backoff": domain_state["backoff_until"] is not None,
                "backoff_remaining": backoff_remaining
            }
    
    def get_all_domain_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status for all domains.
        
        Returns:
            Dictionary with status for all domains
        """
        statuses = {}
        
        with self.lock:
            for domain in self.domain_state:
                statuses[domain] = self.get_domain_status(domain)
        
        return statuses
    
    def should_rotate_ip(self, domain: str) -> Tuple[bool, str]:
        """
        Determine if IP rotation is recommended for this domain.
        
        Args:
            domain: The domain to check
            
        Returns:
            Tuple of (should_rotate, reason)
        """
        with self.lock:
            if domain not in self.domain_state:
                return (False, "No history for this domain")
            
            domain_state = self.domain_state[domain]
            metrics = self.domain_metrics[domain]
            
            # Check for strong blocking signals
            if metrics["block_count"] > 0:
                return (True, f"Domain has blocked requests {metrics['block_count']} times")
            
            if metrics["captcha_count"] > 0:
                return (True, f"Domain has shown CAPTCHAs {metrics['captcha_count']} times")
            
            if domain_state["consecutive_errors"] >= 5:
                return (True, f"Domain has {domain_state['consecutive_errors']} consecutive errors")
            
            if domain_state["backoff_level"] >= 3.0:
                return (True, f"Domain has high backoff level: {domain_state['backoff_level']:.1f}")
            
            # No need to rotate
            return (False, "No rotation needed")
    
    def reset_for_domain(self, domain: str) -> None:
        """
        Reset all throttling state for a domain.
        
        This is useful after IP rotation.
        
        Args:
            domain: The domain to reset
        """
        with self.lock:
            # Reset domain state to default settings
            if domain in self.domain_settings:
                settings = self.domain_settings[domain]
                self.domain_state[domain] = {
                    "current_min_delay": settings["min_delay"],
                    "current_max_delay": settings["max_delay"],
                    "backoff_level": 0,
                    "detection_signals": 0,
                    "request_times": [],
                    "backoff_until": None,
                    "last_error_time": None,
                    "consecutive_errors": 0
                }
            else:
                # Remove domain state if no custom settings
                if domain in self.domain_state:
                    del self.domain_state[domain]
            
            # Reset last request time
            if domain in self.last_request_time:
                del self.last_request_time[domain]
            
            # Reset metrics but keep historical counts
            if domain in self.domain_metrics:
                # Keep historical counts but reset current state
                self.domain_metrics[domain]["response_times"] = []
                self.domain_metrics[domain]["current_delay"] = None
            
            logger.info(f"Reset throttling state for {domain}")
    
    def get_recommendations(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get recommendations for optimizing requests to a domain.
        
        Args:
            domain: The domain to get recommendations for
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        with self.lock:
            if domain not in self.domain_state:
                return []
            
            domain_state = self.domain_state[domain]
            metrics = self.domain_metrics[domain]
            settings = self.get_domain_settings(domain)
            
            # Check if delays are too short
            if metrics["error_count"] > 5 and metrics["request_count"] > 0:
                error_rate = metrics["error_count"] / metrics["request_count"]
                
                if error_rate > 0.1 and settings["min_delay"] < 2.0:
                    recommendations.append({
                        "type": "increase_delay",
                        "message": f"Consider increasing minimum delay (current: {settings['min_delay']}s)",
                        "details": f"Error rate is {error_rate:.1%}, which suggests request rate may be too high",
                        "severity": "medium"
                    })
            
            # Check if we should use random user agents
            if metrics["captcha_count"] > 0:
                recommendations.append({
                    "type": "user_agent_rotation",
                    "message": "Consider using random user agents",
                    "details": f"CAPTCHAs detected ({metrics['captcha_count']} times)",
                    "severity": "high"
                })
            
            # Check if we need to rotate IPs
            should_rotate, reason = self.should_rotate_ip(domain)
            if should_rotate:
                recommendations.append({
                    "type": "ip_rotation",
                    "message": "Consider rotating IP address",
                    "details": reason,
                    "severity": "high"
                })
            
            # Check if we should implement session cookies
            if domain_state["backoff_level"] >= 2.0:
                recommendations.append({
                    "type": "session_cookies",
                    "message": "Consider implementing proper session handling with cookies",
                    "details": f"High backoff level ({domain_state['backoff_level']:.1f}) indicates possible detection",
                    "severity": "medium"
                })
            
            # Check if request pattern is too regular
            response_times = metrics["response_times"]
            if len(response_times) >= 10:
                time_diffs = []
                for i in range(1, len(response_times)):
                    time_diffs.append(abs(response_times[i] - response_times[i-1]))
                
                avg_diff = sum(time_diffs) / len(time_diffs)
                variance = sum((d - avg_diff) ** 2 for d in time_diffs) / len(time_diffs)
                
                if variance < 0.1 and avg_diff < 1.0:
                    recommendations.append({
                        "type": "irregular_timing",
                        "message": "Request timing is too regular",
                        "details": "Low variance in request timing may trigger bot detection",
                        "severity": "medium"
                    })
        
        return recommendations

# Create a global instance for convenience
REQUEST_THROTTLER = RequestThrottler()