"""
Core monitoring system for SmartScrape components.

This module provides classes for monitoring the health and performance
of SmartScrape services, with a focus on resource management components.
"""

import logging
import threading
import time
import json
import datetime
from typing import Dict, Any, Optional, List, Set, Callable
from abc import ABC, abstractmethod
from enum import Enum
import uuid
import requests

from core.service_interface import BaseService
from core.service_registry import ServiceRegistry

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status values for services and components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class HealthCheck(ABC):
    """Base class for all health checks."""
    
    def __init__(self, name: str, service: Optional[BaseService] = None):
        self.name = name
        self.service = service
    
    @abstractmethod
    def check(self) -> Dict[str, Any]:
        """
        Perform the health check.
        
        Returns:
            Dict with health check results including:
            - status: HealthStatus value
            - details: Additional information about the check
            - metrics: Optional check-specific metrics
        """
        pass

class SessionManagerHealthCheck(HealthCheck):
    """Health checks for the SessionManager service."""
    
    def __init__(self, session_manager):
        super().__init__("session_manager_health", session_manager)
        self.test_urls = {
            "http": "https://httpbin.org/get",
            "browser": "https://httpbin.org/html"
        }
    
    def check(self) -> Dict[str, Any]:
        """Perform all session manager health checks."""
        results = {}
        metrics = {}
        
        # Start with overall status as healthy
        status = HealthStatus.HEALTHY
        
        # Run individual checks
        http_result = self.check_http_connectivity()
        if http_result["status"] != HealthStatus.HEALTHY:
            status = http_result["status"]
        results["http_connectivity"] = http_result
        
        browser_result = self.check_browser_sessions()
        if browser_result["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = browser_result["status"]
        results["browser_sessions"] = browser_result
        
        pool_result = self.check_session_pools()
        if pool_result["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = pool_result["status"]
        results["session_pools"] = pool_result
        
        # Collect metrics
        metrics["total_sessions"] = pool_result.get("metrics", {}).get("total_sessions", 0)
        metrics["active_sessions"] = pool_result.get("metrics", {}).get("active_sessions", 0)
        
        return {
            "status": status.value,
            "details": results,
            "metrics": metrics
        }
    
    def check_http_connectivity(self) -> Dict[str, Any]:
        """Test HTTP sessions by making a request to a test URL."""
        metrics = {}
        
        try:
            # Get a session for the test domain
            domain = "httpbin.org"
            start_time = time.time()
            
            # Make a test request
            session = self.service.get_session(domain)
            response = session.get(self.test_urls["http"], timeout=10)
            
            # Calculate metrics
            latency = time.time() - start_time
            metrics["response_time"] = round(latency * 1000, 2)  # in ms
            metrics["status_code"] = response.status_code
            
            if response.status_code == 200:
                return {
                    "status": HealthStatus.HEALTHY,
                    "details": f"HTTP session test successful (latency: {metrics['response_time']}ms)",
                    "metrics": metrics
                }
            else:
                return {
                    "status": HealthStatus.DEGRADED,
                    "details": f"HTTP session test returned non-200 status: {response.status_code}",
                    "metrics": metrics
                }
        except Exception as e:
            logger.warning(f"HTTP session health check failed: {str(e)}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "details": f"HTTP session test error: {str(e)}",
                "metrics": metrics
            }
    
    def check_browser_sessions(self) -> Dict[str, Any]:
        """Test browser sessions if available."""
        metrics = {}
        
        # Check if browser sessions are supported
        if not hasattr(self.service, "get_browser_session"):
            return {
                "status": HealthStatus.UNKNOWN,
                "details": "Browser sessions not supported",
                "metrics": metrics
            }
        
        try:
            # Get a browser session
            start_time = time.time()
            browser = self.service.get_browser_session()
            
            # Navigate to test URL
            page = browser.new_page()
            page.goto(self.test_urls["browser"])
            
            # Calculate metrics
            latency = time.time() - start_time
            metrics["browser_launch_time"] = round(latency * 1000, 2)  # in ms
            
            # Check if page loaded successfully
            title = page.title()
            
            # Close the page
            page.close()
            
            return {
                "status": HealthStatus.HEALTHY,
                "details": f"Browser session test successful (launch time: {metrics['browser_launch_time']}ms)",
                "metrics": metrics
            }
        except Exception as e:
            logger.warning(f"Browser session health check failed: {str(e)}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "details": f"Browser session test error: {str(e)}",
                "metrics": metrics
            }
    
    def check_session_pools(self) -> Dict[str, Any]:
        """Test session pool status and metrics."""
        metrics = {}
        
        try:
            # Get session pool metrics (implementation depends on SessionManager)
            if hasattr(self.service, "_sessions"):
                total_sessions = len(self.service._sessions)
                metrics["total_sessions"] = total_sessions
                metrics["active_sessions"] = total_sessions  # Simple implementation
                
                # Check if the pool is near capacity (if applicable)
                pool_capacity = getattr(self.service, "_max_sessions", None)
                if pool_capacity:
                    metrics["pool_capacity"] = pool_capacity
                    metrics["pool_utilization"] = round((total_sessions / pool_capacity) * 100, 2)
                    
                    if metrics["pool_utilization"] > 90:
                        return {
                            "status": HealthStatus.DEGRADED,
                            "details": f"Session pool at {metrics['pool_utilization']}% capacity",
                            "metrics": metrics
                        }
            
            return {
                "status": HealthStatus.HEALTHY,
                "details": "Session pools operating normally",
                "metrics": metrics
            }
        except Exception as e:
            logger.warning(f"Session pool health check failed: {str(e)}")
            return {
                "status": HealthStatus.DEGRADED,
                "details": f"Session pool check error: {str(e)}",
                "metrics": metrics
            }

class ProxyManagerHealthCheck(HealthCheck):
    """Health checks for the ProxyManager service."""
    
    def __init__(self, proxy_manager):
        super().__init__("proxy_manager_health", proxy_manager)
    
    def check(self) -> Dict[str, Any]:
        """Perform all proxy manager health checks."""
        results = {}
        metrics = {}
        
        # Start with overall status as healthy
        status = HealthStatus.HEALTHY
        
        # Run individual checks
        availability_result = self.check_proxy_availability()
        if availability_result["status"] != HealthStatus.HEALTHY:
            status = availability_result["status"]
        results["proxy_availability"] = availability_result
        
        performance_result = self.check_proxy_performance()
        if performance_result["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = performance_result["status"]
        results["proxy_performance"] = performance_result
        
        success_rate_result = self.check_proxy_success_rate()
        if success_rate_result["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = success_rate_result["status"]
        results["proxy_success_rate"] = success_rate_result
        
        # Collect metrics
        metrics["available_proxies"] = availability_result.get("metrics", {}).get("available_proxies", 0)
        metrics["blacklisted_proxies"] = availability_result.get("metrics", {}).get("blacklisted_proxies", 0)
        metrics["average_proxy_speed"] = performance_result.get("metrics", {}).get("average_speed", 0)
        metrics["overall_success_rate"] = success_rate_result.get("metrics", {}).get("overall_success_rate", 0)
        
        return {
            "status": status.value,
            "details": results,
            "metrics": metrics
        }
    
    def check_proxy_availability(self) -> Dict[str, Any]:
        """Check proxy availability and count."""
        metrics = {}
        
        try:
            # Get proxy metrics
            if hasattr(self.service, "_proxies"):
                all_proxies = list(self.service._proxies)
                
                # Count by status
                available_count = 0
                blacklisted_count = 0
                
                if hasattr(self.service, "ProxyStatus"):  # Check if enum is available
                    ProxyStatus = self.service.ProxyStatus
                    available_count = sum(1 for p in all_proxies if getattr(p, "status", None) != ProxyStatus.BLACKLISTED and getattr(p, "status", None) != ProxyStatus.REMOVED)
                    blacklisted_count = sum(1 for p in all_proxies if getattr(p, "status", None) == ProxyStatus.BLACKLISTED)
                else:
                    # Simpler count if ProxyStatus enum is not accessible
                    available_count = len(all_proxies)
                    blacklisted_count = 0
                
                metrics["total_proxies"] = len(all_proxies)
                metrics["available_proxies"] = available_count
                metrics["blacklisted_proxies"] = blacklisted_count
                
                # Determine status based on availability
                if available_count == 0:
                    return {
                        "status": HealthStatus.UNHEALTHY,
                        "details": "No proxies available",
                        "metrics": metrics
                    }
                elif available_count < 3:  # Adjust threshold as needed
                    return {
                        "status": HealthStatus.DEGRADED,
                        "details": f"Low proxy availability ({available_count} available)",
                        "metrics": metrics
                    }
            
            return {
                "status": HealthStatus.HEALTHY,
                "details": "Proxy availability normal",
                "metrics": metrics
            }
        except Exception as e:
            logger.warning(f"Proxy availability check failed: {str(e)}")
            return {
                "status": HealthStatus.DEGRADED,
                "details": f"Proxy availability check error: {str(e)}",
                "metrics": metrics
            }
    
    def check_proxy_performance(self) -> Dict[str, Any]:
        """Check proxy speed and performance metrics."""
        metrics = {}
        
        try:
            # Get performance metrics if available
            proxy_speeds = []
            
            if hasattr(self.service, "_proxies"):
                for proxy in self.service._proxies:
                    speed = getattr(proxy, "avg_response_time", None)
                    if speed:
                        proxy_speeds.append(speed)
            
            if proxy_speeds:
                avg_speed = sum(proxy_speeds) / len(proxy_speeds)
                metrics["average_speed"] = round(avg_speed, 2)
                metrics["fastest_proxy"] = round(min(proxy_speeds), 2)
                metrics["slowest_proxy"] = round(max(proxy_speeds), 2)
                
                # Determine status based on average speed
                if avg_speed > 5000:  # 5 seconds
                    return {
                        "status": HealthStatus.DEGRADED,
                        "details": f"Slow average proxy speed: {avg_speed}ms",
                        "metrics": metrics
                    }
            
            return {
                "status": HealthStatus.HEALTHY,
                "details": "Proxy performance normal",
                "metrics": metrics
            }
        except Exception as e:
            logger.warning(f"Proxy performance check failed: {str(e)}")
            return {
                "status": HealthStatus.UNKNOWN,
                "details": f"Proxy performance check error: {str(e)}",
                "metrics": metrics
            }
    
    def check_proxy_success_rate(self) -> Dict[str, Any]:
        """Check proxy success rates."""
        metrics = {}
        
        try:
            # Get success rate metrics if available
            success_rates = []
            
            if hasattr(self.service, "_proxies"):
                for proxy in self.service._proxies:
                    success = getattr(proxy, "success_count", 0)
                    fail = getattr(proxy, "failure_count", 0)
                    
                    if success + fail > 0:
                        rate = success / (success + fail)
                        success_rates.append(rate)
            
            if success_rates:
                avg_rate = sum(success_rates) / len(success_rates)
                metrics["overall_success_rate"] = round(avg_rate * 100, 2)
                
                # Determine status based on success rate
                if avg_rate < 0.5:  # Less than 50% success
                    return {
                        "status": HealthStatus.UNHEALTHY,
                        "details": f"Low proxy success rate: {metrics['overall_success_rate']}%",
                        "metrics": metrics
                    }
                elif avg_rate < 0.8:  # Less than 80% success
                    return {
                        "status": HealthStatus.DEGRADED,
                        "details": f"Mediocre proxy success rate: {metrics['overall_success_rate']}%",
                        "metrics": metrics
                    }
            
            return {
                "status": HealthStatus.HEALTHY,
                "details": "Proxy success rate normal",
                "metrics": metrics
            }
        except Exception as e:
            logger.warning(f"Proxy success rate check failed: {str(e)}")
            return {
                "status": HealthStatus.UNKNOWN,
                "details": f"Proxy success rate check error: {str(e)}",
                "metrics": metrics
            }

class RateLimiterHealthCheck(HealthCheck):
    """Health checks for the RateLimiter service."""
    
    def __init__(self, rate_limiter):
        super().__init__("rate_limiter_health", rate_limiter)
    
    def check(self) -> Dict[str, Any]:
        """Perform all rate limiter health checks."""
        results = {}
        metrics = {}
        
        # Start with overall status as healthy
        status = HealthStatus.HEALTHY
        
        # Run individual checks
        rate_status_result = self.check_rate_status()
        if rate_status_result["status"] != HealthStatus.HEALTHY:
            status = rate_status_result["status"]
        results["rate_status"] = rate_status_result
        
        rate_budget_result = self.check_rate_budget()
        if rate_budget_result["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = rate_budget_result["status"]
        results["rate_budget"] = rate_budget_result
        
        domain_blocks_result = self.check_domain_blocks()
        if domain_blocks_result["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = domain_blocks_result["status"]
        results["domain_blocks"] = domain_blocks_result
        
        # Collect metrics
        metrics["domains_tracked"] = rate_status_result.get("metrics", {}).get("domains_tracked", 0)
        metrics["domains_rate_limited"] = domain_blocks_result.get("metrics", {}).get("domains_rate_limited", 0)
        
        return {
            "status": status.value,
            "details": results,
            "metrics": metrics
        }
    
    def check_rate_status(self) -> Dict[str, Any]:
        """Check rate limiter status for all domains."""
        metrics = {}
        
        try:
            domains_tracked = 0
            
            if hasattr(self.service, "_limits"):
                domains_tracked = len(self.service._limits)
            
            metrics["domains_tracked"] = domains_tracked
            
            return {
                "status": HealthStatus.HEALTHY,
                "details": f"Rate limiter tracking {domains_tracked} domains",
                "metrics": metrics
            }
        except Exception as e:
            logger.warning(f"Rate status check failed: {str(e)}")
            return {
                "status": HealthStatus.DEGRADED,
                "details": f"Rate status check error: {str(e)}",
                "metrics": metrics
            }
    
    def check_rate_budget(self) -> Dict[str, Any]:
        """Check remaining rate budget for domains."""
        metrics = {}
        
        try:
            # This is highly dependent on the RateLimiter implementation
            # Adjust based on your actual implementation
            low_budget_domains = []
            
            if hasattr(self.service, "_request_counts") and hasattr(self.service, "_limits"):
                current_time = time.time()
                minute_window = current_time - 60
                
                for domain, counts in self.service._request_counts.items():
                    # Filter counts for last minute
                    recent_count = sum(1 for t in counts if t > minute_window)
                    
                    # Get limit for this domain
                    domain_limit = self.service._get_domain_limits(domain).get("requests_per_minute", 60)
                    
                    # Calculate remaining budget
                    remaining = domain_limit - recent_count
                    
                    if remaining < 0.2 * domain_limit:  # Less than 20% remaining
                        low_budget_domains.append(domain)
            
            metrics["low_budget_domains"] = len(low_budget_domains)
            metrics["low_budget_domain_list"] = low_budget_domains
            
            if low_budget_domains:
                return {
                    "status": HealthStatus.DEGRADED,
                    "details": f"{len(low_budget_domains)} domains have low rate limit budget",
                    "metrics": metrics
                }
            
            return {
                "status": HealthStatus.HEALTHY,
                "details": "Rate limit budgets normal",
                "metrics": metrics
            }
        except Exception as e:
            logger.warning(f"Rate budget check failed: {str(e)}")
            return {
                "status": HealthStatus.UNKNOWN,
                "details": f"Rate budget check error: {str(e)}",
                "metrics": metrics
            }
    
    def check_domain_blocks(self) -> Dict[str, Any]:
        """Check for blocked domains due to rate limiting."""
        metrics = {}
        
        try:
            # This depends on how your RateLimiter tracks blocked domains
            blocked_domains = []
            
            # Example implementation - adjust based on your actual implementation
            if hasattr(self.service, "_blocked_until"):
                current_time = time.time()
                blocked_domains = [domain for domain, until_time in self.service._blocked_until.items() 
                                  if until_time > current_time]
            
            metrics["domains_rate_limited"] = len(blocked_domains)
            metrics["blocked_domain_list"] = blocked_domains
            
            if len(blocked_domains) > 5:  # Adjust threshold as needed
                return {
                    "status": HealthStatus.DEGRADED,
                    "details": f"{len(blocked_domains)} domains currently rate limited",
                    "metrics": metrics
                }
            elif blocked_domains:
                return {
                    "status": HealthStatus.HEALTHY,
                    "details": f"{len(blocked_domains)} domains currently rate limited",
                    "metrics": metrics
                }
            
            return {
                "status": HealthStatus.HEALTHY,
                "details": "No domains currently rate limited",
                "metrics": metrics
            }
        except Exception as e:
            logger.warning(f"Domain blocks check failed: {str(e)}")
            return {
                "status": HealthStatus.UNKNOWN,
                "details": f"Domain blocks check error: {str(e)}",
                "metrics": metrics
            }

class CircuitBreakerHealthCheck(HealthCheck):
    """Health checks for the CircuitBreakerManager service."""
    
    def __init__(self, circuit_breaker_manager):
        super().__init__("circuit_breaker_health", circuit_breaker_manager)
    
    def check(self) -> Dict[str, Any]:
        """Perform all circuit breaker health checks."""
        results = {}
        metrics = {}
        
        # Start with overall status as healthy
        status = HealthStatus.HEALTHY
        
        # Run individual checks
        circuit_status_result = self.check_circuit_status()
        if circuit_status_result["status"] != HealthStatus.HEALTHY:
            status = circuit_status_result["status"]
        results["circuit_status"] = circuit_status_result
        
        open_circuits_result = self.check_open_circuits()
        if open_circuits_result["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = open_circuits_result["status"]
        results["open_circuits"] = open_circuits_result
        
        half_open_circuits_result = self.check_half_open_circuits()
        if half_open_circuits_result["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = half_open_circuits_result["status"]
        results["half_open_circuits"] = half_open_circuits_result
        
        # Collect metrics
        metrics["total_circuits"] = circuit_status_result.get("metrics", {}).get("total_circuits", 0)
        metrics["open_circuits"] = open_circuits_result.get("metrics", {}).get("open_count", 0)
        metrics["half_open_circuits"] = half_open_circuits_result.get("metrics", {}).get("half_open_count", 0)
        
        return {
            "status": status.value,
            "details": results,
            "metrics": metrics
        }
    
    def check_circuit_status(self) -> Dict[str, Any]:
        """Check status of all circuit breakers."""
        metrics = {}
        
        try:
            if not hasattr(self.service, "_circuit_breakers"):
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "No circuit breakers found",
                    "metrics": metrics
                }
            
            total_circuits = len(self.service._circuit_breakers)
            metrics["total_circuits"] = total_circuits
            
            if total_circuits == 0:
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "No circuit breakers registered",
                    "metrics": metrics
                }
            
            return {
                "status": HealthStatus.HEALTHY,
                "details": f"Monitoring {total_circuits} circuit breakers",
                "metrics": metrics
            }
        except Exception as e:
            logger.warning(f"Circuit status check failed: {str(e)}")
            return {
                "status": HealthStatus.DEGRADED,
                "details": f"Circuit status check error: {str(e)}",
                "metrics": metrics
            }
    
    def check_open_circuits(self) -> Dict[str, Any]:
        """Check for open circuits."""
        metrics = {}
        
        try:
            if not hasattr(self.service, "_circuit_breakers"):
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "No circuit breakers found",
                    "metrics": metrics
                }
            
            open_circuits = []
            CircuitState = None
            
            # Try to find the CircuitState enum
            if hasattr(self.service, "CircuitState"):
                CircuitState = self.service.CircuitState
            else:
                # Look for it in any circuit breaker
                for circuit in self.service._circuit_breakers.values():
                    if hasattr(circuit, "_state") and hasattr(circuit._state, "__class__"):
                        CircuitState = circuit._state.__class__
                        break
            
            # Count open circuits
            if CircuitState:
                for name, circuit in self.service._circuit_breakers.items():
                    if circuit.state == CircuitState.OPEN:
                        open_circuits.append(name)
            else:
                # Fallback if we can't find the enum
                for name, circuit in self.service._circuit_breakers.items():
                    if getattr(circuit, "_state", None) == "open":
                        open_circuits.append(name)
            
            metrics["open_count"] = len(open_circuits)
            metrics["open_circuits"] = open_circuits
            
            if len(open_circuits) > 5:  # Adjust threshold as needed
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "details": f"Multiple circuits open ({len(open_circuits)})",
                    "metrics": metrics
                }
            elif len(open_circuits) > 0:
                return {
                    "status": HealthStatus.DEGRADED,
                    "details": f"{len(open_circuits)} circuits open",
                    "metrics": metrics
                }
            
            return {
                "status": HealthStatus.HEALTHY,
                "details": "No open circuits",
                "metrics": metrics
            }
        except Exception as e:
            logger.warning(f"Open circuits check failed: {str(e)}")
            return {
                "status": HealthStatus.UNKNOWN,
                "details": f"Open circuits check error: {str(e)}",
                "metrics": metrics
            }
    
    def check_half_open_circuits(self) -> Dict[str, Any]:
        """Check for half-open circuits (recovery in progress)."""
        metrics = {}
        
        try:
            if not hasattr(self.service, "_circuit_breakers"):
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "No circuit breakers found",
                    "metrics": metrics
                }
            
            half_open_circuits = []
            CircuitState = None
            
            # Try to find the CircuitState enum
            if hasattr(self.service, "CircuitState"):
                CircuitState = self.service.CircuitState
            else:
                # Look for it in any circuit breaker
                for circuit in self.service._circuit_breakers.values():
                    if hasattr(circuit, "_state") and hasattr(circuit._state, "__class__"):
                        CircuitState = circuit._state.__class__
                        break
            
            # Count half-open circuits
            if CircuitState:
                for name, circuit in self.service._circuit_breakers.items():
                    if circuit.state == CircuitState.HALF_OPEN:
                        half_open_circuits.append(name)
            else:
                # Fallback if we can't find the enum
                for name, circuit in self.service._circuit_breakers.items():
                    if getattr(circuit, "_state", None) == "half_open":
                        half_open_circuits.append(name)
            
            metrics["half_open_count"] = len(half_open_circuits)
            metrics["half_open_circuits"] = half_open_circuits
            
            # Half-open is actually good - it means recovery is being attempted
            return {
                "status": HealthStatus.HEALTHY,
                "details": f"{len(half_open_circuits)} circuits in recovery (half-open)",
                "metrics": metrics
            }
        except Exception as e:
            logger.warning(f"Half-open circuits check failed: {str(e)}")
            return {
                "status": HealthStatus.UNKNOWN,
                "details": f"Half-open circuits check error: {str(e)}",
                "metrics": metrics
            }

class SemanticSearchHealthCheck(HealthCheck):
    """Health checks for semantic search capabilities."""
    
    def __init__(self, semantic_search_service=None):
        super().__init__("semantic_search_health", semantic_search_service)
        self.test_queries = [
            "test semantic search functionality",
            "sample query for performance testing"
        ]
    
    def check(self) -> Dict[str, Any]:
        """Perform semantic search health checks."""
        results = {}
        metrics = {}
        
        # Start with overall status as healthy
        status = HealthStatus.HEALTHY
        
        # Run individual checks
        model_status = self.check_model_availability()
        if model_status["status"] != HealthStatus.HEALTHY:
            status = model_status["status"]
        results["model_availability"] = model_status
        
        performance_result = self.check_query_performance()
        if performance_result["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = performance_result["status"]
        results["query_performance"] = performance_result
        
        accuracy_result = self.check_search_accuracy()
        if accuracy_result["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = accuracy_result["status"]
        results["search_accuracy"] = accuracy_result
        
        # Collect metrics
        metrics.update(model_status.get("metrics", {}))
        metrics.update(performance_result.get("metrics", {}))
        metrics.update(accuracy_result.get("metrics", {}))
        
        return {
            "status": status.value,
            "details": results,
            "metrics": metrics
        }
    
    def check_model_availability(self) -> Dict[str, Any]:
        """Check if semantic search models are loaded and available."""
        metrics = {}
        
        try:
            if not self.service:
                # Try to get from service registry
                from core.service_registry import ServiceRegistry
                registry = ServiceRegistry()
                self.service = registry.get_service("semantic_search") or registry.get_service("universal_intent_analyzer")
            
            if not self.service:
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "Semantic search service not found",
                    "metrics": metrics
                }
            
            # Check if models are loaded
            models_loaded = False
            model_names = []
            
            # Check for sentence transformer model
            if hasattr(self.service, 'sentence_transformer') and self.service.sentence_transformer:
                models_loaded = True
                model_names.append("sentence_transformer")
            
            # Check for spaCy model
            if hasattr(self.service, 'nlp') and self.service.nlp:
                models_loaded = True
                model_names.append("spacy_nlp")
            
            metrics["models_loaded"] = len(model_names)
            metrics["model_names"] = model_names
            
            if not models_loaded:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "details": "No semantic search models loaded",
                    "metrics": metrics
                }
            
            return {
                "status": HealthStatus.HEALTHY,
                "details": f"Semantic search models loaded: {', '.join(model_names)}",
                "metrics": metrics
            }
            
        except Exception as e:
            logger.warning(f"Semantic search model check failed: {str(e)}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "details": f"Model availability check error: {str(e)}",
                "metrics": metrics
            }
    
    def check_query_performance(self) -> Dict[str, Any]:
        """Check semantic search query performance."""
        metrics = {}
        
        try:
            if not self.service:
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "Semantic search service not available",
                    "metrics": metrics
                }
            
            total_time = 0
            max_time = 0
            min_time = float('inf')
            successful_queries = 0
            
            for query in self.test_queries:
                try:
                    start_time = time.time()
                    
                    # Try different methods based on service type
                    if hasattr(self.service, 'analyze_intent'):
                        result = self.service.analyze_intent(query)
                    elif hasattr(self.service, 'search'):
                        result = self.service.search(query, limit=5)
                    elif hasattr(self.service, 'calculate_semantic_similarity'):
                        result = self.service.calculate_semantic_similarity(query, "test content")
                    else:
                        continue
                    
                    query_time = time.time() - start_time
                    
                    total_time += query_time
                    max_time = max(max_time, query_time)
                    min_time = min(min_time, query_time)
                    successful_queries += 1
                    
                except Exception as e:
                    logger.debug(f"Query performance test failed for '{query}': {str(e)}")
                    continue
            
            if successful_queries == 0:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "details": "No successful semantic search queries",
                    "metrics": metrics
                }
            
            avg_time = total_time / successful_queries
            
            metrics["average_query_time"] = round(avg_time * 1000, 2)  # ms
            metrics["max_query_time"] = round(max_time * 1000, 2)  # ms
            metrics["min_query_time"] = round(min_time * 1000, 2)  # ms
            metrics["successful_queries"] = successful_queries
            metrics["total_queries"] = len(self.test_queries)
            
            # Determine status based on performance
            if avg_time > 5.0:  # More than 5 seconds is unhealthy
                status = HealthStatus.UNHEALTHY
            elif avg_time > 2.0:  # More than 2 seconds is degraded
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return {
                "status": status.value,
                "details": f"Average query time: {metrics['average_query_time']}ms",
                "metrics": metrics
            }
            
        except Exception as e:
            logger.warning(f"Semantic search performance check failed: {str(e)}")
            return {
                "status": HealthStatus.DEGRADED,
                "details": f"Performance check error: {str(e)}",
                "metrics": metrics
            }
    
    def check_search_accuracy(self) -> Dict[str, Any]:
        """Check semantic search accuracy with known test cases."""
        metrics = {}
        
        try:
            if not self.service:
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "Semantic search service not available",
                    "metrics": metrics
                }
            
            # Test cases with expected behavior
            test_cases = [
                {
                    "query": "restaurant food dining",
                    "expected_entities": ["restaurant", "food"],
                    "expected_intent": "business"
                },
                {
                    "query": "buy laptop computer electronics",
                    "expected_entities": ["laptop", "computer"],
                    "expected_intent": "shopping"
                }
            ]
            
            accurate_results = 0
            total_tests = len(test_cases)
            
            for test_case in test_cases:
                try:
                    if hasattr(self.service, 'analyze_intent'):
                        result = self.service.analyze_intent(test_case["query"])
                        
                        # Check if expected entities are found
                        found_entities = result.get("entities", [])
                        entity_texts = [entity.get("text", "").lower() for entity in found_entities]
                        
                        expected_found = any(expected.lower() in " ".join(entity_texts) 
                                           for expected in test_case["expected_entities"])
                        
                        # Check intent classification
                        intent_correct = (result.get("intent_category", "").lower() == 
                                        test_case["expected_intent"].lower())
                        
                        if expected_found or intent_correct:
                            accurate_results += 1
                    else:
                        # If we can't test accuracy, assume it's working
                        accurate_results += 1
                        
                except Exception as e:
                    logger.debug(f"Accuracy test failed for '{test_case['query']}': {str(e)}")
                    continue
            
            accuracy = (accurate_results / total_tests) * 100 if total_tests > 0 else 0
            
            metrics["accuracy_percentage"] = round(accuracy, 2)
            metrics["accurate_results"] = accurate_results
            metrics["total_tests"] = total_tests
            
            # Determine status based on accuracy
            if accuracy < 50:
                status = HealthStatus.UNHEALTHY
            elif accuracy < 70:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return {
                "status": status.value,
                "details": f"Search accuracy: {accuracy:.1f}%",
                "metrics": metrics
            }
            
        except Exception as e:
            logger.warning(f"Semantic search accuracy check failed: {str(e)}")
            return {
                "status": HealthStatus.DEGRADED,
                "details": f"Accuracy check error: {str(e)}",
                "metrics": metrics
            }


class AISchemaGenerationHealthCheck(HealthCheck):
    """Health checks for AI schema generation capabilities."""
    
    def __init__(self, schema_generator_service=None):
        super().__init__("ai_schema_generation_health", schema_generator_service)
        self.test_samples = [
            {"name": "Test Product", "price": 99.99, "available": True},
            {"title": "Test Article", "author": "Test Author", "date": "2024-01-01"}
        ]
    
    def check(self) -> Dict[str, Any]:
        """Perform AI schema generation health checks."""
        results = {}
        metrics = {}
        
        # Start with overall status as healthy
        status = HealthStatus.HEALTHY
        
        # Run individual checks
        generation_test = self.check_schema_generation()
        if generation_test["status"] != HealthStatus.HEALTHY:
            status = generation_test["status"]
        results["schema_generation"] = generation_test
        
        validation_test = self.check_schema_validation()
        if validation_test["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = validation_test["status"]
        results["schema_validation"] = validation_test
        
        performance_test = self.check_generation_performance()
        if performance_test["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = performance_test["status"]
        results["generation_performance"] = performance_test
        
        # Collect metrics
        metrics.update(generation_test.get("metrics", {}))
        metrics.update(validation_test.get("metrics", {}))
        metrics.update(performance_test.get("metrics", {}))
        
        return {
            "status": status.value,
            "details": results,
            "metrics": metrics
        }
    
    def check_schema_generation(self) -> Dict[str, Any]:
        """Test schema generation functionality."""
        metrics = {}
        
        try:
            if not self.service:
                # Try to get from service registry
                from core.service_registry import ServiceRegistry
                registry = ServiceRegistry()
                self.service = registry.get_service("ai_schema_generator")
            
            if not self.service:
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "AI schema generator service not found",
                    "metrics": metrics
                }
            
            successful_generations = 0
            failed_generations = 0
            
            for sample_data in self.test_samples:
                try:
                    # Test schema generation
                    if hasattr(self.service, 'generate_schema_from_sample'):
                        result = self.service.generate_schema_from_sample([sample_data])
                        if result and hasattr(result, 'schema_model') and result.schema_model:
                            successful_generations += 1
                        else:
                            failed_generations += 1
                    elif hasattr(self.service, 'generate_schema'):
                        result = self.service.generate_schema(sample_data=[sample_data])
                        if result:
                            successful_generations += 1
                        else:
                            failed_generations += 1
                    else:
                        # Service exists but method not available
                        return {
                            "status": HealthStatus.DEGRADED,
                            "details": "Schema generation methods not available",
                            "metrics": metrics
                        }
                        
                except Exception as e:
                    logger.debug(f"Schema generation test failed: {str(e)}")
                    failed_generations += 1
            
            total_attempts = successful_generations + failed_generations
            success_rate = (successful_generations / total_attempts) * 100 if total_attempts > 0 else 0
            
            metrics["successful_generations"] = successful_generations
            metrics["failed_generations"] = failed_generations
            metrics["success_rate"] = round(success_rate, 2)
            
            if success_rate < 50:
                status = HealthStatus.UNHEALTHY
            elif success_rate < 80:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return {
                "status": status.value,
                "details": f"Schema generation success rate: {success_rate:.1f}%",
                "metrics": metrics
            }
            
        except Exception as e:
            logger.warning(f"AI schema generation check failed: {str(e)}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "details": f"Schema generation check error: {str(e)}",
                "metrics": metrics
            }
    
    def check_schema_validation(self) -> Dict[str, Any]:
        """Test schema validation functionality."""
        metrics = {}
        
        try:
            if not self.service:
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "AI schema generator service not available",
                    "metrics": metrics
                }
            
            validation_successes = 0
            validation_attempts = 0
            
            for sample_data in self.test_samples:
                try:
                    validation_attempts += 1
                    
                    # Generate schema first
                    if hasattr(self.service, 'generate_schema_from_sample'):
                        schema_result = self.service.generate_schema_from_sample([sample_data])
                        if schema_result and hasattr(schema_result, 'schema_model') and schema_result.schema_model:
                            # Try to validate the original data against the generated schema
                            try:
                                validated = schema_result.schema_model(**sample_data)
                                validation_successes += 1
                            except Exception as validation_error:
                                logger.debug(f"Schema validation failed: {str(validation_error)}")
                    
                except Exception as e:
                    logger.debug(f"Schema validation test failed: {str(e)}")
            
            validation_rate = (validation_successes / validation_attempts) * 100 if validation_attempts > 0 else 0
            
            metrics["validation_successes"] = validation_successes
            metrics["validation_attempts"] = validation_attempts
            metrics["validation_rate"] = round(validation_rate, 2)
            
            if validation_rate < 70:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return {
                "status": status.value,
                "details": f"Schema validation rate: {validation_rate:.1f}%",
                "metrics": metrics
            }
            
        except Exception as e:
            logger.warning(f"Schema validation check failed: {str(e)}")
            return {
                "status": HealthStatus.DEGRADED,
                "details": f"Validation check error: {str(e)}",
                "metrics": metrics
            }
    
    def check_generation_performance(self) -> Dict[str, Any]:
        """Test schema generation performance."""
        metrics = {}
        
        try:
            if not self.service:
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "AI schema generator service not available",
                    "metrics": metrics
                }
            
            total_time = 0
            successful_operations = 0
            
            for sample_data in self.test_samples:
                try:
                    start_time = time.time()
                    
                    if hasattr(self.service, 'generate_schema_from_sample'):
                        result = self.service.generate_schema_from_sample([sample_data])
                        if result:
                            successful_operations += 1
                    
                    generation_time = time.time() - start_time
                    total_time += generation_time
                    
                except Exception as e:
                    logger.debug(f"Performance test failed: {str(e)}")
                    continue
            
            if successful_operations == 0:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "details": "No successful schema generations for performance test",
                    "metrics": metrics
                }
            
            avg_time = total_time / successful_operations
            
            metrics["average_generation_time"] = round(avg_time * 1000, 2)  # ms
            metrics["successful_operations"] = successful_operations
            
            # Determine status based on performance
            if avg_time > 10.0:  # More than 10 seconds is unhealthy
                status = HealthStatus.UNHEALTHY
            elif avg_time > 5.0:  # More than 5 seconds is degraded
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return {
                "status": status.value,
                "details": f"Average generation time: {metrics['average_generation_time']}ms",
                "metrics": metrics
            }
            
        except Exception as e:
            logger.warning(f"Schema generation performance check failed: {str(e)}")
            return {
                "status": HealthStatus.DEGRADED,
                "details": f"Performance check error: {str(e)}",
                "metrics": metrics
            }


class CacheHealthCheck(HealthCheck):
    """Health checks for multi-tier caching system."""
    
    def __init__(self, cache_service=None):
        super().__init__("cache_health", cache_service)
        self.test_keys = ["test_cache_key_1", "test_cache_key_2", "test_cache_key_3"]
        self.test_data = {"test": "data", "timestamp": time.time()}
    
    def check(self) -> Dict[str, Any]:
        """Perform cache system health checks."""
        results = {}
        metrics = {}
        
        # Start with overall status as healthy
        status = HealthStatus.HEALTHY
        
        # Run individual checks
        connectivity_test = self.check_cache_connectivity()
        if connectivity_test["status"] != HealthStatus.HEALTHY:
            status = connectivity_test["status"]
        results["cache_connectivity"] = connectivity_test
        
        performance_test = self.check_cache_performance()
        if performance_test["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = performance_test["status"]
        results["cache_performance"] = performance_test
        
        hit_rate_test = self.check_hit_miss_ratio()
        if hit_rate_test["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = hit_rate_test["status"]
        results["hit_miss_ratio"] = hit_rate_test
        
        # Collect metrics
        metrics.update(connectivity_test.get("metrics", {}))
        metrics.update(performance_test.get("metrics", {}))
        metrics.update(hit_rate_test.get("metrics", {}))
        
        return {
            "status": status.value,
            "details": results,
            "metrics": metrics
        }
    
    def check_cache_connectivity(self) -> Dict[str, Any]:
        """Test connectivity to cache backends."""
        metrics = {}
        
        try:
            cache_services = []
            
            # Try to find cache services
            if self.service:
                cache_services.append(("primary", self.service))
            
            # Look for additional cache services
            from core.service_registry import ServiceRegistry
            registry = ServiceRegistry()
            
            possible_cache_services = [
                "ai_cache", "cache_manager", "pipeline_cache_manager",
                "memory_cache", "redis_cache", "ai_response_cache"
            ]
            
            for service_name in possible_cache_services:
                service = registry.get_service(service_name)
                if service and service not in [s[1] for s in cache_services]:
                    cache_services.append((service_name, service))
            
            if not cache_services:
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "No cache services found",
                    "metrics": metrics
                }
            
            connected_caches = 0
            failed_caches = 0
            cache_details = {}
            
            for cache_name, cache_service in cache_services:
                try:
                    # Test basic operations
                    test_key = f"health_check_{cache_name}"
                    test_value = {"health_check": True, "timestamp": time.time()}
                    
                    # Test set operation
                    if hasattr(cache_service, 'set'):
                        set_result = cache_service.set(test_key, test_value)
                        if set_result is not False:  # Could be True or None for success
                            # Test get operation
                            get_result = cache_service.get(test_key)
                            if get_result is not None:
                                connected_caches += 1
                                cache_details[cache_name] = "connected"
                                
                                # Clean up test data
                                if hasattr(cache_service, 'delete'):
                                    cache_service.delete(test_key)
                            else:
                                failed_caches += 1
                                cache_details[cache_name] = "get_failed"
                        else:
                            failed_caches += 1
                            cache_details[cache_name] = "set_failed"
                    else:
                        # Cache service doesn't have expected methods
                        failed_caches += 1
                        cache_details[cache_name] = "missing_methods"
                        
                except Exception as e:
                    failed_caches += 1
                    cache_details[cache_name] = f"error: {str(e)}"
                    logger.debug(f"Cache connectivity test failed for {cache_name}: {str(e)}")
            
            total_caches = connected_caches + failed_caches
            
            metrics["connected_caches"] = connected_caches
            metrics["failed_caches"] = failed_caches
            metrics["total_caches"] = total_caches
            metrics["cache_details"] = cache_details
            
            if connected_caches == 0:
                status = HealthStatus.UNHEALTHY
            elif failed_caches > connected_caches:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return {
                "status": status.value,
                "details": f"{connected_caches}/{total_caches} cache backends connected",
                "metrics": metrics
            }
            
        except Exception as e:
            logger.warning(f"Cache connectivity check failed: {str(e)}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "details": f"Connectivity check error: {str(e)}",
                "metrics": metrics
            }
    
    def check_cache_performance(self) -> Dict[str, Any]:
        """Test cache operation performance."""
        metrics = {}
        
        try:
            if not self.service:
                from core.service_registry import ServiceRegistry
                registry = ServiceRegistry()
                self.service = (registry.get_service("ai_cache") or 
                              registry.get_service("cache_manager") or
                              registry.get_service("pipeline_cache_manager"))
            
            if not self.service:
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "No cache service available for performance test",
                    "metrics": metrics
                }
            
            # Test set operations
            set_times = []
            get_times = []
            successful_operations = 0
            
            for i, test_key in enumerate(self.test_keys):
                try:
                    # Test set performance
                    start_time = time.time()
                    if hasattr(self.service, 'set'):
                        result = self.service.set(f"{test_key}_{i}", self.test_data)
                        if result is not False:
                            set_time = time.time() - start_time
                            set_times.append(set_time)
                            
                            # Test get performance
                            start_time = time.time()
                            retrieved = self.service.get(f"{test_key}_{i}")
                            if retrieved is not None:
                                get_time = time.time() - start_time
                                get_times.append(get_time)
                                successful_operations += 1
                            
                            # Clean up
                            if hasattr(self.service, 'delete'):
                                self.service.delete(f"{test_key}_{i}")
                    
                except Exception as e:
                    logger.debug(f"Cache performance test failed for {test_key}: {str(e)}")
                    continue
            
            if successful_operations == 0:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "details": "No successful cache operations for performance test",
                    "metrics": metrics
                }
            
            avg_set_time = sum(set_times) / len(set_times) if set_times else 0
            avg_get_time = sum(get_times) / len(get_times) if get_times else 0
            
            metrics["average_set_time"] = round(avg_set_time * 1000, 2)  # ms
            metrics["average_get_time"] = round(avg_get_time * 1000, 2)  # ms
            metrics["successful_operations"] = successful_operations
            
            # Determine status based on performance
            if avg_set_time > 1.0 or avg_get_time > 0.5:  # Very slow
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return {
                "status": status.value,
                "details": f"Avg set: {metrics['average_set_time']}ms, get: {metrics['average_get_time']}ms",
                "metrics": metrics
            }
            
        except Exception as e:
            logger.warning(f"Cache performance check failed: {str(e)}")
            return {
                "status": HealthStatus.DEGRADED,
                "details": f"Performance check error: {str(e)}",
                "metrics": metrics
            }
    
    def check_hit_miss_ratio(self) -> Dict[str, Any]:
        """Check cache hit/miss ratios."""
        metrics = {}
        
        try:
            cache_services = []
            
            # Collect cache services that provide statistics
            if self.service and hasattr(self.service, 'get_stats'):
                cache_services.append(("primary", self.service))
            
            from core.service_registry import ServiceRegistry
            registry = ServiceRegistry()
            
            possible_cache_services = [
                "ai_cache", "cache_manager", "pipeline_cache_manager",
                "ai_response_cache"
            ]
            
            for service_name in possible_cache_services:
                service = registry.get_service(service_name)
                if service and hasattr(service, 'get_stats'):
                    cache_services.append((service_name, service))
            
            if not cache_services:
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "No cache services with statistics available",
                    "metrics": metrics
                }
            
            total_hits = 0
            total_misses = 0
            cache_stats = {}
            
            for cache_name, cache_service in cache_services:
                try:
                    stats = cache_service.get_stats()
                    if isinstance(stats, dict):
                        hits = stats.get("hits", 0) or stats.get("cache_hits", 0)
                        misses = stats.get("misses", 0) or stats.get("cache_misses", 0)
                        
                        total_hits += hits
                        total_misses += misses
                        
                        hit_rate = (hits / (hits + misses)) * 100 if (hits + misses) > 0 else 0
                        
                        cache_stats[cache_name] = {
                            "hits": hits,
                            "misses": misses,
                            "hit_rate": round(hit_rate, 2),
                            "total_requests": hits + misses
                        }
                        
                except Exception as e:
                    logger.debug(f"Failed to get stats from {cache_name}: {str(e)}")
                    cache_stats[cache_name] = {"error": str(e)}
            
            overall_hit_rate = (total_hits / (total_hits + total_misses)) * 100 if (total_hits + total_misses) > 0 else 0
            
            metrics["total_hits"] = total_hits
            metrics["total_misses"] = total_misses
            metrics["overall_hit_rate"] = round(overall_hit_rate, 2)
            metrics["cache_stats"] = cache_stats
            
            # Determine status based on hit rate
            if overall_hit_rate < 30:  # Very low hit rate
                status = HealthStatus.DEGRADED
            elif overall_hit_rate < 10:  # Extremely low hit rate
                status = HealthStatus.UNHEALTHY
            else:
                status = HealthStatus.HEALTHY
            
            return {
                "status": status.value,
                "details": f"Overall cache hit rate: {overall_hit_rate:.1f}%",
                "metrics": metrics
            }
            
        except Exception as e:
            logger.warning(f"Cache hit/miss ratio check failed: {str(e)}")
            return {
                "status": HealthStatus.DEGRADED,
                "details": f"Hit/miss ratio check error: {str(e)}",
                "metrics": metrics
            }


class ResilienceHealthCheck(HealthCheck):
    """Health checks for resilience measures including CAPTCHA detection."""
    
    def __init__(self, resilience_service=None):
        super().__init__("resilience_health", resilience_service)
    
    def check(self) -> Dict[str, Any]:
        """Perform resilience measures health checks."""
        results = {}
        metrics = {}
        
        # Start with overall status as healthy
        status = HealthStatus.HEALTHY
        
        # Run individual checks
        captcha_detection = self.check_captcha_detection()
        if captcha_detection["status"] != HealthStatus.HEALTHY:
            status = captcha_detection["status"]
        results["captcha_detection"] = captcha_detection
        
        blocking_detection = self.check_blocking_detection()
        if blocking_detection["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = blocking_detection["status"]
        results["blocking_detection"] = blocking_detection
        
        proxy_rotation = self.check_proxy_rotation()
        if proxy_rotation["status"] != HealthStatus.HEALTHY and status != HealthStatus.UNHEALTHY:
            status = proxy_rotation["status"]
        results["proxy_rotation"] = proxy_rotation
        
        # Collect metrics
        metrics.update(captcha_detection.get("metrics", {}))
        metrics.update(blocking_detection.get("metrics", {}))
        metrics.update(proxy_rotation.get("metrics", {}))
        
        return {
            "status": status.value,
            "details": results,
            "metrics": metrics
        }
    
    def check_captcha_detection(self) -> Dict[str, Any]:
        """Check CAPTCHA detection capabilities."""
        metrics = {}
        
        try:
            # Try to find CAPTCHA detection service
            captcha_detector = None
            
            from core.service_registry import ServiceRegistry
            registry = ServiceRegistry()
            
            # Look for CAPTCHA detector in various places
            possible_services = [
                "captcha_detector", "http_utils", "session_manager", "scraper"
            ]
            
            for service_name in possible_services:
                service = registry.get_service(service_name)
                if service:
                    if hasattr(service, 'detect_captcha') or hasattr(service, 'is_captcha_page'):
                        captcha_detector = service
                        break
                    elif hasattr(service, 'captcha_detector'):
                        captcha_detector = service.captcha_detector
                        break
            
            if not captcha_detector:
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "CAPTCHA detection service not found",
                    "metrics": metrics
                }
            
            # Test CAPTCHA detection with known patterns
            test_content = [
                "<div class='g-recaptcha'></div>",  # reCAPTCHA
                "Please complete the CAPTCHA",
                "Verify you are human",
                "<div id='captcha-container'></div>"
            ]
            
            detections = 0
            false_positives = 0
            
            for content in test_content:
                try:
                    if hasattr(captcha_detector, 'detect_captcha'):
                        is_captcha = captcha_detector.detect_captcha(content)
                    elif hasattr(captcha_detector, 'is_captcha_page'):
                        is_captcha = captcha_detector.is_captcha_page(content)
                    else:
                        # Can't test, assume working
                        detections += 1
                        continue
                    
                    if is_captcha:
                        detections += 1
                        
                except Exception as e:
                    logger.debug(f"CAPTCHA detection test failed: {str(e)}")
                    continue
            
            # Test with normal content (should not detect CAPTCHA)
            normal_content = "<html><body><h1>Normal page</h1><p>Regular content</p></body></html>"
            try:
                if hasattr(captcha_detector, 'detect_captcha'):
                    is_captcha = captcha_detector.detect_captcha(normal_content)
                elif hasattr(captcha_detector, 'is_captcha_page'):
                    is_captcha = captcha_detector.is_captcha_page(normal_content)
                else:
                    is_captcha = False
                
                if is_captcha:
                    false_positives += 1
            except Exception:
                pass
            
            detection_rate = (detections / len(test_content)) * 100 if test_content else 0
            
            metrics["captcha_detections"] = detections
            metrics["false_positives"] = false_positives
            metrics["detection_rate"] = round(detection_rate, 2)
            metrics["test_cases"] = len(test_content)
            
            # Determine status
            if detection_rate < 50:
                status = HealthStatus.DEGRADED
            elif false_positives > 0:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return {
                "status": status.value,
                "details": f"CAPTCHA detection rate: {detection_rate:.1f}%",
                "metrics": metrics
            }
            
        except Exception as e:
            logger.warning(f"CAPTCHA detection check failed: {str(e)}")
            return {
                "status": HealthStatus.DEGRADED,
                "details": f"CAPTCHA detection check error: {str(e)}",
                "metrics": metrics
            }
    
    def check_blocking_detection(self) -> Dict[str, Any]:
        """Check blocking detection capabilities."""
        metrics = {}
        
        try:
            # Look for blocking detection capabilities
            from core.service_registry import ServiceRegistry
            registry = ServiceRegistry()
            
            # Check for services that track blocking
            blocking_services = []
            
            possible_services = ["session_manager", "proxy_manager", "http_utils", "scraper"]
            
            for service_name in possible_services:
                service = registry.get_service(service_name)
                if service and (hasattr(service, 'is_blocked') or 
                              hasattr(service, 'check_blocking') or
                              hasattr(service, '_blocked_until')):
                    blocking_services.append((service_name, service))
            
            if not blocking_services:
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "No blocking detection services found",
                    "metrics": metrics
                }
            
            blocked_count = 0
            total_checks = 0
            blocking_details = {}
            
            for service_name, service in blocking_services:
                try:
                    total_checks += 1
                    
                    # Check for blocked domains/IPs
                    if hasattr(service, '_blocked_until'):
                        current_time = time.time()
                        currently_blocked = sum(1 for until_time in service._blocked_until.values() 
                                              if until_time > current_time)
                        if currently_blocked > 0:
                            blocked_count += 1
                        blocking_details[service_name] = currently_blocked
                    elif hasattr(service, 'get_blocked_domains'):
                        blocked_domains = service.get_blocked_domains()
                        if blocked_domains:
                            blocked_count += 1
                        blocking_details[service_name] = len(blocked_domains) if blocked_domains else 0
                    else:
                        # Service exists but can't check specifics
                        blocking_details[service_name] = "available"
                        
                except Exception as e:
                    logger.debug(f"Blocking check failed for {service_name}: {str(e)}")
                    blocking_details[service_name] = f"error: {str(e)}"
            
            metrics["blocking_services"] = len(blocking_services)
            metrics["services_with_blocks"] = blocked_count
            metrics["blocking_details"] = blocking_details
            
            # Status determination
            if blocked_count > total_checks * 0.5:  # More than half have blocks
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return {
                "status": status.value,
                "details": f"{blocked_count}/{len(blocking_services)} services report blocking",
                "metrics": metrics
            }
            
        except Exception as e:
            logger.warning(f"Blocking detection check failed: {str(e)}")
            return {
                "status": HealthStatus.DEGRADED,
                "details": f"Blocking detection check error: {str(e)}",
                "metrics": metrics
            }
    
    def check_proxy_rotation(self) -> Dict[str, Any]:
        """Check proxy rotation effectiveness."""
        metrics = {}
        
        try:
            from core.service_registry import ServiceRegistry
            registry = ServiceRegistry()
            
            proxy_manager = registry.get_service("proxy_manager")
            
            if not proxy_manager:
                return {
                    "status": HealthStatus.UNKNOWN,
                    "details": "Proxy manager service not found",
                    "metrics": metrics
                }
            
            # Check proxy pool status
            if hasattr(proxy_manager, 'get_proxy_stats'):
                stats = proxy_manager.get_proxy_stats()
                
                total_proxies = stats.get("total_proxies", 0)
                active_proxies = stats.get("active_proxies", 0)
                failed_proxies = stats.get("failed_proxies", 0)
                
                metrics["total_proxies"] = total_proxies
                metrics["active_proxies"] = active_proxies
                metrics["failed_proxies"] = failed_proxies
                
                if total_proxies > 0:
                    availability_rate = (active_proxies / total_proxies) * 100
                    metrics["proxy_availability_rate"] = round(availability_rate, 2)
                    
                    if availability_rate < 30:
                        status = HealthStatus.UNHEALTHY
                    elif availability_rate < 60:
                        status = HealthStatus.DEGRADED
                    else:
                        status = HealthStatus.HEALTHY
                else:
                    status = HealthStatus.UNKNOWN
                    
            elif hasattr(proxy_manager, '_proxies') or hasattr(proxy_manager, 'proxies'):
                # Try to get proxy list directly
                proxy_list = getattr(proxy_manager, '_proxies', None) or getattr(proxy_manager, 'proxies', None)
                
                if proxy_list:
                    metrics["total_proxies"] = len(proxy_list)
                    status = HealthStatus.HEALTHY if len(proxy_list) > 0 else HealthStatus.DEGRADED
                else:
                    metrics["total_proxies"] = 0
                    status = HealthStatus.UNKNOWN
            else:
                status = HealthStatus.UNKNOWN
                metrics["total_proxies"] = 0
            
            return {
                "status": status.value,
                "details": f"Proxy rotation system operational with {metrics.get('total_proxies', 0)} proxies",
                "metrics": metrics
            }
            
        except Exception as e:
            logger.warning(f"Proxy rotation check failed: {str(e)}")
            return {
                "status": HealthStatus.DEGRADED,
                "details": f"Proxy rotation check error: {str(e)}",
                "metrics": metrics
            }

class Monitoring(BaseService):
    """
    Service for monitoring the health and performance of SmartScrape components.
    
    This service manages health checks for resource management services,
    collects metrics, and provides overall system health status.
    """
    
    def __init__(self):
        self._initialized = False
        self._config = None
        self._health_checks = {}
        self._metrics_history = {}
        self._alert_thresholds = {}
        self._exporters = []
        self._monitoring_thread = None
        self._shutdown_event = threading.Event()
        self._services = None
        self._registry = ServiceRegistry()
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the monitoring service with configuration."""
        if self._initialized:
            return
            
        logger.info("Initializing monitoring service")
        
        self._config = config or {}
        
        # Initialize configuration
        self._monitoring_interval = self._config.get('monitoring_interval', 60)  # seconds
        self._metrics_retention = self._config.get('metrics_retention', 24 * 60 * 60)  # seconds (1 day default)
        self._alert_thresholds = self._config.get('alert_thresholds', {})
        
        # Configure exporters
        exporter_configs = self._config.get('exporters', [{'type': 'log'}])
        for exporter_config in exporter_configs:
            exporter_type = exporter_config.get('type', 'log')
            if exporter_type == 'log':
                self._exporters.append(self._export_to_log)
            elif exporter_type == 'file':
                path = exporter_config.get('path', 'metrics.json')
                self._exporters.append(lambda data, format: self._export_to_file(data, format, path))
            # Additional exporters can be added here
        
        # Register health checks for resource management services
        self._register_resource_health_checks()
        
        # Setup enhanced alerting for new features
        self._setup_enhanced_alerts()
        
        self._initialized = True
        logger.info("Monitoring service initialized")
        
        # Start monitoring thread if configured to start automatically
        if self._config.get('auto_start', True):
            self.start_monitoring()
    
    def shutdown(self) -> None:
        """Stop monitoring and clean up resources."""
        if not self._initialized:
            return
            
        logger.info("Shutting down monitoring service")
        
        self.stop_monitoring()
        
        # Clear resources
        self._health_checks.clear()
        self._metrics_history.clear()
        
        self._initialized = False
        logger.info("Monitoring service shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "monitoring"
    
    def start_monitoring(self) -> None:
        """Start the background monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Monitoring thread is already running")
            return
            
        logger.info("Starting monitoring thread")
        self._shutdown_event.clear()
        
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="MonitoringThread"
        )
        self._monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop the background monitoring thread."""
        if not self._monitoring_thread:
            return
            
        logger.info("Stopping monitoring thread")
        self._shutdown_event.set()
        
        # Wait for thread to terminate (with timeout)
        if self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
            
        self._monitoring_thread = None
    
    def _monitoring_worker(self) -> None:
        """Worker function for the monitoring thread."""
        logger.info(f"Monitoring thread started (interval: {self._monitoring_interval}s)")
        
        while not self._shutdown_event.is_set():
            try:
                # Run health checks and collect metrics
                health_data = self.run_health_check()
                self._store_metrics(health_data)
                
                # Export metrics if configured
                if self._exporters:
                    self.export_metrics()
                
                # Wait for next check interval or until shutdown
                self._shutdown_event.wait(self._monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring thread: {str(e)}")
                # Wait a bit before retrying to avoid busy loop on error
                self._shutdown_event.wait(10)
        
        logger.info("Monitoring thread stopped")
    
    def run_health_check(self) -> Dict[str, Any]:
        """
        Run health checks for all registered services.
        
        Returns:
            Dict containing health check results and metrics
        """
        results = {}
        system_metrics = {}
        
        logger.debug("Running health checks")
        
        # Run each registered health check
        for check_name, check in self._health_checks.items():
            try:
                result = check.check()
                results[check_name] = result
                
                # Aggregate metrics
                if "metrics" in result:
                    system_metrics[check_name] = result["metrics"]
            except Exception as e:
                logger.error(f"Error running health check {check_name}: {str(e)}")
                results[check_name] = {
                    "status": HealthStatus.UNHEALTHY,
                    "details": f"Exception during health check: {str(e)}",
                    "metrics": {}
                }
        
        # Check overall system health via service registry
        try:
            registry_health = self._registry.get_system_health()
            results["service_registry"] = {
                "status": registry_health["status"],
                "details": registry_health["details"],
                "metrics": {
                    "initialized_services": registry_health["initialized_services"],
                    "registered_services": registry_health["registered_services"]
                }
            }
            
            # Add service-specific health from registry
            for service_name, health in registry_health.get("services", {}).items():
                if service_name not in results:
                    results[service_name] = health
        except Exception as e:
            logger.error(f"Error getting system health from registry: {str(e)}")
        
        # Determine overall health
        overall_status = self._calculate_overall_status(results)
        
        # Add timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        health_data = {
            "timestamp": timestamp,
            "status": overall_status,
            "components": results,
            "metrics": system_metrics
        }
        
        # Check enhanced alerts for new features
        self._check_enhanced_alerts(health_data)
        
        return health_data
    
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect performance metrics from all services.
        
        Returns:
            Dict containing collected metrics
        """
        metrics = {}
        
        # Collect metrics from services
        for service_name, service in self._get_services().items():
            try:
                service_health = service.get_service_health()
                if "metrics" in service_health:
                    metrics[service_name] = service_health["metrics"]
            except Exception as e:
                logger.warning(f"Error collecting metrics from {service_name}: {str(e)}")
        
        # Add timestamp
        metrics["timestamp"] = datetime.datetime.now().isoformat()
        
        return metrics
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get the overall system health status.
        
        Returns:
            Dict containing overall system health and component statuses
        """
        # Run a fresh health check
        return self.run_health_check()
    
    def export_metrics(self, format: str = 'json') -> None:
        """
        Export metrics using configured exporters.
        
        Args:
            format: Output format (currently only 'json' is supported)
        """
        # Get latest metrics
        metrics = self.collect_metrics()
        
        # Run each exporter
        for exporter in self._exporters:
            try:
                exporter(metrics, format)
            except Exception as e:
                logger.error(f"Error in metrics exporter: {str(e)}")
    
    def _export_to_log(self, data: Dict[str, Any], format: str) -> None:
        """Export metrics to log."""
        if format == 'json':
            logger.info(f"Metrics: {json.dumps(data)}")
        else:
            logger.info(f"Metrics: {data}")
    
    def _export_to_file(self, data: Dict[str, Any], format: str, path: str) -> None:
        """Export metrics to a file."""
        try:
            with open(path, 'w') as f:
                if format == 'json':
                    json.dump(data, f, indent=2)
                else:
                    f.write(str(data))
        except Exception as e:
            logger.error(f"Failed to write metrics to file {path}: {str(e)}")
    
    def _get_services(self) -> Dict[str, BaseService]:
        """Get all initialized services from the registry."""
        if self._services is None:
            self._services = {}
            
            # Get registry's internal service dict if available
            if hasattr(self._registry, "_services"):
                self._services = self._registry._services
        
        return self._services
    
    def _calculate_overall_status(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Calculate overall system status from component statuses."""
        if not results:
            return HealthStatus.UNKNOWN
            
        statuses = [result.get("status", HealthStatus.UNKNOWN) for result in results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses and len(statuses) == statuses.count(HealthStatus.UNKNOWN):
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY
    
    def _store_metrics(self, health_data: Dict[str, Any]) -> None:
        """Store metrics in history with timestamp-based expiration."""
        timestamp = time.time()
        self._metrics_history[timestamp] = health_data
        
        # Clean up old metrics
        expiration = timestamp - self._metrics_retention
        expired_keys = [k for k in self._metrics_history if k < expiration]
        for key in expired_keys:
            del self._metrics_history[key]
    
    def _register_resource_health_checks(self) -> None:
        """Register health checks for resource management services."""

        try:
            # Create health checks for each resource service
            registry = ServiceRegistry()
            
            # Session Manager health check
            try:
                session_manager = registry.get_service("session_manager")
                if session_manager:
                    self._health_checks["session_manager"] = SessionManagerHealthCheck(session_manager)
                    logger.debug("Registered SessionManagerHealthCheck")
            except Exception as e:
                logger.warning(f"Could not register SessionManagerHealthCheck: {str(e)}")
            
            # Proxy Manager health check
            try:
                proxy_manager = registry.get_service("proxy_manager")
                if proxy_manager:
                    self._health_checks["proxy_manager"] = ProxyManagerHealthCheck(proxy_manager)
                    logger.debug("Registered ProxyManagerHealthCheck")
            except Exception as e:
                logger.warning(f"Could not register ProxyManagerHealthCheck: {str(e)}")
            
            # Rate Limiter health check
            try:
                rate_limiter = registry.get_service("rate_limiter")
                if rate_limiter:
                    self._health_checks["rate_limiter"] = RateLimiterHealthCheck(rate_limiter)
                    logger.debug("Registered RateLimiterHealthCheck")
            except Exception as e:
                logger.warning(f"Could not register RateLimiterHealthCheck: {str(e)}")
            
            # Circuit Breaker health check
            try:
                circuit_breaker = registry.get_service("circuit_breaker_manager")
                if circuit_breaker:
                    self._health_checks["circuit_breaker"] = CircuitBreakerHealthCheck(circuit_breaker)
                    logger.debug("Registered CircuitBreakerHealthCheck")
            except Exception as e:
                logger.warning(f"Could not register CircuitBreakerHealthCheck: {str(e)}")
            
            # Semantic Search health check
            try:
                semantic_search = registry.get_service("semantic_search")
                if semantic_search:
                    self._health_checks["semantic_search"] = SemanticSearchHealthCheck(semantic_search)
                    logger.debug("Registered SemanticSearchHealthCheck")
            except Exception as e:
                logger.warning(f"Could not register SemanticSearchHealthCheck: {str(e)}")
            
            # AI Schema Generation health check
            try:
                ai_schema_generator = registry.get_service("ai_schema_generator")
                if ai_schema_generator:
                    self._health_checks["ai_schema_generation"] = AISchemaGenerationHealthCheck(ai_schema_generator)
                    logger.debug("Registered AISchemaGenerationHealthCheck")
            except Exception as e:
                logger.warning(f"Could not register AISchemaGenerationHealthCheck: {str(e)}")
            
            # Cache health check
            try:
                cache_service = registry.get_service("cache_manager")
                if cache_service:
                    self._health_checks["cache"] = CacheHealthCheck(cache_service)
                    logger.debug("Registered CacheHealthCheck")
            except Exception as e:
                logger.warning(f"Could not register CacheHealthCheck: {str(e)}")
            
            # Resilience health check
            try:
                # The resilience health check can work without a specific service
                # as it checks various resilience components
                self._health_checks["resilience"] = ResilienceHealthCheck()
                logger.debug("Registered ResilienceHealthCheck")
            except Exception as e:
                logger.warning(f"Could not register ResilienceHealthCheck: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error registering resource health checks: {str(e)}")
    
    def register_health_check(self, name: str, health_check: HealthCheck) -> None:
        """
        Register a custom health check.
        
        Args:
            name: Name for the health check
            health_check: HealthCheck instance
        """
        if not isinstance(health_check, HealthCheck):
            raise TypeError("health_check must be an instance of HealthCheck")
            
        self._health_checks[name] = health_check
        logger.debug(f"Registered custom health check: {name}")
    
    def get_health_history(self, component: Optional[str] = None, 
                          period: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical health data.
        
        Args:
            component: Optional component name to filter results
            period: Optional time period in seconds to limit results
            
        Returns:
            List of health data entries, newest first
        """
        # Filter by time period if specified
        if period:
            cutoff = time.time() - period
            history = {k: v for k, v in self._metrics_history.items() if k >= cutoff}
        else:
            history = self._metrics_history
        
        # Sort by timestamp (newest first)
        sorted_history = [v for k, v in sorted(history.items(), reverse=True)]
        
        # Filter by component if specified
        if component and sorted_history:
            return [{
                "timestamp": entry["timestamp"],
                "component": entry["components"].get(component, {"status": "unknown", "details": "Not found"})
            } for entry in sorted_history if "components" in entry]
        
        return sorted_history
    
    def reset_service(self, service_name: str) -> Dict[str, Any]:
        """
        Reset a service to recover from unhealthy state.
        
        Args:
            service_name: Name of the service to reset
            
        Returns:
            Dict with reset result information
        """
        result = {
            "service": service_name,
            "success": False,
            "details": ""
        }
        
        try:
            registry = ServiceRegistry()
            service = registry.get_service(service_name)
            
            if not service:
                result["details"] = f"Service {service_name} not found"
                return result
            
            # Special handling for different services
            if service_name == "circuit_breaker_manager":
                # Reset all circuits
                for circuit in service._circuit_breakers.values():
                    circuit.reset()
                result["details"] = f"Reset all circuit breakers"
                result["success"] = True
                
            elif service_name == "proxy_manager":
                # Refresh proxies
                if hasattr(service, "_check_all_proxies"):
                    service._check_all_proxies()
                    result["details"] = "Triggered proxy health check"
                    result["success"] = True
                
            elif service_name == "rate_limiter":
                # Clear rate limiting state
                if hasattr(service, "_last_request_time"):
                    service._last_request_time.clear()
                    result["details"] = "Cleared rate limiting state"
                    result["success"] = True
                
            else:
                # Generic restart approach
                try:
                    service.shutdown()
                    service.initialize(service._config)
                    result["details"] = f"Restarted service {service_name}"
                    result["success"] = True
                except Exception as e:
                    result["details"] = f"Failed to restart {service_name}: {str(e)}"
                
        except Exception as e:
            result["details"] = f"Error resetting service {service_name}: {str(e)}"
        
        return result
    
    def get_service_health(self) -> Dict[str, Any]:
        """
        Get the health status of the monitoring service.
        
        Overrides BaseService.get_service_health()
        """
        metrics = {
            "registered_checks": len(self._health_checks),
            "metric_history_entries": len(self._metrics_history),
            "monitoring_active": self._monitoring_thread is not None and self._monitoring_thread.is_alive(),
            "check_interval_seconds": self._monitoring_interval,
            "enhanced_alert_patterns": len(getattr(self, '_enhanced_alert_patterns', {})),
            "last_alerts_triggered": len(getattr(self, '_last_alert_times', {}))
        }
        
        return {
            "status": HealthStatus.HEALTHY if self.is_initialized else HealthStatus.UNHEALTHY,
            "details": "Monitoring service is running" if self.is_initialized else "Monitoring service is not initialized",
            "metrics": metrics
        }
    
    def _setup_enhanced_alerts(self) -> None:
        """Setup enhanced alerting rules for new features."""
        try:
            from core.alerting import Alerting, AlertSeverity
            alerting = None
            
            # Try to get the alerting service from registry
            try:
                alerting = self._registry.get_service("alerting")
            except Exception:
                # Create a basic alerting service if none exists
                alerting = Alerting()
                try:
                    alerting.initialize()
                except Exception as e:
                    logger.warning(f"Could not initialize alerting service: {str(e)}")
                    return
            
            if not alerting:
                logger.warning("No alerting service available for enhanced alerts")
                return
            
            # Enhanced alert patterns for new features
            enhanced_alert_patterns = {
                # Semantic Search Alerts
                "semantic_search_model_failure": {
                    "condition": lambda health: (
                        health.get("semantic_search", {}).get("details", {}).get("model_availability", {}).get("status") == "unhealthy"
                    ),
                    "message": "Semantic search models failed to load or are unavailable",
                    "severity": AlertSeverity.CRITICAL,
                    "cooldown": 300  # 5 minutes
                },
                "semantic_search_performance_degraded": {
                    "condition": lambda health: (
                        health.get("semantic_search", {}).get("details", {}).get("query_performance", {}).get("status") == "degraded"
                    ),
                    "message": "Semantic search query performance is degraded",
                    "severity": AlertSeverity.WARNING,
                    "cooldown": 600  # 10 minutes
                },
                "semantic_search_accuracy_low": {
                    "condition": lambda health: (
                        health.get("semantic_search", {}).get("details", {}).get("search_accuracy", {}).get("status") == "unhealthy"
                    ),
                    "message": "Semantic search accuracy has dropped below acceptable levels",
                    "severity": AlertSeverity.ERROR,
                    "cooldown": 300  # 5 minutes
                },
                
                # AI Schema Generation Alerts
                "ai_schema_generation_failure": {
                    "condition": lambda health: (
                        health.get("ai_schema_generation", {}).get("details", {}).get("schema_generation", {}).get("status") == "unhealthy"
                    ),
                    "message": "AI schema generation is failing consistently",
                    "severity": AlertSeverity.CRITICAL,
                    "cooldown": 300  # 5 minutes
                },
                "ai_schema_validation_failure": {
                    "condition": lambda health: (
                        health.get("ai_schema_generation", {}).get("details", {}).get("schema_validation", {}).get("status") == "unhealthy"
                    ),
                    "message": "AI-generated schemas are failing validation",
                    "severity": AlertSeverity.ERROR,
                    "cooldown": 300  # 5 minutes
                },
                "ai_schema_performance_slow": {
                    "condition": lambda health: (
                        health.get("ai_schema_generation", {}).get("details", {}).get("generation_performance", {}).get("status") == "degraded"
                    ),
                    "message": "AI schema generation performance is slower than expected",
                    "severity": AlertSeverity.WARNING,
                    "cooldown": 600  # 10 minutes
                },
                
                # Cache System Alerts
                "cache_connectivity_failure": {
                    "condition": lambda health: (
                        health.get("cache", {}).get("details", {}).get("cache_connectivity", {}).get("status") == "unhealthy"
                    ),
                    "message": "Critical cache connectivity failure - Redis or persistent cache unavailable",
                    "severity": AlertSeverity.CRITICAL,
                    "cooldown": 180  # 3 minutes
                },
                "cache_hit_rate_low": {
                    "condition": lambda health: (
                        health.get("cache", {}).get("details", {}).get("hit_miss_ratio", {}).get("metrics", {}).get("aggregated_hit_rate", 0) < 30
                    ),
                    "message": "Cache hit rate is critically low - performance may be impacted",
                    "severity": AlertSeverity.WARNING,
                    "cooldown": 900  # 15 minutes
                },
                "cache_performance_degraded": {
                    "condition": lambda health: (
                        health.get("cache", {}).get("details", {}).get("cache_performance", {}).get("status") == "degraded"
                    ),
                    "message": "Cache operation performance is degraded",
                    "severity": AlertSeverity.WARNING,
                    "cooldown": 600  # 10 minutes
                },
                
                # Resilience Measures Alerts
                "captcha_detection_failure": {
                    "condition": lambda health: (
                        health.get("resilience", {}).get("details", {}).get("captcha_detection", {}).get("status") == "unhealthy"
                    ),
                    "message": "CAPTCHA detection system is failing - scraping may be blocked",
                    "severity": AlertSeverity.ERROR,
                    "cooldown": 300  # 5 minutes
                },
                "blocking_incidents_high": {
                    "condition": lambda health: (
                        health.get("resilience", {}).get("details", {}).get("blocking_detection", {}).get("metrics", {}).get("blocked_domains_count", 0) > 5
                    ),
                    "message": "High number of blocking incidents detected across multiple domains",
                    "severity": AlertSeverity.WARNING,
                    "cooldown": 600  # 10 minutes
                },
                "proxy_availability_critical": {
                    "condition": lambda health: (
                        health.get("resilience", {}).get("details", {}).get("proxy_rotation", {}).get("status") == "unhealthy"
                    ),
                    "message": "Proxy availability is critically low - scraping operations may fail",
                    "severity": AlertSeverity.CRITICAL,
                    "cooldown": 300  # 5 minutes
                },
                
                # Combined System Health Alerts
                "multiple_systems_degraded": {
                    "condition": lambda health: sum(
                        1 for component in health.get("components", {}).values() 
                        if component.get("status") in ["degraded", "unhealthy"]
                    ) >= 3,
                    "message": "Multiple critical systems are experiencing issues simultaneously",
                    "severity": AlertSeverity.CRITICAL,
                    "cooldown": 300  # 5 minutes
                },
                "ai_capabilities_offline": {
                    "condition": lambda health: (
                        health.get("semantic_search", {}).get("status") == "unhealthy" and
                        health.get("ai_schema_generation", {}).get("status") == "unhealthy"
                    ),
                    "message": "Both semantic search and AI schema generation are offline - AI capabilities compromised",
                    "severity": AlertSeverity.CRITICAL,
                    "cooldown": 180  # 3 minutes
                }
            }
            
            # Store alert patterns for use in monitoring
            self._enhanced_alert_patterns = enhanced_alert_patterns
            logger.info(f"Setup {len(enhanced_alert_patterns)} enhanced alert patterns for new features")
            
        except Exception as e:
            logger.error(f"Error setting up enhanced alerts: {str(e)}")

    def _check_enhanced_alerts(self, health_data: Dict[str, Any]) -> None:
        """Check enhanced alert conditions and trigger alerts if needed."""
        if not hasattr(self, '_enhanced_alert_patterns'):
            return
            
        try:
            from core.alerting import Alerting
            alerting = self._registry.get_service("alerting")
            if not alerting:
                return
            
            current_time = time.time()
            
            for pattern_name, pattern in self._enhanced_alert_patterns.items():
                try:
                    # Check if alert is in cooldown
                    last_triggered = getattr(self, '_last_alert_times', {}).get(pattern_name, 0)
                    cooldown = pattern.get('cooldown', 300)
                    
                    if current_time - last_triggered < cooldown:
                        continue
                    
                    # Check condition
                    if pattern['condition'](health_data.get('components', {})):
                        # Trigger alert
                        context = {
                            "pattern": pattern_name,
                            "health_data": health_data.get('components', {}),
                            "system_status": health_data.get('status'),
                            "timestamp": health_data.get('timestamp')
                        }
                        
                        alerting.trigger_alert(
                            message=pattern['message'],
                            severity=pattern['severity'],
                            context=context
                        )
                        
                        # Update last triggered time
                        if not hasattr(self, '_last_alert_times'):
                            self._last_alert_times = {}
                        self._last_alert_times[pattern_name] = current_time
                        
                        logger.info(f"Triggered enhanced alert: {pattern_name}")
                        
                except Exception as e:
                    logger.warning(f"Error checking alert pattern {pattern_name}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error checking enhanced alerts: {str(e)}")

def log_execution_metrics(job_id: str, url: str, metrics: Dict[str, Any]) -> None:
    """
    Log execution metrics for monitoring purposes.
    
    Args:
        job_id: Unique identifier for the job
        url: URL being processed
        metrics: Dictionary containing execution metrics
    """
    logger.info(f"EXECUTION_METRICS [Job: {job_id}] [URL: {url}] Metrics: {json.dumps(metrics, indent=2)}")
    
    # You could extend this to store metrics in a database, send to monitoring services, etc.
    # For now, we just log them for debugging purposes
    if 'duration' in metrics:
        logger.info(f"EXECUTION_TIME [Job: {job_id}] [URL: {url}] Duration: {metrics['duration']}s")
    
    if 'status' in metrics:
        logger.info(f"EXECUTION_STATUS [Job: {job_id}] [URL: {url}] Status: {metrics['status']}")
        
    if 'error' in metrics:
        logger.warning(f"EXECUTION_ERROR [Job: {job_id}] [URL: {url}] Error: {metrics['error']}")


def get_monitoring_instance():
    """Get the global monitoring instance."""
    # This function can be used to get a monitoring instance from a registry
    # For now, just return None to indicate monitoring is available
    return None