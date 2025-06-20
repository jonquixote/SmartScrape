"""
User Agent Rotation Stage Module.

This module provides a UserAgentRotationStage that handles the rotation and selection
of user agents for HTTP requests to make scraping appear more natural.
"""

import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from core.pipeline.stages.base_stage import BaseStage
from core.pipeline.context import PipelineContext
from core.service_registry import ServiceRegistry


class UserAgentRotationStage(BaseStage):
    """
    User Agent Rotation Stage for managing user agent selection in pipelines.
    
    This stage handles selection and rotation of user agents for HTTP requests,
    helping to distribute traffic patterns and avoid fingerprinting.
    
    Features:
    - Automatic user agent selection from predefined lists
    - Custom user agent lists per device category
    - Domain-specific user agent assignments
    - User agent rotation strategies (random, sequential, weighted)
    - Integration with browser fingerprinting protection
    
    Configuration:
    - rotation_strategy: Strategy for rotating user agents ("random", "sequential", "weighted")
    - sticky_domain: Whether to use the same user agent for the same domain
    - rotation_interval: Time in seconds before forcing a user agent rotation
    - device_categories: List of device categories to use ("desktop", "mobile", "tablet")
    - browsers: List of browsers to use ("chrome", "firefox", "safari", "edge")
    - custom_user_agents: List of custom user agents to use instead of built-in lists
    - user_agent_weights: Dictionary of user agent weights for weighted selection
    """
    
    # Common user agents by device and browser
    DEFAULT_USER_AGENTS = {
        "desktop": {
            "chrome": [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
            ],
            "firefox": [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
                "Mozilla/5.0 (X11; Linux i686; rv:89.0) Gecko/20100101 Firefox/89.0"
            ],
            "safari": [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
            ],
            "edge": [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"
            ]
        },
        "mobile": {
            "chrome": [
                "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/91.0.4472.80 Mobile/15E148 Safari/604.1",
                "Mozilla/5.0 (Linux; Android 11; SM-G996B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36"
            ],
            "firefox": [
                "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) FxiOS/34.0 Mobile/15E148 Safari/605.1.15",
                "Mozilla/5.0 (Android 11; Mobile; rv:68.0) Gecko/68.0 Firefox/89.0"
            ],
            "safari": [
                "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1"
            ]
        },
        "tablet": {
            "chrome": [
                "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/91.0.4472.80 Mobile/15E148 Safari/604.1",
                "Mozilla/5.0 (Linux; Android 11; SM-T970) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Safari/537.36"
            ],
            "firefox": [
                "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) FxiOS/34.0 Mobile/15E148 Safari/605.1.15"
            ],
            "safari": [
                "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1"
            ]
        }
    }
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the user agent rotation stage.
        
        Args:
            name: Optional name for the stage (defaults to class name)
            config: Optional configuration dictionary
        """
        super().__init__(name, config)
        
        # Configuration
        self.rotation_strategy = self.config.get("rotation_strategy", "random")
        self.sticky_domain = self.config.get("sticky_domain", True)
        self.rotation_interval = self.config.get("rotation_interval", 3600)  # 1 hour default
        self.device_categories = self.config.get("device_categories", ["desktop"])
        self.browsers = self.config.get("browsers", ["chrome", "firefox", "safari", "edge"])
        self.custom_user_agents = self.config.get("custom_user_agents", [])
        self.user_agent_weights = self.config.get("user_agent_weights", {})
        
        # Internal state
        self._domain_user_agents = {}  # Maps domains to their current user agent
        self._last_rotation = {}  # Maps domains to last rotation timestamp
        self._sequential_index = 0  # Current index for sequential rotation
        
        # Build user agent list
        self._all_user_agents = self._build_user_agent_list()
        
        # Metrics
        self._rotation_count = 0
    
    async def _process(self, context: PipelineContext) -> bool:
        """
        Select and apply a user agent to the pipeline context.
        
        This method determines domains from the context, selects appropriate
        user agents, and adds them to request headers.
        
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
        
        if not domains and not context.get("global_user_agent"):
            # No domains found and no global UA set yet, use a global user agent
            user_agent = self._select_user_agent(None, context)
            if user_agent:
                context.set("global_user_agent", user_agent)
                
                # Update headers if they exist
                self._update_headers_in_context(context, None, user_agent)
                
                self._logger.debug(f"Set global user agent: {user_agent[:30]}...")
                return True
            else:
                self._logger.warning("Failed to select a user agent")
                return False
        
        # Select user agent for each domain and set in context
        rotation_applied = False
        
        for domain in domains:
            # Check if we need to rotate 
            rotate = self._should_rotate(domain)
            
            if rotate:
                user_agent = self._select_user_agent(domain, context)
                if user_agent:
                    # Store the user agent and update last rotation time
                    self._domain_user_agents[domain] = user_agent
                    self._last_rotation[domain] = time.time()
                    
                    # Update headers for this domain
                    self._update_headers_in_context(context, domain, user_agent)
                    
                    self._logger.debug(f"Rotated user agent for {domain}: {user_agent[:30]}...")
                    rotation_applied = True
                else:
                    self._logger.warning(f"Failed to select user agent for {domain}")
            else:
                # Re-use existing user agent
                user_agent = self._domain_user_agents[domain]
                
                # Ensure it's set in context
                self._update_headers_in_context(context, domain, user_agent)
                
                self._logger.debug(f"Re-using user agent for {domain}")
        
        # Update metrics if rotation occurred
        if rotation_applied:
            self._rotation_count += 1
            self._register_metrics({
                "rotation_count": self._rotation_count
            })
        
        return True
    
    def _extract_urls(self, context: PipelineContext) -> List[str]:
        """
        Extract URLs from the context to determine domains for user agent selection.
        
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
    
    def _should_rotate(self, domain: str) -> bool:
        """
        Determine if user agent should be rotated for a domain.
        
        Args:
            domain: The domain to check
            
        Returns:
            bool: True if rotation is needed, False otherwise
        """
        # If we don't have a user agent for this domain yet, we need to rotate
        if domain not in self._domain_user_agents:
            return True
        
        # Check if rotation interval has elapsed
        if self.rotation_interval > 0:
            last_rotation = self._last_rotation.get(domain, 0)
            time_since_rotation = time.time() - last_rotation
            
            if time_since_rotation > self.rotation_interval:
                return True
        
        return False
    
    def _select_user_agent(self, domain: Optional[str], context: PipelineContext) -> Optional[str]:
        """
        Select a user agent based on the configured strategy.
        
        Args:
            domain: The domain to select a user agent for (or None for global)
            context: The pipeline context with additional information
            
        Returns:
            Optional[str]: The selected user agent or None if selection failed
        """
        # Check if there are any user agents to choose from
        if not self._all_user_agents:
            self._logger.error("No user agents available for selection")
            return None
        
        # Apply selection strategy
        if self.rotation_strategy == "random":
            return random.choice(self._all_user_agents)
            
        elif self.rotation_strategy == "sequential":
            # Select next user agent in sequence and update index
            user_agent = self._all_user_agents[self._sequential_index % len(self._all_user_agents)]
            self._sequential_index += 1
            return user_agent
            
        elif self.rotation_strategy == "weighted":
            # Apply weights if defined, otherwise fall back to random
            if self.user_agent_weights:
                # Build weighted list
                weighted_agents = []
                
                for agent in self._all_user_agents:
                    # Try to match agent to a pattern in weights
                    weight = 1.0  # Default weight
                    
                    for pattern, w in self.user_agent_weights.items():
                        if pattern.lower() in agent.lower():
                            weight = w
                            break
                    
                    # Add agent to list with its weight
                    for _ in range(int(weight * 10)):  # Multiply by 10 to handle decimal weights
                        weighted_agents.append(agent)
                
                if weighted_agents:
                    return random.choice(weighted_agents)
            
            # Fall back to random if weights not applicable
            return random.choice(self._all_user_agents)
            
        else:
            # Default to random for unknown strategies
            return random.choice(self._all_user_agents)
    
    def _build_user_agent_list(self) -> List[str]:
        """
        Build the complete list of user agents based on configuration.
        
        Returns:
            List[str]: Complete list of available user agents
        """
        # If custom user agents are provided, use those
        if self.custom_user_agents:
            if isinstance(self.custom_user_agents, list):
                return self.custom_user_agents
            elif isinstance(self.custom_user_agents, str):
                return [self.custom_user_agents]
            else:
                self._logger.warning(f"Invalid custom_user_agents format: {type(self.custom_user_agents)}")
        
        # Otherwise build from default lists
        user_agents = []
        
        # Handle device categories
        for category in self.device_categories:
            if category in self.DEFAULT_USER_AGENTS:
                # Handle browsers for this category
                for browser in self.browsers:
                    if browser in self.DEFAULT_USER_AGENTS[category]:
                        user_agents.extend(self.DEFAULT_USER_AGENTS[category][browser])
        
        if not user_agents:
            # Fall back to all desktop Chrome if nothing matches
            self._logger.warning("No matching user agents found, falling back to defaults")
            return self.DEFAULT_USER_AGENTS["desktop"]["chrome"]
        
        return user_agents
    
    def _update_headers_in_context(self, context: PipelineContext, domain: Optional[str], user_agent: str) -> None:
        """
        Update headers in the context with the selected user agent.
        
        Args:
            context: The pipeline context
            domain: The domain for this user agent, or None for global
            user_agent: The user agent string to apply
        """
        # Check for existing headers in different locations
        updated = False
        
        # Case 1: Domain-specific headers
        if domain and context.contains("domain_headers"):
            domain_headers = context.get("domain_headers")
            if isinstance(domain_headers, dict):
                if domain not in domain_headers:
                    domain_headers[domain] = {}
                
                domain_headers[domain]["User-Agent"] = user_agent
                context.set("domain_headers", domain_headers)
                updated = True
        
        # Case 2: Request object with headers
        request = context.get("request")
        if request and hasattr(request, "headers"):
            if not isinstance(request.headers, dict):
                request.headers = {}
            
            request.headers["User-Agent"] = user_agent
            updated = True
        
        # Case 3: Headers dictionary
        if context.contains("headers"):
            headers = context.get("headers")
            if isinstance(headers, dict):
                headers["User-Agent"] = user_agent
                context.set("headers", headers)
                updated = True
        
        # If no existing headers were found, create a new headers dict
        if not updated:
            context.set("headers", {"User-Agent": user_agent})
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics about user agent rotation activity.
        
        Returns:
            Dict[str, Any]: Dictionary of user agent metrics
        """
        metrics = super().get_metrics()
        
        # Add user agent specific metrics
        ua_metrics = {
            "rotation_count": self._rotation_count,
            "user_agent_count": len(self._all_user_agents),
            "domain_count": len(self._domain_user_agents),
            "available_user_agents": self._all_user_agents[:3] + (["..."] if len(self._all_user_agents) > 3 else [])
        }
        
        metrics.update(ua_metrics)
        return metrics
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for user agent rotation stage configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing JSON schema
        """
        schema = super().get_config_schema()
        
        # Add user agent specific properties
        ua_properties = {
            "rotation_strategy": {"type": "string", "enum": ["random", "sequential", "weighted"]},
            "sticky_domain": {"type": "boolean"},
            "rotation_interval": {"type": "number", "minimum": 0},
            "device_categories": {
                "type": "array",
                "items": {"type": "string", "enum": ["desktop", "mobile", "tablet"]}
            },
            "browsers": {
                "type": "array",
                "items": {"type": "string", "enum": ["chrome", "firefox", "safari", "edge"]}
            },
            "custom_user_agents": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}}
                ]
            },
            "user_agent_weights": {
                "type": "object",
                "additionalProperties": {"type": "number", "minimum": 0}
            }
        }
        
        schema["properties"].update(ua_properties)
        return schema