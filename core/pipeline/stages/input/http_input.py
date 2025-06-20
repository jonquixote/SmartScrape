"""
HTTP Input Stage Module.

This module provides an enhanced HTTPInputStage that integrates with resource management
and error handling components created in Batch 5.
"""

import asyncio
import json
import logging
import re
import time
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlencode, urlparse

import aiohttp
import yarl
from aiohttp import ClientResponse, ClientSession, TCPConnector

from core.pipeline.stages.base_stage import BaseStage
from core.pipeline.context import PipelineContext
from core.pipeline.dto import PipelineRequest, PipelineResponse, RequestMethod, ResponseStatus
from core.service_registry import ServiceRegistry
from core.session_manager import SessionManager
from core.rate_limiter import RateLimiter
from core.proxy_manager import ProxyManager
from core.circuit_breaker import OpenCircuitError


class HTTPInputStage(BaseStage):
    """
    Enhanced HTTP Input Stage with resource management and error handling.
    
    This stage integrates with:
    - SessionManager for HTTP session handling and user agent rotation
    - RateLimiter for domain-specific rate limiting
    - ProxyManager for proxy selection and rotation
    - ErrorClassifier for structured error handling
    - CircuitBreaker for service protection
    
    Features:
    - Enhanced error handling with classification and recovery
    - Circuit breaker protection for domains
    - Automatic session management and reuse
    - Intelligent rate limiting with backoff
    - Smart proxy selection and rotation on failures
    - Detailed telemetry and metrics
    - Resource usage tracking
    
    Configuration:
    - url_template: Template string for URL with {variable} placeholders
    - method: HTTP method to use (GET, POST, etc.)
    - headers: Default headers to include in requests
    - params: Default query parameters
    - auth: Authentication configuration (basic, digest, bearer)
    - follow_redirects: Whether to automatically follow redirects
    - max_redirects: Maximum number of redirects to follow
    - timeout: Request timeout in seconds
    - verify_ssl: Whether to verify SSL certificates
    - respect_robots_txt: Whether to respect robots.txt rules
    - use_session_manager: Whether to use SessionManager (default: True)
    - use_rate_limiter: Whether to use RateLimiter (default: True)
    - use_proxy_manager: Whether to use ProxyManager (default: False)
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new HTTP input stage.
        
        Args:
            name (Optional[str]): Name of the stage, defaults to class name
            config (Optional[Dict[str, Any]]): Configuration parameters
        """
        super().__init__(name, config)
        
        # Store HTTP-specific configuration
        self.url_template = self.config.get("url_template", "")
        self.method = self.config.get("method", "GET").upper()
        self.headers = self.config.get("headers", {})
        self.params = self.config.get("params", {})
        self.auth = self.config.get("auth", None)
        self.follow_redirects = self.config.get("follow_redirects", True)
        self.max_redirects = self.config.get("max_redirects", 10)
        self.timeout = self.config.get("timeout", 30)
        self.verify_ssl = self.config.get("verify_ssl", True)
        self.respect_robots_txt = self.config.get("respect_robots_txt", True)
        self.compression = self.config.get("compression", True)
        
        # Resource management configuration
        self.use_session_manager = self.config.get("use_session_manager", True)
        self.use_rate_limiter = self.config.get("use_rate_limiter", True)
        self.use_proxy_manager = self.config.get("use_proxy_manager", False)
        self.proxy_settings = self.config.get("proxy_settings", None)
        
        # Service references (will be loaded when needed)
        self._session_manager = None
        self._rate_limiter = None
        self._proxy_manager = None
        self._url_service = None
        self._session = None
        
        # Stats tracking
        self._request_count = 0
        self._success_count = 0
        self._error_count = 0
        self._retry_count = 0
    
    async def process(self, context: PipelineContext) -> bool:
        """
        Process the input stage, acquiring data from the specified HTTP source.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            bool: True if processing succeeded, False otherwise.
        """
        # Get the input request from context or create default
        request = self._get_request(context)
        
        try:
            # Execute HTTP request with protection
            response = await self._execute_with_protection(
                self._execute_http_request,
                "http_request",
                context,
                request,
                context
            )
            
            # Store the response in context
            if response:
                self._store_response(response, context)
                return True
            
            return False
            
        except Exception as e:
            await self.handle_error(context, e)
            return False
    
    async def _execute_http_request(self, request: PipelineRequest, context: PipelineContext) -> Optional[PipelineResponse]:
        """
        Execute an HTTP request with integrated resource management.
        
        Args:
            request (PipelineRequest): The input request
            context (PipelineContext): The shared pipeline context
            
        Returns:
            Optional[PipelineResponse]: The response or None if failed
        """
        # Get the URL from template or request
        url = self._get_url(request, context)
        if not url:
            self._logger.error("No URL provided for HTTP request")
            return None
        
        # Extract domain for domain-specific handling
        domain = urlparse(url).netloc
        
        # Wait for rate limiting if enabled
        if self.use_rate_limiter:
            await self._apply_rate_limiting(domain)
        
        # Check robots.txt if enabled
        if self.respect_robots_txt and not await self._check_robots_txt(url):
            self._logger.warning(f"URL {url} is disallowed by robots.txt")
            return self._create_error_response(
                url, 
                "URL is disallowed by robots.txt", 
                ResponseStatus.FORBIDDEN
            )
        
        # Get or create a session
        session = await self._get_session(domain)
        
        # Prepare request parameters
        method = self._get_method(request)
        headers = self._merge_headers(request, domain)
        params = self._merge_params(request, context)
        data = self._get_request_body(request)
        proxy = await self._get_proxy(url)
        
        # Update metrics
        self._resource_metrics["network_requests"] += 1
        self._request_count += 1
        
        try:
            # Make the HTTP request
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                proxy=proxy,
                ssl=None if not self.verify_ssl else True,
                allow_redirects=self.follow_redirects,
                max_redirects=self.max_redirects,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                compress=self.compression
            ) as response:
                # Handle rate limiting
                if response.status == HTTPStatus.TOO_MANY_REQUESTS:
                    if self.use_rate_limiter:
                        await self._handle_rate_limited(domain)
                    return await self.handle_rate_limiting(response)
                    
                # Handle authentication challenges
                elif response.status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
                    return await self.handle_authentication(response)
                    
                # Handle redirects (if not following)
                elif 300 <= response.status < 400 and not self.follow_redirects:
                    return await self.handle_redirect(response)
                
                # Handle error responses
                elif response.status >= 400:
                    self._error_count += 1
                    return self._create_error_response_from_http(response)
                
                # Process successful response
                self._success_count += 1
                return await self._process_response(response, url, context)
                
        except aiohttp.ClientError as e:
            self._error_count += 1
            self._logger.error(f"HTTP request error for {url}: {str(e)}")
            return self._create_error_response(
                url, 
                f"HTTP request error: {str(e)}", 
                ResponseStatus.ERROR
            )
        except asyncio.TimeoutError:
            self._error_count += 1
            self._logger.error(f"Timeout for {url}")
            return self._create_error_response(
                url, 
                "Request timed out", 
                ResponseStatus.TIMEOUT
            )
    
    async def _get_session(self, domain: str) -> ClientSession:
        """
        Get an HTTP session for the specified domain.
        
        Args:
            domain (str): The domain to get a session for
            
        Returns:
            ClientSession: The session to use
        """
        # Use SessionManager if enabled
        if self.use_session_manager:
            try:
                if not self._session_manager:
                    self._session_manager = self._get_service("session_manager")
                
                # Get a session from the manager
                session = self._session_manager.get_session(domain)
                return session
            except Exception as e:
                self._logger.warning(f"Error using SessionManager: {str(e)}")
        
        # Fall back to local session
        if self._session is None or self._session.closed:
            # Configure connection options
            connector = TCPConnector(
                ssl=None if not self.verify_ssl else True,
                limit=self.config.get("connection_limit", 100),
                ttl_dns_cache=self.config.get("dns_cache_ttl", 10 * 60),  # 10 minutes
                enable_cleanup_closed=True
            )
            
            self._session = ClientSession(connector=connector)
        
        return self._session
    
    async def _apply_rate_limiting(self, domain: str) -> None:
        """
        Apply rate limiting for the specified domain.
        
        Args:
            domain (str): The domain to apply rate limiting for
        """
        if not self.use_rate_limiter:
            return
            
        try:
            if not self._rate_limiter:
                self._rate_limiter = self._get_service("rate_limiter")
            
            # Wait if needed to comply with rate limits
            wait_needed = self._rate_limiter.wait_if_needed(domain)
            if wait_needed:
                self._logger.debug(f"Rate limited for {domain}")
        except Exception as e:
            self._logger.warning(f"Error using RateLimiter: {str(e)}")
            
            # Fall back to simple rate limiting
            await self._simple_rate_limiting(domain)
    
    async def _simple_rate_limiting(self, domain: str) -> None:
        """
        Apply simple rate limiting when RateLimiter service is unavailable.
        
        Args:
            domain (str): The domain to apply rate limiting for
        """
        # Default rate limit: 1 request per second
        rate_limit = self.config.get("fallback_rate_limit", 1.0)
        min_interval = 1.0 / rate_limit
        
        # Use domain-specific tracking of last request time
        domain_times = getattr(self, "_domain_request_times", {})
        last_time = domain_times.get(domain, 0)
        
        current_time = time.time()
        time_since_last = current_time - last_time
        
        if time_since_last < min_interval:
            delay = min_interval - time_since_last
            await asyncio.sleep(delay)
        
        # Update last request time
        domain_times[domain] = time.time()
        self._domain_request_times = domain_times
    
    async def _handle_rate_limited(self, domain: str) -> None:
        """
        Handle a rate limiting response.
        
        Args:
            domain (str): The domain that rate limited the request
        """
        try:
            if not self._rate_limiter:
                self._rate_limiter = self._get_service("rate_limiter")
            
            # Report rate limiting to adjust future limits
            self._rate_limiter.report_rate_limited(domain)
        except Exception as e:
            self._logger.warning(f"Error reporting rate limiting: {str(e)}")
    
    async def _get_proxy(self, url: str) -> Optional[str]:
        """
        Get a proxy URL for the request using ProxyManager if available.
        
        Args:
            url (str): The request URL
            
        Returns:
            Optional[str]: Proxy URL or None if no proxy should be used
        """
        # Check if we should use a proxy
        if not self.use_proxy_manager and not self.proxy_settings:
            return None
        
        # If static proxy is configured and not using manager, use that
        if not self.use_proxy_manager and isinstance(self.proxy_settings, str):
            return self.proxy_settings
        
        # Try to use proxy manager
        try:
            if self.use_proxy_manager:
                if not self._proxy_manager:
                    self._proxy_manager = self._get_service("proxy_manager")
                
                # Get a proxy from the manager
                return await self._proxy_manager.get_proxy(url)
        except Exception as e:
            self._logger.warning(f"Error using ProxyManager: {str(e)}")
        
        # Fall back to configured static proxy if available
        if isinstance(self.proxy_settings, dict):
            return self.proxy_settings.get("url")
        
        return None
    
    def _merge_headers(self, request: PipelineRequest, domain: str = None) -> Dict[str, str]:
        """
        Merge headers from config and request, using SessionManager if available.
        
        Args:
            request (PipelineRequest): The input request
            domain (str, optional): The domain for the request
            
        Returns:
            Dict[str, str]: The merged headers
        """
        # Try to get headers from SessionManager first
        if self.use_session_manager and domain and self._session_manager:
            try:
                # Assuming session manager provides a get_headers method
                headers = self._session_manager.get_headers(domain)
            except Exception:
                # Fall back to default headers
                headers = {
                    "User-Agent": "SmartScrape/1.0",
                    "Accept": "*/*",
                    "Accept-Encoding": "gzip, deflate" if self.compression else "identity"
                }
        else:
            # Use default headers
            headers = {
                "User-Agent": "SmartScrape/1.0",
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate" if self.compression else "identity"
            }
        
        # Add configured headers
        if self.headers:
            headers.update(self.headers)
        
        # Add request-specific headers (highest priority)
        if request.headers:
            headers.update(request.headers)
        
        return headers
    
    def _get_request(self, context: PipelineContext) -> PipelineRequest:
        """
        Get the input request from context or create a default one.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            PipelineRequest: The input request data.
        """
        # Check if there's a request object in context
        request = context.get("request")
        if request is not None and isinstance(request, PipelineRequest):
            return request
            
        # Create a default request using context data
        return PipelineRequest(
            source=self.config.get("source", ""),
            params=context.data.copy()
        )
    
    def _store_response(self, response: PipelineResponse, context: PipelineContext) -> None:
        """
        Store the response data in the context.
        
        Args:
            response (PipelineResponse): The response data.
            context (PipelineContext): The shared pipeline context.
        """
        # Store the full response object
        context.set("response", response)
        
        # Also store the response data directly in context for easier access
        if response.data:
            context.update(response.data)
        
        # Add stage metrics
        self._register_metrics({
            "requests": self._request_count,
            "success": self._success_count,
            "errors": self._error_count,
            "retries": self._retry_count,
            "success_rate": (self._success_count / self._request_count) if self._request_count > 0 else 0
        })
    
    async def handle_redirect(self, response: ClientResponse) -> PipelineResponse:
        """
        Handle HTTP redirects when automatic following is disabled.
        
        Args:
            response (ClientResponse): The HTTP response
            
        Returns:
            PipelineResponse: Response with redirect information
        """
        location = response.headers.get('Location', '')
        redirect_url = str(yarl.URL(location).join(response.url))
        
        self._logger.info(f"Redirect from {response.url} to {redirect_url}")
        
        # Create a response containing the redirect information
        return PipelineResponse(
            status=ResponseStatus.SUCCESS,
            data={
                "redirect_url": redirect_url,
                "original_url": str(response.url),
                "status_code": response.status,
                "is_redirect": True
            },
            source=str(response.url),
            metadata={
                "redirect_count": len(response.history),
                "redirect_chain": [str(r.url) for r in response.history]
            },
            headers=dict(response.headers),
            status_code=response.status
        )
    
    async def handle_rate_limiting(self, response: ClientResponse) -> PipelineResponse:
        """
        Handle rate limiting (HTTP 429) responses.
        
        Args:
            response (ClientResponse): The HTTP response
            
        Returns:
            PipelineResponse: Response with rate limiting information
        """
        # Extract retry-after header if available
        retry_after = response.headers.get('Retry-After')
        wait_time = None
        
        if retry_after:
            try:
                # Try to parse as an integer (seconds)
                wait_time = int(retry_after)
            except ValueError:
                # Try to parse as a HTTP date
                pass  # Could implement HTTP date parsing if needed
        
        # Use default wait time if no valid retry-after was found
        if wait_time is None:
            wait_time = 60  # Default to 60 seconds wait
            
        self._logger.warning(f"Rate limited at {response.url}. Retry after {wait_time}s")
        
        # Log this rate limiting event to help improve rate limiting strategies
        domain = urlparse(str(response.url)).netloc
        rate_limit_context = {"domain": domain, "wait_time": wait_time, "timestamp": time.time()}
        
        # Create a response indicating rate limiting
        return PipelineResponse(
            status=ResponseStatus.RATE_LIMITED,
            data={"retry_after": wait_time},
            source=str(response.url),
            error_message=f"Rate limited. Retry after {wait_time} seconds",
            metadata={"rate_limiting_context": rate_limit_context},
            headers=dict(response.headers),
            status_code=response.status
        )
    
    async def handle_authentication(self, response: ClientResponse) -> PipelineResponse:
        """
        Handle authentication challenges (401/403 responses).
        
        Args:
            response (ClientResponse): The HTTP response
            
        Returns:
            PipelineResponse: Response with authentication information
        """
        auth_header = response.headers.get('WWW-Authenticate', '')
        
        # Determine auth type from header
        auth_type = "unknown"
        auth_realm = ""
        
        if auth_header:
            if auth_header.startswith('Basic'):
                auth_type = "basic"
                match = re.search(r'realm="([^"]+)"', auth_header)
                if match:
                    auth_realm = match.group(1)
            elif auth_header.startswith('Digest'):
                auth_type = "digest"
            elif auth_header.startswith('Bearer'):
                auth_type = "bearer"
            
        self._logger.warning(f"Authentication required for {response.url} ({auth_type})")
        
        # Create authentication challenge response
        return PipelineResponse(
            status=ResponseStatus.UNAUTHORIZED if response.status == 401 else ResponseStatus.FORBIDDEN,
            data={
                "auth_type": auth_type,
                "auth_realm": auth_realm,
                "auth_header": auth_header
            },
            source=str(response.url),
            error_message=f"Authentication required: {auth_type}",
            metadata={"auth_challenge": True},
            headers=dict(response.headers),
            status_code=response.status
        )
    
    def extract_content_type(self, response: ClientResponse) -> Tuple[str, str]:
        """
        Extract and parse content type and charset from response.
        
        Args:
            response (ClientResponse): The HTTP response
            
        Returns:
            Tuple[str, str]: Tuple of (mime_type, charset)
        """
        content_type = response.headers.get('Content-Type', '')
        mime_type = 'application/octet-stream'
        charset = 'utf-8'
        
        if content_type:
            parts = content_type.split(';')
            mime_type = parts[0].strip().lower()
            
            # Look for charset in Content-Type
            for part in parts[1:]:
                if 'charset=' in part.lower():
                    charset = part.split('=')[1].strip().lower()
                    break
        
        return mime_type, charset
    
    async def validate_response(self, response: ClientResponse, context: PipelineContext) -> bool:
        """
        Validate the integrity and quality of the HTTP response.
        
        Args:
            response (ClientResponse): The HTTP response
            context (PipelineContext): The shared pipeline context
            
        Returns:
            bool: True if the response is valid, False otherwise
        """
        # Check if response has a body
        if response.content_length == 0 and not response.headers.get('Transfer-Encoding') == 'chunked':
            self._logger.warning(f"Empty response from {response.url}")
            return False
        
        # Validate content type if expected type is specified
        expected_type = self.config.get("expected_content_type")
        if expected_type:
            mime_type, _ = self.extract_content_type(response)
            if not mime_type.startswith(expected_type):
                self._logger.warning(
                    f"Unexpected content type: got {mime_type}, expected {expected_type}"
                )
                return False
        
        return True
    
    async def _process_response(self, response: ClientResponse, url: str, 
                              context: PipelineContext) -> PipelineResponse:
        """
        Process a successful HTTP response.
        
        Args:
            response (ClientResponse): The HTTP response
            url (str): The request URL
            context (PipelineContext): The shared pipeline context
            
        Returns:
            PipelineResponse: The processed response
        """
        # Validate the response
        if not await self.validate_response(response, context):
            return self._create_error_response(
                url,
                "Response validation failed",
                ResponseStatus.ERROR
            )
        
        # Extract content type information
        mime_type, charset = self.extract_content_type(response)
        
        # Process response based on content type
        try:
            if mime_type == 'application/json':
                data = await response.json(encoding=charset)
                response_data = {"json": data}
            
            elif mime_type.startswith('text/'):
                text = await response.text(encoding=charset)
                response_data = {"text": text}
                
                # Special handling for HTML
                if mime_type == 'text/html':
                    response_data["html"] = text
            
            elif mime_type.startswith('image/'):
                binary = await response.read()
                response_data = {
                    "binary": binary,
                    "content_type": mime_type
                }
            
            else:
                # Default to binary for unknown types
                binary = await response.read()
                response_data = {
                    "binary": binary,
                    "content_type": mime_type
                }
            
            # Create successful response object
            return PipelineResponse(
                status=ResponseStatus.SUCCESS,
                data=response_data,
                source=url,
                metadata={
                    "mime_type": mime_type,
                    "charset": charset,
                    "content_length": response.content_length,
                    "history": [str(r.url) for r in response.history]
                },
                headers=dict(response.headers),
                status_code=response.status
            )
            
        except Exception as e:
            self._logger.error(f"Error processing response: {str(e)}")
            return self._create_error_response(
                url,
                f"Error processing response: {str(e)}",
                ResponseStatus.ERROR
            )
    
    def _get_url(self, request: PipelineRequest, context: PipelineContext) -> str:
        """
        Get the URL for the HTTP request, applying template variables.
        
        Args:
            request (PipelineRequest): The input request
            context (PipelineContext): The shared pipeline context
            
        Returns:
            str: The resolved URL
        """
        # Try to get URL from different sources in order of precedence
        url = None
        
        # 1. From request source
        if request.source:
            url = request.source
        
        # 2. From config url_template with context variables
        elif self.url_template:
            try:
                # Apply template with context variables
                template_vars = {**context.data, **request.params}
                url = self.url_template.format(**template_vars)
            except KeyError as e:
                self._logger.error(f"Missing template variable: {str(e)}")
                return ""
            except Exception as e:
                self._logger.error(f"Error applying URL template: {str(e)}")
                return ""
        
        # 3. From config url
        else:
            url = self.config.get("url", "")
        
        # Normalize the URL if we have the URL service
        if url and self._url_service:
            return self._url_service.normalize_url(url)
        
        return url
    
    def _get_method(self, request: PipelineRequest) -> str:
        """
        Get the HTTP method to use.
        
        Args:
            request (PipelineRequest): The input request
            
        Returns:
            str: The HTTP method
        """
        # Try request method first, fall back to configured method
        if isinstance(request.method, RequestMethod):
            return request.method.name
        
        return self.method
    
    def _merge_params(self, request: PipelineRequest, context: PipelineContext) -> Dict[str, str]:
        """
        Merge query parameters from config, request, and context.
        
        Args:
            request (PipelineRequest): The input request
            context (PipelineContext): The shared pipeline context
            
        Returns:
            Dict[str, str]: The merged query parameters
        """
        # Start with configured params
        params = dict(self.params)
        
        # Add request params
        if request.params:
            # Only add params that aren't already template variables
            if self.url_template:
                # Avoid adding params that are used in the URL template
                template_vars = set(re.findall(r'\{(\w+)\}', self.url_template))
                for key, value in request.params.items():
                    if key not in template_vars:
                        params[key] = value
            else:
                params.update(request.params)
        
        # Convert any non-string values to strings
        for key, value in params.items():
            if not isinstance(value, str):
                if isinstance(value, (int, float, bool)):
                    params[key] = str(value)
                elif value is None:
                    params[key] = ""
                else:
                    params[key] = json.dumps(value)
        
        return params
    
    def _get_request_body(self, request: PipelineRequest) -> Optional[Any]:
        """
        Get the request body data.
        
        Args:
            request (PipelineRequest): The input request
            
        Returns:
            Optional[Any]: The request body or None
        """
        # Don't include body for methods that shouldn't have one
        method = self._get_method(request)
        if method in ("GET", "HEAD", "OPTIONS"):
            return None
        
        # Use request body if provided
        if request.body is not None:
            return request.body
        
        # Use configured body if available
        return self.config.get("body")
    
    def _create_error_response(self, url: str, message: str, 
                             status: ResponseStatus) -> PipelineResponse:
        """
        Create an error response.
        
        Args:
            url (str): The request URL
            message (str): The error message
            status (ResponseStatus): The error status
            
        Returns:
            PipelineResponse: The error response
        """
        return PipelineResponse(
            status=status,
            data=None,
            source=url,
            error_message=message,
            metadata={"error_type": status.name},
            headers={},
            status_code=None
        )
    
    def _create_error_response_from_http(self, response: ClientResponse) -> PipelineResponse:
        """
        Create an error response from an HTTP error response.
        
        Args:
            response (ClientResponse): The HTTP error response
            
        Returns:
            PipelineResponse: The error response
        """
        url = str(response.url)
        
        # Map HTTP status to response status
        status_mapping = {
            404: ResponseStatus.NOT_FOUND,
            401: ResponseStatus.UNAUTHORIZED,
            403: ResponseStatus.FORBIDDEN,
            429: ResponseStatus.RATE_LIMITED,
            500: ResponseStatus.SERVER_ERROR,
            503: ResponseStatus.SERVER_ERROR,
            504: ResponseStatus.TIMEOUT
        }
        
        status = status_mapping.get(response.status, ResponseStatus.ERROR)
        
        return PipelineResponse(
            status=status,
            data=None,
            source=url,
            error_message=f"HTTP error {response.status}: {response.reason}",
            metadata={"http_status": response.status},
            headers=dict(response.headers),
            status_code=response.status
        )
    
    async def _check_robots_txt(self, url: str) -> bool:
        """
        Check if a URL is allowed by robots.txt.
        
        Args:
            url (str): The URL to check
            
        Returns:
            bool: True if allowed or robots.txt check disabled, False otherwise
        """
        if not self.respect_robots_txt:
            return True
        
        # Try to get URL service from service registry
        try:
            if not self._url_service:
                self._url_service = self._get_service("url_service")
            
            # Use URL service to check robots.txt
            return self._url_service.is_allowed(url)
        except Exception as e:
            self._logger.warning(f"Error checking robots.txt: {str(e)}")
            return True  # Allow on error
    
    async def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        # Close local session if we created one
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for HTTP input stage configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing JSON schema
        """
        schema = super().get_config_schema()
        
        # Add HTTP-specific properties
        http_properties = {
            "url": {"type": "string", "format": "uri"},
            "url_template": {"type": "string"},
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]},
            "headers": {"type": "object", "additionalProperties": {"type": "string"}},
            "params": {"type": "object"},
            "auth": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["basic", "digest", "bearer"]},
                    "username": {"type": "string"},
                    "password": {"type": "string"},
                    "token": {"type": "string"}
                }
            },
            "follow_redirects": {"type": "boolean"},
            "max_redirects": {"type": "integer", "minimum": 0},
            "timeout": {"type": "number", "minimum": 0},
            "verify_ssl": {"type": "boolean"},
            "respect_robots_txt": {"type": "boolean"},
            "use_session_manager": {"type": "boolean"},
            "use_rate_limiter": {"type": "boolean"},
            "use_proxy_manager": {"type": "boolean"},
            "proxy_settings": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "use_proxy_manager": {"type": "boolean"}
                        }
                    }
                ]
            },
            "compression": {"type": "boolean"},
            "expected_content_type": {"type": "string"},
            "fallback_rate_limit": {"type": "number", "minimum": 0.1}
        }
        
        # Update the properties in the schema
        schema["properties"].update(http_properties)
        
        return schema
