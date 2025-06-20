"""
Tests for HTTP Input Stage Module.

This module contains tests for the HTTPInputStage class, testing its ability to acquire
data from HTTP sources with various configurations and error scenarios.
"""

import asyncio
import json
import os
import re
import time
from http import HTTPStatus
from typing import Dict, Any, Optional, List
from unittest import mock
from urllib.parse import urlparse, urljoin

import aiohttp
import pytest
from aiohttp import ClientResponse, ClientSession, web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from core.pipeline.stages.input.http_input import HTTPInputStage
from core.pipeline.context import PipelineContext
from core.pipeline.dto import PipelineRequest, PipelineResponse, ResponseStatus, RequestMethod
from core.service_registry import ServiceRegistry


class MockURLService:
    """Mock URL Service for testing."""
    
    def __init__(self):
        self.normalized_urls = {}
        self.allowed_urls = {}
    
    def normalize_url(self, url: str) -> str:
        """Mock normalize_url method."""
        return self.normalized_urls.get(url, url)
    
    def is_allowed(self, url: str) -> bool:
        """Mock is_allowed method for robots.txt checking."""
        domain = urlparse(url).netloc
        path = urlparse(url).path
        
        # Check if we have explicit allow/disallow for this URL
        if url in self.allowed_urls:
            return self.allowed_urls[url]
        
        # Check if we have domain-level rules
        if domain in self.allowed_urls:
            return self.allowed_urls[domain]
        
        # Default to allowed
        return True
    
    def set_normalization(self, url: str, normalized: str) -> None:
        """Set up URL normalization mapping."""
        self.normalized_urls[url] = normalized
    
    def set_allowed(self, url: str, allowed: bool) -> None:
        """Set up robots.txt allow/disallow mapping."""
        self.allowed_urls[url] = allowed


class MockProxyManager:
    """Mock Proxy Manager for testing."""
    
    def __init__(self):
        self.proxies = {}
        self.default_proxy = None
    
    async def get_proxy(self, url: str) -> Optional[str]:
        """Mock get_proxy method."""
        domain = urlparse(url).netloc
        
        # Check for URL-specific proxy
        if url in self.proxies:
            return self.proxies[url]
        
        # Check for domain-specific proxy
        if domain in self.proxies:
            return self.proxies[domain]
        
        # Return default proxy if set
        return self.default_proxy
    
    def set_proxy(self, url: str, proxy: str) -> None:
        """Set up proxy mapping."""
        self.proxies[url] = proxy
    
    def set_default_proxy(self, proxy: str) -> None:
        """Set default proxy."""
        self.default_proxy = proxy


class MockServiceRegistry:
    """Mock Service Registry for testing."""
    
    def __init__(self):
        self.services = {}
    
    def get_service(self, service_name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Mock get_service method."""
        if service_name not in self.services:
            raise KeyError(f"Service {service_name} not registered")
        return self.services[service_name]
    
    def register_service(self, service_name: str, service: Any) -> None:
        """Register a mock service."""
        self.services[service_name] = service


class MockResponse:
    """Mock aiohttp ClientResponse for testing."""
    
    def __init__(self, url: str, status: int, headers: Optional[Dict[str, str]] = None, 
                body: Optional[bytes] = None, content_type: str = "text/plain",
                history: Optional[List["MockResponse"]] = None):
        """Initialize mock response."""
        self.url = web.URL(url)
        self.status = status
        self.headers = headers or {}
        self._body = body or b""
        
        # Set Content-Type if not in headers
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = content_type
        
        self.content_length = len(self._body)
        self.history = history or []
        self.reason = HTTPStatus(status).phrase
        self.closed = False
    
    async def read(self) -> bytes:
        """Mock read method."""
        return self._body
    
    async def text(self, encoding: Optional[str] = None) -> str:
        """Mock text method."""
        encoding = encoding or "utf-8"
        return self._body.decode(encoding)
    
    async def json(self, encoding: Optional[str] = None) -> Any:
        """Mock json method."""
        text = await self.text(encoding)
        return json.loads(text)
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.closed = True


class TestHTTPInputStage:
    """Test cases for HTTPInputStage."""

    @pytest.fixture
    def url_service(self):
        """Fixture for URL service."""
        service = MockURLService()
        return service
    
    @pytest.fixture
    def proxy_manager(self):
        """Fixture for proxy manager."""
        service = MockProxyManager()
        return service
    
    @pytest.fixture
    def service_registry(self, url_service, proxy_manager):
        """Fixture for service registry with mock services."""
        registry = MockServiceRegistry()
        registry.register_service("url_service", url_service)
        registry.register_service("proxy_manager", proxy_manager)
        return registry
    
    @pytest.fixture
    def http_stage(self):
        """Fixture for basic HTTP input stage."""
        config = {
            "url": "https://example.com/api",
            "method": "GET",
            "headers": {
                "User-Agent": "SmartScrape/1.0 TestSuite"
            },
            "timeout": 5,
            "follow_redirects": True,
            "respect_robots_txt": True
        }
        return HTTPInputStage(name="test_http_stage", config=config)
    
    @pytest.fixture
    def template_stage(self):
        """Fixture for HTTP input stage with URL template."""
        config = {
            "url_template": "https://example.com/api/{endpoint}?q={query}",
            "method": "GET",
            "headers": {
                "User-Agent": "SmartScrape/1.0 TestSuite"
            },
            "timeout": 5
        }
        return HTTPInputStage(name="template_http_stage", config=config)
    
    @pytest.mark.asyncio
    async def test_successful_get_request(self, http_stage, monkeypatch):
        """Test successful GET request."""
        # Mock the session's request method
        async def mock_request(*args, **kwargs):
            headers = {"Content-Type": "application/json"}
            body = json.dumps({"result": "success", "data": {"id": 123, "name": "Test"}}).encode("utf-8")
            return MockResponse(
                url="https://example.com/api",
                status=200,
                headers=headers,
                body=body,
                content_type="application/json"
            )
        
        # Apply the monkeypatch for session.request
        monkeypatch.setattr(ClientSession, "request", mock_request)
        
        # Initialize URL service
        http_stage._url_service = MockURLService()
        
        # Create request and context
        request = PipelineRequest(source="https://example.com/api")
        context = PipelineContext({})
        
        # Execute the stage
        response = await http_stage.acquire_data(request, context)
        
        # Verify the response
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "json" in response.data
        assert response.data["json"]["result"] == "success"
        assert response.status_code == 200
        assert response.source == "https://example.com/api"
    
    @pytest.mark.asyncio
    async def test_url_template_resolution(self, template_stage, monkeypatch):
        """Test URL template resolution with context variables."""
        expected_url = "https://example.com/api/users?q=searchterm"
        
        # Mock the session's request method
        async def mock_request(method, url, **kwargs):
            assert url == expected_url, f"URL not properly resolved: {url} != {expected_url}"
            headers = {"Content-Type": "application/json"}
            body = json.dumps({"result": "success", "users": [{"id": 1, "name": "User 1"}]}).encode("utf-8")
            return MockResponse(
                url=url,
                status=200,
                headers=headers,
                body=body,
                content_type="application/json"
            )
        
        # Apply the monkeypatch for session.request
        monkeypatch.setattr(ClientSession, "request", mock_request)
        
        # Create request and context with template variables
        request = PipelineRequest(params={"endpoint": "users", "query": "searchterm"})
        context = PipelineContext({})
        
        # Execute the stage
        response = await template_stage.acquire_data(request, context)
        
        # Verify the response
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "json" in response.data
        assert response.data["json"]["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_robots_txt_compliance(self, http_stage, monkeypatch):
        """Test robots.txt compliance."""
        # Set up URL service with disallowed URL
        url_service = MockURLService()
        url_service.set_allowed("https://example.com/api", False)
        http_stage._url_service = url_service
        
        # Create request and context
        request = PipelineRequest(source="https://example.com/api")
        context = PipelineContext({})
        
        # Execute the stage
        response = await http_stage.acquire_data(request, context)
        
        # Verify the response indicates forbidden by robots.txt
        assert response is not None
        assert response.status == ResponseStatus.FORBIDDEN
        assert "disallowed by robots.txt" in response.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_rate_limiting_handling(self, http_stage, monkeypatch):
        """Test handling of rate limiting (HTTP 429)."""
        retry_seconds = 60
        
        # Mock the session's request method to return a 429 response
        async def mock_request(*args, **kwargs):
            headers = {"Retry-After": str(retry_seconds)}
            return MockResponse(
                url="https://example.com/api",
                status=429,
                headers=headers,
                body=b"Too Many Requests",
                content_type="text/plain"
            )
        
        # Apply the monkeypatch for session.request
        monkeypatch.setattr(ClientSession, "request", mock_request)
        
        # Create request and context
        request = PipelineRequest(source="https://example.com/api")
        context = PipelineContext({})
        
        # Execute the stage
        response = await http_stage.acquire_data(request, context)
        
        # Verify the response indicates rate limiting
        assert response is not None
        assert response.status == ResponseStatus.RATE_LIMITED
        assert response.data is not None
        assert "retry_after" in response.data
        assert response.data["retry_after"] == retry_seconds
    
    @pytest.mark.asyncio
    async def test_authentication_challenge(self, http_stage, monkeypatch):
        """Test handling of authentication challenges (HTTP 401)."""
        # Mock the session's request method to return a 401 response
        async def mock_request(*args, **kwargs):
            headers = {"WWW-Authenticate": 'Basic realm="API Access"'}
            return MockResponse(
                url="https://example.com/api",
                status=401,
                headers=headers,
                body=b"Unauthorized",
                content_type="text/plain"
            )
        
        # Apply the monkeypatch for session.request
        monkeypatch.setattr(ClientSession, "request", mock_request)
        
        # Create request and context
        request = PipelineRequest(source="https://example.com/api")
        context = PipelineContext({})
        
        # Execute the stage
        response = await http_stage.acquire_data(request, context)
        
        # Verify the response indicates authentication required
        assert response is not None
        assert response.status == ResponseStatus.UNAUTHORIZED
        assert response.data is not None
        assert response.data["auth_type"] == "basic"
        assert response.data["auth_realm"] == "API Access"
    
    @pytest.mark.asyncio
    async def test_request_timeout(self, http_stage, monkeypatch):
        """Test handling of request timeout."""
        # Mock the session's request method to raise a timeout error
        async def mock_request(*args, **kwargs):
            raise asyncio.TimeoutError("Request timed out")
        
        # Apply the monkeypatch for session.request
        monkeypatch.setattr(ClientSession, "request", mock_request)
        
        # Create request and context
        request = PipelineRequest(source="https://example.com/api")
        context = PipelineContext({})
        
        # Execute the stage
        response = await http_stage.acquire_data(request, context)
        
        # Verify the response indicates timeout
        assert response is not None
        assert response.status == ResponseStatus.TIMEOUT
        assert "timed out" in response.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_redirect_handling(self, monkeypatch):
        """Test redirect handling with automatic following disabled."""
        # Create a stage with redirect following disabled
        config = {
            "url": "https://example.com/old",
            "method": "GET",
            "follow_redirects": False
        }
        stage = HTTPInputStage(name="no_redirect_stage", config=config)
        
        # Mock the session's request method to return a redirect
        async def mock_request(*args, **kwargs):
            headers = {"Location": "https://example.com/new"}
            return MockResponse(
                url="https://example.com/old",
                status=302,
                headers=headers,
                body=b"Redirecting...",
                content_type="text/plain"
            )
        
        # Apply the monkeypatch for session.request
        monkeypatch.setattr(ClientSession, "request", mock_request)
        
        # Create request and context
        request = PipelineRequest(source="https://example.com/old")
        context = PipelineContext({})
        
        # Execute the stage
        response = await stage.acquire_data(request, context)
        
        # Verify the response contains redirect information
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert response.data["is_redirect"] is True
        assert response.data["redirect_url"] == "https://example.com/new"
    
    @pytest.mark.asyncio
    async def test_proxy_integration(self, http_stage, monkeypatch):
        """Test integration with proxy manager."""
        # Set up proxy manager with a test proxy
        test_proxy = "http://testproxy:8080"
        proxy_manager = MockProxyManager()
        proxy_manager.set_proxy("https://example.com/api", test_proxy)
        http_stage._proxy_manager = proxy_manager
        
        # Set proxy configuration to use proxy manager
        http_stage.proxy_settings = {"use_proxy_manager": True}
        
        # Mock the session's request method to verify proxy is used
        async def mock_request(method, url, **kwargs):
            # Verify proxy is passed correctly
            assert kwargs.get("proxy") == test_proxy, f"Proxy not properly applied: {kwargs.get('proxy')} != {test_proxy}"
            return MockResponse(
                url=url,
                status=200,
                body=b'{"success": true}',
                content_type="application/json"
            )
        
        # Apply the monkeypatch for session.request
        monkeypatch.setattr(ClientSession, "request", mock_request)
        
        # Create request and context
        request = PipelineRequest(source="https://example.com/api")
        context = PipelineContext({})
        
        # Execute the stage
        response = await http_stage.acquire_data(request, context)
        
        # Verify successful response
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
    
    @pytest.mark.asyncio
    async def test_different_content_types(self, http_stage, monkeypatch):
        """Test handling of different content types."""
        # Test cases for different content types
        content_types = [
            # (content_type, body, expected_data_key)
            ("application/json", b'{"key": "value"}', "json"),
            ("text/html", b"<html><body>Test</body></html>", "html"),
            ("text/plain", b"Plain text content", "text"),
            ("image/jpeg", b"binary-image-data", "binary"),
            ("application/pdf", b"binary-pdf-data", "binary")
        ]
        
        for content_type, body, expected_data_key in content_types:
            # Mock the session's request method for this content type
            async def mock_request(*args, **kwargs):
                headers = {"Content-Type": content_type}
                return MockResponse(
                    url="https://example.com/api",
                    status=200,
                    headers=headers,
                    body=body,
                    content_type=content_type
                )
            
            # Apply the monkeypatch for session.request
            monkeypatch.setattr(ClientSession, "request", mock_request)
            
            # Create request and context
            request = PipelineRequest(source="https://example.com/api")
            context = PipelineContext({})
            
            # Execute the stage
            response = await http_stage.acquire_data(request, context)
            
            # Verify the response contains the expected data key
            assert response is not None
            assert response.status == ResponseStatus.SUCCESS
            assert response.data is not None
            assert expected_data_key in response.data, f"Data key '{expected_data_key}' not found for content type '{content_type}'"
    
    @pytest.mark.asyncio
    async def test_error_responses(self, http_stage, monkeypatch):
        """Test handling of various HTTP error responses."""
        # Test cases for different HTTP errors
        error_cases = [
            # (status, expected_response_status)
            (400, ResponseStatus.ERROR),
            (404, ResponseStatus.NOT_FOUND),
            (500, ResponseStatus.SERVER_ERROR),
            (503, ResponseStatus.SERVER_ERROR)
        ]
        
        for status, expected_response_status in error_cases:
            # Mock the session's request method for this error
            async def mock_request(*args, **kwargs):
                return MockResponse(
                    url="https://example.com/api",
                    status=status,
                    body=b"Error occurred",
                    content_type="text/plain"
                )
            
            # Apply the monkeypatch for session.request
            monkeypatch.setattr(ClientSession, "request", mock_request)
            
            # Create request and context
            request = PipelineRequest(source="https://example.com/api")
            context = PipelineContext({})
            
            # Execute the stage
            response = await http_stage.acquire_data(request, context)
            
            # Verify the response has the expected error status
            assert response is not None
            assert response.status == expected_response_status, f"Expected {expected_response_status} for HTTP {status}, got {response.status}"
            assert response.error_message is not None
    
    @pytest.mark.asyncio
    async def test_client_errors(self, http_stage, monkeypatch):
        """Test handling of client-side errors."""
        # Mock the session's request method to raise a client error
        async def mock_request(*args, **kwargs):
            raise aiohttp.ClientError("Connection error")
        
        # Apply the monkeypatch for session.request
        monkeypatch.setattr(ClientSession, "request", mock_request)
        
        # Create request and context
        request = PipelineRequest(source="https://example.com/api")
        context = PipelineContext({})
        
        # Execute the stage
        response = await http_stage.acquire_data(request, context)
        
        # Verify the response indicates client error
        assert response is not None
        assert response.status == ResponseStatus.ERROR
        assert "connection error" in response.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_validate_source_config(self, http_stage):
        """Test configuration validation."""
        # Valid configuration should pass validation
        assert await http_stage.validate_source_config() is True
        
        # Test invalid method
        http_stage.method = "INVALID_METHOD"
        assert await http_stage.validate_source_config() is False
        
        # Reset to valid method
        http_stage.method = "GET"
        
        # Test missing URL
        http_stage.url_template = ""
        http_stage.config["url"] = ""
        assert await http_stage.validate_source_config() is False
    
    @pytest.mark.asyncio
    async def test_shutdown(self, http_stage):
        """Test resource cleanup on shutdown."""
        # Create a session manually
        http_stage._session = ClientSession()
        
        # Ensure session is open
        assert not http_stage._session.closed
        
        # Call shutdown
        await http_stage.shutdown()
        
        # Verify session is closed
        assert http_stage._session.closed