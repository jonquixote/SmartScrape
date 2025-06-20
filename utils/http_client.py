#!/usr/bin/env python3
"""
Optimized HTTP Client for SmartScrape

This module provides an optimized HTTP client with connection pooling,
timeout management, and resource optimization for web scraping operations.
"""

import aiohttp
import asyncio
import logging
from typing import Dict, Optional, Any
from aiohttp import TCPConnector, ClientTimeout, ClientSession
from contextlib import asynccontextmanager
import random
import time

logger = logging.getLogger(__name__)

# User agent rotation for better scraping success
USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0'
]

class OptimizedHTTPClient:
    """High-performance HTTP client optimized for scraping operations"""
    
    def __init__(self, 
                 max_connections: int = 100,
                 max_connections_per_host: int = 30,
                 dns_cache_ttl: int = 300,
                 keepalive_timeout: int = 30,
                 total_timeout: int = 30,
                 connect_timeout: int = 10,
                 read_timeout: int = 10):
        """
        Initialize optimized HTTP client
        
        Args:
            max_connections: Total connection limit across all hosts
            max_connections_per_host: Maximum connections per single host
            dns_cache_ttl: DNS cache TTL in seconds
            keepalive_timeout: Keep-alive timeout in seconds
            total_timeout: Total request timeout in seconds
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
        """
        
        # Configure connection pool
        self.connector = TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections_per_host,
            ttl_dns_cache=dns_cache_ttl,
            use_dns_cache=True,
            keepalive_timeout=keepalive_timeout,
            enable_cleanup_closed=True,
            force_close=False  # Reuse connections
        )
        
        # Configure timeouts
        self.timeout = ClientTimeout(
            total=total_timeout,
            connect=connect_timeout,
            sock_read=read_timeout
        )
        
        self.session: Optional[ClientSession] = None
        self._session_created = False
        
        logger.info(f"OptimizedHTTPClient initialized: max_conn={max_connections}, per_host={max_connections_per_host}")
    
    async def __aenter__(self) -> 'OptimizedHTTPClient':
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize the HTTP session"""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=self.timeout,
                headers={
                    'User-Agent': random.choice(USER_AGENTS),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
            )
            self._session_created = True
            logger.info("HTTP session initialized")
    
    async def close(self):
        """Close the HTTP session and cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("HTTP session closed")
        
        if self.connector and not self.connector.closed:
            await self.connector.close()
            logger.info("HTTP connector closed")
    
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """
        Perform GET request with optimization
        
        Args:
            url: Target URL
            **kwargs: Additional request parameters
            
        Returns:
            aiohttp.ClientResponse object
        """
        if not self.session:
            await self.initialize()
        
        # Add random user agent rotation for this request
        headers = kwargs.get('headers', {})
        if 'User-Agent' not in headers:
            headers['User-Agent'] = random.choice(USER_AGENTS)
            kwargs['headers'] = headers
        
        return await self.session.get(url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """
        Perform POST request with optimization
        
        Args:
            url: Target URL
            **kwargs: Additional request parameters
            
        Returns:
            aiohttp.ClientResponse object  
        """
        if not self.session:
            await self.initialize()
        
        # Add random user agent rotation for this request
        headers = kwargs.get('headers', {})
        if 'User-Agent' not in headers:
            headers['User-Agent'] = random.choice(USER_AGENTS)
            kwargs['headers'] = headers
        
        return await self.session.post(url, **kwargs)
    
    async def fetch_with_retry(self, url: str, max_retries: int = 3, backoff_factor: float = 1.0, **kwargs) -> Dict[str, Any]:
        """
        Fetch URL with retry logic and error handling
        
        Args:
            url: Target URL
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
            **kwargs: Additional request parameters
            
        Returns:
            Dict containing response data and metadata
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                
                async with await self.get(url, **kwargs) as response:
                    content = await response.text()
                    response_time = time.time() - start_time
                    
                    return {
                        'success': True,
                        'status_code': response.status,
                        'content': content,
                        'headers': dict(response.headers),
                        'url': str(response.url),
                        'response_time': response_time,
                        'attempt': attempt + 1,
                        'content_length': len(content)
                    }
                    
            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
                
            except aiohttp.ClientError as e:
                last_exception = e
                logger.warning(f"Client error on attempt {attempt + 1} for {url}: {e}")
                
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error on attempt {attempt + 1} for {url}: {e}")
            
            # Exponential backoff before retry
            if attempt < max_retries:
                wait_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Retrying {url} in {wait_time:.1f}s (attempt {attempt + 2}/{max_retries + 1})")
                await asyncio.sleep(wait_time)
        
        # All retries failed
        return {
            'success': False,
            'error': str(last_exception),
            'status_code': 0,
            'content': '',
            'headers': {},
            'url': url,
            'response_time': 0.0,
            'attempt': max_retries + 1,
            'content_length': 0
        }
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        if not self.connector:
            return {'error': 'Connector not initialized'}
        
        return {
            'total_connections': len(self.connector._conns),
            'available_connections': sum(len(conns) for conns in self.connector._conns.values()),
            'connector_limit': self.connector.limit,
            'connector_limit_per_host': self.connector.limit_per_host,
            'dns_cache_size': len(self.connector._dns_cache) if hasattr(self.connector, '_dns_cache') else 0,
            'closed': self.connector.closed
        }

# Global HTTP client instance
_global_client: Optional[OptimizedHTTPClient] = None

@asynccontextmanager
async def get_http_client():
    """Get a shared HTTP client instance"""
    global _global_client
    
    if _global_client is None:
        _global_client = OptimizedHTTPClient()
    
    if not _global_client.session or _global_client.session.closed:
        await _global_client.initialize()
    
    try:
        yield _global_client
    finally:
        # Don't close the global client here - it will be reused
        pass

async def cleanup_global_client():
    """Cleanup the global HTTP client"""
    global _global_client
    
    if _global_client:
        await _global_client.close()
        _global_client = None
