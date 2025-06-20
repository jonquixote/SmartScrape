"""
Resource Management Examples for SmartScrape

This module provides practical examples of how to use the resource management
components in SmartScrape, including SessionManager, RateLimiter, and ProxyManager.
"""

import time
import random
import logging
import requests
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('resource_management_examples')

# -----------------------------------------------------------------------------
# SessionManager Examples
# -----------------------------------------------------------------------------

def session_manager_basic_usage():
    """Demonstrate basic SessionManager usage."""
    from core.service_registry import ServiceRegistry
    
    # Get the SessionManager from the service registry
    registry = ServiceRegistry()
    session_manager = registry.get_service("session_manager")
    
    # Example domains
    domains = ["example.com", "httpbin.org", "api.github.com"]
    
    # Get sessions for different domains
    for domain in domains:
        logger.info(f"Getting session for {domain}")
        session = session_manager.get_session(domain)
        
        # Use the session to make a request
        url = f"https://{domain}"
        try:
            response = session.get(url, timeout=10)
            logger.info(f"Request to {url} returned status code {response.status_code}")
        except Exception as e:
            logger.error(f"Request to {url} failed: {str(e)}")
    
    # Cleanup
    session_manager.shutdown()
    logger.info("SessionManager example completed")


def session_manager_browser_example():
    """Demonstrate SessionManager with browser automation."""
    from core.service_registry import ServiceRegistry
    import asyncio
    
    # Get the SessionManager from the service registry
    registry = ServiceRegistry()
    session_manager = registry.get_service("session_manager")
    
    # Define the async function for browser operations
    async def browser_example():
        # Get a browser page for a specific domain
        domain = "example.com"
        logger.info(f"Getting browser page for {domain}")
        
        page = await session_manager.get_browser_page(domain)
        
        # Navigate to the URL
        url = f"https://{domain}"
        logger.info(f"Navigating to {url}")
        await page.goto(url)
        
        # Get the page title
        title = await page.title()
        logger.info(f"Page title: {title}")
        
        # Take a screenshot
        screenshot_path = "example_screenshot.png"
        await page.screenshot(path=screenshot_path)
        logger.info(f"Screenshot saved to {screenshot_path}")
        
        # Close the page
        await page.close()
        logger.info("Browser page closed")
    
    # Run the async example
    asyncio.run(browser_example())
    
    # Cleanup
    session_manager.shutdown()
    logger.info("Browser session example completed")


def session_manager_with_cookies():
    """Demonstrate cookie management with SessionManager."""
    from core.service_registry import ServiceRegistry
    
    # Get the SessionManager from the service registry
    registry = ServiceRegistry()
    session_manager = registry.get_service("session_manager")
    
    # Example domain
    domain = "httpbin.org"
    
    # Get a session for the domain
    logger.info(f"Getting session for {domain}")
    session = session_manager.get_session(domain)
    
    # Set a cookie manually
    session.cookies.set("test_cookie", "cookie_value", domain=domain)
    
    # Make a request to see the cookies
    url = f"https://{domain}/cookies"
    try:
        response = session.get(url)
        logger.info(f"Cookies sent to {url}: {response.json()}")
    except Exception as e:
        logger.error(f"Request to {url} failed: {str(e)}")
    
    # Make another request that sets a cookie
    url = f"https://{domain}/cookies/set/session_cookie/another_value"
    try:
        response = session.get(url, allow_redirects=False)
        logger.info(f"Server set cookie with status code {response.status_code}")
    except Exception as e:
        logger.error(f"Request to {url} failed: {str(e)}")
    
    # Check all cookies in the session
    logger.info(f"All cookies in session: {session.cookies.get_dict()}")
    
    # Cleanup
    session_manager.shutdown()
    logger.info("Cookie management example completed")


# -----------------------------------------------------------------------------
# RateLimiter Examples
# -----------------------------------------------------------------------------

def rate_limiter_basic_usage():
    """Demonstrate basic RateLimiter usage."""
    from core.service_registry import ServiceRegistry
    
    # Get the RateLimiter from the service registry
    registry = ServiceRegistry()
    rate_limiter = registry.get_service("rate_limiter")
    
    # Example domain
    domain = "api.example.com"
    
    # Make multiple requests respecting rate limits
    for i in range(5):
        logger.info(f"Request {i+1} to {domain}")
        
        # Wait if needed according to rate limits
        waited = rate_limiter.wait_if_needed(domain)
        if waited:
            logger.info(f"Waited to respect rate limits for {domain}")
        
        # Simulate a request
        logger.info(f"Making request to {domain}")
        
        # Report success (in a real scenario, this would be called after the actual request)
        rate_limiter.report_success(domain)
    
    # Cleanup
    rate_limiter.shutdown()
    logger.info("RateLimiter basic example completed")


def rate_limiter_adaptive_example():
    """Demonstrate adaptive rate limiting."""
    from core.service_registry import ServiceRegistry
    
    # Get the RateLimiter from the service registry
    registry = ServiceRegistry()
    rate_limiter = registry.get_service("rate_limiter")
    
    # Example domain
    domain = "api.example.com"
    
    # Make some successful requests
    for i in range(3):
        logger.info(f"Successful request {i+1} to {domain}")
        rate_limiter.wait_if_needed(domain)
        rate_limiter.report_success(domain)
    
    # Simulate a rate limit response
    logger.info(f"Received a 429 Too Many Requests from {domain}")
    rate_limiter.report_rate_limited(domain)
    
    # Make more requests - should now be more conservative
    for i in range(3):
        logger.info(f"Post-rate-limit request {i+1} to {domain}")
        waited = rate_limiter.wait_if_needed(domain)
        if waited:
            logger.info(f"Waited to respect adjusted rate limits for {domain}")
        rate_limiter.report_success(domain)
    
    # Cleanup
    rate_limiter.shutdown()
    logger.info("Adaptive rate limiting example completed")


def rate_limiter_with_domains():
    """Demonstrate domain-specific rate limiting."""
    from core.service_registry import ServiceRegistry
    
    # Get the RateLimiter from the service registry
    registry = ServiceRegistry()
    rate_limiter = registry.get_service("rate_limiter")
    
    # Example domains with different characteristics
    domains = {
        "api.highrate.com": {"requests_per_minute": 120},
        "api.lowrate.com": {"requests_per_minute": 20},
        "api.average.com": {"requests_per_minute": 60}
    }
    
    # Update limits for specific domains
    for domain, limits in domains.items():
        rate_limiter.update_limits(domain, limits)
        logger.info(f"Updated rate limits for {domain}: {limits}")
    
    # Make requests to each domain
    for domain in domains:
        for i in range(3):
            logger.info(f"Request {i+1} to {domain}")
            waited = rate_limiter.wait_if_needed(domain)
            if waited:
                logger.info(f"Waited to respect rate limits for {domain}")
            rate_limiter.report_success(domain)
    
    # Cleanup
    rate_limiter.shutdown()
    logger.info("Domain-specific rate limiting example completed")


# -----------------------------------------------------------------------------
# ProxyManager Examples
# -----------------------------------------------------------------------------

def proxy_manager_basic_usage():
    """Demonstrate basic ProxyManager usage."""
    from core.service_registry import ServiceRegistry
    
    # Get the ProxyManager from the service registry
    registry = ServiceRegistry()
    proxy_manager = registry.get_service("proxy_manager")
    
    # Example domain
    domain = "example.com"
    
    # Get a proxy for the domain
    logger.info(f"Getting proxy for {domain}")
    proxy = proxy_manager.get_proxy(domain)
    
    if proxy:
        logger.info(f"Selected proxy: {proxy}")
        
        # Create a requests session and configure it with the proxy
        session = requests.Session()
        session.proxies = proxy.as_dict()
        
        # Make a request through the proxy
        try:
            url = "https://httpbin.org/ip"
            logger.info(f"Making request to {url} through proxy")
            response = session.get(url, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Proxy request succeeded: {response.json()}")
                proxy_manager.report_success(proxy)
            else:
                logger.error(f"Proxy request returned status code {response.status_code}")
                proxy_manager.report_failure(proxy, error=f"Status code {response.status_code}")
        except Exception as e:
            logger.error(f"Proxy request failed: {str(e)}")
            proxy_manager.report_failure(proxy, error=str(e))
    else:
        logger.warning(f"No proxy available for {domain}")
    
    # Cleanup
    proxy_manager.shutdown()
    logger.info("ProxyManager basic example completed")


def proxy_manager_rotation_example():
    """Demonstrate proxy rotation."""
    from core.service_registry import ServiceRegistry
    
    # Get the ProxyManager from the service registry
    registry = ServiceRegistry()
    proxy_manager = registry.get_service("proxy_manager")
    
    # Example domain
    domain = "example.com"
    
    # Make multiple requests with different proxies
    for i in range(3):
        logger.info(f"Request {i+1}")
        
        # Get a proxy for the domain
        proxy = proxy_manager.get_proxy(domain)
        
        if proxy:
            logger.info(f"Selected proxy: {proxy}")
            
            # Create a requests session and configure it with the proxy
            session = requests.Session()
            session.proxies = proxy.as_dict()
            
            # Make a request through the proxy
            try:
                url = "https://httpbin.org/ip"
                logger.info(f"Making request to {url} through proxy")
                response = session.get(url, timeout=10)
                
                if response.status_code == 200:
                    logger.info(f"Proxy request succeeded: {response.json()}")
                    proxy_manager.report_success(proxy)
                else:
                    logger.error(f"Proxy request returned status code {response.status_code}")
                    proxy_manager.report_failure(proxy, error=f"Status code {response.status_code}")
            except Exception as e:
                logger.error(f"Proxy request failed: {str(e)}")
                proxy_manager.report_failure(proxy, error=str(e))
        else:
            logger.warning(f"No proxy available for {domain}")
    
    # Cleanup
    proxy_manager.shutdown()
    logger.info("Proxy rotation example completed")


def proxy_manager_with_tags():
    """Demonstrate proxy selection with tags."""
    from core.service_registry import ServiceRegistry
    
    # Get the ProxyManager from the service registry
    registry = ServiceRegistry()
    proxy_manager = registry.get_service("proxy_manager")
    
    # Example domain
    domain = "example.com"
    
    # Get proxies with specific tags
    tags_to_try = [
        ["residential"],
        ["datacenter"],
        ["residential", "us"],
        []  # No tags (any proxy)
    ]
    
    for tags in tags_to_try:
        tag_str = ", ".join(tags) if tags else "any"
        logger.info(f"Getting proxy for {domain} with tags: {tag_str}")
        
        proxy = proxy_manager.get_proxy(domain, tags=tags)
        
        if proxy:
            logger.info(f"Selected proxy: {proxy} with tags: {proxy.tags}")
        else:
            logger.warning(f"No proxy available for {domain} with tags: {tag_str}")
    
    # Cleanup
    proxy_manager.shutdown()
    logger.info("Tagged proxy selection example completed")


# -----------------------------------------------------------------------------
# Integrated Examples
# -----------------------------------------------------------------------------

def integrated_request_flow():
    """Demonstrate a complete request flow with all resources."""
    from core.service_registry import ServiceRegistry
    from urllib.parse import urlparse
    
    # Get services from registry
    registry = ServiceRegistry()
    session_manager = registry.get_service("session_manager")
    rate_limiter = registry.get_service("rate_limiter")
    proxy_manager = registry.get_service("proxy_manager")
    
    # Example URL to fetch
    url = "https://httpbin.org/get"
    
    # Extract domain from URL
    domain = urlparse(url).netloc
    
    logger.info(f"Starting integrated request to {url}")
    
    # Get a proxy
    proxy = proxy_manager.get_proxy(domain)
    if proxy:
        logger.info(f"Using proxy: {proxy}")
    else:
        logger.info("No proxy available, proceeding with direct connection")
    
    # Respect rate limits
    waited = rate_limiter.wait_if_needed(domain)
    if waited:
        logger.info(f"Waited to respect rate limits for {domain}")
    
    try:
        # Get a session
        session = session_manager.get_session(domain)
        
        # Configure session with proxy if available
        if proxy:
            session.proxies = proxy.as_dict()
        
        # Make the request
        logger.info(f"Making request to {url}")
        response = session.get(url, timeout=10)
        
        # Process the response
        if response.status_code == 200:
            logger.info(f"Request succeeded: {response.json()}")
            
            # Report success
            rate_limiter.report_success(domain)
            if proxy:
                proxy_manager.report_success(proxy)
        else:
            logger.error(f"Request returned status code {response.status_code}")
            
            # Report failure
            if response.status_code == 429:
                rate_limiter.report_rate_limited(domain)
            
            if proxy:
                proxy_manager.report_failure(proxy, error=f"Status code {response.status_code}")
    
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        
        # Report proxy failure if applicable
        if proxy:
            proxy_manager.report_failure(proxy, error=str(e))
    
    # Cleanup
    session_manager.shutdown()
    rate_limiter.shutdown()
    proxy_manager.shutdown()
    
    logger.info("Integrated request flow example completed")


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main():
    """Run examples."""
    logger.info("Starting resource management examples")
    
    examples = [
        # Basic examples
        session_manager_basic_usage,
        rate_limiter_basic_usage,
        proxy_manager_basic_usage,
        
        # Advanced examples
        session_manager_with_cookies,
        rate_limiter_adaptive_example,
        rate_limiter_with_domains,
        proxy_manager_rotation_example,
        proxy_manager_with_tags,
        
        # Integration example
        integrated_request_flow
    ]
    
    # Select which examples to run
    examples_to_run = [
        session_manager_basic_usage,
        rate_limiter_basic_usage,
        integrated_request_flow
    ]
    
    # Run selected examples
    for example in examples_to_run:
        logger.info(f"\n{'=' * 80}\nRunning example: {example.__name__}\n{'=' * 80}")
        try:
            example()
        except Exception as e:
            logger.error(f"Example {example.__name__} failed: {str(e)}")
    
    logger.info("All examples completed")


if __name__ == "__main__":
    main()