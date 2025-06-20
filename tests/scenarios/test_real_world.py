"""
Real-world scenario tests for SmartScrape.

These tests validate the system's performance against real-world scenarios,
including rate limiting, CAPTCHAs, IP blocking, session tracking, and JavaScript
requirements. The tests measure success rates, resource efficiency, error recovery,
data quality, and overall resilience.
"""

import logging
import pytest
import time
import requests
import random
import threading
import json
import os
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from core.service_registry import ServiceRegistry
from core.session_manager import SessionManager
from core.proxy_manager import ProxyManager 
from core.rate_limiter import RateLimiter
from core.retry_manager import RetryManager
from core.circuit_breaker import CircuitBreakerManager, OpenCircuitError
from core.error_classifier import ErrorClassifier, ErrorCategory, ErrorSeverity

# Configure logging for tests
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class TestRealWorldScenarios:
    """Test the system against realistic scenarios."""
    
    @pytest.fixture
    def service_registry(self):
        """Set up service registry with all required services."""
        registry = ServiceRegistry()
        
        # Initialize and register all necessary services
        session_manager = SessionManager()
        session_manager.initialize()
        registry.register_service(session_manager)
        
        proxy_manager = ProxyManager()
        proxy_config = {
            'proxies': [
                {'url': f'http://test-proxy-{i}.example.com', 'type': 'http'} 
                for i in range(10)
            ]
        }
        proxy_manager.initialize(proxy_config)
        registry.register_service(proxy_manager)
        
        rate_limiter = RateLimiter()
        rate_config = {
            'default_limits': {
                'requests_per_minute': 30,
                'requests_per_hour': 300,
                'concurrent_requests': 5
            },
            'domain_limits': {
                'httpbin.org': {
                    'requests_per_minute': 10,
                    'requests_per_hour': 60
                }
            }
        }
        rate_limiter.initialize(rate_config)
        registry.register_service(rate_limiter)
        
        error_classifier = ErrorClassifier()
        error_classifier.initialize()
        registry.register_service(error_classifier)
        
        retry_manager = RetryManager()
        retry_config = {
            'default_policy': {
                'max_attempts': 3,
                'backoff_factor': 0.5,
                'jitter': True
            }
        }
        retry_manager.initialize(retry_config)
        registry.register_service(retry_manager)
        
        circuit_breaker = CircuitBreakerManager()
        circuit_config = {
            'default_settings': {
                'failure_threshold': 5,
                'reset_timeout': 30,
                'half_open_max': 1
            }
        }
        circuit_breaker.initialize(circuit_config)
        registry.register_service(circuit_breaker)
        
        return registry
    
    def test_rate_limited_sites(self, service_registry, monkeypatch):
        """Test performance against sites that employ rate limiting."""
        # Get services
        session_manager = service_registry.get_service('session_manager')
        rate_limiter = service_registry.get_service('rate_limiter')
        retry_manager = service_registry.get_service('retry_manager')
        error_classifier = service_registry.get_service('error_classifier')
        
        # Test domain
        domain = 'rate-limited.example.com'
        base_url = f'https://{domain}/api'
        
        # Configure mock session with rate limiting behavior
        # First 5 requests succeed, then 429s for a while, then succeeds again
        request_count = {'count': 0}
        
        def mock_get_session(domain, force_new=False):
            session = MagicMock()
            
            def rate_limited_get(url, **kwargs):
                request_count['count'] += 1
                
                # Pattern: 5 successes, then 3 rate limits, then success
                pattern_position = (request_count['count'] - 1) % 9
                
                if 5 <= pattern_position < 8:
                    # Rate limited
                    mock_response = MagicMock()
                    mock_response.status_code = 429
                    mock_response.text = "Too Many Requests"
                    mock_response.headers = {
                        'Retry-After': '5',  # Suggests 5 second wait
                        'X-RateLimit-Limit': '5',
                        'X-RateLimit-Remaining': '0',
                        'X-RateLimit-Reset': str(int(time.time()) + 5)
                    }
                    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                        "429 Too Many Requests", response=mock_response
                    )
                    return mock_response
                else:
                    # Success
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.text = f"Response {request_count['count']}"
                    mock_response.json.return_value = {"success": True, "request_id": request_count['count']}
                    return mock_response
                
            session.get = rate_limited_get
            return session
            
        monkeypatch.setattr(session_manager, 'get_session', mock_get_session)
        
        # Test adaptive rate limiting behavior
        metrics = {'requests': 0, 'success': 0, 'rate_limited': 0, 'retry_success': 0}
        
        # Track rate limit adjustments
        rate_limit_adjustments = []
        original_report_rate_limited = rate_limiter.report_rate_limited
        
        def mock_report_rate_limited(domain):
            rate_limit_adjustments.append(domain)
            original_report_rate_limited(domain)
            
        monkeypatch.setattr(rate_limiter, 'report_rate_limited', mock_report_rate_limited)
        
        # Function to make request with retry
        def make_request_with_retry(endpoint):
            endpoint_url = f"{base_url}/{endpoint}"
            metrics['requests'] += 1
            
            # Wait if needed according to rate limiter
            rate_limiter.wait_if_needed(domain)
            
            @retry_manager.retry(domain, max_attempts=5)
            def fetch():
                try:
                    response = session_manager.get_session(domain).get(endpoint_url, timeout=5)
                    response.raise_for_status()
                    return response
                except requests.exceptions.HTTPError as e:
                    if hasattr(e, 'response') and e.response.status_code == 429:
                        # Report rate limiting to adjust limits
                        metrics['rate_limited'] += 1
                        rate_limiter.report_rate_limited(domain)
                        
                        # Extract retry-after if available
                        retry_after = e.response.headers.get('Retry-After')
                        if retry_after:
                            logger.info(f"Rate limited, suggested wait: {retry_after}s")
                        
                    # Classify error
                    error_info = error_classifier.classify_exception(
                        e, {'url': endpoint_url, 'domain': domain}
                    )
                    
                    # Propagate for retry handling
                    raise
                
            try:
                response = fetch()
                metrics['success'] += 1
                return response
            except Exception as e:
                logger.error(f"Failed to fetch {endpoint_url}: {str(e)}")
                return None
        
        # Make multiple requests to the rate-limited API
        endpoints = [f"endpoint{i}" for i in range(20)]
        
        start_time = time.time()
        results = []
        
        for endpoint in endpoints:
            response = make_request_with_retry(endpoint)
            results.append(response is not None)
            
        total_time = time.time() - start_time
        success_rate = sum(results) / len(results)
        
        logger.info(f"Rate limiting test results:")
        logger.info(f"Total requests: {metrics['requests']}")
        logger.info(f"Successful requests: {metrics['success']}")
        logger.info(f"Rate limited responses: {metrics['rate_limited']}")
        logger.info(f"Success rate: {success_rate:.2%}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Rate limit adjustments: {len(rate_limit_adjustments)}")
        
        # Verify that rate limiting worked correctly
        assert success_rate > 0.8, "Success rate should be high despite rate limiting"
        assert len(rate_limit_adjustments) > 0, "Rate limits should have been adjusted"
    
    def test_captcha_handling(self, service_registry, monkeypatch):
        """Test the system's ability to detect and handle CAPTCHAs."""
        # Get services
        session_manager = service_registry.get_service('session_manager')
        proxy_manager = service_registry.get_service('proxy_manager')
        error_classifier = service_registry.get_service('error_classifier')
        
        # Test domain
        domain = 'captcha.example.com'
        base_url = f'https://{domain}'
        
        # Configure CAPTCHA detection in ErrorClassifier
        captcha_patterns = [
            r'captcha',
            r'robot check',
            r'human verification',
            r'are you a robot',
            r'prove you\'re human'
        ]
        
        # Simulate site that sometimes shows CAPTCHAs
        captcha_scenarios = [
            # (path, should_show_captcha, session_based)
            ('/page1', False, False),  # Normal page
            ('/page2', True, False),   # Always CAPTCHA
            ('/page3', False, True),   # CAPTCHA if same session used too much
            ('/page4', True, True),    # CAPTCHA that can be bypassed with new session
        ]
        
        # Track sessions that have seen too many pages
        session_requests = {}
        
        def mock_get_session(domain, force_new=False):
            session = MagicMock()
            session_id = f"session_{random.randint(1000, 9999)}" if force_new else "session_default"
            
            # Initialize session counter if needed
            if session_id not in session_requests:
                session_requests[session_id] = 0
                
            def captcha_aware_get(url, **kwargs):
                session_requests[session_id] += 1
                request_count = session_requests[session_id]
                
                path = url.split(domain)[1] if domain in url else url
                
                # Determine if this request should get a CAPTCHA
                show_captcha = False
                
                for test_path, always_captcha, session_based in captcha_scenarios:
                    if path.startswith(test_path):
                        if always_captcha:
                            show_captcha = True
                        elif session_based and request_count > 2:  # Show CAPTCHA after 2 requests
                            show_captcha = True
                        break
                
                if show_captcha:
                    # Return CAPTCHA page
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.text = f"""
                    <html>
                    <head><title>CAPTCHA Required</title></head>
                    <body>
                        <h1>Please complete this CAPTCHA to continue</h1>
                        <form>
                            <input type="text" name="captcha" placeholder="Enter CAPTCHA">
                            <button type="submit">Submit</button>
                        </form>
                    </body>
                    </html>
                    """
                    return mock_response
                else:
                    # Return normal page
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.text = f"Content from {url}"
                    return mock_response
                
            session.get = captcha_aware_get
            return session
            
        monkeypatch.setattr(session_manager, 'get_session', mock_get_session)
        
        # Mock CAPTCHA detection
        def mock_check_for_captcha(self, response):
            return 'captcha' in response.text.lower()
            
        monkeypatch.setattr(error_classifier.__class__, '_check_for_captcha', mock_check_for_captcha)
        
        # Function to scrape with CAPTCHA awareness
        def scrape_with_captcha_handling(url):
            max_attempts = 3
            attempt = 0
            force_new_session = False
            
            while attempt < max_attempts:
                try:
                    # Get session (new one if previous had CAPTCHA)
                    session = session_manager.get_session(domain, force_new=force_new_session)
                    response = session.get(url, timeout=5)
                    
                    # Check for CAPTCHA
                    if error_classifier._check_for_captcha(response):
                        logger.warning(f"CAPTCHA detected at {url} on attempt {attempt + 1}")
                        
                        # Strategy 1: Try a new session
                        force_new_session = True
                        attempt += 1
                        continue
                    
                    # Success - no CAPTCHA
                    return {
                        'url': url,
                        'success': True,
                        'content': response.text[:50] + "..."  # Truncated content
                    }
                    
                except Exception as e:
                    logger.error(f"Error scraping {url}: {str(e)}")
                    attempt += 1
            
            # All attempts failed
            return {
                'url': url,
                'success': False,
                'error': 'Failed to bypass CAPTCHA after multiple attempts'
            }
        
        # Test all scenarios
        test_urls = [f"{base_url}{path}" for path, _, _ in captcha_scenarios]
        results = []
        
        for url in test_urls:
            result = scrape_with_captcha_handling(url)
            results.append(result)
            logger.info(f"Result for {url}: {'Success' if result['success'] else 'Failure'}")
        
        # Verify results
        success_rate = sum(1 for r in results if r['success']) / len(results)
        logger.info(f"CAPTCHA handling success rate: {success_rate:.2%}")
        
        # Page 1 should always succeed
        assert results[0]['success'], "Page without CAPTCHA should succeed"
        
        # Page 2 will probably fail as it always has CAPTCHA
        # We don't assert this as some advanced CAPTCHA handling might be implemented
        
        # Page 3 and 4 should have some success with session rotation
        assert any(r['success'] for r in results[2:]), "Session rotation should bypass some CAPTCHAs"
    
    def test_ip_blocking_scenarios(self, service_registry, monkeypatch):
        """Test handling of sites that implement IP blocking."""
        # Get services
        session_manager = service_registry.get_service('session_manager')
        proxy_manager = service_registry.get_service('proxy_manager')
        error_classifier = service_registry.get_service('error_classifier')
        
        # Test domain
        domain = 'ip-blocking.example.com'
        base_url = f'https://{domain}'
        
        # Track IPs (proxies) that get blocked
        blocked_ips = set()
        access_counts = {}
        
        # Configure proxy thresholds
        block_threshold = 5  # Block IP after this many requests
        
        # Mock proxy behavior
        def mock_get_proxy(domain):
            # Select proxy based on rotation policy
            proxy_index = random.randint(0, 9)  # Simulate random selection
            proxy_url = f'http://test-proxy-{proxy_index}.example.com'
            return {'url': proxy_url, 'type': 'http'}
            
        monkeypatch.setattr(proxy_manager, 'get_proxy', mock_get_proxy)
        
        # Mock session that tracks IP blocking
        def mock_get_session(domain, force_new=False):
            session = MagicMock()
            
            def ip_aware_get(url, proxies=None, **kwargs):
                ip = proxies.get('http', 'default-ip') if proxies else 'default-ip'
                
                # Initialize access counter for this IP
                if ip not in access_counts:
                    access_counts[ip] = 0
                
                access_counts[ip] += 1
                
                # Check if IP is blocked
                if ip in blocked_ips:
                    mock_response = MagicMock()
                    mock_response.status_code = 403
                    mock_response.text = "Your IP address has been blocked."
                    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                        "403 Forbidden: IP Blocked", response=mock_response
                    )
                    return mock_response
                
                # Check if IP should be blocked due to too many requests
                if access_counts[ip] >= block_threshold:
                    blocked_ips.add(ip)
                    logger.info(f"IP {ip} blocked after {access_counts[ip]} requests")
                    
                    mock_response = MagicMock()
                    mock_response.status_code = 403
                    mock_response.text = "Your IP address has been blocked due to suspicious activity."
                    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                        "403 Forbidden: IP Blocked", response=mock_response
                    )
                    return mock_response
                
                # Normal response
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = f"Content from {url} via {ip}"
                return mock_response
                
            session.get = ip_aware_get
            return session
            
        monkeypatch.setattr(session_manager, 'get_session', mock_get_session)
        
        # Define function to scrape with proxy rotation on IP blocking
        def scrape_with_proxy_rotation(path):
            url = f"{base_url}{path}"
            max_attempts = 8
            attempt = 0
            used_proxies = set()
            
            while attempt < max_attempts:
                try:
                    # Get a proxy (preferably new one)
                    proxy = proxy_manager.get_proxy(domain)
                    proxy_url = proxy['url']
                    
                    # Keep track of proxies we've tried
                    used_proxies.add(proxy_url)
                    
                    # Get a session
                    session = session_manager.get_session(domain)
                    
                    # Make the request
                    response = session.get(url, proxies={
                        'http': proxy_url,
                        'https': proxy_url
                    }, timeout=5)
                    
                    # If successful, return result
                    return {
                        'url': url,
                        'success': True,
                        'content': response.text,
                        'attempts': attempt + 1,
                        'unique_proxies': len(used_proxies)
                    }
                    
                except requests.exceptions.HTTPError as e:
                    # Check if it's an IP blocking error
                    if hasattr(e, 'response') and e.response.status_code == 403:
                        logger.warning(f"Proxy {proxy_url} blocked. Rotating proxy. Attempt {attempt + 1}")
                        
                        # Mark this proxy as failed
                        proxy_manager.mark_proxy_failed(proxy_url)
                        
                        # Try again with a new proxy
                        attempt += 1
                        continue
                    else:
                        # Other HTTP error
                        logger.error(f"HTTP error: {str(e)}")
                        attempt += 1
                        
                except Exception as e:
                    # Other error
                    logger.error(f"Error: {str(e)}")
                    attempt += 1
            
            # All attempts failed
            return {
                'url': url,
                'success': False,
                'error': f'Failed after {max_attempts} attempts with different proxies',
                'attempts': attempt,
                'unique_proxies': len(used_proxies)
            }
        
        # Test IP blocking with many requests
        test_paths = [f"/page{i}" for i in range(20)]
        results = []
        
        for path in test_paths:
            result = scrape_with_proxy_rotation(path)
            results.append(result)
            
            logger.info(f"Result for {path}: {'Success' if result['success'] else 'Failure'} "
                       f"after {result['attempts']} attempts using {result['unique_proxies']} proxies")
        
        # Calculate metrics
        success_rate = sum(1 for r in results if r['success']) / len(results)
        avg_attempts = sum(r['attempts'] for r in results) / len(results)
        avg_proxies = sum(r['unique_proxies'] for r in results) / len(results)
        
        logger.info(f"IP blocking test results:")
        logger.info(f"Success rate: {success_rate:.2%}")
        logger.info(f"Average attempts per request: {avg_attempts:.2f}")
        logger.info(f"Average unique proxies per request: {avg_proxies:.2f}")
        logger.info(f"Total IPs blocked: {len(blocked_ips)}")
        
        # Verify proxy rotation is working to overcome blocking
        assert success_rate > 0.8, "Success rate should be high with proxy rotation"
        assert avg_proxies > 1.0, "Should use multiple proxies on average"
        assert len(blocked_ips) > 0, "Some IPs should have been blocked"
    
    def test_session_tracking_sites(self, service_registry, monkeypatch):
        """Test against sites that track sessions with cookies and require consistent sessions."""
        # Get services
        session_manager = service_registry.get_service('session_manager')
        
        # Test domain
        domain = 'session-tracking.example.com'
        base_url = f'https://{domain}'
        
        # Simulate server-side session tracking
        active_sessions = {}
        
        def mock_get_session(domain, force_new=False):
            # Create real-like session object
            session = MagicMock()
            
            # Generate a unique session ID if forcing new session or if this is first call
            session_id = None
            if force_new or domain not in active_sessions:
                session_id = f"session_{random.randint(10000, 99999)}"
                active_sessions[domain] = {
                    'session_id': session_id,
                    'state': {},
                    'cookies': {'session_id': session_id}
                }
            else:
                session_id = active_sessions[domain]['session_id']
            
            # Add cookies to session
            session.cookies = MagicMock()
            session.cookies.get_dict.return_value = active_sessions[domain]['cookies']
            
            def session_aware_get(url, **kwargs):
                # Extract path from URL
                path = url.split(domain, 1)[1] if domain in url else url
                
                # Create mock response
                mock_response = MagicMock()
                mock_response.status_code = 200
                
                # Initial visit - set cookie
                if path == '/init':
                    active_sessions[domain]['state']['initialized'] = True
                    mock_response.text = "Session initialized"
                    return mock_response
                
                # Login page - store credentials
                elif path == '/login':
                    if 'initialized' not in active_sessions[domain]['state']:
                        # Must visit init first
                        mock_response.status_code = 403
                        mock_response.text = "Must initialize session first"
                        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                            "403 Forbidden: Session not initialized", response=mock_response
                        )
                        return mock_response
                    
                    active_sessions[domain]['state']['logged_in'] = True
                    mock_response.text = "Login successful"
                    return mock_response
                
                # Protected page - require login
                elif path == '/protected':
                    if not active_sessions[domain]['state'].get('logged_in', False):
                        mock_response.status_code = 401
                        mock_response.text = "Unauthorized: Please log in"
                        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                            "401 Unauthorized", response=mock_response
                        )
                        return mock_response
                    
                    mock_response.text = f"Protected content for session {session_id}"
                    return mock_response
                
                # Default - public page
                else:
                    mock_response.text = f"Public content at {path}"
                    return mock_response
                
            session.get = session_aware_get
            return session
            
        monkeypatch.setattr(session_manager, 'get_session', mock_get_session)
        
        # Function to simulate multi-step session flow
        def execute_session_workflow(preserve_session=True, follow_required_flow=True):
            results = {}
            success = True
            error = None
            
            try:
                # Get and maintain the same session
                session = session_manager.get_session(domain, force_new=True)
                
                # Step 1: Initialize session (always required first)
                init_url = f"{base_url}/init"
                response = session.get(init_url)
                results['init'] = response.text
                
                # Step 2: Login (required for protected content)
                login_url = f"{base_url}/login"
                
                # If we're testing a broken flow, try accessing protected content first
                if not follow_required_flow:
                    try:
                        protected_url = f"{base_url}/protected"
                        response = session.get(protected_url)
                        results['protected_before_login'] = response.text
                    except requests.exceptions.HTTPError as e:
                        results['protected_before_login_error'] = str(e)
                
                # Login
                response = session.get(login_url)
                results['login'] = response.text
                
                # Step 3: Access protected content
                protected_url = f"{base_url}/protected"
                
                # If testing not preserving session, get a new session
                if not preserve_session:
                    session = session_manager.get_session(domain, force_new=True)
                
                # Try to access protected content
                try:
                    response = session.get(protected_url)
                    results['protected'] = response.text
                except requests.exceptions.HTTPError as e:
                    results['protected_error'] = str(e)
                    success = False
                    error = str(e)
                
                # Step 4: Access public content
                public_url = f"{base_url}/public"
                response = session.get(public_url)
                results['public'] = response.text
                
            except Exception as e:
                success = False
                error = str(e)
            
            return {
                'success': success,
                'error': error,
                'results': results
            }
        
        # Test different session handling scenarios
        scenarios = [
            ("Correct flow with session preservation", True, True),
            ("Incorrect flow with session preservation", True, False),
            ("Correct flow without session preservation", False, True),
        ]
        
        results = {}
        
        for name, preserve_session, follow_flow in scenarios:
            result = execute_session_workflow(preserve_session, follow_flow)
            results[name] = result
            logger.info(f"Scenario '{name}': {'Success' if result['success'] else 'Failure'}")
            
        # Verify results
        assert results["Correct flow with session preservation"]['success'], \
            "Correct flow with session preservation should succeed"
        
        if not follow_flow:
            assert "protected_before_login_error" in results["Incorrect flow with session preservation"]['results'], \
                "Accessing protected content before login should fail"
        
        assert not results["Correct flow without session preservation"]['success'], \
            "Correct flow without session preservation should fail"
        assert "protected_error" in results["Correct flow without session preservation"]['results'], \
            "Access to protected content with new session should fail"
    
    def test_javascript_dependent_sites(self, service_registry, monkeypatch):
        """Test against sites that require JavaScript for content loading."""
        # Get services
        session_manager = service_registry.get_service('session_manager')
        
        # Test domain
        domain = 'js-required.example.com'
        base_url = f'https://{domain}'
        
        # Create a fake browser session functionality
        browser_available = True  # Toggle to simulate browser availability
        
        def mock_get_browser_session(domain):
            if not browser_available:
                raise RuntimeError("Browser automation not available")
                
            browser_session = MagicMock()
            
            def js_aware_get(url, **kwargs):
                # Create response for browser automation
                mock_response = MagicMock()
                mock_response.status_code = 200
                
                # Different response depending on URL
                path = url.split(domain, 1)[1] if domain in url else url
                
                if path == '/static':
                    # Static page without JS requirements
                    mock_response.text = """
                    <html><body>
                        <h1>Static Content</h1>
                        <p>This content is visible without JavaScript.</p>
                    </body></html>
                    """
                elif path == '/dynamic':
                    # Dynamic page that loads content via JS
                    mock_response.text = """
                    <html><body>
                        <h1>Dynamic Content</h1>
                        <div id="content">Loading...</div>
                        <script>
                            // This script would normally load content dynamically
                            document.getElementById('content').innerHTML = 
                                'This content was loaded via JavaScript.';
                        </script>
                    </body></html>
                    """
                    
                    # With a browser, we'd see the JS-loaded content
                    mock_response.rendered_text = """
                    <html><body>
                        <h1>Dynamic Content</h1>
                        <div id="content">This content was loaded via JavaScript.</div>
                        <script>
                            // Script already executed
                        </script>
                    </body></html>
                    """
                elif path == '/spa':
                    # Single page app requiring JS
                    mock_response.text = """
                    <html><body>
                        <div id="app">Loading application...</div>
                        <script>
                            // This script would initialize the SPA
                            document.getElementById('app').innerHTML = 
                                '<h1>SPA Loaded</h1><p>Single Page App content.</p>';
                        </script>
                    </body></html>
                    """
                    
                    # With a browser, we'd see the SPA content
                    mock_response.rendered_text = """
                    <html><body>
                        <div id="app"><h1>SPA Loaded</h1><p>Single Page App content.</p></div>
                        <script>
                            // Script already executed
                        </script>
                    </body></html>
                    """
                else:
                    # Unknown path
                    mock_response.text = f"<html><body>Unknown path: {path}</body></html>"
                    mock_response.rendered_text = mock_response.text
                
                return mock_response
                
            browser_session.get = js_aware_get
            return browser_session
            
        # Function to get regular HTTP session
        def mock_get_http_session(domain):
            http_session = MagicMock()
            
            def http_get(url, **kwargs):
                # Create response for HTTP session without JS execution
                mock_response = MagicMock()
                mock_response.status_code = 200
                
                # Different response depending on URL
                path = url.split(domain, 1)[1] if domain in url else url
                
                if path == '/static':
                    # Static page without JS requirements
                    mock_response.text = """
                    <html><body>
                        <h1>Static Content</h1>
                        <p>This content is visible without JavaScript.</p>
                    </body></html>
                    """
                elif path == '/dynamic':
                    # Dynamic page that loads content via JS
                    mock_response.text = """
                    <html><body>
                        <h1>Dynamic Content</h1>
                        <div id="content">Loading...</div>
                        <script>
                            // This script would normally load content dynamically
                            document.getElementById('content').innerHTML = 
                                'This content was loaded via JavaScript.';
                        </script>
                    </body></html>
                    """
                    # HTTP session doesn't execute JS, so we see the loading state
                elif path == '/spa':
                    # Single page app requiring JS
                    mock_response.text = """
                    <html><body>
                        <div id="app">Loading application...</div>
                        <script>
                            // This script would initialize the SPA
                            document.getElementById('app').innerHTML = 
                                '<h1>SPA Loaded</h1><p>Single Page App content.</p>';
                        </script>
                    </body></html>
                    """
                    # HTTP session doesn't execute JS, so we see the loading state
                else:
                    # Unknown path
                    mock_response.text = f"<html><body>Unknown path: {path}</body></html>"
                
                return mock_response
                
            http_session.get = http_get
            return http_session
            
        # Extend session manager to provide HTTP and browser sessions
        def mock_get_session(domain, force_new=False, session_type='http'):
            if session_type == 'browser':
                return mock_get_browser_session(domain)
            else:
                return mock_get_http_session(domain)
                
        monkeypatch.setattr(session_manager, 'get_session', mock_get_session)
        
        # Function to scrape content with appropriate session type
        def scrape_with_js_awareness(url, require_js=False):
            path = url.split(domain, 1)[1] if domain in url else url
            logger.info(f"Scraping {path} with require_js={require_js}")
            
            try:
                if require_js:
                    # Try browser session first
                    try:
                        session = session_manager.get_session(domain, session_type='browser')
                        response = session.get(url)
                        
                        # Use rendered_text which includes JS execution results
                        content = getattr(response, 'rendered_text', response.text)
                        
                        return {
                            'url': url,
                            'success': True,
                            'content': content,
                            'session_type': 'browser'
                        }
                    except Exception as e:
                        logger.warning(f"Browser session failed, falling back to HTTP: {str(e)}")
                        # Fall back to HTTP session if browser fails
                
                # Use regular HTTP session
                session = session_manager.get_session(domain, session_type='http')
                response = session.get(url)
                
                return {
                    'url': url,
                    'success': True,
                    'content': response.text,
                    'session_type': 'http'
                }
                
            except Exception as e:
                return {
                    'url': url,
                    'success': False,
                    'error': str(e)
                }
        
        # Test static and dynamic pages with different session types
        test_cases = [
            # (path, require_js, should_contain)
            ('/static', False, 'Static Content'),
            ('/static', True, 'Static Content'),
            ('/dynamic', False, 'Loading...'),
            ('/dynamic', True, 'This content was loaded via JavaScript'),
            ('/spa', False, 'Loading application'),
            ('/spa', True, 'SPA Loaded')
        ]
        
        results = []
        
        for path, require_js, expected_content in test_cases:
            url = f"{base_url}{path}"
            result = scrape_with_js_awareness(url, require_js)
            
            # Add expectation check
            if result['success']:
                result['contains_expected'] = expected_content in result['content']
            else:
                result['contains_expected'] = False
                
            results.append((path, require_js, result))
            
            logger.info(f"Result for {path} (require_js={require_js}): "
                       f"{'Success' if result['success'] else 'Failure'}, "
                       f"Contains expected content: {result['contains_expected']}")
        
        # Verify results
        # Static content should be accessible regardless of JS
        assert results[0][2]['contains_expected'], "Static content should be accessible without JS"
        assert results[1][2]['contains_expected'], "Static content should be accessible with JS"
        
        # Dynamic content should show loading state without JS and full content with JS
        assert results[2][2]['contains_expected'], "Dynamic page should show loading state without JS"
        assert results[3][2]['contains_expected'], "Dynamic page should show full content with JS"
        
        # SPA should show loading state without JS and full content with JS
        assert results[4][2]['contains_expected'], "SPA should show loading state without JS"
        assert results[5][2]['contains_expected'], "SPA should show full content with JS"
        
        # Test browser availability scenario
        nonlocal browser_available
        browser_available = False
        
        # Try SPA with JS required but browser unavailable
        spa_url = f"{base_url}/spa"
        fallback_result = scrape_with_js_awareness(spa_url, require_js=True)
        
        logger.info(f"Browser unavailable test result: {'Success' if fallback_result['success'] else 'Failure'}")
        logger.info(f"Session type used: {fallback_result.get('session_type', 'None')}")
        
        # Verify fallback to HTTP session when browser is unavailable
        assert fallback_result['success'], "Should fall back to HTTP session when browser unavailable"
        assert fallback_result['session_type'] == 'http', "Should use HTTP session as fallback"
        assert 'Loading application' in fallback_result['content'], "Should show loading state with fallback HTTP session"
    
    def test_overall_system_resilience(self, service_registry, monkeypatch):
        """Test overall system resilience against a mix of challenging scenarios."""
        # Get services
        session_manager = service_registry.get_service('session_manager')
        proxy_manager = service_registry.get_service('proxy_manager')
        rate_limiter = service_registry.get_service('rate_limiter')
        retry_manager = service_registry.get_service('retry_manager')
        circuit_breaker = service_registry.get_service('circuit_breaker_manager')
        error_classifier = service_registry.get_service('error_classifier')
        
        # Define test websites with different behaviors/challenges
        test_sites = [
            {
                'domain': 'normal.example.com',
                'challenges': [],
                'expected_success': 0.95,  # 95% success expected
            },
            {
                'domain': 'rate-limited.example.com',
                'challenges': ['rate_limit'],
                'expected_success': 0.8,
            },
            {
                'domain': 'captcha.example.com',
                'challenges': ['captcha'],
                'expected_success': 0.5,
            },
            {
                'domain': 'ip-blocked.example.com',
                'challenges': ['ip_block'],
                'expected_success': 0.7,
            },
            {
                'domain': 'flaky.example.com',
                'challenges': ['flaky'],
                'expected_success': 0.8,
            },
            {
                'domain': 'session-required.example.com',
                'challenges': ['session'],
                'expected_success': 0.9,
            },
            {
                'domain': 'js-required.example.com',
                'challenges': ['javascript'],
                'expected_success': 0.6,
            },
            {
                'domain': 'nightmare.example.com',
                'challenges': ['rate_limit', 'captcha', 'ip_block', 'flaky'],
                'expected_success': 0.3,  # Very challenging site
            }
        ]
        
        # Mock domain-specific session behavior
        domain_sessions = {}
        
        def mock_get_session(domain, force_new=False):
            # Initialize domain state
            if domain not in domain_sessions:
                domain_sessions[domain] = {
                    'request_count': 0,
                    'blocked_ips': set(),
                    'captcha_triggered': False,
                    'session_data': {},
                }
                
            state = domain_sessions[domain]
            state['request_count'] += 1
            
            # Create session mock
            session = MagicMock()
            
            # Configure behavior based on site challenges
            site = next((s for s in test_sites if s['domain'] == domain), None)
            challenges = site['challenges'] if site else []
            
            def challenge_aware_get(url, proxies=None, **kwargs):
                # Extract path
                path = url.split('://', 1)[1].split('/', 1)[1] if '://' in url and '/' in url.split('://', 1)[1] else '/'
                
                # Initialize response
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = f"Content from {url}"
                
                # Track the IP being used
                ip = proxies.get('http', 'default-ip') if proxies else 'default-ip'
                
                # Apply challenges
                
                # Rate limiting
                if 'rate_limit' in challenges and state['request_count'] % 4 == 0:
                    mock_response.status_code = 429
                    mock_response.text = "Too Many Requests"
                    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                        "429 Too Many Requests", response=mock_response
                    )
                    return mock_response
                
                # IP blocking
                if 'ip_block' in challenges:
                    if ip in state['blocked_ips']:
                        mock_response.status_code = 403
                        mock_response.text = "Your IP has been blocked"
                        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                            "403 Forbidden: IP Blocked", response=mock_response
                        )
                        return mock_response
                    
                    # Block IP after some requests
                    if state['request_count'] % 10 == 0:
                        state['blocked_ips'].add(ip)
                
                # CAPTCHA
                if 'captcha' in challenges:
                    # Trigger CAPTCHA based on request pattern
                    if state['request_count'] % 5 == 0:
                        state['captcha_triggered'] = True
                    
                    if state['captcha_triggered'] and not force_new:
                        mock_response.text = "Please complete this CAPTCHA to continue"
                        # HTTP 200 with CAPTCHA content
                        return mock_response
                
                # Flaky connection
                if 'flaky' in challenges and random.random() < 0.3:
                    raise requests.exceptions.ConnectionError("Random connection error")
                
                # Session requirement
                if 'session' in challenges and path == '/login':
                    state['session_data']['logged_in'] = True
                    mock_response.text = "Login successful"
                    return mock_response
                
                if 'session' in challenges and path == '/profile':
                    if not state.get('session_data', {}).get('logged_in', False):
                        mock_response.status_code = 401
                        mock_response.text = "Unauthorized: Please log in"
                        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                            "401 Unauthorized", response=mock_response
                        )
                        return mock_response
                
                # JavaScript requirement
                if 'javascript' in challenges:
                    mock_response.text = "Loading content... (requires JavaScript)"
                    if hasattr(session, 'js_enabled') and session.js_enabled:
                        mock_response.rendered_text = "Content loaded via JavaScript"
                
                # If we get here, return successful response
                return mock_response
                
            session.get = challenge_aware_get
            
            # Add JS capability flag for some sessions
            session.js_enabled = random.random() < 0.5  # 50% chance of JS support
            
            return session
            
        monkeypatch.setattr(session_manager, 'get_session', mock_get_session)
        
        # Integrated scraping function that uses all components
        def scrape_with_resilience(domain, path):
            url = f"https://{domain}{path}"
            
            # Get circuit breaker for this domain
            cb = circuit_breaker.get_circuit_breaker(domain)
            
            # If circuit is open, skip request
            if not cb.allow_request():
                logger.info(f"Circuit open for {domain}, skipping request")
                return {
                    'url': url,
                    'success': False,
                    'error': 'Circuit breaker open',
                    'recovery_action': 'wait'
                }
            
            # Apply rate limiting
            waited = rate_limiter.wait_if_needed(domain)
            if waited:
                logger.info(f"Rate limited for {domain}, waited before proceeding")
            
            # Get a proxy
            proxy = proxy_manager.get_proxy(domain)
            proxy_url = proxy['url']
            
            # Define retry function
            @retry_manager.retry(domain, max_attempts=3)
            def fetch_with_retry():
                session = None
                try:
                    # Get an appropriate session
                    session = session_manager.get_session(domain)
                    
                    # Make the request
                    response = session.get(url, proxies={
                        'http': proxy_url,
                        'https': proxy_url
                    }, timeout=5)
                    
                    # Check for CAPTCHA
                    if 'captcha' in response.text.lower():
                        logger.warning(f"CAPTCHA detected at {url}")
                        
                        # Try a new session to bypass
                        new_session = session_manager.get_session(domain, force_new=True)
                        response = new_session.get(url, proxies={
                            'http': proxy_url,
                            'https': proxy_url
                        }, timeout=5)
                        
                        # If still CAPTCHA, raise exception
                        if 'captcha' in response.text.lower():
                            raise Exception("CAPTCHA could not be bypassed")
                    
                    # Check for HTTP errors
                    response.raise_for_status()
                    
                    # Record success
                    cb.record_success()
                    
                    # Return content (with JS rendering if available)
                    content = getattr(response, 'rendered_text', response.text)
                    return {
                        'url': url,
                        'success': True,
                        'content': content[:50] + "..." if len(content) > 50 else content
                    }
                    
                except requests.exceptions.HTTPError as e:
                    # HTTP error handling
                    if hasattr(e, 'response'):
                        error_info = error_classifier.classify_exception(
                            e, {'url': url, 'domain': domain, 'proxy': proxy_url}
                        )
                        
                        # IP blocking detection
                        if e.response.status_code == 403 and 'block' in e.response.text.lower():
                            logger.warning(f"IP {proxy_url} blocked, marking as failed")
                            proxy_manager.mark_proxy_failed(proxy_url)
                            # Don't record circuit breaker failure for IP blocks
                            raise
                        
                        # Rate limiting detection
                        if e.response.status_code == 429:
                            logger.warning(f"Rate limited at {url}")
                            rate_limiter.report_rate_limited(domain)
                            # Don't record circuit breaker failure for rate limits
                            raise
                    
                    # Record failure
                    cb.record_failure()
                    raise
                    
                except Exception as e:
                    # General error handling
                    logger.error(f"Error fetching {url}: {str(e)}")
                    cb.record_failure()
                    raise
            
            try:
                return fetch_with_retry()
            except Exception as e:
                return {
                    'url': url,
                    'success': False,
                    'error': str(e)
                }
        
        # Test all sites with multiple pages
        results = {}
        
        for site in test_sites:
            domain = site['domain']
            site_results = []
            
            # Try different paths
            paths = [
                '/',                # Home page
                '/login',           # Login page
                '/profile',         # Protected page
                '/page/1',          # Content page
                '/api/data.json',   # API endpoint
            ]
            
            for path in paths:
                result = scrape_with_resilience(domain, path)
                site_results.append(result)
            
            # Calculate success rate
            success_rate = sum(1 for r in site_results if r['success']) / len(site_results)
            
            results[domain] = {
                'success_rate': success_rate,
                'expected_success': site['expected_success'],
                'meets_expectation': success_rate >= site['expected_success'] * 0.8,  # Allow 20% margin
                'results': site_results
            }
            
            logger.info(f"Site {domain}: Success rate {success_rate:.2%} vs expected {site['expected_success']:.2%} - "
                       f"{'MEETS' if results[domain]['meets_expectation'] else 'BELOW'} expectation")
        
        # Summarize overall results
        overall_success = sum(1 for r in results.values() if r['meets_expectation']) / len(results)
        logger.info(f"Overall system resilience: {overall_success:.2%} of sites meet expectations")
        
        # Verify results
        assert overall_success >= 0.75, "At least 75% of sites should meet expectations"
        
        # The normal site should definitely meet expectations
        assert results['normal.example.com']['meets_expectation'], "Normal site should meet expectations"
        
        # At least one challenging site should meet expectations
        challenging_sites = [s for s in test_sites if len(s['challenges']) > 0]
        challenging_results = [results[s['domain']]['meets_expectation'] for s in challenging_sites]
        assert any(challenging_results), "At least one challenging site should meet expectations"