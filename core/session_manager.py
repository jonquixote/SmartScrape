import logging
import time
import random
import threading
from typing import Dict, Any, Optional, List, Union, Callable
import uuid
from urllib.parse import urlparse
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fake_useragent import UserAgent
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page

from core.service_interface import BaseService

logger = logging.getLogger(__name__)

class HttpSession:
    """Wrapper for HTTP sessions with metadata."""
    
    def __init__(self, session_id: str, session: requests.Session, config: Dict[str, Any]):
        self.session_id = session_id
        self.session = session
        self.created_at = time.time()
        self.last_used_at = time.time()
        self.request_count = 0
        self.error_count = 0
        self.config = config
        self.metadata = {}
        
    def update_usage(self):
        """Update session usage statistics."""
        self.last_used_at = time.time()
        self.request_count += 1
        
    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1
        
    @property
    def age(self) -> float:
        """Get the age of the session in seconds."""
        return time.time() - self.created_at
        
    @property
    def idle_time(self) -> float:
        """Get the time since last use in seconds."""
        return time.time() - self.last_used_at


class BrowserSession:
    """Wrapper for browser sessions with metadata."""
    
    def __init__(self, session_id: str, browser: Browser, context: BrowserContext, config: Dict[str, Any]):
        self.session_id = session_id
        self.browser = browser
        self.context = context
        self.pages = {}  # Dictionary of page_id -> Page
        self.created_at = time.time()
        self.last_used_at = time.time()
        self.request_count = 0
        self.error_count = 0
        self.config = config
        self.metadata = {}
        
    def add_page(self, page_id: str, page: Page):
        """Add a page to the session."""
        self.pages[page_id] = page
        
    def get_page(self, page_id: str) -> Optional[Page]:
        """Get a page from the session."""
        return self.pages.get(page_id)
        
    def remove_page(self, page_id: str):
        """Remove a page from the session."""
        if page_id in self.pages:
            try:
                self.pages[page_id].close()
            except Exception as e:
                logger.warning(f"Error closing page {page_id}: {str(e)}")
            del self.pages[page_id]
            
    def update_usage(self):
        """Update session usage statistics."""
        self.last_used_at = time.time()
        self.request_count += 1
        
    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1
        
    @property
    def age(self) -> float:
        """Get the age of the session in seconds."""
        return time.time() - self.created_at
        
    @property
    def idle_time(self) -> float:
        """Get the time since last use in seconds."""
        return time.time() - self.last_used_at


class SessionManager(BaseService):
    """
    Centralized service for managing HTTP and browser sessions.
    
    Provides a unified interface for:
    - Creating and reusing HTTP sessions
    - Creating and managing browser (Playwright) sessions
    - Session configuration and rotation
    - Cookie and state management
    - Resource cleanup
    """
    
    def __init__(self):
        self._initialized = False
        self._config = None
        self._http_sessions = {}  # domain -> [HttpSession]
        self._browser_sessions = {}  # domain -> [BrowserSession]
        self._playwright = None
        self._user_agent_generator = None
        self._lock = threading.RLock()
        self._health_check_thread = None
        self._shutdown_event = threading.Event()
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the session manager with configuration."""
        if self._initialized:
            return
            
        self._config = config or {}
        
        # Set default configuration values if not provided
        self._default_http_config = self._config.get('http', {})
        self._default_http_config.setdefault('timeout', 30)
        self._default_http_config.setdefault('max_sessions_per_domain', 5)
        self._default_http_config.setdefault('session_ttl', 1800)  # 30 minutes
        self._default_http_config.setdefault('user_agent_rotation', 'random')
        self._default_http_config.setdefault('verify_ssl', True)
        
        # Retry configuration
        self._retry_config = self._config.get('retry', {})
        self._retry_config.setdefault('total', 3)
        self._retry_config.setdefault('backoff_factor', 0.3)
        self._retry_config.setdefault('status_forcelist', [500, 502, 503, 504])
        
        # Browser configuration
        self._browser_config = self._config.get('browser', {})
        self._browser_config.setdefault('browser_type', 'chromium')
        self._browser_config.setdefault('headless', True)
        self._browser_config.setdefault('max_sessions_per_domain', 2)
        self._browser_config.setdefault('session_ttl', 1800)  # 30 minutes
        
        # Initialize user agent generator
        try:
            self._user_agent_generator = UserAgent()
        except Exception as e:
            logger.warning(f"Failed to initialize UserAgent, using fallback: {str(e)}")
            self._user_agent_generator = None
            
        # Initialize health check thread if enabled
        if self._config.get('enable_health_checks', True):
            self._health_check_thread = threading.Thread(
                target=self._health_check_worker,
                daemon=True,
                name="SessionManagerHealthCheck"
            )
            self._health_check_thread.start()
        
        self._initialized = True
        logger.info("Session manager initialized")
    
    def shutdown(self) -> None:
        """Close all sessions and clean up resources."""
        if not self._initialized:
            return
            
        logger.info("Shutting down session manager...")
        self._shutdown_event.set()
        
        # Stop health check thread if running
        if self._health_check_thread and self._health_check_thread.is_alive():
            logger.debug("Waiting for health check thread to terminate...")
            self._health_check_thread.join(timeout=5.0)
            
        # Close all HTTP sessions
        with self._lock:
            for domain, sessions in self._http_sessions.items():
                for session in sessions:
                    try:
                        session.session.close()
                    except Exception as e:
                        logger.warning(f"Error closing HTTP session for {domain}: {str(e)}")
            self._http_sessions.clear()
            
            # Close all browser sessions
            for domain, sessions in self._browser_sessions.items():
                for session in sessions:
                    try:
                        # Close all pages
                        for page_id in list(session.pages.keys()):
                            session.remove_page(page_id)
                        
                        # Close browser context and browser
                        session.context.close()
                        session.browser.close()
                    except Exception as e:
                        logger.warning(f"Error closing browser session for {domain}: {str(e)}")
            self._browser_sessions.clear()
            
            # Close Playwright if initialized
            if self._playwright:
                try:
                    self._playwright.stop()
                except Exception as e:
                    logger.warning(f"Error closing Playwright: {str(e)}")
                self._playwright = None
        
        self._initialized = False
        logger.info("Session manager shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "session_manager"
        
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL for session management."""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Handle URLs without scheme
            if not domain and not parsed_url.scheme:
                domain = parsed_url.path.split('/')[0]
                
            # Remove www. prefix for consistent domain matching
            if domain.startswith('www.'):
                domain = domain[4:]
                
            return domain or url  # Fallback to full URL if domain extraction fails
        except Exception as e:
            logger.warning(f"Error extracting domain from {url}: {str(e)}")
            return url

    def _create_http_session(self, domain: str, config: Optional[Dict[str, Any]] = None) -> HttpSession:
        """Create a new HTTP session with the specified configuration."""
        session_config = self._default_http_config.copy()
        if config:
            session_config.update(config)
            
        # Create the requests session
        session = requests.Session()
        
        # Configure retries
        retry_config = self._retry_config.copy()
        if config and 'retry' in config:
            retry_config.update(config['retry'])
            
        retry = Retry(
            total=retry_config['total'],
            backoff_factor=retry_config['backoff_factor'],
            status_forcelist=retry_config['status_forcelist'],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        # Set default headers
        user_agent = self._get_user_agent(domain, session_config.get('user_agent_rotation', 'random'))
        session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1',
        })
        
        # Set additional headers if provided
        if 'headers' in session_config:
            session.headers.update(session_config['headers'])
            
        # Set timeout
        session.timeout = session_config.get('timeout', 30)
        
        # Set SSL verification
        session.verify = session_config.get('verify_ssl', True)
        
        # Create session ID
        session_id = f"{domain}_{str(uuid.uuid4())[:8]}"
        
        # Create and return HttpSession wrapper
        return HttpSession(session_id, session, session_config)
        
    def _get_user_agent(self, domain: str, rotation_strategy: str = 'random') -> str:
        """Get a user agent based on the specified rotation strategy."""
        if rotation_strategy == 'none':
            return "SmartScrape/1.0"
            
        if self._user_agent_generator is None:
            # Fallback user agents if fake-useragent fails
            fallback_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
            ]
            return random.choice(fallback_agents)
            
        try:
            if rotation_strategy == 'random':
                return self._user_agent_generator.random
            elif rotation_strategy == 'chrome':
                return self._user_agent_generator.chrome
            elif rotation_strategy == 'firefox':
                return self._user_agent_generator.firefox
            elif rotation_strategy == 'safari':
                return self._user_agent_generator.safari
            elif rotation_strategy == 'edge':
                return self._user_agent_generator.edge
            else:
                return self._user_agent_generator.random
        except Exception as e:
            logger.warning(f"Error getting user agent with strategy {rotation_strategy}: {str(e)}")
            return "SmartScrape/1.0"

    def get_session(self, url: str, session_type: str = 'http', 
                    force_new: bool = False, config: Optional[Dict[str, Any]] = None) -> Union[HttpSession, BrowserSession]:
        """
        Get or create a session for the specified URL.
        
        Args:
            url: The URL or domain to get a session for
            session_type: Type of session ('http' or 'browser')
            force_new: If True, always create a new session
            config: Optional configuration overrides
            
        Returns:
            An HttpSession or BrowserSession object
        """
        if not self._initialized:
            raise RuntimeError("SessionManager is not initialized")
            
        domain = self._get_domain(url)
        
        if session_type.lower() == 'http':
            return self._get_http_session(domain, force_new, config)
        elif session_type.lower() in ('browser', 'playwright'):
            return self._get_browser_session(domain, force_new, config)
        else:
            raise ValueError(f"Unsupported session type: {session_type}")
    
    def _get_http_session(self, domain: str, force_new: bool = False, 
                          config: Optional[Dict[str, Any]] = None) -> HttpSession:
        """Get or create an HTTP session for the specified domain."""
        with self._lock:
            # Initialize domain entry if it doesn't exist
            if domain not in self._http_sessions:
                self._http_sessions[domain] = []
                
            # Try to find an existing session if not forcing new
            if not force_new and self._http_sessions[domain]:
                # Get the least used session
                session = min(self._http_sessions[domain], key=lambda s: s.request_count)
                session.update_usage()
                return session
                
            # Check if we've reached the maximum number of sessions
            max_sessions = self._default_http_config['max_sessions_per_domain']
            if config and 'max_sessions_per_domain' in config:
                max_sessions = config['max_sessions_per_domain']
                
            if len(self._http_sessions[domain]) >= max_sessions:
                # Find the oldest session and replace it
                oldest_session = min(self._http_sessions[domain], key=lambda s: s.created_at)
                try:
                    oldest_session.session.close()
                except Exception as e:
                    logger.warning(f"Error closing HTTP session: {str(e)}")
                    
                self._http_sessions[domain].remove(oldest_session)
                
            # Create a new session
            session = self._create_http_session(domain, config)
            self._http_sessions[domain].append(session)
            
            return session
    
    def _get_browser_session(self, domain: str, force_new: bool = False, 
                            config: Optional[Dict[str, Any]] = None) -> BrowserSession:
        """Get or create a browser session for the specified domain."""
        with self._lock:
            # Initialize domain entry if it doesn't exist
            if domain not in self._browser_sessions:
                self._browser_sessions[domain] = []
                
            # Try to find an existing session if not forcing new
            if not force_new and self._browser_sessions[domain]:
                # Get the least used session
                session = min(self._browser_sessions[domain], key=lambda s: s.request_count)
                session.update_usage()
                return session
                
            # Check if we've reached the maximum number of sessions
            max_sessions = self._browser_config['max_sessions_per_domain']
            if config and 'max_sessions_per_domain' in config:
                max_sessions = config['max_sessions_per_domain']
                
            if len(self._browser_sessions[domain]) >= max_sessions:
                # Find the oldest session and replace it
                oldest_session = min(self._browser_sessions[domain], key=lambda s: s.created_at)
                try:
                    # Close all pages
                    for page_id in list(oldest_session.pages.keys()):
                        oldest_session.remove_page(page_id)
                    
                    # Close browser context
                    oldest_session.context.close()
                    oldest_session.browser.close()
                except Exception as e:
                    logger.warning(f"Error closing browser session: {str(e)}")
                    
                self._browser_sessions[domain].remove(oldest_session)
                
            # Create a new session
            session = self._create_browser_session(domain, config)
            self._browser_sessions[domain].append(session)
            
            return session
    
    def _create_browser_session(self, domain: str, config: Optional[Dict[str, Any]] = None) -> BrowserSession:
        """Create a new browser session with the specified configuration."""
        # Merge with default config
        session_config = self._browser_config.copy()
        if config:
            session_config.update(config)
            
        # Initialize Playwright if not already done
        if self._playwright is None:
            self._playwright = sync_playwright().start()
            
        # Determine browser type
        browser_type = session_config.get('browser_type', 'chromium').lower()
        browser_launcher = getattr(self._playwright, browser_type)
        
        # Prepare browser launch options
        launch_options = {
            'headless': session_config.get('headless', True)
        }
        
        # Add proxy if configured
        if 'proxy' in session_config:
            launch_options['proxy'] = {
                'server': session_config['proxy']['server']
            }
            if 'username' in session_config['proxy'] and 'password' in session_config['proxy']:
                launch_options['proxy']['username'] = session_config['proxy']['username']
                launch_options['proxy']['password'] = session_config['proxy']['password']
        
        # Launch browser
        browser = browser_launcher.launch(**launch_options)
        
        # Create browser context with configuration
        context_options = {}
        
        # Set viewport if configured
        if 'viewport' in session_config:
            context_options['viewport'] = session_config['viewport']
        else:
            # Random viewport dimensions to prevent fingerprinting
            widths = [1280, 1366, 1440, 1536, 1600, 1920]
            heights = [720, 768, 800, 900, 1080]
            context_options['viewport'] = {
                'width': random.choice(widths),
                'height': random.choice(heights)
            }
            
        # Set user agent if configured
        if 'user_agent' in session_config:
            context_options['user_agent'] = session_config['user_agent']
        else:
            context_options['user_agent'] = self._get_user_agent(
                domain, session_config.get('user_agent_rotation', 'random')
            )
            
        # Set locale, timezone and other fingerprinting options
        context_options['locale'] = session_config.get('locale', 'en-US')
        context_options['timezone_id'] = session_config.get('timezone', 'America/New_York')
        context_options['geolocation'] = session_config.get('geolocation', None)
        context_options['permissions'] = session_config.get('permissions', [])
        
        # Create a browser context
        context = browser.new_context(**context_options)
        
        # Apply stealth mode if available and requested
        stealth_mode = session_config.get('stealth_mode', True)
        if stealth_mode:
            try:
                from playwright_stealth import stealth_sync
                page = context.new_page()
                stealth_sync(page)
                # We'll keep this initial page
                page_id = f"{domain}_main_{str(uuid.uuid4())[:8]}"
            except ImportError:
                logger.warning("playwright_stealth not available, stealth mode disabled")
                page = context.new_page()
                page_id = f"{domain}_main_{str(uuid.uuid4())[:8]}"
        else:
            page = context.new_page()
            page_id = f"{domain}_main_{str(uuid.uuid4())[:8]}"
            
        # Create and set up the session
        session_id = f"{domain}_browser_{str(uuid.uuid4())[:8]}"
        browser_session = BrowserSession(session_id, browser, context, session_config)
        browser_session.add_page(page_id, page)
        
        return browser_session
    
    def get_browser_session(self, url: str, browser_type: str = 'chromium', 
                           config: Optional[Dict[str, Any]] = None) -> BrowserSession:
        """
        Get or create a browser session for the specified URL.
        
        Args:
            url: The URL or domain to get a session for
            browser_type: Type of browser ('chromium', 'firefox', or 'webkit')
            config: Optional configuration overrides
            
        Returns:
            A BrowserSession object
        """
        session_config = config or {}
        session_config['browser_type'] = browser_type
        return self.get_session(url, session_type='browser', config=session_config)
    
    def get_page_in_browser(self, browser_session: BrowserSession, url: str, 
                           page_id: Optional[str] = None) -> tuple:
        """
        Navigate to a URL in a browser session and return the page.
        
        Args:
            browser_session: The browser session to use
            url: The URL to navigate to
            page_id: Optional ID for the page, will create a new ID if not provided
            
        Returns:
            A tuple of (page_id, page)
        """
        if not page_id:
            domain = self._get_domain(url)
            page_id = f"{domain}_page_{str(uuid.uuid4())[:8]}"
            
        # Check if the page already exists
        if page_id in browser_session.pages:
            page = browser_session.pages[page_id]
        else:
            # Create a new page
            page = browser_session.context.new_page()
            browser_session.add_page(page_id, page)
            
        # Navigate to the URL
        try:
            wait_until = browser_session.config.get('wait_until', 'domcontentloaded')
            timeout = browser_session.config.get('timeout', 30000)
            page.goto(url, wait_until=wait_until, timeout=timeout)
        except Exception as e:
            browser_session.record_error()
            raise e
            
        browser_session.update_usage()
        return (page_id, page)
        
    def execute_in_browser(self, browser_session: BrowserSession, script: str, page_id: Optional[str] = None):
        """
        Execute a script in a browser session.
        
        Args:
            browser_session: The browser session to use
            script: The JavaScript to execute
            page_id: Optional ID for the page, will use the first page if not provided
            
        Returns:
            The result of the script execution
        """
        # Get the page to execute on
        if page_id and page_id in browser_session.pages:
            page = browser_session.pages[page_id]
        elif browser_session.pages:
            # Use the first page
            page_id, page = next(iter(browser_session.pages.items()))
        else:
            # Create a new page
            page = browser_session.context.new_page()
            page_id = f"default_page_{str(uuid.uuid4())[:8]}"
            browser_session.add_page(page_id, page)
            
        # Execute the script
        try:
            result = page.evaluate(script)
            browser_session.update_usage()
            return result
        except Exception as e:
            browser_session.record_error()
            raise e
    
    def configure_browser(self, browser_session: BrowserSession, options: Dict[str, Any]):
        """
        Configure browser session with new options.
        
        Args:
            browser_session: The browser session to configure
            options: Dictionary of configuration options
        """
        with self._lock:
            # Update the session configuration
            browser_session.config.update(options)
            
            # Apply certain options that can be changed on existing sessions
            context = browser_session.context
            
            # Update geolocation if provided
            if 'geolocation' in options:
                context.set_geolocation(options['geolocation'])
                
            # Update permissions if provided
            if 'permissions' in options:
                for permission in options['permissions']:
                    context.grant_permissions([permission])
                    
            # Other options like user_agent, viewport, etc. can't be changed
            # without creating a new context, so we just update the config
            # for future reference
    
    def close_session(self, url: str, session_id: Optional[str] = None, session_type: str = 'http'):
        """
        Close a session for the specified URL/domain.
        
        Args:
            url: The URL or domain of the session to close
            session_id: Optional ID of the specific session to close
            session_type: Type of session ('http' or 'browser')
        """
        domain = self._get_domain(url)
        
        with self._lock:
            if session_type.lower() == 'http':
                if domain not in self._http_sessions:
                    return
                    
                if session_id:
                    # Close specific session
                    to_remove = None
                    for session in self._http_sessions[domain]:
                        if session.session_id == session_id:
                            try:
                                session.session.close()
                            except Exception as e:
                                logger.warning(f"Error closing HTTP session {session_id}: {str(e)}")
                            to_remove = session
                            break
                            
                    if to_remove:
                        self._http_sessions[domain].remove(to_remove)
                else:
                    # Close all sessions for domain
                    for session in self._http_sessions[domain]:
                        try:
                            session.session.close()
                        except Exception as e:
                            logger.warning(f"Error closing HTTP session for {domain}: {str(e)}")
                    self._http_sessions[domain] = []
                    
            elif session_type.lower() in ('browser', 'playwright'):
                if domain not in self._browser_sessions:
                    return
                    
                if session_id:
                    # Close specific session
                    to_remove = None
                    for session in self._browser_sessions[domain]:
                        if session.session_id == session_id:
                            try:
                                # Close all pages
                                for page_id in list(session.pages.keys()):
                                    session.remove_page(page_id)
                                
                                # Close browser context and browser
                                session.context.close()
                                session.browser.close()
                            except Exception as e:
                                logger.warning(f"Error closing browser session {session_id}: {str(e)}")
                            to_remove = session
                            break
                            
                    if to_remove:
                        self._browser_sessions[domain].remove(to_remove)
                else:
                    # Close all sessions for domain
                    for session in self._browser_sessions[domain]:
                        try:
                            # Close all pages
                            for page_id in list(session.pages.keys()):
                                session.remove_page(page_id)
                            
                            # Close browser context and browser
                            session.context.close()
                            session.browser.close()
                        except Exception as e:
                            logger.warning(f"Error closing browser session for {domain}: {str(e)}")
                    self._browser_sessions[domain] = []
    
    def close_browser_session(self, session_id: str):
        """Close a specific browser session by ID."""
        with self._lock:
            for domain, sessions in self._browser_sessions.items():
                for session in sessions:
                    if session.session_id == session_id:
                        try:
                            # Close all pages
                            for page_id in list(session.pages.keys()):
                                session.remove_page(page_id)
                            
                            # Close browser context and browser
                            session.context.close()
                            session.browser.close()
                        except Exception as e:
                            logger.warning(f"Error closing browser session {session_id}: {str(e)}")
                            
                        sessions.remove(session)
                        return
    
    def close_all_sessions(self):
        """Close all sessions of all types."""
        with self._lock:
            # Close all HTTP sessions
            for domain, sessions in self._http_sessions.items():
                for session in sessions:
                    try:
                        session.session.close()
                    except Exception as e:
                        logger.warning(f"Error closing HTTP session for {domain}: {str(e)}")
            self._http_sessions.clear()
            
            # Close all browser sessions
            for domain, sessions in self._browser_sessions.items():
                for session in sessions:
                    try:
                        # Close all pages
                        for page_id in list(session.pages.keys()):
                            session.remove_page(page_id)
                        
                        # Close browser context and browser
                        session.context.close()
                        session.browser.close()
                    except Exception as e:
                        logger.warning(f"Error closing browser session for {domain}: {str(e)}")
            self._browser_sessions.clear()
    
    def session_exists(self, url: str, session_id: str, session_type: str = 'http') -> bool:
        """
        Check if a session exists for the specified URL/domain.
        
        Args:
            url: The URL or domain to check
            session_id: ID of the specific session to check
            session_type: Type of session ('http' or 'browser')
            
        Returns:
            True if the session exists, False otherwise
        """
        domain = self._get_domain(url)
        
        with self._lock:
            if session_type.lower() == 'http':
                if domain not in self._http_sessions:
                    return False
                    
                return any(session.session_id == session_id for session in self._http_sessions[domain])
                
            elif session_type.lower() in ('browser', 'playwright'):
                if domain not in self._browser_sessions:
                    return False
                    
                return any(session.session_id == session_id for session in self._browser_sessions[domain])
                
            return False
    
    def rotate_user_agent(self, session, domain: str, rotation_strategy: str = 'random'):
        """
        Rotate the user agent for a session.
        
        Args:
            session: The session to rotate the user agent for (HttpSession or BrowserSession)
            domain: The domain for user agent tracking
            rotation_strategy: The user agent rotation strategy
        """
        new_user_agent = self._get_user_agent(domain, rotation_strategy)
        
        if isinstance(session, HttpSession):
            session.session.headers.update({'User-Agent': new_user_agent})
        elif isinstance(session, BrowserSession):
            # For browser sessions, we can't change the user agent after creation
            # We'll just update the config for future reference
            session.config['user_agent'] = new_user_agent
            logger.info(f"Updated user agent in browser session config. Note: This won't affect existing pages.")
    
    def rotate_proxy(self, session, domain: str):
        """
        Rotate the proxy for a session.
        
        Args:
            session: The session to rotate the proxy for (HttpSession or BrowserSession)
            domain: The domain for proxy tracking
        """
        # This is a placeholder. In a real implementation, you would get a new proxy
        # from a proxy manager or pool.
        logger.warning("Proxy rotation is not fully implemented. Override with a proxy manager integration.")
        
        if isinstance(session, HttpSession):
            # For HTTP sessions, we need to create a new session with the new proxy
            logger.info("HTTP session proxy rotation requires closing the current session and creating a new one.")
        elif isinstance(session, BrowserSession):
            # For browser sessions, we can't change the proxy after creation
            logger.info("Browser proxy rotation requires closing the current session and creating a new one.")
    
    def clear_cookies(self, session, domain: str = None):
        """
        Clear cookies for a session.
        
        Args:
            session: The session to clear cookies for (HttpSession or BrowserSession)
            domain: Optional domain to clear cookies for (browser sessions only)
        """
        if isinstance(session, HttpSession):
            session.session.cookies.clear()
        elif isinstance(session, BrowserSession):
            if domain:
                # Clear cookies for specific domain
                session.context.clear_cookies(domain=domain)
            else:
                # Clear all cookies
                session.context.clear_cookies()
    
    def set_proxy(self, session, proxy: Dict[str, str]):
        """
        Set a proxy for a session.
        
        Args:
            session: The session to set the proxy for (HttpSession only)
            proxy: Proxy configuration dict with keys like 'http', 'https', etc.
        """
        if isinstance(session, HttpSession):
            session.session.proxies.update(proxy)
            session.config['proxy'] = proxy
        elif isinstance(session, BrowserSession):
            logger.warning("Cannot change proxy for an existing browser session. Create a new session instead.")
    
    def get_session_metrics(self, url: str = None) -> Dict[str, Any]:
        """
        Get metrics about active sessions.
        
        Args:
            url: Optional URL to get metrics for specific domain
            
        Returns:
            Dictionary of metrics
        """
        with self._lock:
            metrics = {
                'http_sessions': {
                    'total': sum(len(sessions) for sessions in self._http_sessions.values()),
                    'domains': len(self._http_sessions),
                    'by_domain': {}
                },
                'browser_sessions': {
                    'total': sum(len(sessions) for sessions in self._browser_sessions.values()),
                    'domains': len(self._browser_sessions),
                    'by_domain': {}
                }
            }
            
            if url:
                domain = self._get_domain(url)
                
                # HTTP session metrics for specific domain
                if domain in self._http_sessions:
                    metrics['http_sessions']['by_domain'][domain] = {
                        'count': len(self._http_sessions[domain]),
                        'oldest_age': max(session.age for session in self._http_sessions[domain]) if self._http_sessions[domain] else 0,
                        'total_requests': sum(session.request_count for session in self._http_sessions[domain]),
                        'total_errors': sum(session.error_count for session in self._http_sessions[domain])
                    }
                    
                # Browser session metrics for specific domain
                if domain in self._browser_sessions:
                    metrics['browser_sessions']['by_domain'][domain] = {
                        'count': len(self._browser_sessions[domain]),
                        'pages': sum(len(session.pages) for session in self._browser_sessions[domain]),
                        'oldest_age': max(session.age for session in self._browser_sessions[domain]) if self._browser_sessions[domain] else 0,
                        'total_requests': sum(session.request_count for session in self._browser_sessions[domain]),
                        'total_errors': sum(session.error_count for session in self._browser_sessions[domain])
                    }
            else:
                # HTTP session metrics for all domains
                for domain, sessions in self._http_sessions.items():
                    metrics['http_sessions']['by_domain'][domain] = {
                        'count': len(sessions),
                        'oldest_age': max(session.age for session in sessions) if sessions else 0,
                        'total_requests': sum(session.request_count for session in sessions),
                        'total_errors': sum(session.error_count for session in sessions)
                    }
                    
                # Browser session metrics for all domains
                for domain, sessions in self._browser_sessions.items():
                    metrics['browser_sessions']['by_domain'][domain] = {
                        'count': len(sessions),
                        'pages': sum(len(session.pages) for session in sessions),
                        'oldest_age': max(session.age for session in sessions) if sessions else 0,
                        'total_requests': sum(session.request_count for session in sessions),
                        'total_errors': sum(session.error_count for session in sessions)
                    }
                    
            return metrics
    
    def _health_check_worker(self):
        """Background worker to perform session health checks and cleanup."""
        check_interval = self._config.get('health_check_interval', 300)  # 5 minutes
        
        while not self._shutdown_event.is_set():
            try:
                self._cleanup_expired_sessions()
                
                # Use event with timeout to support clean shutdown
                self._shutdown_event.wait(check_interval)
            except Exception as e:
                logger.error(f"Error in session health check: {str(e)}")
                
                # If an exception occurs, wait a bit before trying again
                self._shutdown_event.wait(60)
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions based on TTL and idle time."""
        with self._lock:
            now = time.time()
            
            # Clean up HTTP sessions
            for domain in list(self._http_sessions.keys()):
                # Create a copy of the list to safely remove items
                sessions = list(self._http_sessions[domain])
                
                for session in sessions:
                    session_ttl = session.config.get('session_ttl', self._default_http_config['session_ttl'])
                    max_idle_time = session.config.get('max_idle_time', session_ttl / 2)
                    
                    # Check if session has expired
                    if now - session.created_at > session_ttl or now - session.last_used_at > max_idle_time:
                        try:
                            session.session.close()
                        except Exception as e:
                            logger.warning(f"Error closing expired HTTP session: {str(e)}")
                            
                        self._http_sessions[domain].remove(session)
                        logger.debug(f"Removed expired HTTP session for {domain} (age: {session.age:.1f}s, idle: {session.idle_time:.1f}s)")
                        
                # Remove empty domain entries
                if not self._http_sessions[domain]:
                    del self._http_sessions[domain]
                    
            # Clean up browser sessions
            for domain in list(self._browser_sessions.keys()):
                # Create a copy of the list to safely remove items
                sessions = list(self._browser_sessions[domain])
                
                for session in sessions:
                    session_ttl = session.config.get('session_ttl', self._browser_config['session_ttl'])
                    max_idle_time = session.config.get('max_idle_time', session_ttl / 2)
                    
                    # Check if session has expired
                    if now - session.created_at > session_ttl or now - session.last_used_at > max_idle_time:
                        try:
                            # Close all pages
                            for page_id in list(session.pages.keys()):
                                session.remove_page(page_id)
                            
                            # Close browser context and browser
                            session.context.close()
                            session.browser.close()
                        except Exception as e:
                            logger.warning(f"Error closing expired browser session: {str(e)}")
                            
                        self._browser_sessions[domain].remove(session)
                        logger.debug(f"Removed expired browser session for {domain} (age: {session.age:.1f}s, idle: {session.idle_time:.1f}s)")
                        
                # Remove empty domain entries
                if not self._browser_sessions[domain]:
                    del self._browser_sessions[domain]