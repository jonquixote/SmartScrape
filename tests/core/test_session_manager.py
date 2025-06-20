import pytest
import time
import threading
from unittest.mock import MagicMock, patch
import requests
from urllib.parse import urlparse

from core.service_registry import ServiceRegistry
from core.session_manager import SessionManager, HttpSession, BrowserSession


@pytest.fixture
def session_manager():
    """Fixture to get a clean SessionManager instance."""
    manager = SessionManager()
    manager.initialize({
        'http': {
            'timeout': 10,
            'max_sessions_per_domain': 3,
            'session_ttl': 600,  # 10 minutes
            'user_agent_rotation': 'random'
        },
        'browser': {
            'browser_type': 'chromium',
            'headless': True,
            'max_sessions_per_domain': 2
        },
        'enable_health_checks': False  # Disable health checks for testing
    })
    
    yield manager
    
    # Clean up
    manager.shutdown()


class TestSessionManager:
    """Test suite for SessionManager."""

    def test_initialization(self):
        """Test SessionManager initialization."""
        manager = SessionManager()
        assert not manager._initialized
        
        manager.initialize()
        assert manager._initialized
        assert manager.name == "session_manager"
        
        # Clean up
        manager.shutdown()
        assert not manager._initialized

    def test_service_registry_integration(self):
        """Test SessionManager integration with ServiceRegistry."""
        registry = ServiceRegistry()
        registry.register_service_class(SessionManager)
        
        # Get the service through the registry
        manager = registry.get_service("session_manager")
        assert manager.is_initialized
        assert manager.name == "session_manager"
        
        # Clean up
        registry.shutdown_all()

    def test_get_domain(self, session_manager):
        """Test domain extraction from URLs."""
        # Full URLs
        assert session_manager._get_domain("https://example.com/path") == "example.com"
        assert session_manager._get_domain("http://sub.example.com/path?query=123") == "sub.example.com"
        
        # Remove www prefix
        assert session_manager._get_domain("https://www.example.com") == "example.com"
        
        # Handle URLs without scheme
        assert session_manager._get_domain("example.com/path") == "example.com"

    @patch('requests.Session')
    def test_create_http_session(self, mock_session, session_manager):
        """Test HTTP session creation."""
        # Set up mock
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        # Create session
        session = session_manager._create_http_session("example.com")
        
        # Verify session creation
        assert isinstance(session, HttpSession)
        assert session.session is mock_session_instance
        assert "example.com" in session.session_id
        
        # Verify headers
        assert "User-Agent" in mock_session_instance.headers
        assert "Accept" in mock_session_instance.headers
        assert "Accept-Language" in mock_session_instance.headers

    @patch('requests.Session')
    def test_get_http_session(self, mock_session, session_manager):
        """Test getting HTTP sessions."""
        # Set up mock
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        # Get a session
        session1 = session_manager.get_session("https://example.com", session_type="http")
        assert isinstance(session1, HttpSession)
        
        # Get another session for the same domain
        session2 = session_manager.get_session("https://example.com", session_type="http")
        assert session2 is session1  # Should reuse the session
        assert session2.request_count == 1  # Usage should be tracked
        
        # Force a new session
        session3 = session_manager.get_session("https://example.com", session_type="http", force_new=True)
        assert session3 is not session1  # Should be a new session
        
        # Get a session for a different domain
        session4 = session_manager.get_session("https://another.com", session_type="http")
        assert session4 is not session1  # Should be a new session
        assert session4 is not session3  # Should be a new session

    @patch('requests.Session')
    def test_max_sessions_per_domain(self, mock_session, session_manager):
        """Test max sessions per domain limit."""
        # Set up mock
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        # Create maximum number of sessions (3 configured in fixture)
        sessions = []
        for i in range(3):
            sessions.append(session_manager.get_session("https://example.com", force_new=True))
            
        # Verify we have 3 sessions
        assert len(session_manager._http_sessions["example.com"]) == 3
        
        # Create one more, which should replace the oldest
        oldest_session = sessions[0]
        new_session = session_manager.get_session("https://example.com", force_new=True)
        
        # Verify we still have 3 sessions, but the oldest was replaced
        assert len(session_manager._http_sessions["example.com"]) == 3
        assert oldest_session not in session_manager._http_sessions["example.com"]
        assert new_session in session_manager._http_sessions["example.com"]

    @patch('requests.Session')
    def test_close_session(self, mock_session, session_manager):
        """Test closing a session."""
        # Set up mock
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        # Create sessions for two domains
        session1 = session_manager.get_session("https://example.com")
        session2 = session_manager.get_session("https://another.com")
        
        # Close specific session
        session_manager.close_session("https://example.com", session1.session_id)
        
        # Verify the domain entry exists but the session is gone
        assert "example.com" in session_manager._http_sessions
        assert len(session_manager._http_sessions["example.com"]) == 0
        
        # Verify the other domain's session is still there
        assert "another.com" in session_manager._http_sessions
        assert len(session_manager._http_sessions["another.com"]) == 1

    @patch('requests.Session')
    def test_close_all_sessions(self, mock_session, session_manager):
        """Test closing all sessions."""
        # Set up mock
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        # Create sessions for multiple domains
        session_manager.get_session("https://example.com")
        session_manager.get_session("https://another.com")
        
        # Close all sessions
        session_manager.close_all_sessions()
        
        # Verify all sessions are closed
        assert len(session_manager._http_sessions) == 0

    @patch('requests.Session')
    def test_session_exists(self, mock_session, session_manager):
        """Test checking if a session exists."""
        # Set up mock
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        # Create a session
        session = session_manager.get_session("https://example.com")
        
        # Check if the session exists
        assert session_manager.session_exists("https://example.com", session.session_id)
        assert not session_manager.session_exists("https://example.com", "non-existent-id")
        assert not session_manager.session_exists("https://another.com", session.session_id)

    @patch('requests.Session')
    def test_rotate_user_agent(self, mock_session, session_manager):
        """Test user agent rotation."""
        # Set up mock
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.headers = {}
        
        # Create a session
        session = session_manager.get_session("https://example.com")
        
        # Get original user agent
        original_ua = session.session.headers.get('User-Agent')
        
        # Mock _get_user_agent to return a fixed value
        with patch.object(session_manager, '_get_user_agent', return_value="NewUserAgent/1.0"):
            # Rotate user agent
            session_manager.rotate_user_agent(session, "example.com")
            
            # Verify user agent was changed
            assert session.session.headers.get('User-Agent') == "NewUserAgent/1.0"

    @patch('requests.Session')
    def test_clear_cookies(self, mock_session, session_manager):
        """Test clearing cookies."""
        # Set up mock
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        # Create a session
        session = session_manager.get_session("https://example.com")
        
        # Clear cookies
        session_manager.clear_cookies(session)
        
        # Verify cookies were cleared
        mock_session_instance.cookies.clear.assert_called_once()

    @patch('requests.Session')
    def test_set_proxy(self, mock_session, session_manager):
        """Test setting a proxy."""
        # Set up mock
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.proxies = {}
        
        # Create a session
        session = session_manager.get_session("https://example.com")
        
        # Set proxy
        proxy = {"http": "http://proxy.example.com:8080"}
        session_manager.set_proxy(session, proxy)
        
        # Verify proxy was set
        assert session.session.proxies == proxy
        assert session.config['proxy'] == proxy

    @patch('requests.Session')
    def test_get_session_metrics(self, mock_session, session_manager):
        """Test getting session metrics."""
        # Set up mock
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        # Create sessions for multiple domains
        session1 = session_manager.get_session("https://example.com")
        session2 = session_manager.get_session("https://another.com")
        
        # Increment usage to create metrics
        session1.update_usage()
        session1.update_usage()
        session1.record_error()
        
        # Get metrics for all domains
        metrics = session_manager.get_session_metrics()
        
        # Verify metrics
        assert metrics['http_sessions']['total'] == 2
        assert metrics['http_sessions']['domains'] == 2
        assert 'example.com' in metrics['http_sessions']['by_domain']
        assert 'another.com' in metrics['http_sessions']['by_domain']
        assert metrics['http_sessions']['by_domain']['example.com']['total_requests'] == 3  # 1 initial + 2 updates
        assert metrics['http_sessions']['by_domain']['example.com']['total_errors'] == 1
        
        # Get metrics for specific domain
        metrics = session_manager.get_session_metrics("https://example.com")
        assert 'example.com' in metrics['http_sessions']['by_domain']
        assert 'another.com' not in metrics['http_sessions']['by_domain']

    @patch('requests.Session')
    @patch('time.time')
    def test_cleanup_expired_sessions(self, mock_time, mock_session, session_manager):
        """Test cleanup of expired sessions."""
        # Set up mocks
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_time.return_value = 1000  # Start time
        
        # Create sessions
        session1 = session_manager.get_session("https://example.com")
        
        # Simulate time passing
        mock_time.return_value = 1000 + session_manager._default_http_config['session_ttl'] + 1
        
        # Run cleanup
        session_manager._cleanup_expired_sessions()
        
        # Verify expired session was removed
        assert "example.com" not in session_manager._http_sessions or len(session_manager._http_sessions["example.com"]) == 0
        
        # Verify session was closed
        mock_session_instance.close.assert_called_once()

    @pytest.mark.skipif(True, reason="Requires a real Playwright instance")
    def test_browser_session_real(self, session_manager):
        """
        Test browser session with real Playwright (skip by default).
        
        To run this test, remove the skipif decorator and make sure Playwright is installed:
        pip install playwright
        python -m playwright install
        """
        # Create a browser session
        session = session_manager.get_browser_session("https://example.com")
        
        # Verify session was created
        assert isinstance(session, BrowserSession)
        assert session.browser is not None
        assert session.context is not None
        assert len(session.pages) == 1
        
        # Close the session
        session_manager.close_browser_session(session.session_id)
        
        # Verify session was closed
        assert "example.com" not in session_manager._browser_sessions

    @patch('core.session_manager.sync_playwright')
    def test_browser_session_mock(self, mock_sync_playwright, session_manager):
        """Test browser session with mocked Playwright."""
        # Set up mocks
        mock_playwright = MagicMock()
        mock_sync_playwright.return_value.start.return_value = mock_playwright
        
        mock_browser_launcher = MagicMock()
        mock_playwright.chromium = mock_browser_launcher
        
        mock_browser = MagicMock()
        mock_browser_launcher.launch.return_value = mock_browser
        
        mock_context = MagicMock()
        mock_browser.new_context.return_value = mock_context
        
        mock_page = MagicMock()
        mock_context.new_page.return_value = mock_page
        
        # Create a browser session
        with patch('core.session_manager.stealth_sync', create=True):
            session = session_manager.get_browser_session("https://example.com")
        
        # Verify session was created
        assert isinstance(session, BrowserSession)
        assert session.browser is mock_browser
        assert session.context is mock_context
        assert len(session.pages) == 1
        
        # Navigate to a page
        session_manager.get_page_in_browser(session, "https://example.com/path")
        
        # Verify page navigation
        assert len(session.pages) == 2
        mock_page.goto.assert_called_once()
        
        # Execute script in browser
        script = "return document.title;"
        session_manager.execute_in_browser(session, script)
        
        # Verify script execution
        mock_page.evaluate.assert_called_once_with(script)
        
        # Configure browser
        options = {'geolocation': {'latitude': 51.507, 'longitude': -0.127}}
        session_manager.configure_browser(session, options)
        
        # Verify browser configuration
        assert session.config['geolocation'] == options['geolocation']
        mock_context.set_geolocation.assert_called_once_with(options['geolocation'])
        
        # Close the session
        session_manager.close_browser_session(session.session_id)
        
        # Verify session was closed
        assert "example.com" not in session_manager._browser_sessions


if __name__ == "__main__":
    pytest.main(["-v", __file__])