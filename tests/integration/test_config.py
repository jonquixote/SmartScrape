"""
Test configuration module for setting up the environment before tests run.

This module pre-registers necessary services to prevent import-time errors.
"""

# Import this module at the beginning of test files to ensure services are registered
# before other modules are imported

import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import core services
from core.service_registry import ServiceRegistry
from core.service_interface import BaseService

# Create a custom URL service for tests
class MockURLService(BaseService):
    """Mock URL service for testing."""
    
    def __init__(self):
        """Initialize the mock URL service."""
        self._initialized = True
        
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "url_service"
        
    def get_domain(self, url):
        """Get domain from URL."""
        return "example.com"
        
    def normalize_url(self, url):
        """Normalize URL."""
        return "https://example.com/"
        
    def is_same_domain(self, url1, url2):
        """Check if URLs are from the same domain."""
        return True
        
    def initialize(self, config=None):
        """Initialize service."""
        self._initialized = True
        
    def shutdown(self):
        """Shutdown service."""
        pass
        
    def get_service_health(self):
        """Get service health."""
        return {"status": "healthy", "details": "Mocked service", "metrics": {}}

    def get_queue(self, name="default"):
        """Get URL queue by name."""
        return MockURLQueue()

    def is_allowed(self, url):
        """Check if the URL is allowed by robots.txt."""
        return True  # For testing, always allow URLs

# Create a custom HTML service for tests
class MockHTMLService(BaseService):
    """Mock HTML service for testing."""
    
    def __init__(self):
        """Initialize the mock HTML service."""
        self._initialized = True
        
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "html_service"
        
    def parse_html(self, html):
        """Parse HTML string."""
        return {"parsed": True}
        
    def extract_links(self, html, base_url=None):
        """Extract links from HTML."""
        return [
            {"url": "https://example.com/link1", "text": "Link 1"},
            {"url": "https://example.com/link2", "text": "Link 2"}
        ]
        
    def find_elements(self, html, selector):
        """Find elements in HTML using selector."""
        return ["element1", "element2"]
        
    def extract_main_content(self, html):
        """Extract main content from HTML."""
        return "<main>Mocked main content</main>"
        
    def initialize(self, config=None):
        """Initialize service."""
        self._initialized = True
        
    def shutdown(self):
        """Shutdown service."""
        pass
        
    def get_service_health(self):
        """Get service health."""
        return {"status": "healthy", "details": "Mocked service", "metrics": {}}

# Create a mock database connection service
class MockDatabaseService(BaseService):
    """Mock database service for testing."""
    
    def __init__(self):
        """Initialize the mock database service."""
        self._initialized = True
        self.connections = {
            "default": {
                "name": "default",
                "type": "sqlite",
                "connection_string": ":memory:"
            }
        }
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "database_service"
    
    def get_connection(self, name="default"):
        """Get a database connection by name."""
        if name not in self.connections:
            raise ValueError(f"Connection {name} not found")
        return self.connections[name]
    
    def execute_query(self, query, parameters=None, connection_name="default"):
        """Execute a query on the specified connection."""
        # Mock query execution
        return [{"id": 1, "name": "Test Result"}]
    
    def initialize(self, config=None):
        """Initialize service."""
        self._initialized = True
    
    def shutdown(self):
        """Shutdown service."""
        pass
    
    def get_service_health(self):
        """Get service health."""
        return {"status": "healthy", "details": "Mocked database service", "metrics": {}}

class MockURLQueue:
    """Mock URL queue for testing."""
    
    def __init__(self):
        """Initialize the mock URL queue."""
        self.urls = []
        self.visited = set()
        
    def add(self, url, priority=0):
        """Add URL to queue."""
        self.urls.append(url)
        return True
        
    def get(self):
        """Get next URL from queue."""
        if not self.urls:
            return None
        return self.urls.pop(0)
        
    def complete(self, url):
        """Mark URL as completed."""
        self.visited.add(url)
        
    def is_visited(self, url):
        """Check if URL has been visited."""
        return url in self.visited
        
    def is_in_progress(self, url):
        """Check if URL is being processed."""
        return False
        
    def clear(self):
        """Clear queue."""
        self.urls = []
        self.visited = set()

# Register mock services to prevent import-time errors
def register_mock_services():
    """Register mock services for tests."""
    # Create a registry instance (singleton)
    registry = ServiceRegistry()
    
    # Create and register the mock URL service
    mock_url_service = MockURLService()
    registry.register_service("url_service", mock_url_service)
    
    # Create and register the mock HTML service
    mock_html_service = MockHTMLService()
    registry.register_service("html_service", mock_html_service)
    
    # Create and register the mock database service
    mock_db_service = MockDatabaseService()
    registry.register_service("database_service", mock_db_service)

# Register services during module import
register_mock_services()