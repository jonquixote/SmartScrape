import unittest
from unittest.mock import patch, MagicMock, Mock
import requests
import datetime
import threading
import time
import os
import tempfile
import json

from core.proxy_manager import (
    ProxyManager, Proxy, ProxyStatus, ProxyType, ProxyAnonymity,
    StaticProxyProvider, FileProxyProvider, APIProxyProvider,
    RoundRobinStrategy, RandomStrategy, WeightedStrategy, LeastUsedStrategy,
    GeoMatchStrategy, AdaptiveStrategy
)


class TestProxy(unittest.TestCase):
    """Test the Proxy class functionality."""
    
    def test_proxy_initialization(self):
        """Test that a proxy can be initialized with proper values."""
        proxy = Proxy("192.168.1.1", 8080)
        self.assertEqual(proxy.address, "192.168.1.1")
        self.assertEqual(proxy.port, 8080)
        self.assertEqual(proxy.proxy_type, ProxyType.HTTP)
        self.assertEqual(proxy.status, ProxyStatus.ACTIVE)
        self.assertEqual(proxy.anonymity, ProxyAnonymity.UNKNOWN)
        
    def test_proxy_url_generation(self):
        """Test that proxy URLs are generated correctly."""
        proxy = Proxy("192.168.1.1", 8080, ProxyType.HTTP)
        self.assertEqual(proxy.url, "http://192.168.1.1:8080")
        
        proxy = Proxy("192.168.1.1", 8080, ProxyType.SOCKS5, "user", "pass")
        self.assertEqual(proxy.url, "socks5://192.168.1.1:8080")
        self.assertEqual(proxy.auth_url, "socks5://user:pass@192.168.1.1:8080")
        
    def test_proxy_active_status(self):
        """Test that proxy active status is determined correctly."""
        proxy = Proxy("192.168.1.1", 8080)
        self.assertTrue(proxy.is_active)
        
        proxy.status = ProxyStatus.BLACKLISTED
        self.assertFalse(proxy.is_active)
        
        proxy.status = ProxyStatus.ACTIVE
        proxy.blacklisted_until = datetime.datetime.now() + datetime.timedelta(hours=1)
        self.assertFalse(proxy.is_active)
        
        proxy.blacklisted_until = datetime.datetime.now() - datetime.timedelta(hours=1)
        self.assertTrue(proxy.is_active)
        
    def test_proxy_metrics(self):
        """Test that proxy metrics like success rate are calculated correctly."""
        proxy = Proxy("192.168.1.1", 8080)
        self.assertEqual(proxy.success_rate, 0.0)
        
        # Record successes and failures
        proxy.record_success(0.5)
        proxy.record_success(1.5)
        proxy.record_failure()
        
        self.assertEqual(proxy.success_count, 2)
        self.assertEqual(proxy.failure_count, 1)
        self.assertAlmostEqual(proxy.success_rate, 2/3, places=2)
        self.assertEqual(proxy.average_response_time, 1.0)
        
    def test_proxy_blacklisting(self):
        """Test that proxy blacklisting works correctly."""
        proxy = Proxy("192.168.1.1", 8080)
        self.assertTrue(proxy.is_active)
        
        proxy.blacklist("testing", datetime.timedelta(minutes=5))
        self.assertEqual(proxy.status, ProxyStatus.BLACKLISTED)
        self.assertFalse(proxy.is_active)
        self.assertEqual(proxy.blacklist_reason, "testing")
        
        proxy.unblacklist()
        self.assertEqual(proxy.status, ProxyStatus.ACTIVE)
        self.assertTrue(proxy.is_active)
        self.assertIsNone(proxy.blacklist_reason)
        
    def test_requests_dict_format(self):
        """Test that proxy configuration for requests is correct."""
        # HTTP proxy
        proxy = Proxy("192.168.1.1", 8080, ProxyType.HTTP)
        requests_dict = proxy.get_dict_for_requests()
        self.assertEqual(requests_dict["http"], "http://192.168.1.1:8080")
        
        # HTTPS proxy with auth
        proxy = Proxy("192.168.1.1", 8080, ProxyType.HTTPS, "user", "pass")
        requests_dict = proxy.get_dict_for_requests()
        self.assertEqual(requests_dict["https"], "https://user:pass@192.168.1.1:8080")
        
        # SOCKS proxy
        proxy = Proxy("192.168.1.1", 8080, ProxyType.SOCKS5)
        requests_dict = proxy.get_dict_for_requests()
        self.assertEqual(requests_dict["http"], "socks5://192.168.1.1:8080")
        self.assertEqual(requests_dict["https"], "socks5://192.168.1.1:8080")
        

class TestProxyProviders(unittest.TestCase):
    """Test the various proxy provider implementations."""
    
    def test_static_provider(self):
        """Test that StaticProxyProvider works correctly."""
        # Test with string format
        config = {
            "proxies": [
                "http://192.168.1.1:8080",
                "socks5://user:pass@192.168.1.2:1080"
            ]
        }
        provider = StaticProxyProvider("test", config)
        proxies = provider.get_proxies()
        
        self.assertEqual(len(proxies), 2)
        self.assertEqual(proxies[0].address, "192.168.1.1")
        self.assertEqual(proxies[0].port, 8080)
        self.assertEqual(proxies[0].proxy_type, ProxyType.HTTP)
        
        self.assertEqual(proxies[1].address, "192.168.1.2")
        self.assertEqual(proxies[1].port, 1080)
        self.assertEqual(proxies[1].proxy_type, ProxyType.SOCKS5)
        self.assertEqual(proxies[1].username, "user")
        self.assertEqual(proxies[1].password, "pass")
        
        # Test with dictionary format
        config = {
            "proxies": [
                {
                    "address": "192.168.1.3",
                    "port": 8888,
                    "type": "https",
                    "username": "admin",
                    "password": "secret",
                    "metadata": {"country": "US"}
                }
            ]
        }
        provider = StaticProxyProvider("test", config)
        proxies = provider.get_proxies()
        
        self.assertEqual(len(proxies), 1)
        self.assertEqual(proxies[0].address, "192.168.1.3")
        self.assertEqual(proxies[0].port, 8888)
        self.assertEqual(proxies[0].proxy_type, ProxyType.HTTPS)
        self.assertEqual(proxies[0].username, "admin")
        self.assertEqual(proxies[0].password, "secret")
        self.assertEqual(proxies[0].metadata.get("country"), "US")
        
    def test_file_provider(self):
        """Test that FileProxyProvider works correctly."""
        # Create a temporary file with proxies
        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
        try:
            temp_file.write("http://192.168.1.1:8080\n")
            temp_file.write("# This is a comment\n")
            temp_file.write("socks5://user:pass@192.168.1.2:1080\n")
            temp_file.close()
            
            config = {
                "file_path": temp_file.name
            }
            provider = FileProxyProvider("test", config)
            proxies = provider.get_proxies()
            
            self.assertEqual(len(proxies), 2)
            self.assertEqual(proxies[0].address, "192.168.1.1")
            self.assertEqual(proxies[0].port, 8080)
            self.assertEqual(proxies[1].address, "192.168.1.2")
            self.assertEqual(proxies[1].port, 1080)
            
            # Test JSON format
            json_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json')
            json.dump([
                "http://192.168.1.3:8080",
                {
                    "address": "192.168.1.4",
                    "port": 9090,
                    "type": "https"
                }
            ], json_file)
            json_file.close()
            
            config = {
                "file_path": json_file.name
            }
            provider = FileProxyProvider("test", config)
            proxies = provider.get_proxies()
            
            self.assertEqual(len(proxies), 2)
            self.assertEqual(proxies[0].address, "192.168.1.3")
            self.assertEqual(proxies[1].address, "192.168.1.4")
            self.assertEqual(proxies[1].port, 9090)
            
        finally:
            # Clean up temp files
            os.unlink(temp_file.name)
            os.unlink(json_file.name)
    
    @patch('requests.get')
    def test_api_provider(self, mock_get):
        """Test that APIProxyProvider works correctly."""
        # Mock the API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "proxies": [
                {
                    "ip": "192.168.1.1",
                    "port": 8080,
                    "protocol": "http",
                    "country": "US"
                },
                {
                    "ip": "192.168.1.2",
                    "port": 1080,
                    "protocol": "socks5",
                    "country": "UK"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        config = {
            "api_url": "https://proxy-api.example.com/v1/proxies",
            "format": "json",
            "json_path": ["proxies"],
            "mapping": {
                "address": "ip",
                "port": "port",
                "type": "protocol",
                "metadata": {
                    "country": "country"
                }
            }
        }
        
        provider = APIProxyProvider("test", config)
        proxies = provider.get_proxies()
        
        self.assertEqual(len(proxies), 2)
        self.assertEqual(proxies[0].address, "192.168.1.1")
        self.assertEqual(proxies[0].port, 8080)
        self.assertEqual(proxies[0].proxy_type, ProxyType.HTTP)
        self.assertEqual(proxies[0].metadata.get("country"), "US")
        
        self.assertEqual(proxies[1].address, "192.168.1.2")
        self.assertEqual(proxies[1].port, 1080)
        self.assertEqual(proxies[1].proxy_type, ProxyType.SOCKS5)
        self.assertEqual(proxies[1].metadata.get("country"), "UK")
        
        # Verify request parameters
        mock_get.assert_called_once_with(
            "https://proxy-api.example.com/v1/proxies",
            headers={},
            params={},
            auth=None,
            timeout=30
        )
        
        # Test authentication
        mock_get.reset_mock()
        config["auth"] = {
            "username": "user",
            "password": "pass"
        }
        
        provider = APIProxyProvider("test", config)
        proxies = provider.get_proxies()
        
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        self.assertEqual(call_kwargs["auth"], ("user", "pass"))


class TestProxyStrategies(unittest.TestCase):
    """Test the various proxy rotation strategies."""
    
    def setUp(self):
        """Set up test proxies for use in strategy tests."""
        self.proxies = [
            Proxy("192.168.1.1", 8080),
            Proxy("192.168.1.2", 8080),
            Proxy("192.168.1.3", 8080),
            Proxy("192.168.1.4", 8080),
            Proxy("192.168.1.5", 8080)
        ]
        
        # Set up some proxy metrics
        self.proxies[0].record_success(0.5)
        self.proxies[0].record_success(0.5)
        
        self.proxies[1].record_success(1.0)
        self.proxies[1].record_failure()
        
        self.proxies[2].record_failure()
        self.proxies[2].record_failure()
        self.proxies[2].blacklist("testing")
        
        self.proxies[3].record_success(0.2)
        self.proxies[3].last_used = datetime.datetime.now() - datetime.timedelta(hours=2)
        
        self.proxies[4].metadata = {"country": "US"}
    
    def test_round_robin_strategy(self):
        """Test that RoundRobinStrategy works correctly."""
        strategy = RoundRobinStrategy("test")
        
        # First selection
        proxy = strategy.select_proxy(self.proxies)
        self.assertEqual(proxy.address, "192.168.1.1")
        
        # Next selection
        proxy = strategy.select_proxy(self.proxies)
        self.assertEqual(proxy.address, "192.168.1.2")
        
        # Skip blacklisted proxy
        proxy = strategy.select_proxy(self.proxies)
        self.assertNotEqual(proxy.address, "192.168.1.3")
        
        # Complete the cycle
        proxy = strategy.select_proxy(self.proxies)
        proxy = strategy.select_proxy(self.proxies)
        
        # Back to the beginning
        proxy = strategy.select_proxy(self.proxies)
        self.assertEqual(proxy.address, "192.168.1.1")
    
    def test_random_strategy(self):
        """Test that RandomStrategy works correctly."""
        strategy = RandomStrategy("test")
        
        # Multiple selections
        selected_addresses = set()
        for _ in range(20):
            proxy = strategy.select_proxy(self.proxies)
            self.assertIsNotNone(proxy)
            self.assertNotEqual(proxy.address, "192.168.1.3")  # Blacklisted
            selected_addresses.add(proxy.address)
            
        # Should have selected multiple different proxies
        self.assertGreater(len(selected_addresses), 1)
    
    def test_weighted_strategy(self):
        """Test that WeightedStrategy works correctly."""
        strategy = WeightedStrategy("test")
        
        # Track selections
        selections = {"192.168.1.1": 0, "192.168.1.2": 0, "192.168.1.4": 0, "192.168.1.5": 0}
        
        # Multiple selections
        for _ in range(100):
            proxy = strategy.select_proxy(self.proxies)
            self.assertIsNotNone(proxy)
            self.assertNotEqual(proxy.address, "192.168.1.3")  # Blacklisted
            selections[proxy.address] += 1
            
        # Proxy 1 should be selected more often (higher success rate)
        self.assertGreater(selections["192.168.1.1"], selections["192.168.1.2"])
    
    def test_least_used_strategy(self):
        """Test that LeastUsedStrategy works correctly."""
        strategy = LeastUsedStrategy("test")
        
        # First selection should be the least recently used
        proxy = strategy.select_proxy(self.proxies)
        self.assertEqual(proxy.address, "192.168.1.5")  # Never used
        
        # Mark it as used
        proxy.last_used = datetime.datetime.now()
        
        # Next selection
        proxy = strategy.select_proxy(self.proxies)
        self.assertEqual(proxy.address, "192.168.1.4")  # Used 2 hours ago
    
    def test_geo_match_strategy(self):
        """Test that GeoMatchStrategy works correctly."""
        strategy = GeoMatchStrategy("test")
        
        # Context with country
        context = {"country": "US"}
        proxy = strategy.select_proxy(self.proxies, context)
        self.assertEqual(proxy.address, "192.168.1.5")  # Has US country metadata
        
        # Context with unknown country
        context = {"country": "UK"}
        proxy = strategy.select_proxy(self.proxies, context)
        self.assertIsNotNone(proxy)  # Should fall back to random
    
    def test_adaptive_strategy(self):
        """Test that AdaptiveStrategy works correctly."""
        strategy = AdaptiveStrategy("test")
        
        # First selection with domain
        context = {"domain": "example.com"}
        proxy1 = strategy.select_proxy(self.proxies, context)
        self.assertIsNotNone(proxy1)
        
        # Update performance
        strategy.update_performance(proxy1, "example.com", True)
        
        # Second selection with same domain
        proxy2 = strategy.select_proxy(self.proxies, context)
        
        # Update performance negatively
        strategy.update_performance(proxy2, "example.com", False)
        
        # Third selection
        proxy3 = strategy.select_proxy(self.proxies, context)
        
        # First proxy should be preferred due to positive performance
        selections = {"192.168.1.1": 0, "192.168.1.2": 0, "192.168.1.4": 0, "192.168.1.5": 0}
        
        # Multiple selections
        for _ in range(50):
            proxy = strategy.select_proxy(self.proxies, context)
            selections[proxy.address] += 1
            
        # First proxy should be selected more often
        self.assertGreater(selections[proxy1.address], selections[proxy2.address])


class TestProxyManager(unittest.TestCase):
    """Test the ProxyManager service."""
    
    def setUp(self):
        """Set up a mock proxy manager for testing."""
        self.manager = ProxyManager()
        
        # Patch the health check thread to prevent actual threading
        patcher = patch.object(ProxyManager, '_start_health_check_thread')
        self.addCleanup(patcher.stop)
        self.mock_start_thread = patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        if self.manager._initialized:
            self.manager.shutdown()
    
    def test_initialization(self):
        """Test that the proxy manager initializes correctly."""
        config = {
            "enable_health_checks": False,
            "max_failures": 3,
            "default_strategy": "random",
            "providers": {
                "static": {
                    "type": "static",
                    "proxies": ["http://192.168.1.1:8080"]
                }
            }
        }
        
        self.manager.initialize(config)
        self.assertTrue(self.manager._initialized)
        self.assertEqual(self.manager._blacklist_threshold, 3)
        self.assertEqual(self.manager._current_strategy.name, "random")
        self.assertEqual(len(self.manager._proxies), 1)
        
    def test_proxy_selection(self):
        """Test that proxy selection works correctly."""
        config = {
            "enable_health_checks": False,
            "providers": {
                "static": {
                    "type": "static",
                    "proxies": [
                        "http://192.168.1.1:8080",
                        "http://192.168.1.2:8080"
                    ]
                }
            }
        }
        
        self.manager.initialize(config)
        
        # Select a proxy
        proxy = self.manager.get_proxy()
        self.assertIsNotNone(proxy)
        self.assertIn(proxy.address, ["192.168.1.1", "192.168.1.2"])
        
        # Select with strategy
        proxy = self.manager.get_proxy(strategy_name="random")
        self.assertIsNotNone(proxy)
        
        # Select with context
        proxy = self.manager.get_proxy(context={"domain": "example.com"})
        self.assertIsNotNone(proxy)
        
        # Verify proxies get rotated
        selections = set()
        for _ in range(10):
            proxy = self.manager.get_proxy(strategy_name="round_robin")
            selections.add(proxy.address)
        
        self.assertEqual(len(selections), 2)  # Both proxies should be used
    
    def test_blacklisting(self):
        """Test that proxy blacklisting works correctly."""
        config = {
            "enable_health_checks": False,
            "max_failures": 2,
            "providers": {
                "static": {
                    "type": "static",
                    "proxies": ["http://192.168.1.1:8080"]
                }
            }
        }
        
        self.manager.initialize(config)
        
        # Get the only proxy
        proxy = self.manager.get_proxy()
        self.assertTrue(proxy.is_active)
        
        # Report failures
        self.manager.report_result(proxy, False, error_type="test")
        self.assertTrue(proxy.is_active)  # 1 failure, not blacklisted yet
        
        self.manager.report_result(proxy, False, error_type="test")
        self.assertFalse(proxy.is_active)  # 2 failures, should be blacklisted
        
        # No active proxies
        proxy = self.manager.get_proxy()
        self.assertIsNone(proxy)
        
        # Unblacklist
        self.manager.unblacklist_all()
        proxy = self.manager.get_proxy()
        self.assertIsNotNone(proxy)
        self.assertTrue(proxy.is_active)
    
    def test_refresh_proxies(self):
        """Test that proxy refreshing works correctly."""
        # Start with an empty provider
        config = {
            "enable_health_checks": False,
            "providers": {
                "static": {
                    "type": "static",
                    "proxies": []
                }
            }
        }
        
        self.manager.initialize(config)
        self.assertEqual(len(self.manager._proxies), 0)
        
        # Add a proxy to the provider
        self.manager._providers["static"] = MagicMock()
        self.manager._providers["static"].get_proxies.return_value = [
            Proxy("192.168.1.1", 8080)
        ]
        
        # Refresh proxies
        self.manager.refresh_proxies()
        self.assertEqual(len(self.manager._proxies), 1)
    
    def test_proxy_performance_tracking(self):
        """Test that proxy performance tracking works correctly."""
        config = {
            "enable_health_checks": False,
            "providers": {
                "static": {
                    "type": "static",
                    "proxies": ["http://192.168.1.1:8080"]
                }
            }
        }
        
        self.manager.initialize(config)
        proxy = self.manager.get_proxy()
        
        # Report success
        self.manager.report_result(proxy, True, response_time=0.5)
        self.assertEqual(proxy.success_count, 1)
        self.assertEqual(proxy.total_response_time, 0.5)
        
        # Report failure
        self.manager.report_result(proxy, False, error_type="connection_error")
        self.assertEqual(proxy.failure_count, 1)
        self.assertEqual(proxy.error_types.get("connection_error"), 1)
    
    @patch.object(ProxyManager, '_verify_proxy')
    def test_health_checks(self, mock_verify):
        """Test that proxy health checks work correctly."""
        config = {
            "enable_health_checks": True,  # Enable health checks
            "providers": {
                "static": {
                    "type": "static",
                    "proxies": [
                        "http://192.168.1.1:8080", 
                        "http://192.168.1.2:8080"
                    ]
                }
            }
        }
        
        # Mock verification results
        def verify_side_effect(proxy):
            return proxy.address == "192.168.1.1"  # Only first proxy is valid
            
        mock_verify.side_effect = verify_side_effect
        
        # Initialize with health checks
        self.manager.initialize(config)
        self.mock_start_thread.assert_called_once()
        
        # Run health check manually
        self.manager._check_proxies()
        
        # Verify that only one proxy is active
        active_proxies = [p for p in self.manager._proxies if p.is_active]
        self.assertEqual(len(active_proxies), 1)
        self.assertEqual(active_proxies[0].address, "192.168.1.1")


if __name__ == '__main__':
    unittest.main()