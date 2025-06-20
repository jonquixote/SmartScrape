"""
End-to-end tests for SmartScrape error handling system.

Tests the complete error flow through ErrorClassifier, AlertService, 
RetryManager and CircuitBreaker components, as well as validation of
error classification, reporting, and recovery mechanisms.
"""

import logging
import pytest
import time
import requests
import random
import os
import json
from unittest.mock import patch, MagicMock, PropertyMock

from core.service_registry import ServiceRegistry
from core.error_classifier import ErrorClassifier
from core.alerting import AlertService
from core.retry_manager import RetryManager
from core.circuit_breaker import CircuitBreakerManager, OpenCircuitError
from core.monitoring import MonitoringService

# Configure logging for tests
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class TestErrorHandlingFlow:
    """Test end-to-end flow through error handling components."""
    
    @pytest.fixture
    def service_registry(self):
        """Set up service registry with all required services."""
        registry = ServiceRegistry()
        
        # Initialize and register all necessary services
        error_classifier = ErrorClassifier()
        error_config = {
            'error_categories': {
                'network': [
                    'ConnectionError', 
                    'Timeout', 
                    'SSLError'
                ],
                'client': [
                    'HTTPError:400', 
                    'HTTPError:401', 
                    'HTTPError:403', 
                    'HTTPError:404'
                ],
                'server': [
                    'HTTPError:500', 
                    'HTTPError:502', 
                    'HTTPError:503', 
                    'HTTPError:504'
                ],
                'rate_limiting': [
                    'HTTPError:429', 
                    'TooManyRedirects'
                ],
                'content': [
                    'ContentParsingError', 
                    'EmptyResponseError', 
                    'InvalidSchemaError'
                ]
            }
        }
        error_classifier.initialize(error_config)
        registry.register_service(error_classifier)
        
        alert_service = AlertService()
        alert_config = {
            'channels': {
                'log': {'enabled': True, 'level': 'ERROR'},
                'email': {'enabled': False},
                'slack': {'enabled': False}
            },
            'alert_thresholds': {
                'network': 10,
                'client': 5,
                'server': 5,
                'rate_limiting': 3,
                'content': 5
            },
            'alert_cooldown': 60  # seconds
        }
        alert_service.initialize(alert_config)
        registry.register_service(alert_service)
        
        retry_manager = RetryManager()
        retry_config = {
            'default_policy': {
                'max_attempts': 3,
                'backoff_factor': 0.5,
                'jitter': True
            },
            'category_policies': {
                'network': {
                    'max_attempts': 5,
                    'backoff_factor': 1.0
                },
                'server': {
                    'max_attempts': 4,
                    'backoff_factor': 2.0
                },
                'rate_limiting': {
                    'max_attempts': 2,
                    'backoff_factor': 5.0
                }
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
            },
            'domain_settings': {
                'flaky-site.example.com': {
                    'failure_threshold': 3,
                    'reset_timeout': 10
                }
            }
        }
        circuit_breaker.initialize(circuit_config)
        registry.register_service(circuit_breaker)
        
        monitoring_service = MonitoringService()
        monitoring_config = {
            'metrics_enabled': True,
            'logging_level': 'INFO'
        }
        monitoring_service.initialize(monitoring_config)
        registry.register_service(monitoring_service)
        
        return registry
    
    def test_error_classification_flow(self, service_registry, monkeypatch):
        """Test complete error classification and handling flow."""
        # Get services
        error_classifier = service_registry.get_service('error_classifier')
        alert_service = service_registry.get_service('alert_service')
        retry_manager = service_registry.get_service('retry_manager')
        circuit_breaker = service_registry.get_service('circuit_breaker_manager')
        
        # Set up mock alerting to track triggered alerts
        alerts = []
        
        def mock_send_alert(category, message, context=None):
            alerts.append({
                'category': category,
                'message': message,
                'context': context
            })
            logger.info(f"Alert triggered: {category} - {message}")
            
        monkeypatch.setattr(alert_service, 'send_alert', mock_send_alert)
        
        # Test errors for different categories
        test_errors = [
            # Network errors
            requests.exceptions.ConnectionError("Failed to establish connection"),
            requests.exceptions.Timeout("Request timed out"),
            requests.exceptions.SSLError("SSL certificate verification failed"),
            
            # Client errors
            requests.exceptions.HTTPError("404 Client Error: Not Found"),
            requests.exceptions.HTTPError("403 Client Error: Forbidden"),
            
            # Server errors
            requests.exceptions.HTTPError("500 Server Error: Internal Server Error"),
            requests.exceptions.HTTPError("503 Server Error: Service Unavailable"),
            
            # Rate limiting
            requests.exceptions.HTTPError("429 Client Error: Too Many Requests"),
            requests.exceptions.TooManyRedirects("Exceeded redirect limit"),
            
            # Content errors
            Exception("ContentParsingError: Failed to parse JSON response"),
            Exception("EmptyResponseError: Empty response received")
        ]
        
        classification_results = []
        
        # Test error classification
        for error in test_errors:
            # Classify the error
            error_category = error_classifier.classify_error(error)
            
            # Track classification result
            classification_results.append({
                'error': str(error),
                'category': error_category
            })
            
            # Simulate retry policy selection based on error category
            retry_policy = retry_manager.get_policy_for_error_category(error_category)
            
            # Trigger alert if needed
            alert_service.record_error(error_category, str(error), {
                'error_type': type(error).__name__,
                'timestamp': time.time()
            })
            
            # For testing, force an alert by simulating multiple errors of the same type
            for _ in range(alert_service._alert_thresholds.get(error_category, 10) + 1):
                alert_service.record_error(error_category, str(error), {
                    'error_type': type(error).__name__,
                    'timestamp': time.time()
                })
        
        # Log classification results
        logger.info("Error classification results:")
        for result in classification_results:
            logger.info(f"Error: {result['error']} -> Category: {result['category']}")
        
        # Verify alerts were triggered for each category
        logger.info(f"Alerts triggered: {len(alerts)}")
        
        alert_categories = set(alert['category'] for alert in alerts)
        expected_categories = {'network', 'client', 'server', 'rate_limiting', 'content'}
        
        # Log alert categories
        logger.info(f"Alert categories: {alert_categories}")
        
        # Verify all expected categories triggered alerts
        for category in expected_categories:
            assert any(alert['category'] == category for alert in alerts), \
                f"No alert triggered for category: {category}"
    
    def test_retry_policies_by_error_type(self, service_registry, monkeypatch):
        """Test different retry policies based on error type."""
        # Get services
        error_classifier = service_registry.get_service('error_classifier')
        retry_manager = service_registry.get_service('retry_manager')
        
        # Function to count retries based on error type
        def count_retries_for_error(error_instance):
            # Classify the error
            error_category = error_classifier.classify_error(error_instance)
            
            # Track retry attempts
            attempts = {'count': 0}
            success = {'achieved': False}
            
            # Create a function to test retry behavior
            @retry_manager.retry_by_error_category(error_category)
            def test_function():
                attempts['count'] += 1
                
                # Succeed after specific number of attempts based on category
                if error_category == 'network' and attempts['count'] >= 4:
                    success['achieved'] = True
                    return "Success after network errors"
                elif error_category == 'server' and attempts['count'] >= 3:
                    success['achieved'] = True
                    return "Success after server errors"
                elif error_category == 'rate_limiting' and attempts['count'] >= 2:
                    success['achieved'] = True
                    return "Success after rate limiting"
                elif attempts['count'] >= 2:  # Default for other categories
                    success['achieved'] = True
                    return "Success after other errors"
                
                # Otherwise, raise the error again
                raise error_instance
            
            # Execute function with retries
            try:
                result = test_function()
                return {
                    'error_type': type(error_instance).__name__,
                    'category': error_category,
                    'attempts': attempts['count'],
                    'success': True,
                    'result': result
                }
            except Exception as e:
                return {
                    'error_type': type(error_instance).__name__,
                    'category': error_category,
                    'attempts': attempts['count'],
                    'success': False,
                    'exception': str(e)
                }
        
        # Test different error types
        test_errors = [
            ('network', requests.exceptions.ConnectionError("Connection refused")),
            ('network', requests.exceptions.Timeout("Request timed out")),
            ('client', requests.exceptions.HTTPError("404 Client Error: Not Found")),
            ('server', requests.exceptions.HTTPError("500 Server Error: Internal Server Error")),
            ('server', requests.exceptions.HTTPError("503 Server Error: Service Unavailable")),
            ('rate_limiting', requests.exceptions.HTTPError("429 Client Error: Too Many Requests")),
            ('content', Exception("ContentParsingError: Invalid JSON"))
        ]
        
        retry_results = []
        
        # Test retry behavior for each error
        for expected_category, error in test_errors:
            result = count_retries_for_error(error)
            retry_results.append(result)
            
            # Verify error was correctly classified
            assert result['category'] == expected_category, \
                f"Error {error} classified as {result['category']}, expected {expected_category}"
        
        # Log retry results
        logger.info("Retry policy results by error category:")
        for result in retry_results:
            logger.info(f"Error type: {result['error_type']}, Category: {result['category']}, " +
                       f"Attempts: {result['attempts']}, Success: {result['success']}")
        
        # Verify retry count matched policy for each category
        network_result = next(r for r in retry_results if r['category'] == 'network')
        assert network_result['attempts'] >= 4, "Network errors should retry at least 4 times"
        
        server_result = next(r for r in retry_results if r['category'] == 'server')
        assert server_result['attempts'] >= 3, "Server errors should retry at least 3 times"
        
        rate_limiting_result = next(r for r in retry_results if r['category'] == 'rate_limiting')
        assert rate_limiting_result['attempts'] >= 2, "Rate limiting errors should retry at least 2 times"
    
    def test_error_recovery_strategies(self, service_registry, monkeypatch):
        """Test recovery strategies for different error types."""
        # Get services
        error_classifier = service_registry.get_service('error_classifier')
        retry_manager = service_registry.get_service('retry_manager')
        circuit_breaker = service_registry.get_service('circuit_breaker_manager')
        
        # Create mock session to simulate different errors
        class MockSession:
            def __init__(self):
                self.call_count = 0
                self.errors = []
                
            def get(self, url, **kwargs):
                self.call_count += 1
                domain = url.split('//', 1)[1].split('/', 1)[0] if '//' in url else 'unknown'
                
                # Check if we have specific errors to raise
                if self.errors:
                    error = self.errors.pop(0)
                    if error:
                        raise error
                
                # Otherwise return success response
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = f'{{"success": true, "domain": "{domain}"}}'
                return mock_response
                
            def add_error(self, error):
                self.errors.append(error)
                
            def clear_errors(self):
                self.errors = []
        
        # Create test domains and URLs
        domains = {
            'network_issues': 'network-test.example.com',
            'client_errors': 'client-error.example.com',
            'server_errors': 'server-error.example.com',
            'rate_limited': 'rate-limited.example.com',
            'content_issues': 'content-issue.example.com'
        }
        
        # Create mock session for testing
        mock_session = MockSession()
        
        # Define recovery strategies for different error types
        recovery_strategies = {
            'network': lambda domain, err: {
                'action': 'retry_with_backoff',
                'message': f"Network error for {domain}, retrying with backoff"
            },
            'client': lambda domain, err: {
                'action': 'check_url_validity', 
                'message': f"Client error for {domain}, checking URL validity"
            },
            'server': lambda domain, err: {
                'action': 'circuit_break_if_persistent',
                'message': f"Server error for {domain}, monitoring for persistent issues"
            },
            'rate_limiting': lambda domain, err: {
                'action': 'delay_and_reduce_concurrency',
                'message': f"Rate limit hit for {domain}, reducing request rate"
            },
            'content': lambda domain, err: {
                'action': 'try_alternative_parsing',
                'message': f"Content error for {domain}, trying alternative parsing"
            }
        }
        
        # Function to process requests with error recovery
        def process_with_recovery(domain, error_to_inject=None):
            url = f"https://{domain}/api"
            
            # Clear any previous errors
            mock_session.clear_errors()
            
            # Add error to inject if specified
            if error_to_inject:
                mock_session.add_error(error_to_inject)
            
            # Get circuit breaker for domain
            cb = circuit_breaker.get_circuit_breaker(domain)
            
            # Check if circuit is open
            if not cb.allow_request():
                return {
                    'domain': domain,
                    'success': False,
                    'recovery': {
                        'action': 'circuit_open',
                        'message': f"Circuit is open for {domain}, skipping request"
                    }
                }
            
            try:
                # Try to make the request with retry handling
                @retry_manager.retry(domain)
                def execute_request():
                    response = mock_session.get(url)
                    return response
                
                response = execute_request()
                
                # Record success
                cb.record_success()
                
                return {
                    'domain': domain,
                    'success': True,
                    'status_code': response.status_code
                }
                
            except Exception as e:
                # Classify the error
                error_category = error_classifier.classify_error(e)
                
                # Record failure in circuit breaker
                cb.record_failure()
                
                # Get recovery strategy based on error category
                recovery = recovery_strategies.get(error_category, lambda d, err: {
                    'action': 'default_recovery',
                    'message': f"Unknown error for {d}, using default recovery"
                })(domain, e)
                
                return {
                    'domain': domain,
                    'success': False,
                    'error': str(e),
                    'category': error_category,
                    'recovery': recovery
                }
        
        # Test recovery for different error types
        test_scenarios = [
            (domains['network_issues'], requests.exceptions.ConnectionError("Connection refused")),
            (domains['client_errors'], requests.exceptions.HTTPError("404 Client Error: Not Found")),
            (domains['server_errors'], requests.exceptions.HTTPError("500 Server Error: Internal Server Error")),
            (domains['rate_limited'], requests.exceptions.HTTPError("429 Client Error: Too Many Requests")),
            (domains['content_issues'], Exception("ContentParsingError: Invalid JSON"))
        ]
        
        recovery_results = []
        
        # Run tests for each scenario
        for domain, error in test_scenarios:
            result = process_with_recovery(domain, error)
            recovery_results.append(result)
        
        # Log recovery results
        logger.info("Error recovery strategy results:")
        for result in recovery_results:
            if not result['success']:
                logger.info(f"Domain: {result['domain']}, Error: {result.get('error', 'N/A')}, " +
                           f"Category: {result.get('category', 'N/A')}, " +
                           f"Recovery: {result['recovery']['action']}")
            else:
                logger.info(f"Domain: {result['domain']}, Request succeeded")
        
        # Verify recovery strategies were applied correctly
        for result in recovery_results:
            assert not result['success'], "Initial request should fail with injected error"
            
            # Verify recovery strategy matches error category
            if 'category' in result:
                expected_action = recovery_strategies[result['category']](result['domain'], None)['action']
                assert result['recovery']['action'] == expected_action, \
                    f"Expected recovery {expected_action} for {result['category']}, got {result['recovery']['action']}"
        
        # Test successful recovery after error resolution
        recovery_after_fix_results = []
        
        for domain, _ in test_scenarios:
            # No error injection this time - should succeed
            result = process_with_recovery(domain)
            recovery_after_fix_results.append(result)
        
        # Log recovery after fix results
        logger.info("Recovery results after error resolution:")
        for result in recovery_after_fix_results:
            if result['success']:
                logger.info(f"Domain: {result['domain']}, Successfully recovered")
            else:
                logger.info(f"Domain: {result['domain']}, Recovery failed: {result.get('recovery', {}).get('action', 'unknown')}")
        
        # Verify recovery was successful
        for result in recovery_after_fix_results:
            assert result['success'], f"Request should succeed after error resolution for {result['domain']}"
    
    def test_error_trend_detection(self, service_registry, monkeypatch):
        """Test the system's ability to detect error trends."""
        # Get services
        error_classifier = service_registry.get_service('error_classifier')
        alert_service = service_registry.get_service('alert_service')
        monitoring_service = service_registry.get_service('monitoring_service')
        
        # Set up trend tracking
        error_trends = {}
        trend_alerts = []
        
        # Mock trend detection in monitoring service
        def mock_record_error(domain, error_type, error_message):
            # Track error counts
            if domain not in error_trends:
                error_trends[domain] = {}
            
            if error_type not in error_trends[domain]:
                error_trends[domain][error_type] = 0
            
            error_trends[domain][error_type] += 1
            
            # Check for trends
            if error_trends[domain][error_type] >= 5:
                # Detect trend and generate alert
                trend_alert = {
                    'domain': domain,
                    'error_type': error_type,
                    'count': error_trends[domain][error_type],
                    'message': f"Error trend detected: {error_type} errors in {domain}"
                }
                trend_alerts.append(trend_alert)
                logger.warning(f"Error trend detected: {error_type} errors in {domain} ({error_trends[domain][error_type]} occurrences)")
            
            return True
            
        monkeypatch.setattr(monitoring_service, 'record_error', mock_record_error)
        
        # Function to simulate errors and record them
        def simulate_error_flow(domain, error_instance, count=1):
            # Classify the error
            error_category = error_classifier.classify_error(error_instance)
            error_type = type(error_instance).__name__
            
            for _ in range(count):
                # Record the error
                alert_service.record_error(error_category, str(error_instance), {
                    'domain': domain,
                    'error_type': error_type
                })
                
                # Monitor the error
                monitoring_service.record_error(domain, error_type, str(error_instance))
            
            return {
                'domain': domain,
                'error_type': error_type,
                'category': error_category,
                'count': count
            }
        
        # Test domains
        test_domains = {
            'spike': 'error-spike.example.com',
            'gradual': 'gradual-errors.example.com',
            'mixed': 'mixed-errors.example.com'
        }
        
        # Simulate a sudden spike in one error type
        spike_result = simulate_error_flow(
            test_domains['spike'],
            requests.exceptions.ConnectionError("Connection refused"),
            count=10
        )
        
        # Simulate gradual increase in another domain
        gradual_results = []
        for i in range(7):
            result = simulate_error_flow(
                test_domains['gradual'],
                requests.exceptions.HTTPError("500 Server Error: Internal Server Error")
            )
            gradual_results.append(result)
        
        # Simulate mixed errors in third domain
        mixed_results = []
        for _ in range(3):
            result = simulate_error_flow(
                test_domains['mixed'],
                requests.exceptions.HTTPError("429 Client Error: Too Many Requests")
            )
            mixed_results.append(result)
        
        for _ in range(3):
            result = simulate_error_flow(
                test_domains['mixed'],
                requests.exceptions.Timeout("Request timed out")
            )
            mixed_results.append(result)
        
        # Log trend detection results
        logger.info("Error trend detection results:")
        for domain, error_counts in error_trends.items():
            logger.info(f"Domain: {domain}")
            for error_type, count in error_counts.items():
                logger.info(f"  {error_type}: {count} occurrences")
        
        logger.info(f"Detected trends: {len(trend_alerts)}")
        for alert in trend_alerts:
            logger.info(f"  Trend alert: {alert['message']} ({alert['count']} occurrences)")
        
        # Verify trend detection
        assert any(alert['domain'] == test_domains['spike'] for alert in trend_alerts), \
            "Should detect trend for error spike domain"
        
        assert any(alert['domain'] == test_domains['gradual'] for alert in trend_alerts), \
            "Should detect trend for gradual errors domain"
        
        # May or may not detect trend for mixed errors, depending on threshold
        
        # Verify counts in error_trends match expected values
        assert error_trends[test_domains['spike']]['ConnectionError'] == 10, \
            "Should record 10 ConnectionError occurrences for spike domain"
        
        assert error_trends[test_domains['gradual']]['HTTPError'] == 7, \
            "Should record 7 HTTPError occurrences for gradual domain"


class TestErrorReporting:
    """Test error reporting and visualization components."""
    
    @pytest.fixture
    def service_registry(self):
        """Set up service registry with all required services."""
        registry = ServiceRegistry()
        
        # Initialize and register monitoring service
        monitoring_service = MonitoringService()
        monitoring_config = {
            'metrics_enabled': True,
            'logging_level': 'INFO',
            'persistence': {
                'enabled': True,
                'path': './test_error_metrics'
            }
        }
        monitoring_service.initialize(monitoring_config)
        registry.register_service(monitoring_service)
        
        # Initialize and register error classifier
        error_classifier = ErrorClassifier()
        error_config = {
            'error_categories': {
                'network': [
                    'ConnectionError', 
                    'Timeout', 
                    'SSLError'
                ],
                'client': [
                    'HTTPError:400', 
                    'HTTPError:401', 
                    'HTTPError:403', 
                    'HTTPError:404'
                ],
                'server': [
                    'HTTPError:500', 
                    'HTTPError:502', 
                    'HTTPError:503', 
                    'HTTPError:504'
                ],
                'rate_limiting': [
                    'HTTPError:429', 
                    'TooManyRedirects'
                ],
                'content': [
                    'ContentParsingError', 
                    'EmptyResponseError', 
                    'InvalidSchemaError'
                ]
            }
        }
        error_classifier.initialize(error_config)
        registry.register_service(error_classifier)
        
        return registry
    
    def setup_method(self):
        """Set up test environment."""
        # Create temporary directory for metrics
        os.makedirs('./test_error_metrics', exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        # Remove temporary metrics directory and files
        if os.path.exists('./test_error_metrics'):
            for file in os.listdir('./test_error_metrics'):
                os.remove(os.path.join('./test_error_metrics', file))
            os.rmdir('./test_error_metrics')
    
    def test_error_metrics_collection(self, service_registry, monkeypatch):
        """Test collection and persistence of error metrics."""
        # Get monitoring service
        monitoring_service = service_registry.get_service('monitoring_service')
        
        # Test domains
        domains = [
            'metrics-test1.example.com',
            'metrics-test2.example.com'
        ]
        
        # Error types to test
        error_types = [
            ('ConnectionError', 'Network error occurred'),
            ('HTTPError:404', 'Not found'),
            ('HTTPError:500', 'Internal server error'),
            ('HTTPError:429', 'Too many requests'),
            ('ContentParsingError', 'Failed to parse response')
        ]
        
        # Store metrics for verification
        expected_metrics = {}
        
        # Record various errors
        for domain in domains:
            domain_metrics = {}
            
            for error_name, error_message in error_types:
                # Record different counts for each error type
                count = random.randint(1, 10)
                
                for _ in range(count):
                    monitoring_service.record_error(domain, error_name, error_message)
                
                domain_metrics[error_name] = count
            
            expected_metrics[domain] = domain_metrics
        
        # Force metrics persistence
        monitoring_service.persist_metrics()
        
        # Verify metrics were persisted
        metrics_files = os.listdir('./test_error_metrics')
        
        logger.info(f"Metrics files: {metrics_files}")
        
        # There should be at least one metrics file
        assert len(metrics_files) > 0, "Should create metrics files"
        
        # Load and verify metrics
        for filename in metrics_files:
            if not filename.endswith('.json'):
                continue
                
            with open(os.path.join('./test_error_metrics', filename), 'r') as f:
                metrics_data = json.load(f)
            
            # Verify metrics data contains expected errors
            if 'domain_errors' in metrics_data:
                for domain, expected_counts in expected_metrics.items():
                    if domain in metrics_data['domain_errors']:
                        domain_data = metrics_data['domain_errors'][domain]
                        
                        for error_type, expected_count in expected_counts.items():
                            if error_type in domain_data:
                                assert domain_data[error_type] >= expected_count, \
                                    f"Expected at least {expected_count} {error_type} errors for {domain}"
        
        # Verify metrics can be retrieved from monitoring service
        error_summary = monitoring_service.get_error_summary()
        
        logger.info("Error metrics summary:")
        for domain, error_counts in error_summary.items():
            logger.info(f"Domain: {domain}")
            for error_type, count in error_counts.items():
                logger.info(f"  {error_type}: {count}")
        
        # Verify summary contains expected data
        for domain, expected_counts in expected_metrics.items():
            assert domain in error_summary, f"Domain {domain} should be in error summary"
            
            for error_type, expected_count in expected_counts.items():
                assert error_type in error_summary[domain], \
                    f"Error type {error_type} should be recorded for {domain}"
                assert error_summary[domain][error_type] >= expected_count, \
                    f"Expected at least {expected_count} {error_type} errors for {domain}"
    
    def test_error_classification_reporting(self, service_registry, monkeypatch):
        """Test error classification reporting."""
        # Get services
        error_classifier = service_registry.get_service('error_classifier')
        monitoring_service = service_registry.get_service('monitoring_service')
        
        # Test errors to classify
        test_errors = [
            requests.exceptions.ConnectionError("Connection refused"),
            requests.exceptions.Timeout("Request timed out"),
            requests.exceptions.SSLError("SSL verification failed"),
            requests.exceptions.HTTPError("404 Client Error: Not Found"),
            requests.exceptions.HTTPError("500 Server Error: Internal Server Error"),
            requests.exceptions.HTTPError("429 Client Error: Too Many Requests"),
            Exception("ContentParsingError: Invalid JSON")
        ]
        
        # Expected classifications
        expected_categories = [
            'network',
            'network',
            'network',
            'client',
            'server',
            'rate_limiting',
            'content'
        ]
        
        # Domain for testing
        domain = 'classification-test.example.com'
        
        # Track classifications
        error_categories = {}
        
        # Classify errors and record them
        for i, error in enumerate(test_errors):
            # Classify error
            category = error_classifier.classify_error(error)
            error_type = type(error).__name__
            
            # Verify classification
            assert category == expected_categories[i], \
                f"Error {error} should be classified as {expected_categories[i]}, got {category}"
            
            # Record error
            monitoring_service.record_error(domain, error_type, str(error))
            
            # Track category counts
            if category not in error_categories:
                error_categories[category] = 0
            error_categories[category] += 1
        
        # Get error summary by category
        category_summary = monitoring_service.get_error_categories()
        
        logger.info("Error category summary:")
        for category, count in category_summary.items():
            logger.info(f"  {category}: {count}")
        
        # Verify category counts
        for category, expected_count in error_categories.items():
            assert category in category_summary, f"Category {category} should be in summary"
            assert category_summary[category] >= expected_count, \
                f"Expected at least {expected_count} errors in {category} category"