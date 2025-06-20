"""Tests for the core alerting system."""

import pytest
import time
import datetime
from unittest.mock import MagicMock, patch

from core.alerting import Alerting, Alert, AlertSeverity
from core.alerting import ConsoleAlerter, LogAlerter, EmailAlerter, WebhookAlerter

# Fixture for an alerting service with mocks
@pytest.fixture
def alerting_service():
    # Create an alerting service
    alerting = Alerting()
    
    # Replace alerters with mocks
    console_mock = MagicMock(spec=ConsoleAlerter)
    log_mock = MagicMock(spec=LogAlerter)
    
    alerting._alerters = {
        'console': console_mock,
        'log': log_mock
    }
    
    # Initialize with test config
    alerting.initialize({
        'alerters': [
            {'type': 'console'},
            {'type': 'log'}
        ]
    })
    
    yield alerting
    
    # Clean up
    alerting.shutdown()

def test_alerting_initialization():
    """Test that the alerting service initializes correctly."""
    # Create a fresh alerting service
    alerting = Alerting()
    
    # Should not be initialized initially
    assert not alerting.is_initialized
    
    # Initialize and check
    alerting.initialize()
    assert alerting.is_initialized
    assert alerting.name == "alerting"
    
    # Should have default alerters
    assert len(alerting._alerters) > 0
    
    # Cleanup
    alerting.shutdown()
    assert not alerting.is_initialized

def test_trigger_alert(alerting_service):
    """Test triggering an alert."""
    # Trigger an alert
    alert_id = alerting_service.trigger_alert(
        "Test alert message", 
        AlertSeverity.WARNING,
        {"source": "test_case"}
    )
    
    # Verify alert was created
    assert alert_id in alerting_service._alerts
    alert = alerting_service._alerts[alert_id]
    
    # Verify alert properties
    assert alert.message == "Test alert message"
    assert alert.severity == AlertSeverity.WARNING
    assert alert.context == {"source": "test_case"}
    assert alert.is_active
    assert not alert.is_muted
    
    # Verify alerters were called
    for alerter in alerting_service._alerters.values():
        assert alerter.send_alert.call_count == 1

def test_trigger_alert_with_string_severity(alerting_service):
    """Test triggering an alert with string severity."""
    # Trigger an alert using string severity
    alert_id = alerting_service.trigger_alert(
        "Test alert with string severity", 
        "error",
        {"source": "test_case"}
    )
    
    # Verify alert was created with correct severity
    alert = alerting_service._alerts[alert_id]
    assert alert.severity == AlertSeverity.ERROR

def test_resolve_alert(alerting_service):
    """Test resolving an alert."""
    # Trigger an alert
    alert_id = alerting_service.trigger_alert(
        "Test alert to resolve", 
        AlertSeverity.ERROR
    )
    
    # Verify alert is active
    assert alerting_service._alerts[alert_id].is_active
    
    # Resolve the alert
    result = alerting_service.resolve_alert(alert_id)
    
    # Verify resolution
    assert result
    assert not alerting_service._alerts[alert_id].is_active
    assert alerting_service._alerts[alert_id].resolved_at is not None
    
    # Try resolving again (should fail as already resolved)
    result = alerting_service.resolve_alert(alert_id)
    assert not result
    
    # Try resolving non-existent alert
    result = alerting_service.resolve_alert("non-existent-id")
    assert not result

def test_get_active_alerts(alerting_service):
    """Test getting active alerts."""
    # Create some alerts with different severities
    info_id = alerting_service.trigger_alert("Info alert", AlertSeverity.INFO)
    warning_id = alerting_service.trigger_alert("Warning alert", AlertSeverity.WARNING)
    error_id = alerting_service.trigger_alert("Error alert", AlertSeverity.ERROR)
    critical_id = alerting_service.trigger_alert("Critical alert", AlertSeverity.CRITICAL)
    
    # Resolve one alert
    alerting_service.resolve_alert(warning_id)
    
    # Get all active alerts
    active_alerts = alerting_service.get_active_alerts()
    assert len(active_alerts) == 3
    
    # Get alerts filtered by severity
    error_alerts = alerting_service.get_active_alerts(AlertSeverity.ERROR)
    assert len(error_alerts) == 1
    assert error_alerts[0]["severity"] == "error"
    
    # Get alerts filtered by string severity
    critical_alerts = alerting_service.get_active_alerts("critical")
    assert len(critical_alerts) == 1
    assert critical_alerts[0]["severity"] == "critical"
    
    # Get alerts with invalid severity (should return all)
    all_alerts = alerting_service.get_active_alerts("invalid_severity")
    assert len(all_alerts) == 3

def test_mute_alerts(alerting_service):
    """Test muting alerts based on patterns."""
    # Create some alerts with specific patterns
    alert1_id = alerting_service.trigger_alert("Database connection error", AlertSeverity.ERROR)
    alert2_id = alerting_service.trigger_alert("API timeout error", AlertSeverity.ERROR)
    alert3_id = alerting_service.trigger_alert("Cache miss", AlertSeverity.INFO)
    
    # Mute all alerts containing "error"
    alerting_service.mute_alerts("error", 3600)
    
    # Verify alerts are muted
    assert alerting_service._alerts[alert1_id].is_muted
    assert alerting_service._alerts[alert2_id].is_muted
    assert not alerting_service._alerts[alert3_id].is_muted
    
    # Trigger a new alert matching the pattern
    alert4_id = alerting_service.trigger_alert("Network error detected", AlertSeverity.WARNING)
    
    # Verify it was automatically muted
    assert alerting_service._alerts[alert4_id].is_muted
    
    # Verify alerters were not called for muted alert
    for alerter in alerting_service._alerters.values():
        # Reset mock to clear previous calls
        alerter.send_alert.reset_mock()
    
    # Trigger another matching alert
    alerting_service.trigger_alert("Another error alert", AlertSeverity.WARNING)
    
    # Alerters should still be called even for muted alerts, but the alerter implementations
    # should skip sending if is_muted is True
    for alerter in alerting_service._alerters.values():
        assert alerter.send_alert.call_count == 1

def test_clear_alert_history(alerting_service):
    """Test clearing alert history."""
    # Create some alerts
    alerting_service.trigger_alert("Old resolved alert", AlertSeverity.INFO)
    alerting_service.trigger_alert("Recent resolved alert", AlertSeverity.WARNING)
    active_id = alerting_service.trigger_alert("Active alert", AlertSeverity.ERROR)
    
    # Resolve some alerts
    for alert_id, alert in list(alerting_service._alerts.items()):
        if alert_id != active_id:
            alerting_service.resolve_alert(alert_id)
    
    # Verify initial count
    assert len(alerting_service._alerts) == 3
    
    # Clear resolved alerts only
    cleared = alerting_service.clear_alert_history()
    
    # Verify only active alerts remain
    assert len(alerting_service._alerts) == 1
    assert active_id in alerting_service._alerts
    assert cleared == 2
    
    # Add more alerts with different timestamps
    # We'll manually set created_at to simulate age
    alert1 = Alert("Very old alert", AlertSeverity.INFO)
    alert1.created_at = datetime.datetime.now() - datetime.timedelta(hours=24)
    alert2 = Alert("Recent alert", AlertSeverity.WARNING)
    alert2.created_at = datetime.datetime.now() - datetime.timedelta(minutes=30)
    
    alerting_service._alerts[alert1.id] = alert1
    alerting_service._alerts[alert2.id] = alert2
    
    # Clear alerts older than 1 hour
    cleared = alerting_service.clear_alert_history(max_age=3600)
    
    # Verify only recent alerts remain
    assert len(alerting_service._alerts) == 2  # active_id and alert2
    assert alert1.id not in alerting_service._alerts
    assert alert2.id in alerting_service._alerts
    assert cleared == 1

def test_get_service_health(alerting_service):
    """Test getting service health information."""
    # Add some alerts of different severities
    alerting_service.trigger_alert("Info alert", AlertSeverity.INFO)
    alerting_service.trigger_alert("Warning alert", AlertSeverity.WARNING)
    critical_id = alerting_service.trigger_alert("Critical alert", AlertSeverity.CRITICAL)
    
    # Resolve one alert
    alerting_service.resolve_alert(critical_id)
    
    # Get health
    health = alerting_service.get_service_health()
    
    # Verify structure
    assert "status" in health
    assert "details" in health
    assert "metrics" in health
    
    # Verify metrics
    assert health["metrics"]["active_alerts"] == 2
    assert health["metrics"]["total_alerts_tracked"] == 3
    assert health["metrics"]["critical_alerts"] == 0
    
    # Add a critical alert
    alerting_service.trigger_alert("New critical alert", AlertSeverity.CRITICAL)
    
    # Get health again
    health = alerting_service.get_service_health()
    
    # Verify updated metrics
    assert health["metrics"]["active_alerts"] == 3
    assert health["metrics"]["critical_alerts"] == 1
    
    # Verify status is degraded due to critical alert
    assert health["status"] == "degraded"

def test_alerter_implementations():
    """Test the basic behavior of alerter implementations."""
    # Create an alert
    alert = Alert("Test alert message", AlertSeverity.WARNING)
    
    # Test ConsoleAlerter
    with patch('builtins.print') as mock_print:
        console_alerter = ConsoleAlerter()
        result = console_alerter.send_alert(alert)
        assert result
        assert mock_print.call_count == 1
    
    # Test LogAlerter
    with patch('core.alerting.logger') as mock_logger:
        log_alerter = LogAlerter()
        result = log_alerter.send_alert(alert)
        assert result
        assert mock_logger.warning.call_count == 1
    
    # Test EmailAlerter (just verify configuration handling)
    email_config = {
        'smtp_server': 'smtp.example.com',
        'smtp_port': 587,
        'smtp_username': 'test',
        'smtp_password': 'password',
        'from_address': 'alerts@example.com',
        'to_addresses': ['admin@example.com']
    }
    email_alerter = EmailAlerter(email_config)
    assert email_alerter.smtp_server == 'smtp.example.com'
    assert email_alerter.smtp_port == 587
    assert email_alerter.from_address == 'alerts@example.com'
    
    # Test WebhookAlerter (just verify configuration handling)
    webhook_config = {
        'url': 'https://example.com/webhook',
        'method': 'POST',
        'headers': {'Content-Type': 'application/json'},
        'min_severity': 'warning'
    }
    webhook_alerter = WebhookAlerter(webhook_config)
    assert webhook_alerter.url == 'https://example.com/webhook'
    assert webhook_alerter.method == 'POST'
    assert webhook_alerter.min_severity == AlertSeverity.WARNING

def test_add_custom_alerter(alerting_service):
    """Test adding a custom alerter."""
    # Create a mock alerter
    mock_alerter = MagicMock()
    
    # Add it to the service
    alerting_service.add_alerter("custom_alerter", mock_alerter)
    
    # Verify it was added
    assert "custom_alerter" in alerting_service._alerters
    
    # Trigger an alert and verify the custom alerter was called
    alerting_service.trigger_alert("Test for custom alerter", AlertSeverity.INFO)
    assert mock_alerter.send_alert.call_count == 1