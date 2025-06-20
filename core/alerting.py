"""
Alerting system for SmartScrape components.

This module provides classes for sending alerts when system components
experience issues or require attention.
"""

import logging
import threading
import time
import json
import smtplib
import email.message
import uuid
import datetime
import os
import requests
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List, Set, Callable, Union

from core.service_interface import BaseService

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"           # Informational alerts
    WARNING = "warning"     # Potential issues
    ERROR = "error"         # System problems
    CRITICAL = "critical"   # Severe problems

class Alert:
    """Represents a system alert."""
    
    def __init__(self, message: str, severity: AlertSeverity, context: Dict[str, Any] = None):
        self.id = str(uuid.uuid4())
        self.message = message
        self.severity = severity
        self.context = context or {}
        self.created_at = datetime.datetime.now()
        self.resolved_at = None
        self.muted_until = None
    
    @property
    def is_active(self) -> bool:
        """Check if the alert is active (not resolved)."""
        return self.resolved_at is None
    
    @property
    def is_muted(self) -> bool:
        """Check if the alert is currently muted."""
        if not self.muted_until:
            return False
        return datetime.datetime.now() < self.muted_until
    
    @property
    def age(self) -> datetime.timedelta:
        """Get the age of the alert."""
        return datetime.datetime.now() - self.created_at
    
    def resolve(self) -> None:
        """Mark the alert as resolved."""
        self.resolved_at = datetime.datetime.now()
    
    def mute(self, duration: int) -> None:
        """
        Mute the alert for a specified duration.
        
        Args:
            duration: Number of seconds to mute the alert
        """
        self.muted_until = datetime.datetime.now() + datetime.timedelta(seconds=duration)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the alert to a dictionary."""
        return {
            "id": self.id,
            "message": self.message,
            "severity": self.severity.value,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "muted_until": self.muted_until.isoformat() if self.muted_until else None,
            "is_active": self.is_active,
            "is_muted": self.is_muted,
            "age_seconds": self.age.total_seconds()
        }

class AlerterBase(ABC):
    """Base class for alert notification channels."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the alerter.
        
        Args:
            config: Configuration for the alerter
        """
        self.config = config or {}
        
    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """
        Send an alert notification.
        
        Args:
            alert: The alert to send
            
        Returns:
            True if the alert was sent successfully, False otherwise
        """
        pass
    
    def format_message(self, alert: Alert) -> str:
        """
        Format an alert message for sending.
        
        Args:
            alert: The alert to format
            
        Returns:
            Formatted message string
        """
        return f"[{alert.severity.value.upper()}] {alert.message}"

class ConsoleAlerter(AlerterBase):
    """Alerter that prints to the console."""
    
    def send_alert(self, alert: Alert) -> bool:
        """Send an alert to the console."""
        if alert.is_muted:
            return True
            
        message = self.format_message(alert)
        
        # Use different colors for different severities
        if alert.severity == AlertSeverity.CRITICAL:
            print(f"\033[91m{message}\033[0m")  # Red
        elif alert.severity == AlertSeverity.ERROR:
            print(f"\033[31m{message}\033[0m")  # Dark Red
        elif alert.severity == AlertSeverity.WARNING:
            print(f"\033[93m{message}\033[0m")  # Yellow
        else:
            print(f"\033[94m{message}\033[0m")  # Blue
            
        return True

class LogAlerter(AlerterBase):
    """Alerter that logs to the standard logger."""
    
    def send_alert(self, alert: Alert) -> bool:
        """Send an alert to the log."""
        if alert.is_muted:
            return True
            
        message = self.format_message(alert)
        
        # Log at appropriate level
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(message, extra=alert.context)
        elif alert.severity == AlertSeverity.ERROR:
            logger.error(message, extra=alert.context)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(message, extra=alert.context)
        else:
            logger.info(message, extra=alert.context)
            
        return True

class EmailAlerter(AlerterBase):
    """Alerter that sends alerts via email."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Get email configuration
        self.smtp_server = self.config.get('smtp_server', 'localhost')
        self.smtp_port = self.config.get('smtp_port', 587)
        self.smtp_username = self.config.get('smtp_username')
        self.smtp_password = self.config.get('smtp_password')
        self.use_ssl = self.config.get('use_ssl', False)
        self.use_tls = self.config.get('use_tls', True)
        
        self.from_address = self.config.get('from_address', 'smartscrape@localhost')
        self.to_addresses = self.config.get('to_addresses', [])
        
        # Set minimum severity for email alerts
        severity_str = self.config.get('min_severity', 'error')
        self.min_severity = getattr(AlertSeverity, severity_str.upper(), AlertSeverity.ERROR)
    
    def send_alert(self, alert: Alert) -> bool:
        """Send an alert via email."""
        if alert.is_muted:
            return True
            
        # Check if severity meets threshold
        if self._severity_value(alert.severity) < self._severity_value(self.min_severity):
            return True
            
        if not self.to_addresses:
            logger.error("No recipient email addresses configured")
            return False
            
        try:
            # Create message
            msg = email.message.EmailMessage()
            msg['Subject'] = f"SmartScrape Alert: {alert.severity.value.upper()} - {alert.message[:50]}"
            msg['From'] = self.from_address
            msg['To'] = ", ".join(self.to_addresses)
            
            # Format message body
            body = f"""
SmartScrape Alert
=================

Severity: {alert.severity.value.upper()}
Time: {alert.created_at.isoformat()}
Alert ID: {alert.id}

Message:
{alert.message}

Context:
{json.dumps(alert.context, indent=2)}
"""
            msg.set_content(body)
            
            # Connect to SMTP server
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                
                if self.use_tls:
                    server.starttls()
            
            # Login if credentials provided
            if self.smtp_username and self.smtp_password:
                server.login(self.smtp_username, self.smtp_password)
            
            # Send message
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False
    
    def _severity_value(self, severity: AlertSeverity) -> int:
        """Get numeric value for severity comparison."""
        values = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.ERROR: 2,
            AlertSeverity.CRITICAL: 3
        }
        return values.get(severity, 0)

class WebhookAlerter(AlerterBase):
    """Alerter that sends alerts to a webhook endpoint."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Get webhook configuration
        self.url = self.config.get('url')
        self.method = self.config.get('method', 'POST').upper()
        self.headers = self.config.get('headers', {})
        self.timeout = self.config.get('timeout', 10)
        
        # Set minimum severity for webhook alerts
        severity_str = self.config.get('min_severity', 'warning')
        self.min_severity = getattr(AlertSeverity, severity_str.upper(), AlertSeverity.WARNING)
    
    def send_alert(self, alert: Alert) -> bool:
        """Send an alert via webhook."""
        if alert.is_muted:
            return True
            
        # Check if severity meets threshold
        if self._severity_value(alert.severity) < self._severity_value(self.min_severity):
            return True
            
        if not self.url:
            logger.error("No webhook URL configured")
            return False
            
        try:
            # Prepare payload
            payload = {
                "alert": alert.to_dict(),
                "system": "SmartScrape",
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Send request
            response = requests.request(
                method=self.method,
                url=self.url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            
            # Check response
            if response.status_code >= 200 and response.status_code < 300:
                return True
            else:
                logger.error(f"Webhook returned non-success status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {str(e)}")
            return False
    
    def _severity_value(self, severity: AlertSeverity) -> int:
        """Get numeric value for severity comparison."""
        values = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.ERROR: 2,
            AlertSeverity.CRITICAL: 3
        }
        return values.get(severity, 0)

class Alerting(BaseService):
    """
    Service for sending alerts from SmartScrape components.
    
    This service manages different alert channels and centralized
    alert handling.
    """
    
    def __init__(self):
        self._initialized = False
        self._config = None
        self._alerters = {}
        self._alerts = {}
        self._alert_patterns = {}
        self._lock = threading.RLock()
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the alerting service with configuration."""
        if self._initialized:
            return
            
        logger.info("Initializing alerting service")
        
        self._config = config or {}
        
        # Configure alert channels
        self._configure_alerters()
        
        self._initialized = True
        logger.info("Alerting service initialized")
    
    def shutdown(self) -> None:
        """Clean up resources."""
        if not self._initialized:
            return
            
        logger.info("Shutting down alerting service")
        
        with self._lock:
            self._alerters.clear()
            self._alerts.clear()
            self._alert_patterns.clear()
        
        self._initialized = False
        logger.info("Alerting service shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "alerting"
    
    def trigger_alert(self, message: str, severity: Union[str, AlertSeverity], 
                      context: Optional[Dict[str, Any]] = None) -> str:
        """
        Trigger a new alert.
        
        Args:
            message: Alert message
            severity: Alert severity (can be string or AlertSeverity enum)
            context: Optional additional context information
            
        Returns:
            Alert ID
        """
        if not self._initialized:
            logger.warning("Alert triggered before alerting service initialized")
            
        # Convert string severity to enum if needed
        if isinstance(severity, str):
            try:
                severity = getattr(AlertSeverity, severity.upper())
            except AttributeError:
                logger.warning(f"Invalid severity: {severity}, using WARNING")
                severity = AlertSeverity.WARNING
        
        with self._lock:
            # Create the alert
            alert = Alert(message, severity, context)
            self._alerts[alert.id] = alert
            
            # Check for alert muting patterns
            for pattern, mute_info in self._alert_patterns.items():
                if pattern in message and mute_info['until'] > datetime.datetime.now():
                    alert.mute(int((mute_info['until'] - datetime.datetime.now()).total_seconds()))
                    logger.debug(f"Alert matching pattern '{pattern}' muted: {alert.id}")
                    break
            
            # Send the alert to all configured alerters
            for alerter in self._alerters.values():
                try:
                    alerter.send_alert(alert)
                except Exception as e:
                    logger.error(f"Error sending alert through {alerter.__class__.__name__}: {str(e)}")
            
            return alert.id
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an active alert.
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            True if the alert was found and resolved, False otherwise
        """
        with self._lock:
            if alert_id not in self._alerts:
                return False
                
            alert = self._alerts[alert_id]
            if not alert.is_active:
                return False
                
            alert.resolve()
            
            return True
    
    def get_active_alerts(self, severity: Optional[Union[str, AlertSeverity]] = None) -> List[Dict[str, Any]]:
        """
        Get all active alerts, optionally filtered by severity.
        
        Args:
            severity: Optional severity filter
            
        Returns:
            List of alert dictionaries
        """
        # Convert string severity to enum if needed
        if isinstance(severity, str):
            try:
                severity = getattr(AlertSeverity, severity.upper())
            except AttributeError:
                severity = None
        
        with self._lock:
            active_alerts = []
            
            for alert in self._alerts.values():
                if not alert.is_active:
                    continue
                    
                if severity is not None and alert.severity != severity:
                    continue
                    
                active_alerts.append(alert.to_dict())
            
            return active_alerts
    
    def mute_alerts(self, pattern: str, duration: int) -> None:
        """
        Mute alerts matching a pattern for a specified duration.
        
        Args:
            pattern: String pattern to match in alert messages
            duration: Duration in seconds to mute matching alerts
        """
        if not pattern:
            return
            
        with self._lock:
            # Set up muting pattern
            until = datetime.datetime.now() + datetime.timedelta(seconds=duration)
            self._alert_patterns[pattern] = {'until': until}
            
            # Mute existing alerts that match
            for alert in self._alerts.values():
                if pattern in alert.message and alert.is_active:
                    alert.mute(duration)
    
    def clear_alert_history(self, max_age: Optional[int] = None) -> int:
        """
        Clear alert history, keeping only recent alerts.
        
        Args:
            max_age: Optional maximum age in seconds to keep
            
        Returns:
            Number of alerts cleared
        """
        with self._lock:
            if max_age is None:
                # Keep active alerts
                before_count = len(self._alerts)
                self._alerts = {id: alert for id, alert in self._alerts.items() if alert.is_active}
                return before_count - len(self._alerts)
            
            # Keep alerts newer than max_age
            cutoff = datetime.datetime.now() - datetime.timedelta(seconds=max_age)
            before_count = len(self._alerts)
            self._alerts = {id: alert for id, alert in self._alerts.items() 
                           if alert.is_active or alert.created_at > cutoff}
            return before_count - len(self._alerts)
    
    def get_service_health(self) -> Dict[str, Any]:
        """
        Get the health status of the alerting service.
        
        Overrides BaseService.get_service_health()
        """
        with self._lock:
            active_count = sum(1 for alert in self._alerts.values() if alert.is_active)
            critical_count = sum(1 for alert in self._alerts.values() 
                               if alert.is_active and alert.severity == AlertSeverity.CRITICAL)
            
            metrics = {
                "active_alerts": active_count,
                "total_alerts_tracked": len(self._alerts),
                "critical_alerts": critical_count,
                "alerters_configured": len(self._alerters)
            }
            
            status = "healthy"
            if critical_count > 0:
                status = "degraded"
            
            return {
                "status": status,
                "details": f"{active_count} active alerts ({critical_count} critical)",
                "metrics": metrics
            }
    
    def _configure_alerters(self) -> None:
        """Configure alert channels from configuration."""
        alerter_configs = self._config.get('alerters', [
            {'type': 'console'},
            {'type': 'log'}
        ])
        
        for alerter_config in alerter_configs:
            alerter_type = alerter_config.get('type', 'log')
            
            try:
                if alerter_type == 'console':
                    self._alerters['console'] = ConsoleAlerter(alerter_config)
                    
                elif alerter_type == 'log':
                    self._alerters['log'] = LogAlerter(alerter_config)
                    
                elif alerter_type == 'email':
                    self._alerters['email'] = EmailAlerter(alerter_config)
                    
                elif alerter_type == 'webhook':
                    self._alerters['webhook'] = WebhookAlerter(alerter_config)
                    
                else:
                    logger.warning(f"Unknown alerter type: {alerter_type}")
                    
            except Exception as e:
                logger.error(f"Error configuring {alerter_type} alerter: {str(e)}")
    
    def add_alerter(self, name: str, alerter: AlerterBase) -> None:
        """
        Add a custom alerter.
        
        Args:
            name: Name for the alerter
            alerter: AlerterBase instance
        """
        with self._lock:
            self._alerters[name] = alerter