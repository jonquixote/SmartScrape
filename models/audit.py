"""
Audit log model for tracking system events.
"""

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime

from models.base import Base


class AuditLog(Base):
    """Audit log model for tracking system events."""
    
    __tablename__ = "audit_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_type = Column(String(50), nullable=False)  # auth, job, system, error
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    resource_type = Column(String(50), nullable=True)  # user, job, template
    resource_id = Column(String(36), nullable=True)
    action = Column(String(50), nullable=False)  # create, read, update, delete, login, etc.
    status = Column(String(20), nullable=False)  # success, failure
    details = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)  # Support IPv6
    user_agent = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<AuditLog {self.id} ({self.event_type}: {self.action})>"


# Helper function to create audit logs
def create_audit_log(db_session, event_type, action, status, user_id=None, 
                     resource_type=None, resource_id=None, details=None,
                     ip_address=None, user_agent=None):
    """Create an audit log entry."""
    log = AuditLog(
        event_type=event_type,
        action=action,
        status=status,
        user_id=user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    db_session.add(log)
    db_session.commit()
    
    return log
