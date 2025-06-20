"""
Models package for SmartScrape database entities.
"""

from models.base import Base, init_db, get_db_session
from models.user import User, APIKey
from models.job import Job, JobResult
from models.template import ScrapingTemplate
from models.audit import AuditLog, create_audit_log

__all__ = [
    'Base',
    'init_db',
    'get_db_session',
    'User',
    'APIKey',
    'Job',
    'JobResult',
    'ScrapingTemplate',
    'AuditLog',
    'create_audit_log'
]
