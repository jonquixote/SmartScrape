"""
Template model for storing scraping templates.
"""

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime

from models.base import Base


class ScrapingTemplate(Base):
    """Template model for storing scraping templates."""
    
    __tablename__ = "scraping_templates"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    domain = Column(String(255), nullable=False, index=True)
    template_type = Column(String(50), nullable=False)  # page, list, search, detail
    selectors = Column(JSON, nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    is_public = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    use_count = Column(Integer, default=0)
    success_rate = Column(Integer, default=0)  # 0-100 percentage
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<ScrapingTemplate {self.name} ({self.domain})>"
