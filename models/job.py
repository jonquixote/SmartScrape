"""
Job model for tracking scraping tasks.
"""

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime

from models.base import Base


class Job(Base):
    """Job model for tracking scraping tasks."""
    
    __tablename__ = "jobs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed
    job_type = Column(String(50), nullable=False)  # simple, adaptive, search-extract
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    url = Column(String(2048), nullable=True)
    query = Column(Text, nullable=True)
    parameters = Column(JSON, nullable=True)  # Store any additional parameters
    result_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="jobs")
    results = relationship("JobResult", back_populates="job", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Job {self.id} ({self.status})>"


class JobResult(Base):
    """Job result model for storing scraping results."""
    
    __tablename__ = "job_results"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    url = Column(String(2048), nullable=True)
    data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    job = relationship("Job", back_populates="results")
    
    def __repr__(self):
        return f"<JobResult {self.id} (Job: {self.job_id})>"
