"""
Database models for SmartScrape

This module defines the SQLAlchemy models for storing extraction results,
domain profiles, and performance metrics.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Boolean, Float, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()

class ExtractionResult(Base):
    """Store extraction results for analysis and caching"""
    __tablename__ = 'extraction_results'
    
    id = Column(Integer, primary_key=True)
    url = Column(String(2048), nullable=False, index=True)
    url_hash = Column(String(64), nullable=False, index=True)  # MD5 hash for faster lookups
    strategy = Column(String(100), nullable=False, index=True)
    content = Column(Text)
    content_length = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    extraction_metadata = Column(JSON)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    success = Column(Boolean, default=False, index=True)
    error_message = Column(Text)
    response_time = Column(Float)
    quality_score = Column(Float)  # Content quality assessment
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional metadata
    user_agent = Column(String(512))
    status_code = Column(Integer)
    final_url = Column(String(2048))  # After redirects
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_url_strategy', 'url_hash', 'strategy'),
        Index('idx_domain_strategy', 'url_hash', 'strategy', 'created_at'),
        Index('idx_success_time', 'success', 'created_at'),
    )

class DomainProfile(Base):
    """Store domain-specific optimization profiles"""
    __tablename__ = 'domain_profiles'
    
    id = Column(Integer, primary_key=True)
    domain = Column(String(255), unique=True, nullable=False, index=True)
    
    # Strategy preferences
    optimal_strategy = Column(String(100))
    fallback_strategies = Column(JSON)  # List of fallback strategies
    
    # Technical characteristics
    requires_js = Column(Boolean, default=False)
    js_frameworks = Column(JSON)  # Detected frameworks
    content_type = Column(String(100))  # news, ecommerce, spa, etc.
    
    # Performance metrics
    avg_response_time = Column(Float)
    success_rate = Column(Float)
    total_extractions = Column(Integer, default=0)
    last_success = Column(DateTime)
    last_failure = Column(DateTime)
    
    # Analysis metadata
    last_analyzed = Column(DateTime, default=datetime.utcnow)
    analysis_confidence = Column(Float)  # Confidence in the profile
    
    # Settings
    custom_settings = Column(JSON)  # Domain-specific extraction settings
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class StrategyPerformance(Base):
    """Track strategy performance metrics over time"""
    __tablename__ = 'strategy_performance'
    
    id = Column(Integer, primary_key=True)
    domain = Column(String(255), nullable=False, index=True)
    strategy = Column(String(100), nullable=False, index=True)
    
    # Performance counters
    total_attempts = Column(Integer, default=0)
    successful_attempts = Column(Integer, default=0)
    failed_attempts = Column(Integer, default=0)
    
    # Timing metrics
    total_response_time = Column(Float, default=0.0)
    avg_response_time = Column(Float)
    min_response_time = Column(Float)
    max_response_time = Column(Float)
    
    # Quality metrics
    total_quality_score = Column(Float, default=0.0)
    avg_quality_score = Column(Float)
    
    # Calculated metrics
    success_rate = Column(Float)
    
    # Time windows
    last_attempt = Column(DateTime)
    first_attempt = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_domain_strategy_perf', 'domain', 'strategy'),
        Index('idx_strategy_success', 'strategy', 'success_rate'),
    )

class ExtractionSession(Base):
    """Track extraction sessions and batch operations"""
    __tablename__ = 'extraction_sessions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    
    # Session metadata
    user_query = Column(Text)  # Original user query
    extraction_type = Column(String(100))  # single, batch, intelligent, etc.
    total_urls = Column(Integer, default=0)
    successful_urls = Column(Integer, default=0)
    failed_urls = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    total_duration = Column(Float)
    
    # Configuration used
    config_snapshot = Column(JSON)
    
    # Results summary
    results_summary = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class SystemMetrics(Base):
    """Store system-level performance metrics"""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50))
    
    # Context
    component = Column(String(100))  # Which component generated the metric
    context = Column(JSON)  # Additional context data
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_metric_time', 'metric_name', 'timestamp'),
        Index('idx_component_metric', 'component', 'metric_name'),
    )
