"""
Base database model and connection utilities for SQLAlchemy.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

import config

# Create SQLAlchemy base model
Base = declarative_base()

# Create engine based on DATABASE_URL from config
engine = create_engine(config.DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session():
    """Context manager to handle database sessions with error handling."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Initialize the database by creating all tables."""
    # Import all models here to ensure they're registered with Base
    from models.user import User
    from models.job import Job
    from models.template import ScrapingTemplate
    from models.audit import AuditLog
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
