"""
Database Manager for SmartScrape

This module provides database operations for storing and retrieving
extraction results, domain profiles, and performance metrics.
"""

import logging
import hashlib
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from models.extraction_models import (
    Base, ExtractionResult, DomainProfile, 
    StrategyPerformance, ExtractionSession, SystemMetrics
)
from config import DATABASE_URL, DATABASE_CONFIG, DATABASE_ENABLED

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for SmartScrape"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or DATABASE_URL
        self.enabled = DATABASE_ENABLED
        self.engine = None
        self.SessionLocal = None
        
        if self.enabled:
            try:
                self._initialize_engine()
            except Exception as e:
                logger.error(f"Database initialization failed: {e}")
                self.enabled = False
    
    def _initialize_engine(self):
        """Initialize the database engine and session factory"""
        engine_kwargs = {}
        
        # Use SQLite-specific settings for SQLite, PostgreSQL settings for PostgreSQL
        if self.database_url.startswith('sqlite'):
            engine_kwargs.update({
                'pool_pre_ping': True,
                'echo': DATABASE_CONFIG.get('echo', False)
            })
        else:
            engine_kwargs.update({
                'pool_size': DATABASE_CONFIG.get('pool_size', 5),
                'max_overflow': DATABASE_CONFIG.get('max_overflow', 10),
                'pool_timeout': DATABASE_CONFIG.get('pool_timeout', 30),
                'pool_recycle': DATABASE_CONFIG.get('pool_recycle', 3600),
                'pool_pre_ping': True,
                'echo': DATABASE_CONFIG.get('echo', False)
            })
        
        self.engine = create_engine(self.database_url, **engine_kwargs)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        logger.info(f"Database engine initialized for {self.database_url}")
    
    async def initialize(self):
        """Create tables if they don't exist"""
        if not self.enabled:
            logger.warning("Database is disabled, skipping initialization")
            return False
            
        try:
            # Run in thread since SQLAlchemy create_all is synchronous
            await asyncio.to_thread(self._create_tables)
            logger.info("Database tables created/verified successfully")
            return True
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.enabled = False
            return False
    
    def _create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Session:
        """Get a database session with automatic cleanup"""
        if not self.enabled:
            raise RuntimeError("Database is not enabled")
            
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    async def save_extraction_result(self, result: Dict) -> Optional[int]:
        """Save extraction result to database"""
        if not self.enabled:
            return None
            
        try:
            return await asyncio.to_thread(self._save_extraction_result_sync, result)
        except Exception as e:
            logger.error(f"Failed to save extraction result: {e}")
            return None
    
    def _save_extraction_result_sync(self, result: Dict) -> int:
        """Synchronous version of save_extraction_result"""
        with self.get_session() as session:
            url = result.get('url', '')
            url_hash = hashlib.md5(url.encode()).hexdigest()
            
            extraction = ExtractionResult(
                url=url,
                url_hash=url_hash,
                strategy=result.get('strategy', 'unknown'),
                content=result.get('content', ''),
                content_length=len(result.get('content', '')),
                word_count=result.get('word_count', 0),
                extraction_metadata=result.get('metadata', {}),  # Use extraction_metadata column
                success=result.get('success', False),
                error_message=result.get('error'),
                response_time=result.get('response_time', 0.0),
                quality_score=result.get('quality_score', 0.0),
                user_agent=result.get('user_agent'),
                status_code=result.get('status_code'),
                final_url=result.get('final_url', url)
            )
            
            session.add(extraction)
            session.flush()  # To get the ID
            return extraction.id
    
    async def update_domain_profile(self, domain: str, profile_data: Dict) -> bool:
        """Update or create domain profile"""
        if not self.enabled:
            return False
            
        try:
            return await asyncio.to_thread(self._update_domain_profile_sync, domain, profile_data)
        except Exception as e:
            logger.error(f"Failed to update domain profile for {domain}: {e}")
            return False
    
    def _update_domain_profile_sync(self, domain: str, profile_data: Dict) -> bool:
        """Synchronous version of update_domain_profile"""
        with self.get_session() as session:
            # Try to get existing profile
            profile = session.query(DomainProfile).filter_by(domain=domain).first()
            
            if profile:
                # Update existing profile
                for key, value in profile_data.items():
                    if hasattr(profile, key):
                        setattr(profile, key, value)
                profile.updated_at = datetime.utcnow()
            else:
                # Create new profile
                profile = DomainProfile(domain=domain, **profile_data)
                session.add(profile)
            
            return True
    
    async def get_domain_profile(self, domain: str) -> Optional[Dict]:
        """Get domain profile"""
        if not self.enabled:
            return None
            
        try:
            return await asyncio.to_thread(self._get_domain_profile_sync, domain)
        except Exception as e:
            logger.error(f"Failed to get domain profile for {domain}: {e}")
            return None
    
    def _get_domain_profile_sync(self, domain: str) -> Optional[Dict]:
        """Synchronous version of get_domain_profile"""
        with self.get_session() as session:
            profile = session.query(DomainProfile).filter_by(domain=domain).first()
            if profile:
                return {
                    'domain': profile.domain,
                    'optimal_strategy': profile.optimal_strategy,
                    'fallback_strategies': profile.fallback_strategies,
                    'requires_js': profile.requires_js,
                    'js_frameworks': profile.js_frameworks,
                    'content_type': profile.content_type,
                    'avg_response_time': profile.avg_response_time,
                    'success_rate': profile.success_rate,
                    'total_extractions': profile.total_extractions,
                    'last_analyzed': profile.last_analyzed,
                    'analysis_confidence': profile.analysis_confidence,
                    'custom_settings': profile.custom_settings
                }
            return None
    
    async def update_strategy_performance(self, domain: str, strategy: str, 
                                        success: bool, response_time: float, 
                                        quality_score: float = None) -> bool:
        """Update strategy performance metrics"""
        if not self.enabled:
            return False
            
        try:
            return await asyncio.to_thread(
                self._update_strategy_performance_sync, 
                domain, strategy, success, response_time, quality_score
            )
        except Exception as e:
            logger.error(f"Failed to update strategy performance: {e}")
            return False
    
    def _update_strategy_performance_sync(self, domain: str, strategy: str, 
                                        success: bool, response_time: float, 
                                        quality_score: float = None) -> bool:
        """Synchronous version of update_strategy_performance"""
        with self.get_session() as session:
            # Get or create performance record
            perf = session.query(StrategyPerformance).filter_by(
                domain=domain, strategy=strategy
            ).first()
            
            if not perf:
                perf = StrategyPerformance(
                    domain=domain,
                    strategy=strategy,
                    total_attempts=0,
                    successful_attempts=0,
                    failed_attempts=0,
                    total_response_time=0.0,
                    total_quality_score=0.0,
                    first_attempt=datetime.utcnow()
                )
                session.add(perf)
            
            # Update metrics
            perf.total_attempts += 1
            perf.last_attempt = datetime.utcnow()
            
            if success:
                perf.successful_attempts += 1
            else:
                perf.failed_attempts += 1
            
            # Update timing metrics
            perf.total_response_time += response_time
            perf.avg_response_time = perf.total_response_time / perf.total_attempts
            
            if perf.min_response_time is None or response_time < perf.min_response_time:
                perf.min_response_time = response_time
            if perf.max_response_time is None or response_time > perf.max_response_time:
                perf.max_response_time = response_time
            
            # Update quality metrics
            if quality_score is not None:
                if perf.total_quality_score is None:
                    perf.total_quality_score = 0.0
                perf.total_quality_score += quality_score
                perf.avg_quality_score = perf.total_quality_score / perf.total_attempts
            
            # Calculate success rate
            perf.success_rate = perf.successful_attempts / perf.total_attempts
            
            perf.updated_at = datetime.utcnow()
            
            return True
    
    async def get_strategy_performance(self, domain: str = None, strategy: str = None) -> List[Dict]:
        """Get strategy performance data"""
        if not self.enabled:
            return []
            
        try:
            return await asyncio.to_thread(self._get_strategy_performance_sync, domain, strategy)
        except Exception as e:
            logger.error(f"Failed to get strategy performance: {e}")
            return []
    
    def _get_strategy_performance_sync(self, domain: str = None, strategy: str = None) -> List[Dict]:
        """Synchronous version of get_strategy_performance"""
        with self.get_session() as session:
            query = session.query(StrategyPerformance)
            
            if domain:
                query = query.filter_by(domain=domain)
            if strategy:
                query = query.filter_by(strategy=strategy)
            
            results = []
            for perf in query.all():
                results.append({
                    'domain': perf.domain,
                    'strategy': perf.strategy,
                    'total_attempts': perf.total_attempts,
                    'successful_attempts': perf.successful_attempts,
                    'failed_attempts': perf.failed_attempts,
                    'success_rate': perf.success_rate,
                    'avg_response_time': perf.avg_response_time,
                    'min_response_time': perf.min_response_time,
                    'max_response_time': perf.max_response_time,
                    'avg_quality_score': perf.avg_quality_score,
                    'last_attempt': perf.last_attempt,
                    'first_attempt': perf.first_attempt
                })
            
            return results
    
    async def create_extraction_session(self, session_data: Dict) -> Optional[str]:
        """Create a new extraction session"""
        if not self.enabled:
            return None
            
        try:
            return await asyncio.to_thread(self._create_extraction_session_sync, session_data)
        except Exception as e:
            logger.error(f"Failed to create extraction session: {e}")
            return None
    
    def _create_extraction_session_sync(self, session_data: Dict) -> str:
        """Synchronous version of create_extraction_session"""
        import uuid
        
        with self.get_session() as session:
            session_id = str(uuid.uuid4())
            
            extraction_session = ExtractionSession(
                session_id=session_id,
                user_query=session_data.get('user_query'),
                extraction_type=session_data.get('extraction_type', 'single'),
                total_urls=session_data.get('total_urls', 0),
                config_snapshot=session_data.get('config_snapshot', {})
            )
            
            session.add(extraction_session)
            return session_id
    
    async def record_system_metric(self, metric_name: str, value: float, 
                                 unit: str = None, component: str = None, 
                                 context: Dict = None) -> bool:
        """Record a system metric"""
        if not self.enabled:
            return False
            
        try:
            return await asyncio.to_thread(
                self._record_system_metric_sync, 
                metric_name, value, unit, component, context
            )
        except Exception as e:
            logger.error(f"Failed to record system metric: {e}")
            return False
    
    def _record_system_metric_sync(self, metric_name: str, value: float, 
                                 unit: str = None, component: str = None, 
                                 context: Dict = None) -> bool:
        """Synchronous version of record_system_metric"""
        with self.get_session() as session:
            metric = SystemMetrics(
                metric_name=metric_name,
                metric_value=value,
                metric_unit=unit,
                component=component,
                context=context or {}
            )
            session.add(metric)
            return True
    
    async def cleanup_old_records(self, days: int = 30) -> Dict[str, int]:
        """Clean up old records to save space"""
        if not self.enabled:
            return {}
            
        try:
            return await asyncio.to_thread(self._cleanup_old_records_sync, days)
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
            return {}
    
    def _cleanup_old_records_sync(self, days: int) -> Dict[str, int]:
        """Synchronous version of cleanup_old_records"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        cleaned_counts = {}
        
        with self.get_session() as session:
            # Clean extraction results
            result = session.query(ExtractionResult).filter(
                ExtractionResult.created_at < cutoff_date
            ).delete()
            cleaned_counts['extraction_results'] = result
            
            # Clean system metrics
            result = session.query(SystemMetrics).filter(
                SystemMetrics.timestamp < cutoff_date
            ).delete()
            cleaned_counts['system_metrics'] = result
            
            # Clean old extraction sessions
            result = session.query(ExtractionSession).filter(
                ExtractionSession.created_at < cutoff_date
            ).delete()
            cleaned_counts['extraction_sessions'] = result
        
        logger.info(f"Cleaned up old records: {cleaned_counts}")
        return cleaned_counts
    
    async def get_extraction_stats(self, days: int = 7) -> Dict:
        """Get extraction statistics for the last N days"""
        if not self.enabled:
            return {}
            
        try:
            return await asyncio.to_thread(self._get_extraction_stats_sync, days)
        except Exception as e:
            logger.error(f"Failed to get extraction stats: {e}")
            return {}
    
    def _get_extraction_stats_sync(self, days: int) -> Dict:
        """Synchronous version of get_extraction_stats"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            # Total extractions
            total_query = session.query(ExtractionResult).filter(
                ExtractionResult.created_at >= start_date
            )
            total_extractions = total_query.count()
            successful_extractions = total_query.filter(ExtractionResult.success == True).count()
            
            # Strategy breakdown
            strategy_stats = session.execute(text("""
                SELECT strategy, 
                       COUNT(*) as total,
                       SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                       AVG(response_time) as avg_response_time
                FROM extraction_results 
                WHERE created_at >= :start_date 
                GROUP BY strategy
            """), {"start_date": start_date}).fetchall()
            
            # Domain breakdown
            domain_stats = session.execute(text("""
                SELECT COUNT(DISTINCT url_hash) as unique_domains,
                       AVG(content_length) as avg_content_length,
                       AVG(quality_score) as avg_quality_score
                FROM extraction_results 
                WHERE created_at >= :start_date AND success = true
            """), {"start_date": start_date}).fetchone()
            
            return {
                'period_days': days,
                'total_extractions': total_extractions,
                'successful_extractions': successful_extractions,
                'success_rate': successful_extractions / max(total_extractions, 1),
                'strategy_breakdown': [
                    {
                        'strategy': row[0],
                        'total': row[1],
                        'successful': row[2],
                        'success_rate': row[2] / max(row[1], 1),
                        'avg_response_time': float(row[3]) if row[3] else 0.0
                    }
                    for row in strategy_stats
                ],
                'unique_domains': domain_stats[0] if domain_stats else 0,
                'avg_content_length': float(domain_stats[1]) if domain_stats and domain_stats[1] else 0.0,
                'avg_quality_score': float(domain_stats[2]) if domain_stats and domain_stats[2] else 0.0
            }

# Global database manager instance
db_manager = DatabaseManager()
