"""
Learning and Adaptation System for SmartScrape

This module implements machine learning capabilities for pattern recognition,
site-specific optimization, and continuous improvement of extraction strategies.
"""

import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import hashlib
import re
from urllib.parse import urlparse

from utils.database_manager import DatabaseManager


class PatternLearningSystem:
    """System for learning and recognizing successful extraction patterns"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.setup_learning_tables()
        
        # Pattern storage
        self.successful_patterns = defaultdict(list)
        self.failed_patterns = defaultdict(list)
        self.site_patterns = defaultdict(dict)
        
        # Performance tracking
        self.selector_performance = defaultdict(lambda: {'success': 0, 'total': 0})
        self.strategy_performance = defaultdict(lambda: {'success': 0, 'total': 0})
        self.site_performance = defaultdict(lambda: {'success': 0, 'total': 0})
        
        # Learning thresholds
        self.min_samples_for_learning = 10
        self.success_threshold = 0.8
        self.pattern_confidence_threshold = 0.7
        
    def setup_learning_tables(self):
        """Setup database tables for learning system"""
        try:
            # Extraction patterns table
            self.db_manager.execute_query("""
                CREATE TABLE IF NOT EXISTS extraction_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    selector_pattern TEXT NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence_score REAL DEFAULT 0.0,
                    pattern_hash TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Site optimization table
            self.db_manager.execute_query("""
                CREATE TABLE IF NOT EXISTS site_optimizations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    optimization_type TEXT NOT NULL,
                    optimization_data TEXT NOT NULL,
                    success_rate REAL DEFAULT 0.0,
                    sample_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Strategy performance table
            self.db_manager.execute_query("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    total_count INTEGER DEFAULT 0,
                    avg_quality_score REAL DEFAULT 0.0,
                    avg_extraction_time REAL DEFAULT 0.0,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Learning feedback table
            self.db_manager.execute_query("""
                CREATE TABLE IF NOT EXISTS learning_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    extraction_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    feedback_score REAL NOT NULL,
                    feedback_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
        except Exception as e:
            print(f"Error setting up learning tables: {e}")
    
    def record_extraction_attempt(self, 
                                domain: str,
                                content_type: str,
                                selectors: List[str],
                                strategy_name: str,
                                success: bool,
                                quality_score: float = 0.0,
                                extraction_time: float = 0.0,
                                extracted_data: Optional[Dict] = None):
        """Record an extraction attempt for learning"""
        try:
            # Record selector patterns
            for selector in selectors:
                pattern_hash = hashlib.md5(f"{domain}:{content_type}:{selector}".encode()).hexdigest()
                
                # Update pattern performance
                if success:
                    self.selector_performance[pattern_hash]['success'] += 1
                self.selector_performance[pattern_hash]['total'] += 1
                
                # Store in database
                confidence = self.selector_performance[pattern_hash]['success'] / max(1, self.selector_performance[pattern_hash]['total'])
                
                self.db_manager.execute_query("""
                    INSERT OR REPLACE INTO extraction_patterns 
                    (domain, content_type, selector_pattern, success_count, failure_count, 
                     confidence_score, pattern_hash, last_used)
                    SELECT ?, ?, ?, 
                           COALESCE((SELECT success_count FROM extraction_patterns WHERE pattern_hash = ?), 0) + ?,
                           COALESCE((SELECT failure_count FROM extraction_patterns WHERE pattern_hash = ?), 0) + ?,
                           ?, ?, CURRENT_TIMESTAMP
                """, (domain, content_type, selector, pattern_hash, 
                      1 if success else 0, pattern_hash, 0 if success else 1, confidence, pattern_hash))
            
            # Record strategy performance
            if success:
                self.strategy_performance[f"{strategy_name}:{domain}:{content_type}"]['success'] += 1
            self.strategy_performance[f"{strategy_name}:{domain}:{content_type}"]['total'] += 1
            
            self.db_manager.execute_query("""
                INSERT OR REPLACE INTO strategy_performance 
                (strategy_name, domain, content_type, success_count, total_count, 
                 avg_quality_score, avg_extraction_time, last_used)
                SELECT ?, ?, ?, 
                       COALESCE((SELECT success_count FROM strategy_performance 
                                WHERE strategy_name = ? AND domain = ? AND content_type = ?), 0) + ?,
                       COALESCE((SELECT total_count FROM strategy_performance 
                                WHERE strategy_name = ? AND domain = ? AND content_type = ?), 0) + 1,
                       ?, ?, CURRENT_TIMESTAMP
            """, (strategy_name, domain, content_type, 
                  strategy_name, domain, content_type, 1 if success else 0,
                  strategy_name, domain, content_type, quality_score, extraction_time))
            
        except Exception as e:
            print(f"Error recording extraction attempt: {e}")
    
    def get_best_selectors_for_domain(self, domain: str, content_type: str, limit: int = 5) -> List[Dict]:
        """Get the most successful selectors for a domain and content type"""
        try:
            results = self.db_manager.fetch_all("""
                SELECT selector_pattern, confidence_score, success_count, failure_count
                FROM extraction_patterns
                WHERE domain = ? AND content_type = ? AND confidence_score >= ?
                ORDER BY confidence_score DESC, success_count DESC
                LIMIT ?
            """, (domain, content_type, self.pattern_confidence_threshold, limit))
            
            return [
                {
                    'selector': row[0],
                    'confidence': row[1],
                    'success_count': row[2],
                    'failure_count': row[3]
                }
                for row in results
            ]
        except Exception as e:
            print(f"Error getting best selectors: {e}")
            return []
    
    def get_best_strategy_for_domain(self, domain: str, content_type: str) -> Optional[str]:
        """Get the most successful strategy for a domain and content type"""
        try:
            result = self.db_manager.fetch_one("""
                SELECT strategy_name, (success_count * 1.0 / total_count) as success_rate
                FROM strategy_performance
                WHERE domain = ? AND content_type = ? AND total_count >= ?
                ORDER BY success_rate DESC, success_count DESC
                LIMIT 1
            """, (domain, content_type, self.min_samples_for_learning))
            
            if result and result[1] >= self.success_threshold:
                return result[0]
            return None
        except Exception as e:
            print(f"Error getting best strategy: {e}")
            return None
    
    def learn_site_patterns(self, domain: str) -> Dict[str, Any]:
        """Learn optimization patterns for a specific site"""
        try:
            site_data = {
                'selectors': {},
                'strategies': {},
                'optimizations': {}
            }
            
            # Get successful selector patterns
            selector_results = self.db_manager.fetch_all("""
                SELECT content_type, selector_pattern, confidence_score
                FROM extraction_patterns
                WHERE domain = ? AND confidence_score >= ?
                ORDER BY content_type, confidence_score DESC
            """, (domain, self.pattern_confidence_threshold))
            
            for content_type, selector, confidence in selector_results:
                if content_type not in site_data['selectors']:
                    site_data['selectors'][content_type] = []
                site_data['selectors'][content_type].append({
                    'selector': selector,
                    'confidence': confidence
                })
            
            # Get successful strategy patterns
            strategy_results = self.db_manager.fetch_all("""
                SELECT content_type, strategy_name, (success_count * 1.0 / total_count) as success_rate
                FROM strategy_performance
                WHERE domain = ? AND total_count >= ?
                ORDER BY content_type, success_rate DESC
            """, (domain, self.min_samples_for_learning))
            
            for content_type, strategy, success_rate in strategy_results:
                if success_rate >= self.success_threshold:
                    site_data['strategies'][content_type] = {
                        'strategy': strategy,
                        'success_rate': success_rate
                    }
            
            # Store learned patterns
            optimization_data = json.dumps(site_data)
            self.db_manager.execute_query("""
                INSERT OR REPLACE INTO site_optimizations 
                (domain, optimization_type, optimization_data, last_updated)
                VALUES (?, 'patterns', ?, CURRENT_TIMESTAMP)
            """, (domain, optimization_data))
            
            return site_data
            
        except Exception as e:
            print(f"Error learning site patterns: {e}")
            return {}
    
    def get_site_optimizations(self, domain: str) -> Dict[str, Any]:
        """Get learned optimizations for a site"""
        try:
            result = self.db_manager.fetch_one("""
                SELECT optimization_data
                FROM site_optimizations
                WHERE domain = ? AND optimization_type = 'patterns' AND is_active = TRUE
                ORDER BY last_updated DESC
                LIMIT 1
            """, (domain,))
            
            if result:
                return json.loads(result[0])
            return {}
        except Exception as e:
            print(f"Error getting site optimizations: {e}")
            return {}
    
    def record_user_feedback(self, extraction_id: str, feedback_type: str, score: float, data: Optional[Dict] = None):
        """Record user feedback for quality improvement"""
        try:
            feedback_data = json.dumps(data) if data else None
            self.db_manager.execute_query("""
                INSERT INTO learning_feedback 
                (extraction_id, feedback_type, feedback_score, feedback_data)
                VALUES (?, ?, ?, ?)
            """, (extraction_id, feedback_type, score, feedback_data))
        except Exception as e:
            print(f"Error recording feedback: {e}")
    
    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze feedback patterns for system improvement"""
        try:
            # Get feedback statistics
            feedback_stats = self.db_manager.fetch_all("""
                SELECT feedback_type, AVG(feedback_score) as avg_score, COUNT(*) as count
                FROM learning_feedback
                WHERE created_at >= datetime('now', '-30 days')
                GROUP BY feedback_type
            """)
            
            # Get recent low-scoring extractions for analysis
            poor_extractions = self.db_manager.fetch_all("""
                SELECT extraction_id, feedback_type, feedback_score, feedback_data
                FROM learning_feedback
                WHERE feedback_score < 0.5 AND created_at >= datetime('now', '-7 days')
                ORDER BY created_at DESC
                LIMIT 20
            """)
            
            return {
                'feedback_stats': [
                    {'type': row[0], 'avg_score': row[1], 'count': row[2]}
                    for row in feedback_stats
                ],
                'poor_extractions': [
                    {
                        'extraction_id': row[0],
                        'feedback_type': row[1],
                        'score': row[2],
                        'data': json.loads(row[3]) if row[3] else None
                    }
                    for row in poor_extractions
                ]
            }
        except Exception as e:
            print(f"Error analyzing feedback: {e}")
            return {'feedback_stats': [], 'poor_extractions': []}


class SiteSpecificOptimizer:
    """Optimizer for site-specific extraction strategies"""
    
    def __init__(self, learning_system: PatternLearningSystem):
        self.learning_system = learning_system
        self.site_configs = {}
        self.load_site_configs()
    
    def load_site_configs(self):
        """Load existing site-specific configurations"""
        try:
            # Load from database
            results = self.learning_system.db_manager.fetch_all("""
                SELECT domain, optimization_data
                FROM site_optimizations
                WHERE optimization_type = 'config' AND is_active = TRUE
            """)
            
            for domain, config_data in results:
                self.site_configs[domain] = json.loads(config_data)
                
        except Exception as e:
            print(f"Error loading site configs: {e}")
    
    def optimize_for_site(self, domain: str, sample_extractions: List[Dict]) -> Dict[str, Any]:
        """Generate site-specific optimization configuration"""
        try:
            optimization = {
                'domain': domain,
                'selectors': {},
                'navigation_hints': {},
                'content_patterns': {},
                'rate_limiting': {},
                'anti_detection': {}
            }
            
            # Analyze URL patterns
            urls = [ext.get('url', '') for ext in sample_extractions if ext.get('url')]
            optimization['navigation_hints'] = self._analyze_url_patterns(urls)
            
            # Analyze content structure
            for extraction in sample_extractions:
                content_type = extraction.get('content_type', 'unknown')
                if content_type not in optimization['content_patterns']:
                    optimization['content_patterns'][content_type] = {}
                
                # Analyze successful selectors
                if extraction.get('success') and extraction.get('selectors'):
                    if content_type not in optimization['selectors']:
                        optimization['selectors'][content_type] = []
                    optimization['selectors'][content_type].extend(extraction['selectors'])
            
            # Determine rate limiting needs
            optimization['rate_limiting'] = self._determine_rate_limits(domain, sample_extractions)
            
            # Store optimization
            config_data = json.dumps(optimization)
            self.learning_system.db_manager.execute_query("""
                INSERT OR REPLACE INTO site_optimizations 
                (domain, optimization_type, optimization_data, last_updated)
                VALUES (?, 'config', ?, CURRENT_TIMESTAMP)
            """, (domain, config_data))
            
            self.site_configs[domain] = optimization
            return optimization
            
        except Exception as e:
            print(f"Error optimizing for site {domain}: {e}")
            return {}
    
    def _analyze_url_patterns(self, urls: List[str]) -> Dict[str, Any]:
        """Analyze URL patterns for navigation hints"""
        patterns = {
            'path_patterns': [],
            'query_patterns': [],
            'fragment_patterns': [],
            'pagination_patterns': []
        }
        
        try:
            for url in urls:
                parsed = urlparse(url)
                
                # Analyze path patterns
                path_parts = [p for p in parsed.path.split('/') if p]
                if path_parts:
                    patterns['path_patterns'].extend(path_parts)
                
                # Look for pagination indicators
                if re.search(r'page=\d+|p=\d+|offset=\d+', parsed.query):
                    patterns['pagination_patterns'].append(parsed.query)
                
                # Analyze query parameters
                if parsed.query:
                    patterns['query_patterns'].append(parsed.query)
            
            # Find common patterns
            for key in patterns:
                if patterns[key]:
                    counter = Counter(patterns[key])
                    patterns[key] = [item for item, count in counter.most_common(5)]
            
            return patterns
        except Exception as e:
            print(f"Error analyzing URL patterns: {e}")
            return patterns
    
    def _determine_rate_limits(self, domain: str, extractions: List[Dict]) -> Dict[str, Any]:
        """Determine appropriate rate limiting for a domain"""
        return {
            'requests_per_second': 2,  # Conservative default
            'concurrent_requests': 1,
            'delay_range': [1, 3],
            'respect_robots_txt': True,
            'use_session': True
        }
    
    def get_site_config(self, domain: str) -> Dict[str, Any]:
        """Get site-specific configuration"""
        return self.site_configs.get(domain, {})


class PerformanceOptimizer:
    """System for optimizing extraction performance"""
    
    def __init__(self, learning_system: PatternLearningSystem):
        self.learning_system = learning_system
        self.performance_metrics = defaultdict(list)
        self.optimization_history = []
    
    def record_performance_metrics(self, 
                                 domain: str,
                                 extraction_time: float,
                                 memory_usage: float,
                                 success_rate: float,
                                 quality_score: float):
        """Record performance metrics for analysis"""
        metrics = {
            'domain': domain,
            'extraction_time': extraction_time,
            'memory_usage': memory_usage,
            'success_rate': success_rate,
            'quality_score': quality_score,
            'timestamp': datetime.now()
        }
        
        self.performance_metrics[domain].append(metrics)
        
        # Keep only recent metrics (last 100 per domain)
        if len(self.performance_metrics[domain]) > 100:
            self.performance_metrics[domain] = self.performance_metrics[domain][-100:]
    
    def analyze_performance_trends(self, domain: str) -> Dict[str, Any]:
        """Analyze performance trends for optimization opportunities"""
        if domain not in self.performance_metrics:
            return {}
        
        metrics = self.performance_metrics[domain]
        if len(metrics) < 10:  # Need sufficient data
            return {}
        
        try:
            recent_metrics = metrics[-20:]  # Last 20 extractions
            older_metrics = metrics[-40:-20] if len(metrics) >= 40 else metrics[:-20]
            
            analysis = {
                'performance_trend': self._calculate_trend(recent_metrics, older_metrics),
                'bottlenecks': self._identify_bottlenecks(recent_metrics),
                'optimization_opportunities': self._find_optimization_opportunities(recent_metrics),
                'recommendations': []
            }
            
            # Generate recommendations
            if analysis['performance_trend']['extraction_time'] > 1.1:  # Getting slower
                analysis['recommendations'].append("Consider selector optimization - extraction time increasing")
            
            if analysis['performance_trend']['memory_usage'] > 1.2:  # Memory increasing
                analysis['recommendations'].append("Implement memory optimization - usage trending up")
            
            if analysis['performance_trend']['success_rate'] < 0.9:  # Success decreasing
                analysis['recommendations'].append("Review extraction strategy - success rate declining")
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing performance trends: {e}")
            return {}
    
    def _calculate_trend(self, recent: List[Dict], older: List[Dict]) -> Dict[str, float]:
        """Calculate performance trend ratios"""
        if not older:
            return {'extraction_time': 1.0, 'memory_usage': 1.0, 'success_rate': 1.0}
        
        try:
            recent_avg = {
                'extraction_time': sum(m['extraction_time'] for m in recent) / len(recent),
                'memory_usage': sum(m['memory_usage'] for m in recent) / len(recent),
                'success_rate': sum(m['success_rate'] for m in recent) / len(recent)
            }
            
            older_avg = {
                'extraction_time': sum(m['extraction_time'] for m in older) / len(older),
                'memory_usage': sum(m['memory_usage'] for m in older) / len(older),
                'success_rate': sum(m['success_rate'] for m in older) / len(older)
            }
            
            return {
                'extraction_time': recent_avg['extraction_time'] / max(0.1, older_avg['extraction_time']),
                'memory_usage': recent_avg['memory_usage'] / max(0.1, older_avg['memory_usage']),
                'success_rate': recent_avg['success_rate'] / max(0.1, older_avg['success_rate'])
            }
        except Exception:
            return {'extraction_time': 1.0, 'memory_usage': 1.0, 'success_rate': 1.0}
    
    def _identify_bottlenecks(self, metrics: List[Dict]) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        try:
            avg_time = sum(m['extraction_time'] for m in metrics) / len(metrics)
            avg_memory = sum(m['memory_usage'] for m in metrics) / len(metrics)
            avg_success = sum(m['success_rate'] for m in metrics) / len(metrics)
            
            if avg_time > 30:  # More than 30 seconds
                bottlenecks.append("Slow extraction time")
            
            if avg_memory > 500:  # More than 500MB
                bottlenecks.append("High memory usage")
            
            if avg_success < 0.8:  # Less than 80% success
                bottlenecks.append("Low success rate")
            
            return bottlenecks
        except Exception:
            return []
    
    def _find_optimization_opportunities(self, metrics: List[Dict]) -> List[str]:
        """Find specific optimization opportunities"""
        opportunities = []
        
        try:
            times = [m['extraction_time'] for m in metrics]
            memory = [m['memory_usage'] for m in metrics]
            
            # Check for high variance (inconsistent performance)
            if len(times) > 5:
                time_variance = max(times) / min(times) if min(times) > 0 else 1
                if time_variance > 3:
                    opportunities.append("Inconsistent extraction times - consider caching")
            
            # Check for memory spikes
            if len(memory) > 5:
                memory_variance = max(memory) / min(memory) if min(memory) > 0 else 1
                if memory_variance > 5:
                    opportunities.append("Memory spikes detected - implement streaming extraction")
            
            return opportunities
        except Exception:
            return []


class LearningOrchestrator:
    """Main orchestrator for the learning and adaptation system"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.pattern_learning = PatternLearningSystem(db_manager)
        self.site_optimizer = SiteSpecificOptimizer(self.pattern_learning)
        self.performance_optimizer = PerformanceOptimizer(self.pattern_learning)
        
        self.learning_enabled = True
        self.auto_optimization = True
    
    def process_extraction_result(self, 
                                extraction_context: Dict[str, Any],
                                result: Dict[str, Any],
                                performance_metrics: Dict[str, Any]):
        """Process extraction result for learning"""
        if not self.learning_enabled:
            return
        
        try:
            domain = urlparse(extraction_context.get('url', '')).netloc
            content_type = extraction_context.get('content_type', 'unknown')
            strategy_name = extraction_context.get('strategy_name', 'unknown')
            selectors = extraction_context.get('selectors', [])
            
            success = result.get('success', False)
            quality_score = result.get('quality_score', 0.0)
            extraction_time = performance_metrics.get('extraction_time', 0.0)
            
            # Record extraction attempt
            self.pattern_learning.record_extraction_attempt(
                domain=domain,
                content_type=content_type,
                selectors=selectors,
                strategy_name=strategy_name,
                success=success,
                quality_score=quality_score,
                extraction_time=extraction_time,
                extracted_data=result.get('data')
            )
            
            # Record performance metrics
            self.performance_optimizer.record_performance_metrics(
                domain=domain,
                extraction_time=extraction_time,
                memory_usage=performance_metrics.get('memory_usage', 0.0),
                success_rate=1.0 if success else 0.0,
                quality_score=quality_score
            )
            
            # Trigger auto-optimization if enabled
            if self.auto_optimization and self._should_trigger_optimization(domain):
                self._run_optimization(domain)
                
        except Exception as e:
            print(f"Error processing extraction result for learning: {e}")
    
    def get_optimized_extraction_config(self, domain: str, content_type: str) -> Dict[str, Any]:
        """Get optimized configuration for extraction"""
        try:
            config = {
                'selectors': [],
                'strategy': None,
                'site_optimizations': {},
                'performance_hints': {}
            }
            
            # Get learned selector patterns
            best_selectors = self.pattern_learning.get_best_selectors_for_domain(domain, content_type)
            config['selectors'] = [s['selector'] for s in best_selectors]
            
            # Get best strategy
            config['strategy'] = self.pattern_learning.get_best_strategy_for_domain(domain, content_type)
            
            # Get site-specific optimizations
            config['site_optimizations'] = self.site_optimizer.get_site_config(domain)
            
            # Get performance optimization hints
            perf_analysis = self.performance_optimizer.analyze_performance_trends(domain)
            if perf_analysis:
                config['performance_hints'] = perf_analysis.get('recommendations', [])
            
            return config
            
        except Exception as e:
            print(f"Error getting optimized extraction config: {e}")
            return {}
    
    def _should_trigger_optimization(self, domain: str) -> bool:
        """Determine if optimization should be triggered for a domain"""
        try:
            # Check if we have enough data
            result = self.pattern_learning.db_manager.fetch_one("""
                SELECT COUNT(*) FROM extraction_patterns WHERE domain = ?
            """, (domain,))
            
            extraction_count = result[0] if result else 0
            
            # Trigger optimization every 50 extractions
            return extraction_count > 0 and extraction_count % 50 == 0
            
        except Exception:
            return False
    
    def _run_optimization(self, domain: str):
        """Run optimization process for a domain"""
        try:
            print(f"Running optimization for domain: {domain}")
            
            # Learn new patterns
            learned_patterns = self.pattern_learning.learn_site_patterns(domain)
            
            # Get recent extractions for analysis
            recent_extractions = self._get_recent_extractions(domain)
            
            # Optimize site configuration
            if recent_extractions:
                self.site_optimizer.optimize_for_site(domain, recent_extractions)
            
            print(f"Optimization completed for {domain}")
            
        except Exception as e:
            print(f"Error running optimization for {domain}: {e}")
    
    def _get_recent_extractions(self, domain: str, limit: int = 20) -> List[Dict]:
        """Get recent extraction data for a domain"""
        try:
            # This would typically come from extraction logs
            # For now, return empty list
            return []
        except Exception:
            return []
    
    def generate_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning system report"""
        try:
            report = {
                'pattern_stats': {},
                'site_optimizations': {},
                'performance_trends': {},
                'feedback_analysis': {},
                'recommendations': []
            }
            
            # Get pattern learning statistics
            pattern_stats = self.pattern_learning.db_manager.fetch_all("""
                SELECT domain, content_type, COUNT(*) as pattern_count,
                       AVG(confidence_score) as avg_confidence
                FROM extraction_patterns
                WHERE confidence_score >= ?
                GROUP BY domain, content_type
                ORDER BY pattern_count DESC
                LIMIT 20
            """, (self.pattern_learning.pattern_confidence_threshold,))
            
            report['pattern_stats'] = [
                {
                    'domain': row[0],
                    'content_type': row[1],
                    'pattern_count': row[2],
                    'avg_confidence': row[3]
                }
                for row in pattern_stats
            ]
            
            # Get feedback analysis
            report['feedback_analysis'] = self.pattern_learning.analyze_feedback_patterns()
            
            # Generate recommendations
            if report['feedback_analysis']['feedback_stats']:
                avg_scores = {stat['type']: stat['avg_score'] for stat in report['feedback_analysis']['feedback_stats']}
                for feedback_type, score in avg_scores.items():
                    if score < 0.7:
                        report['recommendations'].append(f"Improve {feedback_type} - current average: {score:.2f}")
            
            return report
            
        except Exception as e:
            print(f"Error generating learning report: {e}")
            return {}
