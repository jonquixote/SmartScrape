"""
Metrics Collection System for SmartScrape

This module provides comprehensive metrics collection for monitoring
extraction performance, success rates, and system health.
"""

import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from urllib.parse import urlparse
import asyncio
import threading

logger = logging.getLogger(__name__)

@dataclass
class ExtractionMetrics:
    """Metrics for a single extraction attempt"""
    url: str
    strategy: str
    success: bool
    response_time: float
    content_length: int
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: float = None
    attempt_number: int = 1
    cache_hit: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class SystemMetrics:
    """System-level metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    queue_size: int
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class MetricsCollector:
    """Comprehensive metrics collection system"""
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.extraction_metrics: deque = deque(maxlen=max_history_size)
        self.system_metrics: deque = deque(maxlen=1000)  # Keep last 1000 system metrics
        self.lock = threading.Lock()
        
        # Aggregated statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0.0,
            'avg_response_time': 0.0,
            'strategies_used': defaultdict(int),
            'error_types': defaultdict(int),
            'domains_processed': set(),
            'hourly_stats': defaultdict(lambda: {'requests': 0, 'successes': 0, 'failures': 0}),
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Performance tracking
        self.performance_windows = {
            '1min': deque(maxlen=60),
            '5min': deque(maxlen=300),
            '1hour': deque(maxlen=3600)
        }
        
        # Start background metrics collection
        self._start_system_monitoring()
    
    def record_extraction(self, metrics: ExtractionMetrics):
        """Record extraction metrics"""
        with self.lock:
            self.extraction_metrics.append(metrics)
            self._update_aggregated_stats(metrics)
            self._update_performance_windows(metrics)
    
    def record_system_metrics(self, metrics: SystemMetrics):
        """Record system metrics"""
        with self.lock:
            self.system_metrics.append(metrics)
    
    def _update_aggregated_stats(self, metrics: ExtractionMetrics):
        """Update aggregated statistics"""
        self.stats['total_requests'] += 1
        
        if metrics.success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
            if metrics.error_type:
                self.stats['error_types'][metrics.error_type] += 1
        
        # Update response time
        self.stats['total_response_time'] += metrics.response_time
        self.stats['avg_response_time'] = self.stats['total_response_time'] / self.stats['total_requests']
        
        # Update strategy usage
        self.stats['strategies_used'][metrics.strategy] += 1
        
        # Track domains
        domain = urlparse(metrics.url).netloc
        self.stats['domains_processed'].add(domain)
        
        # Track cache performance
        if metrics.cache_hit:
            self.stats['cache_hits'] += 1
        else:
            self.stats['cache_misses'] += 1
        
        # Track hourly stats
        hour_key = int(metrics.timestamp // 3600)
        hourly = self.stats['hourly_stats'][hour_key]
        hourly['requests'] += 1
        if metrics.success:
            hourly['successes'] += 1
        else:
            hourly['failures'] += 1
    
    def _update_performance_windows(self, metrics: ExtractionMetrics):
        """Update sliding window performance metrics"""
        current_time = time.time()
        
        for window_name, window in self.performance_windows.items():
            window.append({
                'timestamp': current_time,
                'success': metrics.success,
                'response_time': metrics.response_time,
                'error_type': metrics.error_type
            })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        with self.lock:
            current_time = time.time()
            
            # Basic statistics
            total_requests = self.stats['total_requests']
            success_rate = (self.stats['successful_requests'] / max(1, total_requests)) * 100
            
            # Most used strategy
            most_used_strategy = None
            if self.stats['strategies_used']:
                most_used_strategy = max(self.stats['strategies_used'].items(), key=lambda x: x[1])[0]
            
            # Most common error
            most_common_error = None
            if self.stats['error_types']:
                most_common_error = max(self.stats['error_types'].items(), key=lambda x: x[1])[0]
            
            # Cache performance
            total_cache_requests = self.stats['cache_hits'] + self.stats['cache_misses']
            cache_hit_rate = (self.stats['cache_hits'] / max(1, total_cache_requests)) * 100
            
            # Recent performance (last hour)
            recent_metrics = [
                m for m in self.extraction_metrics 
                if current_time - m.timestamp < 3600
            ]
            
            recent_success_rate = 0
            recent_avg_response_time = 0
            if recent_metrics:
                recent_successes = sum(1 for m in recent_metrics if m.success)
                recent_success_rate = (recent_successes / len(recent_metrics)) * 100
                recent_avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
            
            # System metrics
            latest_system_metrics = None
            if self.system_metrics:
                latest_system_metrics = asdict(self.system_metrics[-1])
            
            return {
                'summary': {
                    'total_requests': total_requests,
                    'success_rate': round(success_rate, 2),
                    'avg_response_time': round(self.stats['avg_response_time'], 2),
                    'domains_processed': len(self.stats['domains_processed']),
                    'cache_hit_rate': round(cache_hit_rate, 2)
                },
                'recent_performance': {
                    'last_hour_requests': len(recent_metrics),
                    'last_hour_success_rate': round(recent_success_rate, 2),
                    'last_hour_avg_response_time': round(recent_avg_response_time, 2)
                },
                'strategies': dict(self.stats['strategies_used']),
                'errors': dict(self.stats['error_types']),
                'most_used_strategy': most_used_strategy,
                'most_common_error': most_common_error,
                'system_metrics': latest_system_metrics,
                'report_generated_at': current_time
            }
    
    def get_time_series_data(self, window: str = '1hour') -> List[Dict]:
        """Get time series performance data"""
        if window not in self.performance_windows:
            raise ValueError(f"Invalid window: {window}. Valid windows: {list(self.performance_windows.keys())}")
        
        with self.lock:
            current_time = time.time()
            window_data = list(self.performance_windows[window])
            
            # Group by time buckets (5-minute intervals)
            bucket_size = 300  # 5 minutes
            buckets = defaultdict(lambda: {'requests': 0, 'successes': 0, 'total_response_time': 0})
            
            for data_point in window_data:
                bucket_time = int(data_point['timestamp'] // bucket_size) * bucket_size
                bucket = buckets[bucket_time]
                bucket['requests'] += 1
                bucket['total_response_time'] += data_point['response_time']
                if data_point['success']:
                    bucket['successes'] += 1
            
            # Convert to time series format
            time_series = []
            for bucket_time, bucket_data in sorted(buckets.items()):
                time_series.append({
                    'timestamp': bucket_time,
                    'requests': bucket_data['requests'],
                    'success_rate': (bucket_data['successes'] / bucket_data['requests']) * 100,
                    'avg_response_time': bucket_data['total_response_time'] / bucket_data['requests']
                })
            
            return time_series
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get detailed error analysis"""
        with self.lock:
            current_time = time.time()
            
            # Recent errors (last 24 hours)
            recent_errors = [
                m for m in self.extraction_metrics 
                if not m.success and current_time - m.timestamp < 86400
            ]
            
            # Group errors by type
            error_analysis = defaultdict(lambda: {
                'count': 0,
                'recent_count': 0,
                'affected_domains': set(),
                'failed_strategies': defaultdict(int)
            })
            
            for metrics in self.extraction_metrics:
                if not metrics.success and metrics.error_type:
                    error_info = error_analysis[metrics.error_type]
                    error_info['count'] += 1
                    error_info['affected_domains'].add(urlparse(metrics.url).netloc)
                    error_info['failed_strategies'][metrics.strategy] += 1
                    
                    if current_time - metrics.timestamp < 86400:
                        error_info['recent_count'] += 1
            
            # Convert sets to lists for JSON serialization
            for error_type, info in error_analysis.items():
                info['affected_domains'] = list(info['affected_domains'])
                info['failed_strategies'] = dict(info['failed_strategies'])
            
            return {
                'total_errors': len([m for m in self.extraction_metrics if not m.success]),
                'recent_errors': len(recent_errors),
                'error_breakdown': dict(error_analysis),
                'error_rate_trend': self._calculate_error_trend()
            }
    
    def _calculate_error_trend(self) -> str:
        """Calculate error rate trend (improving/degrading/stable)"""
        current_time = time.time()
        
        # Compare last hour vs previous hour
        last_hour_metrics = [
            m for m in self.extraction_metrics 
            if current_time - 3600 < m.timestamp <= current_time
        ]
        
        prev_hour_metrics = [
            m for m in self.extraction_metrics 
            if current_time - 7200 < m.timestamp <= current_time - 3600
        ]
        
        if not last_hour_metrics or not prev_hour_metrics:
            return "insufficient_data"
        
        last_hour_error_rate = (len([m for m in last_hour_metrics if not m.success]) / len(last_hour_metrics)) * 100
        prev_hour_error_rate = (len([m for m in prev_hour_metrics if not m.success]) / len(prev_hour_metrics)) * 100
        
        if last_hour_error_rate < prev_hour_error_rate - 5:
            return "improving"
        elif last_hour_error_rate > prev_hour_error_rate + 5:
            return "degrading"
        else:
            return "stable"
    
    def _start_system_monitoring(self):
        """Start background system monitoring"""
        def monitor_system():
            while True:
                try:
                    import psutil
                    
                    # Collect system metrics
                    cpu_usage = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    
                    system_metrics = SystemMetrics(
                        cpu_usage=cpu_usage,
                        memory_usage=memory.percent,
                        disk_usage=disk.percent,
                        active_connections=len(psutil.net_connections()),
                        queue_size=0  # This would be filled by the actual queue size
                    )
                    
                    self.record_system_metrics(system_metrics)
                    
                except ImportError:
                    logger.warning("psutil not available, system monitoring disabled")
                    break
                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                
                time.sleep(60)  # Monitor every minute
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        report = self.get_performance_report()
        
        if format.lower() == "json":
            return json.dumps(report, indent=2, default=str)
        elif format.lower() == "csv":
            # Simple CSV export of basic metrics
            lines = [
                "metric,value",
                f"total_requests,{report['summary']['total_requests']}",
                f"success_rate,{report['summary']['success_rate']}",
                f"avg_response_time,{report['summary']['avg_response_time']}",
                f"cache_hit_rate,{report['summary']['cache_hit_rate']}"
            ]
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        with self.lock:
            self.extraction_metrics.clear()
            self.system_metrics.clear()
            self.stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_response_time': 0.0,
                'avg_response_time': 0.0,
                'strategies_used': defaultdict(int),
                'error_types': defaultdict(int),
                'domains_processed': set(),
                'hourly_stats': defaultdict(lambda: {'requests': 0, 'successes': 0, 'failures': 0}),
                'cache_hits': 0,
                'cache_misses': 0
            }
            
            for window in self.performance_windows.values():
                window.clear()
            
            logger.info("All metrics have been reset")

# Global metrics collector instance
metrics_collector = MetricsCollector()
