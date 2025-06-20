#!/usr/bin/env python3
"""
Memory Monitor for SmartScrape

This module provides memory monitoring and cleanup functionality to prevent
memory leaks and optimize resource usage during extraction operations.
"""

import psutil
import gc
import logging
import asyncio
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor and manage memory usage for SmartScrape operations"""
    
    def __init__(self, max_memory_mb: int = 1024, cleanup_threshold: float = 0.8):
        """
        Initialize memory monitor
        
        Args:
            max_memory_mb: Maximum memory usage in MB before cleanup
            cleanup_threshold: Memory usage percentage to trigger cleanup (0.0-1.0)
        """
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold = cleanup_threshold
        self.process = psutil.Process()
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(minutes=5)  # Minimum interval between cleanups
        
        logger.info(f"MemoryMonitor initialized: max={max_memory_mb}MB, threshold={cleanup_threshold}")
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage statistics"""
        try:
            memory_info = self.process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': self.process.memory_percent(),   # Process memory percentage
                'available_mb': system_memory.available / 1024 / 1024,
                'system_percent': system_memory.percent,    # System memory usage
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {
                'rss_mb': 0,
                'vms_mb': 0,
                'percent': 0,
                'available_mb': 0,
                'system_percent': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        try:
            memory_usage = self.get_memory_usage()
            
            # Check if we exceed the absolute limit
            if memory_usage['rss_mb'] > self.max_memory_mb:
                logger.warning(f"Memory usage {memory_usage['rss_mb']:.1f}MB exceeds limit {self.max_memory_mb}MB")
                return True
            
            # Check if we exceed the percentage threshold
            if memory_usage['percent'] > (self.cleanup_threshold * 100):
                logger.warning(f"Memory usage {memory_usage['percent']:.1f}% exceeds threshold {self.cleanup_threshold * 100}%")
                return True
            
            # Check if system memory is low
            if memory_usage['system_percent'] > 90:
                logger.warning(f"System memory usage {memory_usage['system_percent']:.1f}% is critical")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking cleanup need: {e}")
            return False
    
    def cleanup(self, force: bool = False) -> Dict:
        """
        Perform memory cleanup
        
        Args:
            force: Force cleanup even if interval hasn't passed
            
        Returns:
            Dict with cleanup results
        """
        now = datetime.now()
        
        # Check cleanup interval unless forced
        if not force and (now - self.last_cleanup) < self.cleanup_interval:
            return {
                'performed': False,
                'reason': 'cleanup_interval_not_met',
                'next_cleanup_available': (self.last_cleanup + self.cleanup_interval).isoformat()
            }
        
        try:
            # Get memory usage before cleanup
            before_memory = self.get_memory_usage()
            
            # Perform garbage collection
            collected = gc.collect()
            
            # Get memory usage after cleanup
            after_memory = self.get_memory_usage()
            
            # Update last cleanup time
            self.last_cleanup = now
            
            # Calculate memory freed
            memory_freed = before_memory['rss_mb'] - after_memory['rss_mb']
            
            result = {
                'performed': True,
                'timestamp': now.isoformat(),
                'objects_collected': collected,
                'memory_before_mb': before_memory['rss_mb'],
                'memory_after_mb': after_memory['rss_mb'],
                'memory_freed_mb': memory_freed,
                'freed_percentage': (memory_freed / max(before_memory['rss_mb'], 1)) * 100
            }
            
            if memory_freed > 0:
                logger.info(f"Memory cleanup freed {memory_freed:.1f}MB ({result['freed_percentage']:.1f}%), collected {collected} objects")
            else:
                logger.info(f"Memory cleanup completed, collected {collected} objects, no significant memory freed")
            
            return result
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return {
                'performed': False,
                'error': str(e),
                'timestamp': now.isoformat()
            }
    
    def auto_cleanup_if_needed(self) -> Optional[Dict]:
        """Automatically cleanup if needed"""
        if self.should_cleanup():
            return self.cleanup()
        return None
    
    def get_memory_summary(self) -> Dict:
        """Get a summary of memory status"""
        usage = self.get_memory_usage()
        
        return {
            'current_usage_mb': usage['rss_mb'],
            'max_limit_mb': self.max_memory_mb,
            'usage_percentage': (usage['rss_mb'] / self.max_memory_mb) * 100,
            'cleanup_needed': self.should_cleanup(),
            'system_memory_percent': usage['system_percent'],
            'last_cleanup': self.last_cleanup.isoformat(),
            'status': self._get_status(usage)
        }
    
    def _get_status(self, usage: Dict) -> str:
        """Get memory status string"""
        if usage['rss_mb'] > self.max_memory_mb:
            return 'critical'
        elif usage['percent'] > (self.cleanup_threshold * 100):
            return 'warning'
        elif usage['system_percent'] > 90:
            return 'system_critical'
        else:
            return 'normal'

# Global memory monitor instance
memory_monitor = MemoryMonitor()

def monitor_memory_during_extraction(func):
    """Decorator to monitor memory usage during extraction operations"""
    
    import functools
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Check memory before extraction
        memory_monitor.auto_cleanup_if_needed()
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Check memory after extraction
            memory_monitor.auto_cleanup_if_needed()
            
            return result
            
        except Exception as e:
            # Force cleanup on error to prevent memory leaks
            cleanup_result = memory_monitor.cleanup(force=True)
            logger.warning(f"Forced memory cleanup after error: {cleanup_result}")
            raise
    
    return wrapper
