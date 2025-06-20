"""
Performance Optimization Module for SmartScrape

This module provides performance optimization utilities including
content change detection, user-defined stop conditions, batching strategies,
and modular architecture optimizations.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""
    operation_id: str
    operation_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    memory_before: float = 0.0
    memory_after: float = 0.0
    memory_peak: float = 0.0
    cpu_usage: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

class ContentChangeDetector:
    """
    Detects content changes for repeated URL visits to avoid
    unnecessary expensive processing.
    """
    
    def __init__(self, cache_size: int = 10000):
        self.cache_size = cache_size
        self.content_hashes: Dict[str, str] = {}
        self.last_checked: Dict[str, datetime] = {}
        self.change_threshold = timedelta(hours=1)  # Minimum time between checks
        self._lock = threading.Lock()
    
    def get_content_hash(self, content: str) -> str:
        """Generate a hash for content comparison."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def has_content_changed(self, url: str, content: str) -> Tuple[bool, str]:
        """
        Check if content has changed since last visit.
        
        Args:
            url: URL being checked
            content: Current content
            
        Returns:
            Tuple of (has_changed, content_hash)
        """
        content_hash = self.get_content_hash(content)
        
        with self._lock:
            # Clean old entries if cache is too large
            if len(self.content_hashes) > self.cache_size:
                self._cleanup_cache()
            
            # Check if we have previous hash
            previous_hash = self.content_hashes.get(url)
            last_check = self.last_checked.get(url)
            
            # Update cache
            self.content_hashes[url] = content_hash
            self.last_checked[url] = datetime.now()
            
            # If no previous hash or enough time has passed, consider it changed
            if previous_hash is None:
                return True, content_hash
            
            if last_check and datetime.now() - last_check < self.change_threshold:
                # Recent check, compare hashes
                return previous_hash != content_hash, content_hash
            
            # Old check, assume changed
            return True, content_hash
    
    def _cleanup_cache(self) -> None:
        """Remove old entries from cache."""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        urls_to_remove = [
            url for url, last_check in self.last_checked.items()
            if last_check < cutoff_time
        ]
        
        for url in urls_to_remove:
            self.content_hashes.pop(url, None)
            self.last_checked.pop(url, None)
    
    def force_refresh(self, url: str) -> None:
        """Force a refresh for a specific URL."""
        with self._lock:
            self.content_hashes.pop(url, None)
            self.last_checked.pop(url, None)

class StopConditionManager:
    """
    Manages user-defined stop conditions for data collection.
    """
    
    def __init__(self):
        self.conditions: Dict[str, Callable[[Dict[str, Any]], bool]] = {}
        self.condition_configs: Dict[str, Dict[str, Any]] = {}
        
        # Register built-in conditions
        self._register_builtin_conditions()
    
    def _register_builtin_conditions(self) -> None:
        """Register built-in stop conditions."""
        
        def item_count_condition(data: Dict[str, Any]) -> bool:
            """Stop when enough items are found."""
            config = self.condition_configs.get('item_count', {})
            target_count = config.get('target_count', 10)
            items = data.get('items', [])
            return len(items) >= target_count
        
        def price_range_condition(data: Dict[str, Any]) -> bool:
            """Stop when enough items in price range are found."""
            config = self.condition_configs.get('price_range', {})
            min_price = config.get('min_price', 0)
            max_price = config.get('max_price', float('inf'))
            target_count = config.get('target_count', 5)
            
            items = data.get('items', [])
            matching_items = [
                item for item in items
                if 'price' in item and min_price <= item['price'] <= max_price
            ]
            return len(matching_items) >= target_count
        
        def time_limit_condition(data: Dict[str, Any]) -> bool:
            """Stop after a time limit."""
            config = self.condition_configs.get('time_limit', {})
            start_time = config.get('start_time', datetime.now())
            time_limit = config.get('time_limit_seconds', 300)  # 5 minutes
            
            return (datetime.now() - start_time).total_seconds() >= time_limit
        
        def data_quality_condition(data: Dict[str, Any]) -> bool:
            """Stop when data quality threshold is met."""
            config = self.condition_configs.get('data_quality', {})
            min_quality_score = config.get('min_quality_score', 0.8)
            min_complete_fields = config.get('min_complete_fields', 0.9)
            
            items = data.get('items', [])
            if not items:
                return False
            
            # Calculate average quality score
            quality_scores = [item.get('quality_score', 0) for item in items]
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            # Calculate field completeness
            if items:
                total_fields = len(items[0]) if items else 0
                if total_fields > 0:
                    field_completeness = sum(
                        len([v for v in item.values() if v is not None]) / total_fields
                        for item in items
                    ) / len(items)
                else:
                    field_completeness = 0
            else:
                field_completeness = 0
            
            return (avg_quality >= min_quality_score and 
                   field_completeness >= min_complete_fields)
        
        self.conditions.update({
            'item_count': item_count_condition,
            'price_range': price_range_condition,
            'time_limit': time_limit_condition,
            'data_quality': data_quality_condition
        })
    
    def register_condition(self, name: str, condition_func: Callable[[Dict[str, Any]], bool],
                          config: Dict[str, Any] = None) -> None:
        """Register a custom stop condition."""
        self.conditions[name] = condition_func
        if config:
            self.condition_configs[name] = config
    
    def configure_condition(self, name: str, config: Dict[str, Any]) -> None:
        """Configure parameters for a stop condition."""
        self.condition_configs[name] = config
    
    def should_stop(self, data: Dict[str, Any], active_conditions: List[str] = None) -> Tuple[bool, List[str]]:
        """
        Check if any stop conditions are met.
        
        Args:
            data: Current collected data
            active_conditions: List of condition names to check
            
        Returns:
            Tuple of (should_stop, list_of_met_conditions)
        """
        if active_conditions is None:
            active_conditions = list(self.conditions.keys())
        
        met_conditions = []
        
        for condition_name in active_conditions:
            if condition_name in self.conditions:
                try:
                    if self.conditions[condition_name](data):
                        met_conditions.append(condition_name)
                except Exception as e:
                    logger.error(f"Error checking condition {condition_name}: {e}")
        
        return len(met_conditions) > 0, met_conditions

class BatchProcessor:
    """
    Handles batching strategies for large data processing,
    particularly for AI processing of aggregated data.
    """
    
    def __init__(self, max_batch_size: int = 8192, overlap_tokens: int = 200):
        self.max_batch_size = max_batch_size
        self.overlap_tokens = overlap_tokens
    
    def estimate_token_count(self, text: str) -> int:
        """Rough estimation of token count (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def split_text_into_batches(self, text: str, context_window: int = 8192) -> List[str]:
        """
        Split large text into batches that fit within context window.
        
        Args:
            text: Text to split
            context_window: Maximum tokens per batch
            
        Returns:
            List of text batches
        """
        # Rough token estimation
        estimated_tokens = self.estimate_token_count(text)
        
        if estimated_tokens <= context_window:
            return [text]
        
        # Split into sentences for better boundaries
        sentences = text.split('. ')
        batches = []
        current_batch = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_token_count(sentence + '. ')
            
            # If single sentence is too large, split by words
            if sentence_tokens > context_window:
                words = sentence.split()
                for word in words:
                    word_tokens = self.estimate_token_count(word + ' ')
                    
                    if current_tokens + word_tokens > context_window - self.overlap_tokens:
                        if current_batch:
                            batches.append(current_batch.strip())
                            # Start new batch with overlap
                            overlap_words = current_batch.split()[-self.overlap_tokens//4:]
                            current_batch = ' '.join(overlap_words) + ' ' + word
                            current_tokens = self.estimate_token_count(current_batch)
                        else:
                            current_batch = word
                            current_tokens = word_tokens
                    else:
                        current_batch += ' ' + word
                        current_tokens += word_tokens
            else:
                # Add sentence if it fits
                if current_tokens + sentence_tokens > context_window - self.overlap_tokens:
                    if current_batch:
                        batches.append(current_batch.strip())
                        current_batch = sentence + '. '
                        current_tokens = sentence_tokens
                else:
                    current_batch += sentence + '. '
                    current_tokens += sentence_tokens
        
        # Add remaining batch
        if current_batch.strip():
            batches.append(current_batch.strip())
        
        return batches
    
    async def process_batches_parallel(self, batches: List[str], 
                                     processor_func: Callable[[str], Any],
                                     max_concurrent: int = 3) -> List[Any]:
        """
        Process batches in parallel with concurrency control.
        
        Args:
            batches: List of text batches to process
            processor_func: Function to process each batch
            max_concurrent: Maximum concurrent operations
            
        Returns:
            List of processing results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch: str) -> Any:
            async with semaphore:
                return await asyncio.get_event_loop().run_in_executor(
                    None, processor_func, batch
                )
        
        tasks = [process_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing batch {i}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def merge_batch_results(self, results: List[Dict[str, Any]], 
                           merge_strategy: str = "union") -> Dict[str, Any]:
        """
        Merge results from multiple batches.
        
        Args:
            results: List of results from batch processing
            merge_strategy: Strategy for merging ("union", "intersection", "majority")
            
        Returns:
            Merged results
        """
        if not results:
            return {}
        
        if len(results) == 1:
            return results[0]
        
        merged = {}
        
        if merge_strategy == "union":
            # Combine all unique items
            all_items = []
            for result in results:
                if 'items' in result:
                    all_items.extend(result['items'])
            
            # Remove duplicates based on a key (if available)
            unique_items = []
            seen_keys = set()
            for item in all_items:
                key = item.get('id') or item.get('url') or str(hash(str(item)))
                if key not in seen_keys:
                    unique_items.append(item)
                    seen_keys.add(key)
            
            merged['items'] = unique_items
            
        elif merge_strategy == "intersection":
            # Only keep items that appear in multiple batches
            item_counts = {}
            for result in results:
                for item in result.get('items', []):
                    key = item.get('id') or item.get('url') or str(hash(str(item)))
                    item_counts[key] = item_counts.get(key, 0) + 1
            
            # Keep items that appear in more than one batch
            threshold = len(results) // 2 + 1
            common_items = []
            for result in results:
                for item in result.get('items', []):
                    key = item.get('id') or item.get('url') or str(hash(str(item)))
                    if item_counts[key] >= threshold:
                        common_items.append(item)
                        break
            
            merged['items'] = common_items
            
        elif merge_strategy == "majority":
            # Use voting for conflicting information
            field_votes = {}
            for result in results:
                for item in result.get('items', []):
                    for field, value in item.items():
                        if field not in field_votes:
                            field_votes[field] = {}
                        field_votes[field][value] = field_votes[field].get(value, 0) + 1
            
            # Create consensus items
            consensus_items = []
            for result in results:
                for item in result.get('items', []):
                    consensus_item = {}
                    for field, value in item.items():
                        votes = field_votes[field]
                        most_voted = max(votes.items(), key=lambda x: x[1])
                        consensus_item[field] = most_voted[0]
                    consensus_items.append(consensus_item)
                    break  # One consensus item per result
            
            merged['items'] = consensus_items
        
        # Merge metadata
        merged['batch_count'] = len(results)
        merged['merge_strategy'] = merge_strategy
        merged['total_items_before_merge'] = sum(len(r.get('items', [])) for r in results)
        
        return merged

class PerformanceOptimizer:
    """
    Main performance optimization coordinator.
    """
    
    def __init__(self):
        self.content_detector = ContentChangeDetector()
        self.stop_manager = StopConditionManager()
        self.batch_processor = BatchProcessor()
        self.metrics: List[PerformanceMetrics] = []
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def create_metrics_context(self, operation_id: str, operation_type: str):
        """Create a performance metrics context manager."""
        return PerformanceMetricsContext(operation_id, operation_type, self.metrics)
    
    def optimize_extraction_pipeline(self, urls: List[str], 
                                   extraction_func: Callable[[str], Dict[str, Any]],
                                   stop_conditions: List[str] = None,
                                   max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Optimize extraction pipeline with all performance enhancements.
        
        Args:
            urls: List of URLs to process
            extraction_func: Function to extract data from each URL
            stop_conditions: Stop conditions to apply
            max_concurrent: Maximum concurrent extractions
            
        Returns:
            Optimized extraction results
        """
        start_time = datetime.now()
        aggregated_data = {'items': [], 'urls_processed': [], 'urls_skipped': []}
        
        # Configure time limit condition
        if stop_conditions:
            self.stop_manager.configure_condition('time_limit', {
                'start_time': start_time,
                'time_limit_seconds': 600  # 10 minutes default
            })
        
        # Process URLs with optimization
        with self._executor as executor:
            future_to_url = {}
            
            for url in urls[:max_concurrent]:  # Start with initial batch
                future = executor.submit(self._extract_with_optimization, url, extraction_func)
                future_to_url[future] = url
            
            remaining_urls = urls[max_concurrent:]
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                
                try:
                    result = future.result()
                    if result and result.get('items'):
                        aggregated_data['items'].extend(result['items'])
                        aggregated_data['urls_processed'].append(url)
                    else:
                        aggregated_data['urls_skipped'].append(url)
                    
                    # Check stop conditions
                    if stop_conditions:
                        should_stop, met_conditions = self.stop_manager.should_stop(
                            aggregated_data, stop_conditions
                        )
                        if should_stop:
                            logger.info(f"Stopping extraction: conditions met {met_conditions}")
                            break
                    
                    # Submit next URL if available
                    if remaining_urls:
                        next_url = remaining_urls.pop(0)
                        future = executor.submit(self._extract_with_optimization, next_url, extraction_func)
                        future_to_url[future] = next_url
                        
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
                    aggregated_data['urls_skipped'].append(url)
        
        # Add optimization metadata
        aggregated_data['optimization_metadata'] = {
            'total_time': (datetime.now() - start_time).total_seconds(),
            'content_change_detection_enabled': True,
            'stop_conditions_used': stop_conditions or [],
            'concurrent_processing': True
        }
        
        return aggregated_data
    
    def _extract_with_optimization(self, url: str, extraction_func: Callable[[str], Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract data with performance optimizations."""
        with self.create_metrics_context(f"extract_{url}", "extraction"):
            # First, get content to check for changes
            try:
                # This is a simplified approach - in practice, you'd want to
                # get content first, then check for changes
                result = extraction_func(url)
                
                if not result:
                    return None
                
                # Simulate content change detection
                content = str(result)  # Simplified content representation
                has_changed, content_hash = self.content_detector.has_content_changed(url, content)
                
                if not has_changed:
                    logger.info(f"Content unchanged for {url}, skipping expensive processing")
                    return None
                
                return result
                
            except Exception as e:
                logger.error(f"Error in optimized extraction for {url}: {e}")
                return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance optimization summary."""
        if not self.metrics:
            return {'message': 'No performance data available'}
        
        successful_metrics = [m for m in self.metrics if m.success]
        failed_metrics = [m for m in self.metrics if not m.success]
        
        total_operations = len(self.metrics)
        success_rate = len(successful_metrics) / total_operations if total_operations > 0 else 0
        
        avg_duration = np.mean([m.duration for m in successful_metrics]) if successful_metrics else 0
        avg_memory_usage = np.mean([m.memory_peak - m.memory_before for m in successful_metrics]) if successful_metrics else 0
        
        return {
            'total_operations': total_operations,
            'success_rate': success_rate,
            'avg_duration_seconds': avg_duration,
            'avg_memory_usage_mb': avg_memory_usage,
            'failed_operations': len(failed_metrics),
            'content_cache_size': len(self.content_detector.content_hashes),
            'optimization_features': [
                'content_change_detection',
                'user_defined_stop_conditions',
                'intelligent_batching',
                'parallel_processing'
            ]
        }


class PerformanceMetricsContext:
    """Context manager for collecting performance metrics."""
    
    def __init__(self, operation_id: str, operation_type: str, metrics_list: List[PerformanceMetrics]):
        self.metrics = PerformanceMetrics(
            operation_id=operation_id,
            operation_type=operation_type,
            start_time=datetime.now()
        )
        self.metrics_list = metrics_list
        self.process = psutil.Process()
    
    def __enter__(self):
        self.metrics.memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
        return self.metrics
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metrics.end_time = datetime.now()
        self.metrics.duration = (self.metrics.end_time - self.metrics.start_time).total_seconds()
        self.metrics.memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
        self.metrics.memory_peak = max(self.metrics.memory_before, self.metrics.memory_after)
        self.metrics.cpu_usage = self.process.cpu_percent()
        
        if exc_type is not None:
            self.metrics.success = False
            self.metrics.error_message = str(exc_val)
        
        self.metrics_list.append(self.metrics)


# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer
