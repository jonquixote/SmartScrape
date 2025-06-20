"""
Metrics Analyzer Module

This module provides comprehensive functionality for performance monitoring
and resource tracking during web scraping operations. It includes:

1. Performance monitoring
    - Execution time tracking
    - CPU usage monitoring
    - Operation performance statistics

2. Resource tracking
    - Memory usage monitoring
    - CPU consumption analysis
    - Bandwidth utilization tracking

3. Bandwidth analysis
    - Network I/O tracking
    - Domain-specific bandwidth usage
    - Optimization recommendations

4. Data collection
    - Metrics collection during scraping operations
    - Storage of historical performance data
    - Trend analysis

5. Visualization capabilities
    - Generation of performance charts
    - Resource utilization plots
    - Time series analysis

Usage:
     metrics = MetricsAnalyzer()
     
     # Start tracking a new operation
     with metrics.track_operation("page_scraping"):
          # Perform scraping operations
          ...
     
     # Get performance report
     report = metrics.generate_report()
     
     # Visualize metrics
     metrics.plot_memory_usage()
     metrics.plot_bandwidth_usage()
"""

import time
import threading
import psutil
import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Generator
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
from functools import wraps
from contextlib import contextmanager

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

from core.service_interface import BaseService

logger = logging.getLogger("MetricsAnalyzer")

class MetricsAnalyzer(BaseService):
    """
    Comprehensive metrics analyzer for monitoring performance and resource usage.
    
    This class provides tools to track and analyze:
    - Execution times
    - CPU usage
    - Memory consumption
    - Bandwidth usage
    - Operation counts and durations
    """
    
    def __init__(self, 
                 metrics_dir: str = "metrics",
                 save_interval: int = 300,
                 keep_history: bool = True,
                 history_limit: int = 7,
                 track_system_resources: bool = True):
        """
        Initialize the metrics analyzer.
        
        Args:
            metrics_dir: Directory to store metrics data
            save_interval: Interval in seconds to periodically save metrics
            keep_history: Whether to maintain historical metrics data
            history_limit: Number of days to keep historical data
            track_system_resources: Whether to track system resources
        """
        self._initialized = False
        self.metrics_dir = metrics_dir
        self.save_interval = save_interval
        self.keep_history = keep_history
        self.history_limit = history_limit
        self.track_system_resources = track_system_resources
        
        # These will be initialized in the initialize() method
        self.metrics = None
        self.operation_timers = None
        self.operation_counts = None
        self.bandwidth_tracker = None
        self._tracking_active = False
        self._tracking_thread = None
        self._save_timer = None
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the service with the given configuration."""
        if self._initialized:
            return
            
        # Apply configuration if provided
        if config:
            self.metrics_dir = config.get('metrics_dir', self.metrics_dir)
            self.save_interval = config.get('save_interval', self.save_interval)
            self.keep_history = config.get('keep_history', self.keep_history)
            self.history_limit = config.get('history_limit', self.history_limit)
            self.track_system_resources = config.get('track_system_resources', self.track_system_resources)
        
        # Create metrics directory if it doesn't exist
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            "operations": defaultdict(list),
            "resources": {
                "memory": [],
                "cpu": [],
                "disk": [],
                "network": []
            },
            "bandwidth": {
                "domains": defaultdict(int),
                "total_received": 0,
                "total_sent": 0,
                "time_series": []
            },
            "errors": defaultdict(int),
            "execution_times": defaultdict(list),
            "timestamps": {
                "start": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat()
            }
        }
        
        # Initialize performance counters
        self.operation_timers = {}
        self.operation_counts = defaultdict(int)
        self.bandwidth_tracker = BandwidthTracker()
        
        # Set up tracking thread if system resource tracking is enabled
        if self.track_system_resources:
            self._tracking_active = True
            self._tracking_thread = threading.Thread(target=self._resource_tracking_loop, daemon=True)
            self._tracking_thread.start()
        else:
            self._tracking_active = False
            self._tracking_thread = None
        
        # Set up periodic save timer
        if self.save_interval > 0:
            self._save_timer = threading.Timer(self.save_interval, self._periodic_save)
            self._save_timer.daemon = True
            self._save_timer.start()
        else:
            self._save_timer = None
            
        self._initialized = True
        logger.info(f"Metrics analyzer initialized with metrics directory: {self.metrics_dir}")
    
    def shutdown(self) -> None:
        """Perform cleanup and shutdown operations."""
        self.cleanup()
        self._initialized = False
        logger.info("MetricsAnalyzer service shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "metrics_analyzer"
    
    def _resource_tracking_loop(self) -> None:
        """Background thread for tracking system resources."""
        try:
            while self._tracking_active:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Sleep interval (collect every 5 seconds)
                time.sleep(5)
                
        except Exception as e:
            logger.error(f"Error in resource tracking loop: {str(e)}")
    
    def _collect_system_metrics(self) -> None:
        """Collect current system metrics."""
        try:
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_data = {
                "timestamp": timestamp,
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            }
            self.metrics["resources"]["memory"].append(memory_data)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_data = {
                "timestamp": timestamp,
                "percent": cpu_percent,
                "count": psutil.cpu_count()
            }
            self.metrics["resources"]["cpu"].append(cpu_data)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_data = {
                "timestamp": timestamp,
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            }
            self.metrics["resources"]["disk"].append(disk_data)
            
            # Network I/O
            network = psutil.net_io_counters()
            network_data = {
                "timestamp": timestamp,
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            self.metrics["resources"]["network"].append(network_data)
            
            # Update bandwidth metrics
            self.metrics["bandwidth"]["total_received"] = network.bytes_recv
            self.metrics["bandwidth"]["total_sent"] = network.bytes_sent
            self.metrics["bandwidth"]["time_series"].append({
                "timestamp": timestamp,
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv
            })
            
            # Trim historical data if needed
            self._trim_metrics_data()
            
            # Update last update timestamp
            self.metrics["timestamps"]["last_update"] = timestamp
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    def _trim_metrics_data(self) -> None:
        """Trim metrics data to prevent excessive memory usage."""
        # Keep only the last 1000 data points for high-frequency metrics
        for resource_type in ["memory", "cpu", "disk", "network"]:
            if len(self.metrics["resources"][resource_type]) > 1000:
                self.metrics["resources"][resource_type] = self.metrics["resources"][resource_type][-1000:]
        
        # Trim bandwidth time series
        if len(self.metrics["bandwidth"]["time_series"]) > 1000:
            self.metrics["bandwidth"]["time_series"] = self.metrics["bandwidth"]["time_series"][-1000:]
    
    def _periodic_save(self) -> None:
        """Periodically save metrics to disk."""
        try:
            self.save_metrics()
        except Exception as e:
            logger.error(f"Error in periodic metrics save: {str(e)}")
        finally:
            # Reschedule the timer if still active
            if self._tracking_active and self.save_interval > 0:
                self._save_timer = threading.Timer(self.save_interval, self._periodic_save)
                self._save_timer.daemon = True
                self._save_timer.start()
    
    def track_request(self, 
                      url: str, 
                      size: int, 
                      duration: float, 
                      status_code: int,
                      request_type: str = "GET",
                      is_successful: bool = True) -> None:
        """
        Track a single HTTP request.
        
        Args:
            url: The requested URL
            size: Size of the response in bytes
            duration: Request duration in seconds
            status_code: HTTP status code
            request_type: Request method (GET, POST, etc.)
            is_successful: Whether the request was successful
        """
        # Extract domain from URL
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
        except Exception:
            domain = "unknown"
        
        # Update domain bandwidth
        if "bandwidth" not in self.metrics:
            self.metrics["bandwidth"] = {
                "domains": defaultdict(int),
                "total_received": 0,
                "total_sent": 0,
                "time_series": []
            }
        if "domains" not in self.metrics["bandwidth"]:
            self.metrics["bandwidth"]["domains"] = defaultdict(int)
        
        self.metrics["bandwidth"]["domains"][domain] += size
        
        # Update bandwidth tracker
        if self.bandwidth_tracker is None:
            self.bandwidth_tracker = BandwidthTracker()
        self.bandwidth_tracker.add_request(domain, size, request_type)
        
        # Update operation metrics
        operation_name = f"request_{request_type.lower()}"
        if self.operation_counts is None:
            self.operation_counts = defaultdict(int)
        self.operation_counts[operation_name] += 1
        
        if "operations" not in self.metrics:
            self.metrics["operations"] = defaultdict(list)
        if "execution_times" not in self.metrics:
            self.metrics["execution_times"] = defaultdict(list)
        if "errors" not in self.metrics:
            self.metrics["errors"] = defaultdict(int)
        
        self.metrics["operations"][operation_name].append({
            "timestamp": datetime.now().isoformat(),
            "url": url,
            "domain": domain,
            "size": size,
            "duration": duration,
            "status_code": status_code,
            "successful": is_successful
        })
        
        # Update execution times
        self.metrics["execution_times"][operation_name].append(duration)
        
        # Track errors if not successful
        if not is_successful:
            error_category = f"status_{status_code}" if status_code else "network_error"
            self.metrics["errors"][error_category] += 1
    
    @contextmanager
    def track_operation(self, operation_name: str) -> Generator[None, None, None]:
        """
        Context manager to track operation execution time.
        
        Args:
            operation_name: Name of the operation to track
            
        Example:
            with metrics.track_operation("parse_page"):
                # code to parse page
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            # Calculate duration
            duration = time.time() - start_time
            
            # Update metrics
            self.operation_counts[operation_name] += 1
            self.metrics["execution_times"][operation_name].append(duration)
            
            self.metrics["operations"][operation_name].append({
                "timestamp": datetime.now().isoformat(),
                "duration": duration
            })
    
    def track_function(self, operation_name: Optional[str] = None):
        """
        Decorator to track function execution time.
        
        Args:
            operation_name: Optional name of the operation, defaults to function name
            
        Example:
            @metrics.track_function("extract_data")
            def extract_data_from_page(html):
                # extraction code
                return data
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Use provided operation name or function name
                op_name = operation_name or func.__name__
                
                with self.track_operation(op_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def track_async_function(self, operation_name: Optional[str] = None):
        """
        Decorator to track async function execution time.
        
        Args:
            operation_name: Optional name of the operation, defaults to function name
            
        Example:
            @metrics.track_async_function("fetch_page")
            async def fetch_page(url):
                # fetch code
                return response
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Use provided operation name or function name
                op_name = operation_name or func.__name__
                
                start_time = time.time()
                
                try:
                    return await func(*args, **kwargs)
                finally:
                    # Calculate duration
                    duration = time.time() - start_time
                    
                    # Update metrics
                    self.operation_counts[op_name] += 1
                    self.metrics["execution_times"][op_name].append(duration)
                    
                    self.metrics["operations"][op_name].append({
                        "timestamp": datetime.now().isoformat(),
                        "duration": duration
                    })
            return wrapper
        return decorator
    
    def record_error(self, error_type: str, details: Optional[str] = None) -> None:
        """
        Record an error.
        
        Args:
            error_type: Type or category of error
            details: Optional error details
        """
        self.metrics["errors"][error_type] += 1
        
        if details:
            # Store detailed error information
            if "error_details" not in self.metrics:
                self.metrics["error_details"] = []
            
            self.metrics["error_details"].append({
                "timestamp": datetime.now().isoformat(),
                "type": error_type,
                "details": details
            })
    
    def record_search_metrics(self, query: str, url: Optional[str] = None, 
                             success: bool = True, execution_time: float = 0.0,
                             result_count: int = 0, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Record metrics for search operations.
        
        Args:
            query: The search query
            url: Optional URL that was searched
            success: Whether the search was successful
            execution_time: Time taken for the search operation
            result_count: Number of results returned
            metrics: Additional metrics dictionary
        """
        # Ensure metrics is initialized
        if self.metrics is None:
            logger.warning("MetricsAnalyzer not initialized, initializing with defaults")
            self.metrics = {
                "search_metrics": [],
                "operations": defaultdict(list),
                "resources": {
                    "memory": [],
                    "cpu": [],
                    "disk": [],
                    "network": []
                },
                "bandwidth": {
                    "domains": defaultdict(int),
                    "total_received": 0,
                    "total_sent": 0,
                    "time_series": []
                },
                "errors": defaultdict(int),
                "execution_times": defaultdict(list),
                "timestamps": {
                    "start": datetime.now().isoformat(),
                    "last_update": datetime.now().isoformat()
                }
            }
            # Initialize operation counts and bandwidth tracker if needed
            if self.operation_counts is None:
                self.operation_counts = defaultdict(int)
            if self.bandwidth_tracker is None:
                self.bandwidth_tracker = BandwidthTracker()
        
        # Initialize search metrics if not exists
        if "search_metrics" not in self.metrics:
            self.metrics["search_metrics"] = []
        
        # Record the search metrics
        search_record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "url": url,
            "success": success,
            "execution_time": execution_time,
            "result_count": result_count
        }
        
        # Add additional metrics if provided
        if metrics:
            search_record.update(metrics)
        
        self.metrics["search_metrics"].append(search_record)
        
        # Update operation tracking
        self.track_request(
            url=url or query,
            size=result_count,
            duration=execution_time,
            status_code=200 if success else 500,
            request_type="SEARCH",
            is_successful=success
        )

    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Dictionary with operation statistics
        """
        if operation_name not in self.metrics["execution_times"] or not self.metrics["execution_times"][operation_name]:
            return {
                "count": 0,
                "total_time": 0,
                "avg_time": 0,
                "min_time": 0,
                "max_time": 0,
                "median_time": 0
            }
        
        execution_times = self.metrics["execution_times"][operation_name]
        
        return {
            "count": self.operation_counts[operation_name],
            "total_time": sum(execution_times),
            "avg_time": sum(execution_times) / len(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "median_time": sorted(execution_times)[len(execution_times) // 2]
        }
    
    def get_all_operation_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all operations.
        
        Returns:
            Dictionary with statistics for all operations
        """
        stats = {}
        
        for operation_name in self.operation_counts.keys():
            stats[operation_name] = self.get_operation_stats(operation_name)
        
        return stats
    
    def get_bandwidth_stats(self) -> Dict[str, Any]:
        """
        Get bandwidth usage statistics.
        
        Returns:
            Dictionary with bandwidth statistics
        """
        return {
            "total_received": self.metrics["bandwidth"]["total_received"],
            "total_sent": self.metrics["bandwidth"]["total_sent"],
            "by_domain": dict(self.metrics["bandwidth"]["domains"]),
            "top_domains": self._get_top_bandwidth_domains(5)
        }
    
    def _get_top_bandwidth_domains(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get top domains by bandwidth usage.
        
        Args:
            limit: Number of top domains to return
            
        Returns:
            List of top domains with usage data
        """
        # Sort domains by bandwidth
        domains = sorted(
            self.metrics["bandwidth"]["domains"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top domains
        return [
            {"domain": domain, "bytes": bytes_used}
            for domain, bytes_used in domains[:limit]
        ]
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get resource usage statistics.
        
        Returns:
            Dictionary with resource usage data
        """
        # Return current resource usage
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent
                },
                "cpu": {
                    "percent": psutil.cpu_percent(interval=0.1),
                    "count": psutil.cpu_count()
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                }
            }
        except Exception as e:
            logger.error(f"Error getting resource usage: {str(e)}")
            return {}
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        return {
            "total_errors": sum(self.metrics["errors"].values()),
            "by_type": dict(self.metrics["errors"])
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary with performance report data
        """
        try:
            # Calculate total runtime
            start_time = datetime.fromisoformat(self.metrics["timestamps"]["start"])
            runtime_seconds = (datetime.now() - start_time).total_seconds()
            
            # Generate report
            report = {
                "runtime": {
                    "start_time": self.metrics["timestamps"]["start"],
                    "current_time": datetime.now().isoformat(),
                    "runtime_seconds": runtime_seconds,
                    "runtime_formatted": str(timedelta(seconds=int(runtime_seconds)))
                },
                "operations": self.get_all_operation_stats(),
                "bandwidth": self.get_bandwidth_stats(),
                "resources": self.get_resource_usage(),
                "errors": self.get_error_stats(),
                "summary": self._generate_summary()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {"error": str(e)}
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        # Calculate total operations
        total_operations = sum(self.operation_counts.values())
        
        # Calculate average operations per minute if runtime > 0
        start_time = datetime.fromisoformat(self.metrics["timestamps"]["start"])
        runtime_minutes = (datetime.now() - start_time).total_seconds() / 60
        
        ops_per_minute = 0
        if runtime_minutes > 0:
            ops_per_minute = total_operations / runtime_minutes
        
        # Calculate overall success rate
        total_errors = sum(self.metrics["errors"].values())
        success_rate = 1.0
        if total_operations > 0:
            success_rate = 1.0 - (total_errors / total_operations)
        
        # Generate summary
        return {
            "total_operations": total_operations,
            "operations_per_minute": ops_per_minute,
            "success_rate": success_rate,
            "total_errors": total_errors,
            "bandwidth_mb": self.metrics["bandwidth"]["total_received"] / (1024 * 1024)
        }
    
    def save_metrics(self, filename: Optional[str] = None) -> str:
        """
        Save current metrics to disk.
        
        Args:
            filename: Optional filename, defaults to timestamp-based name
            
        Returns:
            Path to saved metrics file
        """
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        # Ensure metrics directory exists
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Full path for metrics file
        file_path = os.path.join(self.metrics_dir, filename)
        
        try:
            # Convert defaultdicts to regular dicts for serialization
            serializable_metrics = json.loads(json.dumps(self.metrics, default=str))
            
            # Write metrics to file
            with open(file_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            
            logger.info(f"Metrics saved to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            return ""
    
    def load_metrics(self, file_path: str) -> bool:
        """
        Load metrics from disk.
        
        Args:
            file_path: Path to metrics file
            
        Returns:
            Success status
        """
        try:
            # Read metrics from file
            with open(file_path, 'r') as f:
                loaded_metrics = json.load(f)
            
            # Convert to defaultdicts where needed
            operations = defaultdict(list)
            for key, value in loaded_metrics.get("operations", {}).items():
                operations[key] = value
            
            bandwidth_domains = defaultdict(int)
            for key, value in loaded_metrics.get("bandwidth", {}).get("domains", {}).items():
                bandwidth_domains[key] = value
            
            errors = defaultdict(int)
            for key, value in loaded_metrics.get("errors", {}).items():
                errors[key] = value
            
            execution_times = defaultdict(list)
            for key, value in loaded_metrics.get("execution_times", {}).items():
                execution_times[key] = value
            
            # Update metrics
            self.metrics["operations"] = operations
            self.metrics["resources"] = loaded_metrics.get("resources", self.metrics["resources"])
            
            bandwidth = loaded_metrics.get("bandwidth", {})
            self.metrics["bandwidth"]["domains"] = bandwidth_domains
            self.metrics["bandwidth"]["total_received"] = bandwidth.get("total_received", 0)
            self.metrics["bandwidth"]["total_sent"] = bandwidth.get("total_sent", 0)
            self.metrics["bandwidth"]["time_series"] = bandwidth.get("time_series", [])
            
            self.metrics["errors"] = errors
            self.metrics["execution_times"] = execution_times
            self.metrics["timestamps"] = loaded_metrics.get("timestamps", self.metrics["timestamps"])
            
            # Update operation counts
            for operation, times in execution_times.items():
                self.operation_counts[operation] = len(times)
            
            logger.info(f"Metrics loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading metrics: {str(e)}")
            return False
    
    def reset_metrics(self) -> None:
        """Reset all metrics data."""
        # Save current metrics before reset if keep_history is True
        if self.keep_history:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_metrics(f"metrics_history_{timestamp}.json")
        
        # Reset metrics storage
        self.metrics = {
            "operations": defaultdict(list),
            "resources": {
                "memory": [],
                "cpu": [],
                "disk": [],
                "network": []
            },
            "bandwidth": {
                "domains": defaultdict(int),
                "total_received": 0,
                "total_sent": 0,
                "time_series": []
            },
            "errors": defaultdict(int),
            "execution_times": defaultdict(list),
            "timestamps": {
                "start": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat()
            }
        }
        
        # Reset counters
        self.operation_counts = defaultdict(int)
        self.bandwidth_tracker.reset()
        
        logger.info("Metrics reset")
    
    def plot_memory_usage(self, save_path: Optional[str] = None) -> str:
        """
        Generate a memory usage plot.
        
        Args:
            save_path: Optional path to save the plot, defaults to metrics directory
            
        Returns:
            Path to saved plot
        """
        try:
            # Extract memory data
            memory_data = self.metrics["resources"]["memory"]
            
            if not memory_data:
                logger.warning("No memory data available for plotting")
                return ""
            
            # Convert to pandas DataFrame for easier plotting
            df = pd.DataFrame(memory_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Convert bytes to MB
            for col in ["total", "available", "used"]:
                if col in df.columns:
                    df[f"{col}_mb"] = df[col] / (1024 * 1024)
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            plt.plot(df["timestamp"], df["used_mb"], label="Used Memory (MB)")
            plt.plot(df["timestamp"], df["available_mb"], label="Available Memory (MB)")
            
            # Add percent as secondary axis
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.plot(df["timestamp"], df["percent"], "r--", label="Memory Usage (%)")
            ax2.set_ylabel("Memory Usage (%)")
            ax2.set_ylim(0, 100)
            
            # Format plot
            plt.title("Memory Usage Over Time")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Memory (MB)")
            
            # Format x-axis as time
            date_format = DateFormatter("%H:%M:%S")
            ax1.xaxis.set_major_formatter(date_format)
            plt.xticks(rotation=45)
            
            # Add combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
            
            plt.tight_layout()
            
            # Save plot
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.metrics_dir, f"memory_usage_{timestamp}.png")
            
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"Memory usage plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error generating memory usage plot: {str(e)}")
            return ""
    
    def plot_cpu_usage(self, save_path: Optional[str] = None) -> str:
        """
        Generate a CPU usage plot.
        
        Args:
            save_path: Optional path to save the plot, defaults to metrics directory
            
        Returns:
            Path to saved plot
        """
        try:
            # Extract CPU data
            cpu_data = self.metrics["resources"]["cpu"]
            
            if not cpu_data:
                logger.warning("No CPU data available for plotting")
                return ""
            
            # Convert to pandas DataFrame for easier plotting
            df = pd.DataFrame(cpu_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            plt.plot(df["timestamp"], df["percent"], "g-", label="CPU Usage (%)")
            
            # Format plot
            plt.title("CPU Usage Over Time")
            plt.xlabel("Time")
            plt.ylabel("CPU Usage (%)")
            plt.ylim(0, 100)
            
            # Format x-axis as time
            date_format = DateFormatter("%H:%M:%S")
            plt.gca().xaxis.set_major_formatter(date_format)
            plt.xticks(rotation=45)
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.metrics_dir, f"cpu_usage_{timestamp}.png")
            
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"CPU usage plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error generating CPU usage plot: {str(e)}")
            return ""
    
    def plot_bandwidth_usage(self, save_path: Optional[str] = None) -> str:
        """
        Generate a bandwidth usage plot.
        
        Args:
            save_path: Optional path to save the plot, defaults to metrics directory
            
        Returns:
            Path to saved plot
        """
        try:
            # Extract bandwidth data
            bandwidth_data = self.metrics["bandwidth"]["time_series"]
            
            if not bandwidth_data:
                logger.warning("No bandwidth data available for plotting")
                return ""
            
            # Convert to pandas DataFrame for easier plotting
            df = pd.DataFrame(bandwidth_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Convert bytes to MB
            df["bytes_recv_mb"] = df["bytes_recv"] / (1024 * 1024)
            df["bytes_sent_mb"] = df["bytes_sent"] / (1024 * 1024)
            
            # Calculate rate of change
            df["received_rate"] = df["bytes_recv_mb"].diff() / df["timestamp"].diff().dt.total_seconds()
            df["sent_rate"] = df["bytes_sent_mb"].diff() / df["timestamp"].diff().dt.total_seconds()
            
            # Remove NaN values
            df = df.dropna()
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Cumulative data
            ax1.plot(df["timestamp"], df["bytes_recv_mb"], "b-", label="Received (MB)")
            ax1.plot(df["timestamp"], df["bytes_sent_mb"], "r-", label="Sent (MB)")
            
            ax1.set_title("Cumulative Bandwidth Usage")
            ax1.set_ylabel("Data (MB)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Transfer rate
            ax2.plot(df["timestamp"], df["received_rate"], "b-", label="Download Rate (MB/s)")
            ax2.plot(df["timestamp"], df["sent_rate"], "r-", label="Upload Rate (MB/s)")
            
            ax2.set_title("Bandwidth Transfer Rate")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Rate (MB/s)")
            
            # Format x-axis as time
            date_format = DateFormatter("%H:%M:%S")
            ax2.xaxis.set_major_formatter(date_format)
            plt.xticks(rotation=45)
            
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.metrics_dir, f"bandwidth_usage_{timestamp}.png")
            
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"Bandwidth usage plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error generating bandwidth usage plot: {str(e)}")
            return ""
    
    def plot_operation_times(self, operation_name: Optional[str] = None, save_path: Optional[str] = None) -> str:
        """
        Generate an operation times plot.
        
        Args:
            operation_name: Optional operation name to plot, defaults to plotting all operations
            save_path: Optional path to save the plot, defaults to metrics directory
            
        Returns:
            Path to saved plot
        """
        try:
            # Get operations to plot
            if operation_name:
                operations = {operation_name: self.metrics["execution_times"].get(operation_name, [])}
            else:
                operations = self.metrics["execution_times"]
            
            if not operations or all(len(times) == 0 for times in operations.values()):
                logger.warning("No operation timing data available for plotting")
                return ""
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Plot each operation
            for op_name, times in operations.items():
                if times:
                    plt.boxplot(times, positions=[list(operations.keys()).index(op_name) + 1], 
                               widths=0.6, labels=[op_name])
            
            # Format plot
            plt.title("Operation Execution Times")
            plt.xlabel("Operation")
            plt.ylabel("Time (seconds)")
            plt.grid(True, axis='y', alpha=0.3)
            
            # Rotate x labels if many operations
            if len(operations) > 5:
                plt.xticks(rotation=45, ha="right")
            
            plt.tight_layout()
            
            # Save plot
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                op_suffix = f"_{operation_name}" if operation_name else ""
                save_path = os.path.join(self.metrics_dir, f"operation_times{op_suffix}_{timestamp}.png")
            
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"Operation times plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error generating operation times plot: {str(e)}")
            return ""
    
    def plot_domain_bandwidth(self, limit: int = 10, save_path: Optional[str] = None) -> str:
        """
        Generate a domain bandwidth usage plot.
        
        Args:
            limit: Number of top domains to include
            save_path: Optional path to save the plot, defaults to metrics directory
            
        Returns:
            Path to saved plot
        """
        try:
            # Get domain bandwidth data
            domain_data = self.metrics["bandwidth"]["domains"]
            
            if not domain_data:
                logger.warning("No domain bandwidth data available for plotting")
                return ""
            
            # Sort domains by bandwidth and get top domains
            top_domains = sorted(domain_data.items(), key=lambda x: x[1], reverse=True)[:limit]
            
            # Extract domain names and values
            domains = [d[0] for d in top_domains]
            values = [d[1] / (1024 * 1024) for d in top_domains]  # Convert to MB
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Use horizontal bar chart for better readability
            bars = plt.barh(domains, values, color='skyblue')
            
            # Add value labels to bars
            for i, v in enumerate(values):
                if v >= 1:
                    label = f"{v:.1f} MB"
                else:
                    label = f"{v*1024:.1f} KB"
                plt.text(v + 0.1, i, label, va='center')
            
            # Format plot
            plt.title(f"Top {len(domains)} Domains by Bandwidth Usage")
            plt.xlabel("Bandwidth (MB)")
            plt.ylabel("Domain")
            
            plt.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.metrics_dir, f"domain_bandwidth_{timestamp}.png")
            
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"Domain bandwidth plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error generating domain bandwidth plot: {str(e)}")
            return ""
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate optimization recommendations based on metrics.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        try:
            # Check CPU usage
            cpu_data = self.metrics["resources"]["cpu"]
            if cpu_data:
                recent_cpu = [d["percent"] for d in cpu_data[-10:]]
                avg_cpu = sum(recent_cpu) / len(recent_cpu) if recent_cpu else 0
                
                if avg_cpu > 80:
                    recommendations.append({
                        "type": "cpu_usage",
                        "severity": "high",
                        "message": "CPU usage is very high (>80%). Consider reducing concurrent operations.",
                        "value": avg_cpu
                    })
                elif avg_cpu > 60:
                    recommendations.append({
                        "type": "cpu_usage",
                        "severity": "medium",
                        "message": "CPU usage is elevated (>60%). Monitor system performance.",
                        "value": avg_cpu
                    })
            
            # Check memory usage
            memory_data = self.metrics["resources"]["memory"]
            if memory_data:
                recent_memory = [d["percent"] for d in memory_data[-10:]]
                avg_memory = sum(recent_memory) / len(recent_memory) if recent_memory else 0
                
                if avg_memory > 85:
                    recommendations.append({
                        "type": "memory_usage",
                        "severity": "high",
                        "message": "Memory usage is very high (>85%). Consider limiting data storage or using pagination.",
                        "value": avg_memory
                    })
                elif avg_memory > 70:
                    recommendations.append({
                        "type": "memory_usage",
                        "severity": "medium",
                        "message": "Memory usage is elevated (>70%). Monitor for potential memory issues.",
                        "value": avg_memory
                    })
            
            # Check slow operations
            for op_name, times in self.metrics["execution_times"].items():
                if times and len(times) >= 5:
                    avg_time = sum(times) / len(times)
                    
                    if avg_time > 5.0:
                        recommendations.append({
                            "type": "slow_operation",
                            "severity": "high" if avg_time > 10.0 else "medium",
                            "message": f"Operation '{op_name}' is slow (avg {avg_time:.2f}s). Consider optimization.",
                            "value": avg_time,
                            "operation": op_name
                        })
            
            # Check domains with high bandwidth usage
            domain_data = self.metrics["bandwidth"]["domains"]
            if domain_data:
                for domain, bytes_used in domain_data.items():
                    mb_used = bytes_used / (1024 * 1024)
                    
                    if mb_used > 50:
                        recommendations.append({
                            "type": "high_bandwidth",
                            "severity": "medium",
                            "message": f"Domain '{domain}' has high bandwidth usage ({mb_used:.1f} MB). Consider caching or request limiting.",
                            "value": mb_used,
                            "domain": domain
                        })
            
            # Check error rates
            total_ops = sum(self.operation_counts.values())
            total_errors = sum(self.metrics["errors"].values())
            
            if total_ops > 0 and total_errors > 0:
                error_rate = total_errors / total_ops
                
                if error_rate > 0.1:
                    recommendations.append({
                        "type": "high_error_rate",
                        "severity": "high" if error_rate > 0.2 else "medium",
                        "message": f"High error rate ({error_rate*100:.1f}%). Investigate error handling and retry mechanisms.",
                        "value": error_rate
                    })
            
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {str(e)}")
            
        return recommendations
    
    def cleanup(self) -> None:
        """Clean up resources and ensure metrics are saved."""
        try:
            # Stop tracking thread
            self._tracking_active = False
            
            if self._tracking_thread and self._tracking_thread.is_alive():
                self._tracking_thread.join(timeout=1.0)
            
            # Cancel save timer
            if self._save_timer and self._save_timer.is_alive():
                self._save_timer.cancel()
            
            # Final save of metrics
            self.save_metrics()
            
            # Clean up old metrics files if history limit is set
            if self.keep_history and self.history_limit > 0:
                self._cleanup_old_metrics()
            
            logger.info("Metrics analyzer cleaned up")
            
        except Exception as e:
            logger.error(f"Error during metrics cleanup: {str(e)}")
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics files beyond history limit."""
        try:
            # List all metrics files
            metrics_files = [os.path.join(self.metrics_dir, f) for f in os.listdir(self.metrics_dir) 
                             if f.startswith("metrics_") and f.endswith(".json")]
            
            # Sort by modification time (oldest first)
            metrics_files.sort(key=lambda f: os.path.getmtime(f))
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self.history_limit)
            
            # Remove files older than cutoff date
            for file_path in metrics_files:
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if mod_time < cutoff_date:
                    os.remove(file_path)
                    logger.debug(f"Removed old metrics file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error cleaning up old metrics files: {str(e)}")
    
    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        self.cleanup()

class BandwidthTracker:
    """
    Tracks bandwidth usage by domain and request type.
    
    This class provides detailed tracking of bandwidth usage
    across domains and request types, enabling optimization
    recommendations.
    """
    
    def __init__(self):
        """Initialize the bandwidth tracker."""
        self.domain_bandwidth = defaultdict(int)
        self.request_type_bandwidth = defaultdict(int)
        self.domain_request_counts = defaultdict(int)
        self.request_sizes = []
        self.timestamps = []
    
    def add_request(self, domain: str, size: int, request_type: str = "GET") -> None:
        """
        Add a request to tracking.
        
        Args:
            domain: Domain of the request
            size: Size of the response in bytes
            request_type: Request method (GET, POST, etc.)
        """
        self.domain_bandwidth[domain] += size
        self.request_type_bandwidth[request_type] += size
        self.domain_request_counts[domain] += 1
        
        self.request_sizes.append(size)
        self.timestamps.append(time.time())
    
    def get_domain_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get bandwidth statistics by domain.
        
        Returns:
            Dictionary of domain statistics
        """
        stats = {}
        
        for domain, bytes_used in self.domain_bandwidth.items():
            request_count = self.domain_request_counts[domain]
            
            stats[domain] = {
                "bytes": bytes_used,
                "kb": bytes_used / 1024,
                "mb": bytes_used / (1024 * 1024),
                "request_count": request_count,
                "avg_request_size": bytes_used / request_count if request_count > 0 else 0
            }
        
        return stats
    
    def get_request_type_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get bandwidth statistics by request type.
        
        Returns:
            Dictionary of request type statistics
        """
        stats = {}
        
        for req_type, bytes_used in self.request_type_bandwidth.items():
            stats[req_type] = {
                "bytes": bytes_used,
                "kb": bytes_used / 1024,
                "mb": bytes_used / (1024 * 1024)
            }
        
        return stats
    
    def get_top_domains(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get top domains by bandwidth usage.
        
        Args:
            limit: Number of top domains to return
            
        Returns:
            List of top domains with usage data
        """
        # Sort domains by bandwidth
        domains = sorted(
            self.domain_bandwidth.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top domains
        return [
            {
                "domain": domain,
                "bytes": bytes_used,
                "mb": bytes_used / (1024 * 1024),
                "request_count": self.domain_request_counts[domain]
            }
            for domain, bytes_used in domains[:limit]
        ]
    
    def get_bandwidth_rate(self, window_seconds: float = 60.0) -> float:
        """
        Calculate the recent bandwidth usage rate.
        
        Args:
            window_seconds: Time window in seconds for rate calculation
            
        Returns:
            Bandwidth rate in bytes per second
        """
        if not self.timestamps:
            return 0.0
        
        now = time.time()
        cutoff = now - window_seconds
        
        # Get requests within window
        recent_sizes = []
        for ts, size in zip(self.timestamps, self.request_sizes):
            if ts >= cutoff:
                recent_sizes.append(size)
        
        # Calculate rate
        if recent_sizes:
            return sum(recent_sizes) / window_seconds
        return 0.0
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self.domain_bandwidth = defaultdict(int)
        self.request_type_bandwidth = defaultdict(int)
        self.domain_request_counts = defaultdict(int)
        self.request_sizes = []
        self.timestamps = []

# Create a global instance for convenient access
metrics_analyzer = MetricsAnalyzer()

def track_function(operation_name: Optional[str] = None):
    """
    Decorator to track function execution time using the global metrics analyzer.
    
    Args:
        operation_name: Optional name of the operation, defaults to function name
        
    Example:
        @track_function("extract_data")
        def extract_data_from_page(html):
            # extraction code
            return data
    """
    return metrics_analyzer.track_function(operation_name)

def track_async_function(operation_name: Optional[str] = None):
    """
    Decorator to track async function execution time using the global metrics analyzer.
    
    Args:
        operation_name: Optional name of the operation, defaults to function name
        
    Example:
        @track_async_function("fetch_page")
        async def fetch_page(url):
            # fetch code
            return response
    """
    return metrics_analyzer.track_async_function(operation_name)

@contextmanager
def track_operation(operation_name: str) -> Generator[None, None, None]:
    """
    Context manager to track operation execution time using the global metrics analyzer.
    
    Args:
        operation_name: Name of the operation to track
        
    Example:
        with track_operation("parse_page"):
            # code to parse page
    """
    with metrics_analyzer.track_operation(operation_name):
        yield