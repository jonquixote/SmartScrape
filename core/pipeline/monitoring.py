import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union, Deque

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from core.pipeline.context import PipelineContext


class PipelineMonitor:
    """
    Monitor for pipeline execution metrics and performance.
    
    Tracks real-time and historical data about pipeline executions, including
    performance metrics, resource utilization, and error rates.
    
    Attributes:
        retention_period (int): Number of days to retain historical data
        enable_prometheus (bool): Whether to enable Prometheus metrics export
        pipeline_metrics (Dict): Metrics for each pipeline
        stage_metrics (Dict): Metrics for each stage
        historical_data (Dict): Historical execution data
        active_pipelines (Dict): Currently running pipelines
        error_counts (Dict): Error counts by type and source
    """
    
    def __init__(self, 
                 retention_period: int = 7, 
                 enable_prometheus: bool = False,
                 metrics_storage_path: Optional[str] = None):
        """
        Initialize the monitor.
        
        Args:
            retention_period (int): Days to retain historical data
            enable_prometheus (bool): Whether to enable Prometheus metrics
            metrics_storage_path (Optional[str]): Path to store metrics data
        """
        self.logger = logging.getLogger("pipeline.monitor")
        self.retention_period = retention_period
        self.enable_prometheus = enable_prometheus
        self.metrics_storage_path = metrics_storage_path
        
        # Initialize metrics storage
        self.pipeline_metrics = defaultdict(lambda: {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_duration": 0.0,
            "average_duration": 0.0,
            "execution_history": []
        })
        
        self.stage_metrics = defaultdict(lambda: {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_duration": 0.0,
            "average_duration": 0.0,
            "success_rate": 0.0
        })
        
        self.historical_data = defaultdict(list)
        self.active_pipelines = {}
        self.error_counts = defaultdict(int)
        
        # Resource monitoring
        self.resource_history = {
            "cpu": deque(maxlen=1000),
            "memory": deque(maxlen=1000),
            "disk": deque(maxlen=100),
            "network": deque(maxlen=100)
        }
        
        # Start background tasks
        self.background_tasks = set()
        self._tracking_active = True
        
        if PSUTIL_AVAILABLE:
            self._start_background_task(self._resource_tracking_loop())
        
        # Anomaly detection settings
        self.anomaly_threshold = 2.0  # Standard deviations
        self.anomalies = []
        
        self.logger.info("Pipeline monitor initialized")
    
    def _start_background_task(self, coroutine):
        """Start a background task and track it."""
        task = asyncio.create_task(coroutine)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
    
    async def _resource_tracking_loop(self):
        """Background task to track system resource usage."""
        while self._tracking_active:
            try:
                # Get current resource usage
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                timestamp = time.time()
                
                # Store resource metrics
                self.resource_history["cpu"].append({
                    "timestamp": timestamp,
                    "value": cpu_percent
                })
                
                self.resource_history["memory"].append({
                    "timestamp": timestamp,
                    "value": memory.percent,
                    "total": memory.total,
                    "available": memory.available
                })
                
                self.resource_history["disk"].append({
                    "timestamp": timestamp,
                    "value": disk.percent,
                    "total": disk.total,
                    "free": disk.free
                })
                
                # Network stats if available
                try:
                    network = psutil.net_io_counters()
                    self.resource_history["network"].append({
                        "timestamp": timestamp,
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv
                    })
                except Exception:
                    pass
                    
            except Exception as e:
                self.logger.warning(f"Error in resource tracking: {str(e)}")
                
            await asyncio.sleep(5)  # Sample every 5 seconds
    
    def register_pipeline_start(self, pipeline_id: str, context: PipelineContext) -> None:
        """
        Register the start of a pipeline execution.
        
        Args:
            pipeline_id (str): Unique ID for this execution
            context (PipelineContext): Pipeline context
        """
        pipeline_name = context.metadata.get("pipeline_name", "unknown")
        start_time = time.time()
        
        # Record active pipeline
        self.active_pipelines[pipeline_id] = {
            "pipeline_name": pipeline_name,
            "start_time": start_time,
            "stages_completed": 0,
            "stages_total": len(context.metadata.get("stage_transitions", [])),
            "current_stage": context.metadata.get("current_stage", "initializing"),
            "context": context
        }
        
        self.logger.debug(f"Registered start of pipeline {pipeline_name} with ID {pipeline_id}")
    
    def register_pipeline_end(self, pipeline_id: str, context: PipelineContext) -> None:
        """
        Register the end of a pipeline execution.
        
        Args:
            pipeline_id (str): Unique ID for this execution
            context (PipelineContext): Pipeline context
        """
        if pipeline_id not in self.active_pipelines:
            self.logger.warning(f"Attempted to end unknown pipeline: {pipeline_id}")
            return
        
        # Get pipeline data
        pipeline_data = self.active_pipelines[pipeline_id]
        pipeline_name = pipeline_data["pipeline_name"]
        start_time = pipeline_data["start_time"]
        end_time = time.time()
        duration = end_time - start_time
        
        # Get metrics from context
        metrics = context.get_metrics()
        success = metrics.get("successful_stages", 0) == metrics.get("total_stages", 0)
        
        # Update pipeline metrics
        pipeline_metrics = self.pipeline_metrics[pipeline_name]
        pipeline_metrics["total_executions"] += 1
        
        if success:
            pipeline_metrics["successful_executions"] += 1
        else:
            pipeline_metrics["failed_executions"] += 1
            
            # Record error counts
            errors = context.get_errors()
            for source, error_list in errors.items():
                for error in error_list:
                    error_type = error.split(":", 1)[0] if ":" in error else "unknown"
                    self.error_counts[f"{pipeline_name}:{error_type}"] += 1
        
        pipeline_metrics["total_duration"] += duration
        pipeline_metrics["average_duration"] = (
            pipeline_metrics["total_duration"] / pipeline_metrics["total_executions"]
        )
        
        # Create execution record
        execution_record = {
            "pipeline_id": pipeline_id,
            "duration": duration,
            "success": success,
            "timestamp": start_time,
            "stages": metrics.get("stages", {}),
            "error_count": len(context.get_errors())
        }
        
        # Add to execution history
        pipeline_metrics["execution_history"].append(execution_record)
        
        # Limit history length
        if len(pipeline_metrics["execution_history"]) > 100:
            pipeline_metrics["execution_history"] = pipeline_metrics["execution_history"][-100:]
        
        # Add to historical data
        self.historical_data[pipeline_name].append(execution_record)
        
        # Update stage metrics
        for stage_name, stage_data in metrics.get("stages", {}).items():
            stage_key = f"{pipeline_name}:{stage_name}"
            stage_metrics = self.stage_metrics[stage_key]
            
            stage_metrics["total_executions"] += 1
            stage_success = stage_data.get("status", "") == "success"
            
            if stage_success:
                stage_metrics["successful_executions"] += 1
            else:
                stage_metrics["failed_executions"] += 1
            
            stage_duration = stage_data.get("execution_time", 0)
            stage_metrics["total_duration"] += stage_duration
            
            if stage_metrics["total_executions"] > 0:
                stage_metrics["average_duration"] = (
                    stage_metrics["total_duration"] / stage_metrics["total_executions"]
                )
                stage_metrics["success_rate"] = (
                    stage_metrics["successful_executions"] / stage_metrics["total_executions"]
                )
        
        # Check for anomalies
        self._check_for_anomalies(pipeline_name, execution_record)
        
        # Remove from active pipelines
        del self.active_pipelines[pipeline_id]
        
        self.logger.debug(
            f"Registered end of pipeline {pipeline_name} with ID {pipeline_id}: "
            f"duration={duration:.2f}s, success={success}"
        )
        
        # Persist metrics if storage path is configured
        if self.metrics_storage_path:
            self._persist_metrics()
    
    def register_stage_transition(self, 
                                 pipeline_id: str, 
                                 stage_name: str, 
                                 context: PipelineContext) -> None:
        """
        Register a stage transition in a pipeline execution.
        
        Args:
            pipeline_id (str): Unique ID for this execution
            stage_name (str): Name of the stage being transitioned to
            context (PipelineContext): Pipeline context
        """
        if pipeline_id not in self.active_pipelines:
            return
        
        pipeline_data = self.active_pipelines[pipeline_id]
        pipeline_data["current_stage"] = stage_name
        pipeline_data["stages_completed"] += 1
        
        self.logger.debug(
            f"Pipeline {pipeline_id} transitioned to stage {stage_name} "
            f"({pipeline_data['stages_completed']}/{pipeline_data['stages_total']})"
        )
    
    def real_time_metrics(self) -> Dict[str, Any]:
        """
        Get real-time metrics for all active pipelines.
        
        Returns:
            Dict[str, Any]: Dictionary of real-time metrics
        """
        active_count = len(self.active_pipelines)
        
        # Prepare active pipeline data
        active_pipelines = []
        for pipeline_id, data in self.active_pipelines.items():
            runtime = time.time() - data["start_time"]
            progress = 0
            if data["stages_total"] > 0:
                progress = data["stages_completed"] / data["stages_total"] * 100
                
            active_pipelines.append({
                "id": pipeline_id,
                "name": data["pipeline_name"],
                "runtime": runtime,
                "progress": progress,
                "current_stage": data["current_stage"]
            })
            
        # Get current resource utilization
        current_resources = {}
        if PSUTIL_AVAILABLE:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                current_resources = {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used": memory.used,
                    "memory_total": memory.total,
                    "disk_percent": disk.percent,
                    "disk_free": disk.free,
                    "disk_total": disk.total
                }
            except Exception as e:
                self.logger.warning(f"Error getting resource metrics: {str(e)}")
        
        return {
            "timestamp": time.time(),
            "active_pipelines": active_count,
            "pipelines": active_pipelines,
            "resource_utilization": {
                "current": current_resources
            }
        }
    
    def performance_history(self, pipeline_name: str) -> Dict[str, Any]:
        """
        Get performance history for a specific pipeline.
        
        Args:
            pipeline_name (str): Name of the pipeline
            
        Returns:
            Dict[str, Any]: Dictionary of performance metrics
        """
        if pipeline_name not in self.pipeline_metrics:
            return {
                "pipeline_id": pipeline_name,
                "total_executions": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
                "history": []
            }
        
        metrics = self.pipeline_metrics[pipeline_name]
        success_rate = 0.0
        if metrics["total_executions"] > 0:
            success_rate = metrics["successful_executions"] / metrics["total_executions"]
            
        return {
            "pipeline_id": pipeline_name,
            "total_executions": metrics["total_executions"],
            "success_rate": success_rate,
            "average_duration": metrics["average_duration"],
            "history": metrics["execution_history"]
        }
    
    def stage_statistics(self, stage_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific stage.
        
        Args:
            stage_id (str): ID of the stage in format "pipeline_name:stage_name"
            
        Returns:
            Dict[str, Any]: Dictionary of stage metrics
        """
        if stage_id not in self.stage_metrics:
            pipeline_name, stage_name = stage_id.split(":", 1) if ":" in stage_id else (stage_id, "")
            return {
                "stage_id": stage_id,
                "pipeline_name": pipeline_name,
                "stage_name": stage_name,
                "total_executions": 0,
                "success_rate": 0.0,
                "average_duration": 0.0
            }
        
        metrics = self.stage_metrics[stage_id]
        pipeline_name, stage_name = stage_id.split(":", 1) if ":" in stage_id else (stage_id, "")
        
        return {
            "stage_id": stage_id,
            "pipeline_name": pipeline_name,
            "stage_name": stage_name,
            "total_executions": metrics["total_executions"],
            "successful_executions": metrics["successful_executions"],
            "failed_executions": metrics["failed_executions"],
            "success_rate": metrics["success_rate"],
            "average_duration": metrics["average_duration"]
        }
    
    def error_analysis(self) -> Dict[str, Any]:
        """
        Get error analysis for all monitored pipelines.
        
        Returns:
            Dict[str, Any]: Dictionary of error analysis data
        """
        # Global error counts
        total_errors = sum(self.error_counts.values())
        error_types = defaultdict(int)
        pipeline_errors = defaultdict(int)
        
        for error_key, count in self.error_counts.items():
            if ":" in error_key:
                pipeline, error_type = error_key.split(":", 1)
                error_types[error_type] += count
                pipeline_errors[pipeline] += count
        
        # Per-pipeline error analysis
        pipeline_analysis = {}
        for pipeline_name in self.pipeline_metrics:
            pipeline_exec_count = self.pipeline_metrics[pipeline_name]["total_executions"]
            pipeline_error_count = pipeline_errors.get(pipeline_name, 0)
            
            if pipeline_exec_count > 0:
                error_rate = pipeline_error_count / pipeline_exec_count
            else:
                error_rate = 0.0
                
            pipeline_analysis[pipeline_name] = {
                "error_count": pipeline_error_count,
                "error_rate": error_rate,
                "execution_count": pipeline_exec_count
            }
            
            # Add stage error analysis
            stage_errors = {}
            for stage_key in self.stage_metrics:
                if stage_key.startswith(f"{pipeline_name}:"):
                    _, stage_name = stage_key.split(":", 1)
                    stage_metrics = self.stage_metrics[stage_key]
                    
                    stage_errors[stage_name] = {
                        "error_count": stage_metrics["failed_executions"],
                        "error_rate": 1.0 - stage_metrics["success_rate"],
                        "execution_count": stage_metrics["total_executions"]
                    }
                    
            pipeline_analysis[pipeline_name]["stages"] = stage_errors
        
        return {
            "global_stats": {
                "total_errors": total_errors,
                "error_types": dict(error_types),
                "pipeline_errors": dict(pipeline_errors)
            },
            "pipeline_analysis": pipeline_analysis,
            "anomalies": self.anomalies[-20:] if self.anomalies else []
        }
    
    def resource_utilization(self, minutes: int = 15) -> Dict[str, Any]:
        """
        Get resource utilization metrics.
        
        Args:
            minutes (int): Timespan in minutes to analyze
            
        Returns:
            Dict[str, Any]: Dictionary of resource utilization data
        """
        now = time.time()
        timespan = minutes * 60  # Convert to seconds
        min_timestamp = now - timespan
        
        # Filter records within timespan
        cpu_records = [r for r in self.resource_history["cpu"] if r["timestamp"] >= min_timestamp]
        memory_records = [r for r in self.resource_history["memory"] if r["timestamp"] >= min_timestamp]
        
        # Calculate statistics if records exist
        cpu_stats = {}
        if cpu_records:
            cpu_values = [r["value"] for r in cpu_records]
            cpu_stats = {
                "current": cpu_values[-1] if cpu_values else 0,
                "min": min(cpu_values),
                "max": max(cpu_values),
                "avg": sum(cpu_values) / len(cpu_values)
            }
            
        memory_stats = {}
        if memory_records:
            memory_values = [r["value"] for r in memory_records]
            memory_stats = {
                "current": memory_values[-1] if memory_values else 0,
                "min": min(memory_values),
                "max": max(memory_values),
                "avg": sum(memory_values) / len(memory_values),
                "total": memory_records[-1]["total"] if memory_records else 0
            }
        
        return {
            "timespan_minutes": minutes,
            "sample_count": len(cpu_records),
            "statistics": {
                "cpu": cpu_stats,
                "memory": memory_stats
            },
            "history": {
                "cpu": cpu_records,
                "memory": memory_records
            }
        }
    
    def export_metrics(self, format_type: str = "json") -> str:
        """
        Export metrics in various formats.
        
        Args:
            format_type (str): Format type (json, prometheus, csv)
            
        Returns:
            str: Metrics in the requested format
        """
        if format_type.lower() == "json":
            return self._export_json()
        elif format_type.lower() == "prometheus" and self.enable_prometheus:
            return self._export_prometheus()
        elif format_type.lower() == "csv":
            return self._export_csv()
        else:
            return self._export_json()
    
    def _export_json(self) -> str:
        """Export metrics as JSON."""
        metrics = {
            "pipelines": dict(self.pipeline_metrics),
            "stages": dict(self.stage_metrics),
            "errors": dict(self.error_counts),
            "active_pipelines": self.active_pipelines,
            "resource_utilization": self.resource_utilization(),
            "timestamp": time.time()
        }
        
        # Clean up non-serializable data
        for pipeline_id in metrics["active_pipelines"]:
            if "context" in metrics["active_pipelines"][pipeline_id]:
                metrics["active_pipelines"][pipeline_id]["context"] = str(
                    metrics["active_pipelines"][pipeline_id]["context"]
                )
        
        return json.dumps(metrics, indent=2)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Pipeline metrics
        for pipeline_name, metrics in self.pipeline_metrics.items():
            lines.append(f'# HELP pipeline_executions_total Total number of pipeline executions')
            lines.append(f'# TYPE pipeline_executions_total counter')
            lines.append(f'pipeline_executions_total{{pipeline="{pipeline_name}"}} {metrics["total_executions"]}')
            
            lines.append(f'# HELP pipeline_executions_success Successful pipeline executions')
            lines.append(f'# TYPE pipeline_executions_success counter')
            lines.append(f'pipeline_executions_success{{pipeline="{pipeline_name}"}} {metrics["successful_executions"]}')
            
            lines.append(f'# HELP pipeline_executions_failed Failed pipeline executions')
            lines.append(f'# TYPE pipeline_executions_failed counter')
            lines.append(f'pipeline_executions_failed{{pipeline="{pipeline_name}"}} {metrics["failed_executions"]}')
            
            lines.append(f'# HELP pipeline_execution_duration_seconds_total Total execution time')
            lines.append(f'# TYPE pipeline_execution_duration_seconds_total counter')
            lines.append(f'pipeline_execution_duration_seconds_total{{pipeline="{pipeline_name}"}} {metrics["total_duration"]}')
            
            lines.append(f'# HELP pipeline_execution_duration_seconds_avg Average execution time')
            lines.append(f'# TYPE pipeline_execution_duration_seconds_avg gauge')
            lines.append(f'pipeline_execution_duration_seconds_avg{{pipeline="{pipeline_name}"}} {metrics["average_duration"]}')
        
        # Stage metrics
        for stage_id, metrics in self.stage_metrics.items():
            pipeline_name, stage_name = stage_id.split(":", 1) if ":" in stage_id else (stage_id, "unknown")
            
            lines.append(f'# HELP stage_executions_total Total stage executions')
            lines.append(f'# TYPE stage_executions_total counter')
            lines.append(f'stage_executions_total{{pipeline="{pipeline_name}",stage="{stage_name}"}} {metrics["total_executions"]}')
            
            lines.append(f'# HELP stage_success_rate Stage success rate')
            lines.append(f'# TYPE stage_success_rate gauge')
            lines.append(f'stage_success_rate{{pipeline="{pipeline_name}",stage="{stage_name}"}} {metrics["success_rate"]}')
            
            lines.append(f'# HELP stage_execution_duration_seconds_avg Average stage execution time')
            lines.append(f'# TYPE stage_execution_duration_seconds_avg gauge')
            lines.append(f'stage_execution_duration_seconds_avg{{pipeline="{pipeline_name}",stage="{stage_name}"}} {metrics["average_duration"]}')
        
        # Active pipelines
        lines.append(f'# HELP active_pipelines_total Current active pipelines')
        lines.append(f'# TYPE active_pipelines_total gauge')
        lines.append(f'active_pipelines_total {len(self.active_pipelines)}')
        
        # Resource utilization if available
        if PSUTIL_AVAILABLE:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                lines.append(f'# HELP system_cpu_percent CPU utilization percentage')
                lines.append(f'# TYPE system_cpu_percent gauge')
                lines.append(f'system_cpu_percent {cpu_percent}')
                
                lines.append(f'# HELP system_memory_percent Memory utilization percentage')
                lines.append(f'# TYPE system_memory_percent gauge')
                lines.append(f'system_memory_percent {memory.percent}')
            except Exception:
                pass
        
        return '\n'.join(lines)
    
    def _export_csv(self) -> str:
        """Export metrics as CSV."""
        lines = ["pipeline,total_executions,successful_executions,failed_executions,average_duration,success_rate"]
        
        for pipeline_name, metrics in self.pipeline_metrics.items():
            success_rate = 0
            if metrics["total_executions"] > 0:
                success_rate = metrics["successful_executions"] / metrics["total_executions"]
                
            lines.append(
                f"{pipeline_name},"
                f"{metrics['total_executions']},"
                f"{metrics['successful_executions']},"
                f"{metrics['failed_executions']},"
                f"{metrics['average_duration']},"
                f"{success_rate}"
            )
            
        return '\n'.join(lines)
    
    def _check_for_anomalies(self, pipeline_name: str, execution_record: Dict[str, Any]) -> None:
        """
        Check for performance anomalies.
        
        Args:
            pipeline_name (str): Name of the pipeline
            execution_record (Dict[str, Any]): Execution record to check
        """
        history = self.historical_data.get(pipeline_name, [])
        if len(history) < 5:  # Need at least 5 data points for meaningful analysis
            return
            
        # Calculate mean and standard deviation of durations
        durations = [record["duration"] for record in history[:-1]]  # Exclude current record
        if not durations:
            return
            
        mean_duration = sum(durations) / len(durations)
        std_dev = (sum((d - mean_duration) ** 2 for d in durations) / len(durations)) ** 0.5
        
        if std_dev == 0:  # Avoid division by zero
            return
            
        # Calculate z-score for current execution
        current_duration = execution_record["duration"]
        z_score = (current_duration - mean_duration) / std_dev
        
        # Check if it's an anomaly
        if abs(z_score) > self.anomaly_threshold:
            anomaly_type = "slow" if z_score > 0 else "fast"
            deviation_percent = abs(current_duration - mean_duration) / mean_duration * 100
            
            anomaly = {
                "pipeline_name": pipeline_name,
                "timestamp": time.time(),
                "execution_id": execution_record["pipeline_id"],
                "type": anomaly_type,
                "z_score": z_score,
                "current_duration": current_duration,
                "mean_duration": mean_duration,
                "deviation_percent": deviation_percent,
                "description": (
                    f"Pipeline executed {anomaly_type}er than usual "
                    f"({deviation_percent:.1f}% {'above' if z_score > 0 else 'below'} average)"
                )
            }
            
            self.anomalies.append(anomaly)
            
            # Limit anomalies list size
            if len(self.anomalies) > 100:
                self.anomalies = self.anomalies[-100:]
                
            self.logger.info(
                f"Performance anomaly detected in pipeline '{pipeline_name}': "
                f"{anomaly['description']}"
            )
    
    def _persist_metrics(self) -> None:
        """Persist metrics to storage path."""
        if not self.metrics_storage_path:
            return
            
        try:
            os.makedirs(self.metrics_storage_path, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.metrics_storage_path, f"metrics_{timestamp}.json")
            
            with open(filename, "w") as f:
                f.write(self._export_json())
                
            self.logger.debug(f"Metrics persisted to {filename}")
        except Exception as e:
            self.logger.error(f"Error persisting metrics: {str(e)}")
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.pipeline_metrics = defaultdict(lambda: {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_duration": 0.0,
            "average_duration": 0.0,
            "execution_history": []
        })
        
        self.stage_metrics = defaultdict(lambda: {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_duration": 0.0,
            "average_duration": 0.0,
            "success_rate": 0.0
        })
        
        self.historical_data = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.anomalies = []
        
        self.logger.info("Metrics have been reset")
    
    async def stop(self):
        """Stop all background monitoring tasks."""
        self._tracking_active = False
        
        # Wait for all tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
        self.logger.info("Pipeline monitor stopped")