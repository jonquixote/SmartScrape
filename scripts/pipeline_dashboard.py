#!/usr/bin/env python3
"""
Pipeline Dashboard Script

This script provides a web-based dashboard for monitoring pipeline execution status,
performance metrics, and resource utilization.
"""

import os
import sys
import argparse
import logging
import json
import asyncio
import time
import signal
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import pipeline monitoring and visualization
from core.pipeline.monitoring import PipelineMonitor
from core.pipeline.visualization import PipelineVisualizer
from core.pipeline.registry import PipelineRegistry

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("Warning: aiohttp not available, using simple HTTP server")
    import http.server
    import socketserver

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pipeline_dashboard')


class PipelineDashboard:
    """Dashboard for pipeline monitoring and control."""
    
    def __init__(self, host: str = 'localhost', port: int = 8080,
                 output_dir: str = 'dashboard', refresh_interval: int = 30,
                 enable_prometheus: bool = False, demo_mode: bool = False):
        """
        Initialize the pipeline dashboard.
        
        Args:
            host: Host to bind the server to
            port: Port to listen on
            output_dir: Directory for dashboard output files
            refresh_interval: Dashboard refresh interval in seconds
            enable_prometheus: Whether to enable Prometheus metrics export
            demo_mode: Run with demo data when no pipelines are available
        """
        self.host = host
        self.port = port
        self.output_dir = os.path.abspath(output_dir)
        self.refresh_interval = refresh_interval
        self.enable_prometheus = enable_prometheus
        self.demo_mode = demo_mode
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup monitor and visualizer
        self.monitor = PipelineMonitor(enable_prometheus=enable_prometheus)
        self.visualizer = PipelineVisualizer(output_dir=self.output_dir)
        
        # Try to get the pipeline registry
        try:
            self.registry = PipelineRegistry()
            logger.info(f"Found pipeline registry with {len(self.registry.get_registered_pipelines())} registered pipelines")
        except Exception as e:
            logger.error(f"Error accessing pipeline registry: {str(e)}")
            self.registry = None
        
        # Last update timestamp
        self.last_update = 0
        
        # Running flag for background tasks
        self.running = False
        
        # Web application (if using aiohttp)
        self.app = None
        self.runner = None
        self.site = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals for graceful shutdown."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
        sys.exit(0)
    
    async def start(self):
        """Start the dashboard server and background tasks."""
        self.running = True
        
        # Start background update task
        asyncio.create_task(self._update_dashboard_periodically())
        
        if AIOHTTP_AVAILABLE:
            await self._start_aiohttp_server()
        else:
            self._start_simple_server()
    
    def stop(self):
        """Stop the dashboard server and background tasks."""
        self.running = False
        
        if AIOHTTP_AVAILABLE and self.site:
            asyncio.create_task(self._stop_aiohttp_server())
    
    async def _update_dashboard_periodically(self):
        """Periodically update the dashboard files."""
        while self.running:
            try:
                await self._update_dashboard()
                await asyncio.sleep(self.refresh_interval)
            except Exception as e:
                logger.error(f"Error updating dashboard: {str(e)}")
                await asyncio.sleep(10)  # Wait a bit longer on error
    
    async def _update_dashboard(self):
        """Update the dashboard files."""
        current_time = time.time()
        
        # Skip if it's been less than refresh_interval/2 seconds since the last update
        if current_time - self.last_update < (self.refresh_interval / 2):
            return
        
        # Get pipeline data
        try:
            pipeline_data = self._get_pipeline_data()
            monitor_data = self._get_monitor_data()
            
            # Get dashboard path
            dashboard_path = self.visualizer.create_dashboard(
                pipeline_data, monitor_data, self.output_dir
            )
            
            if dashboard_path:
                logger.info(f"Updated dashboard at {dashboard_path}")
                self.last_update = current_time
            else:
                logger.warning("Failed to update dashboard")
        except Exception as e:
            logger.error(f"Error updating dashboard: {str(e)}")
    
    def _get_pipeline_data(self) -> List[Dict[str, Any]]:
        """
        Get pipeline data for the dashboard.
        
        Returns:
            List of pipeline data dictionaries
        """
        # Try to get data from registry
        pipeline_data = []
        
        if self.registry:
            try:
                pipeline_names = self.registry.get_registered_pipelines()
                
                for name in pipeline_names:
                    pipeline_info = self.registry.get_pipeline_info(name)
                    if pipeline_info:
                        # Add metrics from monitor if available
                        if name in self.monitor.pipeline_metrics:
                            pipeline_info.update(self.monitor.pipeline_metrics[name])
                        
                        pipeline_data.append(pipeline_info)
            except Exception as e:
                logger.error(f"Error getting pipeline data from registry: {str(e)}")
        
        # Use monitor data if available
        if not pipeline_data and self.monitor.pipeline_metrics:
            for name, metrics in self.monitor.pipeline_metrics.items():
                pipeline_data.append({
                    'name': name,
                    **metrics
                })
        
        # Generate demo data if requested and no real data available
        if not pipeline_data and self.demo_mode:
            logger.info("No pipeline data available, generating demo data")
            pipeline_data = self._generate_demo_data()
        
        return pipeline_data
    
    def _get_monitor_data(self) -> Dict[str, Any]:
        """
        Get monitoring data for the dashboard.
        
        Returns:
            Dictionary of monitoring data
        """
        monitor_data = {}
        
        # Get active pipelines
        monitor_data['active_pipelines'] = [
            {
                'id': pipeline_id,
                'name': data['pipeline_name'],
                'runtime': time.time() - data['start_time'],
                'progress': data['stages_completed'] / max(data['stages_total'], 1) * 100,
                'current_stage': data['current_stage']
            }
            for pipeline_id, data in self.monitor.active_pipelines.items()
        ]
        
        # Get real-time metrics
        monitor_data['resource_utilization'] = self.monitor.resource_utilization(minutes=30)
        
        # Get error analysis
        monitor_data['error_analysis'] = self.monitor.error_analysis()
        
        return monitor_data
    
    def _generate_demo_data(self) -> List[Dict[str, Any]]:
        """
        Generate demo pipeline data for testing.
        
        Returns:
            List of demo pipeline data
        """
        from random import random, randint, choice
        from collections import deque
        
        # Generate some example pipelines
        pipeline_types = [
            'ExtractionPipeline', 
            'ValidationPipeline', 
            'TransformationPipeline',
            'AnalysisPipeline'
        ]
        
        demo_data = []
        
        for i, pipeline_type in enumerate(pipeline_types):
            # Generate execution history
            history = deque(maxlen=20)
            current_time = time.time()
            
            for j in range(10):
                # Random success/failure with 80% success rate
                success = random() < 0.8
                
                # Random duration between 1 and 10 seconds
                duration = 1 + random() * 9
                
                # Random number of errors
                error_count = 0 if success else randint(1, 3)
                
                # Random timestamp within the last 24 hours
                timestamp = current_time - randint(0, 86400)
                
                # Generate random stages
                stages = {}
                stage_names = [f"Stage{k}" for k in range(1, randint(3, 6))]
                
                for stage_name in stage_names:
                    # 90% success rate for stages
                    stage_success = random() < 0.9
                    
                    stages[stage_name] = {
                        'duration': 0.2 + random() * 2,
                        'status': 'success' if stage_success else 'failed'
                    }
                
                # Add execution record
                history.append({
                    'pipeline_id': f"demo-{i}-{j}",
                    'duration': duration,
                    'success': success,
                    'timestamp': timestamp,
                    'stages': stages,
                    'error_count': error_count
                })
            
            # Calculate metrics
            success_count = sum(1 for record in history if record['success'])
            total_duration = sum(record['duration'] for record in history)
            
            # Create pipeline data
            pipeline_data = {
                'name': f"{pipeline_type}-{i+1}",
                'description': f"Demo {pipeline_type}",
                'total_executions': len(history),
                'successful_executions': success_count,
                'failed_executions': len(history) - success_count,
                'total_duration': total_duration,
                'average_duration': total_duration / len(history),
                'execution_history': list(history)
            }
            
            demo_data.append(pipeline_data)
        
        return demo_data
    
    async def _start_aiohttp_server(self):
        """Start the web server using aiohttp."""
        # Create web application
        self.app = web.Application()
        
        # Setup routes
        self.app.router.add_static('/', self.output_dir)
        self.app.router.add_get('/api/metrics', self._handle_metrics)
        self.app.router.add_get('/api/data', self._handle_data)
        
        # Add a redirect from / to /dashboard/dashboard.html
        self.app.router.add_get('/', self._handle_root)
        
        # Start server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        dashboard_path = os.path.join(self.output_dir, 'dashboard', 'dashboard.html')
        
        # Make sure we have an initial dashboard before starting server
        if not os.path.exists(dashboard_path):
            await self._update_dashboard()
        
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        
        logger.info(f"Dashboard server started at http://{self.host}:{self.port}/dashboard/dashboard.html")
    
    async def _stop_aiohttp_server(self):
        """Stop the aiohttp server."""
        if self.site:
            await self.site.stop()
        
        if self.runner:
            await self.runner.cleanup()
    
    def _start_simple_server(self):
        """Start a simple HTTP server (fallback when aiohttp is not available)."""
        # Make sure we have an initial dashboard
        asyncio.run(self._update_dashboard())
        
        # Change to output directory
        os.chdir(self.output_dir)
        
        # Create a simple HTTP server
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer((self.host, self.port), handler) as httpd:
            logger.info(f"Dashboard server started at http://{self.host}:{self.port}/dashboard/dashboard.html")
            httpd.serve_forever()
    
    async def _handle_root(self, request):
        """Handle requests to the root path."""
        raise web.HTTPFound('/dashboard/dashboard.html')
    
    async def _handle_metrics(self, request):
        """Handle Prometheus metrics endpoint."""
        if self.enable_prometheus:
            metrics = self.monitor.export_metrics('prometheus')
            return web.Response(text=metrics, content_type='text/plain')
        else:
            return web.Response(text="Prometheus metrics not enabled", status=404)
    
    async def _handle_data(self, request):
        """Handle API data endpoint."""
        data = {
            'pipelines': self._get_pipeline_data(),
            'monitor': self._get_monitor_data(),
            'timestamp': time.time()
        }
        
        return web.json_response(data)


async def main():
    """Main entry point for the dashboard script."""
    parser = argparse.ArgumentParser(description='Pipeline monitoring dashboard')
    
    parser.add_argument('--host', default='localhost', 
                      help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8080, 
                      help='Port to listen on')
    parser.add_argument('--output-dir', default='dashboard', 
                      help='Directory for dashboard output files')
    parser.add_argument('--refresh', type=int, default=30, 
                      help='Dashboard refresh interval in seconds')
    parser.add_argument('--prometheus', action='store_true', 
                      help='Enable Prometheus metrics export')
    parser.add_argument('--demo', action='store_true', 
                      help='Run with demo data when no pipelines are available')
    
    args = parser.parse_args()
    
    # Create dashboard
    dashboard = PipelineDashboard(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
        refresh_interval=args.refresh,
        enable_prometheus=args.prometheus,
        demo_mode=args.demo
    )
    
    # Start dashboard
    await dashboard.start()


if __name__ == '__main__':
    if AIOHTTP_AVAILABLE:
        asyncio.run(main())
    else:
        # Simple server doesn't use asyncio
        dashboard = PipelineDashboard(
            host='localhost',
            port=8080,
            output_dir='dashboard',
            refresh_interval=30,
            demo_mode=True
        )
        dashboard._start_simple_server()