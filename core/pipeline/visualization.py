import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from jinja2 import Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False


class PipelineVisualizer:
    """
    Visualizer for pipeline structure and performance.
    
    This class provides visualization capabilities for pipeline structure,
    execution flow, and performance metrics.
    """
    
    def __init__(self, output_dir: str = "pipeline_visualizations", 
                 enable_interactive: bool = False) -> None:
        """
        Initialize the pipeline visualizer.
        
        Args:
            output_dir: Directory to save visualization files
            enable_interactive: Whether to enable interactive visualizations
        """
        self.logger = logging.getLogger("pipeline.visualizer")
        self.output_dir = output_dir
        self.enable_interactive = enable_interactive
        
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for required libraries
        self.has_matplotlib = MATPLOTLIB_AVAILABLE
        self.has_networkx = NETWORKX_AVAILABLE
        self.has_jinja2 = JINJA2_AVAILABLE
        
        if not self.has_matplotlib:
            self.logger.warning("Matplotlib not available, visualizations will be limited")
        
        if not self.has_networkx:
            self.logger.warning("NetworkX not available, flow diagrams will be limited")
            
        if not self.has_jinja2:
            self.logger.warning("Jinja2 not available, HTML reports will be limited")
            
        # Configure template environment if Jinja2 is available
        if self.has_jinja2:
            template_dir = os.path.join(os.path.dirname(__file__), 'templates')
            # Create template directory if it doesn't exist
            os.makedirs(template_dir, exist_ok=True)
            
            self.template_env = Environment(loader=FileSystemLoader(template_dir))
            
            # Create default templates if they don't exist
            self._create_default_templates(template_dir)
    
    def _create_default_templates(self, template_dir: str) -> None:
        """
        Create default templates for HTML reports.
        
        Args:
            template_dir: Directory to save templates
        """
        # Dashboard template
        dashboard_template = os.path.join(template_dir, 'dashboard.html')
        if not os.path.exists(dashboard_template):
            with open(dashboard_template, 'w') as f:
                f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background-color: white; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        .metric-card { background-color: #f9f9f9; border-radius: 5px; padding: 15px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .metric-label { font-size: 14px; color: #666; }
        .chart-container { height: 300px; margin-top: 20px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        .status-success { color: #28a745; }
        .status-failed { color: #dc3545; }
        .refresh-time { color: #666; font-size: 12px; }
        @media (max-width: 768px) { .metric-grid { grid-template-columns: 1fr; } }
    </style>
    <script>
        function refreshDashboard() {
            location.reload();
        }
        
        // Auto-refresh every 30 seconds
        setInterval(refreshDashboard, 30000);
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Pipeline Dashboard</h1>
            <div>
                <button onclick="refreshDashboard()">Refresh</button>
                <span class="refresh-time">Last updated: {{timestamp}}</span>
            </div>
        </div>
        
        <div class="card">
            <h2>Active Pipelines</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Active Pipelines</div>
                    <div class="metric-value">{{active_pipelines_count}}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Pipelines</div>
                    <div class="metric-value">{{total_pipelines_count}}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Success Rate (24h)</div>
                    <div class="metric-value">{{success_rate}}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Duration (24h)</div>
                    <div class="metric-value">{{avg_duration}}s</div>
                </div>
            </div>
            
            {% if active_pipelines %}
            <h3>Currently Running</h3>
            <table>
                <thead>
                    <tr>
                        <th>Pipeline ID</th>
                        <th>Name</th>
                        <th>Runtime</th>
                        <th>Progress</th>
                        <th>Current Stage</th>
                    </tr>
                </thead>
                <tbody>
                    {% for pipeline in active_pipelines %}
                    <tr>
                        <td>{{pipeline.id}}</td>
                        <td>{{pipeline.name}}</td>
                        <td>{{pipeline.runtime}}s</td>
                        <td>{{pipeline.progress}}%</td>
                        <td>{{pipeline.current_stage or 'N/A'}}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <p>No active pipelines</p>
            {% endif %}
        </div>
        
        <div class="card">
            <h2>Recent Executions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Pipeline</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Timestamp</th>
                        <th>Stages</th>
                    </tr>
                </thead>
                <tbody>
                    {% for execution in recent_executions %}
                    <tr>
                        <td>{{execution.pipeline_name}}</td>
                        <td class="{% if execution.success %}status-success{% else %}status-failed{% endif %}">
                            {{execution.success | string | capitalize}}
                        </td>
                        <td>{{execution.duration}}s</td>
                        <td>{{execution.timestamp}}</td>
                        <td>{{execution.stages_completed}}/{{execution.stages_total}}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h2>Resource Utilization</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Memory Usage</div>
                    <div class="metric-value">{{memory_usage}}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">CPU Usage</div>
                    <div class="metric-value">{{cpu_usage}}%</div>
                </div>
            </div>
            <div class="chart-container">
                <img src="{{resource_chart_path}}" alt="Resource utilization chart" style="max-width: 100%; height: auto;">
            </div>
        </div>
        
        <div class="card">
            <h2>Error Analysis</h2>
            <div class="chart-container">
                <img src="{{error_chart_path}}" alt="Error distribution chart" style="max-width: 100%; height: auto;">
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Error Type</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
                    {% for error_type, stats in error_stats.items() %}
                    <tr>
                        <td>{{error_type}}</td>
                        <td>{{stats.count}}</td>
                        <td>{{stats.percentage}}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>''')
        
        # Pipeline detail template
        pipeline_template = os.path.join(template_dir, 'pipeline_detail.html')
        if not os.path.exists(pipeline_template):
            with open(pipeline_template, 'w') as f:
                f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Detail: {{pipeline_name}}</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background-color: white; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        .metric-card { background-color: #f9f9f9; border-radius: 5px; padding: 15px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .metric-label { font-size: 14px; color: #666; }
        .chart-container { height: 300px; margin-top: 20px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        .status-success { color: #28a745; }
        .status-failed { color: #dc3545; }
        .stage-table { margin-top: 20px; }
        .back-link { margin-bottom: 20px; display: block; }
        @media (max-width: 768px) { .metric-grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <a href="dashboard.html" class="back-link">← Back to Dashboard</a>
        
        <div class="header">
            <h1>Pipeline: {{pipeline_name}}</h1>
        </div>
        
        <div class="card">
            <h2>Overview</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Executions</div>
                    <div class="metric-value">{{total_executions}}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Success Rate</div>
                    <div class="metric-value">{{success_rate}}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Duration</div>
                    <div class="metric-value">{{avg_duration}}s</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Latest Status</div>
                    <div class="metric-value {% if latest_success %}status-success{% else %}status-failed{% endif %}">
                        {{latest_success | string | capitalize}}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Pipeline Structure</h2>
            <div class="chart-container">
                <img src="{{flow_diagram_path}}" alt="Pipeline flow diagram" style="max-width: 100%; height: auto;">
            </div>
        </div>
        
        <div class="card">
            <h2>Performance History</h2>
            <div class="chart-container">
                <img src="{{performance_chart_path}}" alt="Performance history chart" style="max-width: 100%; height: auto;">
            </div>
        </div>
        
        <div class="card">
            <h2>Stage Performance</h2>
            <div class="chart-container">
                <img src="{{stage_chart_path}}" alt="Stage performance chart" style="max-width: 100%; height: auto;">
            </div>
            
            <h3>Stage Details</h3>
            <table class="stage-table">
                <thead>
                    <tr>
                        <th>Stage</th>
                        <th>Avg Duration</th>
                        <th>Success Rate</th>
                        <th>Total Executions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for stage in stages %}
                    <tr>
                        <td>{{stage.name}}</td>
                        <td>{{stage.avg_duration}}s</td>
                        <td>{{stage.success_rate}}%</td>
                        <td>{{stage.total_executions}}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h2>Recent Executions</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Timestamp</th>
                        <th>Errors</th>
                    </tr>
                </thead>
                <tbody>
                    {% for execution in recent_executions %}
                    <tr>
                        <td>{{execution.id}}</td>
                        <td class="{% if execution.success %}status-success{% else %}status-failed{% endif %}">
                            {{execution.success | string | capitalize}}
                        </td>
                        <td>{{execution.duration}}s</td>
                        <td>{{execution.timestamp}}</td>
                        <td>{{execution.error_count}}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        {% if has_anomalies %}
        <div class="card">
            <h2>Performance Anomalies</h2>
            <table>
                <thead>
                    <tr>
                        <th>Type</th>
                        <th>Value</th>
                        <th>Expected</th>
                        <th>Deviation</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody>
                    {% for anomaly in anomalies %}
                    <tr>
                        <td>{{anomaly.type}}</td>
                        <td>{{anomaly.value}}s</td>
                        <td>{{anomaly.expected}}s</td>
                        <td>{{anomaly.z_score}}</td>
                        <td>{{anomaly.timestamp}}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
    </div>
</body>
</html>''')
    
    def generate_flow_diagram(self, pipeline, output_format: str = 'svg',
                             show_config: bool = False) -> Optional[str]:
        """
        Generate a flow diagram for a pipeline structure.
        
        Args:
            pipeline: The pipeline object
            output_format: Output format ('svg', 'png', 'html')
            show_config: Whether to show stage configuration details
            
        Returns:
            Path to the generated diagram file, or None if generation failed
        """
        if not self.has_networkx:
            self.logger.warning("NetworkX not available, cannot generate flow diagram")
            return None
            
        try:
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add pipeline node
            G.add_node("Pipeline", shape="box", style="filled", fillcolor="#ADD8E6")
            
            # Add stage nodes and connections
            for i, stage in enumerate(pipeline.stages):
                stage_name = stage.name
                stage_id = f"stage_{i}"
                
                # Add stage node
                G.add_node(stage_id, label=stage_name, shape="box", style="filled", fillcolor="#E8E8E8")
                
                # Connect pipeline to first stage
                if i == 0:
                    G.add_edge("Pipeline", stage_id)
                # Connect stages in sequence
                elif i > 0:
                    G.add_edge(f"stage_{i-1}", stage_id)
                    
                # Add configuration details if requested
                if show_config and hasattr(stage, 'config'):
                    config_id = f"config_{i}"
                    config_str = "\n".join([f"{k}: {v}" for k, v in stage.config.items()])
                    G.add_node(config_id, label=f"Config:\n{config_str}", shape="note", style="filled", fillcolor="#FFFACD")
                    G.add_edge(stage_id, config_id, style="dashed")
            
            # Generate timestamp for unique filename
            timestamp = int(time.time())
            filename = f"pipeline_flow_{timestamp}.{output_format}"
            output_path = os.path.join(self.output_dir, filename)
            
            # Generate the diagram
            if not MATPLOTLIB_AVAILABLE:
                # Simple text-based output if matplotlib not available
                with open(output_path.replace(output_format, 'txt'), 'w') as f:
                    f.write(f"Pipeline: {pipeline.name}\n")
                    f.write("Stages:\n")
                    for i, stage in enumerate(pipeline.stages):
                        f.write(f"  {i+1}. {stage.name}\n")
                return output_path.replace(output_format, 'txt')
                
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            
            # Draw nodes with labels and styles
            node_labels = {node: data.get('label', node) for node, data in G.nodes(data=True)}
            node_colors = [data.get('fillcolor', '#FFFFFF') for _, data in G.nodes(data=True)]
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, 
                                  node_size=2000, edgecolors='black')
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=1.5, arrowsize=20, alpha=0.7)
            
            plt.title(f"Pipeline: {pipeline.name}")
            plt.axis('off')
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(output_path, format=output_format, bbox_inches='tight')
            plt.close()
            
            return output_path
        
        except Exception as e:
            self.logger.error(f"Error generating flow diagram: {str(e)}")
            return None
    
    def create_performance_chart(self, metrics: Dict[str, Any], 
                               output_format: str = 'svg') -> Optional[str]:
        """
        Create performance visualization from pipeline metrics.
        
        Args:
            metrics: Pipeline metrics
            output_format: Output format ('svg', 'png')
            
        Returns:
            Path to the generated chart file, or None if generation failed
        """
        if not self.has_matplotlib:
            self.logger.warning("Matplotlib not available, cannot create performance chart")
            return None
            
        try:
            # Generate timestamp for unique filename
            timestamp = int(time.time())
            filename = f"performance_chart_{timestamp}.{output_format}"
            output_path = os.path.join(self.output_dir, filename)
            
            # Extract execution history
            pipeline_history = {}
            for pipeline_name, pipeline_data in metrics.get('pipelines', {}).items():
                history = pipeline_data.get('execution_history', [])
                if history:
                    pipeline_history[pipeline_name] = history
            
            if not pipeline_history:
                self.logger.warning("No performance history available for chart")
                return None
                
            # Create multi-panel plot based on available data
            num_pipelines = len(pipeline_history)
            fig, axes = plt.subplots(num_pipelines, 2, figsize=(14, 4 * num_pipelines))
            
            # Handle single pipeline case
            if num_pipelines == 1:
                axes = axes.reshape(1, 2)
                
            for i, (pipeline_name, history) in enumerate(pipeline_history.items()):
                # Convert timestamps to datetime for better display
                timestamps = [datetime.fromtimestamp(entry.get('timestamp', 0)) for entry in history]
                durations = [entry.get('duration', 0) for entry in history]
                success = [entry.get('success', False) for entry in history]
                
                # Create duration plot
                duration_ax = axes[i, 0]
                duration_colors = ['green' if s else 'red' for s in success]
                duration_ax.bar(range(len(durations)), durations, color=duration_colors, alpha=0.7)
                duration_ax.set_title(f"{pipeline_name}: Execution Duration")
                duration_ax.set_xlabel("Execution")
                duration_ax.set_ylabel("Duration (seconds)")
                
                # Add success rate annotation
                success_rate = sum(success) / len(success) * 100 if success else 0
                duration_ax.annotate(f"Success Rate: {success_rate:.1f}%", 
                                   xy=(0.05, 0.95), xycoords='axes fraction',
                                   fontsize=10, ha='left', va='top')
                
                # Create execution timeline
                timeline_ax = axes[i, 1]
                timeline_colors = ['green' if s else 'red' for s in success]
                
                if len(timestamps) > 1:
                    timeline_ax.scatter(timestamps, durations, c=timeline_colors, s=50, alpha=0.7)
                    timeline_ax.plot(timestamps, durations, 'k--', alpha=0.3)
                else:
                    timeline_ax.scatter(timestamps, durations, c=timeline_colors, s=50, alpha=0.7)
                    
                timeline_ax.set_title(f"{pipeline_name}: Execution Timeline")
                timeline_ax.set_xlabel("Timestamp")
                timeline_ax.set_ylabel("Duration (seconds)")
                
                # Format x-axis for better datetime display
                timeline_ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                fig.autofmt_xdate()
                
                # Add trend line if enough data
                if len(timestamps) >= 3:
                    try:
                        import numpy as np
                        from scipy import stats
                        
                        # Convert timestamps to ordinal values for regression
                        x = mdates.date2num(timestamps)
                        y = durations
                        
                        # Calculate trend line
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        trend_line = slope * x + intercept
                        
                        # Plot trend line
                        timeline_ax.plot(timestamps, trend_line, 'b-', linewidth=2, alpha=0.7)
                        
                        # Add annotation
                        if slope > 0:
                            trend_text = f"Trend: Increasing (r={r_value:.2f})"
                        elif slope < 0:
                            trend_text = f"Trend: Decreasing (r={r_value:.2f})"
                        else:
                            trend_text = f"Trend: Stable (r={r_value:.2f})"
                            
                        timeline_ax.annotate(trend_text, xy=(0.05, 0.05), 
                                           xycoords='axes fraction', fontsize=10)
                    except ImportError:
                        pass  # Skip trend line if scipy not available
            
            plt.tight_layout()
            plt.savefig(output_path, format=output_format, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating performance chart: {str(e)}")
            return None
    
    def generate_execution_timeline(self, execution_log: Dict[str, Any], 
                                  output_format: str = 'svg') -> Optional[str]:
        """
        Generate an execution timeline visualization.
        
        Args:
            execution_log: Execution log data
            output_format: Output format ('svg', 'png')
            
        Returns:
            Path to the generated timeline file, or None if generation failed
        """
        if not self.has_matplotlib:
            self.logger.warning("Matplotlib not available, cannot generate execution timeline")
            return None
            
        try:
            # Generate timestamp for unique filename
            timestamp = int(time.time())
            filename = f"execution_timeline_{timestamp}.{output_format}"
            output_path = os.path.join(self.output_dir, filename)
            
            # Extract timeline data
            stages = execution_log.get('stages', {})
            pipeline_name = execution_log.get('pipeline_name', 'Unknown Pipeline')
            
            if not stages:
                self.logger.warning("No stage data available for execution timeline")
                return None
                
            # Prepare stage data for plotting
            stage_names = []
            start_times = []
            durations = []
            statuses = []
            
            pipeline_start = execution_log.get('start_time', 0)
            
            for stage_name, stage_data in stages.items():
                stage_names.append(stage_name)
                
                # Calculate relative start time from pipeline start
                stage_start = stage_data.get('start_time', pipeline_start)
                relative_start = stage_start - pipeline_start
                start_times.append(relative_start)
                
                # Duration
                if 'end_time' in stage_data and 'start_time' in stage_data:
                    duration = stage_data['end_time'] - stage_data['start_time']
                else:
                    duration = stage_data.get('execution_time', 0)
                    
                durations.append(duration)
                
                # Status
                statuses.append(stage_data.get('status', 'unknown'))
            
            # Sort stages by start time
            sorted_data = sorted(zip(stage_names, start_times, durations, statuses), 
                               key=lambda x: x[1])
            stage_names, start_times, durations, statuses = zip(*sorted_data)
            
            # Create colors based on status
            colors = []
            for status in statuses:
                if status == 'success':
                    colors.append('#28a745')  # Green
                elif status == 'failed':
                    colors.append('#dc3545')  # Red
                else:
                    colors.append('#6c757d')  # Gray
            
            # Create execution timeline
            plt.figure(figsize=(12, max(6, len(stage_names) * 0.5)))
            
            # Plot horizontal bars for each stage
            y_pos = range(len(stage_names))
            plt.barh(y_pos, durations, left=start_times, color=colors, alpha=0.8)
            
            # Add stage names and duration labels
            for i, (name, start, duration, status) in enumerate(zip(stage_names, start_times, durations, statuses)):
                # Add stage name
                plt.text(start - 0.1, i, name, ha='right', va='center', fontsize=10)
                
                # Add duration
                plt.text(start + duration / 2, i, f"{duration:.2f}s", 
                       ha='center', va='center', fontsize=9, 
                       color='white' if status != 'unknown' else 'black')
            
            # Set labels and title
            plt.yticks([])  # Hide y-tick labels since we add our own stage names
            plt.xlabel('Time (seconds)')
            plt.title(f"Execution Timeline: {pipeline_name}")
            
            # Add end time marker
            total_duration = max([s + d for s, d in zip(start_times, durations)]) if start_times else 0
            plt.axvline(x=total_duration, color='black', linestyle='--', alpha=0.3)
            plt.text(total_duration + 0.1, len(stage_names) - 1, 
                   f"Total: {total_duration:.2f}s", ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig(output_path, format=output_format, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating execution timeline: {str(e)}")
            return None
    
    def create_dashboard(self, pipelines: List[Dict[str, Any]], 
                        monitor_data: Dict[str, Any] = None,
                        output_dir: Optional[str] = None) -> Optional[str]:
        """
        Create a comprehensive dashboard for pipeline monitoring.
        
        Args:
            pipelines: List of pipeline data
            monitor_data: Data from the PipelineMonitor
            output_dir: Directory to save dashboard files (default: self.output_dir)
            
        Returns:
            Path to the dashboard HTML file, or None if creation failed
        """
        if not self.has_jinja2:
            self.logger.warning("Jinja2 not available, cannot create dashboard")
            return None
            
        try:
            # Use specified output dir or default
            output_dir = output_dir or self.output_dir
            
            # Create dashboard directory
            dashboard_dir = os.path.join(output_dir, 'dashboard')
            os.makedirs(dashboard_dir, exist_ok=True)
            
            # Generate charts for the dashboard
            resource_chart_path = self._generate_resource_chart(monitor_data, dashboard_dir)
            error_chart_path = self._generate_error_chart(monitor_data, dashboard_dir)
            
            # Prepare template variables
            active_pipelines = monitor_data.get('active_pipelines', []) if monitor_data else []
            recent_executions = self._extract_recent_executions(pipelines, monitor_data)
            
            # Calculate aggregate metrics
            total_pipelines = len(pipelines)
            
            # Success rate calculation
            success_count = 0
            total_executions = 0
            total_duration = 0
            for pipeline in pipelines:
                pipeline_success = pipeline.get('successful_executions', 0)
                pipeline_total = pipeline.get('total_executions', 0)
                pipeline_duration = pipeline.get('total_duration', 0)
                
                success_count += pipeline_success
                total_executions += pipeline_total
                total_duration += pipeline_duration
                
            success_rate = (success_count / total_executions * 100) if total_executions > 0 else 0
            avg_duration = (total_duration / total_executions) if total_executions > 0 else 0
            
            # Extract error statistics
            error_stats = {}
            if monitor_data and 'error_analysis' in monitor_data:
                error_data = monitor_data['error_analysis'].get('global_stats', {})
                error_types = error_data.get('error_types', {})
                total_errors = error_data.get('total_errors', 0)
                
                for error_type, count in error_types.items():
                    percentage = (count / total_errors * 100) if total_errors > 0 else 0
                    error_stats[error_type] = {
                        'count': count,
                        'percentage': round(percentage, 1)
                    }
            
            # Current resource utilization
            memory_usage = 0
            cpu_usage = 0
            if monitor_data and 'resource_utilization' in monitor_data:
                resource_data = monitor_data['resource_utilization'].get('current', {})
                memory_usage = resource_data.get('memory_percent', 0)
                cpu_usage = resource_data.get('cpu_percent', 0)
            
            # Render dashboard template
            template = self.template_env.get_template('dashboard.html')
            html = template.render(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                active_pipelines=active_pipelines,
                active_pipelines_count=len(active_pipelines),
                total_pipelines_count=total_pipelines,
                success_rate=round(success_rate, 1),
                avg_duration=round(avg_duration, 2),
                recent_executions=recent_executions,
                memory_usage=round(memory_usage, 1),
                cpu_usage=round(cpu_usage, 1),
                resource_chart_path=os.path.basename(resource_chart_path) if resource_chart_path else '',
                error_chart_path=os.path.basename(error_chart_path) if error_chart_path else '',
                error_stats=error_stats
            )
            
            # Write dashboard HTML
            dashboard_path = os.path.join(dashboard_dir, 'dashboard.html')
            with open(dashboard_path, 'w') as f:
                f.write(html)
                
            # Generate individual pipeline detail pages
            for pipeline in pipelines:
                self._generate_pipeline_detail_page(pipeline, monitor_data, dashboard_dir)
                
            return dashboard_path
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {str(e)}")
            return None
    
    def _generate_resource_chart(self, monitor_data: Dict[str, Any], 
                               output_dir: str) -> Optional[str]:
        """
        Generate resource utilization chart.
        
        Args:
            monitor_data: Monitor data containing resource information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract resource data
            resource_data = monitor_data.get('resource_utilization', {})
            history = resource_data.get('history', [])
            
            if not history:
                return None
                
            # Generate chart
            filename = f"resource_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            timestamps = [datetime.fromtimestamp(entry.get('timestamp', 0)) for entry in history]
            memory_values = [entry.get('memory_percent', 0) for entry in history]
            cpu_values = [entry.get('cpu_percent', 0) for entry in history]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot memory and CPU
            ax.plot(timestamps, memory_values, 'b-', label='Memory %', alpha=0.7)
            ax.plot(timestamps, cpu_values, 'r-', label='CPU %', alpha=0.7)
            
            # Add labels and legend
            ax.set_xlabel('Time')
            ax.set_ylabel('Utilization %')
            ax.set_title('Resource Utilization')
            ax.legend()
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating resource chart: {str(e)}")
            return None
    
    def _generate_error_chart(self, monitor_data: Dict[str, Any], 
                            output_dir: str) -> Optional[str]:
        """
        Generate error distribution chart.
        
        Args:
            monitor_data: Monitor data containing error information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract error data
            error_data = monitor_data.get('error_analysis', {}).get('global_stats', {})
            error_distribution = error_data.get('error_distribution', {})
            
            if not error_distribution:
                return None
                
            # Generate chart
            filename = f"error_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            error_types = list(error_distribution.keys())
            percentages = list(error_distribution.values())
            
            # Create color map
            colors = plt.cm.Set3(range(len(error_types)))
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(percentages, labels=error_types, autopct='%1.1f%%', 
                  colors=colors, shadow=False, startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            plt.title('Error Distribution')
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating error chart: {str(e)}")
            return None
    
    def _extract_recent_executions(self, pipelines: List[Dict[str, Any]], 
                                 monitor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract recent execution data from multiple pipelines.
        
        Args:
            pipelines: List of pipeline data
            monitor_data: Monitor data
            
        Returns:
            List of recent executions across all pipelines
        """
        recent_executions = []
        
        # Extract from monitor data if available
        if monitor_data and 'recent_executions' in monitor_data:
            return monitor_data['recent_executions']
            
        # Otherwise extract from pipeline data
        for pipeline in pipelines:
            pipeline_name = pipeline.get('name', 'Unknown')
            history = pipeline.get('execution_history', [])
            
            for execution in history:
                execution_data = {
                    'pipeline_name': pipeline_name,
                    'success': execution.get('success', False),
                    'duration': round(execution.get('duration', 0), 2),
                    'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    'stages_completed': len(execution.get('stages', {})),
                    'stages_total': len(execution.get('stages', {}))
                }
                recent_executions.append(execution_data)
        
        # Sort by timestamp (most recent first) and limit to 10
        recent_executions.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), reverse=True)
        return recent_executions[:10]
    
    def _generate_pipeline_detail_page(self, pipeline: Dict[str, Any], 
                                     monitor_data: Dict[str, Any],
                                     output_dir: str) -> Optional[str]:
        """
        Generate detail page for a specific pipeline.
        
        Args:
            pipeline: Pipeline data
            monitor_data: Monitor data
            output_dir: Directory to save the detail page
            
        Returns:
            Path to the detail page file, or None if generation failed
        """
        if not self.has_jinja2:
            return None
            
        try:
            pipeline_name = pipeline.get('name', 'Unknown')
            
            # Generate charts for the pipeline
            charts_dir = os.path.join(output_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            # Flow diagram would come from a specific pipeline object, not metrics
            flow_diagram_path = ''
            
            # Performance chart from history
            performance_data = {
                'pipelines': {pipeline_name: pipeline}
            }
            performance_chart_path = self.create_performance_chart(performance_data, 'svg')
            if performance_chart_path:
                # Copy to charts directory and get relative path
                import shutil
                dest_path = os.path.join(charts_dir, os.path.basename(performance_chart_path))
                shutil.copy2(performance_chart_path, dest_path)
                performance_chart_path = os.path.join('charts', os.path.basename(performance_chart_path))
            
            # Stage performance data
            stage_chart_path = self._generate_stage_performance_chart(pipeline, charts_dir)
            if stage_chart_path:
                stage_chart_path = os.path.join('charts', os.path.basename(stage_chart_path))
            
            # Extract pipeline metrics
            total_executions = pipeline.get('total_executions', 0)
            successful_executions = pipeline.get('successful_executions', 0)
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
            avg_duration = pipeline.get('average_duration', 0)
            
            # Get recent executions
            history = list(pipeline.get('execution_history', []))
            history.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            recent_executions = [{
                'id': execution.get('pipeline_id', 'unknown'),
                'success': execution.get('success', False),
                'duration': round(execution.get('duration', 0), 2),
                'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'error_count': execution.get('error_count', 0)
            } for execution in history[:10]]
            
            # Latest execution status
            latest_success = history[0].get('success', False) if history else False
            
            # Stage data
            stages = []
            stage_metrics = {}
            
            # Try to get stage metrics from monitor data
            if monitor_data and 'stage_metrics' in monitor_data:
                for stage_id, stage_data in monitor_data['stage_metrics'].items():
                    if stage_id.startswith(f"{pipeline_name}:"):
                        stage_name = stage_id.split(':', 1)[1]
                        stage_metrics[stage_name] = stage_data
            
            # Extract stage data from execution history if not available in monitor
            if not stage_metrics and history:
                latest_execution = history[0]
                stage_data = latest_execution.get('stages', {})
                
                for stage_name, data in stage_data.items():
                    stage_metrics[stage_name] = {
                        'avg_duration': data.get('duration', 0),
                        'success_rate': 100 if data.get('status') == 'success' else 0,
                        'total_executions': 1
                    }
            
            # Format stage data for template
            for stage_name, data in stage_metrics.items():
                stages.append({
                    'name': stage_name,
                    'avg_duration': round(data.get('avg_duration', 0), 2),
                    'success_rate': round(data.get('success_rate', 0), 1),
                    'total_executions': data.get('total_executions', 0)
                })
            
            # Sort stages by name
            stages.sort(key=lambda x: x['name'])
            
            # Anomaly data
            anomalies = []
            has_anomalies = False
            
            if monitor_data and 'anomalies' in monitor_data:
                pipeline_anomalies = [a for a in monitor_data['anomalies'] 
                                    if a.get('pipeline_name') == pipeline_name]
                
                if pipeline_anomalies:
                    has_anomalies = True
                    anomalies = [{
                        'type': anomaly.get('type', 'unknown'),
                        'value': round(anomaly.get('value', 0), 2),
                        'expected': round(anomaly.get('expected', 0), 2),
                        'z_score': round(anomaly.get('z_score', 0), 2),
                        'timestamp': datetime.fromtimestamp(anomaly.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
                    } for anomaly in pipeline_anomalies]
            
            # Render template
            template = self.template_env.get_template('pipeline_detail.html')
            html = template.render(
                pipeline_name=pipeline_name,
                total_executions=total_executions,
                success_rate=round(success_rate, 1),
                avg_duration=round(avg_duration, 2),
                latest_success=latest_success,
                flow_diagram_path=flow_diagram_path,
                performance_chart_path=performance_chart_path or '',
                stage_chart_path=stage_chart_path or '',
                stages=stages,
                recent_executions=recent_executions,
                has_anomalies=has_anomalies,
                anomalies=anomalies
            )
            
            # Write detail page
            safe_name = pipeline_name.replace(' ', '_').lower()
            detail_path = os.path.join(output_dir, f"{safe_name}_detail.html")
            with open(detail_path, 'w') as f:
                f.write(html)
                
            return detail_path
            
        except Exception as e:
            self.logger.error(f"Error generating pipeline detail page: {str(e)}")
            return None
    
    def _generate_stage_performance_chart(self, pipeline: Dict[str, Any], 
                                       output_dir: str) -> Optional[str]:
        """
        Generate a chart showing performance metrics for each stage.
        
        Args:
            pipeline: Pipeline data
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib:
            return None
            
        try:
            # Extract stage data from most recent execution
            history = list(pipeline.get('execution_history', []))
            if not history:
                return None
                
            # Get the most recent execution with stage data
            for execution in history:
                if 'stages' in execution and execution['stages']:
                    stages = execution['stages']
                    break
            else:
                return None
                
            # Generate chart
            filename = f"stage_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            stage_names = []
            durations = []
            statuses = []
            
            for stage_name, stage_data in stages.items():
                stage_names.append(stage_name)
                durations.append(stage_data.get('duration', 0))
                statuses.append(stage_data.get('status', 'unknown'))
            
            # Sort stages by duration (descending)
            sorted_data = sorted(zip(stage_names, durations, statuses), 
                               key=lambda x: x[1], reverse=True)
            stage_names, durations, statuses = zip(*sorted_data)
            
            # Create colors based on status
            colors = []
            for status in statuses:
                if status == 'success':
                    colors.append('#28a745')  # Green
                elif status == 'failed':
                    colors.append('#dc3545')  # Red
                else:
                    colors.append('#6c757d')  # Gray
            
            # Create horizontal bar chart
            plt.figure(figsize=(10, max(6, len(stage_names) * 0.5)))
            
            y_pos = range(len(stage_names))
            plt.barh(y_pos, durations, color=colors, alpha=0.8)
            
            # Add stage names and duration labels
            for i, (name, duration) in enumerate(zip(stage_names, durations)):
                plt.text(duration + 0.1, i, f"{duration:.2f}s", va='center')
                
            plt.yticks(y_pos, stage_names)
            plt.xlabel('Duration (seconds)')
            plt.title('Stage Performance')
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating stage performance chart: {str(e)}")
            return None
    
    def _generate_resource_chart(self, monitor_data: Dict[str, Any], 
                               output_dir: str) -> Optional[str]:
        """
        Generate resource utilization chart.
        
        Args:
            monitor_data: Monitor data containing resource information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract resource data
            resource_data = monitor_data.get('resource_utilization', {})
            history = resource_data.get('history', [])
            
            if not history:
                return None
                
            # Generate chart
            filename = f"resource_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            timestamps = [datetime.fromtimestamp(entry.get('timestamp', 0)) for entry in history]
            memory_values = [entry.get('memory_percent', 0) for entry in history]
            cpu_values = [entry.get('cpu_percent', 0) for entry in history]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot memory and CPU
            ax.plot(timestamps, memory_values, 'b-', label='Memory %', alpha=0.7)
            ax.plot(timestamps, cpu_values, 'r-', label='CPU %', alpha=0.7)
            
            # Add labels and legend
            ax.set_xlabel('Time')
            ax.set_ylabel('Utilization %')
            ax.set_title('Resource Utilization')
            ax.legend()
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating resource chart: {str(e)}")
            return None
    
    def _generate_error_chart(self, monitor_data: Dict[str, Any], 
                            output_dir: str) -> Optional[str]:
        """
        Generate error distribution chart.
        
        Args:
            monitor_data: Monitor data containing error information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract error data
            error_data = monitor_data.get('error_analysis', {}).get('global_stats', {})
            error_distribution = error_data.get('error_distribution', {})
            
            if not error_distribution:
                return None
                
            # Generate chart
            filename = f"error_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            error_types = list(error_distribution.keys())
            percentages = list(error_distribution.values())
            
            # Create color map
            colors = plt.cm.Set3(range(len(error_types)))
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(percentages, labels=error_types, autopct='%1.1f%%', 
                  colors=colors, shadow=False, startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            plt.title('Error Distribution')
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating error chart: {str(e)}")
            return None
    
    def _extract_recent_executions(self, pipelines: List[Dict[str, Any]], 
                                 monitor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract recent execution data from multiple pipelines.
        
        Args:
            pipelines: List of pipeline data
            monitor_data: Monitor data
            
        Returns:
            List of recent executions across all pipelines
        """
        recent_executions = []
        
        # Extract from monitor data if available
        if monitor_data and 'recent_executions' in monitor_data:
            return monitor_data['recent_executions']
            
        # Otherwise extract from pipeline data
        for pipeline in pipelines:
            pipeline_name = pipeline.get('name', 'Unknown')
            history = pipeline.get('execution_history', [])
            
            for execution in history:
                execution_data = {
                    'pipeline_name': pipeline_name,
                    'success': execution.get('success', False),
                    'duration': round(execution.get('duration', 0), 2),
                    'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    'stages_completed': len(execution.get('stages', {})),
                    'stages_total': len(execution.get('stages', {}))
                }
                recent_executions.append(execution_data)
        
        # Sort by timestamp (most recent first) and limit to 10
        recent_executions.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), reverse=True)
        return recent_executions[:10]
    
    def _generate_pipeline_detail_page(self, pipeline: Dict[str, Any], 
                                     monitor_data: Dict[str, Any],
                                     output_dir: str) -> Optional[str]:
        """
        Generate detail page for a specific pipeline.
        
        Args:
            pipeline: Pipeline data
            monitor_data: Monitor data
            output_dir: Directory to save the detail page
            
        Returns:
            Path to the detail page file, or None if generation failed
        """
        if not self.has_jinja2:
            return None
            
        try:
            pipeline_name = pipeline.get('name', 'Unknown')
            
            # Generate charts for the pipeline
            charts_dir = os.path.join(output_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            # Flow diagram would come from a specific pipeline object, not metrics
            flow_diagram_path = ''
            
            # Performance chart from history
            performance_data = {
                'pipelines': {pipeline_name: pipeline}
            }
            performance_chart_path = self.create_performance_chart(performance_data, 'svg')
            if performance_chart_path:
                # Copy to charts directory and get relative path
                import shutil
                dest_path = os.path.join(charts_dir, os.path.basename(performance_chart_path))
                shutil.copy2(performance_chart_path, dest_path)
                performance_chart_path = os.path.join('charts', os.path.basename(performance_chart_path))
            
            # Stage performance data
            stage_chart_path = self._generate_stage_performance_chart(pipeline, charts_dir)
            if stage_chart_path:
                stage_chart_path = os.path.join('charts', os.path.basename(stage_chart_path))
            
            # Extract pipeline metrics
            total_executions = pipeline.get('total_executions', 0)
            successful_executions = pipeline.get('successful_executions', 0)
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
            avg_duration = pipeline.get('average_duration', 0)
            
            # Get recent executions
            history = list(pipeline.get('execution_history', []))
            history.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            recent_executions = [{
                'id': execution.get('pipeline_id', 'unknown'),
                'success': execution.get('success', False),
                'duration': round(execution.get('duration', 0), 2),
                'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'error_count': execution.get('error_count', 0)
            } for execution in history[:10]]
            
            # Latest execution status
            latest_success = history[0].get('success', False) if history else False
            
            # Stage data
            stages = []
            stage_metrics = {}
            
            # Try to get stage metrics from monitor data
            if monitor_data and 'stage_metrics' in monitor_data:
                for stage_id, stage_data in monitor_data['stage_metrics'].items():
                    if stage_id.startswith(f"{pipeline_name}:"):
                        stage_name = stage_id.split(':', 1)[1]
                        stage_metrics[stage_name] = stage_data
            
            # Extract stage data from execution history if not available in monitor
            if not stage_metrics and history:
                latest_execution = history[0]
                stage_data = latest_execution.get('stages', {})
                
                for stage_name, data in stage_data.items():
                    stage_metrics[stage_name] = {
                        'avg_duration': data.get('duration', 0),
                        'success_rate': 100 if data.get('status') == 'success' else 0,
                        'total_executions': 1
                    }
            
            # Format stage data for template
            for stage_name, data in stage_metrics.items():
                stages.append({
                    'name': stage_name,
                    'avg_duration': round(data.get('avg_duration', 0), 2),
                    'success_rate': round(data.get('success_rate', 0), 1),
                    'total_executions': data.get('total_executions', 0)
                })
            
            # Sort stages by name
            stages.sort(key=lambda x: x['name'])
            
            # Anomaly data
            anomalies = []
            has_anomalies = False
            
            if monitor_data and 'anomalies' in monitor_data:
                pipeline_anomalies = [a for a in monitor_data['anomalies'] 
                                    if a.get('pipeline_name') == pipeline_name]
                
                if pipeline_anomalies:
                    has_anomalies = True
                    anomalies = [{
                        'type': anomaly.get('type', 'unknown'),
                        'value': round(anomaly.get('value', 0), 2),
                        'expected': round(anomaly.get('expected', 0), 2),
                        'z_score': round(anomaly.get('z_score', 0), 2),
                        'timestamp': datetime.fromtimestamp(anomaly.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
                    } for anomaly in pipeline_anomalies]
            
            # Render template
            template = self.template_env.get_template('pipeline_detail.html')
            html = template.render(
                pipeline_name=pipeline_name,
                total_executions=total_executions,
                success_rate=round(success_rate, 1),
                avg_duration=round(avg_duration, 2),
                latest_success=latest_success,
                flow_diagram_path=flow_diagram_path,
                performance_chart_path=performance_chart_path or '',
                stage_chart_path=stage_chart_path or '',
                stages=stages,
                recent_executions=recent_executions,
                has_anomalies=has_anomalies,
                anomalies=anomalies
            )
            
            # Write detail page
            safe_name = pipeline_name.replace(' ', '_').lower()
            detail_path = os.path.join(output_dir, f"{safe_name}_detail.html")
            with open(detail_path, 'w') as f:
                f.write(html)
                
            return detail_path
            
        except Exception as e:
            self.logger.error(f"Error generating pipeline detail page: {str(e)}")
            return None
    
    def _generate_stage_performance_chart(self, pipeline: Dict[str, Any], 
                                       output_dir: str) -> Optional[str]:
        """
        Generate a chart showing performance metrics for each stage.
        
        Args:
            pipeline: Pipeline data
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib:
            return None
            
        try:
            # Extract stage data from most recent execution
            history = list(pipeline.get('execution_history', []))
            if not history:
                return None
                
            # Get the most recent execution with stage data
            for execution in history:
                if 'stages' in execution and execution['stages']:
                    stages = execution['stages']
                    break
            else:
                return None
                
            # Generate chart
            filename = f"stage_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            stage_names = []
            durations = []
            statuses = []
            
            for stage_name, stage_data in stages.items():
                stage_names.append(stage_name)
                durations.append(stage_data.get('duration', 0))
                statuses.append(stage_data.get('status', 'unknown'))
            
            # Sort stages by duration (descending)
            sorted_data = sorted(zip(stage_names, durations, statuses), 
                               key=lambda x: x[1], reverse=True)
            stage_names, durations, statuses = zip(*sorted_data)
            
            # Create colors based on status
            colors = []
            for status in statuses:
                if status == 'success':
                    colors.append('#28a745')  # Green
                elif status == 'failed':
                    colors.append('#dc3545')  # Red
                else:
                    colors.append('#6c757d')  # Gray
            
            # Create horizontal bar chart
            plt.figure(figsize=(10, max(6, len(stage_names) * 0.5)))
            
            y_pos = range(len(stage_names))
            plt.barh(y_pos, durations, color=colors, alpha=0.8)
            
            # Add stage names and duration labels
            for i, (name, duration) in enumerate(zip(stage_names, durations)):
                plt.text(duration + 0.1, i, f"{duration:.2f}s", va='center')
                
            plt.yticks(y_pos, stage_names)
            plt.xlabel('Duration (seconds)')
            plt.title('Stage Performance')
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating stage performance chart: {str(e)}")
            return None
    
    def _generate_resource_chart(self, monitor_data: Dict[str, Any], 
                               output_dir: str) -> Optional[str]:
        """
        Generate resource utilization chart.
        
        Args:
            monitor_data: Monitor data containing resource information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract resource data
            resource_data = monitor_data.get('resource_utilization', {})
            history = resource_data.get('history', [])
            
            if not history:
                return None
                
            # Generate chart
            filename = f"resource_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            timestamps = [datetime.fromtimestamp(entry.get('timestamp', 0)) for entry in history]
            memory_values = [entry.get('memory_percent', 0) for entry in history]
            cpu_values = [entry.get('cpu_percent', 0) for entry in history]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot memory and CPU
            ax.plot(timestamps, memory_values, 'b-', label='Memory %', alpha=0.7)
            ax.plot(timestamps, cpu_values, 'r-', label='CPU %', alpha=0.7)
            
            # Add labels and legend
            ax.set_xlabel('Time')
            ax.set_ylabel('Utilization %')
            ax.set_title('Resource Utilization')
            ax.legend()
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating resource chart: {str(e)}")
            return None
    
    def _generate_error_chart(self, monitor_data: Dict[str, Any], 
                            output_dir: str) -> Optional[str]:
        """
        Generate error distribution chart.
        
        Args:
            monitor_data: Monitor data containing error information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract error data
            error_data = monitor_data.get('error_analysis', {}).get('global_stats', {})
            error_distribution = error_data.get('error_distribution', {})
            
            if not error_distribution:
                return None
                
            # Generate chart
            filename = f"error_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            error_types = list(error_distribution.keys())
            percentages = list(error_distribution.values())
            
            # Create color map
            colors = plt.cm.Set3(range(len(error_types)))
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(percentages, labels=error_types, autopct='%1.1f%%', 
                  colors=colors, shadow=False, startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            plt.title('Error Distribution')
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating error chart: {str(e)}")
            return None
    
    def _extract_recent_executions(self, pipelines: List[Dict[str, Any]], 
                                 monitor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract recent execution data from multiple pipelines.
        
        Args:
            pipelines: List of pipeline data
            monitor_data: Monitor data
            
        Returns:
            List of recent executions across all pipelines
        """
        recent_executions = []
        
        # Extract from monitor data if available
        if monitor_data and 'recent_executions' in monitor_data:
            return monitor_data['recent_executions']
            
        # Otherwise extract from pipeline data
        for pipeline in pipelines:
            pipeline_name = pipeline.get('name', 'Unknown')
            history = pipeline.get('execution_history', [])
            
            for execution in history:
                execution_data = {
                    'pipeline_name': pipeline_name,
                    'success': execution.get('success', False),
                    'duration': round(execution.get('duration', 0), 2),
                    'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    'stages_completed': len(execution.get('stages', {})),
                    'stages_total': len(execution.get('stages', {}))
                }
                recent_executions.append(execution_data)
        
        # Sort by timestamp (most recent first) and limit to 10
        recent_executions.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), reverse=True)
        return recent_executions[:10]
    
    def _generate_pipeline_detail_page(self, pipeline: Dict[str, Any], 
                                     monitor_data: Dict[str, Any],
                                     output_dir: str) -> Optional[str]:
        """
        Generate detail page for a specific pipeline.
        
        Args:
            pipeline: Pipeline data
            monitor_data: Monitor data
            output_dir: Directory to save the detail page
            
        Returns:
            Path to the detail page file, or None if generation failed
        """
        if not self.has_jinja2:
            return None
            
        try:
            pipeline_name = pipeline.get('name', 'Unknown')
            
            # Generate charts for the pipeline
            charts_dir = os.path.join(output_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            # Flow diagram would come from a specific pipeline object, not metrics
            flow_diagram_path = ''
            
            # Performance chart from history
            performance_data = {
                'pipelines': {pipeline_name: pipeline}
            }
            performance_chart_path = self.create_performance_chart(performance_data, 'svg')
            if performance_chart_path:
                # Copy to charts directory and get relative path
                import shutil
                dest_path = os.path.join(charts_dir, os.path.basename(performance_chart_path))
                shutil.copy2(performance_chart_path, dest_path)
                performance_chart_path = os.path.join('charts', os.path.basename(performance_chart_path))
            
            # Stage performance data
            stage_chart_path = self._generate_stage_performance_chart(pipeline, charts_dir)
            if stage_chart_path:
                stage_chart_path = os.path.join('charts', os.path.basename(stage_chart_path))
            
            # Extract pipeline metrics
            total_executions = pipeline.get('total_executions', 0)
            successful_executions = pipeline.get('successful_executions', 0)
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
            avg_duration = pipeline.get('average_duration', 0)
            
            # Get recent executions
            history = list(pipeline.get('execution_history', []))
            history.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            recent_executions = [{
                'id': execution.get('pipeline_id', 'unknown'),
                'success': execution.get('success', False),
                'duration': round(execution.get('duration', 0), 2),
                'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'error_count': execution.get('error_count', 0)
            } for execution in history[:10]]
            
            # Latest execution status
            latest_success = history[0].get('success', False) if history else False
            
            # Stage data
            stages = []
            stage_metrics = {}
            
            # Try to get stage metrics from monitor data
            if monitor_data and 'stage_metrics' in monitor_data:
                for stage_id, stage_data in monitor_data['stage_metrics'].items():
                    if stage_id.startswith(f"{pipeline_name}:"):
                        stage_name = stage_id.split(':', 1)[1]
                        stage_metrics[stage_name] = stage_data
            
            # Extract stage data from execution history if not available in monitor
            if not stage_metrics and history:
                latest_execution = history[0]
                stage_data = latest_execution.get('stages', {})
                
                for stage_name, data in stage_data.items():
                    stage_metrics[stage_name] = {
                        'avg_duration': data.get('duration', 0),
                        'success_rate': 100 if data.get('status') == 'success' else 0,
                        'total_executions': 1
                    }
            
            # Format stage data for template
            for stage_name, data in stage_metrics.items():
                stages.append({
                    'name': stage_name,
                    'avg_duration': round(data.get('avg_duration', 0), 2),
                    'success_rate': round(data.get('success_rate', 0), 1),
                    'total_executions': data.get('total_executions', 0)
                })
            
            # Sort stages by name
            stages.sort(key=lambda x: x['name'])
            
            # Anomaly data
            anomalies = []
            has_anomalies = False
            
            if monitor_data and 'anomalies' in monitor_data:
                pipeline_anomalies = [a for a in monitor_data['anomalies'] 
                                    if a.get('pipeline_name') == pipeline_name]
                
                if pipeline_anomalies:
                    has_anomalies = True
                    anomalies = [{
                        'type': anomaly.get('type', 'unknown'),
                        'value': round(anomaly.get('value', 0), 2),
                        'expected': round(anomaly.get('expected', 0), 2),
                        'z_score': round(anomaly.get('z_score', 0), 2),
                        'timestamp': datetime.fromtimestamp(anomaly.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
                    } for anomaly in pipeline_anomalies]
            
            # Render template
            template = self.template_env.get_template('pipeline_detail.html')
            html = template.render(
                pipeline_name=pipeline_name,
                total_executions=total_executions,
                success_rate=round(success_rate, 1),
                avg_duration=round(avg_duration, 2),
                latest_success=latest_success,
                flow_diagram_path=flow_diagram_path,
                performance_chart_path=performance_chart_path or '',
                stage_chart_path=stage_chart_path or '',
                stages=stages,
                recent_executions=recent_executions,
                has_anomalies=has_anomalies,
                anomalies=anomalies
            )
            
            # Write detail page
            safe_name = pipeline_name.replace(' ', '_').lower()
            detail_path = os.path.join(output_dir, f"{safe_name}_detail.html")
            with open(detail_path, 'w') as f:
                f.write(html)
                
            return detail_path
            
        except Exception as e:
            self.logger.error(f"Error generating pipeline detail page: {str(e)}")
            return None
    
    def _generate_stage_performance_chart(self, pipeline: Dict[str, Any], 
                                       output_dir: str) -> Optional[str]:
        """
        Generate a chart showing performance metrics for each stage.
        
        Args:
            pipeline: Pipeline data
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib:
            return None
            
        try:
            # Extract stage data from most recent execution
            history = list(pipeline.get('execution_history', []))
            if not history:
                return None
                
            # Get the most recent execution with stage data
            for execution in history:
                if 'stages' in execution and execution['stages']:
                    stages = execution['stages']
                    break
            else:
                return None
                
            # Generate chart
            filename = f"stage_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            stage_names = []
            durations = []
            statuses = []
            
            for stage_name, stage_data in stages.items():
                stage_names.append(stage_name)
                durations.append(stage_data.get('duration', 0))
                statuses.append(stage_data.get('status', 'unknown'))
            
            # Sort stages by duration (descending)
            sorted_data = sorted(zip(stage_names, durations, statuses), 
                               key=lambda x: x[1], reverse=True)
            stage_names, durations, statuses = zip(*sorted_data)
            
            # Create colors based on status
            colors = []
            for status in statuses:
                if status == 'success':
                    colors.append('#28a745')  # Green
                elif status == 'failed':
                    colors.append('#dc3545')  # Red
                else:
                    colors.append('#6c757d')  # Gray
            
            # Create horizontal bar chart
            plt.figure(figsize=(10, max(6, len(stage_names) * 0.5)))
            
            y_pos = range(len(stage_names))
            plt.barh(y_pos, durations, color=colors, alpha=0.8)
            
            # Add stage names and duration labels
            for i, (name, duration) in enumerate(zip(stage_names, durations)):
                plt.text(duration + 0.1, i, f"{duration:.2f}s", va='center')
                
            plt.yticks(y_pos, stage_names)
            plt.xlabel('Duration (seconds)')
            plt.title('Stage Performance')
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating stage performance chart: {str(e)}")
            return None
    
    def _generate_resource_chart(self, monitor_data: Dict[str, Any], 
                               output_dir: str) -> Optional[str]:
        """
        Generate resource utilization chart.
        
        Args:
            monitor_data: Monitor data containing resource information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract resource data
            resource_data = monitor_data.get('resource_utilization', {})
            history = resource_data.get('history', [])
            
            if not history:
                return None
                
            # Generate chart
            filename = f"resource_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            timestamps = [datetime.fromtimestamp(entry.get('timestamp', 0)) for entry in history]
            memory_values = [entry.get('memory_percent', 0) for entry in history]
            cpu_values = [entry.get('cpu_percent', 0) for entry in history]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot memory and CPU
            ax.plot(timestamps, memory_values, 'b-', label='Memory %', alpha=0.7)
            ax.plot(timestamps, cpu_values, 'r-', label='CPU %', alpha=0.7)
            
            # Add labels and legend
            ax.set_xlabel('Time')
            ax.set_ylabel('Utilization %')
            ax.set_title('Resource Utilization')
            ax.legend()
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating resource chart: {str(e)}")
            return None
    
    def _generate_error_chart(self, monitor_data: Dict[str, Any], 
                            output_dir: str) -> Optional[str]:
        """
        Generate error distribution chart.
        
        Args:
            monitor_data: Monitor data containing error information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract error data
            error_data = monitor_data.get('error_analysis', {}).get('global_stats', {})
            error_distribution = error_data.get('error_distribution', {})
            
            if not error_distribution:
                return None
                
            # Generate chart
            filename = f"error_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            error_types = list(error_distribution.keys())
            percentages = list(error_distribution.values())
            
            # Create color map
            colors = plt.cm.Set3(range(len(error_types)))
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(percentages, labels=error_types, autopct='%1.1f%%', 
                  colors=colors, shadow=False, startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            plt.title('Error Distribution')
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating error chart: {str(e)}")
            return None
    
    def _extract_recent_executions(self, pipelines: List[Dict[str, Any]], 
                                 monitor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract recent execution data from multiple pipelines.
        
        Args:
            pipelines: List of pipeline data
            monitor_data: Monitor data
            
        Returns:
            List of recent executions across all pipelines
        """
        recent_executions = []
        
        # Extract from monitor data if available
        if monitor_data and 'recent_executions' in monitor_data:
            return monitor_data['recent_executions']
            
        # Otherwise extract from pipeline data
        for pipeline in pipelines:
            pipeline_name = pipeline.get('name', 'Unknown')
            history = pipeline.get('execution_history', [])
            
            for execution in history:
                execution_data = {
                    'pipeline_name': pipeline_name,
                    'success': execution.get('success', False),
                    'duration': round(execution.get('duration', 0), 2),
                    'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    'stages_completed': len(execution.get('stages', {})),
                    'stages_total': len(execution.get('stages', {}))
                }
                recent_executions.append(execution_data)
        
        # Sort by timestamp (most recent first) and limit to 10
        recent_executions.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), reverse=True)
        return recent_executions[:10]
    
    def _generate_pipeline_detail_page(self, pipeline: Dict[str, Any], 
                                     monitor_data: Dict[str, Any],
                                     output_dir: str) -> Optional[str]:
        """
        Generate detail page for a specific pipeline.
        
        Args:
            pipeline: Pipeline data
            monitor_data: Monitor data
            output_dir: Directory to save the detail page
            
        Returns:
            Path to the detail page file, or None if generation failed
        """
        if not self.has_jinja2:
            return None
            
        try:
            pipeline_name = pipeline.get('name', 'Unknown')
            
            # Generate charts for the pipeline
            charts_dir = os.path.join(output_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            # Flow diagram would come from a specific pipeline object, not metrics
            flow_diagram_path = ''
            
            # Performance chart from history
            performance_data = {
                'pipelines': {pipeline_name: pipeline}
            }
            performance_chart_path = self.create_performance_chart(performance_data, 'svg')
            if performance_chart_path:
                # Copy to charts directory and get relative path
                import shutil
                dest_path = os.path.join(charts_dir, os.path.basename(performance_chart_path))
                shutil.copy2(performance_chart_path, dest_path)
                performance_chart_path = os.path.join('charts', os.path.basename(performance_chart_path))
            
            # Stage performance data
            stage_chart_path = self._generate_stage_performance_chart(pipeline, charts_dir)
            if stage_chart_path:
                stage_chart_path = os.path.join('charts', os.path.basename(stage_chart_path))
            
            # Extract pipeline metrics
            total_executions = pipeline.get('total_executions', 0)
            successful_executions = pipeline.get('successful_executions', 0)
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
            avg_duration = pipeline.get('average_duration', 0)
            
            # Get recent executions
            history = list(pipeline.get('execution_history', []))
            history.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            recent_executions = [{
                'id': execution.get('pipeline_id', 'unknown'),
                'success': execution.get('success', False),
                'duration': round(execution.get('duration', 0), 2),
                'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'error_count': execution.get('error_count', 0)
            } for execution in history[:10]]
            
            # Latest execution status
            latest_success = history[0].get('success', False) if history else False
            
            # Stage data
            stages = []
            stage_metrics = {}
            
            # Try to get stage metrics from monitor data
            if monitor_data and 'stage_metrics' in monitor_data:
                for stage_id, stage_data in monitor_data['stage_metrics'].items():
                    if stage_id.startswith(f"{pipeline_name}:"):
                        stage_name = stage_id.split(':', 1)[1]
                        stage_metrics[stage_name] = stage_data
            
            # Extract stage data from execution history if not available in monitor
            if not stage_metrics and history:
                latest_execution = history[0]
                stage_data = latest_execution.get('stages', {})
                
                for stage_name, data in stage_data.items():
                    stage_metrics[stage_name] = {
                        'avg_duration': data.get('duration', 0),
                        'success_rate': 100 if data.get('status') == 'success' else 0,
                        'total_executions': 1
                    }
            
            # Format stage data for template
            for stage_name, data in stage_metrics.items():
                stages.append({
                    'name': stage_name,
                    'avg_duration': round(data.get('avg_duration', 0), 2),
                    'success_rate': round(data.get('success_rate', 0), 1),
                    'total_executions': data.get('total_executions', 0)
                })
            
            # Sort stages by name
            stages.sort(key=lambda x: x['name'])
            
            # Anomaly data
            anomalies = []
            has_anomalies = False
            
            if monitor_data and 'anomalies' in monitor_data:
                pipeline_anomalies = [a for a in monitor_data['anomalies'] 
                                    if a.get('pipeline_name') == pipeline_name]
                
                if pipeline_anomalies:
                    has_anomalies = True
                    anomalies = [{
                        'type': anomaly.get('type', 'unknown'),
                        'value': round(anomaly.get('value', 0), 2),
                        'expected': round(anomaly.get('expected', 0), 2),
                        'z_score': round(anomaly.get('z_score', 0), 2),
                        'timestamp': datetime.fromtimestamp(anomaly.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
                    } for anomaly in pipeline_anomalies]
            
            # Render template
            template = self.template_env.get_template('pipeline_detail.html')
            html = template.render(
                pipeline_name=pipeline_name,
                total_executions=total_executions,
                success_rate=round(success_rate, 1),
                avg_duration=round(avg_duration, 2),
                latest_success=latest_success,
                flow_diagram_path=flow_diagram_path,
                performance_chart_path=performance_chart_path or '',
                stage_chart_path=stage_chart_path or '',
                stages=stages,
                recent_executions=recent_executions,
                has_anomalies=has_anomalies,
                anomalies=anomalies
            )
            
            # Write detail page
            safe_name = pipeline_name.replace(' ', '_').lower()
            detail_path = os.path.join(output_dir, f"{safe_name}_detail.html")
            with open(detail_path, 'w') as f:
                f.write(html)
                
            return detail_path
            
        except Exception as e:
            self.logger.error(f"Error generating pipeline detail page: {str(e)}")
            return None
    
    def _generate_stage_performance_chart(self, pipeline: Dict[str, Any], 
                                       output_dir: str) -> Optional[str]:
        """
        Generate a chart showing performance metrics for each stage.
        
        Args:
            pipeline: Pipeline data
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib:
            return None
            
        try:
            # Extract stage data from most recent execution
            history = list(pipeline.get('execution_history', []))
            if not history:
                return None
                
            # Get the most recent execution with stage data
            for execution in history:
                if 'stages' in execution and execution['stages']:
                    stages = execution['stages']
                    break
            else:
                return None
                
            # Generate chart
            filename = f"stage_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            stage_names = []
            durations = []
            statuses = []
            
            for stage_name, stage_data in stages.items():
                stage_names.append(stage_name)
                durations.append(stage_data.get('duration', 0))
                statuses.append(stage_data.get('status', 'unknown'))
            
            # Sort stages by duration (descending)
            sorted_data = sorted(zip(stage_names, durations, statuses), 
                               key=lambda x: x[1], reverse=True)
            stage_names, durations, statuses = zip(*sorted_data)
            
            # Create colors based on status
            colors = []
            for status in statuses:
                if status == 'success':
                    colors.append('#28a745')  # Green
                elif status == 'failed':
                    colors.append('#dc3545')  # Red
                else:
                    colors.append('#6c757d')  # Gray
            
            # Create horizontal bar chart
            plt.figure(figsize=(10, max(6, len(stage_names) * 0.5)))
            
            y_pos = range(len(stage_names))
            plt.barh(y_pos, durations, color=colors, alpha=0.8)
            
            # Add stage names and duration labels
            for i, (name, duration) in enumerate(zip(stage_names, durations)):
                plt.text(duration + 0.1, i, f"{duration:.2f}s", va='center')
                
            plt.yticks(y_pos, stage_names)
            plt.xlabel('Duration (seconds)')
            plt.title('Stage Performance')
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating stage performance chart: {str(e)}")
            return None
    
    def _generate_resource_chart(self, monitor_data: Dict[str, Any], 
                               output_dir: str) -> Optional[str]:
        """
        Generate resource utilization chart.
        
        Args:
            monitor_data: Monitor data containing resource information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract resource data
            resource_data = monitor_data.get('resource_utilization', {})
            history = resource_data.get('history', [])
            
            if not history:
                return None
                
            # Generate chart
            filename = f"resource_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            timestamps = [datetime.fromtimestamp(entry.get('timestamp', 0)) for entry in history]
            memory_values = [entry.get('memory_percent', 0) for entry in history]
            cpu_values = [entry.get('cpu_percent', 0) for entry in history]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot memory and CPU
            ax.plot(timestamps, memory_values, 'b-', label='Memory %', alpha=0.7)
            ax.plot(timestamps, cpu_values, 'r-', label='CPU %', alpha=0.7)
            
            # Add labels and legend
            ax.set_xlabel('Time')
            ax.set_ylabel('Utilization %')
            ax.set_title('Resource Utilization')
            ax.legend()
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating resource chart: {str(e)}")
            return None
    
    def _generate_error_chart(self, monitor_data: Dict[str, Any], 
                            output_dir: str) -> Optional[str]:
        """
        Generate error distribution chart.
        
        Args:
            monitor_data: Monitor data containing error information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract error data
            error_data = monitor_data.get('error_analysis', {}).get('global_stats', {})
            error_distribution = error_data.get('error_distribution', {})
            
            if not error_distribution:
                return None
                
            # Generate chart
            filename = f"error_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            error_types = list(error_distribution.keys())
            percentages = list(error_distribution.values())
            
            # Create color map
            colors = plt.cm.Set3(range(len(error_types)))
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(percentages, labels=error_types, autopct='%1.1f%%', 
                  colors=colors, shadow=False, startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            plt.title('Error Distribution')
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating error chart: {str(e)}")
            return None
    
    def _extract_recent_executions(self, pipelines: List[Dict[str, Any]], 
                                 monitor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract recent execution data from multiple pipelines.
        
        Args:
            pipelines: List of pipeline data
            monitor_data: Monitor data
            
        Returns:
            List of recent executions across all pipelines
        """
        recent_executions = []
        
        # Extract from monitor data if available
        if monitor_data and 'recent_executions' in monitor_data:
            return monitor_data['recent_executions']
            
        # Otherwise extract from pipeline data
        for pipeline in pipelines:
            pipeline_name = pipeline.get('name', 'Unknown')
            history = pipeline.get('execution_history', [])
            
            for execution in history:
                execution_data = {
                    'pipeline_name': pipeline_name,
                    'success': execution.get('success', False),
                    'duration': round(execution.get('duration', 0), 2),
                    'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    'stages_completed': len(execution.get('stages', {})),
                    'stages_total': len(execution.get('stages', {}))
                }
                recent_executions.append(execution_data)
        
        # Sort by timestamp (most recent first) and limit to 10
        recent_executions.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), reverse=True)
        return recent_executions[:10]
    
    def _generate_pipeline_detail_page(self, pipeline: Dict[str, Any], 
                                     monitor_data: Dict[str, Any],
                                     output_dir: str) -> Optional[str]:
        """
        Generate detail page for a specific pipeline.
        
        Args:
            pipeline: Pipeline data
            monitor_data: Monitor data
            output_dir: Directory to save the detail page
            
        Returns:
            Path to the detail page file, or None if generation failed
        """
        if not self.has_jinja2:
            return None
            
        try:
            pipeline_name = pipeline.get('name', 'Unknown')
            
            # Generate charts for the pipeline
            charts_dir = os.path.join(output_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            # Flow diagram would come from a specific pipeline object, not metrics
            flow_diagram_path = ''
            
            # Performance chart from history
            performance_data = {
                'pipelines': {pipeline_name: pipeline}
            }
            performance_chart_path = self.create_performance_chart(performance_data, 'svg')
            if performance_chart_path:
                # Copy to charts directory and get relative path
                import shutil
                dest_path = os.path.join(charts_dir, os.path.basename(performance_chart_path))
                shutil.copy2(performance_chart_path, dest_path)
                performance_chart_path = os.path.join('charts', os.path.basename(performance_chart_path))
            
            # Stage performance data
            stage_chart_path = self._generate_stage_performance_chart(pipeline, charts_dir)
            if stage_chart_path:
                stage_chart_path = os.path.join('charts', os.path.basename(stage_chart_path))
            
            # Extract pipeline metrics
            total_executions = pipeline.get('total_executions', 0)
            successful_executions = pipeline.get('successful_executions', 0)
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
            avg_duration = pipeline.get('average_duration', 0)
            
            # Get recent executions
            history = list(pipeline.get('execution_history', []))
            history.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            recent_executions = [{
                'id': execution.get('pipeline_id', 'unknown'),
                'success': execution.get('success', False),
                'duration': round(execution.get('duration', 0), 2),
                'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'error_count': execution.get('error_count', 0)
            } for execution in history[:10]]
            
            # Latest execution status
            latest_success = history[0].get('success', False) if history else False
            
            # Stage data
            stages = []
            stage_metrics = {}
            
            # Try to get stage metrics from monitor data
            if monitor_data and 'stage_metrics' in monitor_data:
                for stage_id, stage_data in monitor_data['stage_metrics'].items():
                    if stage_id.startswith(f"{pipeline_name}:"):
                        stage_name = stage_id.split(':', 1)[1]
                        stage_metrics[stage_name] = stage_data
            
            # Extract stage data from execution history if not available in monitor
            if not stage_metrics and history:
                latest_execution = history[0]
                stage_data = latest_execution.get('stages', {})
                
                for stage_name, data in stage_data.items():
                    stage_metrics[stage_name] = {
                        'avg_duration': data.get('duration', 0),
                        'success_rate': 100 if data.get('status') == 'success' else 0,
                        'total_executions': 1
                    }
            
            # Format stage data for template
            for stage_name, data in stage_metrics.items():
                stages.append({
                    'name': stage_name,
                    'avg_duration': round(data.get('avg_duration', 0), 2),
                    'success_rate': round(data.get('success_rate', 0), 1),
                    'total_executions': data.get('total_executions', 0)
                })
            
            # Sort stages by name
            stages.sort(key=lambda x: x['name'])
            
            # Anomaly data
            anomalies = []
            has_anomalies = False
            
            if monitor_data and 'anomalies' in monitor_data:
                pipeline_anomalies = [a for a in monitor_data['anomalies'] 
                                    if a.get('pipeline_name') == pipeline_name]
                
                if pipeline_anomalies:
                    has_anomalies = True
                    anomalies = [{
                        'type': anomaly.get('type', 'unknown'),
                        'value': round(anomaly.get('value', 0), 2),
                        'expected': round(anomaly.get('expected', 0), 2),
                        'z_score': round(anomaly.get('z_score', 0), 2),
                        'timestamp': datetime.fromtimestamp(anomaly.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
                    } for anomaly in pipeline_anomalies]
            
            # Render template
            template = self.template_env.get_template('pipeline_detail.html')
            html = template.render(
                pipeline_name=pipeline_name,
                total_executions=total_executions,
                success_rate=round(success_rate, 1),
                avg_duration=round(avg_duration, 2),
                latest_success=latest_success,
                flow_diagram_path=flow_diagram_path,
                performance_chart_path=performance_chart_path or '',
                stage_chart_path=stage_chart_path or '',
                stages=stages,
                recent_executions=recent_executions,
                has_anomalies=has_anomalies,
                anomalies=anomalies
            )
            
            # Write detail page
            safe_name = pipeline_name.replace(' ', '_').lower()
            detail_path = os.path.join(output_dir, f"{safe_name}_detail.html")
            with open(detail_path, 'w') as f:
                f.write(html)
                
            return detail_path
            
        except Exception as e:
            self.logger.error(f"Error generating pipeline detail page: {str(e)}")
            return None
    
    def _generate_stage_performance_chart(self, pipeline: Dict[str, Any], 
                                       output_dir: str) -> Optional[str]:
        """
        Generate a chart showing performance metrics for each stage.
        
        Args:
            pipeline: Pipeline data
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib:
            return None
            
        try:
            # Extract stage data from most recent execution
            history = list(pipeline.get('execution_history', []))
            if not history:
                return None
                
            # Get the most recent execution with stage data
            for execution in history:
                if 'stages' in execution and execution['stages']:
                    stages = execution['stages']
                    break
            else:
                return None
                
            # Generate chart
            filename = f"stage_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            stage_names = []
            durations = []
            statuses = []
            
            for stage_name, stage_data in stages.items():
                stage_names.append(stage_name)
                durations.append(stage_data.get('duration', 0))
                statuses.append(stage_data.get('status', 'unknown'))
            
            # Sort stages by duration (descending)
            sorted_data = sorted(zip(stage_names, durations, statuses), 
                               key=lambda x: x[1], reverse=True)
            stage_names, durations, statuses = zip(*sorted_data)
            
            # Create colors based on status
            colors = []
            for status in statuses:
                if status == 'success':
                    colors.append('#28a745')  # Green
                elif status == 'failed':
                    colors.append('#dc3545')  # Red
                else:
                    colors.append('#6c757d')  # Gray
            
            # Create horizontal bar chart
            plt.figure(figsize=(10, max(6, len(stage_names) * 0.5)))
            
            y_pos = range(len(stage_names))
            plt.barh(y_pos, durations, color=colors, alpha=0.8)
            
            # Add stage names and duration labels
            for i, (name, duration) in enumerate(zip(stage_names, durations)):
                plt.text(duration + 0.1, i, f"{duration:.2f}s", va='center')
                
            plt.yticks(y_pos, stage_names)
            plt.xlabel('Duration (seconds)')
            plt.title('Stage Performance')
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating stage performance chart: {str(e)}")
            return None
    
    def _generate_resource_chart(self, monitor_data: Dict[str, Any], 
                               output_dir: str) -> Optional[str]:
        """
        Generate resource utilization chart.
        
        Args:
            monitor_data: Monitor data containing resource information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract resource data
            resource_data = monitor_data.get('resource_utilization', {})
            history = resource_data.get('history', [])
            
            if not history:
                return None
                
            # Generate chart
            filename = f"resource_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            timestamps = [datetime.fromtimestamp(entry.get('timestamp', 0)) for entry in history]
            memory_values = [entry.get('memory_percent', 0) for entry in history]
            cpu_values = [entry.get('cpu_percent', 0) for entry in history]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot memory and CPU
            ax.plot(timestamps, memory_values, 'b-', label='Memory %', alpha=0.7)
            ax.plot(timestamps, cpu_values, 'r-', label='CPU %', alpha=0.7)
            
            # Add labels and legend
            ax.set_xlabel('Time')
            ax.set_ylabel('Utilization %')
            ax.set_title('Resource Utilization')
            ax.legend()
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating resource chart: {str(e)}")
            return None
    
    def _generate_error_chart(self, monitor_data: Dict[str, Any], 
                            output_dir: str) -> Optional[str]:
        """
        Generate error distribution chart.
        
        Args:
            monitor_data: Monitor data containing error information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract error data
            error_data = monitor_data.get('error_analysis', {}).get('global_stats', {})
            error_distribution = error_data.get('error_distribution', {})
            
            if not error_distribution:
                return None
                
            # Generate chart
            filename = f"error_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            error_types = list(error_distribution.keys())
            percentages = list(error_distribution.values())
            
            # Create color map
            colors = plt.cm.Set3(range(len(error_types)))
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(percentages, labels=error_types, autopct='%1.1f%%', 
                  colors=colors, shadow=False, startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            plt.title('Error Distribution')
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating error chart: {str(e)}")
            return None
    
    def _extract_recent_executions(self, pipelines: List[Dict[str, Any]], 
                                 monitor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract recent execution data from multiple pipelines.
        
        Args:
            pipelines: List of pipeline data
            monitor_data: Monitor data
            
        Returns:
            List of recent executions across all pipelines
        """
        recent_executions = []
        
        # Extract from monitor data if available
        if monitor_data and 'recent_executions' in monitor_data:
            return monitor_data['recent_executions']
            
        # Otherwise extract from pipeline data
        for pipeline in pipelines:
            pipeline_name = pipeline.get('name', 'Unknown')
            history = pipeline.get('execution_history', [])
            
            for execution in history:
                execution_data = {
                    'pipeline_name': pipeline_name,
                    'success': execution.get('success', False),
                    'duration': round(execution.get('duration', 0), 2),
                    'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    'stages_completed': len(execution.get('stages', {})),
                    'stages_total': len(execution.get('stages', {}))
                }
                recent_executions.append(execution_data)
        
        # Sort by timestamp (most recent first) and limit to 10
        recent_executions.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), reverse=True)
        return recent_executions[:10]
    
    def _generate_pipeline_detail_page(self, pipeline: Dict[str, Any], 
                                     monitor_data: Dict[str, Any],
                                     output_dir: str) -> Optional[str]:
        """
        Generate detail page for a specific pipeline.
        
        Args:
            pipeline: Pipeline data
            monitor_data: Monitor data
            output_dir: Directory to save the detail page
            
        Returns:
            Path to the detail page file, or None if generation failed
        """
        if not self.has_jinja2:
            return None
            
        try:
            pipeline_name = pipeline.get('name', 'Unknown')
            
            # Generate charts for the pipeline
            charts_dir = os.path.join(output_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            # Flow diagram would come from a specific pipeline object, not metrics
            flow_diagram_path = ''
            
            # Performance chart from history
            performance_data = {
                'pipelines': {pipeline_name: pipeline}
            }
            performance_chart_path = self.create_performance_chart(performance_data, 'svg')
            if performance_chart_path:
                # Copy to charts directory and get relative path
                import shutil
                dest_path = os.path.join(charts_dir, os.path.basename(performance_chart_path))
                shutil.copy2(performance_chart_path, dest_path)
                performance_chart_path = os.path.join('charts', os.path.basename(performance_chart_path))
            
            # Stage performance data
            stage_chart_path = self._generate_stage_performance_chart(pipeline, charts_dir)
            if stage_chart_path:
                stage_chart_path = os.path.join('charts', os.path.basename(stage_chart_path))
            
            # Extract pipeline metrics
            total_executions = pipeline.get('total_executions', 0)
            successful_executions = pipeline.get('successful_executions', 0)
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
            avg_duration = pipeline.get('average_duration', 0)
            
            # Get recent executions
            history = list(pipeline.get('execution_history', []))
            history.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            recent_executions = [{
                'id': execution.get('pipeline_id', 'unknown'),
                'success': execution.get('success', False),
                'duration': round(execution.get('duration', 0), 2),
                'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'error_count': execution.get('error_count', 0)
            } for execution in history[:10]]
            
            # Latest execution status
            latest_success = history[0].get('success', False) if history else False
            
            # Stage data
            stages = []
            stage_metrics = {}
            
            # Try to get stage metrics from monitor data
            if monitor_data and 'stage_metrics' in monitor_data:
                for stage_id, stage_data in monitor_data['stage_metrics'].items():
                    if stage_id.startswith(f"{pipeline_name}:"):
                        stage_name = stage_id.split(':', 1)[1]
                        stage_metrics[stage_name] = stage_data
            
            # Extract stage data from execution history if not available in monitor
            if not stage_metrics and history:
                latest_execution = history[0]
                stage_data = latest_execution.get('stages', {})
                
                for stage_name, data in stage_data.items():
                    stage_metrics[stage_name] = {
                        'avg_duration': data.get('duration', 0),
                        'success_rate': 100 if data.get('status') == 'success' else 0,
                        'total_executions': 1
                    }
            
            # Format stage data for template
            for stage_name, data in stage_metrics.items():
                stages.append({
                    'name': stage_name,
                    'avg_duration': round(data.get('avg_duration', 0), 2),
                    'success_rate': round(data.get('success_rate', 0), 1),
                    'total_executions': data.get('total_executions', 0)
                })
            
            # Sort stages by name
            stages.sort(key=lambda x: x['name'])
            
            # Anomaly data
            anomalies = []
            has_anomalies = False
            
            if monitor_data and 'anomalies' in monitor_data:
                pipeline_anomalies = [a for a in monitor_data['anomalies'] 
                                    if a.get('pipeline_name') == pipeline_name]
                
                if pipeline_anomalies:
                    has_anomalies = True
                    anomalies = [{
                        'type': anomaly.get('type', 'unknown'),
                        'value': round(anomaly.get('value', 0), 2),
                        'expected': round(anomaly.get('expected', 0), 2),
                        'z_score': round(anomaly.get('z_score', 0), 2),
                        'timestamp': datetime.fromtimestamp(anomaly.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
                    } for anomaly in pipeline_anomalies]
            
            # Render template
            template = self.template_env.get_template('pipeline_detail.html')
            html = template.render(
                pipeline_name=pipeline_name,
                total_executions=total_executions,
                success_rate=round(success_rate, 1),
                avg_duration=round(avg_duration, 2),
                latest_success=latest_success,
                flow_diagram_path=flow_diagram_path,
                performance_chart_path=performance_chart_path or '',
                stage_chart_path=stage_chart_path or '',
                stages=stages,
                recent_executions=recent_executions,
                has_anomalies=has_anomalies,
                anomalies=anomalies
            )
            
            # Write detail page
            safe_name = pipeline_name.replace(' ', '_').lower()
            detail_path = os.path.join(output_dir, f"{safe_name}_detail.html")
            with open(detail_path, 'w') as f:
                f.write(html)
                
            return detail_path
            
        except Exception as e:
            self.logger.error(f"Error generating pipeline detail page: {str(e)}")
            return None
    
    def _generate_stage_performance_chart(self, pipeline: Dict[str, Any], 
                                       output_dir: str) -> Optional[str]:
        """
        Generate a chart showing performance metrics for each stage.
        
        Args:
            pipeline: Pipeline data
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib:
            return None
            
        try:
            # Extract stage data from most recent execution
            history = list(pipeline.get('execution_history', []))
            if not history:
                return None
                
            # Get the most recent execution with stage data
            for execution in history:
                if 'stages' in execution and execution['stages']:
                    stages = execution['stages']
                    break
            else:
                return None
                
            # Generate chart
            filename = f"stage_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            stage_names = []
            durations = []
            statuses = []
            
            for stage_name, stage_data in stages.items():
                stage_names.append(stage_name)
                durations.append(stage_data.get('duration', 0))
                statuses.append(stage_data.get('status', 'unknown'))
            
            # Sort stages by duration (descending)
            sorted_data = sorted(zip(stage_names, durations, statuses), 
                               key=lambda x: x[1], reverse=True)
            stage_names, durations, statuses = zip(*sorted_data)
            
            # Create colors based on status
            colors = []
            for status in statuses:
                if status == 'success':
                    colors.append('#28a745')  # Green
                elif status == 'failed':
                    colors.append('#dc3545')  # Red
                else:
                    colors.append('#6c757d')  # Gray
            
            # Create horizontal bar chart
            plt.figure(figsize=(10, max(6, len(stage_names) * 0.5)))
            
            y_pos = range(len(stage_names))
            plt.barh(y_pos, durations, color=colors, alpha=0.8)
            
            # Add stage names and duration labels
            for i, (name, duration) in enumerate(zip(stage_names, durations)):
                plt.text(duration + 0.1, i, f"{duration:.2f}s", va='center')
                
            plt.yticks(y_pos, stage_names)
            plt.xlabel('Duration (seconds)')
            plt.title('Stage Performance')
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating stage performance chart: {str(e)}")
            return None
    
    def _generate_resource_chart(self, monitor_data: Dict[str, Any], 
                               output_dir: str) -> Optional[str]:
        """
        Generate resource utilization chart.
        
        Args:
            monitor_data: Monitor data containing resource information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract resource data
            resource_data = monitor_data.get('resource_utilization', {})
            history = resource_data.get('history', [])
            
            if not history:
                return None
                
            # Generate chart
            filename = f"resource_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            timestamps = [datetime.fromtimestamp(entry.get('timestamp', 0)) for entry in history]
            memory_values = [entry.get('memory_percent', 0) for entry in history]
            cpu_values = [entry.get('cpu_percent', 0) for entry in history]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot memory and CPU
            ax.plot(timestamps, memory_values, 'b-', label='Memory %', alpha=0.7)
            ax.plot(timestamps, cpu_values, 'r-', label='CPU %', alpha=0.7)
            
            # Add labels and legend
            ax.set_xlabel('Time')
            ax.set_ylabel('Utilization %')
            ax.set_title('Resource Utilization')
            ax.legend()
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating resource chart: {str(e)}")
            return None
    
    def _generate_error_chart(self, monitor_data: Dict[str, Any], 
                            output_dir: str) -> Optional[str]:
        """
        Generate error distribution chart.
        
        Args:
            monitor_data: Monitor data containing error information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract error data
            error_data = monitor_data.get('error_analysis', {}).get('global_stats', {})
            error_distribution = error_data.get('error_distribution', {})
            
            if not error_distribution:
                return None
                
            # Generate chart
            filename = f"error_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            error_types = list(error_distribution.keys())
            percentages = list(error_distribution.values())
            
            # Create color map
            colors = plt.cm.Set3(range(len(error_types)))
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(percentages, labels=error_types, autopct='%1.1f%%', 
                  colors=colors, shadow=False, startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            plt.title('Error Distribution')
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating error chart: {str(e)}")
            return None
    
    def _extract_recent_executions(self, pipelines: List[Dict[str, Any]], 
                                 monitor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract recent execution data from multiple pipelines.
        
        Args:
            pipelines: List of pipeline data
            monitor_data: Monitor data
            
        Returns:
            List of recent executions across all pipelines
        """
        recent_executions = []
        
        # Extract from monitor data if available
        if monitor_data and 'recent_executions' in monitor_data:
            return monitor_data['recent_executions']
            
        # Otherwise extract from pipeline data
        for pipeline in pipelines:
            pipeline_name = pipeline.get('name', 'Unknown')
            history = pipeline.get('execution_history', [])
            
            for execution in history:
                execution_data = {
                    'pipeline_name': pipeline_name,
                    'success': execution.get('success', False),
                    'duration': round(execution.get('duration', 0), 2),
                    'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    'stages_completed': len(execution.get('stages', {})),
                    'stages_total': len(execution.get('stages', {}))
                }
                recent_executions.append(execution_data)
        
        # Sort by timestamp (most recent first) and limit to 10
        recent_executions.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), reverse=True)
        return recent_executions[:10]
    
    def _generate_pipeline_detail_page(self, pipeline: Dict[str, Any], 
                                     monitor_data: Dict[str, Any],
                                     output_dir: str) -> Optional[str]:
        """
        Generate detail page for a specific pipeline.
        
        Args:
            pipeline: Pipeline data
            monitor_data: Monitor data
            output_dir: Directory to save the detail page
            
        Returns:
            Path to the detail page file, or None if generation failed
        """
        if not self.has_jinja2:
            return None
            
        try:
            pipeline_name = pipeline.get('name', 'Unknown')
            
            # Generate charts for the pipeline
            charts_dir = os.path.join(output_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            # Flow diagram would come from a specific pipeline object, not metrics
            flow_diagram_path = ''
            
            # Performance chart from history
            performance_data = {
                'pipelines': {pipeline_name: pipeline}
            }
            performance_chart_path = self.create_performance_chart(performance_data, 'svg')
            if performance_chart_path:
                # Copy to charts directory and get relative path
                import shutil
                dest_path = os.path.join(charts_dir, os.path.basename(performance_chart_path))
                shutil.copy2(performance_chart_path, dest_path)
                performance_chart_path = os.path.join('charts', os.path.basename(performance_chart_path))
            
            # Stage performance data
            stage_chart_path = self._generate_stage_performance_chart(pipeline, charts_dir)
            if stage_chart_path:
                stage_chart_path = os.path.join('charts', os.path.basename(stage_chart_path))
            
            # Extract pipeline metrics
            total_executions = pipeline.get('total_executions', 0)
            successful_executions = pipeline.get('successful_executions', 0)
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
            avg_duration = pipeline.get('average_duration', 0)
            
            # Get recent executions
            history = list(pipeline.get('execution_history', []))
            history.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            recent_executions = [{
                'id': execution.get('pipeline_id', 'unknown'),
                'success': execution.get('success', False),
                'duration': round(execution.get('duration', 0), 2),
                'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'error_count': execution.get('error_count', 0)
            } for execution in history[:10]]
            
            # Latest execution status
            latest_success = history[0].get('success', False) if history else False
            
            # Stage data
            stages = []
            stage_metrics = {}
            
            # Try to get stage metrics from monitor data
            if monitor_data and 'stage_metrics' in monitor_data:
                for stage_id, stage_data in monitor_data['stage_metrics'].items():
                    if stage_id.startswith(f"{pipeline_name}:"):
                        stage_name = stage_id.split(':', 1)[1]
                        stage_metrics[stage_name] = stage_data
            
            # Extract stage data from execution history if not available in monitor
            if not stage_metrics and history:
                latest_execution = history[0]
                stage_data = latest_execution.get('stages', {})
                
                for stage_name, data in stage_data.items():
                    stage_metrics[stage_name] = {
                        'avg_duration': data.get('duration', 0),
                        'success_rate': 100 if data.get('status') == 'success' else 0,
                        'total_executions': 1
                    }
            
            # Format stage data for template
            for stage_name, data in stage_metrics.items():
                stages.append({
                    'name': stage_name,
                    'avg_duration': round(data.get('avg_duration', 0), 2),
                    'success_rate': round(data.get('success_rate', 0), 1),
                    'total_executions': data.get('total_executions', 0)
                })
            
            # Sort stages by name
            stages.sort(key=lambda x: x['name'])
            
            # Anomaly data
            anomalies = []
            has_anomalies = False
            
            if monitor_data and 'anomalies' in monitor_data:
                pipeline_anomalies = [a for a in monitor_data['anomalies'] 
                                    if a.get('pipeline_name') == pipeline_name]
                
                if pipeline_anomalies:
                    has_anomalies = True
                    anomalies = [{
                        'type': anomaly.get('type', 'unknown'),
                        'value': round(anomaly.get('value', 0), 2),
                        'expected': round(anomaly.get('expected', 0), 2),
                        'z_score': round(anomaly.get('z_score', 0), 2),
                        'timestamp': datetime.fromtimestamp(anomaly.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
                    } for anomaly in pipeline_anomalies]
            
            # Render template
            template = self.template_env.get_template('pipeline_detail.html')
            html = template.render(
                pipeline_name=pipeline_name,
                total_executions=total_executions,
                success_rate=round(success_rate, 1),
                avg_duration=round(avg_duration, 2),
                latest_success=latest_success,
                flow_diagram_path=flow_diagram_path,
                performance_chart_path=performance_chart_path or '',
                stage_chart_path=stage_chart_path or '',
                stages=stages,
                recent_executions=recent_executions,
                has_anomalies=has_anomalies,
                anomalies=anomalies
            )
            
            # Write detail page
            safe_name = pipeline_name.replace(' ', '_').lower()
            detail_path = os.path.join(output_dir, f"{safe_name}_detail.html")
            with open(detail_path, 'w') as f:
                f.write(html)
                
            return detail_path
            
        except Exception as e:
            self.logger.error(f"Error generating pipeline detail page: {str(e)}")
            return None
    
    def _generate_stage_performance_chart(self, pipeline: Dict[str, Any], 
                                       output_dir: str) -> Optional[str]:
        """
        Generate a chart showing performance metrics for each stage.
        
        Args:
            pipeline: Pipeline data
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib:
            return None
            
        try:
            # Extract stage data from most recent execution
            history = list(pipeline.get('execution_history', []))
            if not history:
                return None
                
            # Get the most recent execution with stage data
            for execution in history:
                if 'stages' in execution and execution['stages']:
                    stages = execution['stages']
                    break
            else:
                return None
                
            # Generate chart
            filename = f"stage_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            stage_names = []
            durations = []
            statuses = []
            
            for stage_name, stage_data in stages.items():
                stage_names.append(stage_name)
                durations.append(stage_data.get('duration', 0))
                statuses.append(stage_data.get('status', 'unknown'))
            
            # Sort stages by duration (descending)
            sorted_data = sorted(zip(stage_names, durations, statuses), 
                               key=lambda x: x[1], reverse=True)
            stage_names, durations, statuses = zip(*sorted_data)
            
            # Create colors based on status
            colors = []
            for status in statuses:
                if status == 'success':
                    colors.append('#28a745')  # Green
                elif status == 'failed':
                    colors.append('#dc3545')  # Red
                else:
                    colors.append('#6c757d')  # Gray
            
            # Create horizontal bar chart
            plt.figure(figsize=(10, max(6, len(stage_names) * 0.5)))
            
            y_pos = range(len(stage_names))
            plt.barh(y_pos, durations, color=colors, alpha=0.8)
            
            # Add stage names and duration labels
            for i, (name, duration) in enumerate(zip(stage_names, durations)):
                plt.text(duration + 0.1, i, f"{duration:.2f}s", va='center')
                
            plt.yticks(y_pos, stage_names)
            plt.xlabel('Duration (seconds)')
            plt.title('Stage Performance')
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating stage performance chart: {str(e)}")
            return None
    
    def _generate_resource_chart(self, monitor_data: Dict[str, Any], 
                               output_dir: str) -> Optional[str]:
        """
        Generate resource utilization chart.
        
        Args:
            monitor_data: Monitor data containing resource information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract resource data
            resource_data = monitor_data.get('resource_utilization', {})
            history = resource_data.get('history', [])
            
            if not history:
                return None
                
            # Generate chart
            filename = f"resource_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            timestamps = [datetime.fromtimestamp(entry.get('timestamp', 0)) for entry in history]
            memory_values = [entry.get('memory_percent', 0) for entry in history]
            cpu_values = [entry.get('cpu_percent', 0) for entry in history]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot memory and CPU
            ax.plot(timestamps, memory_values, 'b-', label='Memory %', alpha=0.7)
            ax.plot(timestamps, cpu_values, 'r-', label='CPU %', alpha=0.7)
            
            # Add labels and legend
            ax.set_xlabel('Time')
            ax.set_ylabel('Utilization %')
            ax.set_title('Resource Utilization')
            ax.legend()
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating resource chart: {str(e)}")
            return None
    
    def _generate_error_chart(self, monitor_data: Dict[str, Any], 
                            output_dir: str) -> Optional[str]:
        """
        Generate error distribution chart.
        
        Args:
            monitor_data: Monitor data containing error information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract error data
            error_data = monitor_data.get('error_analysis', {}).get('global_stats', {})
            error_distribution = error_data.get('error_distribution', {})
            
            if not error_distribution:
                return None
                
            # Generate chart
            filename = f"error_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            error_types = list(error_distribution.keys())
            percentages = list(error_distribution.values())
            
            # Create color map
            colors = plt.cm.Set3(range(len(error_types)))
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(percentages, labels=error_types, autopct='%1.1f%%', 
                  colors=colors, shadow=False, startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            plt.title('Error Distribution')
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating error chart: {str(e)}")
            return None
    
    def _extract_recent_executions(self, pipelines: List[Dict[str, Any]], 
                                 monitor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract recent execution data from multiple pipelines.
        
        Args:
            pipelines: List of pipeline data
            monitor_data: Monitor data
            
        Returns:
            List of recent executions across all pipelines
        """
        recent_executions = []
        
        # Extract from monitor data if available
        if monitor_data and 'recent_executions' in monitor_data:
            return monitor_data['recent_executions']
            
        # Otherwise extract from pipeline data
        for pipeline in pipelines:
            pipeline_name = pipeline.get('name', 'Unknown')
            history = pipeline.get('execution_history', [])
            
            for execution in history:
                execution_data = {
                    'pipeline_name': pipeline_name,
                    'success': execution.get('success', False),
                    'duration': round(execution.get('duration', 0), 2),
                    'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    'stages_completed': len(execution.get('stages', {})),
                    'stages_total': len(execution.get('stages', {}))
                }
                recent_executions.append(execution_data)
        
        # Sort by timestamp (most recent first) and limit to 10
        recent_executions.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), reverse=True)
        return recent_executions[:10]
    
    def _generate_pipeline_detail_page(self, pipeline: Dict[str, Any], 
                                     monitor_data: Dict[str, Any],
                                     output_dir: str) -> Optional[str]:
        """
        Generate detail page for a specific pipeline.
        
        Args:
            pipeline: Pipeline data
            monitor_data: Monitor data
            output_dir: Directory to save the detail page
            
        Returns:
            Path to the detail page file, or None if generation failed
        """
        if not self.has_jinja2:
            return None
            
        try:
            pipeline_name = pipeline.get('name', 'Unknown')
            
            # Generate charts for the pipeline
            charts_dir = os.path.join(output_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            # Flow diagram would come from a specific pipeline object, not metrics
            flow_diagram_path = ''
            
            # Performance chart from history
            performance_data = {
                'pipelines': {pipeline_name: pipeline}
            }
            performance_chart_path = self.create_performance_chart(performance_data, 'svg')
            if performance_chart_path:
                # Copy to charts directory and get relative path
                import shutil
                dest_path = os.path.join(charts_dir, os.path.basename(performance_chart_path))
                shutil.copy2(performance_chart_path, dest_path)
                performance_chart_path = os.path.join('charts', os.path.basename(performance_chart_path))
            
            # Stage performance data
            stage_chart_path = self._generate_stage_performance_chart(pipeline, charts_dir)
            if stage_chart_path:
                stage_chart_path = os.path.join('charts', os.path.basename(stage_chart_path))
            
            # Extract pipeline metrics
            total_executions = pipeline.get('total_executions', 0)
            successful_executions = pipeline.get('successful_executions', 0)
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
            avg_duration = pipeline.get('average_duration', 0)
            
            # Get recent executions
            history = list(pipeline.get('execution_history', []))
            history.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            recent_executions = [{
                'id': execution.get('pipeline_id', 'unknown'),
                'success': execution.get('success', False),
                'duration': round(execution.get('duration', 0), 2),
                'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'error_count': execution.get('error_count', 0)
            } for execution in history[:10]]
            
            # Latest execution status
            latest_success = history[0].get('success', False) if history else False
            
            # Stage data
            stages = []
            stage_metrics = {}
            
            # Try to get stage metrics from monitor data
            if monitor_data and 'stage_metrics' in monitor_data:
                for stage_id, stage_data in monitor_data['stage_metrics'].items():
                    if stage_id.startswith(f"{pipeline_name}:"):
                        stage_name = stage_id.split(':', 1)[1]
                        stage_metrics[stage_name] = stage_data
            
            # Extract stage data from execution history if not available in monitor
            if not stage_metrics and history:
                latest_execution = history[0]
                stage_data = latest_execution.get('stages', {})
                
                for stage_name, data in stage_data.items():
                    stage_metrics[stage_name] = {
                        'avg_duration': data.get('duration', 0),
                        'success_rate': 100 if data.get('status') == 'success' else 0,
                        'total_executions': 1
                    }
            
            # Format stage data for template
            for stage_name, data in stage_metrics.items():
                stages.append({
                    'name': stage_name,
                    'avg_duration': round(data.get('avg_duration', 0), 2),
                    'success_rate': round(data.get('success_rate', 0), 1),
                    'total_executions': data.get('total_executions', 0)
                })
            
            # Sort stages by name
            stages.sort(key=lambda x: x['name'])
            
            # Anomaly data
            anomalies = []
            has_anomalies = False
            
            if monitor_data and 'anomalies' in monitor_data:
                pipeline_anomalies = [a for a in monitor_data['anomalies'] 
                                    if a.get('pipeline_name') == pipeline_name]
                
                if pipeline_anomalies:
                    has_anomalies = True
                    anomalies = [{
                        'type': anomaly.get('type', 'unknown'),
                        'value': round(anomaly.get('value', 0), 2),
                        'expected': round(anomaly.get('expected', 0), 2),
                        'z_score': round(anomaly.get('z_score', 0), 2),
                        'timestamp': datetime.fromtimestamp(anomaly.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
                    } for anomaly in pipeline_anomalies]
            
            # Render template
            template = self.template_env.get_template('pipeline_detail.html')
            html = template.render(
                pipeline_name=pipeline_name,
                total_executions=total_executions,
                success_rate=round(success_rate, 1),
                avg_duration=round(avg_duration, 2),
                latest_success=latest_success,
                flow_diagram_path=flow_diagram_path,
                performance_chart_path=performance_chart_path or '',
                stage_chart_path=stage_chart_path or '',
                stages=stages,
                recent_executions=recent_executions,
                has_anomalies=has_anomalies,
                anomalies=anomalies
            )
            
            # Write detail page
            safe_name = pipeline_name.replace(' ', '_').lower()
            detail_path = os.path.join(output_dir, f"{safe_name}_detail.html")
            with open(detail_path, 'w') as f:
                f.write(html)
                
            return detail_path
            
        except Exception as e:
            self.logger.error(f"Error generating pipeline detail page: {str(e)}")
            return None
    
    def _generate_stage_performance_chart(self, pipeline: Dict[str, Any], 
                                       output_dir: str) -> Optional[str]:
        """
        Generate a chart showing performance metrics for each stage.
        
        Args:
            pipeline: Pipeline data
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib:
            return None
            
        try:
            # Extract stage data from most recent execution
            history = list(pipeline.get('execution_history', []))
            if not history:
                return None
                
            # Get the most recent execution with stage data
            for execution in history:
                if 'stages' in execution and execution['stages']:
                    stages = execution['stages']
                    break
            else:
                return None
                
            # Generate chart
            filename = f"stage_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            stage_names = []
            durations = []
            statuses = []
            
            for stage_name, stage_data in stages.items():
                stage_names.append(stage_name)
                durations.append(stage_data.get('duration', 0))
                statuses.append(stage_data.get('status', 'unknown'))
            
            # Sort stages by duration (descending)
            sorted_data = sorted(zip(stage_names, durations, statuses), 
                               key=lambda x: x[1], reverse=True)
            stage_names, durations, statuses = zip(*sorted_data)
            
            # Create colors based on status
            colors = []
            for status in statuses:
                if status == 'success':
                    colors.append('#28a745')  # Green
                elif status == 'failed':
                    colors.append('#dc3545')  # Red
                else:
                    colors.append('#6c757d')  # Gray
            
            # Create horizontal bar chart
            plt.figure(figsize=(10, max(6, len(stage_names) * 0.5)))
            
            y_pos = range(len(stage_names))
            plt.barh(y_pos, durations, color=colors, alpha=0.8)
            
            # Add stage names and duration labels
            for i, (name, duration) in enumerate(zip(stage_names, durations)):
                plt.text(duration + 0.1, i, f"{duration:.2f}s", va='center')
                
            plt.yticks(y_pos, stage_names)
            plt.xlabel('Duration (seconds)')
            plt.title('Stage Performance')
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating stage performance chart: {str(e)}")
            return None
    
    def _generate_resource_chart(self, monitor_data: Dict[str, Any], 
                               output_dir: str) -> Optional[str]:
        """
        Generate resource utilization chart.
        
        Args:
            monitor_data: Monitor data containing resource information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract resource data
            resource_data = monitor_data.get('resource_utilization', {})
            history = resource_data.get('history', [])
            
            if not history:
                return None
                
            # Generate chart
            filename = f"resource_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            timestamps = [datetime.fromtimestamp(entry.get('timestamp', 0)) for entry in history]
            memory_values = [entry.get('memory_percent', 0) for entry in history]
            cpu_values = [entry.get('cpu_percent', 0) for entry in history]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot memory and CPU
            ax.plot(timestamps, memory_values, 'b-', label='Memory %', alpha=0.7)
            ax.plot(timestamps, cpu_values, 'r-', label='CPU %', alpha=0.7)
            
            # Add labels and legend
            ax.set_xlabel('Time')
            ax.set_ylabel('Utilization %')
            ax.set_title('Resource Utilization')
            ax.legend()
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating resource chart: {str(e)}")
            return None
    
    def _generate_error_chart(self, monitor_data: Dict[str, Any], 
                            output_dir: str) -> Optional[str]:
        """
        Generate error distribution chart.
        
        Args:
            monitor_data: Monitor data containing error information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract error data
            error_data = monitor_data.get('error_analysis', {}).get('global_stats', {})
            error_distribution = error_data.get('error_distribution', {})
            
            if not error_distribution:
                return None
                
            # Generate chart
            filename = f"error_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            error_types = list(error_distribution.keys())
            percentages = list(error_distribution.values())
            
            # Create color map
            colors = plt.cm.Set3(range(len(error_types)))
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(percentages, labels=error_types, autopct='%1.1f%%', 
                  colors=colors, shadow=False, startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            plt.title('Error Distribution')
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating error chart: {str(e)}")
            return None
    
    def _extract_recent_executions(self, pipelines: List[Dict[str, Any]], 
                                 monitor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract recent execution data from multiple pipelines.
        
        Args:
            pipelines: List of pipeline data
            monitor_data: Monitor data
            
        Returns:
            List of recent executions across all pipelines
        """
        recent_executions = []
        
        # Extract from monitor data if available
        if monitor_data and 'recent_executions' in monitor_data:
            return monitor_data['recent_executions']
            
        # Otherwise extract from pipeline data
        for pipeline in pipelines:
            pipeline_name = pipeline.get('name', 'Unknown')
            history = pipeline.get('execution_history', [])
            
            for execution in history:
                execution_data = {
                    'pipeline_name': pipeline_name,
                    'success': execution.get('success', False),
                    'duration': round(execution.get('duration', 0), 2),
                    'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    'stages_completed': len(execution.get('stages', {})),
                    'stages_total': len(execution.get('stages', {}))
                }
                recent_executions.append(execution_data)
        
        # Sort by timestamp (most recent first) and limit to 10
        recent_executions.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), reverse=True)
        return recent_executions[:10]
    
    def _generate_pipeline_detail_page(self, pipeline: Dict[str, Any], 
                                     monitor_data: Dict[str, Any],
                                     output_dir: str) -> Optional[str]:
        """
        Generate detail page for a specific pipeline.
        
        Args:
            pipeline: Pipeline data
            monitor_data: Monitor data
            output_dir: Directory to save the detail page
            
        Returns:
            Path to the detail page file, or None if generation failed
        """
        if not self.has_jinja2:
            return None
            
        try:
            pipeline_name = pipeline.get('name', 'Unknown')
            
            # Generate charts for the pipeline
            charts_dir = os.path.join(output_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            # Flow diagram would come from a specific pipeline object, not metrics
            flow_diagram_path = ''
            
            # Performance chart from history
            performance_data = {
                'pipelines': {pipeline_name: pipeline}
            }
            performance_chart_path = self.create_performance_chart(performance_data, 'svg')
            if performance_chart_path:
                # Copy to charts directory and get relative path
                import shutil
                dest_path = os.path.join(charts_dir, os.path.basename(performance_chart_path))
                shutil.copy2(performance_chart_path, dest_path)
                performance_chart_path = os.path.join('charts', os.path.basename(performance_chart_path))
            
            # Stage performance data
            stage_chart_path = self._generate_stage_performance_chart(pipeline, charts_dir)
            if stage_chart_path:
                stage_chart_path = os.path.join('charts', os.path.basename(stage_chart_path))
            
            # Extract pipeline metrics
            total_executions = pipeline.get('total_executions', 0)
            successful_executions = pipeline.get('successful_executions', 0)
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
            avg_duration = pipeline.get('average_duration', 0)
            
            # Get recent executions
            history = list(pipeline.get('execution_history', []))
            history.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            recent_executions = [{
                'id': execution.get('pipeline_id', 'unknown'),
                'success': execution.get('success', False),
                'duration': round(execution.get('duration', 0), 2),
                'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'error_count': execution.get('error_count', 0)
            } for execution in history[:10]]
            
            # Latest execution status
            latest_success = history[0].get('success', False) if history else False
            
            # Stage data
            stages = []
            stage_metrics = {}
            
            # Try to get stage metrics from monitor data
            if monitor_data and 'stage_metrics' in monitor_data:
                for stage_id, stage_data in monitor_data['stage_metrics'].items():
                    if stage_id.startswith(f"{pipeline_name}:"):
                        stage_name = stage_id.split(':', 1)[1]
                        stage_metrics[stage_name] = stage_data
            
            # Extract stage data from execution history if not available in monitor
            if not stage_metrics and history:
                latest_execution = history[0]
                stage_data = latest_execution.get('stages', {})
                
                for stage_name, data in stage_data.items():
                    stage_metrics[stage_name] = {
                        'avg_duration': data.get('duration', 0),
                        'success_rate': 100 if data.get('status') == 'success' else 0,
                        'total_executions': 1
                    }
            
            # Format stage data for template
            for stage_name, data in stage_metrics.items():
                stages.append({
                    'name': stage_name,
                    'avg_duration': round(data.get('avg_duration', 0), 2),
                    'success_rate': round(data.get('success_rate', 0), 1),
                    'total_executions': data.get('total_executions', 0)
                })
            
            # Sort stages by name
            stages.sort(key=lambda x: x['name'])
            
            # Anomaly data
            anomalies = []
            has_anomalies = False
            
            if monitor_data and 'anomalies' in monitor_data:
                pipeline_anomalies = [a for a in monitor_data['anomalies'] 
                                    if a.get('pipeline_name') == pipeline_name]
                
                if pipeline_anomalies:
                    has_anomalies = True
                    anomalies = [{
                        'type': anomaly.get('type', 'unknown'),
                        'value': round(anomaly.get('value', 0), 2),
                        'expected': round(anomaly.get('expected', 0), 2),
                        'z_score': round(anomaly.get('z_score', 0), 2),
                        'timestamp': datetime.fromtimestamp(anomaly.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
                    } for anomaly in pipeline_anomalies]
            
            # Render template
            template = self.template_env.get_template('pipeline_detail.html')
            html = template.render(
                pipeline_name=pipeline_name,
                total_executions=total_executions,
                success_rate=round(success_rate, 1),
                avg_duration=round(avg_duration, 2),
                latest_success=latest_success,
                flow_diagram_path=flow_diagram_path,
                performance_chart_path=performance_chart_path or '',
                stage_chart_path=stage_chart_path or '',
                stages=stages,
                recent_executions=recent_executions,
                has_anomalies=has_anomalies,
                anomalies=anomalies
            )
            
            # Write detail page
            safe_name = pipeline_name.replace(' ', '_').lower()
            detail_path = os.path.join(output_dir, f"{safe_name}_detail.html")
            with open(detail_path, 'w') as f:
                f.write(html)
                
            return detail_path
            
        except Exception as e:
            self.logger.error(f"Error generating pipeline detail page: {str(e)}")
            return None
    
    def _generate_stage_performance_chart(self, pipeline: Dict[str, Any], 
                                       output_dir: str) -> Optional[str]:
        """
        Generate a chart showing performance metrics for each stage.
        
        Args:
            pipeline: Pipeline data
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib:
            return None
            
        try:
            # Extract stage data from most recent execution
            history = list(pipeline.get('execution_history', []))
            if not history:
                return None
                
            # Get the most recent execution with stage data
            for execution in history:
                if 'stages' in execution and execution['stages']:
                    stages = execution['stages']
                    break
            else:
                return None
                
            # Generate chart
            filename = f"stage_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            stage_names = []
            durations = []
            statuses = []
            
            for stage_name, stage_data in stages.items():
                stage_names.append(stage_name)
                durations.append(stage_data.get('duration', 0))
                statuses.append(stage_data.get('status', 'unknown'))
            
            # Sort stages by duration (descending)
            sorted_data = sorted(zip(stage_names, durations, statuses), 
                               key=lambda x: x[1], reverse=True)
            stage_names, durations, statuses = zip(*sorted_data)
            
            # Create colors based on status
            colors = []
            for status in statuses:
                if status == 'success':
                    colors.append('#28a745')  # Green
                elif status == 'failed':
                    colors.append('#dc3545')  # Red
                else:
                    colors.append('#6c757d')  # Gray
            
            # Create horizontal bar chart
            plt.figure(figsize=(10, max(6, len(stage_names) * 0.5)))
            
            y_pos = range(len(stage_names))
            plt.barh(y_pos, durations, color=colors, alpha=0.8)
            
            # Add stage names and duration labels
            for i, (name, duration) in enumerate(zip(stage_names, durations)):
                plt.text(duration + 0.1, i, f"{duration:.2f}s", va='center')
                
            plt.yticks(y_pos, stage_names)
            plt.xlabel('Duration (seconds)')
            plt.title('Stage Performance')
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating stage performance chart: {str(e)}")
            return None
    
    def _generate_resource_chart(self, monitor_data: Dict[str, Any], 
                               output_dir: str) -> Optional[str]:
        """
        Generate resource utilization chart.
        
        Args:
            monitor_data: Monitor data containing resource information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract resource data
            resource_data = monitor_data.get('resource_utilization', {})
            history = resource_data.get('history', [])
            
            if not history:
                return None
                
            # Generate chart
            filename = f"resource_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            timestamps = [datetime.fromtimestamp(entry.get('timestamp', 0)) for entry in history]
            memory_values = [entry.get('memory_percent', 0) for entry in history]
            cpu_values = [entry.get('cpu_percent', 0) for entry in history]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot memory and CPU
            ax.plot(timestamps, memory_values, 'b-', label='Memory %', alpha=0.7)
            ax.plot(timestamps, cpu_values, 'r-', label='CPU %', alpha=0.7)
            
            # Add labels and legend
            ax.set_xlabel('Time')
            ax.set_ylabel('Utilization %')
            ax.set_title('Resource Utilization')
            ax.legend()
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating resource chart: {str(e)}")
            return None
    
    def _generate_error_chart(self, monitor_data: Dict[str, Any], 
                            output_dir: str) -> Optional[str]:
        """
        Generate error distribution chart.
        
        Args:
            monitor_data: Monitor data containing error information
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib or not monitor_data:
            return None
            
        try:
            # Extract error data
            error_data = monitor_data.get('error_analysis', {}).get('global_stats', {})
            error_distribution = error_data.get('error_distribution', {})
            
            if not error_distribution:
                return None
                
            # Generate chart
            filename = f"error_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            error_types = list(error_distribution.keys())
            percentages = list(error_distribution.values())
            
            # Create color map
            colors = plt.cm.Set3(range(len(error_types)))
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(percentages, labels=error_types, autopct='%1.1f%%', 
                  colors=colors, shadow=False, startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            plt.title('Error Distribution')
            plt.tight_layout()
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating error chart: {str(e)}")
            return None
    
    def _extract_recent_executions(self, pipelines: List[Dict[str, Any]], 
                                 monitor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract recent execution data from multiple pipelines.
        
        Args:
            pipelines: List of pipeline data
            monitor_data: Monitor data
            
        Returns:
            List of recent executions across all pipelines
        """
        recent_executions = []
        
        # Extract from monitor data if available
        if monitor_data and 'recent_executions' in monitor_data:
            return monitor_data['recent_executions']
            
        # Otherwise extract from pipeline data
        for pipeline in pipelines:
            pipeline_name = pipeline.get('name', 'Unknown')
            history = pipeline.get('execution_history', [])
            
            for execution in history:
                execution_data = {
                    'pipeline_name': pipeline_name,
                    'success': execution.get('success', False),
                    'duration': round(execution.get('duration', 0), 2),
                    'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    'stages_completed': len(execution.get('stages', {})),
                    'stages_total': len(execution.get('stages', {}))
                }
                recent_executions.append(execution_data)
        
        # Sort by timestamp (most recent first) and limit to 10
        recent_executions.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), reverse=True)
        return recent_executions[:10]
    
    def _generate_pipeline_detail_page(self, pipeline: Dict[str, Any], 
                                     monitor_data: Dict[str, Any],
                                     output_dir: str) -> Optional[str]:
        """
        Generate detail page for a specific pipeline.
        
        Args:
            pipeline: Pipeline data
            monitor_data: Monitor data
            output_dir: Directory to save the detail page
            
        Returns:
            Path to the detail page file, or None if generation failed
        """
        if not self.has_jinja2:
            return None
            
        try:
            pipeline_name = pipeline.get('name', 'Unknown')
            
            # Generate charts for the pipeline
            charts_dir = os.path.join(output_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            # Flow diagram would come from a specific pipeline object, not metrics
            flow_diagram_path = ''
            
            # Performance chart from history
            performance_data = {
                'pipelines': {pipeline_name: pipeline}
            }
            performance_chart_path = self.create_performance_chart(performance_data, 'svg')
            if performance_chart_path:
                # Copy to charts directory and get relative path
                import shutil
                dest_path = os.path.join(charts_dir, os.path.basename(performance_chart_path))
                shutil.copy2(performance_chart_path, dest_path)
                performance_chart_path = os.path.join('charts', os.path.basename(performance_chart_path))
            
            # Stage performance data
            stage_chart_path = self._generate_stage_performance_chart(pipeline, charts_dir)
            if stage_chart_path:
                stage_chart_path = os.path.join('charts', os.path.basename(stage_chart_path))
            
            # Extract pipeline metrics
            total_executions = pipeline.get('total_executions', 0)
            successful_executions = pipeline.get('successful_executions', 0)
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
            avg_duration = pipeline.get('average_duration', 0)
            
            # Get recent executions
            history = list(pipeline.get('execution_history', []))
            history.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            recent_executions = [{
                'id': execution.get('pipeline_id', 'unknown'),
                'success': execution.get('success', False),
                'duration': round(execution.get('duration', 0), 2),
                'timestamp': datetime.fromtimestamp(execution.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'error_count': execution.get('error_count', 0)
            } for execution in history[:10]]
            
            # Latest execution status
            latest_success = history[0].get('success', False) if history else False
            
            # Stage data
            stages = []
            stage_metrics = {}
            
            # Try to get stage metrics from monitor data
            if monitor_data and 'stage_metrics' in monitor_data:
                for stage_id, stage_data in monitor_data['stage_metrics'].items():
                    if stage_id.startswith(f"{pipeline_name}:"):
                        stage_name = stage_id.split(':', 1)[1]
                        stage_metrics[stage_name] = stage_data
            
            # Extract stage data from execution history if not available in monitor
            if not stage_metrics and history:
                latest_execution = history[0]
                stage_data = latest_execution.get('stages', {})
                
                for stage_name, data in stage_data.items():
                    stage_metrics[stage_name] = {
                        'avg_duration': data.get('duration', 0),
                        'success_rate': 100 if data.get('status') == 'success' else 0,
                        'total_executions': 1
                    }
            
            # Format stage data for template
            for stage_name, data in stage_metrics.items():
                stages.append({
                    'name': stage_name,
                    'avg_duration': round(data.get('avg_duration', 0), 2),
                    'success_rate': round(data.get('success_rate', 0), 1),
                    'total_executions': data.get('total_executions', 0)
                })
            
            # Sort stages by name
            stages.sort(key=lambda x: x['name'])
            
            # Anomaly data
            anomalies = []
            has_anomalies = False
            
            if monitor_data and 'anomalies' in monitor_data:
                pipeline_anomalies = [a for a in monitor_data['anomalies'] 
                                    if a.get('pipeline_name') == pipeline_name]
                
                if pipeline_anomalies:
                    has_anomalies = True
                    anomalies = [{
                        'type': anomaly.get('type', 'unknown'),
                        'value': round(anomaly.get('value', 0), 2),
                        'expected': round(anomaly.get('expected', 0), 2),
                        'z_score': round(anomaly.get('z_score', 0), 2),
                        'timestamp': datetime.fromtimestamp(anomaly.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
                    } for anomaly in pipeline_anomalies]
            
            # Render template
            template = self.template_env.get_template('pipeline_detail.html')
            html = template.render(
                pipeline_name=pipeline_name,
                total_executions=total_executions,
                success_rate=round(success_rate, 1),
                avg_duration=round(avg_duration, 2),
                latest_success=latest_success,
                flow_diagram_path=flow_diagram_path,
                performance_chart_path=performance_chart_path or '',
                stage_chart_path=stage_chart_path or '',
                stages=stages,
                recent_executions=recent_executions,
                has_anomalies=has_anomalies,
                anomalies=anomalies
            )
            
            # Write detail page
            safe_name = pipeline_name.replace(' ', '_').lower()
            detail_path = os.path.join(output_dir, f"{safe_name}_detail.html")
            with open(detail_path, 'w') as f:
                f.write(html)
                
            return detail_path
            
        except Exception as e:
            self.logger.error(f"Error generating pipeline detail page: {str(e)}")
            return None
    
    def _generate_stage_performance_chart(self, pipeline: Dict[str, Any], 
                                       output_dir: str) -> Optional[str]:
        """
        Generate a chart showing performance metrics for each stage.
        
        Args:
            pipeline: Pipeline data
            output_dir: Directory to save the chart
            
        Returns:
            Path to the chart file, or None if generation failed
        """
        if not self.has_matplotlib:
            return None
            
        try:
            # Extract stage data from most recent execution
            history = list(pipeline.get('execution_history', []))
            if not history:
                return None
                
            # Get the most recent execution with stage data
            for execution in history:
                if 'stages' in execution and execution['stages']:
                    stages = execution['stages']
                    break
            else:
                return None
                
            # Generate chart
            filename = f"stage_chart_{int(time.time())}.svg"
            output_path = os.path.join(output_dir, filename)
            
            # Prepare data
            stage_names = []
            durations = []
            statuses = []
            
            for stage_name, stage_data in stages.items():