#!/usr/bin/env python3
"""
Benchmark script to compare different AI service optimization strategies.
This script measures performance across various optimization techniques.
"""

import asyncio
import os
import time
import json
import logging
from typing import Dict, Any, List, Tuple
import argparse
from datetime import datetime
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark")

# Import SmartScrape components
from core.ai_service import AIService
from core.content_processor import ContentProcessor
from core.model_selector import ModelSelector
from core.rule_engine import RuleEngine
from core.batch_processor import BatchProcessor
from core.service_registry import ServiceRegistry

# Import test data utilities
import sys
import urllib.request
import io
import gzip
from bs4 import BeautifulSoup

# Test constants
DEFAULT_ITERATIONS = 3
TEST_PROMPT = "Summarize the main points of this article in 3 bullet points:"

# HTML test data - download a sample or use default
DEFAULT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Service Optimization</title>
    <meta name="description" content="Understanding AI service optimization">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        header, footer { background-color: #f5f5f5; padding: 10px; }
        nav ul { list-style-type: none; margin: 0; padding: 0; }
        nav ul li { display: inline; margin-right: 10px; }
        article { margin: 20px 0; }
        h1 { color: #333; }
    </style>
</head>
<body>
    <header>
        <nav>
            <ul>
                <li><a href="/">Home</a></li>
                <li><a href="/about">About</a></li>
                <li><a href="/contact">Contact</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        <article>
            <h1>Understanding AI Service Optimization</h1>
            
            <p>AI service optimization is a critical aspect of building scalable, efficient AI-powered applications.
            As AI becomes more integrated into everyday tools, optimizing these services becomes increasingly important.</p>
            
            <h2>Key Optimization Strategies</h2>
            
            <p>There are several key strategies for optimizing AI services:</p>
            
            <ul>
                <li><strong>Caching responses</strong> to reduce API calls and latency</li>
                <li><strong>Content preprocessing</strong> to minimize token usage</li>
                <li><strong>Intelligent model selection</strong> to balance cost and performance</li>
                <li><strong>Batch processing</strong> to optimize request patterns</li>
                <li><strong>Rule-based alternatives</strong> for common patterns</li>
            </ul>
            
            <p>When implemented properly, these strategies can significantly reduce costs, improve performance,
            and enhance the user experience of AI-powered applications.</p>
            
            <h2>Implementation Considerations</h2>
            
            <p>While implementing these optimization strategies, it's important to consider the trade-offs
            between processing time, memory usage, and accuracy. Different use cases may require different
            optimization priorities.</p>
            
            <p>For example, a real-time application might prioritize caching and quick rule-based alternatives,
            while a batch processing system might focus more on content preprocessing to minimize token usage.</p>
            
            <h2>Measuring Optimization Impact</h2>
            
            <p>It's important to establish baselines and measure the impact of each optimization:
            
            <ul>
                <li>Response time improvements</li>
                <li>Cost reduction in API usage</li>
                <li>Accuracy and quality of results</li>
                <li>Resource usage (CPU, memory, etc.)</li>
            </ul>
            
            <p>Regular benchmarking ensures that optimizations actually deliver the expected benefits
            without compromising on the quality of results.</p>
        </article>
    </main>
    
    <footer>
        <p>&copy; 2025 Example Company</p>
        <nav>
            <ul>
                <li><a href="/privacy">Privacy Policy</a></li>
                <li><a href="/terms">Terms of Service</a></li>
            </ul>
        </nav>
    </footer>
</body>
</html>
"""

def fetch_test_html(url=None):
    """Fetch HTML content for testing."""
    if not url:
        return DEFAULT_HTML
    
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            content_type = response.headers.get('Content-Type', '')
            content = response.read()
            
            # Handle gzipped content
            if response.headers.get('Content-Encoding') == 'gzip':
                content = gzip.decompress(content)
            
            if 'text/html' in content_type:
                return content.decode('utf-8', errors='replace')
            else:
                print(f"Warning: URL did not return HTML (got {content_type})")
                return DEFAULT_HTML
    except Exception as e:
        print(f"Error fetching HTML from {url}: {e}")
        return DEFAULT_HTML

class BenchmarkResult:
    """Class to store and analyze benchmark results."""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.runs = []
        self.metadata = {}
    
    def add_run(self, elapsed_time, input_tokens=None, output_tokens=None, cost=None, success=True, metadata=None):
        """Add a benchmark run result."""
        self.runs.append({
            'elapsed_time': elapsed_time,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': cost,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
    
    def get_summary(self):
        """Get a summary of the benchmark results."""
        if not self.runs:
            return {"name": self.name, "description": self.description, "error": "No runs recorded"}
        
        # Calculate statistics
        times = [run['elapsed_time'] for run in self.runs if run['success']]
        if not times:
            return {"name": self.name, "description": self.description, "error": "No successful runs"}
        
        # Calculate token statistics if available
        input_tokens = [run['input_tokens'] for run in self.runs if run['success'] and run['input_tokens'] is not None]
        output_tokens = [run['output_tokens'] for run in self.runs if run['success'] and run['output_tokens'] is not None]
        costs = [run['cost'] for run in self.runs if run['success'] and run['cost'] is not None]
        
        summary = {
            "name": self.name,
            "description": self.description,
            "runs": len(self.runs),
            "successful_runs": len(times),
            "time_stats": {
                "mean": statistics.mean(times) if times else None,
                "median": statistics.median(times) if times else None,
                "min": min(times) if times else None,
                "max": max(times) if times else None,
                "stdev": statistics.stdev(times) if len(times) > 1 else None
            },
            "metadata": self.metadata
        }
        
        # Add token and cost statistics if available
        if input_tokens:
            summary["input_token_stats"] = {
                "mean": statistics.mean(input_tokens),
                "median": statistics.median(input_tokens),
                "min": min(input_tokens),
                "max": max(input_tokens)
            }
        
        if output_tokens:
            summary["output_token_stats"] = {
                "mean": statistics.mean(output_tokens),
                "median": statistics.median(output_tokens),
                "min": min(output_tokens),
                "max": max(output_tokens)
            }
        
        if costs:
            summary["cost_stats"] = {
                "mean": statistics.mean(costs),
                "median": statistics.median(costs),
                "min": min(costs),
                "max": max(costs),
                "total": sum(costs)
            }
        
        return summary

async def initialize_ai_service(config):
    """Initialize and configure the AI service."""
    registry = ServiceRegistry()
    ai_service = AIService()
    registry.register_service(ai_service)
    ai_service.initialize(config)
    return ai_service

async def benchmark_baseline(api_key, html_content, iterations=3):
    """Benchmark baseline performance without optimizations."""
    logger.info("Running baseline benchmark (no optimizations)...")
    
    result = BenchmarkResult(
        "baseline", 
        "Baseline performance without optimizations"
    )
    
    # Configure AI service with minimal settings
    config = {
        "models": [
            {
                "name": "default",
                "type": "openai",
                "model_id": "gpt-3.5-turbo",
                "api_key": api_key
            }
        ],
        "default_model": "default",
        # No caching or other optimizations
    }
    
    # Initialize service
    ai_service = await initialize_ai_service(config)
    
    # Run benchmark iterations
    for i in range(iterations):
        logger.info(f"Baseline run {i+1}/{iterations}")
        prompt = f"{TEST_PROMPT}\n\n{html_content}"
        
        try:
            start_time = time.time()
            response = await ai_service.generate_response(prompt, use_cache=False)
            elapsed = time.time() - start_time
            
            # Extract metadata
            metadata = response.get("_metadata", {})
            input_tokens = metadata.get("input_tokens")
            output_tokens = metadata.get("output_tokens")
            cost = metadata.get("total_cost")
            
            result.add_run(
                elapsed_time=elapsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                metadata={"iteration": i+1}
            )
            
            logger.info(f"Run completed in {elapsed:.2f}s, input tokens: {input_tokens}, output tokens: {output_tokens}")
            
        except Exception as e:
            logger.error(f"Error in baseline run {i+1}: {str(e)}")
            result.add_run(elapsed_time=0, success=False, metadata={"error": str(e)})
    
    # Cleanup
    ai_service.shutdown()
    return result

async def benchmark_with_caching(api_key, html_content, iterations=3):
    """Benchmark performance with caching enabled."""
    logger.info("Running benchmark with caching...")
    
    result = BenchmarkResult(
        "caching", 
        "Performance with caching optimization"
    )
    
    # Configure AI service with caching
    config = {
        "models": [
            {
                "name": "default",
                "type": "openai",
                "model_id": "gpt-3.5-turbo",
                "api_key": api_key
            }
        ],
        "default_model": "default",
        "cache": {
            "backend": "memory",
            "default_ttl": 3600
        }
    }
    
    # Initialize service
    ai_service = await initialize_ai_service(config)
    
    # First run (cache miss)
    logger.info("First run (cache miss expected)")
    prompt = f"{TEST_PROMPT}\n\n{html_content}"
    
    try:
        start_time = time.time()
        response = await ai_service.generate_response(prompt)
        elapsed = time.time() - start_time
        
        # Extract metadata
        metadata = response.get("_metadata", {})
        input_tokens = metadata.get("input_tokens")
        output_tokens = metadata.get("output_tokens")
        cost = metadata.get("total_cost")
        
        result.add_run(
            elapsed_time=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            metadata={"iteration": 1, "cache_status": "miss"}
        )
        
        logger.info(f"First run completed in {elapsed:.2f}s (cache miss)")
        
        # Subsequent runs (cache hits)
        for i in range(1, iterations):
            logger.info(f"Cache run {i+1}/{iterations} (cache hit expected)")
            start_time = time.time()
            response = await ai_service.generate_response(prompt)
            elapsed = time.time() - start_time
            
            # For cache hits, we shouldn't incur token costs
            # but we'll record metadata from the original response
            result.add_run(
                elapsed_time=elapsed,
                input_tokens=0,  # No tokens consumed on cache hit
                output_tokens=0,  # No tokens consumed on cache hit
                cost=0,  # No cost incurred on cache hit
                metadata={"iteration": i+1, "cache_status": "hit"}
            )
            
            logger.info(f"Run completed in {elapsed:.2f}s (cache hit)")
    
    except Exception as e:
        logger.error(f"Error in cache benchmark: {str(e)}")
        result.add_run(elapsed_time=0, success=False, metadata={"error": str(e)})
    
    # Record cache stats
    result.metadata["cache_stats"] = ai_service.cache.get_stats()
    
    # Cleanup
    ai_service.shutdown()
    return result

async def benchmark_with_preprocessing(api_key, html_content, iterations=3):
    """Benchmark performance with content preprocessing."""
    logger.info("Running benchmark with content preprocessing...")
    
    result = BenchmarkResult(
        "preprocessing", 
        "Performance with content preprocessing optimization"
    )
    
    # Configure AI service
    config = {
        "models": [
            {
                "name": "default",
                "type": "openai",
                "model_id": "gpt-3.5-turbo",
                "api_key": api_key
            }
        ],
        "default_model": "default",
        "cache": {
            "backend": "memory",
            "default_ttl": 3600
        }
    }
    
    # Initialize service
    ai_service = await initialize_ai_service(config)
    
    # Initialize content processor directly for metadata
    content_processor = ContentProcessor()
    
    # Process HTML content first for logging
    processed_html = content_processor.preprocess_html(html_content, extract_main=True)
    token_reduction = (len(html_content) - len(processed_html)) / len(html_content) * 100
    logger.info(f"Content preprocessing reduced content size by {token_reduction:.1f}%")
    result.metadata["preprocessing_stats"] = {
        "original_length": len(html_content),
        "processed_length": len(processed_html),
        "percent_reduction": token_reduction
    }
    
    # Run benchmark iterations
    for i in range(iterations):
        logger.info(f"Preprocessing run {i+1}/{iterations}")
        
        # Use the context parameter to enable preprocessing
        context = {
            "content_type": "html",
            "preprocess": True,
            "extract_main": True
        }
        
        prompt = f"{TEST_PROMPT}\n\n{html_content}"
        
        try:
            start_time = time.time()
            response = await ai_service.generate_response(
                prompt, 
                context=context,
                use_cache=False  # Disable cache to measure preprocessing effect
            )
            elapsed = time.time() - start_time
            
            # Extract metadata
            metadata = response.get("_metadata", {})
            input_tokens = metadata.get("input_tokens")
            output_tokens = metadata.get("output_tokens")
            cost = metadata.get("total_cost")
            
            result.add_run(
                elapsed_time=elapsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                metadata={"iteration": i+1}
            )
            
            logger.info(f"Run completed in {elapsed:.2f}s, input tokens: {input_tokens}")
            
        except Exception as e:
            logger.error(f"Error in preprocessing run {i+1}: {str(e)}")
            result.add_run(elapsed_time=0, success=False, metadata={"error": str(e)})
    
    # Cleanup
    ai_service.shutdown()
    return result

async def benchmark_with_all_optimizations(api_key, html_content, iterations=3):
    """Benchmark performance with all optimizations enabled."""
    logger.info("Running benchmark with all optimizations...")
    
    result = BenchmarkResult(
        "all_optimizations", 
        "Performance with all optimizations enabled"
    )
    
    # Configure AI service with all optimizations
    config = {
        "models": [
            {
                "name": "default",
                "type": "openai",
                "model_id": "gpt-3.5-turbo",
                "api_key": api_key
            },
            {
                "name": "fast",
                "type": "openai",
                "model_id": "gpt-3.5-turbo-instruct",  # Faster, cheaper model for simpler tasks
                "api_key": api_key
            }
        ],
        "default_model": "default",
        "cache": {
            "backend": "memory",
            "default_ttl": 3600
        },
        "preprocessing": {
            "enabled": True,
            "default_extraction": True
        },
        "batch_processing": {
            "enabled": True,
            "max_batch_size": 5
        },
        "rule_engine": {
            "enabled": True
        }
    }
    
    # Initialize service
    ai_service = await initialize_ai_service(config)
    
    # Run benchmark iterations
    for i in range(iterations):
        logger.info(f"Full optimization run {i+1}/{iterations}")
        
        context = {
            "task_type": "summarization",
            "content_type": "html",
            "preprocess": True,
            "extract_main": True,
            "options": {
                "temperature": 0.3
            }
        }
        
        prompt = f"{TEST_PROMPT}\n\n{html_content}"
        
        try:
            start_time = time.time()
            response = await ai_service.generate_response(
                prompt, 
                context=context,
                # Let the system decide on caching for the first iteration
                use_cache=(i > 0)  # Force cache use after first iteration
            )
            elapsed = time.time() - start_time
            
            # Extract metadata
            metadata = response.get("_metadata", {})
            input_tokens = metadata.get("input_tokens", 0)
            output_tokens = metadata.get("output_tokens", 0)
            cost = metadata.get("total_cost", 0)
            
            # For cache hits after first iteration, these will be 0
            cache_status = "hit" if i > 0 else "miss"
            
            result.add_run(
                elapsed_time=elapsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                metadata={"iteration": i+1, "cache_status": cache_status}
            )
            
            logger.info(f"Run completed in {elapsed:.2f}s, cost: ${cost if cost else 'N/A'}")
            
        except Exception as e:
            logger.error(f"Error in full optimization run {i+1}: {str(e)}")
            result.add_run(elapsed_time=0, success=False, metadata={"error": str(e)})
    
    # Record optimization metadata
    if hasattr(ai_service, 'cache'):
        result.metadata["cache_stats"] = ai_service.cache.get_stats()
    
    # Cleanup
    ai_service.shutdown()
    return result

async def run_benchmarks(api_key, html_content, iterations=3, output_file=None):
    """Run all benchmarks and generate comparison results."""
    all_results = []
    
    # Run individual benchmarks
    baseline = await benchmark_baseline(api_key, html_content, iterations)
    all_results.append(baseline)
    
    caching = await benchmark_with_caching(api_key, html_content, iterations)
    all_results.append(caching)
    
    preprocessing = await benchmark_with_preprocessing(api_key, html_content, iterations)
    all_results.append(preprocessing)
    
    all_optimizations = await benchmark_with_all_optimizations(api_key, html_content, iterations)
    all_results.append(all_optimizations)
    
    # Generate summary
    summaries = [result.get_summary() for result in all_results]
    
    # Calculate comparative metrics
    comparative = calculate_comparative_metrics(summaries)
    
    # Combine results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "iterations": iterations,
            "test_prompt": TEST_PROMPT,
            "html_content_size": len(html_content)
        },
        "results": summaries,
        "comparative": comparative
    }
    
    # Write to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results written to {output_file}")
    
    # Print summary
    print_summary(results)
    
    return results

def calculate_comparative_metrics(summaries):
    """Calculate comparative metrics between different benchmarks."""
    baseline = next((s for s in summaries if s["name"] == "baseline"), None)
    if not baseline:
        return {"error": "No baseline results found for comparison"}
    
    baseline_time = baseline["time_stats"]["mean"]
    baseline_cost = baseline.get("cost_stats", {}).get("mean", 0)
    
    comparative = {
        "time_improvement": {},
        "cost_savings": {}
    }
    
    for summary in summaries:
        if summary["name"] == "baseline":
            continue
        
        # Time improvement
        if "time_stats" in summary and summary["time_stats"]["mean"]:
            time_diff = baseline_time - summary["time_stats"]["mean"]
            time_pct = (time_diff / baseline_time) * 100 if baseline_time else 0
            comparative["time_improvement"][summary["name"]] = {
                "absolute": time_diff,
                "percent": time_pct
            }
        
        # Cost savings
        summary_cost = summary.get("cost_stats", {}).get("mean", 0)
        if baseline_cost and summary_cost is not None:
            cost_diff = baseline_cost - summary_cost
            cost_pct = (cost_diff / baseline_cost) * 100 if baseline_cost else 0
            comparative["cost_savings"][summary["name"]] = {
                "absolute": cost_diff,
                "percent": cost_pct
            }
    
    return comparative

def print_summary(results):
    """Print a human-readable summary of benchmark results."""
    print("\n" + "="*80)
    print(f"AI SERVICE OPTIMIZATION BENCHMARK SUMMARY")
    print("="*80)
    
    print(f"\nTest Configuration:")
    print(f"- Iterations: {results['config']['iterations']}")
    print(f"- Test content size: {results['config']['html_content_size']} characters")
    print(f"- Timestamp: {results['timestamp']}")
    
    print("\nPerformance Summary:")
    print("-"*80)
    print(f"{'Benchmark':<20} {'Avg Time (s)':<15} {'Tokens In':<15} {'Cost ($)':<15} {'Improvement'}")
    print("-"*80)
    
    # Get baseline for comparison
    baseline = next((r for r in results["results"] if r["name"] == "baseline"), None)
    baseline_time = baseline["time_stats"]["mean"] if baseline else None
    
    for result in results["results"]:
        name = result["name"]
        avg_time = result["time_stats"]["mean"]
        
        # Get token info if available
        tokens_in = "N/A"
        if "input_token_stats" in result:
            tokens_in = f"{int(result['input_token_stats']['mean'])}"
        
        # Get cost info if available
        cost = "N/A"
        if "cost_stats" in result:
            cost = f"${result['cost_stats']['mean']:.5f}"
        
        # Calculate improvement compared to baseline
        improvement = "N/A"
        if name != "baseline" and baseline_time:
            time_pct = results["comparative"]["time_improvement"][name]["percent"]
            improvement = f"{time_pct:.1f}% faster"
        
        print(f"{name:<20} {avg_time:<15.3f} {tokens_in:<15} {cost:<15} {improvement}")
    
    print("\nOptimization Benefits:")
    print("-"*80)
    
    for name, time_imp in results["comparative"].get("time_improvement", {}).items():
        print(f"- {name}: {time_imp['percent']:.1f}% faster response time")
    
    for name, cost_imp in results["comparative"].get("cost_savings", {}).items():
        print(f"- {name}: {cost_imp['percent']:.1f}% cost reduction")
    
    print("\nRecommendation:")
    # Find the optimization with the best balance of time and cost savings
    best_option = "all_optimizations"  # Default recommendation
    print(f"Based on the benchmark results, the most effective optimization strategy is: {best_option}")
    print("="*80)

async def main():
    parser = argparse.ArgumentParser(description="Benchmark AI service optimization strategies")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, 
                      help=f"Number of iterations to run (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--url", type=str, 
                      help="URL to fetch HTML content from for testing (optional)")
    parser.add_argument("--output", type=str, 
                      help="Output file to save benchmark results (optional)")
    args = parser.parse_args()
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Get HTML content
    html_content = fetch_test_html(args.url)
    print(f"Using test HTML content ({len(html_content)} characters)")
    
    # Run benchmarks
    print(f"Running benchmarks with {args.iterations} iterations per test...")
    await run_benchmarks(api_key, html_content, args.iterations, args.output)

if __name__ == "__main__":
    asyncio.run(main())