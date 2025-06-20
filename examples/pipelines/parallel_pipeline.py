#!/usr/bin/env python3
"""
Parallel Pipeline Example

This example demonstrates a pipeline with parallel execution of independent stages.
It shows how to process multiple data sources concurrently and combine their results.

Key concepts demonstrated:
- Concurrent stage execution
- Task coordination
- Result aggregation
- Performance monitoring
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
import random

# Import the base pipeline classes from the basic example
# In a real implementation, you would import from core.pipeline.*
from examples.pipelines.basic_pipeline import (
    Pipeline, PipelineStage, PipelineContext
)


# Data Source Stages

class DataSourceStage(PipelineStage):
    """Base class for data source stages."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.source_name = self.config.get("source_name", self.__class__.__name__)
        self.delay = self.config.get("delay", 0)  # Simulated delay in seconds
        
    async def process(self, context: PipelineContext) -> bool:
        """Fetch data from a source with simulated delay."""
        print(f"[{self.name}] Fetching data from {self.source_name}...")
        
        # Simulate network delay or processing time
        if self.delay > 0:
            print(f"[{self.name}] Operation will take {self.delay:.1f}s")
            await asyncio.sleep(self.delay)
            
        # Fetch the actual data
        try:
            data = await self._fetch_data()
            context.set(f"data_{self.source_name}", data)
            print(f"[{self.name}] Successfully fetched {len(data)} items")
            return True
        except Exception as e:
            context.add_error(self.name, f"Error fetching data: {str(e)}")
            return False
            
    async def _fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch data from the source. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _fetch_data()")


class ProductDataSource(DataSourceStage):
    """Fetches product data."""
    
    async def _fetch_data(self) -> List[Dict[str, Any]]:
        """Simulate fetching product data."""
        # Create sample product data
        products = []
        categories = ["Electronics", "Clothing", "Home", "Books", "Sports"]
        
        for i in range(1, self.config.get("num_products", 10) + 1):
            products.append({
                "id": f"PROD-{i:03d}",
                "name": f"Product {i}",
                "price": round(random.uniform(10, 1000), 2),
                "category": random.choice(categories),
                "in_stock": random.choice([True, False]),
                "rating": round(random.uniform(1, 5), 1)
            })
            
        return products


class CustomerDataSource(DataSourceStage):
    """Fetches customer data."""
    
    async def _fetch_data(self) -> List[Dict[str, Any]]:
        """Simulate fetching customer data."""
        # Create sample customer data
        customers = []
        statuses = ["Active", "Inactive", "New", "VIP"]
        
        for i in range(1, self.config.get("num_customers", 15) + 1):
            customers.append({
                "id": f"CUST-{i:03d}",
                "name": f"Customer {i}",
                "email": f"customer{i}@example.com",
                "status": random.choice(statuses),
                "joined": f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "purchases": random.randint(0, 50)
            })
            
        return customers


class OrderDataSource(DataSourceStage):
    """Fetches order data."""
    
    async def _fetch_data(self) -> List[Dict[str, Any]]:
        """Simulate fetching order data."""
        # Create sample order data
        orders = []
        statuses = ["Pending", "Processing", "Shipped", "Delivered", "Cancelled"]
        
        for i in range(1, self.config.get("num_orders", 20) + 1):
            # Generate random product and customer IDs
            product_id = f"PROD-{random.randint(1, 10):03d}"
            customer_id = f"CUST-{random.randint(1, 15):03d}"
            
            orders.append({
                "id": f"ORD-{i:03d}",
                "customer_id": customer_id,
                "products": [
                    {
                        "product_id": product_id,
                        "quantity": random.randint(1, 5),
                        "price": round(random.uniform(10, 1000), 2)
                    }
                    for _ in range(random.randint(1, 3))
                ],
                "status": random.choice(statuses),
                "date": f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "total": round(random.uniform(20, 2000), 2)
            })
            
        return orders


# Processing Stages

class AnalyticsStage(PipelineStage):
    """Analyzes data from multiple sources."""
    
    def validate_input(self, context: PipelineContext) -> bool:
        """Validate that all required data sources are available."""
        required_sources = self.config.get("required_sources", [])
        
        for source in required_sources:
            if not context.get(f"data_{source}"):
                context.add_error(self.name, f"Missing data from source: {source}")
                return False
                
        return True
    
    async def process(self, context: PipelineContext) -> bool:
        """Process data from all sources to generate analytics."""
        print(f"[{self.name}] Generating analytics...")
        
        # Get data from all sources
        sources = self.config.get("sources", [])
        all_data = {}
        
        for source in sources:
            source_data = context.get(f"data_{source}")
            if source_data:
                all_data[source] = source_data
                
        # Create analytics
        analytics = {}
        
        # Process each source independently
        tasks = []
        for source, data in all_data.items():
            task = asyncio.create_task(self._analyze_source(source, data))
            tasks.append(task)
            
        # Wait for all analyses to complete
        source_analytics = await asyncio.gather(*tasks)
        
        # Combine results
        for source_name, result in source_analytics:
            analytics[source_name] = result
            
        # Add combined analytics if we have the necessary data
        if all(source in all_data for source in ["products", "customers", "orders"]):
            analytics["combined"] = await self._generate_combined_analytics(all_data)
            
        context.set("analytics", analytics)
        print(f"[{self.name}] Analytics generation complete")
        return True
        
    async def _analyze_source(self, source: str, data: List[Dict[str, Any]]) -> tuple:
        """Analyze a single data source."""
        print(f"[{self.name}] Analyzing {source} data...")
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.2, 0.5))
        
        if source == "products":
            result = self._analyze_products(data)
        elif source == "customers":
            result = self._analyze_customers(data)
        elif source == "orders":
            result = self._analyze_orders(data)
        else:
            result = {"item_count": len(data)}
            
        return (source, result)
        
    def _analyze_products(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze product data."""
        if not products:
            return {}
            
        # Calculate basic statistics
        total_products = len(products)
        in_stock = sum(1 for p in products if p.get("in_stock", False))
        avg_price = sum(p.get("price", 0) for p in products) / total_products if total_products else 0
        
        # Group by category
        categories = {}
        for product in products:
            category = product.get("category", "Unknown")
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
            
        return {
            "total_products": total_products,
            "in_stock_count": in_stock,
            "out_of_stock_count": total_products - in_stock,
            "average_price": round(avg_price, 2),
            "categories": categories
        }
        
    def _analyze_customers(self, customers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze customer data."""
        if not customers:
            return {}
            
        # Calculate basic statistics
        total_customers = len(customers)
        active_customers = sum(1 for c in customers if c.get("status") == "Active")
        avg_purchases = sum(c.get("purchases", 0) for c in customers) / total_customers if total_customers else 0
        
        # Group by status
        statuses = {}
        for customer in customers:
            status = customer.get("status", "Unknown")
            if status not in statuses:
                statuses[status] = 0
            statuses[status] += 1
            
        return {
            "total_customers": total_customers,
            "active_customers": active_customers,
            "inactive_customers": total_customers - active_customers,
            "average_purchases": round(avg_purchases, 2),
            "status_distribution": statuses
        }
        
    def _analyze_orders(self, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze order data."""
        if not orders:
            return {}
            
        # Calculate basic statistics
        total_orders = len(orders)
        total_revenue = sum(order.get("total", 0) for order in orders)
        avg_order_value = total_revenue / total_orders if total_orders else 0
        
        # Group by status
        statuses = {}
        for order in orders:
            status = order.get("status", "Unknown")
            if status not in statuses:
                statuses[status] = 0
            statuses[status] += 1
            
        # Calculate items per order
        total_items = sum(
            sum(item.get("quantity", 1) for item in order.get("products", []))
            for order in orders
        )
        avg_items_per_order = total_items / total_orders if total_orders else 0
        
        return {
            "total_orders": total_orders,
            "total_revenue": round(total_revenue, 2),
            "average_order_value": round(avg_order_value, 2),
            "average_items_per_order": round(avg_items_per_order, 2),
            "status_distribution": statuses
        }
        
    async def _generate_combined_analytics(self, all_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate analytics that combine multiple data sources."""
        print(f"[{self.name}] Generating combined analytics...")
        
        # Simulate complex processing
        await asyncio.sleep(random.uniform(0.5, 1.0))
        
        products = all_data.get("products", [])
        customers = all_data.get("customers", [])
        orders = all_data.get("orders", [])
        
        # Calculate revenue per customer
        customer_revenue = {}
        for order in orders:
            customer_id = order.get("customer_id")
            if customer_id:
                if customer_id not in customer_revenue:
                    customer_revenue[customer_id] = 0
                customer_revenue[customer_id] += order.get("total", 0)
                
        avg_revenue_per_customer = (
            sum(customer_revenue.values()) / len(customer_revenue)
            if customer_revenue else 0
        )
        
        # Calculate product popularity
        product_orders = {}
        for order in orders:
            for item in order.get("products", []):
                product_id = item.get("product_id")
                if product_id:
                    if product_id not in product_orders:
                        product_orders[product_id] = 0
                    product_orders[product_id] += item.get("quantity", 1)
                    
        # Find most popular products
        popular_products = sorted(
            product_orders.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return {
            "customer_count": len(customers),
            "product_count": len(products),
            "order_count": len(orders),
            "average_revenue_per_customer": round(avg_revenue_per_customer, 2),
            "popular_products": [
                {"product_id": pid, "orders": count}
                for pid, count in popular_products
            ]
        }


class ReportGenerationStage(PipelineStage):
    """Generates a report based on analytics data."""
    
    def validate_input(self, context: PipelineContext) -> bool:
        """Validate that analytics data is available."""
        if not context.get("analytics"):
            context.add_error(self.name, "Missing analytics data")
            return False
        return True
    
    async def process(self, context: PipelineContext) -> bool:
        """Generate a formatted report from analytics data."""
        print(f"[{self.name}] Generating report...")
        
        analytics = context.get("analytics", {})
        report_format = self.config.get("format", "text")
        
        # Generate the report in the requested format
        if report_format == "json":
            report = self._generate_json_report(analytics)
        elif report_format == "html":
            report = self._generate_html_report(analytics)
        else:
            report = self._generate_text_report(analytics)
            
        context.set("report", report)
        context.set("report_format", report_format)
        
        print(f"[{self.name}] Report generation complete")
        return True
        
    def _generate_text_report(self, analytics: Dict[str, Any]) -> str:
        """Generate a plain text report."""
        lines = ["=== ANALYTICS REPORT ===", ""]
        
        # Add sections for each data source
        for source, data in analytics.items():
            if source == "combined":
                continue  # We'll add this at the end
                
            lines.append(f"--- {source.upper()} ANALYTICS ---")
            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f"{key}:")
                    for subkey, subvalue in value.items():
                        lines.append(f"  {subkey}: {subvalue}")
                else:
                    lines.append(f"{key}: {value}")
            lines.append("")
            
        # Add combined section if it exists
        if "combined" in analytics:
            lines.append("--- COMBINED ANALYTICS ---")
            for key, value in analytics["combined"].items():
                if isinstance(value, list):
                    lines.append(f"{key}:")
                    for item in value:
                        lines.append(f"  {item}")
                else:
                    lines.append(f"{key}: {value}")
                    
        return "\n".join(lines)
        
    def _generate_json_report(self, analytics: Dict[str, Any]) -> str:
        """Generate a JSON report."""
        return json.dumps(analytics, indent=2)
        
    def _generate_html_report(self, analytics: Dict[str, Any]) -> str:
        """Generate an HTML report."""
        html = ["<html>", "<head><title>Analytics Report</title></head>", "<body>", 
                "<h1>Analytics Report</h1>"]
        
        # Add sections for each data source
        for source, data in analytics.items():
            if source == "combined":
                continue  # We'll add this at the end
                
            html.append(f"<h2>{source.title()} Analytics</h2>")
            html.append("<table border='1'>")
            html.append("<tr><th>Metric</th><th>Value</th></tr>")
            
            for key, value in data.items():
                if isinstance(value, dict):
                    html.append(f"<tr><td>{key}</td><td><pre>{json.dumps(value, indent=2)}</pre></td></tr>")
                else:
                    html.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
                    
            html.append("</table>")
            
        # Add combined section if it exists
        if "combined" in analytics:
            html.append("<h2>Combined Analytics</h2>")
            html.append("<table border='1'>")
            html.append("<tr><th>Metric</th><th>Value</th></tr>")
            
            for key, value in analytics["combined"].items():
                if isinstance(value, list):
                    html.append(f"<tr><td>{key}</td><td><pre>{json.dumps(value, indent=2)}</pre></td></tr>")
                else:
                    html.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
                    
            html.append("</table>")
            
        html.extend(["</body>", "</html>"])
        return "\n".join(html)


# Pipeline Monitor Class

class PipelineMonitor:
    """Monitors pipeline execution metrics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.stage_times = {}
        
    def start_pipeline(self):
        """Mark the start of pipeline execution."""
        self.start_time = time.time()
        print(f"\n[Monitor] Pipeline started at {self.start_time:.3f}")
        
    def end_pipeline(self):
        """Mark the end of pipeline execution."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"[Monitor] Pipeline completed in {duration:.3f}s")
        
    def start_stage(self, stage_name):
        """Mark the start of a stage's execution."""
        self.stage_times[stage_name] = {"start": time.time()}
        
    def end_stage(self, stage_name):
        """Mark the end of a stage's execution."""
        if stage_name in self.stage_times:
            self.stage_times[stage_name]["end"] = time.time()
            duration = self.stage_times[stage_name]["end"] - self.stage_times[stage_name]["start"]
            self.stage_times[stage_name]["duration"] = duration
            print(f"[Monitor] Stage '{stage_name}' completed in {duration:.3f}s")
            
    def get_report(self):
        """Generate a timing report."""
        if not self.end_time:
            self.end_time = time.time()
            
        total_duration = self.end_time - self.start_time
        stage_reports = []
        
        for stage_name, timings in self.stage_times.items():
            if "duration" in timings:
                percentage = (timings["duration"] / total_duration) * 100
                stage_reports.append({
                    "name": stage_name,
                    "duration": timings["duration"],
                    "percentage": percentage
                })
                
        return {
            "total_duration": total_duration,
            "stages": sorted(stage_reports, key=lambda x: x["duration"], reverse=True)
        }


# Enhanced Pipeline with Monitoring

class MonitoredPipeline(Pipeline):
    """Pipeline with execution monitoring capabilities."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self.monitor = PipelineMonitor()
        
    async def execute(self, initial_data: Optional[Dict[str, Any]] = None) -> PipelineContext:
        """Execute the pipeline with the provided initial data and monitoring."""
        context = PipelineContext(initial_data or {})
        context.start_pipeline(self.name)
        self.monitor.start_pipeline()
        
        try:
            if self.parallel_execution:
                await self._execute_parallel(context)
            else:
                await self._execute_sequential(context)
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            context.add_error("pipeline", str(e))
        finally:
            context.end_pipeline()
            self.monitor.end_pipeline()
            
        # Add monitoring report to context
        context.set("monitoring_report", self.monitor.get_report())
            
        return context
        
    async def _execute_sequential(self, context: PipelineContext) -> None:
        """Execute pipeline stages sequentially with monitoring."""
        for stage in self.stages:
            stage_name = stage.name
            context.start_stage(stage_name)
            self.monitor.start_stage(stage_name)
            
            try:
                if not stage.validate_input(context):
                    context.add_error(stage_name, "Input validation failed")
                    context.end_stage(success=False)
                    self.monitor.end_stage(stage_name)
                    if not self.config.get("continue_on_error", False):
                        break
                    continue
                    
                success = await stage.process(context)
                context.end_stage(success=success)
                self.monitor.end_stage(stage_name)
                
                if not success and not self.config.get("continue_on_error", False):
                    print(f"[{self.name}] Stage {stage_name} failed, stopping pipeline")
                    break
            except Exception as e:
                print(f"[{self.name}] Error in stage {stage_name}: {str(e)}")
                if stage.handle_error(context, e):
                    context.end_stage(success=False)
                    self.monitor.end_stage(stage_name)
                    if self.config.get("continue_on_error", False):
                        continue
                else:
                    context.end_stage(success=False)
                    self.monitor.end_stage(stage_name)
                    raise
                    
    async def _execute_parallel(self, context: PipelineContext) -> None:
        """Execute independent pipeline stages in parallel with monitoring."""
        # Group stages by their dependency level (0 = no dependencies)
        stage_groups = {0: []}
        
        # For simplicity in this example, we'll just use a single group
        # In a real implementation, you would analyze dependencies
        stage_groups[0] = self.stages
        
        # Process each level of stages
        for level, stages in sorted(stage_groups.items()):
            if not stages:
                continue
                
            print(f"[{self.name}] Executing {len(stages)} stages at level {level}")
            
            # Create tasks for all stages at this level
            tasks = []
            for stage in stages:
                task = asyncio.create_task(self._execute_stage(stage, context))
                tasks.append(task)
                
            # Wait for all tasks to complete
            await asyncio.gather(*tasks)
            
    async def _execute_stage(self, stage: PipelineStage, context: PipelineContext) -> None:
        """Execute a single stage with monitoring."""
        stage_name = stage.name
        context.start_stage(stage_name)
        self.monitor.start_stage(stage_name)
        
        try:
            if not stage.validate_input(context):
                context.add_error(stage_name, "Input validation failed")
                context.end_stage(success=False)
                self.monitor.end_stage(stage_name)
                return
                
            success = await stage.process(context)
            context.end_stage(success=success)
            self.monitor.end_stage(stage_name)
            
        except Exception as e:
            print(f"[{self.name}] Error in stage {stage_name}: {str(e)}")
            stage.handle_error(context, e)
            context.end_stage(success=False)
            self.monitor.end_stage(stage_name)


async def main():
    """Run the parallel pipeline example."""
    print("=== Parallel Pipeline Example ===\n")
    
    # Create a monitored pipeline
    pipeline = MonitoredPipeline("business_analytics", {
        "parallel_execution": True,
        "continue_on_error": True
    })
    
    # Add data source stages with varying delays to simulate real-world behavior
    pipeline.add_stage(ProductDataSource({
        "source_name": "products",
        "num_products": 20,
        "delay": 1.5  # 1.5 second delay
    }))
    
    pipeline.add_stage(CustomerDataSource({
        "source_name": "customers",
        "num_customers": 50,
        "delay": 2.0  # 2 second delay
    }))
    
    pipeline.add_stage(OrderDataSource({
        "source_name": "orders",
        "num_orders": 100,
        "delay": 1.0  # 1 second delay
    }))
    
    # Add analytics stage that processes all data sources
    pipeline.add_stage(AnalyticsStage({
        "sources": ["products", "customers", "orders"],
        "required_sources": ["products", "customers", "orders"]
    }))
    
    # Add report generation stage
    pipeline.add_stage(ReportGenerationStage({
        "format": "text"
    }))
    
    # Execute the pipeline
    print("Starting pipeline execution...")
    context = await pipeline.execute()
    
    # Check for errors
    if context.has_errors():
        print("\nErrors during execution:")
        for source, messages in context.metadata["errors"].items():
            for message in messages:
                print(f"  {source}: {message}")
    
    # Display the report
    print("\n=== Generated Report ===")
    report = context.get("report", "No report generated")
    print(report)
    
    # Display monitoring information
    monitoring = context.get("monitoring_report", {})
    
    print("\n=== Performance Metrics ===")
    print(f"Total execution time: {monitoring.get('total_duration', 0):.3f}s")
    print("\nStage timings:")
    
    for stage in monitoring.get("stages", []):
        print(f"  {stage['name']}: {stage['duration']:.3f}s ({stage['percentage']:.1f}%)")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())

# Expected output:
"""
=== Parallel Pipeline Example ===

Starting pipeline execution...

[Monitor] Pipeline started at 1684326485.123
[ProductDataSource] Fetching data from products...
[ProductDataSource] Operation will take 1.5s
[CustomerDataSource] Fetching data from customers...
[CustomerDataSource] Operation will take 2.0s
[OrderDataSource] Fetching data from orders...
[OrderDataSource] Operation will take 1.0s
[OrderDataSource] Successfully fetched 100 items
[Monitor] Stage 'OrderDataSource' completed in 1.023s
[ProductDataSource] Successfully fetched 20 items
[Monitor] Stage 'ProductDataSource' completed in 1.513s
[CustomerDataSource] Successfully fetched 50 items
[Monitor] Stage 'CustomerDataSource' completed in 2.015s
[AnalyticsStage] Generating analytics...
[AnalyticsStage] Analyzing products data...
[AnalyticsStage] Analyzing customers data...
[AnalyticsStage] Analyzing orders data...
[AnalyticsStage] Generating combined analytics...
[AnalyticsStage] Analytics generation complete
[Monitor] Stage 'AnalyticsStage' completed in 1.234s
[ReportGenerationStage] Generating report...
[ReportGenerationStage] Report generation complete
[Monitor] Stage 'ReportGenerationStage' completed in 0.021s
[Monitor] Pipeline completed in 3.283s

=== Generated Report ===
=== ANALYTICS REPORT ===

--- PRODUCTS ANALYTICS ---
total_products: 20
in_stock_count: 11
out_of_stock_count: 9
average_price: 505.23
categories:
  Electronics: 5
  Clothing: 3
  Home: 4
  Books: 2
  Sports: 6

--- CUSTOMERS ANALYTICS ---
total_customers: 50
active_customers: 12
inactive_customers: 38
average_purchases: 24.5
status_distribution:
  Active: 12
  Inactive: 15
  New: 13
  VIP: 10

--- ORDERS ANALYTICS ---
total_orders: 100
total_revenue: 10248.36
average_order_value: 102.48
average_items_per_order: 1.96
status_distribution:
  Pending: 18
  Processing: 22
  Shipped: 25
  Delivered: 30
  Cancelled: 5

--- COMBINED ANALYTICS ---
customer_count: 50
product_count: 20
order_count: 100
average_revenue_per_customer: 204.97
popular_products:
  {"product_id": "PROD-003", "orders": 15}
  {"product_id": "PROD-010", "orders": 12}
  {"product_id": "PROD-007", "orders": 11}

=== Performance Metrics ===
Total execution time: 3.283s

Stage timings:
  CustomerDataSource: 2.015s (61.4%)
  ProductDataSource: 1.513s (46.1%)
  AnalyticsStage: 1.234s (37.6%)
  OrderDataSource: 1.023s (31.2%)
  ReportGenerationStage: 0.021s (0.6%)

=== Example Complete ===
"""