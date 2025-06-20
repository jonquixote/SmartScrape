#!/usr/bin/env python3
"""
Basic Pipeline Example

This example shows a simple linear pipeline with three stages:
1. Input stage - Loads sample data
2. Processing stage - Transforms the data
3. Output stage - Formats and displays the results

This demonstrates the fundamental concepts of the pipeline architecture
without external dependencies.
"""

import asyncio
import json
from typing import Any, Dict, Optional

# Core pipeline classes
# These would normally be imported from core.pipeline.*
# but are included here for a self-contained example


class PipelineContext:
    """Shared context for pipeline execution with state tracking."""
    
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None) -> None:
        """Initialize a new context."""
        # Pipeline data
        self.data = initial_data or {}
        
        # Execution metadata
        self.metadata = {
            "pipeline_name": None,
            "start_time": None,
            "end_time": None,
            "current_stage": None,
            "completed_stages": set(),
            "stage_metrics": {},
            "errors": {}
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context."""
        return self.data.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """Set a value in the context."""
        self.data[key] = value
        
    def update(self, values: Dict[str, Any]) -> None:
        """Update multiple values in the context."""
        self.data.update(values)
        
    def start_pipeline(self, pipeline_name: str) -> None:
        """Mark the start of pipeline execution."""
        import time
        self.metadata["pipeline_name"] = pipeline_name
        self.metadata["start_time"] = time.time()
        
    def end_pipeline(self) -> None:
        """Mark the end of pipeline execution."""
        import time
        self.metadata["end_time"] = time.time()
        
    def start_stage(self, stage_name: str) -> None:
        """Mark the start of a stage's execution."""
        import time
        self.metadata["current_stage"] = stage_name
        self.metadata["stage_metrics"][stage_name] = {
            "start_time": time.time(),
            "end_time": None,
            "status": "running",
            "execution_time": 0
        }
        
    def end_stage(self, success: bool = True) -> None:
        """Mark the end of a stage's execution."""
        import time
        stage_name = self.metadata["current_stage"]
        if stage_name:
            self.metadata["completed_stages"].add(stage_name)
            self.metadata["stage_metrics"][stage_name].update({
                "end_time": time.time(),
                "status": "success" if success else "failed",
                "execution_time": time.time() - self.metadata["stage_metrics"][stage_name]["start_time"]
            })
            self.metadata["current_stage"] = None
            
    def add_error(self, source: str, message: str) -> None:
        """Add an error to the context."""
        if source not in self.metadata["errors"]:
            self.metadata["errors"][source] = []
        self.metadata["errors"][source].append(message)
        
    def has_errors(self) -> bool:
        """Check if the context has any errors."""
        return len(self.metadata["errors"]) > 0
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for the pipeline."""
        import time
        total_time = 0
        if self.metadata["start_time"]:
            end_time = self.metadata["end_time"] or time.time()
            total_time = end_time - self.metadata["start_time"]
            
        successful_stages = sum(
            1 for metrics in self.metadata["stage_metrics"].values()
            if metrics["status"] == "success"
        )
        
        return {
            "pipeline_name": self.metadata["pipeline_name"],
            "total_time": total_time,
            "stages": self.metadata["stage_metrics"],
            "successful_stages": successful_stages,
            "total_stages": len(self.metadata["stage_metrics"]),
            "has_errors": self.has_errors()
        }


class PipelineStage:
    """Base interface for all pipeline stages."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the pipeline stage with configuration."""
        self.config = config or {}
        self.name = self.config.get("name", self.__class__.__name__)
        
    async def process(self, context: PipelineContext) -> bool:
        """Process the current stage."""
        raise NotImplementedError("Subclasses must implement process method")
        
    def validate_input(self, context: PipelineContext) -> bool:
        """Validate that the context contains required inputs."""
        return True
        
    def handle_error(self, context: PipelineContext, error: Exception) -> bool:
        """Handle an error that occurred during processing."""
        context.add_error(self.name, str(error))
        return False


class Pipeline:
    """A pipeline that executes a series of stages on a shared context."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize a new pipeline."""
        self.name = name
        self.config = config or {}
        self.stages = []
        
    def add_stage(self, stage: PipelineStage):
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        return self
        
    async def execute(self, initial_data: Optional[Dict[str, Any]] = None) -> PipelineContext:
        """Execute the pipeline with the provided initial data."""
        context = PipelineContext(initial_data or {})
        context.start_pipeline(self.name)
        
        try:
            await self._execute_sequential(context)
        except Exception as e:
            context.add_error("pipeline", str(e))
        finally:
            context.end_pipeline()
            
        return context
        
    async def _execute_sequential(self, context: PipelineContext) -> None:
        """Execute pipeline stages sequentially."""
        for stage in self.stages:
            stage_name = stage.name
            context.start_stage(stage_name)
            
            try:
                if not stage.validate_input(context):
                    context.add_error(stage_name, "Input validation failed")
                    context.end_stage(success=False)
                    if not self.config.get("continue_on_error", False):
                        break
                    continue
                    
                success = await stage.process(context)
                context.end_stage(success=success)
                
                if not success and not self.config.get("continue_on_error", False):
                    break
            except Exception as e:
                if stage.handle_error(context, e):
                    context.end_stage(success=False)
                    if self.config.get("continue_on_error", False):
                        continue
                else:
                    context.end_stage(success=False)
                    raise


# Example Stage Implementations

class SampleDataInputStage(PipelineStage):
    """Input stage that loads sample data."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Load sample data into the context."""
        print(f"[{self.name}] Loading sample data...")
        
        # Get data source from configuration or use default
        data_source = self.config.get("data_source", "sample")
        
        if data_source == "sample":
            # Load built-in sample data
            sample_data = {
                "products": [
                    {
                        "id": "prod-001",
                        "name": "Smartphone X",
                        "price": 799.99,
                        "category": "electronics",
                        "in_stock": True
                    },
                    {
                        "id": "prod-002",
                        "name": "Laptop Pro",
                        "price": 1299.99,
                        "category": "electronics",
                        "in_stock": False
                    },
                    {
                        "id": "prod-003",
                        "name": "Bluetooth Headphones",
                        "price": 99.99,
                        "category": "accessories",
                        "in_stock": True
                    }
                ]
            }
            context.set("raw_data", sample_data)
            print(f"[{self.name}] Loaded {len(sample_data['products'])} sample products")
            return True
        else:
            # Handle other data sources (file, API, etc.)
            context.add_error(self.name, f"Unsupported data source: {data_source}")
            return False


class DataTransformationStage(PipelineStage):
    """Processing stage that transforms the data."""
    
    def validate_input(self, context: PipelineContext) -> bool:
        """Validate that the context contains required inputs."""
        if not context.get("raw_data"):
            context.add_error(self.name, "Missing required input: raw_data")
            return False
        return True
    
    async def process(self, context: PipelineContext) -> bool:
        """Transform the data in the context."""
        print(f"[{self.name}] Transforming data...")
        
        raw_data = context.get("raw_data")
        products = raw_data.get("products", [])
        
        # Apply transformations based on configuration
        should_normalize_prices = self.config.get("normalize_prices", True)
        should_filter_in_stock = self.config.get("filter_in_stock", False)
        should_add_vat = self.config.get("add_vat", False)
        vat_rate = self.config.get("vat_rate", 0.2)  # 20% VAT
        
        transformed_products = []
        
        for product in products:
            # Filter by in-stock status if configured
            if should_filter_in_stock and not product.get("in_stock", False):
                continue
                
            # Create a copy of the product to transform
            transformed_product = product.copy()
            
            # Normalize price format if configured
            if should_normalize_prices:
                price = transformed_product.get("price", 0)
                transformed_product["price"] = round(price, 2)
                
            # Add VAT calculation if configured
            if should_add_vat:
                price = transformed_product.get("price", 0)
                transformed_product["price_with_vat"] = round(price * (1 + vat_rate), 2)
                transformed_product["vat_amount"] = round(price * vat_rate, 2)
                
            transformed_products.append(transformed_product)
        
        # Store transformed data in context
        context.set("transformed_data", {"products": transformed_products})
        
        print(f"[{self.name}] Transformed {len(transformed_products)} products")
        return True


class DataOutputStage(PipelineStage):
    """Output stage that formats and displays the results."""
    
    def validate_input(self, context: PipelineContext) -> bool:
        """Validate that the context contains required inputs."""
        if not context.get("transformed_data"):
            context.add_error(self.name, "Missing required input: transformed_data")
            return False
        return True
    
    async def process(self, context: PipelineContext) -> bool:
        """Format and output the transformed data."""
        print(f"[{self.name}] Formatting output...")
        
        transformed_data = context.get("transformed_data")
        products = transformed_data.get("products", [])
        
        # Get output format from configuration
        output_format = self.config.get("format", "json")
        
        if output_format == "json":
            # Format as JSON
            output = json.dumps(products, indent=2)
            context.set("formatted_output", output)
            
            # Display the output
            print(f"\n--- Formatted Output (JSON) ---\n{output}\n----------------------------")
            
        elif output_format == "text":
            # Format as plain text
            lines = []
            for product in products:
                line = f"Product: {product.get('name', 'Unknown')}"
                line += f" - Price: ${product.get('price', 0):.2f}"
                
                if "price_with_vat" in product:
                    line += f" (With VAT: ${product.get('price_with_vat', 0):.2f})"
                    
                line += f" - {'In Stock' if product.get('in_stock', False) else 'Out of Stock'}"
                lines.append(line)
                
            output = "\n".join(lines)
            context.set("formatted_output", output)
            
            # Display the output
            print(f"\n--- Formatted Output (Text) ---\n{output}\n----------------------------")
            
        else:
            context.add_error(self.name, f"Unsupported output format: {output_format}")
            return False
            
        return True


async def main():
    """Run the basic pipeline example."""
    print("=== Basic Pipeline Example ===\n")
    
    # Create a pipeline
    pipeline = Pipeline("basic_pipeline")
    
    # Add stages
    pipeline.add_stage(SampleDataInputStage())
    pipeline.add_stage(DataTransformationStage({
        "normalize_prices": True,
        "add_vat": True,
        "vat_rate": 0.2
    }))
    pipeline.add_stage(DataOutputStage({
        "format": "text"
    }))
    
    # Execute the pipeline
    context = await pipeline.execute()
    
    # Print metrics
    metrics = context.get_metrics()
    print("\n=== Pipeline Execution Metrics ===")
    print(f"Pipeline: {metrics['pipeline_name']}")
    print(f"Total execution time: {metrics['total_time']:.4f}s")
    print(f"Stage metrics:")
    
    for stage_name, stage_metrics in metrics['stages'].items():
        status = stage_metrics['status']
        time = stage_metrics['execution_time']
        print(f"  {stage_name}: {status} in {time:.4f}s")
    
    # Check for errors
    if context.has_errors():
        print("\n=== Pipeline Errors ===")
        for source, messages in context.metadata["errors"].items():
            for message in messages:
                print(f"  {source}: {message}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())

# Expected output:
"""
=== Basic Pipeline Example ===

[SampleDataInputStage] Loading sample data...
[SampleDataInputStage] Loaded 3 sample products
[DataTransformationStage] Transforming data...
[DataTransformationStage] Transformed 3 products
[DataOutputStage] Formatting output...

--- Formatted Output (Text) ---
Product: Smartphone X - Price: $799.99 (With VAT: $959.99) - In Stock
Product: Laptop Pro - Price: $1299.99 (With VAT: $1559.99) - Out of Stock
Product: Bluetooth Headphones - Price: $99.99 (With VAT: $119.99) - In Stock
----------------------------

=== Pipeline Execution Metrics ===
Pipeline: basic_pipeline
Total execution time: 0.0012s
Stage metrics:
  SampleDataInputStage: success in 0.0004s
  DataTransformationStage: success in 0.0003s
  DataOutputStage: success in 0.0003s

=== Example Complete ===
"""