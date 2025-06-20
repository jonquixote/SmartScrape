import unittest
import asyncio
import time
import sys
import psutil
import gc
import statistics
from unittest.mock import MagicMock

from core.pipeline.pipeline import Pipeline
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext


class TestPipelinePerformance(unittest.TestCase):
    """Performance benchmark tests for the pipeline architecture."""
    
    def setUp(self):
        """Set up test environment."""
        # Number of iterations for benchmarks
        self.iterations = 5
        # Timeout for tests
        self.timeout = 30
        # Force garbage collection before tests
        gc.collect()
        
    def tearDown(self):
        """Clean up after tests."""
        # Force garbage collection after tests
        gc.collect()
    
    def _measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    def _measure_async_execution_time(self, func, *args, **kwargs):
        """Measure execution time of an async function."""
        async def wrapper():
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            return result, end_time - start_time
            
        return asyncio.run(wrapper())
    
    def _measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function."""
        # Get initial memory usage
        process = psutil.Process()
        gc.collect()  # Force garbage collection
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final memory usage
        gc.collect()  # Force garbage collection
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Calculate memory delta
        memory_delta = final_memory - initial_memory
        
        return result, memory_delta
    
    def _create_pipeline(self, num_stages, stage_type=None, config=None):
        """Create a pipeline with a specified number of stages."""
        pipeline = Pipeline(f"benchmark_pipeline_{num_stages}")
        
        for i in range(num_stages):
            if stage_type:
                stage = stage_type(config or {})
            else:
                stage = BenchmarkStage(config or {})
            stage.name = f"stage_{i}"
            pipeline.add_stage(stage)
            
        return pipeline
    
    def test_execution_time_by_stage_count(self):
        """Measure execution time for pipelines with different numbers of stages."""
        stage_counts = [1, 5, 10, 20, 50]
        results = {}
        
        for count in stage_counts:
            pipeline = self._create_pipeline(count)
            
            # Warm-up run
            asyncio.run(pipeline.execute())
            
            # Benchmark runs
            times = []
            for _ in range(self.iterations):
                _, execution_time = self._measure_async_execution_time(pipeline.execute)
                times.append(execution_time)
                
            avg_time = statistics.mean(times)
            results[count] = avg_time
            
            print(f"Pipeline with {count} stages: {avg_time:.4f}s")
        
        # Verify that execution time scales roughly linearly with stage count
        # This is a simple check to ensure that there's no exponential growth
        if len(stage_counts) > 1:
            ratio_first = results[stage_counts[1]] / results[stage_counts[0]]
            ratio_expected = stage_counts[1] / stage_counts[0]
            self.assertLess(ratio_first / ratio_expected, 2.0, 
                           "Execution time should scale roughly linearly with stage count")
    
    def test_sequential_vs_parallel_execution(self):
        """Compare sequential vs. parallel execution performance."""
        num_stages = 10
        heavy_config = {"processing_time": 0.01}  # 10ms per stage
        
        # Create sequential pipeline
        seq_pipeline = self._create_pipeline(
            num_stages, 
            BenchmarkStage, 
            heavy_config
        )
        seq_pipeline.parallel_execution = False
        
        # Create parallel pipeline
        parallel_pipeline = self._create_pipeline(
            num_stages, 
            BenchmarkStage, 
            heavy_config
        )
        parallel_pipeline.parallel_execution = True
        
        # Warm-up runs
        asyncio.run(seq_pipeline.execute())
        asyncio.run(parallel_pipeline.execute())
        
        # Benchmark sequential pipeline
        seq_times = []
        for _ in range(self.iterations):
            _, time_taken = self._measure_async_execution_time(seq_pipeline.execute)
            seq_times.append(time_taken)
        
        seq_avg = statistics.mean(seq_times)
        
        # Benchmark parallel pipeline
        parallel_times = []
        for _ in range(self.iterations):
            _, time_taken = self._measure_async_execution_time(parallel_pipeline.execute)
            parallel_times.append(time_taken)
            
        parallel_avg = statistics.mean(parallel_times)
        
        print(f"Sequential execution: {seq_avg:.4f}s")
        print(f"Parallel execution: {parallel_avg:.4f}s")
        print(f"Speedup: {seq_avg / parallel_avg:.2f}x")
        
        # Verify parallel is faster (may not be true on single-core systems)
        # Add a tolerance factor since parallelization has overhead
        self.assertLessEqual(parallel_avg, seq_avg * 1.2, "Parallel execution should be faster than sequential")
    
    def test_different_stage_implementations(self):
        """Benchmark different stage implementations."""
        # Define different stage types to benchmark
        stage_types = {
            "NoOp": NoOpStage,
            "Light": LightProcessingStage,
            "Medium": MediumProcessingStage,
            "Heavy": HeavyProcessingStage,
            "IO": IOSimulationStage
        }
        
        results = {}
        
        for name, stage_cls in stage_types.items():
            # Create pipeline with this stage type
            pipeline = Pipeline(f"benchmark_{name}")
            for i in range(5):  # Use 5 stages of each type
                stage = stage_cls()
                stage.name = f"{name}_stage_{i}"
                pipeline.add_stage(stage)
                
            # Warm-up run
            asyncio.run(pipeline.execute())
            
            # Benchmark runs
            times = []
            for _ in range(self.iterations):
                _, execution_time = self._measure_async_execution_time(pipeline.execute)
                times.append(execution_time)
                
            avg_time = statistics.mean(times)
            results[name] = avg_time
            
            print(f"{name} stage pipeline: {avg_time:.4f}s")
        
        # Verify the expected performance ordering
        self.assertLess(results["NoOp"], results["Light"], "NoOp should be faster than Light processing")
        self.assertLess(results["Light"], results["Medium"], "Light should be faster than Medium processing")
        self.assertLess(results["Medium"], results["Heavy"], "Medium should be faster than Heavy processing")
    
    def test_data_size_impact(self):
        """Test pipeline performance with varying data sizes."""
        data_sizes = [1, 10, 100, 1000, 10000]
        results = {}
        
        for size in data_sizes:
            # Create a pipeline with data processing stages
            pipeline = Pipeline("data_size_benchmark")
            pipeline.add_stage(DataGeneratorStage({"size": size}))
            for i in range(3):  # Add 3 processing stages
                pipeline.add_stage(DataProcessingStage())
                
            # Warm-up run
            asyncio.run(pipeline.execute())
            
            # Benchmark runs
            times = []
            for _ in range(self.iterations):
                _, execution_time = self._measure_async_execution_time(pipeline.execute)
                times.append(execution_time)
                
            avg_time = statistics.mean(times)
            results[size] = avg_time
            
            print(f"Data size {size}: {avg_time:.4f}s")
        
        # Verify performance scales with data size
        # Simple check to ensure processing time increases with data size
        for i in range(len(data_sizes) - 1):
            self.assertLessEqual(
                results[data_sizes[i]], 
                results[data_sizes[i+1]] * 1.2,  # Allow some variance
                f"Processing time should increase with data size ({data_sizes[i]} vs {data_sizes[i+1]})"
            )
    
    def test_memory_usage(self):
        """Analyze memory usage patterns during pipeline execution."""
        data_sizes = [1, 10, 100, 1000, 10000]
        results = {}
        
        for size in data_sizes:
            # Create pipeline that handles different data sizes
            pipeline = Pipeline("memory_benchmark")
            pipeline.add_stage(DataGeneratorStage({"size": size}))
            pipeline.add_stage(DataHoldingStage())
            pipeline.add_stage(DataProcessingStage())
            
            # Measure memory usage
            def execute_pipeline():
                return asyncio.run(pipeline.execute())
                
            _, memory_delta = self._measure_memory_usage(execute_pipeline)
            results[size] = memory_delta
            
            print(f"Data size {size}: {memory_delta:.2f}MB")
        
        # Verify memory usage scales with data size
        # Simple check that larger data sizes use more memory
        for i in range(len(data_sizes) - 1):
            size_ratio = data_sizes[i+1] / data_sizes[i]
            # Memory shouldn't grow faster than data size
            # (in practice, it might not be linear due to overhead)
            self.assertLess(
                results[data_sizes[i+1]] / (results[data_sizes[i]] + 0.001),  # Avoid div by zero
                size_ratio * 2,  # Allow some overhead
                f"Memory usage should scale reasonably with data size ({data_sizes[i]} vs {data_sizes[i+1]})"
            )
    
    def test_cpu_utilization(self):
        """Profile CPU utilization during pipeline execution."""
        pipelines = {
            "Light": self._create_pipeline(10, LightProcessingStage),
            "Heavy": self._create_pipeline(10, HeavyProcessingStage),
            "Parallel": Pipeline("parallel_cpu", {"parallel_execution": True})
        }
        
        # Add stages to parallel pipeline
        for i in range(10):
            pipelines["Parallel"].add_stage(HeavyProcessingStage())
        
        results = {}
        
        for name, pipeline in pipelines.items():
            # Measure CPU utilization during execution
            process = psutil.Process()
            
            # Warm-up
            asyncio.run(pipeline.execute())
            
            # Start measuring
            start_cpu_times = process.cpu_times()
            start_time = time.time()
            
            # Execute pipeline multiple times for a more stable measurement
            for _ in range(3):
                asyncio.run(pipeline.execute())
                
            # End measuring
            end_cpu_times = process.cpu_times()
            end_time = time.time()
            
            # Calculate CPU utilization
            elapsed = end_time - start_time
            cpu_user = end_cpu_times.user - start_cpu_times.user
            cpu_system = end_cpu_times.system - start_cpu_times.system
            cpu_total = cpu_user + cpu_system
            
            # Calculate percentage
            cpu_percent = (cpu_total / elapsed) * 100.0
            
            results[name] = cpu_percent
            
            print(f"{name} pipeline CPU utilization: {cpu_percent:.2f}%")
        
        # Verify CPU utilization is higher for heavy processing
        self.assertGreater(
            results["Heavy"], 
            results["Light"], 
            "Heavy processing should utilize more CPU"
        )


# Benchmark stages for performance testing

class BenchmarkStage(PipelineStage):
    """Basic benchmark stage with configurable processing time."""
    
    async def process(self, context):
        """Process with a configurable delay."""
        # Get the processing time from config or use default
        processing_time = self.config.get("processing_time", 0.001)  # Default: 1ms
        
        # Simulate processing by sleeping
        await asyncio.sleep(processing_time)
        
        # Simulate a bit of CPU work
        result = 0
        for i in range(1000):
            result += i
            
        return True


class NoOpStage(PipelineStage):
    """Stage that does minimal work."""
    
    async def process(self, context):
        """Do almost nothing."""
        return True


class LightProcessingStage(PipelineStage):
    """Stage that does light processing work."""
    
    async def process(self, context):
        """Do a small amount of work."""
        # Simulate light CPU work
        result = 0
        for i in range(10000):
            result += i
            
        context.set(f"light_result_{self.name}", result)
        return True


class MediumProcessingStage(PipelineStage):
    """Stage that does medium processing work."""
    
    async def process(self, context):
        """Do a moderate amount of work."""
        # Simulate medium CPU work
        result = 0
        for i in range(100000):
            result += i
            result *= 1.0001
            
        context.set(f"medium_result_{self.name}", result)
        return True


class HeavyProcessingStage(PipelineStage):
    """Stage that does heavy processing work."""
    
    async def process(self, context):
        """Do a significant amount of work."""
        # Simulate heavy CPU work
        result = 0
        for i in range(500000):
            result += i
            result *= 1.0001
            
        context.set(f"heavy_result_{self.name}", result)
        return True


class IOSimulationStage(PipelineStage):
    """Stage that simulates I/O operations."""
    
    async def process(self, context):
        """Simulate I/O operations with sleep."""
        # Simulate I/O delay
        await asyncio.sleep(0.05)  # 50ms I/O operation
        
        context.set(f"io_completed_{self.name}", True)
        return True


class DataGeneratorStage(PipelineStage):
    """Stage that generates test data of configurable size."""
    
    async def process(self, context):
        """Generate data of the specified size."""
        size = self.config.get("size", 100)
        
        # Generate a list of dictionaries
        data = [{"index": i, "value": f"test_value_{i}"} for i in range(size)]
        
        context.set("test_data", data)
        context.set("data_size", size)
        return True


class DataProcessingStage(PipelineStage):
    """Stage that processes data in the context."""
    
    async def process(self, context):
        """Process all data items."""
        data = context.get("test_data", [])
        
        # Process each item
        processed_data = []
        for item in data:
            # Simple transformation
            processed_item = {
                "id": item["index"],
                "transformed_value": item["value"].upper(),
                "processed": True
            }
            processed_data.append(processed_item)
            
        context.set("processed_data", processed_data)
        return True


class DataHoldingStage(PipelineStage):
    """Stage that holds onto data to test memory usage."""
    
    async def process(self, context):
        """Hold reference to data to prevent garbage collection."""
        data = context.get("test_data", [])
        
        # Store a reference to the data
        self.held_data = data.copy()
        
        # Hold onto an enlarged copy for memory testing
        enlarged_data = []
        for item in data:
            # Create a larger representation of each item
            enlarged_item = {
                "original": item,
                "extra_data": "x" * 100,  # Add 100 bytes per item
                "more_data": [i for i in range(100)]  # Add another ~400 bytes
            }
            enlarged_data.append(enlarged_item)
            
        context.set("enlarged_data", enlarged_data)
        return True


if __name__ == "__main__":
    unittest.main()