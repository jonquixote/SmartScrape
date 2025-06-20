#!/usr/bin/env python3
"""
Pipeline Converter Utility

This script helps identify and convert legacy components to pipeline stages.
It provides analysis of existing code, generation of pipeline configurations,
and creation of pipeline stage stubs.

Usage:
    python -m scripts.tools.pipeline_converter analyze --path extraction/content_extraction.py
    python -m scripts.tools.pipeline_converter generate-config --component ContentExtractor
    python -m scripts.tools.pipeline_converter generate-stages --component ContentExtractor
    python -m scripts.tools.pipeline_converter interactive
"""

import os
import sys
import ast
import inspect
import importlib
import re
import argparse
import json
import asyncio
from typing import Any, Dict, List, Set, Optional, Tuple, Union, Type, Callable
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pipeline_converter")

# Add project root to path to allow importing project modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Default output directory for generated files
DEFAULT_OUTPUT_DIR = project_root / "core" / "pipeline" / "generated"

class PipelineCandidateAnalyzer:
    """Analyzes code to identify pipeline candidates."""
    
    def __init__(self, path: str):
        """
        Initialize the analyzer with a file path.
        
        Args:
            path: Path to the file to analyze
        """
        self.path = path
        self.module_name = None
        self.raw_code = None
        self.ast_tree = None
        self.candidates = []
        
    def analyze(self) -> List[Dict[str, Any]]:
        """
        Analyze the file for pipeline candidates.
        
        Returns:
            List of candidate information dictionaries
        """
        self._load_file()
        self._parse_code()
        self._find_candidates()
        return self.candidates
        
    def _load_file(self) -> None:
        """Load the file and convert to module path."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File not found: {self.path}")
            
        # Read file content
        with open(self.path, 'r', encoding='utf-8') as file:
            self.raw_code = file.read()
            
        # Convert file path to module name
        rel_path = os.path.relpath(self.path, project_root)
        self.module_name = rel_path.replace('/', '.').replace('\\', '.').replace('.py', '')
        
    def _parse_code(self) -> None:
        """Parse the code into an AST."""
        self.ast_tree = ast.parse(self.raw_code)
        
    def _find_candidates(self) -> None:
        """Find pipeline candidates in the code."""
        # Find class definitions
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.ClassDef):
                class_info = self._analyze_class(node)
                if class_info and class_info.get("pipeline_score", 0) >= 3:
                    self.candidates.append(class_info)
                    
    def _analyze_class(self, node: ast.ClassDef) -> Optional[Dict[str, Any]]:
        """
        Analyze a class definition for pipeline suitability.
        
        Args:
            node: AST node for class definition
            
        Returns:
            Dictionary with class information or None if not a candidate
        """
        methods = []
        pipeline_score = 0
        linear_flow = False
        error_handling = False
        multi_stage = False
        complex_state = False
        
        # Find methods and their characteristics
        for item in node.body:
            if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                method_info = self._analyze_method(item)
                methods.append(method_info)
                
                # Increment pipeline score based on method characteristics
                if method_info.get("has_steps", 0) >= 3:
                    pipeline_score += 1
                    multi_stage = True
                
                if method_info.get("has_linear_flow", False):
                    pipeline_score += 1
                    linear_flow = True
                    
                if method_info.get("has_error_handling", False):
                    pipeline_score += 1
                    error_handling = True
                    
                if method_info.get("has_complex_state", False):
                    pipeline_score += 1
                    complex_state = True
                    
        # Additional class-level analysis
        if hasattr(node, "bases") and node.bases:
            # Check if the class inherits from a relevant base class
            base_names = [
                self._get_name(base) for base in node.bases
                if hasattr(base, "id") or hasattr(base, "attr")
            ]
            for base in base_names:
                if "extractor" in base.lower() or "processor" in base.lower():
                    pipeline_score += 1
        
        return {
            "type": "class",
            "name": node.name,
            "module": self.module_name,
            "file": self.path,
            "pipeline_score": pipeline_score,
            "methods": methods,
            "characteristics": {
                "linear_flow": linear_flow,
                "error_handling": error_handling,
                "multi_stage": multi_stage,
                "complex_state": complex_state
            }
        }
        
    def _analyze_method(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """
        Analyze a method for pipeline characteristics.
        
        Args:
            node: AST node for method definition
            
        Returns:
            Dictionary with method information
        """
        # Count distinct processing steps
        steps = self._identify_processing_steps(node)
        
        # Check for linear flow
        has_linear_flow = self._has_linear_flow(node)
        
        # Check for error handling
        has_error_handling = self._has_error_handling(node)
        
        # Check for complex state management
        has_complex_state = self._has_complex_state(node)
        
        # Count inputs/outputs
        inputs = self._count_inputs(node)
        outputs = self._identify_outputs(node)
        
        return {
            "name": node.name,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "has_steps": len(steps),
            "steps": steps,
            "has_linear_flow": has_linear_flow,
            "has_error_handling": has_error_handling,
            "has_complex_state": has_complex_state,
            "inputs": inputs,
            "outputs": outputs
        }
        
    def _identify_processing_steps(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """Identify distinct processing steps in a method."""
        steps = []
        step_count = 0
        
        # Look for blocks of code separated by comments or distinct operations
        for idx, item in enumerate(ast.iter_child_nodes(node)):
            # Check for comments indicating steps
            if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                comment = item.value.value
                if "step" in comment.lower() or "stage" in comment.lower():
                    step_count += 1
                    steps.append(f"Step {step_count}: {comment}")
            
            # Check for function calls that might indicate processing
            elif isinstance(item, ast.Expr) and isinstance(item.value, ast.Call):
                func_name = self._get_name(item.value.func)
                if any(keyword in func_name.lower() for keyword in 
                       ["process", "extract", "transform", "validate", "normalize"]):
                    step_count += 1
                    steps.append(f"Step {step_count}: {func_name}")
            
            # Check for assignments that might indicate processing
            elif isinstance(item, ast.Assign):
                if len(item.targets) == 1 and isinstance(item.value, ast.Call):
                    func_name = self._get_name(item.value.func)
                    target_name = self._get_name(item.targets[0])
                    if any(keyword in func_name.lower() for keyword in 
                           ["process", "extract", "transform", "validate", "normalize"]):
                        step_count += 1
                        steps.append(f"Step {step_count}: {target_name} = {func_name}")
                        
        # Fallback: If no explicit steps found, count statement blocks
        if not steps:
            # Group statements by type to identify logical blocks
            current_type = None
            for item in ast.iter_child_nodes(node):
                item_type = type(item).__name__
                if item_type != current_type:
                    current_type = item_type
                    step_count += 1
                    steps.append(f"Implicit Step {step_count}: {item_type} block")
                    
        return steps
        
    def _has_linear_flow(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if the method has a linear flow without complex branching."""
        branch_count = 0
        loop_count = 0
        
        # Count branching and loop constructs
        for item in ast.walk(node):
            if isinstance(item, (ast.If, ast.IfExp)):
                branch_count += 1
            elif isinstance(item, (ast.For, ast.While, ast.AsyncFor)):
                loop_count += 1
                
        # Consider linear if limited branching
        return branch_count <= 3 and loop_count <= 2
        
    def _has_error_handling(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if the method has error handling."""
        for item in ast.walk(node):
            if isinstance(item, ast.Try):
                return True
        return False
        
    def _has_complex_state(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if the method manages complex state."""
        attr_assigns = 0
        for item in ast.walk(node):
            if isinstance(item, ast.Attribute) and isinstance(item.ctx, ast.Store):
                if hasattr(item.value, "id") and item.value.id == "self":
                    attr_assigns += 1
        return attr_assigns >= 3
        
    def _count_inputs(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict[str, Any]]:
        """Count and analyze the method inputs."""
        inputs = []
        for arg in node.args.args:
            if arg.arg != "self":
                inputs.append({
                    "name": arg.arg,
                    "has_default": False
                })
                
        # Add defaults
        defaults = node.args.defaults
        if defaults:
            for i in range(len(defaults)):
                idx = len(inputs) - len(defaults) + i
                if idx >= 0 and idx < len(inputs):
                    inputs[idx]["has_default"] = True
                    
        return inputs
        
    def _identify_outputs(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """Identify possible outputs from the method."""
        outputs = []
        
        # Look for return statements
        for item in ast.walk(node):
            if isinstance(item, ast.Return):
                if hasattr(item, "value"):
                    if isinstance(item.value, ast.Dict):
                        # If returning a dictionary, try to get keys
                        for key_idx, key in enumerate(item.value.keys):
                            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                outputs.append(key.value)
                    elif isinstance(item.value, ast.Name):
                        # If returning a variable
                        outputs.append(item.value.id)
                        
        return outputs
        
    def _get_name(self, node: ast.AST) -> str:
        """Extract name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return str(node.__class__.__name__)
            

class PipelineConfigGenerator:
    """Generates pipeline configurations from components."""
    
    def __init__(self, component_name: str, module_path: Optional[str] = None):
        """
        Initialize the generator.
        
        Args:
            component_name: Name of the component to convert
            module_path: Optional module path (will try to infer if not provided)
        """
        self.component_name = component_name
        self.module_path = module_path
        self.component_class = None
        self.component_instance = None
        self.methods = []
        
    def generate_config(self) -> Dict[str, Any]:
        """
        Generate a pipeline configuration for the component.
        
        Returns:
            Dictionary with pipeline configuration
        """
        self._import_component()
        self._analyze_component()
        
        # Create pipeline config
        pipeline_name = f"{self.component_name.lower()}_pipeline"
        
        stages = []
        for method_info in self.methods:
            if self._is_processing_method(method_info):
                # Convert method to stage
                stage_config = self._method_to_stage_config(method_info)
                if stage_config:
                    stages.append(stage_config)
        
        # Create final config
        config = {
            "name": pipeline_name,
            "description": f"Pipeline generated from {self.component_name}",
            "version": "0.1.0",
            "stages": stages,
            "config": {
                "continue_on_error": False,
                "parallel_execution": False
            }
        }
        
        return config
        
    def _import_component(self) -> None:
        """Import the component class."""
        # Try to find the module if not specified
        if not self.module_path:
            self.module_path = self._find_module_for_class(self.component_name)
            
        if not self.module_path:
            raise ValueError(f"Could not find module for {self.component_name}")
            
        try:
            # Import the module and get the class
            module = importlib.import_module(self.module_path)
            self.component_class = getattr(module, self.component_name)
            
            # Try to create an instance for introspection
            try:
                self.component_instance = self.component_class()
            except Exception:
                # If default constructor fails, we'll proceed without an instance
                logger.warning(f"Could not instantiate {self.component_name}, proceeding with class-only analysis")
                
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import {self.component_name} from {self.module_path}: {e}")
            raise
            
    def _find_module_for_class(self, class_name: str) -> Optional[str]:
        """Search project for the module containing the class."""
        # List of common directories to search
        search_dirs = ['extraction', 'strategies', 'components', 'core']
        
        for search_dir in search_dirs:
            # Walk through the directory
            for root, _, files in os.walk(os.path.join(project_root, search_dir)):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        
                        # Check if the class is in this file
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if f"class {class_name}" in content:
                                    # Convert file path to module path
                                    rel_path = os.path.relpath(file_path, project_root)
                                    module_path = rel_path.replace('/', '.').replace('\\', '.').replace('.py', '')
                                    return module_path
                        except Exception:
                            continue
                            
        return None
        
    def _analyze_component(self) -> None:
        """Analyze the component to find pipeline-suitable methods."""
        if not self.component_class:
            return
            
        # Get all methods through introspection
        for name, method in inspect.getmembers(self.component_class, predicate=inspect.isfunction):
            # Skip private methods and special methods
            if name.startswith('_') and not name == '__init__':
                continue
                
            # Analyze method
            method_info = {
                "name": name,
                "is_async": asyncio.iscoroutinefunction(method),
                "signature": inspect.signature(method)
            }
            
            self.methods.append(method_info)
            
    def _is_processing_method(self, method_info: Dict[str, Any]) -> bool:
        """Determine if a method is a processing method suitable for a pipeline stage."""
        name = method_info["name"]
        
        # Skip common non-processing methods
        skip_methods = ['__init__', 'get_name', 'get_version', 'get_description', 'get_config']
        if name in skip_methods:
            return False
            
        # Check for processing method name patterns
        processing_patterns = ['process', 'extract', 'transform', 'validate', 'normalize', 'analyze', 'detect']
        if any(pattern in name.lower() for pattern in processing_patterns):
            return True
            
        # Check signature - processing methods typically have input parameters
        sig = method_info["signature"]
        param_count = len(sig.parameters)
        if param_count > 1:  # More than just 'self'
            return True
            
        return False
        
    def _method_to_stage_config(self, method_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a method to a pipeline stage configuration."""
        name = method_info["name"]
        sig = method_info["signature"]
        
        # Create stage name
        stage_name = name
        if not stage_name.endswith('_stage'):
            stage_name = f"{name}_stage"
            
        # Create stage class name (CamelCase)
        words = stage_name.split('_')
        class_name = ''.join(word.capitalize() for word in words)
        
        # Analyze parameters
        params = []
        for param_name, param in sig.parameters.items():
            if param_name != 'self':
                has_default = param.default != inspect.Parameter.empty
                params.append({
                    "name": param_name,
                    "has_default": has_default
                })
                
        # Create stage config
        return {
            "name": stage_name,
            "class": class_name,
            "config": {
                "method_name": name,
                "input_mapping": {
                    param["name"]: param["name"] for param in params
                }
            }
        }


class PipelineStageGenerator:
    """Generates pipeline stage implementations from components."""
    
    def __init__(self, 
                component_name: str, 
                module_path: Optional[str] = None,
                output_dir: Optional[str] = None):
        """
        Initialize the generator.
        
        Args:
            component_name: Name of the component to convert
            module_path: Optional module path (will try to infer if not provided)
            output_dir: Directory to write generated files
        """
        self.component_name = component_name
        self.module_path = module_path
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self.config_generator = PipelineConfigGenerator(component_name, module_path)
        self.pipeline_config = None
        
    def generate_stages(self) -> Dict[str, str]:
        """
        Generate pipeline stage implementations.
        
        Returns:
            Dictionary mapping file paths to file contents
        """
        self.pipeline_config = self.config_generator.generate_config()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        generated_files = {}
        
        # Generate files for each stage
        module_name = self._get_module_name()
        
        # Generate __init__.py
        init_path = os.path.join(self.output_dir, "__init__.py")
        init_content = self._generate_init()
        generated_files[init_path] = init_content
        
        # Generate stage classes
        for stage in self.pipeline_config["stages"]:
            class_name = stage["class"]
            file_name = f"{stage['name']}.py"
            file_path = os.path.join(self.output_dir, file_name)
            
            # Generate stage content
            content = self._generate_stage_module(
                class_name=class_name,
                stage_config=stage,
                module_name=module_name
            )
            
            generated_files[file_path] = content
            
        # Generate pipeline module
        pipeline_path = os.path.join(self.output_dir, f"{self.pipeline_config['name']}.py")
        pipeline_content = self._generate_pipeline_module()
        generated_files[pipeline_path] = pipeline_content
        
        # Write files
        for path, content in generated_files.items():
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        return generated_files
        
    def _generate_init(self) -> str:
        """Generate __init__.py content."""
        return f'''"""
Pipeline stages generated from {self.component_name}.
Generated by pipeline_converter.py
"""

# Import all generated stages
from .{self.pipeline_config["name"]} import {self.component_name}Pipeline
'''
        
    def _generate_stage_module(self, 
                              class_name: str, 
                              stage_config: Dict[str, Any],
                              module_name: str) -> str:
        """
        Generate a stage module.
        
        Args:
            class_name: Name of the stage class
            stage_config: Stage configuration
            module_name: Original component's module name
            
        Returns:
            Content for the stage module file
        """
        # Extract method name from config
        method_name = stage_config["config"]["method_name"]
        
        # Get input parameters
        input_mapping = stage_config["config"]["input_mapping"]
        input_keys = list(input_mapping.keys())
        
        return f'''"""
Pipeline stage for {method_name} method from {self.component_name}.
Generated by pipeline_converter.py
"""

import logging
from typing import Any, Dict, Optional

from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext
from {module_name} import {self.component_name}

logger = logging.getLogger(__name__)

class {class_name}(PipelineStage):
    """
    Pipeline stage that wraps the {method_name} method from {self.component_name}.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the stage.
        
        Args:
            config: Optional configuration
        """
        super().__init__(config or {})
        self.component = {self.component_name}()
        self.input_mapping = self.config.get("input_mapping", {input_mapping})
    
    def validate_input(self, context: PipelineContext) -> bool:
        """
        Validate that required inputs are in the context.
        
        Args:
            context: Pipeline context
            
        Returns:
            True if validation passes, False otherwise
        """
        for ctx_key in self.input_mapping.keys():
            if ctx_key not in context.data:
                context.add_error(self.name, f"Missing required input: {{ctx_key}}")
                return False
        return True
    
    async def process(self, context: PipelineContext) -> bool:
        """
        Process the stage by calling the component method.
        
        Args:
            context: Pipeline context
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Map context data to method parameters
            params = {{}}
            for ctx_key, param_name in self.input_mapping.items():
                params[param_name] = context.get(ctx_key)
            
            # Call the component method
            method = getattr(self.component, "{method_name}")
            if asyncio.iscoroutinefunction(method):
                result = await method(**params)
            else:
                result = method(**params)
            
            # Store the result
            if isinstance(result, dict):
                context.update(result)
            else:
                context.set("result", result)
            
            # Check for success indicator
            if isinstance(result, dict) and "success" in result:
                return result["success"]
            
            return True
            
        except Exception as e:
            return self.handle_error(context, e)
'''
        
    def _generate_pipeline_module(self) -> str:
        """Generate the pipeline module."""
        # Get stage imports
        imports = []
        stage_instances = []
        
        for stage in self.pipeline_config["stages"]:
            class_name = stage["class"]
            stage_name = stage["name"]
            imports.append(f"from .{stage_name} import {class_name}")
            stage_instances.append(
                f"            {class_name}(self.config.get(\"{stage_name}\", {{}}))"
            )
            
        imports_str = "\n".join(imports)
        stages_str = ",\n".join(stage_instances)
        
        return f'''"""
Pipeline implementation for {self.component_name}.
Generated by pipeline_converter.py
"""

import logging
from typing import Any, Dict, Optional

from core.pipeline.pipeline import Pipeline
from core.pipeline.context import PipelineContext

{imports_str}

logger = logging.getLogger(__name__)

class {self.component_name}Pipeline(Pipeline):
    """
    Pipeline implementation for {self.component_name}.
    """
    
    def __init__(self, name: str = "{self.pipeline_config['name']}", config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the pipeline.
        
        Args:
            name: Pipeline name
            config: Optional configuration
        """
        super().__init__(name, config or {{}})
        self._setup_stages()
    
    def _setup_stages(self) -> None:
        """Set up the pipeline stages."""
        self.add_stages([
{stages_str}
        ])
'''
        
    def _get_module_name(self) -> str:
        """Get the module name for imports."""
        if hasattr(self.config_generator, 'module_path') and self.config_generator.module_path:
            return self.config_generator.module_path
        return "unknown.module"  # Fallback


class InteractiveConverter:
    """Interactive CLI for guided migration."""
    
    def __init__(self):
        """Initialize the interactive converter."""
        self.analyzer = None
        self.config_generator = None
        self.stage_generator = None
        
    def run(self) -> None:
        """Run the interactive converter."""
        print("\n===== SmartScrape Pipeline Converter =====\n")
        print("This tool will help you convert legacy components to pipeline stages.")
        
        while True:
            print("\nOptions:")
            print("1. Analyze component file")
            print("2. Generate pipeline configuration")
            print("3. Generate pipeline stages")
            print("4. Full conversion workflow")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == "1":
                self._run_analyze()
            elif choice == "2":
                self._run_generate_config()
            elif choice == "3":
                self._run_generate_stages()
            elif choice == "4":
                self._run_full_workflow()
            elif choice == "5":
                print("\nExiting converter. Goodbye!")
                break
            else:
                print("\nInvalid choice. Please try again.")
                
    def _run_analyze(self) -> None:
        """Run the analyzer interactively."""
        file_path = input("\nEnter file path to analyze: ")
        
        try:
            self.analyzer = PipelineCandidateAnalyzer(file_path)
            candidates = self.analyzer.analyze()
            
            if not candidates:
                print("\nNo pipeline candidates found in the file.")
                return
                
            print(f"\nFound {len(candidates)} pipeline candidates:")
            for i, candidate in enumerate(candidates, 1):
                score = candidate.get("pipeline_score", 0)
                print(f"{i}. {candidate['name']} (Score: {score}/10)")
                
                # Show characteristics
                chars = candidate.get("characteristics", {})
                print(f"   - Linear flow: {'Yes' if chars.get('linear_flow') else 'No'}")
                print(f"   - Error handling: {'Yes' if chars.get('error_handling') else 'No'}")
                print(f"   - Multi-stage: {'Yes' if chars.get('multi_stage') else 'No'}")
                print(f"   - Complex state: {'Yes' if chars.get('complex_state') else 'No'}")
                
        except Exception as e:
            print(f"\nError analyzing file: {str(e)}")
            
    def _run_generate_config(self) -> None:
        """Generate pipeline configuration interactively."""
        component_name = input("\nEnter component class name: ")
        module_path = input("Enter module path (or leave empty to auto-detect): ")
        
        if not module_path.strip():
            module_path = None
            
        try:
            self.config_generator = PipelineConfigGenerator(component_name, module_path)
            config = self.config_generator.generate_config()
            
            print("\nGenerated pipeline configuration:")
            print(json.dumps(config, indent=2))
            
            # Offer to save
            save = input("\nSave configuration to file? (y/n): ")
            if save.lower() == 'y':
                file_path = input("Enter file path: ")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                print(f"Configuration saved to {file_path}")
                
        except Exception as e:
            print(f"\nError generating configuration: {str(e)}")
            
    def _run_generate_stages(self) -> None:
        """Generate pipeline stages interactively."""
        component_name = input("\nEnter component class name: ")
        module_path = input("Enter module path (or leave empty to auto-detect): ")
        output_dir = input("Enter output directory (or leave empty for default): ")
        
        if not module_path.strip():
            module_path = None
            
        if not output_dir.strip():
            output_dir = DEFAULT_OUTPUT_DIR
            
        try:
            self.stage_generator = PipelineStageGenerator(
                component_name, 
                module_path, 
                output_dir
            )
            
            generated_files = self.stage_generator.generate_stages()
            
            print(f"\nGenerated {len(generated_files)} files:")
            for path in generated_files:
                print(f"- {path}")
                
        except Exception as e:
            print(f"\nError generating stages: {str(e)}")
            
    def _run_full_workflow(self) -> None:
        """Run the full conversion workflow interactively."""
        file_path = input("\nEnter file path to analyze: ")
        
        try:
            # Step 1: Analyze file
            print("\n==== Step 1: Analyzing file ====")
            self.analyzer = PipelineCandidateAnalyzer(file_path)
            candidates = self.analyzer.analyze()
            
            if not candidates:
                print("\nNo pipeline candidates found in the file.")
                return
                
            print(f"\nFound {len(candidates)} pipeline candidates:")
            for i, candidate in enumerate(candidates, 1):
                score = candidate.get("pipeline_score", 0)
                print(f"{i}. {candidate['name']} (Score: {score}/10)")
                
            # Step 2: Select component
            choice = input("\nSelect component to convert (number): ")
            try:
                idx = int(choice) - 1
                if idx < 0 or idx >= len(candidates):
                    print("Invalid selection.")
                    return
                selected = candidates[idx]
            except ValueError:
                print("Invalid input. Please enter a number.")
                return
                
            component_name = selected["name"]
            module_path = selected["module"]
            
            # Step 3: Generate configuration
            print(f"\n==== Step 2: Generating configuration for {component_name} ====")
            self.config_generator = PipelineConfigGenerator(component_name, module_path)
            config = self.config_generator.generate_config()
            
            print("\nGenerated pipeline configuration:")
            print(json.dumps(config, indent=2))
            
            # Step 4: Generate stages
            print(f"\n==== Step 3: Generating pipeline stages ====")
            output_dir = input("Enter output directory (or leave empty for default): ")
            
            if not output_dir.strip():
                output_dir = os.path.join(DEFAULT_OUTPUT_DIR, component_name.lower())
                
            os.makedirs(output_dir, exist_ok=True)
            
            self.stage_generator = PipelineStageGenerator(
                component_name, 
                module_path, 
                output_dir
            )
            
            generated_files = self.stage_generator.generate_stages()
            
            print(f"\nGenerated {len(generated_files)} files:")
            for path in generated_files:
                print(f"- {path}")
                
            print("\nConversion complete!")
            
        except Exception as e:
            print(f"\nError in conversion workflow: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pipeline Converter Utility")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a file for pipeline candidates")
    analyze_parser.add_argument("--path", required=True, help="Path to file to analyze")
    
    # Generate config command
    config_parser = subparsers.add_parser("generate-config", help="Generate pipeline config")
    config_parser.add_argument("--component", required=True, help="Component class name")
    config_parser.add_argument("--module", help="Module path (optional)")
    config_parser.add_argument("--output", help="Output file path (optional)")
    
    # Generate stages command
    stages_parser = subparsers.add_parser("generate-stages", help="Generate pipeline stages")
    stages_parser.add_argument("--component", required=True, help="Component class name")
    stages_parser.add_argument("--module", help="Module path (optional)")
    stages_parser.add_argument("--output-dir", help="Output directory (optional)")
    
    # Interactive command
    subparsers.add_parser("interactive", help="Run interactive mode")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        analyzer = PipelineCandidateAnalyzer(args.path)
        candidates = analyzer.analyze()
        
        print(f"Found {len(candidates)} pipeline candidates:")
        for candidate in candidates:
            print(f"  - {candidate['name']} (Score: {candidate.get('pipeline_score', 0)}/10)")
            
    elif args.command == "generate-config":
        generator = PipelineConfigGenerator(args.component, args.module)
        config = generator.generate_config()
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            print(f"Configuration saved to {args.output}")
        else:
            print(json.dumps(config, indent=2))
            
    elif args.command == "generate-stages":
        generator = PipelineStageGenerator(
            args.component, 
            args.module, 
            args.output_dir
        )
        files = generator.generate_stages()
        
        print(f"Generated {len(files)} files:")
        for path in files:
            print(f"- {path}")
            
    elif args.command == "interactive":
        converter = InteractiveConverter()
        converter.run()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()