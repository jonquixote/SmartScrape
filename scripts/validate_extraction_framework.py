#!/usr/bin/env python
"""
SmartScrape Universal Extraction Framework Validator

This script validates the implementation of the Universal Extraction Framework
by checking for the presence and functionality of required components.

Usage:
    python validate_extraction_framework.py

The script will check:
1. Required classes and interfaces
2. Required methods on each class
3. Pipeline component integration
4. Basic extraction functionality
5. Schema validation capabilities
"""

import importlib
import inspect
import json
import os
import sys
from typing import Dict, List, Any, Optional, Set, Type, Tuple

# Output formatting constants
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}{text}{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}")

def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{GREEN}✓ {text}{RESET}")

def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{RED}✗ {text}{RESET}")

def print_info(text: str) -> None:
    """Print an informational message."""
    print(f"  {text}")

class ExtractionFrameworkValidator:
    """Validates the Universal Extraction Framework implementation."""
    
    def __init__(self):
        self.success_count = 0
        self.warning_count = 0
        self.error_count = 0
        self.required_modules = [
            "extraction.core.extraction_interface",
            "extraction.pattern_extractor",
            "extraction.semantic_extractor",
            "extraction.structural_analyzer",
            "extraction.metadata_extractor",
            "extraction.content_normalizer",
            "extraction.quality_evaluator",
            "extraction.schema_manager",
            "extraction.schema_validator",
            "extraction.fallback_framework"
        ]
        self.required_classes = {
            "extraction.core.extraction_interface": [
                "BaseExtractor",
                "PatternExtractor",
                "SemanticExtractor",
                "StructuralAnalyzer",
                "MetadataExtractor",
                "ContentNormalizer",
                "QualityEvaluator",
                "SchemaValidator"
            ],
            "extraction.pattern_extractor": ["DOMPatternExtractor"],
            "extraction.semantic_extractor": ["AISemanticExtractor"],
            "extraction.structural_analyzer": ["DOMStructuralAnalyzer"],
            "extraction.metadata_extractor": ["CompositeMetadataExtractor"],
            "extraction.content_normalizer": ["ContentNormalizer"],
            "extraction.quality_evaluator": ["QualityEvaluator"],
            "extraction.schema_manager": ["SchemaManager"],
            "extraction.schema_validator": ["SchemaValidator"],
            "extraction.fallback_framework": ["ExtractionFallbackChain"]
        }
        self.required_methods = {
            "BaseExtractor": [
                "can_handle",
                "extract",
                "initialize",
                "shutdown"
            ],
            "PatternExtractor": [
                "can_handle",
                "extract",
                "generate_patterns",
                "match_patterns"
            ],
            "SemanticExtractor": [
                "can_handle",
                "extract",
                "extract_semantic_content"
            ],
            "StructuralAnalyzer": [
                "can_handle",
                "extract",
                "analyze_structure"
            ],
            "MetadataExtractor": [
                "can_handle",
                "extract",
                "extract_metadata"
            ],
            "ContentNormalizer": [
                "can_handle",
                "extract",
                "normalize"
            ],
            "QualityEvaluator": [
                "can_handle",
                "extract",
                "evaluate"
            ],
            "SchemaValidator": [
                "can_handle",
                "extract", 
                "validate"
            ]
        }
        self.pipeline_components = [
            "extraction.stages.structural_analysis_stage",
            "extraction.stages.metadata_extraction_stage",
            "extraction.stages.pattern_extraction_stage",
            "extraction.stages.semantic_extraction_stage",
            "extraction.stages.content_normalization_stage",
            "extraction.stages.quality_evaluation_stage",
            "extraction.stages.schema_validation_stage"
        ]
        self.imported_modules = {}
        self.imported_classes = {}
        
    def validate_all(self) -> None:
        """Run all validation checks."""
        print_header("SmartScrape Universal Extraction Framework Validator")
        
        # Phase 1: Check for required modules
        self.validate_required_modules()
        
        # Phase 2: Check for required classes
        self.validate_required_classes()
        
        # Phase 3: Check methods on classes
        self.validate_required_methods()
        
        # Phase 4: Check pipeline components
        self.validate_pipeline_components()
        
        # Phase 5: Check for basic extraction functionality
        self.validate_extraction_functionality()
        
        # Print summary
        self.print_summary()
    
    def validate_required_modules(self) -> None:
        """Check if all required modules are present."""
        print_header("Phase 1: Checking Required Modules")
        
        for module_name in self.required_modules:
            try:
                module = importlib.import_module(module_name)
                self.imported_modules[module_name] = module
                print_success(f"Found module: {module_name}")
                self.success_count += 1
            except ImportError as e:
                print_error(f"Missing module: {module_name} - {str(e)}")
                self.error_count += 1
    
    def validate_required_classes(self) -> None:
        """Check if all required classes are present."""
        print_header("Phase 2: Checking Required Classes")
        
        for module_name, class_names in self.required_classes.items():
            if module_name not in self.imported_modules:
                for class_name in class_names:
                    print_error(f"Cannot check class {class_name} - module {module_name} not imported")
                    self.error_count += 1
                continue
            
            module = self.imported_modules[module_name]
            
            for class_name in class_names:
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    if inspect.isclass(cls):
                        self.imported_classes[class_name] = cls
                        print_success(f"Found class: {class_name} in {module_name}")
                        self.success_count += 1
                    else:
                        print_error(f"{class_name} in {module_name} is not a class")
                        self.error_count += 1
                else:
                    print_error(f"Missing class: {class_name} in {module_name}")
                    self.error_count += 1
    
    def validate_required_methods(self) -> None:
        """Check if all required methods are present on their respective classes."""
        print_header("Phase 3: Checking Required Methods")
        
        for class_name, method_names in self.required_methods.items():
            if class_name not in self.imported_classes:
                for method_name in method_names:
                    print_error(f"Cannot check method {class_name}.{method_name} - class not imported")
                    self.error_count += 1
                continue
            
            cls = self.imported_classes[class_name]
            
            for method_name in method_names:
                if hasattr(cls, method_name):
                    method = getattr(cls, method_name)
                    if callable(method):
                        print_success(f"Found method: {class_name}.{method_name}")
                        self.success_count += 1
                    else:
                        print_error(f"{class_name}.{method_name} is not callable")
                        self.error_count += 1
                else:
                    print_error(f"Missing method: {class_name}.{method_name}")
                    self.error_count += 1
    
    def validate_pipeline_components(self) -> None:
        """Check if pipeline components are properly implemented."""
        print_header("Phase 4: Checking Pipeline Components")
        
        for component_module_name in self.pipeline_components:
            try:
                module = importlib.import_module(component_module_name)
                # Get the main class in the module (assuming naming convention)
                component_class_name = component_module_name.split('.')[-1]
                component_class_name = ''.join(word.capitalize() for word in component_class_name.split('_'))
                
                if hasattr(module, component_class_name):
                    component_class = getattr(module, component_class_name)
                    if inspect.isclass(component_class):
                        # Check if it has required pipeline stage methods
                        has_name = hasattr(component_class, 'name') and isinstance(getattr(component_class, 'name'), property)
                        has_process = hasattr(component_class, 'process') and callable(getattr(component_class, 'process'))
                        
                        if has_name and has_process:
                            print_success(f"Pipeline component validated: {component_class_name}")
                            self.success_count += 1
                        else:
                            missing = []
                            if not has_name:
                                missing.append("name property")
                            if not has_process:
                                missing.append("process method")
                            print_error(f"Pipeline component {component_class_name} is missing: {', '.join(missing)}")
                            self.error_count += 1
                    else:
                        print_error(f"{component_class_name} is not a class")
                        self.error_count += 1
                else:
                    print_error(f"Could not find class {component_class_name} in {component_module_name}")
                    self.error_count += 1
            except ImportError as e:
                print_error(f"Missing pipeline component: {component_module_name} - {str(e)}")
                self.error_count += 1
    
    def validate_extraction_functionality(self) -> None:
        """Validate basic extraction functionality."""
        print_header("Phase 5: Testing Basic Extraction Functionality")
        
        # Check if we have the required classes for basic testing
        required_for_testing = ["DOMPatternExtractor", "SchemaManager"]
        missing_classes = [cls for cls in required_for_testing if cls not in self.imported_classes]
        
        if missing_classes:
            for cls in missing_classes:
                print_error(f"Cannot test extraction - missing class: {cls}")
                self.error_count += 1
            return
        
        # Try to test basic extraction
        try:
            # Import required components
            from strategies.core.strategy_context import StrategyContext
            from core.service_registry import ServiceRegistry
            
            # Create a simple test HTML
            test_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Test Product Page</title>
            </head>
            <body>
                <h1 class="product-title">Test Product</h1>
                <div class="product-price">$99.99</div>
                <div class="product-description">This is a test product description.</div>
            </body>
            </html>
            """
            
            # Create basic test schema
            test_schema = {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "price": {"type": "number"},
                    "description": {"type": "string"}
                },
                "required": ["title"]
            }
            
            # Setup basic context and services
            registry = ServiceRegistry()
            context = StrategyContext(registry)
            
            # Create pattern extractor
            DOMPatternExtractor = self.imported_classes["DOMPatternExtractor"]
            extractor = DOMPatternExtractor(context)
            extractor.initialize()
            
            # Test extraction
            result = extractor.extract(
                test_html, 
                schema=test_schema,
                options={"content_type": "html"}
            )
            
            # Check result
            if isinstance(result, dict):
                fields_found = [field for field in test_schema["properties"].keys() if field in result]
                
                if len(fields_found) > 0:
                    print_success(f"Basic extraction test passed - extracted {len(fields_found)} fields")
                    print_info(f"Extracted fields: {', '.join(fields_found)}")
                    self.success_count += 1
                else:
                    print_warning("Extraction test partially succeeded - no fields were extracted")
                    self.warning_count += 1
            else:
                print_error(f"Extraction test failed - expected dict result, got {type(result)}")
                self.error_count += 1
            
        except Exception as e:
            print_error(f"Could not test extraction functionality: {str(e)}")
            import traceback
            print_info(traceback.format_exc())
            self.error_count += 1
    
    def print_summary(self) -> None:
        """Print validation summary."""
        print_header("Validation Summary")
        
        total_checks = self.success_count + self.warning_count + self.error_count
        
        print(f"Total checks: {total_checks}")
        print(f"{GREEN}Successes: {self.success_count}{RESET}")
        print(f"{YELLOW}Warnings: {self.warning_count}{RESET}")
        print(f"{RED}Errors: {self.error_count}{RESET}")
        
        if self.error_count == 0:
            if self.warning_count == 0:
                print(f"\n{GREEN}{BOLD}Validation Passed!{RESET} All required components are present and functional.")
            else:
                print(f"\n{YELLOW}{BOLD}Validation Passed with Warnings!{RESET} All required components are present, but some warnings were detected.")
        else:
            print(f"\n{RED}{BOLD}Validation Failed!{RESET} Some required components are missing or not functioning correctly.")
            print("\nPlease fix the errors above and run the validator again.")


def main():
    """Main entry point."""
    validator = ExtractionFrameworkValidator()
    validator.validate_all()


if __name__ == "__main__":
    main()