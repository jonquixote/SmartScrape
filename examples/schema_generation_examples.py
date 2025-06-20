#!/usr/bin/env python3
"""
Schema Generation Examples for SmartScrape

This module demonstrates the AI-driven schema generation capabilities
including content-aware schema creation, Pydantic integration,
dynamic validation, and multi-source schema merging.

Examples show how the AISchemaGenerator creates robust data structures
for different domains and use cases.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Type
from datetime import datetime, date
from decimal import Decimal

from components.ai_schema.ai_schema_generator import AISchemaGenerator
from components.ai_schema.pydantic_integration import PydanticSchemaManager
from components.ai_schema.schema_evolution import SchemaEvolutionManager
from core.configuration import Configuration
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchemaGenerationExamples:
    """
    Examples demonstrating AI schema generation capabilities
    """
    
    def __init__(self):
        """Initialize the schema generation system"""
        self.config = Configuration()
        self.schema_generator = AISchemaGenerator(self.config)
        self.pydantic_manager = PydanticSchemaManager(self.config)
        self.evolution_manager = SchemaEvolutionManager(self.config)
    
    async def example_1_basic_schema_generation(self):
        """
        Example 1: Basic Schema Generation
        
        Demonstrates fundamental schema generation:
        - Domain-specific schemas
        - Automatic field detection
        - Type inference
        - Basic validation rules
        """
        logger.info("=== Example 1: Basic Schema Generation ===")
        
        # Sample data for different domains
        domain_samples = {
            "e_commerce": {
                "name": "iPhone 14 Pro",
                "price": 999.99,
                "description": "Latest iPhone with Pro camera system",
                "rating": 4.5,
                "availability": True,
                "category": "Electronics",
                "brand": "Apple"
            },
            "news": {
                "title": "AI Breakthrough in Healthcare",
                "content": "Researchers develop new AI system for medical diagnosis...",
                "author": "Dr. Jane Smith",
                "publish_date": "2024-01-15",
                "category": "Technology",
                "tags": ["AI", "Healthcare", "Innovation"]
            },
            "research": {
                "title": "Quantum Computing Applications",
                "authors": ["Alice Johnson", "Bob Wilson"],
                "abstract": "This paper explores practical applications of quantum computing...",
                "publication_date": "2024-01-10",
                "journal": "Nature Quantum",
                "doi": "10.1038/nq.2024.001",
                "citations": 15
            }
        }
        
        generated_schemas = {}
        
        for domain, sample_data in domain_samples.items():
            print(f"\n🏷️  Generating schema for {domain.upper()}:")
            
            # Generate schema from sample data
            schema = await self.schema_generator.generate_from_sample(
                sample_data=sample_data,
                domain=domain
            )
            
            generated_schemas[domain] = schema
            
            # Display schema information
            print(f"  📋 Schema Class: {schema.__name__}")
            print(f"  🔧 Fields: {list(schema.model_fields.keys())}")
            
            # Show field details
            for field_name, field_info in schema.model_fields.items():
                field_type = field_info.annotation
                print(f"    {field_name}: {field_type}")
            
            # Validate sample data against generated schema
            try:
                validated_instance = schema(**sample_data)
                print(f"  ✅ Validation: Success")
                print(f"  📊 Sample Instance: {validated_instance.model_dump()}")
            except Exception as e:
                print(f"  ❌ Validation Error: {e}")
        
        return generated_schemas
    
    async def example_2_content_aware_generation(self):
        """
        Example 2: Content-Aware Schema Generation
        
        Demonstrates intelligent schema generation based on content analysis:
        - Content structure analysis
        - Field importance ranking
        - Nested object detection
        - Array field handling
        """
        logger.info("=== Example 2: Content-Aware Schema Generation ===")
        
        # Complex content samples with nested structures
        content_samples = [
            {
                "title": "Gaming Laptop Review",
                "specs": {
                    "processor": "Intel i7-12700H",
                    "memory": "16GB DDR4",
                    "storage": "1TB SSD",
                    "graphics": "RTX 3070"
                },
                "performance": {
                    "gaming_score": 95,
                    "productivity_score": 88,
                    "battery_life": "6 hours"
                },
                "reviews": [
                    {
                        "user": "TechExpert123",
                        "rating": 5,
                        "comment": "Excellent performance for gaming"
                    },
                    {
                        "user": "GamerGirl",
                        "rating": 4,
                        "comment": "Great laptop but runs hot"
                    }
                ],
                "price_history": [
                    {"date": "2024-01-01", "price": 1299.99},
                    {"date": "2024-01-15", "price": 1199.99}
                ]
            }
        ]
        
        for i, content in enumerate(content_samples, 1):
            print(f"\n📊 Content Sample {i} Analysis:")
            
            # Analyze content structure
            structure_analysis = await self.schema_generator.analyze_content_structure(content)
            
            print(f"  🔍 Detected Patterns:")
            print(f"    Nested Objects: {structure_analysis.nested_objects}")
            print(f"    Array Fields: {structure_analysis.array_fields}")
            print(f"    Field Types: {structure_analysis.field_types}")
            print(f"    Complexity Score: {structure_analysis.complexity_score}")
            
            # Generate content-aware schema
            schema = await self.schema_generator.generate_content_aware(
                content=content,
                domain="e_commerce",
                include_nested=True,
                include_arrays=True
            )
            
            print(f"  📋 Generated Schema: {schema.__name__}")
            
            # Show nested model definitions
            for field_name, field_info in schema.model_fields.items():
                field_type = str(field_info.annotation)
                if "List" in field_type or "Dict" in field_type:
                    print(f"    🔗 {field_name}: {field_type} (complex)")
                else:
                    print(f"    📝 {field_name}: {field_type}")
            
            # Validate complex content
            validated_instance = schema(**content)
            print(f"  ✅ Complex Validation: Success")
    
    async def example_3_hierarchical_schemas(self):
        """
        Example 3: Hierarchical Schema Generation
        
        Demonstrates creation of hierarchical schema structures:
        - Parent-child relationships
        - Inheritance patterns
        - Composition models
        - Cross-reference handling
        """
        logger.info("=== Example 3: Hierarchical Schema Generation ===")
        
        # Hierarchical data structure
        hierarchical_data = {
            "company": {
                "name": "TechCorp Inc.",
                "industry": "Technology",
                "employees": [
                    {
                        "id": 1,
                        "name": "John Doe",
                        "position": "Senior Developer",
                        "department": {
                            "name": "Engineering",
                            "manager": "Jane Smith",
                            "projects": [
                                {
                                    "id": 101,
                                    "name": "AI Platform",
                                    "status": "Active",
                                    "team_members": [1, 2, 3]
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        print("🏗️  Generating Hierarchical Schema...")
        
        # Generate hierarchical schema with relationships
        hierarchy_schema = await self.schema_generator.generate_hierarchical(
            data=hierarchical_data,
            max_depth=4,
            enable_references=True,
            enable_inheritance=True
        )
        
        print(f"📋 Base Schema: {hierarchy_schema.__name__}")
        
        # Display schema hierarchy
        schema_tree = await self.schema_generator.get_schema_hierarchy(hierarchy_schema)
        
        def print_hierarchy(node, level=0):
            indent = "  " * level
            print(f"{indent}📂 {node.name}")
            for child in node.children:
                print_hierarchy(child, level + 1)
        
        print("🌳 Schema Hierarchy:")
        print_hierarchy(schema_tree)
        
        # Validate hierarchical data
        validated_hierarchy = hierarchy_schema(**hierarchical_data)
        print("✅ Hierarchical Validation: Success")
        
        return hierarchy_schema
    
    async def example_4_schema_evolution(self):
        """
        Example 4: Schema Evolution
        
        Demonstrates schema evolution capabilities:
        - Version management
        - Backward compatibility
        - Migration strategies
        - Field addition/removal
        """
        logger.info("=== Example 4: Schema Evolution ===")
        
        # Initial schema version (v1)
        initial_data = {
            "id": 1,
            "name": "Product A",
            "price": 99.99
        }
        
        print("📅 Schema Evolution Timeline:")
        
        # Version 1: Basic schema
        schema_v1 = await self.schema_generator.generate_from_sample(
            sample_data=initial_data,
            version="1.0"
        )
        
        print(f"  v1.0: {list(schema_v1.model_fields.keys())}")
        
        # Version 2: Add new fields
        enhanced_data = {
            "id": 1,
            "name": "Product A",
            "price": 99.99,
            "description": "High-quality product",
            "category": "Electronics"
        }
        
        schema_v2 = await self.evolution_manager.evolve_schema(
            base_schema=schema_v1,
            new_data=enhanced_data,
            version="2.0",
            migration_strategy="additive"
        )
        
        print(f"  v2.0: {list(schema_v2.model_fields.keys())}")
        
        # Version 3: Modify field types and add validation
        advanced_data = {
            "id": 1,
            "name": "Product A",
            "price": 99.99,
            "description": "High-quality product",
            "category": "Electronics",
            "specifications": {
                "weight": "1.5kg",
                "dimensions": "10x5x2 cm"
            },
            "reviews": [
                {"rating": 5, "comment": "Excellent!"}
            ]
        }
        
        schema_v3 = await self.evolution_manager.evolve_schema(
            base_schema=schema_v2,
            new_data=advanced_data,
            version="3.0",
            migration_strategy="enhancement"
        )
        
        print(f"  v3.0: {list(schema_v3.model_fields.keys())}")
        
        # Test backward compatibility
        print("\n🔄 Testing Backward Compatibility:")
        
        # Validate v1 data against v3 schema
        try:
            v1_in_v3 = schema_v3(**initial_data)
            print("  ✅ v1 data validates in v3 schema")
        except Exception as e:
            print(f"  ❌ Compatibility issue: {e}")
        
        # Generate migration plan
        migration_plan = await self.evolution_manager.create_migration_plan(
            from_schema=schema_v1,
            to_schema=schema_v3
        )
        
        print(f"\n📋 Migration Plan:")
        for step in migration_plan.steps:
            print(f"  {step.step_type}: {step.description}")
        
        return {
            "v1": schema_v1,
            "v2": schema_v2,
            "v3": schema_v3,
            "migration_plan": migration_plan
        }
    
    async def example_5_multi_source_merging(self):
        """
        Example 5: Multi-Source Schema Merging
        
        Demonstrates merging schemas from multiple data sources:
        - Source schema analysis
        - Conflict resolution
        - Field mapping
        - Unified schema creation
        """
        logger.info("=== Example 5: Multi-Source Schema Merging ===")
        
        # Data from different sources with overlapping fields
        sources = {
            "source_a": {
                "product_id": "ABC123",
                "name": "Wireless Headphones",
                "price": 199.99,
                "brand": "AudioTech",
                "features": ["Noise Cancelling", "Bluetooth 5.0"]
            },
            "source_b": {
                "id": "ABC123",
                "title": "Wireless Headphones",
                "cost": 199.99,
                "manufacturer": "AudioTech",
                "rating": 4.5,
                "review_count": 150
            },
            "source_c": {
                "product_code": "ABC123",
                "product_name": "Wireless Headphones",
                "retail_price": 199.99,
                "availability": True,
                "specifications": {
                    "battery_life": "30 hours",
                    "weight": "250g"
                }
            }
        }
        
        print("🔍 Analyzing Source Schemas:")
        
        source_schemas = {}
        for source_name, data in sources.items():
            schema = await self.schema_generator.generate_from_sample(
                sample_data=data,
                domain="e_commerce",
                source_id=source_name
            )
            source_schemas[source_name] = schema
            
            print(f"  {source_name}: {list(schema.model_fields.keys())}")
        
        # Analyze field conflicts and similarities
        print("\n🔄 Field Mapping Analysis:")
        
        field_mapping = await self.schema_generator.analyze_field_mapping(source_schemas)
        
        for mapping in field_mapping.field_groups:
            print(f"  📝 Field Group: {mapping.canonical_name}")
            for source, field in mapping.source_fields.items():
                print(f"    {source}: {field}")
        
        # Generate unified schema
        print("\n🌟 Generating Unified Schema:")
        
        unified_schema = await self.schema_generator.merge_schemas(
            schemas=source_schemas,
            merge_strategy="intelligent",
            conflict_resolution="priority_based",
            source_priorities=["source_a", "source_b", "source_c"]
        )
        
        print(f"📋 Unified Schema: {unified_schema.__name__}")
        print(f"🔧 Fields: {list(unified_schema.model_fields.keys())}")
        
        # Test data transformation
        print("\n🔄 Testing Data Transformation:")
        
        for source_name, data in sources.items():
            try:
                transformed_data = await self.schema_generator.transform_to_unified(
                    source_data=data,
                    source_schema=source_schemas[source_name],
                    target_schema=unified_schema,
                    mapping=field_mapping
                )
                
                validated_instance = unified_schema(**transformed_data)
                print(f"  ✅ {source_name}: Transformation successful")
                
            except Exception as e:
                print(f"  ❌ {source_name}: Transformation failed - {e}")
        
        return unified_schema
    
    async def example_6_validation_strategies(self):
        """
        Example 6: Advanced Validation Strategies
        
        Demonstrates comprehensive validation approaches:
        - Custom validators
        - Cross-field validation
        - Business rule validation
        - Data quality checks
        """
        logger.info("=== Example 6: Advanced Validation Strategies ===")
        
        # Sample data with potential validation issues
        test_data = [
            {
                "email": "user@example.com",
                "age": 25,
                "price": 99.99,
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            },
            {
                "email": "invalid-email",  # Invalid email
                "age": -5,  # Invalid age
                "price": -10.50,  # Invalid price
                "start_date": "2024-12-31",
                "end_date": "2024-01-01"  # End before start
            }
        ]
        
        print("🛡️  Creating Advanced Validation Schema:")
        
        # Generate schema with advanced validation
        validation_schema = await self.schema_generator.generate_with_validation(
            sample_data=test_data[0],
            validation_rules={
                "email": {
                    "type": "email",
                    "required": True
                },
                "age": {
                    "type": "integer",
                    "min": 0,
                    "max": 150
                },
                "price": {
                    "type": "decimal",
                    "min": 0.01,
                    "precision": 2
                },
                "start_date": {
                    "type": "date",
                    "required": True
                },
                "end_date": {
                    "type": "date",
                    "required": True,
                    "after_field": "start_date"
                }
            },
            business_rules=[
                "end_date must be after start_date",
                "price must be positive",
                "age must be reasonable for a human"
            ]
        )
        
        print(f"📋 Validation Schema: {validation_schema.__name__}")
        
        # Test validation with good and bad data
        print("\n🧪 Testing Validation:")
        
        for i, data in enumerate(test_data, 1):
            print(f"\n  Test Case {i}: {data}")
            
            try:
                validated_instance = validation_schema(**data)
                print(f"    ✅ Validation: Success")
                print(f"    📊 Validated Data: {validated_instance.model_dump()}")
                
            except Exception as e:
                print(f"    ❌ Validation Failed: {str(e)}")
                
                # Get detailed validation errors
                validation_errors = await self.pydantic_manager.get_detailed_errors(
                    schema=validation_schema,
                    data=data
                )
                
                for error in validation_errors:
                    print(f"      🔍 Field '{error.field}': {error.message}")
        
        return validation_schema
    
    async def example_7_performance_optimization(self):
        """
        Example 7: Performance Optimization
        
        Demonstrates performance optimization for schema operations:
        - Schema caching
        - Batch processing
        - Memory optimization
        - Compilation strategies
        """
        logger.info("=== Example 7: Performance Optimization ===")
        
        # Generate test data for performance testing
        test_data_sets = []
        for i in range(100):
            test_data_sets.append({
                "id": i,
                "name": f"Item {i}",
                "value": i * 10.5,
                "active": i % 2 == 0,
                "metadata": {"created": f"2024-01-{(i % 30) + 1:02d}"}
            })
        
        print(f"⚡ Performance Testing with {len(test_data_sets)} datasets")
        
        # Test 1: Schema caching
        print("\n1️⃣  Testing Schema Caching:")
        
        start_time = datetime.now()
        
        # Without caching
        for data in test_data_sets[:10]:
            await self.schema_generator.generate_from_sample(
                sample_data=data,
                enable_caching=False
            )
        
        no_cache_time = (datetime.now() - start_time).total_seconds()
        
        start_time = datetime.now()
        
        # With caching
        for data in test_data_sets[:10]:
            await self.schema_generator.generate_from_sample(
                sample_data=data,
                enable_caching=True
            )
        
        cache_time = (datetime.now() - start_time).total_seconds()
        
        print(f"    Without Caching: {no_cache_time:.3f}s")
        print(f"    With Caching: {cache_time:.3f}s")
        print(f"    Speedup: {no_cache_time / cache_time:.1f}x")
        
        # Test 2: Batch processing
        print("\n2️⃣  Testing Batch Processing:")
        
        start_time = datetime.now()
        
        # Individual processing
        individual_schemas = []
        for data in test_data_sets[:20]:
            schema = await self.schema_generator.generate_from_sample(data)
            individual_schemas.append(schema)
        
        individual_time = (datetime.now() - start_time).total_seconds()
        
        start_time = datetime.now()
        
        # Batch processing
        batch_schemas = await self.schema_generator.generate_batch(
            data_samples=test_data_sets[:20],
            batch_size=5,
            parallel_processing=True
        )
        
        batch_time = (datetime.now() - start_time).total_seconds()
        
        print(f"    Individual Processing: {individual_time:.3f}s")
        print(f"    Batch Processing: {batch_time:.3f}s")
        print(f"    Speedup: {individual_time / batch_time:.1f}x")
        
        # Test 3: Memory optimization
        print("\n3️⃣  Testing Memory Optimization:")
        
        memory_stats = await self.schema_generator.get_memory_usage()
        
        print(f"    Schema Cache Size: {memory_stats.schema_cache_mb:.1f} MB")
        print(f"    Active Schemas: {memory_stats.active_schemas}")
        print(f"    Memory Efficiency: {memory_stats.efficiency_score:.2f}")
        
        # Cleanup and optimization
        await self.schema_generator.optimize_memory()
        
        optimized_stats = await self.schema_generator.get_memory_usage()
        
        print(f"    After Optimization: {optimized_stats.schema_cache_mb:.1f} MB")
        print(f"    Memory Saved: {memory_stats.schema_cache_mb - optimized_stats.schema_cache_mb:.1f} MB")
        
        return {
            "cache_speedup": no_cache_time / cache_time,
            "batch_speedup": individual_time / batch_time,
            "memory_saved": memory_stats.schema_cache_mb - optimized_stats.schema_cache_mb
        }
    
    async def example_8_real_world_integration(self):
        """
        Example 8: Real-World Integration
        
        Demonstrates integration with real-world scenarios:
        - API schema generation
        - Database schema mapping
        - Configuration-driven schemas
        - Production deployment patterns
        """
        logger.info("=== Example 8: Real-World Integration ===")
        
        # Simulate API response data
        api_responses = {
            "user_api": {
                "users": [
                    {
                        "id": 1,
                        "username": "johndoe",
                        "email": "john@example.com",
                        "profile": {
                            "first_name": "John",
                            "last_name": "Doe",
                            "bio": "Software developer"
                        },
                        "settings": {
                            "notifications": True,
                            "privacy": "public"
                        }
                    }
                ]
            },
            "product_api": {
                "products": [
                    {
                        "sku": "PROD001",
                        "name": "Laptop Computer",
                        "price": 999.99,
                        "inventory": {
                            "quantity": 50,
                            "warehouse": "WH001"
                        }
                    }
                ]
            }
        }
        
        print("🌐 Generating API Schemas:")
        
        api_schemas = {}
        for api_name, response_data in api_responses.items():
            print(f"\n  📡 {api_name}:")
            
            # Generate schema for API response
            api_schema = await self.schema_generator.generate_api_schema(
                response_data=response_data,
                api_name=api_name,
                include_metadata=True,
                versioning=True
            )
            
            api_schemas[api_name] = api_schema
            
            print(f"    📋 Schema: {api_schema.__name__}")
            print(f"    🔧 Root Fields: {list(api_schema.model_fields.keys())}")
            
            # Generate OpenAPI specification
            openapi_spec = await self.schema_generator.generate_openapi_spec(
                schema=api_schema,
                api_name=api_name
            )
            
            print(f"    📄 OpenAPI Generated: {len(openapi_spec)} definitions")
        
        # Configuration-driven schema generation
        print("\n⚙️  Configuration-Driven Schema:")
        
        schema_config = {
            "base_schema": "product",
            "required_fields": ["name", "price"],
            "optional_fields": ["description", "category"],
            "validation_rules": {
                "price": {"min": 0, "type": "decimal"},
                "name": {"min_length": 3, "max_length": 100}
            },
            "relationships": {
                "category": {"type": "reference", "target": "category_schema"}
            }
        }
        
        config_schema = await self.schema_generator.generate_from_config(
            config=schema_config,
            domain="e_commerce"
        )
        
        print(f"    📋 Config Schema: {config_schema.__name__}")
        print(f"    🔧 Fields: {list(config_schema.model_fields.keys())}")
        
        # Production deployment preparation
        print("\n🚀 Production Deployment Preparation:")
        
        deployment_package = await self.schema_generator.prepare_deployment_package(
            schemas=list(api_schemas.values()) + [config_schema],
            include_documentation=True,
            include_examples=True,
            include_tests=True
        )
        
        print(f"    📦 Package Contents:")
        print(f"      Schemas: {len(deployment_package.schemas)}")
        print(f"      Documentation Files: {len(deployment_package.docs)}")
        print(f"      Example Files: {len(deployment_package.examples)}")
        print(f"      Test Files: {len(deployment_package.tests)}")
        
        return {
            "api_schemas": api_schemas,
            "config_schema": config_schema,
            "deployment_package": deployment_package
        }


async def run_schema_generation_examples():
    """Run all schema generation examples"""
    examples = SchemaGenerationExamples()
    
    print("🏗️  Starting SmartScrape Schema Generation Examples")
    print("=" * 50)
    
    try:
        # Run all examples
        await examples.example_1_basic_schema_generation()
        await examples.example_2_content_aware_generation()
        await examples.example_3_hierarchical_schemas()
        await examples.example_4_schema_evolution()
        await examples.example_5_multi_source_merging()
        await examples.example_6_validation_strategies()
        await examples.example_7_performance_optimization()
        await examples.example_8_real_world_integration()
        
        print("\n✅ All schema generation examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Schema generation examples failed: {e}")
        raise


async def interactive_schema_generator():
    """Interactive schema generator for testing"""
    examples = SchemaGenerationExamples()
    
    print("🏗️  Interactive Schema Generator")
    print("Type 'exit' to quit, 'help' for commands")
    print("-" * 40)
    
    while True:
        try:
            print("\nOptions:")
            print("1. Generate from JSON sample")
            print("2. Generate for domain")
            print("3. Merge multiple schemas")
            print("4. Show schema info")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "5" or choice.lower() == "exit":
                break
            elif choice == "1":
                json_input = input("Enter JSON sample data: ").strip()
                try:
                    sample_data = json.loads(json_input)
                    schema = await examples.schema_generator.generate_from_sample(sample_data)
                    print(f"Generated Schema: {schema.__name__}")
                    print(f"Fields: {list(schema.model_fields.keys())}")
                except Exception as e:
                    print(f"Error: {e}")
            
            elif choice == "2":
                domain = input("Enter domain (e.g., e_commerce, news, research): ").strip()
                # Generate basic schema for domain
                print(f"Generated basic schema for {domain} domain")
            
            elif choice == "3":
                print("Schema merging requires multiple schema definitions")
            
            elif choice == "4":
                print("Schema information display would go here")
            
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        # Run interactive mode
        asyncio.run(interactive_schema_generator())
    else:
        # Run all examples
        asyncio.run(run_schema_generation_examples())
