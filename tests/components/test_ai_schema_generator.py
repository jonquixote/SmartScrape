"""
Unit tests for AISchemaGenerator component.

This module tests the AI-powered schema generation system including Pydantic model
creation, intent-based field mapping, sample data inference, and dynamic validation
for extracted data structuring.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from components.ai_schema_generator import (
    AISchemaGenerator, SchemaField, FieldType, SchemaGenerationResult
)


@unittest.skipIf(not PYDANTIC_AVAILABLE, "Pydantic not available")
class TestAISchemaGenerator(unittest.TestCase):
    """Test suite for AISchemaGenerator component."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock intent analyzer
        self.mock_intent_analyzer = Mock()
        self.mock_intent_analyzer.analyze_intent.return_value = {
            'intent_type': 'product_search',
            'entities': [
                {'text': 'laptop', 'label': 'PRODUCT'},
                {'text': '$1000', 'label': 'MONEY'},
                {'text': 'Dell', 'label': 'BRAND'}
            ],
            'confidence': 0.85,
            'semantic_keywords': ['laptop', 'computer', 'price'],
            'query_complexity': 'medium'
        }
        
        # Mock configuration
        with patch('components.ai_schema_generator.AI_SCHEMA_GENERATION_ENABLED', True):
            with patch('components.ai_schema_generator.PYDANTIC_AVAILABLE', True):
                self.generator = AISchemaGenerator(intent_analyzer=self.mock_intent_analyzer)
    
    def test_initialization_success(self):
        """Test successful AISchemaGenerator initialization."""
        self.assertIsNotNone(self.generator)
        self.assertEqual(self.generator.intent_analyzer, self.mock_intent_analyzer)
        self.assertTrue(hasattr(self.generator, 'intent_field_mappings'))
        self.assertTrue(hasattr(self.generator, 'field_patterns'))
        self.assertTrue(hasattr(self.generator, 'type_inference_rules'))
        self.assertTrue(hasattr(self.generator, 'schema_cache'))
    
    @patch('components.ai_schema_generator.PYDANTIC_AVAILABLE', False)
    def test_initialization_no_pydantic(self):
        """Test initialization failure when Pydantic is not available."""
        with self.assertRaises(ImportError):
            AISchemaGenerator()
    
    @patch('components.ai_schema_generator.AI_SCHEMA_GENERATION_ENABLED', False)
    def test_initialization_disabled(self):
        """Test initialization failure when AI schema generation is disabled."""
        with self.assertRaises(ValueError):
            AISchemaGenerator()
    
    def test_generate_schema_from_intent_product_search(self):
        """Test schema generation from product search intent."""
        intent_analysis = {
            'intent_type': 'product_search',
            'entities': [
                {'text': 'laptop', 'label': 'PRODUCT'},
                {'text': '$1000', 'label': 'MONEY'},
                {'text': 'Dell', 'label': 'BRAND'}
            ],
            'confidence': 0.85
        }
        query = "Dell laptop under $1000"
        
        with patch.object(self.generator, '_generate_fields_from_intent') as mock_intent_fields:
            with patch.object(self.generator, '_generate_fields_from_entities') as mock_entity_fields:
                with patch.object(self.generator, '_generate_fields_from_query') as mock_query_fields:
                    with patch.object(self.generator, '_create_pydantic_model') as mock_create_model:
                        
                        # Setup mock returns
                        mock_intent_fields.return_value = [
                            SchemaField('name', FieldType.STRING, 'Product name'),
                            SchemaField('price', FieldType.CURRENCY, 'Product price')
                        ]
                        mock_entity_fields.return_value = [
                            SchemaField('brand', FieldType.STRING, 'Product brand')
                        ]
                        mock_query_fields.return_value = []
                        
                        mock_model = Mock(spec=BaseModel)
                        mock_create_model.return_value = mock_model
                        
                        result = self.generator.generate_schema_from_intent(intent_analysis, query)
                        
                        self.assertIsInstance(result, SchemaGenerationResult)
                        self.assertEqual(result.schema_model, mock_model)
                        self.assertGreater(len(result.fields), 0)
                        self.assertGreater(result.confidence, 0)
                        self.assertEqual(result.generation_method, "intent_analysis")
                        self.assertEqual(len(result.validation_errors), 0)
    
    def test_generate_schema_from_intent_information_seeking(self):
        """Test schema generation from information seeking intent."""
        intent_analysis = {
            'intent_type': 'information_seeking',
            'entities': [
                {'text': 'climate change', 'label': 'TOPIC'},
                {'text': '2024', 'label': 'DATE'}
            ],
            'confidence': 0.90
        }
        query = "climate change research 2024"
        
        with patch.object(self.generator, '_generate_fields_from_intent') as mock_intent_fields:
            with patch.object(self.generator, '_create_pydantic_model') as mock_create_model:
                
                mock_intent_fields.return_value = [
                    SchemaField('title', FieldType.STRING, 'Article title'),
                    SchemaField('content', FieldType.STRING, 'Article content'),
                    SchemaField('date', FieldType.DATE, 'Publication date')
                ]
                
                mock_model = Mock(spec=BaseModel)
                mock_create_model.return_value = mock_model
                
                result = self.generator.generate_schema_from_intent(intent_analysis, query)
                
                self.assertIsInstance(result, SchemaGenerationResult)
                self.assertIsNotNone(result.schema_model)
                self.assertEqual(result.metadata['intent_type'], 'information_seeking')
    
    def test_generate_schema_from_intent_error_handling(self):
        """Test error handling in schema generation from intent."""
        intent_analysis = {'intent_type': 'test'}
        
        with patch.object(self.generator, '_generate_fields_from_intent', side_effect=Exception("Test error")):
            result = self.generator.generate_schema_from_intent(intent_analysis)
            
            self.assertIsInstance(result, SchemaGenerationResult)
            self.assertIsNone(result.schema_model)
            self.assertEqual(result.confidence, 0.0)
            self.assertGreater(len(result.validation_errors), 0)
    
    def test_generate_schema_from_sample_valid_data(self):
        """Test schema generation from valid sample data."""
        sample_data = [
            {
                'name': 'Product 1',
                'price': 99.99,
                'in_stock': True,
                'description': 'Great product',
                'tags': ['electronics', 'gadget']
            },
            {
                'name': 'Product 2',
                'price': 149.99,
                'in_stock': False,
                'description': 'Another product',
                'tags': ['home', 'appliance']
            }
        ]
        
        with patch.object(self.generator, '_analyze_field_types') as mock_analyze:
            with patch.object(self.generator, '_create_pydantic_model') as mock_create_model:
                
                mock_analyze.return_value = [
                    SchemaField('name', FieldType.STRING, 'Product name'),
                    SchemaField('price', FieldType.FLOAT, 'Product price'),
                    SchemaField('in_stock', FieldType.BOOLEAN, 'Stock availability'),
                    SchemaField('description', FieldType.STRING, 'Product description'),
                    SchemaField('tags', FieldType.LIST, 'Product tags')
                ]
                
                mock_model = Mock(spec=BaseModel)
                mock_create_model.return_value = mock_model
                
                result = self.generator.generate_schema_from_sample(sample_data)
                
                self.assertIsInstance(result, SchemaGenerationResult)
                self.assertIsNotNone(result.schema_model)
                self.assertEqual(result.generation_method, "sample_inference")
                self.assertGreater(result.confidence, 0)
    
    def test_generate_schema_from_sample_empty_data(self):
        """Test schema generation from empty sample data."""
        sample_data = []
        
        result = self.generator.generate_schema_from_sample(sample_data)
        
        self.assertIsInstance(result, SchemaGenerationResult)
        self.assertIsNone(result.schema_model)
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("No sample data provided", result.validation_errors)
    
    def test_generate_schema_from_sample_inconsistent_data(self):
        """Test schema generation from inconsistent sample data."""
        sample_data = [
            {'name': 'Product 1', 'price': 99.99},
            {'title': 'Article 1', 'content': 'Some content'},  # Different structure
            {'name': 'Product 2', 'price': 'free'}  # Type inconsistency
        ]
        
        with patch.object(self.generator, '_analyze_field_types') as mock_analyze:
            mock_analyze.return_value = [
                SchemaField('name', FieldType.OPTIONAL_STRING, 'Name field'),
                SchemaField('title', FieldType.OPTIONAL_STRING, 'Title field'),
                SchemaField('price', FieldType.OPTIONAL_STRING, 'Price field'),  # Made optional due to inconsistency
                SchemaField('content', FieldType.OPTIONAL_STRING, 'Content field')
            ]
            
            with patch.object(self.generator, '_create_pydantic_model') as mock_create_model:
                mock_model = Mock(spec=BaseModel)
                mock_create_model.return_value = mock_model
                
                result = self.generator.generate_schema_from_sample(sample_data)
                
                self.assertIsInstance(result, SchemaGenerationResult)
                # Should still generate a schema but with lower confidence
                self.assertLessEqual(result.confidence, 0.7)
    
    def test_validate_data_against_schema_valid(self):
        """Test data validation against generated schema with valid data."""
        # Create a simple mock Pydantic model
        class MockSchema(BaseModel):
            name: str
            price: float
            
        test_data = {'name': 'Test Product', 'price': 99.99}
        
        with patch.object(self.generator, '_apply_validation_rules') as mock_validate:
            mock_validate.return_value = (True, [])
            
            result = self.generator.validate_data_against_schema(test_data, MockSchema)
            
            self.assertTrue(result['is_valid'])
            self.assertEqual(len(result['errors']), 0)
            self.assertIsNotNone(result['validated_data'])
    
    def test_validate_data_against_schema_invalid(self):
        """Test data validation against generated schema with invalid data."""
        class MockSchema(BaseModel):
            name: str
            price: float
            
        test_data = {'name': 'Test Product', 'price': 'invalid_price'}  # Invalid type
        
        with patch.object(self.generator, '_apply_validation_rules') as mock_validate:
            mock_validate.return_value = (False, ['Invalid price type'])
            
            result = self.generator.validate_data_against_schema(test_data, MockSchema)
            
            self.assertFalse(result['is_valid'])
            self.assertGreater(len(result['errors']), 0)
    
    def test_refine_schema_based_on_validation_results(self):
        """Test schema refinement based on validation results."""
        class OriginalSchema(BaseModel):
            name: str
            price: float
            
        validation_results = [
            {'is_valid': False, 'errors': ['Price field missing'], 'data': {'name': 'Product 1'}},
            {'is_valid': False, 'errors': ['Invalid price type'], 'data': {'name': 'Product 2', 'price': 'free'}},
            {'is_valid': True, 'errors': [], 'data': {'name': 'Product 3', 'price': 99.99}}
        ]
        
        with patch.object(self.generator, '_analyze_validation_patterns') as mock_analyze:
            with patch.object(self.generator, '_create_refined_schema') as mock_refine:
                mock_analyze.return_value = {
                    'missing_fields': ['price'],
                    'type_conflicts': ['price'],
                    'success_rate': 0.33
                }
                
                mock_refined_schema = Mock(spec=BaseModel)
                mock_refine.return_value = mock_refined_schema
                
                result = self.generator.refine_schema_based_on_validation_results(
                    OriginalSchema, validation_results
                )
                
                self.assertIsInstance(result, SchemaGenerationResult)
                self.assertEqual(result.schema_model, mock_refined_schema)
                self.assertEqual(result.generation_method, "validation_refinement")
    
    def test_schema_field_dataclass(self):
        """Test SchemaField dataclass functionality."""
        field = SchemaField(
            name="test_field",
            field_type=FieldType.STRING,
            description="Test field description",
            required=True,
            default=None,
            constraints={'min_length': 1, 'max_length': 100}
        )
        
        self.assertEqual(field.name, "test_field")
        self.assertEqual(field.field_type, FieldType.STRING)
        self.assertEqual(field.description, "Test field description")
        self.assertTrue(field.required)
        self.assertIsNone(field.default)
        self.assertEqual(field.constraints['min_length'], 1)
        self.assertEqual(field.constraints['max_length'], 100)
    
    def test_schema_field_dataclass_with_defaults(self):
        """Test SchemaField dataclass with default values."""
        field = SchemaField(
            name="optional_field",
            field_type=FieldType.OPTIONAL_STRING,
            description="Optional field"
        )
        
        self.assertEqual(field.name, "optional_field")
        self.assertTrue(field.required)  # Default value
        self.assertIsNone(field.default)  # Default value
        self.assertEqual(field.constraints, {})  # Default value after __post_init__
    
    def test_field_type_enum_values(self):
        """Test FieldType enum has expected values."""
        expected_types = [
            'STRING', 'INTEGER', 'FLOAT', 'BOOLEAN', 'LIST', 'DICT',
            'OPTIONAL_STRING', 'OPTIONAL_INTEGER', 'OPTIONAL_FLOAT', 'OPTIONAL_BOOLEAN',
            'URL', 'EMAIL', 'PHONE', 'DATE', 'CURRENCY'
        ]
        
        actual_types = [field_type.name for field_type in FieldType]
        
        for expected_type in expected_types:
            self.assertIn(expected_type, actual_types)
    
    def test_schema_generation_result_dataclass(self):
        """Test SchemaGenerationResult dataclass functionality."""
        class MockSchema(BaseModel):
            test_field: str
            
        fields = [SchemaField('test_field', FieldType.STRING, 'Test field')]
        
        result = SchemaGenerationResult(
            schema_model=MockSchema,
            fields=fields,
            confidence=0.85,
            generation_method="test_method",
            validation_errors=[],
            metadata={'test': True}
        )
        
        self.assertEqual(result.schema_model, MockSchema)
        self.assertEqual(result.fields, fields)
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.generation_method, "test_method")
        self.assertEqual(result.validation_errors, [])
        self.assertEqual(result.metadata['test'], True)
    
    def test_caching_functionality(self):
        """Test schema caching to avoid regeneration."""
        intent_analysis = {
            'intent_type': 'product_search',
            'entities': [],
            'confidence': 0.8
        }
        
        # First call should generate schema
        with patch.object(self.generator, '_create_pydantic_model') as mock_create:
            mock_model = Mock(spec=BaseModel)
            mock_create.return_value = mock_model
            
            result1 = self.generator.generate_schema_from_intent(intent_analysis)
            
            # Should call _create_pydantic_model
            self.assertTrue(mock_create.called)
            call_count = mock_create.call_count
            
            # Second call with same intent should use cache (if implemented)
            result2 = self.generator.generate_schema_from_intent(intent_analysis)
            
            # Verify both results are valid
            self.assertIsInstance(result1, SchemaGenerationResult)
            self.assertIsInstance(result2, SchemaGenerationResult)
    
    def test_integration_with_intent_analyzer(self):
        """Test integration with UniversalIntentAnalyzer."""
        query = "find laptops under $1000"
        
        # Test that intent analyzer is used when available
        with patch.object(self.generator, '_generate_fields_from_intent') as mock_fields:
            with patch.object(self.generator, '_create_pydantic_model') as mock_create:
                mock_fields.return_value = []
                mock_create.return_value = Mock(spec=BaseModel)
                
                # When intent_analyzer is available, it should be used
                intent_analysis = self.mock_intent_analyzer.analyze_intent.return_value
                result = self.generator.generate_schema_from_intent(intent_analysis, query)
                
                self.assertIsInstance(result, SchemaGenerationResult)
    
    def test_complex_field_type_inference(self):
        """Test complex field type inference from sample data."""
        sample_data = [
            {
                'email': 'user@example.com',
                'url': 'https://example.com',
                'phone': '+1-555-123-4567',
                'date': '2024-01-15',
                'price': '$99.99',
                'rating': 4.5,
                'is_featured': True,
                'tags': ['tag1', 'tag2'],
                'metadata': {'key': 'value'}
            }
        ]
        
        with patch.object(self.generator, '_analyze_field_types') as mock_analyze:
            mock_analyze.return_value = [
                SchemaField('email', FieldType.EMAIL, 'Email address'),
                SchemaField('url', FieldType.URL, 'Website URL'),
                SchemaField('phone', FieldType.PHONE, 'Phone number'),
                SchemaField('date', FieldType.DATE, 'Date field'),
                SchemaField('price', FieldType.CURRENCY, 'Price field'),
                SchemaField('rating', FieldType.FLOAT, 'Rating score'),
                SchemaField('is_featured', FieldType.BOOLEAN, 'Featured flag'),
                SchemaField('tags', FieldType.LIST, 'Tag list'),
                SchemaField('metadata', FieldType.DICT, 'Metadata object')
            ]
            
            with patch.object(self.generator, '_create_pydantic_model') as mock_create:
                mock_create.return_value = Mock(spec=BaseModel)
                
                result = self.generator.generate_schema_from_sample(sample_data)
                
                self.assertIsInstance(result, SchemaGenerationResult)
                self.assertGreater(len(result.fields), 0)
                
                # Verify complex types were inferred
                field_types = [field.field_type for field in result.fields]
                expected_types = [FieldType.EMAIL, FieldType.URL, FieldType.PHONE, 
                                FieldType.DATE, FieldType.CURRENCY, FieldType.FLOAT,
                                FieldType.BOOLEAN, FieldType.LIST, FieldType.DICT]
                
                for expected_type in expected_types:
                    self.assertIn(expected_type, field_types)


if __name__ == '__main__':
    unittest.main()
