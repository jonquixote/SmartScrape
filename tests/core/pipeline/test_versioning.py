"""
Test for pipeline versioning functionality.

This test verifies that the versioning components correctly handle schema versioning,
compatibility checks, and migration between versions.
"""

import unittest
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from core.pipeline.versioning import (
    SchemaVersion,
    VersionRegistry,
    PipelineVersionManager,
    VersionError,
    IncompatibleVersionError,
    MigrationError
)


class VersioningTest(unittest.TestCase):
    """Test suite for versioning functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Sample schema for version 1.0.0
        self.schema_v1 = {
            "type": "object",
            "required": ["name", "version"],
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "stages": {
                    "type": "array",
                    "items": {"type": "object"}
                }
            }
        }
        
        # Sample schema for version 1.1.0 (adding new field)
        self.schema_v1_1 = {
            "type": "object",
            "required": ["name", "version"],
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "stages": {
                    "type": "array",
                    "items": {"type": "object"}
                },
                "timeout": {"type": "integer"}  # New field
            }
        }
        
        # Sample schema for version 2.0.0 (incompatible changes)
        self.schema_v2 = {
            "type": "object",
            "required": ["name", "version", "config"],  # Added required field
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "config": {  # New required structure
                    "type": "object",
                    "properties": {
                        "timeout": {"type": "integer"},
                        "retry": {"type": "integer"}
                    }
                },
                "stages": {
                    "type": "array",
                    "items": {"type": "object"}
                }
            }
        }
        
        # Sample valid config for v1
        self.config_v1 = {
            "name": "test_pipeline",
            "version": "1.0.0",
            "description": "Test pipeline",
            "stages": [
                {"name": "stage1", "type": "input"},
                {"name": "stage2", "type": "process"}
            ]
        }
        
        # Sample valid config for v1.1
        self.config_v1_1 = {
            "name": "test_pipeline",
            "version": "1.1.0",
            "description": "Test pipeline",
            "stages": [
                {"name": "stage1", "type": "input"},
                {"name": "stage2", "type": "process"}
            ],
            "timeout": 60
        }
        
        # Sample valid config for v2
        self.config_v2 = {
            "name": "test_pipeline",
            "version": "2.0.0",
            "description": "Test pipeline",
            "config": {
                "timeout": 60,
                "retry": 3
            },
            "stages": [
                {"name": "stage1", "type": "input"},
                {"name": "stage2", "type": "process"}
            ]
        }
        
        # Define migration functions
        def migrate_v1_to_v1_1(config: Dict[str, Any]) -> Dict[str, Any]:
            """Migrate from v1.0.0 to v1.1.0."""
            new_config = config.copy()
            new_config["version"] = "1.1.0"
            new_config["timeout"] = 30  # Default timeout
            return new_config
        
        def migrate_v1_1_to_v2(config: Dict[str, Any]) -> Dict[str, Any]:
            """Migrate from v1.1.0 to v2.0.0."""
            new_config = config.copy()
            new_config["version"] = "2.0.0"
            new_config["config"] = {
                "timeout": config.get("timeout", 30),
                "retry": 3  # Default retry
            }
            if "timeout" in new_config:
                del new_config["timeout"]  # Moved to config object
            return new_config
        
        # Create the registry and register versions
        self.registry = VersionRegistry("pipeline_config")
        self.registry.register_version("1.0.0", self.schema_v1)
        self.registry.register_version("1.1.0", self.schema_v1_1)
        self.registry.register_version("2.0.0", self.schema_v2, min_compatible_version="2.0.0")
        
        # Register migrations
        self.registry.register_migration("1.0.0", "1.1.0", migrate_v1_to_v1_1)
        self.registry.register_migration("1.1.0", "2.0.0", migrate_v1_1_to_v2)
        
        # Create manager and register the registry
        self.manager = PipelineVersionManager()
        self.manager.registries["pipeline_config"] = self.registry
    
    def test_schema_version_compatibility(self):
        """Test schema version compatibility checks."""
        # Create schema versions
        sv1 = SchemaVersion("1.0.0", self.schema_v1)
        sv1_1 = SchemaVersion("1.1.0", self.schema_v1_1)
        sv2 = SchemaVersion("2.0.0", self.schema_v2, min_compatible_version="2.0.0")
        
        # Test compatibility within same major version
        self.assertTrue(sv1_1.is_compatible_with("1.0.0"))
        self.assertTrue(sv1_1.is_compatible_with("1.1.0"))
        
        # Test incompatibility across major versions
        self.assertFalse(sv2.is_compatible_with("1.0.0"))
        self.assertFalse(sv2.is_compatible_with("1.1.0"))
        
        # Test compatibility with same version
        self.assertTrue(sv1.is_compatible_with("1.0.0"))
        self.assertTrue(sv2.is_compatible_with("2.0.0"))
        
        # Test custom min compatible version
        sv_custom = SchemaVersion("3.0.0", self.schema_v1, min_compatible_version="2.5.0")
        self.assertTrue(sv_custom.is_compatible_with("2.5.0"))
        self.assertTrue(sv_custom.is_compatible_with("2.9.9"))
        self.assertFalse(sv_custom.is_compatible_with("2.4.9"))
    
    def test_schema_validation(self):
        """Test schema validation functionality."""
        # Validate against v1 schema
        errors = self.registry.validate_config(self.config_v1, "1.0.0")
        self.assertEqual(errors, [])
        
        # Validate against v1.1 schema
        errors = self.registry.validate_config(self.config_v1_1, "1.1.0")
        self.assertEqual(errors, [])
        
        # Validate against v2 schema
        errors = self.registry.validate_config(self.config_v2, "2.0.0")
        self.assertEqual(errors, [])
        
        # Validate invalid config (missing required field)
        invalid_config = {
            "version": "2.0.0",
            "description": "Missing name field"
        }
        errors = self.registry.validate_config(invalid_config, "2.0.0")
        self.assertGreater(len(errors), 0)
        
        # Validate with wrong types
        invalid_config = {
            "name": "test",
            "version": "2.0.0",
            "config": "Not an object"  # Should be an object
        }
        errors = self.registry.validate_config(invalid_config, "2.0.0")
        self.assertGreater(len(errors), 0)
    
    def test_migration(self):
        """Test migration between versions."""
        # Migrate from v1 to v1.1
        migrated = self.registry.migrate_config(self.config_v1, "1.0.0", "1.1.0")
        self.assertEqual(migrated["version"], "1.1.0")
        self.assertEqual(migrated["timeout"], 30)
        
        # Validate migrated config against v1.1 schema
        errors = self.registry.validate_config(migrated, "1.1.0")
        self.assertEqual(errors, [])
        
        # Migrate from v1 to v2 (multi-step migration)
        migrated = self.registry.migrate_config(self.config_v1, "1.0.0", "2.0.0")
        self.assertEqual(migrated["version"], "2.0.0")
        self.assertIn("config", migrated)
        self.assertEqual(migrated["config"]["timeout"], 30)
        self.assertEqual(migrated["config"]["retry"], 3)
        
        # Validate migrated config against v2 schema
        errors = self.registry.validate_config(migrated, "2.0.0")
        self.assertEqual(errors, [])
    
    def test_no_migration_path(self):
        """Test error when no migration path exists."""
        # Create isolated registry with no migrations
        isolated_registry = VersionRegistry("isolated")
        isolated_registry.register_version("1.0.0", self.schema_v1)
        isolated_registry.register_version("2.0.0", self.schema_v2)
        
        # Attempt migration should fail
        with self.assertRaises(MigrationError):
            isolated_registry.migrate_config(self.config_v1, "1.0.0", "2.0.0")
    
    def test_version_manager(self):
        """Test version manager functionality."""
        # Get latest version
        latest = self.manager.get_latest_version("pipeline_config")
        self.assertEqual(latest, "2.0.0")
        
        # Validate config
        errors = self.manager.validate_config("pipeline_config", self.config_v1, "1.0.0")
        self.assertEqual(errors, [])
        
        # Migrate config
        migrated = self.manager.migrate_config("pipeline_config", self.config_v1, "1.0.0", "2.0.0")
        self.assertEqual(migrated["version"], "2.0.0")
        
        # Register a new version dynamically
        self.manager.register_version(
            "pipeline_config",
            "2.1.0",
            self.schema_v2,  # Reuse schema
            min_compatible_version="2.0.0"
        )
        
        # New latest version should be 2.1.0
        latest = self.manager.get_latest_version("pipeline_config")
        self.assertEqual(latest, "2.1.0")
    
    def test_extract_version(self):
        """Test version extraction from config."""
        # Simple case with version field
        config = {"version": "1.2.3", "name": "test"}
        version = self.manager.extract_version_from_config(config)
        self.assertEqual(version, "1.2.3")
        
        # Alternative version field
        config = {"schema_version": "2.0.1", "name": "test"}
        version = self.manager.extract_version_from_config(config)
        self.assertEqual(version, "2.0.1")
        
        # Version in string
        config = {"info": "Using configuration format v3.2.1", "name": "test"}
        version = self.manager.extract_version_from_config(config)
        self.assertEqual(version, "3.2.1")
        
        # No valid version
        config = {"name": "test", "value": 123}
        version = self.manager.extract_version_from_config(config)
        self.assertIsNone(version)


if __name__ == "__main__":
    unittest.main()