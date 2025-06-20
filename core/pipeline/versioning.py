"""
Pipeline Versioning Module.

This module provides versioning capabilities for pipelines, including schema versioning,
compatibility checking, and migration utilities.
"""

import json
import logging
import re
import semver
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from datetime import datetime


class VersionError(Exception):
    """Base exception for versioning errors."""
    pass


class IncompatibleVersionError(VersionError):
    """Exception raised when incompatible versions are detected."""
    pass


class MigrationError(VersionError):
    """Exception raised when a migration fails."""
    pass


class SchemaVersion:
    """
    Manages versioning for pipeline schemas.
    
    This class handles semantic versioning for pipeline configurations
    and provides compatibility checking between versions.
    
    Attributes:
        version (str): Semantic version string (e.g., "1.2.3")
        schema (Dict[str, Any]): Schema definition for this version
        min_compatible_version (str): Minimum version compatible with this one
    """
    
    def __init__(self, 
                 version: str, 
                 schema: Dict[str, Any],
                 min_compatible_version: Optional[str] = None):
        """
        Initialize a schema version.
        
        Args:
            version: Semantic version string
            schema: Schema definition for this version
            min_compatible_version: Minimum compatible version, or None to auto-calculate
        """
        # Validate semantic version
        try:
            semver.parse(version)
        except ValueError:
            raise VersionError(f"Invalid semantic version format: {version}")
            
        self.version = version
        self.schema = schema
        
        # Set minimum compatible version (defaults to current major version)
        if min_compatible_version is None:
            major = semver.parse(version)["major"]
            self.min_compatible_version = f"{major}.0.0"
        else:
            try:
                semver.parse(min_compatible_version)
                self.min_compatible_version = min_compatible_version
            except ValueError:
                raise VersionError(f"Invalid min_compatible_version: {min_compatible_version}")
    
    def is_compatible_with(self, other_version: str) -> bool:
        """
        Check if this schema is compatible with another version.
        
        Args:
            other_version: Version string to check compatibility with
            
        Returns:
            bool: True if versions are compatible, False otherwise
        """
        try:
            other = semver.parse(other_version)
            current = semver.parse(self.version)
            min_compatible = semver.parse(self.min_compatible_version)
            
            # Exact same version is always compatible
            if other_version == self.version:
                return True
                
            # Only compatible if other version is >= min_compatible and <= current
            return (semver.compare(other_version, self.min_compatible_version) >= 0 and
                    semver.compare(other_version, self.version) <= 0)
        except ValueError:
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate a configuration against this schema.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List[str]: List of validation errors, empty if valid
        """
        # A proper implementation would use jsonschema validation
        # This is a simplified version
        errors = []
        
        # Check required fields
        for field in self.schema.get("required", []):
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        for field, field_schema in self.schema.get("properties", {}).items():
            if field in config:
                field_type = field_schema.get("type")
                
                if field_type == "string" and not isinstance(config[field], str):
                    errors.append(f"Field {field} should be a string")
                elif field_type == "integer" and not isinstance(config[field], int):
                    errors.append(f"Field {field} should be an integer")
                elif field_type == "number" and not isinstance(config[field], (int, float)):
                    errors.append(f"Field {field} should be a number")
                elif field_type == "boolean" and not isinstance(config[field], bool):
                    errors.append(f"Field {field} should be a boolean")
                elif field_type == "array" and not isinstance(config[field], list):
                    errors.append(f"Field {field} should be an array")
                elif field_type == "object" and not isinstance(config[field], dict):
                    errors.append(f"Field {field} should be an object")
        
        return errors


class VersionRegistry:
    """
    Registry for managing schema versions.
    
    This class stores and manages multiple versions of a schema,
    handles compatibility checks, and provides access to migrations.
    
    Attributes:
        name (str): Name of this version registry
        versions (Dict[str, SchemaVersion]): Map of version strings to SchemaVersion objects
        migrations (Dict[str, Dict[str, Callable]]): Map of version pairs to migration functions
    """
    
    def __init__(self, name: str):
        """
        Initialize a version registry.
        
        Args:
            name: Name of this registry (e.g., "pipeline_config")
        """
        self.name = name
        self.versions = {}
        self.migrations = {}
        self.logger = logging.getLogger(f"pipeline.versioning.{name}")
    
    def register_version(self, 
                       version: str, 
                       schema: Dict[str, Any],
                       min_compatible_version: Optional[str] = None) -> None:
        """
        Register a new schema version.
        
        Args:
            version: Semantic version string
            schema: Schema definition for this version
            min_compatible_version: Minimum compatible version, or None to auto-calculate
        """
        schema_version = SchemaVersion(
            version=version,
            schema=schema,
            min_compatible_version=min_compatible_version
        )
        
        self.versions[version] = schema_version
        self.logger.info(f"Registered schema version {version} for {self.name}")
    
    def register_migration(self, 
                         from_version: str, 
                         to_version: str,
                         migration_func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """
        Register a migration function between two versions.
        
        Args:
            from_version: Source version string
            to_version: Target version string
            migration_func: Function that transforms configs from source to target version
        """
        # Ensure both versions exist
        if from_version not in self.versions:
            raise VersionError(f"Source version {from_version} not registered")
        if to_version not in self.versions:
            raise VersionError(f"Target version {to_version} not registered")
            
        # Store the migration
        if from_version not in self.migrations:
            self.migrations[from_version] = {}
            
        self.migrations[from_version][to_version] = migration_func
        self.logger.info(f"Registered migration from {from_version} to {to_version}")
    
    def get_latest_version(self) -> str:
        """
        Get the latest registered version.
        
        Returns:
            str: Latest version string
        """
        if not self.versions:
            raise VersionError(f"No versions registered for {self.name}")
            
        # Sort versions using semver
        sorted_versions = sorted(
            self.versions.keys(),
            key=lambda x: semver.parse(x),
            reverse=True
        )
        
        return sorted_versions[0]
    
    def is_compatible(self, from_version: str, to_version: str) -> bool:
        """
        Check if two versions are compatible.
        
        Args:
            from_version: First version to check
            to_version: Second version to check
            
        Returns:
            bool: True if versions are compatible, False otherwise
        """
        if from_version not in self.versions:
            return False
        if to_version not in self.versions:
            return False
            
        # Check direct compatibility
        return self.versions[to_version].is_compatible_with(from_version)
    
    def validate_config(self, config: Dict[str, Any], version: str) -> List[str]:
        """
        Validate a configuration against a specific schema version.
        
        Args:
            config: Configuration to validate
            version: Version to validate against
            
        Returns:
            List[str]: List of validation errors, empty if valid
        """
        if version not in self.versions:
            return [f"Unknown version: {version}"]
            
        return self.versions[version].validate_config(config)
    
    def migrate_config(self, 
                     config: Dict[str, Any], 
                     from_version: str,
                     to_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Migrate a configuration from one version to another.
        
        Args:
            config: Source configuration
            from_version: Source version
            to_version: Target version, or None to use latest version
            
        Returns:
            Dict[str, Any]: Migrated configuration
        """
        if from_version not in self.versions:
            raise VersionError(f"Source version {from_version} not registered")
            
        # If no target version specified, use latest
        if to_version is None:
            to_version = self.get_latest_version()
            
        if to_version not in self.versions:
            raise VersionError(f"Target version {to_version} not registered")
            
        # No migration needed for same version
        if from_version == to_version:
            return config.copy()
            
        # Find migration path
        migration_path = self._find_migration_path(from_version, to_version)
        if not migration_path:
            raise MigrationError(
                f"No migration path from {from_version} to {to_version}"
            )
            
        # Apply migrations in sequence
        result = config.copy()
        for i in range(len(migration_path) - 1):
            source = migration_path[i]
            target = migration_path[i + 1]
            
            # Get migration function and apply it
            migration_func = self.migrations[source][target]
            try:
                result = migration_func(result)
                self.logger.info(f"Migrated config from {source} to {target}")
            except Exception as e:
                raise MigrationError(
                    f"Failed to migrate from {source} to {target}: {str(e)}"
                )
                
        return result
    
    def _find_migration_path(self, from_version: str, to_version: str) -> List[str]:
        """
        Find a path of migrations from source to target version.
        
        Uses a breadth-first search to find the shortest migration path.
        
        Args:
            from_version: Source version
            to_version: Target version
            
        Returns:
            List[str]: Ordered list of versions to pass through, or empty if no path
        """
        # If direct migration exists, use it
        if from_version in self.migrations and to_version in self.migrations[from_version]:
            return [from_version, to_version]
            
        # Use breadth-first search to find shortest path
        visited = {from_version}
        queue = [[from_version]]
        
        while queue:
            path = queue.pop(0)
            current = path[-1]
            
            # Check each possible next step
            if current in self.migrations:
                for next_version in self.migrations[current]:
                    if next_version == to_version:
                        # Found a path to target
                        return path + [next_version]
                        
                    if next_version not in visited:
                        visited.add(next_version)
                        queue.append(path + [next_version])
                        
    def set_output_schema(self, 
                         schema: Dict[str, Any], 
                         version: Optional[str] = None) -> None:
        """
        Set the output schema for this pipeline version.
        
        Args:
            schema (Dict): The output schema definition.
            version (Optional[str]): Version string (defaults to current version).
        """
        if version is None:
            version = self.config_schema_version.get_version()
            
        if self.output_schema_version is None:
            self.output_schema_version = SchemaVersion(
                name=f"{self.pipeline_name}_output",
                current_version=version,
                schema=schema
            )
        else:
            self.output_schema_version.update_schema(schema, version)
            
        self.logger.debug(f"Set output schema for version {version}")
    
    def update_version(self, 
                      new_version: str,
                      config_schema: Optional[Dict[str, Any]] = None,
                      migration_func: Optional[Callable] = None,
                      description: str = "",
                      backward_compatible: bool = False) -> None:
        """
        Update the pipeline to a new version.
        
        Args:
            new_version (str): The new version string.
            config_schema (Optional[Dict]): New configuration schema.
            migration_func (Optional[Callable]): Migration function.
            description (str): Description of the new version.
            backward_compatible (bool): Whether this version is compatible with previous ones.
            
        Raises:
            VersionError: If the new version is not greater than the current version.
        """
        current_version = self.config_schema_version.get_version()
        
        # Ensure new version is greater than current version
        if pkg_version.parse(new_version) <= pkg_version.parse(current_version):
            raise VersionError(
                f"New version {new_version} must be greater than current version {current_version}"
            )
        
        # Update configuration schema
        if config_schema is not None:
            self.config_schema_version.update_schema(
                schema=config_schema,
                new_version=new_version,
                description=description,
                migration_func=migration_func
            )
        
        # Update compatible versions list
        if backward_compatible:
            self.compatible_versions.append(new_version)
        else:
            self.compatible_versions = [new_version]
            
        self.logger.info(f"Updated pipeline '{self.pipeline_name}' to version {new_version}")
    
    def check_config_compatibility(self, config: Dict[str, Any], strict: bool = False) -> Tuple[bool, str]:
        """
        Check if a configuration is compatible with the current pipeline version.
        
        Args:
            config (Dict): The configuration to check.
            strict (bool): If True, requires exact version match.
            
        Returns:
            Tuple[bool, str]: (is_compatible, message)
        """
        # Get config version
        config_version = config.get("version", "1.0.0")
        
        # For strict mode, config version must match exactly
        if strict:
            is_compatible = config_version in self.compatible_versions
            message = (f"Configuration version {config_version} {'matches' if is_compatible else 'does not match'} "
                      f"expected version(s): {', '.join(self.compatible_versions)}")
            return is_compatible, message
        
        # For non-strict mode, major version must match and minor version can be less than current
        config_ver = pkg_version.parse(config_version)
        current_ver = pkg_version.parse(self.config_schema_version.get_version())
        
        is_compatible = (
            config_ver.major == current_ver.major and
            config_ver.minor <= current_ver.minor
        )
        
        message = (f"Configuration version {config_version} is "
                  f"{'compatible' if is_compatible else 'incompatible'} with pipeline version "
                  f"{self.config_schema_version.get_version()}")
        
        return is_compatible, message
    
    def migrate_config(self, 
                      config: Dict[str, Any], 
                      to_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Migrate a configuration to a new version.
        
        Args:
            config (Dict): The configuration to migrate.
            to_version (Optional[str]): Target version (defaults to current version).
            
        Returns:
            Dict: The migrated configuration.
            
        Raises:
            MigrationError: If migration fails or no path available.
        """
        # Get config version
        from_version = config.get("version", "1.0.0")
        
        # Use current version if target not specified
        if to_version is None:
            to_version = self.config_schema_version.get_version()
            
        # If versions are the same, no migration needed
        if from_version == to_version:
            return config
            
        # Perform migration
        migrated_config = self.config_schema_version.migrate_data(
            data=config,
            from_version=from_version,
            to_version=to_version
        )
        
        # Update version in migrated config
        migrated_config["version"] = to_version
        
        self.logger.info(f"Migrated configuration from version {from_version} to {to_version}")
        return migrated_config
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """
        Get the version history.
        
        Returns:
            List[Dict]: History of pipeline versions.
        """
        return self.config_schema_version.get_version_history()


class VersionRegistry:
    """
    Registry for managing pipeline and schema versions.
    
    This class provides a centralized way to access and manage versions
    across the application.
    """
    
    def __init__(self):
        """Initialize a new version registry."""
        self._pipeline_versions = {}
        self._schema_versions = {}
        self.logger = logging.getLogger("versioning.registry")
    
    def register_pipeline_version(self, 
                                 pipeline_version: PipelineVersion) -> None:
        """
        Register a pipeline version.
        
        Args:
            pipeline_version (PipelineVersion): The pipeline version to register.
        """
        self._pipeline_versions[pipeline_version.pipeline_name] = pipeline_version
        self.logger.info(f"Registered pipeline version: {pipeline_version.pipeline_name}")
    
    def register_schema_version(self, 
                               schema_version: SchemaVersion) -> None:
        """
        Register a schema version.
        
        Args:
            schema_version (SchemaVersion): The schema version to register.
        """
        self._schema_versions[schema_version.name] = schema_version
        self.logger.info(f"Registered schema version: {schema_version.name}")
    
    def get_pipeline_version(self, 
                            pipeline_name: str) -> Optional[PipelineVersion]:
        """
        Get a pipeline version by name.
        
        Args:
            pipeline_name (str): Name of the pipeline.
            
        Returns:
            Optional[PipelineVersion]: The pipeline version if found, None otherwise.
        """
        return self._pipeline_versions.get(pipeline_name)
    
    def get_schema_version(self, 
                          schema_name: str) -> Optional[SchemaVersion]:
        """
        Get a schema version by name.
        
        Args:
            schema_name (str): Name of the schema.
            
        Returns:
            Optional[SchemaVersion]: The schema version if found, None otherwise.
        """
        return self._schema_versions.get(schema_name)
    
    def list_pipeline_versions(self) -> List[str]:
        """
        List all registered pipeline versions.
        
        Returns:
            List[str]: Names of registered pipeline versions.
        """
        return list(self._pipeline_versions.keys())
    
    def list_schema_versions(self) -> List[str]:
        """
        List all registered schema versions.
        
        Returns:
            List[str]: Names of registered schema versions.
        """
        return list(self._schema_versions.keys())


# Global version registry
_registry = VersionRegistry()


def get_registry() -> VersionRegistry:
    """
    Get the global version registry.
    
    Returns:
        VersionRegistry: The global registry instance.
    """
    return _registry