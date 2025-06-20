"""
Database Output Stage Module.

This module provides specialized output stages for saving data to databases.
"""

import asyncio
import json
import logging
import time
from abc import abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union, Tuple

try:
    import aiosqlite
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

try:
    import aiomysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import asyncpg
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

from core.pipeline.stages.base_stages import OutputStage
from core.pipeline.context import PipelineContext


class DatabaseType(Enum):
    """Supported database types."""
    SQLITE = auto()
    MYSQL = auto()
    POSTGRES = auto()


class BatchMode(Enum):
    """Batch insertion modes."""
    SINGLE = auto()  # One query per item
    BULK = auto()    # One query for all items
    CHUNKED = auto() # Multiple queries with batches of items


class DatabaseOutputStage(OutputStage):
    """
    Base output stage for writing data to databases.
    
    Features:
    - Configurable database connection
    - Transaction management
    - Schema validation
    - Batch insertion
    - Error handling with retry logic
    
    Configuration:
    - db_type: Type of database (SQLITE, MYSQL, POSTGRES)
    - connection_params: Connection parameters for the database
    - table: Target table name
    - schema_validation: Whether to validate schema before insertion
    - batch_mode: Mode for batch insertion (SINGLE, BULK, CHUNKED)
    - batch_size: Size of batches for CHUNKED mode
    - retry_count: Number of retries for transient errors
    - retry_delay: Delay between retries in seconds
    - use_transactions: Whether to use transactions
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new database output stage.
        
        Args:
            name: Name of the stage, defaults to class name if not provided
            config: Configuration parameters for this stage
        """
        super().__init__(name, config)
        
        # Database configuration
        db_type = self.config.get("db_type", "SQLITE")
        try:
            self.db_type = DatabaseType[db_type] if isinstance(db_type, str) else db_type
        except (KeyError, TypeError):
            raise ValueError(f"Unsupported database type: {db_type}")
        
        # Check if the required module is available
        if self.db_type == DatabaseType.SQLITE and not SQLITE_AVAILABLE:
            raise ImportError("aiosqlite is required for SQLite support")
        elif self.db_type == DatabaseType.MYSQL and not MYSQL_AVAILABLE:
            raise ImportError("aiomysql is required for MySQL support")
        elif self.db_type == DatabaseType.POSTGRES and not POSTGRES_AVAILABLE:
            raise ImportError("asyncpg is required for PostgreSQL support")
            
        # Connection parameters
        self.connection_params = self.config.get("connection_params", {})
        
        # Table configuration
        self.table = self.config.get("table")
        if not self.table:
            raise ValueError("table must be specified in configuration")
            
        # Schema validation
        self.schema_validation = self.config.get("schema_validation", True)
        
        # Batch configuration
        batch_mode = self.config.get("batch_mode", "BULK")
        try:
            self.batch_mode = BatchMode[batch_mode] if isinstance(batch_mode, str) else batch_mode
        except (KeyError, TypeError):
            self.batch_mode = BatchMode.BULK
            self.logger.warning(f"Invalid batch mode: {batch_mode}, using BULK")
            
        self.batch_size = self.config.get("batch_size", 100)
        
        # Retry configuration
        self.retry_count = self.config.get("retry_count", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        
        # Transaction configuration
        self.use_transactions = self.config.get("use_transactions", True)
        
        # Set up logger
        self.logger = logging.getLogger(f"pipeline.stages.output.database.{self.name}")
        
        # Connection and cursor
        self.connection = None
        
        # Statistics for monitoring
        self._stats = {
            "rows_inserted": 0,
            "batches_processed": 0,
            "retries": 0,
            "errors": 0
        }
    
    async def validate_output_format(self, context: PipelineContext) -> bool:
        """
        Validate that the data can be inserted into the database.
        
        Args:
            context: The shared pipeline context
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        # Check if table name is specified
        if not self.table:
            self.logger.error("No table specified")
            context.add_error(self.name, "No table specified")
            return False
            
        # Get the data to validate
        data = self._prepare_output_data(context)
        if data is None:
            self.logger.error("No data to insert")
            context.add_error(self.name, "No data to insert")
            return False
            
        # For bulk operations, data must be a list
        if self.batch_mode != BatchMode.SINGLE and not isinstance(data, (list, tuple)):
            self.logger.error("Data must be a list or tuple for bulk insertion")
            context.add_error(self.name, "Invalid data type for bulk insertion")
            return False
            
        # Schema validation if enabled
        if self.schema_validation:
            # This is a basic validation to ensure items have consistent fields
            if isinstance(data, (list, tuple)) and len(data) > 0:
                first_item = data[0]
                if isinstance(first_item, dict):
                    keys = set(first_item.keys())
                    for i, item in enumerate(data[1:], 1):
                        if not isinstance(item, dict):
                            self.logger.error(f"Inconsistent item types at index {i}")
                            context.add_error(self.name, f"Inconsistent item types at index {i}")
                            return False
                            
                        item_keys = set(item.keys())
                        if item_keys != keys:
                            self.logger.error(f"Inconsistent item schema at index {i}")
                            context.add_error(self.name, f"Inconsistent item schema at index {i}")
                            return False
                    
                    # For single items, just check it's a dict
                elif isinstance(data, dict):
                    pass
                else:
                    self.logger.error("Data items must be dictionaries")
                    context.add_error(self.name, "Data items must be dictionaries")
                    return False
        
        # Validate the database connection will work
        try:
            # Try to connect to the database
            await self._connect()
            await self._disconnect()
            return True
        except Exception as e:
            self.logger.error(f"Database connection validation failed: {str(e)}")
            context.add_error(self.name, f"Database connection error: {str(e)}")
            return False
    
    async def deliver_output(self, data: Any, context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        Insert the data into the database.
        
        Args:
            data: The data to insert
            context: The shared pipeline context
            
        Returns:
            Dict containing insertion results or None if insertion failed
        """
        try:
            # Normalize data to list of records
            records = self._normalize_data(data)
            
            # Connect to the database
            await self._connect()
            
            # Start transaction if enabled
            if self.use_transactions and len(records) > 1:
                await self._begin_transaction()
            
            # Insert records based on batch mode
            if self.batch_mode == BatchMode.SINGLE:
                # Insert one by one
                for record in records:
                    await self._insert_record(record)
                
            elif self.batch_mode == BatchMode.BULK:
                # Insert all records in one query
                await self._bulk_insert(records)
                
            elif self.batch_mode == BatchMode.CHUNKED:
                # Insert in batches of batch_size
                for i in range(0, len(records), self.batch_size):
                    batch = records[i:i+self.batch_size]
                    await self._bulk_insert(batch)
                    self._stats["batches_processed"] += 1
            
            # Commit transaction if enabled
            if self.use_transactions and len(records) > 1:
                await self._commit_transaction()
            
            # Disconnect from the database
            await self._disconnect()
            
            return {
                "table": self.table,
                "rows_inserted": self._stats["rows_inserted"],
                "batches_processed": self._stats["batches_processed"],
                "database_type": self.db_type.name
            }
            
        except Exception as e:
            self.logger.error(f"Error inserting data: {str(e)}")
            context.add_error(self.name, f"Database output error: {str(e)}")
            
            # Rollback transaction if active
            if self.use_transactions and self.connection:
                try:
                    await self._rollback_transaction()
                except Exception as rollback_error:
                    self.logger.error(f"Error rolling back transaction: {str(rollback_error)}")
            
            # Disconnect from the database
            await self._disconnect()
            
            return None
    
    def _normalize_data(self, data: Any) -> List[Dict[str, Any]]:
        """
        Normalize data to a list of records.
        
        Args:
            data: The data to normalize
            
        Returns:
            List of records as dictionaries
        """
        if isinstance(data, dict):
            # Single record as dictionary
            return [data]
        elif isinstance(data, (list, tuple)):
            # List of records
            return list(data)
        else:
            # Try to convert to a record
            return [{"value": data}]
    
    async def _connect(self) -> None:
        """Connect to the database."""
        if self.db_type == DatabaseType.SQLITE:
            self.connection = await aiosqlite.connect(**self.connection_params)
            
        elif self.db_type == DatabaseType.MYSQL:
            self.connection = await aiomysql.connect(**self.connection_params)
            
        elif self.db_type == DatabaseType.POSTGRES:
            self.connection = await asyncpg.connect(**self.connection_params)
    
    async def _disconnect(self) -> None:
        """Disconnect from the database."""
        if self.connection:
            await self.connection.close()
            self.connection = None
    
    async def _begin_transaction(self) -> None:
        """Begin a database transaction."""
        if self.db_type == DatabaseType.SQLITE:
            await self.connection.execute("BEGIN TRANSACTION")
            
        elif self.db_type == DatabaseType.MYSQL:
            async with self.connection.cursor() as cursor:
                await cursor.execute("START TRANSACTION")
                
        elif self.db_type == DatabaseType.POSTGRES:
            await self.connection.execute("BEGIN")
    
    async def _commit_transaction(self) -> None:
        """Commit a database transaction."""
        if self.db_type == DatabaseType.SQLITE:
            await self.connection.commit()
            
        elif self.db_type == DatabaseType.MYSQL:
            await self.connection.commit()
            
        elif self.db_type == DatabaseType.POSTGRES:
            await self.connection.execute("COMMIT")
    
    async def _rollback_transaction(self) -> None:
        """Rollback a database transaction."""
        if self.db_type == DatabaseType.SQLITE:
            await self.connection.rollback()
            
        elif self.db_type == DatabaseType.MYSQL:
            await self.connection.rollback()
            
        elif self.db_type == DatabaseType.POSTGRES:
            await self.connection.execute("ROLLBACK")
    
    async def _insert_record(self, record: Dict[str, Any]) -> None:
        """
        Insert a single record into the database.
        
        Args:
            record: The record to insert as a dictionary
        """
        # Get column names and values
        columns = list(record.keys())
        values = [record[col] for col in columns]
        
        # Generate placeholders based on database type
        placeholders = self._get_placeholders(len(columns))
        
        # Build SQL query
        query = f"INSERT INTO {self.table} ({', '.join(columns)}) VALUES ({placeholders})"
        
        # Execute query with retry logic
        attempt = 0
        while True:
            try:
                if self.db_type == DatabaseType.SQLITE:
                    await self.connection.execute(query, values)
                    
                elif self.db_type == DatabaseType.MYSQL:
                    async with self.connection.cursor() as cursor:
                        await cursor.execute(query, values)
                        
                elif self.db_type == DatabaseType.POSTGRES:
                    await self.connection.execute(query, *values)
                
                # Increment statistics
                self._stats["rows_inserted"] += 1
                break
                
            except Exception as e:
                # Check if error is transient and we should retry
                attempt += 1
                if attempt <= self.retry_count and self._is_transient_error(str(e)):
                    self._stats["retries"] += 1
                    delay = self.retry_delay * attempt
                    self.logger.warning(f"Transient error: {str(e)}. Retrying in {delay:.2f}s (attempt {attempt}/{self.retry_count})")
                    await asyncio.sleep(delay)
                else:
                    self._stats["errors"] += 1
                    raise
    
    async def _bulk_insert(self, records: List[Dict[str, Any]]) -> None:
        """
        Insert multiple records in a single query.
        
        Args:
            records: List of records to insert
        """
        if not records:
            return
            
        # Use first record to get column names
        columns = list(records[0].keys())
        
        # Handle database-specific bulk insertions
        if self.db_type == DatabaseType.SQLITE:
            # SQLite doesn't have a native bulk insert, so we use executemany
            placeholders = f"({', '.join(['?' for _ in columns])})"
            query = f"INSERT INTO {self.table} ({', '.join(columns)}) VALUES {placeholders}"
            values = [[record[col] for col in columns] for record in records]
            
            attempt = 0
            while True:
                try:
                    await self.connection.executemany(query, values)
                    self._stats["rows_inserted"] += len(records)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt <= self.retry_count and self._is_transient_error(str(e)):
                        self._stats["retries"] += 1
                        delay = self.retry_delay * attempt
                        self.logger.warning(f"Transient error: {str(e)}. Retrying in {delay:.2f}s (attempt {attempt}/{self.retry_count})")
                        await asyncio.sleep(delay)
                    else:
                        self._stats["errors"] += 1
                        raise
            
        elif self.db_type == DatabaseType.MYSQL:
            # MySQL bulk insert
            placeholders_list = []
            values_flat = []
            
            for record in records:
                placeholders_list.append(f"({', '.join(['%s' for _ in columns])})")
                values_flat.extend([record[col] for col in columns])
                
            query = f"INSERT INTO {self.table} ({', '.join(columns)}) VALUES {', '.join(placeholders_list)}"
            
            attempt = 0
            while True:
                try:
                    async with self.connection.cursor() as cursor:
                        await cursor.execute(query, values_flat)
                    self._stats["rows_inserted"] += len(records)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt <= self.retry_count and self._is_transient_error(str(e)):
                        self._stats["retries"] += 1
                        delay = self.retry_delay * attempt
                        self.logger.warning(f"Transient error: {str(e)}. Retrying in {delay:.2f}s (attempt {attempt}/{self.retry_count})")
                        await asyncio.sleep(delay)
                    else:
                        self._stats["errors"] += 1
                        raise
            
        elif self.db_type == DatabaseType.POSTGRES:
            # PostgreSQL bulk insert
            try:
                # Use the more efficient copy function
                values = []
                for record in records:
                    row = [record[col] for col in columns]
                    values.append(row)
                
                # Copy data to table
                await self.connection.copy_records_to_table(
                    self.table,
                    records=values,
                    columns=columns
                )
                self._stats["rows_inserted"] += len(records)
            except Exception as copy_error:
                # Fall back to standard insert if copy fails
                self.logger.warning(f"Copy operation failed: {str(copy_error)}. Falling back to standard insert.")
                
                placeholders_list = []
                for i, record in enumerate(records):
                    placeholder_set = [f"${j+i*len(columns)+1}" for j in range(len(columns))]
                    placeholders_list.append(f"({', '.join(placeholder_set)})")
                    
                query = f"INSERT INTO {self.table} ({', '.join(columns)}) VALUES {', '.join(placeholders_list)}"
                
                # Flatten values
                values_flat = []
                for record in records:
                    values_flat.extend([record[col] for col in columns])
                
                attempt = 0
                while True:
                    try:
                        await self.connection.execute(query, *values_flat)
                        self._stats["rows_inserted"] += len(records)
                        break
                    except Exception as e:
                        attempt += 1
                        if attempt <= self.retry_count and self._is_transient_error(str(e)):
                            self._stats["retries"] += 1
                            delay = self.retry_delay * attempt
                            self.logger.warning(f"Transient error: {str(e)}. Retrying in {delay:.2f}s (attempt {attempt}/{self.retry_count})")
                            await asyncio.sleep(delay)
                        else:
                            self._stats["errors"] += 1
                            raise
    
    def _get_placeholders(self, count: int) -> str:
        """
        Get placeholder string for SQL parameters based on database type.
        
        Args:
            count: Number of placeholders needed
            
        Returns:
            str: Placeholders string
        """
        if self.db_type == DatabaseType.SQLITE:
            return ", ".join(["?" for _ in range(count)])
            
        elif self.db_type == DatabaseType.MYSQL:
            return ", ".join(["%s" for _ in range(count)])
            
        elif self.db_type == DatabaseType.POSTGRES:
            return ", ".join([f"${i+1}" for i in range(count)])
    
    def _is_transient_error(self, error_message: str) -> bool:
        """
        Check if an error is transient and should be retried.
        
        Args:
            error_message: The error message string
            
        Returns:
            bool: True if the error is transient, False otherwise
        """
        # Check for common transient error patterns
        transient_patterns = [
            "deadlock",
            "lock timeout",
            "connection reset",
            "server closed the connection",
            "connection timed out",
            "too many connections",
            "resource temporarily unavailable",
            "database is locked",
            "operational error"
        ]
        
        error_lower = error_message.lower()
        return any(pattern in error_lower for pattern in transient_patterns)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for configuration.
        
        Returns:
            Dict containing JSON schema for this stage's configuration
        """
        schema = super().get_config_schema()
        
        # Add database-specific properties
        database_properties = {
            "db_type": {
                "type": "string",
                "enum": ["SQLITE", "MYSQL", "POSTGRES"]
            },
            "connection_params": {
                "type": "object",
                "properties": {
                    "host": {"type": "string"},
                    "port": {"type": "integer"},
                    "user": {"type": "string"},
                    "password": {"type": "string"},
                    "database": {"type": "string"},
                    "database_path": {"type": "string"}
                }
            },
            "table": {"type": "string"},
            "schema_validation": {"type": "boolean"},
            "batch_mode": {
                "type": "string",
                "enum": ["SINGLE", "BULK", "CHUNKED"]
            },
            "batch_size": {"type": "integer", "minimum": 1},
            "retry_count": {"type": "integer", "minimum": 0},
            "retry_delay": {"type": "number", "minimum": 0},
            "use_transactions": {"type": "boolean"}
        }
        
        # Update the properties in the schema
        schema["properties"].update(database_properties)
        
        return schema


class SQLiteOutputStage(DatabaseOutputStage):
    """
    Output stage for SQLite databases.
    
    Additional Configuration:
    - database_path: Path to SQLite database file
    - pragma_statements: List of PRAGMA statements to execute before insertion
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new SQLite output stage.
        
        Args:
            name: Name of the stage, defaults to class name if not provided
            config: Configuration parameters for this stage
        """
        # Ensure SQLite database type
        config = config or {}
        config["db_type"] = "SQLITE"
        
        # Get the database path
        database_path = config.get("database_path")
        if not database_path and "connection_params" in config:
            database_path = config["connection_params"].get("database")
        
        if not database_path:
            raise ValueError("database_path must be specified for SQLite")
            
        # Set up connection parameters
        if "connection_params" not in config:
            config["connection_params"] = {}
        config["connection_params"]["database"] = database_path
        
        super().__init__(name, config)
        
        # SQLite-specific configuration
        self.pragma_statements = self.config.get("pragma_statements", [])
    
    async def _connect(self) -> None:
        """Connect to the SQLite database and execute PRAGMA statements."""
        await super()._connect()
        
        # Execute PRAGMA statements
        for pragma in self.pragma_statements:
            await self.connection.execute(pragma)


class MySQLOutputStage(DatabaseOutputStage):
    """
    Output stage for MySQL databases.
    
    Additional Configuration:
    - insert_ignore: Whether to use INSERT IGNORE syntax
    - replace: Whether to use REPLACE syntax instead of INSERT
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new MySQL output stage.
        
        Args:
            name: Name of the stage, defaults to class name if not provided
            config: Configuration parameters for this stage
        """
        # Ensure MySQL database type
        config = config or {}
        config["db_type"] = "MYSQL"
        
        super().__init__(name, config)
        
        # MySQL-specific configuration
        self.insert_ignore = self.config.get("insert_ignore", False)
        self.replace = self.config.get("replace", False)
        
        if self.insert_ignore and self.replace:
            raise ValueError("Cannot use both insert_ignore and replace")
    
    async def _insert_record(self, record: Dict[str, Any]) -> None:
        """
        Insert a single record into MySQL.
        
        Args:
            record: The record to insert as a dictionary
        """
        # Get column names and values
        columns = list(record.keys())
        values = [record[col] for col in columns]
        
        # Generate placeholders
        placeholders = self._get_placeholders(len(columns))
        
        # Build SQL query with MySQL-specific syntax
        if self.replace:
            query = f"REPLACE INTO {self.table} ({', '.join(columns)}) VALUES ({placeholders})"
        elif self.insert_ignore:
            query = f"INSERT IGNORE INTO {self.table} ({', '.join(columns)}) VALUES ({placeholders})"
        else:
            query = f"INSERT INTO {self.table} ({', '.join(columns)}) VALUES ({placeholders})"
        
        # Execute query with retry logic
        attempt = 0
        while True:
            try:
                async with self.connection.cursor() as cursor:
                    await cursor.execute(query, values)
                
                # Increment statistics
                self._stats["rows_inserted"] += 1
                break
                
            except Exception as e:
                attempt += 1
                if attempt <= self.retry_count and self._is_transient_error(str(e)):
                    self._stats["retries"] += 1
                    delay = self.retry_delay * attempt
                    self.logger.warning(f"Transient error: {str(e)}. Retrying in {delay:.2f}s (attempt {attempt}/{self.retry_count})")
                    await asyncio.sleep(delay)
                else:
                    self._stats["errors"] += 1
                    raise
    
    async def _bulk_insert(self, records: List[Dict[str, Any]]) -> None:
        """
        Insert multiple records in a single MySQL query.
        
        Args:
            records: List of records to insert
        """
        if not records:
            return
            
        # Use first record to get column names
        columns = list(records[0].keys())
        
        # MySQL bulk insert
        placeholders_list = []
        values_flat = []
        
        for record in records:
            placeholders_list.append(f"({', '.join(['%s' for _ in columns])})")
            values_flat.extend([record[col] for col in columns])
            
        # Build SQL query with MySQL-specific syntax
        if self.replace:
            query = f"REPLACE INTO {self.table} ({', '.join(columns)}) VALUES {', '.join(placeholders_list)}"
        elif self.insert_ignore:
            query = f"INSERT IGNORE INTO {self.table} ({', '.join(columns)}) VALUES {', '.join(placeholders_list)}"
        else:
            query = f"INSERT INTO {self.table} ({', '.join(columns)}) VALUES {', '.join(placeholders_list)}"
        
        # Execute query with retry logic
        attempt = 0
        while True:
            try:
                async with self.connection.cursor() as cursor:
                    await cursor.execute(query, values_flat)
                self._stats["rows_inserted"] += len(records)
                break
            except Exception as e:
                attempt += 1
                if attempt <= self.retry_count and self._is_transient_error(str(e)):
                    self._stats["retries"] += 1
                    delay = self.retry_delay * attempt
                    self.logger.warning(f"Transient error: {str(e)}. Retrying in {delay:.2f}s (attempt {attempt}/{self.retry_count})")
                    await asyncio.sleep(delay)
                else:
                    self._stats["errors"] += 1
                    raise
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for configuration.
        
        Returns:
            Dict containing JSON schema for this stage's configuration
        """
        schema = super().get_config_schema()
        
        # Add MySQL-specific properties
        mysql_properties = {
            "insert_ignore": {"type": "boolean"},
            "replace": {"type": "boolean"}
        }
        
        # Update the properties in the schema
        schema["properties"].update(mysql_properties)
        
        return schema


class PostgresOutputStage(DatabaseOutputStage):
    """
    Output stage for PostgreSQL databases.
    
    Additional Configuration:
    - upsert_constraint: Constraint for ON CONFLICT clause (upsert operations)
    - upsert_action: Action for ON CONFLICT (DO NOTHING or DO UPDATE SET)
    - upsert_columns: Columns to update in DO UPDATE SET
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new PostgreSQL output stage.
        
        Args:
            name: Name of the stage, defaults to class name if not provided
            config: Configuration parameters for this stage
        """
        # Ensure PostgreSQL database type
        config = config or {}
        config["db_type"] = "POSTGRES"
        
        super().__init__(name, config)
        
        # PostgreSQL-specific configuration
        self.upsert_constraint = self.config.get("upsert_constraint")
        self.upsert_action = self.config.get("upsert_action", "DO NOTHING")
        self.upsert_columns = self.config.get("upsert_columns", [])
        
        # Validate upsert configuration
        if self.upsert_constraint and self.upsert_action == "DO UPDATE SET" and not self.upsert_columns:
            raise ValueError("upsert_columns must be specified for DO UPDATE SET action")
    
    async def _insert_record(self, record: Dict[str, Any]) -> None:
        """
        Insert a single record into PostgreSQL with optional upsert handling.
        
        Args:
            record: The record to insert as a dictionary
        """
        # Get column names and values
        columns = list(record.keys())
        values = [record[col] for col in columns]
        
        # Generate placeholders
        placeholders = self._get_placeholders(len(columns))
        
        # Build SQL query with PostgreSQL-specific upsert syntax
        query = f"INSERT INTO {self.table} ({', '.join(columns)}) VALUES ({placeholders})"
        
        # Add ON CONFLICT clause if upsert is configured
        if self.upsert_constraint:
            query += f" ON CONFLICT ({self.upsert_constraint})"
            
            if self.upsert_action == "DO NOTHING":
                query += " DO NOTHING"
            elif self.upsert_action == "DO UPDATE SET":
                updates = []
                for col in self.upsert_columns:
                    if col in columns:
                        col_index = columns.index(col) + 1
                        updates.append(f"{col} = EXCLUDED.{col}")
                
                if updates:
                    query += f" DO UPDATE SET {', '.join(updates)}"
                else:
                    query += " DO NOTHING"
        
        # Execute query with retry logic
        attempt = 0
        while True:
            try:
                await self.connection.execute(query, *values)
                
                # Increment statistics
                self._stats["rows_inserted"] += 1
                break
                
            except Exception as e:
                attempt += 1
                if attempt <= self.retry_count and self._is_transient_error(str(e)):
                    self._stats["retries"] += 1
                    delay = self.retry_delay * attempt
                    self.logger.warning(f"Transient error: {str(e)}. Retrying in {delay:.2f}s (attempt {attempt}/{self.retry_count})")
                    await asyncio.sleep(delay)
                else:
                    self._stats["errors"] += 1
                    raise
    
    async def _bulk_insert(self, records: List[Dict[str, Any]]) -> None:
        """
        Insert multiple records in a single PostgreSQL query or using COPY.
        
        Args:
            records: List of records to insert
        """
        if not records:
            return
        
        # Use first record to get column names
        columns = list(records[0].keys())
        
        # Try to use COPY for faster inserts if no upsert is needed
        if not self.upsert_constraint:
            try:
                # Prepare values for copy
                values = []
                for record in records:
                    row = [record[col] for col in columns]
                    values.append(row)
                
                # Copy data to table
                await self.connection.copy_records_to_table(
                    self.table,
                    records=values,
                    columns=columns
                )
                self._stats["rows_inserted"] += len(records)
                return
            except Exception as copy_error:
                self.logger.warning(f"Copy operation failed: {str(copy_error)}. Falling back to standard insert.")
        
        # Fall back to standard INSERT with optional ON CONFLICT
        placeholders_list = []
        values_flat = []
        
        for i, record in enumerate(records):
            placeholder_set = [f"${j+i*len(columns)+1}" for j in range(len(columns))]
            placeholders_list.append(f"({', '.join(placeholder_set)})")
            values_flat.extend([record[col] for col in columns])
        
        # Build the query
        query = f"INSERT INTO {self.table} ({', '.join(columns)}) VALUES {', '.join(placeholders_list)}"
        
        # Add ON CONFLICT clause if upsert is configured
        if self.upsert_constraint:
            query += f" ON CONFLICT ({self.upsert_constraint})"
            
            if self.upsert_action == "DO NOTHING":
                query += " DO NOTHING"
            elif self.upsert_action == "DO UPDATE SET":
                updates = []
                for col in self.upsert_columns:
                    if col in columns:
                        updates.append(f"{col} = EXCLUDED.{col}")
                
                if updates:
                    query += f" DO UPDATE SET {', '.join(updates)}"
                else:
                    query += " DO NOTHING"
        
        # Execute query with retry logic
        attempt = 0
        while True:
            try:
                await self.connection.execute(query, *values_flat)
                self._stats["rows_inserted"] += len(records)
                break
            except Exception as e:
                attempt += 1
                if attempt <= self.retry_count and self._is_transient_error(str(e)):
                    self._stats["retries"] += 1
                    delay = self.retry_delay * attempt
                    self.logger.warning(f"Transient error: {str(e)}. Retrying in {delay:.2f}s (attempt {attempt}/{self.retry_count})")
                    await asyncio.sleep(delay)
                else:
                    self._stats["errors"] += 1
                    raise
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for configuration.
        
        Returns:
            Dict containing JSON schema for this stage's configuration
        """
        schema = super().get_config_schema()
        
        # Add PostgreSQL-specific properties
        postgres_properties = {
            "upsert_constraint": {"type": "string"},
            "upsert_action": {
                "type": "string",
                "enum": ["DO NOTHING", "DO UPDATE SET"]
            },
            "upsert_columns": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
        
        # Update the properties in the schema
        schema["properties"].update(postgres_properties)
        
        return schema