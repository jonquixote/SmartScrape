"""
Tests for the database output stage.

This module contains tests for the DatabaseOutputStage and its specialized
subclasses for different database types.
"""

import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

from core.pipeline.context import PipelineContext
from core.pipeline.stages.output.database_output import (
    DatabaseOutputStage, DatabaseType, BatchMode,
    SQLiteOutputStage, MySQLOutputStage, PostgresOutputStage
)


class TestDatabaseOutputStage(unittest.TestCase):
    """Test cases for the DatabaseOutputStage base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a test SQLite database
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        
        # Create a test table
        self.cursor.execute("""
        CREATE TABLE test_table (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER
        )
        """)
        self.connection.commit()
        
        # Test data as list of dictionaries
        self.test_data = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35}
        ]
        
        # Single record
        self.single_record = {"id": 4, "name": "Dave", "age": 40}
        
        self.context = PipelineContext()
        self.context.set("data", self.test_data)
    
    def tearDown(self):
        """Clean up after tests."""
        self.connection.close()
        self.temp_dir.cleanup()
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    def test_init_with_sqlite_config(self):
        """Test initialization with SQLite configuration."""
        config = {
            "db_type": "SQLITE",
            "connection_params": {
                "database": self.db_path
            },
            "table": "test_table"
        }
        
        with patch('core.pipeline.stages.output.database_output.aiosqlite'):
            stage = DatabaseOutputStage(config=config)
            
            self.assertEqual(stage.db_type, DatabaseType.SQLITE)
            self.assertEqual(stage.table, "test_table")
            self.assertEqual(stage.connection_params, {"database": self.db_path})
            self.assertEqual(stage.batch_mode, BatchMode.BULK)
            self.assertEqual(stage.batch_size, 100)
            self.assertEqual(stage.retry_count, 3)
            self.assertTrue(stage.use_transactions)
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', False)
    def test_init_with_unavailable_module(self):
        """Test initialization with unavailable database module."""
        config = {
            "db_type": "SQLITE",
            "connection_params": {
                "database": self.db_path
            },
            "table": "test_table"
        }
        
        with self.assertRaises(ImportError):
            DatabaseOutputStage(config=config)
    
    def test_init_without_table(self):
        """Test initialization without a table name."""
        config = {
            "db_type": "SQLITE",
            "connection_params": {
                "database": self.db_path
            }
        }
        
        with self.assertRaises(ValueError):
            DatabaseOutputStage(config=config)
    
    def test_init_with_invalid_db_type(self):
        """Test initialization with an invalid database type."""
        config = {
            "db_type": "INVALID",
            "connection_params": {},
            "table": "test_table"
        }
        
        with self.assertRaises(ValueError):
            DatabaseOutputStage(config=config)
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiosqlite')
    async def test_validate_output_format_success(self, mock_aiosqlite):
        """Test successful output validation."""
        # Set up mock connection
        mock_connection = AsyncMock()
        mock_aiosqlite.connect.return_value = mock_connection
        
        config = {
            "db_type": "SQLITE",
            "connection_params": {
                "database": self.db_path
            },
            "table": "test_table"
        }
        
        stage = DatabaseOutputStage(config=config)
        
        result = await stage.validate_output_format(self.context)
        self.assertTrue(result)
        
        # Verify connection was attempted
        mock_aiosqlite.connect.assert_called_once()
        mock_connection.close.assert_called_once()
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiosqlite')
    async def test_validate_output_format_connection_error(self, mock_aiosqlite):
        """Test validation with connection error."""
        # Make connection fail
        mock_aiosqlite.connect.side_effect = Exception("Connection failed")
        
        config = {
            "db_type": "SQLITE",
            "connection_params": {
                "database": self.db_path
            },
            "table": "test_table"
        }
        
        stage = DatabaseOutputStage(config=config)
        
        result = await stage.validate_output_format(self.context)
        self.assertFalse(result)
        self.assertTrue(self.context.has_errors())
        
        # Verify connection was attempted
        mock_aiosqlite.connect.assert_called_once()
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiosqlite')
    async def test_validate_output_format_invalid_data(self, mock_aiosqlite):
        """Test validation with invalid data."""
        # Set up mock connection
        mock_connection = AsyncMock()
        mock_aiosqlite.connect.return_value = mock_connection
        
        config = {
            "db_type": "SQLITE",
            "connection_params": {
                "database": self.db_path
            },
            "table": "test_table",
            "schema_validation": True
        }
        
        stage = DatabaseOutputStage(config=config)
        
        # Mix of dictionaries and other types
        invalid_data = [{"id": 1}, "not a dict", {"name": "test"}]
        self.context.set("data", invalid_data)
        
        result = await stage.validate_output_format(self.context)
        self.assertFalse(result)
        self.assertTrue(self.context.has_errors())
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    async def test_normalize_data(self):
        """Test normalizing data to list of records."""
        config = {
            "db_type": "SQLITE",
            "connection_params": {
                "database": self.db_path
            },
            "table": "test_table"
        }
        
        with patch('core.pipeline.stages.output.database_output.aiosqlite'):
            stage = DatabaseOutputStage(config=config)
            
            # Test with list of dicts
            normalized = stage._normalize_data(self.test_data)
            self.assertEqual(normalized, self.test_data)
            
            # Test with single dict
            normalized = stage._normalize_data(self.single_record)
            self.assertEqual(normalized, [self.single_record])
            
            # Test with scalar value
            normalized = stage._normalize_data("test")
            self.assertEqual(normalized, [{"value": "test"}])
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiosqlite')
    async def test_insert_record(self, mock_aiosqlite):
        """Test inserting a single record."""
        # Set up mock connection and cursor
        mock_connection = AsyncMock()
        mock_aiosqlite.connect.return_value = mock_connection
        
        config = {
            "db_type": "SQLITE",
            "connection_params": {
                "database": self.db_path
            },
            "table": "test_table"
        }
        
        stage = DatabaseOutputStage(config=config)
        stage.connection = mock_connection
        
        # Insert a record
        await stage._insert_record(self.single_record)
        
        # Verify execute was called with correct SQL
        mock_connection.execute.assert_called_once()
        args, kwargs = mock_connection.execute.call_args
        sql = args[0]
        
        self.assertIn("INSERT INTO test_table", sql)
        self.assertIn("id", sql)
        self.assertIn("name", sql)
        self.assertIn("age", sql)
        
        # Check that values were passed correctly
        values = args[1]
        self.assertEqual(values, [4, "Dave", 40])
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiosqlite')
    async def test_bulk_insert(self, mock_aiosqlite):
        """Test bulk insertion of records."""
        # Set up mock connection
        mock_connection = AsyncMock()
        mock_aiosqlite.connect.return_value = mock_connection
        
        config = {
            "db_type": "SQLITE",
            "connection_params": {
                "database": self.db_path
            },
            "table": "test_table"
        }
        
        stage = DatabaseOutputStage(config=config)
        stage.connection = mock_connection
        
        # Insert multiple records
        await stage._bulk_insert(self.test_data)
        
        # Verify executemany was called
        mock_connection.executemany.assert_called_once()
        
        # Check that stats were updated
        self.assertEqual(stage._stats["rows_inserted"], 3)
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiosqlite')
    async def test_deliver_output_single_mode(self, mock_aiosqlite):
        """Test delivering output in single insertion mode."""
        # Set up mock connection
        mock_connection = AsyncMock()
        mock_aiosqlite.connect.return_value = mock_connection
        
        config = {
            "db_type": "SQLITE",
            "connection_params": {
                "database": self.db_path
            },
            "table": "test_table",
            "batch_mode": "SINGLE"
        }
        
        stage = DatabaseOutputStage(config=config)
        
        # Deliver output
        result = await stage.deliver_output(self.test_data, self.context)
        
        # Verify result
        self.assertEqual(result["table"], "test_table")
        self.assertEqual(result["rows_inserted"], 3)
        self.assertEqual(result["database_type"], "SQLITE")
        
        # Verify connection was established and closed
        mock_aiosqlite.connect.assert_called_once()
        mock_connection.close.assert_called_once()
        
        # Verify execute was called for each record (3 times)
        self.assertEqual(mock_connection.execute.call_count, 3)
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiosqlite')
    async def test_deliver_output_bulk_mode(self, mock_aiosqlite):
        """Test delivering output in bulk insertion mode."""
        # Set up mock connection
        mock_connection = AsyncMock()
        mock_aiosqlite.connect.return_value = mock_connection
        
        config = {
            "db_type": "SQLITE",
            "connection_params": {
                "database": self.db_path
            },
            "table": "test_table",
            "batch_mode": "BULK"
        }
        
        stage = DatabaseOutputStage(config=config)
        
        # Deliver output
        result = await stage.deliver_output(self.test_data, self.context)
        
        # Verify result
        self.assertEqual(result["table"], "test_table")
        self.assertEqual(result["rows_inserted"], 3)
        
        # Verify executemany was called once for all records
        mock_connection.executemany.assert_called_once()
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiosqlite')
    async def test_deliver_output_chunked_mode(self, mock_aiosqlite):
        """Test delivering output in chunked insertion mode."""
        # Set up mock connection
        mock_connection = AsyncMock()
        mock_aiosqlite.connect.return_value = mock_connection
        
        # Create a larger dataset
        large_data = [{"id": i, "name": f"User{i}", "age": 20 + i} for i in range(150)]
        self.context.set("data", large_data)
        
        config = {
            "db_type": "SQLITE",
            "connection_params": {
                "database": self.db_path
            },
            "table": "test_table",
            "batch_mode": "CHUNKED",
            "batch_size": 50
        }
        
        stage = DatabaseOutputStage(config=config)
        
        # Deliver output
        result = await stage.deliver_output(large_data, self.context)
        
        # Verify result
        self.assertEqual(result["table"], "test_table")
        self.assertEqual(result["rows_inserted"], 150)
        self.assertEqual(result["batches_processed"], 3)  # 150 records in batches of 50
        
        # Verify executemany was called 3 times (once per batch)
        self.assertEqual(mock_connection.executemany.call_count, 3)
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiosqlite')
    async def test_deliver_output_with_transaction(self, mock_aiosqlite):
        """Test delivering output with transaction support."""
        # Set up mock connection
        mock_connection = AsyncMock()
        mock_aiosqlite.connect.return_value = mock_connection
        
        config = {
            "db_type": "SQLITE",
            "connection_params": {
                "database": self.db_path
            },
            "table": "test_table",
            "use_transactions": True
        }
        
        stage = DatabaseOutputStage(config=config)
        
        # Deliver output
        result = await stage.deliver_output(self.test_data, self.context)
        
        # Verify transaction commands were executed
        expected_calls = [
            unittest.mock.call("BEGIN TRANSACTION"),  # Start transaction
            unittest.mock.call(unittest.mock.ANY),    # Insert statement
        ]
        mock_connection.execute.assert_has_calls(expected_calls, any_order=False)
        
        # Verify commit was called
        mock_connection.commit.assert_called_once()
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiosqlite')
    async def test_deliver_output_error_handling(self, mock_aiosqlite):
        """Test error handling during output delivery."""
        # Set up mock connection with an error during execute
        mock_connection = AsyncMock()
        mock_connection.executemany.side_effect = Exception("Database error")
        mock_aiosqlite.connect.return_value = mock_connection
        
        config = {
            "db_type": "SQLITE",
            "connection_params": {
                "database": self.db_path
            },
            "table": "test_table",
            "use_transactions": True
        }
        
        stage = DatabaseOutputStage(config=config)
        
        # Deliver output (should handle the error)
        result = await stage.deliver_output(self.test_data, self.context)
        
        # Verify result is None due to error
        self.assertIsNone(result)
        
        # Verify context has error
        self.assertTrue(self.context.has_errors())
        
        # Verify rollback was attempted
        mock_connection.rollback.assert_called_once()
        
        # Verify connection was closed
        mock_connection.close.assert_called_once()
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    async def test_is_transient_error(self):
        """Test identifying transient database errors."""
        config = {
            "db_type": "SQLITE",
            "connection_params": {
                "database": self.db_path
            },
            "table": "test_table"
        }
        
        with patch('core.pipeline.stages.output.database_output.aiosqlite'):
            stage = DatabaseOutputStage(config=config)
            
            # Test transient errors
            self.assertTrue(stage._is_transient_error("database is locked"))
            self.assertTrue(stage._is_transient_error("SQLITE_BUSY: database is locked"))
            self.assertTrue(stage._is_transient_error("connection reset by peer"))
            self.assertTrue(stage._is_transient_error("too many connections"))
            
            # Test non-transient errors
            self.assertFalse(stage._is_transient_error("syntax error in SQL statement"))
            self.assertFalse(stage._is_transient_error("no such table: test_table"))
            self.assertFalse(stage._is_transient_error("permission denied"))


class TestSQLiteOutputStage(unittest.TestCase):
    """Test cases for the SQLiteOutputStage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        
        # Test data
        self.test_data = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35}
        ]
        
        self.context = PipelineContext()
        self.context.set("data", self.test_data)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiosqlite')
    def test_init_with_path_in_config(self, mock_aiosqlite):
        """Test initialization with database path in config."""
        config = {
            "database_path": self.db_path,
            "table": "test_table"
        }
        
        stage = SQLiteOutputStage(config=config)
        
        # Verify database type is set to SQLite
        self.assertEqual(stage.db_type, DatabaseType.SQLITE)
        
        # Verify database path is correctly set in connection params
        self.assertEqual(stage.connection_params["database"], self.db_path)
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiosqlite')
    def test_init_with_path_in_connection_params(self, mock_aiosqlite):
        """Test initialization with database path in connection params."""
        config = {
            "connection_params": {
                "database": self.db_path
            },
            "table": "test_table"
        }
        
        stage = SQLiteOutputStage(config=config)
        
        # Verify database path is correctly set
        self.assertEqual(stage.connection_params["database"], self.db_path)
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiosqlite')
    def test_init_without_database_path(self, mock_aiosqlite):
        """Test initialization without database path."""
        config = {
            "table": "test_table"
        }
        
        with self.assertRaises(ValueError):
            SQLiteOutputStage(config=config)
    
    @patch('core.pipeline.stages.output.database_output.SQLITE_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiosqlite')
    async def test_connect_with_pragma(self, mock_aiosqlite):
        """Test connection with PRAGMA statements."""
        # Set up mock connection
        mock_connection = AsyncMock()
        mock_aiosqlite.connect.return_value = mock_connection
        
        config = {
            "database_path": self.db_path,
            "table": "test_table",
            "pragma_statements": [
                "PRAGMA journal_mode=WAL",
                "PRAGMA synchronous=NORMAL"
            ]
        }
        
        stage = SQLiteOutputStage(config=config)
        
        # Connect to database
        await stage._connect()
        
        # Verify PRAGMA statements were executed
        self.assertEqual(mock_connection.execute.call_count, 2)
        mock_connection.execute.assert_any_call("PRAGMA journal_mode=WAL")
        mock_connection.execute.assert_any_call("PRAGMA synchronous=NORMAL")


class TestMySQLOutputStage(unittest.TestCase):
    """Test cases for the MySQLOutputStage."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Test data
        self.test_data = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35}
        ]
        
        self.context = PipelineContext()
        self.context.set("data", self.test_data)
    
    @patch('core.pipeline.stages.output.database_output.MYSQL_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiomysql')
    def test_init_with_default_config(self, mock_aiomysql):
        """Test initialization with default configuration."""
        config = {
            "connection_params": {
                "host": "localhost",
                "port": 3306,
                "user": "user",
                "password": "password",
                "database": "test_db"
            },
            "table": "test_table"
        }
        
        stage = MySQLOutputStage(config=config)
        
        # Verify database type is set to MySQL
        self.assertEqual(stage.db_type, DatabaseType.MYSQL)
        
        # Verify default MySQL-specific options
        self.assertFalse(stage.insert_ignore)
        self.assertFalse(stage.replace)
    
    @patch('core.pipeline.stages.output.database_output.MYSQL_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiomysql')
    def test_init_with_custom_config(self, mock_aiomysql):
        """Test initialization with custom configuration."""
        config = {
            "connection_params": {
                "host": "localhost",
                "port": 3306,
                "user": "user",
                "password": "password",
                "database": "test_db"
            },
            "table": "test_table",
            "insert_ignore": True
        }
        
        stage = MySQLOutputStage(config=config)
        
        # Verify MySQL-specific options
        self.assertTrue(stage.insert_ignore)
        self.assertFalse(stage.replace)
    
    @patch('core.pipeline.stages.output.database_output.MYSQL_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiomysql')
    def test_init_with_conflicting_options(self, mock_aiomysql):
        """Test initialization with conflicting options."""
        config = {
            "connection_params": {
                "host": "localhost",
                "port": 3306,
                "user": "user",
                "password": "password",
                "database": "test_db"
            },
            "table": "test_table",
            "insert_ignore": True,
            "replace": True
        }
        
        # Should raise an error because insert_ignore and replace are mutually exclusive
        with self.assertRaises(ValueError):
            MySQLOutputStage(config=config)
    
    @patch('core.pipeline.stages.output.database_output.MYSQL_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiomysql')
    async def test_insert_record_standard(self, mock_aiomysql):
        """Test inserting a record with standard INSERT."""
        # Set up mock connection and cursor
        mock_cursor = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_aiomysql.connect.return_value = mock_connection
        
        config = {
            "connection_params": {
                "host": "localhost",
                "port": 3306,
                "user": "user",
                "password": "password",
                "database": "test_db"
            },
            "table": "test_table"
        }
        
        stage = MySQLOutputStage(config=config)
        stage.connection = mock_connection
        
        # Insert a record
        record = {"id": 1, "name": "Alice", "age": 30}
        await stage._insert_record(record)
        
        # Verify cursor execute was called with correct SQL
        mock_cursor.execute.assert_called_once()
        args, kwargs = mock_cursor.execute.call_args
        sql = args[0]
        
        # Should be standard INSERT
        self.assertIn("INSERT INTO test_table", sql)
        self.assertNotIn("IGNORE", sql)
        self.assertNotIn("REPLACE", sql)
    
    @patch('core.pipeline.stages.output.database_output.MYSQL_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiomysql')
    async def test_insert_record_ignore(self, mock_aiomysql):
        """Test inserting a record with INSERT IGNORE."""
        # Set up mock connection and cursor
        mock_cursor = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_aiomysql.connect.return_value = mock_connection
        
        config = {
            "connection_params": {
                "host": "localhost",
                "port": 3306,
                "user": "user",
                "password": "password",
                "database": "test_db"
            },
            "table": "test_table",
            "insert_ignore": True
        }
        
        stage = MySQLOutputStage(config=config)
        stage.connection = mock_connection
        
        # Insert a record
        record = {"id": 1, "name": "Alice", "age": 30}
        await stage._insert_record(record)
        
        # Verify cursor execute was called with correct SQL
        mock_cursor.execute.assert_called_once()
        args, kwargs = mock_cursor.execute.call_args
        sql = args[0]
        
        # Should be INSERT IGNORE
        self.assertIn("INSERT IGNORE INTO test_table", sql)
    
    @patch('core.pipeline.stages.output.database_output.MYSQL_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.aiomysql')
    async def test_insert_record_replace(self, mock_aiomysql):
        """Test inserting a record with REPLACE."""
        # Set up mock connection and cursor
        mock_cursor = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_aiomysql.connect.return_value = mock_connection
        
        config = {
            "connection_params": {
                "host": "localhost",
                "port": 3306,
                "user": "user",
                "password": "password",
                "database": "test_db"
            },
            "table": "test_table",
            "replace": True
        }
        
        stage = MySQLOutputStage(config=config)
        stage.connection = mock_connection
        
        # Insert a record
        record = {"id": 1, "name": "Alice", "age": 30}
        await stage._insert_record(record)
        
        # Verify cursor execute was called with correct SQL
        mock_cursor.execute.assert_called_once()
        args, kwargs = mock_cursor.execute.call_args
        sql = args[0]
        
        # Should be REPLACE INTO
        self.assertIn("REPLACE INTO test_table", sql)


class TestPostgresOutputStage(unittest.TestCase):
    """Test cases for the PostgresOutputStage."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Test data
        self.test_data = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35}
        ]
        
        self.context = PipelineContext()
        self.context.set("data", self.test_data)
    
    @patch('core.pipeline.stages.output.database_output.POSTGRES_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.asyncpg')
    def test_init_with_default_config(self, mock_asyncpg):
        """Test initialization with default configuration."""
        config = {
            "connection_params": {
                "host": "localhost",
                "port": 5432,
                "user": "user",
                "password": "password",
                "database": "test_db"
            },
            "table": "test_table"
        }
        
        stage = PostgresOutputStage(config=config)
        
        # Verify database type is set to PostgreSQL
        self.assertEqual(stage.db_type, DatabaseType.POSTGRES)
        
        # Verify default PostgreSQL-specific options
        self.assertIsNone(stage.upsert_constraint)
        self.assertEqual(stage.upsert_action, "DO NOTHING")
        self.assertEqual(stage.upsert_columns, [])
    
    @patch('core.pipeline.stages.output.database_output.POSTGRES_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.asyncpg')
    def test_init_with_upsert_config(self, mock_asyncpg):
        """Test initialization with upsert configuration."""
        config = {
            "connection_params": {
                "host": "localhost",
                "port": 5432,
                "user": "user",
                "password": "password",
                "database": "test_db"
            },
            "table": "test_table",
            "upsert_constraint": "id",
            "upsert_action": "DO UPDATE SET",
            "upsert_columns": ["name", "age"]
        }
        
        stage = PostgresOutputStage(config=config)
        
        # Verify PostgreSQL-specific options
        self.assertEqual(stage.upsert_constraint, "id")
        self.assertEqual(stage.upsert_action, "DO UPDATE SET")
        self.assertEqual(stage.upsert_columns, ["name", "age"])
    
    @patch('core.pipeline.stages.output.database_output.POSTGRES_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.asyncpg')
    def test_init_with_incomplete_upsert_config(self, mock_asyncpg):
        """Test initialization with incomplete upsert configuration."""
        config = {
            "connection_params": {
                "host": "localhost",
                "port": 5432,
                "user": "user",
                "password": "password",
                "database": "test_db"
            },
            "table": "test_table",
            "upsert_constraint": "id",
            "upsert_action": "DO UPDATE SET"
            # Missing upsert_columns which should cause an error
        }
        
        # Should raise an error because upsert_columns is required for DO UPDATE SET
        with self.assertRaises(ValueError):
            PostgresOutputStage(config=config)
    
    @patch('core.pipeline.stages.output.database_output.POSTGRES_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.asyncpg')
    async def test_insert_record_standard(self, mock_asyncpg):
        """Test inserting a record with standard INSERT."""
        # Set up mock connection
        mock_connection = AsyncMock()
        mock_asyncpg.connect.return_value = mock_connection
        
        config = {
            "connection_params": {
                "host": "localhost",
                "port": 5432,
                "user": "user",
                "password": "password",
                "database": "test_db"
            },
            "table": "test_table"
        }
        
        stage = PostgresOutputStage(config=config)
        stage.connection = mock_connection
        
        # Insert a record
        record = {"id": 1, "name": "Alice", "age": 30}
        await stage._insert_record(record)
        
        # Verify execute was called with correct SQL
        mock_connection.execute.assert_called_once()
        args, kwargs = mock_connection.execute.call_args
        sql = args[0]
        
        # Should be standard INSERT without ON CONFLICT
        self.assertIn("INSERT INTO test_table", sql)
        self.assertNotIn("ON CONFLICT", sql)
    
    @patch('core.pipeline.stages.output.database_output.POSTGRES_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.asyncpg')
    async def test_insert_record_with_upsert(self, mock_asyncpg):
        """Test inserting a record with ON CONFLICT clause."""
        # Set up mock connection
        mock_connection = AsyncMock()
        mock_asyncpg.connect.return_value = mock_connection
        
        config = {
            "connection_params": {
                "host": "localhost",
                "port": 5432,
                "user": "user",
                "password": "password",
                "database": "test_db"
            },
            "table": "test_table",
            "upsert_constraint": "id",
            "upsert_action": "DO NOTHING"
        }
        
        stage = PostgresOutputStage(config=config)
        stage.connection = mock_connection
        
        # Insert a record
        record = {"id": 1, "name": "Alice", "age": 30}
        await stage._insert_record(record)
        
        # Verify execute was called with correct SQL
        mock_connection.execute.assert_called_once()
        args, kwargs = mock_connection.execute.call_args
        sql = args[0]
        
        # Should include ON CONFLICT DO NOTHING
        self.assertIn("INSERT INTO test_table", sql)
        self.assertIn("ON CONFLICT (id)", sql)
        self.assertIn("DO NOTHING", sql)
    
    @patch('core.pipeline.stages.output.database_output.POSTGRES_AVAILABLE', True)
    @patch('core.pipeline.stages.output.database_output.asyncpg')
    async def test_insert_record_with_update(self, mock_asyncpg):
        """Test inserting a record with ON CONFLICT DO UPDATE SET."""
        # Set up mock connection
        mock_connection = AsyncMock()
        mock_asyncpg.connect.return_value = mock_connection
        
        config = {
            "connection_params": {
                "host": "localhost",
                "port": 5432,
                "user": "user",
                "password": "password",
                "database": "test_db"
            },
            "table": "test_table",
            "upsert_constraint": "id",
            "upsert_action": "DO UPDATE SET",
            "upsert_columns": ["name", "age"]
        }
        
        stage = PostgresOutputStage(config=config)
        stage.connection = mock_connection
        
        # Insert a record
        record = {"id": 1, "name": "Alice", "age": 30}
        await stage._insert_record(record)
        
        # Verify execute was called with correct SQL
        mock_connection.execute.assert_called_once()
        args, kwargs = mock_connection.execute.call_args
        sql = args[0]
        
        # Should include ON CONFLICT DO UPDATE SET
        self.assertIn("INSERT INTO test_table", sql)
        self.assertIn("ON CONFLICT (id)", sql)
        self.assertIn("DO UPDATE SET", sql)
        self.assertIn("name = EXCLUDED.name", sql)
        self.assertIn("age = EXCLUDED.age", sql)


if __name__ == "__main__":
    unittest.main()