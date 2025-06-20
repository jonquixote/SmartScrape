"""
Test Input Stages.

This module tests the file input and HTTP input stages.
"""

import asyncio
import unittest
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch

from core.pipeline.context import PipelineContext
from core.pipeline.stages.input import FileInputStage, HttpInputStage


class TestFileInputStage(unittest.IsolatedAsyncioTestCase):
    """Test case for the FileInputStage."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.write(b'{"test": "data"}')
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up after each test method."""
        os.unlink(self.temp_file.name)
    
    async def test_process_file_exists(self):
        """Test processing when file exists."""
        stage = FileInputStage(name="test_file_input")
        context = PipelineContext({"file_path": self.temp_file.name})
        
        result = await stage.process(context)
        
        self.assertTrue(result)
        self.assertIn("file_content", context.data)
        self.assertEqual(context.get("file_content"), '{"test": "data"}')
    
    async def test_process_file_does_not_exist(self):
        """Test processing when file does not exist."""
        stage = FileInputStage(name="test_file_input")
        context = PipelineContext({"file_path": "non_existent_file.txt"})
        
        result = await stage.process(context)
        
        self.assertFalse(result)
        self.assertIn("error", context.data)
        self.assertIn("FileNotFoundError", context.get("error"))
    
    async def test_process_no_file_path(self):
        """Test processing when no file path is provided."""
        stage = FileInputStage(name="test_file_input")
        context = PipelineContext()
        
        result = await stage.process(context)
        
        self.assertFalse(result)
        self.assertIn("error", context.data)
        self.assertIn("No file path specified", context.get("error"))
    
    async def test_initialize_and_cleanup(self):
        """Test initialize and cleanup methods."""
        stage = FileInputStage(name="test_file_input")
        
        # These should run without errors
        await stage.initialize()
        await stage.cleanup()


class TestHttpInputStage(unittest.IsolatedAsyncioTestCase):
    """Test case for the HttpInputStage."""
    
    async def test_process_success(self):
        """Test successful HTTP request."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = "<html>Test content</html>"
        mock_response.headers = {"Content-Type": "text/html"}
        
        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            stage = HttpInputStage(name="test_http_input")
            context = PipelineContext({"url": "https://example.com"})
            
            result = await stage.process(context)
            
            self.assertTrue(result)
            self.assertIn("html_content", context.data)
            self.assertEqual(context.get("html_content"), "<html>Test content</html>")
            self.assertIn("response_headers", context.data)
            self.assertEqual(context.get("response_headers"), {"Content-Type": "text/html"})
    
    async def test_process_http_error(self):
        """Test HTTP error handling."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text.return_value = "Not Found"
        
        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            stage = HttpInputStage(name="test_http_input")
            context = PipelineContext({"url": "https://example.com/not-found"})
            
            result = await stage.process(context)
            
            self.assertFalse(result)
            self.assertIn("error", context.data)
            self.assertIn("HTTP 404", context.get("error"))
    
    async def test_process_connection_error(self):
        """Test connection error handling."""
        with patch("aiohttp.ClientSession.get", side_effect=Exception("Connection error")):
            stage = HttpInputStage(name="test_http_input")
            context = PipelineContext({"url": "https://example.com"})
            
            result = await stage.process(context)
            
            self.assertFalse(result)
            self.assertIn("error", context.data)
            self.assertIn("Connection error", context.get("error"))
    
    async def test_process_no_url(self):
        """Test processing when no URL is provided."""
        stage = HttpInputStage(name="test_http_input")
        context = PipelineContext()
        
        result = await stage.process(context)
        
        self.assertFalse(result)
        self.assertIn("error", context.data)
        self.assertIn("No URL specified", context.get("error"))
    
    async def test_custom_headers(self):
        """Test processing with custom headers."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = "<html>Test content</html>"
        
        custom_headers = {"User-Agent": "SmartScrape Test", "Accept-Language": "en-US"}
        
        with patch("aiohttp.ClientSession.get", return_value=mock_response) as mock_get:
            stage = HttpInputStage(name="test_http_input")
            context = PipelineContext({
                "url": "https://example.com",
                "headers": custom_headers
            })
            
            result = await stage.process(context)
            
            self.assertTrue(result)
            mock_get.assert_called_once_with(
                "https://example.com", 
                headers=custom_headers,
                timeout=30,
                ssl=None
            )
    
    async def test_timeout_configuration(self):
        """Test configuration of request timeout."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = "<html>Test content</html>"
        
        with patch("aiohttp.ClientSession.get", return_value=mock_response) as mock_get:
            stage = HttpInputStage(name="test_http_input", config={"timeout": 60})
            context = PipelineContext({"url": "https://example.com"})
            
            result = await stage.process(context)
            
            self.assertTrue(result)
            mock_get.assert_called_once_with(
                "https://example.com", 
                headers={},
                timeout=60,
                ssl=None
            )