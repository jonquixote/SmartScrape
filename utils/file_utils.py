"""
Async file operations for SmartScrape.

This module provides non-blocking file operations using aiofiles for:
- Reading/writing files asynchronously
- JSON operations
- CSV operations
- Handling large files efficiently
"""

import os
import json
import csv
import logging
from typing import Dict, List, Any, Optional, Union, BinaryIO, TextIO, Iterable

# Import aiofiles for non-blocking file operations
import aiofiles
import aiofiles.os

# Import retry utilities
from utils.retry_utils import with_file_operation_retry

# Configure logging
logger = logging.getLogger(__name__)

@with_file_operation_retry(max_attempts=3)
async def read_file(filepath: str, mode: str = 'r', encoding: str = 'utf-8') -> str:
    """
    Read a file asynchronously.
    
    Args:
        filepath: Path to the file
        mode: File mode (default: 'r')
        encoding: File encoding (default: 'utf-8')
        
    Returns:
        File contents as a string
    """
    logger.debug(f"Reading file {filepath}")
    async with aiofiles.open(filepath, mode=mode, encoding=encoding) as f:
        return await f.read()

@with_file_operation_retry(max_attempts=3)
async def write_file(filepath: str, content: str, mode: str = 'w', encoding: str = 'utf-8') -> int:
    """
    Write to a file asynchronously.
    
    Args:
        filepath: Path to the file
        content: Content to write
        mode: File mode (default: 'w')
        encoding: File encoding (default: 'utf-8')
        
    Returns:
        Number of bytes written
    """
    logger.debug(f"Writing to file {filepath}")
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    async with aiofiles.open(filepath, mode=mode, encoding=encoding) as f:
        return await f.write(content)

@with_file_operation_retry(max_attempts=3)
async def append_file(filepath: str, content: str, encoding: str = 'utf-8') -> int:
    """
    Append to a file asynchronously.
    
    Args:
        filepath: Path to the file
        content: Content to append
        encoding: File encoding (default: 'utf-8')
        
    Returns:
        Number of bytes written
    """
    logger.debug(f"Appending to file {filepath}")
    async with aiofiles.open(filepath, mode='a', encoding=encoding) as f:
        return await f.write(content)

@with_file_operation_retry(max_attempts=3)
async def read_json(filepath: str, encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Read a JSON file asynchronously.
    
    Args:
        filepath: Path to the JSON file
        encoding: File encoding (default: 'utf-8')
        
    Returns:
        Parsed JSON data
    """
    logger.debug(f"Reading JSON from {filepath}")
    async with aiofiles.open(filepath, mode='r', encoding=encoding) as f:
        content = await f.read()
        return json.loads(content)

@with_file_operation_retry(max_attempts=3)
async def write_json(
    filepath: str, 
    data: Union[Dict[str, Any], List[Any]], 
    encoding: str = 'utf-8',
    indent: int = 2
) -> int:
    """
    Write JSON data to a file asynchronously.
    
    Args:
        filepath: Path to the JSON file
        data: Data to write (dict or list)
        encoding: File encoding (default: 'utf-8')
        indent: JSON indentation (default: 2)
        
    Returns:
        Number of bytes written
    """
    logger.debug(f"Writing JSON to {filepath}")
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    async with aiofiles.open(filepath, mode='w', encoding=encoding) as f:
        # Log Point 3.1: Log just before json.dumps in file_utils.py
        logger.info(f"FILE_UTILS_JSON_SAVE_LOG_3.1 About to write JSON to file: {filepath}")
        logger.info(f"FILE_UTILS_JSON_SAVE_LOG_3.1 Data type: {type(data)}, length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        if isinstance(data, dict):
            data_keys = list(data.keys())
            logger.info(f"FILE_UTILS_JSON_SAVE_LOG_3.1 Data dict keys: {data_keys}")
            if 'data' in data:
                data_content = data.get('data', [])
                logger.info(f"FILE_UTILS_JSON_SAVE_LOG_3.1 Data['data'] type: {type(data_content)}, length: {len(data_content) if hasattr(data_content, '__len__') else 'N/A'}")
        elif isinstance(data, list) and len(data) > 0:
            logger.info(f"FILE_UTILS_JSON_SAVE_LOG_3.1 Sample data[0]: {str(data[0])[:200]}")
        
        json_str = json.dumps(data, indent=indent, ensure_ascii=False)
        return await f.write(json_str)

@with_file_operation_retry(max_attempts=3)
async def read_csv(
    filepath: str, 
    has_header: bool = True, 
    delimiter: str = ',',
    encoding: str = 'utf-8'
) -> List[Dict[str, str]]:
    """
    Read a CSV file asynchronously.
    
    Args:
        filepath: Path to the CSV file
        has_header: Whether the CSV has a header row (default: True)
        delimiter: CSV delimiter (default: ',')
        encoding: File encoding (default: 'utf-8')
        
    Returns:
        List of dictionaries representing the CSV rows
    """
    logger.debug(f"Reading CSV from {filepath}")
    async with aiofiles.open(filepath, mode='r', encoding=encoding, newline='') as f:
        content = await f.read()
        
        # Process the CSV in memory
        reader = csv.reader(content.splitlines(), delimiter=delimiter)
        rows = list(reader)
        
        if not rows:
            return []
        
        if has_header:
            header = rows[0]
            return [dict(zip(header, row)) for row in rows[1:]]
        else:
            return [dict(enumerate(row)) for row in rows]

@with_file_operation_retry(max_attempts=3)
async def write_csv(
    filepath: str, 
    data: List[Dict[str, Any]], 
    fieldnames: Optional[List[str]] = None,
    delimiter: str = ',',
    encoding: str = 'utf-8'
) -> int:
    """
    Write data to a CSV file asynchronously.
    
    Args:
        filepath: Path to the CSV file
        data: List of dictionaries to write
        fieldnames: List of field names (columns). If not provided, will use keys from first row.
        delimiter: CSV delimiter (default: ',')
        encoding: File encoding (default: 'utf-8')
        
    Returns:
        Number of bytes written
    """
    logger.debug(f"Writing CSV to {filepath}")
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    if not data:
        logger.warning(f"No data to write to {filepath}")
        return 0
    
    # If fieldnames not provided, use keys from first row
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    # Create CSV in memory
    rows = []
    rows.append(delimiter.join(fieldnames))
    for row in data:
        csv_row = [str(row.get(field, '')) for field in fieldnames]
        rows.append(delimiter.join(csv_row))
    
    content = '\n'.join(rows)
    
    # Write to file
    async with aiofiles.open(filepath, mode='w', encoding=encoding, newline='') as f:
        return await f.write(content)

@with_file_operation_retry(max_attempts=3)
async def file_exists(filepath: str) -> bool:
    """
    Check if a file exists asynchronously.
    
    Args:
        filepath: Path to the file
        
    Returns:
        True if the file exists, False otherwise
    """
    try:
        await aiofiles.os.stat(filepath)
        return True
    except FileNotFoundError:
        return False

@with_file_operation_retry(max_attempts=3)
async def create_directory(dirpath: str) -> None:
    """
    Create a directory asynchronously (including parents).
    
    Args:
        dirpath: Path to the directory
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

@with_file_operation_retry(max_attempts=3)
async def list_directory(dirpath: str) -> List[str]:
    """
    List contents of a directory asynchronously.
    
    Args:
        dirpath: Path to the directory
        
    Returns:
        List of file and directory names
    """
    return await aiofiles.os.listdir(dirpath)

async def process_large_file(
    filepath: str,
    process_func: callable,
    chunk_size: int = 1024 * 1024,  # 1MB chunks
    encoding: str = 'utf-8'
) -> Any:
    """
    Process a large file in chunks to avoid loading it all into memory.
    
    Args:
        filepath: Path to the file
        process_func: Function to process each chunk (receives chunk of text)
        chunk_size: Size of chunks to read
        encoding: File encoding
        
    Returns:
        Result from the process function
    """
    logger.info(f"Processing large file {filepath} in chunks")
    
    result = None
    async with aiofiles.open(filepath, mode='r', encoding=encoding) as f:
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            
            # Call the processing function on the chunk
            chunk_result = await process_func(chunk)
            
            # Update the overall result (function needs to handle this logic)
            if result is None:
                result = chunk_result
            else:
                result = await process_func(result, chunk_result, is_merge=True)
    
    return result

async def stream_large_file(
    filepath: str,
    encoding: str = 'utf-8',
    chunk_size: int = 1024 * 1024  # 1MB chunks
):
    """
    Stream a large file in chunks (generator).
    
    Args:
        filepath: Path to the file
        encoding: File encoding
        chunk_size: Size of chunks to read
        
    Yields:
        Chunks of the file
    """
    async with aiofiles.open(filepath, mode='r', encoding=encoding) as f:
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            yield chunk