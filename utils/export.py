import json
import logging
import asyncio
import aiofiles
from typing import Dict, List, Any, Optional, Union, BinaryIO
from io import BytesIO
import pandas as pd
import markdown
import html2text

# Import our new async file utilities
from utils.file_utils import write_file, write_json, write_csv
from utils.retry_utils import with_file_operation_retry

# Configure logging
logger = logging.getLogger(__name__)

async def generate_json_export(data: List[Dict[str, Any]], pretty: bool = True) -> str:
    """
    Generate a JSON string from the extracted data.
    
    Args:
        data: List of data items to export
        pretty: Whether to format the JSON with indentation
        
    Returns:
        JSON string representation
    """
    indent = 2 if pretty else None
    return json.dumps(data, indent=indent)

async def generate_csv_export(data: List[Dict[str, Any]]) -> str:
    """
    Generate a CSV string from the extracted data with flattened structure.
    
    Args:
        data: List of data items to export
        
    Returns:
        CSV string representation
    """
    # This is a more sophisticated flattening function for nested data
    flattened_data = []
    
    for item in data:
        base_item = {
            "source_url": item["source_url"],
            "depth": item["depth"],
            "score": item["score"]
        }
        
        # Handle different data types
        if isinstance(item["data"], list):
            # For list data, create separate rows for each item
            for i, subitem in enumerate(item["data"]):
                row = base_item.copy()
                if isinstance(subitem, dict):
                    # Flatten dict items
                    for key, value in subitem.items():
                        # Handle nested values by converting to string
                        if isinstance(value, (dict, list)):
                            row[key] = json.dumps(value)
                        else:
                            row[key] = value
                else:
                    row[f"item_{i}"] = subitem
                flattened_data.append(row)
        elif isinstance(item["data"], dict):
            # For dict data, flatten keys into columns
            row = base_item.copy()
            for key, value in item["data"].items():
                # Handle nested values by converting to string
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value)
                else:
                    row[key] = value
            flattened_data.append(row)
        else:
            # For scalar data
            row = base_item.copy()
            row["data"] = item["data"]
            flattened_data.append(row)
    
    # If no data, return an empty CSV
    if not flattened_data:
        return "source_url,depth,score,data\n"
    
    # Convert to DataFrame and then to CSV
    try:
        df = pd.DataFrame(flattened_data)
        return df.to_csv(index=False)
    except Exception as e:
        logger.error(f"Error generating CSV: {e}")
        # Fallback to basic CSV
        return "source_url,depth,score,data\n" + "\n".join([
            f"{item['source_url']},{item['depth']},{item['score']},{json.dumps(item['data'])}"
            for item in data
        ])

async def generate_excel_export(data: List[Dict[str, Any]]) -> bytes:
    """
    Generate an Excel file from the extracted data with flattened structure.
    
    Args:
        data: List of data items to export
        
    Returns:
        Excel file as bytes
    """
    # Use the same flattened data approach as CSV
    flattened_data = []
    
    for item in data:
        base_item = {
            "source_url": item["source_url"],
            "depth": item["depth"],
            "score": item["score"]
        }
        
        # Handle different data types
        if isinstance(item["data"], list):
            for i, subitem in enumerate(item["data"]):
                row = base_item.copy()
                if isinstance(subitem, dict):
                    for key, value in subitem.items():
                        if isinstance(value, (dict, list)):
                            row[key] = json.dumps(value)
                        else:
                            row[key] = value
                else:
                    row[f"item_{i}"] = subitem
                flattened_data.append(row)
        elif isinstance(item["data"], dict):
            row = base_item.copy()
            for key, value in item["data"].items():
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value)
                else:
                    row[key] = value
            flattened_data.append(row)
        else:
            row = base_item.copy()
            row["data"] = item["data"]
            flattened_data.append(row)
    
    # Convert to DataFrame and use BytesIO for in-memory Excel file
    try:
        df = pd.DataFrame(flattened_data)
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        return excel_buffer.getvalue()
    except Exception as e:
        logger.error(f"Error generating Excel: {e}")
        # Return empty Excel file in case of error
        df = pd.DataFrame({"error": ["Failed to generate Excel file"]})
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        return excel_buffer.getvalue()

async def generate_markdown_export(data: List[Dict[str, Any]]) -> str:
    """
    Generate a Markdown representation of the extracted data.
    
    Args:
        data: List of data items to export
        
    Returns:
        Markdown string representation
    """
    md_lines = ["# Extracted Content\n"]
    
    for i, item in enumerate(data):
        # Add item header with URL and metadata
        md_lines.append(f"## Item {i+1}: {item['source_url']}\n")
        md_lines.append(f"- **Depth**: {item['depth']}")
        md_lines.append(f"- **Score**: {item['score']}")
        md_lines.append("")
        
        # Process the data based on its type
        if isinstance(item["data"], dict):
            # For dictionary data, create sections for each key
            for key, value in item["data"].items():
                # Skip HTML content in markdown export
                if key.endswith('_html') or key == 'html' or key == 'raw_html':
                    continue
                    
                md_lines.append(f"### {key.replace('_', ' ').title()}")
                
                if isinstance(value, str):
                    md_lines.append(f"\n{value}\n")
                elif isinstance(value, list):
                    for list_item in value:
                        md_lines.append(f"- {list_item}")
                    md_lines.append("")
                else:
                    md_lines.append(f"\n{json.dumps(value)}\n")
                    
        elif isinstance(item["data"], list):
            # For list data, create bulleted items
            md_lines.append("### Content")
            for list_item in item["data"]:
                if isinstance(list_item, dict):
                    # For dictionary items in a list, create sub-sections
                    for key, value in list_item.items():
                        if not key.endswith('_html') and key != 'html' and key != 'raw_html':
                            md_lines.append(f"#### {key.replace('_', ' ').title()}")
                            md_lines.append(f"{value}\n")
                else:
                    # For simple list items
                    md_lines.append(f"- {list_item}")
            md_lines.append("")
            
        else:
            # For scalar data
            md_lines.append("### Content")
            md_lines.append(f"\n{item['data']}\n")
            
        md_lines.append("\n---\n")
    
    return "\n".join(md_lines)

async def generate_html_export(data: List[Dict[str, Any]], include_raw_html: bool = False) -> str:
    """
    Generate an HTML representation of the extracted data.
    
    Args:
        data: List of data items to export
        include_raw_html: Whether to include original HTML content
        
    Returns:
        HTML string representation
    """
    # First generate markdown
    md_content = await generate_markdown_export(data)
    
    # Convert to HTML
    html_content = markdown.markdown(md_content)
    
    # If we want to include raw HTML, process the data to embed original HTML
    if include_raw_html:
        # This is a placeholder for where we would insert raw HTML content
        # We'll manually insert raw HTML sections after the main content
        html_raw_sections = []
        
        for i, item in enumerate(data):
            if isinstance(item.get("data"), dict):
                for key, value in item["data"].items():
                    if (key.endswith('_html') or key == 'html' or key == 'raw_html') and isinstance(value, str):
                        html_raw_sections.append(
                            f'<div class="raw-html-section">'
                            f'<h3>Raw HTML for {key} (Item {i+1})</h3>'
                            f'<pre class="html-code">{value.replace("<", "&lt;").replace(">", "&gt;")}</pre>'
                            f'<div class="html-preview"><h4>Preview:</h4>{value}</div>'
                            f'</div>'
                        )
        
        # Append raw HTML sections if any were found
        if html_raw_sections:
            html_content += '<h2>Raw HTML Content</h2>' + ''.join(html_raw_sections)
    
    # Add enhanced styling with better visualization
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SmartScrape Export</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f9f9f9; }}
            h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
            h2 {{ color: #3498db; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 30px; }}
            h3 {{ color: #2980b9; }}
            h4 {{ color: #16a085; }}
            hr {{ margin: 30px 0; border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(0,0,0,0.2), rgba(0,0,0,0)); }}
            .metadata {{ color: #7f8c8d; font-size: 0.9em; background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            pre {{ background-color: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            code {{ font-family: Monaco, Consolas, 'Courier New', monospace; }}
            .raw-html-section {{ margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .html-code {{ max-height: 300px; overflow-y: auto; background-color: #2c3e50; color: #ecf0f1; padding: 15px; }}
            .html-preview {{ margin-top: 15px; padding: 15px; border: 1px dashed #ddd; background-color: white; }}
            .embedded-html {{ border: 1px solid #eee; padding: 10px; margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .export-date {{ text-align: right; color: #7f8c8d; font-size: 0.8em; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            {html_content}
            <div class="export-date">Exported on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
    </body>
    </html>
    """
    
    return styled_html

async def generate_plain_text_export(data: List[Dict[str, Any]]) -> str:
    """
    Generate a plain text representation of the extracted data.
    
    Args:
        data: List of data items to export
        
    Returns:
        Plain text string representation
    """
    # Create markdown first then convert to plain text
    md_content = await generate_markdown_export(data)
    
    # Use html2text to convert markdown to plain text
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0  # No wrapping
    
    plain_text = h.handle(markdown.markdown(md_content))
    
    return plain_text

async def generate_content_structure_export(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a structured representation of the content hierarchy.
    
    Args:
        data: List of data items to export
        
    Returns:
        Dictionary with content structure information
    """
    structure = {
        "total_items": len(data),
        "sources": [],
        "content_types": {},
        "hierarchy": {}
    }
    
    # Analyze data structure
    for item in data:
        # Track sources
        source_url = item.get("source_url", "unknown")
        structure["sources"].append(source_url)
        
        # Analyze content types
        if isinstance(item.get("data"), dict):
            structure["content_types"]["dict"] = structure["content_types"].get("dict", 0) + 1
            
            # Analyze dictionary keys to understand content structure
            for key in item["data"].keys():
                structure["hierarchy"][key] = structure["hierarchy"].get(key, 0) + 1
                
        elif isinstance(item.get("data"), list):
            structure["content_types"]["list"] = structure["content_types"].get("list", 0) + 1
            
            # Analyze list items if they're dictionaries
            for list_item in item["data"]:
                if isinstance(list_item, dict):
                    for key in list_item.keys():
                        structure["hierarchy"][key] = structure["hierarchy"].get(key, 0) + 1
        else:
            structure["content_types"]["scalar"] = structure["content_types"].get("scalar", 0) + 1
    
    return structure

async def _remove_raw_html(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove raw HTML content from the data to reduce export size.
    
    Args:
        data: List of data items to process
        
    Returns:
        Processed data with HTML content removed
    """
    processed_data = []
    
    for item in data:
        item_copy = item.copy()
        
        # Process the data field based on its type
        if isinstance(item_copy.get("data"), dict):
            # For dictionary data, remove HTML keys
            cleaned_dict = {}
            for key, value in item_copy["data"].items():
                if not (key.endswith('_html') or key == 'html' or key == 'raw_html'):
                    cleaned_dict[key] = value
            item_copy["data"] = cleaned_dict
            
        elif isinstance(item_copy.get("data"), list):
            # For list data, process each item if it's a dictionary
            cleaned_list = []
            for list_item in item_copy["data"]:
                if isinstance(list_item, dict):
                    cleaned_item = {}
                    for key, value in list_item.items():
                        if not (key.endswith('_html') or key == 'html' or key == 'raw_html'):
                            cleaned_item[key] = value
                    cleaned_list.append(cleaned_item if cleaned_item else list_item)
                else:
                    cleaned_list.append(list_item)
            item_copy["data"] = cleaned_list
            
        processed_data.append(item_copy)
    
    return processed_data

async def _adapt_to_content_type(data: List[Dict[str, Any]], base_format: str) -> List[Dict[str, Any]]:
    """
    Adapt the export format based on the content type for improved readability.
    
    Args:
        data: List of data items to process
        base_format: The base export format
        
    Returns:
        Processed data adapted to content type
    """
    # This function optimizes the data structure based on the content type
    # and the target export format
    
    processed_data = []
    
    for item in data:
        item_copy = item.copy()
        
        # If it's a dictionary type content
        if isinstance(item_copy.get("data"), dict):
            # For tabular formats, ensure nested structures are serialized
            if base_format.lower() in ['csv', 'excel']:
                for key, value in item_copy["data"].items():
                    if isinstance(value, (dict, list)):
                        item_copy["data"][key] = json.dumps(value)
                        
            # For text formats, convert HTML to text if present
            elif base_format.lower() in ['markdown', 'text']:
                for key, value in list(item_copy["data"].items()):
                    if key.endswith('_html') and isinstance(value, str):
                        h = html2text.HTML2Text()
                        h.ignore_links = False
                        item_copy["data"][key.replace('_html', '_text')] = h.handle(value)
                        
            # For HTML format, enhance embedded HTML if present
            elif base_format.lower() == 'html':
                for key, value in list(item_copy["data"].items()):
                    if key.endswith('_html') and isinstance(value, str):
                        # Add class for styling
                        item_copy["data"][key] = f'<div class="embedded-html">{value}</div>'
        
        # If it's a list type content
        elif isinstance(item_copy.get("data"), list):
            # For table formats, consider flattening to improve readability
            if base_format.lower() in ['csv', 'excel']:
                # If all list items are dictionaries with similar keys, restructure
                if all(isinstance(x, dict) for x in item_copy["data"]) and len(item_copy["data"]) > 0:
                    all_keys = set().union(*(d.keys() for d in item_copy["data"]))
                    # If there are common fields, we can flatten the structure
                    if len(all_keys) < 10:  # Reasonable number of columns
                        flattened = {"source_url": item_copy["source_url"], "depth": item_copy["depth"], "score": item_copy["score"]}
                        for i, list_item in enumerate(item_copy["data"]):
                            for key in all_keys:
                                value = list_item.get(key, "")
                                if isinstance(value, (dict, list)):
                                    value = json.dumps(value)
                                flattened[f"{key}_{i+1}"] = value
                        item_copy["data"] = flattened
                        
            # For markdown or text format, improve list representation
            elif base_format.lower() in ['markdown', 'text']:
                # Convert HTML content in list items if present
                if any(isinstance(x, dict) for x in item_copy["data"]):
                    for i, list_item in enumerate(item_copy["data"]):
                        if isinstance(list_item, dict):
                            for key, value in list(list_item.items()):
                                if key.endswith('_html') and isinstance(value, str):
                                    h = html2text.HTML2Text()
                                    h.ignore_links = False
                                    list_item[key.replace('_html', '_text')] = h.handle(value)
        
        processed_data.append(item_copy)
    
    return processed_data

@with_file_operation_retry(max_attempts=3)
async def export_to_file(
    data: List[Dict[str, Any]],
    filepath: str,
    export_format: str = 'json',
    include_raw_html: bool = False,
    content_specific: bool = False,
    export_structure: bool = False
) -> str:
    """
    Export data to a file asynchronously.
    
    Args:
        data: List of data items to export
        filepath: Path to save the exported file
        export_format: Format to export ('json', 'csv', 'excel', 'markdown', 'html', 'text')
        include_raw_html: Whether to include raw HTML in the export
        content_specific: Whether to adapt export format based on content type
        export_structure: Whether to include content structure data
        
    Returns:
        Path to the saved file
    """
    logger.info(f"Exporting data to {filepath} in {export_format} format")
    
    # Process data to handle raw HTML inclusion/exclusion
    processed_data = data
    if not include_raw_html:
        processed_data = await _remove_raw_html(data)
        
    # Add structure data if requested
    if export_structure:
        structure_data = await generate_content_structure_export(processed_data)
        # Add structure info as first item
        processed_data = [{"content_structure": structure_data}] + processed_data
    
    # Handle content-specific formatting
    if content_specific:
        processed_data = await _adapt_to_content_type(processed_data, export_format)
    
    # Generate content based on format
    if export_format.lower() == 'json':
        # Use our async JSON writer
        await write_json(filepath, processed_data)
    
    elif export_format.lower() == 'csv':
        # Generate CSV string
        csv_content = await generate_csv_export(processed_data)
        # Use our async file writer
        await write_file(filepath, csv_content)
    
    elif export_format.lower() == 'excel':
        # Generate Excel bytes
        excel_bytes = await generate_excel_export(processed_data)
        # Write bytes to file
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(excel_bytes)
    
    elif export_format.lower() == 'markdown':
        # Generate Markdown content
        md_content = await generate_markdown_export(processed_data)
        # Write to file
        await write_file(filepath, md_content)
    
    elif export_format.lower() == 'html':
        # Generate HTML content
        html_content = await generate_html_export(processed_data, include_raw_html)
        # Write to file
        await write_file(filepath, html_content)
    
    elif export_format.lower() == 'text':
        # Generate plain text content
        text_content = await generate_plain_text_export(processed_data)
        # Write to file
        await write_file(filepath, text_content)
    
    else:
        raise ValueError(f"Unsupported export format: {export_format}")
    
    logger.info(f"Successfully exported data to {filepath}")
    return filepath

async def export_multiple_formats(
    data: List[Dict[str, Any]],
    base_filepath: str,
    formats: List[str] = ['json', 'csv'],
    include_raw_html: bool = False,
    content_specific: bool = False,
    export_structure: bool = False
) -> Dict[str, str]:
    """
    Export data to multiple file formats in parallel.
    
    Args:
        data: List of data items to export
        base_filepath: Base path for the exported files (without extension)
        formats: List of formats to export to
        include_raw_html: Whether to include raw HTML in the export
        content_specific: Whether to adapt export format based on content type
        export_structure: Whether to include content structure data
        
    Returns:
        Dictionary of format to filepath
    """
    tasks = []
    result_paths = {}
    
    for fmt in formats:
        filepath = f"{base_filepath}.{fmt}"
        task = export_to_file(
            data, 
            filepath, 
            fmt, 
            include_raw_html,
            content_specific,
            export_structure
        )
        tasks.append(task)
    
    # Run exports in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for i, fmt in enumerate(formats):
        if isinstance(results[i], Exception):
            logger.error(f"Error exporting to {fmt}: {results[i]}")
            result_paths[fmt] = None
        else:
            result_paths[fmt] = results[i]
    
    return result_paths