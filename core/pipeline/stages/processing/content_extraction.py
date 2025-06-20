"""
Content Extraction Pipeline Stages Module.

This module provides pipeline stages for extracting different types of content from documents.
"""

import re
import logging
import html
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple, Union
from bs4 import BeautifulSoup, Tag, NavigableString
import json

from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext


class TextExtractionStage(PipelineStage):
    """
    Pipeline stage for extracting text content from HTML elements.
    
    This stage extracts and normalizes text content from HTML documents, with support for:
    - Configurable text normalization options
    - Handling different text node types
    - Preserving or removing formatting as needed
    - Text chunking for large documents
    
    Configuration options:
        input_key (str): Key in the context to get the HTML content from
        output_key (str): Key in the context to store the extracted text
        selector (str, optional): CSS selector to target specific elements
        preserve_formatting (bool): Whether to preserve some formatting (default: False)
        preserve_links (bool): Whether to preserve links in the text (default: False)
        preserve_images (bool): Whether to include image alt text (default: True)
        normalize_whitespace (bool): Whether to normalize whitespace (default: True)
        normalize_unicode (bool): Whether to normalize Unicode characters (default: True)
        remove_boilerplate (bool): Whether to attempt to remove boilerplate text (default: True)
        chunk_size (int): Maximum size of text chunks (default: 4000)
        chunk_overlap (int): Overlap between consecutive chunks (default: 200)
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a text extraction stage.
        
        Args:
            name: Name of the stage
            config: Configuration parameters
        """
        super().__init__(name, config)
        
        # Extract configuration options with defaults
        self.input_key = self.config.get("input_key", "html_content")
        self.output_key = self.config.get("output_key", "extracted_text")
        self.selector = self.config.get("selector", None)
        
        # Text normalization options
        self.preserve_formatting = self.config.get("preserve_formatting", False)
        self.preserve_links = self.config.get("preserve_links", False)
        self.preserve_images = self.config.get("preserve_images", True)
        self.normalize_whitespace = self.config.get("normalize_whitespace", True)
        self.normalize_unicode = self.config.get("normalize_unicode", True)
        self.remove_boilerplate = self.config.get("remove_boilerplate", True)
        
        # Chunking configuration
        self.chunk_size = self.config.get("chunk_size", 4000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)
        
        # Elements that should be preserved as block-level elements
        self.block_elements = {
            'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'blockquote', 'pre', 'table', 'ul', 'ol', 'li',
            'section', 'article', 'header', 'footer', 'aside',
            'figure', 'figcaption', 'main'
        }
        
        # Elements that we might want to ignore
        self.ignore_elements = {
            'script', 'style', 'noscript', 'iframe', 'svg',
            'canvas', 'template', 'nav', 'menu'
        }
        
    async def validate_input(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains the required HTML content.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if validation passes, False otherwise
        """
        if not context.get(self.input_key):
            self.logger.error(f"Missing required input '{self.input_key}' in context")
            context.add_error(self.name, f"Missing required input '{self.input_key}'")
            return False
        return True
        
    async def process(self, context: PipelineContext) -> bool:
        """
        Extract text content from HTML in the context.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Get the HTML content from the context
            html_content = context.get(self.input_key)
            
            if not html_content or not isinstance(html_content, str):
                self.logger.warning(f"Invalid HTML content in '{self.input_key}'")
                return False
                
            # Parse the HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Apply selector if provided
            root_elements = soup.select(self.selector) if self.selector else [soup]
            
            if not root_elements and self.selector:
                self.logger.warning(f"Selector '{self.selector}' did not match any elements")
                
            # Main content extraction
            if self.remove_boilerplate:
                # Attempt to identify and focus on the main content
                main_content = self._extract_main_content(soup)
                if main_content:
                    root_elements = [main_content]
            
            # Extract and normalize text from the selected elements
            extracted_text = ""
            for element in root_elements:
                extracted_text += self._extract_text_from_element(element)
                
            # Normalize the extracted text according to configuration
            normalized_text = self._normalize_text(extracted_text)
            
            # Chunk the text if necessary
            if self.chunk_size > 0:
                text_chunks = self._chunk_text(normalized_text)
                context.set(self.output_key, text_chunks)
                self.logger.info(f"Extracted text into {len(text_chunks)} chunks")
            else:
                context.set(self.output_key, normalized_text)
                self.logger.info(f"Extracted text of length {len(normalized_text)}")
                
            # Include metadata about the extraction
            context.set(f"{self.output_key}_metadata", {
                "original_length": len(html_content),
                "extracted_length": len(normalized_text),
                "chunk_count": len(text_chunks) if self.chunk_size > 0 else 1,
                "extraction_stage": self.name,
                "selector_used": self.selector
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error extracting text: {str(e)}")
            context.add_error(self.name, f"Text extraction failed: {str(e)}")
            return await self.handle_error(context, e)
            
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[Tag]:
        """
        Attempt to identify and extract the main content area of a page.
        
        Args:
            soup: The BeautifulSoup object representing the HTML
            
        Returns:
            The main content element or None if not identified
        """
        # Try common main content selectors
        main_content_selectors = [
            'main', 'article', '#content', '#main', '.content', '.main', 
            '[role="main"]', '.post', '.entry', '.post-content', '.article-content'
        ]
        
        for selector in main_content_selectors:
            elements = soup.select(selector)
            if elements:
                # Use the largest matching element by content length
                return max(elements, key=lambda e: len(e.get_text()))
                
        # If no obvious main content, try heuristic approach:
        # Find div with most paragraph elements
        paragraphs_by_parent = {}
        for p in soup.find_all('p'):
            parent = p.parent
            if parent not in paragraphs_by_parent:
                paragraphs_by_parent[parent] = []
            paragraphs_by_parent[parent].append(p)
            
        if paragraphs_by_parent:
            # Get the parent with the most paragraphs
            main_parent = max(paragraphs_by_parent.items(), 
                             key=lambda x: len(x[1]))[0]
            return main_parent
            
        # Fallback: return the body
        return soup.body
        
    def _extract_text_from_element(self, element: Union[Tag, NavigableString]) -> str:
        """
        Extract text from a BeautifulSoup element with optional formatting.
        
        Args:
            element: BeautifulSoup Tag or NavigableString
            
        Returns:
            The extracted text string
        """
        # Handle text nodes directly
        if isinstance(element, NavigableString):
            return str(element)
            
        # Skip ignored elements
        if element.name in self.ignore_elements:
            return ""
            
        # Handle special elements
        if element.name == 'br':
            return '\n'
            
        if element.name == 'img' and self.preserve_images and element.get('alt'):
            return f"[Image: {element.get('alt')}]"
            
        if element.name == 'a' and self.preserve_links:
            href = element.get('href', '')
            text = element.get_text().strip()
            if href and text:
                return f"{text} [{href}]"
                
        # Extract text from children recursively
        text_parts = []
        for child in element.children:
            child_text = self._extract_text_from_element(child)
            if child_text:
                text_parts.append(child_text)
        
        # Join text from children
        result = "".join(text_parts)
        
        # Add formatting based on element type if configured
        if self.preserve_formatting:
            # Add line breaks for block elements
            if element.name in self.block_elements:
                result = f"\n{result}\n"
                
            # Add extra spacing for headings
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(element.name[1])
                prefix = '#' * level + ' '
                result = f"\n{prefix}{result}\n"
                
            # Handle list items
            if element.name == 'li':
                result = f"- {result}\n"
                
            # Handle code blocks
            if element.name == 'pre' or element.name == 'code':
                result = f"\n```\n{result}\n```\n"
                
        return result
        
    def _normalize_text(self, text: str) -> str:
        """
        Normalize the extracted text according to configuration.
        
        Args:
            text: The extracted text
            
        Returns:
            The normalized text
        """
        # Decode HTML entities
        text = html.unescape(text)
        
        if self.normalize_whitespace:
            # Replace multiple whitespace with single space
            text = re.sub(r'\s+', ' ', text)
            
            # Normalize newlines (keep max 2 consecutive)
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            # Remove leading/trailing whitespace
            text = text.strip()
            
        if self.normalize_unicode:
            # Normalize Unicode characters to their canonical form
            import unicodedata
            text = unicodedata.normalize('NFKC', text)
            
        return text
        
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split large text into overlapping chunks.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position for current chunk
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
                
            # Try to break at a sentence or paragraph boundary
            # Look backward from end to find a good breaking point
            break_pos = end
            
            # Look for paragraph break
            para_break = text.rfind('\n\n', start, end)
            if para_break != -1 and para_break > start + self.chunk_size // 2:
                break_pos = para_break + 2  # Include the double newline
                
            else:
                # Look for sentence break (period followed by space)
                sentence_break = text.rfind('. ', start, end)
                if sentence_break != -1 and sentence_break > start + self.chunk_size // 2:
                    break_pos = sentence_break + 2  # Include the period and space
                    
            chunks.append(text[start:break_pos])
            
            # Start next chunk with overlap
            start = break_pos - self.chunk_overlap
            
            # Ensure we're making forward progress
            if start <= 0 or start <= chunks[-1]:
                start = break_pos
                
        return chunks


class StructuredDataExtractionStage(PipelineStage):
    """
    Pipeline stage for extracting structured data from HTML elements.
    
    This stage extracts tables, lists, and other structured elements from HTML documents, with support for:
    - Converting HTML structures to JSON/dictionary format
    - Handling nested data structures
    - Schema mapping for standardizing extraction
    - Validation for extracted structures
    
    Configuration options:
        input_key (str): Key in the context to get the HTML content from
        output_key (str): Key in the context to store the extracted structures
        extract_tables (bool): Whether to extract tables (default: True)
        extract_lists (bool): Whether to extract lists (default: True)
        extract_forms (bool): Whether to extract forms (default: False)
        extract_metadata (bool): Whether to extract metadata (default: True)
        schema_mapping (Dict): Optional schema mapping for field standardization
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a structured data extraction stage.
        
        Args:
            name: Name of the stage
            config: Configuration parameters
        """
        super().__init__(name, config)
        
        # Extract configuration options with defaults
        self.input_key = self.config.get("input_key", "html_content")
        self.output_key = self.config.get("output_key", "structured_data")
        
        # What to extract
        self.extract_tables = self.config.get("extract_tables", True)
        self.extract_lists = self.config.get("extract_lists", True)
        self.extract_forms = self.config.get("extract_forms", False)
        self.extract_metadata = self.config.get("extract_metadata", True)
        
        # Schema mapping for standardization
        self.schema_mapping = self.config.get("schema_mapping", {})
        
        # Table extraction settings
        self.table_selector = self.config.get("table_selector", "table")
        self.include_table_captions = self.config.get("include_table_captions", True)
        self.include_table_headers = self.config.get("include_table_headers", True)
        
        # List extraction settings
        self.list_selector = self.config.get("list_selector", "ul, ol, dl")
        self.include_list_items_attributes = self.config.get("include_list_items_attributes", False)
        
        # Form extraction settings
        self.form_selector = self.config.get("form_selector", "form")
        
        # Metadata extraction settings
        self.metadata_selectors = self.config.get("metadata_selectors", {
            "title": "title",
            "meta_description": "meta[name='description']",
            "meta_keywords": "meta[name='keywords']",
            "canonical_url": "link[rel='canonical']"
        })
        
    async def validate_input(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains the required HTML content.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if validation passes, False otherwise
        """
        if not context.get(self.input_key):
            self.logger.error(f"Missing required input '{self.input_key}' in context")
            context.add_error(self.name, f"Missing required input '{self.input_key}'")
            return False
        return True
        
    async def process(self, context: PipelineContext) -> bool:
        """
        Extract structured data from HTML in the context.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Get the HTML content from the context
            html_content = context.get(self.input_key)
            
            if not html_content or not isinstance(html_content, str):
                self.logger.warning(f"Invalid HTML content in '{self.input_key}'")
                return False
                
            # Parse the HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Initialize the structured data container
            structured_data = {}
            
            # Extract different types of structured data
            if self.extract_tables:
                structured_data["tables"] = self._extract_tables(soup)
                
            if self.extract_lists:
                structured_data["lists"] = self._extract_lists(soup)
                
            if self.extract_forms:
                structured_data["forms"] = self._extract_forms(soup)
                
            if self.extract_metadata:
                structured_data["metadata"] = self._extract_metadata(soup)
                
            # Apply schema mapping if provided
            if self.schema_mapping:
                structured_data = self._apply_schema_mapping(structured_data)
                
            # Store the extracted data in the context
            context.set(self.output_key, structured_data)
            
            # Add extraction metadata
            extraction_metadata = {
                "tables_count": len(structured_data.get("tables", [])),
                "lists_count": len(structured_data.get("lists", [])),
                "forms_count": len(structured_data.get("forms", [])),
                "has_metadata": bool(structured_data.get("metadata")),
                "extraction_stage": self.name
            }
            context.set(f"{self.output_key}_metadata", extraction_metadata)
            
            self.logger.info(f"Extracted structured data: {extraction_metadata}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error extracting structured data: {str(e)}")
            context.add_error(self.name, f"Structured data extraction failed: {str(e)}")
            return await self.handle_error(context, e)
            
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract tables from HTML and convert to structured format.
        
        Args:
            soup: The BeautifulSoup object representing the HTML
            
        Returns:
            List of extracted tables as dictionaries
        """
        tables = []
        
        for table_idx, table in enumerate(soup.select(self.table_selector)):
            table_data = {
                "id": f"table_{table_idx}",
                "rows": [],
                "headers": [],
                "caption": "",
                "metadata": {
                    "row_count": 0,
                    "column_count": 0,
                    "has_headers": False
                }
            }
            
            # Extract caption if available
            if self.include_table_captions and table.caption:
                table_data["caption"] = table.caption.get_text().strip()
                
            # Try to find table headers
            headers = []
            if self.include_table_headers:
                # Look for standard th elements in thead
                th_elements = table.select('thead th') or table.select('tr th')
                if th_elements:
                    headers = [th.get_text().strip() for th in th_elements]
                    table_data["headers"] = headers
                    table_data["metadata"]["has_headers"] = True
                    
                # If no headers found, check if first row might be headers
                elif table.select('tr'):
                    first_row = table.select('tr')[0]
                    # If first row has td elements with different style/class, might be headers
                    if first_row.select('td') and not first_row.select('th'):
                        potential_headers = [td.get_text().strip() for td in first_row.select('td')]
                        if any(h for h in potential_headers):
                            table_data["headers"] = potential_headers
                            table_data["metadata"]["has_headers"] = True
            
            # Extract rows
            rows = []
            for row_idx, tr in enumerate(table.select('tr')):
                # Skip row if it's already been identified as a header row
                if row_idx == 0 and table_data["metadata"]["has_headers"] and not tr.select('th'):
                    continue
                    
                # Extract cells from this row
                cells = []
                for cell in tr.select('th, td'):
                    # Get cell text content
                    cell_text = cell.get_text().strip()
                    
                    # Handle colspan and rowspan
                    cell_data = {
                        "text": cell_text,
                        "type": cell.name  # 'th' or 'td'
                    }
                    
                    # Add colspan if present
                    colspan = cell.get('colspan')
                    if colspan and colspan.isdigit() and int(colspan) > 1:
                        cell_data["colspan"] = int(colspan)
                        
                    # Add rowspan if present
                    rowspan = cell.get('rowspan')
                    if rowspan and rowspan.isdigit() and int(rowspan) > 1:
                        cell_data["rowspan"] = int(rowspan)
                    
                    cells.append(cell_data)
                    
                if cells:
                    rows.append(cells)
            
            table_data["rows"] = rows
            table_data["metadata"]["row_count"] = len(rows)
            
            # Calculate column count (max columns in any row)
            if rows:
                table_data["metadata"]["column_count"] = max(len(row) for row in rows)
                
            # Only add non-empty tables
            if rows and table_data["metadata"]["column_count"] > 0:
                tables.append(table_data)
                
        return tables
        
    def _extract_lists(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract lists (ordered, unordered, definition) from HTML.
        
        Args:
            soup: The BeautifulSoup object representing the HTML
            
        Returns:
            List of extracted lists as dictionaries
        """
        lists = []
        
        for list_idx, list_elem in enumerate(soup.select(self.list_selector)):
            list_data = {
                "id": f"list_{list_idx}",
                "type": list_elem.name,  # ul, ol, or dl
                "items": [],
                "metadata": {
                    "item_count": 0,
                    "nested": False
                }
            }
            
            # Extract list items based on list type
            if list_elem.name in ['ul', 'ol']:
                items = []
                for item in list_elem.find_all('li', recursive=False):
                    item_data = {"text": item.get_text().strip()}
                    
                    # Check for nested lists
                    nested_lists = item.select('ul, ol')
                    if nested_lists:
                        list_data["metadata"]["nested"] = True
                        nested_items = []
                        for nested_list in nested_lists:
                            nested_type = nested_list.name
                            nested_items.extend([{
                                "text": li.get_text().strip(),
                                "type": nested_type
                            } for li in nested_list.find_all('li')])
                        item_data["nested_items"] = nested_items
                        
                    # Add attributes if configured
                    if self.include_list_items_attributes and item.attrs:
                        item_data["attributes"] = {k: v for k, v in item.attrs.items()}
                        
                    items.append(item_data)
                    
                list_data["items"] = items
                
            # Handle definition lists differently
            elif list_elem.name == 'dl':
                terms = []
                current_term = None
                
                for child in list_elem.children:
                    if child.name == 'dt':
                        # New term
                        current_term = {
                            "term": child.get_text().strip(),
                            "definitions": []
                        }
                        terms.append(current_term)
                    elif child.name == 'dd' and current_term:
                        # Definition for the current term
                        current_term["definitions"].append(child.get_text().strip())
                        
                list_data["items"] = terms
                
            list_data["metadata"]["item_count"] = len(list_data["items"])
            
            # Only add non-empty lists
            if list_data["items"]:
                lists.append(list_data)
                
        return lists
        
    def _extract_forms(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract forms and form fields from HTML.
        
        Args:
            soup: The BeautifulSoup object representing the HTML
            
        Returns:
            List of extracted forms as dictionaries
        """
        forms = []
        
        for form_idx, form in enumerate(soup.select(self.form_selector)):
            form_data = {
                "id": f"form_{form_idx}",
                "action": form.get('action', ''),
                "method": form.get('method', 'get').upper(),
                "fields": [],
                "metadata": {
                    "field_count": 0,
                    "has_submit": False,
                    "enctype": form.get('enctype', 'application/x-www-form-urlencoded')
                }
            }
            
            # Extract form fields
            fields = []
            
            # Input fields
            for input_field in form.find_all('input'):
                field_type = input_field.get('type', 'text').lower()
                
                if field_type == 'submit':
                    form_data["metadata"]["has_submit"] = True
                    
                field_data = {
                    "name": input_field.get('name', ''),
                    "type": field_type,
                    "value": input_field.get('value', ''),
                    "required": input_field.has_attr('required'),
                    "placeholder": input_field.get('placeholder', '')
                }
                
                # Handle special field types
                if field_type in ['checkbox', 'radio']:
                    field_data["checked"] = input_field.has_attr('checked')
                    
                fields.append(field_data)
                
            # Select fields
            for select in form.find_all('select'):
                options = []
                for option in select.find_all('option'):
                    options.append({
                        "value": option.get('value', ''),
                        "text": option.get_text().strip(),
                        "selected": option.has_attr('selected')
                    })
                    
                field_data = {
                    "name": select.get('name', ''),
                    "type": "select",
                    "required": select.has_attr('required'),
                    "options": options,
                    "multiple": select.has_attr('multiple')
                }
                fields.append(field_data)
                
            # Textarea fields
            for textarea in form.find_all('textarea'):
                field_data = {
                    "name": textarea.get('name', ''),
                    "type": "textarea",
                    "value": textarea.get_text(),
                    "required": textarea.has_attr('required'),
                    "placeholder": textarea.get('placeholder', '')
                }
                fields.append(field_data)
                
            form_data["fields"] = fields
            form_data["metadata"]["field_count"] = len(fields)
            
            # Only add forms with fields
            if fields:
                forms.append(form_data)
                
        return forms
        
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract metadata from HTML document.
        
        Args:
            soup: The BeautifulSoup object representing the HTML
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        
        # Extract title
        if soup.title:
            metadata["title"] = soup.title.get_text().strip()
            
        # Extract standard meta tags
        for name, selector in self.metadata_selectors.items():
            elements = soup.select(selector)
            if elements:
                if selector.startswith('meta'):
                    # For meta tags, get the content attribute
                    metadata[name] = elements[0].get('content', '').strip()
                elif name == 'canonical_url':
                    # For canonical URL, get the href attribute
                    metadata[name] = elements[0].get('href', '').strip()
                else:
                    # Default to text content
                    metadata[name] = elements[0].get_text().strip()
                    
        # Extract all other meta tags
        meta_tags = {}
        for meta in soup.find_all('meta'):
            # Check for different meta tag formats
            name = meta.get('name', meta.get('property', meta.get('http-equiv')))
            content = meta.get('content', '')
            
            if name and content:
                meta_tags[name] = content
                
        metadata["meta_tags"] = meta_tags
        
        # Extract Open Graph metadata
        og_metadata = {}
        for meta in soup.find_all('meta', property=lambda x: x and x.startswith('og:')):
            property_name = meta.get('property', '')[3:]  # Remove 'og:' prefix
            content = meta.get('content', '')
            if property_name and content:
                og_metadata[property_name] = content
                
        if og_metadata:
            metadata["open_graph"] = og_metadata
            
        # Extract structured data (JSON-LD)
        json_ld_data = []
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                json_content = script.string
                if json_content:
                    data = json.loads(json_content)
                    json_ld_data.append(data)
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse JSON-LD data")
                
        if json_ld_data:
            metadata["structured_data"] = json_ld_data
            
        return metadata
        
    def _apply_schema_mapping(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply schema mapping to standardize field names.
        
        Args:
            data: The extracted structured data
            
        Returns:
            Data with standardized field names
        """
        if not self.schema_mapping:
            return data
            
        result = {}
        
        for key, value in data.items():
            # Check if we have a mapping for this top-level key
            if key in self.schema_mapping:
                mapped_key = self.schema_mapping[key]
                result[mapped_key] = value
            else:
                # Keep original key
                result[key] = value
                
            # Handle nested mappings for lists of objects
            if isinstance(value, list) and key + ".items" in self.schema_mapping:
                item_mapping = self.schema_mapping[key + ".items"]
                mapped_items = []
                
                for item in value:
                    if isinstance(item, dict):
                        mapped_item = {}
                        for item_key, item_value in item.items():
                            if item_key in item_mapping:
                                mapped_item[item_mapping[item_key]] = item_value
                            else:
                                mapped_item[item_key] = item_value
                        mapped_items.append(mapped_item)
                    else:
                        mapped_items.append(item)
                        
                result[key if key not in self.schema_mapping else self.schema_mapping[key]] = mapped_items
                
        return result


class PatternExtractionStage(PipelineStage):
    """
    Pipeline stage for extracting data using regex patterns.
    
    This stage extracts data using regular expression patterns, with support for:
    - Named capture groups
    - Common built-in patterns (email, phone, dates, etc.)
    - Pattern combining and prioritization
    - Confidence scores for matches
    
    Configuration options:
        input_key (str): Key in the context to get the content from
        output_key (str): Key in the context to store the extracted patterns
        patterns (Dict[str, str]): Dictionary of named patterns to extract
        built_in_patterns (List[str]): List of built-in patterns to extract
        min_confidence (float): Minimum confidence threshold (0.0-1.0)
        extract_all_matches (bool): Whether to extract all matches or just the first
    """
    
    # Common built-in patterns
    BUILT_IN_PATTERNS = {
        "email": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 0.9),
        "phone_us": (r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 0.8),
        "phone_international": (r'\b\+\d{1,3}[\s.-]?\d{1,14}(?:[\s.-]?\d{1,4})?\b', 0.7),
        "url": (r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)', 0.9),
        "date_iso": (r'\b\d{4}-\d{2}-\d{2}\b', 0.9),
        "date_us": (r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', 0.7),
        "date_eu": (r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b', 0.7),
        "date_text": (r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', 0.8),
        "time": (r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[aApP][mM])?\b', 0.8),
        "price_usd": (r'\$\s*\d+(?:\.\d{2})?\b', 0.9),
        "price_eur": (r'€\s*\d+(?:,\d{2})?\b', 0.9),
        "price_gbp": (r'£\s*\d+(?:\.\d{2})?\b', 0.9),
        "credit_card": (r'\b(?:\d{4}[- ]?){3}\d{4}\b', 0.5),  # Low confidence due to privacy concerns
        "ssn": (r'\b\d{3}-\d{2}-\d{4}\b', 0.5),  # Low confidence due to privacy concerns
        "ip_address": (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 0.8),
        "zip_code_us": (r'\b\d{5}(?:-\d{4})?\b', 0.7),
        "zip_code_ca": (r'\b[A-Za-z]\d[A-Za-z] \d[A-Za-z]\d\b', 0.8),
        "postal_code_uk": (r'\b[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}\b', 0.8),
        "hex_color": (r'#[0-9A-Fa-f]{6}\b', 0.9),
        "hashtag": (r'#[A-Za-z0-9_]+\b', 0.7),
        "mention": (r'@[A-Za-z0-9_]+\b', 0.7)
    }
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a pattern extraction stage.
        
        Args:
            name: Name of the stage
            config: Configuration parameters
        """
        super().__init__(name, config)
        
        # Extract configuration options with defaults
        self.input_key = self.config.get("input_key", "text_content")
        self.output_key = self.config.get("output_key", "extracted_patterns")
        
        # Pattern configuration
        self.patterns = self.config.get("patterns", {})
        self.built_in_patterns = self.config.get("built_in_patterns", [])
        self.min_confidence = float(self.config.get("min_confidence", 0.5))
        self.extract_all_matches = self.config.get("extract_all_matches", True)
        
        # Compile the patterns for efficiency
        self.compiled_patterns = {}
        self._compile_patterns()
        
    async def validate_input(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains the required text content.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if validation passes, False otherwise
        """
        if not context.get(self.input_key):
            self.logger.error(f"Missing required input '{self.input_key}' in context")
            context.add_error(self.name, f"Missing required input '{self.input_key}'")
            return False
        return True
        
    async def process(self, context: PipelineContext) -> bool:
        """
        Extract data using regex patterns from content in the context.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Get the content from the context
            content = context.get(self.input_key)
            
            if not content or not isinstance(content, str):
                self.logger.warning(f"Invalid content in '{self.input_key}'")
                return False
                
            # Initialize results container
            extraction_results = {}
            
            # Extract using custom patterns
            for pattern_name, pattern_info in self.compiled_patterns.items():
                pattern, confidence_base = pattern_info
                matches = self._extract_with_pattern(content, pattern, pattern_name, confidence_base)
                if matches:
                    extraction_results[pattern_name] = matches
                    
            # Store results in context
            context.set(self.output_key, extraction_results)
            
            # Add extraction metadata
            pattern_counts = {name: len(matches) for name, matches in extraction_results.items()}
            extraction_metadata = {
                "total_patterns": len(self.compiled_patterns),
                "patterns_with_matches": len(extraction_results),
                "match_counts": pattern_counts,
                "extraction_stage": self.name
            }
            context.set(f"{self.output_key}_metadata", extraction_metadata)
            
            self.logger.info(f"Extracted patterns: {len(extraction_results)} pattern types with matches")
            return True
            
        except Exception as e:
            self.logger.error(f"Error extracting patterns: {str(e)}")
            context.add_error(self.name, f"Pattern extraction failed: {str(e)}")
            return await self.handle_error(context, e)
            
    def _compile_patterns(self) -> None:
        """
        Compile regex patterns for efficient matching.
        """
        # Compile custom patterns
        for pattern_name, pattern_str in self.patterns.items():
            # Check if confidence is specified
            if isinstance(pattern_str, tuple) and len(pattern_str) == 2:
                pattern_regex, confidence = pattern_str
            else:
                pattern_regex = pattern_str
                confidence = 0.7  # Default confidence
                
            try:
                compiled = re.compile(pattern_regex, re.DOTALL)
                self.compiled_patterns[pattern_name] = (compiled, confidence)
            except re.error:
                self.logger.error(f"Invalid regex pattern '{pattern_name}': {pattern_regex}")
                
        # Add requested built-in patterns
        for pattern_name in self.built_in_patterns:
            if pattern_name in self.BUILT_IN_PATTERNS:
                pattern_str, confidence = self.BUILT_IN_PATTERNS[pattern_name]
                try:
                    compiled = re.compile(pattern_str, re.DOTALL)
                    self.compiled_patterns[pattern_name] = (compiled, confidence)
                except re.error:
                    self.logger.error(f"Invalid built-in pattern '{pattern_name}': {pattern_str}")
            else:
                self.logger.warning(f"Unknown built-in pattern: {pattern_name}")
                
    def _extract_with_pattern(self, 
                             content: str, 
                             pattern: Pattern, 
                             pattern_name: str,
                             confidence_base: float) -> List[Dict[str, Any]]:
        """
        Extract data using a compiled regex pattern.
        
        Args:
            content: Text content to extract from
            pattern: Compiled regex pattern
            pattern_name: Name of the pattern for reference
            confidence_base: Base confidence score for this pattern
            
        Returns:
            List of match dictionaries with text and metadata
        """
        results = []
        
        # Determine if pattern has named groups
        has_named_groups = bool(pattern.groupindex)
        
        # Find all matches or just the first one
        matches = pattern.finditer(content)
        
        for match_idx, match in enumerate(matches):
            match_text = match.group(0)
            start_pos = match.start()
            end_pos = match.end()
            
            # Calculate match-specific confidence score
            # Base it on the base confidence and the match length
            confidence = confidence_base
            
            # Longer matches might be more reliable for some patterns
            if len(match_text) > 20:
                confidence = min(confidence + 0.1, 1.0)
            elif len(match_text) < 4:
                confidence = max(confidence - 0.1, 0.0)
                
            # If the match is surrounded by context that suggests it's valid,
            # slightly increase confidence
            context_before = content[max(0, start_pos-20):start_pos]
            context_after = content[end_pos:min(len(content), end_pos+20)]
            
            # Context-based confidence adjustments
            if pattern_name.startswith("email"):
                # Emails preceded by "email:", "contact:", etc.
                if re.search(r'email|contact|mail|e-mail|reach|address', context_before, re.IGNORECASE):
                    confidence = min(confidence + 0.1, 1.0)
            elif pattern_name.startswith("phone"):
                # Phone numbers preceded by "phone:", "call:", etc.
                if re.search(r'phone|call|tel|telephone|mobile|contact', context_before, re.IGNORECASE):
                    confidence = min(confidence + 0.1, 1.0)
            elif pattern_name.startswith("date"):
                # Dates preceded by date-related words
                if re.search(r'date|published|posted|updated|created|on', context_before, re.IGNORECASE):
                    confidence = min(confidence + 0.1, 1.0)
            elif pattern_name.startswith("price"):
                # Prices preceded by price-related words
                if re.search(r'price|cost|fee|total|amount', context_before, re.IGNORECASE):
                    confidence = min(confidence + 0.1, 1.0)
                    
            # Skip if below confidence threshold
            if confidence < self.min_confidence:
                continue
                
            # Extract named groups if available
            captured_groups = {}
            if has_named_groups:
                for group_name, group_idx in pattern.groupindex.items():
                    if match.group(group_idx):
                        captured_groups[group_name] = match.group(group_idx)
                        
            # Create match result
            match_result = {
                "text": match_text,
                "position": {"start": start_pos, "end": end_pos},
                "confidence": round(confidence, 2)
            }
            
            # Add captured groups if any
            if captured_groups:
                match_result["groups"] = captured_groups
                
            results.append(match_result)
            
            # Stop after first match if not extracting all
            if not self.extract_all_matches:
                break
                
        return results
    
    def add_pattern(self, name: str, pattern: str, confidence: float = 0.7) -> None:
        """
        Add a new pattern at runtime.
        
        Args:
            name: Name of the pattern
            pattern: Regex pattern string
            confidence: Base confidence score (0.0-1.0)
        """
        try:
            compiled = re.compile(pattern, re.DOTALL)
            self.compiled_patterns[name] = (compiled, confidence)
            self.patterns[name] = (pattern, confidence)
        except re.error as e:
            self.logger.error(f"Failed to add pattern '{name}': {str(e)}")
            
    def remove_pattern(self, name: str) -> bool:
        """
        Remove a pattern.
        
        Args:
            name: Name of the pattern to remove
            
        Returns:
            True if pattern was removed, False if not found
        """
        if name in self.compiled_patterns:
            del self.compiled_patterns[name]
            if name in self.patterns:
                del self.patterns[name]
            return True
        return False
        
    def get_pattern_info(self) -> Dict[str, Any]:
        """
        Get information about the current patterns.
        
        Returns:
            Dictionary with pattern information
        """
        pattern_info = {}
        for name, (pattern, confidence) in self.compiled_patterns.items():
            pattern_info[name] = {
                "pattern": pattern.pattern,
                "confidence": confidence,
                "has_groups": bool(pattern.groupindex),
                "groups": list(pattern.groupindex.keys()) if pattern.groupindex else []
            }
        return pattern_info