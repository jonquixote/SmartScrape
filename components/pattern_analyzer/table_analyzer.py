"""
Table Analyzer Module

This module provides functionality to detect and analyze table patterns
on websites, including data tables, pricing tables, comparison tables, and other
structured tabular content.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from bs4 import BeautifulSoup, Tag

from components.pattern_analyzer.base_analyzer import PatternAnalyzer, get_registry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TableAnalyzer")


class TableAnalyzer(PatternAnalyzer):
    """
    Analyzer for detecting and analyzing table patterns on web pages.
    This includes data tables, pricing tables, comparison tables,
    and any structured tabular data.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the table analyzer.
        
        Args:
            confidence_threshold: Minimum confidence level for pattern detection
        """
        super().__init__(confidence_threshold)
        
        # Patterns that indicate a table might be a data table
        self.data_table_patterns = {
            'id': [re.compile(r'data', re.I), re.compile(r'table', re.I), re.compile(r'grid', re.I)],
            'class': [re.compile(r'data', re.I), re.compile(r'table', re.I), re.compile(r'grid', re.I),
                     re.compile(r'listing', re.I), re.compile(r'results', re.I)]
        }
        
        # Common header row indicators
        self.header_patterns = [
            re.compile(r'header', re.I), re.compile(r'heading', re.I), re.compile(r'head', re.I),
            re.compile(r'th\b', re.I), re.compile(r'thead', re.I)
        ]
    
    async def analyze(self, html: str, url: str) -> Dict[str, Any]:
        """
        Analyze a page to detect table patterns.
        
        Args:
            html: HTML content of the page
            url: URL of the page
            
        Returns:
            Dictionary with detected table patterns
        """
        logger.info(f"Analyzing table patterns on {url}")
        soup = self.parse_html(html)
        domain = self.get_domain(url)
        
        # Results will contain all detected table patterns
        results = {
            "tables": [],
            "confidence_scores": {},
            "confidence_score": 0.0,
            "has_tables": False,
            "table_type": "none",
            "patterns": []
        }
        
        # Find all explicit table elements
        explicit_tables = soup.find_all('table')
        
        # If no explicit tables, look for implicit tables (div-based grids)
        if not explicit_tables:
            grid_tables = self._find_grid_tables(soup)
            
            for table_idx, grid_div in enumerate(grid_tables):
                table_data = self._analyze_grid_table(grid_div, table_idx)
                
                if table_data["confidence_score"] >= self.confidence_threshold:
                    results["tables"].append(table_data)
                    results["confidence_scores"][table_data["table_id"]] = table_data["confidence_score"]
                    logger.info(f"Detected grid table: {table_data['table_id']} with confidence {table_data['confidence_score']:.2f}")
                    
                    # Update overall results
                    if table_data["confidence_score"] > results["confidence_score"]:
                        results["confidence_score"] = table_data["confidence_score"]
                        results["has_tables"] = True
                        results["table_type"] = table_data["table_type"]
        
        # Analyze explicit HTML tables
        for table_idx, table in enumerate(explicit_tables):
            table_data = self._analyze_table(table, table_idx)
            
            if table_data["confidence_score"] >= self.confidence_threshold:
                results["tables"].append(table_data)
                results["confidence_scores"][table_data["table_id"]] = table_data["confidence_score"]
                logger.info(f"Detected table: {table_data['table_id']} with confidence {table_data['confidence_score']:.2f}")
                
                # Update overall results
                if table_data["confidence_score"] > results["confidence_score"]:
                    results["confidence_score"] = table_data["confidence_score"]
                    results["has_tables"] = True
                    results["table_type"] = table_data["table_type"]
        
        # Register the best table pattern in the global registry
        if results["tables"]:
            best_table = max(results["tables"], key=lambda x: x["confidence_score"])
            get_registry().register_pattern(
                pattern_type="table",
                url=url,
                pattern_data=best_table,
                confidence=best_table["confidence_score"]
            )
            logger.info(f"Registered table pattern for {domain}")
        
        return results
    
    def _find_grid_tables(self, soup: BeautifulSoup) -> List[Tag]:
        """
        Find grid-like structures that function as tables but don't use the table tag.
        
        Args:
            soup: BeautifulSoup object representing the HTML
            
        Returns:
            List of elements that appear to be grid-based tables
        """
        grid_tables = []
        
        # Look for common grid class patterns
        grid_containers = soup.select('.grid, .table, [class*="grid"], [class*="table"], [role="grid"], [role="table"]')
        
        for container in grid_containers:
            # Skip if it's actually a table tag or contains a table tag
            if container.name == 'table' or container.find('table'):
                continue
                
            # Check if it has repeating row structures
            rows = container.select(':scope > div, :scope > li, :scope > ul')
            
            # If we have at least 2 rows (header + data), it might be a table
            if len(rows) >= 2:
                # Check for similar structure in rows
                if self._check_row_similarity(rows):
                    grid_tables.append(container)
        
        return grid_tables
    
    def _check_row_similarity(self, rows: List[Tag]) -> bool:
        """
        Check if a list of elements has similar structure to function as table rows.
        
        Args:
            rows: List of elements to check
            
        Returns:
            True if the elements have similar structure, False otherwise
        """
        if len(rows) < 2:
            return False
            
        # Get the number of children in each row
        child_counts = []
        for row in rows:
            direct_children = [child for child in row.children if isinstance(child, Tag)]
            child_counts.append(len(direct_children))
            
        # If all rows (except perhaps the header) have the same number of children,
        # it's likely a table structure
        if len(set(child_counts[1:])) <= 1 and child_counts[1:][0] > 0:
            return True
            
        return False
    
    def _analyze_grid_table(self, grid: Tag, table_idx: int) -> Dict[str, Any]:
        """
        Analyze a grid-based table.
        
        Args:
            grid: The grid container element
            table_idx: Index of the grid in the page
            
        Returns:
            Dictionary with grid table data and confidence
        """
        evidence_points = []
        
        # Get grid attributes
        grid_attrs = {
            'id': grid.get('id', ''),
            'class': grid.get('class', []),
            'role': grid.get('role', '')
        }
        
        # Convert classes to string for easier searching
        grid_class_str = ' '.join(grid_attrs['class']) if isinstance(grid_attrs['class'], list) else grid_attrs['class']
        grid_id = grid_attrs['id']
        
        # Check for table/grid indicators in attributes
        if re.search(r'table|grid|data', grid_class_str + ' ' + grid_id, re.I):
            evidence_points.append(0.8)
        
        # Check ARIA role
        if grid_attrs['role'] in ['grid', 'table']:
            evidence_points.append(0.9)
        
        # Find rows
        rows = grid.select(':scope > div, :scope > li, :scope > ul')
        row_count = len(rows)
        
        # More rows suggest a stronger table pattern
        row_evidence = min(0.8, row_count / 10)  # Cap at 0.8
        evidence_points.append(row_evidence)
        
        # Check for header-like first row
        if rows:
            first_row = rows[0]
            if self._is_likely_header_row(first_row):
                evidence_points.append(0.7)
                
            # Check column count - consistent columns are a strong indicator
            if self._has_consistent_columns(rows):
                evidence_points.append(0.9)
        
        # Sample row data to determine structure
        sample_data = self._extract_grid_data(grid, rows, max_rows=5)
        
        # Determine table type
        table_type = self._determine_table_type(grid, sample_data)
        
        # Calculate confidence score
        confidence_score = self.calculate_confidence(evidence_points)
        
        # Generate a selector for the table
        selector = self._generate_table_selector(grid)
        
        return {
            "table_id": grid_attrs['id'] or f"grid_table_{table_idx}",
            "element_type": "div",
            "table_type": table_type,
            "row_count": row_count,
            "column_count": len(sample_data.get("headers", [])),
            "has_headers": bool(sample_data.get("headers")),
            "headers": sample_data.get("headers", []),
            "sample_data": sample_data.get("rows", []),
            "confidence_score": confidence_score,
            "selector": selector,
            "is_grid_table": True
        }
    
    def _analyze_table(self, table: Tag, table_idx: int) -> Dict[str, Any]:
        """
        Analyze an HTML table element.
        
        Args:
            table: The table element
            table_idx: Index of the table in the page
            
        Returns:
            Dictionary with table data and confidence
        """
        evidence_points = []
        
        # Get table attributes
        table_attrs = {
            'id': table.get('id', ''),
            'class': table.get('class', []),
            'role': table.get('role', '')
        }
        
        # Convert classes to string for easier searching
        table_class_str = ' '.join(table_attrs['class']) if isinstance(table_attrs['class'], list) else table_attrs['class']
        table_id = table_attrs['id']
        
        # Check for data table indicators in attributes
        if re.search(r'data|results|stats|comparison|pricing', table_class_str + ' ' + table_id, re.I):
            evidence_points.append(0.7)
        
        # Check for thead element
        has_thead = bool(table.find('thead'))
        if has_thead:
            evidence_points.append(0.8)
        
        # Check for th elements
        th_count = len(table.find_all('th'))
        if th_count > 0:
            th_evidence = min(0.8, th_count / 10)  # Cap at 0.8
            evidence_points.append(th_evidence)
        
        # Count rows and columns
        rows = table.find_all('tr')
        row_count = len(rows)
        
        # More rows suggest a stronger data table
        row_evidence = min(0.7, row_count / 10)  # Cap at 0.7
        evidence_points.append(row_evidence)
        
        # Extract table data
        table_data = self._extract_table_data(table, max_rows=5)
        
        # Determine table type
        table_type = self._determine_table_type(table, table_data)
        
        # Calculate confidence score
        confidence_score = self.calculate_confidence(evidence_points)
        
        # Generate a selector for the table
        selector = self._generate_table_selector(table)
        
        return {
            "table_id": table_attrs['id'] or f"table_{table_idx}",
            "element_type": "table",
            "table_type": table_type,
            "row_count": row_count,
            "column_count": len(table_data.get("headers", [])),
            "has_headers": bool(table_data.get("headers")),
            "headers": table_data.get("headers", []),
            "sample_data": table_data.get("rows", []),
            "confidence_score": confidence_score,
            "selector": selector,
            "is_grid_table": False
        }
    
    def _is_likely_header_row(self, row: Tag) -> bool:
        """
        Check if a row is likely to be a header row.
        
        Args:
            row: The row element
            
        Returns:
            True if the row is likely a header, False otherwise
        """
        # Check if it has th elements
        if row.find('th'):
            return True
            
        # Check class and id attributes
        row_attrs = {
            'id': row.get('id', ''),
            'class': row.get('class', [])
        }
        
        row_class_str = ' '.join(row_attrs['class']) if isinstance(row_attrs['class'], list) else row_attrs['class']
        row_id = row_attrs['id']
        
        for pattern in self.header_patterns:
            if pattern.search(row_class_str + ' ' + row_id):
                return True
        
        # Check if it has bold or strong elements
        if row.find(['b', 'strong']):
            # Only consider it a header if all or most cells have bold/strong
            cells = row.find_all(['td', 'th', 'div'])
            if cells:
                bold_count = sum(1 for cell in cells if cell.find(['b', 'strong']))
                if bold_count / len(cells) > 0.7:
                    return True
        
        return False
    
    def _has_consistent_columns(self, rows: List[Tag]) -> bool:
        """
        Check if rows have consistent column count.
        
        Args:
            rows: List of row elements
            
        Returns:
            True if columns are consistent, False otherwise
        """
        if not rows:
            return False
            
        column_counts = []
        
        for row in rows:
            # For tr elements
            if row.name == 'tr':
                columns = row.find_all(['td', 'th'])
            else:
                # For div or other elements
                columns = [c for c in row.children if isinstance(c, Tag)]
                
            column_counts.append(len(columns))
        
        # Check if all data rows have the same column count
        # (Ignore header row which might be different)
        data_rows_consistent = len(set(column_counts[1:])) <= 1 if len(column_counts) > 1 else True
        
        return data_rows_consistent
    
    def _extract_table_data(self, table: Tag, max_rows: int = 5) -> Dict[str, Any]:
        """
        Extract data from an HTML table.
        
        Args:
            table: The table element
            max_rows: Maximum number of data rows to extract
            
        Returns:
            Dictionary with headers and sample rows
        """
        result = {
            "headers": [],
            "rows": []
        }
        
        rows = table.find_all('tr')
        if not rows:
            return result
            
        # Check for thead/tbody structure
        thead = table.find('thead')
        tbody = table.find('tbody')
        
        # If we have a thead, use its rows for headers
        if thead:
            header_rows = thead.find_all('tr')
            if header_rows:
                # Use the last row in thead for column headers
                header_cells = header_rows[-1].find_all(['th', 'td'])
                result["headers"] = [cell.get_text().strip() for cell in header_cells]
                
                # If headers are empty strings, try to use column indices
                if all(not h for h in result["headers"]):
                    result["headers"] = [f"Column {i+1}" for i in range(len(header_cells))]
        
        # If no thead or no headers found, use the first row
        if not result["headers"] and rows:
            first_row = rows[0]
            header_cells = first_row.find_all(['th', 'td'])
            
            # If the first row has th elements or seems like a header, use it
            if first_row.find('th') or self._is_likely_header_row(first_row):
                result["headers"] = [cell.get_text().strip() for cell in header_cells]
                # Skip this row when collecting data rows
                rows = rows[1:]
                
                # If headers are empty strings, try to use column indices
                if all(not h for h in result["headers"]):
                    result["headers"] = [f"Column {i+1}" for i in range(len(header_cells))]
        
        # Get data rows
        data_rows = rows[:max_rows] if rows else []
        
        for row in data_rows:
            cells = row.find_all(['td', 'th'])
            if cells:
                row_data = [cell.get_text().strip() for cell in cells]
                result["rows"].append(row_data)
        
        return result
    
    def _extract_grid_data(self, grid: Tag, rows: List[Tag], max_rows: int = 5) -> Dict[str, Any]:
        """
        Extract data from a grid-based table.
        
        Args:
            grid: The grid container element
            rows: List of row elements
            max_rows: Maximum number of data rows to extract
            
        Returns:
            Dictionary with headers and sample rows
        """
        result = {
            "headers": [],
            "rows": []
        }
        
        if not rows:
            return result
            
        # Check if the first row is a header
        first_row = rows[0]
        
        if self._is_likely_header_row(first_row):
            # Extract header cells
            if first_row.name == 'ul' or first_row.name == 'ol':
                header_cells = first_row.find_all('li')
            else:
                header_cells = [c for c in first_row.children if isinstance(c, Tag)]
                
            result["headers"] = [cell.get_text().strip() for cell in header_cells]
            
            # Skip the header row when collecting data
            data_rows = rows[1:max_rows+1]
        else:
            # No header, so just use column indices
            # Get the first row to determine column count
            if first_row.name == 'ul' or first_row.name == 'ol':
                first_row_cells = first_row.find_all('li')
            else:
                first_row_cells = [c for c in first_row.children if isinstance(c, Tag)]
                
            result["headers"] = [f"Column {i+1}" for i in range(len(first_row_cells))]
            
            # Use all rows as data
            data_rows = rows[:max_rows]
        
        # Extract data from rows
        for row in data_rows:
            if row.name == 'ul' or row.name == 'ol':
                cells = row.find_all('li')
            else:
                cells = [c for c in row.children if isinstance(c, Tag)]
                
            if cells:
                row_data = [cell.get_text().strip() for cell in cells]
                result["rows"].append(row_data)
        
        return result
    
    def _determine_table_type(self, table: Tag, table_data: Dict[str, Any]) -> str:
        """
        Determine the type of table based on its content and structure.
        
        Args:
            table: The table element
            table_data: Extracted table data
            
        Returns:
            String describing the table type
        """
        # Get attributes for analysis
        attrs = {
            'id': table.get('id', ''),
            'class': table.get('class', [])
        }
        
        attr_str = ' '.join(attrs['class'] if isinstance(attrs['class'], list) else [attrs['class']])
        attr_str += ' ' + attrs['id']
        
        # Check for pricing tables
        if re.search(r'price|pricing|plan', attr_str, re.I):
            return "pricing_table"
            
        # Check for comparison tables
        elif re.search(r'compare|comparison', attr_str, re.I):
            return "comparison_table"
            
        # Check for stats tables
        elif re.search(r'stat|stats|statistics', attr_str, re.I):
            return "statistics_table"
            
        # Check for property/feature tables
        elif re.search(r'property|properties|feature|spec|specification', attr_str, re.I):
            return "property_table"
            
        # Generic data table detection - look at content
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        
        if headers and rows:
            # Check headers for price indicators
            price_headers = sum(1 for h in headers if re.search(r'price|cost|\$|€|£|¥', h, re.I))
            if price_headers > 0:
                return "pricing_table"
                
            # Check for date columns (common in data tables)
            date_headers = sum(1 for h in headers if re.search(r'date|time|period|year|month', h, re.I))
            if date_headers > 0:
                return "data_table"
        
        # Default to generic data table
        return "data_table"
    
    def _generate_table_selector(self, table: Tag) -> str:
        """
        Generate a CSS selector for the table.
        
        Args:
            table: The table element
            
        Returns:
            CSS selector string
        """
        # If it has an ID, use that (most specific)
        if table.get('id'):
            return f"#{table['id']}"
            
        # If it has classes, use the most specific combination
        if table.get('class'):
            # Sort classes by specificity (longer classes are usually more specific)
            classes = sorted(table['class'], key=len, reverse=True)
            
            # Use up to 2 classes for better specificity
            class_selector = '.'.join(classes[:2]) 
            return f".{class_selector}"
            
        # If it's a regular table element, use that with some context
        if table.name == 'table':
            # Try to get context from parent
            parent = table.parent
            if parent and parent.name != 'body' and parent.get('id'):
                return f"#{parent['id']} > table"
            elif parent and parent.name != 'body' and parent.get('class'):
                parent_class = parent['class'][0] if isinstance(parent['class'], list) else parent['class']
                return f".{parent_class} > table"
            
            # Count previous siblings to create an nth-of-type selector
            siblings = list(table.find_previous_siblings('table')) 
            return f"table:nth-of-type({len(siblings) + 1})"
            
        # For non-table elements, generate a more complex selector
        parent = table.parent
        if parent and parent.name != 'body' and parent.get('id'):
            return f"#{parent['id']} > {table.name}"
        elif parent and parent.name != 'body' and parent.get('class'):
            parent_class = parent['class'][0] if isinstance(parent['class'], list) else parent['class']
            return f".{parent_class} > {table.name}"
            
        # Last resort: use tag name with nth-of-type
        siblings = list(table.find_previous_siblings(table.name))
        return f"{table.name}:nth-of-type({len(siblings) + 1})"