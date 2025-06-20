# Pagination Support in SmartScrape

## Overview

SmartScrape now supports automatic pagination for search results, allowing the framework to navigate through multiple pages of results autonomously. This feature is integrated into the FormSearchEngine strategy through the PaginationHandler component.

## Configuration

Pagination can be configured through the following parameters:

- `use_pagination`: Boolean flag to enable/disable pagination (default: `True`)
- `max_pages`: Maximum number of pages to scrape (default: value from configuration or 10)

These parameters can be passed to the `search` method of FormSearchEngine or included in the `options` dictionary passed to the `execute` method.

## Example Usage

```python
# Basic usage with default settings
result = await search_engine.search(query, url, params={
    'use_pagination': True,
    'max_pages': 5
})

# Or through the execute method
result = await search_engine.execute(url, options={
    'query': 'Cleveland properties',
    'use_pagination': True,
    'max_pages': 3
})
```

## How It Works

1. The FormSearchEngine first extracts results from the initial search results page.
2. If pagination is enabled, the PaginationHandler detects the pagination type and next page URL.
3. The engine navigates to subsequent pages, extracting results from each page.
4. Results are combined, with duplicate entries removed.
5. The process continues until one of the following conditions is met:
   - The maximum number of pages (`max_pages`) is reached
   - No next page is available
   - An error occurs during navigation

## Pagination Detection

The PaginationHandler component can detect various types of pagination:

- Next/Previous button navigation
- Numbered page navigation
- Load more/infinite scroll
- URL parameter-based pagination

## Response Format

When pagination is used, the search result includes a `has_pagination` flag indicating whether pagination was enabled:

```json
{
  "success": true,
  "results_url": "https://example.com/search?q=query",
  "result_count": 45,
  "results": [...],
  "has_pagination": true
}
```

## Configuration in FormSearchEngine

The FormSearchEngine initializes pagination support in its constructor:

```python
try:
    self.pagination_handler = PaginationHandler(max_depth=max_pages)
    self.pagination_support_enabled = True
except NameError:
    logger.warning("PaginationHandler not available, pagination support disabled")
    self.pagination_support_enabled = False
```

This allows the engine to gracefully degrade if the PaginationHandler component is not available.
