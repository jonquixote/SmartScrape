# SmartScrape Compatibility Documentation

## Python 3.13 Compatibility

As of May 2025, SmartScrape has been updated to ensure compatibility with Python 3.13. This document outlines the key changes made and considerations to be aware of when developing or deploying the application.

### spaCy Compatibility Issues

**Problem**: spaCy has compatibility issues with Python 3.13. It requires special installation procedures or may not work at all depending on the specific version.

**Solution**: We've implemented an alternative NLP pipeline using NLTK (Natural Language Toolkit) which is fully compatible with Python 3.13. This approach provides similar functionality while ensuring compatibility:

1. Replaced spaCy with NLTK for NLP processing tasks
2. Added a pluggable NLP architecture that can support multiple backends
3. Implemented NLTK-based text analysis for intent parsing
4. Added automatic download of required NLTK data resources

### Dependency Management

All dependencies have been pinned to specific versions in `requirements.txt` to ensure consistent behavior across environments. Key dependencies include:

- nltk==3.8.1 (replacing spaCy)
- pandas==2.2.0 (confirmed compatible with Python 3.13)
- httpx==0.27.2 (for HTTP requests, replacing requests)
- fastapi==0.104.1 (for API endpoints)

### Docker Containerization

For complete isolation and consistent environment setup, we provide a Docker-based deployment option that ensures all dependencies work correctly regardless of the host system.

The multi-stage build process in our Dockerfile:
1. Uses a Python 3.13 base image
2. Installs all dependencies in a controlled environment
3. Sets up required NLTK data
4. Creates a production-ready image with minimal footprint

### Testing with Python 3.13

To test SmartScrape with Python 3.13:

1. Ensure Python 3.13 is installed on your system
2. Create a virtual environment: `python3.13 -m venv test_env`
3. Activate the environment: `source test_env/bin/activate` (Linux/macOS) or `test_env\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run tests: `python -m pytest tests/`

### Known Limitations

- Some advanced language processing features available in spaCy (like dependency parsing or named entity recognition) have been simplified or replaced with NLTK alternatives
- The NLTK-based implementation may have slightly different behavior for certain text analysis tasks

## Other Compatibility Considerations

### Browser Automation

Playwright is used for browser automation and has specific version requirements for its browser drivers. Use `playwright install` to ensure compatible browser versions are installed.

### API Compatibility

All API endpoints maintain backward compatibility with previous SmartScrape versions to ensure seamless upgrades. Any breaking changes will be clearly documented in release notes.