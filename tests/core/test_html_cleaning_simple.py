import pytest
from core.html_service import HTMLService

class TestHTMLCleaningSimple:
    def setup_method(self):
        self.html_service = HTMLService()
        self.html_service.initialize()
    
    def teardown_method(self):
        if hasattr(self, 'html_service'):
            self.html_service.shutdown()

    def test_script_removal_basic(self):
        """Test basic script tag removal."""
        html_with_script = '''
        <html>
        <head>
            <script>alert("test");</script>
            <script type="text/javascript">console.log("test");</script>
        </head>
        <body>
            <p>This is clean content that should be preserved.</p>
            <script>trackPageView();</script>
        </body>
        </html>
        '''
        
        cleaned = self.html_service.clean_html(html_with_script)
        
        # Should preserve content
        assert 'clean content that should be preserved' in cleaned
        
        # Should remove all scripts
        assert 'alert(' not in cleaned
        assert 'console.log(' not in cleaned
        assert 'trackPageView()' not in cleaned

    def test_css_removal_basic(self):
        """Test CSS removal."""
        html_with_css = '''
        <html>
        <head>
            <style>
                .container { max-width: 1200px; margin: 0 auto; }
                .hidden { display: none; }
            </style>
        </head>
        <body>
            <div class="container">
                <p>Content with styling that should be preserved.</p>
            </div>
        </body>
        </html>
        '''
        
        cleaned = self.html_service.clean_html(html_with_css)
        
        # Should preserve content
        assert 'Content with styling that should be preserved' in cleaned
        
        # Should remove CSS
        assert 'max-width: 1200px' not in cleaned
        assert 'display: none' not in cleaned

    def test_malformed_html_basic(self):
        """Test handling of malformed HTML."""
        malformed_html = '''
        <html>
        <body>
            <div><p>Unclosed paragraph<div>Another div</div>
            <span>Clean content</span>
        </body>
        </html>
        '''
        
        try:
            cleaned = self.html_service.clean_html(malformed_html)
            assert 'Clean content' in cleaned
        except Exception as e:
            pytest.fail(f"HTML cleaning failed on malformed HTML: {e}")
