import pytest
from core.html_service import HTMLService

class TestHTMLCleaningCopy:
    def setup_method(self):
        self.html_service = HTMLService()
        self.html_service.initialize()
    
    def teardown_method(self):
        self.html_service.shutdown()
    
    def test_clean_html_copy(self):
        # Test basic cleaning
        html = '''
        <html>
            <head>
                <script>alert("test");</script>
                <style>.test{color:red;}</style>
            </head>
            <body>
                <div style="display:none;">Hidden</div>
                <div><!-- Comment -->Visible</div>
                <div hidden>Also hidden</div>
            </body>
        </html>
        '''
        cleaned = self.html_service.clean_html(html)
    
        # Script, style, hidden div, and comments should be removed
        assert 'script' not in cleaned
        assert 'style' not in cleaned
        assert 'Hidden' not in cleaned
        assert 'Comment' not in cleaned
        assert 'Visible' in cleaned
        assert 'Also hidden' not in cleaned
