import pytest

class TestBasic:
    def test_simple_assertion(self):
        assert True
        
    def test_string_operations(self):
        test_string = "Hello World"
        assert "Hello" in test_string
        assert "World" in test_string
