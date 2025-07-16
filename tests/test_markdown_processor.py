

import pytest
from src.chunkers.markdown_processor import MarkdownProcessor

@pytest.fixture
def processor():
    return MarkdownProcessor()

def test_extract_structure_headers(processor):
    content = "# Header 1\n## Header 2\n### Header 3"
    structure = processor.extract_structure(content)
    assert len(structure['headers']) == 3
    assert structure['headers'][0]['level'] == 1
    assert structure['headers'][1]['title'] == 'Header 2'

def test_extract_structure_code_blocks(processor):
    content = "```python\nprint('Hello')\n```"
    structure = processor.extract_structure(content)
    assert len(structure['code_blocks']) == 2
    assert structure['code_blocks'][0]['language'] == 'python'

def test_extract_structure_tables(processor):
    content = "| Head 1 | Head 2 |\n|---|---|\n| Cell 1 | Cell 2 |"
    structure = processor.extract_structure(content)
    assert len(structure['tables']) == 3

def test_extract_structure_links_and_images(processor):
    content = "[link](http://example.com) ![alt](http://image.com/img.png)"
    structure = processor.extract_structure(content)
    assert len(structure['links']) == 1
    assert len(structure['images']) == 1
    assert structure['links'][0] == ('link', 'http://example.com')
    assert structure['images'][0] == ('alt', 'http://image.com/img.png')

def test_clean_content_whitespace(processor):
    content = "Line 1\n\n\n\nLine 2"
    cleaned = processor.clean_content(content)
    assert cleaned == "Line 1\n\nLine 2"

def test_clean_content_headers(processor):
    content = "#Header 1\n##  Header 2"
    cleaned = processor.clean_content(content)
    assert cleaned == "# Header 1\n## Header 2"

