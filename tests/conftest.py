"""
Pytest configuration and fixtures for the chunking system tests.
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock

from langchain_core.documents import Document

# Test data constants
SAMPLE_MARKDOWN = """# Chapter 1: Introduction

This is the introduction to our document. It contains multiple paragraphs and different types of content.

## Section 1.1: Overview

Here's an overview section with some important information.

### Subsection 1.1.1: Details

More detailed information goes here.

## Section 1.2: Code Examples

Here's some code:

```python
def hello_world():
    print("Hello, World!")
    return True
```

## Section 1.3: Lists and Tables

Here's a list:
- Item 1
- Item 2
- Item 3

And a table:

| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |

## Conclusion

This concludes our sample document.
"""

SMALL_MARKDOWN = """# Simple Doc

Just a simple document with minimal content.

Some text here.
"""

LARGE_MARKDOWN = SAMPLE_MARKDOWN * 10  # Repeat content 10 times


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_markdown_file(temp_dir: Path) -> Path:
    """Create a sample markdown file for testing."""
    file_path = temp_dir / "sample.md"
    file_path.write_text(SAMPLE_MARKDOWN, encoding='utf-8')
    return file_path


@pytest.fixture
def small_markdown_file(temp_dir: Path) -> Path:
    """Create a small markdown file for testing."""
    file_path = temp_dir / "small.md"
    file_path.write_text(SMALL_MARKDOWN, encoding='utf-8')
    return file_path


@pytest.fixture
def large_markdown_file(temp_dir: Path) -> Path:
    """Create a large markdown file for testing."""
    file_path = temp_dir / "large.md"
    file_path.write_text(LARGE_MARKDOWN, encoding='utf-8')
    return file_path


@pytest.fixture
def empty_file(temp_dir: Path) -> Path:
    """Create an empty file for testing."""
    file_path = temp_dir / "empty.md"
    file_path.write_text("", encoding='utf-8')
    return file_path


@pytest.fixture
def nonexistent_file(temp_dir: Path) -> Path:
    """Return path to a non-existent file."""
    return temp_dir / "nonexistent.md"


@pytest.fixture
def sample_chunks() -> list[Document]:
    """Create sample chunks for testing evaluators."""
    return [
        Document(
            page_content="# Chapter 1: Introduction\n\nThis is the introduction to our document.",
            metadata={
                "Header 1": "Chapter 1: Introduction",
                "chunk_index": 0,
                "chunk_tokens": 15,
                "chunk_chars": 65,
                "word_count": 12
            }
        ),
        Document(
            page_content="## Section 1.1: Overview\n\nHere's an overview section with important information.",
            metadata={
                "Header 1": "Chapter 1: Introduction",
                "Header 2": "Section 1.1: Overview", 
                "chunk_index": 1,
                "chunk_tokens": 14,
                "chunk_chars": 73,
                "word_count": 11
            }
        ),
        Document(
            page_content="```python\ndef hello_world():\n    print(\"Hello, World!\")\n    return True\n```",
            metadata={
                "content_type": "code",
                "chunk_index": 2,
                "chunk_tokens": 18,
                "chunk_chars": 70,
                "word_count": 8
            }
        )
    ]


@pytest.fixture
def sample_document_metadata() -> Dict[str, Any]:
    """Sample document metadata for testing."""
    return {
        "source_file": "/path/to/test.md",
        "file_size": 1000,
        "processing_strategy": "hybrid_chunking",
        "book_title": "Test Document"
    }


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    mock = Mock()
    mock.encode.return_value = [1, 2, 3, 4, 5]  # Always return 5 tokens
    return mock


@pytest.fixture
def chunking_config():
    """Sample chunking configuration."""
    return {
        "chunk_size": 100,
        "chunk_overlap": 20,
        "min_chunk_words": 10,
        "max_chunk_words": 200
    }


# Test data for edge cases
@pytest.fixture
def edge_case_content():
    """Various edge case content for testing."""
    return {
        "empty": "",
        "whitespace_only": "   \n\t  \n  ",
        "single_word": "word",
        "no_headers": "Just plain text without any headers or structure.",
        "only_headers": "# Header 1\n## Header 2\n### Header 3",
        "mixed_languages": "Hello world. Bonjour le monde. Hola mundo.",
        "special_chars": "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
        "unicode": "Unicode: 你好世界 🌍 émojis 🎉",
        "long_line": "A" * 1000,  # Very long single line
    }


# Quality evaluation test data
@pytest.fixture
def quality_test_chunks():
    """Chunks specifically designed for quality evaluation testing."""
    return {
        "good_chunks": [
            Document(
                page_content="This is a well-formed chunk with proper sentence structure.",
                metadata={"chunk_index": 0, "chunk_tokens": 12, "word_count": 11}
            ),
            Document(
                page_content="Another good chunk that follows the same pattern.",
                metadata={"chunk_index": 1, "chunk_tokens": 10, "word_count": 9}
            )
        ],
        "poor_chunks": [
            Document(
                page_content="",  # Empty chunk
                metadata={"chunk_index": 0, "chunk_tokens": 0, "word_count": 0}
            ),
            Document(
                page_content="word",  # Very short chunk
                metadata={"chunk_index": 1, "chunk_tokens": 1, "word_count": 1}
            ),
            Document(
                page_content="Incomplete sentence without proper ending",  # No punctuation
                metadata={"chunk_index": 2, "chunk_tokens": 7, "word_count": 7}
            )
        ]
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables and paths."""
    # Ensure we're using test configuration
    monkeypatch.setenv("TESTING", "true")
    
    # Mock external API calls if needed
    # This prevents tests from making real API calls
    pass


# Cleanup fixture
@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts():
    """Clean up any test artifacts after test session."""
    yield
    # Cleanup logic here if needed
    pass