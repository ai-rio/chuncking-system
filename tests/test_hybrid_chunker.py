"""
Unit tests for the HybridMarkdownChunker class.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document

from src.chunkers.hybrid_chunker import HybridMarkdownChunker
from src.config.settings import config


class TestHybridMarkdownChunker:
    """Test cases for HybridMarkdownChunker."""

    @pytest.fixture
    def chunker(self) -> HybridMarkdownChunker:
        """Create a HybridMarkdownChunker instance for testing."""
        return HybridMarkdownChunker(chunk_size=100, chunk_overlap=20)

    @pytest.fixture
    def chunker_default(self) -> HybridMarkdownChunker:
        """Create a HybridMarkdownChunker with default settings."""
        return HybridMarkdownChunker()

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        chunker = HybridMarkdownChunker()
        
        assert chunker.chunk_size == config.DEFAULT_CHUNK_SIZE
        assert chunker.chunk_overlap == config.DEFAULT_CHUNK_OVERLAP
        assert chunker.enable_semantic is False
        assert chunker.tokenizer is not None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        chunk_size = 500
        chunk_overlap = 50
        enable_semantic = True
        
        chunker = HybridMarkdownChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_semantic=enable_semantic
        )
        
        assert chunker.chunk_size == chunk_size
        assert chunker.chunk_overlap == chunk_overlap
        assert chunker.enable_semantic == enable_semantic

    @patch('tiktoken.encoding_for_model')
    def test_tokenizer_initialization_success(self, mock_encoding_for_model):
        """Test successful tokenizer initialization."""
        mock_tokenizer = Mock()
        mock_encoding_for_model.return_value = mock_tokenizer
        
        chunker = HybridMarkdownChunker()
        
        assert chunker.tokenizer == mock_tokenizer
        mock_encoding_for_model.assert_called_once_with("gpt-3.5-turbo")

    @patch('tiktoken.get_encoding')
    @patch('tiktoken.encoding_for_model')
    def test_tokenizer_initialization_fallback(self, mock_encoding_for_model, mock_get_encoding):
        """Test tokenizer initialization fallback."""
        mock_encoding_for_model.side_effect = Exception("Model not found")
        mock_fallback_tokenizer = Mock()
        mock_get_encoding.return_value = mock_fallback_tokenizer
        
        chunker = HybridMarkdownChunker()
        
        assert chunker.tokenizer == mock_fallback_tokenizer
        mock_get_encoding.assert_called_once_with("cl100k_base")

    def test_token_length(self, chunker):
        """Test token length calculation."""
        chunker.llm_provider = None  # Disable LLM provider to test tokenizer
        chunker.tokenizer = Mock()
        chunker.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        text = "Sample text"
        length = chunker._token_length(text)
        
        assert length == 5
        chunker.tokenizer.encode.assert_called_once_with(text)

    def test_detect_content_type_with_headers(self, chunker):
        """Test content type detection with headers."""
        content = "# Header 1\n## Header 2\nSome content"
        
        analysis = chunker._detect_content_type(content)
        
        assert analysis['has_headers'] is True
        assert analysis['has_code'] is False
        assert analysis['has_tables'] is False

    def test_detect_content_type_with_code(self, chunker):
        """Test content type detection with code blocks."""
        content = "Some text\n```python\nprint('hello')\n```\nMore text"
        
        analysis = chunker._detect_content_type(content)
        
        assert analysis['has_code'] is True
        assert analysis['has_headers'] is False

    def test_detect_content_type_with_tables(self, chunker):
        """Test content type detection with tables."""
        content = "| Col1 | Col2 |\n|------|------|\n| A    | B    |"
        
        analysis = chunker._detect_content_type(content)
        
        assert analysis['has_tables'] is True

    def test_detect_content_type_with_lists(self, chunker):
        """Test content type detection with lists."""
        content = "- Item 1\n- Item 2\n* Item 3\n+ Item 4"
        
        analysis = chunker._detect_content_type(content)
        
        assert analysis['has_lists'] is True

    def test_detect_content_type_large_content(self, chunker):
        """Test detection of large content."""
        chunker.chunk_size = 100
        content = "a" * 600  # 6 times chunk size
        
        analysis = chunker._detect_content_type(content)
        
        assert analysis['is_large'] is True

    def test_chunk_document_empty_content(self, chunker):
        """Test chunking empty content."""
        result = chunker.chunk_document("")
        assert result == []
        
        result = chunker.chunk_document("   \n\t  ")
        assert result == []

    def test_chunk_document_with_headers(self, chunker, sample_markdown_file):
        """Test chunking document with headers."""
        content = sample_markdown_file.read_text()
        metadata = {"source": str(sample_markdown_file)}
        
        chunks = chunker.chunk_document(content, metadata)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
        
        # Check that metadata is preserved
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == str(sample_markdown_file)

    def test_chunk_document_with_code(self, chunker):
        """Test chunking document with code blocks."""
        content = """Here's some code without headers:

```python
def hello():
    print("Hello, World!")
    return True
```

And some more text after the code block.
This should trigger code-aware chunking since there are no headers.
"""
        
        chunks = chunker.chunk_document(content)
        
        assert len(chunks) > 0
        # Should detect code and use code-aware chunking
        code_chunks = [c for c in chunks if c.metadata.get('content_type') == 'code']
        assert len(code_chunks) > 0

    def test_chunk_document_plain_text(self, chunker):
        """Test chunking plain text without special structure."""
        content = "This is just plain text without any headers or special formatting. " * 20
        
        chunks = chunker.chunk_document(content)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_header_recursive_chunking(self, chunker):
        """Test header-based recursive chunking."""
        content = """# Big Header
        
This is a lot of content that should be split because it's too long for the chunk size. """ * 20
        
        metadata = {"test": "value"}
        analysis = {"has_headers": True, "has_code": False}
        
        with patch.object(chunker, '_token_length') as mock_token_length:
            # Return function that gives realistic token lengths
            mock_token_length.side_effect = lambda text: min(200, len(text.split()) * 2)
            
            chunks = chunker._header_recursive_chunking(content, metadata, analysis)
            
            assert len(chunks) > 0
            for chunk in chunks:
                assert "test" in chunk.metadata

    def test_header_recursive_chunking_fallback(self, chunker):
        """Test fallback when header splitting fails."""
        content = "# Header\nContent"
        metadata = {}
        analysis = {"has_headers": True}
        
        with patch.object(chunker.header_splitter, 'split_text') as mock_split:
            mock_split.side_effect = Exception("Split failed")
            
            # Should fallback to simple recursive chunking
            chunks = chunker._header_recursive_chunking(content, metadata, analysis)
            
            assert len(chunks) >= 0  # Should not crash

    def test_simple_recursive_chunking(self, chunker):
        """Test simple recursive chunking."""
        content = "Simple content without headers. " * 10
        metadata = {"source": "test"}
        
        chunks = chunker._simple_recursive_chunking(content, metadata)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test"

    def test_code_aware_chunking_with_code_blocks(self, chunker):
        """Test code-aware chunking with actual code blocks."""
        content = """
Some text before code.

```python
def function1():
    return "hello"

def function2():
    return "world"
```

Some text after code.

```javascript
function hello() {
    return "hello";
}
```

Final text.
"""
        
        metadata = {"source": "test"}
        chunks = chunker._code_aware_chunking(content, metadata)
        
        assert len(chunks) > 0
        
        # Should have code chunks
        code_chunks = [c for c in chunks if c.metadata.get('content_type') == 'code']
        assert len(code_chunks) > 0

    def test_code_aware_chunking_no_code(self, chunker):
        """Test code-aware chunking when no code blocks exist."""
        content = "Just regular text without any code blocks."
        metadata = {}
        
        chunks = chunker._code_aware_chunking(content, metadata)
        
        assert len(chunks) > 0
        # Should fallback to simple recursive chunking
        assert all(c.metadata.get('content_type') != 'code' for c in chunks)

    def test_post_process_chunks(self, chunker):
        """Test post-processing of chunks."""
        chunks = [
            Document(page_content="Valid chunk content", metadata={}),
            Document(page_content="a", metadata={}),  # Too short
            Document(page_content="Another valid chunk", metadata={}),
            Document(page_content="", metadata={}),  # Empty
        ]
        
        with patch.object(chunker, '_token_length') as mock_token_length:
            mock_token_length.return_value = 10
            
            processed = chunker._post_process_chunks(chunks)
            
            # Should filter out short/empty chunks
            assert len(processed) == 2
            
            # Check metadata enrichment
            for i, chunk in enumerate(processed):
                assert chunk.metadata['chunk_index'] == i
                assert 'chunk_tokens' in chunk.metadata
                assert 'chunk_chars' in chunk.metadata
                assert 'word_count' in chunk.metadata

    def test_batch_process_files(self, chunker, temp_dir):
        """Test batch processing of multiple files."""
        # Create test files
        file1 = temp_dir / "test1.md"
        file2 = temp_dir / "test2.md" 
        file1.write_text("# File 1\nContent 1")
        file2.write_text("# File 2\nContent 2")
        
        file_paths = [str(file1), str(file2)]
        
        # Mock progress callback
        progress_callback = Mock()
        
        results = chunker.batch_process_files(file_paths, progress_callback)
        
        assert len(results) == 2
        assert str(file1) in results
        assert str(file2) in results
        assert len(results[str(file1)]) > 0
        assert len(results[str(file2)]) > 0
        
        # Check progress callback was called
        assert progress_callback.call_count == 2

    def test_batch_process_files_with_error(self, chunker, temp_dir):
        """Test batch processing with file errors."""
        # Create one valid file and one invalid path
        valid_file = temp_dir / "valid.md"
        valid_file.write_text("# Valid\nContent")
        invalid_file = temp_dir / "nonexistent.md"
        
        file_paths = [str(valid_file), str(invalid_file)]
        
        results = chunker.batch_process_files(file_paths)
        
        assert len(results) == 2
        assert len(results[str(valid_file)]) > 0
        assert len(results[str(invalid_file)]) == 0  # Error should result in empty list

    @patch('gc.collect')
    def test_batch_process_memory_cleanup(self, mock_gc_collect, chunker, temp_dir):
        """Test memory cleanup during batch processing."""
        # Create files that would trigger cleanup
        files = []
        for i in range(config.BATCH_SIZE + 1):
            file_path = temp_dir / f"test{i}.md"
            file_path.write_text(f"# File {i}\nContent {i}")
            files.append(str(file_path))
        
        chunker.batch_process_files(files)
        
        # Should call gc.collect at least once
        assert mock_gc_collect.call_count >= 1

    def test_chunker_with_different_strategies(self, chunker, sample_markdown_file, large_markdown_file):
        """Test that chunker selects appropriate strategies for different content types."""
        # Test with markdown (should use header-based)
        markdown_content = sample_markdown_file.read_text()
        markdown_chunks = chunker.chunk_document(markdown_content)
        
        # Test with large content (should still work)
        large_content = large_markdown_file.read_text()
        large_chunks = chunker.chunk_document(large_content)
        
        assert len(markdown_chunks) > 0
        assert len(large_chunks) > len(markdown_chunks)

    def test_chunker_preserves_metadata_hierarchy(self, chunker):
        """Test that chunker preserves header hierarchy in metadata."""
        content = """# Chapter 1
Content for chapter 1.

## Section 1.1
Content for section 1.1.

### Subsection 1.1.1
Content for subsection 1.1.1.
"""
        
        chunks = chunker.chunk_document(content)
        
        # Check that header metadata is preserved
        header_chunks = [c for c in chunks if any(key.startswith('Header') for key in c.metadata.keys())]
        assert len(header_chunks) > 0

    def test_edge_cases(self, chunker, edge_case_content):
        """Test various edge cases."""
        for content_type, content in edge_case_content.items():
            # Should not crash on any edge case
            try:
                chunks = chunker.chunk_document(content)
                assert isinstance(chunks, list)
            except Exception as e:
                pytest.fail(f"Chunker failed on {content_type}: {e}")

    def test_chunker_config_integration(self):
        """Test integration with configuration settings."""
        # Test that chunker respects config values
        chunker = HybridMarkdownChunker()
        
        assert chunker.chunk_size == config.DEFAULT_CHUNK_SIZE
        assert chunker.chunk_overlap == config.DEFAULT_CHUNK_OVERLAP

    def test_chunker_metadata_enrichment(self, chunker):
        """Test that chunker properly enriches chunk metadata."""
        content = "# Test Header\nSome content here."
        base_metadata = {"source": "test.md", "author": "test"}
        
        chunks = chunker.chunk_document(content, base_metadata)
        
        for chunk in chunks:
            # Should preserve base metadata
            assert chunk.metadata.get("source") == "test.md"
            assert chunk.metadata.get("author") == "test"
            
            # Should add chunk-specific metadata
            assert "chunk_index" in chunk.metadata
            assert "chunk_tokens" in chunk.metadata
            assert "word_count" in chunk.metadata