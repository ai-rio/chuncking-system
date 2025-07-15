"""
Unit tests for the MetadataEnricher class.
"""

import pytest
import hashlib
from unittest.mock import patch, Mock
from datetime import datetime

from langchain_core.documents import Document

from src.utils.metadata_enricher import MetadataEnricher


class TestMetadataEnricher:
    """Test cases for MetadataEnricher."""

    def test_enrich_chunk_basic(self):
        """Test basic chunk enrichment."""
        chunk = Document(
            page_content="Sample content for testing",
            metadata={"original_key": "original_value"}
        )
        
        document_info = {"source_file": "/path/to/test.md"}
        
        enriched = MetadataEnricher.enrich_chunk(chunk, document_info)
        
        # Check that original metadata is preserved
        assert enriched.metadata["original_key"] == "original_value"
        
        # Check that document info is added
        assert enriched.metadata["source_file"] == "/path/to/test.md"
        
        # Check that new metadata is added
        assert "chunk_id" in enriched.metadata
        assert "processed_at" in enriched.metadata
        assert "content_hash" in enriched.metadata
        assert "has_code" in enriched.metadata
        assert "has_urls" in enriched.metadata
        assert "has_headers" in enriched.metadata
        assert "language" in enriched.metadata

    def test_enrich_chunk_no_document_info(self):
        """Test chunk enrichment without document info."""
        chunk = Document(
            page_content="Test content",
            metadata={"test": "value"}
        )
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        # Should still enrich with basic metadata
        assert "chunk_id" in enriched.metadata
        assert "processed_at" in enriched.metadata
        assert "content_hash" in enriched.metadata
        assert enriched.metadata["test"] == "value"

    def test_enrich_chunk_none_document_info(self):
        """Test chunk enrichment with None document info."""
        chunk = Document(
            page_content="Test content",
            metadata={}
        )
        
        enriched = MetadataEnricher.enrich_chunk(chunk, None)
        
        # Should work without errors
        assert "chunk_id" in enriched.metadata
        assert "processed_at" in enriched.metadata

    def test_chunk_id_generation(self):
        """Test chunk ID generation."""
        chunk1 = Document(
            page_content="Same content",
            metadata={"source": "file1.md"}
        )
        
        chunk2 = Document(
            page_content="Same content",
            metadata={"source": "file1.md"}
        )
        
        chunk3 = Document(
            page_content="Different content",
            metadata={"source": "file1.md"}
        )
        
        enriched1 = MetadataEnricher.enrich_chunk(chunk1)
        enriched2 = MetadataEnricher.enrich_chunk(chunk2)
        enriched3 = MetadataEnricher.enrich_chunk(chunk3)
        
        # Same content and source should produce same chunk ID
        assert enriched1.metadata["chunk_id"] == enriched2.metadata["chunk_id"]
        
        # Different content should produce different chunk ID
        assert enriched1.metadata["chunk_id"] != enriched3.metadata["chunk_id"]

    def test_chunk_id_length(self):
        """Test that chunk ID has correct length."""
        chunk = Document(page_content="Test", metadata={})
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        # Should be 12 characters (first 12 of MD5 hash)
        assert len(enriched.metadata["chunk_id"]) == 12

    def test_content_hash_generation(self):
        """Test content hash generation."""
        content = "Test content for hashing"
        chunk = Document(page_content=content, metadata={})
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        expected_hash = hashlib.md5(content.encode()).hexdigest()
        assert enriched.metadata["content_hash"] == expected_hash

    def test_content_hash_different_content(self):
        """Test that different content produces different hashes."""
        chunk1 = Document(page_content="Content 1", metadata={})
        chunk2 = Document(page_content="Content 2", metadata={})
        
        enriched1 = MetadataEnricher.enrich_chunk(chunk1)
        enriched2 = MetadataEnricher.enrich_chunk(chunk2)
        
        assert enriched1.metadata["content_hash"] != enriched2.metadata["content_hash"]

    @patch('src.utils.metadata_enricher.datetime')
    def test_processed_at_timestamp(self, mock_datetime):
        """Test processed_at timestamp generation."""
        mock_now = Mock()
        mock_now.isoformat.return_value = "2023-01-01T12:00:00"
        mock_datetime.now.return_value = mock_now
        
        chunk = Document(page_content="Test", metadata={})
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        assert enriched.metadata["processed_at"] == "2023-01-01T12:00:00"

    def test_has_code_detection_backticks(self):
        """Test code detection with backticks."""
        chunk = Document(
            page_content="Here's some code: ```python\nprint('hello')\n```",
            metadata={}
        )
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        assert enriched.metadata["has_code"] is True

    def test_has_code_detection_def_keyword(self):
        """Test code detection with 'def' keyword."""
        chunk = Document(
            page_content="def my_function():\n    return True",
            metadata={}
        )
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        assert enriched.metadata["has_code"] is True

    def test_has_code_detection_import_keyword(self):
        """Test code detection with 'import' keyword."""
        chunk = Document(
            page_content="import os\nimport sys",
            metadata={}
        )
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        assert enriched.metadata["has_code"] is True

    def test_has_code_detection_no_code(self):
        """Test code detection with no code."""
        chunk = Document(
            page_content="This is just regular text without any code.",
            metadata={}
        )
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        assert enriched.metadata["has_code"] is False

    def test_has_urls_detection_http(self):
        """Test URL detection with http."""
        chunk = Document(
            page_content="Visit http://example.com for more info.",
            metadata={}
        )
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        assert enriched.metadata["has_urls"] is True

    def test_has_urls_detection_https(self):
        """Test URL detection with https."""
        chunk = Document(
            page_content="Check out https://secure-site.com",
            metadata={}
        )
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        assert enriched.metadata["has_urls"] is True

    def test_has_urls_detection_no_urls(self):
        """Test URL detection with no URLs."""
        chunk = Document(
            page_content="This text has no URLs or links.",
            metadata={}
        )
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        assert enriched.metadata["has_urls"] is False

    def test_has_headers_detection_true(self):
        """Test header detection with header."""
        chunk = Document(
            page_content="# This is a header\nContent follows.",
            metadata={}
        )
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        assert enriched.metadata["has_headers"] is True

    def test_has_headers_detection_false(self):
        """Test header detection without header."""
        chunk = Document(
            page_content="This is just regular content without headers.",
            metadata={}
        )
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        assert enriched.metadata["has_headers"] is False

    def test_has_headers_detection_header_not_at_start(self):
        """Test header detection when header is not at start."""
        chunk = Document(
            page_content="Some text\n# Header in middle",
            metadata={}
        )
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        # Current implementation only checks if content starts with #
        assert enriched.metadata["has_headers"] is False

    def test_detect_language_technical(self):
        """Test language detection for technical content."""
        technical_terms = ['import', 'def', 'class', 'function']
        
        for term in technical_terms:
            content = f"This content has {term} keyword"
            language = MetadataEnricher._detect_language(content)
            assert language == 'technical'

    def test_detect_language_english(self):
        """Test language detection for English content."""
        english_words = ['the', 'and', 'to', 'of', 'a']
        
        for word in english_words:
            content = f"Content with {word} word"
            language = MetadataEnricher._detect_language(content)
            assert language == 'english'

    def test_detect_language_unknown(self):
        """Test language detection for unknown content."""
        content = "xyz qrs mnp"  # No recognizable words
        language = MetadataEnricher._detect_language(content)
        assert language == 'unknown'

    def test_detect_language_case_insensitive(self):
        """Test that language detection is case insensitive."""
        content_lower = "import module"
        content_upper = "IMPORT MODULE"
        content_mixed = "Import Module"
        
        assert MetadataEnricher._detect_language(content_lower) == 'technical'
        assert MetadataEnricher._detect_language(content_upper) == 'technical'
        assert MetadataEnricher._detect_language(content_mixed) == 'technical'

    def test_detect_language_priority(self):
        """Test language detection priority (technical > english)."""
        # Content with both technical and english words
        content = "import the module and define function"
        language = MetadataEnricher._detect_language(content)
        
        # Should detect as technical since it has technical terms
        assert language == 'technical'

    def test_metadata_merge_order(self):
        """Test that metadata is merged in correct order."""
        original_metadata = {"key1": "original", "key2": "original"}
        document_info = {"key2": "document", "key3": "document"}
        
        chunk = Document(
            page_content="Test content",
            metadata=original_metadata
        )
        
        enriched = MetadataEnricher.enrich_chunk(chunk, document_info)
        
        # Original metadata should be preserved
        assert enriched.metadata["key1"] == "original"
        
        # Document info should override original where keys conflict
        assert enriched.metadata["key2"] == "document"
        
        # Document info should add new keys
        assert enriched.metadata["key3"] == "document"

    def test_enrich_chunk_immutability(self):
        """Test that enriching chunk doesn't modify original."""
        original_metadata = {"original": "value"}
        chunk = Document(
            page_content="Test content",
            metadata=original_metadata.copy()
        )
        
        original_metadata_copy = chunk.metadata.copy()
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        # Original chunk metadata should be unchanged
        assert chunk.metadata == original_metadata_copy
        
        # Enriched chunk should have additional metadata
        assert len(enriched.metadata) > len(original_metadata_copy)

    def test_enrich_chunk_empty_content(self):
        """Test enriching chunk with empty content."""
        chunk = Document(page_content="", metadata={})
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        # Should still work and add metadata
        assert "chunk_id" in enriched.metadata
        assert "content_hash" in enriched.metadata
        assert enriched.metadata["has_code"] is False
        assert enriched.metadata["has_urls"] is False
        assert enriched.metadata["has_headers"] is False

    def test_enrich_chunk_whitespace_content(self):
        """Test enriching chunk with only whitespace."""
        chunk = Document(page_content="   \n\t  ", metadata={})
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        # Should work and detect no special content
        assert enriched.metadata["has_code"] is False
        assert enriched.metadata["has_urls"] is False
        assert enriched.metadata["has_headers"] is False

    def test_enrich_chunk_unicode_content(self):
        """Test enriching chunk with unicode content."""
        chunk = Document(
            page_content="Unicode test: 你好世界 🌍 émojis 🎉",
            metadata={}
        )
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        # Should handle unicode without errors
        assert "chunk_id" in enriched.metadata
        assert "content_hash" in enriched.metadata

    def test_enrich_chunk_very_long_content(self):
        """Test enriching chunk with very long content."""
        long_content = "A" * 10000
        chunk = Document(page_content=long_content, metadata={})
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        # Should handle long content
        assert "chunk_id" in enriched.metadata
        assert len(enriched.metadata["chunk_id"]) == 12

    def test_multiple_feature_detection(self):
        """Test content with multiple features."""
        content = """# Header
        
Visit https://example.com for more info.

```python
import os
def hello():
    print("Hello world")
```

More content here.
"""
        
        chunk = Document(page_content=content, metadata={})
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        assert enriched.metadata["has_headers"] is True
        assert enriched.metadata["has_urls"] is True
        assert enriched.metadata["has_code"] is True
        assert enriched.metadata["language"] == 'technical'

    def test_enrich_chunk_preserves_content(self):
        """Test that enriching preserves original content."""
        original_content = "Original content that should not change"
        chunk = Document(page_content=original_content, metadata={})
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        assert enriched.page_content == original_content

    def test_static_method_behavior(self):
        """Test that enrich_chunk works as a static method."""
        # Should be able to call without instantiating class
        chunk = Document(page_content="Test", metadata={})
        
        enriched = MetadataEnricher.enrich_chunk(chunk)
        
        assert "chunk_id" in enriched.metadata

    def test_detect_language_static_method(self):
        """Test that _detect_language works as a static method."""
        # Should be able to call without instantiating class
        language = MetadataEnricher._detect_language("import test")
        
        assert language == 'technical'