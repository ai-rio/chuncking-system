"""
Tests for metadata enricher module.

This module provides basic test coverage for the metadata enricher
to improve overall test coverage metrics.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.utils.metadata_enricher import MetadataEnricher
from langchain_core.documents import Document


class TestMetadataEnricher:
    """Test MetadataEnricher implementation."""
    
    def test_metadata_enricher_initialization(self):
        """Test MetadataEnricher initialization."""
        enricher = MetadataEnricher()
        assert enricher is not None
    
    def test_enrich_chunk_metadata_basic(self):
        """Test basic metadata enrichment."""
        enricher = MetadataEnricher()
        
        # Create test chunk
        chunk = Document(
            page_content="# Test Header\n\nThis is test content.",
            metadata={"source": "test.md"}
        )
        
        enriched = enricher.enrich_chunk_metadata(chunk, chunk_index=0)
        
        # Should have additional metadata
        assert "chunk_index" in enriched.metadata
        assert "chunk_id" in enriched.metadata
        assert "processed_at" in enriched.metadata
        assert enriched.metadata["chunk_index"] == 0
    
    def test_enrich_chunk_metadata_preserves_existing(self):
        """Test that existing metadata is preserved."""
        enricher = MetadataEnricher()
        
        existing_metadata = {
            "source": "test.md",
            "page": 1,
            "section": "introduction"
        }
        
        chunk = Document(
            page_content="# Test\n\nContent.",
            metadata=existing_metadata
        )
        
        enriched = enricher.enrich_chunk_metadata(chunk, chunk_index=5)
        
        # Should preserve existing metadata
        assert enriched.metadata["source"] == "test.md"
        assert enriched.metadata["page"] == 1
        assert enriched.metadata["section"] == "introduction"
        
        # Should add new metadata
        assert "chunk_index" in enriched.metadata