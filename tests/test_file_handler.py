"""
Unit tests for the FileHandler class.
"""

import pytest
import json
import pickle
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import pandas as pd

from langchain_core.documents import Document

from src.utils.file_handler import FileHandler


class TestFileHandler:
    """Test cases for FileHandler."""

    def test_find_markdown_files_basic(self, temp_dir):
        """Test finding markdown files in a directory."""
        # Create test files
        (temp_dir / "test1.md").write_text("# Test 1")
        (temp_dir / "test2.markdown").write_text("# Test 2")
        (temp_dir / "test3.txt").write_text("Not markdown")
        (temp_dir / "test4.py").write_text("# Python code")
        
        files = FileHandler.find_markdown_files(str(temp_dir))
        
        assert len(files) == 2
        assert any("test1.md" in f for f in files)
        assert any("test2.markdown" in f for f in files)
        assert not any("test3.txt" in f for f in files)
        assert not any("test4.py" in f for f in files)

    def test_find_markdown_files_all_extensions(self, temp_dir):
        """Test finding all supported markdown extensions."""
        extensions = ['.md', '.markdown', '.mdown', '.mkd']
        
        for i, ext in enumerate(extensions):
            (temp_dir / f"test{i}{ext}").write_text(f"# Test {i}")
        
        files = FileHandler.find_markdown_files(str(temp_dir))
        
        assert len(files) == len(extensions)

    def test_find_markdown_files_case_insensitive(self, temp_dir):
        """Test finding markdown files with different cases."""
        (temp_dir / "test1.MD").write_text("# Test 1")
        (temp_dir / "test2.Md").write_text("# Test 2")
        (temp_dir / "test3.MARKDOWN").write_text("# Test 3")
        
        files = FileHandler.find_markdown_files(str(temp_dir))
        
        assert len(files) == 3

    def test_find_markdown_files_subdirectories(self, temp_dir):
        """Test finding markdown files in subdirectories."""
        # Create subdirectories
        subdir1 = temp_dir / "sub1"
        subdir2 = temp_dir / "sub2"
        subdir1.mkdir()
        subdir2.mkdir()
        
        # Create files in different locations
        (temp_dir / "root.md").write_text("# Root")
        (subdir1 / "sub1.md").write_text("# Sub1")
        (subdir2 / "sub2.md").write_text("# Sub2")
        
        files = FileHandler.find_markdown_files(str(temp_dir))
        
        assert len(files) == 3
        assert any("root.md" in f for f in files)
        assert any(os.path.join("sub1", "sub1.md") in f for f in files)
        assert any(os.path.join("sub2", "sub2.md") in f for f in files)

    def test_find_markdown_files_empty_directory(self, temp_dir):
        """Test finding markdown files in empty directory."""
        files = FileHandler.find_markdown_files(str(temp_dir))
        assert files == []

    def test_find_markdown_files_nonexistent_directory(self):
        """Test finding markdown files in nonexistent directory."""
        with pytest.raises(OSError):
            FileHandler.find_markdown_files("/nonexistent/directory")

    def test_find_markdown_files_sorted(self, temp_dir):
        """Test that found files are sorted."""
        (temp_dir / "z_file.md").write_text("# Z")
        (temp_dir / "a_file.md").write_text("# A")
        (temp_dir / "m_file.md").write_text("# M")
        
        files = FileHandler.find_markdown_files(str(temp_dir))
        
        # Should be sorted alphabetically
        assert files == sorted(files)

    def test_save_chunks_json(self, temp_dir, sample_chunks):
        """Test saving chunks in JSON format."""
        output_path = temp_dir / "chunks.json"
        
        FileHandler.save_chunks(sample_chunks, str(output_path), 'json')
        
        assert output_path.exists()
        
        # Verify content
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert len(data) == len(sample_chunks)
        assert 'content' in data[0]
        assert 'metadata' in data[0]
        assert data[0]['content'] == sample_chunks[0].page_content

    def test_save_chunks_json_creates_directories(self, temp_dir, sample_chunks):
        """Test that save_chunks creates parent directories."""
        output_path = temp_dir / "nested" / "dir" / "chunks.json"
        
        FileHandler.save_chunks(sample_chunks, str(output_path), 'json')
        
        assert output_path.exists()
        assert output_path.parent.exists()

    def test_save_chunks_pickle(self, temp_dir, sample_chunks):
        """Test saving chunks in pickle format."""
        output_path = temp_dir / "chunks.pickle"
        
        FileHandler.save_chunks(sample_chunks, str(output_path), 'pickle')
        
        assert output_path.exists()
        
        # Verify content
        with open(output_path, 'rb') as f:
            data = pickle.load(f)
        
        assert len(data) == len(sample_chunks)
        assert isinstance(data[0], Document)
        assert data[0].page_content == sample_chunks[0].page_content

    def test_save_chunks_csv(self, temp_dir, sample_chunks):
        """Test saving chunks in CSV format."""
        output_path = temp_dir / "chunks.csv"
        
        FileHandler.save_chunks(sample_chunks, str(output_path), 'csv')
        
        assert output_path.exists()
        
        # Verify content
        df = pd.read_csv(output_path)
        
        assert len(df) == len(sample_chunks)
        assert 'chunk_id' in df.columns
        assert 'content' in df.columns
        assert 'source' in df.columns
        assert 'tokens' in df.columns
        assert 'words' in df.columns

    def test_save_chunks_csv_content(self, temp_dir, sample_chunks):
        """Test CSV content mapping."""
        output_path = temp_dir / "chunks.csv"
        
        FileHandler.save_chunks(sample_chunks, str(output_path), 'csv')
        
        df = pd.read_csv(output_path)
        
        # Check first chunk mapping
        assert df.iloc[0]['content'] == sample_chunks[0].page_content
        assert df.iloc[0]['chunk_id'] == 0
        assert df.iloc[0]['tokens'] == sample_chunks[0].metadata.get('chunk_tokens', 0)
        assert df.iloc[0]['words'] == sample_chunks[0].metadata.get('word_count', 0)

    def test_save_chunks_json_unicode(self, temp_dir):
        """Test saving chunks with unicode content."""
        unicode_chunks = [
            Document(
                page_content="Unicode test: 你好世界 🌍 émojis 🎉",
                metadata={"test": "value"}
            )
        ]
        
        output_path = temp_dir / "unicode_chunks.json"
        
        FileHandler.save_chunks(unicode_chunks, str(output_path), 'json')
        
        assert output_path.exists()
        
        # Verify unicode is preserved
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert "你好世界" in data[0]['content']
        assert "🌍" in data[0]['content']

    def test_save_chunks_empty_list(self, temp_dir):
        """Test saving empty chunks list."""
        output_path = temp_dir / "empty_chunks.json"
        
        FileHandler.save_chunks([], str(output_path), 'json')
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert data == []

    def test_load_chunks_json(self, temp_dir, sample_chunks):
        """Test loading chunks from JSON format."""
        output_path = temp_dir / "chunks.json"
        
        # Save first
        FileHandler.save_chunks(sample_chunks, str(output_path), 'json')
        
        # Load back
        loaded_chunks = FileHandler.load_chunks(str(output_path))
        
        assert len(loaded_chunks) == len(sample_chunks)
        assert isinstance(loaded_chunks[0], Document)
        assert loaded_chunks[0].page_content == sample_chunks[0].page_content
        assert loaded_chunks[0].metadata == sample_chunks[0].metadata

    def test_load_chunks_pickle(self, temp_dir, sample_chunks):
        """Test loading chunks from pickle format."""
        output_path = temp_dir / "chunks.pickle"
        
        # Save first
        FileHandler.save_chunks(sample_chunks, str(output_path), 'pickle')
        
        # Load back
        loaded_chunks = FileHandler.load_chunks(str(output_path))
        
        assert len(loaded_chunks) == len(sample_chunks)
        assert isinstance(loaded_chunks[0], Document)
        assert loaded_chunks[0].page_content == sample_chunks[0].page_content

    def test_load_chunks_nonexistent_file(self, temp_dir):
        """Test loading chunks from nonexistent file."""
        nonexistent_path = temp_dir / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            FileHandler.load_chunks(str(nonexistent_path))

    def test_load_chunks_unsupported_format(self, temp_dir):
        """Test loading chunks from unsupported format."""
        unsupported_path = temp_dir / "chunks.txt"
        unsupported_path.write_text("Some content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            FileHandler.load_chunks(str(unsupported_path))

    def test_load_chunks_invalid_json(self, temp_dir):
        """Test loading chunks from invalid JSON file."""
        invalid_json_path = temp_dir / "invalid.json"
        invalid_json_path.write_text("invalid json content")
        
        with pytest.raises(json.JSONDecodeError):
            FileHandler.load_chunks(str(invalid_json_path))

    def test_load_chunks_empty_json(self, temp_dir):
        """Test loading chunks from empty JSON file."""
        empty_json_path = temp_dir / "empty.json"
        empty_json_path.write_text("[]")
        
        loaded_chunks = FileHandler.load_chunks(str(empty_json_path))
        
        assert loaded_chunks == []

    def test_save_load_roundtrip_json(self, temp_dir, sample_chunks):
        """Test save and load roundtrip for JSON."""
        output_path = temp_dir / "roundtrip.json"
        
        # Save and load
        FileHandler.save_chunks(sample_chunks, str(output_path), 'json')
        loaded_chunks = FileHandler.load_chunks(str(output_path))
        
        # Should be identical
        assert len(loaded_chunks) == len(sample_chunks)
        for original, loaded in zip(sample_chunks, loaded_chunks):
            assert original.page_content == loaded.page_content
            assert original.metadata == loaded.metadata

    def test_save_load_roundtrip_pickle(self, temp_dir, sample_chunks):
        """Test save and load roundtrip for pickle."""
        output_path = temp_dir / "roundtrip.pickle"
        
        # Save and load
        FileHandler.save_chunks(sample_chunks, str(output_path), 'pickle')
        loaded_chunks = FileHandler.load_chunks(str(output_path))
        
        # Should be identical
        assert len(loaded_chunks) == len(sample_chunks)
        for original, loaded in zip(sample_chunks, loaded_chunks):
            assert original.page_content == loaded.page_content
            assert original.metadata == loaded.metadata

    def test_csv_metadata_extraction(self, temp_dir):
        """Test CSV format extracts correct metadata."""
        chunks = [
            Document(
                page_content="Test content",
                metadata={
                    'source': '/path/to/file.md',
                    'chunk_tokens': 10,
                    'word_count': 8,
                    'other_metadata': 'value'
                }
            )
        ]
        
        output_path = temp_dir / "metadata_test.csv"
        
        FileHandler.save_chunks(chunks, str(output_path), 'csv')
        
        df = pd.read_csv(output_path)
        
        assert df.iloc[0]['source'] == '/path/to/file.md'
        assert df.iloc[0]['tokens'] == 10
        assert df.iloc[0]['words'] == 8
        # Note: 'other_metadata' won't be in CSV as it's not explicitly mapped

    def test_csv_missing_metadata(self, temp_dir):
        """Test CSV format handles missing metadata gracefully."""
        chunks = [
            Document(
                page_content="Test content",
                metadata={}  # No metadata
            )
        ]
        
        output_path = temp_dir / "no_metadata.csv"
        
        FileHandler.save_chunks(chunks, str(output_path), 'csv')
        
        df = pd.read_csv(output_path)
        
        assert df.iloc[0]['source'] == ''  # Default empty string
        assert df.iloc[0]['tokens'] == 0   # Default 0
        assert df.iloc[0]['words'] == 0    # Default 0

    def test_file_handler_static_methods(self):
        """Test that FileHandler methods work as static methods."""
        # Should be able to call without instantiating class
        test_dir = "/tmp"  # Use system temp directory
        
        # This should not raise an error about calling on instance
        try:
            files = FileHandler.find_markdown_files(test_dir)
            assert isinstance(files, list)
        except OSError:
            # Directory might not exist or be accessible, that's ok
            pass

    def test_large_chunks_handling(self, temp_dir):
        """Test handling of large number of chunks."""
        # Create many chunks
        large_chunk_list = [
            Document(
                page_content=f"Chunk {i} content",
                metadata={"chunk_index": i, "chunk_tokens": 5}
            ) for i in range(1000)
        ]
        
        output_path = temp_dir / "large_chunks.json"
        
        # Should handle large lists without issues
        FileHandler.save_chunks(large_chunk_list, str(output_path), 'json')
        
        assert output_path.exists()
        
        loaded_chunks = FileHandler.load_chunks(str(output_path))
        assert len(loaded_chunks) == 1000

    def test_special_characters_in_content(self, temp_dir):
        """Test handling content with special characters."""
        special_chunks = [
            Document(
                page_content='Content with "quotes" and \'apostrophes\'',
                metadata={"test": "value"}
            ),
            Document(
                page_content="Content with\nnewlines\tand\ttabs",
                metadata={"test": "value"}
            ),
            Document(
                page_content="Content with special chars: @#$%^&*()",
                metadata={"test": "value"}
            )
        ]
        
        output_path = temp_dir / "special_chunks.json"
        
        FileHandler.save_chunks(special_chunks, str(output_path), 'json')
        loaded_chunks = FileHandler.load_chunks(str(output_path))
        
        # Content should be preserved exactly
        for original, loaded in zip(special_chunks, loaded_chunks):
            assert original.page_content == loaded.page_content

    def test_file_permissions_error_handling(self, temp_dir):
        """Test handling of file permission errors."""
        output_path = temp_dir / "readonly.json"
        
        # Create file and make it read-only
        output_path.write_text("{}")
        output_path.chmod(0o444)  # Read-only
        
        chunks = [Document(page_content="test", metadata={})]
        
        try:
            with pytest.raises(PermissionError):
                FileHandler.save_chunks(chunks, str(output_path), 'json')
        finally:
            # Restore write permissions for cleanup
            output_path.chmod(0o644)

    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_save_chunks_permission_error(self, mock_open, temp_dir, sample_chunks):
        """Test save_chunks handles permission errors."""
        output_path = temp_dir / "permission_test.json"
        
        with pytest.raises(PermissionError):
            FileHandler.save_chunks(sample_chunks, str(output_path), 'json')

    def test_path_with_spaces(self, temp_dir, sample_chunks):
        """Test handling paths with spaces."""
        output_path = temp_dir / "path with spaces" / "chunks.json"
        
        FileHandler.save_chunks(sample_chunks, str(output_path), 'json')
        
        assert output_path.exists()
        
        loaded_chunks = FileHandler.load_chunks(str(output_path))
        assert len(loaded_chunks) == len(sample_chunks)