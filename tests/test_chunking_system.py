
import pytest
from pathlib import Path
from src.chunking_system import DocumentChunker, ChunkingConfig

@pytest.fixture
def chunker():
    config = ChunkingConfig(enable_security=False, enable_monitoring=False, enable_caching=False)
    return DocumentChunker(config=config)

@pytest.fixture
def temp_dir_with_files(tmp_path):
    (tmp_path / "file1.md").write_text("content1")
    (tmp_path / "file2.md").write_text("content2")
    return tmp_path

def test_chunk_directory(chunker, temp_dir_with_files):
    results = chunker.chunk_directory(temp_dir_with_files)
    assert len(results) == 2
    assert results[0].success is True
    assert results[1].success is True
