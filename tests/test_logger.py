
import pytest
from unittest.mock import Mock, patch
from src.utils.logger import StructuredLogger, ChunkingLogger, get_logger, get_chunking_logger

@pytest.fixture
def structured_logger():
    with patch('src.utils.logger.logging.getLogger') as mock_get_logger:
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        yield StructuredLogger("test"), mock_logger

@pytest.fixture
def chunking_logger():
    with patch('src.utils.logger.StructuredLogger') as mock_structured_logger:
        mock_logger = Mock()
        mock_structured_logger.return_value = mock_logger
        yield ChunkingLogger("test"), mock_logger

def test_structured_logger_info(structured_logger):
    logger, mock_logger = structured_logger
    logger.info("Test message", key="value")
    mock_logger.log.assert_called_once()

def test_chunking_logger_start_end_operation(chunking_logger):
    logger, mock_logger = chunking_logger
    logger.start_operation("test_op")
    logger.end_operation("test_op")
    assert mock_logger.info.call_count == 2
    assert mock_logger.error.call_count == 0

def test_chunking_logger_log_chunk_stats(chunking_logger):
    logger, mock_logger = chunking_logger
    class MockChunk: 
        def __init__(self, content):
            self.page_content = content
    chunks = [MockChunk("a"), MockChunk("b")]
    logger.log_chunk_stats(chunks)
    mock_logger.info.assert_called_once()

def test_get_logger():
    with patch('src.utils.logger.StructuredLogger') as mock_structured_logger:
        get_logger("test")
        mock_structured_logger.assert_called_once_with("test", "INFO")

def test_get_chunking_logger():
    with patch('src.utils.logger.ChunkingLogger') as mock_chunking_logger:
        get_chunking_logger("test")
        mock_chunking_logger.assert_called_once_with("test")
