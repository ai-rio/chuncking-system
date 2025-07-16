"""
Test cases for project folder auto-creation feature.
Following TDD principles - tests written first to define expected behavior.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.utils.path_utils import MarkdownFileManager, get_markdown_manager


class TestProjectFolderCreation:
    """Test automatic project folder creation functionality."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        input_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()
        
        # Create a test markdown file
        test_file = Path(input_dir) / "test_book.md"
        test_file.write_text("# Test Book\n\nThis is a test book content.")
        
        yield {
            'input_dir': Path(input_dir),
            'output_dir': Path(output_dir),
            'test_file': test_file
        }
        
        # Cleanup
        shutil.rmtree(input_dir)
        shutil.rmtree(output_dir)
    
    def test_create_project_folder_enabled_by_default(self, temp_dirs):
        """Test that project folders are created by default."""
        manager = MarkdownFileManager()
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20250716_123000"
            
            paths = manager.create_markdown_output_paths(
                temp_dirs['test_file'], 
                temp_dirs['output_dir']
            )
        
        # Should create project-specific folder with timestamp
        expected_project_folder = temp_dirs['output_dir'] / "test_book_20250716_123000"
        assert paths['project_folder'] == expected_project_folder
        assert expected_project_folder.exists()
        
        # Verify subdirectories are created within project folder
        assert (expected_project_folder / 'chunks').exists()
        assert (expected_project_folder / 'reports').exists()
        assert (expected_project_folder / 'logs').exists()
    
    def test_create_project_folder_disabled(self, temp_dirs):
        """Test that project folder creation can be disabled."""
        manager = MarkdownFileManager()
        
        paths = manager.create_markdown_output_paths(
            temp_dirs['test_file'], 
            temp_dirs['output_dir'],
            create_project_folder=False
        )
        
        # Should not create project-specific folder
        assert paths['project_folder'] is None
        assert paths['base'] == temp_dirs['output_dir']
    
    def test_project_folder_name_sanitization(self, temp_dirs):
        """Test that unsafe characters in filenames are sanitized."""
        # Create file with unsafe characters
        unsafe_file = temp_dirs['input_dir'] / "test@book#with$unsafe%chars.md"
        unsafe_file.write_text("# Test content")
        
        manager = MarkdownFileManager()
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20250716_123000"
            
            paths = manager.create_markdown_output_paths(
                unsafe_file, 
                temp_dirs['output_dir']
            )
        
        # Verify unsafe characters are replaced
        project_folder_name = paths['project_folder'].name
        assert '@' not in project_folder_name
        assert '#' not in project_folder_name
        assert '$' not in project_folder_name
        assert '%' not in project_folder_name
        # The sanitization removes unsafe characters, so expect something like 'testbookwithunsafechars_20250716_123000'
        assert 'testbook' in project_folder_name.lower()
        assert '20250716_123000' in project_folder_name
    
    def test_project_folder_handles_existing_directories(self, temp_dirs):
        """Test behavior when project folder already exists."""
        manager = MarkdownFileManager()
        
        # Create first project folder
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20250716_123000"
            
            paths1 = manager.create_markdown_output_paths(
                temp_dirs['test_file'], 
                temp_dirs['output_dir']
            )
        
        # Create second project folder with same timestamp (simulating collision)
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20250716_123000"
            
            paths2 = manager.create_markdown_output_paths(
                temp_dirs['test_file'], 
                temp_dirs['output_dir']
            )
        
        # Both should exist without conflicts
        assert paths1['project_folder'].exists()
        assert paths2['project_folder'].exists()
        # They should be able to coexist (mkdir with exist_ok=True)
    
    def test_timestamp_format_consistency(self, temp_dirs):
        """Test that timestamp format is consistent and parseable."""
        manager = MarkdownFileManager()
        
        # Test with current time
        paths = manager.create_markdown_output_paths(
            temp_dirs['test_file'], 
            temp_dirs['output_dir']
        )
        
        project_folder_name = paths['project_folder'].name
        # Extract timestamp part
        timestamp_part = project_folder_name.split('_')[-2:]  # Last two parts should be date_time
        timestamp_str = '_'.join(timestamp_part)
        
        # Verify timestamp format YYYYMMDD_HHMMSS
        try:
            parsed_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            assert isinstance(parsed_time, datetime)
        except ValueError:
            pytest.fail(f"Timestamp format is invalid: {timestamp_str}")
    
    def test_output_paths_structure_with_project_folder(self, temp_dirs):
        """Test that all expected output paths are created within project folder."""
        manager = MarkdownFileManager()
        
        paths = manager.create_markdown_output_paths(
            temp_dirs['test_file'], 
            temp_dirs['output_dir']
        )
        
        # Verify all required paths exist and are within project folder
        required_paths = ['base', 'chunks', 'reports', 'logs', 'temp']
        for path_key in required_paths:
            assert path_key in paths
            assert paths[path_key].exists()
            # Verify path is within project folder
            assert paths['project_folder'] in paths[path_key].parents or paths[path_key] == paths['project_folder']
        
        # Verify file-specific paths
        file_specific_paths = ['chunks_json', 'chunks_csv', 'chunks_pickle', 'quality_report', 'processing_log']
        for path_key in file_specific_paths:
            assert path_key in paths
            # File paths should be within appropriate subdirectories
            if 'chunks' in path_key:
                assert paths['chunks'] in paths[path_key].parents
            elif 'quality_report' in path_key:
                assert paths['reports'] in paths[path_key].parents
            elif 'processing_log' in path_key:
                assert paths['logs'] in paths[path_key].parents
    
    def test_concurrent_project_creation(self, temp_dirs):
        """Test that concurrent project folder creation works correctly."""
        import threading
        import time
        
        manager = MarkdownFileManager()
        results = []
        
        def create_project():
            paths = manager.create_markdown_output_paths(
                temp_dirs['test_file'], 
                temp_dirs['output_dir']
            )
            results.append(paths['project_folder'])
        
        # Create multiple threads to simulate concurrent access
        threads = []
        for i in range(3):
            thread = threading.Thread(target=create_project)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all project folders were created successfully
        assert len(results) == 3
        for project_folder in results:
            assert project_folder.exists()
            assert (project_folder / 'chunks').exists()
            assert (project_folder / 'reports').exists()
    
    def test_project_folder_permissions(self, temp_dirs):
        """Test that project folders are created with correct permissions."""
        manager = MarkdownFileManager()
        
        paths = manager.create_markdown_output_paths(
            temp_dirs['test_file'], 
            temp_dirs['output_dir']
        )
        
        project_folder = paths['project_folder']
        
        # Verify folder is readable and writable
        assert project_folder.is_dir()
        assert project_folder.exists()
        
        # Test that we can create files in the project folder
        test_file = project_folder / "test_write.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"


class TestProjectFolderIntegration:
    """Integration tests for project folder creation with the main system."""
    
    @pytest.fixture
    def temp_setup(self):
        """Setup for integration tests."""
        input_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()
        
        # Create a more realistic test markdown file
        test_file = Path(input_dir) / "integration_test_book.md"
        test_content = """# Integration Test Book

## Chapter 1
This is chapter 1 content with multiple paragraphs.

This is the second paragraph of chapter 1.

## Chapter 2
This is chapter 2 content.

### Section 2.1
Subsection content here.
"""
        test_file.write_text(test_content)
        
        yield {
            'input_dir': Path(input_dir),
            'output_dir': Path(output_dir),
            'test_file': test_file
        }
        
        # Cleanup
        shutil.rmtree(input_dir)
        shutil.rmtree(output_dir)
    
    def test_end_to_end_project_creation(self, temp_setup):
        """Test complete workflow with project folder creation."""
        from src.chunkers.hybrid_chunker import HybridMarkdownChunker
        from src.utils.file_handler import FileHandler
        
        manager = MarkdownFileManager()
        
        # Create project structure
        paths = manager.create_markdown_output_paths(
            temp_setup['test_file'], 
            temp_setup['output_dir']
        )
        
        # Verify we can process a file and save to project folder
        chunker = HybridMarkdownChunker(chunk_size=100, chunk_overlap=20)
        
        # Read file content
        content = temp_setup['test_file'].read_text()
        
        # Create chunks
        chunks = chunker.chunk_document(content, {'source_file': str(temp_setup['test_file'])})
        
        # Save chunks to project folder
        FileHandler.save_chunks(chunks, str(paths['chunks_json']), 'json')
        
        # Verify files were created in project folder
        assert paths['chunks_json'].exists()
        assert paths['chunks_json'].stat().st_size > 0
        
        # Verify project folder structure is maintained
        assert paths['project_folder'].exists()
        assert (paths['project_folder'] / 'chunks').exists()
        assert (paths['project_folder'] / 'reports').exists()