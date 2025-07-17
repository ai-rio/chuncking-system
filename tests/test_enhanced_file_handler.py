import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import List, Dict, Any

from src.utils.enhanced_file_handler import EnhancedFileHandler, FileInfo
from src.utils.file_handler import FileHandler
from src.chunkers.docling_processor import DoclingProcessor, ProcessingResult
from src.exceptions import FileHandlingError, ValidationError


class TestEnhancedFileHandler:
    """Test suite for EnhancedFileHandler format detection and routing"""
    
    def setup_method(self):
        """Setup test environment before each test"""
        self.mock_file_handler = Mock(spec=FileHandler)
        self.mock_docling_processor = Mock(spec=DoclingProcessor)
        self.enhanced_handler = EnhancedFileHandler(
            file_handler=self.mock_file_handler,
            docling_processor=self.mock_docling_processor
        )
        
    def test_format_detection_pdf(self):
        """Test PDF format detection"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b'%PDF-1.4\n')
            tmp_path = tmp.name
        
        try:
            format_type = self.enhanced_handler.detect_format(tmp_path)
            assert format_type == 'pdf'
        finally:
            os.unlink(tmp_path)
    
    def test_format_detection_docx(self):
        """Test DOCX format detection"""
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            tmp.write(b'PK\x03\x04')  # ZIP header for DOCX
            tmp_path = tmp.name
        
        try:
            format_type = self.enhanced_handler.detect_format(tmp_path)
            assert format_type == 'docx'
        finally:
            os.unlink(tmp_path)
    
    def test_format_detection_pptx(self):
        """Test PPTX format detection"""
        with tempfile.NamedTemporaryFile(suffix='.pptx', delete=False) as tmp:
            tmp.write(b'PK\x03\x04')  # ZIP header for PPTX
            tmp_path = tmp.name
        
        try:
            format_type = self.enhanced_handler.detect_format(tmp_path)
            assert format_type == 'pptx'
        finally:
            os.unlink(tmp_path)
    
    def test_format_detection_html(self):
        """Test HTML format detection"""
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            tmp.write(b'<html><body>Test</body></html>')
            tmp_path = tmp.name
        
        try:
            format_type = self.enhanced_handler.detect_format(tmp_path)
            assert format_type == 'html'
        finally:
            os.unlink(tmp_path)
    
    def test_format_detection_image(self):
        """Test image format detection"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(b'\x89PNG\r\n\x1a\n')  # PNG header
            tmp_path = tmp.name
        
        try:
            format_type = self.enhanced_handler.detect_format(tmp_path)
            assert format_type == 'image'
        finally:
            os.unlink(tmp_path)
    
    def test_format_detection_markdown(self):
        """Test Markdown format detection"""
        with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as tmp:
            tmp.write(b'# Test Markdown\n\nContent')
            tmp_path = tmp.name
        
        try:
            format_type = self.enhanced_handler.detect_format(tmp_path)
            assert format_type == 'markdown'
        finally:
            os.unlink(tmp_path)
    
    def test_intelligent_routing_docling_processor(self):
        """Test routing to DoclingProcessor for supported formats"""
        # Mock DoclingProcessor response
        mock_result = ProcessingResult(
            format_type='pdf',
            file_path='/test/file.pdf',
            success=True,
            text='Test content',
            structure={},
            metadata={},
            processing_time=0.1,
            file_size=1024
        )
        self.mock_docling_processor.process_document.return_value = mock_result
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b'%PDF-1.4\n')
            tmp_path = tmp.name
        
        try:
            result = self.enhanced_handler.route_to_processor(tmp_path, 'pdf')
            
            assert result == mock_result
            self.mock_docling_processor.process_document.assert_called_once_with(
                tmp_path, 'pdf'
            )
        finally:
            os.unlink(tmp_path)
    
    def test_intelligent_routing_markdown_processor(self):
        """Test routing to existing MarkdownProcessor for Markdown files"""
        with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as tmp:
            tmp.write(b'# Test Markdown\n\nContent')
            tmp_path = tmp.name
        
        try:
            result = self.enhanced_handler.route_to_processor(tmp_path, 'markdown')
            
            assert result.format_type == 'markdown'
            assert result.success is True
            assert '# Test Markdown' in result.text
        finally:
            os.unlink(tmp_path)
    
    def test_batch_processing_mixed_formats(self):
        """Test batch processing with mixed file formats"""
        # Create test files
        test_files = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            # PDF file
            pdf_path = os.path.join(tmp_dir, 'test.pdf')
            with open(pdf_path, 'wb') as f:
                f.write(b'%PDF-1.4\n')
            test_files.append(pdf_path)
            
            # Markdown file
            md_path = os.path.join(tmp_dir, 'test.md')
            with open(md_path, 'w') as f:
                f.write('# Test Markdown\n\nContent')
            test_files.append(md_path)
            
            # Mock responses
            pdf_result = ProcessingResult(
                format_type='pdf',
                file_path=pdf_path,
                success=True,
                text='PDF content',
                structure={},
                metadata={},
                processing_time=0.1,
                file_size=1024
            )
            self.mock_docling_processor.process_document.return_value = pdf_result
            
            results = self.enhanced_handler.process_batch(test_files)
            
            assert len(results) == 2
            assert results[0].format_type == 'pdf'
            assert results[1].format_type == 'markdown'
            assert all(result.success for result in results)
    
    def test_error_handling_unsupported_formats(self):
        """Test error handling for unsupported file formats"""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp:
            tmp.write(b'unknown format')
            tmp_path = tmp.name
        
        try:
            format_type = self.enhanced_handler.detect_format(tmp_path)
            assert format_type == 'unknown'
            
            # Should raise error when trying to process unsupported format
            with pytest.raises(ValueError, match="Unsupported format"):
                self.enhanced_handler.route_to_processor(tmp_path, 'xyz')
        finally:
            os.unlink(tmp_path)
    
    def test_security_validation_new_formats(self):
        """Test security validation for new file formats"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b'%PDF-1.4\n')
            tmp_path = tmp.name
        
        try:
            # Test file size validation
            is_valid = self.enhanced_handler.validate_file_format(tmp_path, 'pdf')
            assert is_valid is True
            
            # Test with oversized file (mock)
            with patch('os.path.getsize', return_value=100 * 1024 * 1024):  # 100MB
                is_valid = self.enhanced_handler.validate_file_format(tmp_path, 'pdf')
                assert is_valid is False
        finally:
            os.unlink(tmp_path)
    
    def test_find_supported_files(self):
        """Test finding all supported files in a directory"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files
            test_files = {
                'test.pdf': b'%PDF-1.4\n',
                'test.docx': b'PK\x03\x04',
                'test.md': b'# Test Markdown',
                'test.txt': b'Plain text',  # Not supported
                'test.jpg': b'\xff\xd8\xff\xe0'  # JPEG header
            }
            
            for filename, content in test_files.items():
                file_path = os.path.join(tmp_dir, filename)
                with open(file_path, 'wb') as f:
                    f.write(content)
            
            # Mock find_markdown_files for backward compatibility
            self.mock_file_handler.find_markdown_files.return_value = [
                os.path.join(tmp_dir, 'test.md')
            ]
            
            supported_files = self.enhanced_handler.find_supported_files(tmp_dir)
            
            # Should find PDF, DOCX, MD, and JPG (4 files)
            assert len(supported_files) == 4
            file_formats = [f.format_type for f in supported_files]
            assert 'pdf' in file_formats
            assert 'docx' in file_formats
            assert 'markdown' in file_formats
            assert 'image' in file_formats
    
    def test_backward_compatibility_markdown_files(self):
        """Test backward compatibility with existing find_markdown_files"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test markdown files
            test_md = os.path.join(tmp_dir, 'test.md')
            test_markdown = os.path.join(tmp_dir, 'test.markdown')
            
            with open(test_md, 'w') as f:
                f.write('# Test MD')
            with open(test_markdown, 'w') as f:
                f.write('# Test Markdown')
            
            expected_files = [test_md, test_markdown]
            self.mock_file_handler.find_markdown_files.return_value = expected_files
            
            # Should still work through enhanced handler
            supported_files = self.enhanced_handler.find_supported_files(tmp_dir)
            
            # Verify find_markdown_files was called for backward compatibility
            self.mock_file_handler.find_markdown_files.assert_called_once_with(tmp_dir)
            
            # Should contain markdown files
            markdown_files = [f for f in supported_files if f.format_type == 'markdown']
            assert len(markdown_files) == 2
    
    def test_validation_input_types(self):
        """Test input validation for various methods"""
        # Test detect_format with invalid input
        with pytest.raises(ValidationError):
            self.enhanced_handler.detect_format(123)  # Not a string
        
        # Test route_to_processor with invalid input
        with pytest.raises(ValidationError):
            self.enhanced_handler.route_to_processor(None, 'pdf')
        
        # Test process_batch with invalid input
        with pytest.raises(ValidationError):
            self.enhanced_handler.process_batch("not_a_list")
    
    def test_file_info_dataclass(self):
        """Test FileInfo dataclass structure"""
        file_info = FileInfo(
            file_path='/test/file.pdf',
            format_type='pdf',
            file_size=1024,
            mime_type='application/pdf'
        )
        
        assert file_info.file_path == '/test/file.pdf'
        assert file_info.format_type == 'pdf'
        assert file_info.file_size == 1024
        assert file_info.mime_type == 'application/pdf'