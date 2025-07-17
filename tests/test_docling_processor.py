import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import json
import os
import tempfile

from langchain_core.documents import Document

try:
    from docling.datamodel.base_models import ConversionResult, ConversionStatus
    from docling.datamodel.document import DoclingDocument
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    ConversionResult = None
    ConversionStatus = None
    DoclingDocument = None

from src.chunkers.docling_processor import DoclingProcessor
from src.llm.providers.docling_provider import DoclingProvider
from src.llm.providers.base import LLMProviderError
from src.exceptions import ChunkingError


class TestDoclingProcessor:
    """Comprehensive test suite for DoclingProcessor following TDD methodology"""
    
    @pytest.fixture
    def mock_docling_provider(self):
        """Create a mock DoclingProvider for testing"""
        provider = Mock(spec=DoclingProvider)
        provider.provider_name = "docling"
        provider.model = "docling-v1"
        provider.is_available.return_value = True
        return provider
    
    @pytest.fixture
    def mock_conversion_result(self):
        """Create a mock ConversionResult for testing"""
        # Create mock status
        mock_status = Mock()
        mock_status.success = True
        
        # Create mock document
        mock_document = Mock()
        mock_document.export_to_markdown.return_value = "# Sample Document\n\nContent here"
        mock_document.export_to_html.return_value = "<h1>Sample Document</h1><p>Content here</p>"
        mock_document.export_to_json.return_value = '{"title": "Sample Document", "content": "Content here"}'
        
        # Create mock conversion result
        mock_result = Mock()
        mock_result.status = mock_status
        mock_result.document = mock_document
        
        return mock_result
    
    @pytest.fixture
    def docling_processor(self, mock_docling_provider):
        """Create DoclingProcessor instance with mocked provider"""
        return DoclingProcessor(mock_docling_provider)
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF processing result"""
        return {
            "text": "Sample PDF content with structured data",
            "structure": {
                "pages": [
                    {"page_number": 1, "text": "Page 1 content"},
                    {"page_number": 2, "text": "Page 2 content"}
                ],
                "headings": [
                    {"level": 1, "text": "Main Title", "page": 1},
                    {"level": 2, "text": "Subtitle", "page": 1}
                ]
            },
            "metadata": {
                "title": "Sample PDF Document",
                "author": "Test Author",
                "pages": 2,
                "format": "pdf"
            }
        }
    
    @pytest.fixture
    def sample_docx_content(self):
        """Sample DOCX processing result"""
        return {
            "text": "Sample DOCX content with hierarchy",
            "structure": {
                "headings": [
                    {"level": 1, "text": "Document Title", "style": "Heading 1"},
                    {"level": 2, "text": "Section 1", "style": "Heading 2"},
                    {"level": 3, "text": "Subsection 1.1", "style": "Heading 3"}
                ],
                "paragraphs": [
                    {"text": "First paragraph", "style": "Normal"},
                    {"text": "Second paragraph", "style": "Normal"}
                ]
            },
            "metadata": {
                "title": "Sample DOCX Document",
                "author": "Test Author",
                "format": "docx"
            }
        }
    
    def test_processor_initialization(self, mock_docling_provider):
        """Test DoclingProcessor initialization"""
        processor = DoclingProcessor(mock_docling_provider)
        
        assert processor.provider == mock_docling_provider
        assert processor.provider.provider_name == "docling"
        assert processor.supported_formats == ["pdf", "docx", "pptx", "html", "image"]
    
    def test_processor_initialization_invalid_provider(self):
        """Test DoclingProcessor initialization with invalid provider"""
        with pytest.raises(ValueError, match="DoclingProvider instance required"):
            DoclingProcessor(None)
    
    def test_process_document_unsupported_format(self, docling_processor):
        """Test processing with unsupported format"""
        # Create a temporary file to avoid FileNotFoundError
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(b"Sample text content")
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                docling_processor.process_document(tmp_path, "txt")
        finally:
            os.unlink(tmp_path)
    
    def test_process_document_nonexistent_file(self, docling_processor):
        """Test processing with non-existent file"""
        with pytest.raises(FileNotFoundError):
            docling_processor.process_document("/nonexistent/file.pdf", "pdf")
    
    def test_pdf_processing_basic(self, docling_processor, mock_conversion_result):
        """Test basic PDF processing functionality"""
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            # Mock the provider's process_document method
            docling_processor.provider.process_document.return_value = mock_conversion_result
            
            result = docling_processor.process_document(tmp_path, "pdf")
            
            # Result should be a ConversionResult
            assert result == mock_conversion_result
            assert result.status.success is True
            assert result.document.export_to_markdown() == "# Sample Document\n\nContent here"
            
            # Verify provider was called correctly
            docling_processor.provider.process_document.assert_called_once()
            
        finally:
            os.unlink(tmp_path)
    
    def test_docx_processing_with_hierarchy(self, docling_processor, mock_conversion_result):
        """Test DOCX processing with hierarchy preservation"""
        # Create a temporary DOCX file
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
            tmp_file.write(b"Sample DOCX content")
            tmp_path = tmp_file.name
        
        try:
            # Mock the provider's process_document method
            docling_processor.provider.process_document.return_value = mock_conversion_result
            
            result = docling_processor.process_document(tmp_path, "docx")
            
            # Result should be a ConversionResult
            assert result == mock_conversion_result
            assert result.status.success is True
            assert result.document.export_to_markdown() == "# Sample Document\n\nContent here"
            
        finally:
            os.unlink(tmp_path)
    
    def test_pptx_processing_with_slides(self, docling_processor, mock_conversion_result):
        """Test PPTX processing with slides and visual elements"""
        with tempfile.NamedTemporaryFile(suffix='.pptx', delete=False) as tmp_file:
            tmp_file.write(b"Sample PPTX content")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = mock_conversion_result
            
            result = docling_processor.process_document(tmp_path, "pptx")
            
            assert result == mock_conversion_result
            assert result.status.success is True
            assert result.document.export_to_markdown() == "# Sample Document\n\nContent here"
            
        finally:
            os.unlink(tmp_path)
    
    def test_html_processing_with_semantic_structure(self, docling_processor, mock_conversion_result):
        """Test HTML processing with semantic structure preservation"""
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            tmp_file.write(b"<html><body>Sample HTML content</body></html>")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = mock_conversion_result
            
            result = docling_processor.process_document(tmp_path, "html")
            
            assert result == mock_conversion_result
            assert result.status.success is True
            assert result.document.export_to_markdown() == "# Sample Document\n\nContent here"
            
        finally:
            os.unlink(tmp_path)
    
    def test_image_processing_with_vision_capabilities(self, docling_processor, mock_conversion_result):
        """Test image processing with vision capabilities"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_file.write(b"PNG image data")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = mock_conversion_result
            
            result = docling_processor.process_document(tmp_path, "image")
            
            assert result == mock_conversion_result
            assert result.status.success is True
            assert result.document.export_to_markdown() == "# Sample Document\n\nContent here"
            
        finally:
            os.unlink(tmp_path)
    
    def test_error_handling_graceful(self, docling_processor):
        """Test graceful error handling and reporting"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            # Mock provider to raise an error
            docling_processor.provider.process_document.side_effect = LLMProviderError("API Error")
            
            # Should raise the LLMProviderError
            with pytest.raises(LLMProviderError) as exc_info:
                docling_processor.process_document(tmp_path, "pdf")
            
            assert str(exc_info.value) == "API Error"
            
        finally:
            os.unlink(tmp_path)
    
    def test_error_handling_network_timeout(self, docling_processor):
        """Test handling of network timeout errors"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            # Mock provider to raise a timeout error
            docling_processor.provider.process_document.side_effect = OSError("Connection timeout")
            
            # Should raise ChunkingError (wrapping the OSError)
            with pytest.raises(ChunkingError) as exc_info:
                docling_processor.process_document(tmp_path, "pdf")
            
            assert "Connection timeout" in str(exc_info.value)
            
        finally:
            os.unlink(tmp_path)
    
    def test_performance_monitoring_integration(self, docling_processor, mock_conversion_result):
        """Test performance monitoring integration"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = mock_conversion_result
            
            result = docling_processor.process_document(tmp_path, "pdf")
            
            # Verify result is a ConversionResult
            assert result == mock_conversion_result
            assert result.status.success is True
            
        finally:
            os.unlink(tmp_path)
    
    def test_process_document_with_options(self, docling_processor, mock_conversion_result):
        """Test process_document with custom options"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = mock_conversion_result
            
            options = {
                "extract_text": True,
                "extract_structure": True,
                "extract_metadata": True,
                "custom_option": "value"
            }
            
            result = docling_processor.process_document(tmp_path, "pdf", **options)
            
            assert result == mock_conversion_result
            assert result.status.success is True
            assert result.document.export_to_markdown() == "# Sample Document\n\nContent here"
            
            # Verify options were passed to provider
            call_args = docling_processor.provider.process_document.call_args
            assert call_args[1]["extract_text"] is True
            assert call_args[1]["extract_structure"] is True
            assert call_args[1]["extract_metadata"] is True
            assert call_args[1]["custom_option"] == "value"
            
        finally:
            os.unlink(tmp_path)
    
    def test_auto_format_detection(self, docling_processor, mock_conversion_result):
        """Test automatic format detection"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = mock_conversion_result
            
            # Test auto-detection (should detect PDF from extension)
            result = docling_processor.process_document(tmp_path, "auto")
            
            assert result == mock_conversion_result
            assert result.status.success is True
            
        finally:
            os.unlink(tmp_path)
    
    def test_process_document_empty_result(self, docling_processor):
        """Test processing with empty result from provider"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            # Mock provider to return empty conversion result
            mock_status = Mock()
            mock_status.success = True
            
            mock_document = Mock()
            mock_document.export_to_markdown.return_value = ""
            
            mock_result = Mock()
            mock_result.status = mock_status
            mock_result.document = mock_document
            
            docling_processor.provider.process_document.return_value = mock_result
            
            result = docling_processor.process_document(tmp_path, "pdf")
            
            assert result == mock_result
            assert result.status.success is True
            assert result.document.export_to_markdown() == ""
            
        finally:
            os.unlink(tmp_path)
    
    def test_large_file_processing(self, docling_processor, mock_conversion_result):
        """Test processing of large files"""
        # Create a larger temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            # Write 1MB of data
            tmp_file.write(b"Sample PDF content" * 60000)
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = mock_conversion_result
            
            result = docling_processor.process_document(tmp_path, "pdf")
            
            assert result == mock_conversion_result
            assert result.status.success is True
            
        finally:
            os.unlink(tmp_path)
    
    def test_get_supported_formats(self, docling_processor):
        """Test getting supported formats"""
        formats = docling_processor.get_supported_formats()
        expected_formats = ["pdf", "docx", "pptx", "html", "image"]
        assert formats == expected_formats
        
        # Verify it returns a copy, not reference
        formats.append("new_format")
        assert len(docling_processor.get_supported_formats()) == 5
    
    def test_is_format_supported(self, docling_processor):
        """Test format support checking"""
        assert docling_processor.is_format_supported("pdf") is True
        assert docling_processor.is_format_supported("docx") is True
        assert docling_processor.is_format_supported("txt") is False
        assert docling_processor.is_format_supported("unknown") is False
    
    def test_get_provider_info(self, docling_processor):
        """Test getting provider information"""
        info = docling_processor.get_provider_info()
        
        assert info["provider_name"] == "docling"
        assert info["model"] == "docling-v1"
        assert info["is_available"] is True
        assert info["supported_formats"] == ["pdf", "docx", "pptx", "html", "image"]
    
    def test_format_specific_processing_methods(self, docling_processor, mock_conversion_result):
        """Test format-specific processing methods"""
        # Create different mock results for each format
        def create_mock_result(text):
            mock_status = Mock()
            mock_status.success = True
            
            mock_document = Mock()
            mock_document.export_to_markdown.return_value = text
            
            mock_result = Mock()
            mock_result.status = mock_status
            mock_result.document = mock_document
            
            return mock_result
        
        # Test _process_pdf
        pdf_content = b"PDF content"
        docling_processor.provider.process_document.return_value = create_mock_result("PDF result")
        
        result = docling_processor._process_pdf(pdf_content)
        assert result.document.export_to_markdown() == "PDF result"
        
        # Verify correct parameters were passed
        call_args = docling_processor.provider.process_document.call_args
        assert call_args[1]["document_type"] == "pdf"
        assert call_args[1]["extract_text"] is True
        assert call_args[1]["extract_structure"] is True
        assert call_args[1]["extract_metadata"] is True
        
        # Test _process_docx
        docx_content = b"DOCX content"
        docling_processor.provider.process_document.return_value = create_mock_result("DOCX result")
        
        result = docling_processor._process_docx(docx_content)
        assert result.document.export_to_markdown() == "DOCX result"
        
        call_args = docling_processor.provider.process_document.call_args
        assert call_args[1]["document_type"] == "docx"
        assert call_args[1]["preserve_hierarchy"] is True
        assert call_args[1]["extract_styles"] is True
        
        # Test _process_pptx
        pptx_content = b"PPTX content"
        docling_processor.provider.process_document.return_value = create_mock_result("PPTX result")
        
        result = docling_processor._process_pptx(pptx_content)
        assert result.document.export_to_markdown() == "PPTX result"
        
        call_args = docling_processor.provider.process_document.call_args
        assert call_args[1]["document_type"] == "pptx"
        assert call_args[1]["extract_slides"] is True
        assert call_args[1]["extract_visuals"] is True
        
        # Test _process_html
        html_content = "<html>HTML content</html>"
        docling_processor.provider.process_document.return_value = create_mock_result("HTML result")
        
        result = docling_processor._process_html(html_content)
        assert result.document.export_to_markdown() == "HTML result"
        
        call_args = docling_processor.provider.process_document.call_args
        assert call_args[1]["document_type"] == "html"
        assert call_args[1]["preserve_semantic_structure"] is True
        assert call_args[1]["extract_links"] is True
        
        # Test _process_image
        image_content = b"Image content"
        docling_processor.provider.process_document.return_value = create_mock_result("Image result")
        
        result = docling_processor._process_image(image_content)
        assert result.document.export_to_markdown() == "Image result"
        
        call_args = docling_processor.provider.process_document.call_args
        assert call_args[1]["document_type"] == "image"
        assert call_args[1]["use_vision"] is True
        assert call_args[1]["extract_text"] is True
        assert call_args[1]["analyze_layout"] is True
    
    def test_detect_format_by_extension(self, docling_processor):
        """Test format detection by file extension"""
        # Test PDF detection
        assert docling_processor._detect_format("test.pdf") == "pdf"
        assert docling_processor._detect_format("test.PDF") == "pdf"
        
        # Test DOCX detection
        assert docling_processor._detect_format("test.docx") == "docx"
        assert docling_processor._detect_format("test.DOCX") == "docx"
        
        # Test PPTX detection
        assert docling_processor._detect_format("test.pptx") == "pptx"
        
        # Test HTML detection
        assert docling_processor._detect_format("test.html") == "html"
        assert docling_processor._detect_format("test.htm") == "html"
        
        # Test image detection
        assert docling_processor._detect_format("test.png") == "image"
        assert docling_processor._detect_format("test.jpg") == "image"
        assert docling_processor._detect_format("test.jpeg") == "image"
        assert docling_processor._detect_format("test.gif") == "image"
        assert docling_processor._detect_format("test.bmp") == "image"
        assert docling_processor._detect_format("test.tiff") == "image"
        assert docling_processor._detect_format("test.tif") == "image"
        
        # Test unknown format defaults to PDF
        assert docling_processor._detect_format("test.unknown") == "pdf"
    
    def test_format_detection_with_mime_types(self, docling_processor):
        """Test format detection using MIME types"""
        # Create temporary files with different extensions but check MIME type fallback
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Test that MIME type detection works as fallback
            detected = docling_processor._detect_format(tmp_path)
            assert detected == "pdf"
            
        finally:
            os.unlink(tmp_path)
    
    def test_mime_type_detection_fallback(self, docling_processor):
        """Test MIME type detection fallback for unknown extensions"""
        # Create a file with unknown extension but known MIME type
        with tempfile.NamedTemporaryFile(suffix='.unknown', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Test that it falls back to PDF for unknown files
            detected = docling_processor._detect_format(tmp_path)
            assert detected == "pdf"
            
        finally:
            os.unlink(tmp_path)
    
    def test_mime_type_detection_for_known_types(self, docling_processor):
        """Test MIME type detection for known MIME types"""
        # Mock mimetypes.guess_type to return known MIME types
        import mimetypes
        
        # Test with a file that has no extension but has a known MIME type
        original_guess_type = mimetypes.guess_type
        
        def mock_guess_type(path):
            if "pdf_mime" in path:
                return ("application/pdf", None)
            elif "docx_mime" in path:
                return ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", None)
            elif "html_mime" in path:
                return ("text/html", None)
            elif "image_mime" in path:
                return ("image/png", None)
            return (None, None)
        
        mimetypes.guess_type = mock_guess_type
        
        try:
            # Test PDF MIME type detection
            assert docling_processor._detect_format("test_pdf_mime") == "pdf"
            
            # Test DOCX MIME type detection
            assert docling_processor._detect_format("test_docx_mime") == "docx"
            
            # Test HTML MIME type detection  
            assert docling_processor._detect_format("test_html_mime") == "html"
            
            # Test Image MIME type detection
            assert docling_processor._detect_format("test_image_mime") == "image"
            
        finally:
            mimetypes.guess_type = original_guess_type
    
    def test_export_to_markdown(self, docling_processor, mock_conversion_result):
        """Test export to markdown functionality"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = mock_conversion_result
            
            with patch('os.path.exists', return_value=True):
                markdown_content = docling_processor.export_to_markdown(tmp_path)
                
                assert markdown_content == "# Sample Document\n\nContent here"
                mock_conversion_result.document.export_to_markdown.assert_called_once()
                
        finally:
            os.unlink(tmp_path)
    
    def test_export_to_html(self, docling_processor, mock_conversion_result):
        """Test export to HTML functionality"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = mock_conversion_result
            
            with patch('os.path.exists', return_value=True):
                html_content = docling_processor.export_to_html(tmp_path)
                
                assert html_content == "<h1>Sample Document</h1><p>Content here</p>"
                mock_conversion_result.document.export_to_html.assert_called_once()
                
        finally:
            os.unlink(tmp_path)
    
    def test_export_to_json(self, docling_processor, mock_conversion_result):
        """Test export to JSON functionality"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = mock_conversion_result
            
            with patch('os.path.exists', return_value=True):
                json_content = docling_processor.export_to_json(tmp_path)
                
                assert json_content == '{"title": "Sample Document", "content": "Content here"}'
                mock_conversion_result.document.export_to_json.assert_called_once()
                
        finally:
            os.unlink(tmp_path)
    
    def test_export_methods_with_failed_conversion(self, docling_processor):
        """Test export methods when conversion fails"""
        # Create mock failed conversion result
        mock_status = Mock()
        mock_status.success = False
        mock_status.__str__ = Mock(return_value="Conversion failed")
        
        mock_result = Mock()
        mock_result.status = mock_status
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = mock_result
            
            with patch('os.path.exists', return_value=True):
                # Test markdown export with failed conversion
                markdown_content = docling_processor.export_to_markdown(tmp_path)
                assert "Error processing document" in markdown_content
                
                # Test HTML export with failed conversion
                html_content = docling_processor.export_to_html(tmp_path)
                assert "Error processing document" in html_content
                
                # Test JSON export with failed conversion
                json_content = docling_processor.export_to_json(tmp_path)
                assert "Error processing document" in json_content
                
        finally:
            os.unlink(tmp_path)