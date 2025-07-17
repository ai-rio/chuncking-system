import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import json
import os
import tempfile

from src.chunkers.docling_processor import DoclingProcessor, ProcessingResult
from src.llm.providers.docling_provider import DoclingProvider
from src.llm.providers.base import LLMProviderError


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
    
    def test_pdf_processing_basic(self, docling_processor, sample_pdf_content):
        """Test basic PDF processing functionality"""
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            # Mock the provider's process_document method
            docling_processor.provider.process_document.return_value = sample_pdf_content
            
            result = docling_processor.process_document(tmp_path, "pdf")
            
            assert isinstance(result, ProcessingResult)
            assert result.format_type == "pdf"
            assert result.file_path == tmp_path
            assert result.success is True
            assert result.text == "Sample PDF content with structured data"
            assert result.metadata["format"] == "pdf"
            assert result.metadata["pages"] == 2
            assert len(result.structure["pages"]) == 2
            assert len(result.structure["headings"]) == 2
            
            # Verify provider was called correctly
            docling_processor.provider.process_document.assert_called_once()
            call_args = docling_processor.provider.process_document.call_args
            assert call_args[1]["document_type"] == "pdf"
            
        finally:
            os.unlink(tmp_path)
    
    def test_docx_processing_with_hierarchy(self, docling_processor, sample_docx_content):
        """Test DOCX processing with hierarchy preservation"""
        # Create a temporary DOCX file
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
            tmp_file.write(b"Sample DOCX content")
            tmp_path = tmp_file.name
        
        try:
            # Mock the provider's process_document method
            docling_processor.provider.process_document.return_value = sample_docx_content
            
            result = docling_processor.process_document(tmp_path, "docx")
            
            assert isinstance(result, ProcessingResult)
            assert result.format_type == "docx"
            assert result.success is True
            assert result.text == "Sample DOCX content with hierarchy"
            assert result.metadata["format"] == "docx"
            
            # Verify hierarchy preservation
            headings = result.structure["headings"]
            assert len(headings) == 3
            assert headings[0]["level"] == 1
            assert headings[1]["level"] == 2
            assert headings[2]["level"] == 3
            assert headings[0]["text"] == "Document Title"
            
        finally:
            os.unlink(tmp_path)
    
    def test_pptx_processing_with_slides(self, docling_processor):
        """Test PPTX processing with slides and visual elements"""
        sample_pptx_content = {
            "text": "Sample PPTX content with slides",
            "structure": {
                "slides": [
                    {
                        "slide_number": 1,
                        "title": "Slide 1 Title",
                        "content": "Slide 1 content",
                        "layout": "title_slide"
                    },
                    {
                        "slide_number": 2,
                        "title": "Slide 2 Title",
                        "content": "Slide 2 content",
                        "layout": "content_slide"
                    }
                ],
                "visual_elements": [
                    {"type": "image", "slide": 1, "description": "Company logo"},
                    {"type": "chart", "slide": 2, "description": "Sales data"}
                ]
            },
            "metadata": {
                "title": "Sample PPTX Presentation",
                "author": "Test Author",
                "slides": 2,
                "format": "pptx"
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pptx', delete=False) as tmp_file:
            tmp_file.write(b"Sample PPTX content")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = sample_pptx_content
            
            result = docling_processor.process_document(tmp_path, "pptx")
            
            assert result.format_type == "pptx"
            assert result.success is True
            assert result.metadata["slides"] == 2
            assert len(result.structure["slides"]) == 2
            assert len(result.structure["visual_elements"]) == 2
            
        finally:
            os.unlink(tmp_path)
    
    def test_html_processing_with_semantic_structure(self, docling_processor):
        """Test HTML processing with semantic structure preservation"""
        sample_html_content = {
            "text": "Sample HTML content with semantic structure",
            "structure": {
                "semantic_elements": [
                    {"tag": "header", "text": "Page Header"},
                    {"tag": "nav", "text": "Navigation Menu"},
                    {"tag": "main", "text": "Main Content"},
                    {"tag": "footer", "text": "Page Footer"}
                ],
                "headings": [
                    {"level": 1, "text": "Main Heading", "tag": "h1"},
                    {"level": 2, "text": "Section Heading", "tag": "h2"}
                ]
            },
            "metadata": {
                "title": "Sample HTML Document",
                "format": "html"
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            tmp_file.write(b"<html><body>Sample HTML content</body></html>")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = sample_html_content
            
            result = docling_processor.process_document(tmp_path, "html")
            
            assert result.format_type == "html"
            assert result.success is True
            assert len(result.structure["semantic_elements"]) == 4
            assert len(result.structure["headings"]) == 2
            
        finally:
            os.unlink(tmp_path)
    
    def test_image_processing_with_vision_capabilities(self, docling_processor):
        """Test image processing with vision capabilities"""
        sample_image_content = {
            "text": "Extracted text from image using vision capabilities",
            "structure": {
                "visual_elements": [
                    {"type": "text_block", "text": "Header Text", "confidence": 0.95},
                    {"type": "text_block", "text": "Body Text", "confidence": 0.87},
                    {"type": "image", "description": "Chart or diagram", "confidence": 0.92}
                ],
                "layout": {
                    "orientation": "portrait",
                    "text_regions": 2,
                    "image_regions": 1
                }
            },
            "metadata": {
                "format": "image",
                "image_type": "png",
                "dimensions": {"width": 800, "height": 600}
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_file.write(b"PNG image data")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = sample_image_content
            
            result = docling_processor.process_document(tmp_path, "image")
            
            assert result.format_type == "image"
            assert result.success is True
            assert len(result.structure["visual_elements"]) == 3
            assert result.metadata["image_type"] == "png"
            
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
            
            result = docling_processor.process_document(tmp_path, "pdf")
            
            assert result.success is False
            assert result.error_message == "API Error"
            assert result.text == ""
            assert result.structure == {}
            assert result.metadata == {}
            
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
            
            result = docling_processor.process_document(tmp_path, "pdf")
            
            assert result.success is False
            assert "Connection timeout" in result.error_message
            
        finally:
            os.unlink(tmp_path)
    
    def test_performance_monitoring_integration(self, docling_processor, sample_pdf_content):
        """Test performance monitoring integration"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = sample_pdf_content
            
            result = docling_processor.process_document(tmp_path, "pdf")
            
            # Verify performance metrics are captured
            assert hasattr(result, 'processing_time')
            assert result.processing_time > 0
            assert hasattr(result, 'file_size')
            assert result.file_size > 0
            
        finally:
            os.unlink(tmp_path)
    
    def test_process_document_with_options(self, docling_processor, sample_pdf_content):
        """Test process_document with custom options"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = sample_pdf_content
            
            options = {
                "extract_text": True,
                "extract_structure": True,
                "extract_metadata": True,
                "custom_option": "value"
            }
            
            result = docling_processor.process_document(tmp_path, "pdf", **options)
            
            assert result.success is True
            
            # Verify options were passed to provider
            call_args = docling_processor.provider.process_document.call_args
            assert call_args[1]["extract_text"] is True
            assert call_args[1]["extract_structure"] is True
            assert call_args[1]["extract_metadata"] is True
            assert call_args[1]["custom_option"] == "value"
            
        finally:
            os.unlink(tmp_path)
    
    def test_auto_format_detection(self, docling_processor, sample_pdf_content):
        """Test automatic format detection"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = sample_pdf_content
            
            # Test auto-detection (should detect PDF from extension)
            result = docling_processor.process_document(tmp_path, "auto")
            
            assert result.format_type == "pdf"
            assert result.success is True
            
        finally:
            os.unlink(tmp_path)
    
    def test_process_document_empty_result(self, docling_processor):
        """Test processing with empty result from provider"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"Sample PDF content")
            tmp_path = tmp_file.name
        
        try:
            # Mock provider to return empty result
            docling_processor.provider.process_document.return_value = {}
            
            result = docling_processor.process_document(tmp_path, "pdf")
            
            assert result.success is True
            assert result.text == ""
            assert result.structure == {}
            assert result.metadata == {}
            
        finally:
            os.unlink(tmp_path)
    
    def test_large_file_processing(self, docling_processor, sample_pdf_content):
        """Test processing of large files"""
        # Create a larger temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            # Write 1MB of data
            tmp_file.write(b"Sample PDF content" * 60000)
            tmp_path = tmp_file.name
        
        try:
            docling_processor.provider.process_document.return_value = sample_pdf_content
            
            result = docling_processor.process_document(tmp_path, "pdf")
            
            assert result.success is True
            assert result.file_size > 1000000  # > 1MB
            
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
    
    def test_format_specific_processing_methods(self, docling_processor):
        """Test format-specific processing methods"""
        # Test _process_pdf
        pdf_content = b"PDF content"
        docling_processor.provider.process_document.return_value = {"text": "PDF result"}
        
        result = docling_processor._process_pdf(pdf_content)
        assert result["text"] == "PDF result"
        
        # Verify correct parameters were passed
        call_args = docling_processor.provider.process_document.call_args
        assert call_args[1]["document_type"] == "pdf"
        assert call_args[1]["extract_text"] is True
        assert call_args[1]["extract_structure"] is True
        assert call_args[1]["extract_metadata"] is True
        
        # Test _process_docx
        docx_content = b"DOCX content"
        docling_processor.provider.process_document.return_value = {"text": "DOCX result"}
        
        result = docling_processor._process_docx(docx_content)
        assert result["text"] == "DOCX result"
        
        call_args = docling_processor.provider.process_document.call_args
        assert call_args[1]["document_type"] == "docx"
        assert call_args[1]["preserve_hierarchy"] is True
        assert call_args[1]["extract_styles"] is True
        
        # Test _process_pptx
        pptx_content = b"PPTX content"
        docling_processor.provider.process_document.return_value = {"text": "PPTX result"}
        
        result = docling_processor._process_pptx(pptx_content)
        assert result["text"] == "PPTX result"
        
        call_args = docling_processor.provider.process_document.call_args
        assert call_args[1]["document_type"] == "pptx"
        assert call_args[1]["extract_slides"] is True
        assert call_args[1]["extract_visuals"] is True
        
        # Test _process_html
        html_content = "<html>HTML content</html>"
        docling_processor.provider.process_document.return_value = {"text": "HTML result"}
        
        result = docling_processor._process_html(html_content)
        assert result["text"] == "HTML result"
        
        call_args = docling_processor.provider.process_document.call_args
        assert call_args[1]["document_type"] == "html"
        assert call_args[1]["preserve_semantic_structure"] is True
        assert call_args[1]["extract_links"] is True
        
        # Test _process_image
        image_content = b"Image content"
        docling_processor.provider.process_document.return_value = {"text": "Image result"}
        
        result = docling_processor._process_image(image_content)
        assert result["text"] == "Image result"
        
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