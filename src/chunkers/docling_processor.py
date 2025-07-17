import os
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import mimetypes

from src.llm.providers.docling_provider import DoclingProvider
from src.llm.providers.base import LLMProviderError


@dataclass
class ProcessingResult:
    """Result of document processing operation"""
    format_type: str
    file_path: str
    success: bool
    text: str
    structure: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float
    file_size: int
    error_message: str = ""


class DoclingProcessor:
    """
    DoclingProcessor component for multi-format document processing.
    
    Handles PDF, DOCX, PPTX, HTML, and image files using Docling's AI capabilities.
    Follows existing patterns from MarkdownProcessor while leveraging DoclingProvider.
    """
    
    def __init__(self, docling_provider: DoclingProvider):
        """
        Initialize DoclingProcessor.
        
        Args:
            docling_provider: DoclingProvider instance for API interactions
            
        Raises:
            ValueError: If docling_provider is None or invalid
        """
        if not isinstance(docling_provider, DoclingProvider):
            raise ValueError("DoclingProvider instance required")
        
        self.provider = docling_provider
        self.supported_formats = ["pdf", "docx", "pptx", "html", "image"]
        
        # Format-specific MIME type mappings
        self.mime_mappings = {
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
            "text/html": "html",
            "image/jpeg": "image",
            "image/png": "image",
            "image/gif": "image",
            "image/bmp": "image",
            "image/tiff": "image"
        }
    
    def process_document(
        self, 
        file_path: str, 
        format_type: str = "auto",
        **kwargs
    ) -> ProcessingResult:
        """
        Process document using Docling capabilities.
        
        Args:
            file_path: Path to the document file
            format_type: Document format ("pdf", "docx", "pptx", "html", "image", "auto")
            **kwargs: Additional processing options
            
        Returns:
            ProcessingResult containing processed document data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is not supported
        """
        start_time = time.time()
        
        # Validate file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file size for performance monitoring
        file_size = os.path.getsize(file_path)
        
        # Auto-detect format if needed
        if format_type == "auto":
            format_type = self._detect_format(file_path)
        
        # Validate format
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}")
        
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Process document using provider
            result = self.provider.process_document(
                document_content=file_content,
                document_type=format_type,
                **kwargs
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract and structure the result
            return ProcessingResult(
                format_type=format_type,
                file_path=file_path,
                success=True,
                text=result.get("text", ""),
                structure=result.get("structure", {}),
                metadata=result.get("metadata", {}),
                processing_time=processing_time,
                file_size=file_size
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)
            
            return ProcessingResult(
                format_type=format_type,
                file_path=file_path,
                success=False,
                text="",
                structure={},
                metadata={},
                processing_time=processing_time,
                file_size=file_size,
                error_message=error_message
            )
    
    def _detect_format(self, file_path: str) -> str:
        """
        Auto-detect document format from file extension and MIME type.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Detected format string
        """
        # Try extension first
        _, ext = os.path.splitext(file_path.lower())
        ext_mappings = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".pptx": "pptx",
            ".html": "html",
            ".htm": "html",
            ".png": "image",
            ".jpg": "image",
            ".jpeg": "image",
            ".gif": "image",
            ".bmp": "image",
            ".tiff": "image",
            ".tif": "image"
        }
        
        if ext in ext_mappings:
            return ext_mappings[ext]
        
        # Fallback to MIME type detection
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type in self.mime_mappings:
            return self.mime_mappings[mime_type]
        
        # Default to PDF if uncertain
        return "pdf"
    
    def _process_pdf(self, content: bytes, **kwargs) -> Dict[str, Any]:
        """
        Process PDF document content.
        
        Args:
            content: PDF file content as bytes
            **kwargs: Additional processing options
            
        Returns:
            Processed PDF data with text, structure, and metadata
        """
        return self.provider.process_document(
            document_content=content,
            document_type="pdf",
            extract_text=kwargs.get("extract_text", True),
            extract_structure=kwargs.get("extract_structure", True),
            extract_metadata=kwargs.get("extract_metadata", True),
            **kwargs
        )
    
    def _process_docx(self, content: bytes, **kwargs) -> Dict[str, Any]:
        """
        Process DOCX document content with hierarchy preservation.
        
        Args:
            content: DOCX file content as bytes
            **kwargs: Additional processing options
            
        Returns:
            Processed DOCX data with hierarchy and structure
        """
        return self.provider.process_document(
            document_content=content,
            document_type="docx",
            preserve_hierarchy=kwargs.get("preserve_hierarchy", True),
            extract_styles=kwargs.get("extract_styles", True),
            **kwargs
        )
    
    def _process_pptx(self, content: bytes, **kwargs) -> Dict[str, Any]:
        """
        Process PPTX presentation content with slides and visual elements.
        
        Args:
            content: PPTX file content as bytes
            **kwargs: Additional processing options
            
        Returns:
            Processed PPTX data with slides and visual elements
        """
        return self.provider.process_document(
            document_content=content,
            document_type="pptx",
            extract_slides=kwargs.get("extract_slides", True),
            extract_visuals=kwargs.get("extract_visuals", True),
            **kwargs
        )
    
    def _process_html(self, content: str, **kwargs) -> Dict[str, Any]:
        """
        Process HTML content with semantic structure preservation.
        
        Args:
            content: HTML content as string
            **kwargs: Additional processing options
            
        Returns:
            Processed HTML data with semantic structure
        """
        return self.provider.process_document(
            document_content=content,
            document_type="html",
            preserve_semantic_structure=kwargs.get("preserve_semantic_structure", True),
            extract_links=kwargs.get("extract_links", True),
            **kwargs
        )
    
    def _process_image(self, content: bytes, **kwargs) -> Dict[str, Any]:
        """
        Process image content with vision capabilities.
        
        Args:
            content: Image file content as bytes
            **kwargs: Additional processing options
            
        Returns:
            Processed image data with extracted text and visual analysis
        """
        return self.provider.process_document(
            document_content=content,
            document_type="image",
            use_vision=kwargs.get("use_vision", True),
            extract_text=kwargs.get("extract_text", True),
            analyze_layout=kwargs.get("analyze_layout", True),
            **kwargs
        )
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported document formats.
        
        Returns:
            List of supported format strings
        """
        return self.supported_formats.copy()
    
    def is_format_supported(self, format_type: str) -> bool:
        """
        Check if a format is supported.
        
        Args:
            format_type: Format to check
            
        Returns:
            True if format is supported, False otherwise
        """
        return format_type in self.supported_formats
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the underlying provider.
        
        Returns:
            Provider information dictionary
        """
        return {
            "provider_name": self.provider.provider_name,
            "model": self.provider.model,
            "is_available": self.provider.is_available(),
            "supported_formats": self.supported_formats
        }