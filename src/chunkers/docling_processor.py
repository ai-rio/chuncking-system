import os
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import mimetypes

from langchain_core.documents import Document

try:
    from docling.datamodel.base_models import ConversionResult
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    ConversionResult = None

from src.llm.providers.docling_provider import DoclingProvider
from src.llm.providers.base import LLMProviderError
from src.exceptions import ChunkingError


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
    
    Handles PDF, DOCX, PPTX, HTML, and image files using DoclingProvider.
    """
    
    def __init__(self, provider: 'DoclingProvider'):
        """
        Initialize DoclingProcessor with a DoclingProvider.
        
        Args:
            provider: DoclingProvider instance for document processing
            
        Raises:
            ValueError: If provider is None or invalid
        """
        if provider is None:
            raise ValueError("DoclingProvider instance required")
        self.provider = provider
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
    ) -> List[Document]:
        """
        Process document using DoclingProvider and return Document chunks.
        
        Args:
            file_path: Path to the document file
            format_type: Document format ("pdf", "docx", "pptx", "html", "image", "auto")
            **kwargs: Additional processing options
            
        Returns:
            List of Document chunks
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is not supported
            ChunkingError: If document processing fails
        """
        # Validate file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect format if needed
        if format_type == "auto":
            format_type = self._detect_format(file_path)
        
        # Validate format
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}")
        
        try:
            # Call provider to process the document
            conversion_result = self.provider.process_document(file_path, document_type=format_type, **kwargs)
            
            # Convert ConversionResult to Document chunks
            return self._convert_to_document_chunks(conversion_result, file_path, format_type)
            
        except LLMProviderError as e:
            # Re-raise provider errors for ProductionPipeline to handle
            raise e
        except Exception as e:
            # Re-raise other exceptions for ProductionPipeline to handle
            raise ChunkingError(f"Document processing failed: {str(e)}") from e
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the provider.
        
        Returns:
            Provider information dictionary
        """
        base_info = {
            "provider_name": "docling",
            "model": "docling-v1",
            "is_available": True,
            "supported_formats": self.supported_formats
        }
        
        # Add provider-specific info if available
        if hasattr(self.provider, 'get_provider_info'):
            provider_info = self.provider.get_provider_info()
            base_info.update(provider_info)
            
        return base_info
    
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
    
    def export_to_markdown(self, file_path: str) -> str:
        """
        Export document to Markdown format.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Markdown representation of the document
        """
        try:
            result = self.process_document(file_path)
            if result.status.success:
                # Use DoclingDocument's export_to_markdown method
                return result.document.export_to_markdown()
            else:
                return f"Error processing document: {result.status}"
        except Exception as e:
            return f"Error exporting to markdown: {str(e)}"
    
    def export_to_html(self, file_path: str) -> str:
        """
        Export document to HTML format.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            HTML representation of the document
        """
        try:
            result = self.process_document(file_path)
            if result.status.success:
                # Use DoclingDocument's export_to_html method
                return result.document.export_to_html()
            else:
                return f"Error processing document: {result.status}"
        except Exception as e:
            return f"Error exporting to HTML: {str(e)}"
    
    def export_to_json(self, file_path: str) -> str:
        """
        Export document to JSON format.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            JSON representation of the document
        """
        try:
            result = self.process_document(file_path)
            if result.status.success:
                # Use DoclingDocument's export_to_json method
                return result.document.export_to_json()
            else:
                return f"Error processing document: {result.status}"
        except Exception as e:
            return f"Error exporting to JSON: {str(e)}"
    
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
    
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get information about the processor configuration.
        
        Returns:
            Dictionary containing processor information
        """
        return {
            "processor_type": "DoclingProcessor",
            "provider_available": hasattr(self, 'provider') and self.provider is not None,
            "supported_formats": self.supported_formats,
            "provider_info": self.provider.get_provider_info() if hasattr(self.provider, 'get_provider_info') else {},
            "version": "1.0.0"
        }
    
    def _convert_to_document_chunks(self, conversion_result, file_path: str, format_type: str) -> List[Document]:
        """
        Convert ConversionResult to Document chunks.
        
        Args:
            conversion_result: ConversionResult from DoclingProvider
            file_path: Path to the source file
            format_type: Document format type
            
        Returns:
            List of Document chunks
        """
        chunks = []
        
        # Handle mock provider case (returns dict instead of ConversionResult)
        if isinstance(conversion_result, dict):
            # Mock provider returns a dict with text, structure, metadata
            content = conversion_result.get('text', '')
            metadata = conversion_result.get('metadata', {})
            metadata.update({
                'source': file_path,
                'format': format_type,
                'chunk_index': 0
            })
            
            chunks.append(Document(
                page_content=content,
                metadata=metadata
            ))
            return chunks
        
        # Handle real ConversionResult
        if hasattr(conversion_result, 'document') and conversion_result.document:
            # Extract text content from the document
            if hasattr(conversion_result.document, 'export_to_markdown'):
                content = conversion_result.document.export_to_markdown()
            elif hasattr(conversion_result.document, 'text'):
                content = conversion_result.document.text
            else:
                content = str(conversion_result.document)
            
            # Create metadata
            metadata = {
                'source': file_path,
                'format': format_type,
                'chunk_index': 0,
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
            
            # Add any additional metadata from conversion result
            if hasattr(conversion_result, 'metadata'):
                metadata.update(conversion_result.metadata)
            
            chunks.append(Document(
                page_content=content,
                metadata=metadata
            ))
        else:
            # Fallback: create empty document with error info
            chunks.append(Document(
                page_content="",
                metadata={
                    'source': file_path,
                    'format': format_type,
                    'chunk_index': 0,
                    'error': 'Failed to extract content from ConversionResult'
                }
            ))
        
        return chunks
    
    def _process_pdf(self, file_path: str) -> 'ConversionResult':
        """
        Process PDF file using the provider.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            ConversionResult from docling
        """
        return self.provider.process_document(
            file_path,
            document_type="pdf",
            extract_text=True,
            extract_structure=True,
            extract_metadata=True
        )
    
    def _process_docx(self, file_path: str) -> 'ConversionResult':
        """
        Process DOCX file using the provider.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            ConversionResult from docling
        """
        return self.provider.process_document(
            file_path,
            document_type="docx",
            preserve_hierarchy=True,
            extract_styles=True
        )
    
    def _process_pptx(self, file_path: str) -> 'ConversionResult':
        """
        Process PPTX file using the provider.
        
        Args:
            file_path: Path to PPTX file
            
        Returns:
            ConversionResult from docling
        """
        return self.provider.process_document(
            file_path,
            document_type="pptx",
            extract_slides=True,
            extract_visuals=True
        )
    
    def _process_html(self, file_path: str) -> 'ConversionResult':
        """
        Process HTML file using the provider.
        
        Args:
            file_path: Path to HTML file
            
        Returns:
            ConversionResult from docling
        """
        return self.provider.process_document(
            file_path,
            document_type="html",
            preserve_semantic_structure=True,
            extract_links=True
        )
    
    def _process_image(self, file_path: str) -> 'ConversionResult':
        """
        Process image file using the provider.
        
        Args:
            file_path: Path to image file
            
        Returns:
            ConversionResult from docling
        """
        return self.provider.process_document(
            file_path,
            document_type="image",
            use_vision=True,
            extract_text=True,
            analyze_layout=True
        )