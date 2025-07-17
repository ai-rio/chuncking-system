import os
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import mimetypes

from langchain_core.documents import Document

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
        Process document using DoclingProvider and return list of Document objects.
        
        Args:
            file_path: Path to the document file
            format_type: Document format ("pdf", "docx", "pptx", "html", "image", "auto")
            **kwargs: Additional processing options
            
        Returns:
            List of Document objects containing processed content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is not supported
            ChunkingError: If document processing fails
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
            # Call provider to process the document
            provider_result = self.provider.process_document(file_path, document_type=format_type, **kwargs)
            
            processing_time = time.time() - start_time
            
            # Create Document objects from provider response
            # Handle empty provider result case
            if not provider_result:
                return []
            
            # Merge provider metadata with base metadata
            base_metadata = {
                "source": file_path,
                "format": format_type,
                "file_size": file_size,
                "processing_time": processing_time
            }
            
            # Add provider metadata and structure data at top level
            provider_metadata = provider_result.get('metadata', {})
            provider_structure = provider_result.get('structure', {})
            merged_metadata = {**base_metadata, **provider_metadata, **provider_structure}
            
            # Create a single Document from the processed text
            document = Document(
                page_content=provider_result.get('text', ''),
                metadata=merged_metadata
            )
            
            return [document]
            
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
            if result.success:
                # Try to get markdown from provider or use text as fallback
                if hasattr(self.provider, 'export_to_markdown'):
                    return self.provider.export_to_markdown(file_path)
                else:
                    return result.text
            else:
                return f"Error processing document: {result.error_message}"
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
            if result.success:
                # Try to get HTML from provider or use text as fallback
                if hasattr(self.provider, 'export_to_html'):
                    return self.provider.export_to_html(file_path)
                else:
                    return f"<html><body><pre>{result.text}</pre></body></html>"
            else:
                return f"Error processing document: {result.error_message}"
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
            if result.success:
                # Try to get JSON from provider or use structured data
                if hasattr(self.provider, 'export_to_json'):
                    return self.provider.export_to_json(file_path)
                else:
                    import json
                    return json.dumps({
                        "text": result.text,
                        "structure": result.structure,
                        "metadata": result.metadata
                    })
            else:
                return f"Error processing document: {result.error_message}"
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
    
    def _process_pdf(self, content: bytes) -> Dict[str, Any]:
        """
        Process PDF content using the provider.
        
        Args:
            content: PDF content as bytes
            
        Returns:
            Processing result dictionary
        """
        return self.provider.process_document(
            content,
            document_type="pdf",
            extract_text=True,
            extract_structure=True,
            extract_metadata=True
        )
    
    def _process_docx(self, content: bytes) -> Dict[str, Any]:
        """
        Process DOCX content using the provider.
        
        Args:
            content: DOCX content as bytes
            
        Returns:
            Processing result dictionary
        """
        return self.provider.process_document(
            content,
            document_type="docx",
            preserve_hierarchy=True,
            extract_styles=True
        )
    
    def _process_pptx(self, content: bytes) -> Dict[str, Any]:
        """
        Process PPTX content using the provider.
        
        Args:
            content: PPTX content as bytes
            
        Returns:
            Processing result dictionary
        """
        return self.provider.process_document(
            content,
            document_type="pptx",
            extract_slides=True,
            extract_visuals=True
        )
    
    def _process_html(self, content: str) -> Dict[str, Any]:
        """
        Process HTML content using the provider.
        
        Args:
            content: HTML content as string
            
        Returns:
            Processing result dictionary
        """
        return self.provider.process_document(
            content,
            document_type="html",
            preserve_semantic_structure=True,
            extract_links=True
        )
    
    def _process_image(self, content: bytes) -> Dict[str, Any]:
        """
        Process image content using the provider.
        
        Args:
            content: Image content as bytes
            
        Returns:
            Processing result dictionary
        """
        return self.provider.process_document(
            content,
            document_type="image",
            use_vision=True,
            extract_text=True,
            analyze_layout=True
        )