import os
import mimetypes
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from src.utils.file_handler import FileHandler
from src.chunkers.docling_processor import DoclingProcessor
from src.exceptions import FileHandlingError, ValidationError


@dataclass
class FileInfo:
    """Information about a file for processing"""
    file_path: str
    format_type: str
    file_size: int
    mime_type: str


class EnhancedFileHandler:
    """
    Enhanced file handler with format detection and intelligent routing.
    
    Extends existing FileHandler functionality to support multi-format processing
    while maintaining backward compatibility with existing Markdown workflows.
    """
    
    def __init__(self, file_handler: FileHandler, docling_processor: DoclingProcessor):
        """
        Initialize EnhancedFileHandler.
        
        Args:
            file_handler: Existing FileHandler instance for backward compatibility
            docling_processor: DoclingProcessor instance for multi-format processing
            
        Raises:
            ValueError: If required components are None or invalid
        """
        if not isinstance(file_handler, FileHandler):
            raise ValueError("FileHandler instance required")
        if not isinstance(docling_processor, DoclingProcessor):
            raise ValueError("DoclingProcessor instance required")
        
        self.file_handler = file_handler
        self.docling_processor = docling_processor
        
        # Supported format mappings
        self.format_mappings = {
            # Extension-based mappings
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.pptx': 'pptx',
            '.html': 'html',
            '.htm': 'html',
            '.md': 'markdown',
            '.markdown': 'markdown',
            '.mdown': 'markdown',
            '.mkd': 'markdown',
            '.png': 'image',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.gif': 'image',
            '.bmp': 'image',
            '.tiff': 'image',
            '.tif': 'image'
        }
        
        # MIME type mappings for additional validation
        self.mime_mappings = {
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
            'text/html': 'html',
            'text/markdown': 'markdown',
            'image/jpeg': 'image',
            'image/png': 'image',
            'image/gif': 'image',
            'image/bmp': 'image',
            'image/tiff': 'image'
        }
        
        # File size limits (in bytes)
        self.size_limits = {
            'pdf': 50 * 1024 * 1024,      # 50MB
            'docx': 25 * 1024 * 1024,     # 25MB
            'pptx': 50 * 1024 * 1024,     # 50MB
            'html': 10 * 1024 * 1024,     # 10MB
            'markdown': 5 * 1024 * 1024,   # 5MB
            'image': 20 * 1024 * 1024      # 20MB
        }
    
    def detect_format(self, file_path: str) -> str:
        """
        Detect file format based on extension and MIME type.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Detected format string ('pdf', 'docx', 'pptx', 'html', 'image', 'markdown', 'unknown')
            
        Raises:
            ValidationError: If file_path is not a string
            FileNotFoundError: If file doesn't exist
        """
        if not isinstance(file_path, str):
            raise ValidationError(
                "File path must be a string",
                field="file_path",
                value=type(file_path)
            )
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Primary detection via file extension
        _, ext = os.path.splitext(file_path.lower())
        if ext in self.format_mappings:
            return self.format_mappings[ext]
        
        # Secondary detection via MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type in self.mime_mappings:
            return self.mime_mappings[mime_type]
        
        # Return unknown if no match
        return 'unknown'
    
    def route_to_processor(self, file_path: str, format_type: str):
        """
        Route file to appropriate processor based on format type.
        
        Args:
            file_path: Path to the file to process
            format_type: Detected format type
            
        Returns:
            ProcessingResult object from the appropriate processor
            
        Raises:
            ValidationError: If inputs are invalid
            ValueError: If format is not supported
        """
        if not isinstance(file_path, str) or not file_path:
            raise ValidationError(
                "File path must be a non-empty string",
                field="file_path",
                value=file_path
            )
        
        if not isinstance(format_type, str) or not format_type:
            raise ValidationError(
                "Format type must be a non-empty string",
                field="format_type",
                value=format_type
            )
        
        # Route to DoclingProcessor for supported formats
        if format_type in ['pdf', 'docx', 'pptx', 'html', 'image']:
            return self.docling_processor.process_document(file_path, format_type)
        
        # Route to existing MarkdownProcessor for Markdown files
        elif format_type == 'markdown':
            return self._process_markdown(file_path)
        
        # Unsupported format
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def find_supported_files(self, directory: str) -> List[FileInfo]:
        """
        Find all supported files in a directory.
        
        Args:
            directory: Directory path to search
            
        Returns:
            List of FileInfo objects for supported files
            
        Raises:
            ValidationError: If directory is not a string
            FileHandlingError: If directory operations fail
        """
        if not isinstance(directory, str):
            raise ValidationError(
                "Directory path must be a string",
                field="directory",
                value=type(directory)
            )
        
        if not os.path.exists(directory):
            raise FileHandlingError(
                f"Directory not found: {directory}",
                file_path=directory,
                operation="find_supported_files"
            )
        
        if not os.path.isdir(directory):
            raise FileHandlingError(
                "Path is not a directory",
                file_path=directory,
                operation="find_supported_files"
            )
        
        supported_files = []
        
        try:
            # Walk through directory recursively
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    
                    # Detect format for each file
                    format_type = self.detect_format(file_path)
                    
                    # Include if format is supported
                    if format_type != 'unknown':
                        file_size = os.path.getsize(file_path)
                        mime_type, _ = mimetypes.guess_type(file_path)
                        
                        file_info = FileInfo(
                            file_path=file_path,
                            format_type=format_type,
                            file_size=file_size,
                            mime_type=mime_type or 'unknown'
                        )
                        supported_files.append(file_info)
            
            # Maintain backward compatibility by calling find_markdown_files
            # This ensures existing functionality continues to work
            try:
                markdown_files = self.file_handler.find_markdown_files(directory)
                
                # Add any markdown files that might have been missed
                existing_paths = {f.file_path for f in supported_files}
                for md_file in markdown_files:
                    if md_file not in existing_paths:
                        file_size = os.path.getsize(md_file)
                        file_info = FileInfo(
                            file_path=md_file,
                            format_type='markdown',
                            file_size=file_size,
                            mime_type='text/markdown'
                        )
                        supported_files.append(file_info)
            except Exception:
                # If backward compatibility fails, continue with detected files
                pass
            
            return sorted(supported_files, key=lambda x: x.file_path)
            
        except Exception as e:
            raise FileHandlingError(
                f"Failed to scan directory for supported files: {str(e)}",
                file_path=directory,
                operation="find_supported_files"
            ) from e
    
    def validate_file_format(self, file_path: str, expected_format: str) -> bool:
        """
        Validate file format and security constraints.
        
        Args:
            file_path: Path to the file to validate
            expected_format: Expected format type
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            # Basic existence check
            if not os.path.exists(file_path):
                return False
            
            # Format detection validation
            detected_format = self.detect_format(file_path)
            if detected_format != expected_format:
                return False
            
            # File size validation
            file_size = os.path.getsize(file_path)
            if expected_format in self.size_limits:
                if file_size > self.size_limits[expected_format]:
                    return False
            
            # Additional security checks can be added here
            # (e.g., content validation, malware scanning)
            
            return True
            
        except Exception:
            return False
    
    def process_batch(self, file_paths: List[str]) -> List[Any]:
        """
        Process multiple files in batch.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of ProcessingResult objects
            
        Raises:
            ValidationError: If file_paths is not a list
        """
        if not isinstance(file_paths, list):
            raise ValidationError(
                "File paths must be a list",
                field="file_paths",
                value=type(file_paths)
            )
        
        from src.chunkers.docling_processor import ProcessingResult
        results = []
        
        for file_path in file_paths:
            try:
                # Detect format
                format_type = self.detect_format(file_path)
                
                # Validate format
                if not self.validate_file_format(file_path, format_type):
                    # Create failed result as ProcessingResult object
                    error_result = ProcessingResult(
                        format_type=format_type,
                        file_path=file_path,
                        success=False,
                        text="",
                        structure={},
                        metadata={
                            "source": file_path,
                            "format": format_type,
                            "error": "File validation failed",
                            "processing_time": 0,
                            "file_size": 0
                        },
                        processing_time=0,
                        file_size=0
                    )
                    results.append(error_result)
                    continue
                
                # Route to processor (returns ProcessingResult)
                batch_result = self.route_to_processor(file_path, format_type)
                results.append(batch_result)
                
            except Exception as e:
                # Create failed result for any errors
                error_result = ProcessingResult(
                    format_type="unknown",
                    file_path=file_path,
                    success=False,
                    text="",
                    structure={},
                    metadata={
                        "source": file_path,
                        "format": "unknown",
                        "error": str(e),
                        "processing_time": 0,
                        "file_size": 0
                    },
                    processing_time=0,
                    file_size=0
                )
                results.append(error_result)
        
        return results
    
    def _process_markdown(self, file_path: str):
        """
        Process Markdown file using existing patterns.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            ProcessingResult object for the processed Markdown file
        """
        import time
        from src.chunkers.docling_processor import ProcessingResult
        
        start_time = time.time()
        
        try:
            # Read markdown file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Process markdown (simplified - can be enhanced with actual markdown processor)
            processing_time = time.time() - start_time
            
            # Return as ProcessingResult object
            return ProcessingResult(
                format_type='markdown',
                file_path=file_path,
                success=True,
                text=content,
                structure={},
                metadata={
                    "source": file_path,
                    "format": "markdown",
                    "file_size": file_size,
                    "processing_time": processing_time
                },
                processing_time=processing_time,
                file_size=file_size
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                format_type='markdown',
                file_path=file_path,
                success=False,
                text="",
                structure={},
                metadata={
                    "source": file_path,
                    "format": "markdown",
                    "error": str(e),
                    "processing_time": processing_time,
                    "file_size": 0
                },
                processing_time=processing_time,
                file_size=0
            )
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of all supported file formats.
        
        Returns:
            List of supported format strings
        """
        return list(set(self.format_mappings.values()))
    
    def is_format_supported(self, format_type: str) -> bool:
        """
        Check if a format is supported.
        
        Args:
            format_type: Format to check
            
        Returns:
            True if format is supported, False otherwise
        """
        return format_type in self.get_supported_formats()