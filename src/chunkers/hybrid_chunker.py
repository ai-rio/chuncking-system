from typing import List, Dict, Any, Optional, Tuple
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    PythonCodeTextSplitter
)
from langchain_core.documents import Document
import re
import tiktoken
from src.config.settings import config
from src.exceptions import (
    TokenizationError,
    ProcessingError,
    ValidationError,
    FileHandlingError,
    ConfigurationError,
    MemoryError,
    BatchProcessingError
)
from src.utils.performance import PerformanceMonitor, MemoryOptimizer, BatchProcessor, monitor_performance
import os # Import os for basename in batch_process_files
import gc
from src.utils.logger import get_logger

class HybridMarkdownChunker:
    """
    Hybrid chunking system optimized for i3/16GB hardware
    Combines header-based and recursive splitting strategies
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        enable_semantic: bool = False
    ):
        self.chunk_size = chunk_size or config.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.DEFAULT_CHUNK_OVERLAP
        self.enable_semantic = enable_semantic
        
        # Initialize logger
        self.logger = get_logger(__name__)

        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as e:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as fallback_error:
                raise TokenizationError(
                    "Failed to initialize tokenizer",
                    model="gpt-3.5-turbo or cl100k_base"
                ) from fallback_error

        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.memory_optimizer = MemoryOptimizer()
        self.batch_processor = BatchProcessor(
            batch_size=config.BATCH_SIZE,
            memory_optimizer=self.memory_optimizer
        )
        
        # Initialize splitters
        self._init_splitters()

    def _init_splitters(self):
        """Initialize all text splitters"""
        try:
            # Validate configuration
            if self.chunk_size <= 0:
                raise ConfigurationError(
                    "Chunk size must be positive",
                    config_key="chunk_size",
                    config_value=self.chunk_size
                )
            
            if self.chunk_overlap >= self.chunk_size:
                raise ConfigurationError(
                    "Chunk overlap must be smaller than chunk size",
                    config_key="chunk_overlap",
                    config_value=self.chunk_overlap
                )

            # Header-based splitter for Markdown structure
            self.header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=config.HEADER_LEVELS,
                strip_headers=False
            )

            # Recursive splitter for general text
            self.recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=self._token_length,
                separators=config.SEPARATORS
            )

            # Code-specific splitter
            self.code_splitter = PythonCodeTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize text splitters: {str(e)}"
            ) from e

    def _token_length(self, text: str) -> int:
        """Calculate token length using tiktoken"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            raise TokenizationError(
                "Failed to calculate token length",
                text_length=len(text)
            ) from e

    def _detect_content_type(self, content: str) -> Dict[str, bool]:
        """Analyze content to determine optimal chunking strategy"""
        return {
            'has_headers': bool(re.search(r'^#+\s', content, re.MULTILINE)),
            'has_code': '```' in content,
            'has_tables': '|' in content and '---' in content,
            'has_lists': bool(re.search(r'^\s*[-*+]\s', content, re.MULTILINE)),
            'is_large': len(content) > self.chunk_size * 5
        }

    def chunk_document(
        self,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Main chunking method using hybrid approach
        """
        try:
            with self.performance_monitor.monitor_operation(
                "chunk_document",
                content_length=len(content) if isinstance(content, str) else 0,
                chunk_size=self.chunk_size
            ):
                # Input validation
                if not isinstance(content, str):
                    raise ValidationError(
                        "Content must be a string",
                        field="content",
                        value=type(content)
                    )
                
                if not content.strip():
                    return []

                metadata = metadata or {}
                
                # Validate metadata
                if not isinstance(metadata, dict):
                    raise ValidationError(
                        "Metadata must be a dictionary",
                        field="metadata",
                        value=type(metadata)
                    )

                # Check memory usage before processing
                self.memory_optimizer.cleanup_if_needed()

                content_analysis = self._detect_content_type(content)

                # Choose strategy based on content analysis
                if content_analysis['has_headers']:
                    return self._header_recursive_chunking(content, metadata, content_analysis)
                elif content_analysis['has_code']:
                    return self._code_aware_chunking(content, metadata)
                else:
                    return self._simple_recursive_chunking(content, metadata)
                
        except (ValidationError, ConfigurationError, TokenizationError) as e:
            # Re-raise known exceptions
            raise
        except Exception as e:
            raise ProcessingError(
                f"Document chunking failed: {str(e)}",
                stage="chunk_document"
            ) from e

    def _header_recursive_chunking(
        self,
        content: str,
        metadata: Dict[str, Any],
        analysis: Dict[str, bool]
    ) -> List[Document]:
        """Hybrid header-based + recursive chunking (recommended approach)"""

        # Phase 1: Split by headers
        try:
            header_chunks = self.header_splitter.split_text(content)
        except Exception as e:
            self.logger.warning(
                f"Header splitting failed: {str(e)}. Falling back to recursive.",
                stage="header_splitting"
            )
            # Fallback to simple recursive chunking instead of raising exception
            return self._simple_recursive_chunking(content, metadata)

        # Phase 2: Refine large sections
        final_chunks = []

        for chunk in header_chunks:
            chunk_tokens = self._token_length(chunk.page_content)

            # Merge metadata
            chunk_metadata = {**metadata, **chunk.metadata}

            if chunk_tokens > self.chunk_size:
                # Split large sections further
                sub_chunks = self.recursive_splitter.split_documents([chunk])

                # Preserve header metadata in sub-chunks
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata.update(chunk_metadata)
                    final_chunks.append(sub_chunk)
            else:
                # Keep appropriately sized chunks
                chunk.metadata = chunk_metadata
                final_chunks.append(chunk)

        return self._post_process_chunks(final_chunks)

    def _simple_recursive_chunking(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """Simple recursive chunking for unstructured content"""

        document = Document(page_content=content, metadata=metadata)
        chunks = self.recursive_splitter.split_documents([document])

        return self._post_process_chunks(chunks)

    def _code_aware_chunking(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """Specialized chunking for code-heavy documents"""

        # Split code blocks and regular text separately
        code_pattern = r'```[\s\S]*?```'
        code_blocks = re.findall(code_pattern, content)

        if code_blocks:
            # Process code blocks with code splitter
            chunks = []
            remaining_content = content

            for code_block in code_blocks:
                # Remove code block from content
                remaining_content = remaining_content.replace(code_block, '[CODE_BLOCK]', 1)

                # Chunk the code block
                code_doc = Document(
                    page_content=code_block,
                    metadata={**metadata, 'content_type': 'code'}
                )
                code_chunks = self.code_splitter.split_documents([code_doc])
                chunks.extend(code_chunks)

            # Chunk remaining text
            if remaining_content.strip():
                text_chunks = self._simple_recursive_chunking(remaining_content, metadata)
                chunks.extend(text_chunks)

            return self._post_process_chunks(chunks)
        else:
            return self._simple_recursive_chunking(content, metadata)

    def _post_process_chunks(self, chunks: List[Document]) -> List[Document]:
        """Post-process chunks for quality and consistency"""

        processed_chunks = []
        chunk_index = 0  # Track actual chunk index separately

        for chunk in chunks:
            # Skip very small chunks
            if len(chunk.page_content.strip()) < config.MIN_CHUNK_WORDS:
                continue

            # Add chunk index (use separate counter for valid chunks)
            chunk.metadata['chunk_index'] = chunk_index
            chunk.metadata['chunk_tokens'] = self._token_length(chunk.page_content)
            chunk.metadata['chunk_chars'] = len(chunk.page_content)

            # Add content analysis
            chunk.metadata['word_count'] = len(chunk.page_content.split())

            processed_chunks.append(chunk)
            chunk_index += 1  # Increment only for valid chunks

        return processed_chunks

    def batch_process_files(
        self,
        file_paths: List[str],
        progress_callback=None
    ) -> Dict[str, List[Document]]:
        """Process multiple files efficiently using BatchProcessor with optimized memory management"""

        if not file_paths:
            return {}

        # Validate input
        if not isinstance(file_paths, list):
            raise ValidationError(
                "file_paths must be a list",
                field="file_paths",
                value=type(file_paths)
            )

        def process_single_file(file_path: str) -> Tuple[str, List[Document]]:
            """Process a single file and return file path with chunks"""
            try:
                # Validate file path
                if not isinstance(file_path, str):
                    raise ValidationError(
                        "File path must be a string",
                        field="file_path",
                        value=type(file_path)
                    )

                if not os.path.exists(file_path):
                    raise FileHandlingError(
                        f"File not found: {file_path}",
                        file_path=file_path
                    )

                # Read file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    raise FileHandlingError(
                        f"Failed to read file: {file_path}",
                        file_path=file_path,
                        original_error=str(e)
                    ) from e

                # Prepare metadata
                metadata = {
                    'source': file_path,
                    'file_name': os.path.basename(file_path),
                    'file_size': len(content)
                }

                # Process the document
                chunks = self.chunk_document(content, metadata)
                
                self.logger.debug(
                    "File processed successfully",
                    file_path=file_path,
                    chunks_created=len(chunks),
                    content_size=len(content)
                )

                return file_path, chunks

            except (ValidationError, FileHandlingError, ProcessingError) as e:
                # Re-raise known exceptions
                self.logger.error(
                    "Known error processing file during batch processing",
                    file_path=file_path,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise BatchProcessingError(
                    f"Failed to process file {file_path}: {str(e)}",
                    file_path=file_path,
                    original_error=e
                ) from e
            except Exception as e:
                # Handle unexpected errors
                self.logger.error(
                    "Unexpected error processing file during batch processing",
                    file_path=file_path,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise BatchProcessingError(
                    f"Unexpected error processing file {file_path}: {str(e)}",
                    file_path=file_path,
                    original_error=e
                ) from e

        # Create progress callback wrapper for BatchProcessor
        def batch_progress_callback(processed_count: int, total_count: int):
            if progress_callback:
                # Find the current file being processed (approximate)
                current_file = file_paths[min(processed_count - 1, len(file_paths) - 1)] if processed_count > 0 else ""
                progress_callback(processed_count, total_count, current_file)

        try:
            # Use BatchProcessor for optimized processing
            batch_results = self.batch_processor.process_batches(
                items=file_paths,
                processor_func=process_single_file,
                progress_callback=batch_progress_callback
            )

            # Convert results to the expected format
            results = {}
            
            for i, result in enumerate(batch_results):
                file_path = file_paths[i]
                if result is not None:
                    # result is a tuple of (file_path, chunks)
                    _, chunks = result
                    results[file_path] = chunks
                else:
                    # Processing failed, store empty list
                    results[file_path] = []
                    self.logger.warning(
                        "File processing failed, stored empty result",
                        file_path=file_path
                    )
                
                # Trigger memory cleanup periodically
                if (i + 1) % config.BATCH_SIZE == 0:
                    gc.collect()

            self.logger.info(
                "Batch processing completed",
                total_files=len(file_paths),
                successful_files=sum(1 for chunks in results.values() if chunks),
                total_chunks=sum(len(chunks) for chunks in results.values())
            )

            return results

        except Exception as e:
            self.logger.error(
                "Batch processing failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise BatchProcessingError(
                f"Batch processing failed: {str(e)}"
            ) from e
    
    def get_performance_report(self) -> str:
        """
        Get comprehensive performance report for chunking operations.
        
        Returns:
            Markdown formatted performance report
        """
        chunker_report = self.performance_monitor.generate_performance_report()
        batch_report = self.batch_processor.get_performance_report()
        
        combined_report = f"""
# Chunking System Performance Report

## Chunker Performance
{chunker_report}

## Batch Processing Performance  
{batch_report}
"""
        return combined_report
    
    def clear_performance_metrics(self):
        """
        Clear all performance metrics.
        """
        self.performance_monitor.clear_metrics()
        self.batch_processor.performance_monitor.clear_metrics()

    def get_batch_performance_report(self) -> str:
        """
        Get performance report for batch processing operations.
        
        Returns:
            Performance report as formatted string
        """
        return self.batch_processor.get_performance_report()