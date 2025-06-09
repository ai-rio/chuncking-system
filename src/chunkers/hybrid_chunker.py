from typing import List, Dict, Any, Optional
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    PythonCodeTextSplitter
)
from langchain_core.documents import Document
import re
import tiktoken
from src.config.settings import config
import os # Import os for basename in batch_process_files

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

        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize splitters
        self._init_splitters()

    def _init_splitters(self):
        """Initialize all text splitters"""

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

    def _token_length(self, text: str) -> int:
        """Calculate token length using tiktoken"""
        return len(self.tokenizer.encode(text))

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
        if not content.strip():
            return []

        metadata = metadata or {}
        content_analysis = self._detect_content_type(content)

        # Choose strategy based on content analysis
        if content_analysis['has_headers']:
            return self._header_recursive_chunking(content, metadata, content_analysis)
        elif content_analysis['has_code']:
            return self._code_aware_chunking(content, metadata)
        else:
            return self._simple_recursive_chunking(content, metadata)

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
            print(f"Header splitting failed: {e}. Falling back to recursive.")
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

        for i, chunk in enumerate(chunks):
            # Skip very small chunks
            if len(chunk.page_content.strip()) < config.MIN_CHUNK_WORDS:
                continue

            # Add chunk index
            chunk.metadata['chunk_index'] = i
            chunk.metadata['chunk_tokens'] = self._token_length(chunk.page_content)
            chunk.metadata['chunk_chars'] = len(chunk.page_content)

            # Add content analysis
            chunk.metadata['word_count'] = len(chunk.page_content.split())

            processed_chunks.append(chunk)

        return processed_chunks

    def batch_process_files(
        self,
        file_paths: List[str],
        progress_callback=None
    ) -> Dict[str, List[Document]]:
        """Process multiple files efficiently for i3/16GB system"""

        results = {}

        for i, file_path in enumerate(file_paths):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                metadata = {
                    'source': file_path,
                    'file_name': os.path.basename(file_path)
                }

                chunks = self.chunk_document(content, metadata)
                results[file_path] = chunks

                if progress_callback:
                    progress_callback(i + 1, len(file_paths), file_path)

                # Memory cleanup for i3 system
                if i % config.BATCH_SIZE == 0:
                    import gc
                    gc.collect()

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results[file_path] = []

        return results