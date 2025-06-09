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
    Combines header-based, recursive, code-aware, and now robust table-aware splitting strategies.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        enable_semantic: bool = False
    ):
        self.chunk_size = chunk_size or config.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.DEFAULT_CHUNK_OVERLAP
        self.enable_semantic = enable_semantic # Still here for future semantic integration

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
        """
        Analyze content to determine general content features.
        The actual splitting logic will handle the precise parsing.
        """
        return {
            'has_headers': bool(re.search(r'^#+\s', content, re.MULTILINE)),
            'has_code': '```' in content,
            # Simpler table detection, full parsing in _table_aware_chunking
            'has_tables': bool(re.search(r'^\s*\|.*\|\s*\n\s*\|[-: ]+\|\s*\n', content, re.MULTILINE)),
            'has_lists': bool(re.search(r'^\s*[-*+]\s', content, re.MULTILINE)),
            'is_large': len(content) > self.chunk_size * 5
        }

    def chunk_document(
        self,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Main chunking method using hybrid approach.
        Now prioritizing a sequential processing of text, tables, and code blocks.
        """
        if not content.strip():
            return []

        metadata = metadata or {}
        
        # Sequence of processing: Tables, then Code, then Headers/Recursive
        # This ensures specialized handling for structured content.
        return self._sequential_complex_content_chunking(content, metadata)

    def _sequential_complex_content_chunking(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Sequentially processes content, prioritizing tables, then code blocks,
        and finally general text/headers.
        """
        all_chunks = []
        remaining_content = content
        
        # Regex to find markdown tables: More robust pattern.
        # It looks for a header line, then a separator line, then zero or more data rows.
        # It also handles optional leading/trailing spaces around pipes and within cells.
        table_pattern = re.compile(
            r'(^\s*\|(?:[^|\n]+\|)+\s*\n'  # Header line (e.g., | H1 | H2 |)
            r'^\s*\|(?:[-: ]+\|\s*)+\s*\n'  # Separator line (e.g., |----|----|)
            r'(?:^\s*\|(?:[^|\n]+\|)+\s*\n)*)', # Zero or more data rows (e.g., | D1 | D2 |)
            re.MULTILINE
        )

        # Regex to find code blocks
        code_pattern = re.compile(r'```[\s\S]*?```')

        cursor = 0
        while cursor < len(remaining_content):
            
            # Try to find the next table
            table_match = table_pattern.search(remaining_content, cursor)
            
            # Try to find the next code block
            code_match = code_pattern.search(remaining_content, cursor)

            next_table_start = table_match.start() if table_match else len(remaining_content)
            next_code_start = code_match.start() if code_match else len(remaining_content)

            # Process text before the next special block
            if min(next_table_start, next_code_start) > cursor:
                text_segment = remaining_content[cursor : min(next_table_start, next_code_start)]
                if text_segment.strip():
                    text_analysis = self._detect_content_type(text_segment)
                    if text_analysis['has_headers']:
                        all_chunks.extend(self._header_recursive_chunking(text_segment, metadata, text_analysis))
                    else:
                        all_chunks.extend(self._simple_recursive_chunking(text_segment, metadata))
                cursor = min(next_table_start, next_code_start)
            
            # Process the special block (table or code) that comes next
            if cursor == next_table_start and table_match:
                table_content = table_match.group(0) # Get the full matched table string
                all_chunks.extend(self._table_aware_chunking(table_content, metadata))
                cursor = table_match.end()
            elif cursor == next_code_start and code_match:
                code_content = code_match.group(0) # Get the full matched code string
                all_chunks.extend(self._code_aware_chunking(code_content, metadata))
                cursor = code_match.end()
            else:
                # No more special blocks found, break or handle remaining text
                break

        # Handle any remaining text at the end of the document
        if cursor < len(remaining_content) and remaining_content[cursor:].strip():
            final_text_segment = remaining_content[cursor:]
            text_analysis = self._detect_content_type(final_text_segment)
            if text_analysis['has_headers']:
                all_chunks.extend(self._header_recursive_chunking(final_text_segment, metadata, text_analysis))
            else:
                all_chunks.extend(self._simple_recursive_chunking(final_text_segment, metadata))

        return self._post_process_chunks(all_chunks)


    def _header_recursive_chunking(
        self,
        content: str,
        metadata: Dict[str, Any],
        analysis: Dict[str, bool]
    ) -> List[Document]:
        """Hybrid header-based + recursive chunking (recommended approach)"""

        # Phase 1: Split by headers
        try:
            # CORRECTED: Use split_text for MarkdownHeaderTextSplitter
            header_chunks = self.header_splitter.split_text(content)
        except Exception as e:
            print(f"Header splitting failed: {e}. Falling back to recursive.")
            return self._simple_recursive_chunking(content, metadata)

        # Phase 2: Refine large sections
        final_chunks = []

        for chunk in header_chunks:
            chunk_tokens = self._token_length(chunk.page_content)

            # Merge metadata
            # Langchain's MarkdownHeaderTextSplitter already merges some metadata,
            # but we want to ensure base metadata is always present.
            chunk.metadata = {**metadata, **chunk.metadata}

            if chunk_tokens > self.chunk_size:
                # Split large sections further
                # Ensure recursive splitter gets a Document object with existing metadata
                sub_chunks = self.recursive_splitter.split_documents([chunk])

                # Preserve all existing metadata, including headers, in sub-chunks
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata.update(chunk.metadata) # Update with parent chunk's metadata
                    final_chunks.append(sub_chunk)
            else:
                # Keep appropriately sized chunks
                final_chunks.append(chunk)

        return final_chunks # Post-processing moved to _sequential_complex_content_chunking


    def _simple_recursive_chunking(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """Simple recursive chunking for unstructured content"""

        document = Document(page_content=content, metadata=metadata)
        chunks = self.recursive_splitter.split_documents([document])

        return chunks # Post-processing moved to _sequential_complex_content_chunking

    def _code_aware_chunking(
        self,
        content: str, # This content is now just the code block
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """Specialized chunking for an isolated code block"""

        code_doc = Document(
            page_content=content,
            metadata={**metadata, 'content_type': 'code'}
        )
        code_chunks = self.code_splitter.split_documents([code_doc])
        return code_chunks # Post-processing moved to _sequential_complex_content_chunking

    def _parse_markdown_table(self, table_content: str) -> Dict[str, Any]:
        """
        Parses a Markdown table string into a structured dictionary.
        This version is more robust in identifying the header and rows
        by first finding the separator line.
        """
        lines = table_content.split('\n') # Do not strip here, maintain original line integrity
        
        header_line_index = -1
        separator_line_index = -1
        
        # Find the separator line first
        for i, line in enumerate(lines):
            if re.match(r'^\s*\|([-: ]+\|\s*)+\s*$', line.strip()):
                separator_line_index = i
                break

        if separator_line_index == -1:
            return {'header': [], 'rows': []}

        # The header is the line directly above the separator
        if separator_line_index > 0:
            header_line_index = separator_line_index - 1
            header_str = lines[header_line_index].strip()
            # Ensure the header line also contains pipes
            if not re.match(r'^\s*\|.*\|\s*$', header_str):
                return {'header': [], 'rows': []}
            header = [h.strip() for h in header_str.split('|') if h.strip()]
        else:
            return {'header': [], 'rows': []}

        # Extract rows (starting from line after separator)
        rows = []
        for line_num in range(separator_line_index + 1, len(lines)):
            line = lines[line_num].strip()
            if re.match(r'^\s*\|.*\|\s*$', line): # Ensure it's a table row (contains pipes)
                row_data = [d.strip() for d in line.split('|') if d.strip()]
                if row_data:
                    rows.append(row_data)
                
        return {'header': header, 'rows': rows, 'raw_table': table_content.strip()}

    def _table_aware_chunking(
        self,
        content: str, # This content is now just the table block
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Specialized chunking for an isolated Markdown table.
        Attempts to keep tables intact, or splits row-by-row if too large.
        """
        table_metadata = {**metadata, 'content_type': 'table'}
        parsed_table = self._parse_markdown_table(content)
        
        if not parsed_table['rows'] and not parsed_table['header']:
            # If parsing failed or no rows/header found, fall back to recursive on raw content
            return self._simple_recursive_chunking(content, metadata)

        header_str = ""
        if parsed_table['header']:
            header_str = "| " + " | ".join(parsed_table['header']) + " |\n"
            header_str += "|---" * len(parsed_table['header']) + "|\n"


        # Check if the entire table (including header) fits within TABLE_CHUNK_MAX_TOKENS
        full_table_tokens = self._token_length(content)
        if full_table_tokens <= config.TABLE_CHUNK_MAX_TOKENS:
            return [Document(page_content=content.strip(), metadata=table_metadata)]
        else:
            # Split table by rows
            chunks = []
            current_chunk_content = header_str if config.TABLE_MERGE_HEADER_WITH_ROWS else ""
            current_chunk_tokens = self._token_length(current_chunk_content)
            
            for row_idx, row in enumerate(parsed_table['rows']):
                row_str = "| " + " | ".join(row) + " |\n"
                row_tokens = self._token_length(row_str)

                # If adding this row exceeds max tokens, finalize current chunk and start new one
                # Add a small buffer to avoid off-by-one token issues
                if current_chunk_tokens + row_tokens > config.TABLE_CHUNK_MAX_TOKENS + 5: 
                    if current_chunk_content.strip() and self._token_length(current_chunk_content.strip()) >= config.MIN_CHUNK_WORDS: 
                        chunks.append(Document(page_content=current_chunk_content.strip(), metadata=table_metadata))
                    
                    # Start new chunk, potentially with header repeated
                    current_chunk_content = header_str if config.TABLE_MERGE_HEADER_WITH_ROWS else ""
                    current_chunk_tokens = self._token_length(current_chunk_content)
                
                current_chunk_content += row_str
                current_chunk_tokens += row_tokens
            
            # Add any remaining content in the last chunk
            if current_chunk_content.strip() and self._token_length(current_chunk_content.strip()) >= config.MIN_CHUNK_WORDS:
                chunks.append(Document(page_content=current_chunk_content.strip(), metadata=table_metadata))
            
            # If no chunks were created from rows (e.g., a single very long row),
            # ensure at least one chunk is created by simply splitting the raw table.
            if not chunks and content.strip():
                # Fallback to simple recursive splitting for the raw table content
                # This handles cases where a single row is larger than TABLE_CHUNK_MAX_TOKENS
                # and ensures content isn't lost.
                return self._simple_recursive_chunking(content, table_metadata)


        return chunks # Post-processing moved to _sequential_complex_content_chunking


    def _post_process_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Post-process chunks for quality and consistency,
        and assign a sequential global chunk_index.
        """
        processed_chunks = []
        global_chunk_index = 0 # Initialize a global index

        for chunk in chunks:
            # Skip very small chunks based on word count
            if len(chunk.page_content.strip().split()) < config.MIN_CHUNK_WORDS:
                continue

            # Assign sequential global chunk index
            chunk.metadata['chunk_index'] = global_chunk_index
            global_chunk_index += 1 # Increment for the next valid chunk

            chunk.metadata['chunk_tokens'] = self._token_length(chunk.page_content)
            chunk.metadata['chunk_chars'] = len(chunk.page_content)
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

                # Use the new sequential processing method
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
