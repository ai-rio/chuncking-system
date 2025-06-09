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
import os
import numpy as np # Needed for semantic similarity calculations

# Import SentenceTransformer for semantic embeddings
from sentence_transformers import SentenceTransformer, util

class HybridMarkdownChunker:
    """
    Hybrid chunking system optimized for i3/16GB hardware.
    Combines header-based, recursive, code-aware, table-aware, and now semantic splitting strategies.
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

        # Initialize embedding model if semantic chunking is enabled
        self.embedding_model = None
        if self.enable_semantic and config.EMBEDDING_MODEL:
            try:
                self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
                print(f"Loaded embedding model: {config.EMBEDDING_MODEL}")
            except Exception as e:
                print(f"Error loading embedding model {config.EMBEDDING_MODEL}: {e}. Disabling semantic chunking.")
                self.enable_semantic = False # Disable semantic if model fails to load

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
        Simplified and robust header detection.
        """
        # Header detection: Use re.search with re.MULTILINE to find '# ' at the beginning of any line.
        # This is the most reliable way to check for headers, regardless of leading newlines/whitespace.
        has_headers = bool(re.search(r'^\s*#+\s', content, re.MULTILINE))

        return {
            'has_headers': has_headers,
            'has_code': '```' in content,
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
        Sequentially processes content, prioritizing tables, then code blocks.
        For general text, it prioritizes header-based splitting, and then semantic
        splitting for sub-sections without headers.
        """
        all_chunks = []
        remaining_content = content
        
        # Regex to find markdown tables: More robust pattern.
        table_pattern = re.compile(
            r'(^\s*\|(?:[^|\n]+\|)+\s*\n'  # Header line
            r'^\s*\|(?:[-: ]+\|\s*)+\s*\n'  # Separator line
            r'(?:^\s*\|(?:[^|\n]+\|)+\s*\n)*)', # Zero or more data rows
            re.MULTILINE
        )

        # Regex to find code blocks
        code_pattern = re.compile(r'```[\s\S]*?```')

        cursor = 0
        iteration = 0
        while cursor < len(remaining_content):
            iteration += 1
            print(f"\n--- Sequential Chunking Iteration {iteration} ---")
            print(f"Current cursor position: {cursor}")
            print(f"Remaining content length from cursor: {len(remaining_content) - cursor}")
            
            # Try to find the next table
            table_match = table_pattern.search(remaining_content, cursor)
            
            # Try to find the next code block
            code_match = code_pattern.search(remaining_content, cursor)

            next_table_start = table_match.start() if table_match else len(remaining_content)
            next_code_start = code_match.start() if code_match else len(remaining_content)

            print(f"Next table starts at: {next_table_start if table_match else 'N/A'}")
            print(f"Next code block starts at: {next_code_start if code_match else 'N/A'}")

            # Process text before the next special block
            if min(next_table_start, next_code_start) > cursor:
                text_segment = remaining_content[cursor : min(next_table_start, next_code_start)]
                print(f"Processing text segment from {cursor} to {min(next_table_start, next_code_start)} (Length: {len(text_segment)})")
                print(f"Text Segment (first 100 chars): {text_segment[:100].strip()}...")
                if text_segment.strip():
                    text_analysis = self._detect_content_type(text_segment)
                    
                    if text_analysis['has_headers']:
                        print("  -> Text segment has headers, using header-recursive chunking.")
                        header_split_chunks = self._header_recursive_chunking(text_segment, metadata, text_analysis)
                        
                        for h_chunk in header_split_chunks:
                            h_chunk_analysis = self._detect_content_type(h_chunk.page_content)
                            if self.enable_semantic and \
                               not h_chunk_analysis['has_headers'] and \
                               not h_chunk_analysis['has_code'] and \
                               not h_chunk_analysis['has_tables'] and \
                               not h_chunk_analysis['has_lists']:
                                print("    -> Applying semantic chunking to a header-derived prose sub-segment.")
                                all_chunks.extend(self._semantic_chunking(h_chunk.page_content, h_chunk.metadata))
                            else:
                                all_chunks.append(h_chunk)
                    else: # No headers in this text segment
                        if self.enable_semantic:
                            print("  -> Text segment has no headers, applying semantic chunking.")
                            all_chunks.extend(self._semantic_chunking(text_segment, metadata))
                        else:
                            print("  -> Text segment no headers, semantic disabled, using simple recursive chunking.")
                            all_chunks.extend(self._simple_recursive_chunking(text_segment, metadata))
                cursor = min(next_table_start, next_code_start)
            
            # Process the special block (table or code) that comes next
            if cursor == next_table_start and table_match:
                table_content = table_match.group(0) # Get the full matched table string
                print(f"Processing TABLE content from {cursor} to {table_match.end()} (Length: {len(table_content)})")
                all_chunks.extend(self._table_aware_chunking(table_content, metadata))
                cursor = table_match.end()
            elif cursor == next_code_start and code_match:
                code_content = code_match.group(0) # Get the full matched code string
                print(f"Processing CODE content from {cursor} to {code_match.end()} (Length: {len(code_content)})")
                all_chunks.extend(self._code_aware_chunking(code_content, metadata))
                cursor = code_match.end()
            else:
                print(f"No more special blocks found starting at cursor {cursor}. Breaking loop.")
                break

        # Handle any remaining text at the end of the document
        if cursor < len(remaining_content) and remaining_content[cursor:].strip():
            final_text_segment = remaining_content[cursor:]
            print(f"\n--- Final Text Segment Handling ---")
            print(f"Processing final text segment from {cursor} (Length: {len(final_text_segment)})")
            text_analysis = self._detect_content_type(final_text_segment)
            
            if text_analysis['has_headers']:
                print("  -> Final text segment has headers, using header-recursive chunking.")
                header_split_chunks = self._header_recursive_chunking(final_text_segment, metadata, text_analysis)
                for h_chunk in header_split_chunks:
                    h_chunk_analysis = self._detect_content_type(h_chunk.page_content)
                    if self.enable_semantic and \
                       not h_chunk_analysis['has_headers'] and \
                       not h_chunk_analysis['has_code'] and \
                       not h_chunk_analysis['has_tables'] and \
                       not h_chunk_analysis['has_lists']:
                        print("    -> Applying semantic chunking to a final header-derived prose sub-segment.")
                        all_chunks.extend(self._semantic_chunking(h_chunk.page_content, h_chunk.metadata))
                    else:
                        all_chunks.append(h_chunk)
            else:
                if self.enable_semantic:
                    print("  -> Final text segment no headers, applying semantic chunking.")
                    all_chunks.extend(self._semantic_chunking(final_text_segment, metadata))
                else:
                    print("  -> Final text segment no headers, semantic disabled, using simple recursive chunking.")
                    all_chunks.extend(self._simple_recursive_chunking(final_text_segment, metadata))
        else:
            print("\nNo remaining text to process at the end.")

        return self._post_process_chunks(all_chunks)


    def _header_recursive_chunking(
        self,
        content: str,
        metadata: Dict[str, Any],
        analysis: Dict[str, bool]
    ) -> List[Document]:
        """Hybrid header-based + recursive chunking (recommended approach)"""

        try:
            header_chunks = self.header_splitter.split_text(content)
        except Exception as e:
            print(f"Header splitting failed: {e}. Falling back to recursive.")
            return self._simple_recursive_chunking(content, metadata)

        final_chunks = []
        for chunk in header_chunks:
            chunk_tokens = self._token_length(chunk.page_content)
            
            # Ensure base metadata is passed and merged with header metadata
            # MarkdownHeaderTextSplitter already adds header info to metadata
            chunk.metadata = {**metadata, **chunk.metadata}

            if chunk_tokens > self.chunk_size:
                sub_chunks = self.recursive_splitter.split_documents([chunk])
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata.update(chunk.metadata) # Retain header metadata
                    final_chunks.append(sub_chunk)
            else:
                final_chunks.append(chunk)
        return final_chunks

    def _simple_recursive_chunking(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """Simple recursive chunking for unstructured content"""
        document = Document(page_content=content, metadata=metadata)
        chunks = self.recursive_splitter.split_documents([document])
        return chunks

    def _code_aware_chunking(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """Specialized chunking for an isolated code block"""
        code_doc = Document(
            page_content=content,
            metadata={**metadata, 'content_type': 'code'}
        )
        code_chunks = self.code_splitter.split_documents([code_doc])
        return code_chunks

    def _parse_markdown_table(self, table_content: str) -> Dict[str, Any]:
        """
        Parses a Markdown table string into a structured dictionary.
        """
        lines = table_content.split('\n')
        header_line_index = -1
        separator_line_index = -1
        
        for i, line in enumerate(lines):
            if re.match(r'^\s*\|([-: ]+\|\s*)+\s*$', line.strip()):
                separator_line_index = i
                break

        if separator_line_index == -1:
            return {'header': [], 'rows': []}

        if separator_line_index > 0:
            header_line_index = separator_line_index - 1
            header_str = lines[header_line_index].strip()
            if not re.match(r'^\s*\|.*\|\s*$', header_str):
                return {'header': [], 'rows': []}
            header = [h.strip() for h in header_str.split('|') if h.strip()]
        else:
            return {'header': [], 'rows': []}

        rows = []
        for line_num in range(separator_line_index + 1, len(lines)):
            line = lines[line_num].strip()
            if re.match(r'^\s*\|.*\|\s*$', line):
                row_data = [d.strip() for d in line.split('|') if d.strip()]
                if row_data:
                    rows.append(row_data)
                
        return {'header': header, 'rows': rows, 'raw_table': table_content.strip()}

    def _table_aware_chunking(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Specialized chunking for an isolated Markdown table.
        """
        table_metadata = {**metadata, 'content_type': 'table'}
        parsed_table = self._parse_markdown_table(content)
        
        if not parsed_table['rows'] and not parsed_table['header']:
            return self._simple_recursive_chunking(content, metadata)

        header_str = ""
        if parsed_table['header']:
            header_str = "| " + " | ".join(parsed_table['header']) + " |\n"
            header_str += "|---" * len(parsed_table['header']) + "|\n"

        full_table_tokens = self._token_length(content)
        if full_table_tokens <= config.TABLE_CHUNK_MAX_TOKENS:
            return [Document(page_content=content.strip(), metadata=table_metadata)]
        else:
            chunks = []
            current_chunk_content = header_str if config.TABLE_MERGE_HEADER_WITH_ROWS else ""
            current_chunk_tokens = self._token_length(current_chunk_content)
            
            for row_idx, row in enumerate(parsed_table['rows']):
                row_str = "| " + " | ".join(row) + " |\n"
                row_tokens = self._token_length(row_str)

                if current_chunk_tokens + row_tokens > config.TABLE_CHUNK_MAX_TOKENS + 5: 
                    if current_chunk_content.strip() and self._token_length(current_chunk_content.strip()) >= config.MIN_CHUNK_WORDS: 
                        chunks.append(Document(page_content=current_chunk_content.strip(), metadata=table_metadata))
                    
                    current_chunk_content = header_str if config.TABLE_MERGE_HEADER_WITH_ROWS else ""
                    current_chunk_tokens = self._token_length(current_chunk_content)
                
                current_chunk_content += row_str
                current_chunk_tokens += row_tokens
            
            if current_chunk_content.strip() and self._token_length(current_chunk_content.strip()) >= config.MIN_CHUNK_WORDS:
                chunks.append(Document(page_content=current_chunk_content.strip(), metadata=table_metadata))
            
            if not chunks and content.strip():
                return self._simple_recursive_chunking(content, table_metadata)

        return chunks

    def _semantic_chunking(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Performs semantic chunking on a given text segment.
        Splits text into sentences, embeds them, and then groups them
        based on semantic similarity using a fixed threshold.
        """
        print(f"  -> Entering _semantic_chunking for content length: {len(content)}")
        if not self.embedding_model:
            print("  -> Semantic chunking is enabled but embedding model not loaded. Falling back to recursive.")
            return self._simple_recursive_chunking(content, metadata)

        sentences = re.split(r'(?<=[.!?])\s+', content) # Basic sentence splitting
        sentences = [s.strip() for s in sentences if s.strip()] # Clean and filter empty sentences
        
        print(f"  -> Semantic chunking: {len(sentences)} sentences identified.")

        if len(sentences) == 0: # Handle case where no valid sentences are found
            print("  -> Semantic chunking: No valid sentences found. Falling back to recursive.")
            return self._simple_recursive_chunking(content, metadata)
        elif len(sentences) < 2:
            print("  -> Semantic chunking: Less than 2 sentences, cannot compute similarity. Falling back to recursive.")
            return self._simple_recursive_chunking(content, metadata) # Not enough sentences for similarity analysis

        try:
            # Generate embeddings for each sentence
            sentence_embeddings = self.embedding_model.encode(sentences, convert_to_tensor=True)
            print(f"  -> Semantic chunking: Generated {len(sentence_embeddings)} embeddings.")

            # Calculate cosine similarity between adjacent sentences
            # The .diag() method extracts the diagonal elements (adjacent similarities)
            cosine_scores_adjacent = util.cos_sim(sentence_embeddings[:-1], sentence_embeddings[1:]).diag()
            print(f"  -> Semantic chunking: cosine_scores_adjacent shape: {cosine_scores_adjacent.shape}")


            chunks: List[Document] = []
            current_chunk_sentences = []
            current_chunk_text = ""

            for i, sentence in enumerate(sentences):
                sentence_tokens = self._token_length(sentence)

                is_semantic_break = False
                # Check for semantic break only if there's a previous sentence to compare to
                if i > 0:
                    similarity = cosine_scores_adjacent[i-1].item() # Access the scalar value
                    if similarity < config.SEMANTIC_SIMILARITY_THRESHOLD:
                        is_semantic_break = True
                        print(f"    -> Semantic break detected at sentence {i} (similarity: {similarity:.2f})")

                # Decide to break if:
                # 1. Current chunk text + new sentence exceeds max chunk size
                # 2. Semantic break is detected AND we have enough content for a chunk
                if (current_chunk_text and self._token_length(current_chunk_text + " " + sentence) > self.chunk_size) or \
                   (is_semantic_break and len(current_chunk_sentences) > 0 and self._token_length(current_chunk_text) >= config.MIN_CHUNK_WORDS):
                    if current_chunk_text.strip():
                        print(f"    -> Semantic chunk finalizing (tokens: {self._token_length(current_chunk_text)}), starting new chunk.")
                        chunk_doc = Document(
                            page_content=current_chunk_text.strip(),
                            metadata={**metadata, 'chunking_strategy': 'semantic'}
                        )
                        chunks.append(chunk_doc)
                    current_chunk_sentences = [] # Reset for new chunk
                    current_chunk_text = "" # Reset for new chunk

                current_chunk_sentences.append(sentence)
                current_chunk_text += (" " if current_chunk_text else "") + sentence

            # Add the last chunk if any content remains
            if current_chunk_text.strip():
                print(f"  -> Semantic chunking: Adding final semantic chunk (tokens: {self._token_length(current_chunk_text)}).")
                chunk_doc = Document(
                    page_content=current_chunk_text.strip(),
                    metadata={**metadata, 'chunking_strategy': 'semantic'}
                )
                chunks.append(chunk_doc)
            
            print(f"  -> _semantic_chunking finished. Generated {len(chunks)} semantic chunks.")
            return chunks if chunks else self._simple_recursive_chunking(content, metadata) # Fallback if semantic yields no chunks

        except Exception as e:
            print(f"  -> Error during semantic chunking: {e}. Falling back to recursive chunking.")
            return self._simple_recursive_chunking(content, metadata)


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

                chunks = self.chunk_document(content, metadata)
                results[file_path] = chunks

                if progress_callback:
                    progress_callback(i + 1, len(file_paths), file_path)

                if i % config.BATCH_SIZE == 0:
                    import gc
                    gc.collect()

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results[file_path] = []

        return results
