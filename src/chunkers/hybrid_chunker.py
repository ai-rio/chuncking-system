import re
import uuid
from typing import List, Dict, Any, Optional
from collections import deque

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    PythonCodeTextSplitter
)
from langchain_core.documents import Document
import tiktoken
import src.config.settings as config
import os
import numpy as np

# Import SentenceTransformer for semantic embeddings
from sentence_transformers import SentenceTransformer, util

# Import google.generativeai for LLM calls
import google.generativeai as genai
import asyncio

class HybridChunker:
    """
    Hybrid chunking system optimized for i3/16GB hardware.
    Combines header-based, recursive, code-aware, table-aware, semantic,
    and now image-aware splitting strategies.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        enable_semantic: bool = False
    ):
        """
        Initializes the HybridChunker with specified chunk size and overlap.

        Args:
            chunk_size (int): The target size for each text chunk.
            chunk_overlap (int): The number of characters to overlap between consecutive chunks.
            enable_semantic (bool): Flag to enable/disable semantic chunking.
        """
        self.chunk_size = chunk_size or config.config.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.config.DEFAULT_CHUNK_OVERLAP
        self.enable_semantic = enable_semantic
        self.current_document_id = None # To be set by main processing pipeline

        # Configure Gemini API if key is available
        if config.config.GEMINI_API_KEY:
            genai.configure(api_key=config.config.GEMINI_API_KEY)
        else:
            print("Warning: GEMINI_API_KEY not set. LLM-based features (summaries, image descriptions) will use mock data.")

        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception: # Fallback for tiktoken model not found
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize embedding model if semantic chunking is enabled
        self.embedding_model = None
        if self.enable_semantic and config.config.EMBEDDING_MODEL:
            try:
                self.embedding_model = SentenceTransformer(config.config.EMBEDDING_MODEL)
                print(f"Loaded embedding model: {config.config.EMBEDDING_MODEL}")
            except Exception as e:
                print(f"Error loading embedding model {config.config.EMBEDDING_MODEL}: {e}. Disabling semantic chunking.")
                self.enable_semantic = False

        # Initialize splitters
        self._init_splitters()

    def _init_splitters(self):
        """Initialize all text splitters"""

        # Header-based splitter for Markdown structure
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.config.HEADER_LEVELS,
            strip_headers=False
        )

        # Recursive splitter for general text
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
            separators=config.config.SEPARATORS
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
        Analyzes content to determine general content features.
        Now includes image detection.
        """
        has_headers = bool(re.search(r'^\s*#+\s', content, re.MULTILINE))
        has_images = bool(re.search(r'!\[.*?\]\((.*?)\)', content)) # Detects markdown images

        return {
            'has_headers': has_headers,
            'has_code': '```' in content,
            'has_tables': bool(re.search(r'^\s*\|.*\|\s*\n\s*\|[-: ]+\|\s*\n', content, re.MULTILINE)),
            'has_lists': bool(re.search(r'^\s*[-*+]\s', content, re.MULTILINE)),
            'has_images': has_images, # New detection
            'is_large': len(content) > self.chunk_size * 5
        }

    async def chunk_document(
        self,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Main chunking method using hybrid approach, now an async method
        to accommodate LLM calls for image processing.
        """
        if not content.strip():
            return []

        self.current_document_id = metadata.get("document_id", str(uuid.uuid4()))
        metadata = metadata or {}
        
        # Sequence of processing: Tables, then Code, then Images, then Headers/Recursive/Semantic
        processed_chunks = await self._sequential_complex_content_chunking(content, metadata)
        return self._post_process_chunks(processed_chunks)

    async def _sequential_complex_content_chunking(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Sequentially processes content, prioritizing tables, then code blocks,
        then images. For general text, it prioritizes header-based splitting,
        and then semantic splitting for sub-sections without headers.
        """
        all_chunks = []
        remaining_content = content
        
        table_pattern = re.compile(
            r'(^\s*\|(?:[^|\n]+\|)+\s*\n'
            r'^\s*\|(?:[-: ]+\|\s*)+\s*\n'
            r'(?:^\s*\|(?:[^|\n]+\|)+\s*\n)*)',
            re.MULTILINE
        )

        code_pattern = re.compile(r'```[\s\S]*?```')
        image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

        cursor = 0
        iteration = 0
        while cursor < len(remaining_content):
            iteration += 1
            print(f"\n--- Sequential Chunking Iteration {iteration} ---")
            print(f"Current cursor position: {cursor}")
            print(f"Remaining content length from cursor: {len(remaining_content) - cursor}")
            
            table_match = table_pattern.search(remaining_content, cursor)
            code_match = code_pattern.search(remaining_content, cursor)

            # --- DEBUG PRINTS FOR IMAGE DETECTION ---
            print(f"  -> IMAGE_DEBUG: Content around cursor for image search (first 100 chars from cursor): '{remaining_content[cursor:cursor+100].strip()}'")
            print(f"  -> IMAGE_DEBUG: Full content length: {len(remaining_content)}")
            print(f"  -> IMAGE_DEBUG: Cursor: {cursor}")
            
            image_match = image_pattern.search(remaining_content, cursor)
            if image_match:
                print(f"  -> IMAGE_DEBUG: FOUND IMAGE at {image_match.start()}: {image_match.group(0)}")
            else:
                print(f"  -> IMAGE_DEBUG: NO IMAGE found from cursor {cursor}")
            # --- END DEBUG PRINTS ---

            next_table_start = table_match.start() if table_match else len(remaining_content)
            next_code_start = code_match.start() if code_match else len(remaining_content)
            next_image_start = image_match.start() if image_match else len(remaining_content)

            # Determine the start of the next special block (table, code, or image)
            next_special_start = min(next_table_start, next_code_start, next_image_start)

            print(f"Next table starts at: {next_table_start if table_match else 'N/A'}")
            print(f"Next code block starts at: {next_code_start if code_match else 'N/A'}")
            print(f"Next image starts at: {next_image_start if image_match else 'N/A'}")


            # Process text before the next special block
            if next_special_start > cursor:
                text_segment = remaining_content[cursor : next_special_start]
                print(f"Processing text segment from {cursor} to {next_special_start} (Length: {len(text_segment)})")
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
                               not h_chunk_analysis['has_lists'] and \
                               not h_chunk_analysis['has_images']:
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
                cursor = next_special_start
            
            # Process the special block (table, code, or image) that comes next
            if cursor == next_table_start and table_match:
                table_content = table_match.group(0)
                print(f"Processing TABLE content from {cursor} to {table_match.end()} (Length: {len(table_content)})")
                all_chunks.extend(self._table_aware_chunking(table_content, metadata))
                cursor = table_match.end()
            elif cursor == next_code_start and code_match:
                code_content = code_match.group(0)
                print(f"Processing CODE content from {cursor} to {code_match.end()} (Length: {len(code_content)})")
                all_chunks.extend(self._code_aware_chunking(code_content, metadata))
                cursor = code_match.end()
            elif cursor == next_image_start and image_match:
                image_markdown = image_match.group(0)
                print(f"Processing IMAGE content from {cursor} to {image_match.end()} (Length: {len(image_markdown)})")
                image_chunks = await self._image_aware_processing(image_markdown, metadata)
                all_chunks.extend(image_chunks)
                cursor = image_match.end()
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
                       not h_chunk_analysis['has_lists'] and \
                       not h_chunk_analysis['has_images']:
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

        return all_chunks

    def _header_recursive_chunking(
        self,
        content: str,
        base_metadata: Dict[str, Any],
        content_analysis: Dict[str, bool]
    ) -> List[Document]:
        """
        Chunks the document by headers, maintaining semantic boundaries.
        Within header sections, it further applies appropriate chunking strategies.
        """
        try:
            # MarkdownHeaderTextSplitter will return a list of Documents with header metadata
            header_sections = self.header_splitter.split_text(content)
        except Exception as e:
            print(f"Header splitting failed: {e}. Falling back to recursive.")
            # If header splitting fails, treat the whole content as prose
            return self._simple_recursive_chunking(content, base_metadata)

        all_chunks = []

        for section_doc in header_sections:
            section_content = section_doc.page_content
            # The header metadata is already in section_doc.metadata
            combined_metadata = {**base_metadata, **section_doc.metadata}

            # Re-analyze the section content for its internal structure
            section_analysis = self._detect_content_type(section_content)

            # Prioritize sub-chunking strategies within header sections
            if section_analysis['has_tables']:
                all_chunks.extend(self._table_aware_chunking(section_content, combined_metadata))
            elif section_analysis['has_code']:
                all_chunks.extend(self._code_aware_chunking(section_content, combined_metadata))
            elif section_analysis['has_images']:
                # Note: _image_aware_processing is async, need to await if called directly here
                # For now, it's safer to assume _sequential_complex_content_chunking handles outer logic
                # or _image_aware_chunking placeholder takes care of it by calling recursive.
                all_chunks.extend(self._simple_recursive_chunking(section_content, combined_metadata)) # Fallback
            else:
                # If no specific structures, apply simple recursive chunking to the section
                all_chunks.extend(self._simple_recursive_chunking(section_content, combined_metadata))
        return all_chunks

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
        """Parses a Markdown table string into a structured dictionary."""
        lines = table_content.split('\n')
        header_line_index = -1
        separator_line_index = -1
        for i, line in enumerate(lines):
            if re.match(r'^\s*\|([-: ]+\|\s*)+\s*$', line.strip()):
                separator_line_index = i
                break
        if separator_line_index == -1: return {'header': [], 'rows': []}
        if separator_line_index > 0:
            header_line_index = separator_line_index - 1
            header_str = lines[header_line_index].strip()
            if not re.match(r'^\s*\|.*\|\s*$', header_str): return {'header': [], 'rows': []}
            header = [h.strip() for h in header_str.split('|') if h.strip()]
        else: return {'header': [], 'rows': []}
        rows = []
        for line_num in range(separator_line_index + 1, len(lines)):
            line = lines[line_num].strip()
            if re.match(r'^\s*\|.*\|\s*$', line):
                row_data = [d.strip() for d in line.split('|') if d.strip()]
                if row_data: rows.append(row_data)
        return {'header': header, 'rows': rows, 'raw_table': table_content.strip()}

    def _table_aware_chunking(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """Specialized chunking for an isolated Markdown table."""
        table_metadata = {**metadata, 'content_type': 'table'}
        parsed_table = self._parse_markdown_table(content)
        if not parsed_table['rows'] and not parsed_table['header']: return self._simple_recursive_chunking(content, metadata)
        header_str = ""
        if parsed_table['header']:
            header_str = "| " + " | ".join(parsed_table['header']) + " |\n"
            header_str += "|---" * len(parsed_table['header']) + "|\n"
        full_table_tokens = self._token_length(content)
        if full_table_tokens <= config.config.TABLE_CHUNK_MAX_TOKENS:
            return [Document(page_content=content.strip(), metadata=table_metadata)]
        else:
            chunks = []
            current_chunk_content = header_str if config.config.TABLE_MERGE_HEADER_WITH_ROWS else ""
            current_chunk_tokens = self._token_length(current_chunk_content)
            for row_idx, row in enumerate(parsed_table['rows']):
                row_str = "| " + " | ".join(row) + " |\n"
                row_tokens = self._token_length(row_str)
                if current_chunk_tokens + row_tokens > config.config.TABLE_CHUNK_MAX_TOKENS + 5:
                    if current_chunk_content.strip() and self._token_length(current_chunk_content.strip()) >= config.config.MIN_CHUNK_WORDS:
                        chunks.append(Document(page_content=current_chunk_content.strip(), metadata=table_metadata))
                    current_chunk_content = header_str if config.config.TABLE_MERGE_HEADER_WITH_ROWS else ""
                    current_chunk_tokens = self._token_length(current_chunk_content)
                current_chunk_content += row_str
                current_chunk_tokens += row_tokens
            if current_chunk_content.strip() and self._token_length(current_chunk_content.strip()) >= config.config.MIN_CHUNK_WORDS:
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
        """
        if not self.enable_semantic or not self.embedding_model:
            print("  -> Semantic chunking is disabled or embedding model not loaded. Falling back to recursive.")
            return self._simple_recursive_chunking(content, metadata)

        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            return self._simple_recursive_chunking(content, metadata)
        elif len(sentences) < 2:
            return self._simple_recursive_chunking(content, metadata)

        try:
            sentence_embeddings = self.embedding_model.encode(sentences, convert_to_tensor=True)
            cosine_scores_adjacent = util.cos_sim(sentence_embeddings[:-1], sentence_embeddings[1:]).diag()

            chunks: List[Document] = []
            current_chunk_sentences = deque() # Use deque for efficient appending/popping
            current_chunk_text = ""

            for i, sentence in enumerate(sentences):
                sentence_tokens = self._token_length(sentence)

                is_semantic_break = False
                if i > 0:
                    similarity = cosine_scores_adjacent[i-1].item()
                    if similarity < config.config.SEMANTIC_SIMILARITY_THRESHOLD:
                        is_semantic_break = True

                # Check if adding the current sentence would exceed chunk size OR if a semantic break occurs
                # and the current chunk has enough content.
                if (self._token_length(current_chunk_text + " " + sentence) > self.chunk_size and current_chunk_text) or \
                   (is_semantic_break and len(current_chunk_sentences) > 0 and self._token_length(current_chunk_text) >= config.config.MIN_CHUNK_WORDS):
                    if current_chunk_text.strip():
                        chunk_doc = Document(
                            page_content=current_chunk_text.strip(),
                            metadata={**metadata, 'chunking_strategy': 'semantic'}
                        )
                        chunks.append(chunk_doc)
                    current_chunk_sentences.clear() # Clear the deque
                    current_chunk_text = ""

                current_chunk_sentences.append(sentence)
                current_chunk_text = " ".join(current_chunk_sentences) # Reconstruct text from deque

            # Add any remaining content as a final chunk
            if current_chunk_text.strip():
                chunk_doc = Document(
                    page_content=current_chunk_text.strip(),
                    metadata={**metadata, 'chunking_strategy': 'semantic'}
                )
                chunks.append(chunk_doc)
            
            return chunks if chunks else self._simple_recursive_chunking(content, metadata)

        except Exception as e:
            print(f"Error during semantic chunking: {e}. Falling back to recursive chunking.")
            return self._simple_recursive_chunking(content, metadata)


    async def _image_aware_processing(
        self,
        image_markdown: str,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Processes image markdown by generating an LLM description and replacing the image
        with its description in a new chunk.
        """
        print(f"  -> Entering _image_aware_processing for image markdown: {image_markdown[:50]}...")
        
        if not config.config.ENABLE_LLM_IMAGE_DESCRIPTION:
            print("  -> LLM image description is disabled. Skipping.")
            return [Document(page_content=image_markdown, metadata={**metadata, 'content_type': 'image_markdown', 'has_images': True})] # Added has_images=True

        if not config.config.GEMINI_API_KEY:
            print("  -> GEMINI_API_KEY not set. Cannot generate LLM image description. Using alt text if available or placeholder.")
            alt_text_match = re.search(r'!\[(.*?)\]', image_markdown)
            alt_text = alt_text_match.group(1) if alt_text_match else "an image"
            description = f"Description of {alt_text}."
            return [Document(page_content=description, metadata={**metadata, 'content_type': 'image_description_fallback', 'has_images': True})] # Added has_images=True
        
        # Extract alt text and URL if available for context
        alt_text_match = re.search(r'!\[(.*?)\]', image_markdown)
        image_url_match = re.search(r'\]\((.*?)\)', image_markdown)
        
        alt_text = alt_text_match.group(1) if alt_text_match else "an image"
        image_url = image_url_match.group(1) if image_url_match else None

        prompt_parts = [config.config.LLM_IMAGE_DESCRIPTION_PROMPT]
        if alt_text and alt_text != "image":
            prompt_parts.append(f"The image's alt text is: '{alt_text}'.")
        if image_url:
            prompt_parts.append(f"The image URL is: '{image_url}'.")
        prompt_parts.append("Please provide the description.")

        full_prompt = "\n".join(prompt_parts)

        model = genai.GenerativeModel(config.config.LLM_IMAGE_MODEL)

        try:
            # Using asyncio.to_thread to run blocking model.generate_content in a separate thread
            chatHistory = [{"role": "user", "parts": [{"text": full_prompt}]}]
            
            response = await asyncio.to_thread(model.generate_content, chatHistory)
            description_text = response.candidates[0].content.parts[0].text
            print(f"  -> LLM Image Description generated: {description_text[:50]}...")

        except Exception as e:
            print(f"  -> Error generating LLM image description: {e}. Falling back to alt text.")
            description_text = f"Description of {alt_text}."

        # Create a new document with the LLM-generated description
        image_description_chunk = Document(
            page_content=description_text,
            metadata={
                **metadata,
                'chunk_id': f"{self.current_document_id}-{uuid.uuid4()}",
                'chunk_type': 'visual',
                'source_segment_type': 'image',
                'image_alt_text': alt_text,
                'image_url': image_url,
                'enriched_by_llm': True,
                'summary': description_text, # Summary is the description itself
                'has_images': True # Explicitly mark this chunk as containing an image
            }
        )
        return [image_description_chunk]

    def _post_process_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Post-process chunks for quality and consistency,
        and assign a sequential global chunk_index.
        """
        processed_chunks = []
        global_chunk_index = 0

        for chunk in chunks:
            # Ensure chunk content meets minimum word count unless it's a structural element
            # This logic can be refined based on specific requirements for structural chunks
            if len(chunk.page_content.strip().split()) < config.config.MIN_CHUNK_WORDS and \
               chunk.metadata.get('chunk_type') not in ['structural', 'visual']:
                continue

            chunk.metadata['chunk_index'] = global_chunk_index
            global_chunk_index += 1

            chunk.metadata['chunk_tokens'] = self._token_length(chunk.page_content)
            chunk.metadata['chunk_chars'] = len(chunk.page_content)
            chunk.metadata['word_count'] = len(chunk.page_content.split())

            processed_chunks.append(chunk)

        return processed_chunks

    async def batch_process_files(
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

                # chunk_document is now async
                chunks = await self.chunk_document(content, metadata)
                results[file_path] = chunks

                if progress_callback:
                    progress_callback(i + 1, len(file_paths), file_path)

                if i % config.config.BATCH_SIZE == 0:
                    import gc
                    gc.collect()

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results[file_path] = []

        return results
