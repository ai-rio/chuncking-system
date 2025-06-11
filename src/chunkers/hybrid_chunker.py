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
import src.config.settings as settings_module  # Changed import for config
import os
import numpy as np

# Import SentenceTransformer for semantic embeddings
from sentence_transformers import SentenceTransformer, util

# Import google.generativeai for LLM calls
import google.generativeai as genai
import asyncio

# Import MetadataEnricher here so HybridChunker can receive it
from src.utils.metadata_enricher import MetadataEnricher

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
        enable_semantic: bool = False,
        metadata_enricher: Optional[MetadataEnricher] = None
    ):
        """
        Initializes the HybridChunker with specified chunk size and overlap.
    
        Args:
            chunk_size (int): The target size for each text chunk.
            chunk_overlap (int): The number of characters to overlap between consecutive chunks.
            enable_semantic (bool): Flag to enable/disable semantic chunking.
            metadata_enricher (MetadataEnricher): An instance of MetadataEnricher for LLM calls.
        """
        self.chunk_size = chunk_size or settings_module.config.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings_module.config.DEFAULT_CHUNK_OVERLAP
        self.enable_semantic = enable_semantic
        self.current_document_id = None
        self.metadata_enricher = metadata_enricher if metadata_enricher is not None else MetadataEnricher()
    
        # Configure Gemini API if key is available
        if settings_module.config.GEMINI_API_KEY:
            genai.configure(api_key=settings_module.config.GEMINI_API_KEY)
        else:
            print("Warning: GEMINI_API_KEY not set. LLM-based features (summaries, image descriptions) will use mock data.")

        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception: # Fallback for tiktoken model not found
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize embedding model if semantic chunking is enabled
        self.embedding_model = None
        if self.enable_semantic and settings_module.config.EMBEDDING_MODEL:
            try:
                self.embedding_model = SentenceTransformer(settings_module.config.EMBEDDING_MODEL)
                print(f"Loaded embedding model: {settings_module.config.EMBEDDING_MODEL}")
            except Exception as e:
                print(f"Error loading embedding model {settings_module.config.EMBEDDING_MODEL}: {e}. Disabling semantic chunking.")
                self.enable_semantic = False

        # Initialize splitters
        self._init_splitters()

    def _init_splitters(self):
        """Initialize all text splitters"""
    
        # Header-based splitter for Markdown structure
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=settings_module.config.HEADER_LEVELS,
            strip_headers=False
        )
    
        # Recursive splitter for general text
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
            separators=settings_module.config.SEPARATORS
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
        
        # Process the content through sequential and complex chunking strategies
        raw_chunks = await self._sequential_complex_content_chunking(content, metadata)
        
        # Apply post-processing to add token counts and chunk index
        final_chunks = self._post_process_chunks(raw_chunks)
        
        return final_chunks


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
        while cursor < len(remaining_content):
            
            table_match = table_pattern.search(remaining_content, cursor)
            code_match = code_pattern.search(remaining_content, cursor)
            image_match = image_pattern.search(remaining_content, cursor) 

            next_table_start = table_match.start() if table_match else len(remaining_content)
            next_code_start = code_match.start() if code_match else len(remaining_content)
            next_image_start = image_match.start() if image_match else len(remaining_content)

            # Determine the start of the next special block (table, code, or image)
            next_special_start = min(next_table_start, next_code_start, next_image_start)

            # Process text before the next special block
            if next_special_start > cursor:
                text_segment = remaining_content[cursor : next_special_start]
                if text_segment.strip():
                    text_analysis = self._detect_content_type(text_segment)
                    
                    if text_analysis['has_headers']:
                        all_chunks.extend(await self._header_recursive_chunking(text_segment, metadata, text_analysis))
                    else: # No headers in this text segment
                        if self.enable_semantic:
                            all_chunks.extend(await self._semantic_chunking(text_segment, metadata))
                        else:
                            all_chunks.extend(self._simple_recursive_chunking(text_segment, metadata))
                cursor = next_special_start
            
            # Process the special block (table, code, or image) that comes next
            if cursor == next_table_start and table_match:
                table_content = table_match.group(0)
                all_chunks.extend(self._table_aware_chunking(table_content, metadata))
                cursor = table_match.end()
            elif cursor == next_code_start and code_match:
                code_content = code_match.group(0)
                all_chunks.extend(self._code_aware_chunking(code_content, metadata))
                cursor = code_match.end()
            elif cursor == next_image_start and image_match:
                image_markdown = image_match.group(0)
                image_doc = await self._image_aware_processing(
                    image_markdown,
                    metadata
                )
                all_chunks.extend(image_doc)
                cursor = image_match.end()
            else:
                break

        # Handle any remaining text at the end of the document
        if cursor < len(remaining_content) and remaining_content[cursor:].strip():
            final_text_segment = remaining_content[cursor:]
            text_analysis = self._detect_content_type(final_text_segment)
            
            if text_analysis['has_headers']:
                all_chunks.extend(await self._header_recursive_chunking(final_text_segment, metadata, text_analysis))
            else:
                if self.enable_semantic:
                    all_chunks.extend(await self._semantic_chunking(final_text_segment, metadata))
                else:
                    all_chunks.extend(self._simple_recursive_chunking(final_text_segment, metadata))
        
        return all_chunks

    async def _header_recursive_chunking( # Corrected: Added async
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
                # Note: This branch is for images found within header sections,
                # so we still delegate to metadata_enricher.describe_image
                image_match = re.search(r'!\[(.*?)\]\((.*?)\)', section_content)
                if image_match:
                    image_doc = await self._image_aware_processing( # THIS IS NOW CORRECTLY AWAITed
                        image_match.group(0),
                        combined_metadata
                    )
                    all_chunks.extend(image_doc)
                else:
                     all_chunks.extend(self._simple_recursive_chunking(section_content, combined_metadata))
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
        # In the table chunking method around line 347
        full_table_tokens = self._token_length(content)
        if full_table_tokens <= settings_module.config.TABLE_CHUNK_MAX_TOKENS:
            return [Document(page_content=content.strip(), metadata=table_metadata)]
        else:
            chunks = []
            current_chunk_content = header_str if settings_module.config.TABLE_MERGE_HEADER_WITH_ROWS else ""
            current_chunk_tokens = self._token_length(current_chunk_content)
            for row_idx, row in enumerate(parsed_table['rows']):
                row_str = "| " + " | ".join(row) + " |\n"
                row_tokens = self._token_length(row_str)
                if current_chunk_tokens + row_tokens > settings_module.config.TABLE_CHUNK_MAX_TOKENS + 5:
                    if current_chunk_content.strip() and self._token_length(current_chunk_content.strip()) >= settings_module.config.MIN_CHUNK_WORDS:
                        chunks.append(Document(page_content=current_chunk_content.strip(), metadata=table_metadata))
                    current_chunk_content = header_str if settings_module.config.TABLE_MERGE_HEADER_WITH_ROWS else ""
                    current_chunk_tokens = self._token_length(current_chunk_content)
                current_chunk_content += row_str
                current_chunk_tokens += row_tokens
            if current_chunk_content.strip() and self._token_length(current_chunk_content.strip()) >= settings_module.config.MIN_CHUNK_WORDS:
                chunks.append(Document(page_content=current_chunk_content.strip(), metadata=table_metadata))
            if not chunks and content.strip():
                return self._simple_recursive_chunking(content, table_metadata)
        return chunks

    async def _semantic_chunking( # Marked as async
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
                # In the semantic chunking method around line 403
                if i > 0:
                    similarity = cosine_scores_adjacent[i-1].item()
                    if similarity < settings_module.config.SEMANTIC_SIMILARITY_THRESHOLD:
                        is_semantic_break = True

                # Check if adding the current sentence would exceed chunk size OR if a semantic break occurs
                # and the current chunk has enough content.
                if (self._token_length(current_chunk_text + " " + sentence) > self.chunk_size and current_chunk_text) or \
                   (is_semantic_break and len(current_chunk_sentences) > 0 and self._token_length(current_chunk_text) >= settings_module.config.MIN_CHUNK_WORDS):
                    if current_chunk_text.strip():
                        chunk_doc = Document(
                            page_content=current_chunk_text.strip(),
                            metadata={
                                **metadata,
                                'chunking_strategy': 'semantic',
                                'chunk_type': 'prose',         # ADDED THIS LINE
                                'source_segment_type': 'text'  # ADDED THIS LINE
                            }
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
                    metadata={
                        **metadata,
                        'chunking_strategy': 'semantic',
                        'chunk_type': 'prose',         # ADDED THIS LINE
                        'source_segment_type': 'text'  # ADDED THIS LINE
                    }
                )
                chunks.append(chunk_doc)
            
            return chunks if chunks else self._simple_recursive_chunking(content, metadata)

        except Exception as e:
            print(f"Error during semantic chunking: {e}. Falling back to recursive chunking.")
            return self._simple_recursive_chunking(content, metadata)


    async def _image_aware_processing( # This method is async
        self,
        image_markdown: str,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Processes image markdown by delegating to MetadataEnricher to generate an LLM description
        and creates a Document object for the image.
        """
        print(f"  -> Entering _image_aware_processing for image markdown: {image_markdown[:50]}...")
        
        # Extract alt_text and image_url early to ensure they are always defined
        alt_text_match = re.search(r'!\[(.*?)\]', image_markdown)
        alt_text = alt_text_match.group(1).strip() if alt_text_match else "an image"
        image_url_match = re.search(r'\]\((.*?)\)', image_markdown)
        image_url = image_url_match.group(1).strip() if image_url_match else ""

        # Delegate the actual description generation and caching to self.metadata_enricher
        image_description_content = await self.metadata_enricher.describe_image( # CORRECTLY AWAITed
            alt_text=alt_text,
            image_url=image_url
        )
        
        # Construct the Document object for the image chunk with its metadata
        image_doc = Document(page_content=image_description_content)
        image_doc.metadata.update(metadata) # Inherit document-level metadata
        image_doc.metadata['chunk_type'] = 'visual'
        image_doc.metadata['source_segment_type'] = 'image'
        image_doc.metadata['has_images'] = True # Explicitly mark for evaluator
        image_doc.metadata['image_alt_text'] = alt_text # Store original alt text
        image_doc.metadata['image_url'] = image_url # Store original image URL
        image_doc.metadata['summary'] = image_description_content # Description acts as summary

        return [image_doc] # Return as a list containing one Document


    async def enrich_chunks_with_llm_summaries_and_metadata(self, chunks: List[Document]) -> List[Document]:
        """
        Asynchronously enriches a list of chunks with LLM-generated summaries and structured metadata.
        This replaces the previous enrich_chunks_with_llm_summaries.
        """
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            # Process summaries and structured metadata for text-based chunks
            if chunk.metadata.get('source_segment_type') == 'text' or \
               chunk.metadata.get('chunk_type') == 'prose': # Ensure prose and other text types are summarized
                summary = await self.summarize_chunk(chunk.page_content)
                chunk.metadata['summary'] = summary
                
                # Also extract structured metadata for these text-based chunks
                # In the enrichment method around line 497
                if settings_module.config.ENABLE_LLM_METADATA_EXTRACTION:
                    extracted_metadata = await self.extract_metadata_from_chunk(chunk.page_content)
                    # Ensure extracted_metadata is a dict before updating
                    if isinstance(extracted_metadata, dict):
                        chunk.metadata.update(extracted_metadata)
                    else:
                        print(f"Warning: Extracted metadata for chunk {i} was not a dict: {extracted_metadata}")
                
                print(f"LLM Summary & Metadata processed for text chunk {i}.")
            elif chunk.metadata.get('source_segment_type') == 'image':
                # Image chunks already processed by _image_aware_processing and have a summary (description)
                # No further enrichment needed here unless explicitly desired.
                print(f"Image chunk {i} processed (description already handled during chunking).")
            else:
                # For other structural chunks (table, code) - just process summary for now if needed.
                # Can extend to extract metadata from tables/code if desired.
                summary = await self.summarize_chunk(chunk.page_content)
                chunk.metadata['summary'] = summary
                print(f"LLM Summary processed for structural chunk {i}.")

            enriched_chunks.append(chunk)
        return enriched_chunks

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

                # In the batch processing method around line 546
                if i % settings_module.config.BATCH_SIZE == 0:
                    import gc
                    gc.collect()

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results[file_path] = []

        return results

    def _post_process_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Post-processes chunks to add chunk_tokens and chunk_index metadata.
        
        Args:
            chunks: List of Document objects to process
            
        Returns:
            List of Document objects with enriched metadata
        """
        final_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Validate chunk structure
            if not isinstance(chunk.page_content, str):
                print(f"Warning: chunk {i} page_content is not a string, skipping token calculation")
                chunk.metadata['chunk_tokens'] = 0
            else:
                # Calculate token count
                chunk.metadata['chunk_tokens'] = self._token_length(chunk.page_content)
            
            # Ensure metadata is a dictionary
            if not isinstance(chunk.metadata, dict):
                chunk.metadata = {}
            
            # Add chunk index
            chunk.metadata['chunk_index'] = i
            
            final_chunks.append(chunk)
        
        return final_chunks
