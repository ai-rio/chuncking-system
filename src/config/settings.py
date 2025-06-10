# src/config/settings.py
from pydantic_settings import BaseSettings
from typing import List, Dict, Any
import os

class ChunkingConfig(BaseSettings):
    # Chunking parameters optimized for i3/16GB and Gemini token constraints
    DEFAULT_CHUNK_SIZE: int = 800 # Set a target for tokens, allowing space for prompt
    DEFAULT_CHUNK_OVERLAP: int = 150 # Moderate overlap for context
    MAX_CHUNK_SIZE: int = 1200 # Absolute max for very long sentences/paragraphs
    MIN_CHUNK_WORDS: int = 10 # Adjusted to be slightly lower for more granular chunks
    MAX_CHUNK_WORDS: int = 600 # Adjusted for new chunk size

    # Markdown header levels to split on, comprehensive for a book structure
    HEADER_LEVELS: List[tuple] = [
        ("#", "Part"),       # e.g., "Part I: The New Reality of Markets"
        ("##", "Chapter"),    # e.g., "## 1: THE DISCOVERY JOURNEY"
        ("###", "Section"),   # For potential sections within chapters if they use H3
        ("####", "Sub-section") # For deeper nesting
    ]

    # Recursive splitter separators (priority order)
    SEPARATORS: List[str] = ["\n\n", "\n", ". ", "? ", "! ", " ", ""] # Added sentence endings
    
    # File handling
    INPUT_DIR: str = "data/input/markdown_files"
    OUTPUT_DIR: str = "data/output"
    TEMP_DIR: str = "data/temp"
    
    # Processing settings for i3 CPU
    BATCH_SIZE: int = 10  # Process files in small batches
    ENABLE_PARALLEL: bool = False  # Keep False for i3 (GPU-optional libraries are commented out)
    
    # LLM Integration
    OPENAI_API_KEY: str = "" # Keep for potential OpenAI models
    GEMINI_API_KEY: str = "" # For Gemini API calls (if direct API use, otherwise handled by Canvas env)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2" # Lightweight embedding model for semantic
    EMBEDDING_API_KEY: str = "" # To avoid pydantic validation error if in .env

    # LLM-based metadata enrichment for chunks
    ENABLE_LLM_METADATA_ENRICHMENT: bool = True
    LLM_METADATA_MODEL: str = "gemini-2.0-flash" # Model to use for summarization
    LLM_SUMMARY_PROMPT: str = "Summarize the following text concisely in 1-2 sentences, focusing on its main topic and key points, for improved search and retrieval accuracy:"

    # NEW: LLM-based Image Processing
    ENABLE_LLM_IMAGE_DESCRIPTION: bool = True # Flag to enable image description generation
    LLM_IMAGE_MODEL: str = "gemini-2.0-flash" # Model to use for image description (can be multimodal if needed)
    LLM_IMAGE_DESCRIPTION_PROMPT: str = "Describe the content of this image (or figure) in a concise sentence or two for a document retrieval system. Focus on key elements and context."
    
    # Quality thresholds
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.3 # Adjusted previously

    # Table-specific chunking settings
    TABLE_CHUNK_MAX_TOKENS: int = 75 
    TABLE_MERGE_HEADER_WITH_ROWS: bool = True
    
    class Config:
        env_file = ".env"

# Global config instance
config = ChunkingConfig()
