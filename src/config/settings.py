# src/config/settings.py
from pydantic_settings import BaseSettings # Already updated in previous step
from typing import List, Dict, Any
import os

class ChunkingConfig(BaseSettings):
    # Chunking parameters optimized for i3/16GB and Gemini token constraints
    # Targeting a smaller chunk size for RAG effectiveness within Gemini limits
    DEFAULT_CHUNK_SIZE: int = 800 # Set a target for tokens, allowing space for prompt
    DEFAULT_CHUNK_OVERLAP: int = 150 # Moderate overlap for context
    MAX_CHUNK_SIZE: int = 1200 # Absolute max for very long sentences/paragraphs
    MIN_CHUNK_SIZE: int = 50 # Minimum chunk size in characters

    # Markdown header levels to split on, comprehensive for a book structure
    HEADER_LEVELS: List[tuple] = [
        ("#", "Part"),       # e.g., "Part I: The New Reality of Markets"
        ("##", "Chapter"),    # e.g., "## 1: THE DISCOVERY JOURNEY"
        ("###", "Section"),   # For potential sections within chapters if they use H3
        ("####", "Sub-section") # For deeper nesting
        # Add more levels if your book has deeper hierarchy (e.g., "#####", "######")
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
    
    # LLM Integration (optional, but keep OPENAI_API_KEY for potential future use)
    OPENAI_API_KEY: str = ""
    # EMBEDDING_MODEL is commented out in requirements.txt
    # EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2" # Lightweight

    # Quality thresholds
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.8
    MIN_CHUNK_WORDS: int = 10 # Adjusted to be slightly lower for more granular chunks
    MAX_CHUNK_WORDS: int = 600 # Adjusted for new chunk size
    
    class Config:
        env_file = ".env"

# Global config instance
config = ChunkingConfig()
