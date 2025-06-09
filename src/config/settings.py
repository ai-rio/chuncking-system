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
        ("#", "Part"),
        ("##", "Chapter"),
        ("###", "Section"),
        ("####", "Sub-section")
    ]

    # Recursive splitter separators (priority order)
    SEPARATORS: List[str] = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    
    # File handling
    INPUT_DIR: str = "data/input/markdown_files"
    OUTPUT_DIR: str = "data/output"
    TEMP_DIR: str = "data/temp"
    
    # Processing settings for i3 CPU
    BATCH_SIZE: int = 10
    ENABLE_PARALLEL: bool = False
    
    # LLM Integration
    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_API_KEY: str = ""
    
    # Quality thresholds
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.3 # ADJUSTED: Lowered threshold for more coherent semantic chunks

    # Table-specific chunking settings
    TABLE_CHUNK_MAX_TOKENS: int = 75 
    TABLE_MERGE_HEADER_WITH_ROWS: bool = True
    
    class Config:
        env_file = ".env"

config = ChunkingConfig()
