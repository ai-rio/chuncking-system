# src/config/settings.py
from pydantic_settings import BaseSettings
from typing import List, Optional
from src.utils.security import SecurityConfig

class ChunkingConfig(BaseSettings):
    # Chunking parameters
    DEFAULT_CHUNK_SIZE: int = 800
    DEFAULT_CHUNK_OVERLAP: int = 150
    MAX_CHUNK_SIZE: int = 1200
    MIN_CHUNK_SIZE: int = 50
    MIN_CHUNK_WORDS: int = 10
    MAX_CHUNK_WORDS: int = 600
    
    # Allow dynamic chunk size and overlap parameters
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

    # Markdown header levels
    HEADER_LEVELS: List[tuple] = [
        ("#", "Part"),
        ("##", "Chapter"),
        ("###", "Section"),
        ("####", "Sub-section")
    ]

    # Recursive splitter separators
    SEPARATORS: List[str] = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    # File handling
    INPUT_DIR: str = "data/input/markdown_files"
    OUTPUT_DIR: str = "data/output"
    TEMP_DIR: str = "data/temp"

    # Processing settings
    BATCH_SIZE: int = 10
    ENABLE_PARALLEL: bool = False

    # LLM Integration
    OPENAI_API_KEY: str = ""

    # Quality thresholds
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.8

    # Phase 3 Features
    enable_caching: bool = True
    enable_security: bool = True
    enable_monitoring: bool = True
    cache_ttl_hours: int = 24
    max_file_size_mb: int = 100
    quality_threshold: float = 0.7
    security_config: Optional[SecurityConfig] = None

    class Config:
        env_file = ".env"

# Global config instance
config = ChunkingConfig()

