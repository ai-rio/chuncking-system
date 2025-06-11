# src/config/settings.py
from pydantic_settings import BaseSettings
from typing import List, Dict, Any
import os

class ChunkingConfig(BaseSettings):
    # Chunking parameters optimized for i3/16GB and Gemini token constraints
    DEFAULT_CHUNK_SIZE: int = 250
    DEFAULT_CHUNK_OVERLAP: int = 150
    MAX_CHUNK_SIZE: int = 1200
    MIN_CHUNK_WORDS: int = 10
    MAX_CHUNK_WORDS: int = 600

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
    GEMINI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_API_KEY: str = ""

    # LLM-based metadata enrichment for chunks
    ENABLE_LLM_METADATA_ENRICHMENT: bool = True
    LLM_METADATA_MODEL: str = "gemini-2.0-flash"
    LLM_SUMMARY_PROMPT: str = "Summarize the following text concisely in 1-2 sentences, focusing on its main topic and key points, for improved search and retrieval accuracy:"

    # LLM-based Image Processing
    ENABLE_LLM_IMAGE_DESCRIPTION: bool = True
    LLM_IMAGE_MODEL: str = "gemini-2.0-flash"
    LLM_IMAGE_DESCRIPTION_PROMPT: str = "Describe the content of this image (or figure) in a concise sentence or two for a document retrieval system. Focus on key elements and context."
    
    # Quality thresholds
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.27

    # Table-specific chunking settings
    TABLE_CHUNK_MAX_TOKENS: int = 75 
    TABLE_MERGE_HEADER_WITH_ROWS: bool = True

    # LLM Caching Settings
    ENABLE_LLM_CACHE: bool = True
    LLM_CACHE_DIR: str = "data/cache/llm_responses"

    # NEW: LLM-based Automated Metadata Extraction Settings (STRONGER PROMPT)
    ENABLE_LLM_METADATA_EXTRACTION: bool = True
    LLM_EXTRACTION_MODEL: str = "gemini-2.0-flash"
    LLM_EXTRACTION_PROMPT: str = """
    Extract the following structured metadata from the provided text chunk.
    Your output MUST be a JSON object with only two keys: 'main_topic' and 'key_entities'.
    'main_topic': a concise, general topic of the text (string, max 10 words).
    'key_entities': a list of important noun phrases or named entities from the text (list of strings).
    Do NOT include any conversational text, explanations, or markdown formatting outside the JSON.
    Ensure the JSON is correctly formatted.

    Example:
    Text: "Apple Inc. announced its new iPhone 15 at the Cupertino event in September."
    Output: {{"main_topic": "Apple iPhone launch", "key_entities": ["Apple Inc.", "iPhone 15", "Cupertino", "September"]}}

    Text: {text_content}
    Output:
    """
    
    class Config:
        env_file = ".env"

# Global config instance
config = ChunkingConfig()
