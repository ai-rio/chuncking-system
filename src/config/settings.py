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
    Extract structured metadata (main_topic and key_entities) *strictly from the provided text chunk*.
    Your output MUST be a JSON object with only two keys: 'main_topic' and 'key_entities'.

    'main_topic': A concise, highly specific topic that accurately reflects the *primary subject matter or purpose* discussed in the provided text (string, max 12 words).
    - If the text is a document's main title or a major section header, infer the *overall subject* of the content that section introduces or summarizes.
    - If the text is a conclusion or summary, extract the core topic of the *findings or discussion presented within that text*.
    - For very short or generic titles/headers without rich accompanying text, infer the broader context (e.g., "Mixed media document handling").

    'key_entities': A list of important, highly specific noun phrases, named entities, or technical terms *directly mentioned and central to the core concepts* within the provided text chunk (list of strings).
    - Prioritize unique concepts, specific technologies, names, places, or domain-specific terminology.
    - **Exclude generic, common words (e.g., "document", "text", "images", "chapter", "section", "overview", "summary", "conclusion", "project", "system", "research") unless they are the *explicit and specific subject* of discussion within the chunk (e.g., "image processing techniques").**
    - If no highly specific and relevant entities are found within the provided text, return an empty list.

    Do NOT include any conversational text, explanations, or markdown formatting outside the JSON.
    Ensure the JSON is correctly formatted.

    Example:
    Text: "# **Document with Images**"
    Output: {{"main_topic": "Mixed media document handling", "key_entities": ["document processing", "image integration"]}}

    Example:
    Text: "# This document contains text and some images that need to be processed.  \\n## **Section 1: Introduction to AI**"
    Output: {{"main_topic": "Introduction to Artificial Intelligence", "key_entities": ["Artificial Intelligence", "AI", "machine learning", "smart assistants"]}}

    Example:
    Text: "# Artificial intelligence (AI) is transforming many aspects of our lives. From smart assistants to complex data analysis, AI is becoming increasingly prevalent.\\nHere's an example of an AI model's architecture:"
    Output: {{"main_topic": "Impact and prevalence of Artificial Intelligence", "key_entities": ["Artificial intelligence", "AI", "data analysis", "AI model architecture"]}}

    Example:
    Text: "## **Conclusion**\\n**Overview**\\nThis project began with the overarching goal of designing and implementing a novel method for automatically identifying and classifying different types of logical fallacies in text. Through the research and experimentation detailed in the preceding chapters, we have developed a system that utilizes a combination of natural language processing (NLP) techniques, including semantic analysis, dependency parsing, and machine learning classifiers, to achieve this objective. The system demonstrates promising results in distinguishing between various fallacy types, such as ad hominem, straw man, and appeal to emotion, outperforming several baseline models. While challenges remain, particularly in handling nuanced and context-dependent fallacies, the framework established in this project provides a strong foundation for future research in automated fallacy detection and critical thinking education.\\n\\n**Summary:**\\nThis project developed an NLP-based system for automated detection and classification of logical fallacies in text using semantic analysis, dependency parsing, and machine learning, achieving promising results despite challenges with nuanced fallacies. The system's framework provides a strong foundation for future research in automated fallacy detection and critical thinking education.\\n"
    Output: {{"main_topic": "Logical fallacy detection system development", "key_entities": ["logical fallacies", "NLP", "semantic analysis", "dependency parsing", "machine learning classifiers", "ad hominem", "straw man", "appeal to emotion"]}}
    
    Text: {text_content}
    Output:
    """
    
    class Config:
        env_file = ".env"

# Global config instance
config = ChunkingConfig()
