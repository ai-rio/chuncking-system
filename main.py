import os
import json
import asyncio # Import asyncio for running async functions
from src.chunkers.hybrid_chunker import HybridMarkdownChunker
from src.chunkers.evaluators import ChunkQualityEvaluator
from src.utils.file_handler import FileHandler
from src.utils.metadata_enricher import MetadataEnricher
from src.config.settings import config
from langchain_core.documents import Document

# Define file paths
INPUT_FILE = os.path.join(config.INPUT_DIR, "sample_image_document.md") # <<< CHANGED THIS LINE
OUTPUT_CHUNKS_FILE = os.path.join(config.OUTPUT_DIR, "chunks", "sample_image_document_chunks.json") # <<< CHANGED THIS LINE
QUALITY_REPORT_FILE = os.path.join(config.OUTPUT_DIR, "reports", "sample_image_document_quality_report.md") # <<< CHANGED THIS LINE
PROCESSING_SUMMARY_FILE = os.path.join(config.OUTPUT_DIR, "reports", "sample_image_document_processing_summary.json") # <<< CHANGED THIS LINE

# Define the async main function
async def main_async():
    print("🚀 Starting Hybrid Chunking System...")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(OUTPUT_CHUNKS_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(QUALITY_REPORT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSING_SUMMARY_FILE), exist_ok=True)

    # Enable semantic and ensure LLM image description is enabled via settings
    chunker = HybridMarkdownChunker(enable_semantic=True)
    evaluator = ChunkQualityEvaluator()
    metadata_enricher = MetadataEnricher()

    # Create dummy document for demonstration if it doesn't exist (adjusted for image doc)
    if not os.path.exists(INPUT_FILE):
        print(f"Creating dummy test file: {INPUT_FILE}")
        dummy_content = """
# Document with Images

This document contains text and some images that need to be processed.

## Section 1: Introduction to AI

Artificial intelligence (AI) is transforming many aspects of our lives. From smart assistants to complex data analysis, AI is becoming increasingly prevalent.

Here's an example of an AI model's architecture:
![Neural Network Architecture](https://placehold.co/600x400/FF0000/FFFFFF?text=Neural%20Network)
A visual representation of a deep neural network, showing layers of interconnected nodes.

## Section 2: Data Visualization

Data visualization is key to understanding complex datasets. Charts and graphs help us interpret trends and patterns.

This chart illustrates market trends over the last quarter:
![Market Trends Chart](https://placehold.co/800x500/00FF00/000000?text=Market%20Trends%20Q1)
A bar chart depicting sales performance across different product categories for the first quarter.

## Conclusion

Images play a crucial role in conveying information effectively.
        """
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(dummy_content)

    try:
        # Load content from the input file
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Initial metadata for the document
        initial_metadata = {
            'source_file': os.path.basename(INPUT_FILE),
            'document_type': 'markdown'
        }

        print(f"Chunking document: {INPUT_FILE}")
        # Call the async chunk_document method
        chunks = await chunker.chunk_document(content, initial_metadata)
        print(f"Generated {len(chunks)} chunks.")

        # Enrich chunks with LLM-generated summaries (this is already async)
        print("Enriching chunks with LLM summaries...")
        enriched_chunks = await metadata_enricher.enrich_chunks_with_llm_summaries(chunks)
        print(f"Finished enriching {len(enriched_chunks)} chunks.")

        FileHandler.save_chunks(enriched_chunks, OUTPUT_CHUNKS_FILE, format='json')
        print(f"Chunks saved to {OUTPUT_CHUNKS_FILE}")

        report = evaluator.generate_report(enriched_chunks, QUALITY_REPORT_FILE)
        print("--- Quality Report ---")
        print(report)
        print("----------------------")
        print(f"Quality report saved to {QUALITY_REPORT_FILE}")

        processing_summary = {
            "file_name": os.path.basename(INPUT_FILE),
            "total_chunks": len(enriched_chunks),
            "chunking_strategy_applied": "Hybrid (Table-aware, Header-Recursive, Semantic, LLM-Enriched, Image-Processed)", # Updated description
            "average_chunk_tokens": sum(c.metadata.get('chunk_tokens', 0) for c in enriched_chunks) / len(enriched_chunks) if enriched_chunks else 0
        }
        with open(PROCESSING_SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(processing_summary, f, indent=2, ensure_ascii=False)
        print(f"Processing summary saved to {PROCESSING_SUMMARY_FILE}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main_async())
