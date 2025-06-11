import os
import json
import asyncio
import re
import sys # Import sys
import traceback # Import the traceback module for detailed error logging

# Add the project root to Python's sys.path
project_root_path = os.path.abspath(os.path.dirname(__file__))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)
    print(f"Added '{project_root_path}' to sys.path.")


from src.chunkers.hybrid_chunker import HybridChunker
from src.chunkers.evaluators import ChunkQualityEvaluator
from src.utils.file_handler import FileHandler
from src.utils.metadata_enricher import MetadataEnricher
import src.config.settings as settings_module # Changed import statement
from langchain_core.documents import Document

# Access config instance through settings_module.config
config = settings_module.config

# Define file paths
INPUT_FILE = os.path.join(config.INPUT_DIR, "sample_document.md")
OUTPUT_CHUNKS_FILE = os.path.join(config.OUTPUT_DIR, "chunks", "sample_document_chunks.json")
QUALITY_REPORT_FILE = os.path.join(config.OUTPUT_DIR, "reports", "sample_document_quality_report.md")
PROCESSING_SUMMARY_FILE = os.path.join(config.OUTPUT_DIR, "reports", "sample_document_processing_summary.json")

# Define the async main function
async def main_async():
    print("🚀 Starting Hybrid Chunking System...")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(OUTPUT_CHUNKS_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(QUALITY_REPORT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSING_SUMMARY_FILE), exist_ok=True)
    # Ensure cache directory exists too
    os.makedirs(config.LLM_CACHE_DIR, exist_ok=True)


    # Initialize MetadataEnricher first, as it's now passed to HybridChunker
    metadata_enricher = MetadataEnricher()
    
    # Pass the metadata_enricher instance to HybridChunker
    chunker = HybridChunker(enable_semantic=True, metadata_enricher=metadata_enricher)
    evaluator = ChunkQualityEvaluator()


    # Create dummy document for demonstration if it doesn't exist (adjusted for image doc)
    if not os.path.exists(INPUT_FILE):
        print(f"Creating dummy test file: {INPUT_FILE}")
        # Use fixed, well-defined URLs for consistency and strip any potential hidden chars
        dummy_content = """
# Document with Images

This document contains text and some images that need to be processed.

## Section 1: Introduction to AI

Artificial intelligence (AI) is transforming many aspects of our lives. From smart assistants to complex data analysis, AI is becoming increasingly prevalent.

Here's an example of an AI model's architecture:
![Neural Network Architecture](https://example.com/images/neural_network.png)
A visual representation of a deep neural network, showing layers of interconnected nodes.

## Section 2: Data Visualization

Data visualization is key to understanding complex datasets. Charts and graphs help us interpret trends and patterns.

This chart illustrates market trends over the last quarter:
![Market Trends Chart](https://example.com/images/market_trends.png)
A bar chart depicting sales performance across different product categories for the first quarter.

## Conclusion

Images play a crucial role in conveying information effectively.
        """.strip() # .strip() here to remove any leading/trailing newlines in the multi-line string
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(dummy_content)

    try:
        # Load content from the input file
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            content = f.read().replace('\r', '') # Crucial: remove carriage returns on read
        
        # Initial metadata for the document
        initial_metadata = {
            'source_file': os.path.basename(INPUT_FILE),
            'document_type': 'markdown'
        }

        print(f"Chunking document: {INPUT_FILE}")
        # Call the async chunk_document method
        chunks = await chunker.chunk_document(content, initial_metadata)
        print(f"Generated {len(chunks)} chunks.")

        # Enrich chunks with LLM-generated summaries and structured metadata
        print("Enriching chunks with LLM summaries and structured metadata...")
        
        # Add a try-except block here to catch the specific error
        try:
            enriched_chunks = await metadata_enricher.enrich_chunks_with_llm_summaries_and_metadata(chunks)
        except Exception as e:
            print(f"\nCRITICAL ERROR during chunk enrichment in main_async: {e}")
            traceback.print_exc() # Print the full traceback
            sys.exit(1) # Exit with an error code

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
            "chunking_strategy_applied": "Hybrid (Table-aware, Header-Recursive, Semantic, LLM-Enriched, Image-Processed, Metadata-Extracted)", # Updated description
            "average_chunk_tokens": sum(c.metadata.get('chunk_tokens', 0) for c in enriched_chunks) / len(enriched_chunks) if enriched_chunks else 0
        }
        with open(PROCESSING_SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(processing_summary, f, indent=2, ensure_ascii=False)
        print(f"Processing summary saved to {PROCESSING_SUMMARY_FILE}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
    except Exception as e:
        # This broad catch-all is now less likely to hide critical errors from enrich_chunks,
        # as a more specific handler is above.
        print(f"An unexpected error occurred in main_async (general catch): {e}")
        traceback.print_exc() # Still good to have for other potential errors

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main_async())
