# main.py
import os
import json
from src.chunkers.hybrid_chunker import HybridMarkdownChunker
from src.chunkers.evaluators import ChunkQualityEvaluator
from src.utils.file_handler import FileHandler # Ensure this is imported and used correctly
from src.utils.metadata_enricher import MetadataEnricher # If MetadataEnricher is used elsewhere, keep it. Otherwise, it can be removed if unused.
from src.config.settings import config
from langchain_core.documents import Document # Import Document

# Define file paths
INPUT_FILE = os.path.join(config.INPUT_DIR, "sample_table_document.md") # Your test file
OUTPUT_CHUNKS_FILE = os.path.join(config.OUTPUT_DIR, "chunks", "sample_table_document_chunks.json")
QUALITY_REPORT_FILE = os.path.join(config.OUTPUT_DIR, "reports", "sample_table_document_quality_report.md")
PROCESSING_SUMMARY_FILE = os.path.join(config.OUTPUT_DIR, "reports", "sample_table_document_processing_summary.json")

def main():
    print("🚀 Starting Hybrid Chunking System...")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(OUTPUT_CHUNKS_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(QUALITY_REPORT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSING_SUMMARY_FILE), exist_ok=True)

    chunker = HybridMarkdownChunker()
    evaluator = ChunkQualityEvaluator()

    # Create dummy document for demonstration if it doesn't exist
    if not os.path.exists(INPUT_FILE):
        print(f"Creating dummy test file: {INPUT_FILE}")
        dummy_content = """
# Document with Tables

This is some introductory text before a table.

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Row 1 Col 1 | Row 1 Col 2 | Row 1 Col 3 |
| Row 2 Col 1 | Row 2 Col 2 | Row 2 Col 3 |
| This is a very long piece of text that spans multiple words in a single cell, to test how the chunker handles long content within a table. It should ideally split this row if it exceeds the token limit set for table chunks. | Another cell | Last cell of a long row |
| Row 4 Col 1 | Row 4 Col 2 | Row 4 Col 3 |

Some text after the first table.

## Another Section with a Small Table

| Key | Value |
|-----|-------|
| Apple | Fruit |
| Carrot| Veggie|

End of document.
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
        # Chunk the document
        chunks = chunker.chunk_document(content, initial_metadata)
        print(f"Generated {len(chunks)} chunks.")

        # Save chunks to JSON (for inspection)
        # Using the correct method from FileHandler: save_chunks
        FileHandler.save_chunks(chunks, OUTPUT_CHUNKS_FILE, format='json')
        print(f"Chunks saved to {OUTPUT_CHUNKS_FILE}")

        # Generate and save quality report
        report = evaluator.generate_report(chunks, QUALITY_REPORT_FILE)
        print("--- Quality Report ---")
        print(report)
        print("----------------------")
        print(f"Quality report saved to {QUALITY_REPORT_FILE}")

        # Generate and save processing summary (placeholder, implement as needed)
        processing_summary = {
            "file_name": os.path.basename(INPUT_FILE),
            "total_chunks": len(chunks),
            "chunking_strategy_applied": "Hybrid (Table-aware, Header-Recursive)",
            "average_chunk_tokens": sum(c.metadata.get('chunk_tokens', 0) for c in chunks) / len(chunks) if chunks else 0
        }
        # Correctly saving processing summary using json.dump
        with open(PROCESSING_SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(processing_summary, f, indent=2, ensure_ascii=False)
        print(f"Processing summary saved to {PROCESSING_SUMMARY_FILE}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
