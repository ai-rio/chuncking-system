# main.py
#!/usr/bin/env python3
"""
Main application for document chunking system
Optimized for i3/16GB hardware for processing entire books,
with consideration for Gemini token constraints.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
import json

# Add src to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import necessary components (ensure these imports are correct and accessible)
from src.chunkers.hybrid_chunker import HybridMarkdownChunker
from src.utils.file_handler import FileHandler
from src.utils.metadata_enricher import MetadataEnricher
from src.chunkers.evaluators import ChunkQualityEvaluator  # Fixed: changed 'evaluator' to 'evaluators'
from src.config.settings import config

def progress_callback(current: int, total: int, filename: str):
    """Callback for progress tracking during file processing."""
    print(f"Progress: {current}/{total} - Processing: {os.path.basename(filename)}")

def main():
    print("--- Starting main function execution ---") # Debug point 1
    parser = argparse.ArgumentParser(description="Document Chunking System for Books")
    parser.add_argument(
        '--input-file',
        help="Path to the single Markdown book file to be chunked. E.g., 'data/input/markdown_files/decoupling_book.md'"
    )
    parser.add_argument(
        '--output-dir',
        default=config.OUTPUT_DIR,
        help="Output directory for generated chunks and reports."
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=config.DEFAULT_CHUNK_SIZE, # This now defaults to 800 tokens as per settings.py
        help="Maximum chunk size in tokens, designed to fit Gemini token constraints for RAG."
    )
    parser.add_argument(
        '--format',
        choices=['json', 'csv', 'pickle'],
        default='json',
        help="Output format for the generated chunks."
    )
    
    args = parser.parse_args()
    print(f"--- Arguments parsed: {args} ---") # Debug point 2

    if not args.input_file:
        print("❌ Error: An input book file must be specified using --input-file.")
        print("Example: python main.py --input-file data/input/markdown_files/decoupling_book.md")
        return

    book_file_path = args.input_file
    if not os.path.exists(book_file_path):
        print(f"❌ Error: Input file not found at {book_file_path}")
        return

    # Setup output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'chunks')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'reports')).mkdir(parents=True, exist_ok=True)

    print("🚀 Document Chunking System Starting for Book Processing...")
    print(f"📚 Input Book File: {book_file_path}")
    print(f"📁 Output Directory: {args.output_dir}")
    print(f"⚙️  Target Chunk Size: {args.chunk_size} tokens (for Gemini constraint)")
    print(f"📄 Output Format: {args.format}")
    print("--- Initializing chunker and evaluator... ---") # Debug point 3

    try:
        # Initialize chunker and evaluator
        chunker = HybridMarkdownChunker(
            chunk_size=args.chunk_size,
            # enable_semantic=False # Explicitly set to False as libraries are not installed
        )
        evaluator = ChunkQualityEvaluator()
        print("--- Chunker and evaluator initialized successfully. ---") # Debug point 4
    except Exception as e:
        print(f"❌ Error during chunker/evaluator initialization: {e}")
        import traceback
        traceback.print_exc() # Print full traceback
        return

    print("\n🔄 Processing book file...")

    total_chunks = 0
    all_chunks_for_evaluation = [] # Collect all chunks for overall quality evaluation
    
    try:
        # Read the entire book content
        print(f"--- Attempting to read file: {book_file_path} ---") # Debug point 5
        with open(book_file_path, 'r', encoding='utf-8') as f:
            book_content = f.read()
        print("--- File read successfully. ---") # Debug point 6

        # Prepare base metadata for the book
        book_metadata = {
            'source_file': book_file_path,
            'file_size': len(book_content),
            'processing_strategy': 'hybrid_book_chunking',
            'book_title': Path(book_file_path).stem.replace('_', ' ').title() # Derive title from filename
        }

        # Chunk the entire book content
        print("--- Starting document chunking... ---") # Debug point 7
        chunks = chunker.chunk_document(book_content, book_metadata)
        print(f"--- Document chunking completed. Generated {len(chunks)} chunks. ---") # Debug point 8

        # Enrich metadata for each chunk
        enriched_chunks = []
        print("--- Enriching metadata... ---") # Debug point 9
        for chunk in chunks:
            # The chunk's metadata already contains 'Header X' from MarkdownHeaderTextSplitter
            # Add general book metadata as well
            enriched_chunk = MetadataEnricher.enrich_chunk(chunk, book_metadata)
            enriched_chunks.append(enriched_chunk)
        print("--- Metadata enrichment completed. ---") # Debug point 10
        
        all_chunks_for_evaluation.extend(enriched_chunks) # Add to list for overall evaluation

        # Save chunks
        output_filename = f"{Path(book_file_path).stem}_chunks.{args.format}"
        output_path = os.path.join(args.output_dir, 'chunks', output_filename)
        
        print(f"--- Attempting to save chunks to: {output_path} ---") # Debug point 11
        FileHandler.save_chunks(enriched_chunks, output_path, args.format)
        print("--- Chunks saved successfully. ---") # Debug point 12

        # Update statistics
        total_chunks += len(enriched_chunks)
        print(f"   ✅ Generated {len(enriched_chunks)} chunks for '{os.path.basename(book_file_path)}'")

    except Exception as e:
        print(f"❌ Error during file processing of {book_file_path}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback
    
    print(f"\n✅ Book processing completed!")
    print(f"📊 Total chunks generated: {total_chunks}")

    # Perform overall quality evaluation on all generated chunks
    if all_chunks_for_evaluation:
        print("--- Starting overall quality evaluation... ---") # Debug point 13
        try:
            overall_evaluation_metrics = evaluator.evaluate_chunks(all_chunks_for_evaluation)
            print(f"📈 Overall Chunk Quality Score: {overall_evaluation_metrics.get('overall_score', 0):.1f}/100")
            
            # Save a detailed overall evaluation report
            overall_report_path = os.path.join(args.output_dir, 'reports', f"{Path(book_file_path).stem}_quality_report.md")
            evaluator.generate_report(all_chunks_for_evaluation, overall_report_path)
            print("--- Overall quality evaluation completed and report generated. ---") # Debug point 14
        except Exception as e:
            print(f"❌ Error during overall quality evaluation: {e}")
            import traceback
            traceback.print_exc() # Print full traceback
    else:
        print("⚠️ No chunks generated, skipping overall quality evaluation.")

    # Save a simple processing summary JSON for this book
    processing_summary_path = os.path.join(args.output_dir, 'reports', f"{Path(book_file_path).stem}_processing_summary.json")
    print(f"--- Attempting to save processing summary to: {processing_summary_path} ---") # Debug point 15
    try:
        with open(processing_summary_path, 'w') as f:
            json.dump([
                {
                    'file': os.path.basename(book_file_path),
                    'chunks': total_chunks,
                    'avg_chunk_chars': sum(len(c.page_content) for c in all_chunks_for_evaluation) // total_chunks if total_chunks > 0 else 0,
                    'quality_score': overall_evaluation_metrics.get('overall_score', 0) if all_chunks_for_evaluation else 0
                }
            ], f, indent=2)
        print(f"--- Processing summary saved successfully. ---") # Debug point 16
    except Exception as e:
        print(f"❌ Error saving processing summary: {e}")
        import traceback
        traceback.print_exc() # Print full traceback
    
    print(f"📋 Processing report saved to: {processing_summary_path}")
    print("--- End of main function execution ---") # Debug point 17


if __name__ == "__main__":
    main()
