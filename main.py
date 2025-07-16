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

# Import logging utilities
from src.utils.logger import setup_logging, get_chunking_logger, ChunkingLogger
from src.utils.validators import (
    validate_file_path, validate_directory_path, validate_chunk_size, 
    validate_chunk_overlap, validate_output_format
)
from src.utils.path_utils import get_markdown_manager, get_advanced_quality_enhancement_manager

# Setup logging
app_logger = setup_logging(level="INFO")
chunking_logger = get_chunking_logger()
path_manager = get_markdown_manager()

# Import necessary components (ensure these imports are correct and accessible)
from src.chunkers.hybrid_chunker import HybridMarkdownChunker
from src.utils.file_handler import FileHandler
from src.utils.metadata_enricher import MetadataEnricher
from src.chunkers.evaluators import ChunkQualityEvaluator, EnhancedChunkQualityEvaluator
from src.config.settings import config
from src.exceptions import ChunkingError, FileHandlingError, ValidationError

def progress_callback(current: int, total: int, filename: str):
    """Callback for progress tracking during file processing."""
    chunking_logger.log_batch_progress(current, total, os.path.basename(filename))

def main():
    chunking_logger.start_operation("main_application")
    app_logger.info("Starting document chunking system")
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
        '--chunk-overlap',
        type=int,
        default=config.DEFAULT_CHUNK_OVERLAP,
        help="Number of tokens to overlap between chunks for better context preservation."
    )
    parser.add_argument(
        '--format',
        choices=['json', 'csv', 'pickle'],
        default='json',
        help="Output format for the generated chunks."
    )
    parser.add_argument(
        '--create-project-folder',
        action='store_true',
        default=True,
        help="Create a timestamped project folder for each run."
    )
    parser.add_argument(
        '--auto-enhance',
        action='store_true',
        default=False,
        help="Automatically enhance chunks if quality is below threshold."
    )
    parser.add_argument(
        '--jina-api-key',
        help="Jina AI API key (automatically enables Jina embeddings). Or set JINA_API_KEY environment variable."
    )
    parser.add_argument(
        '--jina-model',
        default="jina-embeddings-v2-base-en",
        help="Jina embedding model to use."
    )
    parser.add_argument(
        '--hybrid-mode',
        action='store_true',
        default=False,
        help="Use hybrid mode comparing TF-IDF and Jina embeddings."
    )
    
    args = parser.parse_args()
    app_logger.debug("Arguments parsed", args=vars(args))

    # Validate input file
    try:
        if not args.input_file:
            raise ValidationError(
                "An input book file must be specified using --input-file",
                field="input_file"
            )
        
        # Validate input file path
        input_file_path = validate_file_path(
            args.input_file, 
            must_exist=True, 
            extensions=path_manager.MARKDOWN_EXTENSIONS
        )
        
        # Validate chunk parameters
        chunk_size = validate_chunk_size(args.chunk_size)
        chunk_overlap = validate_chunk_overlap(args.chunk_overlap, chunk_size)
        output_format = validate_output_format(args.format)
        
        app_logger.info("Input validation completed", 
                       input_file=str(input_file_path),
                       chunk_size=chunk_size,
                       output_format=output_format)
                       
    except (ValidationError, FileHandlingError) as e:
        app_logger.error("Input validation failed", error=str(e), error_type=type(e).__name__)
        chunking_logger.end_operation("main_application", success=False, error="Input validation failed")
        return
    except Exception as e:
        app_logger.error("Unexpected error during validation", error=str(e))
        chunking_logger.end_operation("main_application", success=False, error="Unexpected validation error")
        return

    # Setup output directories using path manager
    try:
        output_paths = path_manager.create_markdown_output_paths(
            input_file_path, 
            args.output_dir, 
            create_project_folder=args.create_project_folder
        )
        app_logger.info("Output directory structure created", 
                       base_dir=str(output_paths['base']),
                       chunks_dir=str(output_paths['chunks']),
                       reports_dir=str(output_paths['reports']),
                       project_folder=str(output_paths.get('project_folder', 'None')))
    except Exception as e:
        app_logger.error("Failed to create output directories", error=str(e), output_dir=args.output_dir)
        chunking_logger.end_operation("main_application", success=False, error="Directory creation failed")
        return

    app_logger.info("🚀 Document Chunking System Starting for Book Processing...")
    app_logger.info("Configuration loaded", 
                   input_file=input_file_path,
                   output_dir=args.output_dir,
                   chunk_size=args.chunk_size,
                   output_format=args.format,
                   create_project_folder=args.create_project_folder,
                   auto_enhance=args.auto_enhance)
    
    chunking_logger.start_operation("initialization")

    try:
        # Initialize chunker and evaluator with validated parameters
        chunker = HybridMarkdownChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # enable_semantic=False # Explicitly set to False as libraries are not installed
        )
        # Initialize quality evaluator - use Enhanced version if Jina API key is provided
        jina_api_key = args.jina_api_key or os.getenv('JINA_API_KEY')
        if jina_api_key:
            app_logger.info("Initializing Enhanced Quality Evaluator with Jina AI embeddings")
            evaluator = EnhancedChunkQualityEvaluator(
                use_jina_embeddings=True,
                jina_api_key=jina_api_key,
                jina_model=args.jina_model,
                fallback_to_tfidf=True,
                enable_embedding_cache=True,
                hybrid_mode=args.hybrid_mode
            )
        else:
            app_logger.info("Using standard TF-IDF based quality evaluator")
            evaluator = ChunkQualityEvaluator()
        chunking_logger.end_operation("initialization", success=True)
        app_logger.debug("Chunker and evaluator initialized successfully")
    except Exception as e:
        chunking_logger.log_error(e, "initialization")
        chunking_logger.end_operation("initialization", success=False)
        app_logger.debug("Full traceback:", exc_info=True)
        chunking_logger.end_operation("main_application", success=False, error="Initialization failed")
        return

    chunking_logger.start_operation("file_processing", file_path=input_file_path)

    total_chunks = 0
    all_chunks_for_evaluation = [] # Collect all chunks for overall quality evaluation
    
    try:
        # Read the entire book content
        chunking_logger.start_operation("file_reading", file_path=str(input_file_path))
        app_logger.debug("Attempting to read file", file_path=str(input_file_path))
        
        with open(input_file_path, 'r', encoding='utf-8') as f:
            book_content = f.read()
        
        file_size = len(book_content)
        chunking_logger.end_operation("file_reading", success=True, file_size_bytes=file_size)
        app_logger.debug("File read successfully", file_size_bytes=file_size)

        # Prepare base metadata for the book
        book_metadata = {
            'source_file': str(input_file_path),
            'file_size': file_size,
            'processing_strategy': 'hybrid_book_chunking',
            'book_title': input_file_path.stem.replace('_', ' ').title() # Derive title from filename
        }

        # Chunk the entire book content
        chunking_logger.start_operation("chunking", strategy="hybrid")
        app_logger.info("Starting document chunking...", file_path=input_file_path)
        
        chunks = chunker.chunk_document(book_content, book_metadata)
        
        chunking_logger.end_operation("chunking", success=True, chunks_generated=len(chunks))
        chunking_logger.log_chunk_stats(chunks, "hybrid_chunking")
        app_logger.info("Document chunking completed", chunks_generated=len(chunks))

        # Enrich metadata for each chunk
        chunking_logger.start_operation("metadata_enrichment")
        enriched_chunks = []
        app_logger.debug("Enriching metadata...")
        
        for i, chunk in enumerate(chunks):
            try:
                # The chunk's metadata already contains 'Header X' from MarkdownHeaderTextSplitter
                # Add general book metadata as well
                enriched_chunk = MetadataEnricher.enrich_chunk(chunk, book_metadata)
                enriched_chunks.append(enriched_chunk)
            except Exception as e:
                app_logger.error("Failed to enrich chunk metadata", chunk_index=i, error=str(e))
                # Use original chunk if enrichment fails
                enriched_chunks.append(chunk)
        
        chunking_logger.end_operation("metadata_enrichment", success=True, chunks_processed=len(enriched_chunks))
        app_logger.debug("Metadata enrichment completed")
        
        all_chunks_for_evaluation.extend(enriched_chunks) # Add to list for overall evaluation

        # Save chunks using proper output paths
        if output_format == 'json':
            output_path = output_paths['chunks_json']
        elif output_format == 'csv':
            output_path = output_paths['chunks_csv']
        else:  # pickle
            output_path = output_paths['chunks_pickle']
        
        app_logger.debug("Attempting to save chunks", output_path=str(output_path))
        FileHandler.save_chunks(enriched_chunks, str(output_path), output_format)
        app_logger.debug("Chunks saved successfully")

        # Update statistics
        total_chunks += len(enriched_chunks)
        chunking_logger.log_file_processed(str(input_file_path), len(enriched_chunks), file_size)

    except Exception as e:
        chunking_logger.log_error(e, "file_processing", file_path=str(input_file_path))
        app_logger.debug("Full traceback:", exc_info=True)
        chunking_logger.end_operation("file_processing", success=False)
        chunking_logger.end_operation("main_application", success=False, error="File processing failed")
        return
    
    chunking_logger.end_operation("file_processing", success=True, chunks_generated=total_chunks)
    app_logger.info("Book processing completed successfully", total_chunks=total_chunks)

    # Perform overall quality evaluation on all generated chunks
    if all_chunks_for_evaluation:
        chunking_logger.start_operation("quality_evaluation")
        app_logger.info("Starting overall quality evaluation...")
        try:
            overall_evaluation_metrics = evaluator.evaluate_chunks(all_chunks_for_evaluation)
            chunking_logger.log_quality_metrics(overall_evaluation_metrics, str(input_file_path))
            
            # Save a detailed overall evaluation report
            overall_report_path = output_paths['quality_report']
            evaluator.generate_report(all_chunks_for_evaluation, str(overall_report_path))
            chunking_logger.end_operation("quality_evaluation", success=True, report_path=str(overall_report_path))
            
            # Auto-enhance chunks if quality is poor and auto-enhance is enabled
            if args.auto_enhance and overall_evaluation_metrics.get('overall_score', 0) < 60:
                chunking_logger.start_operation("quality_enhancement")
                app_logger.info("Quality score below threshold, starting auto-enhancement...")
                try:
                    enhancement_manager = get_advanced_quality_enhancement_manager(path_manager)
                    enhancement_results = enhancement_manager.comprehensive_enhancement(
                        book_content,
                        all_chunks_for_evaluation,
                        overall_evaluation_metrics,
                        output_paths,
                        book_metadata
                    )
                    
                    app_logger.info("Quality enhancement completed",
                                   original_score=enhancement_results['original_score'],
                                   final_score=enhancement_results['final_score'],
                                   improvements=enhancement_results['improvements_made'])
                    
                    # Generate enhanced quality report
                    if enhancement_results['final_score'] > enhancement_results['original_score']:
                        enhanced_report_path = output_paths['reports'] / f"{input_file_path.stem}_enhanced_quality_report.md"
                        evaluator.generate_report(enhancement_results['final_chunks'], str(enhanced_report_path))
                        app_logger.info("Enhanced quality report generated", report_path=str(enhanced_report_path))
                    
                    chunking_logger.end_operation("quality_enhancement", success=True)
                except Exception as e:
                    chunking_logger.log_error(e, "quality_enhancement")
                    app_logger.debug("Full traceback:", exc_info=True)
                    chunking_logger.end_operation("quality_enhancement", success=False)
                    
        except Exception as e:
            chunking_logger.log_error(e, "quality_evaluation")
            app_logger.debug("Full traceback:", exc_info=True)
            chunking_logger.end_operation("quality_evaluation", success=False)
    else:
        app_logger.warning("No chunks generated, skipping overall quality evaluation")

    # Save a simple processing summary JSON for this book
    processing_summary_path = output_paths['reports'] / f"{input_file_path.stem}_processing_summary.json"
    app_logger.debug("Attempting to save processing summary", summary_path=str(processing_summary_path))
    try:
        with open(processing_summary_path, 'w') as f:
            json.dump([
                {
                    'file': input_file_path.name,
                    'chunks': total_chunks,
                    'avg_chunk_chars': sum(len(c.page_content) for c in all_chunks_for_evaluation) // total_chunks if total_chunks > 0 else 0,
                    'quality_score': overall_evaluation_metrics.get('overall_score', 0) if all_chunks_for_evaluation else 0
                }
            ], f, indent=2)
        app_logger.debug("Processing summary saved successfully")
    except Exception as e:
        app_logger.error("Error saving processing summary", error=str(e), summary_path=str(processing_summary_path))
        app_logger.debug("Full traceback:", exc_info=True)
    
    # Generate performance report
    try:
        performance_report_path = output_paths['reports'] / f"{input_file_path.stem}_performance_report.md"
        performance_report = chunker.get_performance_report()
        
        with open(performance_report_path, 'w', encoding='utf-8') as f:
            f.write(performance_report)
        
        app_logger.info("Performance report generated", report_path=str(performance_report_path))
    except Exception as e:
        app_logger.error("Failed to generate performance report", error=str(e))
    
    app_logger.info("Processing completed successfully", 
                   processing_summary_path=str(processing_summary_path),
                   total_chunks=total_chunks,
                   performance_report=str(performance_report_path) if 'performance_report_path' in locals() else None)
    
    chunking_logger.end_operation("main_application", success=True, total_chunks=total_chunks)


if __name__ == "__main__":
    main()