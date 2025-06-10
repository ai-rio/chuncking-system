import os
import json
import asyncio
import sys
import numpy as np # Added this import

# Add the parent directory to the sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chunkers.hybrid_chunker import HybridChunker
from src.chunkers.evaluators import ChunkQualityEvaluator
import src.config.settings as config # Import the settings module

async def tune_semantic_chunking():
    """
    Automates the semantic chunking fine-tuning process.
    It iterates through a range of SEMANTIC_SIMILARITY_THRESHOLD values
    and reports the semantic coherence score for each.
    """
    print("🚀 Starting Semantic Chunking Fine-Tuning Automation...")

    # Define the range of thresholds to test
    # The RAG Specialist suggests a range that includes both lower and higher similarities
    threshold_values = np.arange(0.01, 0.41, 0.02).round(2) # From 0.01 to 0.40, step 0.02

    # Using sample_image_document.md for demonstration, but ideally, you'd use a purely prose document
    INPUT_FILE_FOR_TUNING = os.path.join(config.config.INPUT_DIR, "sample_prose_document.md")

    # Ensure the input file exists
    if not os.path.exists(INPUT_FILE_FOR_TUNING):
        print(f"Error: Input file for tuning not found at {INPUT_FILE_FOR_TUNING}")
        print("Please ensure 'sample_prose_document.md' is in 'data/input/markdown_files/' or update INPUT_FILE_FOR_TUNING.")
        return

    # Load content once
    with open(INPUT_FILE_FOR_TUNING, 'r', encoding='utf-8') as f:
        content_to_chunk = f.read()

    results = []
    evaluator = ChunkQualityEvaluator()

    for threshold in threshold_values:
        print(f"\n--- Testing SEMANTIC_SIMILARITY_THRESHOLD: {threshold:.2f} ---")

        # Dynamically set the threshold for this iteration
        # IMPORTANT: When modifying global settings like this, ensure it's done before
        # any component that *uses* this setting is initialized or re-initialized if needed.
        config.config.SEMANTIC_SIMILARITY_THRESHOLD = float(threshold)

        # Re-initialize HybridChunker to ensure it picks up the new threshold
        # (especially important if _init_splitters or other methods depend on it directly)
        chunker = HybridChunker(enable_semantic=True)

        try:
            # Initial metadata for the document
            initial_metadata = {
                'source_file': os.path.basename(INPUT_FILE_FOR_TUNING),
                'document_type': 'markdown',
                'test_threshold': threshold # Add threshold to metadata for debugging
            }

            # Chunk the document (this is async)
            chunks = await chunker.chunk_document(content_to_chunk, initial_metadata)
            print(f"Generated {len(chunks)} chunks for threshold {threshold:.2f}.")

            # Evaluate the chunks
            metrics = evaluator.evaluate_chunks(chunks)
            coherence_score = metrics['semantic_coherence']['coherence_score']
            avg_similarity = metrics['semantic_coherence']['avg_similarity']
            
            print(f"  -> Semantic Coherence Score: {coherence_score:.3f}")
            print(f"  -> Average Similarity: {avg_similarity:.3f}")

            results.append({
                'threshold': threshold,
                'coherence_score': coherence_score,
                'avg_similarity': avg_similarity,
                'total_chunks': len(chunks)
            })

        except Exception as e:
            print(f"Error processing with threshold {threshold:.2f}: {e}")
            results.append({
                'threshold': threshold,
                'coherence_score': 0.0,
                'avg_similarity': 0.0,
                'total_chunks': 0,
                'error': str(e)
            })

    print("\n=== Semantic Fine-Tuning Results ===")
    for result in results:
        print(f"Threshold: {result['threshold']:.2f}, Coherence Score: {result['coherence_score']:.3f}, Avg Similarity: {result['avg_similarity']:.3f}, Chunks: {result['total_chunks']}")

    # Optional: Find the best threshold
    best_result = max(results, key=lambda x: x['coherence_score'])
    print(f"\nRecommended Threshold: {best_result['threshold']:.2f} (Coherence Score: {best_result['coherence_score']:.3f})")

if __name__ == "__main__":
    asyncio.run(tune_semantic_chunking())

