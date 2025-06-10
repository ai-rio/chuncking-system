import os
import json
import asyncio
import sys
import numpy as np

# Add the parent directory to the sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chunkers.hybrid_chunker import HybridChunker
from src.chunkers.evaluators import ChunkQualityEvaluator
import src.config.settings as config # Import the settings module
from langchain_core.documents import Document # Import Document class

async def tune_semantic_chunking():
    """
    Automates the semantic chunking fine-tuning process.
    It iterates through a range of SEMANTIC_SIMILARITY_THRESHOLD values
    and reports the semantic coherence score for each.
    """
    print("🚀 Starting Semantic Chunking Fine-Tuning Automation...")

    # Define the range of thresholds to test
    # Adjusted to explore more granularly in the lower-mid range
    threshold_values = np.arange(0.01, 0.51, 0.01).round(2)

    # Using sample_prose_document.md for proper semantic tuning
    INPUT_FILE_FOR_TUNING = os.path.join(config.config.INPUT_DIR, "sample_prose_document.md")

    # Ensure the input file exists
    if not os.path.exists(INPUT_FILE_FOR_TUNING):
        print(f"Error: Input file for tuning not found at {INPUT_FILE_FOR_TUNING}")
        print("Please ensure 'sample_prose_document.md' is in 'data/input/markdown_files/' or update INPUT_FILE_FOR_TUNING.")
        return

    # Load content once and remove carriage returns for consistency
    with open(INPUT_FILE_FOR_TUNING, 'r', encoding='utf-8') as f:
        content_to_chunk = f.read().replace('\r', '')

    results = []
    evaluator = ChunkQualityEvaluator()

    for threshold in threshold_values:
        print(f"\n--- Testing SEMANTIC_SIMILARITY_THRESHOLD: {threshold:.2f} ---")

        # Dynamically set the threshold for this iteration
        config.config.SEMANTIC_SIMILARITY_THRESHOLD = float(threshold)

        # Re-initialize HybridChunker to ensure it picks up the new threshold
        chunker = HybridChunker(enable_semantic=True)

        try:
            # Initial metadata for the document
            initial_metadata = {
                'source_file': os.path.basename(INPUT_FILE_FOR_TUNING),
                'document_type': 'markdown',
                'test_threshold': threshold
            }

            # Chunk the document (this is async)
            raw_chunks = await chunker.chunk_document(content_to_chunk, initial_metadata)
            
            # Manually apply post-processing steps from the removed _post_process_chunks
            processed_chunks = []
            global_chunk_index = 0
            for raw_chunk in raw_chunks:
                # Add basic metadata for evaluation purposes (tokens, chars, words)
                raw_chunk.metadata['chunk_index'] = global_chunk_index
                global_chunk_index += 1
                raw_chunk.metadata['chunk_tokens'] = chunker._token_length(raw_chunk.page_content)
                raw_chunk.metadata['chunk_chars'] = len(raw_chunk.page_content)
                raw_chunk.metadata['word_count'] = len(raw_chunk.page_content.split())
                
                # Filter very short chunks, similar to original _post_process_chunks logic
                if len(raw_chunk.page_content.strip().split()) < config.config.MIN_CHUNK_WORDS and \
                   raw_chunk.metadata.get('chunk_type') not in ['structural', 'visual']:
                    continue # Skip very short non-structural chunks

                processed_chunks.append(raw_chunk)

            print(f"Generated {len(processed_chunks)} chunks for threshold {threshold:.2f} after post-processing.")

            if not processed_chunks:
                coherence_score = 0.0
                avg_similarity = 0.0
            else:
                # Evaluate the chunks
                metrics = evaluator.evaluate_chunks(processed_chunks)
                coherence_score = metrics['semantic_coherence']['coherence_score']
                avg_similarity = metrics['semantic_coherence']['avg_similarity']
            
            print(f"  -> Semantic Coherence Score: {coherence_score:.3f}")
            print(f"  -> Average Similarity: {avg_similarity:.3f}")

            results.append({
                'threshold': threshold,
                'coherence_score': coherence_score,
                'avg_similarity': avg_similarity,
                'total_chunks': len(processed_chunks)
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

    # Find the best threshold based on coherence score (excluding 0-chunk cases)
    best_results_filtered = [r for r in results if r['total_chunks'] > 0]
    if best_results_filtered:
        # Prioritize coherence, but you might adjust this logic for optimal RAG balance
        best_result = max(best_results_filtered, key=lambda x: x['coherence_score'])
        print(f"\nRecommended Threshold: {best_result['threshold']:.2f} (Coherence Score: {best_result['coherence_score']:.3f}, Chunks: {best_result['total_chunks']})")
    else:
        print("\nNo valid chunks generated across thresholds. Consider adjusting chunking parameters.")

if __name__ == "__main__":
    asyncio.run(tune_semantic_chunking())

