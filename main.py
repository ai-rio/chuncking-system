import os
import json
import asyncio # Import asyncio for running async functions
from src.chunkers.hybrid_chunker import HybridMarkdownChunker
from src.chunkers.evaluators import ChunkQualityEvaluator
from src.utils.file_handler import FileHandler
from src.utils.metadata_enricher import MetadataEnricher # Import the MetadataEnricher
from src.config.settings import config
from langchain_core.documents import Document

# Define file paths
INPUT_FILE = os.path.join(config.INPUT_DIR, "sample_prose_document.md")
OUTPUT_CHUNKS_FILE = os.path.join(config.OUTPUT_DIR, "chunks", "sample_prose_document_chunks.json")
QUALITY_REPORT_FILE = os.path.join(config.OUTPUT_DIR, "reports", "sample_prose_document_quality_report.md")
PROCESSING_SUMMARY_FILE = os.path.join(config.OUTPUT_DIR, "reports", "sample_prose_document_processing_summary.json")

# Define the async main function
async def main_async():
    print("🚀 Starting Hybrid Chunking System...")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(OUTPUT_CHUNKS_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(QUALITY_REPORT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSING_SUMMARY_FILE), exist_ok=True)

    chunker = HybridMarkdownChunker(enable_semantic=True)
    evaluator = ChunkQualityEvaluator()
    metadata_enricher = MetadataEnricher() # Initialize the MetadataEnricher

    # Create dummy document for demonstration if it doesn't exist
    if not os.path.exists(INPUT_FILE):
        print(f"Creating dummy test file: {INPUT_FILE}")
        dummy_content = """
# The Future of Artificial Intelligence

Artificial intelligence (AI) is rapidly transforming industries and daily life. From self-driving cars to advanced medical diagnostics, AI's capabilities are expanding at an unprecedented pace. This technology promises to solve complex problems, enhance efficiency, and unlock new frontiers of innovation. However, its development also presents ethical and societal challenges that require careful consideration.

## Machine Learning and Deep Learning

At the core of many AI advancements are machine learning (ML) and deep learning (DL). Machine learning algorithms enable systems to learn from data without explicit programming. Deep learning, a subset of ML, uses neural networks with multiple layers to learn complex patterns. These techniques are behind breakthroughs in image recognition, natural language processing, and predictive analytics. The sheer volume of data available today fuels these algorithms, allowing them to achieve remarkable accuracy.

### The Role of Large Language Models (LLMs)

Large Language Models (LLMs) like Gemini, GPT, and Llama have revolutionized natural language understanding and generation. These models are trained on vast datasets of text and code, allowing them to perform tasks such as translation, summarization, and creative writing. They represent a significant leap towards more human-like AI interactions. The ability of LLMs to generate coherent and contextually relevant text has opened new possibilities for applications in customer service, content creation, and education.

## Ethical Considerations and Societal Impact

The increasing power of AI also brings forth critical ethical considerations. Issues such as algorithmic bias, privacy concerns, job displacement, and the potential for misuse of AI technologies are paramount. Ensuring fairness, transparency, and accountability in AI systems is essential for their responsible deployment. Discussions around AI governance and regulation are gaining momentum as societies grapple with the profound impact of these technologies.

### AI in Healthcare

AI is poised to transform healthcare by assisting with drug discovery, personalized treatment plans, and early disease detection. AI-powered tools can analyze vast amounts of patient data to identify patterns that human doctors might miss. This can lead to more accurate diagnoses and more effective interventions, ultimately improving patient outcomes. The integration of AI into healthcare systems requires robust validation and careful integration to ensure patient safety and data security.

## Conclusion

The journey of AI is far from over. As researchers continue to push the boundaries of what's possible, AI will undoubtedly become even more integrated into the fabric of society. Navigating its complexities, harnessing its potential, and mitigating its risks will be a collective endeavor, requiring collaboration across technology, policy, and ethics. The future is exciting, but it also demands thoughtful stewardship of this powerful technology.
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
        chunks = chunker.chunk_document(content, initial_metadata)
        print(f"Generated {len(chunks)} chunks.")

        # --- NEW STEP: Enrich chunks with LLM-generated summaries ---
        print("Enriching chunks with LLM summaries...")
        enriched_chunks = await metadata_enricher.enrich_chunks_with_llm_summaries(chunks)
        print(f"Finished enriching {len(enriched_chunks)} chunks.")
        # -----------------------------------------------------------

        FileHandler.save_chunks(enriched_chunks, OUTPUT_CHUNKS_FILE, format='json') # Save enriched chunks
        print(f"Chunks saved to {OUTPUT_CHUNKS_FILE}")

        report = evaluator.generate_report(enriched_chunks, QUALITY_REPORT_FILE) # Evaluate enriched chunks
        print("--- Quality Report ---")
        print(report)
        print("----------------------")
        print(f"Quality report saved to {QUALITY_REPORT_FILE}")

        processing_summary = {
            "file_name": os.path.basename(INPUT_FILE),
            "total_chunks": len(enriched_chunks), # Use enriched chunks count
            "chunking_strategy_applied": "Hybrid (Table-aware, Header-Recursive, Semantic, LLM-Enriched)", # Updated description
            "average_chunk_tokens": sum(c.metadata.get('chunk_tokens', 0) for c in enriched_chunks) / len(enriched_chunks) if enriched_chunks else 0
        }
        with open(PROCESSING_SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(processing_summary, f, indent=2, ensure_ascii=False)
        print(f"Processing summary saved to {PROCESSING_SUMMARY_FILE}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Call the async main function
if __name__ == "__main__":
    asyncio.run(main_async())
