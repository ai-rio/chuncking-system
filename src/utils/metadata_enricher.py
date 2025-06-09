import os
import json
import asyncio
from typing import List, Dict, Any
from langchain_core.documents import Document
from src.config.settings import config

class MetadataEnricher:
    """
    Enriches document chunks with additional metadata, including LLM-generated summaries.
    """

    def __init__(self):
        # Placeholder for any initialization needed, e.g., LLM client setup if not using fetch directly
        pass

    async def enrich_chunks_with_llm_summaries(self, chunks: List[Document]) -> List[Document]:
        """
        Asynchronously enriches chunks with LLM-generated summaries.
        Uses a mock API call for demonstration purposes.
        """
        if not config.ENABLE_LLM_METADATA_ENRICHMENT:
            print("LLM metadata enrichment is disabled in settings.")
            return chunks

        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            original_content = chunk.page_content
            
            # Skip summarization for very short chunks or structural elements
            if len(original_content.split()) < config.MIN_CHUNK_WORDS * 2 and \
               chunk.metadata.get('content_type') not in ['code', 'table'] and \
               not self._is_header_chunk(chunk):
                print(f"Skipping LLM summary for short or structural chunk {i}.")
                enriched_chunks.append(chunk)
                continue

            try:
                # Construct the prompt for the LLM summary
                prompt = f"{config.LLM_SUMMARY_PROMPT}\n\nTEXT:\n{original_content}"
                
                # Prepare the payload for the Gemini API call
                chatHistory = [{"role": "user", "parts": [{"text": prompt}]}]
                payload = {"contents": chatHistory}
                
                # Use Canvas environment's API key (empty string for auto-injection)
                apiKey = "" 
                apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/{config.LLM_METADATA_MODEL}:generateContent?key={apiKey}"
                
                # Simulate the fetch call to the Gemini API
                # In a real async environment, you would use an aiohttp client or similar.
                # For this environment, we'll use a synchronous fetch-like pattern or a mock.
                # Since this is Python, we'll simulate the response structure.
                
                # --- MOCK LLM RESPONSE ---
                # In a real environment, you'd perform an actual HTTP request here.
                # For demonstration in this environment, we'll simulate a concise summary.
                # If an actual API call is needed, the `requests` library would be used
                # or the `google.generativeai` client, which requires proper async setup.
                
                # For direct execution within a synchronous context, this mock is necessary.
                # If an async context were available (e.g., FastAPI, aiohttp), it would be:
                # response = await aiohttp.ClientSession().post(apiUrl, json=payload)
                # result = await response.json()
                
                # For now, let's provide a simple mock summary for non-API key based interaction
                # and then, if the user explicitly wants to test the LLM call with a key,
                # we can adjust for that or for a proper async environment if available.

                # Simple placeholder/mock for summary generation
                mock_summary_text = f"Summary of chunk {i}: {original_content[:min(100, len(original_content))]}..."
                
                # To make an actual async call in a non-async main function
                # we'd need a specific async library and loop or a different structure.
                # For this interactive environment, the safest is to explicitly use `google.generativeai` client
                # or assume a helper function that can do sync calls to async endpoints if available.
                
                # Assuming `google.generativeai` is installed and configured (e.g., from .env)
                # If you prefer a pure `fetch` HTTP request simulation for now, let me know.
                
                # Using google.generativeai client as it's typically easier than raw http fetch in Python
                # Need to import google.generativeai and configure it with the API key.
                # Let's assume google-generativeai is installed (pip install google-generativeai)
                # and we can load the key from config.

                # Since `google.generativeai` client is preferred for Gemini, let's update this:
                import google.generativeai as genai
                
                if config.GEMINI_API_KEY:
                    genai.configure(api_key=config.GEMINI_API_KEY)
                else:
                    print("Warning: GEMINI_API_KEY not set in settings. Skipping actual LLM call for summary.")
                    summary_text = mock_summary_text
                
                if config.GEMINI_API_KEY:
                    # Creating a model instance for text generation
                    model = genai.GenerativeModel(config.LLM_METADATA_MODEL)
                    
                    try:
                        # Make the API call
                        response = await asyncio.to_thread(model.generate_content, chatHistory)
                        summary_text = response.candidates[0].content.parts[0].text
                        print(f"LLM Summary generated for chunk {i}.")
                    except Exception as llm_e:
                        print(f"Error generating LLM summary for chunk {i}: {llm_e}. Using mock summary.")
                        summary_text = mock_summary_text
                else:
                    summary_text = mock_summary_text # Fallback to mock if API key is missing

                new_metadata = chunk.metadata.copy()
                new_metadata['llm_summary'] = summary_text
                new_metadata['enriched_by_llm'] = True
                
                enriched_chunks.append(Document(page_content=original_content, metadata=new_metadata))

            except Exception as e:
                print(f"Error during LLM enrichment for chunk {i}: {e}. Keeping original chunk.")
                enriched_chunks.append(chunk)

        return enriched_chunks

    def _is_header_chunk(self, chunk: Document) -> bool:
        """Helper to check if a chunk is primarily a header."""
        # Check metadata from MarkdownHeaderTextSplitter
        if any(key in chunk.metadata for key in ["Part", "Chapter", "Section", "Sub-section"]):
            return True
        # Also check if content itself is just a header
        if re.match(r'^\s*#+\s', chunk.page_content.strip()):
            return True
        return False
