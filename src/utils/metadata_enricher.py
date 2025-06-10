import google.generativeai as genai
import os
import json
import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from collections.abc import Iterable

import src.config.settings as config

class MetadataEnricher:
    """
    Enriches document chunks with additional metadata, such as LLM-generated summaries
    and image descriptions, with integrated caching for LLM calls.
    """

    def __init__(self):
        # Configure Gemini API if key is available
        if config.config.GEMINI_API_KEY:
            genai.configure(api_key=config.config.GEMINI_API_KEY)
        else:
            print("Warning: GEMINI_API_KEY not set. LLM-based features (summaries, image descriptions) will use mock data.")

        # Ensure cache directory exists if caching is enabled
        if config.config.ENABLE_LLM_CACHE:
            os.makedirs(config.config.LLM_CACHE_DIR, exist_ok=True)
            print(f"LLM cache directory initialized: {config.config.LLM_CACHE_DIR}")

    def _get_cache_key(self, prompt_parts: List[Any]) -> str:
        """
        Generates a unique cache key (SHA256 hash) for a given list of prompt parts.
        This now converts the entire list to a sorted JSON string for ultimate consistency.
        """
        # Convert the entire list of prompt parts to a sorted JSON string
        # This ensures that the order and exact representation are consistent for hashing
        # For non-dict parts, str() converts them to strings for JSON serialization.
        consistent_string = json.dumps(prompt_parts, sort_keys=True, ensure_ascii=False)
        hasher = hashlib.sha256()
        hasher.update(consistent_string.encode('utf-8'))
        return hasher.hexdigest()

    def _read_from_cache(self, cache_key: str) -> Optional[str]:
        """Reads a cached response from the file system."""
        if not config.config.ENABLE_LLM_CACHE:
            return None

        cache_file_path = os.path.join(config.config.LLM_CACHE_DIR, f"{cache_key}.json")
        
        if os.path.exists(cache_file_path):
            try:
                with open(cache_file_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    response_from_cache = cached_data.get("response")
                    
                    if response_from_cache is not None:
                        response_from_cache = response_from_cache.strip() # Strip whitespace from cached response

                    return response_from_cache
            except Exception as e:
                print(f"Error reading from cache file {cache_file_path}: {e}")
                return None
        return None

    def _write_to_cache(self, cache_key: str, response_text: str):
        """Writes a response to the file system cache."""
        if not config.config.ENABLE_LLM_CACHE:
            return

        cache_file_path = os.path.join(config.config.LLM_CACHE_DIR, f"{cache_key}.json")
        try:
            with open(cache_file_path, 'w', encoding='utf-8') as f:
                json.dump({"response": response_text}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error writing to cache file {cache_file_path}: {e}")

    async def summarize_chunk(self, text_content: str) -> str:
        """
        Generates a concise summary for a given text chunk using an LLM,
        with integrated caching.
        """
        if not config.config.ENABLE_LLM_METADATA_ENRICHMENT:
            return "Summary generation disabled."

        if not config.config.GEMINI_API_KEY:
            return "Mock summary: " + text_content[:50] + "..."

        prompt_parts = [config.config.LLM_SUMMARY_PROMPT, text_content]
        cache_key = self._get_cache_key(prompt_parts)
        
        cached_summary = self._read_from_cache(cache_key)
        if cached_summary:
            print(f"LLM Summary: (CACHED) for chunk: {text_content[:30]}...")
            return cached_summary

        model = genai.GenerativeModel(config.config.LLM_METADATA_MODEL)
        try:
            chatHistory = [{"role": "user", "parts": [{"text": config.config.LLM_SUMMARY_PROMPT + "\n" + text_content}]}]
            response = await asyncio.to_thread(model.generate_content, chatHistory)
            summary_text = response.candidates[0].content.parts[0].text
            self._write_to_cache(cache_key, summary_text)
            print(f"LLM Summary generated for chunk: {text_content[:30]}...")
            return summary_text
        except Exception as e:
            print(f"Error generating LLM summary for chunk: {text_content[:30]}... Error: {e}")
            return "Error summarizing chunk."

    async def describe_image(self, alt_text: str = None, image_url: str = None) -> str:
        """
        Generates a description for an image using an LLM,
        with integrated caching.
        """
        if not config.config.ENABLE_LLM_IMAGE_DESCRIPTION:
            return "Image description generation disabled."

        if not config.config.GEMINI_API_KEY:
            fallback_description = f"Mock description of {alt_text or 'an image'}"
            return fallback_description

        # Normalize alt_text and image_url before creating prompt_parts for consistent hashing
        normalized_alt_text = alt_text.strip().replace('\r', '') if alt_text else ""
        normalized_image_url = image_url.strip().replace('\r', '') if image_url else ""

        # Construct prompt parts consistently for caching
        prompt_parts_for_key = [
            config.config.LLM_IMAGE_DESCRIPTION_PROMPT,
            f"alt_text:{normalized_alt_text}",
            f"image_url:{normalized_image_url}"
        ]

        cache_key = self._get_cache_key(prompt_parts_for_key)

        cached_description = self._read_from_cache(cache_key)
        if cached_description:
            print(f"LLM Image Description: (CACHED) for alt text: {alt_text or 'N/A'}")
            return cached_description

        actual_llm_prompt_parts = [config.config.LLM_IMAGE_DESCRIPTION_PROMPT]
        if normalized_alt_text:
            actual_llm_prompt_parts.append(f"The image's alt text is: '{normalized_alt_text}'.")
        if normalized_image_url:
            actual_llm_prompt_parts.append(f"The image URL is: '{normalized_image_url}'.")
        full_prompt_to_llm = "\n".join(actual_llm_prompt_parts)

        model = genai.GenerativeModel(config.config.LLM_IMAGE_MODEL)

        try:
            chatHistory = [{"role": "user", "parts": [{"text": full_prompt_to_llm}]}]
            response = await asyncio.to_thread(model.generate_content, chatHistory)
            description_text = response.candidates[0].content.parts[0].text
            self._write_to_cache(cache_key, description_text)
            print(f"LLM Image Description generated for alt text: {alt_text or 'N/A'}")
            return description_text
        except Exception as e:
            print(f"Error generating LLM image description for '{alt_text or image_url}'. Error: {e}")
            return f"Error describing image with alt text '{alt_text or 'N/A'}'."

    async def enrich_chunks_with_llm_summaries(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Asynchronously enriches a list of chunks with LLM-generated summaries.
        This function iterates through the chunks and calls the summarize_chunk method.
        """
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            # The summarize_chunk method now handles caching internally
            summary = await self.summarize_chunk(chunk.page_content)
            chunk.metadata['summary'] = summary
            enriched_chunks.append(chunk)
            print(f"LLM Summary processed for chunk {i}.")
        return enriched_chunks
