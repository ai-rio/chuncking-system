import google.generativeai as genai
import os
import json
import asyncio
import hashlib
import re # Import regex module
from typing import List, Dict, Any, Optional
from collections.abc import Iterable

import google.generativeai as genai
from src.config.settings import config

class MetadataEnricher:
    """
    Enriches document chunks with additional metadata, such as LLM-generated summaries
    and image descriptions, with integrated caching for LLM calls.
    Now also includes automated structured metadata extraction.
    """

    def __init__(self):
        # Initialize Gemini API
        if config.GEMINI_API_KEY:  # This should work with current import
            genai.configure(api_key=config.GEMINI_API_KEY)
        else:
            print("Warning: GEMINI_API_KEY not found. LLM features will be mocked.")
        
        self.enable_cache = config.ENABLE_LLM_CACHE
        self.cache_dir = config.LLM_CACHE_DIR

        # Ensure cache directory exists if caching is enabled
        if config.ENABLE_LLM_CACHE:
            os.makedirs(config.LLM_CACHE_DIR, exist_ok=True)
            print(f"LLM cache directory initialized: {config.LLM_CACHE_DIR}")

    def _get_cache_key(self, prompt_parts: List[Any]) -> str:
        """
        Generates a unique cache key (SHA256 hash) for a given list of prompt parts.
        This now converts the entire list to a sorted JSON string for ultimate consistency.
        """
        # Convert the entire list of prompt parts to a sorted JSON string
        consistent_string = json.dumps(prompt_parts, sort_keys=True, ensure_ascii=False)
        hasher = hashlib.sha256()
        hasher.update(consistent_string.encode('utf-8'))
        return hasher.hexdigest()

    def _read_from_cache(self, cache_key: str) -> Optional[str]:
        """Reads a cached response from the file system."""
        if not config.ENABLE_LLM_CACHE:
            return None

        cache_file_path = os.path.join(config.LLM_CACHE_DIR, f"{cache_key}.json")
        
        if os.path.exists(cache_file_path):
            try:
                with open(cache_file_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    response_from_cache = cached_data.get("response")
                    
                    if response_from_cache is not None:
                        # Attempt to parse as JSON, but fall back to raw string if not valid JSON
                        try:
                            # If it was stored as a JSON string representation of a dict, load it
                            parsed_response = json.loads(response_from_cache)
                            if isinstance(parsed_response, dict):
                                return response_from_cache # Return the JSON string for consistency with _write_to_cache
                            else:
                                return response_from_cache # Not a dict, return as is
                        except json.JSONDecodeError:
                            return response_from_cache # Not valid JSON, return as is
                    return None
            except Exception as e:
                print(f"Error reading from cache file {cache_file_path}: {e}")
                return None
        return None

    def _write_to_cache(self, cache_key: str, response_text: str):
        """Writes a response to the file system cache."""
        if not config.ENABLE_LLM_CACHE:
            return

        cache_file_path = os.path.join(config.LLM_CACHE_DIR, f"{cache_key}.json")
        try:
            # Store the response as a simple string in the cache
            with open(cache_file_path, 'w', encoding='utf-8') as f:
                json.dump({"response": response_text}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error writing to cache file {cache_file_path}: {e}")

    def _parse_json_from_llm_output(self, raw_llm_output: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to parse JSON from the raw LLM output.
        First tries direct parsing, then uses regex to find a JSON block.
        """
        try:
            # Attempt direct parsing first
            parsed_data = json.loads(raw_llm_output)
            if isinstance(parsed_data, dict):
                return parsed_data
        except json.JSONDecodeError:
            pass # Continue to regex attempt if direct parsing fails

        # Fallback: Use regex to find a JSON-like block (e.g., between {} or ```json ... ```)
        # This regex tries to capture the content between the first '{' and last '}'
        # and also handles markdown code blocks for JSON.
        match = re.search(r"```json\s*(\{.*\})\s*```", raw_llm_output, re.DOTALL)
        if not match:
            match = re.search(r"(\{.*\})", raw_llm_output, re.DOTALL)
        
        if match:
            json_candidate = match.group(1)
            try:
                parsed_data = json.loads(json_candidate)
                if isinstance(parsed_data, dict):
                    return parsed_data
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON from regex extracted candidate: {json_candidate[:100]}... Error: {e}")

        return None # Return None if no valid JSON could be parsed

    async def summarize_chunk(self, text_content: str) -> str:
        """
        Generates a concise summary for a given text chunk using an LLM,
        with integrated caching.
        """
        if not config.ENABLE_LLM_METADATA_ENRICHMENT: # Check the general enrichment flag
            return "Summary generation disabled."

        if not config.GEMINI_API_KEY:
            return "Mock summary: " + text_content[:50] + "..."

        prompt_parts = [config.LLM_SUMMARY_PROMPT, text_content]
        cache_key = self._get_cache_key(prompt_parts)
        
        cached_summary = self._read_from_cache(cache_key)
        if cached_summary:
            print(f"LLM Summary: (CACHED) for chunk: {text_content[:30]}...")
            return cached_summary

        model = genai.GenerativeModel(config.LLM_METADATA_MODEL)
        try:
            chatHistory = [{"role": "user", "parts": [{"text": config.LLM_SUMMARY_PROMPT + "\n" + text_content}]}]
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
        if not config.ENABLE_LLM_IMAGE_DESCRIPTION:
            return "Image description generation disabled."

        if not config.GEMINI_API_KEY:
            fallback_description = f"Mock description of {alt_text or 'an image'}"
            return fallback_description

        # Normalize alt_text and image_url before creating prompt_parts for consistent hashing
        normalized_alt_text = alt_text.strip().replace('\r', '') if alt_text else ""
        normalized_image_url = image_url.strip().replace('\r', '') if image_url else ""

        # Construct prompt parts consistently for caching
        prompt_parts_for_key = [
            config.LLM_IMAGE_DESCRIPTION_PROMPT,
            f"alt_text:{normalized_alt_text}",
            f"image_url:{normalized_image_url}"
        ]

        cache_key = self._get_cache_key(prompt_parts_for_key)

        cached_description = self._read_from_cache(cache_key)
        if cached_description:
            print(f"LLM Image Description: (CACHED) for alt text: {alt_text or 'N/A'}")
            return cached_description

        actual_llm_prompt_parts = [config.LLM_IMAGE_DESCRIPTION_PROMPT]
        if normalized_alt_text:
            actual_llm_prompt_parts.append(f"The image's alt text is: '{normalized_alt_text}'.")
        if normalized_image_url:
            actual_llm_prompt_parts.append(f"The image URL is: '{normalized_image_url}'.")
        full_prompt_to_llm = "\n".join(actual_llm_prompt_parts)

        model = genai.GenerativeModel(config.LLM_IMAGE_MODEL)

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

    async def extract_metadata_from_chunk(self, text_content: str) -> Dict[str, Any]:
        """
        Extracts structured metadata (main_topic, key_entities) from a text chunk using an LLM,
        with integrated caching and robust JSON parsing.
        """
        if not config.ENABLE_LLM_METADATA_EXTRACTION:
            return {"main_topic": "Metadata extraction disabled.", "key_entities": []}
    
        if not config.GEMINI_API_KEY:
            return {"main_topic": "Mock metadata extraction.", "key_entities": ["mock_entity_1", "mock_entity_2"]}
    
        # Apply contextual prefix for summary content
        processed_content = text_content
        if "Summary:" in text_content or "Overview" in text_content or len(text_content.strip()) < 200:
            processed_content = f"Summary of document section: {text_content}"
        
        prompt_with_text = config.LLM_EXTRACTION_PROMPT.format(text_content=processed_content)
        prompt_parts = [prompt_with_text] # For cache key, use the formatted prompt
        cache_key = self._get_cache_key(prompt_parts)

        cached_response_str = self._read_from_cache(cache_key)
        if cached_response_str:
            print(f"LLM Metadata Extraction: (CACHED) for chunk: {text_content[:30]}...")
            # Attempt to parse the cached string as JSON
            parsed_cached_data = self._parse_json_from_llm_output(cached_response_str)
            if parsed_cached_data:
                return parsed_cached_data
            else:
                # If cached data isn't valid JSON, return it as a raw string for main_topic
                print(f"Warning: Cached metadata for chunk {text_content[:30]}... was not valid JSON. Returning raw string.")
                return {"main_topic": cached_response_str.strip(), "key_entities": [], "cached": True}


        model = genai.GenerativeModel(config.LLM_EXTRACTION_MODEL)
        try:
            chatHistory = [{"role": "user", "parts": [{"text": prompt_with_text}]}]
            # Add this before the API call in extract_metadata_from_chunk
            await asyncio.sleep(1)  # Add 1 second delay between API calls
            response = await asyncio.to_thread(model.generate_content, chatHistory)
            extracted_text = response.candidates[0].content.parts[0].text
            
            # --- DEBUG: Print raw LLM output for metadata extraction ---
            print(f"DEBUG Metadata Extraction: Raw LLM output for chunk '{text_content[:30]}...':\n'{extracted_text}'")
            # --- END DEBUG ---

            parsed_metadata = self._parse_json_from_llm_output(extracted_text)

            if parsed_metadata:
                self._write_to_cache(cache_key, json.dumps(parsed_metadata, ensure_ascii=False)) # Cache the JSON string
                print(f"LLM Metadata Extraction generated and parsed for chunk: {text_content[:30]}...")
                return parsed_metadata
            else:
                # If JSON parsing fails, log a warning and fall back to raw text for main_topic
                print(f"Warning: LLM output for chunk {text_content[:30]}... was not valid JSON. Falling back to raw text.")
                main_topic_content = extracted_text.strip()
                self._write_to_cache(cache_key, main_topic_content) # Cache the raw string
                return {"main_topic": main_topic_content, "key_entities": [], "json_parse_error": True}

        except Exception as e:
            print(f"Error generating LLM metadata for chunk: {text_content[:30]}... Error: {e}")
            return {"main_topic": f"Error extracting metadata: {e}", "key_entities": [], "extraction_error": True}

    async def enrich_chunks_with_llm_summaries_and_metadata(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Asynchronously enriches a list of chunks with LLM-generated summaries and structured metadata.
        This replaces the previous enrich_chunks_with_llm_summaries.
        """
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            # --- DEBUG: Print chunk.metadata initial state within the loop ---
            print(f"DEBUG: Chunk {i} metadata initial state - Type: {type(chunk.metadata)}, Content: {chunk.metadata}")
            # --- END DEBUG ---

            # Determine if the chunk is text-based and not a visual, table, or code chunk
            # This condition ensures we process only relevant text chunks for metadata and summary
            is_text_chunk = not (
                chunk.metadata.get('chunk_type') in ['visual', 'structural'] or
                chunk.metadata.get('source_segment_type') in ['image', 'table', 'code']
            )

            if is_text_chunk:
                # Process summary for this text chunk
                summary = await self.summarize_chunk(chunk.page_content)
                chunk.metadata['summary'] = summary
                
                # Also extract structured metadata if enabled
                if config.ENABLE_LLM_METADATA_EXTRACTION:
                    try:
                        # Use summary for metadata extraction if original content is sparse
                        content_for_metadata = chunk.page_content
                        
                        # Check if summary indicates insufficient content or error
                        summary_indicates_error = any(phrase in summary.lower() for phrase in [
                            "i'm sorry", "insufficient", "need more content", "error summarizing", 
                            "summary generation disabled", "mock summary"
                        ])
                        
                        if len(chunk.page_content.strip()) < 200 or "##" in chunk.page_content[:50]:
                            if not summary_indicates_error:
                                # For sparse content (headers, short sections), use the richer summary
                                content_for_metadata = f"Summary of document section: {summary}"
                            else:
                                # If summary failed, enhance the original content with context
                                content_for_metadata = f"Document section title or header: {chunk.page_content.strip()}"
                        
                        extracted_metadata_dict = await self.extract_metadata_from_chunk(content_for_metadata)
                        
                        # --- DEBUG: Print type and content BEFORE chunk.metadata.update ---
                        print(f"DEBUG: Before chunk.metadata.update - Type of extracted_metadata_dict: {type(extracted_metadata_dict)}, Content: {extracted_metadata_dict}")
                        # --- END DEBUG ---
                        
                        if isinstance(extracted_metadata_dict, dict):
                            # Ensure chunk.metadata is a dictionary before updating
                            if not isinstance(chunk.metadata, dict):
                                print(f"CRITICAL WARNING: chunk.metadata for chunk {i} is not a dictionary ({type(chunk.metadata)}). Attempting to re-initialize.")
                                chunk.metadata = {} # Re-initialize if it's not a dict
                            chunk.metadata.update(extracted_metadata_dict)
                        else:
                            # This case should ideally not happen with the updated extract_metadata_from_chunk
                            print(f"ERROR: Extracted metadata for chunk {i} was NOT a dictionary. Type: {type(extracted_metadata_dict)}, Content: {extracted_metadata_dict}")
                            # Provide a default value to prevent crash
                            chunk.metadata['main_topic'] = f"Failed to extract structured metadata (type error: {type(extracted_metadata_dict)})."
                            chunk.metadata['key_entities'] = []

                    except Exception as extraction_or_update_err:
                        print(f"CRITICAL ERROR during metadata extraction or update for chunk {i}: {extraction_or_update_err}")
                        print(f"Problematic extracted_metadata_dict (if available): Type={type(extracted_metadata_dict) if 'extracted_metadata_dict' in locals() else 'N/A'}, Content={extracted_metadata_dict if 'extracted_metadata_dict' in locals() else 'N/A'}")
                        print(f"Current chunk.metadata before failed operation: {chunk.metadata}")
                        raise # Re-raise the exception after logging for full traceback
                
                print(f"LLM Summary & Metadata processed for text chunk {i}.")
            elif chunk.metadata.get('source_segment_type') == 'image' or \
                 chunk.metadata.get('chunk_type') == 'visual':
                # Image chunks are handled during chunking (image_aware_processing)
                # and already have their descriptions set as summary.
                print(f"Image chunk {i} processed (description already handled during chunking).")
            else:
                # Fallback for other structural chunks (table, code) - summarize only for now
                summary = await self.summarize_chunk(chunk.page_content)
                chunk.metadata['summary'] = summary
                print(f"LLM Summary processed for structural chunk {i}.")

            enriched_chunks.append(chunk)
        return enriched_chunks
