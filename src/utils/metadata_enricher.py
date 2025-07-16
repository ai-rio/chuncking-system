from typing import Dict, Any, List
from langchain_core.documents import Document
import hashlib
from datetime import datetime

class MetadataEnricher:
    """Enrich chunks with additional metadata"""
    
    @staticmethod
    def enrich_chunk(chunk: Document, document_info: Dict[str, Any] = None) -> Document:
        """Add comprehensive metadata to a chunk"""
        
        document_info = document_info or {}
        
        # Generate unique chunk ID
        chunk_id = hashlib.md5(
            f"{chunk.page_content[:100]}{chunk.metadata.get('source', '')}".encode()
        ).hexdigest()[:12]
        
        enriched_metadata = {
            **chunk.metadata,
            'chunk_id': chunk_id,
            'processed_at': datetime.now().isoformat(),
            'content_hash': hashlib.md5(chunk.page_content.encode()).hexdigest(),
            **document_info
        }
        
        # Content analysis
        content = chunk.page_content
        enriched_metadata.update({
            'has_code': '```' in content or 'def ' in content or 'import ' in content,
            'has_urls': 'http://' in content or 'https://' in content,
            'has_headers': content.strip().startswith('#'),
            'language': MetadataEnricher._detect_language(content)
        })
        
        # Return new Document to preserve immutability
        return Document(
            page_content=chunk.page_content,
            metadata=enriched_metadata
        )
    
    def enrich_chunk_metadata(self, chunk: Document, chunk_index: int = 0) -> Document:
        """Enrich chunk metadata with index and additional information"""
        # Generate unique chunk ID
        chunk_id = hashlib.md5(
            f"{chunk.page_content[:100]}{chunk.metadata.get('source', '')}{chunk_index}".encode()
        ).hexdigest()[:12]
        
        enriched_metadata = {
            **chunk.metadata,
            'chunk_index': chunk_index,
            'chunk_id': chunk_id,
            'processed_at': datetime.now().isoformat()
        }
        
        return Document(
            page_content=chunk.page_content,
            metadata=enriched_metadata
        )
    
    @staticmethod
    def _detect_language(content: str) -> str:
        """Simple language detection"""
        # Basic heuristics - can be enhanced with proper language detection
        if any(word in content.lower() for word in ['import', 'def', 'class', 'function']):
            return 'technical'
        elif any(word in content.lower() for word in ['the', 'and', 'to', 'of', 'a']):
            return 'english'
        else:
            return 'unknown'
