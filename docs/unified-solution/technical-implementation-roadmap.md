# Technical Implementation Roadmap: Enterprise AI Unified Solution

## Overview

This document provides detailed technical implementation guidance for integrating PocketFlow, NextJS-FastAPI, Milvus, and the existing chunking system into a unified enterprise-grade AI solution.

**Implementation Timeline**: 12 weeks across 3 phases  
**Team Size**: 8-10 engineers  
**Architecture**: Microservices with unified API gateway  

---

## Phase 1: Foundation Integration (Weeks 1-4)

### Week 1: Environment Setup & Milvus Integration

#### Day 1-2: Development Environment

**Infrastructure Setup:**
```bash
# Docker Compose for development environment
# docker-compose.dev.yml
version: '3.8'
services:
  milvus-etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  milvus-minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - minio:/minio_data
    command: minio server /minio_data --console-address ":9001"

  milvus-standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: milvus-etcd:2379
      MINIO_ADDRESS: milvus-minio:9000
    volumes:
      - milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "milvus-etcd"
      - "milvus-minio"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: unified_ai
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  etcd:
  minio:
  milvus:
  redis_data:
  postgres_data:
```

**Project Structure Setup:**
```bash
# Create unified solution structure
mkdir -p unified-ai-solution/{
  backend/{api,core,services,models,utils},
  frontend/{components,pages,hooks,utils},
  shared/{types,schemas,constants},
  infrastructure/{docker,k8s,terraform},
  tests/{unit,integration,e2e},
  docs/{api,deployment,user-guide}
}
```

#### Day 3-5: Milvus Vector Store Implementation

**Core Vector Store Service:**
```python
# backend/services/vector_store.py
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, Index
)
import numpy as np
from datetime import datetime
import uuid
from backend.core.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class MilvusVectorStore:
    """Enterprise-grade Milvus vector store implementation."""
    
    def __init__(self, collection_name: str = "document_chunks"):
        self.collection_name = collection_name
        self.collection: Optional[Collection] = None
        self.embedding_dim = settings.EMBEDDING_DIMENSION
        self._connect()
        self._ensure_collection()
    
    def _connect(self):
        """Establish connection to Milvus."""
        try:
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                user=settings.MILVUS_USER,
                password=settings.MILVUS_PASSWORD
            )
            logger.info("Connected to Milvus successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        else:
            self._create_collection()
    
    def _create_collection(self):
        """Create new collection with optimized schema."""
        # Define collection schema
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=36,
                is_primary=True,
                description="Unique chunk identifier"
            ),
            FieldSchema(
                name="document_id",
                dtype=DataType.VARCHAR,
                max_length=36,
                description="Parent document identifier"
            ),
            FieldSchema(
                name="chunk_index",
                dtype=DataType.INT64,
                description="Chunk position in document"
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="Chunk text content"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dim,
                description="Text embedding vector"
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
                description="Chunk metadata and quality metrics"
            ),
            FieldSchema(
                name="quality_score",
                dtype=DataType.FLOAT,
                description="Chunk quality score"
            ),
            FieldSchema(
                name="created_at",
                dtype=DataType.INT64,
                description="Creation timestamp"
            )
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Document chunks with embeddings for semantic search"
        )
        
        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using='default',
            shards_num=2
        )
        
        # Create index for vector field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        # Create index for document_id for efficient filtering
        self.collection.create_index(
            field_name="document_id",
            index_params={"index_type": "TRIE"}
        )
        
        logger.info(f"Created new collection: {self.collection_name}")
    
    async def insert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Insert document chunks with embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings count mismatch")
        
        # Prepare data for insertion
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        current_time = int(datetime.now().timestamp() * 1000)
        
        data = [
            chunk_ids,  # id
            [chunk["document_id"] for chunk in chunks],  # document_id
            [chunk["chunk_index"] for chunk in chunks],  # chunk_index
            [chunk["content"] for chunk in chunks],  # content
            embeddings,  # embedding
            [chunk.get("metadata", {}) for chunk in chunks],  # metadata
            [chunk.get("quality_score", 0.0) for chunk in chunks],  # quality_score
            [current_time] * len(chunks)  # created_at
        ]
        
        try:
            # Insert data
            insert_result = self.collection.insert(data)
            
            # Flush to ensure data is written
            self.collection.flush()
            
            logger.info(
                f"Inserted {len(chunks)} chunks into {self.collection_name}",
                document_ids=list(set(chunk["document_id"] for chunk in chunks))
            )
            
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Failed to insert chunks: {e}")
            raise
    
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        document_id: Optional[str] = None,
        quality_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity."""
        try:
            # Load collection if not loaded
            if not self.collection.has_index():
                self.collection.load()
            
            # Prepare search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Build expression for filtering
            expressions = []
            if document_id:
                expressions.append(f'document_id == "{document_id}"')
            if quality_threshold > 0:
                expressions.append(f'quality_score >= {quality_threshold}')
            
            expression = " and ".join(expressions) if expressions else None
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expression,
                output_fields=[
                    "id", "document_id", "chunk_index", "content",
                    "metadata", "quality_score", "created_at"
                ]
            )
            
            # Format results
            formatted_results = []
            for hit in results[0]:
                formatted_results.append({
                    "id": hit.entity.get("id"),
                    "document_id": hit.entity.get("document_id"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "content": hit.entity.get("content"),
                    "metadata": hit.entity.get("metadata"),
                    "quality_score": hit.entity.get("quality_score"),
                    "similarity_score": hit.score,
                    "created_at": hit.entity.get("created_at")
                })
            
            logger.info(
                f"Found {len(formatted_results)} similar chunks",
                query_length=len(query_embedding),
                limit=limit,
                document_id=document_id
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks for a specific document."""
        try:
            # Delete by expression
            expr = f'document_id == "{document_id}"'
            delete_result = self.collection.delete(expr)
            
            # Flush to ensure deletion
            self.collection.flush()
            
            logger.info(
                f"Deleted chunks for document {document_id}",
                deleted_count=delete_result.delete_count
            )
            
            return delete_result.delete_count
            
        except Exception as e:
            logger.error(f"Failed to delete document chunks: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            stats = self.collection.get_stats()
            return {
                "total_entities": stats["row_count"],
                "collection_name": self.collection_name,
                "schema": self.collection.schema.to_dict(),
                "indexes": [index.to_dict() for index in self.collection.indexes]
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
```

### Week 2: Enhanced Chunking Pipeline

#### Enhanced Document Processor Integration

```python
# backend/services/document_processor.py
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path

from src.chunking_system import DocumentChunker, ChunkingResult
from backend.services.vector_store import MilvusVectorStore
from backend.services.embedding_service import EmbeddingService
from backend.core.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class UnifiedDocumentProcessor:
    """Enhanced document processor with vector storage integration."""
    
    def __init__(self):
        # Initialize existing chunking system
        self.chunker = DocumentChunker(
            config=settings.CHUNKING_CONFIG
        )
        
        # Initialize new components
        self.vector_store = MilvusVectorStore()
        self.embedding_service = EmbeddingService()
        
        logger.info("UnifiedDocumentProcessor initialized")
    
    async def process_document(
        self,
        file_path: Path,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process document through unified pipeline."""
        try:
            logger.info(
                f"Starting document processing",
                document_id=document_id,
                file_path=str(file_path)
            )
            
            # Step 1: Chunk document using existing system
            chunking_result = self.chunker.chunk_file(
                file_path=file_path,
                metadata=metadata
            )
            
            if not chunking_result.success:
                raise Exception(f"Chunking failed: {chunking_result.error_message}")
            
            # Step 2: Prepare chunks for vector storage
            prepared_chunks = self._prepare_chunks_for_storage(
                chunks=chunking_result.chunks,
                document_id=document_id,
                metadata=metadata or {}
            )
            
            # Step 3: Generate embeddings
            embeddings = await self.embedding_service.generate_embeddings(
                texts=[chunk["content"] for chunk in prepared_chunks]
            )
            
            # Step 4: Store in vector database
            chunk_ids = await self.vector_store.insert_chunks(
                chunks=prepared_chunks,
                embeddings=embeddings
            )
            
            # Step 5: Compile processing result
            result = {
                "document_id": document_id,
                "chunk_count": len(prepared_chunks),
                "chunk_ids": chunk_ids,
                "processing_time_ms": chunking_result.processing_time_ms,
                "quality_metrics": chunking_result.quality_metrics,
                "performance_metrics": chunking_result.performance_metrics,
                "success": True,
                "metadata": {
                    **metadata or {},
                    "file_path": str(file_path),
                    "processing_timestamp": chunking_result.metadata.get("timestamp"),
                    "chunking_strategy": chunking_result.metadata.get("strategy")
                }
            }
            
            logger.info(
                f"Document processing completed successfully",
                document_id=document_id,
                chunk_count=len(prepared_chunks),
                processing_time_ms=chunking_result.processing_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Document processing failed",
                document_id=document_id,
                error=str(e),
                file_path=str(file_path)
            )
            raise
    
    def _prepare_chunks_for_storage(
        self,
        chunks: List[Any],
        document_id: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Prepare chunks for vector storage."""
        prepared_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Extract chunk content and metadata
            if hasattr(chunk, 'page_content'):
                content = chunk.page_content
                chunk_metadata = getattr(chunk, 'metadata', {})
            elif isinstance(chunk, dict):
                content = chunk.get('content', chunk.get('text', str(chunk)))
                chunk_metadata = chunk.get('metadata', {})
            else:
                content = str(chunk)
                chunk_metadata = {}
            
            # Prepare chunk for storage
            prepared_chunk = {
                "document_id": document_id,
                "chunk_index": i,
                "content": content,
                "metadata": {
                    **metadata,
                    **chunk_metadata,
                    "chunk_length": len(content),
                    "chunk_words": len(content.split())
                },
                "quality_score": chunk_metadata.get('quality_score', 0.0)
            }
            
            prepared_chunks.append(prepared_chunk)
        
        return prepared_chunks
    
    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        document_id: Optional[str] = None,
        quality_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search documents using semantic similarity."""
        try:
            # Generate query embedding
            query_embeddings = await self.embedding_service.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Search similar chunks
            results = await self.vector_store.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                document_id=document_id,
                quality_threshold=quality_threshold
            )
            
            logger.info(
                f"Document search completed",
                query_length=len(query),
                results_count=len(results),
                document_id=document_id
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            raise
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document and all its chunks."""
        try:
            deleted_count = await self.vector_store.delete_document_chunks(document_id)
            
            logger.info(
                f"Document deleted successfully",
                document_id=document_id,
                deleted_chunks=deleted_count
            )
            
            return deleted_count > 0
            
        except Exception as e:
            logger.error(
                f"Document deletion failed",
                document_id=document_id,
                error=str(e)
            )
            raise
```

### Week 3: Embedding Service Implementation

```python
# backend/services/embedding_service.py
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from backend.core.config import settings
from backend.utils.logger import get_logger
from backend.utils.cache import CacheManager

logger = get_logger(__name__)

class EmbeddingService:
    """Multi-provider embedding service with caching."""
    
    def __init__(self):
        self.provider = settings.EMBEDDING_PROVIDER
        self.model_name = settings.EMBEDDING_MODEL
        self.cache_manager = CacheManager()
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        
        # Initialize provider-specific components
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "local":
            self._init_local_model()
        elif self.provider == "jina":
            self._init_jina()
        
        logger.info(
            f"EmbeddingService initialized",
            provider=self.provider,
            model=self.model_name
        )
    
    def _init_openai(self):
        """Initialize OpenAI embedding client."""
        import openai
        self.openai_client = openai.AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY
        )
    
    def _init_local_model(self):
        """Initialize local transformer model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def _init_jina(self):
        """Initialize Jina AI client."""
        self.jina_api_key = settings.JINA_API_KEY
        self.jina_url = "https://api.jina.ai/v1/embeddings"
    
    async def generate_embeddings(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for list of texts."""
        if not texts:
            return []
        
        # Check cache first
        if use_cache:
            cached_embeddings = await self._get_cached_embeddings(texts)
            if len(cached_embeddings) == len(texts):
                return cached_embeddings
        
        # Generate embeddings based on provider
        if self.provider == "openai":
            embeddings = await self._generate_openai_embeddings(texts)
        elif self.provider == "local":
            embeddings = await self._generate_local_embeddings(texts)
        elif self.provider == "jina":
            embeddings = await self._generate_jina_embeddings(texts)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
        
        # Cache results
        if use_cache:
            await self._cache_embeddings(texts, embeddings)
        
        logger.info(
            f"Generated embeddings",
            provider=self.provider,
            text_count=len(texts),
            embedding_dim=len(embeddings[0]) if embeddings else 0
        )
        
        return embeddings
    
    async def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            # Process in batches to respect rate limits
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                response = await self.openai_client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise
    
    async def _generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local transformer model."""
        try:
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling of last hidden states
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    embeddings = embeddings.cpu().numpy().tolist()
                
                all_embeddings.extend(embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            raise
    
    async def _generate_jina_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Jina AI API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.jina_api_key}",
                "Content-Type": "application/json"
            }
            
            all_embeddings = []
            
            async with aiohttp.ClientSession() as session:
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i + self.batch_size]
                    
                    payload = {
                        "model": self.model_name,
                        "input": batch
                    }
                    
                    async with session.post(
                        self.jina_url,
                        headers=headers,
                        json=payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            batch_embeddings = [item["embedding"] for item in result["data"]]
                            all_embeddings.extend(batch_embeddings)
                        else:
                            error_text = await response.text()
                            raise Exception(f"Jina API error: {response.status} - {error_text}")
                    
                    # Small delay between batches
                    if i + self.batch_size < len(texts):
                        await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Jina embedding generation failed: {e}")
            raise
    
    async def _get_cached_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Retrieve cached embeddings for texts."""
        cached_embeddings = []
        
        for text in texts:
            cache_key = f"embedding:{self.provider}:{self.model_name}:{hash(text)}"
            cached = await self.cache_manager.get(cache_key)
            cached_embeddings.append(cached)
        
        return cached_embeddings
    
    async def _cache_embeddings(self, texts: List[str], embeddings: List[List[float]]):
        """Cache embeddings for future use."""
        for text, embedding in zip(texts, embeddings):
            cache_key = f"embedding:{self.provider}:{self.model_name}:{hash(text)}"
            await self.cache_manager.set(
                cache_key,
                embedding,
                ttl=settings.EMBEDDING_CACHE_TTL
            )
```

### Week 4: Integration Testing & Quality Metrics

```python
# tests/integration/test_unified_pipeline.py
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

from backend.services.document_processor import UnifiedDocumentProcessor
from backend.services.vector_store import MilvusVectorStore
from backend.services.embedding_service import EmbeddingService

@pytest.fixture
async def document_processor():
    """Create document processor for testing."""
    processor = UnifiedDocumentProcessor()
    yield processor
    # Cleanup after test
    await processor.vector_store.collection.drop()

@pytest.fixture
def sample_markdown_file(tmp_path):
    """Create sample markdown file for testing."""
    content = """
# Test Document

This is a test document for the unified AI solution.

## Section 1

This section contains important information about the system architecture.
The system integrates multiple components including PocketFlow, Milvus, and NextJS.

## Section 2

This section discusses implementation details and best practices.
The integration follows enterprise-grade patterns and security standards.

### Subsection 2.1

Detailed technical specifications and requirements.
"""
    
    file_path = tmp_path / "test_document.md"
    file_path.write_text(content)
    return file_path

class TestUnifiedPipeline:
    """Test unified document processing pipeline."""
    
    @pytest.mark.asyncio
    async def test_document_processing_pipeline(self, document_processor, sample_markdown_file):
        """Test complete document processing pipeline."""
        document_id = "test-doc-001"
        metadata = {
            "title": "Test Document",
            "author": "Test Author",
            "category": "technical"
        }
        
        # Process document
        result = await document_processor.process_document(
            file_path=sample_markdown_file,
            document_id=document_id,
            metadata=metadata
        )
        
        # Verify processing result
        assert result["success"] is True
        assert result["document_id"] == document_id
        assert result["chunk_count"] > 0
        assert len(result["chunk_ids"]) == result["chunk_count"]
        assert "quality_metrics" in result
        assert "performance_metrics" in result
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, document_processor, sample_markdown_file):
        """Test semantic search functionality."""
        document_id = "test-doc-002"
        
        # Process document first
        await document_processor.process_document(
            file_path=sample_markdown_file,
            document_id=document_id
        )
        
        # Search for relevant content
        search_results = await document_processor.search_documents(
            query="system architecture and components",
            limit=5
        )
        
        # Verify search results
        assert len(search_results) > 0
        assert all("similarity_score" in result for result in search_results)
        assert all("content" in result for result in search_results)
        assert all(result["similarity_score"] > 0 for result in search_results)
    
    @pytest.mark.asyncio
    async def test_document_specific_search(self, document_processor, sample_markdown_file):
        """Test search within specific document."""
        document_id = "test-doc-003"
        
        # Process document
        await document_processor.process_document(
            file_path=sample_markdown_file,
            document_id=document_id
        )
        
        # Search within specific document
        search_results = await document_processor.search_documents(
            query="implementation details",
            document_id=document_id,
            limit=3
        )
        
        # Verify all results are from the specified document
        assert len(search_results) > 0
        assert all(result["document_id"] == document_id for result in search_results)
    
    @pytest.mark.asyncio
    async def test_document_deletion(self, document_processor, sample_markdown_file):
        """Test document deletion functionality."""
        document_id = "test-doc-004"
        
        # Process document
        result = await document_processor.process_document(
            file_path=sample_markdown_file,
            document_id=document_id
        )
        
        initial_chunk_count = result["chunk_count"]
        assert initial_chunk_count > 0
        
        # Delete document
        deletion_success = await document_processor.delete_document(document_id)
        assert deletion_success is True
        
        # Verify document is deleted (search should return no results)
        search_results = await document_processor.search_documents(
            query="test content",
            document_id=document_id
        )
        
        assert len(search_results) == 0

class TestVectorStore:
    """Test Milvus vector store operations."""
    
    @pytest.fixture
    async def vector_store(self):
        """Create vector store for testing."""
        store = MilvusVectorStore(collection_name="test_collection")
        yield store
        # Cleanup
        await store.collection.drop()
    
    @pytest.mark.asyncio
    async def test_collection_creation(self, vector_store):
        """Test collection creation and schema."""
        stats = vector_store.get_collection_stats()
        
        assert stats["collection_name"] == "test_collection"
        assert "schema" in stats
        assert "indexes" in stats
    
    @pytest.mark.asyncio
    async def test_chunk_insertion_and_search(self, vector_store):
        """Test chunk insertion and similarity search."""
        # Prepare test data
        chunks = [
            {
                "document_id": "doc1",
                "chunk_index": 0,
                "content": "This is about machine learning and AI",
                "metadata": {"section": "introduction"},
                "quality_score": 0.8
            },
            {
                "document_id": "doc1",
                "chunk_index": 1,
                "content": "Deep learning models require large datasets",
                "metadata": {"section": "methodology"},
                "quality_score": 0.9
            }
        ]
        
        # Mock embeddings (in real scenario, these would be generated)
        embeddings = [
            [0.1, 0.2, 0.3] * 512,  # 1536-dimensional vector
            [0.2, 0.3, 0.4] * 512
        ]
        
        # Insert chunks
        chunk_ids = await vector_store.insert_chunks(chunks, embeddings)
        assert len(chunk_ids) == 2
        
        # Search for similar content
        query_embedding = [0.15, 0.25, 0.35] * 512
        results = await vector_store.search_similar(
            query_embedding=query_embedding,
            limit=2
        )
        
        assert len(results) == 2
        assert all("similarity_score" in result for result in results)

class TestEmbeddingService:
    """Test embedding service functionality."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create embedding service for testing."""
        with patch('backend.core.config.settings') as mock_settings:
            mock_settings.EMBEDDING_PROVIDER = "local"
            mock_settings.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
            mock_settings.EMBEDDING_BATCH_SIZE = 2
            
            service = EmbeddingService()
            return service
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, embedding_service):
        """Test embedding generation for text list."""
        texts = [
            "This is a test sentence about AI",
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks"
        ]
        
        embeddings = await embedding_service.generate_embeddings(texts)
        
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_embedding_caching(self, embedding_service):
        """Test embedding caching functionality."""
        texts = ["Test sentence for caching"]
        
        # First call should generate embeddings
        embeddings1 = await embedding_service.generate_embeddings(texts, use_cache=True)
        
        # Second call should use cache
        embeddings2 = await embedding_service.generate_embeddings(texts, use_cache=True)
        
        assert embeddings1 == embeddings2

# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for the unified system."""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_processing_performance(self, document_processor, sample_markdown_file):
        """Benchmark document processing performance."""
        import time
        
        start_time = time.time()
        
        result = await document_processor.process_document(
            file_path=sample_markdown_file,
            document_id="perf-test-001"
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance assertions
        assert processing_time < 30.0  # Should complete within 30 seconds
        assert result["chunk_count"] > 0
        
        # Log performance metrics
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Chunks generated: {result['chunk_count']}")
        print(f"Time per chunk: {processing_time / result['chunk_count']:.3f} seconds")
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_search_performance(self, document_processor, sample_markdown_file):
        """Benchmark search performance."""
        import time
        
        # Setup: process document
        await document_processor.process_document(
            file_path=sample_markdown_file,
            document_id="search-perf-001"
        )
        
        # Benchmark search
        start_time = time.time()
        
        results = await document_processor.search_documents(
            query="system architecture implementation",
            limit=10
        )
        
        end_time = time.time()
        search_time = end_time - start_time
        
        # Performance assertions
        assert search_time < 1.0  # Should complete within 1 second
        assert len(results) > 0
        
        # Log performance metrics
        print(f"Search time: {search_time:.3f} seconds")
        print(f"Results returned: {len(results)}")
```

---

## Phase 2: Intelligence Layer (Weeks 5-8)

### Week 5: PocketFlow Integration Foundation

#### PocketFlow Agent Framework Setup

```python
# backend/services/pocketflow_service.py
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
from datetime import datetime

# PocketFlow imports (assuming package structure)
from pocketflow import Flow, Node, Agent, AsyncLLMNode
from pocketflow.communication import MessageBus
from pocketflow.batch import BatchProcessor as PFBatchProcessor

from backend.services.document_processor import UnifiedDocumentProcessor
from backend.services.vector_store import MilvusVectorStore
from backend.core.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class DocumentAnalysisNode(Node):
    """PocketFlow node for document analysis."""
    
    def __init__(self, llm_provider: str = "openai"):
        super().__init__()
        self.llm_provider = llm_provider
        self.document_processor = UnifiedDocumentProcessor()
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document analysis."""
        try:
            document_id = input_data["document_id"]
            analysis_type = input_data.get("analysis_type", "comprehensive")
            
            # Retrieve document chunks
            chunks = await self.document_processor.search_documents(
                query="",  # Empty query to get all chunks
                document_id=document_id,
                limit=1000  # Get all chunks
            )
            
            if not chunks:
                return {
                    "document_id": document_id,
                    "analysis": None,
                    "error": "No chunks found for document"
                }
            
            # Perform analysis based on type
            if analysis_type == "comprehensive":
                analysis = await self._comprehensive_analysis(chunks)
            elif analysis_type == "summary":
                analysis = await self._summary_analysis(chunks)
            elif analysis_type == "key_topics":
                analysis = await self._topic_analysis(chunks)
            else:
                analysis = await self._basic_analysis(chunks)
            
            return {
                "document_id": document_id,
                "analysis_type": analysis_type,
                "analysis": analysis,
                "chunk_count": len(chunks),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return {
                "document_id": input_data.get("document_id"),
                "analysis": None,
                "error": str(e)
            }
    
    async def _comprehensive_analysis(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive document analysis."""
        # Combine all chunk content
        full_content = "\n\n".join([chunk["content"] for chunk in chunks])
        
        # Create analysis prompt
        prompt = f"""
Analyze the following document comprehensively. Provide:
1. Executive Summary (2-3 sentences)
2. Key Topics (list of main topics)
3. Document Structure (outline of sections)
4. Important Insights (key findings or conclusions)
5. Content Quality Assessment (readability, completeness)

Document Content:
{full_content[:8000]}  # Limit content to avoid token limits

Provide your analysis in JSON format.
"""
        
        # Use LLM for analysis (this would integrate with your LLM factory)
        from src.llm.factory import LLMFactory
        llm = LLMFactory.get_provider(self.llm_provider)
        
        analysis_result = await llm.generate(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.3
        )
        
        try:
            import json
            analysis = json.loads(analysis_result)
        except json.JSONDecodeError:
            # Fallback to structured text analysis
            analysis = {
                "executive_summary": "Analysis completed",
                "key_topics": ["General content analysis"],
                "document_structure": "Standard document structure",
                "insights": "Document processed successfully",
                "quality_assessment": "Good",
                "raw_analysis": analysis_result
            }
        
        return analysis
    
    async def _summary_analysis(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate document summary."""
        # Select top chunks by quality score
        top_chunks = sorted(
            chunks,
            key=lambda x: x.get("quality_score", 0),
            reverse=True
        )[:10]
        
        content = "\n\n".join([chunk["content"] for chunk in top_chunks])
        
        prompt = f"""
Create a concise summary of the following document content:

{content[:6000]}

Provide a summary that captures the main points and key information.
"""
        
        from src.llm.factory import LLMFactory
        llm = LLMFactory.get_provider(self.llm_provider)
        
        summary = await llm.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.3
        )
        
        return {
            "summary": summary,
            "chunks_analyzed": len(top_chunks),
            "total_chunks": len(chunks)
        }
    
    async def _topic_analysis(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key topics from document."""
        # Use TF-IDF or similar for topic extraction
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        import numpy as np
        
        # Extract content
        texts = [chunk["content"] for chunk in chunks]
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top terms
        scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = scores.argsort()[-20:][::-1]
        top_topics = [feature_names[i] for i in top_indices]
        
        return {
            "key_topics": top_topics[:10],
            "topic_scores": [float(scores[i]) for i in top_indices[:10]],
            "analysis_method": "TF-IDF"
        }
    
    async def _basic_analysis(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform basic statistical analysis."""
        total_words = sum(len(chunk["content"].split()) for chunk in chunks)
        avg_chunk_length = total_words / len(chunks) if chunks else 0
        avg_quality = sum(chunk.get("quality_score", 0) for chunk in chunks) / len(chunks) if chunks else 0
        
        return {
            "total_chunks": len(chunks),
            "total_words": total_words,
            "average_chunk_length": avg_chunk_length,
            "average_quality_score": avg_quality,
            "analysis_type": "basic_statistics"
        }

class QueryProcessingNode(Node):
    """PocketFlow node for intelligent query processing."""
    
    def __init__(self, llm_provider: str = "openai"):
        super().__init__()
        self.llm_provider = llm_provider
        self.document_processor = UnifiedDocumentProcessor()
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query processing with RAG."""
        try:
            query = input_data["query"]
            document_id = input_data.get("document_id")
            context_limit = input_data.get("context_limit", 5)
            
            # Search for relevant chunks
            relevant_chunks = await self.document_processor.search_documents(
                query=query,
                document_id=document_id,
                limit=context_limit
            )
            
            if not relevant_chunks:
                return {
                    "query": query,
                    "response": "No relevant information found.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Build RAG prompt
            context = "\n\n".join([
                f"Source {i+1}: {chunk['content']}"
                for i, chunk in enumerate(relevant_chunks)
            ])
            
            prompt = f"""
Based on the following context, answer the user's question accurately and comprehensively.

Context:
{context}

Question: {query}

Provide a detailed answer based on the context. If the context doesn't contain enough information to answer the question, say so clearly.
"""
            
            # Generate response using LLM
            from src.llm.factory import LLMFactory
            llm = LLMFactory.get_provider(self.llm_provider)
            
            response = await llm.generate(
                prompt=prompt,
                max_tokens=800,
                temperature=0.3
            )
            
            # Calculate confidence based on similarity scores
            avg_similarity = sum(
                chunk["similarity_score"] for chunk in relevant_chunks
            ) / len(relevant_chunks)
            
            return {
                "query": query,
                "response": response,
                "sources": [
                    {
                        "chunk_id": chunk["id"],
                        "content": chunk["content"][:200] + "...",
                        "similarity_score": chunk["similarity_score"],
                        "document_id": chunk["document_id"]
                    }
                    for chunk in relevant_chunks
                ],
                "confidence": min(avg_similarity * 100, 100),
                "context_chunks_used": len(relevant_chunks)
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "query": input_data.get("query"),
                "response": f"Error processing query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }

class DocumentProcessingFlow(Flow):
    """Complete document processing flow using PocketFlow."""
    
    def __init__(self):
        # Initialize nodes
        self.analysis_node = DocumentAnalysisNode()
        self.query_node = QueryProcessingNode()
        
        # Define flow structure
        super().__init__([
            self.analysis_node,
            self.query_node
        ])
        
        logger.info("DocumentProcessingFlow initialized")
    
    async def analyze_document(self, document_id: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze document using the flow."""
        input_data = {
            "document_id": document_id,
            "analysis_type": analysis_type
        }
        
        result = await self.analysis_node.execute(input_data)
        
        logger.info(
            f"Document analysis completed",
            document_id=document_id,
            analysis_type=analysis_type,
            success=result.get("error") is None
        )
        
        return result
    
    async def process_query(self, query: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Process query using RAG."""
        input_data = {
            "query": query,
            "document_id": document_id
        }
        
        result = await self.query_node.execute(input_data)
        
        logger.info(
            f"Query processed",
            query_length=len(query),
            document_id=document_id,
            confidence=result.get("confidence", 0)
        )
        
        return result

class PocketFlowService:
    """Main service for PocketFlow integration."""
    
    def __init__(self):
        self.processing_flow = DocumentProcessingFlow()
        self.message_bus = MessageBus()
        self.active_flows: Dict[str, Flow] = {}
        
        logger.info("PocketFlowService initialized")
    
    async def create_analysis_flow(self, flow_id: str) -> str:
        """Create new analysis flow instance."""
        flow = DocumentProcessingFlow()
        self.active_flows[flow_id] = flow
        
        logger.info(f"Created analysis flow: {flow_id}")
        return flow_id
    
    async def execute_document_analysis(
        self,
        document_id: str,
        analysis_type: str = "comprehensive",
        flow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute document analysis."""
        if flow_id and flow_id in self.active_flows:
            flow = self.active_flows[flow_id]
        else:
            flow = self.processing_flow
        
        return await flow.analyze_document(document_id, analysis_type)
    
    async def execute_query_processing(
        self,
        query: str,
        document_id: Optional[str] = None,
        flow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute query processing with RAG."""
        if flow_id and flow_id in self.active_flows:
            flow = self.active_flows[flow_id]
        else:
            flow = self.processing_flow
        
        return await flow.process_query(query, document_id)
    
    async def batch_process_documents(
        self,
        document_ids: List[str],
        analysis_type: str = "summary"
    ) -> List[Dict[str, Any]]:
        """Process multiple documents in batch."""
        batch_processor = PFBatchProcessor(batch_size=5)
        
        async def process_single(doc_id: str) -> Dict[str, Any]:
            return await self.execute_document_analysis(doc_id, analysis_type)
        
        results = await batch_processor.process_batch(
            items=document_ids,
            processor_func=process_single
        )
        
        logger.info(
            f"Batch processing completed",
            document_count=len(document_ids),
            analysis_type=analysis_type
        )
        
        return results
    
    def cleanup_flow(self, flow_id: str):
        """Clean up flow instance."""
        if flow_id in self.active_flows:
            del self.active_flows[flow_id]
            logger.info(f"Cleaned up flow: {flow_id}")
```

### Week 6: Multi-Agent Collaboration

```python
# backend/services/multi_agent_system.py
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
from enum import Enum

from pocketflow import Agent, Flow, Node
from pocketflow.communication import MessageBus, Message

from backend.services.pocketflow_service import PocketFlowService
from backend.services.document_processor import UnifiedDocumentProcessor
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class AgentRole(Enum):
    """Agent roles in the multi-agent system."""
    ANALYZER = "analyzer"
    SUMMARIZER = "summarizer"
    QUALITY_CHECKER = "quality_checker"
    COORDINATOR = "coordinator"

class DocumentAnalyzerAgent(Agent):
    """Agent specialized in document analysis."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.role = AgentRole.ANALYZER
        self.document_processor = UnifiedDocumentProcessor()
        self.specialization = "comprehensive_analysis"
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming messages."""
        if message.type == "analyze_document":
            return await self._handle_analysis_request(message)
        elif message.type == "extract_topics":
            return await self._handle_topic_extraction(message)
        else:
            logger.warning(f"Unknown message type: {message.type}")
            return None
    
    async def _handle_analysis_request(self, message: Message) -> Message:
        """Handle document analysis request."""
        try:
            document_id = message.data["document_id"]
            analysis_depth = message.data.get("depth", "standard")
            
            # Retrieve document chunks
            chunks = await self.document_processor.search_documents(
                query="",
                document_id=document_id,
                limit=1000
            )
            
            # Perform analysis based on depth
            if analysis_depth == "deep":
                analysis = await self._deep_analysis(chunks)
            else:
                analysis = await self._standard_analysis(chunks)
            
            return Message(
                type="analysis_complete",
                data={
                    "document_id": document_id,
                    "analysis": analysis,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                },
                sender=self.agent_id
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return Message(
                type="analysis_error",
                data={"error": str(e), "document_id": document_id},
                sender=self.agent_id
            )
    
    async def _deep_analysis(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform deep document analysis."""
        # Advanced analysis with multiple perspectives
        content_analysis = await self._analyze_content_structure(chunks)
        semantic_analysis = await self._analyze_semantic_patterns(chunks)
        quality_analysis = await self._analyze_content_quality(chunks)
        
        return {
            "analysis_type": "deep",
            "content_structure": content_analysis,
            "semantic_patterns": semantic_analysis,
            "quality_assessment": quality_analysis,
            "chunk_count": len(chunks)
        }
    
    async def _standard_analysis(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform standard document analysis."""
        total_words = sum(len(chunk["content"].split()) for chunk in chunks)
        avg_quality = sum(chunk.get("quality_score", 0) for chunk in chunks) / len(chunks)
        
        return {
            "analysis_type": "standard",
            "total_chunks": len(chunks),
            "total_words": total_words,
            "average_quality": avg_quality,
            "readability_score": await self._calculate_readability(chunks)
        }

class DocumentSummarizerAgent(Agent):
    """Agent specialized in document summarization."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.role = AgentRole.SUMMARIZER
        self.document_processor = UnifiedDocumentProcessor()
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process summarization requests."""
        if message.type == "create_summary":
            return await self._handle_summarization(message)
        return None
    
    async def _handle_summarization(self, message: Message) -> Message:
        """Handle document summarization."""
        try:
            document_id = message.data["document_id"]
            summary_type = message.data.get("summary_type", "executive")
            max_length = message.data.get("max_length", 500)
            
            # Get document chunks
            chunks = await self.document_processor.search_documents(
                query="",
                document_id=document_id,
                limit=1000
            )
            
            # Create summary based on type
            if summary_type == "executive":
                summary = await self._create_executive_summary(chunks, max_length)
            elif summary_type == "technical":
                summary = await self._create_technical_summary(chunks, max_length)
            else:
                summary = await self._create_general_summary(chunks, max_length)
            
            return Message(
                type="summary_complete",
                data={
                    "document_id": document_id,
                    "summary": summary,
                    "summary_type": summary_type,
                    "agent_id": self.agent_id
                },
                sender=self.agent_id
            )
            
        except Exception as e:
            return Message(
                type="summary_error",
                data={"error": str(e)},
                sender=self.agent_id
            )

class QualityCheckerAgent(Agent):
    """Agent specialized in quality assessment."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.role = AgentRole.QUALITY_CHECKER
        self.quality_thresholds = {
            "readability": 0.7,
            "completeness": 0.8,
            "coherence": 0.75
        }
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process quality check requests."""
        if message.type == "check_quality":
            return await self._handle_quality_check(message)
        return None
    
    async def _handle_quality_check(self, message: Message) -> Message:
        """Handle quality assessment."""
        try:
            content = message.data.get("content")
            analysis_result = message.data.get("analysis")
            
            quality_metrics = {
                "readability_score": await self._assess_readability(content),
                "completeness_score": await self._assess_completeness(analysis_result),
                "coherence_score": await self._assess_coherence(content),
                "overall_quality": 0.0
            }
            
            # Calculate overall quality
            quality_metrics["overall_quality"] = (
                quality_metrics["readability_score"] * 0.3 +
                quality_metrics["completeness_score"] * 0.4 +
                quality_metrics["coherence_score"] * 0.3
            )
            
            # Determine if quality meets thresholds
            quality_passed = all(
                quality_metrics[f"{metric}_score"] >= threshold
                for metric, threshold in self.quality_thresholds.items()
            )
            
            return Message(
                type="quality_check_complete",
                data={
                    "quality_metrics": quality_metrics,
                    "quality_passed": quality_passed,
                    "recommendations": await self._generate_recommendations(quality_metrics),
                    "agent_id": self.agent_id
                },
                sender=self.agent_id
            )
            
        except Exception as e:
            return Message(
                type="quality_check_error",
                data={"error": str(e)},
                sender=self.agent_id
            )

class CoordinatorAgent(Agent):
    """Agent that coordinates multi-agent workflows."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.role = AgentRole.COORDINATOR
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.agent_registry: Dict[str, Agent] = {}
    
    def register_agent(self, agent: Agent):
        """Register an agent with the coordinator."""
        self.agent_registry[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id}")
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process coordination requests."""
        if message.type == "start_workflow":
            return await self._handle_workflow_start(message)
        elif message.type == "workflow_step_complete":
            return await self._handle_step_completion(message)
        return None
    
    async def _handle_workflow_start(self, message: Message) -> Message:
        """Start a new multi-agent workflow."""
        workflow_id = message.data["workflow_id"]
        workflow_type = message.data["workflow_type"]
        document_id = message.data["document_id"]
        
        # Define workflow steps based on type
        if workflow_type == "comprehensive_analysis":
            steps = [
                {"agent_role": AgentRole.ANALYZER, "task": "analyze_document"},
                {"agent_role": AgentRole.SUMMARIZER, "task": "create_summary"},
                {"agent_role": AgentRole.QUALITY_CHECKER, "task": "check_quality"}
            ]
        else:
            steps = [{"agent_role": AgentRole.ANALYZER, "task": "analyze_document"}]
        
        # Initialize workflow state
        self.active_workflows[workflow_id] = {
            "type": workflow_type,
            "document_id": document_id,
            "steps": steps,
            "current_step": 0,
            "results": {},
            "status": "running",
            "started_at": datetime.now().isoformat()
        }
        
        # Start first step
        await self._execute_workflow_step(workflow_id)
        
        return Message(
            type="workflow_started",
            data={"workflow_id": workflow_id, "total_steps": len(steps)},
            sender=self.agent_id
        )
    
    async def _execute_workflow_step(self, workflow_id: str):
        """Execute the current step of a workflow."""
        workflow = self.active_workflows[workflow_id]
        current_step = workflow["current_step"]
        
        if current_step >= len(workflow["steps"]):
            # Workflow complete
            workflow["status"] = "completed"
            workflow["completed_at"] = datetime.now().isoformat()
            return
        
        step = workflow["steps"][current_step]
        agent_role = step["agent_role"]
        task = step["task"]
        
        # Find appropriate agent
        target_agent = None
        for agent in self.agent_registry.values():
            if hasattr(agent, 'role') and agent.role == agent_role:
                target_agent = agent
                break
        
        if target_agent:
            # Send task to agent
            task_message = Message(
                type=task,
                data={
                    "document_id": workflow["document_id"],
                    "workflow_id": workflow_id,
                    "previous_results": workflow["results"]
                },
                sender=self.agent_id
            )
            
            result = await target_agent.process_message(task_message)
            if result:
                workflow["results"][f"step_{current_step}"] = result.data
                workflow["current_step"] += 1
                
                # Continue to next step
                await self._execute_workflow_step(workflow_id)

class MultiAgentOrchestrator:
    """Orchestrator for multi-agent document processing."""
    
    def __init__(self):
        self.message_bus = MessageBus()
        self.coordinator = CoordinatorAgent("coordinator_001")
        
        # Initialize specialized agents
        self.agents = {
            "analyzer_001": DocumentAnalyzerAgent("analyzer_001"),
            "summarizer_001": DocumentSummarizerAgent("summarizer_001"),
            "quality_checker_001": QualityCheckerAgent("quality_checker_001")
        }
        
        # Register agents with coordinator
        for agent in self.agents.values():
            self.coordinator.register_agent(agent)
        
        logger.info("MultiAgentOrchestrator initialized")
    
    async def process_document_comprehensive(
        self,
        document_id: str,
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process document using multi-agent collaboration."""
        if not workflow_id:
            workflow_id = f"workflow_{document_id}_{int(datetime.now().timestamp())}"
        
        # Start comprehensive analysis workflow
        start_message = Message(
            type="start_workflow",
            data={
                "workflow_id": workflow_id,
                "workflow_type": "comprehensive_analysis",
                "document_id": document_id
            },
            sender="orchestrator"
        )
        
        await self.coordinator.process_message(start_message)
        
        # Wait for workflow completion
        max_wait_time = 300  # 5 minutes
        wait_interval = 1
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            workflow = self.coordinator.active_workflows.get(workflow_id)
            if workflow and workflow["status"] == "completed":
                return {
                    "workflow_id": workflow_id,
                    "document_id": document_id,
                    "results": workflow["results"],
                    "processing_time": elapsed_time,
                    "success": True
                }
            
            await asyncio.sleep(wait_interval)
            elapsed_time += wait_interval
        
        # Timeout
        return {
            "workflow_id": workflow_id,
            "document_id": document_id,
            "error": "Workflow timeout",
            "success": False
        }
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a running workflow."""
        workflow = self.coordinator.active_workflows.get(workflow_id)
        if not workflow:
            return {"error": "Workflow not found"}
        
        return {
             "workflow_id": workflow_id,
             "status": workflow["status"],
             "current_step": workflow["current_step"],
             "total_steps": len(workflow["steps"]),
             "progress_percentage": (workflow["current_step"] / len(workflow["steps"])) * 100
         }
```

### Week 7: Advanced RAG Implementation

```python
# backend/services/advanced_rag.py
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.services.vector_store import MilvusVectorStore
from backend.services.embedding_service import EmbeddingService
from backend.services.pocketflow_service import PocketFlowService
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class AdvancedRAGService:
    """Advanced RAG with multi-step reasoning and context optimization."""
    
    def __init__(self):
        self.vector_store = MilvusVectorStore()
        self.embedding_service = EmbeddingService()
        self.pocketflow_service = PocketFlowService()
        self.context_window_size = 4000  # tokens
        self.max_chunks_per_query = 10
        
        logger.info("AdvancedRAGService initialized")
    
    async def process_complex_query(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        reasoning_depth: str = "standard"
    ) -> Dict[str, Any]:
        """Process complex queries with multi-step reasoning."""
        try:
            # Step 1: Query decomposition
            sub_queries = await self._decompose_query(query)
            
            # Step 2: Multi-step retrieval
            all_contexts = []
            for sub_query in sub_queries:
                contexts = await self._retrieve_contexts(
                    sub_query,
                    document_ids,
                    limit=self.max_chunks_per_query
                )
                all_contexts.extend(contexts)
            
            # Step 3: Context optimization
            optimized_context = await self._optimize_context(
                query,
                all_contexts,
                self.context_window_size
            )
            
            # Step 4: Multi-step reasoning
            if reasoning_depth == "deep":
                response = await self._deep_reasoning(query, optimized_context)
            else:
                response = await self._standard_reasoning(query, optimized_context)
            
            # Step 5: Response validation
            validation_result = await self._validate_response(
                query,
                response,
                optimized_context
            )
            
            return {
                "query": query,
                "sub_queries": sub_queries,
                "response": response,
                "contexts_used": len(optimized_context),
                "reasoning_depth": reasoning_depth,
                "validation": validation_result,
                "confidence_score": validation_result.get("confidence", 0.0),
                "processing_steps": {
                    "decomposition": len(sub_queries),
                    "retrieval": len(all_contexts),
                    "optimization": len(optimized_context),
                    "reasoning": reasoning_depth
                }
            }
            
        except Exception as e:
            logger.error(f"Complex query processing failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "success": False
            }
    
    async def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries."""
        decomposition_prompt = f"""
Analyze the following query and break it down into simpler sub-queries that can be answered independently.
Each sub-query should focus on a specific aspect of the main question.

Main Query: {query}

Provide 2-4 focused sub-queries as a JSON list.
Example: ["What is X?", "How does Y work?", "What are the benefits of Z?"]
"""
        
        from src.llm.factory import LLMFactory
        llm = LLMFactory.get_provider("openai")
        
        result = await llm.generate(
            prompt=decomposition_prompt,
            max_tokens=300,
            temperature=0.3
        )
        
        try:
            import json
            sub_queries = json.loads(result)
            if isinstance(sub_queries, list):
                return sub_queries
        except json.JSONDecodeError:
            pass
        
        # Fallback: return original query
        return [query]
    
    async def _retrieve_contexts(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant contexts for a query."""
        # Generate query embedding
        query_embeddings = await self.embedding_service.generate_embeddings([query])
        query_embedding = query_embeddings[0]
        
        # Search for each document or globally
        all_results = []
        
        if document_ids:
            for doc_id in document_ids:
                results = await self.vector_store.search_similar(
                    query_embedding=query_embedding,
                    limit=limit,
                    document_id=doc_id
                )
                all_results.extend(results)
        else:
            results = await self.vector_store.search_similar(
                query_embedding=query_embedding,
                limit=limit * 2  # Get more results for global search
            )
            all_results.extend(results)
        
        # Remove duplicates and sort by similarity
        unique_results = {}
        for result in all_results:
            chunk_id = result["id"]
            if chunk_id not in unique_results or result["similarity_score"] > unique_results[chunk_id]["similarity_score"]:
                unique_results[chunk_id] = result
        
        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: x["similarity_score"],
            reverse=True
        )
        
        return sorted_results[:limit]
    
    async def _optimize_context(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        max_tokens: int
    ) -> List[Dict[str, Any]]:
        """Optimize context selection for the query."""
        if not contexts:
            return []
        
        # Calculate relevance scores
        query_embeddings = await self.embedding_service.generate_embeddings([query])
        query_embedding = np.array(query_embeddings[0])
        
        context_embeddings = await self.embedding_service.generate_embeddings(
            [ctx["content"] for ctx in contexts]
        )
        
        # Calculate similarity matrix for diversity
        context_matrix = np.array(context_embeddings)
        similarity_matrix = cosine_similarity(context_matrix)
        
        # Select diverse, relevant contexts
        selected_contexts = []
        selected_indices = set()
        current_tokens = 0
        
        # Sort by relevance first
        sorted_contexts = sorted(
            enumerate(contexts),
            key=lambda x: x[1]["similarity_score"],
            reverse=True
        )
        
        for idx, context in sorted_contexts:
            if idx in selected_indices:
                continue
            
            # Estimate token count (rough approximation)
            estimated_tokens = len(context["content"].split()) * 1.3
            
            if current_tokens + estimated_tokens > max_tokens:
                break
            
            # Check diversity (avoid too similar contexts)
            is_diverse = True
            for selected_idx in selected_indices:
                if similarity_matrix[idx][selected_idx] > 0.85:  # Too similar
                    is_diverse = False
                    break
            
            if is_diverse:
                selected_contexts.append(context)
                selected_indices.add(idx)
                current_tokens += estimated_tokens
        
        return selected_contexts
    
    async def _standard_reasoning(
        self,
        query: str,
        contexts: List[Dict[str, Any]]
    ) -> str:
        """Standard reasoning with retrieved contexts."""
        if not contexts:
            return "I don't have enough information to answer this question."
        
        context_text = "\n\n".join([
            f"Source {i+1}: {ctx['content']}"
            for i, ctx in enumerate(contexts)
        ])
        
        reasoning_prompt = f"""
Based on the provided context, answer the following question comprehensively and accurately.

Context:
{context_text}

Question: {query}

Instructions:
1. Use only information from the provided context
2. If the context doesn't contain sufficient information, state this clearly
3. Provide specific references to sources when possible
4. Structure your answer logically

Answer:
"""
        
        from src.llm.factory import LLMFactory
        llm = LLMFactory.get_provider("openai")
        
        response = await llm.generate(
            prompt=reasoning_prompt,
            max_tokens=800,
            temperature=0.3
        )
        
        return response
    
    async def _deep_reasoning(
        self,
        query: str,
        contexts: List[Dict[str, Any]]
    ) -> str:
        """Deep reasoning with multi-step analysis."""
        if not contexts:
            return "I don't have enough information to answer this question."
        
        # Step 1: Initial analysis
        initial_response = await self._standard_reasoning(query, contexts)
        
        # Step 2: Critical analysis
        critical_prompt = f"""
Analyze the following response for accuracy, completeness, and logical consistency:

Original Question: {query}
Response: {initial_response}

Provide:
1. Accuracy assessment
2. Completeness evaluation
3. Logical consistency check
4. Suggestions for improvement
"""
        
        from src.llm.factory import LLMFactory
        llm = LLMFactory.get_provider("openai")
        
        critical_analysis = await llm.generate(
            prompt=critical_prompt,
            max_tokens=400,
            temperature=0.3
        )
        
        # Step 3: Refined response
        refinement_prompt = f"""
Based on the critical analysis, provide an improved response to the original question.

Original Question: {query}
Initial Response: {initial_response}
Critical Analysis: {critical_analysis}

Provide a refined, comprehensive answer that addresses any identified issues:
"""
        
        refined_response = await llm.generate(
            prompt=refinement_prompt,
            max_tokens=1000,
            temperature=0.3
        )
        
        return refined_response
    
    async def _validate_response(
        self,
        query: str,
        response: str,
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate response quality and accuracy."""
        validation_prompt = f"""
Evaluate the following response for quality and accuracy:

Question: {query}
Response: {response}

Rate the response on a scale of 0-100 for:
1. Accuracy (based on context)
2. Completeness (addresses all aspects of question)
3. Clarity (easy to understand)
4. Relevance (directly answers the question)

Provide scores as JSON: {{"accuracy": X, "completeness": Y, "clarity": Z, "relevance": W, "overall": average}}
"""
        
        from src.llm.factory import LLMFactory
        llm = LLMFactory.get_provider("openai")
        
        validation_result = await llm.generate(
            prompt=validation_prompt,
            max_tokens=200,
            temperature=0.1
        )
        
        try:
            import json
            scores = json.loads(validation_result)
            return {
                "scores": scores,
                "confidence": scores.get("overall", 0) / 100,
                "validation_method": "llm_evaluation"
            }
        except json.JSONDecodeError:
            return {
                "scores": {"overall": 50},
                "confidence": 0.5,
                "validation_method": "fallback",
                "raw_validation": validation_result
            }
```

### Week 8: Performance Optimization

```python
# backend/services/performance_optimizer.py
from typing import List, Dict, Any, Optional
import asyncio
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import psutil
import GPUtil

from backend.utils.logger import get_logger
from backend.core.config import settings

logger = get_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    processing_time: float
    throughput: float
    error_rate: float
    cache_hit_rate: float

class PerformanceOptimizer:
    """System performance optimization and monitoring."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_strategies = {
            "batch_processing": self._optimize_batch_processing,
            "caching": self._optimize_caching,
            "parallel_processing": self._optimize_parallel_processing,
            "memory_management": self._optimize_memory_management
        }
        self.thread_pool = ThreadPoolExecutor(max_workers=settings.MAX_WORKER_THREADS)
        
        logger.info("PerformanceOptimizer initialized")
    
    async def monitor_system_performance(self) -> PerformanceMetrics:
        """Monitor current system performance."""
        start_time = time.time()
        
        # CPU and Memory metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU metrics (if available)
        gpu_usage = None
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
        except Exception:
            pass
        
        processing_time = time.time() - start_time
        
        metrics = PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            processing_time=processing_time,
            throughput=0.0,  # Will be calculated based on actual operations
            error_rate=0.0,  # Will be tracked separately
            cache_hit_rate=0.0  # Will be tracked separately
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    async def optimize_system_performance(self) -> Dict[str, Any]:
        """Apply performance optimizations based on current metrics."""
        current_metrics = await self.monitor_system_performance()
        optimizations_applied = []
        
        # Apply optimizations based on current performance
        if current_metrics.cpu_usage > 80:
            await self._optimize_parallel_processing()
            optimizations_applied.append("parallel_processing")
        
        if current_metrics.memory_usage > 85:
            await self._optimize_memory_management()
            optimizations_applied.append("memory_management")
        
        # Always optimize caching
        await self._optimize_caching()
        optimizations_applied.append("caching")
        
        return {
            "current_metrics": current_metrics,
            "optimizations_applied": optimizations_applied,
            "recommendations": await self._generate_recommendations(current_metrics)
        }
    
    async def _optimize_batch_processing(self) -> Dict[str, Any]:
        """Optimize batch processing parameters."""
        # Analyze historical performance to determine optimal batch sizes
        if len(self.metrics_history) < 10:
            return {"status": "insufficient_data"}
        
        avg_cpu = sum(m.cpu_usage for m in self.metrics_history[-10:]) / 10
        avg_memory = sum(m.memory_usage for m in self.metrics_history[-10:]) / 10
        
        # Adjust batch sizes based on resource usage
        if avg_cpu > 70 or avg_memory > 70:
            new_batch_size = max(settings.EMBEDDING_BATCH_SIZE // 2, 1)
        else:
            new_batch_size = min(settings.EMBEDDING_BATCH_SIZE * 2, 32)
        
        # Update settings (this would need to be persisted)
        settings.EMBEDDING_BATCH_SIZE = new_batch_size
        
        return {
            "status": "optimized",
            "new_batch_size": new_batch_size,
            "avg_cpu": avg_cpu,
            "avg_memory": avg_memory
        }
    
    async def _optimize_caching(self) -> Dict[str, Any]:
        """Optimize caching strategies."""
        # This would integrate with your cache manager
        # For now, we'll simulate cache optimization
        
        cache_optimizations = {
            "embedding_cache_ttl": settings.EMBEDDING_CACHE_TTL,
            "result_cache_size": 1000,
            "precompute_common_queries": True
        }
        
        return {
            "status": "optimized",
            "optimizations": cache_optimizations
        }
    
    async def _optimize_parallel_processing(self) -> Dict[str, Any]:
        """Optimize parallel processing configuration."""
        cpu_count = psutil.cpu_count()
        current_workers = settings.MAX_WORKER_THREADS
        
        # Adjust worker count based on CPU usage
        recent_cpu = sum(m.cpu_usage for m in self.metrics_history[-5:]) / 5 if self.metrics_history else 50
        
        if recent_cpu > 80:
            new_workers = max(current_workers - 2, 2)
        elif recent_cpu < 50:
            new_workers = min(current_workers + 2, cpu_count * 2)
        else:
            new_workers = current_workers
        
        settings.MAX_WORKER_THREADS = new_workers
        
        return {
            "status": "optimized",
            "old_workers": current_workers,
            "new_workers": new_workers,
            "cpu_usage": recent_cpu
        }
    
    async def _optimize_memory_management(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Get memory info after cleanup
        memory_after = psutil.virtual_memory()
        
        optimizations = {
            "garbage_collection": "performed",
            "memory_after_cleanup": memory_after.percent,
            "recommendations": []
        }
        
        if memory_after.percent > 80:
            optimizations["recommendations"].extend([
                "Consider reducing batch sizes",
                "Implement more aggressive caching cleanup",
                "Consider scaling horizontally"
            ])
        
        return optimizations
    
    async def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if metrics.cpu_usage > 80:
            recommendations.append("High CPU usage detected. Consider scaling horizontally or optimizing algorithms.")
        
        if metrics.memory_usage > 85:
            recommendations.append("High memory usage detected. Consider implementing memory pooling or reducing batch sizes.")
        
        if metrics.gpu_usage and metrics.gpu_usage > 90:
            recommendations.append("GPU utilization is very high. Consider batch optimization or additional GPU resources.")
        
        if metrics.processing_time > 10:
            recommendations.append("Processing time is high. Consider implementing async processing or caching.")
        
        if not recommendations:
            recommendations.append("System performance is optimal.")
        
        return recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary over time."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        return {
            "avg_cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            "avg_memory_usage": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            "avg_processing_time": sum(m.processing_time for m in recent_metrics) / len(recent_metrics),
            "total_samples": len(self.metrics_history),
            "monitoring_duration": len(self.metrics_history) * 60  # Assuming 1 minute intervals
        }
```

---

## Phase 3: Frontend & Deployment (Weeks 9-12)

### Week 9: NextJS Frontend Foundation

#### Modern React Components with TypeScript

```typescript
// frontend/types/api.ts
export interface DocumentProcessingRequest {
  file: File;
  metadata?: Record<string, any>;
  analysisType?: 'comprehensive' | 'summary' | 'key_topics';
}

export interface DocumentProcessingResponse {
  documentId: string;
  chunkCount: number;
  chunkIds: string[];
  processingTimeMs: number;
  qualityMetrics: QualityMetrics;
  success: boolean;
  metadata: Record<string, any>;
}

export interface QualityMetrics {
  averageQuality: number;
  readabilityScore: number;
  completenessScore: number;
  coherenceScore: number;
}

export interface SearchRequest {
  query: string;
  documentId?: string;
  limit?: number;
  qualityThreshold?: number;
  reasoningDepth?: 'standard' | 'deep';
}

export interface SearchResponse {
  query: string;
  response: string;
  sources: SearchSource[];
  confidence: number;
  processingSteps?: ProcessingSteps;
}

export interface SearchSource {
  chunkId: string;
  content: string;
  similarityScore: number;
  documentId: string;
  metadata: Record<string, any>;
}

export interface ProcessingSteps {
  decomposition: number;
  retrieval: number;
  optimization: number;
  reasoning: string;
}
```

```typescript
// frontend/hooks/useDocumentProcessing.ts
import { useState, useCallback } from 'react';
import { DocumentProcessingRequest, DocumentProcessingResponse } from '../types/api';
import { apiClient } from '../utils/apiClient';

export const useDocumentProcessing = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const processDocument = useCallback(async (
    request: DocumentProcessingRequest
  ): Promise<DocumentProcessingResponse | null> => {
    setIsProcessing(true);
    setError(null);
    setProcessingProgress(0);

    try {
      const formData = new FormData();
      formData.append('file', request.file);
      if (request.metadata) {
        formData.append('metadata', JSON.stringify(request.metadata));
      }
      if (request.analysisType) {
        formData.append('analysisType', request.analysisType);
      }

      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProcessingProgress(prev => Math.min(prev + 10, 90));
      }, 500);

      const response = await apiClient.post<DocumentProcessingResponse>(
        '/api/documents/process',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      clearInterval(progressInterval);
      setProcessingProgress(100);
      
      return response.data;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Processing failed');
      return null;
    } finally {
      setIsProcessing(false);
    }
  }, []);

  return {
     processDocument,
     isProcessing,
     processingProgress,
     error,
   };
 };
 ```

```typescript
// frontend/components/DocumentUpload.tsx
import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, CheckCircle } from 'lucide-react';
import { useDocumentProcessing } from '../hooks/useDocumentProcessing';
import { DocumentProcessingResponse } from '../types/api';

interface DocumentUploadProps {
  onProcessingComplete?: (result: DocumentProcessingResponse) => void;
  acceptedFileTypes?: string[];
  maxFileSize?: number;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({
  onProcessingComplete,
  acceptedFileTypes = ['.md', '.txt', '.pdf', '.docx'],
  maxFileSize = 10 * 1024 * 1024, // 10MB
}) => {
  const { processDocument, isProcessing, processingProgress, error } = useDocumentProcessing();
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [processingResult, setProcessingResult] = useState<DocumentProcessingResponse | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploadedFile(file);
    setProcessingResult(null);

    const result = await processDocument({
      file,
      metadata: {
        uploadedAt: new Date().toISOString(),
        originalName: file.name,
        size: file.size,
      },
      analysisType: 'comprehensive',
    });

    if (result) {
      setProcessingResult(result);
      onProcessingComplete?.(result);
    }
  }, [processDocument, onProcessingComplete]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedFileTypes.reduce((acc, type) => ({ ...acc, [type]: [] }), {}),
    maxSize: maxFileSize,
    multiple: false,
    disabled: isProcessing,
  });

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
          ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        {isProcessing ? (
          <div className="space-y-4">
            <div className="animate-spin mx-auto w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full" />
            <div className="space-y-2">
              <p className="text-sm font-medium text-gray-700">Processing document...</p>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${processingProgress}%` }}
                />
              </div>
              <p className="text-xs text-gray-500">{processingProgress}% complete</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <Upload className="mx-auto w-12 h-12 text-gray-400" />
            <div>
              <p className="text-lg font-medium text-gray-700">
                {isDragActive ? 'Drop your document here' : 'Upload a document'}
              </p>
              <p className="text-sm text-gray-500 mt-1">
                Drag and drop or click to select ({acceptedFileTypes.join(', ')})
              </p>
              <p className="text-xs text-gray-400 mt-1">
                Max file size: {Math.round(maxFileSize / (1024 * 1024))}MB
              </p>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" />
          <div>
            <h4 className="text-sm font-medium text-red-800">Processing Error</h4>
            <p className="text-sm text-red-700 mt-1">{error}</p>
          </div>
        </div>
      )}

      {processingResult && (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-start space-x-3">
            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <h4 className="text-sm font-medium text-green-800">Document Processed Successfully</h4>
              <div className="mt-2 space-y-1 text-sm text-green-700">
                <p>Document ID: <code className="bg-green-100 px-1 rounded">{processingResult.documentId}</code></p>
                <p>Chunks created: {processingResult.chunkCount}</p>
                <p>Processing time: {processingResult.processingTimeMs}ms</p>
                <p>Average quality: {(processingResult.qualityMetrics.averageQuality * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
```

```typescript
// frontend/components/SearchInterface.tsx
import React, { useState, useCallback } from 'react';
import { Search, Brain, Clock, Target } from 'lucide-react';
import { SearchRequest, SearchResponse } from '../types/api';
import { apiClient } from '../utils/apiClient';

interface SearchInterfaceProps {
  documentId?: string;
  onSearchComplete?: (result: SearchResponse) => void;
}

export const SearchInterface: React.FC<SearchInterfaceProps> = ({
  documentId,
  onSearchComplete,
}) => {
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [searchConfig, setSearchConfig] = useState({
    reasoningDepth: 'standard' as 'standard' | 'deep',
    qualityThreshold: 0.7,
    limit: 10,
  });

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;

    setIsSearching(true);
    try {
      const searchRequest: SearchRequest = {
        query: query.trim(),
        documentId,
        ...searchConfig,
      };

      const response = await apiClient.post<SearchResponse>('/api/search', searchRequest);
      setSearchResult(response.data);
      onSearchComplete?.(response.data);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setIsSearching(false);
    }
  }, [query, documentId, searchConfig, onSearchComplete]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6">
      {/* Search Input */}
      <div className="space-y-4">
        <div className="relative">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about your documents..."
            className="w-full p-4 pr-12 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={3}
            disabled={isSearching}
          />
          <button
            onClick={handleSearch}
            disabled={!query.trim() || isSearching}
            className="absolute bottom-3 right-3 p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSearching ? (
              <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
            ) : (
              <Search className="w-4 h-4" />
            )}
          </button>
        </div>

        {/* Search Configuration */}
        <div className="flex flex-wrap gap-4 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-2">
            <Brain className="w-4 h-4 text-gray-600" />
            <label className="text-sm font-medium text-gray-700">Reasoning:</label>
            <select
              value={searchConfig.reasoningDepth}
              onChange={(e) => setSearchConfig(prev => ({ ...prev, reasoningDepth: e.target.value as 'standard' | 'deep' }))}
              className="text-sm border border-gray-300 rounded px-2 py-1"
            >
              <option value="standard">Standard</option>
              <option value="deep">Deep Analysis</option>
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <Target className="w-4 h-4 text-gray-600" />
            <label className="text-sm font-medium text-gray-700">Quality Threshold:</label>
            <input
              type="range"
              min="0.1"
              max="1"
              step="0.1"
              value={searchConfig.qualityThreshold}
              onChange={(e) => setSearchConfig(prev => ({ ...prev, qualityThreshold: parseFloat(e.target.value) }))}
              className="w-20"
            />
            <span className="text-sm text-gray-600">{(searchConfig.qualityThreshold * 100).toFixed(0)}%</span>
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-700">Results:</span>
            <input
              type="number"
              min="1"
              max="20"
              value={searchConfig.limit}
              onChange={(e) => setSearchConfig(prev => ({ ...prev, limit: parseInt(e.target.value) }))}
              className="w-16 text-sm border border-gray-300 rounded px-2 py-1"
            />
          </div>
        </div>
      </div>

      {/* Search Results */}
      {searchResult && (
        <div className="space-y-6">
          {/* Main Response */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="flex items-start justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Response</h3>
              <div className="flex items-center space-x-4 text-sm text-gray-500">
                <div className="flex items-center space-x-1">
                  <Target className="w-4 h-4" />
                  <span>Confidence: {(searchResult.confidence * 100).toFixed(1)}%</span>
                </div>
                {searchResult.processingSteps && (
                  <div className="flex items-center space-x-1">
                    <Clock className="w-4 h-4" />
                    <span>Steps: {searchResult.processingSteps.reasoning}</span>
                  </div>
                )}
              </div>
            </div>
            <div className="prose max-w-none">
              <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">{searchResult.response}</p>
            </div>
          </div>

          {/* Sources */}
          {searchResult.sources.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <h4 className="text-md font-semibold text-gray-900 mb-4">Sources ({searchResult.sources.length})</h4>
              <div className="space-y-4">
                {searchResult.sources.map((source, index) => (
                  <div key={source.chunkId} className="border border-gray-100 rounded-lg p-4">
                    <div className="flex items-start justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700">Source {index + 1}</span>
                      <div className="flex items-center space-x-2 text-xs text-gray-500">
                        <span>Similarity: {(source.similarityScore * 100).toFixed(1)}%</span>
                        <span>•</span>
                        <span>ID: {source.chunkId.slice(0, 8)}...</span>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 leading-relaxed">{source.content}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Processing Details */}
          {searchResult.processingSteps && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">Processing Details</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Decomposition:</span>
                  <span className="ml-2 font-medium">{searchResult.processingSteps.decomposition}</span>
                </div>
                <div>
                  <span className="text-gray-600">Retrieval:</span>
                  <span className="ml-2 font-medium">{searchResult.processingSteps.retrieval}</span>
                </div>
                <div>
                  <span className="text-gray-600">Optimization:</span>
                  <span className="ml-2 font-medium">{searchResult.processingSteps.optimization}</span>
                </div>
                <div>
                  <span className="text-gray-600">Reasoning:</span>
                  <span className="ml-2 font-medium">{searchResult.processingSteps.reasoning}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
```

### Week 10: Advanced Frontend Features

```typescript
// frontend/components/Dashboard.tsx
import React, { useState, useEffect } from 'react';
import { FileText, Search, BarChart3, Settings, Upload } from 'lucide-react';
import { DocumentUpload } from './DocumentUpload';
import { SearchInterface } from './SearchInterface';
import { DocumentProcessingResponse, SearchResponse } from '../types/api';
import { apiClient } from '../utils/apiClient';

interface DashboardStats {
  totalDocuments: number;
  totalChunks: number;
  totalSearches: number;
  averageQuality: number;
}

export const Dashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'upload' | 'search' | 'analytics'>('upload');
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recentDocuments, setRecentDocuments] = useState<DocumentProcessingResponse[]>([]);
  const [recentSearches, setRecentSearches] = useState<SearchResponse[]>([]);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      const [statsResponse, documentsResponse, searchesResponse] = await Promise.all([
        apiClient.get<DashboardStats>('/api/stats'),
        apiClient.get<DocumentProcessingResponse[]>('/api/documents/recent'),
        apiClient.get<SearchResponse[]>('/api/searches/recent'),
      ]);

      setStats(statsResponse.data);
      setRecentDocuments(documentsResponse.data);
      setRecentSearches(searchesResponse.data);
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    }
  };

  const handleDocumentProcessed = (result: DocumentProcessingResponse) => {
    setRecentDocuments(prev => [result, ...prev.slice(0, 4)]);
    loadDashboardData(); // Refresh stats
  };

  const handleSearchComplete = (result: SearchResponse) => {
    setRecentSearches(prev => [result, ...prev.slice(0, 4)]);
    loadDashboardData(); // Refresh stats
  };

  const tabs = [
    { id: 'upload' as const, label: 'Upload Documents', icon: Upload },
    { id: 'search' as const, label: 'Search & Query', icon: Search },
    { id: 'analytics' as const, label: 'Analytics', icon: BarChart3 },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <FileText className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-xl font-semibold text-gray-900">Enterprise AI Document System</h1>
            </div>
            <button className="p-2 text-gray-400 hover:text-gray-600">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      {/* Stats Overview */}
      {stats && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="bg-white rounded-lg p-6 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Total Documents</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.totalDocuments}</p>
                </div>
                <FileText className="w-8 h-8 text-blue-600" />
              </div>
            </div>
            <div className="bg-white rounded-lg p-6 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Total Chunks</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.totalChunks.toLocaleString()}</p>
                </div>
                <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                  <div className="w-4 h-4 bg-green-600 rounded" />
                </div>
              </div>
            </div>
            <div className="bg-white rounded-lg p-6 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Total Searches</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.totalSearches}</p>
                </div>
                <Search className="w-8 h-8 text-purple-600" />
              </div>
            </div>
            <div className="bg-white rounded-lg p-6 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Avg Quality</p>
                  <p className="text-2xl font-bold text-gray-900">{(stats.averageQuality * 100).toFixed(1)}%</p>
                </div>
                <BarChart3 className="w-8 h-8 text-orange-600" />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm
                    ${activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }
                  `}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Tab Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'upload' && (
          <div className="space-y-8">
            <DocumentUpload onProcessingComplete={handleDocumentProcessed} />
            
            {recentDocuments.length > 0 && (
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Documents</h3>
                <div className="bg-white shadow-sm rounded-lg overflow-hidden">
                  <ul className="divide-y divide-gray-200">
                    {recentDocuments.map((doc) => (
                      <li key={doc.documentId} className="p-4 hover:bg-gray-50">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm font-medium text-gray-900">{doc.documentId}</p>
                            <p className="text-sm text-gray-500">
                              {doc.chunkCount} chunks • Quality: {(doc.qualityMetrics.averageQuality * 100).toFixed(1)}%
                            </p>
                          </div>
                          <div className="text-sm text-gray-500">
                            {doc.processingTimeMs}ms
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'search' && (
          <div className="space-y-8">
            <SearchInterface onSearchComplete={handleSearchComplete} />
            
            {recentSearches.length > 0 && (
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Searches</h3>
                <div className="space-y-4">
                  {recentSearches.map((search, index) => (
                    <div key={index} className="bg-white border border-gray-200 rounded-lg p-4">
                      <div className="flex items-start justify-between mb-2">
                        <p className="text-sm font-medium text-gray-900 truncate">{search.query}</p>
                        <span className="text-xs text-gray-500">
                          {(search.confidence * 100).toFixed(1)}% confidence
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 line-clamp-2">{search.response}</p>
                      <p className="text-xs text-gray-500 mt-2">
                        {search.sources.length} sources
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'analytics' && (
          <div className="space-y-8">
            <div className="bg-white rounded-lg p-6 shadow-sm">
              <h3 className="text-lg font-medium text-gray-900 mb-4">System Analytics</h3>
              <p className="text-gray-600">Analytics dashboard coming soon...</p>
              {/* This would include charts, performance metrics, usage patterns, etc. */}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
```

### Week 11: Production Deployment

#### Docker Compose for Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # Frontend (NextJS)
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - NEXT_PUBLIC_API_URL=http://backend:8000
    depends_on:
      - backend
    restart: unless-stopped
    networks:
      - app-network

  # Backend (FastAPI)
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/enterprise_ai
      - REDIS_URL=redis://redis:6379
      - MILVUS_HOST=milvus-standalone
      - MILVUS_PORT=19530
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - postgres
      - redis
      - milvus-standalone
    restart: unless-stopped
    volumes:
      - ./data/uploads:/app/uploads
      - ./data/logs:/app/logs
    networks:
      - app-network

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=enterprise_ai
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - app-network

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - app-network

  # Milvus Vector Database
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks:
      - app-network

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks:
      - app-network

  milvus-standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.3
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
      - ./configs/milvus.yaml:/milvus/configs/milvus.yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"
    networks:
      - app-network

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend
    restart: unless-stopped
    networks:
      - app-network

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    networks:
      - app-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - app-network

volumes:
  postgres_data:
  redis_data:
  milvus_data:
  etcd_data:
  minio_data:
  prometheus_data:
  grafana_data:

networks:
  app-network:
    driver: bridge
```

#### Production Nginx Configuration

```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream frontend {
        server frontend:3000;
    }

    upstream backend {
        server backend:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=2r/s;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    server {
        listen 80;
        server_name _;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name _;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Increase timeouts for long-running operations
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 300s;
        }

        # File upload endpoints
        location /api/documents/process {
            limit_req zone=upload burst=5 nodelay;
            
            client_max_body_size 50M;
            
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 600s;
        }

        # Health checks
        location /health {
            proxy_pass http://backend/health;
            access_log off;
        }
    }
}
```

### Week 12: Final Integration & Testing

#### Comprehensive Integration Tests

```python
# tests/integration/test_full_pipeline.py
import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any

from backend.services.unified_document_processor import UnifiedDocumentProcessor
from backend.services.advanced_rag import AdvancedRAGService
from backend.services.performance_optimizer import PerformanceOptimizer
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class TestFullPipeline:
    """Integration tests for the complete enterprise AI pipeline."""
    
    @pytest.fixture
    async def setup_services(self):
        """Setup all required services for testing."""
        processor = UnifiedDocumentProcessor()
        rag_service = AdvancedRAGService()
        optimizer = PerformanceOptimizer()
        
        # Initialize services
        await processor.initialize()
        
        return {
            "processor": processor,
            "rag_service": rag_service,
            "optimizer": optimizer
        }
    
    @pytest.mark.asyncio
    async def test_document_processing_pipeline(self, setup_services):
        """Test complete document processing pipeline."""
        services = await setup_services
        processor = services["processor"]
        
        # Test document processing
        test_content = """
        # Enterprise AI Solutions
        
        ## Introduction
        This document outlines our enterprise AI strategy and implementation approach.
        
        ## Key Components
        - Vector databases for semantic search
        - Large language models for reasoning
        - Multi-agent systems for complex workflows
        
        ## Benefits
        - Improved decision making
        - Automated document analysis
        - Enhanced knowledge discovery
        """
        
        result = await processor.process_document(
            content=test_content,
            document_id="test-doc-001",
            metadata={"source": "test", "type": "markdown"}
        )
        
        # Assertions
        assert result["success"] is True
        assert result["document_id"] == "test-doc-001"
        assert result["chunk_count"] > 0
        assert len(result["chunk_ids"]) == result["chunk_count"]
        assert result["quality_metrics"]["average_quality"] > 0.5
        
        logger.info(f"Document processing test passed: {result}")
        
        return result
    
    @pytest.mark.asyncio
    async def test_search_and_reasoning_pipeline(self, setup_services):
        """Test search and reasoning pipeline."""
        services = await setup_services
        rag_service = services["rag_service"]
        
        # First process a document
        doc_result = await self.test_document_processing_pipeline(setup_services)
        document_id = doc_result["document_id"]
        
        # Test search with standard reasoning
        search_result = await rag_service.process_complex_query(
            query="What are the key components of enterprise AI solutions?",
            document_ids=[document_id],
            reasoning_depth="standard"
        )
        
        # Assertions
        assert "error" not in search_result
        assert search_result["query"] == "What are the key components of enterprise AI solutions?"
        assert len(search_result["sub_queries"]) >= 1
        assert search_result["confidence_score"] > 0.3
        assert "vector databases" in search_result["response"].lower() or "language models" in search_result["response"].lower()
        
        logger.info(f"Search and reasoning test passed: {search_result}")
        
        return search_result
    
    @pytest.mark.asyncio
    async def test_deep_reasoning_pipeline(self, setup_services):
        """Test deep reasoning capabilities."""
        services = await setup_services
        rag_service = services["rag_service"]
        
        # First process a document
        doc_result = await self.test_document_processing_pipeline(setup_services)
        document_id = doc_result["document_id"]
        
        # Test search with deep reasoning
        search_result = await rag_service.process_complex_query(
            query="How do the benefits of enterprise AI relate to the key components mentioned?",
            document_ids=[document_id],
            reasoning_depth="deep"
        )
        
        # Assertions
        assert "error" not in search_result
        assert search_result["reasoning_depth"] == "deep"
        assert search_result["confidence_score"] > 0.2
        assert len(search_result["response"]) > 100  # Deep reasoning should be more detailed
        
        logger.info(f"Deep reasoning test passed: {search_result}")
        
        return search_result
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self, setup_services):
        """Test performance optimization features."""
        services = await setup_services
        optimizer = services["optimizer"]
        
        # Monitor performance
        metrics = await optimizer.monitor_system_performance()
        
        # Assertions
        assert metrics.cpu_usage >= 0
        assert metrics.memory_usage >= 0
        assert metrics.processing_time >= 0
        
        # Test optimization
        optimization_result = await optimizer.optimize_system_performance()
        
        assert "current_metrics" in optimization_result
        assert "optimizations_applied" in optimization_result
        assert "recommendations" in optimization_result
        
        logger.info(f"Performance optimization test passed: {optimization_result}")
        
        return optimization_result
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, setup_services):
        """Test concurrent document processing."""
        services = await setup_services
        processor = services["processor"]
        
        # Create multiple test documents
        test_documents = [
            ("doc-1", "# Document 1\nThis is the first test document with AI content."),
            ("doc-2", "# Document 2\nThis is the second test document with ML content."),
            ("doc-3", "# Document 3\nThis is the third test document with NLP content."),
        ]
        
        # Process documents concurrently
        tasks = [
            processor.process_document(
                content=content,
                document_id=doc_id,
                metadata={"source": "concurrent_test"}
            )
            for doc_id, content in test_documents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assertions
        assert len(results) == 3
        for result in results:
            assert not isinstance(result, Exception)
            assert result["success"] is True
            assert result["chunk_count"] > 0
        
        logger.info(f"Concurrent processing test passed: {len(results)} documents processed")
        
        return results
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, setup_services):
        """Test error handling and recovery mechanisms."""
        services = await setup_services
        processor = services["processor"]
        rag_service = services["rag_service"]
        
        # Test invalid document processing
        invalid_result = await processor.process_document(
            content="",  # Empty content
            document_id="invalid-doc",
            metadata={}
        )
        
        # Should handle gracefully
        assert "error" in invalid_result or invalid_result["success"] is False
        
        # Test search with non-existent document
        search_result = await rag_service.process_complex_query(
            query="Test query",
            document_ids=["non-existent-doc"],
            reasoning_depth="standard"
        )
        
        # Should handle gracefully
        assert "error" in search_result or "don't have enough information" in search_result.get("response", "")
        
        logger.info("Error handling test passed")
        
        return True
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, setup_services):
        """Test complete end-to-end workflow."""
        services = await setup_services
        
        # Step 1: Process multiple documents
        doc_results = await self.test_concurrent_processing(setup_services)
        document_ids = [result["document_id"] for result in doc_results]
        
        # Step 2: Perform searches across all documents
        search_queries = [
            "What AI technologies are mentioned?",
            "Compare the different documents",
            "What are the main topics discussed?"
        ]
        
        search_results = []
        for query in search_queries:
            result = await services["rag_service"].process_complex_query(
                query=query,
                document_ids=document_ids,
                reasoning_depth="standard"
            )
            search_results.append(result)
        
        # Step 3: Monitor performance
        perf_result = await self.test_performance_optimization(setup_services)
        
        # Assertions
        assert len(search_results) == 3
        for result in search_results:
            assert "error" not in result
            assert result["confidence_score"] > 0.1
        
        assert perf_result["current_metrics"].cpu_usage >= 0
        
        logger.info("End-to-end workflow test passed")
        
        return {
            "documents_processed": len(doc_results),
            "searches_completed": len(search_results),
            "performance_optimized": True,
            "total_chunks": sum(result["chunk_count"] for result in doc_results),
            "average_confidence": sum(result["confidence_score"] for result in search_results) / len(search_results)
        }

if __name__ == "__main__":
    # Run integration tests
    pytest.main(["-v", __file__])
```

---

## Deployment Checklist

### Pre-Deployment
- [ ] All integration tests passing
- [ ] Performance benchmarks meet requirements
- [ ] Security audit completed
- [ ] SSL certificates configured
- [ ] Environment variables set
- [ ] Database migrations ready
- [ ] Monitoring dashboards configured

### Production Deployment
- [ ] Deploy infrastructure (Docker Compose)
- [ ] Initialize databases (PostgreSQL, Milvus)
- [ ] Configure reverse proxy (Nginx)
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure logging aggregation
- [ ] Test health endpoints
- [ ] Verify SSL/TLS configuration
- [ ] Load test the system

### Post-Deployment
- [ ] Monitor system metrics
- [ ] Verify all services are healthy
- [ ] Test document processing pipeline
- [ ] Test search and reasoning capabilities
- [ ] Monitor performance and optimize
- [ ] Set up alerting rules
- [ ] Document operational procedures
- [ ] Train operations team

---

## Success Metrics

### Technical Performance
- **Document Processing**: < 5 seconds per document
- **Search Response Time**: < 2 seconds for standard queries
- **System Uptime**: > 99.9%
- **Concurrent Users**: Support 100+ simultaneous users

### Quality Metrics
- **Search Accuracy**: > 85% relevant results
- **Response Quality**: > 80% user satisfaction
- **System Reliability**: < 0.1% error rate

### Business Value
- **Knowledge Discovery**: 50% faster information retrieval
- **Decision Support**: 40% improvement in decision quality
- **Operational Efficiency**: 60% reduction in manual document analysis

This completes the comprehensive 12-week technical implementation roadmap for the enterprise-grade AI solution integrating PocketFlow, NextJS-FastAPI, Milvus, and the existing chunking system.
```