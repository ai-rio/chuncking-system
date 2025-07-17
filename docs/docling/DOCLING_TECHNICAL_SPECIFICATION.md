# Docling Integration Technical Specification

## Executive Summary

This document specifies the technical implementation of Docling integration into our existing chunking system, enabling multi-format document processing while maintaining enterprise-grade performance, security, and monitoring capabilities.

---

## Architecture Overview

### **Current System Architecture**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Current Chunking System                      │
├─────────────────────────────────────────────────────────────────┤
│ Input: Markdown Files Only                                     │
│ ├── FileHandler (UTF-8 text files)                            │
│ ├── HybridMarkdownChunker (Markdown-specific)                 │
│ ├── LLM Providers (OpenAI, Anthropic, Jina)                  │
│ ├── Quality Evaluator (Basic metrics)                         │
│ └── Security/Monitoring (File validation)                     │
└─────────────────────────────────────────────────────────────────┘
```

### **Target Architecture with Docling**
```
┌─────────────────────────────────────────────────────────────────┐
│                Enhanced Multi-Format System                     │
├─────────────────────────────────────────────────────────────────┤
│ Input: PDF, DOCX, PPTX, HTML, Images, Markdown                │
│ ├── Enhanced FileHandler                                      │
│ │   ├── FormatDetector                                       │
│ │   ├── DoclingProcessor (NEW - Local Library)              │
│ │   └── MarkdownProcessor (Existing)                        │
│ ├── Enhanced HybridChunker                                    │
│ │   ├── Docling HybridChunker Integration                   │
│ │   ├── Document Structure Preservation                     │
│ │   └── Legacy Markdown Chunker                             │
│ ├── Enhanced LLM Providers                                    │
│ │   ├── DoclingProvider (External API - Optional)           │
│ │   ├── OpenAI Provider (Enhanced)                          │
│ │   ├── Anthropic Provider (Enhanced)                       │
│ │   └── Jina Provider (Enhanced)                            │
│ ├── Enhanced Quality Evaluator                                │
│ │   ├── Multi-Format Quality Metrics                        │
│ │   ├── Document Structure Evaluation                       │
│ │   └── Format-Specific Assessment                          │
│ └── Enhanced Security/Monitoring                              │
│     ├── Multi-Format Security Validation                     │
│     ├── Local Processing Monitoring                          │
│     └── Performance Tracking for Large Documents             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### **Important Architecture Note**

The Docling integration consists of two distinct components:

1. **DoclingProcessor** (Primary) - Uses the local Docling library for document processing
2. **DoclingProvider** (Optional) - Provides external API integration for additional LLM capabilities

The core document processing functionality is handled by DoclingProcessor using the local Docling library. DoclingProvider is an optional external API integration that extends the LLM provider ecosystem.

### **1. DoclingProcessor** 
*New Component - Core Integration*

#### **Location**: `src/chunkers/docling_processor.py`

#### **Interface**
```python
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker

class DoclingProcessor:
    """Multi-format document processor using Docling library."""
    
    def __init__(self, 
                 chunker_tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize Docling processor with chunker configuration."""
        
    def process_document(self, 
                        file_path: str, 
                        format_type: str = "auto",
                        **kwargs) -> List[Document]:
        """Process document and return chunked Document objects."""
        
    def _detect_format(self, file_path: str) -> str:
        """Auto-detect document format from file extension and MIME type."""
        
    def export_to_markdown(self, file_path: str) -> str:
        """Export document to Markdown format."""
        
    def export_to_html(self, file_path: str) -> str:
        """Export document to HTML format."""
        
    def export_to_json(self, file_path: str) -> str:
        """Export document to JSON format."""
        
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        
    def is_format_supported(self, format_type: str) -> bool:
        """Check if a format is supported."""
        
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about the processor."""
```

#### **Key Features**
- **Multi-Format Support**: PDF, DOCX, PPTX, HTML, Images
- **Local Processing**: Uses Docling library directly without API calls
- **Hybrid Chunking**: Integrates with Docling's HybridChunker for optimal results
- **Structure Preservation**: Maintains document hierarchy and formatting
- **Metadata Enrichment**: Extracts comprehensive document metadata
- **Error Handling**: Graceful degradation with mock processing fallback
- **Library Fallback**: When Docling library is unavailable, provides mock processing
- **Format Detection**: Automatic detection using file extensions and MIME types
- **Export Options**: Supports export to markdown, HTML, and JSON formats

#### **Configuration**
```python
DOCLING_CONFIG = {
    'allowed_formats': [
        InputFormat.PDF,
        InputFormat.DOCX, 
        InputFormat.PPTX,
        InputFormat.HTML,
        InputFormat.IMAGE
    ],
    'chunker_config': {
        'tokenizer': 'sentence-transformers/all-MiniLM-L6-v2',
        'chunk_size': 1000,
        'chunk_overlap': 200
    },
    'processing_options': {
        'format_detection': 'auto',
        'preserve_structure': True,
        'export_formats': ['markdown', 'html', 'json']
    },
    'performance': {
        'timeout_seconds': 300,
        'max_file_size_mb': 100
    },
    'fallback': {
        'enable_mock_processing': True,
        'mock_content_template': 'Mock {format} content from {filename}',
        'fallback_on_library_missing': True
    }
}
```

#### **Library Availability Detection**
```python
# DoclingProcessor automatically detects library availability
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.chunking import HybridChunker
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    # Mock classes are used when library is not available
```

---

### **2. Enhanced FileHandler**
*Modified Component*

#### **Location**: `src/utils/file_handler.py`

#### **New Methods**
```python
class FileHandler:
    # Existing methods remain unchanged...
    
    def detect_document_format(self, file_path: Path) -> str:
        """Detect document format using file extension and MIME type."""
        
    def validate_multi_format_file(self, file_path: Path) -> bool:
        """Validate file for multi-format processing."""
        
    def get_format_processor(self, file_path: Path) -> BaseProcessor:
        """Get appropriate processor for file format."""
        
    def extract_document_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract format-specific metadata."""
```

#### **Format Detection Logic**
```python
FORMAT_MAPPING = {
    # Docling-supported formats
    'application/pdf': ('pdf', DoclingProcessor),
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ('docx', DoclingProcessor),
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': ('pptx', DoclingProcessor),
    'text/html': ('html', DoclingProcessor),
    'image/png': ('image', DoclingProcessor),
    'image/jpeg': ('image', DoclingProcessor),
    'image/gif': ('image', DoclingProcessor),
    'image/bmp': ('image', DoclingProcessor),
    'image/tiff': ('image', DoclingProcessor),
    
    # Legacy formats
    'text/markdown': ('markdown', MarkdownProcessor),
    'text/plain': ('markdown', MarkdownProcessor)
}
```

---

### **3. DoclingProvider**
*Modified Component - LLM Integration*

#### **Location**: `src/llm/providers/docling_provider.py`

#### **Interface**
```python
from src.llm.providers.base import BaseLLMProvider
from typing import List, Optional, Dict, Any
import requests
import json

class DoclingProvider(BaseLLMProvider):
    """LLM provider for external Docling API services."""
    
    @property
    def provider_name(self) -> str:
        return "docling"
    
    def __init__(self, 
                 api_key: str,
                 model: str = "docling-v1",
                 base_url: str = "https://api.docling.ai/v1",
                 embedding_model: str = "docling-embeddings-v1"):
        """Initialize Docling provider for external API access."""
        
    def generate_completion(self, 
                           prompt: str,
                           max_tokens: Optional[int] = None,
                           temperature: float = 0.7,
                           **kwargs) -> LLMResponse:
        """Generate text completion using Docling API."""
        
    def generate_embeddings(self, 
                           texts: List[str],
                           **kwargs) -> EmbeddingResponse:
        """Generate embeddings using Docling API."""
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using heuristic approximation."""
        
    def process_document(self, 
                        document_content: str,
                        document_type: str = "auto",
                        **kwargs) -> Dict[str, Any]:
        """Process document content using Docling API capabilities."""
```

#### **API Configuration**
```python
DOCLING_API_CONFIG = {
    'base_url': 'https://api.docling.ai/v1',
    'models': {
        'text': 'docling-v1',
        'embeddings': 'docling-embeddings-v1',
        'document_processing': 'docling-document-processor'
    },
    'timeouts': {
        'completion': 30,
        'embedding': 30,
        'document_processing': 60
    },
    'token_limits': {
        'docling-v1': 8192,
        'docling-large': 16384,
        'docling-small': 4096
    }
}
```

---

### **4. Enhanced HybridChunker**
*Modified Component*

#### **Location**: `src/chunkers/hybrid_chunker.py`

#### **New Integration Methods**
```python
class HybridMarkdownChunker:
    # Existing methods remain unchanged...
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 enable_docling: bool = True,
                 docling_chunker_tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Enhanced initialization with Docling support."""
        
    def chunk_docling_document(self, 
                              docling_doc: Any,
                              metadata: Dict[str, Any]) -> List[Document]:
        """Chunk document using Docling's HybridChunker."""
        
    def integrate_docling_processor(self, 
                                   docling_processor: DoclingProcessor) -> None:
        """Integrate DoclingProcessor for multi-format support."""
        
    def process_with_docling(self, 
                            file_path: str,
                            format_type: str = "auto") -> List[Document]:
        """Process document using integrated DoclingProcessor."""
        
    def enhance_chunks_with_structure(self, 
                                     chunks: List[Document],
                                     original_metadata: Dict[str, Any]) -> List[Document]:
        """Enhance chunks with document structure information."""
```

#### **Processing Strategy Selection**
```python
PROCESSING_STRATEGIES = {
    'pdf': {
        'processor': 'docling',
        'chunker': 'hybrid',
        'config': {
            'preserve_tables': True,
            'preserve_images': True,
            'extract_metadata': True
        }
    },
    'docx': {
        'processor': 'docling',
        'chunker': 'hybrid',
        'config': {
            'preserve_headings': True,
            'preserve_styles': True,
            'extract_metadata': True
        }
    },
    'pptx': {
        'processor': 'docling',
        'chunker': 'hybrid',
        'config': {
            'preserve_slides': True,
            'extract_images': True,
            'extract_metadata': True
        }
    },
    'markdown': {
        'processor': 'legacy',
        'chunker': 'hybrid_markdown',
        'config': {
            'header_based': True,
            'preserve_code_blocks': True
        }
    }
}
```

---

### **5. Enhanced ChunkQualityEvaluator**
*Modified Component*

#### **Location**: `src/chunkers/evaluators.py`

#### **New Quality Metrics**
```python
class ChunkQualityEvaluator:
    # Existing methods remain unchanged...
    
    def evaluate_multi_format_chunks(self, 
                                    chunks: List[Any],
                                    document_format: str,
                                    original_document: Any = None) -> Dict[str, Any]:
        """Evaluate chunks with format-specific quality metrics."""
        
    def evaluate_document_structure_preservation(self, 
                                               chunks: List[Any],
                                               original_structure: Dict[str, Any]) -> float:
        """Evaluate how well document structure is preserved."""
        
    def evaluate_visual_content_quality(self, 
                                       chunks: List[Any]) -> Dict[str, Any]:
        """Evaluate quality of visual content processing."""
        
    def calculate_boundary_preservation_score(self, 
                                            chunks: List[Any]) -> float:
        """Calculate how well semantic boundaries are preserved."""
        
    def assess_context_continuity(self, chunks: List[Any]) -> float:
        """Assess context continuity between adjacent chunks."""
```

#### **Quality Metrics Framework**
```python
QUALITY_METRICS = {
    'structure_preservation': {
        'weight': 0.25,
        'thresholds': {'excellent': 0.9, 'good': 0.75, 'poor': 0.5}
    },
    'semantic_coherence': {
        'weight': 0.30,
        'thresholds': {'excellent': 0.85, 'good': 0.70, 'poor': 0.5}
    },
    'boundary_preservation': {
        'weight': 0.20,
        'thresholds': {'excellent': 0.90, 'good': 0.75, 'poor': 0.6}
    },
    'context_continuity': {
        'weight': 0.15,
        'thresholds': {'excellent': 0.85, 'good': 0.70, 'poor': 0.5}
    },
    'visual_content_quality': {
        'weight': 0.10,
        'thresholds': {'excellent': 0.80, 'good': 0.65, 'poor': 0.4}
    }
}
```

---

## Data Models and Schemas

### **Enhanced Chunk Schema**
```python
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class EnhancedChunkMetadata(BaseModel):
    """Enhanced metadata for multi-format chunks."""
    
    # Existing fields
    chunk_index: int
    chunk_tokens: int
    chunk_chars: int
    word_count: int
    chunk_id: str
    processed_at: datetime
    
    # New multi-format fields
    document_format: str
    source_page: Optional[int] = None
    source_slide: Optional[int] = None
    structural_elements: List[str] = []
    visual_elements: List[Dict[str, Any]] = []
    content_types: List[str] = []
    
    # Docling-specific fields
    docling_item_id: Optional[str] = None
    docling_hierarchy: Optional[Dict[str, Any]] = None
    provenance_items: List[Dict[str, Any]] = []
    
    # Quality metrics
    structure_preservation_score: Optional[float] = None
    boundary_quality_score: Optional[float] = None
    visual_content_score: Optional[float] = None

class EnhancedChunk(BaseModel):
    """Enhanced chunk with multi-format support."""
    
    content: str
    metadata: EnhancedChunkMetadata
    
    # Visual elements
    images: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []
    formulas: List[Dict[str, Any]] = []
    code_blocks: List[Dict[str, Any]] = []
    
    # Structure information
    parent_headers: List[str] = []
    structural_context: Optional[Dict[str, Any]] = None
```

### **Document Processing Result Schema**
```python
class DocumentProcessingResult(BaseModel):
    """Result of multi-format document processing."""
    
    success: bool
    document_format: str
    chunks: List[EnhancedChunk]
    
    # Processing metadata
    processing_time_ms: float
    memory_usage_mb: float
    
    # Document structure
    document_structure: Dict[str, Any]
    visual_elements_count: int
    total_pages: Optional[int] = None
    
    # Quality assessment
    overall_quality_score: float
    quality_breakdown: Dict[str, float]
    
    # Error information
    errors: List[str] = []
    warnings: List[str] = []
    
    # Docling-specific
    docling_document_id: Optional[str] = None
    vision_processing_used: bool = False
    enrichment_models_used: List[str] = []
```

---

## Performance Specifications

### **Processing Benchmarks**
```python
PERFORMANCE_TARGETS = {
    'pdf': {
        'small_doc': {'size': '<5MB', 'target_time_ms': 3000, 'memory_mb': 150},
        'medium_doc': {'size': '5-20MB', 'target_time_ms': 15000, 'memory_mb': 300},
        'large_doc': {'size': '>20MB', 'target_time_ms': 60000, 'memory_mb': 600}
    },
    'docx': {
        'small_doc': {'size': '<2MB', 'target_time_ms': 2000, 'memory_mb': 100},
        'medium_doc': {'size': '2-10MB', 'target_time_ms': 10000, 'memory_mb': 200},
        'large_doc': {'size': '>10MB', 'target_time_ms': 30000, 'memory_mb': 400}
    },
    'pptx': {
        'small_doc': {'size': '<5MB', 'target_time_ms': 3000, 'memory_mb': 150},
        'medium_doc': {'size': '5-25MB', 'target_time_ms': 15000, 'memory_mb': 300},
        'large_doc': {'size': '>25MB', 'target_time_ms': 45000, 'memory_mb': 500}
    },
    'html': {
        'small_doc': {'size': '<1MB', 'target_time_ms': 1000, 'memory_mb': 50},
        'medium_doc': {'size': '1-5MB', 'target_time_ms': 5000, 'memory_mb': 100},
        'large_doc': {'size': '>5MB', 'target_time_ms': 15000, 'memory_mb': 200}
    },
    'image': {
        'small_doc': {'size': '<2MB', 'target_time_ms': 2000, 'memory_mb': 100},
        'medium_doc': {'size': '2-10MB', 'target_time_ms': 8000, 'memory_mb': 200},
        'large_doc': {'size': '>10MB', 'target_time_ms': 20000, 'memory_mb': 400}
    }
}
```

### **Memory Management**
- **Local Processing**: All processing happens locally without API calls
- **File Size Limits**: Configurable limits per format type
- **Memory Cleanup**: Automatic cleanup after each document
- **Fallback Processing**: Mock processing when Docling unavailable

### **Concurrency**
- **Document Processing**: Sequential processing for memory efficiency
- **Chunk Generation**: Parallel processing for independent chunks
- **Format Detection**: Concurrent format detection for batch processing

---

## Security Specifications

### **File Validation**
```python
SECURITY_LIMITS = {
    'max_file_size_mb': {
        'pdf': 100,
        'docx': 50,
        'pptx': 100,
        'html': 10,
        'image': 20
    },
    'allowed_mime_types': [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'text/html',
        'image/png',
        'image/jpeg',
        'image/tiff'
    ],
    'scan_for_malware': True,
    'validate_structure': True
}
```

### **Content Security**
- **Malicious Content Detection**: Scan for embedded scripts, macros
- **Privacy Protection**: Redact sensitive information if detected
- **Access Control**: Validate file permissions and ownership
- **Audit Logging**: Log all processing activities

---

## Monitoring and Observability

### **New Metrics**
```python
DOCLING_METRICS = {
    'document_processing_duration_seconds': 'Histogram of document processing times by format',
    'document_processing_total': 'Counter of total documents processed by format',
    'document_processing_failures_total': 'Counter of processing failures by format and error type',
    'docling_library_available': 'Gauge indicating if Docling library is available',
    'mock_processing_total': 'Counter of documents processed with mock fallback',
    'chunk_quality_score': 'Histogram of chunk quality scores by format',
    'memory_usage_bytes': 'Gauge of memory usage during processing',
    'file_size_bytes': 'Histogram of processed file sizes by format',
    'format_detection_duration_seconds': 'Histogram of format detection times'
}
```

### **Health Checks**
- **Docling Library Health**: Verify DocumentConverter functionality
- **Format Support Health**: Test supported format processing
- **Memory Health**: Monitor memory usage and cleanup
- **Processing Performance**: Monitor document processing times

### **Alerting Rules**
- **High Failure Rate**: >5% processing failures in 5 minutes
- **Slow Processing**: >2x normal processing time for document type
- **Memory Issues**: >80% memory usage sustained for 2 minutes
- **Library Errors**: >3 consecutive Docling library failures

---

## Testing Strategy

### **Unit Tests**
- DoclingProcessor component testing
- Format detection and validation
- Vision model integration mocking
- Error handling and edge cases

### **Integration Tests**
- End-to-end document processing
- Multi-format batch processing
- Quality evaluation validation
- Performance benchmarking

### **Load Tests**
- Concurrent document processing
- Large document handling
- Memory usage under load
- Vision model API rate limiting

### **Security Tests**
- Malicious file handling
- Content validation
- Access control verification
- Privacy protection validation

---

*This technical specification provides the detailed implementation guidance needed for successful Docling integration while maintaining the existing system's enterprise-grade capabilities.*