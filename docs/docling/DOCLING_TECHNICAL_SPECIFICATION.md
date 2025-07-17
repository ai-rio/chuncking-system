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
│ │   ├── DoclingProcessor (NEW)                              │
│ │   └── MarkdownProcessor (Existing)                        │
│ ├── Enhanced HybridChunker                                    │
│ │   ├── Docling HybridChunker Integration                   │
│ │   ├── Docling HierarchicalChunker                        │
│ │   └── Legacy Markdown Chunker                             │
│ ├── Enhanced LLM Providers                                    │
│ │   ├── DoclingProvider (NEW - Vision Models)               │
│ │   ├── OpenAI Provider (Enhanced)                          │
│ │   ├── Anthropic Provider (Enhanced)                       │
│ │   └── Jina Provider (Enhanced)                            │
│ ├── Enhanced Quality Evaluator                                │
│ │   ├── Document Structure Metrics                          │
│ │   ├── Visual Content Evaluation                           │
│ │   └── Multi-Modal Quality Assessment                      │
│ └── Enhanced Security/Monitoring                              │
│     ├── Multi-Format Security Validation                     │
│     ├── Vision Model Monitoring                              │
│     └── Performance Tracking for Large Documents             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### **1. DoclingProcessor** 
*New Component - Core Integration*

#### **Location**: `src/chunkers/docling_processor.py`

#### **Interface**
```python
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from src.chunkers.base_processor import BaseProcessor

class DoclingProcessor(BaseProcessor):
    """Multi-format document processor using Docling."""
    
    def __init__(self, 
                 allowed_formats: Optional[List[InputFormat]] = None,
                 pipeline_options: Optional[Dict[str, Any]] = None,
                 enable_vision: bool = True,
                 enable_ocr: bool = True):
        """Initialize Docling processor with configuration."""
        
    def process_document(self, 
                        file_path: Path, 
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process document and return structured content."""
        
    def detect_format(self, file_path: Path) -> InputFormat:
        """Detect document format from file."""
        
    def convert_to_chunks(self, 
                         docling_document: Any, 
                         chunk_config: Dict[str, Any]) -> List[Any]:
        """Convert Docling document to chunks."""
        
    def extract_metadata(self, docling_document: Any) -> Dict[str, Any]:
        """Extract comprehensive metadata from document."""
```

#### **Key Features**
- **Multi-Format Support**: PDF, DOCX, PPTX, HTML, Images
- **Vision Processing**: Image description, table extraction, formula detection
- **Structure Preservation**: Headers, lists, tables, code blocks
- **Metadata Enrichment**: Document structure, content types, provenance
- **Error Handling**: Graceful degradation for unsupported features

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
    'pipeline_options': {
        'do_ocr': True,
        'do_table_structure': True,
        'generate_picture_images': True,
        'picture_description_options': {
            'model': 'granite-vision',
            'prompt': 'Describe this image in detail for document processing.'
        }
    },
    'performance': {
        'batch_size': 5,
        'max_memory_mb': 2048,
        'timeout_seconds': 300
    }
}
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
        """Detect document format using MIME type and extension."""
        
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
    'application/pdf': ('pdf', DoclingProcessor),
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ('docx', DoclingProcessor),
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': ('pptx', DoclingProcessor),
    'text/html': ('html', DoclingProcessor),
    'image/png': ('image', DoclingProcessor),
    'image/jpeg': ('image', DoclingProcessor),
    'text/markdown': ('markdown', MarkdownProcessor),
    'text/plain': ('markdown', MarkdownProcessor)
}
```

---

### **3. DoclingProvider**
*New Component - LLM Integration*

#### **Location**: `src/llm/providers/docling_provider.py`

#### **Interface**
```python
from src.llm.providers.base import BaseLLMProvider
from docling.pipeline.vlm_pipeline import VlmPipeline

class DoclingProvider(BaseLLMProvider):
    """LLM provider for Docling vision and enrichment models."""
    
    @property
    def provider_name(self) -> str:
        return "docling"
    
    def __init__(self, 
                 vision_model: str = "granite-vision",
                 enable_picture_description: bool = True,
                 enable_code_understanding: bool = True,
                 enable_formula_detection: bool = True):
        """Initialize Docling provider with vision capabilities."""
        
    def count_tokens(self, text: str, include_images: int = 0) -> int:
        """Count tokens including image processing overhead."""
        
    def process_with_vision(self, 
                           document_path: Path,
                           options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process document with vision models."""
        
    def describe_images(self, image_items: List[Any]) -> List[str]:
        """Generate descriptions for document images."""
        
    def detect_formulas(self, document_content: str) -> List[Dict[str, Any]]:
        """Detect and process mathematical formulas."""
        
    def understand_code(self, code_blocks: List[str]) -> List[Dict[str, Any]]:
        """Analyze and understand code blocks."""
```

#### **Vision Model Configuration**
```python
VISION_MODELS = {
    'granite-vision': {
        'model_id': 'ibm-granite/granite-3.1-8b-instruct',
        'description_prompt': 'Describe this image in detail for document processing.',
        'max_tokens': 150,
        'temperature': 0.1
    },
    'smol-vlm': {
        'model_id': 'HuggingFaceTB/SmolVLM-Instruct',
        'description_prompt': 'Provide a concise description of this image.',
        'max_tokens': 100,
        'temperature': 0.0
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
                 docling_chunker_type: str = "hybrid"):
        """Enhanced initialization with Docling support."""
        
    def chunk_docling_document(self, 
                              docling_doc: Any,
                              metadata: Dict[str, Any]) -> List[Any]:
        """Chunk document using Docling's chunking capabilities."""
        
    def apply_docling_hybrid_chunking(self, 
                                     docling_doc: Any) -> List[Any]:
        """Apply Docling's HybridChunker with tokenization-aware refinements."""
        
    def apply_docling_hierarchical_chunking(self, 
                                           docling_doc: Any) -> List[Any]:
        """Apply Docling's HierarchicalChunker for structure-aware chunking."""
        
    def preserve_document_structure(self, 
                                   chunks: List[Any],
                                   docling_doc: Any) -> List[Any]:
        """Enhance chunks with document structure information."""
```

#### **Chunking Strategy Selection**
```python
CHUNKING_STRATEGIES = {
    'pdf': {
        'primary': 'docling_hybrid',
        'fallback': 'docling_hierarchical',
        'config': {
            'preserve_tables': True,
            'preserve_images': True,
            'merge_list_items': True
        }
    },
    'docx': {
        'primary': 'docling_hierarchical',
        'fallback': 'docling_hybrid',
        'config': {
            'preserve_headings': True,
            'preserve_styles': True
        }
    },
    'markdown': {
        'primary': 'legacy_hybrid',
        'fallback': 'recursive',
        'config': {
            'header_based': True
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
        'small_doc': {'pages': '<10', 'target_time_ms': 5000, 'memory_mb': 200},
        'medium_doc': {'pages': '10-100', 'target_time_ms': 30000, 'memory_mb': 500},
        'large_doc': {'pages': '>100', 'target_time_ms': 120000, 'memory_mb': 1000}
    },
    'docx': {
        'small_doc': {'pages': '<10', 'target_time_ms': 3000, 'memory_mb': 150},
        'medium_doc': {'pages': '10-50', 'target_time_ms': 15000, 'memory_mb': 300},
        'large_doc': {'pages': '>50', 'target_time_ms': 60000, 'memory_mb': 600}
    },
    'pptx': {
        'small_doc': {'slides': '<20', 'target_time_ms': 4000, 'memory_mb': 180},
        'medium_doc': {'slides': '20-100', 'target_time_ms': 20000, 'memory_mb': 400},
        'large_doc': {'slides': '>100', 'target_time_ms': 80000, 'memory_mb': 800}
    }
}
```

### **Memory Management**
- **Streaming Processing**: For documents >50MB
- **Batch Processing**: Process images in batches of 5
- **Memory Cleanup**: Automatic cleanup after each document
- **Cache Management**: LRU cache for frequently processed documents

### **Concurrency**
- **Document Processing**: Max 3 concurrent documents
- **Vision Model Calls**: Max 2 concurrent API calls
- **Chunk Generation**: Parallel processing for independent chunks

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
    'vision_model_requests_total': 'Counter of vision model API requests',
    'vision_model_response_time_seconds': 'Histogram of vision model response times',
    'chunk_quality_score': 'Histogram of chunk quality scores by format',
    'memory_usage_bytes': 'Gauge of memory usage during processing',
    'concurrent_documents': 'Gauge of currently processing documents'
}
```

### **Health Checks**
- **Docling Service Health**: Verify DocumentConverter functionality
- **Vision Model Health**: Test vision model API connectivity
- **Memory Health**: Monitor memory usage and cleanup
- **Processing Queue Health**: Monitor document processing queue

### **Alerting Rules**
- **High Failure Rate**: >5% processing failures in 5 minutes
- **Slow Processing**: >2x normal processing time for document type
- **Memory Issues**: >80% memory usage sustained for 2 minutes
- **Vision Model Errors**: >3 consecutive vision model API failures

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