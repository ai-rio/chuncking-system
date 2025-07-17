# Data Models and Schema Changes

## **New Data Models**

### **DoclingProcessingResult**

**Purpose**: Encapsulate Docling-specific processing outputs including document structure, vision content, and metadata  
**Integration**: Extends existing ChunkingResult pattern while maintaining backward compatibility with current chunk metadata structure

**Key Attributes:**
- `document_type`: str - Detected format (pdf, docx, pptx, html, image)
- `vision_content`: Optional[List[Dict]] - Extracted visual elements (tables, figures, images) with descriptions
- `structure_data`: Optional[Dict] - Document hierarchy and formatting information from Docling analysis
- `processing_metadata`: Dict - Docling API response metadata including confidence scores and processing time
- `fallback_used`: bool - Indicates if Docling processing failed and fallback was employed

**Relationships:**
- **With Existing**: Integrates as optional extension to current chunk metadata in ChunkingResult dataclass
- **With New**: Used by DoclingProcessor and referenced in enhanced quality evaluation metrics

### **DoclingProviderConfig**

**Purpose**: Configuration management for Docling API integration following existing Pydantic settings patterns  
**Integration**: Extends current provider configuration system used by OpenAI, Anthropic, and Jina providers

**Key Attributes:**
- `api_key`: SecretStr - Docling API authentication credential
- `api_base_url`: HttpUrl - Docling service endpoint with default fallback
- `enable_vision_processing`: bool - Toggle for image and visual content analysis
- `max_file_size_mb`: int - File size limit for security validation (default: 50MB)
- `timeout_seconds`: int - API request timeout (default: 120s)

**Relationships:**
- **With Existing**: Inherits from existing provider configuration base class and integrates with current settings management
- **With New**: Used by DoclingProvider for initialization and operation configuration

### **MultiFormatChunk**

**Purpose**: Enhanced chunk representation supporting multi-format document metadata while maintaining existing chunk interface  
**Integration**: Backward-compatible extension of current chunk dictionary structure

**Key Attributes:**
- `source_format`: str - Original document format for processing pipeline routing
- `docling_metadata`: Optional[DoclingProcessingResult] - Docling-specific processing information
- `visual_elements`: Optional[List[Dict]] - Associated images, tables, figures with position information
- `structure_context`: Optional[Dict] - Document hierarchy context (headings, sections, page numbers)
- `quality_indicators`: Dict - Multi-format quality metrics and confidence scores

**Relationships:**
- **With Existing**: Maintains complete compatibility with existing chunk processing, storage, and export functionality
- **With New**: Enhanced by DoclingProcessor and evaluated by extended quality assessment system

## **Schema Integration Strategy**

**Database Changes Required:**
- **New Tables**: None - using additive approach to existing chunk metadata structure
- **Modified Tables**: None - extending ChunkingResult dataclass with optional fields maintains backward compatibility
- **New Indexes**: None required - existing chunk processing and storage patterns sufficient
- **Migration Strategy**: Zero-downtime deployment - new fields are optional and ignored by existing code

**Backward Compatibility:**
- All existing chunk processing code continues functioning without modification due to optional field approach
- Current chunk export formats (JSON, CSV, Pickle) automatically include new fields when present, exclude when absent
- Existing quality evaluation metrics operate on traditional fields, enhanced metrics operate on extended data when available
- Current API contracts preserved - new metadata accessible through existing interfaces as optional extensions
