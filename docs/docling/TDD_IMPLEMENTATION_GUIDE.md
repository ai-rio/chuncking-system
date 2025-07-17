# Test-Driven Development (TDD) Guide - Docling Integration

## TDD Philosophy for Docling Integration

**Core Principle**: Write tests first, then implement the minimum code to make tests pass, then refactor for quality. This ensures every line of code is justified by a test and requirement.

### **TDD Cycle: Red-Green-Refactor**

```
🔴 RED    → Write a failing test that defines desired behavior
🟢 GREEN  → Write minimal code to make the test pass
🔵 REFACTOR → Improve code quality while keeping tests green
```

### **TDD Benefits for Multi-Format Integration**
- **Confidence**: Every feature is tested before implementation
- **Design Quality**: Tests force good API design
- **Regression Protection**: Prevents breaking existing functionality
- **Documentation**: Tests serve as living specification
- **Faster Debugging**: Immediate feedback on changes

---

## TDD Implementation Strategy

### **Test Pyramid for Docling Integration**

```
                    🔺 E2E Tests (10%)
                   /                 \
                  /   Integration     \
                 /     Tests (20%)     \
                /                       \
               /_________________________\
                    Unit Tests (70%)
```

#### **Unit Tests (70% - Fast, Isolated)**
- Individual component behavior
- Mock external dependencies
- Fast execution (<1ms per test)
- High coverage of edge cases

#### **Integration Tests (20% - Component Interaction)**
- Component interaction validation
- Real file processing with test data
- Database and cache integration
- API endpoint testing

#### **End-to-End Tests (10% - Full System)**
- Complete user workflows
- Real document processing
- Performance validation
- Cross-format compatibility

---

## TDD Test Structure Standards

### **Test Organization Pattern**
```
tests/
├── unit/
│   ├── chunkers/
│   │   ├── test_docling_processor.py
│   │   ├── test_hybrid_chunker_tdd.py
│   │   └── test_format_detector.py
│   ├── llm/
│   │   ├── test_docling_provider.py
│   │   └── test_vision_models.py
│   └── utils/
│       ├── test_file_handler_multiformat.py
│       └── test_security_validation.py
├── integration/
│   ├── test_document_processing_pipeline.py
│   ├── test_multi_format_chunking.py
│   └── test_quality_evaluation_tdd.py
├── e2e/
│   ├── test_pdf_processing_workflow.py
│   ├── test_batch_processing.py
│   └── test_performance_benchmarks.py
└── fixtures/
    ├── sample_documents/
    ├── expected_outputs/
    └── test_data_generators.py
```

### **Test Naming Convention**
```python
# Pattern: test_[action]_[condition]_[expected_result]
def test_process_pdf_with_images_returns_chunks_with_descriptions():
def test_detect_format_invalid_file_raises_validation_error():
def test_chunk_large_document_within_memory_limits():
```

---

## TDD Implementation by Sprint

### **Sprint 1: TDD Foundation Setup**

#### **DOC-001-TDD: TDD Infrastructure Setup**
**Story Points**: 5  
**TDD Focus**: Establish testing infrastructure

**Red Phase - Write Failing Tests**:
```python
# tests/unit/test_tdd_infrastructure.py
import pytest
from src.chunkers.docling_processor import DoclingProcessor

class TestTDDInfrastructure:
    def test_docling_processor_can_be_imported(self):
        """Test that DoclingProcessor can be imported (will fail initially)."""
        processor = DoclingProcessor()
        assert processor is not None
    
    def test_docling_processor_has_process_document_method(self):
        """Test that process_document method exists."""
        processor = DoclingProcessor()
        assert hasattr(processor, 'process_document')
    
    def test_docling_processor_implements_base_interface(self):
        """Test that DoclingProcessor implements BaseProcessor."""
        from src.chunkers.base_processor import BaseProcessor
        processor = DoclingProcessor()
        assert isinstance(processor, BaseProcessor)
```

**Green Phase - Minimal Implementation**:
```python
# src/chunkers/base_processor.py (create if doesn't exist)
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path

class BaseProcessor(ABC):
    @abstractmethod
    def process_document(self, file_path: Path, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        pass

# src/chunkers/docling_processor.py (create minimal version)
from .base_processor import BaseProcessor
from typing import Dict, Any
from pathlib import Path

class DoclingProcessor(BaseProcessor):
    def process_document(self, file_path: Path, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        return {"status": "not_implemented"}
```

**Refactor Phase**: Clean up imports and structure

---

#### **DOC-002-TDD: Format Detection with TDD**
**Story Points**: 8  
**TDD Focus**: Format detection logic

**Red Phase - Comprehensive Test Suite**:
```python
# tests/unit/chunkers/test_format_detector.py
import pytest
from pathlib import Path
from src.utils.format_detector import FormatDetector
from src.exceptions import ValidationError

class TestFormatDetector:
    @pytest.fixture
    def detector(self):
        return FormatDetector()
    
    def test_detect_pdf_format_by_extension(self, detector):
        """Test PDF format detection by file extension."""
        file_path = Path("test.pdf")
        result = detector.detect_format(file_path)
        assert result == "pdf"
    
    def test_detect_docx_format_by_mime_type(self, detector):
        """Test DOCX format detection by MIME type."""
        # This will fail initially - drives implementation
        file_path = Path("document.docx")
        result = detector.detect_format(file_path)
        assert result == "docx"
    
    def test_detect_markdown_format_legacy_support(self, detector):
        """Test markdown format detection for backward compatibility."""
        file_path = Path("readme.md")
        result = detector.detect_format(file_path)
        assert result == "markdown"
    
    def test_unsupported_format_raises_validation_error(self, detector):
        """Test that unsupported formats raise ValidationError."""
        file_path = Path("document.xyz")
        with pytest.raises(ValidationError, match="Unsupported format"):
            detector.detect_format(file_path)
    
    @pytest.mark.parametrize("extension,expected", [
        (".pdf", "pdf"),
        (".docx", "docx"),
        (".pptx", "pptx"),
        (".html", "html"),
        (".md", "markdown"),
        (".txt", "markdown")
    ])
    def test_format_detection_parametrized(self, detector, extension, expected):
        """Parametrized test for multiple format detection."""
        file_path = Path(f"test{extension}")
        result = detector.detect_format(file_path)
        assert result == expected
```

**Green Phase - Minimal Implementation**:
```python
# src/utils/format_detector.py
from pathlib import Path
from src.exceptions import ValidationError

class FormatDetector:
    FORMAT_MAPPING = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.pptx': 'pptx',
        '.html': 'html',
        '.md': 'markdown',
        '.txt': 'markdown'
    }
    
    def detect_format(self, file_path: Path) -> str:
        extension = file_path.suffix.lower()
        if extension in self.FORMAT_MAPPING:
            return self.FORMAT_MAPPING[extension]
        raise ValidationError(f"Unsupported format: {extension}")
```

**Refactor Phase**: Add MIME type detection, error handling improvements

---

### **Sprint 2: Core PDF Processing with TDD**

#### **DOC-011-TDD: PDF Processing Pipeline**
**Story Points**: 13  
**TDD Focus**: PDF document processing

**Red Phase - Test-First PDF Processing**:
```python
# tests/unit/chunkers/test_docling_processor_pdf.py
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from src.chunkers.docling_processor import DoclingProcessor
from src.exceptions import ProcessingError

class TestDoclingProcessorPDF:
    @pytest.fixture
    def processor(self):
        return DoclingProcessor()
    
    @pytest.fixture
    def sample_pdf_path(self):
        return Path("tests/fixtures/sample.pdf")
    
    def test_process_pdf_returns_structured_content(self, processor, sample_pdf_path):
        """Test that PDF processing returns structured content."""
        result = processor.process_document(sample_pdf_path)
        
        assert result["success"] is True
        assert "content" in result
        assert "metadata" in result
        assert result["format"] == "pdf"
        assert len(result["content"]) > 0
    
    def test_process_pdf_extracts_text_content(self, processor, sample_pdf_path):
        """Test that text content is extracted from PDF."""
        result = processor.process_document(sample_pdf_path)
        
        content = result["content"]
        assert isinstance(content, str)
        assert len(content.strip()) > 0
        # Should contain some expected text from test PDF
    
    def test_process_pdf_detects_tables(self, processor):
        """Test that tables are detected in PDF documents."""
        # Mock PDF with tables
        with patch('src.chunkers.docling_processor.DocumentConverter') as mock_converter:
            mock_doc = Mock()
            mock_doc.tables = [{"content": "table1"}, {"content": "table2"}]
            mock_converter.return_value.convert.return_value.document = mock_doc
            
            result = processor.process_document(Path("test.pdf"))
            
            assert "tables" in result["metadata"]
            assert len(result["metadata"]["tables"]) == 2
    
    def test_process_corrupted_pdf_raises_processing_error(self, processor):
        """Test that corrupted PDFs raise ProcessingError."""
        corrupted_pdf = Path("tests/fixtures/corrupted.pdf")
        
        with pytest.raises(ProcessingError, match="Failed to process PDF"):
            processor.process_document(corrupted_pdf)
    
    def test_process_password_protected_pdf_raises_error(self, processor):
        """Test that password-protected PDFs are handled appropriately."""
        protected_pdf = Path("tests/fixtures/protected.pdf")
        
        with pytest.raises(ProcessingError, match="Password protected"):
            processor.process_document(protected_pdf)
    
    @pytest.mark.performance
    def test_process_large_pdf_within_time_limit(self, processor):
        """Test that large PDF processing meets performance requirements."""
        import time
        large_pdf = Path("tests/fixtures/large_document.pdf")
        
        start_time = time.time()
        result = processor.process_document(large_pdf)
        processing_time = time.time() - start_time
        
        assert result["success"] is True
        assert processing_time < 30.0  # 30 second limit for large docs
```

**Green Phase - Implement PDF Processing**:
```python
# src/chunkers/docling_processor.py (expand implementation)
import time
from pathlib import Path
from typing import Dict, Any, Optional
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from src.exceptions import ProcessingError
from .base_processor import BaseProcessor

class DoclingProcessor(BaseProcessor):
    def __init__(self):
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF]
        )
    
    def process_document(self, file_path: Path, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            # Validate file exists and is readable
            if not file_path.exists():
                raise ProcessingError(f"File not found: {file_path}")
            
            # Convert document using Docling
            conversion_result = self.converter.convert(str(file_path))
            
            if conversion_result.status != "success":
                raise ProcessingError(f"Failed to process PDF: {conversion_result.status}")
            
            doc = conversion_result.document
            
            # Extract content and metadata
            content = doc.export_to_markdown()
            
            # Extract tables if present
            tables = []
            if hasattr(doc, 'tables') and doc.tables:
                tables = [{"content": str(table)} for table in doc.tables]
            
            return {
                "success": True,
                "content": content,
                "format": "pdf",
                "metadata": {
                    "file_path": str(file_path),
                    "tables": tables,
                    "page_count": getattr(doc, 'page_count', 0),
                    "processing_time": time.time()
                }
            }
            
        except Exception as e:
            if "password" in str(e).lower():
                raise ProcessingError(f"Password protected PDF: {file_path}")
            raise ProcessingError(f"Failed to process PDF {file_path}: {str(e)}")
```

**Refactor Phase**: Extract methods, improve error handling, add logging

---

### **Sprint 2: Vision Model Integration with TDD**

#### **DOC-015-TDD: Vision Model Processing**
**Story Points**: 13  
**TDD Focus**: Image description and visual content processing

**Red Phase - Vision Processing Tests**:
```python
# tests/unit/llm/test_docling_provider_vision.py
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from src.llm.providers.docling_provider import DoclingProvider
from src.exceptions import VisionProcessingError

class TestDoclingProviderVision:
    @pytest.fixture
    def provider(self):
        return DoclingProvider(enable_vision=True)
    
    def test_describe_image_returns_description(self, provider):
        """Test that image description is generated."""
        image_data = {"path": "test.jpg", "format": "jpeg"}
        
        with patch.object(provider, '_call_vision_model') as mock_vision:
            mock_vision.return_value = "A chart showing quarterly sales data"
            
            description = provider.describe_image(image_data)
            
            assert isinstance(description, str)
            assert len(description) > 10
            assert "chart" in description.lower()
    
    def test_process_pdf_with_images_includes_descriptions(self, provider):
        """Test that PDF processing includes image descriptions."""
        pdf_path = Path("tests/fixtures/pdf_with_images.pdf")
        
        with patch('src.llm.providers.docling_provider.VlmPipeline') as mock_pipeline:
            mock_result = Mock()
            mock_result.document.images = [
                {"id": "img1", "description": "Sales chart"},
                {"id": "img2", "description": "Product diagram"}
            ]
            mock_pipeline.return_value.process.return_value = mock_result
            
            result = provider.process_with_vision(pdf_path)
            
            assert "images" in result
            assert len(result["images"]) == 2
            assert result["images"][0]["description"] == "Sales chart"
    
    def test_vision_processing_disabled_returns_empty_descriptions(self, provider):
        """Test behavior when vision processing is disabled."""
        provider_no_vision = DoclingProvider(enable_vision=False)
        image_data = {"path": "test.jpg"}
        
        description = provider_no_vision.describe_image(image_data)
        
        assert description == ""
    
    def test_vision_model_failure_raises_appropriate_error(self, provider):
        """Test that vision model failures are handled appropriately."""
        image_data = {"path": "invalid.jpg"}
        
        with patch.object(provider, '_call_vision_model') as mock_vision:
            mock_vision.side_effect = Exception("Vision model API error")
            
            with pytest.raises(VisionProcessingError):
                provider.describe_image(image_data)
    
    @pytest.mark.parametrize("image_format,expected_processing", [
        ("jpeg", True),
        ("png", True),
        ("gif", False),  # Not supported
        ("bmp", True)
    ])
    def test_image_format_support(self, provider, image_format, expected_processing):
        """Test support for different image formats."""
        image_data = {"path": f"test.{image_format}", "format": image_format}
        
        if expected_processing:
            # Should process without error
            with patch.object(provider, '_call_vision_model') as mock_vision:
                mock_vision.return_value = "Image description"
                result = provider.describe_image(image_data)
                assert len(result) > 0
        else:
            # Should raise error for unsupported format
            with pytest.raises(VisionProcessingError, match="Unsupported format"):
                provider.describe_image(image_data)
```

**Green Phase - Implement Vision Processing**:
```python
# src/llm/providers/docling_provider.py
from typing import Dict, Any, List, Optional
from pathlib import Path
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import VlmPipelineOptions
from src.llm.providers.base import BaseLLMProvider
from src.exceptions import VisionProcessingError

class DoclingProvider(BaseLLMProvider):
    SUPPORTED_IMAGE_FORMATS = {'jpeg', 'jpg', 'png', 'bmp', 'tiff'}
    
    def __init__(self, 
                 enable_vision: bool = True,
                 vision_model: str = "granite-vision"):
        self.enable_vision = enable_vision
        self.vision_model = vision_model
        
        if self.enable_vision:
            self.vlm_pipeline = VlmPipeline(
                pipeline_options=VlmPipelineOptions(
                    vlm_model=vision_model
                )
            )
    
    @property
    def provider_name(self) -> str:
        return "docling"
    
    def describe_image(self, image_data: Dict[str, Any]) -> str:
        """Generate description for an image."""
        if not self.enable_vision:
            return ""
        
        image_format = image_data.get("format", "").lower()
        if image_format not in self.SUPPORTED_IMAGE_FORMATS:
            raise VisionProcessingError(f"Unsupported format: {image_format}")
        
        try:
            description = self._call_vision_model(image_data)
            return description
        except Exception as e:
            raise VisionProcessingError(f"Vision processing failed: {str(e)}")
    
    def process_with_vision(self, document_path: Path, 
                           options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process document with vision models."""
        if not self.enable_vision:
            return {"images": [], "vision_enabled": False}
        
        try:
            result = self.vlm_pipeline.process(str(document_path))
            
            images = []
            if hasattr(result.document, 'images'):
                for img in result.document.images:
                    images.append({
                        "id": getattr(img, 'id', ''),
                        "description": getattr(img, 'description', ''),
                        "path": getattr(img, 'path', '')
                    })
            
            return {
                "images": images,
                "vision_enabled": True,
                "model_used": self.vision_model
            }
            
        except Exception as e:
            raise VisionProcessingError(f"Document vision processing failed: {str(e)}")
    
    def _call_vision_model(self, image_data: Dict[str, Any]) -> str:
        """Call vision model API - implement based on actual API."""
        # This would call the actual vision model
        # For now, return mock response
        return f"Image description for {image_data['path']}"
    
    def count_tokens(self, text: str, include_images: int = 0) -> int:
        """Count tokens including image processing overhead."""
        base_tokens = len(text.split()) * 1.3  # Rough approximation
        image_tokens = include_images * 150  # Approximate tokens per image
        return int(base_tokens + image_tokens)
```

**Refactor Phase**: Add configuration management, improve error handling, add caching

---

### **Sprint 3: Quality Evaluation with TDD**

#### **DOC-028-TDD: Advanced Quality Metrics**
**Story Points**: 13  
**TDD Focus**: Multi-format quality evaluation

**Red Phase - Quality Evaluation Tests**:
```python
# tests/unit/chunkers/test_quality_evaluator_multiformat.py
import pytest
from src.chunkers.evaluators import MultiFormatQualityEvaluator
from src.chunkers.quality_metrics import QualityMetrics

class TestMultiFormatQualityEvaluator:
    @pytest.fixture
    def evaluator(self):
        return MultiFormatQualityEvaluator()
    
    @pytest.fixture
    def pdf_chunks(self):
        """Sample PDF-derived chunks for testing."""
        return [
            {
                "content": "# Introduction\nThis document covers...",
                "metadata": {
                    "format": "pdf",
                    "page": 1,
                    "has_images": True,
                    "structural_elements": ["header", "paragraph"]
                }
            },
            {
                "content": "## Section 1\nDetailed analysis shows...",
                "metadata": {
                    "format": "pdf", 
                    "page": 2,
                    "has_tables": True,
                    "structural_elements": ["header", "paragraph", "table"]
                }
            }
        ]
    
    def test_evaluate_structure_preservation_pdf(self, evaluator, pdf_chunks):
        """Test structure preservation evaluation for PDF chunks."""
        score = evaluator.evaluate_structure_preservation(pdf_chunks, "pdf")
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Should be high for well-structured chunks
        assert score > 0.7
    
    def test_evaluate_boundary_preservation(self, evaluator, pdf_chunks):
        """Test boundary preservation scoring."""
        score = evaluator.evaluate_boundary_preservation(pdf_chunks)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Headers should create good boundaries
        assert score > 0.6
    
    def test_evaluate_visual_content_quality(self, evaluator, pdf_chunks):
        """Test visual content quality assessment."""
        quality_metrics = evaluator.evaluate_visual_content_quality(pdf_chunks)
        
        assert "image_description_quality" in quality_metrics
        assert "table_preservation_quality" in quality_metrics
        assert "overall_visual_quality" in quality_metrics
        
        # Should detect images and tables
        assert quality_metrics["images_detected"] > 0
        assert quality_metrics["tables_detected"] > 0
    
    def test_calculate_overall_quality_score(self, evaluator, pdf_chunks):
        """Test overall quality score calculation."""
        quality_result = evaluator.evaluate_chunks(pdf_chunks, document_format="pdf")
        
        assert "overall_score" in quality_result
        assert "component_scores" in quality_result
        assert "recommendations" in quality_result
        
        overall_score = quality_result["overall_score"]
        assert 0.0 <= overall_score <= 100.0
    
    @pytest.mark.parametrize("format_type,expected_min_score", [
        ("pdf", 70.0),      # PDFs should score well with structure
        ("docx", 75.0),     # DOCX has good structure preservation
        ("markdown", 80.0), # Markdown should score highest
        ("html", 65.0)      # HTML may have more noise
    ])
    def test_format_specific_quality_thresholds(self, evaluator, format_type, expected_min_score):
        """Test that quality evaluation considers format-specific expectations."""
        # Create format-specific test chunks
        chunks = self._create_test_chunks_for_format(format_type)
        
        result = evaluator.evaluate_chunks(chunks, document_format=format_type)
        
        # Should meet format-specific minimum quality
        assert result["overall_score"] >= expected_min_score
    
    def test_quality_recommendations_provided(self, evaluator, pdf_chunks):
        """Test that quality evaluation provides actionable recommendations."""
        # Create low-quality chunks to trigger recommendations
        low_quality_chunks = [
            {
                "content": "text without structure",
                "metadata": {"format": "pdf", "structural_elements": []}
            }
        ]
        
        result = evaluator.evaluate_chunks(low_quality_chunks, document_format="pdf")
        
        assert len(result["recommendations"]) > 0
        assert any("structure" in rec.lower() for rec in result["recommendations"])
    
    def _create_test_chunks_for_format(self, format_type: str):
        """Helper to create format-specific test chunks."""
        base_chunk = {
            "content": "# Header\nSample content for testing",
            "metadata": {
                "format": format_type,
                "structural_elements": ["header", "paragraph"]
            }
        }
        return [base_chunk]
```

**Green Phase - Implement Quality Evaluation**:
```python
# src/chunkers/evaluators.py (extend existing class)
from typing import List, Dict, Any, Optional
import statistics
from .quality_metrics import QualityMetrics

class MultiFormatQualityEvaluator(ChunkQualityEvaluator):
    """Enhanced quality evaluator for multi-format documents."""
    
    FORMAT_WEIGHTS = {
        "pdf": {
            "structure_preservation": 0.30,
            "boundary_preservation": 0.25,
            "visual_content": 0.20,
            "semantic_coherence": 0.25
        },
        "docx": {
            "structure_preservation": 0.35,
            "boundary_preservation": 0.20,
            "visual_content": 0.15,
            "semantic_coherence": 0.30
        },
        "markdown": {
            "structure_preservation": 0.25,
            "boundary_preservation": 0.20,
            "visual_content": 0.05,
            "semantic_coherence": 0.50
        }
    }
    
    def evaluate_chunks(self, chunks: List[Dict[str, Any]], 
                       document_format: str = "unknown") -> Dict[str, Any]:
        """Evaluate chunks with format-specific quality metrics."""
        
        if not chunks:
            return {"overall_score": 0.0, "error": "No chunks to evaluate"}
        
        # Calculate component scores
        structure_score = self.evaluate_structure_preservation(chunks, document_format)
        boundary_score = self.evaluate_boundary_preservation(chunks)
        visual_quality = self.evaluate_visual_content_quality(chunks)
        semantic_score = self._calculate_semantic_coherence(chunks)
        
        # Apply format-specific weights
        weights = self.FORMAT_WEIGHTS.get(document_format, self.FORMAT_WEIGHTS["markdown"])
        
        overall_score = (
            structure_score * weights["structure_preservation"] +
            boundary_score * weights["boundary_preservation"] +
            visual_quality["overall_visual_quality"] * weights["visual_content"] +
            semantic_score * weights["semantic_coherence"]
        ) * 100
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            structure_score, boundary_score, visual_quality, semantic_score
        )
        
        return {
            "overall_score": round(overall_score, 1),
            "component_scores": {
                "structure_preservation": round(structure_score * 100, 1),
                "boundary_preservation": round(boundary_score * 100, 1),
                "visual_content_quality": round(visual_quality["overall_visual_quality"] * 100, 1),
                "semantic_coherence": round(semantic_score * 100, 1)
            },
            "visual_metrics": visual_quality,
            "recommendations": recommendations,
            "format": document_format
        }
    
    def evaluate_structure_preservation(self, chunks: List[Dict[str, Any]], 
                                      document_format: str) -> float:
        """Evaluate how well document structure is preserved."""
        if not chunks:
            return 0.0
        
        structure_scores = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            structural_elements = metadata.get("structural_elements", [])
            
            # Score based on structural richness
            if "header" in structural_elements:
                structure_scores.append(1.0)
            elif "paragraph" in structural_elements:
                structure_scores.append(0.7)
            else:
                structure_scores.append(0.3)
        
        return statistics.mean(structure_scores) if structure_scores else 0.0
    
    def evaluate_boundary_preservation(self, chunks: List[Dict[str, Any]]) -> float:
        """Evaluate quality of chunk boundaries."""
        if len(chunks) < 2:
            return 1.0  # Single chunk doesn't have boundary issues
        
        boundary_scores = []
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Check if chunks have clear semantic boundaries
            current_ends_with_structure = self._ends_with_structural_element(current_chunk)
            next_starts_with_structure = self._starts_with_structural_element(next_chunk)
            
            if current_ends_with_structure or next_starts_with_structure:
                boundary_scores.append(1.0)
            else:
                boundary_scores.append(0.5)
        
        return statistics.mean(boundary_scores) if boundary_scores else 0.0
    
    def evaluate_visual_content_quality(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate quality of visual content processing."""
        images_detected = 0
        tables_detected = 0
        images_with_descriptions = 0
        tables_with_structure = 0
        
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            
            if metadata.get("has_images"):
                images_detected += 1
                if metadata.get("image_descriptions"):
                    images_with_descriptions += 1
            
            if metadata.get("has_tables"):
                tables_detected += 1
                if metadata.get("table_structure"):
                    tables_with_structure += 1
        
        # Calculate visual quality scores
        image_quality = (
            images_with_descriptions / images_detected 
            if images_detected > 0 else 1.0
        )
        
        table_quality = (
            tables_with_structure / tables_detected 
            if tables_detected > 0 else 1.0
        )
        
        overall_visual_quality = (image_quality + table_quality) / 2
        
        return {
            "images_detected": images_detected,
            "tables_detected": tables_detected,
            "image_description_quality": image_quality,
            "table_preservation_quality": table_quality,
            "overall_visual_quality": overall_visual_quality
        }
    
    def _calculate_semantic_coherence(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate semantic coherence score."""
        # Use existing semantic coherence calculation
        return super().calculate_semantic_coherence(chunks)
    
    def _generate_recommendations(self, structure_score: float, boundary_score: float,
                                visual_quality: Dict[str, Any], semantic_score: float) -> List[str]:
        """Generate actionable recommendations for improvement."""
        recommendations = []
        
        if structure_score < 0.7:
            recommendations.append("Consider adjusting chunking strategy to better preserve document structure")
        
        if boundary_score < 0.6:
            recommendations.append("Improve chunk boundary detection to maintain semantic coherence")
        
        if visual_quality["overall_visual_quality"] < 0.5:
            recommendations.append("Enable vision models to improve image and table processing")
        
        if semantic_score < 0.6:
            recommendations.append("Increase chunk overlap or adjust chunk size for better semantic flow")
        
        return recommendations
    
    def _ends_with_structural_element(self, chunk: Dict[str, Any]) -> bool:
        """Check if chunk ends with a structural element."""
        content = chunk.get("content", "")
        return content.strip().endswith(('.', '!', '?', ':', '\n#'))
    
    def _starts_with_structural_element(self, chunk: Dict[str, Any]) -> bool:
        """Check if chunk starts with a structural element."""
        content = chunk.get("content", "")
        return content.strip().startswith(('#', '##', '###', '-', '*', '1.'))
```

**Refactor Phase**: Extract quality calculation methods, add configuration, improve performance

---

## TDD Testing Standards and Practices

### **Test Fixtures and Data Management**

```python
# tests/conftest.py - Shared fixtures
import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "fixtures" / "sample_documents"

@pytest.fixture(scope="session")  
def sample_pdf():
    """Provide sample PDF for testing."""
    return Path(__file__).parent / "fixtures" / "sample_documents" / "test_document.pdf"

@pytest.fixture
def temp_directory():
    """Provide temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_docling_converter():
    """Mock Docling DocumentConverter for unit tests."""
    with patch('src.chunkers.docling_processor.DocumentConverter') as mock:
        mock_instance = Mock()
        mock_instance.convert.return_value = Mock(
            status="success",
            document=Mock()
        )
        mock.return_value = mock_instance
        yield mock_instance
```

### **Test Data Generation**

```python
# tests/fixtures/test_data_generators.py
from pathlib import Path
from typing import List, Dict, Any
import json

class TestDataGenerator:
    """Generate test data for TDD development."""
    
    @staticmethod
    def create_sample_chunks(format_type: str, count: int = 5) -> List[Dict[str, Any]]:
        """Create sample chunks for testing."""
        chunks = []
        for i in range(count):
            chunk = {
                "content": f"# Section {i+1}\nThis is sample content for testing chunk {i+1}.",
                "metadata": {
                    "chunk_index": i,
                    "format": format_type,
                    "chunk_id": f"test_chunk_{i}",
                    "structural_elements": ["header", "paragraph"]
                }
            }
            chunks.append(chunk)
        return chunks
    
    @staticmethod
    def create_pdf_chunk_with_images() -> Dict[str, Any]:
        """Create PDF chunk with image metadata for testing."""
        return {
            "content": "Figure 1 shows the quarterly results.",
            "metadata": {
                "format": "pdf",
                "has_images": True,
                "image_descriptions": ["Chart showing quarterly sales data"],
                "page": 1,
                "structural_elements": ["paragraph", "image"]
            }
        }
    
    @staticmethod
    def create_quality_test_scenarios() -> Dict[str, List[Dict[str, Any]]]:
        """Create different quality scenarios for testing."""
        return {
            "high_quality": [
                {
                    "content": "# Introduction\nWell-structured content with clear headers.",
                    "metadata": {
                        "format": "pdf",
                        "structural_elements": ["header", "paragraph"],
                        "semantic_score": 0.9
                    }
                }
            ],
            "low_quality": [
                {
                    "content": "random text without structure or meaning fragmented",
                    "metadata": {
                        "format": "pdf", 
                        "structural_elements": [],
                        "semantic_score": 0.3
                    }
                }
            ]
        }
```

### **Performance Testing with TDD**

```python
# tests/performance/test_performance_benchmarks.py
import pytest
import time
from pathlib import Path
from src.chunkers.docling_processor import DoclingProcessor

class TestPerformanceBenchmarks:
    """Performance tests following TDD principles."""
    
    @pytest.mark.performance
    def test_pdf_processing_time_benchmark(self):
        """Test that PDF processing meets time benchmarks."""
        processor = DoclingProcessor()
        pdf_path = Path("tests/fixtures/medium_document.pdf")  # ~50 pages
        
        start_time = time.time()
        result = processor.process_document(pdf_path)
        processing_time = time.time() - start_time
        
        # Test passes/fails based on performance requirement
        assert result["success"] is True
        assert processing_time < 30.0, f"Processing took {processing_time:.2f}s, expected <30s"
    
    @pytest.mark.performance
    def test_memory_usage_within_limits(self):
        """Test that memory usage stays within acceptable limits."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        processor = DoclingProcessor()
        # Process multiple documents to test memory accumulation
        for i in range(5):
            pdf_path = Path(f"tests/fixtures/test_doc_{i}.pdf")
            processor.process_document(pdf_path)
        
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not increase memory by more than 500MB
        assert memory_increase < 500, f"Memory increased by {memory_increase:.1f}MB"
    
    @pytest.mark.performance  
    @pytest.mark.parametrize("concurrent_docs", [1, 3, 5])
    def test_concurrent_processing_performance(self, concurrent_docs):
        """Test performance under concurrent processing load."""
        import concurrent.futures
        import threading
        
        processor = DoclingProcessor()
        pdf_paths = [Path(f"tests/fixtures/test_doc_{i}.pdf") for i in range(concurrent_docs)]
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_docs) as executor:
            futures = [executor.submit(processor.process_document, path) for path in pdf_paths]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # All should succeed
        assert all(result["success"] for result in results)
        
        # Total time should be reasonable for concurrent processing
        expected_max_time = 60 * concurrent_docs / 3  # Assume 3x efficiency from concurrency
        assert total_time < expected_max_time
```

### **TDD Code Coverage Requirements**

```python
# pytest.ini (update configuration)
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --cov=src
    --cov-report=html:htmlcov
    --cov-report=term-missing
    --cov-report=xml
    --cov-fail-under=90
    --cov-branch
    -v
    --tb=short
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower, real dependencies)
    e2e: End-to-end tests (slowest, full system)
    performance: Performance benchmark tests
    security: Security validation tests
    tdd: Tests written following TDD methodology
```

### **TDD Development Workflow**

#### **Daily TDD Cycle**
1. **Morning**: Review failing tests from previous day
2. **Red Phase**: Write failing tests for new feature (15-30 min)
3. **Green Phase**: Write minimal code to pass tests (30-60 min)
4. **Refactor Phase**: Improve code quality (15-30 min)
5. **Commit**: Commit working code with tests
6. **Repeat**: 3-4 cycles per day per developer

#### **TDD Sprint Planning**
- **Sprint Planning**: Define test scenarios before coding tasks
- **Daily Standups**: Discuss failing tests and TDD progress
- **Code Reviews**: Verify tests were written before implementation
- **Sprint Demo**: Show test coverage and quality metrics
- **Retrospective**: Review TDD effectiveness and improvements

#### **TDD Quality Gates**
- **Pre-commit**: All tests must pass
- **Code Review**: Test-first development verified
- **CI/CD**: 90% test coverage required
- **Deployment**: All integration tests passing

This TDD approach ensures that every line of code for the Docling integration is justified by a test, leading to higher quality, better design, and more maintainable code.