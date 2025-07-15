# Document Chunking System 🚀

An **enterprise-grade hybrid document chunking system** optimized for RAG (Retrieval-Augmented Generation) applications. This production-ready system intelligently processes large documents into optimal chunks while preserving semantic structure, maintaining quality metrics, and providing comprehensive observability.

## ✨ Enterprise Features

### 🔀 **Advanced Chunking Engine**
- **Hybrid Chunking Strategies**: Combines header-based, recursive, and code-aware chunking
- **📊 Quality Evaluation**: Comprehensive chunk quality analysis with detailed reporting
- **🤖 RAG-Optimized**: Designed for optimal performance with language models (Gemini, GPT, etc.)
- **🏷️ Metadata Enrichment**: Automatic metadata enhancement with content analysis
- **🔍 Content Type Detection**: Automatic detection of headers, code, lists, and tables

### 🛡️ **Production-Ready Infrastructure** 
- **⚡ Performance Optimized**: Memory-efficient processing with intelligent caching
- **🔒 Security Hardened**: Input validation, file sanitization, and security auditing
- **⚙️ Batch Processing**: Efficient processing of multiple documents with progress tracking
- **📄 Multiple Output Formats**: JSON, CSV, and Pickle support

### 📈 **Enterprise Observability** (Phase 4)
- **🔍 Distributed Tracing**: Correlation IDs and structured logging across requests
- **📊 Advanced Metrics**: Prometheus-compatible metrics with custom business logic
- **💊 Health Monitoring**: Comprehensive health checks with dependency tracking
- **🚨 Intelligent Alerting**: SLA monitoring with automated alert management
- **📋 Production Dashboards**: Grafana dashboards for real-time system visibility
- **🔧 HTTP Endpoints**: REST API for health checks, metrics, and system status

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd chuncking-system

# Install dependencies (recommended: use uv for faster installation)
uv pip install -r requirements.txt

# Or use pip
pip install -r requirements.txt

# Install development dependencies (optional)
uv pip install pytest pytest-cov pytest-mock black flake8 mypy
```

### Basic Usage

```bash
# Process a single book/document
python main.py --input-file data/input/markdown_files/your_book.md

# Specify output directory and chunk size
python main.py \
  --input-file data/input/markdown_files/your_book.md \
  --output-dir data/output \
  --chunk-size 800 \
  --format json
```

### Python API Usage

```python
from src.chunkers.hybrid_chunker import HybridMarkdownChunker
from src.chunkers.evaluators import ChunkQualityEvaluator
from src.utils.file_handler import FileHandler

# Initialize the chunker
chunker = HybridMarkdownChunker(
    chunk_size=800,    # Target chunk size in tokens
    chunk_overlap=150  # Overlap between chunks
)

# Load your document
with open('your_document.md', 'r') as f:
    content = f.read()

# Chunk the document
chunks = chunker.chunk_document(content, {
    'source_file': 'your_document.md',
    'book_title': 'Your Book Title'
})

# Evaluate chunk quality
evaluator = ChunkQualityEvaluator()
quality_metrics = evaluator.evaluate_chunks(chunks)
print(f"Quality Score: {quality_metrics['overall_score']:.1f}/100")

# Save chunks
FileHandler.save_chunks(chunks, 'output/chunks.json', 'json')
```

## 📋 Requirements

### System Requirements
- **Python**: 3.11+
- **Memory**: 4GB RAM minimum, 16GB recommended
- **Storage**: 1GB free space for processing large documents

### Dependencies
- `langchain>=0.3.26` - Text processing and splitting
- `scikit-learn>=1.7.0` - Quality evaluation metrics
- `tiktoken>=0.9.0` - Token counting
- `pydantic>=2.11.7` - Configuration management
- `pandas>=2.3.1` - Data handling and CSV export
- `numpy>=2.3.1` - Numerical operations

See `requirements.txt` for complete dependency list.

## 🏗️ Enterprise Architecture

```
src/
├── api/                       # 🔧 Phase 4: HTTP Health & Monitoring Endpoints
│   ├── health_endpoints.py    #   REST API for health, metrics, system status
│   └── __init__.py           
├── chunkers/                  # 🔀 Core Chunking Engine
│   ├── hybrid_chunker.py      #   Main hybrid chunking logic
│   ├── evaluators.py          #   Quality evaluation and scoring
│   └── markdown_processor.py  #   Markdown-specific processing
├── config/
│   └── settings.py            # Configuration management
├── utils/                     # 🛡️ Production Infrastructure
│   ├── cache.py              #   Intelligent caching system (Phase 3)
│   ├── file_handler.py       #   File I/O operations
│   ├── logger.py             #   Structured logging infrastructure
│   ├── metadata_enricher.py  #   Metadata enhancement
│   ├── monitoring.py         #   System health monitoring (Phase 3)
│   ├── observability.py     #   📈 Enterprise observability (Phase 4)
│   ├── path_utils.py         #   Secure path handling
│   ├── performance.py        #   Performance optimization
│   ├── security.py           #   Security validation & auditing
│   └── validators.py         #   Input validation utilities
├── exceptions.py             #   Custom exception hierarchy
└── chunking_system.py       #   Main system orchestrator

dashboards/                   # 📋 Production Monitoring
├── grafana-dashboard.json    #   Real-time system dashboard
├── prometheus-alerts.yml     #   Comprehensive alerting rules
└── prometheus.yml           #   Metrics collection configuration
```

### Core Components

#### 🔀 HybridMarkdownChunker
The main chunking engine that:
- Analyzes content type (headers, code, tables, lists)
- Selects optimal chunking strategy
- Preserves document structure
- Maintains semantic coherence

#### 📊 ChunkQualityEvaluator
Comprehensive quality assessment including:
- Size distribution analysis
- Content quality metrics
- Semantic coherence scoring
- Structure preservation evaluation

#### 🏷️ MetadataEnricher
Automatic metadata enhancement:
- Content type detection
- Language identification
- Unique chunk IDs
- Processing timestamps

#### 📈 ObservabilityManager (Phase 4)
Enterprise observability infrastructure:
- **Distributed Tracing**: Correlation IDs and trace context propagation
- **Structured Logging**: JSON-formatted logs with contextual information
- **Metrics Registry**: Custom metrics with Prometheus export
- **Health Registry**: Component health monitoring with dependency tracking
- **Dashboard Generation**: Automated Grafana dashboard creation

#### 🔒 SecurityManager (Phase 3)
Production security features:
- **Input Validation**: File type, size, and path validation
- **Security Auditing**: Comprehensive file security scanning
- **Checksum Validation**: File integrity verification
- **Path Sanitization**: Directory traversal prevention

## ⚙️ Configuration

The system uses Pydantic-based configuration in `src/config/settings.py`:

```python
# Key configuration options
DEFAULT_CHUNK_SIZE = 800      # Target tokens per chunk
DEFAULT_CHUNK_OVERLAP = 150   # Token overlap between chunks
MIN_CHUNK_WORDS = 10         # Minimum words per chunk
MAX_CHUNK_WORDS = 600        # Maximum words per chunk

# Chunking strategies
HEADER_LEVELS = [
    ("#", "Part"),
    ("##", "Chapter"), 
    ("###", "Section"),
    ("####", "Sub-section")
]
```

### Environment Variables

Create a `.env` file:

```bash
# Optional: OpenAI API key for advanced features
OPENAI_API_KEY=your_openai_api_key_here

# Processing settings
BATCH_SIZE=10
ENABLE_PARALLEL=false
```

## 📄 Output Formats

### JSON Format (Default)
```json
[
  {
    "content": "# Chapter 1: Introduction\n\nThis is the first chapter...",
    "metadata": {
      "Header 1": "Chapter 1: Introduction",
      "chunk_index": 0,
      "chunk_tokens": 245,
      "chunk_chars": 1150,
      "word_count": 198,
      "chunk_id": "abc123def456",
      "processed_at": "2024-01-15T10:30:00",
      "has_code": false,
      "has_headers": true,
      "language": "english"
    }
  }
]
```

### CSV Format
| chunk_id | content | source | tokens | words |
|----------|---------|---------|---------|--------|
| 0 | Chapter content... | book.md | 245 | 198 |

### Quality Report
```markdown
# Chunk Quality Evaluation Report

## Summary
- **Total Chunks**: 45
- **Overall Quality Score**: 87.3/100

## Size Distribution
- **Average Characters**: 1,150
- **Average Words**: 198
- **Size Consistency**: 0.85

## Recommendations
- ✅ Excellent chunking quality!
- Size distribution is well-balanced
- Strong semantic coherence maintained
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test files
pytest tests/test_hybrid_chunker.py -v
pytest tests/test_evaluators.py -v
```

### Test Coverage
The project includes comprehensive tests for:
- ✅ **Phase 1**: Complete testing infrastructure (Unit, Integration, Performance)
- ✅ **Phase 2**: Documentation and code quality validation
- ✅ **Phase 3**: Security, caching, and monitoring tests
- ✅ **Phase 4**: Observability, health endpoints, and metrics tests

**Enterprise Test Suite**: 
- **Core Components**: 95%+ coverage for chunking engine
- **Security & Performance**: Comprehensive validation of Phase 3 features
- **Observability**: Full test coverage for Phase 4 monitoring infrastructure
- **Integration**: End-to-end workflow testing with real-world scenarios

## 📈 Performance

### Benchmarks (i3 CPU, 16GB RAM)
- **Small Document** (10-50 pages): ~5-15 seconds
- **Medium Book** (200-500 pages): ~30-90 seconds  
- **Large Book** (1000+ pages): ~2-5 minutes

### Memory Usage
- **Base Memory**: ~100-200MB
- **Per 1MB Document**: ~50-100MB additional
- **Peak Memory**: Typically <2GB for large books

### Enterprise Optimization Features
- **Intelligent Caching**: Multi-level caching with TTL and LRU eviction (Phase 3)
- **Performance Monitoring**: Real-time metrics and performance tracking (Phase 4)
- **Memory Optimization**: Batch processing with automatic memory cleanup
- **Security Validation**: File sanitization without performance degradation
- **Progress Tracking**: Real-time progress monitoring for long operations
- **Resource Management**: Adaptive resource allocation based on system load

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
uv pip install pytest pytest-cov pytest-mock black flake8 mypy

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
flake8 src/ tests/

# Type checking
mypy src/
```

### Running Quality Checks

```bash
# Code formatting
black --check src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/

# Test coverage
pytest --cov=src --cov-report=html --cov-fail-under=80
```

## 🔧 Advanced Usage

### Custom Chunking Strategies

```python
# Custom chunk size for specific use cases
chunker = HybridMarkdownChunker(
    chunk_size=1200,    # Larger chunks for summarization
    chunk_overlap=200,  # More overlap for context
    enable_semantic=True  # Enable semantic chunking (requires additional deps)
)
```

### Batch Processing

```python
# Process multiple files
file_paths = ['book1.md', 'book2.md', 'book3.md']

def progress_callback(current, total, filename):
    print(f"Processing {current}/{total}: {filename}")

results = chunker.batch_process_files(file_paths, progress_callback)
```

### Quality Thresholds

```python
# Set custom quality thresholds
evaluator = ChunkQualityEvaluator()
metrics = evaluator.evaluate_chunks(chunks)

if metrics['overall_score'] < 70:
    print("⚠️ Low quality chunks detected!")
    print("Consider adjusting chunk size or overlap settings")
```

## 📈 Enterprise Observability (Phase 4)

### Health Check Endpoints

```python
from src.api.health_endpoints import HealthEndpoint, create_flask_blueprint

# Standalone health server
from src.api.health_endpoints import run_standalone_server
run_standalone_server(host="0.0.0.0", port=8000)

# Flask integration
app.register_blueprint(create_flask_blueprint(), url_prefix='/monitoring')

# FastAPI integration  
from src.api.health_endpoints import create_fastapi_router
app.include_router(create_fastapi_router(), prefix="/monitoring")
```

**Available Endpoints:**
- `GET /health` - Basic health check
- `GET /health/detailed` - Comprehensive component health
- `GET /health/ready` - Kubernetes readiness probe
- `GET /health/live` - Kubernetes liveness probe
- `GET /metrics` - Prometheus metrics export
- `GET /metrics/json` - JSON metrics format
- `GET /system/info` - System information
- `GET /system/performance` - Performance statistics

### Distributed Tracing & Logging

```python
from src.utils.observability import get_structured_logger, trace_operation

# Structured logging with correlation IDs
logger = get_structured_logger(__name__)

with trace_operation("document_processing", document_id="doc123"):
    logger.info("Starting document processing", document_size=len(content))
    chunks = chunker.chunk_document(content)
    logger.info("Processing completed", chunk_count=len(chunks))
```

### Custom Metrics Collection

```python
from src.utils.observability import record_metric, MetricType

# Record business metrics
record_metric("documents_processed", 1, MetricType.COUNTER, "documents")
record_metric("processing_duration", duration_ms, MetricType.HISTOGRAM, "milliseconds")
record_metric("queue_size", current_queue_size, MetricType.GAUGE, "items")
```

### Production Monitoring Setup

```bash
# Start health check server
python -m src.api.health_endpoints

# Deploy Grafana dashboard
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @dashboards/grafana-dashboard.json

# Configure Prometheus alerts
cp dashboards/prometheus-alerts.yml /etc/prometheus/rules/
systemctl reload prometheus
```

## 🔧 Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'langchain_core'"
```bash
# Install missing dependencies
uv pip install langchain langchain-core langchain-text-splitters
```

#### "Memory Error during processing"
```bash
# Reduce chunk size and batch size
python main.py --input-file large_book.md --chunk-size 400
```

#### "Empty chunks generated"
- Check input file encoding (should be UTF-8)
- Verify file is not corrupted
- Ensure file contains actual content

#### "Low quality scores"
- Increase chunk overlap: `--chunk-overlap 200`
- Adjust chunk size for your content type
- Check for malformed markdown structure

### Debug Mode

```bash
# Enable verbose logging
export PYTHONPATH=/path/to/chuncking-system
python main.py --input-file book.md --verbose
```

### Performance Issues

1. **Slow processing**: Reduce `BATCH_SIZE` in settings
2. **High memory usage**: Decrease `DEFAULT_CHUNK_SIZE`
3. **Poor quality scores**: Increase `DEFAULT_CHUNK_OVERLAP`

### Monitoring & Observability Issues

#### "Health check endpoints not responding"
```bash
# Check if monitoring server is running
curl http://localhost:8000/health

# Start standalone monitoring server
python -m src.api.health_endpoints --host 0.0.0.0 --port 8000
```

#### "Metrics not appearing in Prometheus"
```bash
# Verify metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus configuration
promtool check config dashboards/prometheus.yml
```

#### "High alert volume"
- Review alert thresholds in `dashboards/prometheus-alerts.yml`
- Adjust SLA targets based on your requirements
- Enable alert grouping and notification routing

## 📚 Examples

### Processing Academic Papers
```bash
python main.py \
  --input-file papers/research_paper.md \
  --chunk-size 600 \
  --chunk-overlap 100 \
  --format json
```

### Processing Fiction Books
```bash
python main.py \
  --input-file books/novel.md \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --format csv
```

### Processing Technical Documentation
```bash
python main.py \
  --input-file docs/technical_manual.md \
  --chunk-size 800 \
  --chunk-overlap 150 \
  --format json
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for public methods
- Maintain test coverage >80%

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Development Phases

- **✅ Phase 1**: Testing Infrastructure & Core Components
- **✅ Phase 2**: Documentation & Code Quality  
- **✅ Phase 3**: Performance Optimization & Security
- **✅ Phase 4**: Enterprise Observability & Monitoring

## 🙏 Acknowledgments

### Core Technologies
- [LangChain](https://github.com/langchain-ai/langchain) for text processing utilities
- [scikit-learn](https://scikit-learn.org/) for machine learning metrics
- [tiktoken](https://github.com/openai/tiktoken) for accurate token counting
- [Pydantic](https://pydantic-docs.helpmanual.io/) for configuration management

### Observability Stack
- [Prometheus](https://prometheus.io/) for metrics collection and alerting
- [Grafana](https://grafana.com/) for visualization and dashboards
- [OpenTelemetry](https://opentelemetry.io/) for distributed tracing standards

## 📞 Enterprise Support

### Community Support
- **Issues**: [GitHub Issues](https://github.com/your-org/chuncking-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/chuncking-system/discussions)
- **Documentation**: [Wiki](https://github.com/your-org/chuncking-system/wiki)

### Production Deployment
- **Monitoring Runbooks**: See `dashboards/` directory for operational procedures
- **SLA Documentation**: Prometheus alerts define service level objectives
- **Health Checks**: Use `/health` endpoints for load balancer configuration
- **Metrics Integration**: Compatible with standard observability platforms

---

**🚀 Enterprise-grade document chunking for the RAG community**