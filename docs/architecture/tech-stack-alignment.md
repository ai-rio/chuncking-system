# Tech Stack Alignment

## **Existing Technology Stack**

| Category | Current Technology | Version | Usage in Enhancement | Notes |
|----------|-------------------|---------|---------------------|--------|
| Runtime | Python | 3.11+ | Core platform for all Docling components | Required minimum maintained |
| Core Framework | LangChain | 0.3.26+ | Extended for multi-format text processing | Existing chunking logic preserved |
| Configuration | Pydantic | 2.11.7+ | DoclingProvider settings and validation | Follows existing config patterns |
| LLM - OpenAI | OpenAI SDK | 1.95.1+ | Continues existing functionality | No changes to existing integration |
| LLM - Anthropic | Anthropic SDK | 0.7.0+ | Continues existing functionality | No changes to existing integration |
| LLM - Jina | Jina AI (HTTP) | Current | Continues existing functionality | No changes to existing integration |
| Document Processing | mistune | 3.1.3+ | Markdown processing preserved | Docling adds parallel processing path |
| Quality Metrics | scikit-learn | 1.7.0+ | Enhanced with multi-format metrics | Existing evaluation logic maintained |
| Data Processing | pandas | 2.3.1+ | Extended for multi-format metadata | Current CSV export functionality preserved |
| Numerical Operations | numpy | 2.3.1+ | Enhanced for vision content analysis | Existing numerical processing maintained |
| Testing Framework | pytest | 7.0.0+ | Extended with Docling integration tests | 95% coverage requirement maintained |
| Monitoring | Prometheus | Current | Enhanced with Docling metrics | Existing metrics infrastructure preserved |
| Observability | Grafana | Current | Extended dashboards for multi-format | Current dashboard functionality maintained |
| Containerization | Docker | Current | Enhanced with Docling dependencies | Existing deployment patterns preserved |

## **New Technology Additions**

| Technology | Version | Purpose | Rationale | Integration Method |
|------------|---------|---------|-----------|-------------------|
| Docling SDK | Latest stable | Multi-format document processing and vision capabilities | Required for PDF, DOCX, PPTX, HTML, image processing functionality | New DoclingProvider implements existing BaseLLMProvider interface |
| Python-magic | 0.4.27+ | Enhanced file type detection for multi-format routing | Reliable MIME type detection for secure format validation | Integrated into enhanced FileHandler format detection logic |
| Pillow (PIL) | 10.0.0+ | Image format validation and basic processing | Security validation for image files before Docling processing | Extends existing security validation framework |
