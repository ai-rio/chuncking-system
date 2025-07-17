# Source Tree Integration

## **Existing Project Structure**

```plaintext
chunking-system/
├── src/
│   ├── api/                          # HTTP Health & Monitoring Endpoints
│   │   ├── health_endpoints.py       # REST API for system status
│   │   └── __init__.py              
│   ├── chunkers/                     # Core Chunking Engine
│   │   ├── hybrid_chunker.py         # Main chunking logic
│   │   ├── adaptive_chunker.py       # Strategy optimization
│   │   ├── strategy_optimizer.py     # Content analysis 
│   │   ├── strategy_tester.py        # Multi-strategy testing
│   │   ├── evaluators.py             # Quality evaluation
│   │   ├── markdown_processor.py     # Markdown-specific processing
│   │   └── __init__.py              
│   ├── config/
│   │   └── settings.py               # Pydantic configuration
│   ├── llm/                          # Multi-LLM Provider Support
│   │   ├── factory.py               # Provider factory
│   │   └── providers/               # LLM implementations
│   │       ├── base.py              # Abstract provider interface
│   │       ├── openai_provider.py   # OpenAI integration
│   │       ├── anthropic_provider.py # Anthropic Claude
│   │       ├── jina_provider.py     # Jina AI integration
│   │       └── __init__.py          
│   ├── utils/                        # Production Infrastructure
│   │   ├── cache.py                 # Multi-tier caching
│   │   ├── file_handler.py          # File I/O operations
│   │   ├── logger.py                # Structured logging
│   │   ├── metadata_enricher.py     # Metadata enhancement
│   │   ├── monitoring.py            # System monitoring
│   │   ├── observability.py         # Enterprise observability
│   │   ├── path_utils.py            # Path handling and security
│   │   ├── performance.py           # Performance optimization
│   │   ├── security.py              # Security validation
│   │   ├── validators.py            # Input validation
│   │   └── llm_quality_enhancer.py  # LLM-powered quality enhancement
│   ├── exceptions.py                # Custom exception hierarchy
│   └── chunking_system.py          # Main system orchestrator
├── tests/                           # Comprehensive Test Suite
├── docs/                           # Documentation
├── main.py                         # Application entry point
└── requirements.txt                # Dependencies
```

## **New File Organization**

```plaintext
chunking-system/
├── src/
│   ├── chunkers/                     # Core Chunking Engine
│   │   ├── docling_processor.py     # NEW: Core Docling integration
│   │   ├── hybrid_chunker.py        # ENHANCED: Multi-format integration
│   │   ├── evaluators.py            # ENHANCED: Multi-format quality metrics
│   │   └── (existing files...)      # All existing files preserved
│   ├── llm/
│   │   └── providers/               # LLM implementations
│   │       ├── docling_provider.py  # NEW: Docling LLM provider
│   │       └── (existing files...)  # All existing providers preserved
│   ├── utils/
│   │   ├── file_handler.py          # ENHANCED: Multi-format detection and routing
│   │   ├── security.py              # ENHANCED: Multi-format validation
│   │   └── (existing files...)      # All existing utilities preserved
│   ├── config/
│   │   └── settings.py              # ENHANCED: Docling configuration options
├── tests/
│   ├── test_chunkers/
│   │   ├── test_docling_processor.py # NEW: Docling processor tests
│   │   ├── test_multi_format_integration.py # NEW: Integration tests
│   │   └── (existing tests...)      # All existing tests preserved
│   ├── test_llm/
│   │   ├── test_docling_provider.py # NEW: Docling provider tests
│   │   └── (existing tests...)      # All existing LLM tests preserved
│   └── test_utils/
│       ├── test_enhanced_file_handler.py # NEW: Multi-format file handling tests
│       └── (existing tests...)      # All existing utility tests preserved
├── docs/
│   ├── architecture.md              # THIS DOCUMENT: Docling integration architecture
│   ├── prd.md                      # Brownfield PRD for implementation
│   ├── brownfield-architecture.md  # Current system analysis
│   └── docling/                    # Existing Docling planning documentation
└── requirements.txt                # ENHANCED: Docling dependencies added
```

## **Integration Guidelines**

- **File Naming**: Follow existing snake_case Python conventions established in current codebase (e.g., `docling_processor.py`, `docling_provider.py`), maintain consistency with current module naming patterns like `openai_provider.py`, `anthropic_provider.py`

- **Folder Organization**: Place new components in existing folders following established patterns - processors in `src/chunkers/`, providers in `src/llm/providers/`, maintain current logical grouping and avoid creating new top-level directories

- **Import/Export Patterns**: Follow existing import structure with absolute imports from src root, update `__init__.py` files to include new components following current patterns, maintain existing module exposure and API surface consistency
