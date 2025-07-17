# Coding Standards and Conventions

## **Existing Standards Compliance**

**Code Style**: Black formatting with 88-character line length targeting Python 3.11, comprehensive type hints with mypy strict checking enabled, snake_case naming conventions for functions and variables, PascalCase for classes following established patterns in current codebase

**Linting Rules**: flake8 compliance for code quality, mypy strict type checking for all function definitions, comprehensive docstrings required for public methods following existing documentation patterns, import organization following current module structure

**Testing Patterns**: pytest framework with 95% coverage requirement maintained, comprehensive unit tests for all new components, integration tests validating existing system compatibility, TDD approach following established test organization in test_chunkers/, test_llm/, test_utils/ directories

**Documentation Style**: Comprehensive docstrings with parameter and return type documentation, inline comments for complex logic following current commenting patterns, README updates maintaining existing documentation structure and style

## **Enhancement-Specific Standards**

- **Docling Integration Pattern**: All Docling-related components must implement graceful fallback mechanisms when Docling API unavailable, ensuring system continues operating with existing functionality
- **Multi-Format Validation**: New file type validation must extend existing security framework patterns, following current PathSanitizer and FileValidator approaches for consistency
- **Provider Interface Compliance**: DoclingProvider must strictly implement BaseLLMProvider interface, maintaining compatibility with existing factory registration and provider switching mechanisms
- **Quality Evaluation Extensions**: Enhanced quality metrics must preserve existing evaluation patterns while adding multi-format capabilities, ensuring backward compatibility with current assessment workflows
- **Configuration Management**: All new settings must follow existing Pydantic model patterns with environment variable integration, maintaining current configuration validation and type safety approaches

## **Critical Integration Rules**

- **Existing API Compatibility**: All current Python API methods, CLI arguments, and return types must remain unchanged. New functionality accessible through optional parameters only, preserving existing method signatures and behavior exactly

- **Database Integration**: Chunk metadata extensions must use optional fields only, ensuring existing ChunkingResult processing continues without modification. No breaking changes to current data structures or export formats

- **Error Handling**: All new components must integrate with existing exception hierarchy defined in src/exceptions.py, following current error categorization and handling patterns. Docling-specific errors must inherit from established base exceptions

- **Logging Consistency**: All new components must use existing structured logging infrastructure from src/utils/logger.py, maintaining current log formatting, correlation ID patterns, and observability integration for consistent monitoring
