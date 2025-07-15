# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.6.1

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set work directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser README.md ./

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/cache \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Install the application
RUN python -m pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import src.chunking_system; print('OK')" || exit 1

# Expose port (if running as web service)
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.chunking_system"]

# Development stage
FROM builder as development

# Install all dependencies including dev dependencies
RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# Install additional development tools
RUN pip install \
    pytest-xdist \
    pytest-benchmark \
    pre-commit

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY pyproject.toml poetry.lock README.md ./
COPY .pre-commit-config.yaml ./

# Install the application in development mode
RUN poetry install

# Set up pre-commit hooks
RUN git init . && pre-commit install || true

# Default command for development
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Testing stage
FROM development as testing

# Run tests and generate coverage report
RUN python -m pytest tests/ \
    --cov=src \
    --cov-report=html \
    --cov-report=xml \
    --cov-report=term-missing \
    --junitxml=test-results.xml

# Security scanning stage
FROM development as security

# Install security tools
RUN pip install \
    bandit \
    safety \
    pip-audit

# Run security scans
RUN bandit -r src/ -f json -o bandit-report.json || true
RUN safety check --json --output safety-report.json || true
RUN pip-audit --format=json --output=pip-audit-report.json || true