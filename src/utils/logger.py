"""
Logging configuration and utilities for the document chunking system.

This module provides structured logging with configurable levels and formats,
replacing print statements throughout the codebase.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class StructuredLogger:
    """
    Structured logger with context support for the chunking system.
    """
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers with structured formatting."""
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"chunking_system_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional context."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional context."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional context."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional context."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with optional context."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with structured context."""
        if kwargs:
            context = json.dumps(kwargs, default=str)
            formatted_message = f"{message} | Context: {context}"
        else:
            formatted_message = message
        
        self.logger.log(level, formatted_message)


class ChunkingLogger:
    """
    Specialized logger for chunking operations with performance tracking.
    """
    
    def __init__(self, name: str = "chunking"):
        self.logger = StructuredLogger(name)
        self.operation_start_times: Dict[str, datetime] = {}
    
    def start_operation(self, operation: str, **context):
        """Start tracking an operation."""
        self.operation_start_times[operation] = datetime.now()
        self.logger.info(f"Starting {operation}", operation=operation, **context)
    
    def end_operation(self, operation: str, success: bool = True, **context):
        """End tracking an operation and log duration."""
        if operation in self.operation_start_times:
            duration = datetime.now() - self.operation_start_times[operation]
            duration_ms = duration.total_seconds() * 1000
            
            log_level = "info" if success else "error"
            getattr(self.logger, log_level)(
                f"Completed {operation}",
                operation=operation,
                duration_ms=round(duration_ms, 2),
                success=success,
                **context
            )
            
            del self.operation_start_times[operation]
        else:
            self.logger.warning(f"No start time found for operation: {operation}")
    
    def log_chunk_stats(self, chunks: list, operation: str = "chunking"):
        """Log chunk statistics."""
        if not chunks:
            self.logger.warning("No chunks generated", operation=operation)
            return
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        
        self.logger.info(
            "Chunk statistics",
            operation=operation,
            total_chunks=len(chunks),
            avg_chunk_size=round(avg_size, 2),
            min_chunk_size=min(chunk_sizes),
            max_chunk_size=max(chunk_sizes)
        )
    
    def log_file_processed(self, file_path: str, chunks_count: int, file_size: int):
        """Log file processing completion."""
        self.logger.info(
            "File processed successfully",
            file_path=file_path,
            chunks_generated=chunks_count,
            file_size_bytes=file_size
        )
    
    def log_quality_metrics(self, metrics: Dict[str, Any], file_path: Optional[str] = None):
        """Log quality evaluation metrics."""
        self.logger.info(
            "Quality evaluation completed",
            file_path=file_path,
            overall_score=metrics.get('overall_score', 0),
            total_chunks=metrics.get('total_chunks', 0),
            content_quality=metrics.get('content_quality', {}),
            semantic_coherence=metrics.get('semantic_coherence', {})
        )
    
    def log_error(self, error: Exception, operation: str, **context):
        """Log error with full context."""
        self.logger.error(
            f"Error in {operation}: {str(error)}",
            operation=operation,
            error_type=type(error).__name__,
            **context
        )
    
    def log_batch_progress(self, current: int, total: int, filename: str):
        """Log batch processing progress."""
        progress_pct = (current / total) * 100 if total > 0 else 0
        self.logger.info(
            "Batch processing progress",
            current_file=current,
            total_files=total,
            progress_percent=round(progress_pct, 1),
            current_filename=filename
        )


def setup_logging(level: str = "INFO", log_to_file: bool = True) -> StructuredLogger:
    """
    Setup application-wide logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file in addition to console
    
    Returns:
        StructuredLogger instance for the main application
    """
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[]  # We'll add our own handlers
    )
    
    # Create main application logger
    app_logger = StructuredLogger("chunking_system", level)
    
    return app_logger


def get_logger(name: str, level: str = "INFO") -> StructuredLogger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level
    
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, level)


def get_chunking_logger(name: str = "chunking") -> ChunkingLogger:
    """
    Get a specialized chunking logger.
    
    Args:
        name: Logger name
    
    Returns:
        ChunkingLogger instance
    """
    return ChunkingLogger(name)


# Default logger instances for convenience
default_logger = StructuredLogger("chunking_system")
chunking_logger = ChunkingLogger("chunking")