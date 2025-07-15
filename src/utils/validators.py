"""
Input validation utilities for the document chunking system.

This module provides decorators and functions for validating inputs
to ensure data integrity and provide clear error messages.
"""

import os
import functools
from pathlib import Path
from typing import Any, Callable, List, Dict, Union, Optional
from src.exceptions import ValidationError, FileHandlingError


def validate_file_path(file_path: str, must_exist: bool = True, extensions: Optional[List[str]] = None) -> Path:
    """
    Validate a file path and return a Path object.
    
    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist
        extensions: List of allowed file extensions (e.g., ['.md', '.txt'])
    
    Returns:
        Path object
        
    Raises:
        ValidationError: If validation fails
        FileHandlingError: If file doesn't exist or is inaccessible
    """
    if not isinstance(file_path, (str, Path)):
        raise ValidationError(
            "File path must be a string or Path object",
            field="file_path",
            value=type(file_path)
        )
    
    path = Path(file_path)
    
    # Check if path is absolute or make it absolute
    if not path.is_absolute():
        path = path.resolve()
    
    if must_exist:
        if not path.exists():
            raise FileHandlingError(
                "File does not exist",
                file_path=str(path),
                operation="validate_file_path"
            )
        
        if not path.is_file():
            raise FileHandlingError(
                "Path is not a file",
                file_path=str(path),
                operation="validate_file_path"
            )
        
        # Check file is readable
        try:
            with open(path, 'r', encoding='utf-8') as f:
                f.read(1)  # Try to read one character
        except (PermissionError, UnicodeDecodeError) as e:
            raise FileHandlingError(
                f"File is not readable: {str(e)}",
                file_path=str(path),
                operation="validate_file_path"
            ) from e
    
    # Check file extension if specified
    if extensions:
        if path.suffix.lower() not in [ext.lower() for ext in extensions]:
            raise ValidationError(
                f"File must have one of these extensions: {extensions}",
                field="file_path",
                value=path.suffix
            )
    
    return path


def validate_directory_path(dir_path: str, must_exist: bool = True, create_if_missing: bool = False) -> Path:
    """
    Validate a directory path and return a Path object.
    
    Args:
        dir_path: Directory path to validate
        must_exist: Whether the directory must exist
        create_if_missing: Whether to create the directory if it doesn't exist
    
    Returns:
        Path object
        
    Raises:
        ValidationError: If validation fails
        FileHandlingError: If directory operations fail
    """
    if not isinstance(dir_path, (str, Path)):
        raise ValidationError(
            "Directory path must be a string or Path object",
            field="dir_path",
            value=type(dir_path)
        )
    
    path = Path(dir_path)
    
    # Make path absolute
    if not path.is_absolute():
        path = path.resolve()
    
    if must_exist:
        if not path.exists():
            if create_if_missing:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise FileHandlingError(
                        f"Failed to create directory: {str(e)}",
                        file_path=str(path),
                        operation="create_directory"
                    ) from e
            else:
                raise FileHandlingError(
                    "Directory does not exist",
                    file_path=str(path),
                    operation="validate_directory_path"
                )
        
        if path.exists() and not path.is_dir():
            raise FileHandlingError(
                "Path is not a directory",
                file_path=str(path),
                operation="validate_directory_path"
            )
    
    return path


def validate_chunk_size(chunk_size: int, min_size: int = 50, max_size: int = 8000) -> int:
    """
    Validate chunk size parameter.
    
    Args:
        chunk_size: Chunk size to validate
        min_size: Minimum allowed chunk size
        max_size: Maximum allowed chunk size
    
    Returns:
        Validated chunk size
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(chunk_size, int):
        raise ValidationError(
            "Chunk size must be an integer",
            field="chunk_size",
            value=type(chunk_size)
        )
    
    if chunk_size < min_size:
        raise ValidationError(
            f"Chunk size must be at least {min_size}",
            field="chunk_size",
            value=chunk_size
        )
    
    if chunk_size > max_size:
        raise ValidationError(
            f"Chunk size must be at most {max_size}",
            field="chunk_size",
            value=chunk_size
        )
    
    return chunk_size


def validate_chunk_overlap(chunk_overlap: int, chunk_size: int) -> int:
    """
    Validate chunk overlap parameter.
    
    Args:
        chunk_overlap: Chunk overlap to validate
        chunk_size: Chunk size for comparison
    
    Returns:
        Validated chunk overlap
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(chunk_overlap, int):
        raise ValidationError(
            "Chunk overlap must be an integer",
            field="chunk_overlap",
            value=type(chunk_overlap)
        )
    
    if chunk_overlap < 0:
        raise ValidationError(
            "Chunk overlap must be non-negative",
            field="chunk_overlap",
            value=chunk_overlap
        )
    
    if chunk_overlap >= chunk_size:
        raise ValidationError(
            "Chunk overlap must be smaller than chunk size",
            field="chunk_overlap",
            value=chunk_overlap
        )
    
    return chunk_overlap


def validate_output_format(output_format: str) -> str:
    """
    Validate output format parameter.
    
    Args:
        output_format: Output format to validate
    
    Returns:
        Validated output format
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(output_format, str):
        raise ValidationError(
            "Output format must be a string",
            field="output_format",
            value=type(output_format)
        )
    
    valid_formats = ['json', 'csv', 'pickle']
    if output_format.lower() not in valid_formats:
        raise ValidationError(
            f"Output format must be one of: {valid_formats}",
            field="output_format",
            value=output_format
        )
    
    return output_format.lower()


def validate_content(content: str, min_length: int = 1, max_length: int = 50_000_000) -> str:
    """
    Validate content string.
    
    Args:
        content: Content to validate
        min_length: Minimum content length
        max_length: Maximum content length
    
    Returns:
        Validated content
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(content, str):
        raise ValidationError(
            "Content must be a string",
            field="content",
            value=type(content)
        )
    
    if len(content) < min_length:
        raise ValidationError(
            f"Content must be at least {min_length} characters",
            field="content",
            value=len(content)
        )
    
    if len(content) > max_length:
        raise ValidationError(
            f"Content must be at most {max_length} characters",
            field="content",
            value=len(content)
        )
    
    return content


def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate metadata dictionary.
    
    Args:
        metadata: Metadata to validate
    
    Returns:
        Validated metadata
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(metadata, dict):
        raise ValidationError(
            "Metadata must be a dictionary",
            field="metadata",
            value=type(metadata)
        )
    
    # Check for prohibited keys that might cause issues
    prohibited_keys = ['__class__', '__module__', '__dict__']
    for key in metadata.keys():
        if key in prohibited_keys:
            raise ValidationError(
                f"Metadata key '{key}' is prohibited",
                field="metadata",
                value=key
            )
        
        if not isinstance(key, str):
            raise ValidationError(
                "Metadata keys must be strings",
                field="metadata",
                value=type(key)
            )
    
    return metadata


# Validation decorators for functions
def validate_inputs(**validators):
    """
    Decorator to validate function inputs.
    
    Usage:
        @validate_inputs(
            file_path=lambda x: validate_file_path(x, extensions=['.md']),
            chunk_size=lambda x: validate_chunk_size(x)
        )
        def process_file(file_path, chunk_size):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each specified parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        validated_value = validator(value)
                        bound_args.arguments[param_name] = validated_value
                    except (ValidationError, FileHandlingError) as e:
                        # Add function context to the error
                        e.details = e.details or {}
                        e.details['function'] = func.__name__
                        e.details['parameter'] = param_name
                        raise
            
            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator


def safe_path_join(*paths: Union[str, Path]) -> Path:
    """
    Safely join path components and prevent directory traversal attacks.
    
    Args:
        *paths: Path components to join
    
    Returns:
        Safely joined Path object
        
    Raises:
        ValidationError: If path traversal is detected
    """
    if not paths:
        raise ValidationError("At least one path component must be provided")
    
    # Convert all to Path objects
    path_parts = [Path(p) for p in paths]
    
    # Check for directory traversal attempts
    for part in path_parts[1:]:  # Skip the first (base) path
        if part.is_absolute():
            raise ValidationError(
                "Absolute paths not allowed in path components",
                field="path_component",
                value=str(part)
            )
        
        if '..' in part.parts:
            raise ValidationError(
                "Parent directory references not allowed",
                field="path_component", 
                value=str(part)
            )
    
    # Join paths
    result = path_parts[0]
    for part in path_parts[1:]:
        result = result / part
    
    return result


def validate_file_size(file_path: Path, max_size_mb: int = 100) -> int:
    """
    Validate file size and return size in bytes.
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum size in MB
    
    Returns:
        File size in bytes
        
    Raises:
        ValidationError: If file is too large
        FileHandlingError: If file cannot be accessed
    """
    try:
        size_bytes = file_path.stat().st_size
    except Exception as e:
        raise FileHandlingError(
            f"Cannot access file: {str(e)}",
            file_path=str(file_path),
            operation="get_file_size"
        ) from e
    
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if size_bytes > max_size_bytes:
        raise ValidationError(
            f"File size ({size_bytes / 1024 / 1024:.1f} MB) exceeds maximum allowed size ({max_size_mb} MB)",
            field="file_size",
            value=size_bytes
        )
    
    return size_bytes