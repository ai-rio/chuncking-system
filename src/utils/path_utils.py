"""
Path handling utilities for the document chunking system.

This module provides consistent, secure path operations with proper
validation and cross-platform compatibility.
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Generator
from src.exceptions import FileHandlingError, ValidationError
from src.utils.validators import validate_file_path, validate_directory_path, safe_path_join


class PathManager:
    """
    Centralized path management with security and validation.
    """
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize PathManager with optional base directory.
        
        Args:
            base_dir: Base directory for relative path operations
        """
        if base_dir:
            self.base_dir = validate_directory_path(base_dir, must_exist=True)
        else:
            self.base_dir = Path.cwd()
    
    def resolve_path(self, path: Union[str, Path], relative_to_base: bool = False) -> Path:
        """
        Resolve a path to absolute form with validation.
        
        Args:
            path: Path to resolve
            relative_to_base: Whether to resolve relative to base_dir
        
        Returns:
            Resolved absolute Path
        """
        if isinstance(path, str):
            path = Path(path)
        
        if relative_to_base and not path.is_absolute():
            path = self.base_dir / path
        
        return path.resolve()
    
    def create_output_structure(self, base_output_dir: Union[str, Path]) -> dict:
        """
        Create standard output directory structure.
        
        Args:
            base_output_dir: Base output directory
        
        Returns:
            Dictionary with created directory paths
        """
        base_path = validate_directory_path(base_output_dir, must_exist=False, create_if_missing=True)
        
        directories = {
            'base': base_path,
            'chunks': base_path / 'chunks',
            'reports': base_path / 'reports',
            'logs': base_path / 'logs',
            'temp': base_path / 'temp'
        }
        
        # Create all directories
        for name, dir_path in directories.items():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise FileHandlingError(
                    f"Failed to create {name} directory",
                    file_path=str(dir_path),
                    operation="create_directory"
                ) from e
        
        return directories
    
    def find_files(
        self,
        directory: Union[str, Path],
        patterns: List[str],
        recursive: bool = True,
        case_sensitive: bool = False
    ) -> List[Path]:
        """
        Find files matching patterns in directory.
        
        Args:
            directory: Directory to search
            patterns: List of glob patterns (e.g., ['*.md', '*.txt'])
            recursive: Whether to search recursively
            case_sensitive: Whether pattern matching is case sensitive
        
        Returns:
            List of matching file paths
        """
        dir_path = validate_directory_path(directory, must_exist=True)
        
        found_files = []
        
        for pattern in patterns:
            if recursive:
                glob_pattern = f"**/{pattern}"
                matches = dir_path.rglob(pattern)
            else:
                matches = dir_path.glob(pattern)
            
            for match in matches:
                if match.is_file():
                    if not case_sensitive:
                        # Check if file already exists with different case
                        normalized_path = str(match).lower()
                        if not any(str(f).lower() == normalized_path for f in found_files):
                            found_files.append(match)
                    else:
                        found_files.append(match)
        
        return sorted(found_files)
    
    def get_safe_filename(self, filename: str, max_length: int = 255) -> str:
        """
        Create a safe filename by removing/replacing problematic characters.
        
        Args:
            filename: Original filename
            max_length: Maximum filename length
        
        Returns:
            Safe filename
        """
        # Remove or replace problematic characters
        safe_chars = []
        for char in filename:
            if char.isalnum() or char in '-_.()[]{}':
                safe_chars.append(char)
            elif char in ' \t':
                safe_chars.append('_')
            # Skip other characters
        
        safe_filename = ''.join(safe_chars)
        
        # Ensure filename isn't empty
        if not safe_filename:
            safe_filename = 'untitled'
        
        # Truncate if too long
        if len(safe_filename) > max_length:
            # Try to preserve extension
            parts = safe_filename.rsplit('.', 1)
            if len(parts) == 2 and len(parts[1]) <= 10:  # Reasonable extension length
                base, ext = parts
                max_base_length = max_length - len(ext) - 1
                safe_filename = base[:max_base_length] + '.' + ext
            else:
                safe_filename = safe_filename[:max_length]
        
        return safe_filename
    
    def generate_unique_path(self, base_path: Union[str, Path], suffix: str = '') -> Path:
        """
        Generate a unique file path by adding a counter if needed.
        
        Args:
            base_path: Base file path
            suffix: Optional suffix to add before extension
        
        Returns:
            Unique file path
        """
        path = Path(base_path)
        
        if suffix:
            # Insert suffix before extension
            stem = path.stem + suffix
            path = path.parent / (stem + path.suffix)
        
        if not path.exists():
            return path
        
        # Add counter to make unique
        counter = 1
        while True:
            new_stem = f"{path.stem}_{counter}"
            new_path = path.parent / (new_stem + path.suffix)
            
            if not new_path.exists():
                return new_path
            
            counter += 1
            
            # Prevent infinite loop
            if counter > 9999:
                raise FileHandlingError(
                    "Cannot generate unique filename after 9999 attempts",
                    file_path=str(base_path),
                    operation="generate_unique_path"
                )
    
    def calculate_relative_path(self, target_path: Union[str, Path], base_path: Union[str, Path]) -> Path:
        """
        Calculate relative path from base to target.
        
        Args:
            target_path: Target path
            base_path: Base path
        
        Returns:
            Relative path from base to target
        """
        target = Path(target_path).resolve()
        base = Path(base_path).resolve()
        
        try:
            return target.relative_to(base)
        except ValueError:
            # Paths are not relative to each other
            return target
    
    def ensure_parent_directory(self, file_path: Union[str, Path]) -> Path:
        """
        Ensure the parent directory of a file path exists.
        
        Args:
            file_path: File path
        
        Returns:
            Validated file path
        """
        path = Path(file_path)
        parent = path.parent
        
        if not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise FileHandlingError(
                    f"Failed to create parent directory: {str(e)}",
                    file_path=str(parent),
                    operation="create_parent_directory"
                ) from e
        
        return path
    
    def get_file_info(self, file_path: Union[str, Path]) -> dict:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to file
        
        Returns:
            Dictionary with file information
        """
        path = validate_file_path(file_path, must_exist=True)
        
        try:
            stat = path.stat()
            
            return {
                'path': str(path),
                'name': path.name,
                'stem': path.stem,
                'suffix': path.suffix,
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'modified_timestamp': stat.st_mtime,
                'is_readable': os.access(path, os.R_OK),
                'is_writable': os.access(path, os.W_OK),
                'parent_directory': str(path.parent)
            }
        except Exception as e:
            raise FileHandlingError(
                f"Failed to get file information: {str(e)}",
                file_path=str(path),
                operation="get_file_info"
            ) from e
    
    def cleanup_empty_directories(self, directory: Union[str, Path], keep_root: bool = True) -> int:
        """
        Remove empty directories recursively.
        
        Args:
            directory: Directory to clean up
            keep_root: Whether to keep the root directory even if empty
        
        Returns:
            Number of directories removed
        """
        dir_path = validate_directory_path(directory, must_exist=True)
        removed_count = 0
        
        # Walk directory tree bottom-up
        for current_dir in reversed(list(dir_path.rglob('*'))):
            if current_dir.is_dir() and not any(current_dir.iterdir()):
                # Directory is empty
                if not keep_root or current_dir != dir_path:
                    try:
                        current_dir.rmdir()
                        removed_count += 1
                    except Exception:
                        # Ignore errors (directory might not be empty anymore)
                        pass
        
        return removed_count


class MarkdownFileManager(PathManager):
    """
    Specialized path manager for Markdown files.
    """
    
    MARKDOWN_EXTENSIONS = ['.md', '.markdown', '.mdown', '.mkd', '.mdx']
    
    def find_markdown_files(self, directory: Union[str, Path], recursive: bool = True) -> List[Path]:
        """
        Find all Markdown files in directory.
        
        Args:
            directory: Directory to search
            recursive: Whether to search recursively
        
        Returns:
            List of Markdown file paths
        """
        patterns = [f"*{ext}" for ext in self.MARKDOWN_EXTENSIONS]
        return self.find_files(directory, patterns, recursive=recursive)
    
    def is_markdown_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if file is a Markdown file based on extension.
        
        Args:
            file_path: Path to check
        
        Returns:
            True if file is Markdown
        """
        path = Path(file_path)
        return path.suffix.lower() in self.MARKDOWN_EXTENSIONS
    
    def create_markdown_output_paths(self, input_file: Union[str, Path], output_dir: Union[str, Path]) -> dict:
        """
        Create output paths for Markdown processing.
        
        Args:
            input_file: Input Markdown file
            output_dir: Output directory
        
        Returns:
            Dictionary with output paths
        """
        input_path = validate_file_path(input_file, must_exist=True, extensions=self.MARKDOWN_EXTENSIONS)
        output_structure = self.create_output_structure(output_dir)
        
        base_name = input_path.stem
        safe_name = self.get_safe_filename(base_name)
        
        return {
            **output_structure,
            'chunks_json': output_structure['chunks'] / f"{safe_name}_chunks.json",
            'chunks_csv': output_structure['chunks'] / f"{safe_name}_chunks.csv",
            'chunks_pickle': output_structure['chunks'] / f"{safe_name}_chunks.pickle",
            'quality_report': output_structure['reports'] / f"{safe_name}_quality_report.md",
            'processing_log': output_structure['logs'] / f"{safe_name}_processing.log"
        }


# Convenience functions
def get_path_manager(base_dir: Optional[Union[str, Path]] = None) -> PathManager:
    """Get a PathManager instance."""
    return PathManager(base_dir)


def get_markdown_manager(base_dir: Optional[Union[str, Path]] = None) -> MarkdownFileManager:
    """Get a MarkdownFileManager instance."""
    return MarkdownFileManager(base_dir)