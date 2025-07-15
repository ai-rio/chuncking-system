"""Security utilities for the document chunking system.

This module provides security features including input validation,
file sanitization, and security auditing capabilities.
"""

import os
import re
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
from dataclasses import dataclass
from datetime import datetime
from src.utils.logger import get_logger
from src.exceptions import ValidationError, FileHandlingError, ChunkingError


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    max_file_size_mb: int = 100
    max_total_size_mb: int = 1000
    allowed_extensions: Set[str] = None
    blocked_extensions: Set[str] = None
    max_path_length: int = 260
    allow_hidden_files: bool = False
    enable_content_validation: bool = True
    max_filename_length: int = 255
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = {
                '.md', '.txt', '.rst', '.org', '.tex',
                '.markdown', '.mdown', '.mkd', '.mdx'
            }
        
        if self.blocked_extensions is None:
            self.blocked_extensions = {
                '.exe', '.bat', '.cmd', '.com', '.scr',
                '.pif', '.vbs', '.js', '.jar', '.app',
                '.deb', '.pkg', '.dmg', '.iso'
            }


class PathSanitizer:
    """Utilities for sanitizing and validating file paths."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.logger = get_logger(__name__)
        
        # Dangerous path patterns
        self.dangerous_patterns = [
            r'\.\.',  # Directory traversal
            r'[<>:"|?*]',  # Invalid filename characters
            r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$',  # Windows reserved names
            r'^\.',  # Hidden files (if not allowed)
        ]
    
    def sanitize_path(self, path: Union[str, Path]) -> Path:
        """
        Sanitize and validate a file path.
        
        Args:
            path: Path to sanitize
            
        Returns:
            Sanitized Path object
            
        Raises:
            ValidationError: If path is invalid or dangerous
        """
        if isinstance(path, str):
            path = Path(path)
        
        # Convert to absolute path and resolve
        try:
            path = path.resolve()
        except (OSError, ValueError) as e:
            raise ValidationError(f"Invalid path: {e}", field="path", value=str(path))
        
        # Check path length
        if len(str(path)) > self.config.max_path_length:
            raise ValidationError(
                f"Path too long: {len(str(path))} > {self.config.max_path_length}",
                field="path",
                value=str(path)
            )
        
        # Check filename length
        if len(path.name) > self.config.max_filename_length:
            raise ValidationError(
                f"Filename too long: {len(path.name)} > {self.config.max_filename_length}",
                field="filename",
                value=path.name
            )
        
        # Check for dangerous patterns
        path_str = str(path)
        for pattern in self.dangerous_patterns:
            if re.search(pattern, path_str, re.IGNORECASE):
                if pattern == r'^\.' and self.config.allow_hidden_files:
                    continue
                raise ValidationError(
                    f"Dangerous path pattern detected: {pattern}",
                    field="path",
                    value=path_str
                )
        
        # Check file extension
        extension = path.suffix.lower()
        
        if extension in self.config.blocked_extensions:
            raise ValidationError(
                f"Blocked file extension: {extension}",
                field="extension",
                value=extension
            )
        
        if self.config.allowed_extensions and extension not in self.config.allowed_extensions:
            raise ValidationError(
                f"File extension not allowed: {extension}",
                field="extension",
                value=extension
            )
        
        return path
    
    def validate_directory_traversal(self, path: Path, base_dir: Path) -> bool:
        """
        Check if path attempts directory traversal outside base directory.
        
        Args:
            path: Path to check
            base_dir: Base directory that should contain the path
            
        Returns:
            True if path is safe, False otherwise
        """
        try:
            path.resolve().relative_to(base_dir.resolve())
            return True
        except ValueError:
            return False


class FileValidator:
    """Utilities for validating file content and properties."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.logger = get_logger(__name__)
    
    def validate_file_size(self, file_path: Path) -> None:
        """
        Validate file size against limits.
        
        Args:
            file_path: Path to file
            
        Raises:
            ValidationError: If file is too large
        """
        if not file_path.exists():
            raise FileHandlingError(f"File does not exist: {file_path}", file_path=str(file_path))
        
        file_size = file_path.stat().st_size
        max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            raise ValidationError(
                f"File too large: {file_size / (1024*1024):.1f}MB > {self.config.max_file_size_mb}MB",
                field="file_size",
                value=file_size
            )
    
    def validate_total_size(self, file_paths: List[Path]) -> None:
        """
        Validate total size of multiple files.
        
        Args:
            file_paths: List of file paths
            
        Raises:
            ValidationError: If total size exceeds limit
        """
        total_size = 0
        for file_path in file_paths:
            if file_path.exists():
                total_size += file_path.stat().st_size
        
        max_total_bytes = self.config.max_total_size_mb * 1024 * 1024
        
        if total_size > max_total_bytes:
            raise ValidationError(
                f"Total file size too large: {total_size / (1024*1024):.1f}MB > {self.config.max_total_size_mb}MB",
                field="total_size",
                value=total_size
            )
    
    def validate_mime_type(self, file_path: Path) -> str:
        """
        Validate and return MIME type of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            MIME type string
            
        Raises:
            ValidationError: If MIME type is not allowed
        """
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        if mime_type is None:
            # Try to detect from content
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(512)
                
                # Simple text detection
                try:
                    header.decode('utf-8')
                    mime_type = 'text/plain'
                except UnicodeDecodeError:
                    raise ValidationError(
                        "Cannot determine file type or file is binary",
                        field="mime_type",
                        value="unknown"
                    )
            except Exception as e:
                raise FileHandlingError(f"Cannot read file: {e}", file_path=str(file_path))
        
        # Check if MIME type is allowed
        allowed_types = {
            'text/plain', 'text/markdown', 'text/x-markdown',
            'text/x-rst', 'application/x-tex', 'text/org'
        }
        
        if mime_type not in allowed_types and not mime_type.startswith('text/'):
            raise ValidationError(
                f"MIME type not allowed: {mime_type}",
                field="mime_type",
                value=mime_type
            )
        
        return mime_type
    
    def validate_content_safety(self, file_path: Path) -> None:
        """
        Validate file content for safety (basic checks).
        
        Args:
            file_path: Path to file
            
        Raises:
            ValidationError: If content appears unsafe
        """
        if not self.config.enable_content_validation:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first 10KB for analysis
                content = f.read(10240)
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r'<script[^>]*>',  # JavaScript
                r'javascript:',     # JavaScript URLs
                r'data:.*base64',   # Base64 data URLs
                r'\\x[0-9a-fA-F]{2}',  # Hex encoded data
                r'eval\s*\(',       # eval() calls
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    self.logger.warning(
                        "Suspicious content pattern detected",
                        pattern=pattern,
                        file_path=str(file_path)
                    )
                    # Note: We log but don't block, as these might be legitimate in documentation
        
        except Exception as e:
            self.logger.warning(f"Content validation failed: {e}", file_path=str(file_path))


class ChecksumValidator:
    """Utilities for file integrity validation using checksums."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """
        Calculate hash of file content.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256', 'sha512')
            
        Returns:
            Hex digest of file hash
        """
        hash_obj = hashlib.new(algorithm)
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_obj.update(chunk)
        except Exception as e:
            raise FileHandlingError(f"Cannot read file for hashing: {e}", file_path=str(file_path))
        
        return hash_obj.hexdigest()
    
    def verify_file_integrity(self, file_path: Path, expected_hash: str, algorithm: str = 'sha256') -> bool:
        """
        Verify file integrity against expected hash.
        
        Args:
            file_path: Path to file
            expected_hash: Expected hash value
            algorithm: Hash algorithm used
            
        Returns:
            True if hash matches, False otherwise
        """
        try:
            actual_hash = self.calculate_file_hash(file_path, algorithm)
            return actual_hash.lower() == expected_hash.lower()
        except Exception as e:
            self.logger.error(f"Hash verification failed: {e}", file_path=str(file_path))
            return False
    
    def create_integrity_manifest(self, file_paths: List[Path], algorithm: str = 'sha256') -> Dict[str, str]:
        """
        Create integrity manifest for multiple files.
        
        Args:
            file_paths: List of file paths
            algorithm: Hash algorithm to use
            
        Returns:
            Dictionary mapping file paths to hashes
        """
        manifest = {}
        
        for file_path in file_paths:
            try:
                file_hash = self.calculate_file_hash(file_path, algorithm)
                manifest[str(file_path)] = file_hash
            except Exception as e:
                self.logger.error(f"Failed to hash file: {e}", file_path=str(file_path))
                manifest[str(file_path)] = None
        
        return manifest


class SecurityAuditor:
    """Security auditing and reporting utilities."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.logger = get_logger(__name__)
        self.path_sanitizer = PathSanitizer(config)
        self.file_validator = FileValidator(config)
        self.checksum_validator = ChecksumValidator()
    
    def audit_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Perform comprehensive security audit of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Audit report dictionary
        """
        file_path = Path(file_path)
        audit_report = {
            'file_path': str(file_path),
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'warnings': [],
            'errors': [],
            'overall_status': 'unknown'
        }
        
        try:
            # Path sanitization check
            try:
                sanitized_path = self.path_sanitizer.sanitize_path(file_path)
                audit_report['checks']['path_sanitization'] = 'passed'
                audit_report['sanitized_path'] = str(sanitized_path)
            except ValidationError as e:
                audit_report['checks']['path_sanitization'] = 'failed'
                audit_report['errors'].append(f"Path validation: {e}")
            
            # File existence check
            if file_path.exists():
                audit_report['checks']['file_exists'] = 'passed'
                
                # File size validation
                try:
                    self.file_validator.validate_file_size(file_path)
                    audit_report['checks']['file_size'] = 'passed'
                except ValidationError as e:
                    audit_report['checks']['file_size'] = 'failed'
                    audit_report['errors'].append(f"File size: {e}")
                
                # MIME type validation
                try:
                    mime_type = self.file_validator.validate_mime_type(file_path)
                    audit_report['checks']['mime_type'] = 'passed'
                    audit_report['mime_type'] = mime_type
                except ValidationError as e:
                    audit_report['checks']['mime_type'] = 'failed'
                    audit_report['errors'].append(f"MIME type: {e}")
                
                # Content safety check
                try:
                    self.file_validator.validate_content_safety(file_path)
                    audit_report['checks']['content_safety'] = 'passed'
                except ValidationError as e:
                    audit_report['checks']['content_safety'] = 'warning'
                    audit_report['warnings'].append(f"Content safety: {e}")
                
                # File integrity (hash)
                try:
                    file_hash = self.checksum_validator.calculate_file_hash(file_path)
                    audit_report['checks']['integrity'] = 'passed'
                    audit_report['file_hash'] = file_hash
                except Exception as e:
                    audit_report['checks']['integrity'] = 'failed'
                    audit_report['errors'].append(f"Integrity check: {e}")
                
            else:
                audit_report['checks']['file_exists'] = 'failed'
                audit_report['errors'].append("File does not exist")
            
            # Determine overall status
            if audit_report['errors']:
                audit_report['overall_status'] = 'failed'
            elif audit_report['warnings']:
                audit_report['overall_status'] = 'warning'
            else:
                audit_report['overall_status'] = 'passed'
        
        except Exception as e:
            audit_report['overall_status'] = 'error'
            audit_report['errors'].append(f"Audit error: {e}")
            self.logger.error(f"Security audit failed: {e}", file_path=str(file_path))
        
        return audit_report
    
    def audit_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> Dict[str, Any]:
        """
        Perform security audit of all files in a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to audit subdirectories
            
        Returns:
            Comprehensive audit report
        """
        directory_path = Path(directory_path)
        
        audit_report = {
            'directory_path': str(directory_path),
            'timestamp': datetime.now().isoformat(),
            'recursive': recursive,
            'file_audits': {},
            'summary': {
                'total_files': 0,
                'passed': 0,
                'warnings': 0,
                'failed': 0,
                'errors': 0
            }
        }
        
        try:
            # Get all files
            if recursive:
                files = list(directory_path.rglob('*'))
            else:
                files = list(directory_path.glob('*'))
            
            files = [f for f in files if f.is_file()]
            audit_report['summary']['total_files'] = len(files)
            
            # Audit each file
            for file_path in files:
                file_audit = self.audit_file(file_path)
                audit_report['file_audits'][str(file_path)] = file_audit
                
                # Update summary
                status = file_audit['overall_status']
                if status == 'passed':
                    audit_report['summary']['passed'] += 1
                elif status == 'warning':
                    audit_report['summary']['warnings'] += 1
                elif status == 'failed':
                    audit_report['summary']['failed'] += 1
                else:
                    audit_report['summary']['errors'] += 1
        
        except Exception as e:
            self.logger.error(f"Directory audit failed: {e}", directory_path=str(directory_path))
            audit_report['error'] = str(e)
        
        return audit_report


# Global security instances
default_security_config = SecurityConfig()
default_path_sanitizer = PathSanitizer()
default_file_validator = FileValidator()
default_checksum_validator = ChecksumValidator()
default_security_auditor = SecurityAuditor()


def secure_path(path: Union[str, Path]) -> Path:
    """Convenience function for path sanitization."""
    return default_path_sanitizer.sanitize_path(path)


def validate_file(file_path: Union[str, Path]) -> None:
    """Convenience function for file validation."""
    file_path = Path(file_path)
    default_file_validator.validate_file_size(file_path)
    default_file_validator.validate_mime_type(file_path)
    default_file_validator.validate_content_safety(file_path)


def audit_file_security(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Convenience function for file security audit."""
    return default_security_auditor.audit_file(file_path)