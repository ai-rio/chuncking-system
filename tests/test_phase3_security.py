"""Tests for Phase 3 security implementation."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.utils.security import (
    SecurityConfig, PathSanitizer, FileValidator, 
    ChecksumValidator, SecurityAuditor
)
from src.config.settings import ChunkingConfig
from src.exceptions import ValidationError, SecurityError


class TestSecurityConfig:
    """Test SecurityConfig dataclass."""
    
    def test_default_config(self):
        """Test default security configuration."""
        config = SecurityConfig()
        
        assert config.max_file_size_mb == 100  # 100MB
        assert config.allowed_extensions == {'.md', '.txt', '.rst', '.org', '.tex', '.markdown', '.mdown', '.mkd', '.mdx'}
        assert config.blocked_extensions == {'.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.js', '.jar', '.app', '.deb', '.pkg', '.dmg', '.iso'}
        assert config.max_path_length == 260  # Default from SecurityConfig
        assert config.allow_hidden_files is False
        assert config.enable_content_validation is True
    
    def test_custom_config(self):
        """Test custom security configuration."""
        config = SecurityConfig(
            max_file_size_mb=50,  # 50MB
            allowed_extensions={'.md', '.txt'},
            blocked_paths={'/etc', '/tmp'},
            enable_checksum_validation=False,
            enable_content_scanning=False,
            max_path_length=2048
        )
        
        assert config.max_file_size_mb == 50
        assert config.allowed_extensions == {'.md', '.txt'}
        assert config.blocked_paths == {'/etc', '/tmp'}
        assert config.enable_checksum_validation is False
        assert config.enable_content_scanning is False
        assert config.max_path_length == 2048


class TestPathSanitizer:
    """Test PathSanitizer implementation."""
    
    def test_valid_path_sanitization(self):
        """Test sanitization of valid paths."""
        config = SecurityConfig()
        sanitizer = PathSanitizer(config)
        
        # Test normal path
        path = "/home/user/documents/test.md"
        sanitized = sanitizer.sanitize_path(path)
        assert sanitized == Path(path).resolve()
    
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        config = SecurityConfig()
        sanitizer = PathSanitizer(config)
        
        # Test path traversal attempts
        dangerous_paths = [
            "../../../etc/passwd",
            "/home/user/../../../etc/passwd",
            "documents/../../etc/passwd",
            "test.md/../../../etc/passwd"
        ]
        
        for dangerous_path in dangerous_paths:
            with pytest.raises(ValidationError, match="Path traversal detected"):
                sanitizer.sanitize_path(dangerous_path)
    
    def test_blocked_path_prevention(self):
        """Test prevention of access to blocked paths."""
        config = SecurityConfig(blocked_paths={'/etc', '/proc', '/sys'})
        sanitizer = PathSanitizer(config)
        
        blocked_paths = [
            "/etc/passwd",
            "/proc/version",
            "/sys/kernel/version",
            "/etc/shadow"
        ]
        
        for blocked_path in blocked_paths:
            with pytest.raises(ValidationError, match="Access to blocked path"):
                sanitizer.sanitize_path(blocked_path)
    
    def test_path_length_validation(self):
        """Test path length validation."""
        config = SecurityConfig(max_path_length=50)
        sanitizer = PathSanitizer(config)
        
        # Test path that's too long
        long_path = "/" + "a" * 100 + "/test.md"
        
        with pytest.raises(ValidationError, match="Path too long"):
            sanitizer.sanitize_path(long_path)
    
    def test_valid_path_validation(self):
        """Test validation of valid paths."""
        config = SecurityConfig()
        sanitizer = PathSanitizer(config)
        
        valid_paths = [
            "/home/user/documents/test.md",
            "./documents/test.txt",
            "documents/subdoc/test.rst",
            "/tmp/safe/test.markdown"
        ]
        
        for valid_path in valid_paths:
            # Should not raise exception
            result = sanitizer.is_safe_path(valid_path)
            assert result is True
    
    def test_unsafe_path_validation(self):
        """Test validation of unsafe paths."""
        config = SecurityConfig(blocked_paths={'/etc', '/proc'})
        sanitizer = PathSanitizer(config)
        
        unsafe_paths = [
            "/etc/passwd",
            "../../../etc/passwd",
            "/proc/version"
        ]
        
        for unsafe_path in unsafe_paths:
            result = sanitizer.is_safe_path(unsafe_path)
            assert result is False


class TestFileValidator:
    """Test FileValidator implementation."""
    
    def test_file_size_validation(self, tmp_path):
        """Test file size validation."""
        config = SecurityConfig(max_file_size_mb=0.001)  # 1KB limit (1024 bytes = 0.001 MB)
        validator = FileValidator(config)
        
        # Create small file (should pass)
        small_file = tmp_path / "small.md"
        small_file.write_text("Small content")
        
        validator.validate_file_size(small_file)
        
        # Create large file (should fail)
        large_file = tmp_path / "large.md"
        large_file.write_text("x" * 2048)  # 2KB
        
        with pytest.raises(ValidationError, match="File size exceeds limit"):
            validator.validate_file_size(large_file)
    
    def test_file_extension_validation(self, tmp_path):
        """Test file extension validation."""
        config = SecurityConfig(allowed_extensions={'.md', '.txt'})
        validator = FileValidator(config)
        
        # Valid extensions
        valid_files = [
            tmp_path / "test.md",
            tmp_path / "test.txt"
        ]
        
        for file_path in valid_files:
            file_path.write_text("content")
            validator.validate_file_extension(file_path)
        
        # Invalid extensions
        invalid_files = [
            tmp_path / "test.exe",
            tmp_path / "test.py",
            tmp_path / "test.sh"
        ]
        
        for file_path in invalid_files:
            file_path.write_text("content")
            with pytest.raises(ValidationError, match="File extension not allowed"):
                validator.validate_file_extension(file_path)
    
    @patch('magic.from_file')
    def test_mime_type_validation(self, mock_magic, tmp_path):
        """Test MIME type validation."""
        config = SecurityConfig()
        validator = FileValidator(config)
        
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test content")
        
        # Test valid MIME type
        mock_magic.return_value = "text/plain"
        validator.validate_mime_type(test_file)  # Should not raise
        
        # Test invalid MIME type
        mock_magic.return_value = "application/x-executable"
        with pytest.raises(ValidationError, match="MIME type not allowed"):
            validator.validate_mime_type(test_file)
    
    def test_content_safety_validation(self, tmp_path):
        """Test content safety validation."""
        config = SecurityConfig()
        validator = FileValidator(config)
        
        # Safe content
        safe_file = tmp_path / "safe.md"
        safe_file.write_text("# Safe Document\n\nThis is safe content.")
        
        validator.validate_content_safety(safe_file)  # Should not raise
        
        # Potentially unsafe content
        unsafe_file = tmp_path / "unsafe.md"
        unsafe_content = "<script>alert('xss')</script>\n" + "\x00\x01\x02"  # Binary data
        unsafe_file.write_bytes(unsafe_content.encode('utf-8', errors='ignore'))
        
        with pytest.raises(ValidationError, match="Potentially unsafe content detected"):
            validator.validate_content_safety(unsafe_file)
    
    def test_comprehensive_file_validation(self, tmp_path):
        """Test comprehensive file validation."""
        config = SecurityConfig(
            max_file_size=1024,
            allowed_extensions={'.md'},
            enable_content_scanning=True
        )
        validator = FileValidator(config)
        
        # Valid file
        valid_file = tmp_path / "valid.md"
        valid_file.write_text("# Valid Document\n\nValid content.")
        
        with patch('magic.from_file', return_value="text/plain"):
            validator.validate_file(valid_file)  # Should not raise
        
        # Invalid file (wrong extension)
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("Content")
        
        with pytest.raises(ValidationError):
            validator.validate_file(invalid_file)


class TestChecksumValidator:
    """Test ChecksumValidator implementation."""
    
    def test_checksum_calculation(self, tmp_path):
        """Test checksum calculation."""
        validator = ChecksumValidator()
        
        test_file = tmp_path / "test.md"
        test_content = "# Test Document\n\nTest content for checksum."
        test_file.write_text(test_content)
        
        checksum = validator.calculate_checksum(test_file)
        
        # Verify checksum is a valid SHA-256 hash
        assert len(checksum) == 64
        assert all(c in '0123456789abcdef' for c in checksum)
        
        # Verify consistency
        checksum2 = validator.calculate_checksum(test_file)
        assert checksum == checksum2
    
    def test_checksum_validation(self, tmp_path):
        """Test checksum validation."""
        validator = ChecksumValidator()
        
        test_file = tmp_path / "test.md"
        test_content = "# Test Document\n\nTest content."
        test_file.write_text(test_content)
        
        # Calculate expected checksum
        expected_checksum = validator.calculate_checksum(test_file)
        
        # Validate with correct checksum
        validator.validate_checksum(test_file, expected_checksum)  # Should not raise
        
        # Validate with incorrect checksum
        wrong_checksum = "0" * 64
        with pytest.raises(ValidationError, match="File checksum validation failed"):
            validator.validate_checksum(test_file, wrong_checksum)
    
    def test_checksum_with_file_modification(self, tmp_path):
        """Test checksum changes when file is modified."""
        validator = ChecksumValidator()
        
        test_file = tmp_path / "test.md"
        
        # Original content
        test_file.write_text("Original content")
        checksum1 = validator.calculate_checksum(test_file)
        
        # Modified content
        test_file.write_text("Modified content")
        checksum2 = validator.calculate_checksum(test_file)
        
        assert checksum1 != checksum2


class TestSecurityAuditor:
    """Test SecurityAuditor implementation."""
    
    def test_file_audit_success(self, tmp_path):
        """Test successful file audit."""
        config = SecurityConfig()
        auditor = SecurityAuditor(config)
        
        # Create valid file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Document\n\nValid content.")
        
        with patch('magic.from_file', return_value="text/plain"):
            audit_result = auditor.audit_file(test_file)
        
        assert audit_result['file_path'] == str(test_file)
        assert audit_result['is_safe'] is True
        assert audit_result['issues'] == []
        assert 'file_size' in audit_result
        assert 'checksum' in audit_result
    
    def test_file_audit_with_issues(self, tmp_path):
        """Test file audit with security issues."""
        config = SecurityConfig(
            max_file_size_mb=0.00001,  # Very small limit (10 bytes = 0.00001 MB)
            allowed_extensions={'.txt'}  # Only .txt allowed
        )
        auditor = SecurityAuditor(config)
        
        # Create problematic file
        test_file = tmp_path / "test.md"  # Wrong extension
        test_file.write_text("This content is too long for the limit")  # Too large
        
        audit_result = auditor.audit_file(test_file)
        
        assert audit_result['is_safe'] is False
        assert len(audit_result['issues']) >= 2  # Size and extension issues
        
        # Check for specific issues
        issues_text = ' '.join(audit_result['issues'])
        assert 'File size exceeds limit' in issues_text
        assert 'File extension not allowed' in issues_text
    
    def test_directory_audit(self, tmp_path):
        """Test directory audit functionality."""
        config = SecurityConfig()
        auditor = SecurityAuditor(config)
        
        # Create test files
        (tmp_path / "valid.md").write_text("# Valid\n\nContent")
        (tmp_path / "invalid.exe").write_text("Invalid content")
        
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.md").write_text("# Nested\n\nContent")
        
        with patch('magic.from_file', return_value="text/plain"):
            audit_results = auditor.audit_directory(tmp_path)
        
        assert len(audit_results) == 3  # 3 files total
        
        # Check that we have both safe and unsafe files
        safe_files = [r for r in audit_results if r['is_safe']]
        unsafe_files = [r for r in audit_results if not r['is_safe']]
        
        assert len(safe_files) >= 1  # At least the .md files
        assert len(unsafe_files) >= 1  # The .exe file
    
    def test_audit_summary(self, tmp_path):
        """Test audit summary generation."""
        config = SecurityConfig()
        auditor = SecurityAuditor(config)
        
        # Create mixed files
        (tmp_path / "safe1.md").write_text("Safe content 1")
        (tmp_path / "safe2.md").write_text("Safe content 2")
        (tmp_path / "unsafe.exe").write_text("Unsafe content")
        
        with patch('magic.from_file', return_value="text/plain"):
            audit_results = auditor.audit_directory(tmp_path)
        
        summary = auditor.generate_audit_summary(audit_results)
        
        assert summary['total_files'] == 3
        assert summary['safe_files'] >= 2
        assert summary['unsafe_files'] >= 1
        assert summary['total_issues'] >= 1
        assert 'common_issues' in summary


class TestDocumentChunkerSecurity:
    """Test security integration in DocumentChunker."""
    
    def test_chunker_with_security_enabled(self, tmp_path):
        """Test DocumentChunker with security enabled."""
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            enable_caching=False,
            enable_security=True,
            enable_monitoring=False
        )
        
        chunker = DocumentChunker(config)
        
        # Create valid file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Document\n\nThis is a safe test document.")
        
        with patch('magic.from_file', return_value="text/plain"):
            result = chunker.chunk_file(test_file)
        
        assert result.success is True
        assert len(result.chunks) > 0
        assert result.security_audit is not None
        assert result.security_audit['is_safe'] is True
    
    def test_chunker_security_validation_failure(self, tmp_path):
        """Test DocumentChunker with security validation failure."""
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            enable_caching=False,
            enable_security=True,
            enable_monitoring=False,
            security_config=SecurityConfig(
                allowed_extensions={'.txt'}  # Only .txt allowed
            )
        )
        
        chunker = DocumentChunker(config)
        
        # Create file with invalid extension
        test_file = tmp_path / "test.md"  # .md not allowed
        test_file.write_text("# Test Document\n\nContent.")
        
        result = chunker.chunk_file(test_file)
        
        assert result.success is False
        assert len(result.chunks) == 0
        assert "security validation failed" in result.error_message.lower()
    
    def test_chunker_with_security_disabled(self, tmp_path):
        """Test DocumentChunker with security disabled."""
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            enable_caching=False,
            enable_security=False,
            enable_monitoring=False
        )
        
        chunker = DocumentChunker(config)
        
        # Create file that would fail security checks
        test_file = tmp_path / "test.exe"  # Normally not allowed
        test_file.write_text("Content that would normally be blocked")
        
        # Should still process since security is disabled
        result = chunker.chunk_file(test_file)
        
        # Note: This might still fail due to file reading issues,
        # but it shouldn't fail due to security validation
        assert result.security_audit is None
    
    def test_directory_chunking_with_security(self, tmp_path):
        """Test directory chunking with security validation."""
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            enable_caching=False,
            enable_security=True,
            enable_monitoring=False
        )
        
        chunker = DocumentChunker(config)
        
        # Create mixed files
        (tmp_path / "safe.md").write_text("# Safe Document\n\nSafe content.")
        (tmp_path / "unsafe.exe").write_text("Unsafe content")
        
        with patch('magic.from_file', return_value="text/plain"):
            results = chunker.chunk_directory(tmp_path)
        
        # Should have results for both files
        assert len(results) == 2
        
        # Check that safe file was processed successfully
        safe_results = [r for r in results if r.file_path.name == "safe.md"]
        assert len(safe_results) == 1
        assert safe_results[0].success is True
        
        # Check that unsafe file was rejected
        unsafe_results = [r for r in results if r.file_path.name == "unsafe.exe"]
        assert len(unsafe_results) == 1
        assert unsafe_results[0].success is False


class TestSecurityPerformance:
    """Test security validation performance."""
    
    def test_path_sanitization_performance(self):
        """Test path sanitization performance."""
        config = SecurityConfig()
        sanitizer = PathSanitizer(config)
        
        import time
        
        # Test with many paths
        test_paths = [f"/home/user/doc_{i}/test.md" for i in range(1000)]
        
        start_time = time.time()
        for path in test_paths:
            sanitizer.is_safe_path(path)
        end_time = time.time()
        
        # Should complete quickly
        assert (end_time - start_time) < 1.0  # Under 1 second
    
    def test_file_validation_performance(self, tmp_path):
        """Test file validation performance."""
        config = SecurityConfig()
        validator = FileValidator(config)
        
        # Create test files
        test_files = []
        for i in range(10):  # Fewer files for I/O operations
            test_file = tmp_path / f"test_{i}.md"
            test_file.write_text(f"# Document {i}\n\nContent for document {i}.")
            test_files.append(test_file)
        
        import time
        
        start_time = time.time()
        with patch('magic.from_file', return_value="text/plain"):
            for test_file in test_files:
                validator.validate_file(test_file)
        end_time = time.time()
        
        # Should complete reasonably quickly
        assert (end_time - start_time) < 5.0  # Under 5 seconds


@pytest.fixture
def security_test_files(tmp_path):
    """Create test files for security testing."""
    files = {
        'safe_md': tmp_path / "safe.md",
        'safe_txt': tmp_path / "safe.txt",
        'unsafe_exe': tmp_path / "unsafe.exe",
        'large_file': tmp_path / "large.md",
        'binary_file': tmp_path / "binary.md"
    }
    
    # Create files
    files['safe_md'].write_text("# Safe Markdown\n\nSafe content.")
    files['safe_txt'].write_text("Safe text content.")
    files['unsafe_exe'].write_text("Potentially unsafe executable content.")
    files['large_file'].write_text("x" * 10000)  # Large file
    files['binary_file'].write_bytes(b"\x00\x01\x02\x03Binary content")  # Binary
    
    return files


class TestSecurityIntegration:
    """Integration tests for security components."""
    
    def test_full_security_pipeline(self, security_test_files):
        """Test complete security validation pipeline."""
        config = SecurityConfig(
            max_file_size_mb=5,  # 5MB limit 
            allowed_extensions={'.md', '.txt'},
            enable_checksum_validation=True,
            enable_content_scanning=True
        )
        
        auditor = SecurityAuditor(config)
        
        # Test each file
        results = {}
        for name, file_path in security_test_files.items():
            try:
                with patch('magic.from_file', return_value="text/plain"):
                    results[name] = auditor.audit_file(file_path)
            except Exception as e:
                results[name] = {'error': str(e), 'is_safe': False}
        
        # Verify expected results
        assert results['safe_md']['is_safe'] is True
        assert results['safe_txt']['is_safe'] is True
        assert results['unsafe_exe']['is_safe'] is False  # Wrong extension
        assert results['large_file']['is_safe'] is False  # Too large
        # binary_file might pass or fail depending on content validation
    
    def test_security_with_chunking_workflow(self, tmp_path):
        """Test security integration with full chunking workflow."""
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            enable_caching=True,
            enable_security=True,
            enable_monitoring=False,
            security_config=SecurityConfig(
                max_file_size_mb=1,
                allowed_extensions={'.md'},
                enable_content_scanning=True
            )
        )
        
        chunker = DocumentChunker(config)
        
        # Create test directory with mixed files
        (tmp_path / "good.md").write_text("# Good Document\n\nGood content.")
        (tmp_path / "bad.txt").write_text("Bad extension")
        (tmp_path / "toolarge.md").write_text("x" * 2000)  # Too large
        
        with patch('magic.from_file', return_value="text/plain"):
            results = chunker.chunk_directory(tmp_path)
        
        # Should have 3 results
        assert len(results) == 3
        
        # Check individual results
        good_result = next(r for r in results if r.file_path.name == "good.md")
        bad_result = next(r for r in results if r.file_path.name == "bad.txt")
        large_result = next(r for r in results if r.file_path.name == "toolarge.md")
        
        assert good_result.success is True
        assert bad_result.success is False
        assert large_result.success is False
        
        # Verify security audits were performed
        assert good_result.security_audit is not None
        assert bad_result.security_audit is not None
        assert large_result.security_audit is not None