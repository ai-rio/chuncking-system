
import pytest
from pathlib import Path
from src.utils.path_utils import PathManager, MarkdownFileManager

@pytest.fixture
def path_manager(tmp_path):
    return PathManager(base_dir=tmp_path)

@pytest.fixture
def markdown_file_manager(tmp_path):
    return MarkdownFileManager(base_dir=tmp_path)

@pytest.fixture
def temp_md_file(tmp_path):
    file = tmp_path / "test.md"
    file.write_text("---\ntitle: Test Document\n---\n# Test")
    return file

def test_path_manager_resolve_path(path_manager, tmp_path):
    path = path_manager.resolve_path("test.txt", relative_to_base=True)
    assert path == tmp_path / "test.txt"

def test_path_manager_create_output_structure(path_manager, tmp_path):
    output_dir = tmp_path / "output"
    structure = path_manager.create_output_structure(output_dir)
    assert "chunks" in structure
    assert (output_dir / "chunks").exists()

def test_path_manager_find_files(path_manager, tmp_path):
    (tmp_path / "test1.txt").touch()
    (tmp_path / "test2.log").touch()
    files = path_manager.find_files(tmp_path, ["*.txt"])
    assert len(files) == 1
    assert files[0].name == "test1.txt"

def test_path_manager_get_safe_filename(path_manager):
    assert path_manager.get_safe_filename("a/b c.txt") == "ab_c.txt"

def test_path_manager_generate_unique_path(path_manager, tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.touch()
    unique_path = path_manager.generate_unique_path(test_file)
    assert unique_path.name == "test_1.txt"

def test_path_manager_calculate_relative_path(path_manager, tmp_path):
    target = tmp_path / "a/b/c.txt"
    base = tmp_path / "a"
    relative_path = path_manager.calculate_relative_path(target, base)
    assert relative_path == Path("b/c.txt")

def test_path_manager_ensure_parent_directory(path_manager, tmp_path):
    test_file = tmp_path / "new_dir/test.txt"
    path_manager.ensure_parent_directory(test_file)
    assert (tmp_path / "new_dir").exists()

def test_path_manager_get_file_info(path_manager, temp_md_file):
    info = path_manager.get_file_info(temp_md_file)
    assert info['name'] == "test.md"
    assert info['size'] > 0

def test_path_manager_cleanup_empty_directories(path_manager, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    removed_count = path_manager.cleanup_empty_directories(tmp_path)
    assert removed_count == 1
    assert not empty_dir.exists()

def test_markdown_file_manager_find_markdown_files(markdown_file_manager, temp_md_file):
    files = markdown_file_manager.find_markdown_files(temp_md_file.parent)
    assert len(files) == 1
    assert files[0].name == "test.md"

def test_markdown_file_manager_is_markdown_file(markdown_file_manager, temp_md_file):
    assert markdown_file_manager.is_markdown_file(temp_md_file) is True
    assert markdown_file_manager.is_markdown_file("test.txt") is False

def test_markdown_file_manager_create_output_paths(markdown_file_manager, temp_md_file, tmp_path):
    output_dir = tmp_path / "output"
    paths = markdown_file_manager.create_markdown_output_paths(temp_md_file, output_dir)
    assert "chunks_json" in paths
    assert paths["chunks_json"].name == "test_chunks.json"

def test_markdown_file_manager_read_write_file(markdown_file_manager, tmp_path):
    test_file = tmp_path / "test.md"
    content = "# Hello"
    markdown_file_manager.write_file(test_file, content)
    read_content = markdown_file_manager.read_file(test_file)
    assert read_content == content

def test_markdown_file_manager_get_file_metadata(markdown_file_manager, temp_md_file):
    metadata = markdown_file_manager.get_file_metadata(temp_md_file)
    assert metadata['title'] == "Test Document"

def test_markdown_file_manager_validate_markdown_file(markdown_file_manager, temp_md_file):
    assert markdown_file_manager.validate_markdown_file(temp_md_file) is True

def test_markdown_file_manager_backup_restore(markdown_file_manager, temp_md_file):
    backup_path = markdown_file_manager.backup_file(temp_md_file)
    assert backup_path.exists()
    (temp_md_file).write_text("modified")
    markdown_file_manager.restore_backup(backup_path, temp_md_file)
    assert temp_md_file.read_text() == "---\ntitle: Test Document\n---\n# Test"
