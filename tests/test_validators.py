
import pytest
from pathlib import Path
from src.utils.validators import (
    validate_file_path,
    validate_directory_path,
    validate_chunk_size,
    validate_chunk_overlap,
    validate_output_format,
    validate_content,
    validate_metadata,
    validate_inputs,
    safe_path_join,
    validate_file_size
)
from src.exceptions import ValidationError, FileHandlingError

@pytest.fixture
def temp_file(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("hello")
    return file

def test_validate_file_path_success(temp_file):
    path = validate_file_path(str(temp_file))
    assert isinstance(path, Path)
    assert path.exists()

def test_validate_file_path_not_exist():
    with pytest.raises(FileHandlingError):
        validate_file_path("non_existent_file.txt")

def test_validate_file_path_wrong_extension(temp_file):
    with pytest.raises(ValidationError):
        validate_file_path(str(temp_file), extensions=['.md'])

def test_validate_directory_path_success(tmp_path):
    path = validate_directory_path(str(tmp_path))
    assert isinstance(path, Path)
    assert path.is_dir()

def test_validate_directory_path_not_exist():
    with pytest.raises(FileHandlingError):
        validate_directory_path("non_existent_dir")

def test_validate_directory_path_create(tmp_path):
    new_dir = tmp_path / "new_dir"
    path = validate_directory_path(str(new_dir), create_if_missing=True)
    assert path.exists()
    assert path.is_dir()

def test_validate_chunk_size_success():
    assert validate_chunk_size(1000) == 1000

def test_validate_chunk_size_too_small():
    with pytest.raises(ValidationError):
        validate_chunk_size(10)

def test_validate_chunk_size_too_large():
    with pytest.raises(ValidationError):
        validate_chunk_size(10000)

def test_validate_chunk_overlap_success():
    assert validate_chunk_overlap(200, 1000) == 200

def test_validate_chunk_overlap_negative():
    with pytest.raises(ValidationError):
        validate_chunk_overlap(-1, 1000)

def test_validate_chunk_overlap_too_large():
    with pytest.raises(ValidationError):
        validate_chunk_overlap(1000, 1000)

def test_validate_output_format_success():
    assert validate_output_format("json") == "json"
    assert validate_output_format("CSV") == "csv"

def test_validate_output_format_invalid():
    with pytest.raises(ValidationError):
        validate_output_format("xml")

def test_validate_content_success():
    assert validate_content("This is a test.") == "This is a test."

def test_validate_content_too_short():
    with pytest.raises(ValidationError):
        validate_content("")

def test_validate_metadata_success():
    assert validate_metadata({"key": "value"}) == {"key": "value"}

def test_validate_metadata_prohibited_key():
    with pytest.raises(ValidationError):
        validate_metadata({"__class__": "test"})

def test_validate_inputs_decorator():
    @validate_inputs(chunk_size=validate_chunk_size)
    def my_func(chunk_size):
        return chunk_size

    assert my_func(1000) == 1000
    with pytest.raises(ValidationError):
        my_func(10)

def test_safe_path_join_success():
    path = safe_path_join("/a/b", "c", "d.txt")
    assert str(path) == "/a/b/c/d.txt"

def test_safe_path_join_traversal():
    with pytest.raises(ValidationError):
        safe_path_join("/a/b", "../c")

def test_validate_file_size_success(temp_file):
    size = validate_file_size(temp_file)
    assert size == 5

def test_validate_file_size_too_large(temp_file):
    with pytest.raises(ValidationError):
        validate_file_size(temp_file, max_size_mb=0.000001)
