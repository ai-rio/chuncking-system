#!/usr/bin/env python3
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from src.utils.file_handler import FileHandler

# Create temp directory and file
temp_dir = Path(tempfile.mkdtemp())
output_path = temp_dir / "readonly.json"

# Create file and make it read-only
output_path.write_text("{}")
output_path.chmod(0o444)  # Read-only

print(f"File permissions: {oct(output_path.stat().st_mode)}")
print(f"File exists: {output_path.exists()}")
print(f"File is writable: {output_path.stat().st_mode & 0o200 != 0}")

chunks = [Document(page_content="test", metadata={})]

try:
    FileHandler.save_chunks(chunks, str(output_path), 'json')
    print("ERROR: No PermissionError was raised!")
except PermissionError as e:
    print(f"SUCCESS: PermissionError raised: {e}")
except Exception as e:
    print(f"ERROR: Different exception raised: {type(e).__name__}: {e}")
finally:
    # Restore write permissions for cleanup
    output_path.chmod(0o644)
    output_path.unlink()
    temp_dir.rmdir()