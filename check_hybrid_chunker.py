import sys
import os
import inspect

# Add the parent directory to the sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.chunkers.hybrid_chunker import HybridChunker
    print(f"Successfully imported HybridChunker from: {inspect.getfile(HybridChunker)}")

    # Check for the method in the class
    if hasattr(HybridChunker, '_post_process_chunks'):
        print("'_post_process_chunks' method IS present in HybridChunker class.")
    else:
        print("'_post_process_chunks' method is NOT present in HybridChunker class.")
        print("Available public methods/attributes in HybridChunker class:")
        for name in dir(HybridChunker):
            if not name.startswith('__') and callable(getattr(HybridChunker, name)):
                print(f"  - {name}")

    # Check for the method in an instance (for completeness)
    try:
        instance = HybridChunker()
        if hasattr(instance, '_post_process_chunks'):
            print("'_post_process_chunks' method IS present in HybridChunker instance.")
        else:
            print("'_post_process_chunks' method is NOT present in HybridChunker instance.")
    except Exception as e:
        print(f"Could not instantiate HybridChunker: {e}")


except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure your src/chunkers/hybrid_chunker.py file exists and has no syntax errors.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")