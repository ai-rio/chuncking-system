import os
import json
import pickle
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from langchain_core.documents import Document
from src.exceptions import (
    FileHandlingError,
    ValidationError,
    ProcessingError
)

class FileHandler:
    """Handle file operations for the chunking system"""
    
    @staticmethod
    def find_markdown_files(directory: str) -> List[str]:
        """Find all Markdown files in directory"""
        try:
            if not isinstance(directory, str):
                raise ValidationError(
                    "Directory path must be a string",
                    field="directory",
                    value=type(directory)
                )
                
            if not os.path.exists(directory):
                raise OSError(f"Directory not found: {directory}")
            
            if not os.path.isdir(directory):
                raise FileHandlingError(
                    "Path is not a directory",
                    file_path=directory,
                    operation="find_files"
                )
                
            markdown_extensions = ['.md', '.markdown', '.mdown', '.mkd']
            files = []
            
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    if any(filename.lower().endswith(ext) for ext in markdown_extensions):
                        files.append(os.path.join(root, filename))
            
            return sorted(files)
            
        except (ValidationError, FileHandlingError, OSError) as e:
            raise
        except Exception as e:
            raise FileHandlingError(
                f"Failed to scan directory: {str(e)}",
                file_path=directory,
                operation="find_files"
            ) from e
    
    @staticmethod
    def save_chunks(
        chunks: List[Document], 
        output_path: str, 
        format: str = 'json'
    ):
        """Save chunks in specified format"""
        try:
            # Validate inputs
            if not isinstance(chunks, list):
                raise ValidationError(
                    "Chunks must be a list",
                    field="chunks",
                    value=type(chunks)
                )
                
            if not isinstance(output_path, str):
                raise ValidationError(
                    "Output path must be a string",
                    field="output_path",
                    value=type(output_path)
                )
                
            if format not in ['json', 'csv', 'pickle']:
                raise ValidationError(
                    "Format must be 'json', 'csv', or 'pickle'",
                    field="format",
                    value=format
                )
            
            # Create output directory
            try:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise FileHandlingError(
                    f"Failed to create output directory: {str(e)}",
                    file_path=str(Path(output_path).parent),
                    operation="create_directory"
                ) from e
            
            if format == 'json':
                chunks_data = []
                for i, chunk in enumerate(chunks):
                    if not isinstance(chunk, Document):
                        raise ValidationError(
                            f"Chunk {i} is not a Document instance",
                            field="chunks",
                            value=type(chunk)
                        )
                    chunks_data.append({
                        'content': chunk.page_content,
                        'metadata': chunk.metadata
                    })
                
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
                    print(f"DEBUG: FileHandler.save_chunks - Successfully wrote to {output_path}")
                    print(f"DEBUG: FileHandler.save_chunks - File exists after write: {Path(output_path).exists()}")
                except PermissionError:
                    raise
        
            elif format == 'pickle':
                try:
                    with open(output_path, 'wb') as f:
                        pickle.dump(chunks, f)
                except PermissionError:
                    raise
            
            elif format == 'csv':
                df_data = []
                for i, chunk in enumerate(chunks):
                    if not isinstance(chunk, Document):
                        raise ValidationError(
                            f"Chunk {i} is not a Document instance",
                            field="chunks",
                            value=type(chunk)
                        )
                    # Ensure all values are properly defaulted to avoid NaN
                    source = chunk.metadata.get('source')
                    if source is None or pd.isna(source):
                        source = ''
                    
                    tokens = chunk.metadata.get('chunk_tokens')
                    if tokens is None or pd.isna(tokens):
                        tokens = 0
                        
                    words = chunk.metadata.get('word_count')
                    if words is None or pd.isna(words):
                        words = 0
                    
                    df_data.append({
                        'chunk_id': i,
                        'content': chunk.page_content,
                        'source': str(source),  # Ensure string type
                        'tokens': int(tokens),  # Ensure int type
                        'words': int(words)     # Ensure int type
                    })
                
                df = pd.DataFrame(df_data)
                # Ensure no NaN values remain
                df = df.fillna({'source': '', 'tokens': 0, 'words': 0})
                # Convert to proper types
                df['source'] = df['source'].astype(str)
                df['tokens'] = df['tokens'].astype(int)
                df['words'] = df['words'].astype(int)
                
                # Save with proper handling to prevent NaN on read
                try:
                    df.to_csv(output_path, index=False, na_rep='')
                except PermissionError:
                    raise
                
        except (ValidationError, FileHandlingError, PermissionError) as e:
            raise
        except Exception as e:
            raise FileHandlingError(
                f"Failed to save chunks: {str(e)}",
                file_path=output_path,
                operation="save_chunks"
            ) from e
    
    @staticmethod
    def load_chunks(file_path: str) -> List[Document]:
        """Load chunks from file"""
        try:
            # Validate input
            if not isinstance(file_path, str):
                raise ValidationError(
                    "File path must be a string",
                    field="file_path",
                    value=type(file_path)
                )
                
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    raise FileHandlingError(
                        "JSON file must contain a list of chunks",
                        file_path=file_path,
                        operation="load_chunks"
                    )
                
                chunks = []
                for i, item in enumerate(data):
                    if not isinstance(item, dict) or 'content' not in item:
                        raise FileHandlingError(
                            f"Invalid chunk format at index {i}",
                            file_path=file_path,
                            operation="load_chunks"
                        )
                    chunks.append(
                        Document(
                            page_content=item['content'], 
                            metadata=item.get('metadata', {})
                        )
                    )
                return chunks
            
            elif file_path.endswith('.pickle'):
                with open(file_path, 'rb') as f:
                    chunks = pickle.load(f)
                    
                if not isinstance(chunks, list):
                    raise FileHandlingError(
                        "Pickle file must contain a list of chunks",
                        file_path=file_path,
                        operation="load_chunks"
                    )
                    
                return chunks
            
            else:
                raise ValueError(f"Unsupported file format: {file_path}. Use .json or .pickle")
                
        except (ValidationError, FileHandlingError, FileNotFoundError, ValueError) as e:
            raise
        except Exception as e:
            raise FileHandlingError(
                f"Failed to load chunks: {str(e)}",
                file_path=file_path,
                operation="load_chunks"
            ) from e
