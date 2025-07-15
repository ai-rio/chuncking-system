import os
import json
import pickle
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from langchain_core.documents import Document

class FileHandler:
    """Handle file operations for the chunking system"""
    
    @staticmethod
    def find_markdown_files(directory: str) -> List[str]:
        """Find all Markdown files in directory"""
        if not os.path.exists(directory):
            raise OSError(f"Directory not found: {directory}")
            
        markdown_extensions = ['.md', '.markdown', '.mdown', '.mkd']
        files = []
        
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in markdown_extensions):
                    files.append(os.path.join(root, filename))
        
        return sorted(files)
    
    @staticmethod
    def save_chunks(
        chunks: List[Document], 
        output_path: str, 
        format: str = 'json'
    ):
        """Save chunks in specified format"""
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            chunks_data = []
            for chunk in chunks:
                chunks_data.append({
                    'content': chunk.page_content,
                    'metadata': chunk.metadata
                })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        elif format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(chunks, f)
        
        elif format == 'csv':
            df_data = []
            for i, chunk in enumerate(chunks):
                df_data.append({
                    'chunk_id': i,
                    'content': chunk.page_content,
                    'source': chunk.metadata.get('source', ''),
                    'tokens': chunk.metadata.get('chunk_tokens', 0),
                    'words': chunk.metadata.get('word_count', 0)
                })
            
            df = pd.DataFrame(df_data)
            # Replace NaN with empty strings
            df = df.fillna('')
            df.to_csv(output_path, index=False)
    
    @staticmethod
    def load_chunks(file_path: str) -> List[Document]:
        """Load chunks from file"""
        
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return [
                Document(page_content=item['content'], metadata=item['metadata'])
                for item in data
            ]
        
        elif file_path.endswith('.pickle'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
