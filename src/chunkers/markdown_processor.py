import mistune
from typing import Dict, List, Any
import re

class MarkdownProcessor:
    """
    Enhanced Markdown processing with structure preservation
    """

    def __init__(self):
        self.parser = mistune.create_markdown(
            escape=False,
            plugins=['strikethrough', 'footnotes', 'table']
        )

    def extract_structure(self, content: str) -> Dict[str, Any]:
        """Extract structural information from Markdown"""

        structure = {
            'headers': [],
            'code_blocks': [],
            'tables': [],
            'links': [],
            'images': []
        }

        lines = content.split('\n')

        for i, line in enumerate(lines):
            # Extract headers
            header_match = re.match(r'^(#+)\s+(.*)', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2)
                structure['headers'].append({
                    'level': level,
                    'title': title,
                    'line': i
                })

            # Extract code blocks
            if line.strip().startswith('```'):
                structure['code_blocks'].append({'line': i, 'language': line.strip()[3:]})

            # Extract tables
            if '|' in line:
                structure['tables'].append({'line': i, 'content': line})

            # Extract links and images
            links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', line)
            images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', line)

            structure['links'].extend(links)
            structure['images'].extend(images)

        return structure

    def clean_content(self, content: str) -> str:
        """Clean and normalize Markdown content"""

        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Normalize headers
        content = re.sub(r'^(#+)\s*(.+)', r'\1 \2', content, flags=re.MULTILINE)

        return content.strip()