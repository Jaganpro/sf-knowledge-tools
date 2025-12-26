"""
Markdown generator for exporting RAG results.

Uses Jinja2 templates for consistent, customizable output.
Generates clean markdown suitable for sf-skills references.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
import re

from ..query.rag_engine import RAGResult, Citation


class MarkdownGenerator:
    """
    Generate markdown documents from RAG query results.

    Features:
    - Jinja2 templates for customization
    - Automatic content organization
    - Citation formatting
    - Code block detection and formatting
    """

    DEFAULT_TEMPLATE = "skill-reference.md"

    def __init__(self, template_dir: Path = None, template_name: str = None):
        """
        Initialize the markdown generator.

        Args:
            template_dir: Directory containing Jinja2 templates
            template_name: Name of template to use
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"

        self.template_dir = Path(template_dir)
        self.template_name = template_name or self.DEFAULT_TEMPLATE

        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Add custom filters
        self.env.filters['slugify'] = self._slugify
        self.env.filters['truncate_smart'] = self._truncate_smart
        self.env.filters['format_code'] = self._format_code_blocks
        self.env.filters['extract_headers'] = self._extract_headers

    def generate(
        self,
        topic: str,
        skill: str,
        result: RAGResult,
        extra_context: Dict = None
    ) -> str:
        """
        Generate a markdown document from RAG results.

        Args:
            topic: Topic/title for the document
            skill: Target skill name (sf-apex, sf-flow, etc.)
            result: RAGResult from query
            extra_context: Additional template variables

        Returns:
            Formatted markdown string
        """
        # Prepare template context
        context = self._prepare_context(topic, skill, result)

        if extra_context:
            context.update(extra_context)

        # Load and render template
        try:
            template = self.env.get_template(self.template_name)
        except Exception:
            # Fall back to default inline template
            template = self.env.from_string(self._get_default_template())

        return template.render(**context)

    def _prepare_context(
        self,
        topic: str,
        skill: str,
        result: RAGResult
    ) -> Dict[str, Any]:
        """Prepare context dictionary for template rendering."""

        # Group chunks by chapter/section
        organized_content = self._organize_content(result.chunks)

        # Format sources
        sources = self._format_sources(result.citations)

        # Extract key topics from chunks
        all_topics = set()
        for chunk in result.chunks:
            all_topics.update(chunk.get('topic_tags', []))

        return {
            'title': topic,
            'skill': skill,
            'extraction_date': datetime.now().strftime('%Y-%m-%d'),
            'sources': sources,
            'source_documents': result.get_unique_sources(),
            'topics': sorted(list(all_topics)),
            'sections': organized_content,
            'raw_chunks': result.chunks,
            'total_chunks': len(result.chunks),
            'query': result.query
        }

    def _organize_content(self, chunks: List[Dict]) -> List[Dict]:
        """
        Organize chunks into a hierarchical structure.

        Returns list of sections, each with:
        - title: Section title
        - content: Combined content
        - subsections: Nested subsections
        - page_range: Source page numbers
        """
        sections = {}

        for chunk in chunks:
            chapter = chunk.get('chapter', 'General')
            section = chunk.get('section', '')

            # Create chapter if not exists
            if chapter not in sections:
                sections[chapter] = {
                    'title': chapter,
                    'subsections': {},
                    'content': [],
                    'page_start': chunk.get('page_start', 0),
                    'page_end': chunk.get('page_end', 0)
                }

            sec = sections[chapter]

            # Update page range
            sec['page_start'] = min(sec['page_start'], chunk.get('page_start', 0))
            sec['page_end'] = max(sec['page_end'], chunk.get('page_end', 0))

            if section:
                # Add to subsection
                if section not in sec['subsections']:
                    sec['subsections'][section] = {
                        'title': section,
                        'content': [],
                        'page_start': chunk.get('page_start', 0),
                        'page_end': chunk.get('page_end', 0)
                    }
                sec['subsections'][section]['content'].append(chunk)
                sec['subsections'][section]['page_end'] = max(
                    sec['subsections'][section]['page_end'],
                    chunk.get('page_end', 0)
                )
            else:
                # Add directly to chapter
                sec['content'].append(chunk)

        # Convert to list and sort
        result = []
        for chapter, data in sections.items():
            # Convert subsections to list
            data['subsections'] = list(data['subsections'].values())
            result.append(data)

        return result

    def _format_sources(self, citations: List[Citation]) -> List[Dict]:
        """Format citations as source references."""
        sources = []
        seen = set()

        for citation in citations:
            key = citation.document_filename
            if key in seen:
                continue
            seen.add(key)

            sources.append({
                'title': citation.document_title or citation.document_filename,
                'filename': citation.document_filename,
                'pages': f"p. {citation.page_start}" if citation.page_start == citation.page_end
                        else f"pp. {citation.page_start}-{citation.page_end}",
                'chapter': citation.chapter,
                'section': citation.section
            })

        return sources

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to URL-safe slug."""
        slug = text.lower().strip()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_]+', '-', slug)
        return slug

    @staticmethod
    def _truncate_smart(text: str, length: int = 200) -> str:
        """Truncate text at sentence boundary if possible."""
        if len(text) <= length:
            return text

        # Try to break at sentence
        truncated = text[:length]
        last_period = truncated.rfind('.')
        if last_period > length // 2:
            return truncated[:last_period + 1]

        return truncated + "..."

    @staticmethod
    def _format_code_blocks(text: str) -> str:
        """Ensure code blocks have proper language hints."""
        # Add language hints to bare code blocks
        def add_language(match):
            code = match.group(1)
            # Detect language
            if re.search(r'\b(public|private|class|trigger)\b', code):
                return f"```apex\n{code}```"
            if re.search(r'\b(SELECT|FROM|WHERE)\b', code, re.I):
                return f"```soql\n{code}```"
            if re.search(r'<[a-z]+[^>]*>', code):
                return f"```xml\n{code}```"
            if re.search(r'\{.*:.*\}', code, re.DOTALL):
                return f"```json\n{code}```"
            return f"```\n{code}```"

        return re.sub(r'```\n(.*?)```', add_language, text, flags=re.DOTALL)

    @staticmethod
    def _extract_headers(text: str) -> List[str]:
        """Extract markdown headers from text."""
        headers = re.findall(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
        return headers

    def _get_default_template(self) -> str:
        """Return default template as string."""
        return '''---
title: "{{ title }}"
extracted_from:
{% for source in sources %}
  - {{ source.filename }} ({{ source.pages }})
{% endfor %}
extraction_date: {{ extraction_date }}
skill: {{ skill }}
topics: [{{ topics | join(', ') }}]
---

# {{ title }}

> Extracted from official Salesforce documentation and enterprise patterns.

{% for section in sections %}
## {{ section.title }}

{% for chunk in section.content %}
{{ chunk.text }}

{% endfor %}
{% for subsection in section.subsections %}
### {{ subsection.title }}

{% for chunk in subsection.content %}
{{ chunk.text }}

{% endfor %}
{% endfor %}
{% endfor %}

## Sources

{% for source in sources %}
- {{ source.title }}{% if source.chapter %}, {{ source.chapter }}{% endif %}, {{ source.pages }}
{% endfor %}

---
*Auto-generated by sf-knowledge-tools*
'''


def create_skill_reference_template():
    """Create the default skill reference template file."""
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)

    template_path = template_dir / "skill-reference.md"
    template_content = '''---
title: "{{ title }}"
extracted_from:
{% for source in sources %}
  - {{ source.filename }} ({{ source.pages }})
{% endfor %}
extraction_date: {{ extraction_date }}
skill: {{ skill }}
topics: [{{ topics | join(', ') }}]
---

# {{ title }}

> Extracted from official Salesforce documentation and enterprise patterns.

## Overview

{% if sections %}
This document covers the following topics from {{ source_documents | length }} source document(s):

{% for section in sections %}
- **{{ section.title }}**{% if section.subsections %} ({{ section.subsections | length }} subsections){% endif %}
{% endfor %}
{% endif %}

{% for section in sections %}
---

## {{ section.title }}

{% if section.page_start %}
*Source: Pages {{ section.page_start }}-{{ section.page_end }}*
{% endif %}

{% for chunk in section.content %}
{{ chunk.text | trim }}

{% endfor %}
{% for subsection in section.subsections %}
### {{ subsection.title }}

{% for chunk in subsection.content %}
{{ chunk.text | trim }}

{% endfor %}
{% endfor %}
{% endfor %}

---

## Key Takeaways

Based on the extracted content:

{% for topic in topics[:5] %}
- {{ topic | title }} patterns and best practices are covered
{% endfor %}

## Sources

The content in this document was extracted from:

{% for source in sources %}
- **{{ source.title }}**
  - {{ source.pages }}
  {% if source.chapter %}- Chapter: {{ source.chapter }}{% endif %}
{% endfor %}

---
*Auto-generated by sf-knowledge-tools on {{ extraction_date }}*
'''

    template_path.write_text(template_content)
    return template_path
