"""
Semantic chunking for extracted PDF content.

Strategy (100% offline, no LLM):
1. Split on semantic boundaries (headers, paragraphs)
2. Keep code blocks intact
3. Target ~1000 tokens per chunk with 100 token overlap
4. Preserve document structure (chapter/section context)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Iterator, Tuple
import re
from ..ingester.pdf_extractor import PageContent


@dataclass
class Chunk:
    """Represents a semantic chunk of content."""
    id: str                          # Unique chunk ID
    document_id: str                 # Parent document ID
    text: str                        # Chunk text content

    # Location metadata
    page_start: int                  # Starting page (1-indexed)
    page_end: int                    # Ending page (1-indexed)
    chunk_index: int                 # Position in document
    total_chunks: int                # Total chunks in document (filled later)

    # Document structure
    chapter: str = ""                # Current chapter heading
    section: str = ""                # Current section heading
    subsection: str = ""             # Current subsection heading

    # Content classification
    content_type: str = "prose"      # prose, code, table, list, mixed

    # Linking (filled after all chunks created)
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    # Stats
    token_count: int = 0
    text_length: int = 0

    # Topic tags (extracted from content)
    topic_tags: List[str] = field(default_factory=list)


class SemanticChunker:
    """
    Rule-based semantic chunker that respects document structure.

    Splitting hierarchy (in order of preference):
    1. Chapter boundaries (always split)
    2. Section boundaries (always split if chunk > min_size)
    3. Paragraph boundaries (split if chunk > target_size)
    4. Sentence boundaries (fallback)
    """

    # Header patterns for semantic boundary detection
    CHAPTER_PATTERNS = [
        r'^#{1}\s+(.+)$',                          # Markdown H1
        r'^Chapter\s+\d+[:\.]?\s*(.+)?$',          # "Chapter 1: Title"
        r'^CHAPTER\s+\d+[:\.]?\s*(.+)?$',          # "CHAPTER 1: TITLE"
        r'^Part\s+\d+[:\.]?\s*(.+)?$',             # "Part 1: Title"
    ]

    SECTION_PATTERNS = [
        r'^#{2}\s+(.+)$',                          # Markdown H2
        r'^(\d+\.)\s+(.+)$',                       # "1. Section Title"
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):?\s*$', # "Section Title:" (title case)
    ]

    SUBSECTION_PATTERNS = [
        r'^#{3,4}\s+(.+)$',                        # Markdown H3/H4
        r'^(\d+\.\d+)\s+(.+)$',                    # "1.1 Subsection"
        r'^(\d+\.\d+\.\d+)\s+(.+)$',               # "1.1.1 Sub-subsection"
    ]

    # Code block detection
    CODE_FENCE_PATTERN = r'^```'
    CODE_INDENT_PATTERN = r'^(?:    |\t)'

    # Salesforce-specific topic patterns
    TOPIC_PATTERNS = {
        'apex': r'\b(?:Apex|trigger|SOQL|SOSL|DML|governor\s+limit|bulk|queueable|batch|future|schedulable)\b',
        'lwc': r'\b(?:Lightning\s+Web\s+Component|LWC|@wire|@api|@track|lightning-|slds)\b',
        'flow': r'\b(?:Flow|Screen\s+Flow|Record-Triggered|Scheduled|Autolaunched|Process\s+Builder)\b',
        'integration': r'\b(?:REST|SOAP|callout|HttpRequest|Named\s+Credential|External\s+Service|Platform\s+Event)\b',
        'security': r'\b(?:CRUD|FLS|sharing|with\s+sharing|without\s+sharing|permission|profile|role)\b',
        'testing': r'\b(?:@isTest|Test\.startTest|Test\.stopTest|assert|mock|stub|test\s+class)\b',
    }

    def __init__(
        self,
        target_tokens: int = 1000,
        max_tokens: int = 1500,
        min_tokens: int = 200,
        overlap_tokens: int = 100
    ):
        """
        Initialize the chunker.

        Args:
            target_tokens: Target chunk size in tokens
            max_tokens: Maximum chunk size before forced split
            min_tokens: Minimum chunk size (avoid tiny chunks)
            overlap_tokens: Token overlap between adjacent chunks
        """
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens

        # Approximate tokens = chars / 4 (rough estimate for English)
        self.target_chars = target_tokens * 4
        self.max_chars = max_tokens * 4
        self.min_chars = min_tokens * 4
        self.overlap_chars = overlap_tokens * 4

    def chunk_document(
        self,
        pages: List[PageContent],
        document_id: str,
        toc_map: dict = None
    ) -> List[Chunk]:
        """
        Chunk an entire document from extracted pages.

        Args:
            pages: List of PageContent from PDF extractor
            document_id: Unique document identifier
            toc_map: Optional table of contents mapping

        Returns:
            List of Chunk objects
        """
        # Combine all pages into structured segments
        segments = self._extract_segments(pages, toc_map)

        # Create chunks from segments
        chunks = []
        chunk_index = 0

        for segment in segments:
            segment_chunks = self._chunk_segment(
                segment=segment,
                document_id=document_id,
                start_index=chunk_index
            )
            chunks.extend(segment_chunks)
            chunk_index += len(segment_chunks)

        # Fill in total_chunks and link chunks
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.total_chunks = total
            if i > 0:
                chunk.prev_chunk_id = chunks[i - 1].id
                chunks[i - 1].next_chunk_id = chunk.id

        return chunks

    def _extract_segments(
        self,
        pages: List[PageContent],
        toc_map: dict = None
    ) -> List[dict]:
        """
        Extract structural segments from pages.

        Returns list of segments with:
        - text: segment content
        - page_start/page_end: page range
        - chapter/section/subsection: document structure
        - content_type: prose/code/table/mixed
        """
        segments = []
        current_segment = {
            'text': '',
            'page_start': 1,
            'page_end': 1,
            'chapter': '',
            'section': '',
            'subsection': '',
            'content_type': 'prose',
            'has_code': False,
            'has_table': False
        }

        for page in pages:
            lines = page.text.split('\n')

            for line in lines:
                # Check for chapter boundary
                chapter_match = self._match_header(line, self.CHAPTER_PATTERNS)
                if chapter_match:
                    # Save current segment if not empty
                    if current_segment['text'].strip():
                        current_segment['content_type'] = self._classify_content(current_segment)
                        segments.append(current_segment.copy())

                    # Start new chapter segment
                    current_segment = {
                        'text': line + '\n',
                        'page_start': page.page_num,
                        'page_end': page.page_num,
                        'chapter': chapter_match,
                        'section': '',
                        'subsection': '',
                        'content_type': 'prose',
                        'has_code': False,
                        'has_table': False
                    }
                    continue

                # Check for section boundary
                section_match = self._match_header(line, self.SECTION_PATTERNS)
                if section_match:
                    # Save if segment is large enough
                    if len(current_segment['text']) > self.min_chars:
                        current_segment['content_type'] = self._classify_content(current_segment)
                        segments.append(current_segment.copy())

                        current_segment = {
                            'text': line + '\n',
                            'page_start': page.page_num,
                            'page_end': page.page_num,
                            'chapter': current_segment['chapter'],
                            'section': section_match,
                            'subsection': '',
                            'content_type': 'prose',
                            'has_code': False,
                            'has_table': False
                        }
                        continue

                # Check for subsection
                subsection_match = self._match_header(line, self.SUBSECTION_PATTERNS)
                if subsection_match:
                    current_segment['subsection'] = subsection_match

                # Track code blocks
                if re.match(self.CODE_FENCE_PATTERN, line):
                    current_segment['has_code'] = True

                # Append line to current segment
                current_segment['text'] += line + '\n'
                current_segment['page_end'] = page.page_num

            # Track tables from page metadata
            if page.has_tables:
                current_segment['has_table'] = True

        # Don't forget the last segment
        if current_segment['text'].strip():
            current_segment['content_type'] = self._classify_content(current_segment)
            segments.append(current_segment)

        return segments

    def _match_header(self, line: str, patterns: List[str]) -> Optional[str]:
        """Try to match a line against header patterns."""
        line = line.strip()
        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                # Return the captured title or the full line
                groups = match.groups()
                if groups:
                    return ' '.join(g for g in groups if g).strip()
                return line
        return None

    def _classify_content(self, segment: dict) -> str:
        """Classify segment content type."""
        if segment['has_code'] and segment['has_table']:
            return 'mixed'
        if segment['has_code']:
            return 'code'
        if segment['has_table']:
            return 'table'

        # Check for list content
        text = segment['text']
        list_lines = len(re.findall(r'^[\s]*[-•*]\s', text, re.MULTILINE))
        total_lines = len(text.split('\n'))
        if total_lines > 0 and list_lines / total_lines > 0.5:
            return 'list'

        return 'prose'

    def _chunk_segment(
        self,
        segment: dict,
        document_id: str,
        start_index: int
    ) -> List[Chunk]:
        """
        Split a segment into appropriately-sized chunks.
        """
        text = segment['text']

        # If segment is small enough, return as single chunk
        if len(text) <= self.max_chars:
            if len(text.strip()) < self.min_chars:
                return []  # Skip tiny segments

            return [self._create_chunk(
                text=text,
                document_id=document_id,
                chunk_index=start_index,
                segment=segment
            )]

        # Split large segments
        chunks = []
        paragraphs = self._split_into_paragraphs(text)

        current_text = ""
        current_start_page = segment['page_start']

        for para in paragraphs:
            # Would adding this paragraph exceed max?
            if len(current_text) + len(para) > self.max_chars:
                # Save current chunk if big enough
                if len(current_text.strip()) >= self.min_chars:
                    chunk = self._create_chunk(
                        text=current_text,
                        document_id=document_id,
                        chunk_index=start_index + len(chunks),
                        segment={**segment, 'page_start': current_start_page}
                    )
                    chunks.append(chunk)

                    # Start new chunk with overlap
                    overlap = self._get_overlap(current_text)
                    current_text = overlap + para
                else:
                    current_text += para
            else:
                current_text += para

        # Handle remaining text
        if len(current_text.strip()) >= self.min_chars:
            chunk = self._create_chunk(
                text=current_text,
                document_id=document_id,
                chunk_index=start_index + len(chunks),
                segment={**segment, 'page_start': current_start_page}
            )
            chunks.append(chunk)
        elif chunks and current_text.strip():
            # Append to last chunk if too small
            chunks[-1].text += current_text
            chunks[-1].text_length = len(chunks[-1].text)
            chunks[-1].token_count = self._estimate_tokens(chunks[-1].text)

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs, keeping code blocks intact.
        """
        paragraphs = []
        current = ""
        in_code_block = False

        for line in text.split('\n'):
            # Track code fence boundaries
            if re.match(self.CODE_FENCE_PATTERN, line):
                in_code_block = not in_code_block
                current += line + '\n'

                # If closing a code block, that's a paragraph boundary
                if not in_code_block:
                    paragraphs.append(current)
                    current = ""
                continue

            # Inside code block - keep accumulating
            if in_code_block:
                current += line + '\n'
                continue

            # Empty line = paragraph boundary (outside code)
            if not line.strip():
                if current.strip():
                    paragraphs.append(current + '\n')
                    current = ""
                continue

            current += line + '\n'

        # Don't forget trailing content
        if current.strip():
            paragraphs.append(current)

        return paragraphs

    def _get_overlap(self, text: str) -> str:
        """Get the overlap portion from the end of text."""
        if len(text) <= self.overlap_chars:
            return text

        # Try to break at sentence boundary
        overlap_region = text[-self.overlap_chars * 2:]
        sentences = re.split(r'(?<=[.!?])\s+', overlap_region)

        if len(sentences) > 1:
            # Return last complete sentences that fit
            overlap = ""
            for sent in reversed(sentences):
                if len(overlap) + len(sent) <= self.overlap_chars:
                    overlap = sent + " " + overlap
                else:
                    break
            return overlap.strip() + '\n\n' if overlap else text[-self.overlap_chars:]

        return text[-self.overlap_chars:]

    def _create_chunk(
        self,
        text: str,
        document_id: str,
        chunk_index: int,
        segment: dict
    ) -> Chunk:
        """Create a Chunk object with all metadata."""
        import hashlib

        # Generate deterministic chunk ID
        chunk_id = hashlib.sha256(
            f"{document_id}:{chunk_index}:{text[:100]}".encode()
        ).hexdigest()[:16]

        # Extract topic tags
        topic_tags = self._extract_topics(text)

        return Chunk(
            id=chunk_id,
            document_id=document_id,
            text=text.strip(),
            page_start=segment['page_start'],
            page_end=segment['page_end'],
            chunk_index=chunk_index,
            total_chunks=0,  # Filled in later
            chapter=segment.get('chapter', ''),
            section=segment.get('section', ''),
            subsection=segment.get('subsection', ''),
            content_type=segment.get('content_type', 'prose'),
            token_count=self._estimate_tokens(text),
            text_length=len(text),
            topic_tags=topic_tags
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (chars / 4 for English)."""
        return len(text) // 4

    def _extract_topics(self, text: str) -> List[str]:
        """Extract Salesforce-specific topics from text."""
        topics = []
        text_lower = text.lower()

        for topic, pattern in self.TOPIC_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                topics.append(topic)

        return topics


def chunk_pages(
    pages: List[PageContent],
    document_id: str,
    toc_map: dict = None,
    **kwargs
) -> List[Chunk]:
    """
    Convenience function to chunk extracted pages.

    Args:
        pages: List of PageContent from PDF extractor
        document_id: Unique document identifier
        toc_map: Optional table of contents mapping
        **kwargs: Passed to SemanticChunker constructor

    Returns:
        List of Chunk objects
    """
    chunker = SemanticChunker(**kwargs)
    return chunker.chunk_document(pages, document_id, toc_map)
