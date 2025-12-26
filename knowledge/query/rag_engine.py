"""
RAG (Retrieval-Augmented Generation) engine with citations.

This engine retrieves relevant chunks and formats them for export.
Note: This is 100% offline - no LLM calls. The "generation" part
refers to generating structured output from retrieved chunks.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json

from ..storage.vector_store import VectorStore
from ..embedder.embedding_client import EmbeddingClient, EmbeddingConfig
from ..chunker.semantic_chunker import Chunk


@dataclass
class Citation:
    """A citation pointing to source material."""
    document_title: str
    document_filename: str
    page_start: int
    page_end: int
    chapter: str = ""
    section: str = ""
    chunk_id: str = ""

    def format(self, style: str = "markdown") -> str:
        """Format citation for display."""
        if style == "markdown":
            location = f"p. {self.page_start}"
            if self.page_end > self.page_start:
                location = f"pp. {self.page_start}-{self.page_end}"

            parts = [self.document_title]
            if self.chapter:
                parts.append(f"Chapter: {self.chapter}")
            if self.section:
                parts.append(f"Section: {self.section}")
            parts.append(location)

            return " | ".join(parts)

        return f"{self.document_filename}, {location}"


@dataclass
class RAGResult:
    """Result from a RAG query."""
    query: str
    chunks: List[Dict[str, Any]]
    citations: List[Citation]
    total_matches: int
    search_time_ms: float

    # Grouped content by chapter/section
    grouped_content: Dict[str, List[Dict]] = field(default_factory=dict)

    def get_combined_text(self, separator: str = "\n\n---\n\n") -> str:
        """Get all chunk texts combined."""
        return separator.join(c['text'] for c in self.chunks)

    def get_unique_sources(self) -> List[str]:
        """Get list of unique source documents."""
        seen = set()
        sources = []
        for citation in self.citations:
            if citation.document_filename not in seen:
                seen.add(citation.document_filename)
                sources.append(citation.document_title or citation.document_filename)
        return sources


class RAGEngine:
    """
    Retrieval engine that combines semantic and keyword search.

    Features:
    - Hybrid search (vector + FTS5)
    - Automatic context expansion
    - Citation generation
    - Content grouping by document structure
    """

    def __init__(
        self,
        db_path: Path,
        embedding_client: EmbeddingClient = None,
        default_k: int = 5,
        hybrid_weight: float = 0.7
    ):
        """
        Initialize the RAG engine.

        Args:
            db_path: Path to SQLite database
            embedding_client: Optional pre-configured embedding client
            default_k: Default number of results to return
            hybrid_weight: Weight for vector vs keyword search (0-1)
        """
        self.db_path = Path(db_path)
        self.default_k = default_k
        self.hybrid_weight = hybrid_weight

        # Initialize components
        self.vector_store = VectorStore(self.db_path)
        self.embedder = embedding_client or EmbeddingClient(EmbeddingConfig())

    def query(
        self,
        query_text: str,
        k: int = None,
        skill_filter: str = None,
        category_filter: str = None,
        content_type_filter: str = None,
        include_context: bool = True,
        context_window: int = 1
    ) -> RAGResult:
        """
        Execute a RAG query.

        Args:
            query_text: Search query
            k: Number of results (default: self.default_k)
            skill_filter: Filter by skill (apex, flow, lwc, etc.)
            category_filter: Filter by document category
            content_type_filter: Filter by content type (prose, code, table)
            include_context: Include surrounding chunks
            context_window: Number of chunks before/after to include

        Returns:
            RAGResult with chunks and citations
        """
        import time
        start_time = time.time()

        k = k or self.default_k

        # Generate query embedding
        query_embedding = self.embedder.embed_query(query_text)

        # Build filters
        filters = {}
        if category_filter:
            filters['category'] = category_filter
        if content_type_filter:
            filters['content_type'] = content_type_filter

        # Perform hybrid search
        results = self.vector_store.hybrid_search(
            query_text=query_text,
            query_embedding=query_embedding.tolist(),
            k=k * 2 if include_context else k,  # Get extra for dedup after context
            vector_weight=self.hybrid_weight,
            filters=filters if filters else None
        )

        # Expand context if requested
        if include_context and results:
            results = self._expand_context(results, context_window)

        # Limit to k results
        results = results[:k]

        # Generate citations
        citations = self._generate_citations(results)

        # Group by document structure
        grouped = self._group_by_structure(results)

        elapsed_ms = (time.time() - start_time) * 1000

        return RAGResult(
            query=query_text,
            chunks=results,
            citations=citations,
            total_matches=len(results),
            search_time_ms=elapsed_ms,
            grouped_content=grouped
        )

    def _expand_context(
        self,
        results: List[Dict],
        window: int = 1
    ) -> List[Dict]:
        """
        Expand results to include surrounding chunks.
        """
        expanded = []
        seen_ids = set()

        for result in results:
            chunk_id = result.get('chunk_id') or result.get('id', '')

            # Get context chunks
            context = self.vector_store.get_chunk_context(
                chunk_id=chunk_id,
                before=window,
                after=window
            )

            for chunk in context:
                cid = chunk.get('chunk_id') or chunk.get('id', '')
                if cid not in seen_ids:
                    # Mark context chunks
                    if cid != chunk_id:
                        chunk['is_context'] = True
                        chunk['context_for'] = chunk_id
                    else:
                        chunk['is_context'] = False
                        chunk['score'] = result.get('score', 0)

                    # Ensure chunk_id is set for consistency
                    chunk['chunk_id'] = cid
                    expanded.append(chunk)
                    seen_ids.add(cid)

        # Sort by document position
        expanded.sort(key=lambda x: (
            x.get('document_id', ''),
            x.get('page_start', 0),
            x.get('chunk_index', 0)
        ))

        return expanded

    def _generate_citations(self, results: List[Dict]) -> List[Citation]:
        """Generate citations for search results."""
        citations = []
        seen = set()

        for result in results:
            # Create unique key for dedup
            key = (
                result.get('document_id'),
                result.get('page_start'),
                result.get('chapter', '')
            )

            if key in seen:
                continue
            seen.add(key)

            # Get document info
            doc_info = self.vector_store.get_document(result.get('document_id', ''))

            citations.append(Citation(
                document_title=doc_info.get('title', '') if doc_info else '',
                document_filename=doc_info.get('filename', '') if doc_info else '',
                page_start=result.get('page_start', 0),
                page_end=result.get('page_end', 0),
                chapter=result.get('chapter', ''),
                section=result.get('section', ''),
                chunk_id=result.get('chunk_id') or result.get('id', '')
            ))

        return citations

    def _group_by_structure(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group results by document structure."""
        grouped = {}

        for result in results:
            chapter = result.get('chapter', 'Uncategorized')
            section = result.get('section', '')

            key = chapter
            if section:
                key = f"{chapter} > {section}"

            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)

        return grouped

    def search_by_topic(
        self,
        topic: str,
        skill: str = None,
        k: int = 10
    ) -> RAGResult:
        """
        Search for chunks related to a specific topic.

        Optimized for broader topic-based retrieval rather than
        specific question answering.
        """
        # Use topic as the query, but weight keywords more heavily
        return self.query(
            query_text=topic,
            k=k,
            skill_filter=skill,
            include_context=True,
            context_window=2  # More context for topic exploration
        )

    def get_document_summary(self, document_id: str) -> Dict:
        """
        Get a summary of chunks from a specific document.
        """
        chunks = self.vector_store.get_chunks_by_document(document_id)

        # Group by chapter/section
        structure = {}
        for chunk in chunks:
            chapter = chunk.get('chapter', 'Main Content')
            section = chunk.get('section', '')

            if chapter not in structure:
                structure[chapter] = {'sections': {}, 'chunks': []}

            if section:
                if section not in structure[chapter]['sections']:
                    structure[chapter]['sections'][section] = []
                structure[chapter]['sections'][section].append(chunk)
            else:
                structure[chapter]['chunks'].append(chunk)

        return {
            'document_id': document_id,
            'total_chunks': len(chunks),
            'structure': structure
        }

    def log_query(self, result: RAGResult, skill_context: str = None):
        """Log a query for analytics."""
        self.vector_store.log_query(
            query_text=result.query,
            result_count=result.total_matches,
            top_chunk_ids=[c.get('chunk_id') or c.get('id', '') for c in result.chunks[:5]],
            latency_ms=int(result.search_time_ms),
            skill_context=skill_context
        )

    def get_stats(self) -> Dict:
        """Get RAG engine statistics."""
        store_stats = self.vector_store.get_stats()
        model_info = self.embedder.get_model_info()

        return {
            **store_stats,
            'embedding_model': model_info,
            'default_k': self.default_k,
            'hybrid_weight': self.hybrid_weight
        }


def create_rag_engine(
    db_path: Path,
    model_name: str = "BAAI/bge-large-en-v1.5",
    **kwargs
) -> RAGEngine:
    """
    Factory function to create a configured RAG engine.

    Args:
        db_path: Path to SQLite database
        model_name: Embedding model name
        **kwargs: Additional RAGEngine options

    Returns:
        Configured RAGEngine instance
    """
    from ..embedder.embedding_client import create_client

    embedder = create_client(model_name=model_name)
    return RAGEngine(db_path=db_path, embedding_client=embedder, **kwargs)
