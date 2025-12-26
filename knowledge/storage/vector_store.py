"""
Vector store implementation using SQLite + sqlite-vec.

Provides:
- Document and chunk storage
- Vector similarity search
- Hybrid search (vector + FTS5 keyword)
- Metadata filtering
"""

import sqlite3
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class Document:
    """Represents an ingested PDF document."""
    id: str
    filename: str
    filepath: str
    title: Optional[str] = None
    file_size_bytes: int = 0
    page_count: int = 0
    content_hash: str = ""
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    status: str = "pending"


@dataclass
class Chunk:
    """Represents a text chunk from a document."""
    id: str
    document_id: str
    text: str
    page_start: int
    page_end: int
    chunk_index: int
    total_chunks: int
    chapter: str = ""
    section: str = ""
    subsection: str = ""
    content_type: str = "prose"
    topic_tags: Optional[List[str]] = None
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None


class VectorStore:
    """
    SQLite + sqlite-vec vector storage implementation.

    Features:
    - Document deduplication via content hash
    - Vector similarity search
    - Hybrid search (vector + keyword)
    - Metadata filtering
    - Transaction support
    """

    def __init__(self, db_path: Path, embedding_dim: int = 1024):
        """
        Initialize the vector store.

        Args:
            db_path: Path to SQLite database file
            embedding_dim: Dimension of embedding vectors (default: 1024 for bge-large)
        """
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = self._connect()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """Connect to database and load sqlite-vec extension."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        # Load sqlite-vec extension
        try:
            conn.enable_load_extension(True)
            # Try common extension locations
            for ext_name in ["vec0", "sqlite_vec", "sqlite-vec"]:
                try:
                    conn.load_extension(ext_name)
                    break
                except sqlite3.OperationalError:
                    continue
            else:
                # If none found, try importing sqlite_vec which auto-loads
                import sqlite_vec
                sqlite_vec.load(conn)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load sqlite-vec extension: {e}\n"
                "Install with: uv add sqlite-vec"
            ) from e

        return conn

    def _init_schema(self):
        """Initialize database schema."""
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path) as f:
            # Execute schema (excluding the vec0 virtual table which needs special handling)
            self.conn.executescript(f.read())

        # Create the vector table separately
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[{self.embedding_dim}]
            )
        """)

        self.conn.commit()

    # ═══════════════════════════════════════════════════════════════════════
    # Document Operations
    # ═══════════════════════════════════════════════════════════════════════

    def add_document(
        self,
        doc_id: str = None,
        filename: str = None,
        filepath: str = None,
        title: str = None,
        content_hash: str = None,
        page_count: int = None,
        file_size: int = None,
        category: str = None,
        tags: List[str] = None
    ) -> str:
        """
        Register a new document for processing.

        Args:
            doc_id: Document ID (generated if not provided)
            filename: PDF filename
            filepath: Full path to PDF file
            title: Optional document title
            content_hash: SHA256 hash of content
            page_count: Number of pages
            file_size: File size in bytes
            category: Category (apex, flow, lwc, etc.)
            tags: List of topic tags

        Returns:
            Document ID
        """
        # Generate ID if not provided
        if not doc_id:
            doc_id = f"doc_{content_hash[:16] if content_hash else hashlib.sha256(filepath.encode()).hexdigest()[:16]}"

        self.conn.execute("""
            INSERT INTO documents (
                id, filename, filepath, title, content_hash,
                file_size_bytes, page_count, category, tags, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """, (
            doc_id,
            filename,
            filepath,
            title or (Path(filepath).stem if filepath else 'Untitled'),
            content_hash,
            file_size or 0,
            page_count or 0,
            category,
            json.dumps(tags or [])
        ))
        self.conn.commit()

        return doc_id

    def update_document_status(self, doc_id: str, status: str,
                                error_message: str = None, page_count: int = None):
        """Update document processing status."""
        updates = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        params = [status]

        if error_message:
            updates.append("error_message = ?")
            params.append(error_message)

        if page_count is not None:
            updates.append("page_count = ?")
            params.append(page_count)

        params.append(doc_id)

        self.conn.execute(
            f"UPDATE documents SET {', '.join(updates)} WHERE id = ?",
            params
        )
        self.conn.commit()

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        )
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result["tags"] = json.loads(result["tags"]) if result["tags"] else []
            return result
        return None

    def get_document_by_hash(self, content_hash: str) -> Optional[Dict]:
        """Get document by content hash."""
        cursor = self.conn.execute(
            "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
        )
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result["tags"] = json.loads(result["tags"]) if result["tags"] else []
            return result
        return None

    def document_exists(self, filepath: Path) -> bool:
        """Check if document already ingested (by content hash)."""
        content_hash = self._compute_hash(filepath)
        return self.get_document_by_hash(content_hash) is not None

    def list_documents(self, status: str = None, category: str = None) -> List[Dict]:
        """List documents with optional filtering."""
        query = "SELECT * FROM documents WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY created_at DESC"

        cursor = self.conn.execute(query, params)
        results = []
        for row in cursor:
            result = dict(row)
            result["tags"] = json.loads(result["tags"]) if result["tags"] else []
            results.append(result)
        return results

    # ═══════════════════════════════════════════════════════════════════════
    # Chunk Operations
    # ═══════════════════════════════════════════════════════════════════════

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """
        Add chunks with their embeddings in a transaction.

        Args:
            chunks: List of Chunk objects
            embeddings: List of embedding vectors (must match chunks length)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match")

        cursor = self.conn.cursor()

        try:
            for chunk, embedding in zip(chunks, embeddings):
                # Insert chunk
                cursor.execute("""
                    INSERT INTO chunks (
                        id, document_id, text, text_length, token_count,
                        page_start, page_end, chunk_index, total_chunks,
                        chapter, section, subsection, content_type,
                        topic_tags, prev_chunk_id, next_chunk_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.id,
                    chunk.document_id,
                    chunk.text,
                    len(chunk.text),
                    len(chunk.text.split()),  # Rough token count
                    chunk.page_start,
                    chunk.page_end,
                    chunk.chunk_index,
                    chunk.total_chunks,
                    chunk.chapter,
                    chunk.section,
                    chunk.subsection,
                    chunk.content_type,
                    json.dumps(chunk.topic_tags or []),
                    chunk.prev_chunk_id,
                    chunk.next_chunk_id
                ))

                # Insert embedding
                cursor.execute("""
                    INSERT INTO chunk_embeddings (chunk_id, embedding)
                    VALUES (?, ?)
                """, (chunk.id, json.dumps(embedding)))

            self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to add chunks: {e}") from e

    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        """Get chunk by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        )
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result["topic_tags"] = json.loads(result["topic_tags"]) if result["topic_tags"] else []
            return result
        return None

    def get_chunks_by_document(self, doc_id: str) -> List[Dict]:
        """Get all chunks for a document."""
        cursor = self.conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (doc_id,)
        )
        results = []
        for row in cursor:
            result = dict(row)
            result["topic_tags"] = json.loads(result["topic_tags"]) if result["topic_tags"] else []
            results.append(result)
        return results

    def add_chunk(
        self,
        chunk_id: str,
        document_id: str,
        text: str,
        embedding: List[float],
        page_start: int = 0,
        page_end: int = 0,
        chunk_index: int = 0,
        total_chunks: int = 0,
        chapter: str = "",
        section: str = "",
        subsection: str = "",
        content_type: str = "prose",
        topic_tags: List[str] = None,
        prev_chunk_id: str = None,
        next_chunk_id: str = None
    ):
        """
        Add a single chunk with its embedding.

        Args:
            chunk_id: Unique chunk identifier
            document_id: Parent document ID
            text: Chunk text content
            embedding: Embedding vector
            page_start: Starting page number
            page_end: Ending page number
            chunk_index: Position in document
            total_chunks: Total chunks in document
            chapter: Chapter heading
            section: Section heading
            subsection: Subsection heading
            content_type: prose/code/table/list
            topic_tags: List of topic tags
            prev_chunk_id: Previous chunk ID
            next_chunk_id: Next chunk ID
        """
        # Insert chunk
        self.conn.execute("""
            INSERT INTO chunks (
                id, document_id, text, text_length, token_count,
                page_start, page_end, chunk_index, total_chunks,
                chapter, section, subsection, content_type,
                topic_tags, prev_chunk_id, next_chunk_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk_id,
            document_id,
            text,
            len(text),
            len(text) // 4,  # Rough token estimate
            page_start,
            page_end,
            chunk_index,
            total_chunks,
            chapter,
            section,
            subsection,
            content_type,
            json.dumps(topic_tags or []),
            prev_chunk_id,
            next_chunk_id
        ))

        # Insert embedding
        self.conn.execute("""
            INSERT INTO chunk_embeddings (chunk_id, embedding)
            VALUES (?, ?)
        """, (chunk_id, json.dumps(embedding)))

        self.conn.commit()

    def delete_document(self, doc_id: str):
        """Delete a document and all its chunks."""
        # Chunks will be deleted by CASCADE
        self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        # Clean up orphaned embeddings
        self.conn.execute("""
            DELETE FROM chunk_embeddings
            WHERE chunk_id NOT IN (SELECT id FROM chunks)
        """)
        self.conn.commit()

    def get_chunk_context(
        self,
        chunk_id: str,
        before: int = 1,
        after: int = 1
    ) -> List[Dict]:
        """
        Get a chunk and its surrounding context.

        Args:
            chunk_id: The central chunk ID
            before: Number of chunks before to include
            after: Number of chunks after to include

        Returns:
            List of chunks in order
        """
        # Get the central chunk
        chunk = self.get_chunk(chunk_id)
        if not chunk:
            return []

        doc_id = chunk["document_id"]
        chunk_index = chunk["chunk_index"]

        # Get range of chunks
        start_index = max(0, chunk_index - before)
        end_index = chunk_index + after

        cursor = self.conn.execute("""
            SELECT * FROM chunks
            WHERE document_id = ?
              AND chunk_index >= ?
              AND chunk_index <= ?
            ORDER BY chunk_index
        """, (doc_id, start_index, end_index))

        results = []
        for row in cursor:
            result = dict(row)
            result["topic_tags"] = json.loads(result["topic_tags"]) if result["topic_tags"] else []
            result["chunk_id"] = result["id"]  # Alias for consistency
            results.append(result)

        return results

    # ═══════════════════════════════════════════════════════════════════════
    # Search Operations
    # ═══════════════════════════════════════════════════════════════════════

    def search(self,
               query_embedding: List[float],
               k: int = 10,
               filters: Dict[str, Any] = None,
               include_context: bool = True) -> List[Dict]:
        """
        Semantic vector search.

        Args:
            query_embedding: Query vector (1024 dimensions)
            k: Number of results
            filters: Optional filters (document_id, chapter, content_type, category)
            include_context: Include surrounding chunk text

        Returns:
            List of results with chunk data, similarity score, and citations
        """
        start_time = time.time()

        # Vector similarity search
        cursor = self.conn.execute("""
            SELECT
                c.*,
                d.filename,
                d.title as document_title,
                d.category,
                e.distance
            FROM chunk_embeddings e
            JOIN chunks c ON e.chunk_id = c.id
            JOIN documents d ON c.document_id = d.id
            WHERE e.embedding MATCH ?
              AND k = ?
            ORDER BY e.distance
        """, (json.dumps(query_embedding), k * 2))  # Get more for filtering

        results = []
        for row in cursor:
            result = dict(row)

            # Apply filters
            if filters:
                if filters.get("document_id") and result["document_id"] != filters["document_id"]:
                    continue
                if filters.get("chapter") and result["chapter"] != filters["chapter"]:
                    continue
                if filters.get("content_type") and result["content_type"] != filters["content_type"]:
                    continue
                if filters.get("category") and result["category"] != filters["category"]:
                    continue

            # Convert distance to similarity (cosine distance to similarity)
            result["similarity"] = 1 - result["distance"]
            result["topic_tags"] = json.loads(result["topic_tags"]) if result["topic_tags"] else []

            # Build citation
            result["citation"] = {
                "source": result["filename"],
                "page": result["page_start"],
                "chapter": result["chapter"],
                "section": result["section"]
            }

            # Include surrounding context if requested
            if include_context:
                result["context"] = self._get_surrounding_context(
                    result["prev_chunk_id"],
                    result["next_chunk_id"]
                )

            results.append(result)

            if len(results) >= k:
                break

        # Log query
        latency_ms = int((time.time() - start_time) * 1000)
        self._log_query("", None, len(results),
                       [r["id"] for r in results[:5]], latency_ms)

        return results

    def hybrid_search(self,
                      query_text: str,
                      query_embedding: List[float],
                      k: int = 10,
                      vector_weight: float = 0.7,
                      filters: Dict[str, Any] = None) -> List[Dict]:
        """
        Hybrid search combining vector similarity and keyword matching.

        Args:
            query_text: Text query for FTS
            query_embedding: Query vector for semantic search
            k: Number of results
            vector_weight: Weight for vector vs keyword (0-1)
            filters: Optional metadata filters

        Returns:
            List of results ranked by combined score
        """
        start_time = time.time()

        # Vector search (get more for fusion)
        vector_results = self.search(query_embedding, k * 2, filters, include_context=False)

        # FTS keyword search (sanitize query for FTS5 syntax)
        fts_query = self._sanitize_fts_query(query_text)
        fts_scores = {}

        if fts_query:  # Only run FTS if we have valid query terms
            try:
                fts_cursor = self.conn.execute("""
                    SELECT c.id, bm25(chunks_fts) as fts_score
                    FROM chunks_fts
                    JOIN chunks c ON chunks_fts.rowid = c.rowid
                    WHERE chunks_fts MATCH ?
                    ORDER BY bm25(chunks_fts)
                    LIMIT ?
                """, (fts_query, k * 2))
                fts_scores = {row["id"]: row["fts_score"] for row in fts_cursor}
            except sqlite3.OperationalError:
                # Fall back to vector-only if FTS fails
                pass

        # Reciprocal Rank Fusion
        combined_scores = {}
        for i, result in enumerate(vector_results):
            chunk_id = result["id"]
            vector_rank = i + 1
            fts_rank = (list(fts_scores.keys()).index(chunk_id) + 1
                       if chunk_id in fts_scores else 100)

            # RRF formula
            rrf_score = (vector_weight / (60 + vector_rank)) + \
                       ((1 - vector_weight) / (60 + fts_rank))

            combined_scores[chunk_id] = (rrf_score, result)

        # Sort by combined score
        sorted_results = sorted(combined_scores.values(), key=lambda x: -x[0])

        # Add context and return top k
        final_results = []
        for _, result in sorted_results[:k]:
            result["context"] = self._get_surrounding_context(
                result.get("prev_chunk_id"),
                result.get("next_chunk_id")
            )
            final_results.append(result)

        # Log query
        latency_ms = int((time.time() - start_time) * 1000)
        self._log_query(query_text, None, len(final_results),
                       [r["id"] for r in final_results[:5]], latency_ms)

        return final_results

    def _get_surrounding_context(self, prev_id: str, next_id: str) -> Dict:
        """Get text from surrounding chunks."""
        context = {"prev": None, "next": None}

        if prev_id:
            cursor = self.conn.execute(
                "SELECT text FROM chunks WHERE id = ?", (prev_id,)
            )
            row = cursor.fetchone()
            if row:
                text = row["text"]
                context["prev"] = text[:300] + "..." if len(text) > 300 else text

        if next_id:
            cursor = self.conn.execute(
                "SELECT text FROM chunks WHERE id = ?", (next_id,)
            )
            row = cursor.fetchone()
            if row:
                text = row["text"]
                context["next"] = text[:300] + "..." if len(text) > 300 else text

        return context

    def _log_query(self, query_text: str, skill_context: str,
                   result_count: int, top_ids: List[str], latency_ms: int):
        """Log query for analytics (internal)."""
        try:
            self.conn.execute("""
                INSERT INTO query_log (query_text, skill_context, result_count,
                                       top_chunk_ids, latency_ms)
                VALUES (?, ?, ?, ?, ?)
            """, (query_text, skill_context, result_count,
                  json.dumps(top_ids), latency_ms))
            self.conn.commit()
        except Exception:
            pass  # Don't fail on logging errors

    def log_query(
        self,
        query_text: str,
        result_count: int,
        top_chunk_ids: List[str],
        latency_ms: int,
        skill_context: str = None
    ):
        """
        Log a query for analytics (public interface).

        Args:
            query_text: The search query
            result_count: Number of results returned
            top_chunk_ids: IDs of top result chunks
            latency_ms: Query latency in milliseconds
            skill_context: Optional skill context
        """
        self._log_query(query_text, skill_context, result_count, top_chunk_ids, latency_ms)

    # ═══════════════════════════════════════════════════════════════════════
    # Export Operations
    # ═══════════════════════════════════════════════════════════════════════

    def log_export(self, topic: str, skill: str, output_path: str, chunk_ids: List[str]):
        """Log an export for tracking."""
        self.conn.execute("""
            INSERT INTO exports (topic, skill, output_path, chunk_ids)
            VALUES (?, ?, ?, ?)
        """, (topic, skill, output_path, json.dumps(chunk_ids)))
        self.conn.commit()

    # ═══════════════════════════════════════════════════════════════════════
    # Statistics
    # ═══════════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}

        # Document count
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM documents")
        stats["document_count"] = cursor.fetchone()["count"]

        # Chunk count
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM chunks")
        stats["chunk_count"] = cursor.fetchone()["count"]

        # Embedding count
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM chunk_embeddings")
        stats["embedding_count"] = cursor.fetchone()["count"]

        # Database size
        if self.db_path.exists():
            stats["db_size_mb"] = self.db_path.stat().st_size / (1024 * 1024)
        else:
            stats["db_size_mb"] = 0

        # By category
        cursor = self.conn.execute("""
            SELECT category, COUNT(*) as count
            FROM documents
            GROUP BY category
        """)
        stats["categories"] = {row["category"] or "uncategorized": row["count"] for row in cursor}

        # By status
        cursor = self.conn.execute("""
            SELECT status, COUNT(*) as count
            FROM documents
            GROUP BY status
        """)
        stats["by_status"] = {row["status"]: row["count"] for row in cursor}

        # Recent queries
        cursor = self.conn.execute("""
            SELECT query_text, result_count, latency_ms, created_at
            FROM query_log
            ORDER BY created_at DESC
            LIMIT 10
        """)
        stats["recent_queries"] = [dict(row) for row in cursor]

        return stats

    # ═══════════════════════════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of file content."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _sanitize_fts_query(self, query: str) -> str:
        """
        Sanitize query text for FTS5 syntax.

        FTS5 has special syntax characters that need escaping:
        - Removes: * ? " ( ) : ^ -
        - Converts to simple word search with OR logic
        """
        import re

        # Remove FTS5 special characters
        sanitized = re.sub(r'[*?"():^-]', ' ', query)

        # Split into words and rejoin with OR for broader matching
        words = sanitized.split()
        if not words:
            return ""

        # Use simple word matching (implicit AND by default in FTS5)
        # Quote each word to treat as literal
        return ' '.join(f'"{w}"' for w in words if len(w) > 1)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
