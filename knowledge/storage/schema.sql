-- sf-knowledge-tools Database Schema
-- SQLite + sqlite-vec for vector similarity search

-- ═══════════════════════════════════════════════════════════════════════════
-- DOCUMENTS TABLE
-- Tracks ingested PDF documents with deduplication via content hash
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL,
    title TEXT,

    -- Document metadata
    file_size_bytes INTEGER,
    page_count INTEGER,

    -- Version tracking (SHA256 of PDF content)
    content_hash TEXT NOT NULL UNIQUE,
    version INTEGER DEFAULT 1,

    -- Processing status: pending, processing, completed, failed
    status TEXT DEFAULT 'pending',
    error_message TEXT,

    -- Categorization
    category TEXT,                    -- apex, flow, lwc, etc.
    tags TEXT,                        -- JSON array of tags

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ═══════════════════════════════════════════════════════════════════════════
-- CHUNKS TABLE
-- Stores text chunks with full metadata for filtering and context
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,

    -- Content
    text TEXT NOT NULL,
    text_length INTEGER,
    token_count INTEGER,

    -- Location in source document
    page_start INTEGER,
    page_end INTEGER,
    chunk_index INTEGER,
    total_chunks INTEGER,

    -- Document structure
    chapter TEXT,
    section TEXT,
    subsection TEXT,

    -- Content classification: prose, code, table, list
    content_type TEXT DEFAULT 'prose',

    -- Topic tags for filtering (JSON array)
    topic_tags TEXT,

    -- Chunk linking for context retrieval
    prev_chunk_id TEXT,
    next_chunk_id TEXT,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign key
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- ═══════════════════════════════════════════════════════════════════════════
-- VECTOR EMBEDDINGS TABLE (sqlite-vec virtual table)
-- Stores 1024-dimensional embeddings for semantic search
-- ═══════════════════════════════════════════════════════════════════════════

-- Note: This table is created programmatically after loading sqlite-vec extension
-- CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
--     chunk_id TEXT PRIMARY KEY,
--     embedding FLOAT[1024]
-- );

-- ═══════════════════════════════════════════════════════════════════════════
-- FULL-TEXT SEARCH INDEX (FTS5)
-- For hybrid search combining vector similarity + keyword matching
-- ═══════════════════════════════════════════════════════════════════════════

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    chapter,
    section,
    content='chunks',
    content_rowid='rowid'
);

-- Triggers to keep FTS index in sync with chunks table
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, chapter, section)
    VALUES (NEW.rowid, NEW.text, NEW.chapter, NEW.section);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, chapter, section)
    VALUES('delete', OLD.rowid, OLD.text, OLD.chapter, OLD.section);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, chapter, section)
    VALUES('delete', OLD.rowid, OLD.text, OLD.chapter, OLD.section);
    INSERT INTO chunks_fts(rowid, text, chapter, section)
    VALUES (NEW.rowid, NEW.text, NEW.chapter, NEW.section);
END;

-- ═══════════════════════════════════════════════════════════════════════════
-- QUERY LOG TABLE
-- For analytics and debugging
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    skill_context TEXT,
    result_count INTEGER,
    top_chunk_ids TEXT,               -- JSON array
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ═══════════════════════════════════════════════════════════════════════════
-- EXPORT HISTORY TABLE
-- Tracks what has been exported for PR tracking
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS exports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    skill TEXT NOT NULL,
    output_path TEXT,
    chunk_ids TEXT,                   -- JSON array of source chunks
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ═══════════════════════════════════════════════════════════════════════════
-- INDEXES
-- ═══════════════════════════════════════════════════════════════════════════

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON chunks(chapter);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section);
CREATE INDEX IF NOT EXISTS idx_chunks_content_type ON chunks(content_type);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_start, page_end);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
