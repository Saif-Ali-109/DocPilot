CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS chunks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content       TEXT NOT NULL,
    heading_path  TEXT,
    source_file   TEXT NOT NULL,
    chunk_index   INT NOT NULL,
    language      TEXT NOT NULL DEFAULT 'en',  -- derived from source_file by pipeline/backfill
    metadata      JSONB DEFAULT '{}',
    embedding     vector(384) NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Phase 5 corpus hardening: prevents double-inserted chunks.
-- Two overlapping/crashed ingest runs once produced full-file twin rows
-- (identical source_file + heading_path + chunk_index). This btree index is
-- NOT an ANN index — it only enforces logical-chunk uniqueness. It must be
-- created *after* the corpus is deduplicated: creating it on a dirty table
-- fails (unique violation). Use `python -m docpilot dedupe` to clean first.
CREATE UNIQUE INDEX IF NOT EXISTS chunks_unique_triple
    ON chunks (source_file, heading_path, chunk_index);

-- Phase 1: NO ANN index. Use exact (brute-force) cosine search.
-- Corpus is ~1-3k rows; ANN would hurt recall at this scale.
-- Add HNSW/IVFFlat later when corpus grows, with lists ≈ rows/1000.

-- Phase H (pre-Phase-6 hardening): lexical side of hybrid retrieval.
-- Expression GIN index over the English tsvector of chunk content so
-- Postgres full-text search (plainto_tsquery + @@) is index-backed.
-- Referenced by retriever/lexical.py (PostgresFTSSearcher).
CREATE INDEX IF NOT EXISTS chunks_content_fts
    ON chunks USING GIN (to_tsvector('english', content));
