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

-- Phase 1: NO ANN index. Use exact (brute-force) cosine search.
-- Corpus is ~1-3k rows; ANN would hurt recall at this scale.
-- Add HNSW/IVFFlat later when corpus grows, with lists ≈ rows/1000.
