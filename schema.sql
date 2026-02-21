-- schema.sql — Standalone DDL for the exam_bank table.
-- Run this directly against PostgreSQL if you prefer manual setup.

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the main table
CREATE TABLE IF NOT EXISTS exam_bank (
    id              SERIAL PRIMARY KEY,

    -- metadata
    year            INTEGER       NOT NULL,
    season          VARCHAR(20)   NOT NULL,
    paper_code      VARCHAR(20)   NOT NULL,
    question_num    INTEGER       NOT NULL,
    total_marks     INTEGER       NOT NULL DEFAULT 0,

    -- classification
    topic           VARCHAR(100)  NOT NULL DEFAULT 'Unknown', 
    subtopic        VARCHAR(100)  NOT NULL DEFAULT 'Unknown',

    -- content
    question_text        TEXT NOT NULL,
    mark_scheme_text     TEXT NOT NULL DEFAULT '',
    examiner_report_text TEXT NOT NULL DEFAULT '',

    -- embedding (OpenAI text-embedding-3-small → 1536 dimensions)
    embedding       vector(1536),

    -- prevent accidental duplicates
    UNIQUE (year, season, paper_code, question_num)
);

-- 3. HNSW index for fast approximate nearest-neighbour search (cosine)
CREATE INDEX IF NOT EXISTS idx_exam_bank_embedding
    ON exam_bank
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- 4. B-tree indexes for metadata filtering
CREATE INDEX IF NOT EXISTS idx_exam_bank_year      ON exam_bank (year);
CREATE INDEX IF NOT EXISTS idx_exam_bank_topic     ON exam_bank (topic);
CREATE INDEX IF NOT EXISTS idx_exam_bank_subtopic  ON exam_bank (subtopic);
CREATE INDEX IF NOT EXISTS idx_exam_bank_paper     ON exam_bank (paper_code);
