-- =============================================================
-- 03 · RAG Foundation · Table Creation DDL
-- New tables in this chapter:
--   knowledge_chunks (authoritative source for knowledge base content)
--   qa_extraction_staging (staging table for QA extracted from historical conversations)
--
-- Vectors are stored in the Milvus collection "knowledge" rather than MySQL,
-- so Milvus schema is not included in this DDL.
-- MySQL stores the source text and vectorization synchronization status.
--
-- category + questions + answer are concatenated into the text used for embedding.
-- All other fields are metadata only and are not included in the embedding text.
-- =============================================================

-- Ensure all text and comments are interpreted as utf8mb4.
-- A MySQL client using latin1 by default may otherwise double-encode non-ASCII text.
SET NAMES utf8mb4;

CREATE TABLE knowledge_chunks (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'Chunk primary key; aligned with the primary key in the Milvus collection',
  category         VARCHAR(255)    NOT NULL                COMMENT 'Category or parent heading path; included in embedding text',
  questions        TEXT            NOT NULL                COMMENT 'Questions or section title; multiple questions are separated by line breaks; included in embedding text',
  answer           TEXT            NOT NULL                COMMENT 'Main answer content; included in embedding text',
  section_path     VARCHAR(512)    NULL                    COMMENT 'Section path used for traceability; metadata only and not included in embedding text',
  content_type     VARCHAR(32)     NULL                    COMMENT 'Content type such as faq, policy, or manual; metadata only',
  is_key_clause    TINYINT(1)      NOT NULL DEFAULT 0      COMMENT 'Whether this chunk is a key clause: 0 = no, 1 = yes; metadata only',
  prev_chunk_id    BIGINT UNSIGNED NULL                    COMMENT 'Pointer to the previous chunk; metadata only',
  next_chunk_id    BIGINT UNSIGNED NULL                    COMMENT 'Pointer to the next chunk; metadata only',
  vector_id        VARCHAR(64)     NULL                    COMMENT 'Primary key in the Milvus knowledge collection; populated after vector insertion',
  vectorize_status ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT 'Vectorization status: pending or completed; used to support idempotent dual writes',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
  updated_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last updated time',
  PRIMARY KEY (id),
  KEY idx_category (category),
  KEY idx_vectorize_status (vectorize_status),
  CONSTRAINT fk_chunks_prev
    FOREIGN KEY (prev_chunk_id)
    REFERENCES knowledge_chunks (id)
    ON DELETE SET NULL,
  CONSTRAINT fk_chunks_next
    FOREIGN KEY (next_chunk_id)
    REFERENCES knowledge_chunks (id)
    ON DELETE SET NULL
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COMMENT='Authoritative source table for knowledge base chunks';

CREATE TABLE qa_extraction_staging (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'Staging row primary key',
  batch_no         VARCHAR(64)     NOT NULL                COMMENT 'Extraction batch identifier; conversations are processed in batches to reduce cross-conversation contamination and enable batch-level traceability',
  source_ref       VARCHAR(255)    NULL                    COMMENT 'Source conversation or exported file identifier; used for traceability and not copied into the final knowledge base',
  question         TEXT            NOT NULL                COMMENT 'User question extracted from the conversation by the LLM',
  answer           TEXT            NOT NULL                COMMENT 'Customer support answer extracted from the conversation by the LLM',
  status           ENUM('extracted','kept','discarded') NOT NULL DEFAULT 'extracted' COMMENT 'Processing status: extracted and awaiting deduplication, retained after deduplication, or discarded after deduplication',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Extraction timestamp',
  PRIMARY KEY (id),
  KEY idx_batch_no (batch_no),
  KEY idx_status (status)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COMMENT='Offline staging table for QA pairs extracted from historical conversations; used for batch extraction and global deduplication before retained items are written to knowledge_chunks; may be cleared after knowledge base construction';
