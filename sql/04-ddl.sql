-- =============================================================
-- ch04 · Advanced RAG · Table Creation DDL
-- New tables in this chapter:
--   low_confidence_questions (low-confidence question pool)
--   faith_cases (hallucination case ledger)
--
-- During generation, if the model determines that the available knowledge
-- is insufficient to answer reliably, it refuses to answer and stores the
-- user's original question in this pool. This becomes an entry point for
-- the ch09 data flywheel.
--
-- In this chapter, pool insertion is determined only by the useful
-- self-evaluation result (only useful=false is inserted; the field itself
-- is not stored separately). ch09 will ALTER this table to add fields
-- used for aggregation and merging.
-- =============================================================

-- Ensure ENUM values, DEFAULT values, and COMMENT text are parsed as utf8mb4.
-- Otherwise, a MySQL client using latin1 by default may double-encode
-- non-ASCII text and corrupt stored ENUM values.
SET NAMES utf8mb4;

CREATE TABLE low_confidence_questions (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'Primary key',
  conversation_id BIGINT UNSIGNED NULL                    COMMENT 'Source conversation',
  raw_question    TEXT            NOT NULL                COMMENT 'Original user wording, including emotional or colloquial expressions',
  source          ENUM('retrieval_low_conf','self_check','user_feedback') NOT NULL COMMENT 'Entry source: low retrieval confidence / insufficient generation self-check / unresolved user feedback',
  reason          TEXT            NULL                    COMMENT 'Reason why the system determined it could not answer reliably; retained for later review',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Time added to the pool',
  PRIMARY KEY (id),
  KEY idx_source (source),
  KEY idx_created_at (created_at),
  CONSTRAINT fk_lcq_conversation
    FOREIGN KEY (conversation_id)
    REFERENCES conversations (id)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COMMENT='Low-confidence question pool';

-- ch04 hallucination case ledger:
-- every hallucination identified by the faithfulness judge is stored here
-- for long-term management instead of existing only in a single evaluation report.
--
-- Why this table is needed:
-- reports are generated artifacts and may be overwritten by the next run.
-- Without a persistent ledger, previously identified hallucination cases
-- and their resolution states would disappear.
--
-- The real value of these cases comes from tracking them across evaluation runs.
-- If the same question is repeatedly judged as hallucinated, it indicates
-- that the corresponding knowledge gap has still not been fixed correctly.
--
-- One row per evaluation question (uk_eval_id):
-- when the same question is judged as hallucinated again, no new row is created.
-- Instead, the answer, reason, and citation snapshot are updated to the latest
-- version and seen_count is incremented.
--
-- If a case previously marked as resolved is judged as hallucinated again,
-- its status automatically returns to unresolved. This is considered a recurrence,
-- not a new issue, and the case must reappear in the pending work list.
--
-- citations stores the complete Top-K evidence set provided to the model
-- during that evaluation run. Citation markers such as [n] in the generated
-- answer correspond to the sequence numbers in this list.
--
-- When the judge says that a claim is unsupported by the evidence,
-- reviewers must be able to inspect exactly what evidence was available
-- to the model at that time, rather than retaining only the judge's conclusion.
--
-- It must also be possible to distinguish between all evidence that was
-- available and the subset actually cited by the answer. Evidence that was
-- not cited is still relevant to later analysis.
CREATE TABLE faith_cases (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'Primary key',
  eval_id       VARCHAR(16)     NOT NULL                COMMENT 'Evaluation dataset question ID, e.g. A43; one row per question',
  bucket        VARCHAR(24)     NOT NULL                COMMENT 'Evaluation bucket: A_policy / B_model / C_colloquial / E_multi',
  query         VARCHAR(512)    NOT NULL                COMMENT 'Original user question',
  strategy      VARCHAR(24)     NOT NULL DEFAULT 'hybrid_rerank' COMMENT 'Retrieval strategy used to generate this answer',
  answer        TEXT            NOT NULL                COMMENT 'Generated answer version judged to contain hallucination',
  reason        TEXT            NOT NULL                COMMENT 'Reason provided by the judge, identifying the hallucinated claim',
  citations     JSON            NULL                    COMMENT 'Snapshot of the complete Top-K evidence set provided to the model in this run: [{n,chunk_id,section_path,question,answer}]; citation marker [n] in the answer refers to the sequence number in this list; the answer usually cites only two or three entries; legacy rows without snapshots remain NULL',
  judge_model   VARCHAR(64)     NULL                    COMMENT 'Judge model used for this case',
  status        ENUM('unresolved','resolved','no_action_needed') NOT NULL DEFAULT 'unresolved' COMMENT 'Resolution status; updated manually',
  seen_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT 'Total number of times this case has been judged as hallucinated across runs',
  first_seen_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Time this case was first judged as hallucinated',
  last_seen_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Most recent time this case was judged as hallucinated',
  resolution    VARCHAR(300)    NULL                    COMMENT 'Resolution note: resolved cases must explain how the issue was fixed; no-action-needed cases must explain why no change is required; cleared when a case returns to unresolved; an empty resolution means the case has no recorded disposition',
  resolved_at   DATETIME        NULL                    COMMENT 'Most recent time the case was marked resolved or no_action_needed; retained after recurrence so the recurrence can be identified',
  PRIMARY KEY (id),
  UNIQUE KEY uk_eval_id (eval_id),
  KEY idx_status (status),
  KEY idx_last_seen_at (last_seen_at)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COMMENT='ch04 faithfulness hallucination case ledger';
