-- =============================================================
-- 07 · Layered Conversation Context · Table Migration DDL
--
-- Two changes:
--   1. Add an anchor column to conversations to mark the start of Layer 1.
--   2. Create a table for segmented conversation summaries.
--
-- Layer boundaries are represented by message IDs; data is not moved:
--
--   id <= summary_upto_msg_id
--       Already included in the summary.
--
--   summary_upto_msg_id < id <= layer1_from_msg_id
--       Layer 2: rendered in a partially compressed form.
--
--   id > layer1_from_msg_id
--       Layer 1: preserved in its original form.
--
-- Summaries are stored one segment per row and are append-only.
-- Once a segment has been compressed, it is never repeatedly summarized.
-- This ensures that each fact undergoes lossy compression only once.
--
-- If the system instead repeatedly rewrites one rolling summary,
-- the fifth version would contain information from the earliest messages
-- that has already been compressed five times.
-- If an order number were judged unimportant and dropped during any one
-- of those rounds, it would be impossible to determine afterward
-- when or why the information disappeared.
-- =============================================================

SET NAMES utf8mb4;

ALTER TABLE conversations
  ADD COLUMN layer1_from_msg_id BIGINT UNSIGNED NULL
    COMMENT 'Start anchor for Layer 1 (original messages); messages after this ID remain in original form, while earlier messages are rendered in partially compressed form'
    AFTER summary_upto_msg_id;

CREATE TABLE IF NOT EXISTS conversation_summaries (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_id BIGINT UNSIGNED NOT NULL,
  seq             INT             NOT NULL COMMENT 'Segment sequence number, starting from 1',
  from_msg_id     BIGINT UNSIGNED NOT NULL COMMENT 'First message ID covered by this summary segment; inclusive range',
  upto_msg_id     BIGINT UNSIGNED NOT NULL COMMENT 'Last message ID covered by this summary segment; inclusive range',
  content         TEXT            NOT NULL,
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_conv_seq (conversation_id, seq),
  KEY idx_conv_upto (conversation_id, upto_msg_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Segmented conversation summaries; one append-only row per summary segment';
