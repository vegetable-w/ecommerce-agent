-- =============================================================
-- 07 · Layered Conversation Context · Table Migration DDL
--
-- One change: create a table for segmented conversation summaries.
--
-- The layer boundary is a message ID; data is never moved:
--
--   id <= summary_upto_msg_id
--       Layer 2: already folded into the summary.
--
--   id > summary_upto_msg_id
--       Layer 1: preserved in its original form and replayed verbatim.
--
-- A single boundary is enough because the summarizer never advances it all
-- the way to the newest message: it stops short by context_window_turns, so
-- the most recent turns always remain on the Layer 1 side. An earlier draft
-- of this file also carried a layer1_from_msg_id anchor for a third,
-- partially compressed tier. That tier was never part of the design, so the
-- column would have stayed NULL forever and misled anyone reading the schema.
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
