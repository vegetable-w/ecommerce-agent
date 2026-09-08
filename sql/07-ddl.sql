-- =============================================================
-- 07 · Conversation Context Management · Table Migration DDL
-- Step 1 of 2.
-- This step adds two columns to the conversations table introduced in 02:
-- a projected summary assembled from recent summary segments,
-- and an anchor indicating the latest message covered by the summary.
--
-- The additional anchor required for layered context management,
-- together with the segmented summary table, is defined in 07-layers.sql.
-- Both files must be applied.
-- =============================================================

-- Ensure COMMENT text is parsed as utf8mb4.
-- Otherwise, a MySQL client using latin1 by default may double-encode
-- non-ASCII text and corrupt stored comments.
SET NAMES utf8mb4;

ALTER TABLE conversations
  ADD COLUMN summary             TEXT            NULL COMMENT 'Projected summary assembled from the most recent summary segments; attached after the user message together with retrieved evidence during prompt construction' AFTER status,
  ADD COLUMN summary_upto_msg_id BIGINT UNSIGNED NULL COMMENT 'Latest message ID already covered by the summary; the sliding window starts from the following message' AFTER summary;
