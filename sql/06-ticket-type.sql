-- =============================================================
-- 06 · Intent Recognition · tickets.ticket_type ENUM extension
--
-- Refund requests reuse the existing tickets table instead of a
-- dedicated refund table (spec §9 D2). A new ENUM value 'refund'
-- is added so that a refund request can be told apart from the
-- three ticket types introduced in ch02.
--
-- The three existing values must be listed again and kept in the
-- same order: MODIFY COLUMN replaces the whole column definition,
-- so any value omitted here would be dropped and every existing
-- row holding it would be truncated to an empty string.
--
-- 'refund' is appended last for the same reason: the ORM
-- (app/db/models.py) declares the values in this order and
-- tests/test_models.py compares the two as ordered tuples.
--
-- Values stay English identifiers; the Japanese label shown on
-- screen lives in app/core/labels.py (ch02 decision: English in
-- the database, Japanese only in the display layer).
-- =============================================================

-- Ensure ENUM values and COMMENT text are parsed as utf8mb4.
-- Otherwise, a MySQL client using latin1 by default may double-encode
-- non-ASCII text and corrupt stored ENUM values.
SET NAMES utf8mb4;

ALTER TABLE tickets
  MODIFY COLUMN ticket_type ENUM('after_sales','complaint','inquiry','refund') NOT NULL COMMENT 'Ticket type';
