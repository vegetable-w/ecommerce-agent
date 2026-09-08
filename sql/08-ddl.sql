-- =============================================================
-- 08 · Tool System · Table Creation DDL
-- New table in this chapter: tool_audit_logs (tool call audit trail)
--
-- Every tool call that passes through the unified execution engine writes
-- one row here: which tool ran, where it came from (built-in or MCP server),
-- the arguments it received, how it ended, how long it took, and how many
-- times it was retried.
--
-- Deliberately no FOREIGN KEY on conversation_id.
-- An audit row is the record of what happened, including what happened when
-- something went wrong. Constraining the write to referential integrity would
-- drop exactly the rows that matter most: calls made outside a conversation
-- (evaluation scripts, smoke tests, warm-up probes) and calls whose
-- conversation was removed afterwards. The column is therefore a plain
-- nullable BIGINT UNSIGNED that merely records the conversation the call
-- belonged to, if there was one.
--
-- status stays an English identifier, like every other ENUM in this schema.
-- The Japanese labels shown on screen live in app/core/labels.py
-- (TOOL_AUDIT_STATUS). English in the database, Japanese only in the
-- display layer (ch02 decision).
--
-- Indexes: idx_conversation_id supports "show me every tool call made in
-- this conversation", the lookup this table exists for. idx_created_at
-- supports time-window reads: success/failure rates over the last N days
-- and retention deletion of old rows. The AUTO_INCREMENT primary key
-- already orders rows by insertion, but it cannot answer a
-- `WHERE created_at >= ...` range without a scan, which is the shape those
-- reads actually take. No index on status: five values over the whole table
-- is too low a cardinality to help, and it would only add write cost to a
-- table that takes an insert on every single tool call.
-- =============================================================

-- Ensure ENUM values, DEFAULT values, and COMMENT text are parsed as utf8mb4.
-- Otherwise, a MySQL client using latin1 by default may double-encode
-- non-ASCII text and corrupt stored ENUM values.
SET NAMES utf8mb4;

CREATE TABLE tool_audit_logs (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'Primary key',
  conversation_id BIGINT UNSIGNED NULL                    COMMENT 'Conversation the call belonged to; no FK, so a call made outside any conversation is still recorded',
  tool_call_id    VARCHAR(64)     NULL                    COMMENT 'Tool call request ID issued by the model, used to match this row against the corresponding tool message',
  tool_name       VARCHAR(128)    NOT NULL                COMMENT 'Tool name as registered in the registry, e.g. query_order',
  tool_source     ENUM('builtin','mcp') NOT NULL          COMMENT 'Where the tool came from: process-local built-in tool or a tool exposed by an MCP server',
  mcp_server      VARCHAR(64)     NULL                    COMMENT 'MCP server name when tool_source is mcp, e.g. logistics; null for built-in tools',
  arguments       JSON            NULL                    COMMENT 'Arguments the tool was called with, after schema validation; null when the call was blocked before arguments were resolved',
  result_summary  TEXT            NULL                    COMMENT 'Shortened form of the tool result kept for review; the full result is not stored',
  status          ENUM('success','failed','timeout','validation_blocked','permission_denied') NOT NULL COMMENT 'How the call ended: completed / raised / exceeded its timeout / rejected by argument validation / rejected by the permission check',
  error_message   VARCHAR(512)    NULL                    COMMENT 'Error detail for a call that did not succeed; truncated to fit',
  retry_count     INT UNSIGNED    NOT NULL DEFAULT 0      COMMENT 'Number of retries performed before this outcome; 0 means the first attempt decided it',
  duration_ms     INT UNSIGNED    NULL                    COMMENT 'Wall-clock duration of the call in milliseconds; null when the call never started',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Time the call was recorded',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  KEY idx_created_at (created_at)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COMMENT='Tool call audit trail';
