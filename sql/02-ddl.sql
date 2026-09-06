-- =============================================================
-- 02 · Function Calling Toolchain · Table Creation DDL
-- Tables created in this chapter: faq / conversations / messages / tickets
-- Product, order, and logistics data are mocked inside tools, so no tables are created for them
-- The entire database uses ENGINE=InnoDB and CHARSET=utf8mb4
-- Table creation order: conversations first, then messages / tickets that depend on it
-- =============================================================

-- Conversation container: unified identity for a conversation; both messages and tickets reference it
CREATE TABLE conversations (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'Conversation primary key',
  user_id     VARCHAR(64)     NOT NULL                COMMENT 'User identifier',
  status      ENUM('in_progress','escalated','closed') NOT NULL DEFAULT 'in_progress' COMMENT 'Processing status',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Start time',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last updated time',
  PRIMARY KEY (id),
  KEY idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Customer support conversation';

-- Message log: each conversation contains N messages; role aligns with the Chat Completions protocol
CREATE TABLE messages (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'Message primary key',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT 'Associated conversation',
  role            ENUM('user','assistant','tool') NOT NULL COMMENT 'Role: user / assistant / tool result',
  content         TEXT            NULL                     COMMENT 'Message content; may be null when assistant only issues a tool call',
  tool_calls      JSON            NULL                     COMMENT 'Tool call requests contained in an assistant message',
  tool_call_id    VARCHAR(64)     NULL                     COMMENT 'Request ID corresponding to a tool message, used to match tool results correctly',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_messages_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Conversation message log';

-- FAQ question-answer pairs: data source for query_faq;
-- starting from 03, retrieval will move to a vector store and this table will serve as the original source data
CREATE TABLE faq (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'FAQ primary key',
  question    VARCHAR(512)    NOT NULL                COMMENT 'Question',
  answer      TEXT            NOT NULL                COMMENT 'Answer',
  category    VARCHAR(64)     NOT NULL                COMMENT 'Category',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last updated time',
  PRIMARY KEY (id),
  KEY idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Frequently asked questions';

-- Human support ticket: created by create_ticket; ticket number is used as the business primary key
CREATE TABLE tickets (
  ticket_no       VARCHAR(32)     NOT NULL                COMMENT 'Ticket number, e.g. T20260701008',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT 'Associated conversation; can be used to trace the original conversation',
  description     TEXT            NOT NULL                COMMENT 'Issue description',
  ticket_type     ENUM('after_sales','complaint','inquiry') NOT NULL COMMENT 'Ticket type',
  status          ENUM('pending','resolved') NOT NULL DEFAULT 'pending' COMMENT 'Processing status',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
  PRIMARY KEY (ticket_no),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_tickets_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Human support ticket';
