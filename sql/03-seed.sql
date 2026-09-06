-- 03 · 合成の過去会話。ナレッジ抽出 job(kb-mine)の入力にする。
-- status は英語識別子。ENUM('in_progress','escalated','closed') なので日本語を入れると
-- strict mode で即エラーになる(02 章の計画で踏んだ罠)。
SET NAMES utf8mb4;

SELECT COUNT(*) AS before_conversations FROM conversations;

INSERT INTO conversations (user_id, status) VALUES
  ('seed-mine-1', 'closed'),
  ('seed-mine-2', 'closed'),
  ('seed-mine-3', 'closed');

SET @c1 = (SELECT id FROM conversations WHERE user_id='seed-mine-1' ORDER BY id DESC LIMIT 1);
SET @c2 = (SELECT id FROM conversations WHERE user_id='seed-mine-2' ORDER BY id DESC LIMIT 1);
SET @c3 = (SELECT id FROM conversations WHERE user_id='seed-mine-3' ORDER BY id DESC LIMIT 1);

INSERT INTO messages (conversation_id, role, content) VALUES
  -- 再利用できるルール(抽出されるべき)
  (@c1, 'user',      '海外への発送はできますか'),
  (@c1, 'assistant', '現在は日本国内のみへの発送となっております。海外発送には対応しておりません。'),
  -- 個別事例(抽出すべきでない)+ ルール(抽出されるべき)が混在する会話
  (@c2, 'user',      '注文番号 1042 がまだ届きません'),
  (@c2, 'assistant', 'お調べしたところ、現在配送中です。到着まで今しばらくお待ちください。'),
  (@c2, 'user',      'ちなみに何時までの注文なら当日発送ですか'),
  (@c2, 'assistant', '平日15時までにご注文いただいた在庫商品は、当日中に発送いたします。'),
  -- 雑談のみ(抽出すべきでない)
  (@c3, 'user',      'ありがとうございました'),
  (@c3, 'assistant', 'こちらこそ、ご利用ありがとうございます。またお気軽にお問い合わせください。');

SELECT COUNT(*) AS after_conversations FROM conversations;
SELECT COUNT(*) AS seeded_messages FROM messages
  WHERE conversation_id IN (@c1, @c2, @c3);
