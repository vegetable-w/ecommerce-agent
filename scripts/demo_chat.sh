#!/usr/bin/env bash
# 受け入れ確認:2ターンのストリーミング対話。2ターン目で1ターン目のコンテキストを保持できること
#
# 注意(Windows/Git Bash): 日本語を含むJSONを curl に直接コマンドライン引数として渡すと、
# MSYSがネイティブのcurl.exeを起動する際に引数がコンソールのコードページ(cp932)経由で
# 変換され、UTF-8バイト列が壊れる(サーバー側で「There was an error parsing the body」に
# なる)。そのため、ペイロードは一旦UTF-8のテンポラリファイルに書き出し、
# curl -d @file で読み込ませることで回避する。
set -euo pipefail
SID="demo-$$"
TMP1="$(mktemp)"
TMP2="$(mktemp)"
trap 'rm -f "$TMP1" "$TMP2"' EXIT

printf '{"session_id": "%s", "message": "私は山田太郎です。昨日そちらの自動猫トイレを買いました"}' "$SID" > "$TMP1"
printf '{"session_id": "%s", "message": "私の名前と、買ったものを覚えていますか？"}' "$SID" > "$TMP2"

echo "=== 1ターン目:名前と購入商品を伝える ==="
curl -sN http://localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d @"$TMP1"
echo; echo "=== 2ターン目:コンテキスト確認(私の名前は？何を買った？) ==="
curl -sN http://localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d @"$TMP2"
echo
