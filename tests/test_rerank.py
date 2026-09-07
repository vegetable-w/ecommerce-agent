"""上流の /rerank client。実際の上流は叩かず _post を差し替える。

エラー時の契約は「例外を投げず [] を返す」。リランクは並べ替えの補助であり、
ここで落ちて問い合わせ全体を落としてはいけない(呼び出し側は [] を
「リランクなし」と解釈してハイブリッド検索の並びへ縮退する)。
"""

import httpx
import pytest

from app.core import rerank as rr


def _fake_post(payload=None, status=200, exc=None, body=None, seen=None):
    """_post の差し替えを作る。exc を渡すとその例外を送出する。"""

    async def _post(url, json, headers, timeout):
        if seen is not None:
            seen.update(url=url, json=json, headers=headers, timeout=timeout)
        if exc is not None:
            raise exc
        req = httpx.Request("POST", url)
        if body is not None:
            return httpx.Response(status, content=body, request=req)
        return httpx.Response(status, json=payload, request=req)

    return _post


_UNSORTED = {"results": [
    {"index": 0, "relevance_score": 0.1},
    {"index": 1, "relevance_score": 0.9},
    {"index": 2, "relevance_score": 0.5},
]}


async def test_rerank_orders_and_truncates(monkeypatch):
    """上流が降順で返すとは限らない前提で、必ずこちらで並べ替えてから truncate する。"""
    monkeypatch.setattr(rr, "_post", _fake_post(_UNSORTED))
    assert await rr.rerank("q", ["a", "b", "c"], top_n=2) == [(1, 0.9), (2, 0.5)]


async def test_rerank_without_top_n_returns_all(monkeypatch):
    monkeypatch.setattr(rr, "_post", _fake_post(_UNSORTED))
    assert await rr.rerank("q", ["a", "b", "c"]) == [(1, 0.9), (2, 0.5), (0, 0.1)]


async def test_rerank_empty_docs_does_not_call_upstream(monkeypatch):
    async def _boom(*a, **k):
        raise AssertionError("空の docs で上流を呼んではいけない")

    monkeypatch.setattr(rr, "_post", _boom)
    assert await rr.rerank("q", []) == []


async def test_rerank_sends_model_query_and_documents(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(rr, "_post", _fake_post(_UNSORTED, seen=seen))
    await rr.rerank("送料はいくら", ["a", "b", "c"], top_n=2)
    assert seen["url"].endswith("/rerank")
    assert seen["json"]["query"] == "送料はいくら"
    assert seen["json"]["documents"] == ["a", "b", "c"]
    assert seen["json"]["model"] == "BAAI/bge-reranker-v2-m3"
    assert seen["json"]["top_n"] == 2
    # 鍵そのものは assert しない(失敗時に pytest が両辺を展開して漏らすため)
    assert seen["headers"]["Authorization"].startswith("Bearer ")
    assert seen["timeout"] > 0


@pytest.mark.parametrize("status", [401, 429, 500, 502])
async def test_rerank_http_error_returns_empty(monkeypatch, status):
    monkeypatch.setattr(rr, "_post", _fake_post({"error": "x"}, status=status))
    assert await rr.rerank("q", ["a", "b"]) == []


async def test_rerank_timeout_returns_empty(monkeypatch):
    monkeypatch.setattr(rr, "_post", _fake_post(exc=httpx.ReadTimeout("timed out")))
    assert await rr.rerank("q", ["a", "b"]) == []


async def test_rerank_connect_error_returns_empty(monkeypatch):
    monkeypatch.setattr(rr, "_post", _fake_post(exc=httpx.ConnectError("refused")))
    assert await rr.rerank("q", ["a", "b"]) == []


@pytest.mark.parametrize("payload", [
    {},                                                   # results ごと無い
    {"results": [{"index": 0}]},                          # score が無い
    {"results": [{"relevance_score": 0.5}]},              # index が無い
    {"results": [{"index": 0, "relevance_score": "x"}]},  # score が数値でない
    {"results": "なにか"},                                 # results が list でない
])
async def test_rerank_malformed_body_returns_empty(monkeypatch, payload):
    monkeypatch.setattr(rr, "_post", _fake_post(payload))
    assert await rr.rerank("q", ["a", "b"]) == []


async def test_rerank_non_json_body_returns_empty(monkeypatch):
    monkeypatch.setattr(rr, "_post", _fake_post(body=b"<html>gateway</html>"))
    assert await rr.rerank("q", ["a", "b"]) == []


async def test_rerank_drops_out_of_range_index(monkeypatch):
    """docs の範囲外の index は捨てる。呼び出し側は hits[idx] で引くため IndexError になる。"""
    monkeypatch.setattr(rr, "_post", _fake_post({"results": [
        {"index": 5, "relevance_score": 0.99},
        {"index": 1, "relevance_score": 0.4},
    ]}))
    assert await rr.rerank("q", ["a", "b"]) == [(1, 0.4)]
