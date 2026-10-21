"""The repeat-guard repair: an aborted greedy completion is retried once."""

import io
import json
import urllib.error

from code.llm.ollama_client import OllamaClient


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_repeat_limit_500_is_retried_with_penalty(tmp_path, monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        payload = json.loads(request.data)
        calls.append(payload["options"])
        if len(calls) == 1:
            body = b'{"error":"prediction aborted, token repeat limit reached"}'
            raise urllib.error.HTTPError(request.full_url, 500, "Server Error", {}, io.BytesIO(body))
        return _FakeResponse(json.dumps({"message": {"content": "repaired answer"}, "eval_count": 12}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OllamaClient(tmp_path / "cache.jsonl", model="llama3.1:8b")
    record = client.chat([{"role": "user", "content": "hi"}])

    assert record["response"] == "repaired answer"
    assert record["repaired"] is True
    assert calls[0].get("repeat_penalty") is None and calls[1]["repeat_penalty"] == 1.1
    replayed = OllamaClient(tmp_path / "cache.jsonl", model="llama3.1:8b", replay=True).chat([{"role": "user", "content": "hi"}])
    assert replayed["response"] == "repaired answer"


def test_other_500_is_not_retried(tmp_path, monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(request.full_url, 500, "Server Error", {}, io.BytesIO(b'{"error":"boom"}'))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OllamaClient(tmp_path / "cache.jsonl", model="llama3.1:8b")
    try:
        client.chat([{"role": "user", "content": "hi"}])
    except urllib.error.HTTPError:
        pass
    else:
        raise AssertionError("a non-repeat 500 must propagate")
    assert len(calls) == 1
