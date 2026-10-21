"""Cached client for the local Ollama chat endpoint.

Every call is keyed by a hash of model, messages, options, and the thinking
flag, and appended to a JSONL cache, so runs are replayable without the
endpoint. Replay mode fails loudly on a cache miss instead of silently
regenerating, which keeps the prompt-cache guarantee of the reproducibility
policy. Thinking is disabled by default so reasoning models spend the token
budget on the answer.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

# Ollama aborts a completion when the same trimmed token repeats more than one
# hundred times and, since 0.34, surfaces the abort as an HTTP 500 instead of
# returning the partial text. Greedy decoding on a checklist prompt triggers
# it, so the caller retries once with a repeat penalty and records the repair.
REPEAT_LIMIT_MARK = "token repeat limit"
REPEAT_PENALTY_FALLBACK = 1.1


class OllamaClient:
    """Minimal chat client with a JSONL cache and a replay mode."""

    def __init__(
        self,
        cache_path: Path,
        model: str = "qwen3.5:4b",
        endpoint: str = "http://localhost:11434/api/chat",
        replay: bool = False,
        timeout: float = 900.0,
        think: bool = False,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.model = model
        self.endpoint = endpoint
        self.replay = replay
        self.timeout = timeout
        self.think = think
        self.hits = 0
        self.misses = 0
        self.seconds = 0.0
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                self.cache[record["key"]] = record

    def _key(self, messages: list[dict], options: dict) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "options": options,
            "think": self.think,
            "endpoint": self.endpoint,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def chat(self, messages: list[dict], temperature: float = 0.0, seed: int = 0, num_predict: int = 256) -> dict:
        """Return the cached or fresh completion record for these messages.

        A completion aborted by the endpoint's repeat guard is retried once with
        a repeat penalty, and the record keeps the repair flag and the penalty
        that produced it so the provenance of every explanation is explicit.
        """
        options = {"temperature": temperature, "seed": seed, "num_predict": num_predict}
        key = self._key(messages, options)
        repair_options = {**options, "repeat_penalty": REPEAT_PENALTY_FALLBACK}
        repair_key = self._key(messages, repair_options)
        # A repaired completion is stored under the penalty key, so replay must
        # look there too or the reproducibility gate fails on that record.
        for cached in (key, repair_key):
            if cached in self.cache:
                self.hits += 1
                return self.cache[cached]
        if self.replay:
            raise RuntimeError(f"replay mode and cache miss for prompt {key[:16]}")
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": options,
        }
        started = time.time()
        repaired = False
        try:
            data = self._post(payload)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace") if error.fp is not None else ""
            if error.code != 500 or REPEAT_LIMIT_MARK not in body:
                raise
            options, key = repair_options, repair_key
            payload["options"] = options
            data = self._post(payload)
            repaired = True
        elapsed = time.time() - started
        self.seconds += elapsed
        self.misses += 1
        record = {
            "key": key,
            "model": self.model,
            "messages": messages,
            "options": options,
            "think": self.think,
            "response": data["message"]["content"],
            "repaired": repaired,
            "seconds": round(elapsed, 2),
            "eval_count": data.get("eval_count"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_duration": data.get("eval_duration"),
            "total_duration": data.get("total_duration"),
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        with self.cache_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        self.cache[key] = record
        return record
