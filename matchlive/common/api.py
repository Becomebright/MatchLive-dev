"""Cached calls to the LLM API (DESIGN.md §2 principles 8-9), through an OpenAI-compatible endpoint.

A request is a list of parts — text strings and JPEG images as bytes — so callers never
build SDK objects. Every response is cached on disk under the hash of (model name +
request + config), with the model version the API reported and the latency it measured.
Replaying from the cache is what lets offline runs apply a result at "issue time +
measured latency" (DESIGN.md §6.10) reproducibly, and lets ablations share calls whose
inputs are identical. The key does not depend on the endpoint: the same model answers the
same request wherever it is reached, so answers cached before a change of endpoint stay valid.

The endpoint (base URL and key) is read from the local file named by `api.endpoint`
(configs/api.local.yaml, gitignored: the key never goes into the repo); it is only needed when
a call misses the cache. Without it a miss raises NoAPIKey, so a replay from a shared cache
either reproduces the cached run or stops.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from matchlive.common.config import REPO_ROOT


class NoAPIKey(RuntimeError):
    """A request missed the cache and there is no API key to make it: never retried or skipped."""


@dataclass(frozen=True)
class APIResult:
    call_id: str        # the cache key; observation ids cite it (DESIGN.md §5.2)
    text: str
    model: str          # model version as reported by the API
    latency: float      # seconds, measured when the call was actually made
    cached: bool


class APIClient:
    def __init__(self, config: dict[str, Any], cache_dir: Path):
        # `config` is the `api` section of the YAML config.
        self.config = config
        self.cache_dir = cache_dir / config["cache"]
        self._client = None
        self._lock = threading.Lock()

    def generate(self, parts: Sequence[str | bytes], json_output: bool = False) -> APIResult:
        """One generation over `parts` (str = text, bytes = a JPEG image); `json_output`
        asks for a JSON answer."""
        model = self.config["model"]
        # Images enter the key by content hash, so the key stays small and identical
        # images always give the same key.
        request = {"model": model, "json": json_output,
                   "parts": [p if isinstance(p, str) else "jpeg:" + hashlib.sha256(p).hexdigest() for p in parts]}
        call_id = hashlib.sha256(json.dumps(request).encode()).hexdigest()[:16]
        path = self.cache_dir / f"{call_id}.json"
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            return APIResult(call_id, d["text"], d["model"], d["latency"], cached=True)

        with self._lock:
            # One SDK client, created once even when several threads call at the same time.
            if self._client is None:
                self._client = self._connect()
        content = [{"type": "text", "text": p} if isinstance(p, str) else
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(p).decode()}}
                   for p in parts]
        extra = {"response_format": {"type": "json_object"}} if json_output else {}
        # Transient errors (overload, quota, timeouts) are retried with backoff; the latency kept is
        # the successful attempt's, as a cached replay has nothing else to go on.
        for attempt in range(self.config.get("max_attempts", 1)):
            try:
                t0 = time.perf_counter()
                resp = self._client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}],
                                                            **extra)
                latency = time.perf_counter() - t0
                message = resp.choices[0].message if resp.choices else None
                text = (message.content if message else None) or _from_tool_calls(message)
                # Now and then an answer comes back with no text at all; it is not cached.
                if not text:
                    raise ValueError(f"empty response (finish {resp.choices[0].finish_reason if resp.choices else None})")
                break
            except Exception as e:  # noqa: BLE001 — any API failure is retried the same way
                if attempt + 1 == self.config.get("max_attempts", 1):
                    raise
                print(f"api retry {attempt + 1}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                time.sleep(min(5 * 2 ** attempt, self.config.get("max_backoff", 120)))
        d = {"text": text, "model": resp.model or model, "latency": latency,
             "usage": resp.usage.model_dump(mode="json") if resp.usage else None,
             "endpoint": str(self._client.base_url)}
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so an interrupted run never leaves a truncated cache entry. The temp
        # name is per process and thread: parallel runs (ablations) may make the same request.
        tmp = path.with_suffix(f".{os.getpid()}-{threading.get_ident()}.tmp")
        with open(tmp, "w") as f:
            json.dump(d, f)
        tmp.replace(path)
        return APIResult(call_id, d["text"], d["model"], latency, cached=False)

    def _connect(self):
        """The SDK client for the endpoint in the local file `api.endpoint`."""
        path = REPO_ROOT / self.config["endpoint"]
        endpoint = yaml.safe_load(path.read_text()) if path.exists() else None
        if not endpoint or not endpoint.get("api_key"):
            # Replaying from a shared cache needs no key; a request the cache lacks means the run
            # has diverged from the cached one, which must stop the run, not be retried.
            raise NoAPIKey(f"a request missed the cache and there is no endpoint and key in {path}")
        from openai import OpenAI
        # Retries are ours (above); without a timeout an overloaded server can leave a request
        # hanging for good.
        return OpenAI(base_url=endpoint["base_url"], api_key=endpoint["api_key"], max_retries=0,
                      timeout=self.config.get("timeout", 180))


def _from_tool_calls(message) -> str:
    """The JSON text of an answer given as tool calls. The arbiter and QA prompts describe their
    tools in words and ask for {"requests": [...]} (or a verdict / an answer) as JSON; a model may
    emit the same thing as tool calls instead, with no text."""
    calls = [c.function for c in (getattr(message, "tool_calls", None) or []) if getattr(c, "function", None)]
    if not calls:
        return ""
    args = [json.loads(c.arguments or "{}") for c in calls]
    if len(calls) == 1 and calls[0].name in ("verdict", "answer"):
        return json.dumps({"verdict": args[0]} if calls[0].name == "verdict" else {"answer": args[0].get("answer")})
    return json.dumps({"requests": [{"tool": c.name, "args": a} for c, a in zip(calls, args)]})
