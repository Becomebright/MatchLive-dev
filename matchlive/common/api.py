"""Cached calls to the LLM API (DESIGN.md §2 principles 8-9), through the google.genai SDK.

A request is a list of parts — text strings and JPEG images as bytes — so callers never
build SDK objects. Every response is cached on disk under the hash of (model name +
request + config), with the model version the API reported and the latency it measured.
Replaying from the cache is what lets offline runs apply a result at "issue time +
measured latency" (DESIGN.md §6.10) reproducibly, and lets ablations share calls whose
inputs are identical.

The API key is read from the environment variable named in the config (never from a
file in the repo); it is only needed when a call misses the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


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

        from google import genai
        from google.genai import types
        # One SDK client, created once even when several threads call at the same time (a
        # second one would close the first when it is dropped).
        with self._lock:
            if self._client is None:
                # Without a timeout an overloaded server can leave a request hanging for good.
                self._client = genai.Client(api_key=os.environ[self.config["api_key_env"]],
                                            http_options=types.HttpOptions(timeout=int(1000 * self.config.get("timeout", 180))))
        contents = [types.Part.from_text(text=p) if isinstance(p, str)
                    else types.Part.from_bytes(data=p, mime_type="image/jpeg") for p in parts]
        config = types.GenerateContentConfig(response_mime_type="application/json" if json_output else None)
        # Transient errors (overload, quota) are retried with backoff; the latency kept is the
        # successful attempt's, as a cached replay has nothing else to go on.
        for attempt in range(self.config.get("max_attempts", 1)):
            try:
                t0 = time.perf_counter()
                resp = self._client.models.generate_content(model=model, contents=contents, config=config)
                latency = time.perf_counter() - t0
                text = resp.text or _from_function_calls(resp)
                # Now and then an answer comes back with no text part at all; it is not cached.
                if not text:
                    cand = resp.candidates[0] if resp.candidates else None
                    kinds = [k for p in (cand.content.parts or []) for k, v in p.model_dump().items() if v is not None] \
                        if cand and cand.content else []
                    raise ValueError(f"empty response (finish {cand.finish_reason if cand else None}, parts {kinds})")
                break
            except Exception as e:  # noqa: BLE001 — any API failure is retried the same way
                if attempt + 1 == self.config.get("max_attempts", 1):
                    raise
                print(f"api retry {attempt + 1}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                time.sleep(min(5 * 2 ** attempt, 120))
        d = {"text": text, "model": resp.model_version or model, "latency": latency,
             "usage": resp.usage_metadata.model_dump(mode="json") if resp.usage_metadata else None}
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so an interrupted run never leaves a truncated cache entry.
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(d, f)
        tmp.replace(path)
        return APIResult(call_id, d["text"], d["model"], latency, cached=False)


def _from_function_calls(resp) -> str:
    """The JSON text of an answer given as native function calls. The arbiter and QA prompts
    describe their tools in words and ask for {"requests": [...]} (or a verdict / an answer) as
    JSON; the model sometimes emits the same thing as function calls instead, with no text."""
    parts = (resp.candidates[0].content.parts or []) if resp.candidates and resp.candidates[0].content else []
    calls = [p.function_call for p in parts if p.function_call]
    if not calls:
        return ""
    if len(calls) == 1 and calls[0].name in ("verdict", "answer"):
        args = dict(calls[0].args or {})
        return json.dumps({"verdict": args} if calls[0].name == "verdict" else {"answer": args.get("answer")})
    return json.dumps({"requests": [{"tool": c.name, "args": dict(c.args or {})} for c in calls]})
