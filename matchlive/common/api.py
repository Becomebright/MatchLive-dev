"""Cached calls to the LLM API (DESIGN.md §2 principles 8-9), through an OpenAI-compatible endpoint.

A request is a list of parts — text strings and JPEG images as bytes — so callers never
build SDK objects. Every response is cached on disk under the hash of (model name +
request + config), with the model version the API reported and the latency it measured.
Replaying from the cache is what lets offline runs apply a result at "issue time +
measured latency" (DESIGN.md §6.10) reproducibly, and lets ablations share calls whose
inputs are identical. The key does not depend on the endpoint: the same model answers the
same request wherever it is reached, so answers cached before a change of endpoint stay valid.
Text-to-speech for the commentary (speech(), a voice-cloning model) is cached the same way, as MP3
files; it goes through the same endpoints, or through its own (`api.speech_endpoint`, a voice model
served locally).

The endpoints (base URL and key each) are read from the local file named by `api.endpoint`
(configs/api.local.yaml, gitignored: keys never go into the repo), in order: a failed request is
retried on the next one at once (that request only), and the next one takes over for good when
the current one runs out of quota. They are only needed when a call
misses the cache. Without them a miss raises NoAPIKey, so a replay from a shared cache either
reproduces the cached run or stops.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import threading
import time
import wave
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


@dataclass(frozen=True)
class SpeechResult:
    call_id: str
    path: Path          # the MP3 in the speech cache
    latency: float      # seconds, measured when the call was actually made
    cached: bool


class APIClient:
    def __init__(self, config: dict[str, Any], cache_dir: Path):
        # `config` is the `api` section of the YAML config.
        self.config = config
        self.cache_dir = cache_dir / config["cache"]
        self.speech_dir = cache_dir / config.get("speech_cache", "tts")
        self._clients: list | None = None       # one SDK client per endpoint, in the local file's order
        self._out: set[int] = set()              # endpoints out of quota: no request goes to them again
        self._speech: APIClient | None = None    # over api.speech_endpoint, made on the first speech request
        self._lock = threading.Lock()

    def generate(self, parts: Sequence[str | bytes], json_output: bool = False,
                 reasoning: str | None = None) -> APIResult:
        """One generation over `parts` (str = text, bytes = a JPEG image); `json_output`
        asks for a JSON answer; `reasoning` ("low", ...) sets the model's reasoning effort
        (by default the model's own), for a caller that cannot wait for a long think."""
        model = self.config["model"]
        # Images enter the key by content hash, so the key stays small and identical
        # images always give the same key.
        request = {"model": model, "json": json_output,
                   "parts": [p if isinstance(p, str) else "jpeg:" + hashlib.sha256(p).hexdigest() for p in parts]}
        if reasoning is not None:               # in the key only when set: earlier keys are unchanged
            request["reasoning"] = reasoning
        if self.config.get("extra_body"):       # server-specific fields, e.g. thinking off on a local vLLM
            request["extra_body"] = self.config["extra_body"]
        call_id = hashlib.sha256(json.dumps(request).encode()).hexdigest()[:16]
        path = self.cache_dir / f"{call_id}.json"
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            return APIResult(call_id, d["text"], d["model"], d["latency"], cached=True)

        content = [{"type": "text", "text": p} if isinstance(p, str) else
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(p).decode()}}
                   for p in parts]
        extra = {"response_format": {"type": "json_object"}} if json_output else {}
        if reasoning is not None:
            extra["reasoning_effort"] = reasoning
        if self.config.get("extra_body"):
            extra["extra_body"] = self.config["extra_body"]

        def call(client):
            resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}], **extra)
            message = resp.choices[0].message if resp.choices else None
            text = (message.content if message else None) or _from_tool_calls(message)
            # Now and then an answer comes back with no text at all; it is not cached.
            if not text:
                raise ValueError(f"empty response (finish {resp.choices[0].finish_reason if resp.choices else None})")
            return resp, text

        (resp, text), latency, client = self._call(call)
        d = {"text": text, "model": resp.model or model, "latency": latency,
             "usage": resp.usage.model_dump(mode="json") if resp.usage else None,
             "endpoint": str(client.base_url)}
        _write(path, json.dumps(d).encode())
        return APIResult(call_id, d["text"], d["model"], latency, cached=False)

    def speech(self, text: str, voice: str, reference: str, model: str) -> SpeechResult:
        """`text` spoken by a voice-cloning model (fishaudio/drama-3) in the voice and delivery of a
        reference recording (`reference`: <name>.wav under the repo root, its transcript in
        <name>.txt), as MP3, cached like generate() under the hash of the request, in
        paths.cache/<api.speech_cache>. The request goes to the endpoints of `api.speech_endpoint`
        when it is set (a model served locally), else to the usual ones."""
        ref = REPO_ROOT / reference
        ref_audio, ref_text = ref.with_suffix(".wav").read_bytes(), ref.with_suffix(".txt").read_text().strip()
        # The key as before the cloning model ("instructions" held the reference's name), so the
        # cached audio stays valid; the recording by content too: a re-recorded one is a new request.
        request = {"model": model, "voice": voice, "instructions": reference, "text": text, "format": "mp3",
                   "reference": hashlib.sha256(ref_audio + ref_text.encode()).hexdigest()}
        call_id = hashlib.sha256(json.dumps(request).encode()).hexdigest()[:16]
        path = self.speech_dir / f"{call_id}.mp3"
        meta = path.with_suffix(".json")
        if path.exists() and meta.exists():
            return SpeechResult(call_id, path, json.loads(meta.read_text())["latency"], cached=True)

        def call(client):
            wav = client.audio.speech.create(
                model=model, voice=voice, input=text, response_format="wav",
                extra_body={"ref_audio": "data:audio/wav;base64," + base64.b64encode(ref_audio).decode(),
                            "ref_text": ref_text}).read()
            with wave.open(io.BytesIO(wav)) as w:
                seconds = w.getnframes() / w.getframerate()
            # drama-3 now and then goes on generating silence up to its token limit (vLLM-Omni's
            # recipe): speech far longer than the words need (about 0.4 s a word) is a failed take,
            # and sampling differs from call to call, so a retry usually comes out right.
            if seconds > 0.6 * len(text.split()) + 3:
                raise ValueError(f"{seconds:.1f} s of audio for {len(text.split())} words")
            return subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "mp3", "pipe:1"], input=wav,
                                  capture_output=True, check=True).stdout

        with self._lock:
            if self.config.get("speech_endpoint") and self._speech is None:
                # The same client over the speech endpoints alone: a local TTS server must not take
                # over the other calls when an endpoint fails, nor they its requests.
                self._speech = APIClient({**self.config, "endpoint": self.config["speech_endpoint"]}, Path())
        audio, latency, client = (self._speech or self)._call(call)
        _write(path, audio)
        _write(meta, json.dumps({"latency": latency, "model": model, "endpoint": str(client.base_url)}).encode())
        return SpeechResult(call_id, path, latency, cached=False)

    def transcribe(self, audio: Path, model: str) -> dict:
        """Speech-to-text of an audio file with segment timestamps ({"text", "segments": [{"start",
        "end", "text"}]}), cached under the hash of the model and the audio. For evaluation only:
        the live system never listens to the broadcast (DESIGN.md §11 D1)."""
        data = audio.read_bytes()
        call_id = hashlib.sha256(json.dumps({"model": model}).encode() + data).hexdigest()[:16]
        path = self.speech_dir / "transcripts" / f"{call_id}.json"
        if path.exists():
            return json.loads(path.read_text())

        def call(client):
            r = client.audio.transcriptions.create(model=model, file=(audio.name, data), response_format="verbose_json",
                                                   timestamp_granularities=["segment"])
            return {"text": r.text, "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in r.segments or []]}

        out, _, _ = self._call(call)
        _write(path, json.dumps(out).encode())
        return out

    def _call(self, call):
        """Run `call(sdk client)` against the endpoints. A request starts on the first endpoint still
        in quota; a failed attempt (overload, timeout, an empty answer) is retried on the next one
        at once — for this request only. Once every endpoint has failed it in turn, it waits
        (backoff) before the next round. An endpoint out of quota is dropped for good, for every
        request. Returns (what `call` returned, its latency from the first attempt — failed attempts
        and backoff included, as a live run would wait; all a cached replay has to go on — and the
        client that answered)."""
        with self._lock:
            # The SDK clients, created once even when several threads call at the same time.
            if self._clients is None:
                self._clients = self._connect()
        n = len(self._clients)
        step = 0
        t0 = time.perf_counter()
        for attempt in range(self.config.get("max_attempts", 1)):
            with self._lock:
                # The endpoints still in quota, in order (all of them if none is: nothing else to try).
                live = [i for i in range(n) if i not in self._out] or list(range(n))
            current = live[step % len(live)]
            client = self._clients[current]
            try:
                out = call(client)
                return out, time.perf_counter() - t0, client
            except Exception as e:  # noqa: BLE001 — any API failure is retried the same way
                if attempt + 1 == self.config.get("max_attempts", 1):
                    raise
                print(f"api retry {attempt + 1} ({client.base_url.host}): {type(e).__name__}: {str(e)[:120]}", flush=True)
                if _out_of_quota(e) and current not in self._out:
                    with self._lock:
                        self._out.add(current)
                    print(f"api: {client.base_url} is out of quota, no longer used", flush=True)
                    continue        # the same step now lands on the next endpoint still in quota
                step += 1
                if step % len(live) == 0:
                    # Every endpoint has failed this request once more: back off before the next round.
                    time.sleep(min(5 * 2 ** (step // len(live) - 1), self.config.get("max_backoff", 120)))

    def _connect(self) -> list:
        """One SDK client per endpoint of the local file `api.endpoint`, in its order."""
        path = REPO_ROOT / self.config["endpoint"]
        local = yaml.safe_load(path.read_text()) if path.exists() else None
        endpoints = [e for e in (local or {}).get("endpoints") or [] if e.get("api_key")]
        if not endpoints:
            # Replaying from a shared cache needs no key; a request the cache lacks means the run
            # has diverged from the cached one, which must stop the run, not be retried.
            raise NoAPIKey(f"a request missed the cache and there is no endpoint and key in {path}")
        from openai import OpenAI
        # Retries are ours (above); without a timeout an overloaded server can leave a request
        # hanging for good.
        return [OpenAI(base_url=e["base_url"], api_key=e["api_key"], max_retries=0,
                       timeout=self.config.get("timeout", 180)) for e in endpoints]


def _write(path: Path, data: bytes) -> None:
    """Write-then-rename, so an interrupted run never leaves a truncated cache entry. The temp name is
    per process and thread: parallel runs (ablations) may make the same request."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}-{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _out_of_quota(e: Exception) -> bool:
    """An error that retrying the same endpoint will not fix: the key refused (401 / 402 / 403) or
    its quota or balance used up (often a 429 that says so)."""
    status = getattr(e, "status_code", None)
    text = str(e).lower()
    return status in (401, 402, 403) or any(w in text for w in ("quota", "insufficient", "balance", "credit",
                                                                "余额", "额度"))


def _from_tool_calls(message) -> str:
    """The JSON text of an answer given as tool calls. The arbiter and QA prompts describe their
    tools in words and ask for {"requests": [...]} (or a verdict / an answer) as JSON; a model may
    emit the same thing as tool calls instead, with no text."""
    calls = [c.function for c in (getattr(message, "tool_calls", None) or []) if getattr(c, "function", None)]
    if not calls:
        return ""
    args = [json.loads(c.arguments or "{}") for c in calls]
    # Some endpoints prefix the name with a namespace ("default:get_frames", "ToolSet.get_frames").
    names = [c.name.split(":")[-1].split(".")[-1] for c in calls]
    if len(calls) == 1 and names[0] in ("verdict", "answer"):
        return json.dumps({"verdict": args[0]} if names[0] == "verdict" else {"answer": args[0].get("answer")})
    return json.dumps({"requests": [{"tool": n, "args": a} for n, a in zip(names, args)]})
