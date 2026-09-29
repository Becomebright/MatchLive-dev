"""Run the event analyzer over every fixed window of a half (DESIGN.md §6.2-6.3).

    python scripts/run_events.py --match "2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea" --half 1

Reads the half's raw perception cache (run scripts/run_perception.py first), replays identity
voting once, then walks the video forward window by window, sending each window's request
to the API (several in flight at once) and writing one JSON line per window to
outputs/events/<match>/half<h>.jsonl: the window, the call id, the model version, the
measured latency, and the parsed candidates. Calls are cached (cache/api/), so a re-run
only pays for windows whose request changed.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from matchlive.common.api import APIClient
from matchlive.common.cache import load_perception
from matchlive.common.config import load_config
from matchlive.common.match import Match
from matchlive.events.analyzer import build_request, parse_candidates
from matchlive.events.windows import fixed_windows
from matchlive.runtime.runner import MatchRunner, resolve_frames


def call(client: APIClient, request, window, players_of):
    # Transient API errors (rate limits, timeouts) are retried with backoff; a window that
    # still fails is recorded as failed rather than stopping the whole half.
    for attempt in range(4):
        try:
            result = client.generate(request.parts, json_output=True)
            # One line per attempt as it happens, so the error rate can be watched live.
            print(f"ok {window} {result.latency:.1f}s{' cached' if result.cached else ''}", flush=True)
            return {"window": window, "call_id": result.call_id, "model": result.model,
                    "latency": result.latency, "cached": result.cached,
                    "candidates": [{**c.__dict__, "players": [players_of.get(key) for _, key in c.participants]}
                                   for c in parse_candidates(result.text, request)]}
        except Exception as e:  # noqa: BLE001 — any API failure is retried the same way
            error = f"{type(e).__name__}: {e}"
            print(f"error {window} attempt {attempt + 1}: {error[:150]}", flush=True)
            time.sleep(10 * 2 ** attempt)
    return {"window": window, "error": error}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--half", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8, help="API calls in flight at once")
    args = ap.parse_args()

    config = load_config()
    ev = config["events"]
    match = Match(config, args.match)
    arrays, meta = load_perception(MatchRunner(config).run_perception(match, args.half))
    fps = meta["fps"]
    windows = fixed_windows(meta["n_frames"] / fps, ev["window_seconds"], ev["window_overlap"])
    step = 1 / ev["sample_fps"]
    # Sample times on one grid for the whole half, so overlapping windows share frames.
    grid = [k * step for k in range(int(meta["n_frames"] / fps / step) + 1)]
    frame_of = {s: int(round(s * fps)) for s in grid}
    resolved = resolve_frames(config, match, arrays, set(frame_of.values()))
    perceived = {s: resolved[f] for s, f in frame_of.items() if f in resolved}

    client = APIClient(config["api"], config["paths"]["cache"])
    out_path = config["paths"]["outputs"] / "events" / args.match / f"half{args.half}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Walk the video forward once, keeping only the sampled frames still needed by the
    # current or a later window; each finished window's request is built and submitted.
    cap = cv2.VideoCapture(str(match.video_path(args.half)))
    wanted = {f: s for s, f in frame_of.items()}
    frames: dict[float, object] = {}
    futures, frame_idx, t0 = [], 0, time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        for window in windows:
            while frame_idx <= int(round(window[1] * fps)):
                if not cap.grab():
                    break
                if frame_idx in wanted:
                    ok, image = cap.retrieve()
                    if ok:
                        frames[wanted[frame_idx]] = image
                frame_idx += 1
            shown = [(s, frames[s]) for s in grid if window[0] <= s < window[1] and s in frames]
            request = build_request(args.half, window, shown, perceived, match.prematch, ev["sample_fps"])
            # The player behind each track as last seen in the window — what the model was
            # told — recorded next to the candidates for scoring attribution.
            players_of = {}
            for s, _ in shown:
                for t in perceived.get(s, ((), None, ()))[0]:
                    if t.player_id:
                        players_of[(args.half, *t.key)] = t.player_id
            futures.append(pool.submit(call, client, request, window, players_of))
            # The next window starts `window_overlap` s before this one ends.
            next_start = window[1] - ev["window_overlap"]
            frames = {s: im for s, im in frames.items() if s >= next_start}
        results = [f.result() for f in futures]

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in results if "error" not in r]
    n_cand = sum(len(r["candidates"]) for r in ok)
    lat = sorted(r["latency"] for r in ok)
    print(out_path)
    print(f"{len(windows)} windows, {len(ok)} answered ({sum(r['cached'] for r in ok)} from cache), "
          f"{len(results) - len(ok)} failed, {n_cand} candidates, {time.time() - t0:.0f} s wall")
    if lat:
        print(f"latency s: p50 {lat[len(lat) // 2]:.1f}  p90 {lat[int(len(lat) * 0.9)]:.1f}  max {lat[-1]:.1f}")
    for r in results:
        if "error" in r:
            print("failed", r["window"], r["error"][:200])


if __name__ == "__main__":
    main()
