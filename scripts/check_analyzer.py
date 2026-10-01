"""Visual check of the event analyzer on one window: what it is shown, and what it answers.

    python scripts/check_analyzer.py --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" --half 1 --start 1000

Writes outputs/check_analyzer/<match>_h<half>_<start>s/: frames.jpg (the labelled frames as
a contact sheet), prompt.txt (the request's text) and, when the API answers (a cached
answer, or a live call through the endpoint in configs/api.local.yaml), answer.json and the parsed
candidates on stdout.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from matchlive.common.api import APIClient, NoAPIKey
from matchlive.common.cache import load_perception
from matchlive.common.config import load_config
from matchlive.common.match import Match
from matchlive.events.analyzer import build_request, parse_candidates
from matchlive.runtime.runner import MatchRunner, resolve_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--start", type=float, required=True, help="window start, second within the half")
    ap.add_argument("--no-call", action="store_true", help="only write the request, never call the API")
    args = ap.parse_args()

    config = load_config()
    ev = config["events"]
    match = Match(config, args.match)
    arrays, meta = load_perception(MatchRunner(config).run_perception(match, args.half))
    fps = meta["fps"]
    window = (args.start, args.start + ev["window_seconds"])
    seconds = [window[0] + k / ev["sample_fps"] for k in range(int(ev["window_seconds"] * ev["sample_fps"]))]
    frame_of = {s: int(round(s * fps)) for s in seconds}

    resolved = resolve_frames(config, match, arrays, set(frame_of.values()))
    perceived = {s: resolved[f] for s, f in frame_of.items() if f in resolved}
    cap = cv2.VideoCapture(str(match.video_path(args.half)))
    frames = []
    for s in seconds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_of[s])
        ok, image = cap.read()
        if ok:
            frames.append((s, image))
    request = build_request(args.half, window, frames, perceived, match.prematch, ev["sample_fps"])

    out_dir = config["paths"]["outputs"] / "check_analyzer" / f"{args.match[:10]}_h{args.half}_{int(args.start)}s"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = request.parts[0]
    (out_dir / "prompt.txt").write_text(text)
    tiles = []
    for jpg in request.parts[2::2]:
        tile = cv2.resize(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR), (480, 270))
        cv2.putText(tile, str(len(tiles) + 1), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out_dir / "frames.jpg"), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
    print(out_dir)
    print(text.split("Player IDs:")[1])

    client = APIClient(config["api"], config["paths"]["cache"])
    if args.no_call:
        print("\n(no API call: --no-call)")
        return
    try:
        result = client.generate(request.parts, json_output=True)
    except NoAPIKey as e:
        print(f"\n(no API call: {e})")
        return
    (out_dir / "answer.json").write_text(result.text)
    print(f"\n{result.model}  latency {result.latency:.1f} s  cached {result.cached}  call {result.call_id}")
    for c in parse_candidates(result.text, request):
        print(json.dumps(c.__dict__, default=str))


if __name__ == "__main__":
    main()
