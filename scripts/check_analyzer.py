"""Visual check of the event analyzer on one window of a live run: what it is shown, and what it answers.

    python scripts/check_analyzer.py --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" --run full --half 1 --start 1000

The frames carry the identities the live run gave them (its perceived_half<h>.pkl: a second in two
windows as the later one saw it).

Writes outputs/check_analyzer/<match>_h<half>_<start>s/: frames.jpg (the labelled frames as
a contact sheet), prompt.txt (the request's text) and, when the API answers (a cached
answer, or a live call through the endpoint in configs/api.local.yaml), answer.json and the parsed
candidates on stdout.
"""

from __future__ import annotations

import argparse
import json
import pickle

import cv2
import numpy as np

from matchlive.common.api import APIClient, NoAPIKey
from matchlive.common.config import load_config, run_overrides
from matchlive.common.match import Match
from matchlive.events.analyzer import build_request, parse_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--run", default="full", help="live run name (scripts/run_live.py --name)")
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--start", type=float, required=True, help="window start, second within the half")
    ap.add_argument("--no-call", action="store_true", help="only write the request, never call the API")
    args = ap.parse_args()

    run_dir = load_config()["paths"]["outputs"] / "live" / args.match / args.run
    config = load_config(overrides=run_overrides(run_dir))
    ev = config["events"]
    match = Match(config, args.match)
    perceived = pickle.load(open(run_dir / f"perceived_half{args.half}.pkl", "rb"))
    window = (args.start, args.start + ev["window_seconds"])
    cap = cv2.VideoCapture(str(match.video_path(args.half)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for s in sorted(s for s in perceived if window[0] <= s < window[1]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(s * fps)))
        ok, image = cap.read()
        if ok:
            frames.append((s, image))
    request = build_request(args.half, window, frames, perceived, match.prematch, ev["sample_fps"],
                            config["perception"]["inject"], ev["ask_confidence"])

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
