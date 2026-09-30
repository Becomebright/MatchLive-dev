"""The broadcast's own (human) commentary as timed text: the reference our commentary is measured
against (scripts/eval_commentary.py). For evaluation only — the live system never listens to the
broadcast (DESIGN.md §11 D1).

    python scripts/transcribe_broadcast.py [--match M ...] [--model whisper-1]

Each half's soundtrack is cut into 10-minute mono pieces and transcribed with segment timestamps
(APIClient.transcribe, cached). Writes outputs/reference/<match>/half<h>.json: [{"start", "end",
"text"}] in seconds of the half video.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from matchlive.common.api import APIClient
from matchlive.common.config import load_config
from matchlive.common.match import Match

MATCHES = ("2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United", "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool")
PIECE = 600               # s of audio per request (well under the upload limit as 48 kbit/s mono MP3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", action="append", default=None, help="default: both benchmark matches")
    ap.add_argument("--model", default="whisper-1", help="a speech-to-text model that returns segment times")
    args = ap.parse_args()
    config = load_config()
    client = APIClient(config["api"], config["paths"]["cache"])
    for name in args.match or MATCHES:
        match = Match(config, name)
        out_dir = config["paths"]["outputs"] / "reference" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for half in (1, 2):
            video = match.video_path(half)
            length = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                           str(video)], capture_output=True, text=True, check=True).stdout)
            segments = []
            with tempfile.TemporaryDirectory() as tmp:
                for start in range(0, int(length) + 1, PIECE):
                    piece = Path(tmp) / f"{start}.mp3"
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(PIECE), "-i", str(video),
                                    "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(piece)], check=True)
                    result = client.transcribe(piece, args.model)
                    segments += [{"start": start + s["start"], "end": start + s["end"], "text": s["text"].strip()}
                                 for s in result["segments"] if s["text"].strip()]
            (out_dir / f"half{half}.json").write_text(json.dumps(segments, indent=0))
            words = sum(len(s["text"].split()) for s in segments)
            talk = sum(s["end"] - s["start"] for s in segments)
            print(f"{name[-22:]} half {half}: {len(segments)} segments, {words} words, speech {talk / length:.0%} of "
                  f"{length / 60:.0f} min")


if __name__ == "__main__":
    main()
