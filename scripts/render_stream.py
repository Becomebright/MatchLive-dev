"""The delayed live stream with commentary (DESIGN.md §6.9, §10 phase 7): the broadcast video with
our commentary burned in as subtitles and spoken over the broadcast's own sound.

    python scripts/render_stream.py --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" [--run full]
        [--half 1] [--start 930 --end 1110] [--hls]

Input: a half video and the commentary of a live run on it (scripts/run_commentary.py). Output, in
outputs/stream/<match>/<run>/: half<h>[_<start>-<end>].mp4 — or with --hls, an HLS stream of the same
(<name>/index.m3u8 and 6-s segments, playable by any HLS player). It is what a viewer of the live
broadcast would get with the system in the loop: the output runs `commentary.output_delay` s behind
the feed, so it is laid out on the video's own timeline, each line on air from its scheduled time
(runtime.runner.run_commentary) for as long as its audio lasts. The broadcast's sound (its own commentary
and the crowd) stays underneath at `render.background_volume`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


from matchlive.common.config import load_config, run_overrides
from matchlive.common.match import Match

RATE = 48000              # Hz, stereo float: the commentary track is built at the broadcast's audio rate


def ass(lines: list[dict], start: float, font_size: int, delay: float) -> str:
    """The lines as ASS subtitles, timed from `start`: white on a translucent box at the bottom, and a
    small tag at the top saying the stream runs `delay` s behind the feed."""
    def stamp(s):
        s = max(0.0, s)
        return f"{int(s // 3600)}:{int(s // 60 % 60):02d}:{s % 60:05.2f}"
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Line,DejaVu Sans,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H90000000,0,0,0,0,100,100,0,0,3,8,0,2,80,80,36,1
Style: Tag,DejaVu Sans,{font_size * 2 // 3},&H00FFFFFF,&H000000FF,&H00000000,&H90000000,1,0,0,0,100,100,0,0,3,4,0,8,20,20,12,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    end = max([l["show"][2] - start for l in lines] + [0.0]) + 3600
    events = [f"Dialogue: 0,{stamp(0)},{stamp(end)},Tag,,0,0,0,,MatchLive commentary \\N(live, {delay:.0f} s delay)"]
    for line in lines:
        text = line["text"].replace("{", "(").replace("}", ")").replace("\n", " ")
        events.append(f"Dialogue: 0,{stamp(line['show'][1] - start)},{stamp(line['show'][2] - start)},Line,,0,0,0,,{text}")
    return head + "\n".join(events) + "\n"


def voice_track(lines: list[dict], start: float, duration: float) -> np.ndarray:
    """The commentary as one stereo track over [start, start + duration] of the half."""
    track = np.zeros((int(duration * RATE), 2), dtype=np.float32)
    for line in lines:
        if not line.get("audio"):
            continue
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", line["audio"], "-f", "f32le", "-ac", "2", "-ar", str(RATE), "-"],
                             capture_output=True, check=True).stdout
        clip = np.frombuffer(pcm, dtype=np.float32).reshape(-1, 2)
        at = int((line["show"][1] - start) * RATE)
        if at < 0:
            clip, at = clip[-at:], 0
        n = min(len(clip), len(track) - at)
        if n > 0:
            track[at:at + n] += clip[:n]
    return track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--run", default="full", help="live run whose commentary to use (scripts/run_commentary.py)")
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--start", type=float, default=0.0, help="s into the half (default: from kick-off)")
    ap.add_argument("--end", type=float, default=None, help="s into the half (default: the whole half)")
    ap.add_argument("--hls", action="store_true", help="write an HLS stream instead of an MP4")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a setting, e.g. commentary.output_delay=90")
    args = ap.parse_args()

    run_dir = load_config()["paths"]["outputs"] / "live" / args.match / args.run
    config = load_config(overrides=run_overrides(run_dir) + args.set)
    match = Match(config, args.match)
    cfg, render = config["commentary"], config["render"]
    lines = [json.loads(l) for l in open(config["paths"]["outputs"] / "commentary" / args.match / args.run / "lines.jsonl")]

    video = match.video_path(args.half)
    length = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                   str(video)], capture_output=True, text=True, check=True).stdout)
    start, end = args.start, min(args.end or length, length)
    shown = [l for l in lines if l["show"][0] == args.half and l["show"][2] > start and l["show"][1] < end]

    name = f"half{args.half}" + (f"_{int(start)}-{int(end)}" if (start, end) != (0.0, length) else "")
    out_dir = config["paths"]["outputs"] / "stream" / args.match / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "subs.ass").write_text(ass(shown, start, render["font_size"], cfg["output_delay"]))
        voice_track(shown, start, end - start).tofile(tmp / "voice.f32")
        graph = (f"[0:v]ass={tmp / 'subs.ass'}[v];"
                 f"[0:a]aresample={RATE},volume={render['background_volume']}[bg];"
                 f"[bg][1:a]amix=inputs=2:duration=first:normalize=0[a]")
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(end - start), "-i", str(video),
               "-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i", str(tmp / "voice.f32"),
               "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-c:a", "aac", "-b:a", "160k"]
        if args.hls:
            hls = out_dir / name
            hls.mkdir(exist_ok=True)
            cmd += ["-f", "hls", "-hls_time", "6", "-hls_playlist_type", "vod",
                    "-hls_segment_filename", str(hls / "seg_%05d.ts"), str(hls / "index.m3u8")]
            out = hls / "index.m3u8"
        else:
            out = out_dir / f"{name}.mp4"
            cmd += ["-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)
    print(out)
    print(f"{len(shown)} lines on air over {end - start:.0f} s ({sum(bool(l.get('audio')) for l in shown)} voiced)")


if __name__ == "__main__":
    main()
