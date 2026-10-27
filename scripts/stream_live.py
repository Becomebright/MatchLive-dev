"""The delayed live stream with commentary, streamed as it plays (DESIGN.md §6.10): the half video,
read at its real rate as if it were the live feed, goes out `commentary.output_delay` (D) s behind
it, with our commentary spoken over the broadcast's own sound and drawn as subtitles.

    python scripts/stream_live.py --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" --run full_2xl_window \
        --name script_h2 --half 2 [--start 300 --end 420] [--serve 8090] [--keep]

Then, on your own machine: `ssh -N -L 8090:localhost:8090 <server>` and open http://localhost:8090/.

Output, in outputs/stream/<match>/<name>/live_half<h>/: a live HLS stream (index.m3u8, 2-s segments,
the last minute kept unless --keep) and index.html, a player page (hls.js); --serve serves that folder
on localhost:<port>.

Everything is on the video's timeline. Video second t goes out at live time t + D; a phrase is on air
from its scheduled start (`show`, runtime.runner.run_commentary) and becomes known at its ready time
(`t_ready`, live time). The schedule only puts a phrase on air from ready - D on, so by the time the
output reaches video second t every phrase starting at or before t is known: the stream never waits,
and its picture, subtitles and sound are what scripts/render_stream.py renders offline, just made
frame by frame. The phrases come from a finished commentary run (lines.jsonl) here, released at their
ready times as a live run would emit them; a phrase that would arrive after its slot is counted as
missed (none should be).

One ffmpeg encodes the whole stream: raw frames on its stdin, the mixed sound as PCM through a named
pipe. Both inputs are timed by count (frames at 25 fps, samples at 48 kHz) from the same start, so
picture and sound stay in step by construction, and one continuous AAC encoder leaves no seams.
The session starts as if the system had been running for D s already: the first frame out is video
second --start, without first waiting D s; nothing in the timing depends on where the session begins.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from matchlive.common.config import load_config, run_overrides
from matchlive.common.match import Match
from render_stream import RATE, ambience

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

PLAYER = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MatchLive stream</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #0d1210; color: #dce6e0; font: 14px system-ui, sans-serif;
         display: grid; place-items: center; min-height: 100vh; }
  main { width: min(100vw - 32px, 1280px); display: grid; gap: 10px; }
  video { width: 100%; background: #000; border-radius: 4px; }
  p { margin: 0; color: #8fa399; }
</style></head>
<body><main>
  <video id="v" controls autoplay muted playsinline></video>
  <p id="s">Connecting to the stream…  (unmute to hear the commentary)</p>
</main>
<script src="https://cdn.jsdelivr.net/npm/hls.js@1.5.17/dist/hls.min.js"></script>
<script>
  const v = document.getElementById("v"), s = document.getElementById("s");
  if (window.Hls && Hls.isSupported()) {
    const hls = new Hls({ liveSyncDurationCount: 3 });
    hls.loadSource("index.m3u8"); hls.attachMedia(v);
    hls.on(Hls.Events.MANIFEST_PARSED, () => { s.textContent = "Live. Unmute to hear the commentary."; v.play(); });
    hls.on(Hls.Events.ERROR, (_, d) => { if (d.fatal) s.textContent = "Stream error: " + d.details + " (reload once the stream is up)"; });
  } else { v.src = "index.m3u8"; }   // Safari plays HLS natively
</script>
</body></html>
"""


def load_clip(path: str) -> np.ndarray:
    """A phrase's audio as stereo float32 at RATE."""
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(RATE), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.float32).reshape(-1, 2)


def band(text: str, width: int, font: ImageFont.FreeTypeFont, top: bool = False) -> tuple[int, np.ndarray]:
    """A subtitle as an RGBA strip the width of the frame: white text on a translucent box, at the
    bottom (or, for the tag, the top). Returns (the strip's first row, the strip)."""
    lines, words = [], text.split()
    while words:                                       # wrap to 80% of the width
        n = len(words)
        while n > 1 and font.getlength(" ".join(words[:n])) > 0.8 * width:
            n -= 1
        lines.append(" ".join(words[:n])); words = words[n:]
    pad, gap = font.size // 2, font.size // 4
    h = len(lines) * (font.size + gap) - gap + 2 * pad
    w = int(max(font.getlength(l) for l in lines)) + 2 * pad
    img = Image.new("RGBA", (width, h + 2 * pad))
    d = ImageDraw.Draw(img)
    x0 = (width - w) // 2
    d.rounded_rectangle([x0, pad, x0 + w, pad + h], radius=pad // 2, fill=(0, 0, 0, 150))
    for i, l in enumerate(lines):
        d.text(((width - font.getlength(l)) / 2, 2 * pad + i * (font.size + gap)), l, font=font, fill=(255, 255, 255, 255))
    return (0 if top else 720 - img.height - font.size), np.asarray(img)


def draw(frame: np.ndarray, at: int, strip: np.ndarray) -> None:
    """Alpha-blend an RGBA strip onto the frame (in place) from row `at`."""
    rows = frame[at:at + strip.shape[0]]
    a = strip[..., 3:4].astype(np.float32) / 255
    rows[:] = (rows * (1 - a) + strip[..., :3] * a).astype(np.uint8)


def writer(pipe, items: queue.Queue) -> None:
    """Write what the loop produces to one of ffmpeg's inputs, in its own thread so that a full pipe on
    one input never stalls the other."""
    while (b := items.get()) is not None:
        pipe.write(b)
    pipe.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--run", default="full", help="the live run (scripts/run_live.py --name)")
    ap.add_argument("--name", default=None, help="the commentary folder (scripts/run_commentary.py --name; default: the run)")
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--start", type=float, default=0.0, help="s into the half (default: from kick-off)")
    ap.add_argument("--end", type=float, default=None, help="s into the half (default: the whole half)")
    ap.add_argument("--serve", type=int, default=None, metavar="PORT", help="serve the stream on localhost:PORT")
    ap.add_argument("--keep", action="store_true", help="keep every segment (for checking the stream afterwards)")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = ap.parse_args()

    run_dir = load_config()["paths"]["outputs"] / "live" / args.match / args.run
    config = load_config(overrides=run_overrides(run_dir) + args.set)
    match = Match(config, args.match)
    cfg, render = config["commentary"], config["render"]
    name = args.name or args.run
    delay = cfg["output_delay"]
    lines = [json.loads(l) for l in open(config["paths"]["outputs"] / "commentary" / args.match / name / "lines.jsonl")]
    video = match.video_path(args.half)
    length = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                   str(video)], capture_output=True, text=True, check=True).stdout)
    start, end = args.start, min(args.end or length, length)
    fps, size = 25, (1280, 720)                        # the 720p match videos (DESIGN.md §4)
    per_frame = RATE // fps                            # audio samples per frame

    # The phrases of this half, in the order a live run would emit them (ready time).
    feed = sorted((l for l in lines if l["show"][0] == args.half and l["show"][2] > start and l["show"][1] < end),
                  key=lambda l: l["t_ready"][1])

    out = config["paths"]["outputs"] / "stream" / args.match / name / f"live_half{args.half}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    (out / "index.html").write_text(PLAYER)
    if args.serve:
        class Handler(http.server.SimpleHTTPRequestHandler):
            # The system's MIME table may take .ts for a Qt translation file.
            extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                              ".ts": "video/mp2t", ".m3u8": "application/vnd.apple.mpegurl"}

            def log_message(self, *a):            # quiet: a player asks every second
                pass
        handler = partial(Handler, directory=str(out))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.serve), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"serving {out} on http://localhost:{args.serve}/ (from your machine: ssh -N -L "
              f"{args.serve}:localhost:{args.serve} <this server>)", flush=True)

    # The sound under ours, as in render_stream.py: the crowd and the pitch, or the whole broadcast quieter.
    if render["ambience"]:
        under_src = ambience(video, config["paths"]["cache"] / "ambience" / args.match / f"half{args.half}.flac")
        volume = render["ambience_volume"]
    else:
        under_src, volume = video, render["background_volume"]
    under = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(under_src), "-vn", "-f", "f32le",
                              "-ac", "2", "-ar", str(RATE), "-"], stdout=subprocess.PIPE)
    frames = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(video), "-an", "-f", "rawvideo",
                               "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-"],
                              stdout=subprocess.PIPE)

    fifo = out / "audio.fifo"
    os.mkfifo(fifo)
    hls = ["-f", "hls", "-hls_time", "2", "-hls_segment_filename", str(out / "seg_%06d.ts")]
    hls += ["-hls_list_size", "0"] if args.keep else ["-hls_list_size", "30", "-hls_flags", "delete_segments"]
    encoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps),
         "-thread_queue_size", "64", "-i", "pipe:0",
         "-f", "f32le", "-ar", str(RATE), "-ac", "2", "-thread_queue_size", "256", "-i", str(fifo),
         "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
         "-g", str(2 * fps), "-keyint_min", str(2 * fps), "-sc_threshold", "0", "-b:v", "3M",
         "-c:a", "aac", "-b:a", "160k", *hls, str(out / "index.m3u8")], stdin=subprocess.PIPE)
    vq, aq = queue.Queue(maxsize=50), queue.Queue(maxsize=50)
    threads = [threading.Thread(target=writer, args=(encoder.stdin, vq)),
               threading.Thread(target=lambda: writer(open(fifo, "wb"), aq))]
    for t in threads:
        t.start()

    font = ImageFont.truetype(FONT, render["font_size"])
    tag = band(f"MatchLive commentary (live, {delay:.0f} s delay)", size[0], ImageFont.truetype(FONT, render["font_size"] * 2 // 3), top=True)
    known: list[dict] = []                             # phrases released so far, in order of their start
    clips: list[tuple[int, np.ndarray]] = []           # (first sample, audio) of the voiced ones
    strips: dict[int, tuple[int, np.ndarray]] = {}     # subtitle strip per phrase (by id)
    aired, missed, late_s, nxt = 0, 0, 0.0, 0
    n_frames = int((end - start) * fps)
    t0 = time.perf_counter()
    for k in range(n_frames):
        t = start + k / fps                            # video second of this frame
        # Live time is t + D: release every phrase ready by then.
        while nxt < len(feed) and feed[nxt]["t_ready"][1] <= t + delay:
            l = feed[nxt]; nxt += 1
            if l["t_ready"][1] > l["show"][1] + delay + 1 / fps:   # ready after its slot went out
                missed += 1
                continue
            aired += 1
            known.append(l); known.sort(key=lambda x: x["show"][1])
            if l.get("audio"):
                clips.append((round((l["show"][1] - start) * RATE), load_clip(l["audio"])))
        # Picture: the frame, the tag, and the phrase on air (up a second after it is said, unless the
        # next one starts first; the same rule as render_stream.ass).
        raw = frames.stdout.read(size[0] * size[1] * 3)
        if len(raw) < size[0] * size[1] * 3:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(size[1], size[0], 3).copy()
        draw(frame, *tag)
        for i, l in enumerate(known):
            until = l["show"][2] + 1.0
            if i + 1 < len(known):
                until = max(l["show"][2], min(until, known[i + 1]["show"][1]))
            if l["show"][1] <= t < until:
                key = id(l)
                if key not in strips:
                    strips[key] = band(l["text"].replace("\n", " "), size[0], font)
                draw(frame, *strips[key])
        # Sound: this frame's stretch of the broadcast underneath, plus every phrase overlapping it.
        s0 = k * per_frame
        bg = under.stdout.read(per_frame * 8)
        block = np.zeros((per_frame, 2), dtype=np.float32)
        if bg:
            b = np.frombuffer(bg, dtype=np.float32).reshape(-1, 2)
            block[:len(b)] = b * volume
        for at, clip in clips:
            lo, hi = max(s0, at), min(s0 + per_frame, at + len(clip))
            if lo < hi:
                block[lo - s0:hi - s0] += clip[lo - at:hi - at]
        clips = [(at, c) for at, c in clips if at + len(c) > s0 + per_frame]
        known = [l for l in known if l["show"][2] + 1.0 > t]   # done with: off the screen for good
        # Out at its time: frame k leaves at session start + k / fps (live time t + D).
        wait = t0 + k / fps - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        else:
            late_s = max(late_s, -wait)
        vq.put(frame.tobytes()); aq.put(block.tobytes())
        if k % (10 * fps) == 0:
            print(f"{args.half}H {t:7.1f}s out | phrases on air so far {aired}, missed {missed} | "
                  f"max lag {late_s:.2f}s", flush=True)
    vq.put(None); aq.put(None)
    for t in threads:
        t.join()
    encoder.wait()
    for p in (under, frames):
        p.kill()
    fifo.unlink()
    print(f"done: {k + 1} frames ({(k + 1) / fps:.0f} s) in {time.perf_counter() - t0:.0f} s; phrases on air "
          f"{aired}, missed {missed}; max lag behind the clock {late_s:.2f} s -> {out / 'index.m3u8'}")
    if args.serve:
        print("the stream has ended; still serving (Ctrl-C to stop)", flush=True)
        threading.Event().wait()


if __name__ == "__main__":
    main()
