"""The broadcast's own (human) commentary as timed text: the reference our commentary is measured
against (scripts/eval_commentary.py). For evaluation only — the live system never listens to the
broadcast (DESIGN.md §11 D1).

    python scripts/transcribe_broadcast.py [--match M ...] [--model whisper-1]

Each half's soundtrack is cut into 10-minute mono pieces and transcribed with segment timestamps
(APIClient.transcribe, cached). Writes outputs/reference/<match>/half<h>.json: [{"start", "end",
"text"}] in seconds of the half video.

With `--names`, each half is transcribed again by the open Qwen3-ASR (local GPU), told the
match's player and team names, which Whisper mostly gets wrong (Clyne heard as "Klein"), and
timed word by word by Qwen3-ForcedAligner: half<h>.names.json, the same format.
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
    ap.add_argument("--names", action="store_true", help="transcribe again with Qwen3-ASR and the names")
    args = ap.parse_args()
    if args.names:
        return names(args.match or MATCHES)
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


def names(matches, asr_id="Qwen/Qwen3-ASR-1.7B-hf", aligner_id="Qwen/Qwen3-ForcedAligner-0.6B-hf", piece=30.0,
          batch=8):
    """Each half again with the open Qwen3-ASR, told the match's names, in pieces of about `piece`
    s cut at the quietest moment near each boundary; Qwen3-ForcedAligner times every word, and
    words are grouped into segments at sentence ends and pauses."""
    import numpy as np
    import torch
    from transformers import AutoModelForMultimodalLM, AutoModelForTokenClassification, AutoProcessor
    config = load_config()
    asr_proc = AutoProcessor.from_pretrained(asr_id)
    asr = AutoModelForMultimodalLM.from_pretrained(asr_id, dtype=torch.bfloat16).to("cuda").eval()
    al_proc = AutoProcessor.from_pretrained(aligner_id)
    aligner = AutoModelForTokenClassification.from_pretrained(aligner_id, dtype=torch.bfloat16).to("cuda").eval()
    sr = 16000
    for name in matches:
        match = Match(config, name)
        out_dir = config["paths"]["outputs"] / "reference" / name
        vocab = sorted({n for p in match.prematch["players"] for n in (p["short_name"], p["long_name"])})
        prompt = "Vocabulary: " + ", ".join(vocab + list(match.prematch["teams"].values())) + "."
        for half in (1, 2):
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(match.video_path(half)), "-vn", "-ac", "1",
                                  "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True).stdout
            audio = np.frombuffer(raw, dtype=np.float32)
            # Cut points: the quietest 0.1 s within 3 s of every `piece` s.
            energy = np.sqrt(np.mean(audio[:len(audio) // 1600 * 1600].reshape(-1, 1600) ** 2, axis=1))
            cuts = [0]
            while cuts[-1] + piece * 10 + 30 < len(energy):
                lo = cuts[-1] + int(piece * 10) - 30
                cuts.append(lo + int(np.argmin(energy[lo:lo + 60])))
            cuts.append(len(audio) // 1600)
            pieces = [(a / 10, audio[a * 1600:b * 1600]) for a, b in zip(cuts, cuts[1:])]
            segments = []
            for i in range(0, len(pieces), batch):
                starts, clips = zip(*pieces[i:i + batch])
                inputs = asr_proc.apply_transcription_request(audio=list(clips), prompt=[prompt] * len(clips),
                                                              language=["English"] * len(clips)).to("cuda", torch.bfloat16)
                with torch.inference_mode():
                    out = asr.generate(**inputs, max_new_tokens=320, do_sample=False)
                texts = [t.strip() for t in asr_proc.decode(out[:, inputs["input_ids"].shape[1]:],
                                                            return_format="transcription_only")]
                inputs, word_lists = al_proc.prepare_forced_aligner_inputs(audio=list(clips), transcript=texts,
                                                                           language="English")
                inputs = inputs.to("cuda", torch.bfloat16)
                with torch.inference_mode():
                    logits = aligner(**inputs).logits
                timed = al_proc.decode_forced_alignment(logits, inputs["input_ids"], word_lists,
                                                        aligner.config.timestamp_token_id)
                for start, text, words in zip(starts, texts, timed):
                    # The aligner's words lose punctuation; take the transcript's own tokens when they
                    # line up one to one, so sentence ends can split segments.
                    tokens = text.split()
                    if len(tokens) != len(words):
                        tokens = [w["text"] for w in words]
                    seg = None
                    for tok, w in zip(tokens, words):
                        t0, t1 = start + w["start_time"], start + w["end_time"]
                        if seg is None or t0 - seg["end"] > 0.6:
                            seg = {"start": t0, "end": t1, "text": tok}
                            segments.append(seg)
                        else:
                            seg["end"], seg["text"] = t1, seg["text"] + " " + tok
                        if tok[-1] in ".?!":
                            seg = None
            segments = [s | {"start": round(s["start"], 2), "end": round(s["end"], 2)} for s in segments]
            (out_dir / f"half{half}.names.json").write_text(json.dumps(segments, indent=0))
            words = sum(len(s["text"].split()) for s in segments)
            print(f"{name[-22:]} half {half}: {len(pieces)} pieces, {len(segments)} segments, {words} words", flush=True)


if __name__ == "__main__":
    main()
