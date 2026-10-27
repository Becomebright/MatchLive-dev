"""The commentator's voice (DESIGN.md §6.9): each phrase of the script spoken by a text-to-speech
model, at the energy the script gives it.

One model call per phrase through the runtime's API client (speech(), cached as MP3). The model
clones a reference recording's voice and delivery, and the energy ("calm", "lively", "rising",
"excited", chosen by the script's generator from the state of play) picks the recording made in
that manner, so the voice rises as an attack nears the box and a shot is called at full pitch, and
a remark in a lull is said in a relaxed one.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class Utterance:
    path: Path              # the MP3
    duration: float         # s of speech
    latency: float          # s the synthesis took


class NoVoice(Exception):
    """Raised by a speech function that has no voice at all (a dry run): the phrase stays a subtitle."""


class Voice:
    def __init__(self, config: Mapping[str, Any], speak: Callable[..., Any]):
        # `config` is the commentary.voice section; `speak(text, voice, reference, model)` is one
        # text-to-speech call returning an object with .path and .latency
        # (matchlive.common.api.APIClient.speech).
        self.cfg = config
        self.speak = speak

    def say(self, text: str, energy: str) -> Utterance:
        tones = self.cfg["tones"]
        result = self.speak(text, self.cfg["voice"], tones.get(energy, tones["lively"]), self.cfg["model"])
        return Utterance(result.path, audio_duration(result.path), result.latency)


def audio_duration(path: Path) -> float:
    """Seconds of audio in a file (ffprobe)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())
