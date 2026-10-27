"""Window-level identity (DESIGN.md §6.1 identity (f)): the analyzer is shown a window's frames
only once the window is over, so a track is named from the votes of the whole window — a number
read only in the window's last seconds also names the track in the frames before it."""

import copy

import cv2
import numpy as np

from matchlive.common.config import load_config
from matchlive.common.types import Track
from matchlive.perception.identity import IdentityResolver

BLUE, RED = (255, 0, 0), (0, 0, 255)            # BGR
ROSTER = {("home", "7"): "h7", ("away", "9"): "a9"}


def lab(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def reads(number):
    # One frame's jersey read of a track (top label, its probability): `number` read clearly, or
    # nothing readable.
    return np.array([np.nan, np.nan] if number is None else [int(number), 0.95], np.float32)


def window(first_read_frame, n_frames=30):
    """Two tracks over n_frames; both are read clearly (home #7, away #9) from `first_read_frame` on.
    Returns the resolver after voting every frame, the frames' tracks, and the causal answers."""
    cfg = copy.deepcopy(load_config()["perception"]["identity"])
    cfg["kit_burn_in"] = 2
    resolver = IdentityResolver(cfg, {"home": lab(BLUE), "away": lab(RED)})
    tracks = [Track(0, "player", 1, (0, 0, 20, 60), 1.0), Track(0, "player", 2, (50, 0, 20, 60), 1.0)]
    causal = []
    for f in range(n_frames):
        seen = (("7", "9") if f >= first_read_frame else (None, None))
        causal.append(resolver.update(f, tracks, np.stack([reads(seen[0]), reads(seen[1])]),
                                      np.stack([lab(BLUE), lab(RED)]), ROSTER, confirmed={}))
    return resolver, tracks, causal


def test_a_late_read_names_the_whole_window():
    resolver, tracks, causal = window(first_read_frame=25)
    # Frame by frame, nobody is named before the number has been read twice (min_number_votes): frame 25 is the first read ...
    assert [t.player_id for t in causal[0]] == [None, None]
    assert [t.player_id for t in causal[25]] == [None, None]
    # ... but at the window's end the same frames are named from the whole window's votes.
    named = resolver.resolve(tracks, ROSTER, confirmed={})
    assert [t.player_id for t in named] == ["h7", "a9"]
    assert [t.team for t in named] == ["home", "away"]


def test_a_window_without_reads_stays_unnamed():
    resolver, tracks, _ = window(first_read_frame=1000)
    assert [t.player_id for t in resolver.resolve(tracks, ROSTER, confirmed={})] == [None, None]


def test_memory_confirmed_identity_still_wins():
    resolver, tracks, _ = window(first_read_frame=0)
    named = resolver.resolve(tracks, ROSTER, confirmed={tracks[0].key: "a9"})
    assert named[0].player_id == "a9" and named[0].identity_source == "entry"
