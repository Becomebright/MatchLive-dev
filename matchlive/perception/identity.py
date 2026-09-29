"""Resolve player identity (team + jersey number -> player) for tracked players (ported from v2).

Two parts:
  * kit_color(): the per-crop kit-colour observation, computed during the video pass and
    stored in the raw perception cache (it needs pixels);
  * IdentityResolver: streaming, per-frame vote over those cached observations — call
    update() once per frame in order; identities at frame t use only frames <= t.

Team comes from kit colour matched in LAB space, number from the jersey classifier; both
are accumulated as per-track votes (tracks are keyed (shot_id, track_id)).

Kit colour differs from v2, measured on SoccerNet-GSR ground-truth crops: v2 k-means'd
the saturated pixels into dominant colours, which throws away white and black kits
(nearest-kit accuracy 0.74 per crop, 0.5 ms); the mean LAB colour of every non-grass,
non-skin pixel gets 0.95 at 0.05 ms.

Metadata kit colours are crude (pure "#0000FF" is far from a broadcast Chelsea blue, and
closer to the red kit's real colour than that blue is), so outfield colours are not
matched against them directly. Outfield colours seen so far are split into two clusters
(k-means once `kit_burn_in` observations are in, then online updates); the metadata
colours only name the clusters, by the cheaper of the two home/away pairings — which
needs their relative order to be right, not their values. Seeding prototypes with the
metadata colours and adapting them instead collapses: every early claim goes to one kit.

Changed from v2 (DESIGN.md §6.1):
  (a) numbers are voted only among `roster` (the players on the pitch); a confident read
      of a number outside it is not dropped but reported by off_roster_reads() as
      evidence for the consistency check;
  (b) within one frame a player is given to at most one track (Hungarian assignment);
  (c) `confirmed` track identities (from memory) are applied as given;
  (d) goalkeeper kit colours are matched only against goalkeeper tracks and resolved back
      to home / away, instead of acting as a 3rd and 4th team.
A referee detected as a player gets the nearer team's colour; it only becomes a player
if a roster number is read off it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from matchlive.common.types import Track
from matchlive.perception.jersey import UNREADABLE, label_to_number


def kit_bgr(jersey_colors: Mapping[str, list]) -> dict[str, tuple[int, int, int]]:
    """metadata `jersey_colors` ({kit: [name, "#rrggbb"]}) -> {kit: BGR}."""
    out = {}
    for kit, values in jersey_colors.items():
        hexv = next(v for v in values if v.startswith("#"))
        out[kit] = (int(hexv[5:7], 16), int(hexv[3:5], 16), int(hexv[1:3], 16))
    return out


def kit_lab(jersey_colors: Mapping[str, list]) -> dict[str, np.ndarray]:
    """metadata `jersey_colors` -> {kit: LAB colour}, the space kit colours are compared in
    (Euclidean distance in LAB tracks perceived colour difference far better than in BGR)."""
    return {kit: cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
            for kit, bgr in kit_bgr(jersey_colors).items()}


def roster_index(players: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    """metadata `players` entries -> {(team_side, jersey_number): player hash}."""
    return {(str(p["team_side"]), str(p["jersey_number"])): str(p["hash"])
            for p in players if str(p.get("jersey_number") or "-1") != "-1"}


def kit_color(crop: np.ndarray) -> np.ndarray:
    """Mean LAB colour of the crop's kit pixels (3,), or NaN if too few survive."""
    # Drop green grass and skin tones, keyed off HSV ranges (v2's filters); everything
    # else — including white and black kits — counts as kit.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    grass = (hue >= 35) & (hue <= 90) & (sat > 45)
    skin = (hue <= 25) & (sat >= 30) & (sat <= 180) & (val >= 65)
    pixels = crop[~grass & ~skin]
    if len(pixels) < 12:
        return np.full(3, np.nan, dtype=np.float32)
    # Convert only the kept pixels (as one 1xN image): same values, a third cheaper.
    return cv2.cvtColor(pixels[None], cv2.COLOR_BGR2LAB)[0].mean(axis=0).astype(np.float32)


class IdentityResolver:
    def __init__(self, config: dict[str, Any], kits: Mapping[str, np.ndarray]):
        # `config` is the perception.identity section of the YAML config.
        # `kits`: LAB colour per kit — "home", "away", optionally "home_goalkeeper" and
        #   "away_goalkeeper" (see kit_lab). Outfield ones only name the colour clusters.
        self.config = config
        self.home_away = (kits["home"], kits["away"]) if "home" in kits and "away" in kits else None
        self.keeper_kits = {k: kits[k] for k in ("home_goalkeeper", "away_goalkeeper") if k in kits}
        # Outfield colour clusters: observations until k-means, then the two centres.
        self.colors_seen: list[np.ndarray] = []
        self.centers: np.ndarray | None = None
        # Per track key: team votes ("home"/"away" for keepers, cluster 0/1 for outfield
        # players, named only when read so one naming covers the whole history) and
        # jersey-number votes.
        self.team_votes: dict[tuple, Counter] = defaultdict(Counter)
        self.number_votes: dict[tuple, Counter] = defaultdict(Counter)

    def update(self, frame_idx: int, tracks: Sequence[Track], jersey_probs: np.ndarray,
               colors: np.ndarray, roster: Mapping[tuple[str, str], str],
               confirmed: Mapping[tuple, str]) -> list[Track]:
        """Cast this frame's votes and return `tracks` with team / player_id /
        identity_source filled in. `jersey_probs` (N, 101) and `colors` (N, 3) are the
        cached observations of these tracks (NaN = none). `roster` ((team, number) ->
        player_id, see roster_index) and `confirmed` (track key -> player_id) come from
        memory and are passed every frame, since both change during a match."""
        cfg = self.config
        # A vote every `vote_every` frames (v2 used 15; reading every frame is affordable
        # now, so the default is 1).
        if frame_idx % cfg["vote_every"] == 0:
            for track, probs, color in zip(tracks, jersey_probs, colors):
                self._vote_team(track, color)
                if not np.isnan(probs[0]):
                    label = int(np.argmax(probs))
                    if label != UNREADABLE and probs[label] >= cfg["jersey_min_prob"]:
                        self.number_votes[track.key][label_to_number(label)] += 1

        # Each track's team, and its candidate players scored by number votes.
        teams = [self._team(t.key) for t in tracks]
        candidates: list[dict[str, int]] = []
        for track, team in zip(tracks, teams):
            cands = {}
            for number, votes in self.number_votes[track.key].items():
                if votes < cfg["min_number_votes"]:
                    continue
                # Unknown team (e.g. a keeper without keeper kit colours in the metadata):
                # the number alone decides, but only if exactly one roster player has it.
                sides = [s for s in ([team] if team else ("home", "away")) if (s, number) in roster]
                if len(sides) == 1:
                    cands[roster[(sides[0], number)]] = votes
            candidates.append(cands)

        # (c) confirmed identities first, then (b) one player per track in this frame:
        # maximise the total number of votes behind the chosen (track, player) pairs.
        assigned: dict[int, tuple[str, str]] = {
            i: (confirmed[t.key], "entry") for i, t in enumerate(tracks) if t.key in confirmed}
        taken = {player for player, _ in assigned.values()}
        open_tracks = [i for i in range(len(tracks)) if i not in assigned and candidates[i]]
        players = sorted({p for i in open_tracks for p in candidates[i] if p not in taken})
        if open_tracks and players:
            votes = np.array([[candidates[i].get(p, 0) for p in players] for i in open_tracks], dtype=float)
            if cfg["one_player_per_track"]:
                rows, cols = linear_sum_assignment(-votes)
            else:   # v2 behaviour: every track takes its own best candidate
                rows, cols = np.arange(len(open_tracks)), votes.argmax(1)
            for r, c in zip(rows, cols):
                if votes[r, c] > 0:
                    assigned[open_tracks[r]] = (players[c], "vote")

        side_of = {player: side for (side, _), player in roster.items()}
        out = []
        for i, (track, team) in enumerate(zip(tracks, teams)):
            player, source = assigned.get(i, (None, None))
            # A number-decided player also fixes an unknown team.
            out.append(replace(track, team=side_of.get(player, team), player_id=player, identity_source=source))
        return out

    def off_roster_reads(self, teams: Mapping[tuple, str | None], roster: Mapping[tuple[str, str], str],
                         squad: Mapping[tuple[str, str], str]) -> list[tuple[tuple, str]]:
        """(a) Evidence for the consistency check: (track key, player) for every track of known
        team whose most-voted number (at least `min_number_votes` votes) names a player of the
        whole `squad` who is not in `roster` (the players on the pitch) — and is worn by nobody
        on the pitch in either team. A number some player on the pitch wears is far likelier
        that player with the kit colour misjudged: on the two local benchmark matches, such
        reads came from ~250 tracks before any substitution, the others almost only after a
        real one. `teams` maps track key -> team (from the tracks update() returned)."""
        worn = {number for _, number in roster}
        out = []
        for key, team in teams.items():
            votes = self.number_votes.get(key)
            if not team or not votes:
                continue
            number, n = votes.most_common(1)[0]
            if n >= self.config["min_number_votes"] and number not in worn and (team, number) in squad:
                out.append((key, squad[(team, number)]))
        return out

    def _vote_team(self, track: Track, color: np.ndarray) -> None:
        if np.isnan(color[0]):
            return
        key = track.key
        if track.role == "goalkeeper":
            # (d) keepers are matched against keeper kits only (too few keepers on screen
            # to cluster); without keeper kit colours in the metadata, no colour vote.
            kit = self._nearest(color, self.keeper_kits)
            if kit is not None:
                self.team_votes[key][kit.removesuffix("_goalkeeper")] += 1
            return
        if self.home_away is None:
            return
        if self.centers is None:
            self.colors_seen.append(color)
            if len(self.colors_seen) < self.config["kit_burn_in"]:
                return
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
            _, _, self.centers = cv2.kmeans(np.stack(self.colors_seen), 2, None, criteria, 5,
                                            cv2.KMEANS_PP_CENTERS)
            self.colors_seen = []
        cluster = self._nearest(color, {0: self.centers[0], 1: self.centers[1]})
        if cluster is not None:
            self.team_votes[key][cluster] += 1
            # Online k-means step: the claimed colour pulls its cluster centre towards it.
            self.centers[cluster] += self.config["kit_adapt_rate"] * (color - self.centers[cluster])

    def _team(self, key: tuple) -> str | None:
        votes = self.team_votes.get(key)
        if not votes:
            return None
        top = votes.most_common(1)[0][0]
        if isinstance(top, str):        # a keeper's vote is already a team
            return top
        # Name the clusters by the cheaper pairing with the metadata home/away colours.
        (home, away), (c0, c1) = self.home_away, self.centers
        straight = np.linalg.norm(c0 - home) + np.linalg.norm(c1 - away)
        swapped = np.linalg.norm(c0 - away) + np.linalg.norm(c1 - home)
        return ("home", "away")[top] if straight <= swapped else ("away", "home")[top]

    def _nearest(self, color: np.ndarray, protos: Mapping) -> Any:
        """The nearest prototype's key, or None when the choice is ambiguous: the relative
        gap between nearest and farthest must reach `team_min_margin` (one prototype is
        always confident)."""
        if not protos:
            return None
        dist = {k: float(np.linalg.norm(color - p)) for k, p in protos.items()}
        near, far = min(dist.values()), max(dist.values())
        margin = (far - near) / (far + near + 1e-6) if len(dist) > 1 else 1.0
        return min(dist, key=dist.get) if margin >= self.config["team_min_margin"] else None
