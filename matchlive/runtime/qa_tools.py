"""Read-only tools the runtime lends the QA agent (DESIGN.md §6.9), bounded by the question time.

Every answer uses only what was known when the question was asked (`now`): memory as known by
then (a prefix of the op log, so no entry, status or verdict from later), perception and frames
up to then. Memory, perception and video are held here, not by the agent (DESIGN.md §8).

The shared memory tools (get_state, list_events, get_player, why, get_event_evidence) read the
match memory; get_frames is the arbiter's frame tool; get_current_view shows the last seconds
with each tracked player labelled and identified, as the event analyzer is shown them.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import replace
from typing import Any, Mapping

import cv2
import numpy as np

from matchlive.arbiter.agent import Evidence
from matchlive.common.match import Match
from matchlive.common.types import Track
from matchlive.events.analyzer import clock, labelled_frames
from matchlive.memory.store import MatchMemory
from matchlive.memory.types import Event, Time
from matchlive.runtime.evidence import EvidenceTools


class QATools:
    def __init__(self, config: dict[str, Any], match: Match, mem: MatchMemory,
                 perceived: Mapping[int, Mapping[float, tuple]], sources: Mapping[str, Mapping[str, Any]]):
        # `mem`: the run's memory (its op log); `perceived`: half -> sampled second -> the frame
        # as the analyzer was told about it (runtime.runner.run_live); `sources`: observation id
        # -> the analyzer window and candidate behind it.
        self.config, self.cfg = config, config["qa"]
        self.match, self.mem, self.perceived, self.sources = match, mem, perceived, sources
        self.prematch = match.prematch
        self.players = {p["hash"]: p for p in self.prematch["players"]}
        self.replay = config["loop"]["replay"]
        self.video = EvidenceTools(config, match)
        for half in (1, 2):         # loaded up front: questions are answered from several threads
            self.video._load(half)
        # How long the first half ran, to reach back across half-time ("the last 10 minutes").
        self.first_half = max(perceived[1])
        # Each track's identity as voted over time: (half, shot, track) -> ([second], [(team, player)]),
        # so a frame from the past can be labelled with what was known of its players by now.
        self.voted: dict[tuple, tuple[list, list]] = {}
        for half, frames in perceived.items():
            for s in sorted(frames):
                for tr in frames[s][0]:
                    seconds, ids = self.voted.setdefault((half, *tr.key), ([], []))
                    seconds.append(s)
                    ids.append((tr.team, tr.player_id))

    def call(self, now: Time, tool: str, args: Mapping[str, Any]) -> Evidence:
        try:
            if tool == "get_state":
                return self._state(now)
            if tool == "list_events":
                return self._list_events(now, args)
            if tool == "get_player":
                return self._player(now, str(args["name"]))
            if tool == "why":
                return self._why(now, str(args["entry"]))
            if tool == "get_event_evidence":
                return self._event_evidence(now, str(args["entry"]))
            if tool == "get_frames":
                return self._frames(now, int(args["half"]), float(args["start"]), float(args["end"]))
            if tool == "get_current_view":
                return self._frames(now, now[0], now[1] - self.cfg["current_view"], now[1])
        except (KeyError, TypeError, ValueError, IndexError) as e:
            return Evidence(f"error: bad arguments for {tool} ({type(e).__name__}: {e})")
        return Evidence(f"error: unknown tool {tool!r}")

    # ---- the whole log, for qa.mode dump_log (DESIGN.md §7, v2's baseline) ----

    def dump_log(self, now: Time, status: bool = True) -> str:
        """The squads and every event known by `now`. `status` off drops the confirmed / pending
        tags: the events baseline reads a run without the consistency check, where every
        analyzer candidate is its own entry and nothing is pending (results §17)."""
        lines = [self.squads(), "", "Event log so far:"] + [self._line(r, status) for r in self._events(now)]
        return "\n".join(lines)

    def squads(self) -> str:
        teams = self.prematch["teams"]
        lines = [f"Squads (number, name, starting XI or substitute):"]
        for side in ("home", "away"):
            lines.append(f"{teams[side]} ({side}): " + ", ".join(
                self._name(p["hash"]) + ("" if p["starting"] else " (sub)")
                for p in self.prematch["players"] if p["team_side"] == side))
        return "\n".join(lines)

    # ---- the last seconds of raw video, for the video baseline (results §17) ----

    def raw_view(self, now: Time) -> list[str | bytes]:
        """The sampled frames of the last qa.current_view seconds up to `now`, as broadcast: no
        player labels, no identities. ["Frame k (clock)", JPEG, ...]."""
        half, end = now
        seconds = sorted(s for s in self.perceived[half] if end - self.cfg["current_view"] <= s <= end)
        parts: list[str | bytes] = []
        for k, (second, image) in enumerate(self._read(half, seconds[-self.cfg["max_frames"]:]), start=1):
            # The size and quality the labelled frames are shown at (events.analyzer.labelled_frames).
            ok, jpg = cv2.imencode(".jpg", cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA),
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            parts += [f"Frame {k} (half {half}, {clock(second)})", jpg.tobytes()]
        return parts

    # ---- memory ----

    def _events(self, now: Time) -> list[dict]:
        """Every event entry known by `now` and not retracted, with its best time estimate, in
        time order. An entry counts from when it was written, even if its interval (the time
        it has certainly happened) runs past now."""
        past = self.mem.at(now).mem
        track_players = past.track_players()
        out = []
        for eid, e in past.entries.items():
            if not isinstance(e, Event):
                continue
            status = past.status(eid)
            if status == "retracted":
                continue
            half, second, exact = self._when(past, eid, e)
            out.append({"id": eid, "event": e, "status": status, "half": half, "second": second, "exact": exact,
                        "players": past.event_players(eid, track_players)})
        return sorted(out, key=lambda r: (r["half"], r["second"]))

    def _when(self, past: MatchMemory, eid: str, e: Event) -> tuple[int, float, bool]:
        """(half, second, exact): the median second at which the analyzer saw it live, or the
        middle of its interval (verdicts add events there); a replay sighting only bounds it."""
        seen = sorted(self.sources[j.obs_id]["candidate"]["second"] for _, j in past.justifications[eid]
                      if j.obs_id in self.sources and past.holds(j)
                      and not (self.replay and self.sources[j.obs_id]["candidate"]["replay"]))
        (half, lo), (_, hi) = e.t_interval
        if seen:
            return half, seen[len(seen) // 2], True
        first = past.justifications[eid][0][1].obs_id
        replay_only = first in self.sources and self.sources[first]["candidate"]["replay"] and self.replay
        return half, (lo + hi) / 2, not replay_only

    def _line(self, r: dict, status: bool = True) -> str:
        e, teams = r["event"], self.prematch["teams"]
        (_, lo), (_, hi) = e.t_interval
        when = (f"half {r['half']} {clock(r['second'])} ({minute_phrase(r['half'], r['second'])})" if r["exact"]
                else f"half {r['half']}, seen in a replay: between {clock(lo)} and {clock(hi)}")
        team = f"{teams[e.team]} ({e.team})" if e.team else "team unknown"
        who = "; ".join(f"{p.role}: {self._name(pid) if pid else 'unidentified'}"
                        for p, pid in zip(e.participants, r["players"])) or "no players named"
        tag = f" [{r['status']}]" if status else ""
        return f"{r['id']}{tag} {e.type} — {team} — {when} — {who}"

    def _name(self, pid: str) -> str:
        p = self.players[pid]
        return f"#{p['jersey_number']} {p['long_name']}"

    def _state(self, now: Time) -> Evidence:
        view = self.mem.at(now)
        teams = self.prematch["teams"]
        events = self._events(now)
        lines = [f"Now: half {now[0]}, {clock(now[1])}.",
                 f"Score (confirmed goals): {teams['home']} {view.score['home']} - {view.score['away']} {teams['away']}."]
        pending_goals = [self._line(r) for r in events if r["event"].type == "goal" and r["status"] == "pending"]
        if pending_goals:
            lines += ["Pending goals (not counted in the score):"] + pending_goals
        for side in ("home", "away"):
            on = ", ".join(self._name(p) + (f" ({self.players[p]['position']})" if self.players[p]["position"] else "")
                           for p in view.on_field[side])
            lines.append(f"{teams[side]} on the pitch: {on}")
            lines.append(f"{teams[side]} bench: " + ", ".join(self._name(p) for p in view.bench[side]))
        for kind in ("substitution", "yellow_card", "red_card"):
            rows = [self._line(r) for r in events if r["event"].type == kind]
            lines += [f"{kind.replace('_', ' ')}s so far:"] + (rows or ["(none)"])
        lines.append("Event counts so far (confirmed + pending):")
        for side in ("home", "away"):
            counts: dict[str, list[int]] = {}
            for r in events:
                if r["event"].team == side:
                    c = counts.setdefault(r["event"].type, [0, 0])
                    c[r["status"] != "confirmed"] += 1
            lines.append(f"{teams[side]}: " + ", ".join(f"{k} {c[0]}+{c[1]}" for k, c in sorted(counts.items())))
        return Evidence("\n".join(lines))

    def _list_events(self, now: Time, args: Mapping[str, Any]) -> Evidence:
        events = self._events(now)
        if args.get("type"):
            types = args["type"] if isinstance(args["type"], list) else [args["type"]]
            events = [r for r in events if r["event"].type in types]
        if args.get("team"):
            side = self._side(str(args["team"]))
            events = [r for r in events if r["event"].team == side]
        if args.get("player"):
            ids = self._find(str(args["player"]))
            events = [r for r in events if set(r["players"]) & ids]
        if args.get("status"):
            events = [r for r in events if r["status"] == args["status"]]
        if args.get("last_minutes"):
            since = now[1] - 60 * float(args["last_minutes"])
            start = (now[0], since) if since >= 0 or now[0] == 1 else (1, self.first_half + since)
            events = [r for r in events if (r["half"], r["second"]) >= start]
        elif args.get("half"):
            h = int(args["half"])
            lo, hi = float(args.get("start", 0)), float(args.get("end", 1e9))
            events = [r for r in events if r["half"] == h and lo <= r["second"] <= hi]
        lines = [self._line(r) for r in events]
        cut = 80
        head = f"{len(lines)} events" + (f" (the last {cut} shown)" if len(lines) > cut else "")
        return Evidence(head + ":\n" + ("\n".join(lines[-cut:]) or "(none)"))

    def _player(self, now: Time, name: str) -> Evidence:
        ids = self._find(name)
        if not ids:
            return Evidence(f"error: no player matches {name!r}")
        view = self.mem.at(now)
        teams = self.prematch["teams"]
        out = []
        for pid in ids:
            p = self.players[pid]
            side = p["team_side"]
            where = "on the pitch" if pid in view.on_field[side] else "on the bench" if pid in view.bench[side] \
                else "off the pitch (substituted or sent off)"
            out.append(f"{self._name(pid)}: {teams[side]} ({side}), {p['position'] or 'no position given'}, "
                       f"{'starting XI' if p['starting'] else 'substitute'}; now {where}.")
            rows = [self._line(r) for r in self._events(now) if pid in r["players"]]
            out += [f"  {row}" for row in rows] or ["  (no event names this player so far)"]
        return Evidence("\n".join(out))

    def _why(self, now: Time, entry: str) -> Evidence:
        past = self.mem.at(now).mem
        if entry not in past.entries or not isinstance(past.entries[entry], Event):
            return Evidence(f"error: no event {entry} (known by now)")
        lines = [f"{entry} [{past.status(entry)}] rests on:"]
        for _, j in past.justifications[entry]:
            state = "holds" if past.holds(j) else "rejected"
            lines.append(f"- {self._source(past, j.obs_id)} [{state}]")
        cases = past.open_cases(entry)
        if cases:
            lines.append("It is disputed in open case(s) " + ", ".join(cases) + ".")
        return Evidence("\n".join(lines))

    def _source(self, past: MatchMemory, obs_id: str) -> str:
        if obs_id.startswith("verdict:"):
            case = past.cases.get(obs_id.removeprefix("verdict:"))
            summary = (case.verdict or {}).get("summary", "") if case and isinstance(case.verdict, dict) else ""
            return f"an arbiter verdict: {summary}"
        o = self.sources.get(obs_id)
        if o is None:
            return obs_id
        c, (w0, w1) = o["candidate"], o["window"]
        return (f"the event analyzer, watching {clock(w0)}-{clock(w1)}, saw {c['type']} by {c['team']} at "
                f"{clock(c['second'])}{' in a replay' if c['replay'] else ' live'}")

    def _event_evidence(self, now: Time, entry: str) -> Evidence:
        rows = [r for r in self._events(now) if r["id"] == entry]
        if not rows:
            return Evidence(f"error: no event {entry} (known by now)")
        r = rows[0]
        (_, lo), (_, hi) = r["event"].t_interval
        start, end = (r["second"] - 6, r["second"] + 4) if r["exact"] else (lo, hi)
        ev = self._frames(now, r["half"], max(0.0, start), end)
        ev.text = self._line(r) + "\n" + ev.text
        return ev

    def _frames(self, now: Time, half: int, start: float, end: float) -> Evidence:
        """Up to max_frames sampled frames spread over [start, end] of `half`, cut at `now`,
        each tracked player labelled with a local ID and identified as far as known by now."""
        if (half, start) > now:
            return Evidence("error: that part of the match has not been played yet")
        if half == now[0]:
            end = min(end, now[1])
        perceived = self.perceived[half]
        seconds = sorted(s for s in perceived if start <= s <= end)
        if not seconds:
            return Evidence("error: no frames in that span")
        n = self.cfg["max_frames"]
        seconds = [seconds[i] for i in np.linspace(0, len(seconds) - 1, min(n, len(seconds))).round().astype(int)]
        frames, known = self._read(half, seconds), {}
        track_players = self.mem.at(now).mem.track_players()
        for s, _ in frames:
            tracks, carrier, numbers = perceived[s]
            known[s] = ([self._as_known(half, tr, now, track_players) for tr in tracks], carrier, numbers)
        block, parts, _ = labelled_frames(frames, known, self.prematch)
        return Evidence(f"{len(frames)} frames of half {half}, {clock(frames[0][0])}-{clock(frames[-1][0])}; "
                        f"player IDs are local to these frames.\n{block}",
                        [p for p in parts if isinstance(p, bytes)])

    def _read(self, half: int, seconds: list[float]) -> list[tuple[float, np.ndarray]]:
        """The video frames at `seconds` of `half` (duplicates once; unreadable ones skipped)."""
        cap = cv2.VideoCapture(str(self.match.video_path(half)))
        frames = []
        for s in dict.fromkeys(seconds):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(s * self.video.fps[half])))
            ok, image = cap.read()
            if ok:
                frames.append((s, image))
        cap.release()
        return frames

    def _as_known(self, half: int, tr: Track, now: Time, track_players: Mapping[tuple, str]) -> Track:
        """A track as known by `now`: memory's confirmed identity, else its latest vote up to now
        (a player seen in an old frame may have been identified later, but before now)."""
        key = (half, *tr.key)
        if key in track_players:
            return replace(tr, player_id=track_players[key])
        seconds, ids = self.voted.get(key, ([], []))
        i = bisect_right(seconds, now[1]) - 1 if half == now[0] else len(seconds) - 1
        if i < 0:
            return tr
        team, player = ids[i]
        return replace(tr, team=team or tr.team, player_id=player or tr.player_id)

    # ---- names ----

    def _side(self, team: str) -> str | None:
        if team in ("home", "away"):
            return team
        teams = self.prematch["teams"]
        return next((s for s in ("home", "away") if team.lower() in teams[s].lower() or teams[s].lower() in team.lower()), None)

    def _find(self, name: str) -> set[str]:
        """Players whose name contains `name` (any spelling in the squad list), or "#10 home"-style."""
        key = name.lower().lstrip("#").strip()
        return {pid for pid, p in self.players.items()
                if any(key and key in str(p[f]).lower() for f in ("long_name", "short_name", "name"))
                or key == str(p["jersey_number"])}


def minute_phrase(half: int, second: float) -> str:
    """15:04 of half 2 -> "16th minute of the 2nd half"; 45:45 -> "1st minute of 2nd-half stoppage time"."""
    minute = int(second // 60)
    stoppage = minute >= 45
    n = minute - 45 + 1 if stoppage else minute + 1
    nth = f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
    h = "1st" if half == 1 else "2nd"
    return f"{nth} minute of {h}-half stoppage time" if stoppage else f"{nth} minute of the {h} half"
