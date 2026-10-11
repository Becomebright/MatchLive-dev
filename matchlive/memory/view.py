"""The match as of one moment: rule inferences computed from the counted events (DESIGN.md §5.2).

Nothing here is stored between reads. A View is built from the memory's current entries
and justifications, so every inference — the score, who is on the pitch, two yellows
making a red, per-team and per-player counts — follows the latest evidence by construction.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from matchlive.memory.types import Event, Time

if TYPE_CHECKING:
    from matchlive.memory.store import MatchMemory


@dataclass(frozen=True)
class Card:
    entry_id: str                   # the card event; for a derived red, the second yellow
    kind: str                       # "yellow_card" | "red_card"
    team: str | None
    player_id: str | None           # None: the booked player is not identified yet
    derived: bool                   # a red inferred from a second yellow, not observed
    time: Time


@dataclass(frozen=True)
class EventRecord:
    entry_id: str
    event: Event
    status: str                     # "confirmed" | "pending" | "retracted"
    players: tuple[str | None, ...]  # the player behind each participant, None if unknown


class View:
    def __init__(self, mem: MatchMemory, event_time: Time):
        self.mem = mem
        self.time = event_time
        roster = mem.roster
        # An event counts once it has certainly happened, i.e. by its interval's end; the
        # same moment orders events (a stable sort keeps write order on ties).
        happened = sorted(((entry_id, e, players) for entry_id, (e, players) in mem.counted_events().items()
                           if e.t_interval[1] <= event_time),
                          key=lambda x: x[1].t_interval[1])

        self.score = {"home": 0, "away": 0}
        self.team_stats = {"home": Counter(), "away": Counter()}    # counted events by type
        self.player_stats: dict[str, Counter] = defaultdict(Counter)  # events where the player is the main actor
        self.cards: list[Card] = []
        # Lineups start from the pre-match roster and change only through counted
        # substitutions and sendings-off.
        self.on_field = {"home": [], "away": []}
        self.bench = {"home": [], "away": []}
        for pid, p in roster.items():
            (self.on_field if p["starting"] else self.bench)[p["team_side"]].append(pid)
        yellows: Counter = Counter()

        for entry_id, e, players in happened:
            when = e.t_interval[1]
            if e.team in self.score:
                self.team_stats[e.team][e.type] += 1
                if e.type == "goal":
                    self.score[e.team] += 1
            by_role = {part.role: pid for part, pid in zip(e.participants, players)}
            if by_role.get("player"):
                self.player_stats[by_role["player"]][e.type] += 1

            if e.type in ("yellow_card", "red_card"):
                booked = by_role.get("player")
                self.cards.append(Card(entry_id, e.type, e.team, booked, False, when))
                # A card whose player is not identified counts for the team only: it can
                # neither make a second yellow nor send anyone off.
                if booked is None:
                    continue
                if e.type == "yellow_card":
                    yellows[booked] += 1
                    if yellows[booked] == 2:
                        self.cards.append(Card(entry_id, "red_card", e.team, booked, True, when))
                        self._send_off(booked)
                else:
                    self._send_off(booked)
            elif e.type == "substitution":
                out, sub = by_role.get("out"), by_role.get("in")
                # Both players must be known to change the lineups.
                if out and sub:
                    for side in ("home", "away"):
                        if out in self.on_field[side]:
                            self.on_field[side].remove(out)
                        if sub in self.bench[side]:
                            self.bench[side].remove(sub)
                    self.on_field[roster[sub]["team_side"]].append(sub)

    def _send_off(self, player_id: str) -> None:
        # A player sent off leaves the pitch for good; a sent-off substitute can no longer come on.
        for side in ("home", "away"):
            for lineup in (self.on_field[side], self.bench[side]):
                if player_id in lineup:
                    lineup.remove(player_id)

    def events(self, type: str | None = None, team: str | None = None, player: str | None = None,
               since: Time | None = None, until: Time | None = None, status: str | None = None
               ) -> list[EventRecord]:
        """Event entries of any status that have happened by this view's time, in time order."""
        track_players = self.mem.track_players()
        out = []
        for entry_id, e in self.mem.entries.items():
            if not isinstance(e, Event):
                continue
            end = e.t_interval[1]
            if end > self.time or (since is not None and end < since) or (until is not None and end > until):
                continue
            if (type is not None and e.type != type) or (team is not None and e.team != team):
                continue
            s = self.mem.status(entry_id)
            if status is not None and s != status:
                continue
            players = self.mem.event_players(entry_id, track_players)
            if player is not None and player not in players:
                continue
            out.append(EventRecord(entry_id, e, s, players))
        return sorted(out, key=lambda r: r.event.t_interval[1])

    def why(self, entry_id: str) -> dict:
        """What this belief rests on: its justifications (and which were rejected, when), and
        the open cases it is caught up in."""
        mem = self.mem
        return {
            "entry": mem.entries[entry_id],
            "status": mem.status(entry_id),
            "justifications": [{"obs_id": j.obs_id, "kind": j.kind, "sufficient": j.sufficient,
                                "added": t, "rejected": mem.rejected_at(j)}
                               for t, j in mem.justifications[entry_id]],
            "open_cases": mem.open_cases(entry_id),
        }
