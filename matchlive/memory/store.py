"""Match memory: an append-only op log of beliefs and their justifications (DESIGN.md §6.7).

Nothing derived is stored. An entry's status comes from its justifications on every read
(DESIGN.md §5.2), and so do the rule inferences (score, lineups, cards, stats — see
matchlive.memory.view) and the player behind every event participant. A correction —
rejecting an observation, confirming a different identity — therefore reaches everything
derived from it without any dependency tracking: the next read simply sees the new state.

Reading the past replays a prefix of the log: t_known never decreases, so the ops known by
time t are exactly the first ones up to t (DESIGN.md §5.4). The log is JSONL, one op per
line, flushed as it is written; opening an existing log replays it.

With `cascade` off (DESIGN.md §7, the forward-only baseline) entry statuses still follow
their justifications, but every event, once counted by the inferences, stays counted with
the players it was counted with: a retracted yellow card keeps its derived red, a corrected
identity leaves the old player credited.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from matchlive.memory.types import Case, Event, Identity, Justification, Op, Participant, Time
from matchlive.memory.view import View


class MatchMemory:
    def __init__(self, roster: Iterable[Mapping[str, Any]], log_path: str | Path | None = None,
                 cascade: bool = True):
        # `roster` is Match.prematch `players`: pre-match facts only (side, starting flag, hash),
        # so nothing after kick-off leaks into the lineups (DESIGN.md §6.8).
        self.roster: dict[str, Mapping[str, Any]] = {str(p["hash"]): p for p in roster}
        self.cascade = cascade
        self.ops: list[Op] = []
        self.entries: dict[str, Event | Identity] = {}            # insertion order = write order
        self.justifications: dict[str, list[tuple[Time, Justification]]] = {}
        self.rejected: dict[str, Time] = {}                        # obs_id -> when it was rejected
        self.cases: dict[str, Case] = {}
        # Only used with cascade off: every event ever counted -> the players it was counted with.
        self._counted_once: dict[str, tuple[str | None, ...]] = {}
        # Memories rebuilt from the first n ops, by n. The log only grows, so a prefix never changes.
        self._past: dict[int, MatchMemory] = {}

        self._log = None
        if log_path is not None:
            log_path = Path(log_path)
            if log_path.exists():
                with open(log_path) as f:
                    for line in f:
                        d = json.loads(line)
                        self._apply(Op(d["op_id"], tuple(d["t_known"]), d["kind"], d["payload"]))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(log_path, "a")

    # ---- writes (each one is an op, DESIGN.md §5.4) ----

    def add_entry(self, t_known: Time, entry: Event | Identity, justification: Justification) -> str:
        """Write a new belief with its first justification; return its entry id."""
        # An entry without a justification would be retracted from the start, so the first
        # one comes with it.
        kind = "event" if isinstance(entry, Event) else "identity"
        entry_id = f"{kind[0].upper()}{len(self.entries) + 1}"
        self._record(t_known, "add_entry", {"entry_id": entry_id, "entry": {"kind": kind, **asdict(entry)},
                                            "justification": asdict(justification)})
        return entry_id

    def add_justification(self, t_known: Time, entry_id: str, justification: Justification) -> None:
        self._record(t_known, "add_justification", {"entry_id": entry_id,
                                                    "justification": asdict(justification)})

    def reject_observation(self, t_known: Time, obs_id: str) -> None:
        """Rule an observation out (a misread number, an overturned verdict "verdict:<case id>");
        every justification citing it stops holding."""
        self._record(t_known, "reject_observation", {"obs_id": obs_id})

    def open_case(self, t_known: Time, members: Iterable[str], constraint: str,
                  evidence: Iterable[str] = ()) -> str:
        case_id = f"C{len(self.cases) + 1}"
        self._record(t_known, "open_case", {"case_id": case_id, "members": list(members),
                                            "constraint": constraint, "evidence": list(evidence)})
        return case_id

    def add_evidence(self, t_known: Time, case_id: str, evidence: Iterable[str]) -> None:
        self._record(t_known, "add_evidence", {"case_id": case_id, "evidence": list(evidence)})

    def close_case(self, t_known: Time, case_id: str, verdict: Any = None) -> None:
        # The verdict's effects are written as ops of their own (justifications with obs_id
        # "verdict:<case id>", rejections, new entries); `verdict` is kept for audit only.
        self._record(t_known, "close_case", {"case_id": case_id, "verdict": verdict})

    def _record(self, t_known: Time, kind: str, payload: dict[str, Any]) -> None:
        t_known = (int(t_known[0]), float(t_known[1]))
        if self.ops and t_known < self.ops[-1].t_known:
            raise ValueError(f"t_known {t_known} is earlier than the last op's {self.ops[-1].t_known}")
        op = Op(len(self.ops), t_known, kind, payload)
        # Applied before it is logged, so an op that fails (e.g. an unknown entry id) never
        # reaches the log. Live writes and log replay go through the same _apply().
        self._apply(op)
        if self._log is not None:
            self._log.write(json.dumps(asdict(op)) + "\n")
            self._log.flush()

    def _apply(self, op: Op) -> None:
        p = op.payload
        if op.kind == "add_entry":
            self.entries[p["entry_id"]] = _decode_entry(p["entry"])
            self.justifications[p["entry_id"]] = [(op.t_known, Justification(**p["justification"]))]
        elif op.kind == "add_justification":
            self.justifications[p["entry_id"]].append((op.t_known, Justification(**p["justification"])))
        elif op.kind == "reject_observation":
            self.rejected.setdefault(p["obs_id"], op.t_known)
        elif op.kind == "open_case":
            self.cases[p["case_id"]] = Case(p["case_id"], tuple(p["members"]), p["constraint"],
                                            list(p["evidence"]))
        elif op.kind == "add_evidence":
            self.cases[p["case_id"]].evidence.extend(p["evidence"])
        elif op.kind == "close_case":
            case = self.cases[p["case_id"]]
            case.status, case.verdict = "closed", p["verdict"]
        else:
            raise ValueError(f"unknown op kind {op.kind!r}")
        self.ops.append(op)
        if not self.cascade:
            # Forward-only: fold whatever the inferences count now into what they have ever
            # counted. A participant's player, once known, is kept; one still unknown may be
            # filled in later (that adds knowledge, it does not revise it).
            for entry_id, (_, players) in self._confirmed_events().items():
                old = self._counted_once.get(entry_id)
                if old is not None:
                    players = tuple(o or n for o, n in zip(old, players))
                self._counted_once[entry_id] = players

    # ---- derived on read ----

    def status(self, entry_id: str) -> str:
        """"confirmed" | "pending" | "retracted", from the entry's justifications (DESIGN.md §5.2)."""
        holding = [j for _, j in self.justifications[entry_id] if j.obs_id not in self.rejected]
        if not holding:
            return "retracted"
        if any(j.sufficient for j in holding) and not self.open_cases(entry_id):
            return "confirmed"
        return "pending"

    def open_cases(self, entry_id: str) -> list[str]:
        """The open cases this entry is a member of; while there are any it is pending."""
        return [c.case_id for c in self.cases.values() if c.status == "open" and entry_id in c.members]

    def track_players(self) -> dict[tuple, str]:
        """{track_key: player_id} from the confirmed Identity entries. Two confirmed identities
        for one track only happen with the consistency check off; the later-written wins."""
        return {e.track_key: e.player_id for entry_id, e in self.entries.items()
                if isinstance(e, Identity) and self.status(entry_id) == "confirmed"}

    def event_players(self, entry_id: str, track_players: Mapping[tuple, str] | None = None
                      ) -> tuple[str | None, ...]:
        """The player behind each participant of an event (None if not yet identified)."""
        if not self.cascade and entry_id in self._counted_once:
            return self._counted_once[entry_id]
        return resolve_players(self.entries[entry_id], self.track_players() if track_players is None else track_players)

    def counted_events(self) -> dict[str, tuple[Event, tuple[str | None, ...]]]:
        """The events the rule inferences count, each with its participants' players."""
        if self.cascade:
            return self._confirmed_events()
        return {entry_id: (self.entries[entry_id], players) for entry_id, players in self._counted_once.items()}

    def _confirmed_events(self) -> dict[str, tuple[Event, tuple[str | None, ...]]]:
        track_players = self.track_players()
        return {entry_id: (e, resolve_players(e, track_players)) for entry_id, e in self.entries.items()
                if isinstance(e, Event) and self.status(entry_id) == "confirmed"}

    # ---- queries (DESIGN.md §5.5) ----

    def at(self, event_time: Time, known_by: Time | None = None) -> View:
        """The match as of `event_time`, using only what was known by `known_by` (default:
        `event_time` itself — no peeking into the future)."""
        known_by = event_time if known_by is None else known_by
        # t_known never decreases, so what was known by then is a prefix of the log.
        n = bisect_right(self.ops, known_by, key=lambda op: op.t_known)
        mem = self
        if n < len(self.ops):
            mem = self._past.get(n)
            if mem is None:
                mem = MatchMemory(self.roster.values(), cascade=self.cascade)
                for op in self.ops[:n]:
                    mem._apply(op)
                self._past[n] = mem
        return View(mem, event_time)


def resolve_players(event: Event, track_players: Mapping[tuple, str]) -> tuple[str | None, ...]:
    # A participant named directly wins; otherwise its track's confirmed identity, if any.
    return tuple(p.player_id or track_players.get(p.track_key) for p in event.participants)


def _decode_entry(d: Mapping[str, Any]) -> Event | Identity:
    # Ops carry entries as plain dicts (as written to the JSONL log), so JSON arrays come
    # back as lists; keys and times are tuples again here. Every write passes through, so
    # this is also where the track-key contract is checked.
    if d["kind"] == "identity":
        return Identity(_track_key(d["track_key"]), d["player_id"])
    return Event(
        d["type"], d["team"], (tuple(d["t_interval"][0]), tuple(d["t_interval"][1])),
        tuple(Participant(p["role"], None if p["track_key"] is None else _track_key(p["track_key"]),
                          p["player_id"]) for p in d["participants"]))


def _track_key(key) -> tuple:
    # Outside perception a track is (half, shot_id, track_id) (DESIGN.md §4); perception's
    # own (shot_id, track_id) would collide across halves and never match an identity.
    if len(key) != 3:
        raise ValueError(f"track key {key!r} is not (half, shot_id, track_id)")
    return tuple(key)
