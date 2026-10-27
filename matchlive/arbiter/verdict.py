"""Apply an arbiter verdict to memory, behind the guardrails of DESIGN.md §6.6.

A verdict is a set of writes, each citing the evidence the arbiter's tools returned:
  support   {"entry": id}                  the entry happened / is right: a sufficient
                                           justification "verdict:<case id>"
  reject    {"observation": id}            an observation is wrong (a replay taken for a new
                                           event, a misread number, an earlier verdict)
  add_event {"type", "team", "half", "second", "participants": [{"role", "track" | "player"}]}
                                           an event nobody proposed (a foul before a free kick; a
                                           substitution inferred from a substitute's number read
                                           on the pitch, whose players are named directly);
                                           if memory already holds one like it outside this case
                                           (same type and half, compatible team, overlapping
                                           time) the write supports that entry instead of adding
                                           a second one — two cases finding the same missed
                                           event. The case's own members are left out: an event
                                           added next to them is the arbiter splitting two events
                                           that had been merged. So is an entry whose every
                                           observation the verdict rejects: that is a correction.
                                           An added event whose participants are all named (a
                                           player, or a track the verdict identifies) replaces the
                                           entry it matches, whose own observations are rejected:
                                           it says who took part, which that entry may not (an
                                           entry an earlier verdict holds up is supported instead).
  An identity written or supported is the only one standing on its track: the observations of
  every other claim on that track are rejected.
  An observation may be rejected if memory cites it or it belongs to the case (its members
  and evidence, e.g. the number reads of an off_roster_read case); an entry id is no
  observation.
  identity  {"track": [half, shot, track], "player": hash}
                                           who a track is
An empty list dismisses the case. Every write rests on the observation "verdict:<case id>",
so a later verdict overturns this one by rejecting that id (DESIGN.md §5.2).

The verdict is all or nothing: if any write fails a check it is void, nothing is written and
the case stays open (its members stay pending). The checks:
  - every write cites evidence the tools actually returned for this case;
  - it names things that exist (entries, observations, tracks, roster players);
  - it does not look into the future (an added event happened by the time the case opened);
  - it breaks no hard constraint (one player on two tracks seen in the same frame).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping

from matchlive.events.consistency import live_events, matching
from matchlive.memory.store import MatchMemory
from matchlive.memory.types import EVENT_TYPES, Event, Identity, Justification, Participant, Time


def apply_verdict(mem: MatchMemory, t_known: Time, case_id: str, verdict: Mapping[str, Any],
                  evidence_ids: Iterable[str], t_bound: Time, live_margin: Mapping[str, float],
                  track_exists: Callable[[tuple], bool], cooccur: Callable[[tuple, tuple], bool]) -> str | None:
    """Check `verdict` and, if it passes, write it at `t_known` and close the case. Returns
    None when applied, else why it was void. `t_bound` is what was known when the case
    opened; `live_margin` is events.live_margin (an added event's interval, per type, as for a
    live sighting); `track_exists` / `cooccur` answer from the perception cache (a track seen
    up to t_bound; two tracks sharing a frame)."""
    evidence_ids = set(evidence_ids)
    actions = verdict.get("actions")
    if not isinstance(actions, list):
        return "no action list"
    # Observations a verdict may reject: those memory cites, and the case's own (e.g. the
    # off-pitch number reads an off_roster_read case is made of).
    case = mem.cases[case_id]
    cited = {j.obs_id for js in mem.justifications.values() for _, j in js} \
        | {m for m in (*case.members, *case.evidence) if m not in mem.entries}
    confirmed_tracks = defaultdict(list)
    for eid, e in mem.entries.items():
        if isinstance(e, Identity) and mem.status(eid) == "confirmed":
            confirmed_tracks[e.player_id].append(e.track_key)
    # Step-1 matching candidates for added events: everything standing outside this case, except
    # entries this verdict takes down (every observation behind them rejected) — rejecting an
    # event and adding it again corrects what it says (e.g. names both players of a substitution).
    rejects = {a.get("observation") for a in actions if isinstance(a, dict) and a.get("op") == "reject"}
    outside = [(eid, e) for eid, e in live_events(mem) if eid not in case.members
               and not all(j.obs_id in rejects for _, j in mem.justifications[eid])]
    # Tracks this verdict identifies: an added event whose participants are all named (a player,
    # or a track identified here) says who took part, which a matching entry may not.
    named_tracks = {tuple(int(x) for x in a["track"]) for a in actions if isinstance(a, dict)
                    and a.get("op") == "identity" and isinstance(a.get("track"), list)
                    and all(isinstance(x, (int, float)) or str(x).isdigit() for x in a["track"])}

    writes = []
    for a in actions:
        op = a.get("op") if isinstance(a, dict) else None
        ev = a.get("evidence") if isinstance(a, dict) else None
        if not isinstance(ev, list) or not ev or not set(ev) <= evidence_ids:
            return f"{op}: cites no evidence returned for this case"
        if op == "support":
            if a.get("entry") not in mem.entries:
                return f"support: unknown entry {a.get('entry')!r}"
            writes.append(("support", (a["entry"], ())))
            e = mem.entries[a["entry"]]
            if isinstance(e, Identity):
                writes += others_on_track(mem, e.track_key, e.player_id)
        elif op == "reject":
            if a.get("observation") not in cited:
                return f"reject: unknown observation {a.get('observation')!r}"
            writes.append(("reject", a["observation"]))
        elif op == "add_event":
            if a.get("type") not in EVENT_TYPES or a.get("team") not in ("home", "away", None):
                return "add_event: bad type or team"
            try:
                when = (int(a["half"]), float(a["second"]))
                # A participant is a track, or a roster player named directly (e.g. both sides
                # of a substitution nobody saw).
                parts = tuple(Participant(str(p["role"]), track_key=tuple(int(x) for x in p["track"]))
                              if "track" in p else Participant(str(p["role"]), player_id=str(p["player"]))
                              for p in a.get("participants") or [])
            except (KeyError, TypeError, ValueError):
                return "add_event: bad time or participants"
            if when > t_bound or when[1] < 0:
                return "add_event: not a time known when the case opened"
            if any(p.track_key is not None and not track_exists(p.track_key) for p in parts):
                return "add_event: unknown track"
            if any(p.player_id is not None and p.player_id not in mem.roster for p in parts):
                return "add_event: unknown player"
            margin = live_margin.get(a["type"], live_margin["default"])
            event = Event(a["type"], a.get("team"),
                          ((when[0], max(0.0, when[1] - margin)), (when[0], when[1] + margin)), parts)
            # Matched like any other sighting (DESIGN.md §6.4 step 1), so two verdicts that find
            # the same missed event do not add it twice. If the verdict names everyone who took
            # part, it replaces the matching entry instead: that entry's observations are
            # rejected, since the verdict's account of the same event is the fuller one (e.g. a
            # substitution seen with neither player identified, which cannot change the lineup).
            # An entry an earlier verdict holds up is not replaced (the verdict's own writes stand).
            # The observations rejected are the entry's own sightings: what stands on them (a
            # paired partner's sighting, a re-check) falls with them, and the partner's own
            # sighting stays — a foul is not retracted with the free kick it paired with.
            same = matching(outside, event)
            if same and parts and all(p.player_id or p.track_key in named_tracks for p in parts) \
                    and not any(j.kind == "verdict" and mem.holds(j) for _, j in mem.justifications[same[0]]):
                writes += [("reject", j.obs_id) for _, j in mem.justifications[same[0]] if mem.holds(j) and j.of is None]
                writes.append(("add_event", event))
                outside = [(eid, e) for eid, e in outside if eid != same[0]]
            else:
                # Supporting an entry, the verdict's participants join it like any agreeing claim's.
                writes.append(("support", (same[0], parts)) if same else ("add_event", event))
        elif op == "identity":
            try:
                key, player = tuple(int(x) for x in a["track"]), str(a["player"])
            except (KeyError, TypeError, ValueError):
                return "identity: bad track or player"
            if player not in mem.roster or len(key) != 3 or not track_exists(key):
                return "identity: unknown player or track"
            # Hard constraint: one player cannot be two tracks in the same frame.
            if any(other != key and cooccur(key, other) for other in confirmed_tracks.get(player, [])):
                return "identity: player already on another track in the same frame"
            writes.append(("identity", Identity(key, player)))
            writes += others_on_track(mem, key, player)
        else:
            return f"unknown op {op!r}"

    obs = f"verdict:{case_id}"
    for op, x in writes:
        if op == "support":
            entry_id, parts = x
            mem.add_justification(t_known, entry_id, Justification(obs, "verdict", True, parts))
        elif op == "reject":
            mem.reject_observation(t_known, x)
        else:
            mem.add_entry(t_known, x, Justification(obs, "verdict", True))
    mem.close_case(t_known, case_id, {"summary": verdict.get("summary", ""), "actions": actions})
    return None


def others_on_track(mem: MatchMemory, key: tuple, player: str) -> list[tuple[str, str]]:
    """The rejections that leave `player` the only identity standing on track `key`: one track is
    one player, so the observations behind every other claim on it are wrong."""
    return [("reject", j.obs_id) for eid, e in mem.entries.items()
            if isinstance(e, Identity) and e.track_key == key and e.player_id != player
            for _, j in mem.justifications[eid] if mem.holds(j)]


def reject_newest(mem: MatchMemory, t_known: Time, case_id: str) -> None:
    """The loop.arbiter-off baseline (DESIGN.md §7): no investigation, the newest belief in
    the conflict loses — its own observations are rejected, and the case is closed."""
    case = mem.cases[case_id]
    # A member that is an observation (a new sighting disagreeing with an entry) is the
    # newest belief; otherwise the entry written last.
    newest = next((m for m in case.members if m not in mem.entries), None) \
        or max(case.members, key=list(mem.entries).index)
    for obs in ([newest] if newest not in mem.entries else [j.obs_id for _, j in mem.justifications[newest]]):
        mem.reject_observation(t_known, obs)
    mem.close_case(t_known, case_id, "baseline: newest belief rejected")

