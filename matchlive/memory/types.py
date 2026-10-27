"""Match-memory records: beliefs (entries), their justifications, conflict cases and log ops.

DESIGN.md §5.2-5.4. An entry's content is written once and never changes; everything that
does change over a match is expressed as justifications being added or their observations
being rejected. So the content types below carry no id and no justification list — the
store (matchlive.memory.store) keeps those next to the content.

Times are match times (half, second) (DESIGN.md §4); plain tuples compare in match order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Time = tuple[int, float]                        # (half, second of that half's video)

# v2's ten event classes plus injury (play stopped for an injured player); the analyzer asks
# for these and verdicts may only add these.
EVENT_TYPES = ("shot", "goal", "yellow_card", "red_card", "foul", "substitution",
               "penalty", "free_kick", "offside", "corner", "injury")


@dataclass(frozen=True)
class Participant:
    # Roles the rule inferences read (DESIGN.md §5.2): "player" is an event's main actor
    # (the scorer, the booked player, ...); a substitution has "out" and "in". Other roles
    # (e.g. "fouled") may appear but take no part in inferences.
    role: str
    # Usually a track, keyed (half, shot_id, track_id) (DESIGN.md §4), whose player is
    # looked up through the Identity entries on every read, so correcting an identity
    # re-attributes the event; the arbiter may name the player directly instead (e.g. an
    # inferred substitution).
    track_key: tuple | None = None
    player_id: str | None = None                # roster hash
    # Tracks other sightings of the same event saw in this role (different windows track the
    # same player under different keys): the role's player is the first of track_key,
    # alternates that has a confirmed identity. Filled in by memory from the justifications
    # that brought them (MatchMemory._rebuild), never written with the entry.
    alternates: tuple[tuple, ...] = ()


@dataclass(frozen=True)
class Event:
    type: str                                   # one of EVENT_TYPES
    # "home" | "away" | None. For a goal it is the side credited with it, own goals
    # included, so the score is a plain count per team.
    team: str | None
    # When it happened, as far as the proposer can tell: the sighting ± events.live_margin for a
    # live candidate (or a verdict's event), [window start - replay_lookback, window end] for a
    # replay candidate. Inferences time an event by the interval's end, the moment it has
    # certainly happened.
    t_interval: tuple[Time, Time]
    participants: tuple[Participant, ...] = ()


@dataclass(frozen=True)
class Identity:
    # "This track is this player."
    track_key: tuple
    player_id: str


@dataclass(frozen=True)
class Justification:
    # One observation supporting an entry: an event-analysis candidate, a jersey read, a
    # replay analysis — or a verdict, whose id is "verdict:<case id>" so a later verdict
    # overturns it by rejecting that id. The observation itself lives in the API or
    # perception cache; only its id is kept here.
    obs_id: str
    kind: str                                   # "observation" | "recheck" | "verdict"
    sufficient: bool                            # enough on its own to confirm (DESIGN.md §6.4)
    # The participants a matching observation names (DESIGN.md §6.4 step 1). While it holds,
    # they join the entry: a role the entry lacks is added, a role it has gets the track as an
    # alternate. Rejecting the observation takes them out again.
    participants: tuple[Participant, ...] = ()
    # A re-check (kind "recheck", DESIGN.md §6.4) looks at the same footage again, so it is no
    # independent source: it holds only while the observation it re-checks (`of`) does too.
    of: str | None = None


@dataclass
class Case:
    # A set of beliefs that cannot all hold (an ATMS nogood). While the case is open, every
    # entry among its members is pending (DESIGN.md §5.2).
    case_id: str
    members: tuple[str, ...]                    # entry ids and/or observation ids
    constraint: str                             # id of the violated constraint (DESIGN.md §6.4)
    evidence: list[str] = field(default_factory=list)
    status: str = "open"                        # "open" | "closed"
    verdict: Any = None


@dataclass(frozen=True)
class Op:
    # The only way memory is written (DESIGN.md §5.4). t_known is the match time at which
    # the op took effect; it never decreases along the log.
    op_id: int
    t_known: Time
    kind: str       # add_entry | add_justification | reject_observation | open_case | add_evidence | close_case
    payload: dict[str, Any]
