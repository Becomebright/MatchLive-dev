"""Live commentary (DESIGN.md §6.9): what the policy puts on air follows each entry's status —
a confirmed event names its player, a pending one only its team, and when something already said
stops holding (retracted, or a different player), the next line corrects it with the reason.

Each test writes a few ops to a small synthetic memory and asks the policy what to say; no model
is called (the generator only phrases the material the policy chose).
"""

from matchlive.app.commentary import CommentaryPolicy, Item
from matchlive.common.config import load_config
from matchlive.memory.store import MatchMemory
from matchlive.memory.types import Event, Identity, Justification, Participant

PLAYERS = [{"hash": h, "team_side": side, "starting": i <= 3, "jersey_number": str(i), "position": "forward",
            "long_name": f"{side.capitalize()} Player{i}", "short_name": f"Player{i}", "name": f"{side} {i}"}
           for side, prefix in (("home", "h"), ("away", "a")) for i in range(1, 5) for h in [f"{prefix}{i}"]]
PREMATCH = {"teams": {"home": "Homeside", "away": "Awayside"},
            "jersey_colors": {"home": ["blue", "#0000FF"], "away": ["red", "#FF0000"]},
            "sides": {1: {"home": "left", "away": "right"}, 2: {"home": "right", "away": "left"}},
            "players": PLAYERS}
TRACK = (1, 4, 7)


def T(second):
    return (1, second)


def seen(obs_id, sufficient=True):
    return Justification(obs_id, "observation", sufficient)


def goal(t, **participants):
    parts = tuple(Participant(role, track_key=v) if isinstance(v, tuple) else Participant(role, player_id=v)
                  for role, v in participants.items())
    return Event("goal", "home", ((1, t - 10), (1, t + 10)), parts)


def policy():
    p = CommentaryPolicy(load_config()["commentary"], PREMATCH, sources={})
    # The half is already opened and a line was just said, so neither a background line nor a
    # recap gets in the way.
    p.said(Item("background", [], ""), "Welcome.", T(0))
    return p


def air(p, mem, t):
    """Ask the policy at match time t and put its item on air; return the item."""
    item = p.decide(mem.at(T(t)))
    if item is not None:
        p.said(item, f"line about {item.entry}", T(t))
    return item


def test_pending_event_is_hedged_and_names_nobody():
    mem = MatchMemory(PLAYERS)
    mem.add_entry(T(95), Identity(TRACK, "h1"), seen("vote:h1"))
    eid = mem.add_entry(T(100), goal(80, player=TRACK), seen("live#1", sufficient=False))
    item = policy().decide(mem.at(T(100)))
    assert (item.kind, item.entry, item.status) == ("news", eid, "pending")
    assert item.named == {}
    assert not any("Player1" in f for f in item.facts)
    assert "hedge" in item.rule and "Homeside" in item.facts[0]


def test_confirmed_event_names_its_player():
    mem = MatchMemory(PLAYERS)
    mem.add_entry(T(95), Identity(TRACK, "h1"), seen("vote:h1"))
    eid = mem.add_entry(T(100), goal(80, player=TRACK), seen("live#1"))
    item = policy().decide(mem.at(T(100)))
    assert (item.kind, item.entry, item.status) == ("news", eid, "confirmed")
    assert item.named == {"player": "h1"}
    assert any("Home Player1" in f for f in item.facts)


def test_retracted_fact_is_corrected_with_the_verdict():
    mem = MatchMemory(PLAYERS)
    first = mem.add_entry(T(100), goal(80, player="h1"), seen("live#1"))
    p = policy()
    assert air(p, mem, 100).entry == first
    second = mem.add_entry(T(150), goal(130, player="h1"), seen("live#2"))
    assert air(p, mem, 150).entry == second           # aired as fact: a second goal
    # The arbiter finds the second goal was the replay of the first.
    case = mem.open_case(T(160), [first, second], "kickoff_after_goal")
    mem.reject_observation(T(200), "live#2")
    mem.close_case(T(200), case, {"summary": "The second goal is a replay of the first.",
                                  "actions": [{"op": "reject", "observation": "live#2"}]})
    item = p.decide(mem.at(T(200)))
    assert (item.kind, item.entry, item.status) == ("correction", second, "retracted")
    assert any("replay of the first" in f for f in item.facts)
    assert mem.at(T(200)).score == {"home": 1, "away": 0}
    # Once corrected, it is not corrected again.
    p.said(item, "Correction: only one goal.", T(200))
    assert p.decide(mem.at(T(202))) is None


def test_player_named_on_air_then_changed_is_corrected():
    mem = MatchMemory(PLAYERS)
    mem.add_entry(T(95), Identity(TRACK, "h1"), seen("vote:h1"))
    eid = mem.add_entry(T(100), goal(80, player=TRACK), seen("live#1"))
    p = policy()
    assert air(p, mem, 100).named == {"player": "h1"}
    # A closer look reads the other number on that track: the scorer was h2.
    mem.reject_observation(T(300), "vote:h1")
    mem.add_entry(T(300), Identity(TRACK, "h2"), Justification("verdict:C9", "verdict", True))
    item = p.decide(mem.at(T(300)))
    assert (item.kind, item.entry) == ("correction", eid)
    assert item.named == {"player": "h2"}
    assert any("actually #2 Home Player2" in f for f in item.facts)


def test_pending_goal_is_aired_again_once_confirmed():
    mem = MatchMemory(PLAYERS)
    eid = mem.add_entry(T(100), goal(80, player="h1"), seen("live#1", sufficient=False))
    p = policy()
    assert air(p, mem, 100).named == {}
    assert p.decide(mem.at(T(110))) is None            # nothing new
    mem.add_justification(T(140), eid, seen("replay#1"))
    item = p.decide(mem.at(T(140)))
    assert (item.kind, item.entry, item.named) == ("confirmation", eid, {"player": "h1"})


def test_hedged_minor_event_retracted_is_not_worth_a_correction():
    mem = MatchMemory(PLAYERS)
    shot = mem.add_entry(T(100), Event("shot", "away", ((1, 90), (1, 100))), seen("live#1", sufficient=False))
    p = policy()
    assert air(p, mem, 100).entry == shot
    mem.reject_observation(T(150), "live#1")
    assert p.decide(mem.at(T(150))) is None
