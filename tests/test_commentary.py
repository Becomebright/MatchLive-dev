"""Live commentary (DESIGN.md §6.9): what the policy puts on air follows each entry's status —
a confirmed event names its player, a pending one only its team (and waits a little for its
confirmation first), a disputed one waits for the verdict — and when something already said stops
holding (retracted, or a different player), the next line corrects it with the reason. Nothing
after the moment on air is said; with nothing else to say, the line describes the play.

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
CONFIG = load_config()["commentary"]
HOLD = CONFIG["hold"]


def T(second):
    return (1, second)


def seen(obs_id, sufficient=True):
    return Justification(obs_id, "observation", sufficient)


def goal(t, **participants):
    parts = tuple(Participant(role, track_key=v) if isinstance(v, tuple) else Participant(role, player_id=v)
                  for role, v in participants.items())
    return Event("goal", "home", ((1, t - 10), (1, t + 10)), parts)


def policy():
    p = CommentaryPolicy(CONFIG, PREMATCH, sources={})
    # The half is already opened, and a context line was just said.
    p.said(Item("background", [], ""), "Welcome.", T(0))
    return p


def ask(p, mem, t, on_air=None):
    """What the policy says next with memory known at t, for air at `on_air` (default: t)."""
    return p.decide(mem.at(T(t)), T(on_air if on_air is not None else t))


def air(p, mem, t):
    item = ask(p, mem, t)
    p.said(item, f"line about {item.entry}", T(t))
    return item


def is_news(item, entry):
    return item.kind == "news" and item.entry == entry


def test_confirmed_event_names_its_player():
    mem = MatchMemory(PLAYERS)
    mem.add_entry(T(95), Identity(TRACK, "h1"), seen("vote:h1"))
    eid = mem.add_entry(T(100), goal(80, player=TRACK), seen("live#1"))
    item = ask(policy(), mem, 100)
    assert is_news(item, eid) and item.status == "confirmed"
    assert item.named == {"player": "h1"}
    assert any("Home Player1" in f for f in item.facts)


def test_pending_goal_waits_for_confirmation_then_is_hedged_and_names_nobody():
    mem = MatchMemory(PLAYERS)
    mem.add_entry(T(95), Identity(TRACK, "h1"), seen("vote:h1"))
    eid = mem.add_entry(T(100), goal(80, player=TRACK), seen("live#1", sufficient=False))
    p = policy()
    assert not is_news(ask(p, mem, 100 + HOLD - 1), eid)        # still waiting for a second sighting
    item = ask(p, mem, 100 + HOLD)
    assert is_news(item, eid) and item.status == "pending"
    assert item.named == {} and not any("Player1" in f for f in item.facts)
    assert "hedge" in item.rule and "Homeside" in item.facts[0]


def test_disputed_event_waits_for_the_verdict():
    mem = MatchMemory(PLAYERS)
    first = mem.add_entry(T(160), goal(150, player="h1"), seen("live#1"))
    second = mem.add_entry(T(180), goal(175, player="h1"), seen("live#2"))
    case = mem.open_case(T(180), [first, second], "kickoff_after_goal")
    p = policy()
    # Neither goal is said while it is disputed (one of them is most likely a replay).
    assert not ask(p, mem, 200).kind == "news"
    # The verdict: the second one was the replay of the first.
    mem.reject_observation(T(210), "live#2")
    mem.add_justification(T(210), first, Justification(f"verdict:{case}", "verdict", True))
    mem.close_case(T(210), case, {"summary": "A replay.", "actions": []})
    assert is_news(ask(p, mem, 210), first)


def test_pending_minor_event_is_said_only_once_confirmed():
    mem = MatchMemory(PLAYERS)
    shot = mem.add_entry(T(100), Event("shot", "away", ((1, 90), (1, 100))), seen("live#1", sufficient=False))
    p = policy()
    assert not is_news(ask(p, mem, 100 + 2 * HOLD), shot)        # half of these are false
    mem.add_justification(T(120), shot, seen("replay#1"))
    assert is_news(ask(p, mem, 120), shot)
    # A minor event confirmed long after the viewer saw it is stale: it is not news any more.
    late = mem.add_entry(T(130), Event("corner", "home", ((1, 60), (1, 70))), seen("live#2"))
    assert not is_news(ask(p, mem, 130), late)


def test_nothing_after_the_moment_on_air_is_said():
    mem = MatchMemory(PLAYERS)
    eid = mem.add_entry(T(155), goal(150, player="h1"), seen("live#1"))
    p = policy()
    # Memory knows of the goal at 160, but the viewer is at 100: no news, and the score is still 0-0.
    item = ask(p, mem, 160, on_air=100)
    assert not is_news(item, eid)
    assert item.kind == "recap" and any("Homeside 0, Awayside 0" in f for f in item.facts)
    # Once the viewer has seen it, it is news.
    assert is_news(ask(p, mem, 160, on_air=152), eid)


def test_with_nothing_else_to_say_it_describes_the_play():
    p = policy()
    mem = MatchMemory(PLAYERS)
    p.said(Item("recap", [], ""), "Goalless so far.", T(100))       # a context line was just said,
    p.said(Item("analysis", [], ""), "They are sitting deep.", T(105))   # and the co-commentator spoke
    item = ask(p, mem, 110)
    assert item.kind == "play"
    assert item.t_event == T(110 - CONFIG["play_span"])            # the pictures it is written over


def test_after_a_goal_is_called_the_co_commentator_gives_his_view():
    mem = MatchMemory(PLAYERS)
    g = mem.add_entry(T(170), goal(160, player="h1"), seen("live#1"))
    p = policy()
    p.said(Item("analysis", [], ""), "They are sitting deep.", T(165))
    assert is_news(air(p, mem, 170), g)
    item = ask(p, mem, 176)
    assert item.kind == "analysis"
    assert item.t_event == T(176 - CONFIG["analysis_span"])        # the replays of it


def test_retracted_fact_is_corrected_with_the_verdict():
    mem = MatchMemory(PLAYERS)
    first = mem.add_entry(T(100), goal(80, player="h1"), seen("live#1"))
    p = policy()
    assert air(p, mem, 100).entry == first
    second = mem.add_entry(T(150), goal(140, player="h1"), seen("live#2"))
    assert air(p, mem, 150).entry == second           # aired as fact: a second goal
    # The arbiter finds the second goal was the replay of the first.
    case = mem.open_case(T(160), [first, second], "kickoff_after_goal")
    mem.reject_observation(T(200), "live#2")
    mem.close_case(T(200), case, {"summary": "E6 at 02:10.5 (track 1/4/7) is a replay of the first goal, E5.",
                                  "actions": [{"op": "reject", "observation": "live#2"}]})
    item = ask(p, mem, 200)
    assert (item.kind, item.entry, item.status) == ("correction", second, "retracted")
    reason = next(f for f in item.facts if f.startswith("What turned out"))
    assert "replay of the first goal" in reason
    assert not any(code in reason for code in ("E6", "E5", "02:10", "track"))   # no system codes on air
    assert mem.at(T(200)).score == {"home": 1, "away": 0}
    # Once corrected, it is not corrected again.
    p.said(item, "No, only one goal.", T(200))
    assert ask(p, mem, 202).kind != "correction"


def test_player_named_on_air_then_changed_is_corrected():
    mem = MatchMemory(PLAYERS)
    mem.add_entry(T(95), Identity(TRACK, "h1"), seen("vote:h1"))
    eid = mem.add_entry(T(100), goal(80, player=TRACK), seen("live#1"))
    p = policy()
    assert air(p, mem, 100).named == {"player": "h1"}
    # A closer look reads the other number on that track: the scorer was h2.
    mem.reject_observation(T(300), "vote:h1")
    mem.add_entry(T(300), Identity(TRACK, "h2"), Justification("verdict:C9", "verdict", True))
    item = ask(p, mem, 300)
    assert (item.kind, item.entry) == ("correction", eid)
    assert item.named == {"player": "h2"}
    assert any("actually Player2" in f for f in item.facts)


def test_pending_goal_is_aired_again_once_confirmed():
    mem = MatchMemory(PLAYERS)
    eid = mem.add_entry(T(100), goal(80, player="h1"), seen("live#1", sufficient=False))
    p = policy()
    assert air(p, mem, 100 + HOLD).named == {}
    assert ask(p, mem, 100 + HOLD + 2).entry != eid                 # nothing new about it
    mem.add_justification(T(140), eid, seen("replay#1"))
    item = ask(p, mem, 140)
    assert (item.kind, item.entry, item.named) == ("confirmation", eid, {"player": "h1"})


def test_one_incident_is_aired_once_but_a_card_always_is():
    mem = MatchMemory(PLAYERS)
    foul = mem.add_entry(T(100), Event("foul", "away", ((1, 90), (1, 100))), seen("live#1"))
    p = policy()
    assert air(p, mem, 100).entry == foul
    # The free kick after it, and the same foul reported again, only repeat what was said.
    kick = mem.add_entry(T(130), Event("free_kick", "home", ((1, 115), (1, 125))), seen("live#2"))
    again = mem.add_entry(T(131), Event("foul", "away", ((1, 95), (1, 105))), seen("live#3"))
    assert ask(p, mem, 131).entry not in (kick, again)
    # A card for the foul is news of its own.
    card = mem.add_entry(T(140), Event("yellow_card", "away", ((1, 100), (1, 110))), seen("live#4"))
    assert is_news(ask(p, mem, 140), card)


def test_a_line_that_missed_its_moment_is_not_picked_again():
    mem = MatchMemory(PLAYERS)
    eid = mem.add_entry(T(100), goal(80, player="h1"), seen("live#1", sufficient=False))
    p = policy()
    air(p, mem, 100 + HOLD)                                        # aired hedged
    mem.add_justification(T(140), eid, seen("replay#1"))
    item = ask(p, mem, 140)
    assert (item.kind, item.entry) == ("confirmation", eid)
    p.dropped(item)                                                # not ready in time for its moment
    assert ask(p, mem, 141).entry != eid


def test_minor_event_before_a_goal_on_air_is_overtaken_by_it():
    mem = MatchMemory(PLAYERS)
    g = mem.add_entry(T(170), goal(160, player="h1"), seen("live#1"))
    p = policy()
    assert is_news(air(p, mem, 170), g)
    # The free kick the goal came from is confirmed only now: once the goal is called, it is old news.
    kick = mem.add_entry(T(175), Event("free_kick", "home", ((1, 140), (1, 160))), seen("live#2"))
    mem.add_justification(T(176), kick, seen("replay#2"))
    assert not is_news(ask(p, mem, 176), kick)


def test_a_context_line_says_something_new():
    mem = MatchMemory(PLAYERS)
    p = policy()
    every = CONFIG["context_every"]
    item = air(p, mem, every)
    assert item.kind == "recap" and any("Homeside 0, Awayside 0" in f for f in item.facts)
    # Nothing has changed: the score is not said again.
    assert ask(p, mem, 2 * every).kind != "recap"
    # A goal changes it: once the goal and the co-commentator's view are on air, the new score is.
    g = mem.add_entry(T(2 * every + 10), goal(2 * every, player="h1"), seen("live#1"))
    assert is_news(air(p, mem, 2 * every + 10), g)
    assert air(p, mem, 2 * every + 20).kind == "analysis"
    item = ask(p, mem, 3 * every + 30)
    assert item.kind == "recap" and any("Homeside 1, Awayside 0" in f for f in item.facts)


def test_pending_goal_seen_only_in_a_replay_waits_for_its_confirmation():
    mem = MatchMemory(PLAYERS)
    # A goal sighting in a replay, with no live one: most often a goal already called, shown again.
    p = CommentaryPolicy(CONFIG, PREMATCH, sources={"replay#1": {"candidate": {"replay": True, "second": 90}}})
    p.said(Item("background", [], ""), "Welcome.", T(0))
    eid = mem.add_entry(T(100), goal(80, player="h1"), seen("replay#1", sufficient=False))
    assert not is_news(ask(p, mem, 100 + 2 * HOLD), eid)
    mem.add_justification(T(120), eid, seen("live#2"))
    assert is_news(ask(p, mem, 120), eid)


def test_an_entry_a_verdict_replaced_with_a_fuller_one_is_confirmed_not_corrected():
    mem = MatchMemory(PLAYERS)
    change = Event("substitution", "home", ((1, 70), (1, 100)), (Participant("in", track_key=TRACK),))
    old = mem.add_entry(T(100), change, seen("live#1", sufficient=False))
    p = policy()
    assert air(p, mem, 100 + HOLD).entry == old                   # "looks like a change coming"
    # The arbiter names both players: the old entry is replaced by the fuller one (arbiter/verdict.py).
    mem.reject_observation(T(150), "live#1")
    new = mem.add_entry(T(150), Event("substitution", "home", ((1, 80), (1, 90)),
                                      (Participant("in", player_id="h4"), Participant("out", player_id="h1"))),
                        Justification("verdict:C1", "verdict", True))
    item = ask(p, mem, 150)
    assert (item.kind, item.entry) == ("confirmation", new)
    assert item.named == {"in": "h4", "out": "h1"}
