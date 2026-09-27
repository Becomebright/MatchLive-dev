"""Phase-1 acceptance (DESIGN.md §10): match memory keeps beliefs with their evidence, and
every inference follows the evidence as it is added and ruled out.

Each test plays a short match script against a small synthetic roster: home h1-h3 start
with h4 on the bench, away a1-a3 start with a4 on the bench.
"""

import pytest

from matchlive.memory.store import MatchMemory
from matchlive.memory.types import Event, Identity, Justification, Participant

ROSTER = ([{"hash": f"h{i}", "team_side": "home", "starting": i <= 3} for i in range(1, 5)]
          + [{"hash": f"a{i}", "team_side": "away", "starting": i <= 3} for i in range(1, 5)])


def seen(obs_id, sufficient=True):
    return Justification(obs_id, "observation", sufficient)


def event(type, team, t, **participants):
    # A live candidate over the 10 s segment ending at first-half second t; participants
    # are given as role=player_id, or role=("track", key) to reference a track.
    parts = tuple(Participant(role, track_key=v[1]) if isinstance(v, tuple) else Participant(role, player_id=v)
                  for role, v in participants.items())
    return Event(type, team, ((1, t - 10), (1, t)), parts)


def T(second):
    return (1, second)


def test_status_follows_justifications():
    mem = MatchMemory(ROSTER)
    # Seen once, with the scorer's number unreadable: not enough to confirm.
    goal = mem.add_entry(T(100), event("goal", "home", 100, player="h1"), seen("live#1", sufficient=False))
    assert mem.status(goal) == "pending"
    assert mem.at(T(100)).score == {"home": 0, "away": 0}

    # The replay shows it clearly: a second, sufficient justification confirms it.
    mem.add_justification(T(130), goal, seen("replay#1"))
    assert mem.status(goal) == "confirmed"
    assert mem.at(T(130)).score == {"home": 1, "away": 0}

    # Justifications are alternatives: ruling out the replay leaves the live sighting.
    mem.reject_observation(T(200), "replay#1")
    assert mem.status(goal) == "pending"
    # With every justification ruled out the belief is retracted.
    mem.reject_observation(T(210), "live#1")
    assert mem.status(goal) == "retracted"
    assert mem.at(T(210)).score == {"home": 0, "away": 0}
    assert [j["rejected"] for j in mem.at(T(210)).why(goal)["justifications"]] == [T(210), T(200)]


def test_open_case_holds_members_pending():
    mem = MatchMemory(ROSTER)
    goal = mem.add_entry(T(100), event("goal", "away", 100, player="a1"), seen("live#1"))
    assert mem.at(T(100)).score["away"] == 1

    # A contradicting source opens a case: the goal is disputed and stops counting.
    case = mem.open_case(T(120), [goal, "replay#1"], "same_claim_agrees", ["replay#1"])
    assert mem.status(goal) == "pending"
    assert mem.at(T(120)).score["away"] == 0
    assert mem.at(T(120)).why(goal)["open_cases"] == [case]

    mem.close_case(T(150), case, verdict="goal stands")
    assert mem.status(goal) == "confirmed"
    assert mem.at(T(150)).score["away"] == 1


def test_later_verdict_overturns_earlier_verdict():
    mem = MatchMemory(ROSTER)
    # The arbiter infers a substitution nobody saw (h1 off, h4 on) ...
    c1 = mem.open_case(T(300), ["read#bench-number"], "off_roster_number")
    sub = mem.add_entry(T(310), event("substitution", "home", 290, out="h1", **{"in": "h4"}),
                        Justification(f"verdict:{c1}", "verdict", True))
    mem.close_case(T(310), c1, verdict="missed substitution")
    assert "h4" in mem.at(T(310)).on_field["home"] and "h1" not in mem.at(T(310)).on_field["home"]

    # ... and a later verdict rules that inference out: the lineup reverts.
    mem.reject_observation(T(400), f"verdict:{c1}")
    assert mem.status(sub) == "retracted"
    view = mem.at(T(400))
    assert view.on_field["home"] == ["h1", "h2", "h3"] and view.bench["home"] == ["h4"]


def test_retracting_a_yellow_removes_the_derived_red():
    # Acceptance: withdrawing a yellow removes the red and the dismissal that two yellows
    # implied, while a directly observed red card is untouched.
    mem = MatchMemory(ROSTER)
    mem.add_entry(T(100), event("yellow_card", "home", 100, player="h2"), seen("live#y1"))
    mem.add_entry(T(500), event("yellow_card", "home", 500, player="h2"), seen("live#y2"))
    mem.add_entry(T(600), event("red_card", "away", 600, player="a3"), seen("live#r1"))

    view = mem.at(T(600))
    assert [(c.player_id, c.kind, c.derived) for c in view.cards] == [
        ("h2", "yellow_card", False), ("h2", "yellow_card", False), ("h2", "red_card", True),
        ("a3", "red_card", False)]
    assert "h2" not in view.on_field["home"] and "a3" not in view.on_field["away"]

    # The replay shows the second booking went to someone else's foul: rule it out.
    mem.reject_observation(T(700), "live#y2")
    view = mem.at(T(700))
    assert [(c.player_id, c.kind, c.derived) for c in view.cards] == [
        ("h2", "yellow_card", False), ("a3", "red_card", False)]
    assert "h2" in view.on_field["home"]
    assert "a3" not in view.on_field["away"]


def test_correcting_an_identity_moves_the_event_to_the_new_player():
    # Acceptance: when the identity of a track is withdrawn, the events that reference the
    # track change participant, and the stats follow.
    mem = MatchMemory(ROSTER)
    track = (1, 12, 7)
    mem.add_entry(T(100), Identity(track, "h1"), seen("jersey#t7"))
    goal = mem.add_entry(T(110), event("goal", "home", 110, player=("track", track)), seen("live#1"))
    view = mem.at(T(110))
    assert view.events(type="goal")[0].players == ("h1",)
    assert view.player_stats["h1"]["goal"] == 1

    # The number was misread; the right identity comes from a close-up.
    mem.reject_observation(T(200), "jersey#t7")
    mem.add_entry(T(200), Identity(track, "h3"), seen("jersey#t7-closeup"))
    view = mem.at(T(200))
    assert mem.status(goal) == "confirmed"          # the goal itself still stands
    assert view.events(type="goal")[0].players == ("h3",)
    assert view.player_stats["h1"]["goal"] == 0 and view.player_stats["h3"]["goal"] == 1
    assert view.score == {"home": 1, "away": 0}


def test_queries_see_only_what_was_known():
    mem = MatchMemory(ROSTER)
    # A goal around second 90, written once the segment was analysed at second 100.
    mem.add_entry(T(100), event("goal", "home", 90, player="h1"), seen("live#1"))
    mem.reject_observation(T(300), "live#1")

    # At second 95 the goal had happened but was not yet known.
    assert mem.at(T(95)).score["home"] == 0
    # In hindsight (known by second 200) the goal had happened by second 95 ...
    assert mem.at(T(95), known_by=T(200)).score["home"] == 1
    # ... and between being written and being ruled out, it counted.
    assert mem.at(T(250)).score["home"] == 1
    assert mem.at(T(300)).score["home"] == 0
    # Second half: tuples order across halves.
    assert mem.at((2, 0.0)).score["home"] == 0
    with pytest.raises(ValueError):
        mem.reject_observation(T(299), "anything")    # t_known may not go back


def test_substitution_updates_lineups():
    mem = MatchMemory(ROSTER)
    sub = mem.add_entry(T(1000), event("substitution", "away", 1000, out="a2", **{"in": "a4"}), seen("live#s1"))
    view = mem.at(T(1000))
    assert view.on_field["away"] == ["a1", "a3", "a4"] and view.bench["away"] == []
    # Before it happened, the starting lineup.
    assert mem.at(T(900), known_by=T(1000)).on_field["away"] == ["a1", "a2", "a3"]

    mem.reject_observation(T(1100), "live#s1")
    assert mem.status(sub) == "retracted"
    assert mem.at(T(1100)).on_field["away"] == ["a1", "a2", "a3"]


def test_cascade_off_keeps_inferences_forward_only():
    # The loop.cascade=off baseline (GameSight-style): the retracted yellow card itself is
    # marked retracted, but what was inferred from it, and who was credited, never changes.
    mem = MatchMemory(ROSTER, cascade=False)
    mem.add_entry(T(100), event("yellow_card", "home", 100, player="h2"), seen("live#y1"))
    y2 = mem.add_entry(T(500), event("yellow_card", "home", 500, player="h2"), seen("live#y2"))
    track = (1, 3, 1)
    mem.add_entry(T(510), Identity(track, "a1"), seen("jersey#t1"))
    mem.add_entry(T(520), event("goal", "away", 520, player=("track", track)), seen("live#g1"))

    mem.reject_observation(T(700), "live#y2")
    mem.reject_observation(T(700), "jersey#t1")
    mem.add_entry(T(700), Identity(track, "a2"), seen("jersey#t1-closeup"))

    view = mem.at(T(700))
    assert mem.status(y2) == "retracted"
    assert ("h2", "red_card", True) in [(c.player_id, c.kind, c.derived) for c in view.cards]
    assert "h2" not in view.on_field["home"]
    assert view.player_stats["a1"]["goal"] == 1 and view.player_stats["a2"]["goal"] == 0
