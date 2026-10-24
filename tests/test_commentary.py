"""Live commentary (DESIGN.md §6.9): what the policy gives the script of a stretch follows each
entry's status — a confirmed event names its player, a pending one only its team (and waits a
little for its confirmation first), a disputed one waits for the verdict — and when something
already said stops holding (retracted, or a different player), the next stretch corrects it with
the reason. Each event is placed at the moment it happens; nothing after the stretch is said; what
a script left out is offered again while it is still news; how the match stands and pre-match
background are offered only for a lull.

Each test writes a few ops to a small synthetic memory and asks the policy for a stretch's
material; no model is called (the generator only phrases the material over the pictures).
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
SEGMENT = CONFIG["segment"]


def T(second):
    return (1, second)


def seen(obs_id, sufficient=True):
    return Justification(obs_id, "observation", sufficient)


def goal(t, **participants):
    parts = tuple(Participant(role, track_key=v) if isinstance(v, tuple) else Participant(role, player_id=v)
                  for role, v in participants.items())
    return Event("goal", "home", ((1, t - 10), (1, t + 10)), parts)


def policy(prematch=PREMATCH, sources=None):
    p = CommentaryPolicy(CONFIG, prematch, sources=sources or {})
    # The half is already opened.
    p.said(Item("opener", [], ""), "Welcome.", T(0))
    return p


def ask(p, mem, t, end=None):
    """The material for the stretch ending at `end` (default: t) with memory known at t."""
    end = t if end is None else end
    return p.material(mem.at(T(t)), (1, end - SEGMENT, end))


def news(items):
    return [i.entry for i in items if i.kind == "news"]


def air(p, mem, t, end=None):
    """The stretch's must-say items, said as the script would."""
    items = ask(p, mem, t, end)
    for i in items:
        if i.must:
            p.said(i, f"line about {i.entry}", i.at)
    return items


def one(items, kind):
    found = [i for i in items if i.kind == kind]
    assert len(found) == 1, f"expected one {kind}, got {[i.kind for i in items]}"
    return found[0]


def test_confirmed_event_names_its_player_at_the_moment_it_happens():
    mem = MatchMemory(PLAYERS)
    mem.add_entry(T(95), Identity(TRACK, "h1"), seen("vote:h1"))
    eid = mem.add_entry(T(100), goal(85, player=TRACK), seen("live#1"))
    item = one(ask(mem=mem, p=policy(), t=100, end=90), "news")
    assert item.entry == eid and item.status == "confirmed" and item.must
    assert item.named == {"player": "h1"}
    assert any("Home Player1" in f for f in item.facts)
    assert item.at == T(85)                                      # where it happens in the stretch [80, 90)


def test_an_event_seen_before_the_stretch_is_said_first_thing():
    mem = MatchMemory(PLAYERS)
    eid = mem.add_entry(T(100), goal(70, player="h1"), seen("live#1"))
    item = one(ask(policy(), mem, 100, end=90), "news")
    assert item.entry == eid and item.at == T(90 - SEGMENT)


def test_pending_goal_waits_for_confirmation_then_is_hedged_and_names_nobody():
    mem = MatchMemory(PLAYERS)
    mem.add_entry(T(95), Identity(TRACK, "h1"), seen("vote:h1"))
    eid = mem.add_entry(T(100), goal(80, player=TRACK), seen("live#1", sufficient=False))
    p = policy()
    assert eid not in news(ask(p, mem, 100 + HOLD - 1))          # still waiting for a second sighting
    item = one(ask(p, mem, 100 + HOLD), "news")
    assert item.entry == eid and item.status == "pending"
    assert item.named == {} and not any("Player1" in f for f in item.facts)
    assert "hedge" in item.rule and "Homeside" in item.facts[0]


def test_disputed_event_waits_for_the_verdict():
    mem = MatchMemory(PLAYERS)
    first = mem.add_entry(T(160), goal(150, player="h1"), seen("live#1"))
    second = mem.add_entry(T(180), goal(175, player="h1"), seen("live#2"))
    case = mem.open_case(T(180), [first, second], "kickoff_after_goal")
    p = policy()
    # Neither goal is said while it is disputed (one of them is most likely a replay).
    assert news(ask(p, mem, 200)) == []
    # The verdict: the second one was the replay of the first.
    mem.reject_observation(T(210), "live#2")
    mem.add_justification(T(210), first, Justification(f"verdict:{case}", "verdict", True))
    mem.close_case(T(210), case, {"summary": "A replay.", "actions": []})
    assert news(ask(p, mem, 210)) == [first]


def test_pending_minor_event_is_said_only_once_confirmed():
    mem = MatchMemory(PLAYERS)
    shot = mem.add_entry(T(100), Event("shot", "away", ((1, 90), (1, 100))), seen("live#1", sufficient=False))
    p = policy()
    assert shot not in news(ask(p, mem, 100 + 2 * HOLD))         # half of these are false
    mem.add_justification(T(120), shot, seen("replay#1"))
    assert shot in news(ask(p, mem, 120))
    # A minor event confirmed long after the viewer saw it is stale: it is not news any more.
    late = mem.add_entry(T(130), Event("corner", "home", ((1, 60), (1, 70))), seen("live#2"))
    assert late not in news(ask(p, mem, 130))


def test_nothing_after_the_stretch_is_said():
    mem = MatchMemory(PLAYERS)
    eid = mem.add_entry(T(155), goal(150, player="h1"), seen("live#1"))
    p = policy()
    # Memory knows of the goal at 160, but the stretch ends at 100: no news, and the score is still 0-0.
    items = ask(p, mem, 160, end=100)
    assert news(items) == []
    assert any("Homeside 0, Awayside 0" in f for f in one(items, "recap").facts)
    # Once the viewer sees it, it is news.
    assert news(ask(p, mem, 160, end=152)) == [eid]


def test_a_must_say_item_left_out_is_offered_again_while_it_is_news():
    mem = MatchMemory(PLAYERS)
    shot = mem.add_entry(T(100), Event("shot", "home", ((1, 80), (1, 90))), seen("live#1"))
    p = policy()
    assert shot in news(ask(p, mem, 100, end=90))                # the script leaves it out
    assert shot in news(ask(p, mem, 100, end=100))               # the next stretch is given it again
    assert shot not in news(ask(p, mem, 100, end=90 + CONFIG["recent_minor"] + 1))   # by then it is stale


def test_retracted_fact_is_corrected_with_the_verdict():
    mem = MatchMemory(PLAYERS)
    first = mem.add_entry(T(100), goal(80, player="h1"), seen("live#1"))
    p = policy()
    assert news(air(p, mem, 100)) == [first]
    second = mem.add_entry(T(150), goal(140, player="h1"), seen("live#2"))
    assert news(air(p, mem, 150)) == [second]                    # aired as fact: a second goal
    # The arbiter finds the second goal was the replay of the first.
    case = mem.open_case(T(160), [first, second], "kickoff_after_goal")
    mem.reject_observation(T(200), "live#2")
    mem.close_case(T(200), case, {"summary": "E6 at 02:10.5 (track 1/4/7) is a replay of the first goal, E5.",
                                  "actions": [{"op": "reject", "observation": "live#2"}]})
    item = one(ask(p, mem, 200), "correction")
    assert (item.entry, item.status, item.at) == (second, "retracted", T(200 - SEGMENT))   # first thing
    reason = next(f for f in item.facts if f.startswith("What turned out"))
    assert "replay of the first goal" in reason
    assert not any(code in reason for code in ("E6", "E5", "02:10", "track"))   # no system codes on air
    assert mem.at(T(200)).score == {"home": 1, "away": 0}
    # Once corrected, it is not corrected again.
    p.said(item, "No, only one goal.", item.at)
    assert not any(i.kind == "correction" for i in ask(p, mem, 210))


def test_player_named_on_air_then_changed_is_corrected():
    mem = MatchMemory(PLAYERS)
    mem.add_entry(T(95), Identity(TRACK, "h1"), seen("vote:h1"))
    eid = mem.add_entry(T(100), goal(80, player=TRACK), seen("live#1"))
    p = policy()
    assert one(air(p, mem, 100), "news").named == {"player": "h1"}
    # A closer look reads the other number on that track: the scorer was h2.
    mem.reject_observation(T(300), "vote:h1")
    mem.add_entry(T(300), Identity(TRACK, "h2"), Justification("verdict:C9", "verdict", True))
    item = one(ask(p, mem, 300), "correction")
    assert item.entry == eid and item.named == {"player": "h2"}
    assert any("actually Player2" in f for f in item.facts)


def test_pending_goal_is_aired_again_once_confirmed():
    mem = MatchMemory(PLAYERS)
    eid = mem.add_entry(T(100), goal(80, player="h1"), seen("live#1", sufficient=False))
    p = policy()
    assert one(air(p, mem, 100 + HOLD), "news").named == {}
    assert not any(i.entry == eid for i in ask(p, mem, 100 + HOLD + 2))   # nothing new about it
    mem.add_justification(T(140), eid, seen("replay#1"))
    item = one(ask(p, mem, 140), "confirmation")
    assert (item.entry, item.named) == (eid, {"player": "h1"})


def test_one_incident_is_aired_once_but_a_card_always_is():
    mem = MatchMemory(PLAYERS)
    foul = mem.add_entry(T(100), Event("foul", "away", ((1, 90), (1, 100))), seen("live#1"))
    p = policy()
    assert news(air(p, mem, 100)) == [foul]
    # The free kick after it, and the same foul reported again, only repeat what was said.
    kick = mem.add_entry(T(130), Event("free_kick", "home", ((1, 115), (1, 125))), seen("live#2"))
    again = mem.add_entry(T(131), Event("foul", "away", ((1, 95), (1, 105))), seen("live#3"))
    assert not set(news(ask(p, mem, 131))) & {kick, again}
    # A card for the foul is news of its own.
    card = mem.add_entry(T(140), Event("yellow_card", "away", ((1, 100), (1, 110))), seen("live#4"))
    assert card in news(ask(p, mem, 140))


def test_one_incident_within_a_stretch_is_said_once():
    mem = MatchMemory(PLAYERS)
    foul = mem.add_entry(T(100), Event("foul", "away", ((1, 82), (1, 84))), seen("live#1"))
    kick = mem.add_entry(T(100), Event("free_kick", "home", ((1, 86), (1, 88))), seen("live#2"))
    said = news(ask(policy(), mem, 100, end=90))
    assert len(said) == 1 and said[0] in (foul, kick)


def test_minor_event_before_a_goal_on_air_is_overtaken_by_it():
    mem = MatchMemory(PLAYERS)
    g = mem.add_entry(T(170), goal(160, player="h1"), seen("live#1"))
    p = policy()
    assert g in news(air(p, mem, 170))
    # The free kick the goal came from is confirmed only now: once the goal is called, it is old news.
    kick = mem.add_entry(T(175), Event("free_kick", "home", ((1, 140), (1, 160))), seen("live#2"))
    mem.add_justification(T(176), kick, seen("replay#2"))
    assert kick not in news(ask(p, mem, 176))


def test_how_the_match_stands_is_offered_for_a_lull_and_says_something_new():
    mem = MatchMemory(PLAYERS)
    p = policy()
    every = CONFIG["context_every"]
    # A stretch starting `context_every` s after the half was opened.
    item = one(ask(p, mem, every + SEGMENT), "recap")
    assert not item.must and any("Homeside 0, Awayside 0" in f for f in item.facts)
    p.said(item, "Goalless so far.", item.at)
    # Not offered again before `context_every` s; then nothing has changed, so not the score again.
    assert not any(i.kind == "recap" for i in ask(p, mem, every + 30))
    assert not any(i.kind == "recap" for i in ask(p, mem, 2 * every + SEGMENT))
    # A goal changes it: the new score is offered.
    g = mem.add_entry(T(2 * every + 10), goal(2 * every, player="h1"), seen("live#1"))
    assert g in news(air(p, mem, 2 * every + 10))
    item = one(ask(p, mem, 3 * every + 30), "recap")
    assert any("Homeside 1, Awayside 0" in f for f in item.facts)


def test_pending_goal_seen_only_in_a_replay_waits_for_its_confirmation():
    mem = MatchMemory(PLAYERS)
    # A goal sighting in a replay, with no live one: most often a goal already called, shown again.
    p = policy(sources={"replay#1": {"candidate": {"replay": True, "second": 90}}})
    eid = mem.add_entry(T(100), goal(80, player="h1"), seen("replay#1", sufficient=False))
    assert eid not in news(ask(p, mem, 100 + 2 * HOLD))
    mem.add_justification(T(120), eid, seen("live#2"))
    assert eid in news(ask(p, mem, 120))


def test_an_entry_a_verdict_replaced_with_a_fuller_one_is_confirmed_not_corrected():
    mem = MatchMemory(PLAYERS)
    change = Event("substitution", "home", ((1, 70), (1, 100)), (Participant("in", track_key=TRACK),))
    old = mem.add_entry(T(100), change, seen("live#1", sufficient=False))
    p = policy()
    assert news(air(p, mem, 100 + HOLD)) == [old]                # "looks like a change coming"
    # The arbiter names both players: the old entry is replaced by the fuller one (arbiter/verdict.py).
    mem.reject_observation(T(150), "live#1")
    new = mem.add_entry(T(150), Event("substitution", "home", ((1, 80), (1, 90)),
                                      (Participant("in", player_id="h4"), Participant("out", player_id="h1"))),
                        Justification("verdict:C1", "verdict", True))
    items = ask(p, mem, 150)
    assert not any(i.kind == "correction" for i in items)
    item = one(items, "confirmation")
    assert item.entry == new and item.named == {"in": "h4", "out": "h1"}


# Pre-match background (Match.prematch["background"], DESIGN.md §6.8).
BACKGROUND = {"teams": {"home": {"manager": "Home Manager", "captain": None, "facts": ["Homeside are 2nd going in."]},
                        "away": {"manager": None, "captain": None, "facts": []}},
              "players": {"h1": ["He joined from Elsewhere in 2014.", "He has 20 caps."]}}


def test_a_players_pre_match_facts_are_given_once_each():
    mem = MatchMemory(PLAYERS)
    p = policy(PREMATCH | {"background": BACKGROUND})
    mem.add_entry(T(100), goal(80, player="h1"), seen("live#1"))
    facts = "\n".join(one(air(p, mem, 100), "news").facts)
    assert "joined from Elsewhere" in facts and "20 caps" not in facts
    second = mem.add_entry(T(300), goal(280, player="h1"), seen("live#2"))
    item = one(air(p, mem, 300), "news")
    assert item.entry == second
    facts = "\n".join(item.facts)
    assert "20 caps" in facts and "joined from Elsewhere" not in facts


def test_pre_match_facts_fill_a_lull_when_the_match_has_nothing_new():
    mem = MatchMemory(PLAYERS)
    p = policy(PREMATCH | {"background": BACKGROUND})
    every = CONFIG["context_every"]
    item = one(ask(p, mem, every + SEGMENT), "recap")            # the score, first
    p.said(item, "Goalless.", item.at)
    item = one(ask(p, mem, 2 * every + SEGMENT), "recap")        # nothing new in the match
    assert any("2nd going in" in f for f in item.facts)
    p.said(item, "They came in second.", item.at)
    assert not any(i.kind == "recap" for i in ask(p, mem, 3 * every + SEGMENT))   # said once


def test_background_over_the_pictures_is_for_a_lull_and_said_once():
    p = policy(PREMATCH | {"background": BACKGROUND})
    pictures = "Player IDs:\nP1: Homeside (home), #1 Player1\nNearest to the ball (frame:ID): 1:P1 2:P1"
    item = one(p.on_screen(pictures, T(100)), "background")
    assert not item.must and "joined from Elsewhere" in item.facts[0]
    # Not said (no lull): offered again; said: the next fact, but only `background_every` s later.
    assert "joined from Elsewhere" in one(p.on_screen(pictures, T(110)), "background").facts[0]
    p.said(item, "He came from Elsewhere.", T(110))
    assert p.on_screen(pictures, T(120)) == []
    assert "20 caps" in one(p.on_screen(pictures, T(110 + CONFIG["background_every"])), "background").facts[0]
