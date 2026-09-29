"""Phase-2 consistency check (DESIGN.md §6.4): each rule fires on the situation it exists
for, and not on ordinary play.

Each test feeds a short script of event candidates, as the analyzer would, into a memory
with a small synthetic roster: home h1-h3 start with h4 on the bench, away a1-a3 start with
a4 on the bench. Times are seconds of the first half.
"""

import copy

from matchlive.common.config import load_config
from matchlive.events.analyzer import Candidate
from matchlive.events.consistency import ConsistencyChecker
from matchlive.memory.store import MatchMemory

ROSTER = ([{"hash": f"h{i}", "team_side": "home", "starting": i <= 3} for i in range(1, 5)]
          + [{"hash": f"a{i}", "team_side": "away", "starting": i <= 3} for i in range(1, 5)])
CONFIG = load_config()


def setup(consistency=True, replay=True):
    config = copy.deepcopy(CONFIG)
    config["loop"]["consistency"] = consistency
    config["loop"]["replay"] = replay
    return Feed(MatchMemory(ROSTER), ConsistencyChecker(config))


class Feed:
    """Feeds candidates as analyzer answers arriving at `t_known`; each call is one source."""

    def __init__(self, mem, checker):
        self.mem, self.checker, self.n = mem, checker, 0

    def __call__(self, t_known, type, team, second, replay=False, window=None, **players):
        # players: role=(track_id, player_id or None); tracks live in shot 1 of half 1.
        self.n += 1
        parts = tuple((role, (1, 1, track)) for role, (track, _) in players.items())
        cand = Candidate(type, team, second, parts, (second,), replay)
        window = window or (second - 15, second + 15)
        self.checker.tick(self.mem, (1, t_known))
        return self.checker.propose(self.mem, (1, t_known), 1, window, cand, f"obs:call{self.n}#0",
                                    [p for _, p in players.values()])


def cases(mem, constraint=None):
    return [c for c in mem.cases.values() if c.status == "open" and (constraint is None or c.constraint == constraint)]


def test_second_sighting_joins_the_entry_and_confirms_it():
    feed = setup(); mem, checker = feed.mem, feed.checker
    g = feed(130, "goal", "home", 100, player=(7, "h1"))
    assert mem.status(g) == "pending"                          # one source is not enough
    # The next window overlaps and sees the same goal: same entry, now confirmed.
    assert feed(155, "goal", "home", 104, player=(7, "h1")) == g
    assert mem.status(g) == "confirmed"
    assert mem.at((1, 160)).score == {"home": 1, "away": 0}


def test_event_is_confirmed_even_when_its_actor_is_unidentified():
    # Whether it happened and who did it are separate beliefs (DESIGN.md §6.4): two windows
    # agreeing on a goal confirm it, and the score counts it, with no scorer identified.
    feed = setup(); mem, checker = feed.mem, feed.checker
    g = feed(130, "goal", "away", 100, player=(7, None))
    feed(155, "goal", "away", 104, player=(9, None))
    assert mem.status(g) == "confirmed"
    assert mem.at((1, 160)).score == {"home": 0, "away": 1}


def test_replay_candidate_matches_the_goal_it_repeats():
    feed = setup(); mem, checker = feed.mem, feed.checker
    g = feed(130, "goal", "home", 100, player=(7, "h1"))
    # Seen again 50 s later in a replay: its interval reaches back W seconds, so it matches.
    assert feed(190, "goal", "home", 150, replay=True, window=(140, 170), player=(9, "h1")) == g
    assert not cases(mem)


def test_replay_off_takes_the_replay_for_a_second_goal():
    # loop.replay off (DESIGN.md §7): the same replay is a live sighting 50 s after the goal,
    # so it is not matched to it — it becomes a second goal, which only the kick-off rule catches.
    feed = setup(replay=False); mem, checker = feed.mem, feed.checker
    g = feed(130, "goal", "home", 100, player=(7, "h1"))
    assert feed(190, "goal", "home", 150, replay=True, window=(140, 170), player=(9, "h1")) != g
    [case] = cases(mem, "kickoff_after_goal")
    assert g in case.members


def test_two_shots_in_one_window_stay_two_events():
    feed = setup(); mem, checker = feed.mem, feed.checker
    a = feed(130, "shot", "home", 100)
    b = feed(130, "shot", "away", 122)
    assert a != b


def test_unflagged_goal_replay_opens_a_kickoff_case():
    # The duplicate that cost precision in results.md §11: a replay 30 s after the goal,
    # not recognised as one, arrives as a second live goal.
    feed = setup(); mem, checker = feed.mem, feed.checker
    g1 = feed(130, "goal", "home", 100, player=(7, "h1"))
    feed(145, "goal", "home", 100, player=(7, "h1"))          # confirmed by a second window
    g2 = feed(170, "goal", "home", 130)
    [case] = cases(mem, "kickoff_after_goal")
    assert set(case.members) == {g1, g2}
    assert mem.at((1, 200)).score["home"] == 0                  # both held pending until the arbiter rules
    # A goal after a real restart is no conflict.
    feed(400, "goal", "home", 380)
    assert len(cases(mem, "kickoff_after_goal")) == 1


def test_duplicate_card_is_caught_before_it_becomes_a_red():
    feed = setup(); mem, checker = feed.mem, feed.checker
    y1 = feed(130, "yellow_card", "home", 100, player=(5, "h2"))
    y2 = feed(155, "yellow_card", "home", 125, player=(6, "h2"))
    [case] = cases(mem, "duplicate_card")
    assert set(case.members) == {y1, y2}
    assert not any(c.derived for c in mem.at((1, 200)).cards)   # no red card inferred
    # Two different identified players carded close together is ordinary play.
    feed(300, "yellow_card", "away", 280, player=(8, "a1"))
    feed(320, "yellow_card", "away", 300, player=(9, "a2"))
    assert len(cases(mem, "duplicate_card")) == 1


def test_foul_and_its_free_kick_confirm_each_other():
    feed = setup(); mem, checker = feed.mem, feed.checker
    f = feed(130, "foul", "away", 100)
    k = feed(175, "free_kick", "home", 140)                     # 40 s later: the next window
    assert mem.status(f) == "confirmed" and mem.status(k) == "confirmed"
    checker.tick(mem, (1, 1000))
    assert not cases(mem)


def test_missing_partners_open_cases_only_after_the_wait():
    feed = setup(); mem, checker = feed.mem, feed.checker
    f = feed(130, "foul", "away", 100)
    checker.tick(mem, (1, 200))                                 # the restart could still be coming
    assert not cases(mem, "restart_after_foul")
    checker.tick(mem, (1, 300))                                 # 110 + 150 s: no restart came
    assert [c.members for c in cases(mem, "restart_after_foul")] == [(f,)]
    k = feed(700, "free_kick", "home", 660)                     # long after: not that foul's restart
    checker.tick(mem, (1, 1000))
    assert [c.members for c in cases(mem, "infringement_before_free_kick")] == [(k,)]
    # The infringing side never takes the free kick: a same-team pair is not a pair.
    feed2 = setup(); mem2, checker2 = feed2.mem, feed2.checker
    feed2(130, "foul", "away", 100)
    feed2(175, "free_kick", "away", 140)
    checker2.tick(mem2, (1, 1000))
    assert len(cases(mem2)) == 2


def test_substitution_must_take_a_player_off_the_pitch_and_on_from_the_bench():
    feed = setup(); mem, checker = feed.mem, feed.checker
    feed(130, "substitution", "home", 100, out=(1, "h1"), **{"in": (2, "h4")})
    feed(160, "substitution", "home", 100, out=(1, "h1"), **{"in": (2, "h4")})   # confirmed
    assert not cases(mem, "substitution_invalid")
    # h4 is already on and h1 already off: the same swap again is impossible.
    # (Lineups follow confirmed entries only, so the first swap had to be confirmed.)
    s2 = feed(900, "substitution", "home", 870, out=(3, "h1"), **{"in": (4, "h4")})
    assert [c.members for c in cases(mem, "substitution_invalid")] == [(s2,)]


def test_player_who_left_cannot_take_part_again():
    feed = setup(); mem, checker = feed.mem, feed.checker
    feed(130, "substitution", "home", 100, out=(1, "h1"), **{"in": (2, "h4")})
    feed(160, "substitution", "home", 100, out=(1, "h1"), **{"in": (2, "h4")})   # confirmed
    shot = feed(600, "shot", "home", 570, player=(3, "h1"))
    assert [c.members for c in cases(mem, "player_off_pitch")] == [(shot,)]
    # A card is not play: a substituted player can still be booked.
    feed(700, "yellow_card", "home", 670, player=(4, "h1"))
    assert len(cases(mem, "player_off_pitch")) == 1


def test_sources_naming_different_players_disagree():
    feed = setup(); mem, checker = feed.mem, feed.checker
    g = feed(130, "goal", "home", 100, player=(7, "h1"))
    feed(150, "goal", "home", 104, player=(8, "h2"))
    [case] = cases(mem, "claims_disagree")
    assert g in case.members and mem.status(g) == "pending"


def test_consistency_off_is_the_v2_baseline():
    feed = setup(consistency=False); mem, checker = feed.mem, feed.checker
    feed(130, "goal", "home", 100)
    feed(170, "goal", "home", 130)
    checker.tick(mem, (1, 1000))
    assert not mem.cases
    assert mem.at((1, 200)).score["home"] == 2                  # the replay counted as a goal
