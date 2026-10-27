"""Phase-3 arbiter (DESIGN.md §6.6): a verdict settles its case and changes what memory
infers; a verdict that breaks a guardrail changes nothing; the budget bounds the effort.

The model is scripted (no API): each test plays the answers an arbiter would give. Cases come
from the real consistency check, on the same synthetic roster as tests/test_consistency.py.
"""

import copy
import json

from matchlive.arbiter.agent import Arbiter, Evidence, build_packet
from matchlive.arbiter.verdict import apply_verdict, reject_newest
from matchlive.common.config import load_config
from matchlive.events.analyzer import Candidate
from matchlive.events.consistency import ConsistencyChecker
from matchlive.memory.store import MatchMemory
from matchlive.memory.types import Identity

ROSTER = ([{"hash": f"h{i}", "team_side": "home", "starting": i <= 3} for i in range(1, 5)]
          + [{"hash": f"a{i}", "team_side": "away", "starting": i <= 3} for i in range(1, 5)])
CONFIG = load_config()


class Script:
    """A model that plays scripted answers, one per call."""

    def __init__(self, *answers):
        self.answers, self.calls = [json.dumps(a) for a in answers], 0

    def __call__(self, parts):
        self.calls += 1
        text = self.answers[min(self.calls, len(self.answers)) - 1]
        return type("Result", (), {"text": text, "latency": 10.0})()


def tool(name, args):
    return Evidence(f"{name} {args}")


def match_with(*proposals, tick=None):
    """Memory after the consistency check has seen `proposals`: (t_known, type, team, second,
    [player id or None]) with one track per proposal."""
    mem, checker = MatchMemory(ROSTER), ConsistencyChecker(copy.deepcopy(CONFIG))
    for n, (t_known, type, team, second, *player) in enumerate(proposals, 1):
        parts = (("player", (1, 1, n)),) if player else ()
        cand = Candidate(type, team, second, parts, False)
        checker.tick(mem, (1, t_known))
        checker.propose(mem, (1, t_known), 1, (second - 15, second + 15), cand, f"obs:call{n}#0", player)
    if tick:
        checker.tick(mem, (1, tick))
    return mem


def rule(mem, script, t_known=(1, 500), cooccur=lambda a, b: False):
    """Run the arbiter on the only open case and apply its verdict; return (case, void reason)."""
    [case] = [c for c in mem.cases.values() if c.status == "open"]
    decision = Arbiter(CONFIG["arbiter"], script).decide("packet", tool)
    void = apply_verdict(mem, t_known, case.case_id, decision.verdict or {}, decision.evidence_ids,
                         (1, 400), CONFIG["events"]["live_margin"], lambda key: True, cooccur)
    return case, void


def goal_and_replay():
    # A goal seen by two windows, then its replay 30 s later taken for a new goal.
    return match_with((130, "goal", "home", 100), (145, "goal", "home", 101), (170, "goal", "home", 130))


def test_verdict_rejects_the_replay_and_confirms_the_goal():
    mem = goal_and_replay()
    real, dup = "E1", "E2"
    case, void = rule(mem, Script(
        {"requests": [{"tool": "get_frames", "args": {"half": 1, "start": 90, "end": 140}}]},
        {"verdict": {"summary": "the second goal is a replay", "actions": [
            {"op": "reject", "observation": "obs:call3#0", "evidence": ["ev1"]},
            {"op": "support", "entry": real, "evidence": ["ev1"]}]}}))
    assert void is None and case.status == "closed"
    assert mem.status(dup) == "retracted" and mem.status(real) == "confirmed"
    assert mem.at((1, 600)).score == {"home": 1, "away": 0}


def test_verdict_adds_the_free_kick_nobody_reported():
    # A foul with no restart in the next minute: the arbiter finds the free kick and adds it.
    mem = match_with((130, "foul", "away", 100), tick=300)
    case, void = rule(mem, Script(
        {"requests": [{"tool": "get_frames", "args": {"half": 1, "start": 100, "end": 160}}]},
        {"verdict": {"summary": "free kick taken at 02:20", "actions": [
            {"op": "support", "entry": "E1", "evidence": ["ev1"]},
            {"op": "add_event", "type": "free_kick", "team": "home", "half": 1, "second": 140, "evidence": ["ev1"]}]}}))
    assert void is None and case.status == "closed"
    added = [r for r in mem.at((1, 600)).events(type="free_kick")]
    assert len(added) == 1 and added[0].status == "confirmed"
    assert mem.status("E1") == "confirmed"


def test_two_verdicts_finding_the_same_missed_event_add_it_once():
    # A foul with no restart, and a free kick with no infringement: both arbiters find the
    # same missing piece — here, each other — but also both "add" the same missed foul.
    mem = match_with((130, "foul", "away", 100), tick=300)
    add = {"op": "add_event", "type": "free_kick", "team": "home", "half": 1, "second": 140, "evidence": ["ev1"]}
    rule(mem, Script({"requests": [{"tool": "get_frames", "args": {}}]}, {"verdict": {"actions": [add]}}))
    n = len(mem.at((1, 600)).events(type="free_kick"))
    mem.open_case((1, 500), ["E1"], "restart_after_foul")        # a second case over the same spot
    _, void = rule(mem, Script({"requests": [{"tool": "get_frames", "args": {}}]},
                               {"verdict": {"actions": [{**add, "second": 143}]}}))
    assert void is None and n == 1 == len(mem.at((1, 600)).events(type="free_kick"))


def test_verdict_can_split_two_events_that_were_merged():
    # Two shots 13 s apart were merged into one entry; the arbiter keeps the first and adds
    # the second next to it — which must not be folded back into the case's own member.
    mem = match_with((130, "shot", "home", 100, "h1"), (150, "shot", "home", 113, "h2"))
    case, void = rule(mem, Script({"requests": [{"tool": "get_frames", "args": {}}]}, {"verdict": {"actions": [
        {"op": "support", "entry": "E2", "evidence": ["ev1"]},
        {"op": "add_event", "type": "shot", "team": "home", "half": 1, "second": 113, "evidence": ["ev1"]}]}}))
    assert void is None and len(mem.at((1, 600)).events(type="shot")) == 2


def test_verdict_names_the_player_behind_a_track():
    # Two sources disagree on the scorer; the arbiter reads the shirt and settles it.
    mem = match_with((130, "goal", "home", 100, "h1"), (150, "goal", "home", 104, "h2"))
    case, void = rule(mem, Script(
        {"requests": [{"tool": "jersey_reads", "args": {"track": [1, 1, 2]}}]},
        {"verdict": {"summary": "the scorer is h2", "actions": [
            {"op": "reject", "observation": "vote:1/1/1:h1", "evidence": ["ev1"]},
            {"op": "identity", "track": [1, 1, 1], "player": "h2", "evidence": ["ev1"]},
            {"op": "support", "entry": "E2", "evidence": ["ev1"]}]}}))    # ids: I1, E2 (the goal), I3
    assert void is None
    [goal] = mem.at((1, 600)).events(type="goal")
    assert goal.players == ("h2",)


def test_verdict_without_evidence_changes_nothing():
    mem = goal_and_replay()
    n_ops = len(mem.ops)
    case, void = rule(mem, Script({"verdict": {"summary": "a hunch", "actions": [
        {"op": "reject", "observation": "obs:call3#0", "evidence": ["ev9"]}]}}))
    assert void and case.status == "open" and len(mem.ops) == n_ops
    assert mem.at((1, 600)).score == {"home": 1, "away": 0}      # the replay still held pending


def test_an_entry_id_is_no_observation_to_reject():
    mem = goal_and_replay()
    case, void = rule(mem, Script({"requests": [{"tool": "get_frames", "args": {}}]},
                                  {"verdict": {"actions": [{"op": "reject", "observation": "E2", "evidence": ["ev1"]}]}}))
    assert void and case.status == "open"


def test_an_identity_ruled_on_is_the_only_one_on_its_track():
    # Two sources disagree on the scorer; the arbiter names track 1/1/1 without rejecting the vote.
    mem = match_with((130, "goal", "home", 100, "h1"), (150, "goal", "home", 104, "h2"))
    _, void = rule(mem, Script({"requests": [{"tool": "jersey_reads", "args": {"track": [1, 1, 1]}}]},
                               {"verdict": {"actions": [{"op": "identity", "track": [1, 1, 1], "player": "h2",
                                                         "evidence": ["ev1"]}]}}))
    assert void is None and mem.track_players()[(1, 1, 1)] == "h2"
    assert [mem.status(eid) for eid, e in mem.entries.items()
            if isinstance(e, Identity) and e.track_key == (1, 1, 1) and e.player_id == "h1"] == ["retracted"]


def test_replacing_a_paired_free_kick_does_not_retract_its_foul():
    mem = match_with((130, "foul", "away", 100), (150, "free_kick", "home", 110))
    mem.open_case((1, 300), ["E1"], "restart_after_foul")          # a case about the foul
    # The verdict names the taker: the fuller account replaces the free kick entry.
    _, void = rule(mem, Script({"requests": [{"tool": "get_frames", "args": {}}]},
                               {"verdict": {"actions": [{"op": "add_event", "type": "free_kick", "team": "home", "half": 1,
                                                         "second": 110, "participants": [{"role": "player", "player": "h1"}],
                                                         "evidence": ["ev1"]}]}}))
    assert void is None and mem.status("E2") == "retracted" and mem.status("E1") != "retracted"
    assert [r.players for r in mem.at((1, 600)).events(type="free_kick", status="confirmed")] == [("h1",)]


def test_the_case_shows_what_memory_knows_before_its_interval_ends():
    # A substitution seen at 290 counts from 335 on; a case opened at 310 still lists it.
    mem = match_with((300, "substitution", "home", 290))
    mem.open_case((1, 310), ["read:1/1/9:h4"], "off_roster_read", ["read:1/1/9:h4"])
    case = mem.cases["C1"]
    players = [p | {"jersey_number": str(i), "short_name": p["hash"]} for i, p in enumerate(ROSTER)]
    prematch = {"teams": {"home": "H", "away": "A"}, "jersey_colors": {"home": ["blue", "#0000FF"], "away": ["red", "#FF0000"]},
                "sides": {1: {"home": "left", "away": "right"}, 2: {"home": "right", "away": "left"}}, "players": players}
    packet = build_packet(mem, case, (1, 310), {}, prematch, 30)
    assert "- E1: team home" in packet.split("Substitutions memory holds so far:")[1]


def test_verdict_cannot_add_an_event_from_the_future_or_break_a_hard_constraint():
    mem = match_with((130, "foul", "away", 100), tick=300)
    _, void = rule(mem, Script(
        {"requests": [{"tool": "get_frames", "args": {}}]},
        {"verdict": {"actions": [{"op": "add_event", "type": "free_kick", "team": "home", "half": 1,
                                  "second": 450, "evidence": ["ev1"]}]}}))
    assert void and "known" in void                              # the case opened at 400
    # One player on two tracks seen in the same frame.
    mem = match_with((130, "goal", "home", 100, "h1"), (150, "goal", "home", 104, "h2"))
    _, void = rule(mem, Script(
        {"requests": [{"tool": "jersey_reads", "args": {}}]},
        {"verdict": {"actions": [{"op": "identity", "track": [1, 1, 3], "player": "h1", "evidence": ["ev1"]}]}}),
        cooccur=lambda a, b: True)
    assert void and "same frame" in void


def test_budget_bounds_the_rounds():
    mem = goal_and_replay()
    script = Script({"requests": [{"tool": "get_frames", "args": {}}] * 5})
    decision = Arbiter(CONFIG["arbiter"], script).decide("packet", tool)
    assert decision.verdict is None and script.calls == CONFIG["arbiter"]["max_rounds"]
    assert len(decision.evidence_ids) == CONFIG["arbiter"]["max_rounds"] * CONFIG["arbiter"]["max_requests"]


def test_a_later_verdict_can_overturn_an_earlier_one():
    mem = match_with((130, "foul", "away", 100), tick=300)
    rule(mem, Script(
        {"requests": [{"tool": "get_frames", "args": {}}]},
        {"verdict": {"actions": [
            {"op": "add_event", "type": "free_kick", "team": "home", "half": 1, "second": 140, "evidence": ["ev1"]}]}}))
    assert len(mem.at((1, 600)).events(type="free_kick", status="confirmed")) == 1
    # A later case finds that free kick was misseen: rejecting the verdict removes what it added.
    mem.reject_observation((1, 700), "verdict:C1")
    assert mem.at((1, 800)).events(type="free_kick", status="confirmed") == []


def test_baseline_rejects_the_newest_belief():
    # loop.arbiter off (DESIGN.md §7): no look at the footage, the later goal simply loses.
    mem = goal_and_replay()
    [case] = [c for c in mem.cases.values() if c.status == "open"]
    reject_newest(mem, (1, 500), case.case_id)
    assert case.status == "closed" and mem.status("E2") == "retracted"
    # Right here only because the first goal was seen twice; nothing was checked.
    assert mem.at((1, 600)).score == {"home": 1, "away": 0}
