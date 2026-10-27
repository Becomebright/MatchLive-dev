"""Phase-4 loop.state_to_perception (DESIGN.md §6.1 identity (a), §6.4): memory's lineup constrains
who a track can be, and a number worn by nobody on the pitch becomes evidence that reaches the
arbiter — which can infer a substitution nobody saw, and so change the lineup perception votes over.

Synthetic roster: home h1-h3 start (#1-#3) with h4 (#10) on the bench; away a1-a3 start (#1-#3)
with a4 (#20) on the bench.
"""

import copy
import json

import cv2
import numpy as np

from matchlive.arbiter.agent import Arbiter, Evidence
from matchlive.arbiter.verdict import apply_verdict
from matchlive.common.config import load_config
from matchlive.common.types import Track
from matchlive.events.consistency import ConsistencyChecker
from matchlive.memory.store import MatchMemory
from matchlive.perception.identity import IdentityResolver, roster_index

NUMBERS = {"h1": "1", "h2": "2", "h3": "3", "h4": "10", "a1": "1", "a2": "2", "a3": "3", "a4": "20"}
ROSTER = [{"hash": h, "team_side": "home" if h[0] == "h" else "away", "starting": h[1] != "4",
           "jersey_number": n} for h, n in NUMBERS.items()]
CONFIG = load_config()
BLUE, RED = (255, 0, 0), (0, 0, 255)            # BGR


def lab(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def on_pitch(*extra):
    return roster_index([p for p in ROSTER if p["starting"] or p["hash"] in extra])


def reads(number):
    # One frame's jersey read of a track (top label, its probability): `number` read clearly, or
    # nothing readable.
    return np.array([np.nan, np.nan] if number is None else [int(number), 0.95], np.float32)


def play(resolver, frames, roster):
    """Feed frames of (home track's number, away track's number); return the last frame's tracks."""
    tracks = [Track(0, "player", 1, (0, 0, 20, 60), 1.0), Track(0, "player", 2, (50, 0, 20, 60), 1.0)]
    for f, (home_number, away_number) in enumerate(frames):
        out = resolver.update(f, tracks, np.stack([reads(home_number), reads(away_number)]),
                              np.stack([lab(BLUE), lab(RED)]), roster, confirmed={})
    return out


def resolver():
    cfg = copy.deepcopy(CONFIG["perception"]["identity"])
    cfg["kit_burn_in"] = 2          # two tracks' colours are enough to form the two kits here
    return IdentityResolver(cfg, {"home": lab(BLUE), "away": lab(RED)})


def test_votes_only_among_players_on_the_pitch():
    r = resolver()
    home, away = play(r, [("10", "2")] * 3, on_pitch())
    # #10 is the substitute h4: not on the pitch, so the home track stays unnamed ...
    assert home.team == "home" and home.player_id is None
    assert away.player_id == "a2"
    # ... and the read is evidence of an off-pitch player, since nobody on the pitch wears 10.
    assert r.off_roster_reads({home.key: home.team, away.key: away.team}, on_pitch(), roster_index(ROSTER)) \
        == [(home.key, "h4")]
    # Once memory has h4 on the pitch, the same votes name him.
    home, _ = play(r, [("10", "2")], on_pitch("h4"))
    assert home.player_id == "h4"


def test_a_number_worn_on_the_pitch_is_no_evidence():
    # The home track reads #2: a home #2 would be h2, on the pitch; an away #2 is on the pitch too.
    # Nothing off the pitch is implied, whatever the kit colour said.
    r = resolver()
    home, away = play(r, [("2", None)] * 3, on_pitch())
    assert r.off_roster_reads({home.key: home.team, away.key: away.team}, on_pitch(), roster_index(ROSTER)) == []


class Script:
    def __init__(self, *answers):
        self.answers, self.calls = [json.dumps(a) for a in answers], 0

    def __call__(self, parts):
        self.calls += 1
        return type("Result", (), {"text": self.answers[min(self.calls, len(self.answers)) - 1], "latency": 10.0})()


def test_substitute_read_on_k_tracks_leads_to_an_inferred_substitution():
    mem, checker = MatchMemory(ROSTER), ConsistencyChecker(copy.deepcopy(CONFIG))
    k = CONFIG["consistency"]["off_roster_k"]
    # h4's number read on k different tracks: one case, sent to the arbiter only at the k-th.
    escalated = [checker.off_roster_read(mem, (2, 600 + 10 * n), f"read:2/5/{n}:h4", "h4") for n in range(k + 1)]
    [case] = mem.cases.values()
    assert escalated[:k - 1] == [None] * (k - 1) and escalated[k - 1] == case.case_id and escalated[k] is None
    assert len(case.evidence) == k + 1
    # A different substitute's number is a different case.
    checker.off_roster_read(mem, (2, 700), "read:2/6/1:a4", "a4")
    assert len(mem.cases) == 2

    # The arbiter looks at the tracks and infers the substitution nobody saw: h4 on for h3.
    script = Script({"requests": [{"tool": "get_track_crops", "args": {"track": [2, 5, 0]}}]},
                    {"verdict": {"summary": "h4 came on for h3", "actions": [
                        {"op": "add_event", "type": "substitution", "team": "home", "half": 2, "second": 560,
                         "participants": [{"role": "out", "player": "h3"}, {"role": "in", "player": "h4"}],
                         "evidence": ["ev1"]}]}})
    decision = Arbiter(CONFIG["arbiter"], script).decide("packet", lambda tool, args: Evidence(tool))
    void = apply_verdict(mem, (2, 720), case.case_id, decision.verdict, decision.evidence_ids, (2, 700),
                         CONFIG["events"]["live_margin"], lambda key: True, lambda a, b: False)
    assert void is None and case.status == "closed"
    view = mem.at((2, 800))
    assert "h4" in view.on_field["home"] and "h3" not in view.on_field["home"]
    # So identity now votes with h4 on the pitch (the runtime rebuilds the roster from this view).
    assert ("home", "10") in roster_index([p for p in ROSTER if p["hash"] in view.on_field["home"]])


def test_arbiter_names_the_players_of_a_seen_substitution():
    # The analyzer saw a home substitution but neither player was identified, so the lineup
    # cannot change; then h4's number keeps turning up on the pitch.
    config = copy.deepcopy(CONFIG)
    mem, checker = MatchMemory(ROSTER), ConsistencyChecker(config)
    from matchlive.events.analyzer import Candidate
    cand = Candidate("substitution", "home", 500, (("out", (2, 3, 1)), ("in", (2, 3, 2))), False)
    for n in (1, 2):                                   # two windows: confirmed, players unknown
        checker.propose(mem, (2, 520 + 10 * n), 2, (480, 510 + 10 * n), cand, f"obs:sub{n}#0", [None, None])
    [sub] = [eid for eid, e in mem.entries.items() if e.type == "substitution"]
    assert mem.status(sub) == "confirmed" and "h4" not in mem.at((2, 600)).on_field["home"]
    for n in range(config["consistency"]["off_roster_k"]):
        case_id = checker.off_roster_read(mem, (2, 600 + n), f"read:2/5/{n}:h4", "h4")
    # The verdict adds the same substitution with both players named. It replaces the entry
    # it matches (whose observations are rejected) instead of being folded into it, which would
    # drop the players; one of the number reads is ruled a misread.
    verdict = {"summary": "h4 on for h3", "actions": [
        {"op": "reject", "observation": "read:2/5/0:h4", "evidence": ["ev1"]},
        {"op": "add_event", "type": "substitution", "team": "home", "half": 2, "second": 500,
         "participants": [{"role": "out", "player": "h3"}, {"role": "in", "player": "h4"}], "evidence": ["ev1"]}]}
    assert apply_verdict(mem, (2, 650), case_id, verdict, ["ev1"], (2, 610), config["events"]["live_margin"],
                         lambda key: True, lambda a, b: False) is None
    view = mem.at((2, 700))
    assert mem.status(sub) == "retracted"
    assert "h4" in view.on_field["home"] and "h3" not in view.on_field["home"]
    assert view.team_stats["home"]["substitution"] == 1
