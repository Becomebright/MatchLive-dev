"""Ground-truth events of a match from its SoccerNet labels, for evaluating live runs.

    python scripts/ground_truth.py --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool"   # prints it

Two label files in the match folder (docs/SETUP.md §4), used for evaluation only — the live
system never reads them:
  - Labels-v2.json (SoccerNet-v2 action spotting): every event of its classes, with the team and
    the time in ms into the half video (our (half, second), DESIGN.md §4). Exhaustive for its
    classes, which map onto ours as in V2_TYPE; `team` is the side performing the action (for a
    foul, the side committing it: the free kick after it goes to the other side in 54 of 54
    pairs on our three matches), as in the event analyzer's convention.
  - Labels-caption.json (SoccerNet-Caption): commentary with player names, for who scored, who
    was booked, and who came on and off. A caption is matched to its Labels-v2 event by type,
    team and time (captions lag the action by up to a minute or two).
Injuries are not scored: Labels-v2 has no such class.

A substitution caption names two players; the one on the pitch at that moment goes off, the
one on the bench comes on (the lineups replayed from the starting XI through the captions in
time order).
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Mapping

from matchlive.common.config import load_config
from matchlive.common.match import Match

V2_TYPE = {"Goal": "goal", "Shots on target": "shot", "Shots off target": "shot", "Yellow card": "yellow_card",
           "Red card": "red_card", "Yellow->red card": "red_card", "Substitution": "substitution",
           "Corner": "corner", "Foul": "foul", "Offside": "offside", "Direct free-kick": "free_kick",
           "Indirect free-kick": "free_kick", "Penalty": "penalty"}
# Caption labels that name the main actor of a Labels-v2 event.
CAPTION_TYPE = {"soccer-ball": "goal", "own-goal": "goal", "y-card": "yellow_card", "r-card": "red_card",
                "yr-card": "red_card", "substitution": "substitution"}


def ground_truth(match: Match) -> list[dict[str, Any]]:
    """The match's true events in time order: {"type", "team", "time": (half, second), "visible",
    "players": {role: player hash}} — roles as in memory (DESIGN.md §5.2): "player" for a goal's
    scorer or a card's booked player, "out" / "in" for a substitution; other events name nobody."""
    with open(match.dir / "Labels-v2.json") as f:
        anns = json.load(f)["annotations"]
    events = [{"type": V2_TYPE[a["label"]], "team": a["team"] if a["team"] in ("home", "away") else None,
               "time": (int(a["gameTime"][0]), int(a["position"]) / 1000), "visible": a["visibility"] == "visible",
               "players": {}}
              for a in anns if a["label"] in V2_TYPE]
    events.sort(key=lambda e: e["time"])
    _name_players(match, events)
    return events


def _name_players(match: Match, events: list[dict]) -> None:
    with open(match.dir / "Labels-caption.json") as f:
        caps = json.load(f)["annotations"]
    players = match.prematch["players"]
    teams = match.prematch["teams"]
    lineup = {p["hash"]: "on" if p["starting"] else "bench" for p in players}
    caps = sorted((c for c in caps if c["label"] in CAPTION_TYPE),
                  key=lambda c: (int(c["gameTime"][0]), _seconds(c["gameTime"])))
    used: set[int] = set()
    for c in caps:
        kind = CAPTION_TYPE[c["label"]]
        half, t = int(c["gameTime"][0]), _seconds(c["gameTime"])
        named = _mentions(c["description"], players)
        if not named:
            continue
        if kind == "substitution":
            out = next((p for p in named if lineup[p] == "on"), None)
            sub = next((p for p in named if lineup[p] == "bench"), None)
            if not (out and sub):
                continue
            lineup[out], lineup[sub] = "off", "on"
            team, roles = _side(players, sub), {"out": out, "in": sub}
        else:
            # The main actor is the one the caption tags with a team: "Chris Smalling (Manchester United)".
            tagged = [p for p in named if any(f"{n} ({teams[_side(players, p)]}" in c["description"]
                                              for n in _names(players, p))]
            actor = (tagged or named)[0]
            team, roles = _side(players, actor), {"player": actor}
            if kind == "goal" and c["label"] == "own-goal":
                team = "away" if team == "home" else "home"
        # The nearest unnamed event of this type and team; captions come after the action.
        near = [(abs(t - e["time"][1]), i) for i, e in enumerate(events)
                if i not in used and e["type"] == kind and e["time"][0] == half and e["team"] == team
                and -90 <= t - e["time"][1] <= 180]
        if near:
            i = min(near)[1]
            used.add(i)
            events[i]["players"] = roles
        elif kind == "substitution":
            # A substitution Labels-v2 lacks still changes the lineups.
            events.append({"type": kind, "team": team, "time": (half, t), "visible": False, "players": roles})
    events.sort(key=lambda e: e["time"])


def lineups(match: Match, truth: list[dict], t: tuple[int, float]) -> dict[str, set[str]]:
    """The players truly on the pitch at match time `t`, per side (substitutions only: none of
    our matches has a sending-off)."""
    on = {"home": set(), "away": set()}
    for p in match.prematch["players"]:
        if p["starting"]:
            on[p["team_side"]].add(p["hash"])
    for e in truth:
        if e["type"] == "substitution" and e["time"] <= t and "in" in e["players"]:
            on[e["team"]].discard(e["players"]["out"])
            on[e["team"]].add(e["players"]["in"])
    return on


def score(truth: list[dict], t: tuple[int, float]) -> dict[str, int]:
    """The true score at match time `t`."""
    out = {"home": 0, "away": 0}
    for e in truth:
        if e["type"] == "goal" and e["time"] <= t:
            out[e["team"]] += 1
    return out


def _names(players, pid) -> list[str]:
    p = next(p for p in players if p["hash"] == pid)
    return list(dict.fromkeys([p["long_name"], p["short_name"]]))


def _mentions(text: str, players) -> list[str]:
    """Roster players named in `text`, in order of first mention."""
    found = []
    for p in players:
        spots = [m.start() for n in (p["long_name"], p["short_name"])
                 for m in re.finditer(r"\b" + re.escape(n) + r"\b", text)]
        if spots:
            found.append((min(spots), p["hash"]))
    return [pid for _, pid in sorted(found)]


def _side(players, pid) -> str:
    return next(p["team_side"] for p in players if p["hash"] == pid)


def _seconds(game_time: str) -> float:
    minutes, seconds = game_time.split(" - ")[1].split(":")
    return int(minutes) * 60 + float(seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    args = ap.parse_args()
    config = load_config()
    match = Match(config, args.match)
    truth = ground_truth(match)
    names: Mapping[str, str] = {p["hash"]: p["short_name"] for p in match.prematch["players"]}
    for e in truth:
        if e["type"] in ("goal", "yellow_card", "red_card", "substitution", "penalty"):
            who = ", ".join(f"{r} {names[p]}" for r, p in e["players"].items())
            print(f"{e['time'][0]}H {e['time'][1] / 60:5.1f}' {e['type']:13s} {e['team']}"
                  f"{'' if e['visible'] else ' (not shown)'} {who}")
    from collections import Counter
    print(Counter(e["type"] for e in truth))


if __name__ == "__main__":
    main()
