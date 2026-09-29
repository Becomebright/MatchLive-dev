"""Live commentary (DESIGN.md §6.9): a policy that decides, from memory as known now, what to say
next, and a generator that turns the chosen material into one spoken line.

The policy is plain rules over memory — no model call — and speaks one line at a time. Each time
it is asked (the runtime asks every `tick` s while no line is being spoken) it picks, in order:
  1. correction    something already said no longer holds: an event aired as fact (or an aired
                   goal, card, substitution, penalty) has been retracted, or a player named on air
                   is no longer the one memory holds; the reason is what memory says overturned it
                   (the arbiter's verdict);
  2. confirmation  a goal, card, substitution or penalty aired while pending is now confirmed, or
                   its player has since been identified;
  3. news          the most important event memory has learnt of that has not been aired and
                   happened within the last `recent` s — except a minor one that only echoes an
                   event aired within `echo` s (the same kind by the same side, or the other half
                   of one incident: a foul and its free kick, a foul and its card);
  4. background    the first line of each half: the teams, the score, pre-match facts;
  5. recap         after `recap_after` s without a line: the score and how the match is going;
  6. silence.
Wording follows each event's status (DESIGN.md §6.9): a confirmed event names its players (those
memory has identified); a pending one names only the team — and a pending goal, card, penalty or
substitution is also hedged ("looks like ..."). What was said about each entry is kept, so a later
change of status or player is noticed and aired.

The generator makes one cached model call per line from the material alone (facts, the wording
rule, the last lines said) and may not add names or facts. It holds no memory; the runtime lends
the policy a memory view each time (DESIGN.md §8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from matchlive.events.analyzer import parse_json
from matchlive.memory.types import Time
from matchlive.memory.view import EventRecord, View

# How an event type is spoken of; the material says it in these words, with its side where known.
EVENT_WORDS = {"goal": "goal", "shot": "shot", "yellow_card": "yellow card", "red_card": "red card",
               "foul": "foul", "substitution": "substitution", "penalty": "penalty", "free_kick": "free kick",
               "offside": "offside", "corner": "corner", "injury": "stoppage for an injured player"}
WITH_TEAM = {"goal": "a goal for {team}", "shot": "a shot by {team}", "yellow_card": "a yellow card for a {team} player",
             "red_card": "a red card for a {team} player", "foul": "a foul by {team}",
             "substitution": "a {team} substitution", "penalty": "a penalty for {team}",
             "free_kick": "a free kick for {team}", "offside": "{team} caught offside", "corner": "a corner for {team}",
             "injury": "play stopped for an injured {team} player"}
# The roles a line may name: the main actors only (a fouled player is often the fouler misread).
ROLE_WORDS = {"player": "player", "in": "coming on", "out": "going off"}
# Two events that are one incident on the pitch: aired once, the other is an echo.
INCIDENTS = {frozenset(p) for p in (("foul", "free_kick"), ("offside", "free_kick"), ("foul", "penalty"),
                                    ("foul", "yellow_card"), ("foul", "red_card"))}


@dataclass
class Aired:
    """What was said about one event entry, as of the last line about it."""
    status: str                             # the entry's status when it was aired
    named: dict[str, str]                   # role -> player named aloud
    text: str                               # the line itself


@dataclass
class Item:
    """The material for one line, as the policy chose it."""
    kind: str                               # correction | confirmation | news | background | recap
    facts: list[str]
    rule: str                               # how to word it (DESIGN.md §6.9)
    entry: str | None = None
    status: str | None = None               # the entry's status now
    named: dict[str, str] = field(default_factory=dict)   # role -> player the line may name
    t_event: Time | None = None             # when the event happened (best estimate)


class CommentaryPolicy:
    def __init__(self, config: Mapping[str, Any], prematch: Mapping[str, Any],
                 sources: Mapping[str, Mapping[str, Any]]):
        # `config` is the `commentary` section; `sources` maps an observation id to the analyzer
        # window and candidate behind it (to say when an event was seen, and in a replay).
        self.cfg = config
        self.prematch = prematch
        self.sources = sources
        self.players = {p["hash"]: p for p in prematch["players"]}
        self.aired: dict[str, Aired] = {}
        self.last_line: Time | None = None
        self.last_recap: list[str] = []           # the last recap's facts, bar the clock
        self.halves_opened: set[int] = set()

    # ---- deciding ----

    def decide(self, view: View) -> Item | None:
        """What to say now, from `view` (the match as known now); None to stay silent."""
        now, mem = view.time, view.mem
        records = {r.entry_id: r for r in view.events()}
        confirm, announce = self.cfg["confirm"], self.cfg["announce"]

        # 1-2. What was aired and has changed since.
        for eid, a in self.aired.items():
            r = records.get(eid)
            if r is None:
                continue
            named = self._named(r)
            ty = r.event.type
            if r.status == "retracted" and a.status != "retracted":
                if a.status == "confirmed" or ty in confirm:
                    return self._correction(view, r, a, named, "retracted")
                continue
            # A name said on air is corrected only once memory holds a different player; while the
            # event is merely disputed (pending, so no name may be said) the line waits.
            changed = {role: pid for role, pid in a.named.items() if named.get(role) not in (None, pid)}
            if changed:
                return self._correction(view, r, a, named, "player")
            new_names = {role: pid for role, pid in named.items() if role not in a.named}
            if ty in confirm and ((a.status == "pending" and r.status == "confirmed") or new_names):
                return self._event_item(view, r, "confirmation", named, a)

        # 3. News: the most important unaired event of the last `recent` s.
        fresh = [r for r in records.values() if r.entry_id not in self.aired and r.status != "retracted"
                 and r.event.type in announce and r.event.t_interval[1][0] == now[0]
                 and now[1] - r.event.t_interval[1][1] <= self.cfg["recent"] and not self._echoes(r, records)]
        if fresh:
            r = min(fresh, key=lambda r: (announce.index(r.event.type), -r.event.t_interval[1][1]))
            return self._event_item(view, r, "news", self._named(r))

        # 4. The first line of a half.
        if now[0] not in self.halves_opened:
            return self._background(view)

        # 5. A recap after a silence, if the state has moved on since the last one.
        if self.last_line is None or self.last_line[0] != now[0] or now[1] - self.last_line[1] >= self.cfg["recap_after"]:
            item = self._recap(view)
            if item.facts[1:] != self.last_recap:
                return item
        return None

    def said(self, item: Item, text: str, now: Time) -> None:
        """Record that `item` went on air as `text` at match time `now`."""
        self.last_line = now
        if item.kind == "background":
            self.halves_opened.add(now[0])
        if item.kind == "recap":
            self.last_recap = item.facts[1:]
        if item.entry is not None:
            self.aired[item.entry] = Aired(item.status, dict(item.named), text)

    def _echoes(self, r: EventRecord, records: Mapping[str, EventRecord]) -> bool:
        """A minor event within `echo` s of an aired one that it only repeats: the same kind by the
        same side (often one event reported twice), or the other half of one incident. A goal,
        card, penalty or substitution is always news."""
        e = r.event
        if e.type in self.cfg["confirm"]:
            return False
        for eid in self.aired:
            o = records.get(eid)
            if o is None or o.status == "retracted" or o.event.t_interval[1][0] != e.t_interval[1][0] \
                    or abs(o.event.t_interval[1][1] - e.t_interval[1][1]) > self.cfg["echo"]:
                continue
            same_side = o.event.team is None or e.team is None or o.event.team == e.team
            if (o.event.type == e.type and same_side) or frozenset((o.event.type, e.type)) in INCIDENTS:
                return True
        return False

    # ---- material ----

    def _named(self, r: EventRecord) -> dict[str, str]:
        """The players a line may name: only on a confirmed event, only those memory identified."""
        if r.status != "confirmed":
            return {}
        return {p.role: pid for p, pid in zip(r.event.participants, r.players) if pid and p.role in ROLE_WORDS}

    def _event_item(self, view: View, r: EventRecord, kind: str, named: dict[str, str],
                    before: Aired | None = None) -> Item:
        e = r.event
        half, second, replay = self._when(view, r)
        # News is about the play on screen now: no minute, and the score only with a goal.
        facts = [f"Event: {self._phrase(e)}" + (" (seen in a replay)" if replay else "") + "."]
        for role, pid in named.items():
            facts.append(f"{ROLE_WORDS[role].capitalize()}: {self._player(pid, view, e.type)}")
        if e.type == "goal":
            facts.append(self._score(view))
        if r.status == "confirmed":
            facts.append(f"Status: confirmed ({self._support(view, r.entry_id)}).")
            rule = "Say it as fact" + (", naming the player(s) given." if named else "; no player is identified, so name none.")
        elif e.type in self.cfg["confirm"]:
            facts.append("Status: not confirmed yet (seen once so far, or disputed).")
            rule = "It is not confirmed: hedge it (e.g. \"looks like\") and name no player, only the team."
        else:
            facts.append("Status: not confirmed yet.")
            rule = "Name no player, only the team."
        if kind == "confirmation":
            facts.append(f"You said earlier: \"{before.text}\"")
            rule = ("Earlier you mentioned this tentatively or without the player; now it is confirmed: say so"
                    + (", naming the player(s) given." if named else "."))
        return Item(kind, facts, rule, r.entry_id, r.status, named, (half, second))

    def _correction(self, view: View, r: EventRecord, before: Aired, named: dict[str, str], what: str) -> Item:
        e = r.event
        half, second, _ = self._when(view, r)
        facts = [f"You said earlier: \"{before.text}\"", f"About: {self._phrase(e)}, {self._clock(half, second)}."]
        reason = self._reason(view, r.entry_id)
        if what == "retracted":
            facts.append(f"It turned out not to be right: {reason}")
            rule = "Correct yourself briefly and plainly (e.g. \"Correction: ...\"), giving the reason."
        else:
            for role, pid in before.named.items():
                if named.get(role) not in (None, pid):
                    facts.append(f"You named {self._player(pid, view, brief=True)} as the {ROLE_WORDS[role]}; "
                                 f"it was actually {self._player(named[role], view, brief=True)}.")
            facts.append(f"Why: {reason}")
            rule = "Correct the name briefly and plainly (e.g. \"Correction: ...\"), giving the reason."
        if e.type == "goal":
            facts.append(self._score(view))
        return Item("correction", facts, rule, r.entry_id, r.status, named, (half, second))

    def _background(self, view: View) -> Item:
        half, teams, kits = view.time[0], self.prematch["teams"], self.prematch["jersey_colors"]
        facts = [f"{'Kick-off' if half == 1 else 'The second half is under way'}: {teams['home']} ({kits['home'][0]}, home) "
                 f"against {teams['away']} ({kits['away'][0]}, away)."]
        if half == 1:
            for side in ("home", "away"):
                starters = [p for p in self.prematch["players"] if p["team_side"] == side and p["starting"]]
                forwards = [p["short_name"] for p in starters if p["position"] == "forward"]
                facts.append(f"{teams[side]} start with {', '.join(forwards) or 'their usual XI'} up front.")
        else:
            facts.append(self._score(view))
        return Item("background", facts, "Open the half in one sentence.", t_event=None)

    def _recap(self, view: View) -> Item:
        teams = self.prematch["teams"]
        facts = [f"It is {self._clock(*view.time)}.", self._score(view)]
        for side in ("home", "away"):
            stats = view.team_stats[side]
            parts = [f"{stats[t]} {EVENT_WORDS[t]}{'s' if stats[t] != 1 else ''}" for t in ("shot", "corner") if stats[t]]
            if parts:
                facts.append(f"{teams[side]} so far: {', '.join(parts)} (confirmed only).")
        return Item("recap", facts, "Sum up the state of the match in one sentence.")

    # ---- helpers ----

    def _when(self, view: View, r: EventRecord) -> tuple[int, float, bool]:
        """(half, second, replay only): the median live sighting, else the middle of the interval."""
        mem = view.mem
        seen = [self.sources[j.obs_id]["candidate"] for _, j in mem.justifications[r.entry_id]
                if j.obs_id in self.sources and j.obs_id not in mem.rejected]
        live = sorted(c["second"] for c in seen if not c["replay"])
        (half, lo), (_, hi) = r.event.t_interval
        if live:
            return half, live[len(live) // 2], False
        return half, (lo + hi) / 2, bool(seen)

    def _support(self, view: View, eid: str) -> str:
        """What confirmed an entry, in words: an arbiter's check, a replay, or two sightings."""
        mem = view.mem
        holding = [j for _, j in mem.justifications[eid] if j.obs_id not in mem.rejected]
        if any(j.kind == "verdict" for j in holding):
            return "checked against the footage by the arbiter"
        if any(self.sources.get(j.obs_id, {}).get("candidate", {}).get("replay") for j in holding):
            return "the replay shows it"
        return "seen by two independent sightings"

    def _reason(self, view: View, eid: str) -> str:
        """Why an entry changed: the summary of the verdict that rejected its observations or
        identified its players, if any."""
        mem = view.mem
        obs = {j.obs_id for _, j in mem.justifications[eid]}
        tracks = {tuple(p.track_key) for p in mem.entries[eid].participants if p.track_key}
        for case in reversed(list(mem.cases.values())):
            v = case.verdict
            if case.status != "closed" or not isinstance(v, dict):
                continue
            for a in v.get("actions", []):
                if not isinstance(a, dict):
                    continue
                if a.get("observation") in obs or (a.get("op") == "identity" and tuple(a.get("track") or ()) in tracks):
                    return v.get("summary") or "a second look at the footage"
        if mem.open_cases(eid):
            return "it is now disputed and under review"
        return "a second look at the footage"

    def _score(self, view: View) -> str:
        teams = self.prematch["teams"]
        return f"Score now: {teams['home']} {view.score['home']}-{view.score['away']} {teams['away']}."

    def _phrase(self, e) -> str:
        """"a corner for Chelsea"; "a corner" when the side is unknown."""
        if e.team is None:
            return f"a {EVENT_WORDS[e.type]}"
        return WITH_TEAM[e.type].format(team=self.prematch["teams"][e.team])

    def _player(self, pid: str, view: View, ty: str | None = None, brief: bool = False) -> str:
        p = self.players[pid]
        name = f"#{p['jersey_number']} {p['long_name']}"
        role = p["position"] or ("substitute" if not p["starting"] else "")
        name += f" ({self.prematch['teams'][p['team_side']]}{', ' + role if role and not brief else ''})"
        # Background from the match so far (DESIGN.md §6.9): the count of this kind of event,
        # when it is not the player's first.
        n = view.player_stats.get(pid, {}).get(ty, 0) if ty else 0
        if n >= 2 and not brief:
            nth = f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
            name += f"; his {nth} {EVENT_WORDS[ty]} of the match"
        return name

    @staticmethod
    def _clock(half: int, second: float) -> str:
        """The minute of play as a commentator says it: "the 16th minute of the second half",
        "first-half stoppage time"."""
        minute = int(second // 60) + 1
        name = "first" if half == 1 else "second"
        if minute > 45:
            return f"{name}-half stoppage time"
        nth = f"{minute}{'th' if 10 <= minute % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(minute % 10, 'th')}"
        return f"the {nth} minute of the {name} half"


PROMPT = """You are the live TV commentator of a football match. Say ONE spoken line of at most {max_words} words in
{language}, made only from the material below.
- Use only the facts given: never add a player, a number, a time or any detail that is not in the material.
- {rule}
- Sound like live commentary, not a report: vary how your lines start and are built; no hashtags or emojis.

Your last lines (do not repeat them):
{history}

Material ({kind}):
{facts}

Answer with JSON only: {{"line": "..."}}"""


class Commentator:
    def __init__(self, config: Mapping[str, Any], generate: Callable[[Sequence[str | bytes]], Any]):
        # `config` is the `commentary` section; `generate(parts)` is one JSON-answer model call
        # returning an object with .text, .latency and .call_id (matchlive.common.api.APIClient).
        self.cfg = config
        self.generate = generate

    def line(self, item: Item, history: Sequence[str]) -> tuple[str, float, str]:
        """(the line, the call's latency in s, its call id)."""
        prompt = PROMPT.format(max_words=self.cfg["max_words"], language=self.cfg["language"], rule=item.rule,
                               kind=item.kind, facts="\n".join(f"- {f}" for f in item.facts),
                               history="\n".join(f"- {h}" for h in history[-self.cfg["history"]:]) or "(none yet)")
        result = self.generate([prompt])
        text = parse_json(result.text).get("line") or result.text.strip()
        return str(text).strip(), result.latency, result.call_id


def speaking_time(text: str, words_per_second: float) -> float:
    """How long a line takes to say, in s."""
    return max(1.5, len(text.split()) / words_per_second)
