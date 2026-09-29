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
                   of one incident: a goal and the shot it came from, a foul and its free kick or
                   card);
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

import re
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
# Two events that are one incident on the pitch (the shot a goal came from; a foul and its free kick
# or card): aired once, the other is an echo.
INCIDENTS = {frozenset(p) for p in (("goal", "shot"), ("foul", "free_kick"), ("offside", "free_kick"), ("foul", "penalty"),
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
        # News is the play on screen now: no minute, and the score only with a goal.
        facts = [f"What: {self._phrase(e)}" + (" — being shown again in a replay" if replay else "") + "."]
        for role, pid in named.items():
            facts.append(f"{ROLE_WORDS[role].capitalize()}: {self._player(pid, view, e.type)}")
        if r.status == "confirmed":
            if e.type == "goal":                  # the score counts confirmed goals only
                facts.append(self._score(view))
            facts.append(f"Certain: {self._support(view, r.entry_id)}.")
            rule = "Call it as it happens" + (", naming the player(s) given." if named
                                              else "; you don't know who it was, so name nobody.")
        elif e.type in self.cfg["confirm"]:
            facts.append("Not certain yet.")
            rule = "You are not sure yet: hedge it (\"looks like...\", \"is that...?\") and name nobody, only the team."
        else:
            # A pending minor event is simply said without a player (DESIGN.md §6.9: the team only).
            rule = "Name nobody, only the team."
        if kind == "confirmation":
            facts.append(f"What you said before: \"{before.text}\"")
            rule = ("Now it is certain: say so the way a commentator would"
                    + (", naming the player(s) given." if named else "."))
        return Item(kind, facts, rule, r.entry_id, r.status, named, (half, second))

    def _correction(self, view: View, r: EventRecord, before: Aired, named: dict[str, str], what: str) -> Item:
        e = r.event
        half, second, _ = self._when(view, r)
        facts = [f"What you said before: \"{before.text}\"", f"About: {self._phrase(e)}, in {self._clock(half, second)}."]
        reason = _plain(self._reason(view, r.entry_id))
        if what == "retracted":
            facts.append(f"What turned out: {reason}")
            rule = ("Put it right the way a commentator does on air (\"No, ...\", \"Actually, ...\"), briefly, "
                    "with the reason in plain words.")
        else:
            for role, pid in before.named.items():
                if named.get(role) not in (None, pid):
                    facts.append(f"You named {self._player(pid, view, brief=True)} as the {ROLE_WORDS[role]}; "
                                 f"it was actually {self._player(named[role], view, brief=True)}.")
            facts.append(f"What turned out: {reason}")
            rule = "Put the name right the way a commentator does on air, briefly."
        if e.type == "goal":
            facts.append(self._score(view))
        return Item("correction", facts, rule, r.entry_id, r.status, named, (half, second))

    def _background(self, view: View) -> Item:
        half, teams, kits = view.time[0], self.prematch["teams"], self.prematch["jersey_colors"]
        facts = [f"{'Kick-off' if half == 1 else 'The second half is under way'}: {teams['home']}, in {kits['home'][0]} "
                 f"at home, against {teams['away']} in {kits['away'][0]}."]
        if half == 1:
            fronts = []
            for side in ("home", "away"):
                starters = [p for p in self.prematch["players"] if p["team_side"] == side and p["starting"]]
                forwards = [p["short_name"] for p in starters if p["position"] == "forward"]
                if forwards:
                    fronts.append(f"{' and '.join(forwards)} for {teams[side]}")
            if fronts:
                facts.append("Up front: " + "; ".join(fronts) + ".")
        else:
            facts.append(self._score(view))
        return Item("background", facts, "Welcome the viewers and open the half in one or two short spoken sentences.",
                    t_event=None)

    def _recap(self, view: View) -> Item:
        teams = self.prematch["teams"]
        facts = [f"Time: {self._clock(*view.time)}.", self._score(view)]
        for side in ("home", "away"):
            stats = view.team_stats[side]
            parts = [f"{stats[t]} {EVENT_WORDS[t]}{'s' if stats[t] != 1 else ''}" for t in ("shot", "corner") if stats[t]]
            if parts:
                facts.append(f"{teams[side]} so far: {', '.join(parts)}.")
        return Item("recap", facts, "Remind the viewers how the match stands, in one easy sentence; don't list every number.")

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
        """What made an entry certain, as a commentator would put it: a second look (the arbiter's
        check), the replay, or no doubt about it (two sightings agree)."""
        mem = view.mem
        holding = [j for _, j in mem.justifications[eid] if j.obs_id not in mem.rejected]
        if any(j.kind == "verdict" for j in holding):
            return "a second look at the pictures confirms it"
        if any(self.sources.get(j.obs_id, {}).get("candidate", {}).get("replay") for j in holding):
            return "the replay shows it"
        return "no doubt about it"

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
                    return v.get("summary") or "a second look at the pictures"
        if mem.open_cases(eid):
            return "it is now in doubt"
        return "a second look at the pictures"

    def _score(self, view: View) -> str:
        teams = self.prematch["teams"]
        return f"Score now: {teams['home']} {view.score['home']}, {teams['away']} {view.score['away']}."

    def _phrase(self, e) -> str:
        """"a corner for Chelsea"; "a corner" when the side is unknown."""
        if e.team is None:
            return f"a {EVENT_WORDS[e.type]}"
        return WITH_TEAM[e.type].format(team=self.prematch["teams"][e.team])

    def _player(self, pid: str, view: View, ty: str | None = None, brief: bool = False) -> str:
        """A player as a commentator names them — by the name fans use — with the background the
        line may draw on (DESIGN.md §6.9): full name, side and position, and how many of this kind
        of event they have had in the match when it is not their first."""
        p = self.players[pid]
        team = self.prematch["teams"][p["team_side"]]
        if brief:
            return f"{p['short_name']} ({team})"
        role = p["position"] or ("substitute" if not p["starting"] else "player")
        out = f"{p['short_name']} (full name {p['long_name']}; {team} {role})"
        n = view.player_stats.get(pid, {}).get(ty, 0) if ty else 0
        if n >= 2:
            out += f"; his {_ordinal(n)} {EVENT_WORDS[ty]} of the match"
        return out

    @staticmethod
    def _clock(half: int, second: float) -> str:
        """The minute of play as a commentator says it: "the 16th minute of the second half",
        "first-half stoppage time"."""
        minute = int(second // 60) + 1
        name = "first" if half == 1 else "second"
        if minute > 45:
            return f"{name}-half stoppage time"
        return f"the {_ordinal(minute)} minute of the {name} half"


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _plain(reason: str) -> str:
    """A verdict's summary without the system's codes — entry and case ids, track keys, shirt
    numbers, clock times to the tenth — which a commentator would never say."""
    reason = re.sub(r"\(?\btracks? \d+/\d+/\d+\)?", "", reason)
    reason = re.sub(r"\(?\b[ECI]\d+\b\)?|#\d+|\(\s*\)|\b\d{1,2}:\d{2}(\.\d+)?\b", "", reason)
    return re.sub(r"\s+([,.;])", r"\1", re.sub(r"\s{2,}", " ", reason)).strip()


PROMPT = """You are the TV commentator of this football match, speaking live on air in {language}. Give your next line:
what you would actually say into the microphone now, at most {max_words} words.
- Speak, don't write: short spoken sentences, present tense, the rhythm of live commentary. Many lines are just a
  few words ("Corner to Liverpool.", "Costa... and it's in!"); save the big lines for the big moments.
- Call players by the name given first; never say shirt numbers, positions in brackets, clock times to the second,
  or anything about how you know it ("sightings", "the system", "the arbiter", "memory").
- Use only what the material gives: never add a player, a number or a detail that is not in it.
- {rule}
- Don't repeat your recent lines or start the same way as them.

Your recent lines:
{history}

Material ({kind}):
{facts}

Answer with JSON only: {{"line": "...", "tone": "excited" | "lively" | "calm"}} — the tone to say it in."""


class Commentator:
    def __init__(self, config: Mapping[str, Any], generate: Callable[[Sequence[str | bytes]], Any]):
        # `config` is the `commentary` section; `generate(parts)` is one JSON-answer model call
        # returning an object with .text, .latency and .call_id (matchlive.common.api.APIClient).
        self.cfg = config
        self.generate = generate

    def line(self, item: Item, history: Sequence[str]) -> tuple[str, str, float, str]:
        """(the line, the tone to say it in, the call's latency in s, its call id)."""
        prompt = PROMPT.format(max_words=self.cfg["max_words"], language=self.cfg["language"], rule=item.rule,
                               kind=item.kind, facts="\n".join(f"- {f}" for f in item.facts),
                               history="\n".join(f"- {h}" for h in history[-self.cfg["history"]:]) or "(none yet)")
        result = self.generate([prompt])
        answer = parse_json(result.text)
        text = str(answer.get("line") or result.text).strip()
        tone = answer.get("tone") if answer.get("tone") in ("excited", "lively", "calm") else "lively"
        return text, tone, result.latency, result.call_id


def speaking_time(text: str, words_per_second: float) -> float:
    """How long a line takes to say, in s, when there is no audio to measure."""
    return max(1.5, len(text.split()) / words_per_second)


def schedule(lines: list[dict], config: Mapping[str, Any]) -> None:
    """Set each line's "show" (half, start, end) on its half video, in place: the broadcast output
    runs `output_delay` s behind the feed (DESIGN.md §6.9 D3), so a line ready at t is shown with
    the video at t - output_delay — but never before the event it speaks of is on screen, nor over
    the previous line. It lasts as long as its audio (or its estimated speaking time)."""
    prev = {1: 0.0, 2: 0.0}
    for line in lines:
        half, ready = line["t_ready"]
        start = ready - config["output_delay"]
        if line.get("t_event"):
            start = max(start, line["t_event"][1] + 1.0)
        start = max(start, prev[half], 0.0)
        end = start + (line.get("duration") or speaking_time(line["text"], config["words_per_second"]))
        prev[half] = end
        line["show"] = (half, start, end)
