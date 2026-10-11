"""Live commentary (DESIGN.md §6.9): a policy that decides, from memory as known now, what to say
next, and a generator that turns the chosen material into one spoken line.

The output runs `output_delay` s behind the live feed, so each line is about the moment the viewer
will be watching when it goes on air, while memory is known up to the feed: that much hindsight on
every moment, and nothing said about what the viewer has not seen yet. The policy is plain rules
over memory — no model call — and picks, for the next line, in order:
  1. correction    something already said no longer holds: an event aired as fact (or an aired
                   goal, card, substitution, penalty) has been retracted, or a player named on air
                   is no longer the one memory holds; the reason is what memory says overturned it
                   (the arbiter's verdict). An entry a verdict replaced with a fuller one (the same
                   event, with who took part) is not retracted as far as the viewer is concerned:
                   what was said of it carries over to the new entry;
  2. confirmation  a goal, card, substitution or penalty aired while pending is now confirmed, or
                   its player has since been identified;
  3. background    the first line of each half: the teams, the score, pre-match facts (the venue,
                   the managers, where each side stands going in);
  4. news          the most important event the viewer has seen in the last `recent` s that has not
                   been aired — except a minor one that only echoes an event aired within `echo` s
                   (the same kind by the same side, or the other half of one incident: a goal and
                   the shot it came from, a foul and its free kick or card), a disputed one (a member
                   of an open case: it waits for the verdict), and a pending one before `hold` s (a
                   pending minor one not at all, unless `pending_minor`: half of them are false; nor
                   a pending one seen only in replays: a replay re-shows what the viewer has seen,
                   so it is most often an event already called, or none);
  5. analysis      the co-commentator's view, right after a goal, card or change has been called;
  6. context       every `context_every` s: one angle on how the match stands (score, counts,
                   who has the pressure, bookings, the busiest player) that has something new —
                   or, when the match has nothing new, a pre-match fact about a side;
  7. analysis      every `analysis_every` s, the co-commentator's view of the play;
  8. play          otherwise, the play on screen, written over the frames on air (the generator
                   is told not to call events: those come from memory, once certain).
Wording follows each event's status (DESIGN.md §6.9): a confirmed event names its players (those
memory has identified); a pending one names only the team — and a pending goal, card, penalty or
substitution is also hedged ("looks like ..."). What was said about each entry is kept, so a later
change of status or player is noticed and aired.
Pre-match background (Match.prematch["background"], DESIGN.md §6.8) is the only background a line
may use: a player's facts come with the first lines that name him, or with the pictures that show
him (one fact every `background_every` s at most), and each fact is offered once.

The generator makes one cached model call per line from the material alone (facts, the wording
rule, the last lines said; the frames for a play line) and may not add names or facts. It holds no
memory; the runtime lends the policy a memory view each time (DESIGN.md §8).
"""

from __future__ import annotations

import re
from collections import Counter
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


def replacement(mem: Any, eid: str) -> str | None:
    """The entry a verdict replaced a retracted event entry with, if any. A verdict that names
    everyone who took part in an event replaces the entry it matches: it rejects that entry's
    observations and adds the fuller event, all at the time the case closes (DESIGN.md §6.6,
    arbiter/verdict.py). What was said of the old entry still holds through the new one."""
    e = mem.entries[eid]
    when = [mem.rejected_at(j) for _, j in mem.justifications[eid]]
    if not when or None in when:
        return None                                    # not retracted
    t = max(when)
    for other, o in mem.entries.items():
        if other == eid or getattr(o, "type", None) != e.type or o.team != e.team:
            continue
        t0, j0 = mem.justifications[other][0]
        # Added by a verdict in the same write as the last rejection, and about the same moment.
        if t0 == t and j0.kind == "verdict" and o.t_interval[0] <= e.t_interval[1] and e.t_interval[0] <= o.t_interval[1]:
            return other
    return None


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
    event_type: str | None = None           # the event's type, for a line about an event
    angle: tuple | None = None              # (angle, value) a context line is about


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
        self.skipped: set[str] = set()           # news that could not go on air in time
        self.context_said: dict[str, tuple] = {}  # context angle -> (value, video second) last said
        self.last_line: Time | None = None       # video time of the last line on air
        self.last_context: Time | None = None    # ... and of the last line on how the match stands
        self.halves_opened: set[int] = set()
        self.last_big: Time | None = None        # when the last goal, card, penalty or change aired happened
        self.big_since_analysis = False          # ... and the co-commentator has not spoken since
        self.last_analysis: Time | None = None   # video time of the co-commentator's last line
        self.bg = prematch.get("background") or {}   # pre-match background (DESIGN.md §6.8)
        self.offered: set[str] = set()           # background facts already given to a line (each once)
        self.last_bg: Time | None = None         # video time a fact was last given over the pictures

    # ---- deciding ----

    def decide(self, view: View, on_air: Time) -> Item:
        """What to say next, to go on air at `on_air` — the video time the viewer is watching then.
        `view` is the match as known now: the output runs `output_delay` s behind the live feed, so
        memory already knows about the moment on air, with that much hindsight. Nothing that happens
        after the moment on air is said (no spoilers). Never silent: with nothing else to say, the
        line describes the play on screen."""
        mem = view.mem
        at_air = mem.at(on_air, known_by=view.time)       # the match as of the moment on air
        records = {r.entry_id: r for r in view.events()}
        confirm, announce = self.cfg["confirm"], self.cfg["announce"]

        # An aired entry a verdict replaced with a fuller one: what was said carries over to the
        # new entry (which may add the names), and the old one needs no correction.
        for eid, a in list(self.aired.items()):
            r = records.get(eid)
            if a.status != "retracted" and r is not None and r.status == "retracted":
                new = replacement(mem, eid)
                if new is not None and new not in self.aired:
                    self.aired[new] = a
                    self.aired[eid] = Aired("retracted", {}, a.text)

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

        # 3. The first line of a half.
        if on_air[0] not in self.halves_opened:
            return self._background(at_air)

        # 4. News: the most important unaired event the viewer has seen in the last `recent` s.
        fresh = [r for r in records.values() if r.entry_id not in self.aired and r.entry_id not in self.skipped
                 and r.status != "retracted"
                 and r.event.type in announce and self._seen(view, r, on_air) and not self._echoes(r, records)
                 and self._ready(view, r)]
        if fresh:
            r = min(fresh, key=lambda r: (announce.index(r.event.type), -r.event.t_interval[1][1]))
            return self._event_item(view, r, "news", self._named(r))

        # 5. The co-commentator's view right after a goal, card or change has been called.
        if self.big_since_analysis:
            return self._analysis(at_air, on_air)

        # 6. Now and then, how the match stands.
        if self.last_context is None or self.last_context[0] != on_air[0] \
                or on_air[1] - self.last_context[1] >= self.cfg["context_every"]:
            item = self._recap(at_air)
            if item is not None:
                return item

        # 7. Now and then, the co-commentator's view of the play.
        if self.last_analysis is None or self.last_analysis[0] != on_air[0] \
                or on_air[1] - self.last_analysis[1] >= self.cfg["analysis_every"]:
            return self._analysis(at_air, on_air)

        # 8. Otherwise, the play on screen.
        return self._play(at_air, on_air)

    def said(self, item: Item, text: str, on_air: Time) -> None:
        """Record that `item` went on air as `text` at video time `on_air`."""
        self.last_line = on_air
        if item.kind == "background":
            self.halves_opened.add(on_air[0])
            self.last_analysis = on_air
        if item.kind in ("background", "recap"):
            self.last_context = on_air
        if item.angle is not None:
            self.context_said[item.angle[0]] = (item.angle[1], on_air[1])
            if item.angle[0].startswith("background:"):
                self.offered.add(item.angle[0].removeprefix("background:"))
        if item.kind == "analysis":
            self.last_analysis, self.big_since_analysis = on_air, False
        if item.entry is not None:
            self.aired[item.entry] = Aired(item.status, dict(item.named), text)
        if item.kind in ("news", "confirmation") and item.event_type in self.cfg["confirm"]:
            self.last_big, self.big_since_analysis = item.t_event, True

    def dropped(self, item: Item, on_air: Time) -> None:
        """`item` could not go on air in time (for the moment `on_air`): its moment has passed, so it
        is not picked again — news is skipped, a confirmation or correction takes what memory now
        holds as said, and a context line or half opener waits its turn as if said (asked again at
        once it would be the same, cached, call — and as late)."""
        if item.kind == "recap":
            self.last_context = on_air
        elif item.kind == "background":
            self.halves_opened.add(on_air[0])
        if item.entry is None:
            return
        if item.kind == "news":
            self.skipped.add(item.entry)
        elif item.entry in self.aired:
            self.aired[item.entry] = Aired(item.status, dict(item.named), self.aired[item.entry].text)

    def _seen(self, view: View, r: EventRecord, on_air: Time) -> bool:
        """Whether the viewer has seen the event by `on_air`, at most `recent` s before it (`recent_minor`
        for a minor event: later, it is stale)."""
        half, second, _ = self._when(view, r)
        if r.event.type in self.cfg["confirm"]:
            return half == on_air[0] and 0 <= on_air[1] - second <= self.cfg["recent"]
        # A minor event from before the last big one aired is overtaken by it (the free kick a goal
        # came from is not news once the goal has been called).
        if self.last_big is not None and (half, second) < self.last_big:
            return False
        return half == on_air[0] and 0 <= on_air[1] - second <= self.cfg["recent_minor"]

    def _ready(self, view: View, r: EventRecord) -> bool:
        """Whether a not-yet-aired event may go on air now. A confirmed one may. A disputed one (a
        member of an open case) waits for the case to be settled: the dispute is most often a replay
        taken for a new goal, or one card reported twice. A pending one waits `hold` s for its
        confirmation, and then goes on air hedged only if it is a goal, card, penalty or
        substitution — or any type, with `pending_minor` — and was seen live: one seen only in
        replays waits for its confirmation (a replay re-shows what the viewer has already seen)."""
        if r.status == "confirmed":
            return True
        if view.mem.open_cases(r.entry_id) or self._when(view, r)[2]:
            return False
        written = view.mem.justifications[r.entry_id][0][0]
        waited = view.time[1] - written[1] if written[0] == view.time[0] else float("inf")
        return waited >= self.cfg["hold"] and (r.event.type in self.cfg["confirm"] or self.cfg["pending_minor"])

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
            if e.type == "goal":                  # the score counts confirmed goals only, up to this one
                facts.append(self._score(view.mem.at(e.t_interval[1], known_by=view.time)))
            support = self._support(view, r.entry_id)
            if support:
                facts.append(f"Seen: {support}.")
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
        return Item(kind, facts, rule, r.entry_id, r.status, named, (half, second), e.type)

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
            facts.append(self._score(view.mem.at(e.t_interval[1], known_by=view.time)))
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
            if self.bg.get("venue"):
                facts.append(f"Venue: {self.bg['venue']}.")
            for side in ("home", "away"):
                bg = self.bg.get("teams", {}).get(side, {})
                if bg.get("manager"):
                    facts.append(f"{teams[side]}'s manager: {bg['manager']}.")
                facts += [f"Background: {f}" for f in self._take(bg.get("facts", [])[:1])]
        else:
            facts.append(self._score(view))
        return Item("background", facts, "Welcome the viewers and open the half in two or three short spoken sentences"
                                         + (": where we are, who manages the sides and how they come into it." if half == 1 else "."),
                    t_event=None)

    def _recap(self, view: View) -> Item | None:
        """One angle on how the match stands at the moment on air (`view` is the match as of then):
        the score, a side's counts, who has had the recent pressure, a booking, the busiest player.
        An angle is offered only when there is something new in it since it was last said (the
        score or the pressure changed, a new booking, ...) or it has not been said for a while; the
        angles take turns. None when there is nothing new to say."""
        teams = self.prematch["teams"]
        half, now = view.time
        angles = []                                 # (angle, value, fact)
        angles.append(("score", tuple(view.score.values()), self._score(view)))
        recent = Counter(r.event.team for r in view.events(since=(half, max(0.0, now - 600)), status="confirmed")
                         if r.event.team and r.event.type in ("shot", "corner", "goal", "free_kick"))
        if sum(recent.values()) >= 3:
            lead = max(recent, key=recent.get)
            angles.append(("pressure", lead, f"Last ten minutes: {teams[lead]} have had most of the attacking play "
                                             f"({recent[lead]} of {sum(recent.values())} shots, corners and free kicks)."))
        for side in ("home", "away"):
            stats = view.team_stats[side]
            parts = [f"{stats[t]} {EVENT_WORDS[t]}{'s' if stats[t] != 1 else ''}"
                     for t in ("shot", "corner", "free_kick", "offside") if stats[t]]
            if parts:
                angles.append((f"counts:{side}", tuple(parts), f"{teams[side]} so far: {', '.join(parts)}."))
        for c in view.cards:
            if c.player_id:
                angles.append((f"booked:{c.player_id}", c.kind, f"Booked earlier: {self._player(c.player_id, view, brief=True)}."))
        busiest = sorted(((n, pid) for pid, st in view.player_stats.items() for ty, n in st.items() if ty == "shot"),
                         reverse=True)[:1]
        if busiest and busiest[0][0] >= 2:
            n, pid = busiest[0]
            angles.append(("busiest", (pid, n), f"Most shots so far: {self._player(pid, view, brief=True)}, {n}."))
        # What is new: an angle never said, or said with another value, or not said for `context_stale` s.
        fresh = [a for a in angles if a[0] not in self.context_said or self.context_said[a[0]][0] != a[1]
                 or now - self.context_said[a[0]][1] >= self.cfg["context_stale"]]
        # When the match has nothing new: a pre-match fact about a side (each once).
        if not fresh:
            for side in ("home", "away"):
                bg = self.bg.get("teams", {}).get(side, {})
                facts = bg.get("facts", []) + ([f"{teams[side]}'s captain: {bg['captain']}."] if bg.get("captain") else [])
                fresh += [(f"background:{f}", None, f"Background: {f}") for f in facts if f not in self.offered]
            fresh = fresh[:1] if fresh and fresh[0][0].startswith("background:") else fresh
        if not fresh:
            return None
        # Take turns: the angle said longest ago (never said first).
        angle, value, fact = min(fresh, key=lambda a: self.context_said.get(a[0], (None, -1.0))[1])
        facts = [f"Time: {self._clock(half, now)}.", fact]
        rule = ("Work this pre-match background into one natural spoken remark." if angle.startswith("background:")
                else "Work this into a natural remark on how the match stands; don't list numbers.")
        return Item("recap", facts, rule, t_event=None, angle=(angle, value))

    def _analysis(self, view: View, on_air: Time) -> Item:
        """The co-commentator's view, written over the last `analysis_span` s of pictures (after a
        goal, card or change: the replays of it)."""
        facts = [f"Time: {self._clock(*on_air)}.", self._score(view)]
        shapes = self.bg.get("formations")
        if shapes and self._take([f"formations:{on_air[0]}"]):
            teams = self.prematch["teams"]
            facts.append(f"Background: the line-ups — {teams['home']} {shapes['home']}, {teams['away']} {shapes['away']}.")
        return Item("analysis", facts, "Give your view on what we have just seen.",
                    t_event=(on_air[0], max(0.0, on_air[1] - self.cfg["analysis_span"])))

    def _play(self, view: View, on_air: Time) -> Item:
        """A line over the pictures on air: the runtime lends the generator the last `play_span` s of
        frames up to `on_air`, with who each tracked player is as far as known."""
        facts = [f"Time: {self._clock(*on_air)}.", self._score(view)]
        return Item("play", facts, "Describe the play in the pictures as it happens.",
                    t_event=(on_air[0], max(0.0, on_air[1] - self.cfg["play_span"])))

    # ---- helpers ----

    def _when(self, view: View, r: EventRecord) -> tuple[int, float, bool]:
        """(half, second, replay only): the median live sighting, else the middle of the interval."""
        mem = view.mem
        seen = [self.sources[j.obs_id]["candidate"] for _, j in mem.justifications[r.entry_id]
                if j.obs_id in self.sources and mem.holds(j)]
        live = sorted(c["second"] for c in seen if not c["replay"])
        (half, lo), (_, hi) = r.event.t_interval
        if live:
            return half, live[len(live) // 2], False
        return half, (lo + hi) / 2, bool(seen)

    def _support(self, view: View, eid: str) -> str:
        """What made an entry certain, as a commentator would put it: a second look (the arbiter's
        check) or the replay; nothing when two sightings simply agree."""
        mem = view.mem
        holding = [j for _, j in mem.justifications[eid] if mem.holds(j)]
        if any(j.kind == "verdict" for j in holding):
            return "a second look at the pictures confirms it"
        if any(self.sources.get(j.obs_id, {}).get("candidate", {}).get("replay") for j in holding):
            return "the replay shows it"
        return ""

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
        country = self.bg.get("nationality", {}).get(pid)
        out = f"{p['short_name']} (full name {p['long_name']}; {team} {role}" + (f", from {country})" if country else ")")
        n = view.player_stats.get(pid, {}).get(ty, 0) if ty else 0
        if n >= 2:
            out += f"; his {_ordinal(n)} {EVENT_WORDS[ty]} of the match"
        for fact in self._take(self.bg.get("players", {}).get(pid, [])):
            out += f"; background you may use: {fact}"
        return out

    def on_screen(self, pictures: str, on_air: Time) -> list[str]:
        """Background on a player the pictures identify (in their who-is-who text), for a play or
        co-commentator line — the one on the ball most, if he has a fact not yet given: one fact,
        at most every `background_every` s."""
        if self.last_bg is not None and self.last_bg[0] == on_air[0] \
                and on_air[1] - self.last_bg[1] < self.cfg["background_every"]:
            return []
        # Who is who ("P3: Chelsea (home), #21 Matic"), and who is nearest the ball in each frame
        # ("Nearest to the ball (...): 1:P3 2:P3 3:- ..."): the player on the ball most, first.
        local = {}
        for line in pictures.splitlines():
            for p in self.prematch["players"]:
                if line.startswith("P") and f"#{p['jersey_number']} {p['short_name']}" in line:
                    local[line.split(":")[0]] = p
        ball = next((line for line in pictures.splitlines() if line.startswith("Nearest to the ball")), "")
        carried = Counter(re.findall(r"\d+:(P\d+)", ball))
        shown = sorted(local, key=lambda pid: (-carried[pid], int(pid[1:])))
        for p in (local[pid] for pid in shown):
            for fact in self._take(self.bg.get("players", {}).get(p["hash"], [])):
                self.last_bg = on_air
                return [f"Background to say in this line, on {p['short_name']}: {fact}"]
        return []

    def _take(self, facts: Sequence[str], n: int = 1) -> list[str]:
        """The first `n` facts not yet given to a line, now marked as given."""
        new = [f for f in facts if f not in self.offered][:n]
        self.offered.update(new)
        return new

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
- Use only what the material gives: never add a player, a number or a detail that is not in it, and no background
  of your own about players, clubs or seasons (it may be wrong, or tell what happens later). Background given in the
  material is pre-match knowledge: use it when it fits, in your own words.
- {rule}
- Don't repeat your recent lines or start the same way as them.

Your recent lines:
{history}

Material ({kind}):
{facts}

Answer with JSON only: {{"line": "...", "tone": "excited" | "lively" | "calm"}} — the tone to say it in."""


PLAY_PROMPT = """You are the TV commentator of this football match, speaking live on air in {language}. Below are the
last {span:.0f} seconds of broadcast pictures, in order; tracked players carry a label (P1, P2, ...) and the text
before the pictures says who each one is, as far as is known. Say what you would say over these pictures right now,
the way a live TV commentator talks: in short bursts — one to three phrases said one after another as the play
moves on, each usually two to six words, {play_words} words in all at most (a phrase giving background may be longer).
- A phrase is a name as he gets the ball, what he does with it, a quick reaction, a question, the tempo or the crowd,
  or what an on-screen graphic shows: ["Kante.", "Wins it back...", "Hazard!"]; ["Lovely ball.", "Mane's away down
  the right."]; ["Liverpool knocking it about at the back."]; ["Stamford Bridge getting restless."]. Present tense,
  British TV style; vary the form, and don't list every pass.
- Don't call goals, shots, saves, fouls, cards, offsides, corners or free kicks: those are called separately, once
  they are certain. If the pictures show a stoppage, a close-up, the crowd or a replay, say something that fits,
  from the facts given.
- Name a player only when the text identifies him; otherwise say the team. Never say shirt numbers or the labels.
- Say only what the pictures show and the facts give: no background of your own about players, clubs or seasons
  ("his first goal for the club", "since he joined"). Don't repeat your recent lines or start the same way as them.
- If the facts give background to say in this line (pre-match knowledge about a player on screen), this line is
  about him: say it now, briefly and in your own words, tied to what he is doing (["Kante.", "Thirty-two million
  from Leicester in the summer — and he's won it back again."]).

Your recent lines:
{history}

Facts:
{facts}

Answer with JSON only: {{"phrases": ["...", ...], "tone": "excited" | "lively" | "calm"}} — the phrases in the order
you say them, and the tone to say them in."""


ANALYSIS_PROMPT = """You are the co-commentator of this football match (a former player, alongside the main commentator),
speaking live on air in {language}. Below are the last {span:.0f} seconds of broadcast pictures, in order (after a
goal or a big moment these are usually its replays); tracked players carry a label (P1, P2, ...) and the text before
the pictures says who each one is, as far as is known. Give your view on what we have just seen: why it happened,
who did well or badly, how the teams are set up — one or two short, conversational sentences, at most {max_words}
words, the way a pundit talks ("That's lovely movement from Hazard, he's pulled the full-back all the way out.").
- Say only what the pictures show and the facts give: no background of your own about players, clubs or seasons;
  background given in the facts (the line-ups, a player's past) is pre-match knowledge: draw on it when it explains
  what we saw.
- Name a player only when the text identifies him; otherwise say the team. Never say shirt numbers or the labels.
- Don't call new goals, cards or fouls; don't repeat the recent lines or start the same way as them.

Recent lines on air:
{history}

Facts:
{facts}

Answer with JSON only: {{"line": "...", "tone": "excited" | "lively" | "calm"}} — the tone to say it in."""


class Commentator:
    def __init__(self, config: Mapping[str, Any], generate: Callable[[Sequence[str | bytes]], Any]):
        # `config` is the `commentary` section; `generate(parts)` is one JSON-answer model call
        # returning an object with .text, .latency and .call_id (matchlive.common.api.APIClient).
        self.cfg = config
        self.generate = generate

    def line(self, item: Item, history: Sequence[str], frames: Any = None
             ) -> tuple[str, str, float, str, list[str] | None]:
        """(the line, the tone to say it in, the call's latency in s, its call id, its phrases). `frames`
        (a play or co-commentator line) are the pictures on air: an object with .text (who is who)
        and .images (JPEGs). A play line comes as a burst of short phrases, said one after another:
        the line is the phrases joined; for other lines the phrases are None."""
        recent = "\n".join(f"- {h}" for h in history[-self.cfg["history"]:]) or "(none yet)"
        facts = "\n".join(f"- {f}" for f in item.facts)
        if item.kind == "play" and frames is not None:
            prompt = PLAY_PROMPT.format(language=self.cfg["language"], span=self.cfg["play_span"],
                                        max_words=self.cfg["max_words"], play_words=self.cfg["play_words"],
                                        history=recent, facts=facts)
            result = self.generate([prompt, frames.text, *frames.images])
        elif item.kind == "analysis" and frames is not None:
            prompt = ANALYSIS_PROMPT.format(language=self.cfg["language"], span=self.cfg["analysis_span"],
                                            max_words=self.cfg["max_words"], history=recent, facts=facts)
            result = self.generate([prompt, frames.text, *frames.images])
        else:
            prompt = PROMPT.format(max_words=self.cfg["max_words"], language=self.cfg["language"], rule=item.rule,
                                   kind=item.kind, facts=facts, history=recent)
            result = self.generate([prompt])
        answer = parse_json(result.text)
        text, phrases = _said(answer, result.text)
        latency = result.latency
        if BACKGROUND.search(text) and "background" not in facts.lower():
            # Background of the model's own (a record, a career, a season): it may be wrong, or give
            # away what happens later. Asked once more, told what to leave out.
            again = self.generate([*([prompt] if item.kind not in ("play", "analysis") or frames is None
                                     else [prompt, frames.text, *frames.images]),
                                   f'Your line was: "{text}". It uses background that neither the pictures nor the facts give '
                                   "(records, careers, seasons, clubs' history). Say it again without it."])
            latency += again.latency
            answer = parse_json(again.text)
            text, phrases = _said(answer, text)
        tone = answer.get("tone") if answer.get("tone") in ("excited", "lively", "calm") else "lively"
        return text, tone, latency, result.call_id, phrases


def _said(answer: Mapping[str, Any], fallback: str) -> tuple[str, list[str] | None]:
    """(the line, its phrases): a burst of phrases joined, or the one line."""
    phrases = answer.get("phrases")
    if isinstance(phrases, list):
        phrases = [str(p).strip() for p in phrases if str(p).strip()]
        if phrases:
            return " ".join(phrases), phrases
    return str(answer.get("line") or fallback).strip(), None


# Phrases that only background knowledge could supply: a player's record or career, a club's season or history.
BACKGROUND = re.compile(r"\b(first|second|third|\d+(st|nd|rd|th))\b[^.!?]{0,30}\b(goal|appearance|start)s?\b[^.!?]{0,30}"
                        r"\b(for|in|since|of)\b|\bfor the club\b|\bthis season\b|\blast season\b|\bcareer\b|"
                        r"\bsince (he|joining|arriving|signing)\b|\bin (his|their) (time|history)\b", re.I)


def speaking_time(text: str, words_per_second: float) -> float:
    """How long a line takes to say, in s, when there is no audio to measure."""
    return max(1.5, len(text.split()) / words_per_second)
