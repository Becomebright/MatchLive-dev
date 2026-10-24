"""Live commentary (DESIGN.md §6.9): a policy that decides, from memory as known now, what the
commentary over the next stretch of video must say and may say, and a generator that writes that
stretch as a timed script of short spoken phrases over its pictures.

The output runs `output_delay` s behind the live feed, and the commentary is written `segment` s of
video at a time, shortly before that stretch goes on air (runtime.runner.run_commentary). By then
the feed has run past the end of the stretch, so the generator sees all of its pictures and writes
each phrase at the frame it is said at — the name as a player gets the ball, the shot as it is
struck — the way a live commentator talks over the play, not after it. Memory is known up to the
feed too, so it already knows about most of what happens in the stretch, with more than a minute of
hindsight. Nothing that happens after the stretch is said (no spoilers).

The policy is plain rules over memory — no model call. For a stretch it gives the generator:
  must say, each at the moment it belongs to:
    correction    something already said no longer holds: an event aired as fact (or an aired
                  goal, card, substitution, penalty) has been retracted, or a player named on air
                  is no longer the one memory holds; the reason is what memory says overturned it
                  (the arbiter's verdict). An entry a verdict replaced with a fuller one (the same
                  event, with who took part) is not retracted as far as the viewer is concerned:
                  what was said of it carries over to the new entry;
    confirmation  a goal, card, substitution or penalty aired while pending is now confirmed, or
                  its player has since been identified;
    opener        the first stretch of each half: the teams, the score, pre-match facts (the
                  venue, the managers, where each side stands going in);
    news          every event memory holds that the viewer sees in the stretch — or saw in the
                  last `recent` s and has not been told about — except a minor one that only
                  echoes an event aired within `echo` s or another one of the stretch (the same
                  kind by the same side, or the other half of one incident: a goal and the shot it
                  came from, a foul and its free kick or card), a disputed one (a member of an open
                  case: it waits for the verdict), and a pending one before `hold` s (a pending
                  minor one not at all, unless `pending_minor`: half of them are false; nor a
                  pending one seen only in replays: a replay re-shows what the viewer has seen, so
                  it is most often an event already called, or none);
  may say, in a lull (the ball out of play, a replay, a close-up):
    recap         every `context_every` s: one angle on how the match stands (score, counts, who
                  has the pressure, bookings, the busiest player) that has something new — or,
                  when the match has nothing new, a pre-match fact about a side;
    background    a pre-match fact on a player the pictures identify (one every `background_every`
                  s at most).
A must-say item the script leaves out (or that cannot go on air in time) is not marked as said, so
it is offered again with the next stretch while it is still news.

The pictures themselves are the generator's to describe: the ball's progress, and the on-ball
actions any viewer sees as they happen (a shot, a save, a block, a cross, a tackle). The decisions
that stay on the match's record — a goal, a card, a penalty, a substitution, a foul, a free kick, a
corner, an offside — come only from memory, worded by each event's status (DESIGN.md §6.9): a
confirmed event names its players (those memory has identified); a pending one names only the team
— and a pending goal, card, penalty or substitution is also hedged ("looks like ..."). What was said
about each entry is kept, so a later change of status or player is noticed and aired.
Pre-match background (Match.prematch["background"], DESIGN.md §6.8) is the only background a phrase
may use: a player's facts come with the first events that name him, or over the pictures that
show him, and each fact is offered once.

The generator makes one cached model call per stretch from the pictures and the material alone,
and may not add names or facts. It holds no memory; the runtime lends the policy a memory view
each time (DESIGN.md §8).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from matchlive.events.analyzer import clock, parse_json
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
# The voice follows the play: each phrase is said at one of these energies (commentary.voice.tones).
ENERGIES = ("calm", "lively", "rising", "excited")


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
    """What was said about one event entry, as of the last phrases about it."""
    status: str                             # the entry's status when it was aired
    named: dict[str, str]                   # role -> player named aloud
    text: str                               # what was said


@dataclass
class Item:
    """One piece of material for a stretch, as the policy chose it."""
    kind: str                               # correction | confirmation | opener | news | recap | background
    facts: list[str]
    rule: str                               # how to word it (DESIGN.md §6.9)
    entry: str | None = None
    status: str | None = None               # the entry's status now
    named: dict[str, str] = field(default_factory=dict)   # role -> player the line may name
    t_event: Time | None = None             # when the event happened (best estimate)
    event_type: str | None = None           # the event's type, for a line about an event
    angle: tuple | None = None              # (angle, value) a recap or background line is about
    at: Time | None = None                  # the video time to say it at
    must: bool = True                       # must be said in its stretch (else: only in a lull)


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
        self.context_said: dict[str, tuple] = {}  # context angle -> (value, video second) last said
        self.last_context: Time | None = None    # video time of the last line on how the match stands
        self.halves_opened: set[int] = set()
        self.last_big: Time | None = None        # when the last goal, card, penalty or change aired happened
        self.bg = prematch.get("background") or {}   # pre-match background (DESIGN.md §6.8)
        self.offered: set[str] = set()           # background facts already given to a line (each once)
        self.last_bg: Time | None = None         # video time a fact was last said over the pictures

    # ---- deciding ----

    def material(self, view: View, segment: tuple[int, float, float]) -> list[Item]:
        """What the commentary over the stretch `segment` = (half, start, end) of video must say and
        may say, each item at the moment it belongs to. `view` is the match as known now: the
        output runs `output_delay` s behind the live feed, so memory already knows about the
        stretch, with that much hindsight. Nothing that happens after the stretch is said."""
        half, start, end = segment
        mem = view.mem
        opening = mem.at((half, start), known_by=view.time)     # the match as the stretch begins
        records = {r.entry_id: r for r in view.events()}
        confirm, announce = self.cfg["confirm"], self.cfg["announce"]
        items: list[Item] = []

        # An aired entry a verdict replaced with a fuller one: what was said carries over to the
        # new entry (which may add the names), and the old one needs no correction.
        for eid, a in list(self.aired.items()):
            r = records.get(eid)
            if a.status != "retracted" and r is not None and r.status == "retracted":
                new = replacement(mem, eid)
                if new is not None and new not in self.aired:
                    self.aired[new] = a
                    self.aired[eid] = Aired("retracted", {}, a.text)

        # 1-2. What was aired and has changed since: said first thing in the stretch.
        for eid, a in self.aired.items():
            r = records.get(eid)
            if r is None:
                continue
            named = self._named(r)
            ty = r.event.type
            if r.status == "retracted" and a.status != "retracted":
                if a.status == "confirmed" or ty in confirm:
                    items.append(self._correction(view, r, a, named, "retracted", (half, start)))
                continue
            # A name said on air is corrected only once memory holds a different player; while the
            # event is merely disputed (pending, so no name may be said) the line waits.
            changed = {role: pid for role, pid in a.named.items() if named.get(role) not in (None, pid)}
            if changed:
                items.append(self._correction(view, r, a, named, "player", (half, start)))
                continue
            new_names = {role: pid for role, pid in named.items() if role not in a.named}
            if ty in confirm and ((a.status == "pending" and r.status == "confirmed") or new_names):
                items.append(self._event_item(view, r, "confirmation", named, (half, start), a))

        # 3. The first stretch of a half opens it.
        if half not in self.halves_opened:
            items.append(self._opener(opening, (half, start)))

        # 4. News: every unaired event the viewer sees by the end of the stretch (and saw at most
        # `recent` s before it), the most important first when one echoes another.
        fresh = [r for r in records.values() if r.entry_id not in self.aired and r.status != "retracted"
                 and r.event.type in announce and self._seen(view, r, (half, end)) and self._ready(view, r)]
        chosen: list[EventRecord] = []
        for r in sorted(fresh, key=lambda r: (announce.index(r.event.type), self._when(view, r)[1])):
            if not self._echoes(r, records, chosen):
                chosen.append(r)
        for r in sorted(chosen, key=lambda r: self._when(view, r)[1]):
            # Said at the moment it happens; one the viewer saw before the stretch, first thing.
            items.append(self._event_item(view, r, "news", self._named(r), (half, max(start, self._when(view, r)[1]))))

        # 5. Now and then, how the match stands (in a lull).
        if self.last_context is None or self.last_context[0] != half \
                or start - self.last_context[1] >= self.cfg["context_every"]:
            item = self._recap(opening)
            if item is not None:
                items.append(item)
        return items

    def where(self, view: View, segment: tuple[int, float, float]) -> str:
        """The time and the score as the stretch begins, for the generator."""
        half, start, _ = segment
        return f"Time: {self._clock(half, start)}. {self._score(view.mem.at((half, start), known_by=view.time))}"

    def said(self, item: Item, text: str, at: Time) -> None:
        """Record that `item` went on air as `text` (its phrases), from video time `at`."""
        if item.kind == "opener":
            self.halves_opened.add(at[0])
        if item.kind in ("opener", "recap"):
            self.last_context = at
        if item.kind == "background":
            self.last_bg = at
        if item.angle is not None:
            self.context_said[item.angle[0]] = (item.angle[1], at[1])
            if item.angle[0].startswith("background:"):
                self.offered.add(item.angle[0].removeprefix("background:"))
        if item.entry is not None:
            self.aired[item.entry] = Aired(item.status, dict(item.named), text)
        if item.kind in ("news", "confirmation") and item.event_type in self.cfg["confirm"]:
            self.last_big = item.t_event

    def _seen(self, view: View, r: EventRecord, upto: Time) -> bool:
        """Whether the viewer has seen the event by `upto`, at most `recent` s before it (`recent_minor`
        for a minor event: later, it is stale)."""
        half, second, _ = self._when(view, r)
        if half != upto[0] or second > upto[1]:
            return False
        if r.event.type in self.cfg["confirm"]:
            return upto[1] - second <= self.cfg["recent"]
        # A minor event from before the last big one aired is overtaken by it (the free kick a goal
        # came from is not news once the goal has been called).
        if self.last_big is not None and (half, second) < self.last_big:
            return False
        return upto[1] - second <= self.cfg["recent_minor"]

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

    def _echoes(self, r: EventRecord, records: Mapping[str, EventRecord], chosen: Sequence[EventRecord]) -> bool:
        """A minor event within `echo` s of an aired one, or of one already chosen for this stretch,
        that it only repeats: the same kind by the same side (often one event reported twice), or
        the other half of one incident. A goal, card, penalty or substitution is always news."""
        e = r.event
        if e.type in self.cfg["confirm"]:
            return False
        for o in [records.get(eid) for eid in self.aired] + list(chosen):
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

    def _event_item(self, view: View, r: EventRecord, kind: str, named: dict[str, str], at: Time,
                    before: Aired | None = None) -> Item:
        e = r.event
        half, second, replay = self._when(view, r)
        # News is the play on screen: no minute, and the score only with a goal.
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
        return Item(kind, facts, rule, r.entry_id, r.status, named, (half, second), e.type, at=at)

    def _correction(self, view: View, r: EventRecord, before: Aired, named: dict[str, str], what: str,
                    at: Time) -> Item:
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
        return Item("correction", facts, rule, r.entry_id, r.status, named, (half, second), at=at)

    def _opener(self, view: View, at: Time) -> Item:
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
        return Item("opener", facts, "Welcome the viewers and open the half in two or three short spoken sentences"
                                     + (" (where we are, who manages the sides and how they come into it)" if half == 1 else "")
                                     + ", then follow the play.", at=at)

    def _recap(self, view: View) -> Item | None:
        """One angle on how the match stands at the start of the stretch (`view` is the match as of
        then): the score, a side's counts, who has had the recent pressure, a booking, the busiest
        player. An angle is offered only when there is something new in it since it was last said
        (the score or the pressure changed, a new booking, ...) or it has not been said for a while;
        the angles take turns. None when there is nothing new to say."""
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
        rule = ("Work this pre-match background into one natural spoken remark." if angle.startswith("background:")
                else "Work this into a natural remark on how the match stands; don't list numbers.")
        return Item("recap", [fact], rule, t_event=None, angle=(angle, value), at=view.time, must=False)

    def on_screen(self, pictures: str, at: Time) -> list[Item]:
        """Background on a player the pictures identify (in their who-is-who text) — the one on the
        ball most, if he has a fact not yet given — for a lull in the stretch: one fact, at most
        every `background_every` s."""
        if self.last_bg is not None and self.last_bg[0] == at[0] \
                and at[1] - self.last_bg[1] < self.cfg["background_every"]:
            return []
        # Who is who ("P3: Chelsea (home), #21 Matic"), and who is nearest the ball in each frame
        # ("Nearest to the ball (...): 1:P3 2:P3 3:- ..."): the player on the ball most, first.
        local = {}
        for line in pictures.splitlines():
            for p in self.prematch["players"]:
                if line.startswith("P") and f"({p['team_side']}), #{p['jersey_number']} {p['short_name']}" in line:
                    local[line.split(":")[0]] = p
        ball = next((line for line in pictures.splitlines() if line.startswith("Nearest to the ball")), "")
        carried = Counter(re.findall(r"\d+:(P\d+)", ball))
        shown = sorted(local, key=lambda pid: (-carried[pid], int(pid[1:])))
        for p in (local[pid] for pid in shown):
            for fact in self.bg.get("players", {}).get(p["hash"], []):
                if fact not in self.offered:
                    return [Item("background", [f"Background on {p['short_name']}: {fact}"],
                                 "Say it briefly in your own words while he is on screen, tied to what he is doing.",
                                 angle=(f"background:{fact}", None), at=at, must=False)]
        return []

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


PROMPT = """You are the TV commentator of this football match, speaking live on air in {language}. Below are the next
{span:.0f} seconds of the broadcast pictures, one frame every {step:.1f} s, each with its time. Tracked players carry a
label (P1, P2, ...); the text before the frames says who each one is, as far as is known, who is nearest the ball in
each frame, and where the camera cuts. Write what you say over these seconds as a timed script: each phrase starts at
the frame where you say it.

Talk the way a live TV commentator does:
- Follow the ball. Name a player as he gets it ("Kante."), say what he does with it ("Out to Willian."), react as it
  happens ("Oh, good save!"). Most phrases are one to six words, said as the play moves, not after it.
- Call what any viewer can see, at the frame where it happens: a shot (at the strike, not after it), a save, a
  block, a cross, a header, a tackle, a clearance. Don't call goals, fouls, free kicks, corners, offsides, cards,
  penalties or substitutions yourself: those are the referee's and the match record's, and come in the material
  below with how sure they are — say each the way its rule says. A replay is a replay ("Another look at it..."), not
  a new chance.
- Give each phrase the energy the play has at that moment: "calm" (a lull: the ball out, a replay, a stoppage, a
  remark), "lively" (the ball moving), "rising" (an attack nearing the box), "excited" (a shot, a save, a goal, a big
  chance). The voice follows it.
- Time it: a phrase takes about a third of a second a word to say, and the next one starts after it. Leave the
  pictures quiet where nothing happens, but don't go silent through an attack.
- Name a player only when the text identifies him or the material names him; otherwise say the team. Never say shirt
  numbers or the labels. Say only what the pictures and the material give: no background of your own about players,
  clubs or seasons (it may be wrong, or tell what happens later). At most {max_words} words a phrase.
- In a lull, and only then, you may use the optional material (how the match stands, pre-match background), briefly
  and in your own words.
- Carry on from your recent phrases: don't repeat them or start the same way.

{where}

Your recent phrases:
{history}

Material you must say, each at about the frame given (put it where it happens in the pictures):
{must}

Optional material, only for a lull:
{may}

Answer with JSON only: {{"script": [{{"frame": 3, "say": "...", "energy": "calm" | "lively" | "rising" | "excited",
"material": "M1" | null}}, ...]}} — the phrases in order; "material" names the item a phrase says, if any."""


@dataclass
class Phrase:
    """One phrase of a script."""
    at: float                               # the video second of the frame it is said at
    text: str
    energy: str                             # one of ENERGIES
    item: Item | None                       # the material it says, if any


class Commentator:
    def __init__(self, config: Mapping[str, Any], generate: Callable[[Sequence[str | bytes]], Any]):
        # `config` is the `commentary` section; `generate(parts)` is one JSON-answer model call
        # returning an object with .text, .latency and .call_id (matchlive.common.api.APIClient).
        self.cfg = config
        self.generate = generate

    def script(self, items: Sequence[Item], frames: Any, where: str, history: Sequence[str]
               ) -> tuple[list[Phrase], float, str]:
        """(the phrases in time order, the call's latency in s, its call id). `frames` are the
        stretch's pictures: an object with .text (who is who), .images (JPEGs) and .seconds (the
        video second of each); `where` is the time and score as the stretch begins."""
        seconds = frames.seconds
        label = {f"M{i + 1}": item for i, item in enumerate(items)}

        def frame(t):                       # the frame nearest a video time (1-based)
            return min(range(len(seconds)), key=lambda k: abs(seconds[k] - t[1])) + 1

        must = "\n".join(f"- {k}, at frame {frame(it.at)} ({it.kind}): {' '.join(it.facts)} How: {it.rule}"
                         for k, it in label.items() if it.must) or "(none)"
        may = "\n".join(f"- {k} ({it.kind}): {' '.join(it.facts)} How: {it.rule}"
                        for k, it in label.items() if not it.must) or "(none)"
        recent = "\n".join(f"- {h}" for h in history[-self.cfg["history"]:]) or "(none yet)"
        step = seconds[1] - seconds[0] if len(seconds) > 1 else 0.5
        prompt = PROMPT.format(language=self.cfg["language"], span=len(seconds) * step, step=step,
                               max_words=self.cfg["max_words"], where=where, history=recent, must=must, may=may)
        parts = [prompt, frames.text]
        for k, (s, jpg) in enumerate(zip(seconds, frames.images), start=1):
            parts += [f"Frame {k} ({clock(s)})", jpg]
        result = self.generate(parts)
        script = _script(parse_json(result.text), seconds, label)
        latency = result.latency
        said = " ".join(p.text for p in script)
        if BACKGROUND.search(said) and not any("background" in f.lower() for it in items for f in it.facts):
            # Background of the model's own (a record, a career, a season): it may be wrong, or give
            # away what happens later. Asked once more, told what to leave out.
            again = self.generate([*parts, f'Your script said: "{said}". It uses background that neither the pictures nor '
                                           "the material give (records, careers, seasons, clubs' history). Write it again "
                                           "without it."])
            latency += again.latency
            script = _script(parse_json(again.text), seconds, label) or script
        return script, latency, result.call_id


def _script(answer: Mapping[str, Any], seconds: Sequence[float], label: Mapping[str, Item]) -> list[Phrase]:
    """The model's script -> phrases in time order; a phrase with no text or an unknown frame is
    dropped, an unknown energy is "lively", an unknown material item is none."""
    out = []
    phrases = answer.get("script")
    for p in phrases if isinstance(phrases, list) else []:
        if not isinstance(p, dict):
            continue
        text, k = str(p.get("say") or "").strip(), p.get("frame")
        if not text or not isinstance(k, int) or not 1 <= k <= len(seconds):
            continue
        energy = p.get("energy") if p.get("energy") in ENERGIES else "lively"
        out.append(Phrase(seconds[k - 1], text, energy, label.get(p.get("material"))))
    return sorted(out, key=lambda p: p.at)


# Phrases that only background knowledge could supply: a player's record or career, a club's season or history.
BACKGROUND = re.compile(r"\b(first|second|third|\d+(st|nd|rd|th))\b[^.!?]{0,30}\b(goal|appearance|start)s?\b[^.!?]{0,30}"
                        r"\b(for|in|since|of)\b|\bfor the club\b|\bthis season\b|\blast season\b|\bcareer\b|"
                        r"\bsince (he|joining|arriving|signing)\b|\bin (his|their) (time|history)\b", re.I)


def speaking_time(text: str, words_per_second: float) -> float:
    """How long a phrase takes to say, in s, when there is no audio to measure."""
    return max(0.6, len(text.split()) / words_per_second)
