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

The policy is plain rules over memory — no model call. For a stretch it gives the generator what
it must say, each item at the moment it belongs to:
    correction    something already said no longer holds: an event aired as fact (or an aired
                  goal, card, substitution, penalty) has been retracted, or a player named on air
                  is no longer the one memory holds; the reason is what memory says overturned it
                  (the arbiter's verdict). An entry a verdict replaced with a fuller one (the same
                  event, with who took part) is not retracted as far as the viewer is concerned:
                  what was said of it carries over to the new entry;
    confirmation  a goal, card, substitution or penalty aired while pending is now confirmed, or
                  its player has since been identified, at most `confirm_within` s after it happened
                  (later, the play has moved on and it goes unsaid);
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
                  it is most often an event already called, or none).
How the match stands is no item of its own: every stretch comes with the match as it stands when
the stretch begins (`where`) — the score and the match record (goals, cards, substitutions; one
aired while pending is listed as not confirmed yet, and not counted in the score) — and the
generator says it when the play allows. One state, so the score in one place never disagrees with
another (a goal in the stretch gives the score after it, by that name).
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
may use. No rule picks it: the knowledge base goes with every stretch, each fact with an id
(`knowledge`), and the generator chooses which fact fits a lull, if any, and marks the phrase that
tells it; a fact once told, or chosen by a stretch still being written, leaves the list, so none is
told twice, and the list says when the last one was told. The starting line-ups go with
it (`lineups`): each starter's line and side in the formation, so the generator knows whose wing
is whose.

The generator makes one cached model call per stretch from the pictures, the material and the
knowledge base alone, and may not add names or facts: a phrase naming a roster player whom neither
the pictures' who-is-who, the material nor the fact it tells names, or saying a track label, is
dropped. It holds no memory; the runtime lends the policy a memory view
each time (DESIGN.md §8).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Mapping, Sequence

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
    kind: str                               # correction | confirmation | opener | news
    facts: list[str]
    rule: str                               # how to word it (DESIGN.md §6.9)
    entry: str | None = None
    status: str | None = None               # the entry's status now
    named: dict[str, str] = field(default_factory=dict)   # role -> player the line may name
    t_event: Time | None = None             # when the event happened (best estimate)
    event_type: str | None = None           # the event's type, for a line about an event
    at: Time | None = None                  # the video time to say it at


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
        self.halves_opened: set[int] = set()
        self.last_big: Time | None = None        # when the last goal, card, penalty or change aired happened
        self.bg = prematch.get("background") or {}   # pre-match background (DESIGN.md §6.8)
        # The whole knowledge base for the generator to pick from, each fact with an id ("K12") and
        # the team or player it is about in front: the sides' managers, captains and facts, then
        # every player's (bench included), home side first.
        teams = prematch["teams"]
        facts = []
        for side in ("home", "away"):
            bg = self.bg.get("teams", {}).get(side, {})
            facts += [f"{teams[side]}: {k} {bg[k]}." for k in ("manager", "captain") if bg.get(k)]
            facts += [f"{teams[side]}: {f}" for f in bg.get("facts", [])]
        for side in ("home", "away"):
            facts += [f"{p['short_name']} ({teams[side]}): {f}" for pid, fs in self.bg.get("players", {}).items()
                      if (p := self.players.get(pid)) and p["team_side"] == side for f in fs]
        self.knowledge: dict[str, str] = {f"K{i + 1}": f for i, f in enumerate(facts)}
        self.facts_told: dict[str, Time] = {}    # fact id -> when it went on air
        # The starting line-ups, each starter by line and side ("Azpilicueta (defender, left)").
        forms = self.bg.get("formations", {})
        self.lineups = "\n".join(
            f"- {teams[side]}" + (f" ({forms[side]})" if forms.get(side) else "") + ": " + ", ".join(
                f"{p['short_name']} ({p['position']}" + (f", {p['side']})" if p.get("side") else ")")
                for p in sorted((p for p in prematch["players"] if p["team_side"] == side and p["starting"]),
                                key=lambda p: p.get("slot") or 0))
            for side in ("home", "away"))

    # ---- deciding ----

    def material(self, view: View, segment: tuple[int, float, float], held: Collection[str]) -> list[Item]:
        """What the commentary over the stretch `segment` = (half, start, end) of video must say,
        each item at the moment it belongs to. `view` is the match as known now: the
        output runs `output_delay` s behind the live feed, so memory already knows about the
        stretch, with that much hindsight. Nothing that happens after the stretch is said.
        `held` are the entries given to stretches still being written: not news again, and an
        event that only echoes one of them is no news either."""
        half, start, end = segment
        mem = view.mem
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
            # A name said on air is corrected once memory holds a different player, or none, for the
            # confirmed event; while the event is merely disputed (pending, so no name may be said)
            # the line waits.
            changed = {role: pid for role, pid in a.named.items()
                       if named.get(role) != pid and (r.status == "confirmed" or named.get(role) is not None)}
            if changed:
                items.append(self._correction(view, r, a, named, "player", (half, start)))
                continue
            new_names = {role: pid for role, pid in named.items() if role not in a.named}
            # Confirmed long after it happened, it goes unsaid: the play has moved on, and a line about
            # a moment minutes back only puzzles the viewer (the match record still lists it).
            h, second, _ = self._when(view, r)
            if ty in confirm and ((a.status == "pending" and r.status == "confirmed") or new_names) \
                    and h == half and start - second <= self.cfg["confirm_within"]:
                items.append(self._event_item(view, r, "confirmation", named, (half, start), a))

        # 3. The first stretch of a half opens it.
        if half not in self.halves_opened:
            items.append(self._opener(view, (half, start)))

        # 4. News: every unaired event the viewer sees by the end of the stretch (and saw at most
        # `recent` s before it), the most important first when one echoes another.
        fresh = [r for r in records.values() if r.entry_id not in self.aired and r.entry_id not in held
                 and r.status != "retracted" and r.event.type in announce and self._seen(view, r, (half, end))
                 and self._ready(view, r)]
        chosen: list[EventRecord] = []
        for r in sorted(fresh, key=lambda r: (announce.index(r.event.type), self._when(view, r)[1])):
            if not self._echoes(r, records, chosen + [records[eid] for eid in held if eid in records]):
                chosen.append(r)
        for r in sorted(chosen, key=lambda r: self._when(view, r)[1]):
            # Said at the moment it happens; one the viewer saw before the stretch, first thing.
            items.append(self._event_item(view, r, "news", self._named(r), (half, max(start, self._when(view, r)[1]))))

        return items

    def where(self, view: View, segment: tuple[int, float, float]) -> str:
        """The match as it stands when the stretch begins, for the generator: the time, which way
        each team attacks this half, the score and, with `story`, the match record — what the
        viewer has seen before the stretch, as memory knows it now: its goals, cards and
        substitutions, each placed at its sighting, as the news is. "confirmed" lists the counted
        entries, and those aired while pending as not confirmed yet (the viewer has heard of them;
        the score does not count them) until `confirm_within` s have passed without a
        confirmation; "all" every pending entry too. No counts of shots, corners or fouls: memory
        confirms only some of them, so a count is short of the true one (results §41)."""
        half, start, _ = segment
        teams = self.prematch["teams"]
        records = []
        for r in view.events():
            h, second, _ = self._when(view, r)
            if r.event.type not in ("goal", "yellow_card", "red_card", "substitution", "penalty") or (h, second) >= (half, start):
                continue
            if self._counted(r):
                records.append((r, h, second, True))
            elif r.status == "pending" and (self.cfg["story"] == "all" or r.entry_id in self.aired) \
                    and h == half and start - second <= self.cfg["confirm_within"]:
                records.append((r, h, second, False))
        score = self._goals(view, (half, start))
        out = (f"Time: {self._clock(half, start)}.\n{self._ends(half)}\n"
               f"Score as these seconds begin: {teams['home']} {score['home']}, {teams['away']} {score['away']}"
               + "".join(f" (not counted yet: {self._phrase(r.event)}, not confirmed)"
                         for r, _, _, counted in records if not counted and r.event.type == "goal") + ".")
        if self.cfg["story"] == "off":
            return out
        lines = []
        for r, h, second, counted in records:
            who = {part.role: pid for part, pid in zip(r.event.participants, r.players) if pid}
            if not counted:
                what = f"{self._phrase(r.event)}, not confirmed yet"
            elif r.event.type == "substitution" and "in" in who and "out" in who:
                what = f"{teams[self.players[who['in']]['team_side']]} substitution, {self.players[who['in']]['short_name']} " \
                       f"on for {self.players[who['out']]['short_name']}"
            elif "player" in who:
                what = f"{EVENT_WORDS[r.event.type]}, {self._player(who['player'], brief=True)}"
            else:
                what = self._phrase(r.event)
            lines.append(f"- {self._clock(h, second)}: {what}")
        return out + "\nThe match record so far:\n" + ("\n".join(lines) or "- no goals, cards or substitutions yet")

    def _counted(self, r: EventRecord) -> bool:
        """Whether an event counts in the score and the record: confirmed, or aired as confirmed and
        not retracted since — a dispute opened later (a replay taken for a new goal) leaves it
        pending in memory, but does not take back what the viewer was told."""
        a = self.aired.get(r.entry_id)
        return r.status == "confirmed" or (r.status == "pending" and a is not None and a.status == "confirmed")

    def _goals(self, view: View, before: Time) -> dict[str, int]:
        """The score from the counted goals sighted before `before`."""
        score = {"home": 0, "away": 0}
        for r in view.events(type="goal"):
            if r.event.team in score and self._counted(r) and self._when(view, r)[:2] < before:
                score[r.event.team] += 1
        return score

    def _ends(self, half: int) -> str:
        """Which way each team attacks this half, in the main camera's wide picture, and so where its
        left and right wings are: the camera stands on one touchline, so a team attacking to the
        right of the picture has its left wing on the far touchline."""
        teams, sides = self.prematch["teams"], self.prematch["sides"][half]
        out = []
        for side in ("home", "away"):
            attack = {"left": "right", "right": "left"}[sides[side]]
            near, far = ("left", "right") if attack == "left" else ("right", "left")
            out.append(f"{teams[side]} attack towards the goal on the {attack} of the main camera's wide picture: "
                       f"their {far} wing is the far touchline, their {near} wing the near one.")
        return "This half, " + " ".join(out)

    def said(self, item: Item, text: str, at: Time) -> None:
        """Record that `item` went on air as `text` (its phrases), from video time `at`."""
        if item.kind == "opener":
            self.halves_opened.add(at[0])
        if item.entry is not None:
            self.aired[item.entry] = Aired(item.status, dict(item.named), text)
        if item.kind in ("news", "confirmation") and item.event_type in self.cfg["confirm"]:
            self.last_big = item.t_event

    def told(self, fact: str, at: Time) -> None:
        """Record that the pre-match fact `fact` (its id) went on air at video time `at`."""
        self.facts_told.setdefault(fact, at)

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
        # News is the play on screen: no minute, and the score only with a goal. News first offered
        # after its moment (late, or offered again) is said as just past, not called again.
        past = f", {at[1] - second:.0f} s before these pictures" if kind == "news" and at[1] - second > self.cfg["late"] else ""
        facts = [f"What: {self._phrase(e)}{past}" + (" — being shown again in a replay" if replay else "") + "."]
        for role, pid in named.items():
            facts.append(f"{ROLE_WORDS[role].capitalize()}: {self._player(pid)}")
        if r.status == "confirmed":
            if e.type == "goal":                  # the score counts confirmed goals only, up to this one
                score = self._goals(view, (half, second))
                if e.team in score:
                    score[e.team] += 1
                facts.append(self._score(score, "after this goal"))
            rule = ("Say it as just past (\"... there\"), not as if it were happening now" if past else "Call it as it happens") \
                + (", naming the player(s) given." if named else "; you don't know who it was, so name nobody.")
        elif e.type in self.cfg["confirm"]:
            facts.append("Not certain yet.")
            rule = "You are not sure yet: hedge it (\"looks like...\", \"is that...?\")" + (" as just past" if past else "") \
                + " and name nobody, only the team."
        else:
            # A pending minor event is simply said without a player (DESIGN.md §6.9: the team only).
            rule = "Name nobody, only the team" + (", and say it as just past." if past else ".")
        if kind == "confirmation":
            # Confirmed after it happened, with the play moved on: said as news about that earlier
            # moment, not called again as if it were happening now.
            facts[0] = f"What: {self._phrase(e)}, earlier — {at[1] - second:.0f} s ago, not in these pictures."
            facts.append(f"What you said before: \"{before.text}\"")
            rule = ("The play has moved on since: in a pause in this stretch, tell the viewer calmly what is now "
                    "certain about that earlier moment, making clear it is the one from before (\"That goal a "
                    "minute ago is given to ...\", \"And that change is confirmed: ...\")"
                    + (", naming the player(s) given" if named else "") + ". Not as if it were happening now. If these seconds have no pause (the ball "
                    "in play, an attack on), leave it out: it comes again with the next stretch.")
        return Item(kind, facts, rule, r.entry_id, r.status, named, (half, second), e.type, at=at)

    def _correction(self, view: View, r: EventRecord, before: Aired, named: dict[str, str], what: str,
                    at: Time) -> Item:
        e = r.event
        half, second, _ = self._when(view, r)
        facts = [f"What you said before: \"{before.text}\"", f"About: {self._phrase(e)}, in {self._clock(half, second)}."]
        reason = self._reason(view, r.entry_id)
        if what == "retracted":
            facts.append(f"What turned out: {reason}")
            rule = ("Put it right the way a commentator does on air (\"No, ...\", \"Actually, ...\"), briefly, "
                    "with the reason in plain words.")
        else:
            for role, pid in before.named.items():
                if named.get(role) != pid:
                    facts.append(f"You named {self._player(pid, brief=True)} as the {ROLE_WORDS[role]}; "
                                 + (f"it was actually {self._player(named[role], brief=True)}." if named.get(role)
                                    else "it is no longer known who it was."))
            facts.append(f"What turned out: {reason}")
            rule = "Put the name right the way a commentator does on air, briefly."
        if e.type == "goal":
            facts.append(self._score(self._goals(view, (half, second)), "without it"))
        return Item("correction", facts, rule, r.entry_id, r.status, named, (half, second), at=at)

    def _opener(self, view: View, at: Time) -> Item:
        half, teams, kits = at[0], self.prematch["teams"], self.prematch["jersey_colors"]
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
        else:
            facts.append(self._score(self._goals(view, at), "at half-time"))
        return Item("opener", facts, "Welcome the viewers and open the half in two or three short spoken sentences"
                                     + (" (where we are, who manages the sides and how they come into it)" if half == 1 else "")
                                     + ", then follow the play.", at=at)

    # ---- helpers ----

    def _when(self, view: View, r: EventRecord) -> tuple[int, float, bool]:
        """(half, second, replay only): the median live sighting, else the middle of the interval.
        Only sightings of this type count: a foul's paired free kick stands behind it too."""
        mem = view.mem
        seen = [c for _, j in mem.justifications[r.entry_id] if j.obs_id in self.sources and mem.holds(j)
                and (c := self.sources[j.obs_id]["candidate"])["type"] == r.event.type]
        live = sorted(c["second"] for c in seen if not c["replay"])
        (half, lo), (_, hi) = r.event.t_interval
        if live:
            return half, live[len(live) // 2], False
        return half, (lo + hi) / 2, bool(seen)

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

    def _score(self, score: Mapping[str, int], when: str) -> str:
        """"Score after this goal: Chelsea 0, Liverpool 2." — always saying which moment it is the
        score of, so it is never taken for the score at another one."""
        teams = self.prematch["teams"]
        return f"Score {when}: {teams['home']} {score['home']}, {teams['away']} {score['away']}."

    def _phrase(self, e) -> str:
        """"a corner for Chelsea"; "a corner" when the side is unknown."""
        if e.team is None:
            return f"a {EVENT_WORDS[e.type]}"
        return WITH_TEAM[e.type].format(team=self.prematch["teams"][e.team])

    def _player(self, pid: str, brief: bool = False) -> str:
        """A player as a commentator names them — by the name fans use — with what the line may
        draw on (DESIGN.md §6.9): full name, side and position. No count of their events in the
        match: memory confirms only some of them (results §41)."""
        p = self.players[pid]
        team = self.prematch["teams"][p["team_side"]]
        if brief:
            return f"{p['short_name']} ({team})"
        role = p["position"] or ("substitute" if not p["starting"] else "player")
        country = self.bg.get("nationality", {}).get(pid)
        return f"{p['short_name']} (full name {p['long_name']}; {team} {role}" + (f", from {country})" if country else ")")

    @staticmethod
    def _clock(half: int, second: float) -> str:
        """A match time as the generator reads every time: "1H 35:30.0" (the half and its clock,
        as the frames and the phrases are stamped)."""
        return f"{half}H {clock(second)}"


PROMPT = """You are the TV commentator of this football match, speaking live on air in {language}. At the end are the next
{span:.0f} seconds of the broadcast pictures, one frame every {step:.1f} s, each with its time. Tracked players carry a
label (P1, P2, ...); the text before the frames says who each one is, as far as is known, who has the ball at his
feet in each frame, and where the camera cuts. Write what you say over these seconds as a timed script: each phrase starts at
the frame where you say it.

Talk the way a live TV commentator does:
- Follow the play, not every touch: the viewer sees the passing. Name a player when what he does matters (a run at
  the defence, a cross, a shot, a tackle, a pass that opens the play), not each one who touches the ball, and react
  as it happens ("Oh, good save!"). Between those moments, talk about the play as a commentator and his
  co-commentator do: which side is on top, where the space is, what a team keeps trying, what the score asks of
  them, drawing on what you have seen in these minutes, concretely ("Third time Willian has been crowded out down
  that side"), not a stock line you could say at any moment ("Liverpool comfortable here"). Let a thought run on over
  two or three phrases ("Willian to the byline," "looks for the cross," "and Milner gets there first."), and vary how
  phrases begin rather than opening each with a name. Most phrases are five to ten words. Say how the match stands
  (the score, who is on top, what a side needs; from the score and the match record below) at most once a minute,
  in a lull: after that, something new or nothing. When
  you can't tell who has the ball, say the team ("Chelsea work it wide"), not a shirt colour or "one of theirs".
- Call what any viewer can see, at the frame where it happens: a shot (at the strike, not after it), a save, a
  block, a cross, a header, a tackle, a clearance. Don't call goals, fouls, free kicks, corners, offsides, cards,
  penalties or substitutions yourself: those are the referee's and the match record's, and come in the material
  below with how sure they are — say each the way its rule says. A replay (a cut to a slower or closer view of a moment
  already seen) is a replay ("Another look at it..."), not a new chance.
- Give each phrase the energy the play has at that moment: "calm" (a lull: the ball out, a replay, a stoppage, a
  remark), "lively" (the ball moving), "rising" (an attack nearing the box), "excited" (a shot, a save, a goal, a big
  chance). The voice follows it.
- Time it: a phrase takes about {base:.1f} s plus {word:.1f} s a word to say, and the next one starts only after it ends
  (a phrase of four words fills about {four:.0f} frames). These {span:.0f} seconds hold about {phrases:.0f} phrases and {room:.0f}
  words in all, and most stretches need fewer: a phrase that does not fit is not said. Say a few full phrases rather
  than many short ones. Leave the pictures quiet where nothing happens, but don't go silent through an attack.
- Left and right are a team's own, as it attacks (its left wing, its right-back), never the screen's: take them
  from which way each team attacks this half (given with the time below) and the line-ups, and only in the main
  camera's wide view; a close-up or a replay may be shot from another angle. When unsure, say no side.
- Name a player only when the text with the pictures identifies him (a goalkeeper it marks is his side's keeper in
  the line-ups), the material or the match record names him, or the fact you tell is about him (being in the
  line-ups is not enough); otherwise say the team. Never say shirt numbers or the labels.
  Say what the pictures, the material and what you know before the match (below) give: what you know before the
  match is the knowledge of players, clubs and seasons to trust and use. At most {max_words} words a phrase.
- A lull (the ball out, a replay, a close-up), and only a lull, is where a fact you know before the match goes:
  one that fits the moment (the player on the ball or just substituted, the side pressing, the score), told the
  way a commentator tells a story, not as a fact card of bare dates and fees, and mark the phrase with its id. Not
  every lull: one fact every few minutes is plenty (when the last one was told is given with them).
- The match record lists the goals, cards and substitutions so far. Use it to understand the game and for your
  remarks. The score is the one given here, never the scoreboard in the pictures: it changes only some seconds
  after a goal. You have no counts of shots, corners or fouls: don't give any.
- Carry on from what was just said and from the play just before your stretch: don't say again what was said or
  given to say there, don't start the same way, and pick the play up where it is. Your first frames are often the
  moment your last phrases described ("Willian on the byline"): go on to what happens next ("...and the cross is
  cut out"), never describe it again.

The starting line-ups, each player by his line in the formation and his side of it:
{lineups}

What you know before the match, each fact with its id (the ones already told are left out{last}):
{knowledge}

{where}

Your recent phrases, each with when it began:
{history}

Just before your stretch, still being said (its words are not known yet):
{before}

Material you must say, each at about the frame given (put it where it happens in the pictures):
{must}

The pictures:"""

ANSWER = """Answer with JSON only: {"script": [{"frame": 3, "say": "...", "energy": "calm" | "lively" | "rising" | "excited",
"material": "M1" | null, "fact": "K1" | null}, ...]} — the phrases in order; "material" names the item a phrase
says, "fact" the fact from what you know before the match it tells, if any."""


@dataclass
class Phrase:
    """One phrase of a script."""
    at: float                               # the video second of the frame it is said at
    text: str
    energy: str                             # one of ENERGIES
    item: Item | None                       # the material it says, if any
    fact: str | None = None                 # the id of the pre-match fact it tells, if any


class Commentator:
    def __init__(self, config: Mapping[str, Any], generate: Callable[[Sequence[str | bytes]], Any],
                 players: Sequence[Mapping[str, Any]], knowledge: Mapping[str, str], lineups: str):
        # `config` is the `commentary` section; `generate(parts)` is one JSON-answer model call
        # returning an object with .text, .latency and .call_id (matchlive.common.api.APIClient);
        # `players` is the roster (Match.prematch["players"]), whose names a phrase may say only
        # when the pictures, the material or the fact it tells give them; `knowledge` is the
        # pre-match knowledge base, fact id -> fact (CommentaryPolicy.knowledge), of which a
        # stretch is shown the facts not told yet; `lineups` the starting line-ups
        # (CommentaryPolicy.lineups).
        self.cfg = config
        self.generate = generate
        self.players = players
        self.knowledge = knowledge
        self.lineups = lineups

    def script(self, half: int, items: Sequence[Item], frames: Any, where: str, history: Sequence[tuple[float, str]],
               before: Sequence[tuple[float, float, str, Sequence[Item]]],
               told: Mapping[str, Time]) -> tuple[list[Phrase], float, str]:
        """(the phrases in time order, the call's latency in s, its call id). `frames` are the
        stretch's pictures of `half`: an object with .text (who is who), .images (JPEGs) and .seconds (the
        video second of each); `where` is the time and score as the stretch begins; `history` is
        what was said, as (video second it began, text); `before` are the stretches just before
        this one whose scripts are still being written, as (start, end, their pictures' text,
        their material) — a script takes longer than a stretch lasts, so without them a stretch
        would not know what the seconds before it are about; `told` are the pre-match facts already
        told (or given to a stretch still being written), fact id -> when: left out of the
        knowledge base it is shown, with when the last one was told, so it can space them."""
        seconds = frames.seconds
        label = {f"M{i + 1}": item for i, item in enumerate(items)}

        def frame(t):                       # the frame nearest a video time (1-based)
            return min(range(len(seconds)), key=lambda k: abs(seconds[k] - t[1])) + 1

        # A confirmation is about a moment already past: it has no frame of its own, and waits for a pause.
        must = "\n".join(f"- {k}, " + ("at a pause in these seconds" if it.kind == "confirmation" else f"at frame {frame(it.at)}")
                         + f" ({it.kind}): {' '.join(it.facts)} How: {it.rule}"
                         for k, it in label.items()) or "(none)"
        knowledge = {k: f for k, f in self.knowledge.items() if k not in told}
        last = max(told.values(), default=None)
        at = CommentaryPolicy._clock
        recent = "\n".join(f"- {at(half, t)} {text}" for t, text in history[-self.cfg["history"]:]) or "(none yet)"
        earlier = "\n".join(f"- {at(half, a)}-{clock(b)}: on the ball {_on_ball(text) or '(not seen)'}"
                            + (f"; given to say: {' '.join(f for it in its for f in it.facts)}" if its else "")
                            for a, b, text, its in before) or "(nothing)"
        step = seconds[1] - seconds[0] if len(seconds) > 1 else 0.5
        # How much speech the stretch holds: `room` of its time, in phrases of the broadcast's
        # length, each taking `phrase_time` (a + b * words) plus the gap after it.
        (base, word), n = self.cfg["phrase_time"], self.cfg["phrase_words"]
        phrases = len(seconds) * step * self.cfg["room"] / (base + n * word + self.cfg["gap"])
        prompt = PROMPT.format(language=self.cfg["language"], span=len(seconds) * step, step=step,
                               base=base, word=word, four=(base + 4 * word + self.cfg["gap"]) / step,
                               phrases=phrases, room=phrases * n, max_words=self.cfg["max_words"],
                               lineups=self.lineups,
                               knowledge="\n".join(f"- {k} {f}" for k, f in knowledge.items()) or "(nothing)",
                               last=f"; the last one was told at {at(*last)}" if last else "; none yet",
                               where=where, history=recent, before=earlier, must=must)
        parts = [prompt, frames.text]
        for k, (s, jpg) in enumerate(zip(seconds, frames.images), start=1):
            parts += [f"Frame {k} ({at(half, s)})", jpg]
        parts.append(ANSWER)
        result = self.generate(parts)
        script = _script(parse_json(result.text), seconds, label, knowledge)
        # Names only from the pictures' who-is-who (a goalkeeper it marks is his side's starting
        # keeper), the material, the match record and the fact a phrase tells; never a track label ("P3").
        given = _named("\n".join([frames.text, where] + [f for it in items for f in it.facts]), self.players)
        given |= {p["hash"] for p in self.players if p["starting"] and p["position"] == "goalkeeper"
                  and re.search(rf"^P\d+: [^;\n]*\({p['team_side']}\), goalkeeper", frames.text, re.M)}
        kept = []
        for p in script:
            about = _named(self.knowledge[p.fact], self.players) if p.fact else set()
            if not _named(p.text, self.players) <= given | about or re.search(r"\bP\d+\b", p.text):
                print(f"{at(half, p.at)} names what it was not given: dropped {p.text!r}", flush=True)
                continue
            kept.append(p)
        return kept, result.latency, result.call_id


def _script(answer: Mapping[str, Any], seconds: Sequence[float], label: Mapping[str, Item],
            knowledge: Mapping[str, str]) -> list[Phrase]:
    """The model's script -> phrases in time order; a phrase with no text or an unknown frame is
    dropped, an unknown energy is "lively", an unknown material item or fact is none. An answer
    with no script raises (the stretch is lost, and logged so)."""
    out = []
    for p in answer["script"]:
        text, k = str(p.get("say") or "").strip(), p.get("frame")
        if not text or not isinstance(k, int) or not 1 <= k <= len(seconds):
            continue
        energy = p.get("energy") if p.get("energy") in ENERGIES else "lively"
        fact = p.get("fact") if p.get("fact") in knowledge else None
        out.append(Phrase(seconds[k - 1], text, energy, label.get(p.get("material")), fact))
    return sorted(out, key=lambda p: p.at)


def _on_ball(pictures: str) -> str:
    """Who had the ball over a stretch, from its pictures' text: the player with the ball at his
    feet frame by frame, by name where the who-is-who has one, else by team ("Milner, Liverpool,
    Costa"), repeats merged."""
    who = {}
    for line in pictures.splitlines():
        m = re.match(r"(P\d+): ([^;]+)", line)
        if m and "team unknown" not in m[2]:
            # "Liverpool (away), #7 Milner" -> "Milner"; "Chelsea (home)" -> "Chelsea"
            who[m[1]] = m[2].split("#", 1)[1].split(" ", 1)[1] if "#" in m[2] else m[2].split(" (")[0]
    ball = next((line for line in pictures.splitlines() if line.startswith("Ball at a player's feet")), "")
    seq = [who[p] for p in re.findall(r"\d+:(P\d+)", ball) if p in who]
    return ", ".join(n for i, n in enumerate(seq) if i == 0 or n != seq[i - 1])


def _named(text: str, players: Sequence[Mapping[str, Any]]) -> set[str]:
    """The roster players (their hashes) `text` names, by full or short name, accents and case aside
    ("Mané" is Mane)."""
    text = _fold(text)
    return {p["hash"] for p in players
            if any(re.search(r"\b" + re.escape(_fold(n)) + r"\b", text) for n in (p["long_name"], p["short_name"]))}


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).casefold()


def speaking_time(text: str, words_per_second: float) -> float:
    """How long a phrase takes to say, in s, when there is no audio to measure."""
    return max(0.6, len(text.split()) / words_per_second)
