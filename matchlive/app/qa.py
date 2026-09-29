"""Streaming question answering (DESIGN.md §6.9): a question asked during the match, answered from
what the system knows at that moment.

Two modes (qa.mode, DESIGN.md §7):
  tools     the agent reads memory by status through the shared tools (plus the video up to now)
            in rounds of plain JSON, like the arbiter: {"requests": [...]} for more information,
            or {"answer": "B"}. Every round is one cached model call.
  dump_log  v2's baseline: the whole event log known so far goes to the model in one call.
The tools are lent by the runtime (runtime.qa_tools), bounded by the question time; this module
holds no memory, perception or video (DESIGN.md §8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from matchlive.arbiter.agent import Evidence
from matchlive.events.analyzer import parse_json

# What memory is and how the question's words map onto it — shared by both modes.
ABOUT_MEMORY = """About the match memory:
- An event reaches memory 30-90 s after it happens (the video is analysed in 30-s windows, then checked), so
  what is happening now or happened in the last minute or two is not in it yet.
- Each event is confirmed (two independent sightings agree, or an arbiter checked it) or pending (seen once so
  far, or disputed). Pending events are often real; do not ignore them, but a pending event right next to one
  of the same type and team may be the same thing reported twice (e.g. from a replay).
- The players on an event come from reading shirt numbers and are often unidentified.
- A goal also counts as a shot. Cards are yellow and red cards. A corner or free kick counts when it is taken.
- Times are the clock of each half: "half 2, 15:04" is 15 min 4 s into the second half. The Nth minute of a half
  runs from (N-1):00 to (N-1):59, so 15:04 of the second half is "4 seconds into the 16th minute of the second
  half". Stoppage time starts at 45:00 of a half: 45:45 of the second half is "45 seconds into the 1st minute of
  second-half stoppage time"."""

INSTRUCTIONS = """You answer questions about a football (soccer) match while it is being broadcast live. You see
the match through tools: the match memory (the events an automatic system has recognised so far) and the video up
to now. Nothing after now can be seen. Round {round} of at most {max_rounds}.

Tools (at most {max_requests} requests per round):
- get_state {{}}: the score, who is on the pitch and on the bench, substitutions and cards so far, each team's
  event counts.
- list_events {{"type": "shot", "team": "home", "player": "Costa", "half": 1, "start": 600, "end": 900}}: the events
  memory holds, oldest first; every filter is optional ("start"/"end" are seconds of that half; use
  {{"last_minutes": 10}} instead of half/start/end for the last N minutes up to now, across half-time).
- get_player {{"name": "Costa"}}: a player's number, team, position, where they are now, and their events.
- why {{"entry": "E12"}}: what an event rests on (which analyses saw it, arbiter verdicts).
- get_frames {{"half": 1, "start": 600, "end": 620}}: up to {max_frames} broadcast frames spread over a span before
  now; tracked players are labelled P1, P2, ... and the text says who each is, as far as shirt-number reading
  knows by now, and who is nearest the ball.
- get_event_evidence {{"entry": "E12"}}: the same, for the seconds around an event.
- get_current_view {{}}: the same, for the last {current_view} seconds up to now.

{about}
- For what memory does not hold yet, and for who did something when memory does not say, look at the video:
  get_current_view for now, get_event_evidence or get_frames for the past.

Answer with JSON only: {{"requests": [{{"tool": "get_state", "args": {{}}}}]}} to gather information, or
{{"answer": "B"}} with the letter of the option you choose.{last}"""

DUMP_INSTRUCTIONS = """You answer questions about a football (soccer) match while it is being broadcast live.
Below is everything an automatic system has recorded about the match so far: the squads and its event log.
Nothing after now is known.

{about}

Answer with JSON only: {{"answer": "B"}} with the letter of the option you choose."""


@dataclass
class Answer:
    choice: str | None              # the option letter, None if the model gave none in budget
    calls: int
    latency: float                  # seconds, summed over the calls
    tools: list[str] = field(default_factory=list)


def instructions(config: Mapping[str, Any], round: int) -> str:
    """INSTRUCTIONS for round `round` under the `qa` config section."""
    last = "\nThis is the last round: answer now." if round == config["max_rounds"] else ""
    return INSTRUCTIONS.format(round=round, max_rounds=config["max_rounds"], max_requests=config["max_requests"],
                               max_frames=config["max_frames"], current_view=config["current_view"],
                               about=ABOUT_MEMORY, last=last)


def question_text(question: str, options: Sequence[str], now_text: str, match_text: str) -> str:
    return f"{match_text}\nIt is now {now_text}.\n\nQuestion: {question}\nOptions:\n" + "\n".join(options)


class QAAgent:
    def __init__(self, config: Mapping[str, Any], generate: Callable[[Sequence[str | bytes]], Any]):
        # `config` is the `qa` section; `generate(parts)` is one JSON-answer model call returning
        # an object with .text and .latency (matchlive.common.api.APIClient).
        self.cfg = config
        self.generate = generate

    def answer(self, question: str, letters: Sequence[str],
               call_tool: Callable[[str, Mapping[str, Any]], Evidence]) -> Answer:
        """qa.mode tools: rounds of tool requests until an answer, at most max_rounds calls."""
        evidence: list[tuple[str, Evidence]] = []
        out = Answer(None, 0, 0.0)
        for r in range(1, self.cfg["max_rounds"] + 1):
            parts: list[str | bytes] = [instructions(self.cfg, r), question]
            for request, ev in evidence:
                parts += [f"[{request}] {ev.text}", *ev.images]
            result = self.generate(parts)
            out.calls, out.latency = out.calls + 1, out.latency + result.latency
            reply = parse_json(result.text)
            out.choice = _letter(reply.get("answer"), letters)
            if out.choice:
                return out
            requests = reply.get("requests")
            if r == self.cfg["max_rounds"] or not isinstance(requests, list) or not requests:
                return out
            for req in requests[: self.cfg["max_requests"]]:
                if not isinstance(req, dict):
                    continue
                tool, args = str(req.get("tool")).strip(), req.get("args") or {}
                out.tools.append(tool)
                evidence.append((f"{tool} {args}", call_tool(tool, args)))
        return out

    def answer_from_log(self, question: str, letters: Sequence[str], log: str) -> Answer:
        """qa.mode dump_log: one call over the whole log."""
        result = self.generate([DUMP_INSTRUCTIONS.format(about=ABOUT_MEMORY), log, question])
        return Answer(_letter(parse_json(result.text).get("answer"), letters), 1, result.latency)


def _letter(answer: Any, letters: Sequence[str]) -> str | None:
    # "B", or the option as written ("B. 1", "B)"): its leading letter.
    text = str(answer or "").strip()
    return text[0] if text and text[0] in letters and (len(text) == 1 or not text[1].isalpha()) else None
