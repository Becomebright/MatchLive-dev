"""Run the arbiter over a match with verdicts from the arbiter subagent, round by round.

    python scripts/run_arbiter.py --match "2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea"

Replays the cached event candidates through the consistency check and the arbiter. A case
whose verdict is already known (outputs/arbiter/<match>/<key>/verdict.json) gets it, applied behind the usual guardrails at the case's opening time +
arbiter.fixed_latency. A case without one is left open and written out for the subagent:
<key>/task.md (what to do), packet.txt, state.pkl (memory as of the case's opening, for the
tools), meta.json. The subagent then gathers evidence with scripts/arbiter_tool.py and saves
its verdict there. Verdicts change memory, so later cases can change or appear: run again
until no case is pending.

Case and entry ids are handed out in order, so one verdict that adds an entry renumbers every
later one. The key is therefore the hash of the packet with each entry id replaced by the
entry's first observation (stable across runs: it names an analyzer call) and the case id
dropped; a cached verdict's entry ids are translated through the memory saved with it.

Prints the cases (with verdicts or why none applied), the pending ones, and the outcome on
the types the captions cover completely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from collections import Counter

from eval_events import ground_truth   # scripts/eval_events.py
from matchlive.arbiter.agent import Decision, instructions
from matchlive.common.config import load_config
from matchlive.common.match import Match
from matchlive.memory.types import Event
from matchlive.runtime.runner import replay_events

TASK = """You are the arbiter for one conflict case in a football-match memory. Everything about the case is in
{dir}/packet.txt — read it first.

The instructions a model arbiter gets are below. You are that arbiter, with one difference: instead of answering
with {{"requests": ...}}, you call the tools yourself, one at a time, with
    conda run -n matchlive python scripts/arbiter_tool.py "{dir}" <tool> '<json args>'
It prints the evidence id (ev1, ev2, ...), a text, and the paths of any pictures: look at them with Read.
You have at most {max_tools} tool calls. When you have decided, save your verdict (the object that goes under
"verdict": summary and actions, each action citing the evidence ids that show it) with
    conda run -n matchlive python scripts/arbiter_tool.py "{dir}" verdict '<json>'
then reply with one line: your summary.

Rules of this setup: read only packet.txt and the pictures the tool prints; run no other command and open no other
file (in particular nothing about later parts of the match — the tools already stop at the time the case opened).

--- arbiter instructions ---
{instructions}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    args = ap.parse_args()

    config = load_config()
    arb = config["arbiter"]
    match = Match(config, args.match)
    run_dir = config["paths"]["outputs"] / "events" / args.match
    results = {h: [json.loads(line) for line in open(run_dir / f"half{h}.jsonl")] for h in (1, 2)}
    root = config["paths"]["outputs"] / "arbiter" / args.match
    budget = arb["max_rounds"] * arb["max_requests"]      # tool calls, as the API arbiter would get
    pending = []

    def decide(mem, case, packet, t):
        case_dir = root / case_key(mem, packet)
        verdict_path = case_dir / "verdict.json"
        if verdict_path.exists():
            log = case_dir / "evidence.jsonl"
            evidence = [json.loads(line)["id"] for line in open(log)] if log.exists() else []
            with open(case_dir / "state.pkl", "rb") as f:
                verdict = translate(json.loads(verdict_path.read_text()), pickle.load(f), mem)
            # One model call per tool call plus the final answer, as the API arbiter would count.
            return Decision(verdict, evidence, len(evidence) + 1, arb["fixed_latency"])
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "packet.txt").write_text(packet)
        (case_dir / "meta.json").write_text(json.dumps({"match": args.match, "case": case.case_id, "t_bound": t,
                                                        "max_tools": budget}))
        with open(case_dir / "state.pkl", "wb") as f:
            pickle.dump(mem, f)
        (case_dir / "task.md").write_text(TASK.format(dir=case_dir, max_tools=budget,
                                                      instructions=instructions(arb, 1)))
        pending.append((case.case_id, case.constraint, str(case_dir)))
        return Decision(None, [], 0, 0.0, "pending: no verdict yet")

    mem, records = replay_events(config, match, results, decide=decide)

    print("cases:")
    for r in records:
        v = r["verdict"]
        outcome = r["void"] or f"applied: {v.get('summary', '')}" if v else r["void"]
        print(f"  {r['case']:5s} {r['constraint']:30s} {outcome}")
    print(f"\n{len(pending)} pending:")
    for case_id, constraint, d in pending:
        print(f"  {case_id:5s} {constraint:30s} {d}")

    events = {eid: e for eid, e in mem.entries.items() if isinstance(e, Event) and mem.status(eid) != "retracted"}
    truth = ground_truth(match.dir)
    print(f"\n{len(events)} event entries: " + ", ".join(f"{k} {v}" for k, v in Counter(mem.status(e) for e in events).items())
          + f"; cases open {sum(c.status == 'open' for c in mem.cases.values())} / {len(mem.cases)}")
    for ty in ("goal", "yellow_card", "substitution"):
        ents = [e for e in events if events[e].type == ty]
        print(f"  {ty:13s} true {sum(t['type'] == ty for t in truth)}  entries {len(ents)}  confirmed {sum(mem.status(e) == 'confirmed' for e in ents)}")
    print(f"final score in memory: {mem.at((3, 0.0)).score}  (true: {match.metadata['score']})")
    (root / "records.json").write_text(json.dumps(records, indent=1, default=str))


def first_obs(mem, entry_id):
    return mem.justifications[entry_id][0][1].obs_id


def case_key(mem, packet):
    """Hash of the packet with entry ids replaced by their first observation and the case id dropped."""
    text = re.sub(r"^Case C\d+", "Case", packet)
    text = re.sub(r"verdict:C\d+", "verdict", text)
    text = re.sub(r"\b[EI]\d+\b", lambda m: first_obs(mem, m.group()) if m.group() in mem.entries else m.group(), text)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def translate(verdict, old, mem):
    """A verdict written against memory `old`, with its entry ids renamed to `mem`'s."""
    by_obs = {first_obs(mem, eid): eid for eid in mem.entries}
    actions = [{**a, "entry": by_obs.get(first_obs(old, a["entry"]), a["entry"])}
               if isinstance(a, dict) and a.get("entry") in old.entries else a
               for a in verdict.get("actions", [])]
    return {**verdict, "actions": actions}


if __name__ == "__main__":
    main()
