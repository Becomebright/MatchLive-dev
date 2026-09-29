"""The arbiter subagent's only way to gather evidence and deliver its verdict for one case.

    python scripts/arbiter_tool.py <case dir> get_frames '{"half": 1, "start": 1830, "end": 1860}'
    python scripts/arbiter_tool.py <case dir> verdict '{"summary": "...", "actions": [...]}'

<case dir> is written by scripts/run_arbiter.py (packet.txt, task.md, state.pkl, meta.json).
Every tool answer is bounded by the time the case opened (matchlive.runtime.evidence), gets
the next evidence id ("ev3"), is logged to evidence.jsonl, and its pictures are saved next to
it; the command prints the id, the text and the picture paths. `verdict` checks the JSON's
shape and saves it as verdict.json for scripts/run_arbiter.py to apply (behind the guardrails
of matchlive.arbiter.verdict).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from matchlive.common.config import load_config
from matchlive.common.match import Match
from matchlive.runtime.evidence import EvidenceTools


def main():
    case_dir, tool, raw = Path(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "{}"
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"error: the last argument must be JSON ({e})")
    meta = json.loads((case_dir / "meta.json").read_text())
    log = case_dir / "evidence.jsonl"
    n_before = sum(1 for _ in open(log)) if log.exists() else 0

    if tool == "verdict":
        if not isinstance(args, dict) or not isinstance(args.get("actions"), list):
            sys.exit('error: a verdict is {"summary": "...", "actions": [...]}')
        (case_dir / "verdict.json").write_text(json.dumps(args, indent=1))
        print(f"verdict saved ({len(args['actions'])} actions)")
        return
    if n_before >= meta["max_tools"]:
        sys.exit(f"error: the evidence budget ({meta['max_tools']} tool calls) is spent; give your verdict now")

    config = load_config()
    match = Match(config, meta["match"])
    with open(case_dir / "state.pkl", "rb") as f:
        mem = pickle.load(f)
    ev = EvidenceTools(config, match).call(mem, tuple(meta["t_bound"]), tool, args)
    ev_id = f"ev{n_before + 1}"
    paths = []
    for k, jpg in enumerate(ev.images):
        path = case_dir / f"{ev_id}_{k:02d}.jpg"
        path.write_bytes(jpg)
        paths.append(str(path))
    with open(log, "a") as f:
        f.write(json.dumps({"id": ev_id, "tool": tool, "args": args, "text": ev.text, "images": paths}) + "\n")
    print(f"[{ev_id}] {ev.text}")
    for path in paths:
        print(f"picture: {path}")


if __name__ == "__main__":
    main()
