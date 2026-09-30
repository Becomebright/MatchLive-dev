"""Blind comparison of our commentary with the broadcast's own (human) commentary, by a model judge.

    python scripts/judge_commentary.py [--runs v2 ...] [--windows 20] [--judge gpt-5] [--match M ...]

For each match, `--windows` stretches of 60 s of video are drawn (seeded, both halves) where both
the human commentators and ours speak. The judge — a different model family from the generator —
reads the two transcripts of that minute, labelled A and B in both orders, and answers which one
sounds more like professional live TV commentary (natural, fluent, engaging, informative) and which
one it thinks the automatic system wrote. Reported per run: our win rate against the humans (0.5 =
as good; an order split counts half) and how often the judge spots the machine (0.5 = cannot tell).
The human transcripts come from scripts/transcribe_broadcast.py (speech recognition, so the judge
is told to overlook its errors). Evaluation only; every judgement is a cached model call.
"""

from __future__ import annotations

import argparse
import json
import random

from eval_memory import MATCHES                          # scripts/eval_memory.py
from matchlive.common.api import APIClient
from matchlive.common.config import load_config
from matchlive.common.match import Match
from matchlive.events.analyzer import parse_json

PROMPT = """Below are two transcripts of live TV commentary on the same 60 seconds of a football match ({teams},
{minute}). One comes from the broadcaster's commentators (it is a speech-recognition transcript, so overlook its
recognition errors and broken lines); the other was produced by an automatic system.

Transcript A:
{a}

Transcript B:
{b}

1. Which sounds more like professional live TV commentary — natural, fluent, engaging and informative for the viewer?
2. Which do you think the automatic system produced?
Answer with JSON only: {{"better": "A" | "B", "machine": "A" | "B", "why": "one sentence"}}"""


def minute(half: int, second: float) -> str:
    return f"{'first' if half == 1 else 'second'} half, minute {int(second // 60) + 1}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["v2"], help="commentary folders under outputs/commentary/<match>/")
    ap.add_argument("--windows", type=int, default=20, help="60-s stretches per match")
    ap.add_argument("--judge", default="gpt-5", help="the judge model (another family than the generator)")
    ap.add_argument("--match", action="append", default=None, help="default: both benchmark matches")
    args = ap.parse_args()
    config = load_config()
    out = config["paths"]["outputs"]
    judge = APIClient(dict(config["api"], model=args.judge), config["paths"]["cache"])
    for run in args.runs:
        wins, spotted, n = 0.0, 0.0, 0
        reasons = []
        for name in args.match or MATCHES:
            lines = [json.loads(l) for l in open(out / "commentary" / name / run / "lines.jsonl")]
            prematch = Match(config, name).prematch["teams"]
            teams = f"{prematch['home']} vs {prematch['away']}"
            human = {h: json.loads((out / "reference" / name / f"half{h}.json").read_text()) for h in (1, 2)}
            rng = random.Random(f"{name}/{run}")
            candidates = [(h, float(s)) for h in (1, 2) for s in range(60, int(max(x["end"] for x in human[h])) - 60, 30)]
            rng.shuffle(candidates)
            taken = 0
            for h, start in candidates:
                if taken == args.windows:
                    break
                ours = " ".join(l["text"] for l in lines if l["show"][0] == h and start <= l["show"][1] < start + 60)
                theirs = " ".join(x["text"] for x in human[h] if start <= x["start"] < start + 60)
                if len(ours.split()) < 15 or len(theirs.split()) < 15:
                    continue
                taken += 1
                votes = []
                for a, b, ours_is in ((ours, theirs, "A"), (theirs, ours, "B")):
                    r = judge.generate([PROMPT.format(teams=teams, minute=minute(h, start), a=a, b=b)], json_output=True)
                    answer = parse_json(r.text)
                    votes.append((answer.get("better") == ours_is, answer.get("machine") == ours_is))
                    reasons.append(answer.get("why", ""))
                wins += sum(v[0] for v in votes) / 2
                spotted += sum(v[1] for v in votes) / 2
                n += 1
        print(f"{run}: {n} minutes judged by {args.judge}; ours preferred {wins / max(n, 1):.2f} "
              f"(0.5 = as good as the broadcast), machine spotted {spotted / max(n, 1):.2f} (0.5 = cannot tell)")
        for why in reasons[:6]:
            print("   -", why)


if __name__ == "__main__":
    main()
