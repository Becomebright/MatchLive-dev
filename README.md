# MatchLive

Live football commentary and question answering from a match video stream, built as a
neuro-symbolic truth-maintenance loop over perception and a match memory — see `docs/DESIGN.md` §1.

| File | What it covers |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | design source of truth (Chinese): paper claim, architecture, data contracts, modules, phase plan |
| [`docs/SETUP.md`](docs/SETUP.md) | setting up a new machine: environment, model weights, data, caches, a first run to check it |
| [`docs/results.md`](docs/results.md) | measured results: what was run, with which script, and the numbers |
| [`CLAUDE.md`](CLAUDE.md) | how to work in this repo: layout, commands, conventions, git rules |

## Streaming QA

The streaming-QA benchmark (`dataset/streamingqa/part1/`, 817 multiple-choice questions on two
matches) asks questions while a match is playing. MatchLive plays each match as a live stream,
knowing only the pre-match facts: team names, the squads (shirt numbers, starting XI, positions),
kit colours, and which end each team defends. A question at time *t* is answered from what the
system knew at *t*: its match memory as written by then, and the broadcast frames before *t*.
Method and numbers: `docs/results.md` §14; later versions of the benchmark: §15–§16.

Two steps per match, each a script:

| Step | Script | Writes |
|---|---|---|
| 1. Play the match live: perception → event analysis → consistency check → arbiter, on one match clock | `scripts/run_live.py` | `outputs/live/<match>/<run>/`: memory op log, analyzer windows, arbiter cases, identities per sampled frame |
| 2. Ask every benchmark question of that match at its time | `scripts/run_qa.py` | `outputs/qa/<match>/<run>/<mode>.jsonl`, plus an accuracy table on stdout |

`scripts/eval_live.py` summarises a step-1 run (event entries, cases and verdicts, score and
substitutions at the final whistle).

### Reproduce the reported results from the shared caches

With the prerequisites, environment and assets of `docs/SETUP.md` §0–3 (including `cache/*`), the recorded
runs replay exactly: no GPU work and no API key, every model answer comes from `cache/api/`. If
anything diverged, a request would miss the cache and the run would stop with `NoAPIKey`.

```bash
conda activate matchlive
# without configs/api.local.yaml (a fresh clone has none) nothing but the cache can be used
pytest tests/                               # 35 tests
M="2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool"
python scripts/run_live.py --match "$M"     # ~3 min
python scripts/eval_live.py --match "$M"    # 221 windows, 24 cases, score in memory 1-2
python scripts/run_qa.py --match "$M" --mode tools       # ~2 min: 0.923
python scripts/run_qa.py --match "$M" --mode dump_log    # seconds: 0.783
python scripts/run_qa.py --match "$M" --mode blind       # seconds: 0.602
# the 1001 version of the benchmark (a folder of per-match files): writes <mode>_1001.jsonl
python scripts/run_qa.py --match "$M" --mode tools --benchmark dataset/streamingqa/part1_1001 --tag 1001   # 0.879
```
The other match is `"2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United"`. Expected accuracy:

| Benchmark | Match | tools (MatchLive) | dump_log (whole event log, no video) | blind (question and options only) |
|---|---|---|---|---|
| `part1/benchmark-0926.jsonl` | Chelsea – Liverpool (405 questions) | 0.923 | 0.783 | 0.602 |
| | Chelsea – Manchester United (412) | 0.893 | 0.801 | 0.631 |
| | Both (817) | 0.908 | 0.792 | 0.617 |
| `part1_1001/` | Chelsea – Liverpool (347) | 0.879 | 0.674 | 0.493 |
| | Chelsea – Manchester United (353) | 0.861 | 0.722 | 0.595 |
| | Both (700) | 0.870 | 0.699 | 0.544 |

The 0926 numbers differ slightly from `docs/results.md` §14 (0.909 there): since then a void
arbiter verdict gets one more try, which changes the Manchester United memory from 2H 40:32, and a
JSON answer followed by stray text is parsed (§15).

`run_qa.py --half 1` or `--limit N` answer a subset, written to its own file.

### Run from scratch (a new match, model or setting)

1. Perception, once per half, on the GPU (about 25 min per half on an RTX 4060 Ti):
   `python scripts/run_perception.py --match "$M" --half 1` (and `--half 2`).
2. Put the endpoint and key in `configs/api.local.yaml` (docs/SETUP.md §2), then
   `python scripts/run_live.py --match "$M" --name <run>`.
   The model calls are sequential on the match clock, so a match takes 1–3 h of wall time,
   depending on how loaded the API is (every answer is cached; a re-run with the same inputs
   replays in minutes).
3. `python scripts/run_qa.py --match "$M" --run <run> --mode tools`.

Settings live in `configs/default.yaml`: the model (`api.model`), the loop switches of
`DESIGN.md` §7 (`loop.*`, e.g. an ablation turns one off and runs under its own `--name`), and the
QA budget (`qa.*`). `run_live.py --set loop.arbiter=false` changes a setting for one run and records
it in the run's `run.json`, so the QA and evaluation scripts read the run the same way.
