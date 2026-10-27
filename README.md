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

The streaming-QA benchmark (`dataset/streamingqa/part1_1001/`, 700 multiple-choice questions on two
matches; the 0926 and 0929 versions are deprecated) asks questions while a match is playing. MatchLive plays each match as a live stream,
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

### Reproduce the reported results from the caches

The reference run is `full_2xl_window` (`docs/results.md` §33): the RF-DETR-2XL detector, the jersey
reader retrained with SoccerNet-GSR, and identity resolved per window (`DESIGN.md` §6.1 (f)). With
the raw perception cache of both halves and `cache/api/` in place, it replays exactly: no GPU work and
no API key, every model answer comes from the cache. If anything diverged, a request would miss the
cache and the run would stop with `NoAPIKey`.

**Today this works only on the development machine**: the perception caches and `cache/api.tar.gz` in
the Hugging Face repo are still those of the earlier runs, which do not match the current default
config (`docs/SETUP.md` §3). Elsewhere, perceive on a GPU first (below) and expect API calls, not a replay,
until the new caches are uploaded.

```bash
conda activate matchlive
pytest tests/                               # 64 tests
M="2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool"
python scripts/run_live.py --match "$M" --name full_2xl_window    # ~3 min replaying from the caches
python scripts/eval_live.py --match "$M" --run full_2xl_window    # 221 windows, 29 cases, score in memory 1-2
python scripts/run_qa.py --match "$M" --run full_2xl_window --mode tools \
    --benchmark dataset/streamingqa/part1_1001 --tag 1001         # ~2 min: 0.876
```
The other match is `"2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United"`. Accuracy of the reference run
(`tools` mode, benchmark 1001, gemini-3.8-flash; single runs, about ±1 point of noise):

| Match | Questions | tools (MatchLive) |
|---|---|---|
| Chelsea – Liverpool | 347 | 0.876 |
| Chelsea – Manchester United | 353 | 0.881 |
| Both | 700 | 0.879 |

The runs before 2026-10-19 (RF-DETR-L, the previous jersey reader, per-frame identity) scored 0.879,
0.861 and 0.870 on the same questions. They are kept in `outputs/` but no longer replay: the detector,
the jersey reader and the identity resolution all changed what the analyzer is shown, so its requests
differ from the cached ones. The `blind` and `dump_log` baselines were last run on the 0926 benchmark
(`docs/results.md` §14–§16).

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
it in the run's `run.json`, so the QA and evaluation scripts read the run the same way;
`scripts/run_ablations.sh` plays the five single-switch ablations on both matches.

## Live commentary: video in, video with commentary out

Two more steps turn a step-1 run into the broadcast with our commentary, subtitled and spoken:

| Step | Script | Writes |
|---|---|---|
| 3. Commentate, 10 s of video at a time: a policy reads memory as known then and gives what must be said — a correction of something already said, a confirmation, the news — worded by status (a confirmed event names its player, a pending one only its team), and what may be said in a lull (how the match stands, pre-match background); a model writes the stretch as a script of short phrases over all its frames, each at the frame it is said at, with its energy, and a text-to-speech model voices each phrase | `scripts/run_commentary.py --match "$M" [--run full]` | `outputs/commentary/<match>/<run>/`: `lines.jsonl` (one record per phrase: text, energy, audio, the frame it was written for, when it is on air), subtitles per half |
| 4. Render the delayed live output: the half video with the phrases burned in and spoken over the broadcast's crowd and pitch sound (its commentators taken out by source separation) | `scripts/render_stream.py --match "$M" --half 1 [--start S --end E] [--hls]` | `outputs/stream/<match>/<run>/`: an MP4, or an HLS stream (`index.m3u8` + segments) |
| 4b. Stream it live instead: the half video read at its real rate as the feed, sent out D s behind it frame by frame (HLS), the phrases released at their ready times; view it from your machine through an SSH tunnel. **Not yet checked in actual use.** | `scripts/stream_live.py --match "$M" [--run R --name N] --half 2 [--start S --end E] --serve 8090`, then locally `ssh -N -L 8090:localhost:8090 <server>` and open http://localhost:8090/ | `outputs/stream/<match>/<run>/live_half<h>/`: `index.m3u8` (the last minute of 2-s segments, all with `--keep`) and `index.html` |

The output runs `commentary.output_delay` (120 s) behind the feed, which absorbs the time memory
needs to learn of an event, a stretch's length and its script (`DESIGN.md` §6.9 D3).
Rendering needs an ffmpeg with libass (the conda-forge build of `docs/SETUP.md` §2 has it), and
Demucs for the source separation (`docs/SETUP.md` §2).

## Evaluation against the SoccerNet labels

With the labels of `docs/SETUP.md` §4 in the match folders:

| Script | What it measures |
|---|---|
| `scripts/eval_memory.py --runs full no_arbiter ...` | memory over knowledge time: events counted and held, corrections, score / lineups / counts, identities |
| `scripts/inject_errors.py` | error injection: a substitution never reported, a card given to the wrong player — does memory recover, and how soon |
| `scripts/run_live.py --name gate0.9 --set events.ask_confidence=true --set consistency.gate=0.9` then `eval_memory.py` | the confidence-gating baseline: its own live run (perception, analyzer and memory), a sighting kept only at confidence ≥ τ |
| `scripts/eval_commentary.py` | commentary claims (fact / hedged / names / corrections) next to a status-blind commentator; timing (shots called as they happen, speaking rate) next to the broadcast's commentators |

Numbers: `docs/results.md`.
