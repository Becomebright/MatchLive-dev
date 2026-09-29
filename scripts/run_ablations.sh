#!/usr/bin/env bash
# The main ablations of DESIGN.md §7 on both benchmark matches: each run turns one loop switch
# off, plays the match live (scripts/run_live.py) and answers the streaming-QA benchmark in tools
# mode (scripts/run_qa.py). Runs go in parallel, one process per (match, ablation); logs in
# outputs/live/logs/<match tag>_<run>.log.
#
#   bash scripts/run_ablations.sh [run ...]              # default: all five, both matches
#   MATCH_TAGS="mu" bash scripts/run_ablations.sh no_arbiter
#
# A request is retried up to 40 times (backoff capped at 120 s): on an overloaded API the default
# 8 attempts lose windows, and a lost window would make the runs differ by API luck, not by switch.
set -u
cd "$(dirname "$0")/.."
# The editable install may point at another checkout (e.g. from a git worktree): use this one.
export PYTHONPATH="$PWD"

declare -A SET=(
  [no_replay]="loop.replay=false"
  [no_state_to_perception]="loop.state_to_perception=false"
  [no_arbiter]="loop.arbiter=false"
  [no_cascade]="loop.cascade=false"
  [no_consistency]="loop.consistency=false"
)
declare -A MATCH=(
  [mu]="2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United"
  [liv]="2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool"
)
RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(no_replay no_state_to_perception no_arbiter no_cascade no_consistency)

mkdir -p outputs/live/logs
for tag in ${MATCH_TAGS:-mu liv}; do
  for run in "${RUNS[@]}"; do
    log="outputs/live/logs/${tag}_${run}.log"
    (
      conda run -n matchlive --no-capture-output python scripts/run_live.py --match "${MATCH[$tag]}" --name "$run" \
        --set "${SET[$run]}" --set api.max_attempts=40 \
      && conda run -n matchlive --no-capture-output python scripts/run_qa.py --match "${MATCH[$tag]}" --run "$run" --mode tools
    ) > "$log" 2>&1 &
  done
done
wait
