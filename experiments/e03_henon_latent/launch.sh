#!/usr/bin/env bash
# Runs one stage of e03 as N parallel processes, each on every N-th job.
# Usage: bash experiments/e03_henon_latent/launch.sh <stage> <processes> <threads> [run.py options]
#   e.g. bash experiments/e03_henon_latent/launch.sh select 24 2
# Run `uv sync` once before: the processes call the project's Python directly,
# so parallel starts do not race on the environment.
# Logs: logs/e03_<stage>_<i>.log. Waits until every process has finished.
set -euo pipefail

stage=$1
n=$2
threads=${3:-2}
shift 3 || true
root=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$root/logs"
cd "$root/experiments/e03_henon_latent"

for i in $(seq 1 "$n"); do
    OMP_NUM_THREADS=$threads "$root/.venv/bin/python" -u run.py "$stage" \
        --shard "$i/$n" --threads "$threads" "$@" \
        > "$root/logs/e03_${stage}_${i}.log" 2>&1 &
done
wait
echo "stage $stage done: $(ls results/runs/$stage/*.json | wc -l) runs"
