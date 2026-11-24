#!/usr/bin/env bash
# CPU-GPU equivalence check of e03 (rule: research note, section A, revised
# 2026-11-24). Reruns five select jobs on the club GPU and on the club CPU,
# each device in its own copy under ~/equiv, never in ~/anchor/.../results.
# Usage: equiv.sh setup | run
set -uo pipefail

E=$HOME/equiv
SRC=$E/src
PY=$HOME/anchor/.venv/bin/python
COMMIT=ab30086e9e80ae63614c8218d15b647ffe36d781
THREADS=2
# k0.5_L2_onestep_main added once after the light group was inconclusive
# (2026-11-24); run.py skips the jobs whose .pt exists, so only it trains
JOBS=(k0.5_L1_onestep_main k0.5_L1_C_main k0.5_gru_C_main k1.0_L1_C_main k1.0_L2_C_main k0.5_L2_onestep_main)

setup() {
    test ! -e "$E" || { echo "$E exists"; exit 1; }
    git clone -q "$HOME/anchor" "$SRC" || exit 1
    test "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" || { echo "wrong commit"; exit 1; }
    for d in cuda cpu; do
        # a fresh copy with an empty results/runs: every job trains, and the
        # select results of the repository are never read or overwritten
        mkdir -p "$E/$d/results/runs" "$E/logs"
        cp "$SRC"/experiments/e03_henon_latent/*.py "$E/$d/"
    done
    (cd "$E/cuda" && PYTHONPATH=$SRC/src "$PY" -c "
import anchor, torch, platform
from run import plan, run_name
names = [run_name(*j) for j in plan()]
for q in '${JOBS[*]}'.split():
    assert [n for n in names if q in n] == [q], q
print('anchor', anchor.__file__)
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('machine', platform.node(), 'commit $COMMIT')
") | tee "$E/env.txt"
}

run() {
    local pids=() d j
    rm -f "$E/DONE" "$E/FAILED"
    mkdir -p "$E/logs/first_run"
    mv -n "$E"/logs/*.log "$E/logs/first_run/" 2>/dev/null
    for d in cuda cpu; do
        for j in "${JOBS[@]}"; do
            (
                cd "$E/$d" && OMP_NUM_THREADS=$THREADS PYTHONPATH=$SRC/src "$PY" -u run.py select \
                    --only "$j" --device "$d" --threads "$THREADS" > "$E/logs/${d}_${j}.log" 2>&1
            ) &
            pids+=($!)
        done
    done
    local failed=0
    for p in "${pids[@]}"; do wait "$p" || failed=1; done
    ls "$E"/*/results/runs/select/*.json | wc -l
    if [ "$failed" = 1 ]; then echo failed > "$E/FAILED"; else echo finished > "$E/DONE"; fi
}

case ${1:-} in
setup) setup ;;
run) run ;;
*) echo "usage: equiv.sh setup|run"; exit 2 ;;
esac
