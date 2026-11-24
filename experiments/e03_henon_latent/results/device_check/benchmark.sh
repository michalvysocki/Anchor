#!/usr/bin/env bash
# Timing benchmark of the e03 final stage, CPU against GPU. Results are
# discarded: every process works in its own copy under ~/bench_e03.
# Usage: bench.sh setup | preflight | run
set -uo pipefail

B=$HOME/bench_e03
SRC=$HOME/anchor_bench
EXP=$SRC/experiments/e03_henon_latent
PY=$HOME/anchor/.venv/bin/python
STEPS=1000
THREADS=2
LIGHT=k0.5_L2_onestep_small
MEDIUM=k0.5_L4_DB+F_main
GRU=k0.5_gru_C_main
HEAVY=k0.5_L1_TDB_main

setup() {
    test ! -e "$SRC" && test ! -e "$B" || { echo "target exists"; exit 1; }
    git clone -q "$HOME/anchor" "$SRC" || exit 1
    for i in $(seq 1 14); do
        mkdir -p "$B/w$i/results/runs"
        cp "$EXP"/*.py "$B/w$i/"
        ln -s "$EXP/results/runs/select" "$B/w$i/results/runs/select"
    done
    mkdir -p "$B/logs"
    echo "commit $(git -C "$SRC" rev-parse HEAD)"
}

# measure <label> <device> <job> <processes> <steps>
measure() {
    local label=$1 device=$2 job=$3 n=$4 steps=$5 pids=() i
    local load0 t0 t1
    load0=$(cut -d' ' -f1 /proc/loadavg)
    t0=$(date +%s.%N)
    for i in $(seq 1 "$n"); do
        rm -rf "$B/w$i/results/runs/final"
        (
            cd "$B/w$i" && OMP_NUM_THREADS=$THREADS PYTHONPATH=$SRC/src "$PY" -u run.py final \
                --only "$job" --steps "$steps" --device "$device" --threads "$THREADS" \
                > "$B/logs/${label}_${device}_n${n}_p${i}.log" 2>&1
        ) &
        pids+=($!)
    done
    for i in "${pids[@]}"; do
        wait "$i" || { echo "FAILED $label $device n=$n"; exit 1; }
    done
    t1=$(date +%s.%N)
    for i in $(seq 1 "$n"); do
        local s
        s=$(sed -nE 's/.* steps +[0-9]+ +([0-9.]+)s test mse.*/\1/p' \
            "$B/logs/${label}_${device}_n${n}_p${i}.log" | tail -1)
        test -n "$s" || { echo "NO TIME $label $device n=$n p=$i"; exit 1; }
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$device" "$job" "$n" "$i" \
            "$steps" "$s" "$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }')" "$load0" >> "$B/results.tsv"
    done
    echo "$(date +%T) done $label $device n=$n"
}

sampler() {
    while true; do
        printf '%s\t%s\t%s\n' "$(date +%T)" "$(cut -d' ' -f1 /proc/loadavg)" \
            "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)" \
            >> "$B/samples.tsv"
        sleep 30
    done
}

case ${1:-} in
setup) setup ;;
preflight)
    rm -f "$B/results.tsv"
    measure preflight cpu "$LIGHT" 1 20
    measure preflight cuda "$LIGHT" 1 20
    cat "$B/results.tsv"
    rm -f "$B/results.tsv"
    ;;
run)
    sampler & S=$!
    trap 'kill $S 2>/dev/null' EXIT
    printf 'label\tdevice\tjob\tprocesses\tprocess\tsteps\tseconds\twall\tload_before\n' > "$B/results.tsv"
    for pair in light:$LIGHT medium:$MEDIUM gru:$GRU heavy:$HEAVY; do
        for d in cuda cpu; do
            measure "${pair%%:*}" "$d" "${pair#*:}" 1 "$STEPS"
        done
    done
    for n in 2 4 8; do measure scale cuda "$MEDIUM" "$n" "$STEPS"; done
    for n in 4 7 10 14; do measure scale cpu "$MEDIUM" "$n" "$STEPS"; done
    echo finished > "$B/DONE"
    ;;
*) echo "usage: bench.sh setup|preflight|run"; exit 2 ;;
esac
