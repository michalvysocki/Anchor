"""
Per-horizon report of the gap stage (README, findings 7 and 19): test MSE of
onestep and direct at every k for every configuration, the horizons where
direct predicts better than the mean (MSE below PREDICTABLE of Var(y)), and
the gap over several ranges of k. Read before the selection stage, to fix
the range of the primary metric and the configurations of H1 and H4.
"""

import json

import numpy as np
from design import KAPPAS, WINDOWS, config_name, is_markov
from run import RUNS_DIR, run_name

PREDICTABLE = 0.95
RANGES = ((1, 2), (1, 3), (1, 4), (1, 6), (1, 10))


def mse(kappa, window, variant):
    path = RUNS_DIR / "gap" / (run_name(kappa, window, variant, "main") + ".json")
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    return np.array(r["test"]["mse"][0]), r["steps"], r["stopped_at"][0]


def main():
    out = {}
    for kappa in KAPPAS:
        for window in (*WINDOWS, None):
            one, direct = mse(kappa, window, "onestep"), mse(kappa, window, "direct")
            name = config_name(kappa, window)
            if one is None or direct is None:
                print(f"{name:10s} missing")
                continue
            (o, o_steps, o_stop), (d, d_steps, d_stop) = one, direct
            predictable = [k + 1 for k in range(len(d)) if d[k] < PREDICTABLE]
            gaps = {
                f"{a}..{b}": float(
                    (o[a - 1 : b].mean() - d[a - 1 : b].mean()) / o[a - 1 : b].mean()
                )
                for a, b in RANGES
            }
            out[name] = {
                "markov": is_markov(kappa, window),
                "onestep": o.tolist(),
                "direct": d.tolist(),
                "predictable_k": predictable,
                "relative_gap": gaps,
                "steps": {"onestep": o_steps, "direct": d_steps},
                "capped": {"onestep": o_stop < 0, "direct": d_stop < 0},
            }
            print(
                f"\n{name}  markov={is_markov(kappa, window)}  steps onestep {o_steps} direct {d_steps}"
                f"{'  (CAPPED)' if o_stop < 0 or d_stop < 0 else ''}"
            )
            print("  k        " + " ".join(f"{k:6d}" for k in range(1, len(o) + 1)))
            print("  onestep  " + " ".join(f"{x:6.3f}" for x in o))
            print("  direct   " + " ".join(f"{x:6.3f}" for x in d))
            print(f"  direct < {PREDICTABLE} Var(y) at k = {predictable}")
            print(
                "  relative gap (onestep - direct) / onestep over k = "
                + "  ".join(f"{r}: {g:+.3f}" for r, g in gaps.items())
            )
    (RUNS_DIR.parent / "gap_report.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
