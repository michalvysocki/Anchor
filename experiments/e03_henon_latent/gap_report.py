"""
Per-horizon report of the gap stage (README, Gap measured before the Anchor
runs): rollout MSE of onestep and direct at every k on the VALIDATION set,
computed from the saved models of the gap stage, the horizons where direct
predicts better than the mean (MSE below PREDICTABLE of Var(y)), and the
relative gap over several ranges of k. Design decisions (range of the
primary metric, configurations of H1 and H4) are taken on these validation
numbers; the test numbers of the same runs are printed for the record only.
"""

import json

import numpy as np
import torch
from design import KAPPAS, SHORT_HORIZONS, SIZES, WINDOWS, config_name, is_markov
from run import RUNS_DIR, Data, run_name

from anchor.latent import DirectModels, WorldModels

PREDICTABLE = 0.95
RANGES = ((1, 2), (1, 3), (1, 4), (1, 6), (1, 10))


def validation_mse(kappa, window, variant) -> np.ndarray | None:
    path = RUNS_DIR / "gap" / (run_name(kappa, window, variant, "main") + ".pt")
    if not path.exists():
        return None
    if variant == "direct":
        m = DirectModels(window, [0], SIZES)
    else:
        m = WorldModels(variant, window, SHORT_HORIZONS, [0], SIZES)
    m.load_state_dict(torch.load(path)["state"])
    val = Data(kappa, window, "large", [0]).val
    total = 0
    with torch.no_grad():
        for i in range(0, len(val), 4096):
            part = val[i : i + 4096][None]
            total = total + m.rollout_mse(part)["mse"][0] * part.shape[1]
    return (total / len(val)).numpy()


def test_mse(kappa, window, variant) -> np.ndarray | None:
    path = RUNS_DIR / "gap" / (run_name(kappa, window, variant, "main") + ".json")
    if not path.exists():
        return None
    return np.array(json.loads(path.read_text())["test"]["mse"][0])


def relative_gaps(o, d):
    return {
        f"{a}..{b}": float(
            (o[a - 1 : b].mean() - d[a - 1 : b].mean()) / o[a - 1 : b].mean()
        )
        for a, b in RANGES
    }


def main():
    torch.set_num_threads(2)
    out = {}
    for kappa in KAPPAS:
        for window in (*WINDOWS, None):
            name = config_name(kappa, window)
            o, d = (
                validation_mse(kappa, window, "onestep"),
                validation_mse(kappa, window, "direct"),
            )
            if o is None or d is None:
                print(f"{name:10s} missing")
                continue
            ot, dt = (
                test_mse(kappa, window, "onestep"),
                test_mse(kappa, window, "direct"),
            )
            predictable = [k + 1 for k in range(len(d)) if d[k] < PREDICTABLE]
            out[name] = {
                "markov": is_markov(kappa, window),
                "onestep_val": o.tolist(),
                "direct_val": d.tolist(),
                "predictable_k_val": predictable,
                "relative_gap_val": relative_gaps(o, d),
                "relative_gap_test": relative_gaps(ot, dt),
            }
            print(f"\n{name}  markov={is_markov(kappa, window)}  (validation)")
            print("  k        " + " ".join(f"{k:6d}" for k in range(1, len(o) + 1)))
            print("  onestep  " + " ".join(f"{x:6.3f}" for x in o))
            print("  direct   " + " ".join(f"{x:6.3f}" for x in d))
            print(f"  direct < {PREDICTABLE} Var(y) at k = {predictable}")
            print(
                "  relative gap over k = "
                + "  ".join(
                    f"{r}: {g:+.4f}" for r, g in out[name]["relative_gap_val"].items()
                )
                + f"   (test 1..4: {out[name]['relative_gap_test']['1..4']:+.4f})"
            )
    (RUNS_DIR.parent / "gap_report.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
