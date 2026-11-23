"""
Exploratory analysis, not pre-registered and not part of e03: decomposition
of the rollout MSE of the calibration pilot into shift, miscalibration and
lost information, per seed and k. Reads the saved measures; no training. See
README.md in this directory.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "calibration" / "results" / "pilot.json"

ARMS = ("henon_C6", "henon_CL6")
RANGES = {"k7-8": (7, 8), "k15-20": (15, 20)}
TERMS = ("shift", "miscalibration", "information")
TOLERANCE = 0.01


def decompose(r):
    """MSE and its three terms, each (seed, k)."""
    mse = np.array(r["mse_k_seed"])
    b, s = np.array(r["bias_k_seed"]), np.array(r["sharpness_k_seed"])
    rho, v = np.array(r["corr_k_seed"]), np.array(r["var_y_k"])[None]
    return mse, {
        "shift": b**2,
        "miscalibration": v * (s - rho) ** 2,
        "information": v * (1 - rho**2),
    }


def summarize(mse, terms):
    shares = {t: terms[t] / mse for t in TERMS}
    out = {
        "max_sum_error": float(
            (np.abs(sum(terms.values()) - mse) / mse).max()
        ),
        "median_share_k": {
            t: np.median(shares[t], axis=0).tolist() for t in TERMS
        },
        "median_mse_k": np.median(mse, axis=0).tolist(),
        "ranges": {},
    }
    for name, (a, b) in RANGES.items():
        per_seed = np.stack([shares[t][:, a - 1 : b].mean(1) for t in TERMS])
        largest = per_seed.argmax(0)
        out["ranges"][name] = {
            "share_seed": {t: per_seed[i].tolist() for i, t in enumerate(TERMS)},
            "median_share": {
                t: float(np.median(per_seed[i])) for i, t in enumerate(TERMS)
            },
            "seeds_largest": {
                t: int((largest == i).sum()) for i, t in enumerate(TERMS)
            },
        }
    return out


def main():
    pilot = json.loads(PILOT.read_text())
    results = {arm: summarize(*decompose(pilot[arm])) for arm in ARMS}
    for arm in ARMS:
        early = np.array(results[arm]["ranges"]["k7-8"]["share_seed"]["miscalibration"])
        late = np.array(results[arm]["ranges"]["k15-20"]["share_seed"]["miscalibration"])
        results[arm]["miscalibration_growth_median"] = float(np.median(late - early))
    out = HERE / "results" / "decomposition.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    for arm in ARMS:
        r = results[arm]
        print(f"\n{arm}  max sum error {r['max_sum_error']:.2e} (tolerance {TOLERANCE})")
        for name, row in r["ranges"].items():
            shares = "  ".join(f"{t} {row['median_share'][t]:.3f}" for t in TERMS)
            counts = "  ".join(f"{t} {row['seeds_largest'][t]}" for t in TERMS)
            print(f"  {name:7s} median share: {shares}\n          largest in seeds: {counts}")
        print(f"  miscalibration growth k7-8 -> k15-20: {r['miscalibration_growth_median']:+.3f}")
        print("  k    MSE  " + "  ".join(f"{t[:6]:>6s}" for t in TERMS))
        for k in range(1, 21):
            row = "  ".join(
                f"{r['median_share_k'][t][k - 1]:6.3f}" for t in TERMS
            )
            print(f"  {k:2d} {r['median_mse_k'][k - 1]:6.3f}  {row}")


if __name__ == "__main__":
    main()
