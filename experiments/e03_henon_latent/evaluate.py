"""
Evaluation of e03 (README, Metrics and Hypotheses): paired verdicts on the
test set of the final stage, the gap of every configuration, H1-H5 and H-M*,
and the critic of H5 fitted on the saved onestep models. Writes
results/test_results.json and prints the tables.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from design import (
    ABLATION_CONFIGS,
    CRITIC_HIDDEN,
    EVALUATION_SEEDS,
    KAPPAS,
    LAMBDAS,
    LATENT,
    PRIMARY_K,
    SHORT_HORIZONS,
    SIZES,
    VARIANCE_CONFIGS,
    WINDOWS,
    config_name,
    is_markov,
)
from run import RUNS_DIR, Data, run_name, selected_lambda

from anchor.ensemble import EnsembleAdam, EnsembleMLP
from anchor.latent import WorldModels, anchor_pairs, base
from anchor.stats import mean_ci, paired_log_ratio, verdict

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MARGIN = math.log(1.02)
# configurations with a gap below this fraction of the onestep error are
# excluded from H1 and H4 (README, finding 7), decided on the gap stage
GAP_THRESHOLD = 0.01
CRITIC_STEPS = 3000


# ---------------------------------------------------------------- loading


def load(stage, kappa, window, variant, arm):
    path = RUNS_DIR / stage / (run_name(kappa, window, variant, arm) + ".json")
    return json.loads(path.read_text()) if path.exists() else None


def primary(kappa, window, variant, arm="main", lam=None) -> dict[int, float] | None:
    """
    Test rollout MSE (mean over the horizons PRIMARY_K) per evaluation seed, at the
    selected lambda unless lam is given. The long arm shares onestep and
    direct with the main arm.
    """

    if arm == "long" and base(variant) in ("onestep", "direct"):
        arm = "main"
    r = load("final", kappa, window, variant, arm)
    if r is None:
        return None
    if lam is None:
        lam = selected_lambda(kappa, window, variant, arm)
    a, b = PRIMARY_K
    out = {}
    for l, s, mse in zip(r["lams"], r["seeds"], r["test"]["mse"]):
        if abs(l - lam) < 1e-6:
            out[s] = float(np.mean(mse[a - 1 : b]))
    return out


def compare(a: dict | None, b: dict | None) -> dict | None:
    """Paired log ratio a / b over the shared seeds, its 95% interval and verdict."""
    if a is None or b is None:
        return None
    seeds = sorted(set(a) & set(b))
    lr = paired_log_ratio([a[s] for s in seeds], [b[s] for s in seeds])
    finite = np.isfinite(lr)
    if finite.sum() < 2:
        return {
            "mean": float("nan"),
            "half": float("nan"),
            "verdict": "diverged",
            "n": int(finite.sum()),
        }
    mean, half = mean_ci(lr[finite])
    return {
        "mean": float(mean),
        "half": float(half),
        "verdict": verdict(float(mean), float(half), MARGIN),
        "n": int(finite.sum()),
    }


def mean_of(d):
    return None if d is None else float(np.nanmean(list(d.values())))


# ---------------------------------------------------------------- gaps


def gap_stage(kappa, window) -> float | None:
    """Relative gap (onestep - direct) / onestep from the gap stage (one seed)."""
    one, direct = (
        load("gap", kappa, window, "onestep", "main"),
        load("gap", kappa, window, "direct", "main"),
    )
    if one is None or direct is None:
        return None
    o, d = np.mean(one["test"]["mse"]), np.mean(direct["test"]["mse"])
    return float((o - d) / o)


def configs(gru=False):
    return (
        [(k, None) for k in KAPPAS]
        if gru
        else [(k, w) for k in KAPPAS for w in WINDOWS]
    )


def gap_configs():
    """Main-arm window configurations with a gap that H1 and H4 can resolve."""
    out = []
    for k, w in configs():
        if is_markov(k, w):
            continue
        g = gap_stage(k, w)
        if g is None or g >= GAP_THRESHOLD:
            out.append((k, w))
    return out


# ---------------------------------------------------------------- hypotheses


def h1():
    """
    C+F vs C in the gap configurations, with the control of the README: the
    flag's improvement in the Markov controls bounds what it does for reasons
    other than the Markov gap.
    """

    rows = {}
    for k, w in gap_configs():
        rows[config_name(k, w)] = compare(primary(k, w, "C+F"), primary(k, w, "C"))
    controls = {}
    for k, w in configs():
        if is_markov(k, w):
            controls[config_name(k, w)] = compare(
                primary(k, w, "C+F"), primary(k, w, "C")
            )
    control = max(
        (
            -r["mean"]
            for r in controls.values()
            if r is not None and np.isfinite(r["mean"])
        ),
        default=float("nan"),
    )
    better = [n for n, r in rows.items() if r is not None and r["verdict"] == "better"]
    attributed = [n for n in better if -rows[n]["mean"] > control]
    return {
        "rows": rows,
        "controls": controls,
        "control_improvement": control,
        "better": len(better),
        "attributed": attributed,
        "n": len(rows),
        "holds": len(better) >= 5,
    }


def h2():
    rows, harm = {}, 0
    for k, w in configs():
        if not is_markov(k, w):
            continue
        ref = primary(k, w, "onestep")
        for v in ("onestep+F", "C", "C+F", "DB", "DB+F", "M*", "M*+F"):
            lams = [0.0] if base(v) == "onestep" else LAMBDAS
            for lam in lams:
                r = compare(primary(k, w, v, lam=lam), ref)
                rows[f"{config_name(k, w)} {v} λ={lam:g}"] = r
                harm += r is not None and r["verdict"] == "material harm"
    return {"rows": rows, "material_harm": harm, "holds": harm == 0}


def h3():
    rows = {}
    for k, w in ABLATION_CONFIGS:
        for f in ("", "+F"):
            rows[f"{config_name(k, w)} main DB{f} vs D{f}"] = compare(
                primary(k, w, "DB" + f), primary(k, w, "D" + f)
            )
    for k, w in VARIANCE_CONFIGS:
        for arm in ("small", "long"):
            for f in ("", "+F"):
                rows[f"{config_name(k, w)} {arm} DB{f} vs D{f}"] = compare(
                    primary(k, w, "DB" + f, arm), primary(k, w, "D" + f, arm)
                )
    return {"rows": rows}


def h4():
    """
    H4: DB+F vs C+F (decision); H4b: DB vs C without the flag (secondary,
    added before any test result), in the main gap configurations and the
    variance-regime arm.
    """

    rows, secondary = {}, {}
    for k, w in gap_configs():
        name = f"{config_name(k, w)} main"
        rows[name] = compare(primary(k, w, "DB+F"), primary(k, w, "C+F"))
        secondary[name] = compare(primary(k, w, "DB"), primary(k, w, "C"))
    holds = all(
        r is not None and r["verdict"] in ("harmless", "better") for r in rows.values()
    )
    for k, w in VARIANCE_CONFIGS:
        for arm in ("small", "long"):
            name = f"{config_name(k, w)} {arm}"
            rows[name] = compare(primary(k, w, "DB+F", arm), primary(k, w, "C+F", arm))
            secondary[name] = compare(primary(k, w, "DB", arm), primary(k, w, "C", arm))
    return {"rows": rows, "holds": holds, "H4b": secondary}


def mstar_settings():
    """(kappa, window, arm, flag suffix, fixed targets) where H-M* is judged."""
    out = []
    for k, w in configs() + configs(gru=True):
        for f in ("", "+F"):
            fixed = ["C", "DB"] + (["D"] if (k, w) in ABLATION_CONFIGS else [])
            out.append((k, w, "main", f, fixed))
    for k, w in VARIANCE_CONFIGS:
        for arm in ("small", "long"):
            for f in ("", "+F"):
                out.append((k, w, arm, f, ["C", "D", "DB"]))
    return out


def h_mstar():
    rows, worse_somewhere = {}, {}
    ok = True
    for k, w, arm, f, fixed in mstar_settings():
        vals = {t: primary(k, w, t + f, arm) for t in fixed}
        vals = {t: v for t, v in vals.items() if v}
        if not vals:
            continue
        best = min(vals, key=lambda t: mean_of(vals[t]))
        r = compare(primary(k, w, "M*" + f, arm), vals[best])
        key = f"{config_name(k, w)} {arm} M*{f} vs best ({best}{f})"
        rows[key] = r
        ok &= r is not None and r["mean"] + r["half"] < MARGIN
        for t, v in vals.items():
            c = compare(v, vals[best])
            if t != best and c is not None and c["mean"] - c["half"] > 0:
                worse_somewhere.setdefault(t, []).append(key)
    every_fixed_loses = all(t in worse_somewhere for t in ("C", "D", "DB"))
    return {
        "rows": rows,
        "never_worse": ok,
        "fixed_worse": worse_somewhere,
        "holds": ok and every_fixed_loses,
    }


# ---------------------------------------------------------------- critic (H5)


def critic(kappa, window) -> dict | None:
    """
    Critic and raw residual of D's targets at the trained onestep models
    (final stage, evaluation seeds): a head on sg[phi(z_t)] regresses the
    residual rollout_k(z_t) - T_{k,p} on the training windows of each seed;
    reported on the test windows, averaged over pairs and seeds.
    """

    path = RUNS_DIR / "final" / (run_name(kappa, window, "onestep", "main") + ".pt")
    if not path.exists():
        return None
    seeds = list(EVALUATION_SEEDS)
    m = WorldModels("D", window, SHORT_HORIZONS, seeds, SIZES)
    state = torch.load(path)["state"]
    m.load_state_dict(state, strict=False)
    pairs = anchor_pairs(SHORT_HORIZONS)
    head = EnsembleMLP([LATENT, CRITIC_HIDDEN, len(pairs)], [(s, 7) for s in seeds])
    opt = EnsembleAdam(list(head.parameters()), torch.full((len(seeds),), 1e-3))
    data = Data(kappa, window, "large", seeds)
    gen = torch.Generator().manual_seed(0)

    def residuals(w):
        with torch.no_grad():
            z = m.encode(w, 0)
            us = m.rollout(z, max(SHORT_HORIZONS))
            boot = m.bootstraps(w)
            r = torch.stack([m.decode(us[k - 1]) - boot[k, p] for k, p in pairs], -1)
        return z, r

    for _ in range(CRITIC_STEPS):
        idx = torch.randint(0, data.train.shape[1], (len(seeds), 512), generator=gen)
        w = data.train[torch.arange(len(seeds))[:, None], idx]
        z, r = residuals(w)
        loss = (head(z) - r).pow(2).mean((1, 2)).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    w = data.test[None].expand(len(seeds), -1, -1)
    z, r = residuals(w)
    with torch.no_grad():
        h = head(z)
    return {
        "critic": float(h.pow(2).mean()),
        "residual": float(r.pow(2).mean()),
    }


def h5():
    rows = {}
    for k, w in configs() + configs(gru=True):
        c = critic(k, w)
        one, direct = primary(k, w, "onestep"), primary(k, w, "direct")
        if c is None or one is None or direct is None:
            continue
        rows[config_name(k, w)] = {
            **c,
            "gap": mean_of(one) - mean_of(direct),
            "markov": is_markov(k, w),
        }
    if len(rows) < 3:
        return {"rows": rows}
    names = list(rows)

    def rank(xs):
        return np.argsort(np.argsort(xs))

    crit = rank([rows[n]["critic"] for n in names])
    gap = rank([rows[n]["gap"] for n in names])
    spearman = float(np.corrcoef(crit, gap)[0, 1])
    markov = [rows[n]["critic"] for n in names if rows[n]["markov"]]
    others = [rows[n]["critic"] for n in names if not rows[n]["markov"]]
    below = bool(markov and others and max(markov) < min(others))
    return {
        "rows": rows,
        "spearman": spearman,
        "markov_below": below,
        "holds": spearman >= 0.8 and below,
    }


# ---------------------------------------------------------------- descriptive


def fractions():
    """Fraction of the gap closed per configuration and main variant (selected lambda)."""
    out = {}
    for k, w in configs() + configs(gru=True):
        one, direct = (
            mean_of(primary(k, w, "onestep")),
            mean_of(primary(k, w, "direct")),
        )
        if one is None or direct is None or one - direct <= 0:
            continue
        row = {}
        for v in ("onestep+F", "C", "C+F", "DB", "DB+F", "M*", "M*+F"):
            val = mean_of(primary(k, w, v))
            if val is not None:
                row[v] = (one - val) / (one - direct)
        out[config_name(k, w)] = row
    return out


def health():
    """R4: runs stopped by the step cap and diverged rollouts in the final stage."""
    capped, diverged, runs = 0, [], 0
    for path in sorted((RUNS_DIR / "final").glob("*.json")):
        r = json.loads(path.read_text())
        runs += 1
        capped += sum(s < 0 for s in r["stopped_at"])
        d = float(np.mean(r["test"]["diverged"]))
        if d > 0.5:
            diverged.append(path.stem)
    return {"runs": runs, "models_capped": capped, "mostly_diverged": diverged}


def main():
    out = {
        "gaps_stage": {
            config_name(k, w): gap_stage(k, w) for k, w in configs() + configs(gru=True)
        },
        "gap_configs": [config_name(k, w) for k, w in gap_configs()],
        "H1": h1(),
        "H2": h2(),
        "H3": h3(),
        "H4": h4(),
        "H-M*": h_mstar(),
        "H5": h5(),
        "fractions": fractions(),
        "health": health(),
    }
    (RESULTS_DIR / "test_results.json").write_text(json.dumps(out, indent=2))
    for name in ("H1", "H2", "H3", "H4", "H-M*", "H5"):
        h = out[name]
        print(f"\n== {name}: holds={h.get('holds')}")
        if name == "H4":
            for key, r in h["H4b"].items():
                if r is not None:
                    print(
                        f"  H4b {key:41s} {r['mean']:+.4f} ± {r['half']:.4f}  {r['verdict']}"
                    )
        if name == "H1":
            print(
                f"  better {h['better']} of {h['n']}; control improvement {h['control_improvement']:+.4f}; attributed {h['attributed']}"
            )
            for key, r in h["controls"].items():
                if r is not None:
                    print(
                        f"  control {key:37s} {r['mean']:+.4f} ± {r['half']:.4f}  {r['verdict']}"
                    )
        for key, r in h.get("rows", {}).items():
            if r is None:
                print(f"  {key:45s} missing")
            elif "verdict" in r:
                print(f"  {key:45s} {r['mean']:+.4f} ± {r['half']:.4f}  {r['verdict']}")
            else:
                print(f"  {key:45s} {r}")
    print("\n== fraction of the gap closed")
    for c, row in out["fractions"].items():
        print(f"  {c:12s} " + "  ".join(f"{v} {f:.2f}" for v, f in row.items()))
    print("\n== health", out["health"])


if __name__ == "__main__":
    main()
