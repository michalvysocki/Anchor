"""
CPU-GPU equivalence check of e03, rule of 2026-11-24 (research note,
section A). Compares the select results of five jobs rerun on the club GPU
and the club CPU with the Mac select results, model by model.

Per group (one job, or the pooled jobs of GROUPS) with a gap rule:
d = (MSE_device - MSE_Mac) / gap per model, MSE the
test MSE averaged over k = 1..4, gap the validation gap of the configuration
(onestep - direct over k = 1..4, gap_report.json). Verdict on the 90% CI of
the mean of d against +-0.125:
  within        the interval lies inside [-0.125, 0.125]
  outside       the whole interval lies beyond one bound (a shown shift)
  inconclusive  anything else (decided by the user 2026-11-24, as in TOST)
The Markov control is descriptive: log(GPU / club CPU), log(club CPU / Mac).

Usage: python equiv_analyze.py <mac select dir> <gpu select dir> <cpu select dir> <gap_report.json> <out.json>
"""

import json
import sys
from pathlib import Path

import numpy as np

JOBS = {
    "k0.5_L1_onestep_main": "k0.5_L1",
    "k0.5_L1_C_main": "k0.5_L1",
    "k0.5_gru_C_main": "k0.5_gru",
    "k1.0_L1_C_main": "k1.0_L1",
    # added once after the light group was inconclusive (2026-11-24)
    "k0.5_L2_onestep_main": "k0.5_L2",
}
# the light group is judged on its jobs pooled, each model in units of its
# own configuration's gap; every other job is a group of its own
GROUPS = {"light": ["k0.5_L1_onestep_main", "k0.5_L2_onestep_main"]}
MARKOV = "k1.0_L2_C_main"
BOUND = 0.125
LAMBDAS = (0.1, 1.0, 10.0)
PRIMARY_K = (1, 4)
# one-sided 95% quantiles of Student's t (two-sided 90% interval), by df
T90 = dict(enumerate([np.inf, 6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812], 0))


def load(d, job):
    r = json.loads((Path(d) / f"{job}.json").read_text())
    return {
        "mse": np.array(r["test"]["mse"], float)[:, :4].mean(1),
        "diverged": np.array(r["test"]["diverged"], float),
        "stopped_at": r["stopped_at"],
        "lams": np.array(r["lams"], float),
        "seeds": r["seeds"],
        "val_k": np.array(r["curve"][-1]["val_mse_k"], float),
        "device": r["device"],
        "machine": r["machine"],
    }


def selected_lambda(x):
    """The rule of run.py selected_lambda, on one result."""
    if np.all(x["lams"] == 0):
        # onestep and direct are trained at lambda 0 only
        return 0.0
    a, b = PRIMARY_K
    val = x["val_k"][:, a - 1 : b].mean(1).reshape(len(LAMBDAS), -1).mean(1)
    val = np.where(np.isfinite(val), val, np.inf)
    return float(LAMBDAS[int(np.argmin(val))])


def interval(d):
    d = d[np.isfinite(d)]
    n = len(d)
    half = T90[n - 1] * d.std(ddof=1) / np.sqrt(n)
    return float(d.mean()), float(d.mean() - half), float(d.mean() + half), n


def verdict(lo, hi):
    if -BOUND <= lo and hi <= BOUND:
        return "within"
    return "outside" if lo > BOUND or hi < -BOUND else "inconclusive"


def outcome(gpu, cpu):
    """Outcome of one job; the rule does not cover GPU outside with an inconclusive control."""
    if gpu == "within":
        return "within"
    if gpu == "inconclusive":
        return "inconclusive"
    return {"within": "fail", "outside": "platform"}.get(cpu, "not covered")


def main(mac_dir, gpu_dir, cpu_dir, gap_path, out_path):
    gaps = json.loads(Path(gap_path).read_text())
    report = {"bound": BOUND, "jobs": {}}
    for job, cfg in JOBS.items():
        g = gaps[cfg]
        gap = float(np.mean(g["onestep_val"][:4]) - np.mean(g["direct_val"][:4]))
        mac = load(mac_dir, job)
        row = {"gap": gap, "mac_selected_lambda": selected_lambda(mac)}
        for name, d in (("gpu", gpu_dir), ("cpu", cpu_dir)):
            x = load(d, job)
            assert np.array_equal(x["lams"], mac["lams"]) and x["seeds"] == mac["seeds"], job
            d = (x["mse"] - mac["mse"]) / gap
            mean, lo, hi, n = interval(d)
            row[name] = {
                "d_gap_units": d.tolist(),
                "device": x["device"],
                "machine": x["machine"],
                "mean_gap_units": mean,
                "ci90": [lo, hi],
                "n": n,
                "verdict": verdict(lo, hi),
                "selected_lambda": selected_lambda(x),
                "max_diverged": float(x["diverged"].max()),
                "stopped_at": x["stopped_at"],
            }
        row["mac_max_diverged"] = float(mac["diverged"].max())
        row["mac_stopped_at"] = mac["stopped_at"]
        report["jobs"][job] = row

    mac, gpu, cpu = (load(d, MARKOV) for d in (mac_dir, gpu_dir, cpu_dir))
    report["markov_descriptive"] = {
        "job": MARKOV,
        "log_gpu_over_cpu": interval(np.log(gpu["mse"] / cpu["mse"]))[:3],
        "log_cpu_over_mac": interval(np.log(cpu["mse"] / mac["mse"]))[:3],
        "log_gpu_over_mac": interval(np.log(gpu["mse"] / mac["mse"]))[:3],
    }

    # outcomes of the rule, per group
    pooled = {j for js in GROUPS.values() for j in js}
    groups = {**GROUPS, **{j: [j] for j in JOBS if j not in pooled}}
    report["groups"] = {}
    for name, js in groups.items():
        row = {"jobs": js}
        for dev in ("gpu", "cpu"):
            d = np.concatenate([report["jobs"][j][dev]["d_gap_units"] for j in js])
            mean, lo, hi, n = interval(d)
            row[dev] = {"mean_gap_units": mean, "ci90": [lo, hi], "n": n, "verdict": verdict(lo, hi)}
        row["outcome"] = outcome(row["gpu"]["verdict"], row["cpu"]["verdict"])
        report["groups"][name] = row
    o = {r["outcome"] for r in report["groups"].values()}
    if "fail" in o:
        decision = "GPU fails: final on the professor's CPU"
    elif "not covered" in o:
        decision = "not covered by the rule: GPU outside, control inconclusive; the user decides"
    elif "inconclusive" in o:
        # jobs were added once already (light group); decided by the user
        # 2026-11-24 before the added job ran
        decision = "inconclusive after the added job: final stays on the GPU"
    elif "platform" in o:
        decision = "platform effect: final stays on the GPU, recorded in the e03 README"
    else:
        decision = "final on the GPU"
    report["decision"] = decision
    Path(out_path).write_text(json.dumps(report, indent=2))

    print(f"{'job':22s} {'gap':>7s}  {'device':4s} {'mean':>7s} {'90% CI':>17s}  verdict       lambda (Mac)  max div")
    for job, r in report["jobs"].items():
        for name in ("gpu", "cpu"):
            x = r[name]
            print(
                f"{job:22s} {r['gap']:7.4f}  {name:4s} {x['mean_gap_units']:+7.3f} "
                f"[{x['ci90'][0]:+7.3f}, {x['ci90'][1]:+7.3f}]  {x['verdict']:12s}  "
                f"{x['selected_lambda']:4g} ({r['mac_selected_lambda']:g})   {x['max_diverged']:.3f}"
            )
    print("groups (the decision is taken on these):")
    for name, r in report["groups"].items():
        for dev in ("gpu", "cpu"):
            x = r[dev]
            print(
                f"  {name:22s} {dev:4s} n={x['n']:2d} {x['mean_gap_units']:+7.3f} "
                f"[{x['ci90'][0]:+7.3f}, {x['ci90'][1]:+7.3f}]  {x['verdict']}"
            )
        print(f"  {'':22s} outcome: {r['outcome']}")
    m = report["markov_descriptive"]
    for k in ("log_gpu_over_cpu", "log_cpu_over_mac", "log_gpu_over_mac"):
        mean, lo, hi = m[k]
        print(f"{MARKOV} {k:17s} {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  (descriptive)")
    print("decision:", decision)


if __name__ == "__main__":
    main(*sys.argv[1:6])
