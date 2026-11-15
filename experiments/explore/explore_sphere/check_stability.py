"""
Check of the two pilots' disagreement on CL+S in the Markov control
(κ = 1, L = 2): test MSE over k = 1..4 was 0.013 in pilot 1 and 0.143 in
pilot 2, while CL agreed to four digits. Hypothesis: the two code paths
compute the sphere penalty in a different order of floating-point sums, and
training through chaotic rollouts amplifies the difference until a seed
lands in a different minimum.

Trains CL+S with each code path and each thread count of the pilots and
reports the validation MSE (k = 1..4) of every seed separately. If the
hypothesis holds, the per-seed values differ between runs mostly through
one or two seeds, and the difference follows the code path (or the
threads), not the method.
"""

import json
import sys

import numpy as np
import pilot
import pilot2
import torch


def per_seed(m, data):
    w = data.val[None].expand(len(pilot.SEEDS), -1, -1)
    with torch.no_grad():
        us = m.rollout(m.encode(w, 0), 4)
        preds = torch.stack([m.decode(u) for u in us])
        ys = torch.stack([m.y(w, k) for k in range(1, 5)])
    return (preds - ys).pow(2).mean((0, 2)).tolist()  # (M,)


def main():
    out = {}
    for path, threads in (("pilot1", 4), ("pilot2", 2), ("pilot1", 2), ("pilot2", 4)):
        torch.set_num_threads(threads)
        if path == "pilot1":
            m, data = pilot.train("CL+S", 1.0, 2)
        else:
            m, data = pilot2.train("CL+S", 1.0, 2)
        key = f"{path} code, {threads} threads"
        out[key] = per_seed(m, data)
        print(
            f"{key:24s} per-seed MSE k=1..4 {np.round(out[key], 4).tolist()}",
            flush=True,
        )
        (pilot.OUT / "check_stability.json").write_text(json.dumps(out, indent=2))
        if len(sys.argv) > 1 and sys.argv[1] == "--quick" and len(out) == 2:
            break


if __name__ == "__main__":
    main()
