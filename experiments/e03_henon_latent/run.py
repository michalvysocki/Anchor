"""
Training of e03 (README, Setup and Training). Stages:

- gap: onestep and direct, one selection seed, every configuration (finding 7)
- select: every variant x lambda x selection seeds
- final: every variant at its selected lambda x evaluation seeds
- smoke: a few hundred steps of every variant, for shapes, timing and gradients

Each call trains one (configuration, variant, arm) as a batch of independent
models and writes results/runs/<stage>/<name>.json.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from data import observations, windows
from design import (
    ABLATION_CONFIGS,
    ABLATIONS,
    BATCH,
    CLIP,
    CRITIC_LR_FACTOR,
    EVAL_EVERY,
    EVALUATION_SEEDS,
    GRU_VARIANTS,
    KAPPAS,
    LAMBDAS,
    LONG_HORIZONS,
    LR,
    LR_FACTOR,
    LR_STOP,
    MAX_STEPS,
    MIN_REL_IMPROVEMENT,
    MSTAR_EVERY,
    MSTAR_WARMUP,
    N_REF_TRAJ,
    N_TEST_TRAJ,
    N_TRAJ,
    N_VAL_TRAJ,
    PATIENCE,
    REF_TAG,
    SELECTION_SEEDS,
    SHORT_HORIZONS,
    SIZES,
    TEST_TAG,
    TRAIN_TAG,
    VAL_TAG,
    VARIANCE_CONFIGS,
    VARIANCE_VARIANTS,
    WINDOWS,
    config_name,
    is_markov,
    main_variants,
)

from anchor.ensemble import EnsembleAdam
from anchor.latent import DirectModels, WorldModels, base

EXPERIMENT_DIR = Path(__file__).resolve().parent
RUNS_DIR = EXPERIMENT_DIR / "results" / "runs"
ARMS = {
    "main": ("large", SHORT_HORIZONS),
    "small": ("small", SHORT_HORIZONS),
    "long": ("large", LONG_HORIZONS),
}


def plan() -> list[tuple[float, int | None, str, str]]:
    """
    Every (kappa, window, variant, arm) of the design; window None is the
    GRU arm. The direct reference is trained wherever onestep is, so the
    fraction of the gap closed uses a reference with the same seeds.
    """
    out = [
        (k, w, v, "main")
        for k in KAPPAS
        for w in WINDOWS
        for v in (*main_variants(), "direct")
    ]
    out += [(k, w, v, "main") for k, w in ABLATION_CONFIGS for v in ABLATIONS]
    out += [(k, None, v, "main") for k in KAPPAS for v in (*GRU_VARIANTS, "direct")]
    out += [
        (k, w, v, arm)
        for k, w in VARIANCE_CONFIGS
        for arm in ("small", "long")
        # the long arm has the large data of the main arm, whose onestep and
        # direct runs it shares (neither depends on the training horizons)
        for v in (
            (*VARIANCE_VARIANTS, "direct")
            if arm == "small"
            else [v for v in VARIANCE_VARIANTS if v != "onestep"]
        )
    ]
    return out


class Data:
    """Training windows per seed, plus the shared validation, test and reference sets."""

    def __init__(
        self, kappa: float, window: int | None, data_arm: str, seeds: list[int]
    ):
        past = SIZES.past(window)
        # trajectories depend on the arm and seed only, so configurations of
        # one seed share them; the noise also depends on kappa (observations)
        arm_idx = list(N_TRAJ).index(data_arm)
        self.train = torch.stack(
            [
                windows(
                    observations(N_TRAJ[data_arm], kappa, TRAIN_TAG, arm_idx, s)[1],
                    past,
                )
                for s in seeds
            ]
        )
        _, y_val = observations(N_VAL_TRAJ, kappa, VAL_TAG)
        self.val = windows(y_val, past)
        x_test, y_test = observations(N_TEST_TRAJ, kappa, TEST_TAG)
        self.test, self.test_clean = windows(y_test, past), windows(x_test, past)
        self.ref = None
        self.past, self.kappa = past, kappa

    def reference(self):
        if self.ref is None:
            self.ref = windows(
                observations(N_REF_TRAJ, self.kappa, REF_TAG)[1], self.past
            )
        return self.ref

    def batch(self, gens: list[torch.Generator], model_seed: torch.Tensor):
        """
        One batch per model, (M, BATCH, width). Each seed has its own
        generator, so a model's batches depend only on its seed: the models
        of one seed (any lambda, any variant, any run size) see the same
        sequence of batches from the same training set.
        """

        n = self.train.shape[1]
        idx = torch.stack(
            [torch.randint(0, n, (BATCH,), generator=g) for g in gens]
        ).to(self.train.device)
        return self.train[model_seed[:, None], idx[model_seed]]


def expand(w: torch.Tensor, n_models: int) -> torch.Tensor:
    return w[None].expand(n_models, -1, -1)


def evaluate(models, w: torch.Tensor, lam: torch.Tensor, chunk: int = 4096):
    """Own loss and rollout MSE (k = 1..10) on a shared set, per model."""
    n_models = lam.shape[0]
    own, mse, count = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(w), chunk):
            part = expand(w[i : i + chunk], n_models)
            loss, _ = models.losses(part, lam)
            n = part.shape[1]
            own = own + loss * n
            mse = mse + models.rollout_mse(part)["mse"].mean(-1) * n
            count += n
    return own / count, mse / count


def train(
    kappa,
    window,
    variant,
    arm,
    lams,
    seeds,
    max_steps=MAX_STEPS,
    log_grads=False,
    device="cpu",
):
    data_arm, horizons = ARMS[arm]
    if variant == "C+F_h34":
        horizons = (3, 4)
    n_models = len(lams) * len(seeds)
    model_seeds = [s for _ in lams for s in seeds]
    lam = torch.tensor([l for l in lams for _ in seeds], device=device)
    data = Data(kappa, window, data_arm, seeds)
    data.train = data.train.to(device)
    # row of data.train (and generator) of every model
    model_seed = torch.tensor([seeds.index(s) for s in model_seeds], device=device)
    data.val, data.test, data.test_clean = (
        x.to(device) for x in (data.val, data.test, data.test_clean)
    )
    # initialization keyed by the seed alone: the models of one seed start
    # from the same parameters at every lambda and in every run size
    init = model_seeds
    if variant == "direct":
        models = DirectModels(window, init, SIZES)
    else:
        models = WorldModels(variant, window, horizons, init, SIZES)
    models.to(device)
    if hasattr(models, "weights"):
        models.weights = {k: v.to(device) for k, v in models.weights.items()}
    opt = EnsembleAdam(
        models.model_parameters(), torch.full((n_models,), LR, device=device), clip=CLIP
    )
    crit = (
        EnsembleAdam(
            models.critic_parameters(),
            torch.full((n_models,), LR * CRITIC_LR_FACTOR, device=device),
            clip=CLIP,
        )
        if models.critic_parameters()
        else None
    )
    gens = [
        torch.Generator().manual_seed(
            int(
                np.random.SeedSequence(
                    [round(100 * kappa), window or 0, list(N_TRAJ).index(data_arm), s]
                ).generate_state(1)[0]
            )
        )
        for s in seeds
    ]
    # reference batches of the oracle have their own generator, so every
    # variant of a configuration sees the same sequence of training batches
    ref_gen = torch.Generator().manual_seed(int(1e6 * kappa) + (window or 0) + 1)
    best = torch.full((n_models,), float("inf"), device=device)
    stale = torch.zeros(n_models, dtype=torch.long, device=device)
    stopped_at = torch.full((n_models,), -1, device=device)
    curve, grads = [], []
    n_windows = data.train.shape[1]
    t0 = time.time()
    for step in range(1, max_steps + 1):
        w = data.batch(gens, model_seed)
        if base(variant) == "M*" and step > MSTAR_WARMUP and step % MSTAR_EVERY == 1:
            models.update_weights(w, n_windows)
        loss, closs = models.losses(w, lam)
        params = models.model_parameters()
        g = torch.autograd.grad(
            loss.sum(), params, allow_unused=True, retain_graph=crit is not None
        )
        for p, gi in zip(params, g):
            p.grad = gi
        if crit is not None:
            cw = (
                expand(
                    data.reference()[
                        torch.randint(
                            0, len(data.reference()), (4 * BATCH,), generator=ref_gen
                        )
                    ].to(device),
                    n_models,
                )
                if variant.endswith("_oracle")
                else w
            )
            closs = (
                models.critic_loss(cw, models.encode(cw, 0)) if cw is not w else closs
            )
            cg = torch.autograd.grad(closs.sum(), crit.params, allow_unused=True)
            for p, gi in zip(crit.params, cg):
                p.grad = gi
            crit.step()
        if log_grads and step % 50 == 0:
            one = models.losses(w, torch.zeros_like(lam))[0]
            g_one = torch.autograd.grad(
                one.sum(), params, allow_unused=True, retain_graph=True
            )
            grads.append(
                {
                    "step": step,
                    "total": per_model_norm(g, n_models).tolist(),
                    "one_step": per_model_norm(g_one, n_models).tolist(),
                }
            )
        opt.step()
        if step % EVAL_EVERY == 0 or step == max_steps:
            own, val_mse = evaluate(models, data.val, lam)
            improved = own < best * (1 - MIN_REL_IMPROVEMENT)
            best = torch.where(improved, own, best)
            stale = torch.where(improved, torch.zeros_like(stale), stale + 1)
            reduce = (stale >= PATIENCE) & (opt.lr > 0)
            opt.lr = torch.where(reduce, opt.lr * LR_FACTOR, opt.lr)
            stale = torch.where(reduce, torch.zeros_like(stale), stale)
            done = (opt.lr < LR_STOP) & (opt.lr > 0)
            stopped_at = torch.where(
                done & (stopped_at < 0), torch.full_like(stopped_at, step), stopped_at
            )
            opt.lr = torch.where(done, torch.zeros_like(opt.lr), opt.lr)
            if crit is not None:
                crit.lr = opt.lr * CRITIC_LR_FACTOR
            curve.append(
                {
                    "step": step,
                    "own": own.tolist(),
                    "val_mse": val_mse.tolist(),
                    "lr": opt.lr.tolist(),
                }
            )
            if (opt.lr == 0).all():
                break
        opt.zero_grad()
        if crit is not None:
            crit.zero_grad()
    test = {}
    with torch.no_grad():
        parts = [
            models.rollout_mse(
                expand(data.test[i : i + 4096], n_models),
                expand(data.test_clean[i : i + 4096], n_models),
            )
            for i in range(0, len(data.test), 4096)
        ]
        sizes = [min(4096, len(data.test) - i) for i in range(0, len(data.test), 4096)]
        for key in parts[0]:
            test[key] = (
                sum(p[key] * s for p, s in zip(parts, sizes)) / sum(sizes)
            ).tolist()
    result = {
        "kappa": kappa,
        "window": window,
        "variant": variant,
        "arm": arm,
        "horizons": list(horizons),
        "lams": [float(x) for x in lam],
        "seeds": model_seeds,
        "steps": step,
        "stopped_at": stopped_at.tolist(),
        "seconds": time.time() - t0,
        "device": device,
        "machine": platform.node(),
        "threads": torch.get_num_threads(),
        "test": test,
        "curve": curve,
        "grads": grads,
        "weights": {k: v.tolist() for k, v in getattr(models, "weights", {}).items()}
        if base(variant) == "M*"
        else None,
    }
    return result, models


def per_model_norm(grads, n_models):
    return sum(
        g.reshape(n_models, -1).pow(2).sum(1) for g in grads if g is not None
    ).sqrt()


def run_name(kappa, window, variant, arm):
    return f"{config_name(kappa, window)}_{variant}_{arm}"


def save(stage, result, models):
    """Results as JSON and the trained parameters (CPU) next to them."""
    out = (
        RUNS_DIR
        / stage
        / (
            run_name(
                result["kappa"], result["window"], result["variant"], result["arm"]
            )
            + ".json"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))
    state = {k: v.detach().cpu() for k, v in models.state_dict().items()}
    torch.save(
        {"state": state, "weights": getattr(models, "weights", None)},
        out.with_suffix(".pt"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke", "gap", "select", "final"])
    parser.add_argument("--only", default="", help="substring filter on run names")
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--shard", default="1/1", help="i/n: every n-th job from i")
    parser.add_argument(
        "--force", action="store_true", help="rerun jobs whose results exist"
    )
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    if args.stage == "gap":
        jobs = [
            (k, w, v, "main")
            for k in KAPPAS
            for w in (*WINDOWS, None)
            for v in ("onestep", "direct")
        ]
    else:
        jobs = plan()
    jobs = [j for j in jobs if args.only in run_name(*j)]
    for kappa, window, variant, arm in jobs[shard - 1 :: n_shards]:
        name = run_name(kappa, window, variant, arm)
        # resume after an interruption: a job counts as done once its
        # parameters are saved (they are written after the JSON)
        if not args.force and (RUNS_DIR / args.stage / f"{name}.pt").exists():
            print(f"{name:40s} done, skipped", flush=True)
            continue
        if args.stage == "smoke":
            lams, seeds = [1.0], [0]
        elif args.stage == "gap":
            lams, seeds = [0.0], [SELECTION_SEEDS[0]]
        elif args.stage == "select":
            lams = [0.0] if base(variant) in ("onestep", "direct") else list(LAMBDAS)
            seeds = list(SELECTION_SEEDS)
        else:
            seeds = list(EVALUATION_SEEDS)
            if (
                arm == "main"
                and window is not None
                and is_markov(kappa, window)
                and base(variant) not in ("onestep", "direct")
            ):
                # H2 judges harm at every lambda in the Markov controls
                lams = list(LAMBDAS)
            else:
                lams = [selected_lambda(kappa, window, variant, arm)]
        result, models = train(
            kappa,
            window,
            variant,
            arm,
            lams,
            seeds,
            max_steps=args.steps,
            log_grads=args.stage == "smoke",
            device=args.device,
        )
        save(args.stage, result, models)
        print(
            f"{name:40s} steps {result['steps']:6d} {result['seconds']:7.1f}s test mse {np.mean(result['test']['mse']):.4f}",
            flush=True,
        )


def selected_lambda(kappa, window, variant, arm) -> float:
    """Lowest mean validation rollout MSE (last evaluation) over the selection seeds."""
    if base(variant) == "onestep" or variant == "direct":
        return 0.0
    r = json.loads(
        (
            RUNS_DIR / "select" / (run_name(kappa, window, variant, arm) + ".json")
        ).read_text()
    )
    val = np.array(r["curve"][-1]["val_mse"]).reshape(len(LAMBDAS), -1).mean(1)
    # a lambda whose training diverged (NaN) must never be selected
    val = np.where(np.isfinite(val), val, np.inf)
    return float(LAMBDAS[int(np.argmin(val))])


if __name__ == "__main__":
    main()
