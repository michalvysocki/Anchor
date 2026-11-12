"""
Linear world model on a window of observations and its population losses.

The latent is the window z_t = (y_{t-L+1}, ..., y_t). The one-step model
predicts y_{t+1} = theta . z_t and shifts the window, so its rollout is the
companion matrix M(theta) applied k times. Every prediction and target is a
linear function of the stacked vector Y = (y_{t-L+1}, ..., y_{t+K}), so
every expected squared error is a quadratic form in the covariance of Y.
This gives the population losses and their stop-gradient fixed points
exactly, without sampling; the same functions on sampled windows give the
training losses.
"""

import numpy as np
import torch

# horizons of the auxiliary losses and anchor points: p >= 2 because at
# the one-step optimum p = 1 carries no signal (Remark in the paper)
AUX_HORIZONS = (3, 4)


def bootstrap_pairs(horizons: tuple[int, ...] = AUX_HORIZONS) -> list[tuple[int, int]]:
    """(k, p) pairs of the D target, sampled as k uniform, then p uniform on 2..k-1."""
    return [(k, p) for k in horizons for p in range(2, k)]


def pair_weights(horizons: tuple[int, ...] = AUX_HORIZONS) -> list[float]:
    """Sampling probability of each pair of bootstrap_pairs."""
    return [1 / len(horizons) / (k - 2) for k, _ in bootstrap_pairs(horizons)]


def companion(theta: torch.Tensor) -> torch.Tensor:
    """M(theta): shifts the window by one and appends theta . z."""
    window = theta.shape[-1]
    shift = torch.zeros(window, window, dtype=theta.dtype)
    shift[:-1, 1:] = torch.eye(window - 1, dtype=theta.dtype)

    return torch.cat([shift[:-1], theta[None]], dim=0)


def rollout_row(theta: torch.Tensor, steps: int) -> torch.Tensor:
    """Row vector r with r . z_t = prediction of y_{t+steps} by the rollout."""
    m = companion(theta)
    row = torch.zeros(theta.shape[-1], dtype=theta.dtype)
    row[-1] = 1.0

    return row @ torch.linalg.matrix_power(m, steps)


def embed(row: torch.Tensor, start: int, size: int) -> torch.Tensor:
    """Places a row acting on z_{t+start} into a functional on Y of length size."""
    out = torch.zeros(size, dtype=row.dtype)
    out[start : start + row.shape[-1]] = row

    return out


def truth(window: int, k: int, size: int, dtype=torch.float64) -> torch.Tensor:
    """Functional picking y_{t+k} from Y."""
    out = torch.zeros(size, dtype=dtype)
    out[window + k - 1] = 1.0

    return out


def quadratic(alpha: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """E[(alpha . Y)^2] for a zero-mean Y with covariance cov."""
    return alpha @ cov @ alpha


def one_step_loss(theta: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    size = cov.shape[0]
    pred = embed(rollout_row(theta, 1), 0, size)

    return quadratic(pred - truth(theta.shape[-1], 1, size), cov)


def multistep_loss(theta: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """C: rollout to k against the true y_{t+k}, k uniform on AUX_HORIZONS."""
    size, window = cov.shape[0], theta.shape[-1]
    terms = [
        quadratic(embed(rollout_row(theta, k), 0, size) - truth(window, k, size), cov)
        for k in AUX_HORIZONS
    ]

    return torch.stack(terms).mean()


def bootstrap_loss(
    theta: torch.Tensor, cov: torch.Tensor, target_theta: torch.Tensor | None = None
) -> torch.Tensor:
    """
    D: rollout to k against sg[rollout of k - p from the true window at
    t + p]. The target uses target_theta, theta under stop-gradient by
    default; passing it explicitly lets fixed_point differentiate the
    stop-gradient update field itself.
    """

    size = cov.shape[0]
    if target_theta is None:
        target_theta = theta.detach()
    terms = [
        w
        * quadratic(
            embed(rollout_row(theta, k), 0, size)
            - embed(rollout_row(target_theta, k - p), p, size),
            cov,
        )
        for (k, p), w in zip(bootstrap_pairs(), pair_weights())
    ]

    return torch.stack(terms).sum()


AUX_LOSSES = {"onestep": None, "C": multistep_loss, "D": bootstrap_loss}


def anchor_points(k: int, first: int = 2) -> list[int]:
    """
    Anchor points of the mixture: p = first..k, where p = k is the true
    future (C). first = 2 by default; first = 1 is the ablation that keeps
    the point that carries no signal at the one-step optimum.
    """
    return list(range(first, k + 1))


def first_point(k: int, w: np.ndarray) -> int:
    """First anchor point of a weight vector for horizon k (weights end at p = k)."""
    return k - len(w) + 1


def uniform_weights(horizons: tuple[int, ...] = AUX_HORIZONS) -> dict[int, np.ndarray]:
    """Equal weights over the anchor points of every horizon (variant M)."""
    return {k: np.full(k - 1, 1 / (k - 1)) for k in horizons}


def target_row(theta: torch.Tensor, k: int, p: int, size: int) -> torch.Tensor:
    """Functional of the target T_p for horizon k: y_{t+k} if p = k, else g^{k-p}(z_{t+p})."""
    if p == k:
        return truth(theta.shape[-1], k, size, theta.dtype)

    return embed(rollout_row(theta, k - p), p, size)


def mixture_loss(
    theta: torch.Tensor,
    cov: torch.Tensor,
    weights: dict[int, np.ndarray],
    target_theta: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Anchor: rollout to k against sg[sum_p w_p T_p] over the anchor points,
    k uniform on AUX_HORIZONS. Weights on p = k only give C.
    """

    size = cov.shape[0]
    if target_theta is None:
        target_theta = theta.detach()
    terms = []
    for k, w in weights.items():
        target = sum(
            float(wp) * target_row(target_theta, k, p, size)
            for wp, p in zip(w, anchor_points(k, first_point(k, w)))
        )
        terms.append(quadratic(embed(rollout_row(theta, k), 0, size) - target, cov))

    return torch.stack(terms).mean()


def target_moments(
    theta: np.ndarray, cov: np.ndarray, k: int, first: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bias and variance of the anchor targets of horizon k given z_t.

    The bias of T_p is E[T_p | z_t] - E[y_{t+k} | z_t], the difference of two
    regressions on z_t (two critics); its second moments form B. The
    variance is the covariance of the regression residuals, V.

    Returns:
        tuple: B and V, both of size (k - first + 1, k - first + 1), in
        units of Var(y)
    """

    window, size = len(theta), cov.shape[0]
    cov_n = cov / cov[0, 0]
    past = cov_n[:window, :window]
    th = torch.as_tensor(theta, dtype=torch.float64)
    rows = np.stack(
        [target_row(th, k, p, size).numpy() for p in anchor_points(k, first)]
    )
    betas = np.linalg.solve(past, (rows @ cov_n)[:, :window].T).T
    bias = betas - betas[-1]
    residuals = rows.copy()
    residuals[:, :window] -= betas

    return bias @ past @ bias.T, residuals @ cov_n @ residuals.T


def simplex_minimizer(q: np.ndarray) -> np.ndarray:
    """argmin w' q w over the probability simplex, by an active-set method."""
    active = np.ones(len(q), dtype=bool)
    while True:
        sub = q[np.ix_(active, active)]
        w_sub = np.linalg.lstsq(sub, np.ones(active.sum()), rcond=None)[0]
        w_sub = w_sub / w_sub.sum()
        if (w_sub >= -1e-12).all():
            w = np.zeros(len(q))
            w[active] = np.clip(w_sub, 0, None)
            return w / w.sum()
        idx = np.flatnonzero(active)
        active[idx[np.argmin(w_sub)]] = False


def bias_variance_weights(
    theta: np.ndarray, cov: np.ndarray, n_eff: float, first: int = 2
) -> dict[int, np.ndarray]:
    """
    Variant M*: weights minimizing the error of the mixed target,
    w' (B + V / n_eff) w, for every horizon. As n_eff grows the variance
    term vanishes and the true future (C, zero bias) is chosen.
    """

    weights = {}
    for k in AUX_HORIZONS:
        b, v = target_moments(theta, cov, k, first)
        weights[k] = simplex_minimizer(b + v / n_eff)

    return weights


def update_field(
    variant: str,
    lam: float,
    cov: torch.Tensor,
    weights: dict[int, np.ndarray] | None = None,
):
    """
    G(theta): gradient of one_step_loss + lam * aux with respect to the
    trained theta, with the target evaluated at the same theta. Training
    with stop-gradient follows -G; its zeros are the fixed points. The
    mixtures (M, M* and its ablations) take their anchor weights from
    `weights`; the ablation D_nosg is D with the gradient flowing through
    the target as well.
    """

    def total(trained: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = one_step_loss(trained, cov)
        if variant == "C":
            loss = loss + lam * multistep_loss(trained, cov)
        elif variant == "D":
            loss = loss + lam * bootstrap_loss(trained, cov, target)
        elif variant == "D_nosg":
            loss = loss + lam * bootstrap_loss(trained, cov, trained)
        elif variant.startswith("M"):
            loss = loss + lam * mixture_loss(trained, cov, weights, target)
        return loss

    return lambda theta: torch.func.grad(total)(theta, theta)


def fixed_point(
    variant: str,
    lam: float,
    cov: np.ndarray,
    window: int,
    n_eff: float | None = None,
    n_continuation: int = 40,
    tol: float = 1e-12,
) -> dict:
    """
    Stop-gradient fixed point of one_step_loss + lam * aux, followed by
    Newton's method from the one-step optimum while lam is raised
    geometrically, so the branch connected to lam = 0 is tracked. For M*
    the weights are recomputed at the current theta at every continuation
    step and then until they stop changing (they are under stop-gradient,
    like the target).

    Args:
        n_eff (float | None): effective sample size of M*'s weights. In the
            population the oracle ablation M*_oracle is M* itself.

    Returns:
        dict: "theta"; "residual", the norm of the update field there;
        "stable", whether gradient descent with a small step is attracted
        to it (every eigenvalue of the Jacobian of the field, at fixed
        weights, has a positive real part); "weights" for M and M*
    """

    cov_t = torch.as_tensor(cov / cov[0, 0], dtype=torch.float64)
    theta = torch.tensor(one_step_solution(cov, window), dtype=torch.float64)
    if variant == "onestep" or lam == 0:
        lams = [0.0]
    else:
        lams = np.geomspace(lam * 1e-4, lam, n_continuation)

    def current_weights():
        if variant == "M":
            return uniform_weights()
        if variant.startswith("M*"):
            first = 1 if variant == "M*_p1" else 2
            return bias_variance_weights(theta.numpy(), cov, n_eff, first)
        return None

    def newton(current_lam, weights):
        nonlocal theta
        field = update_field(variant, current_lam, cov_t, weights)
        for _ in range(100):
            g = field(theta)
            if g.norm() < tol:
                break
            theta = theta - torch.linalg.solve(torch.func.jacrev(field)(theta), g)

    for current in lams:
        newton(float(current), current_weights())
    weights = current_weights()
    if variant.startswith("M*"):
        for _ in range(100):
            newton(float(lams[-1]), weights)
            new = current_weights()
            change = max(np.abs(new[k] - weights[k]).max() for k in new)
            weights = new
            if change < 1e-10:
                break
    field = update_field(variant, float(lams[-1]), cov_t, weights)
    eig = torch.linalg.eigvals(torch.func.jacrev(field)(theta))

    return {
        "theta": theta.numpy(),
        "residual": field(theta).norm().item(),
        "stable": bool((eig.real > 0).all()),
        "weights": weights,
    }


def one_step_solution(cov: np.ndarray, window: int) -> np.ndarray:
    """Least-squares one-step coefficients on the window (population)."""
    past = cov[:window, :window]
    cross = cov[:window, window]

    return np.linalg.solve(past, cross)


def rollout_mse(theta: np.ndarray, cov: np.ndarray, horizon: int) -> np.ndarray:
    """MSE of the rollout at k = 1..horizon, in units of Var(y)."""
    window = len(theta)
    cov_t = torch.as_tensor(cov / cov[0, 0], dtype=torch.float64)
    th = torch.as_tensor(theta, dtype=torch.float64)
    size = cov.shape[0]

    return np.array(
        [
            quadratic(
                embed(rollout_row(th, k), 0, size) - truth(window, k, size), cov_t
            ).item()
            for k in range(1, horizon + 1)
        ]
    )


def direct_mse(cov: np.ndarray, window: int, horizon: int) -> np.ndarray:
    """MSE of the best linear prediction of y_{t+k} from the window, units of Var(y)."""
    past = cov[:window, :window]
    out = []
    for k in range(1, horizon + 1):
        cross = cov[:window, window + k - 1]
        out.append(cov[0, 0] - cross @ np.linalg.solve(past, cross))

    return np.array(out) / cov[0, 0]


def critic_and_residual(theta: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    """
    Mean over the D pairs of the squared critic, E[(E[R | z_t])^2], and of
    the raw squared residual E[R^2], in units of Var(y). The critic is the
    regression of the D target on z_t, so it carries no target variance.
    """

    window = len(theta)
    size = cov.shape[0]
    cov_n = cov / cov[0, 0]
    th = torch.as_tensor(theta, dtype=torch.float64)
    past = cov_n[:window, :window]
    critic, residual = 0.0, 0.0
    for (k, p), w in zip(bootstrap_pairs(), pair_weights()):
        pred = embed(rollout_row(th, k), 0, size).numpy()
        target = embed(rollout_row(th, k - p), p, size).numpy()
        diff = pred - target
        # E[R | z_t] = coefficients of the regression of R on z_t
        beta = np.linalg.solve(past, (cov_n @ diff)[:window])
        critic += w * beta @ past @ beta
        residual += w * diff @ cov_n @ diff

    return critic, residual


def max_horizon_size(window: int, horizon: int) -> int:
    """Length of Y needed for a window and an evaluation horizon."""
    return window + max(horizon, *AUX_HORIZONS)
