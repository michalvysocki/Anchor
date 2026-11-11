import numpy as np
import torch
from torch import nn

from anchor.models import UnitStep, predict

# two-sided 95% Student t quantile for 4 degrees of freedom (5 seeds)
T_95_DF4 = 2.776
DIVERGENCE_THRESHOLD = 10.0


def median_error(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Median over windows of the Euclidean error at each horizon, averaged
    over horizons. Non-finite predictions count as infinite errors.

    Args:
        preds (torch.Tensor): predictions of size (n, k_max, point_dim)
        targets (torch.Tensor): true points of size (n, k_max, point_dim)

    Returns:
        torch.Tensor: scalar error
    """

    errors = torch.linalg.vector_norm(preds - targets, dim=-1)
    errors = torch.nan_to_num(errors, nan=float("inf"))

    return errors.median(dim=0).values.mean()


def direct_median_error(model: nn.Module, windows: torch.Tensor) -> torch.Tensor:
    """Median-based error of direct predictions F(z, k), used to select checkpoints."""
    n, length, point_dim = windows.shape
    k_max = length - 1
    k = torch.arange(1, k_max + 1, device=windows.device).repeat(n)
    starts = windows[:, 0].repeat_interleave(k_max, dim=0)
    preds = predict(model, starts, k, k_max).reshape(n, k_max, point_dim)

    return median_error(preds, windows[:, 1:])


def rollout_median_error(model: nn.Module, windows: torch.Tensor) -> torch.Tensor:
    """Median-based error of free-running rollouts, used to select checkpoints."""
    z = windows[:, 0]
    states = []
    for _ in range(windows.shape[1] - 1):
        z = model(z)
        states.append(z)

    return median_error(torch.stack(states, dim=1), windows[:, 1:])


def path_errors(
    model: torch.nn.Module, windows: torch.Tensor, path: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Euclidean errors of one prediction path against true future points.
    A rollout counts as diverged from the step where a coordinate exceeds
    DIVERGENCE_THRESHOLD or becomes NaN, and its error is infinite from then on.

    Args:
        model (torch.nn.Module): horizon-conditioned model F(z, k)
        windows (torch.Tensor): test windows of size (n, k_max + 1, point_dim)
        path (str): "direct" for F(z, k) or "rollout" for F(., 1)^k

    Returns:
        tuple[np.ndarray, np.ndarray]: errors of size (n, k_max) and the
        predictions of size (n, k_max, point_dim)
    """

    n, length, _ = windows.shape
    k_max = length - 1
    z = windows[:, 0]

    with torch.no_grad():
        if path == "direct":
            preds = torch.stack(
                [
                    predict(model, z, torch.full((n,), k, dtype=torch.int64), k_max)
                    for k in range(1, k_max + 1)
                ],
                dim=1,
            )
        else:
            unit = UnitStep(model, k_max)
            states = []
            for _ in range(k_max):
                z = unit(z)
                states.append(z)
            preds = torch.stack(states, dim=1)

    preds = preds.numpy()
    errors = np.linalg.norm(preds - windows[:, 1:].numpy(), axis=-1)
    diverged = ~np.isfinite(preds).all(axis=-1) | (
        np.abs(preds) > DIVERGENCE_THRESHOLD
    ).any(axis=-1)
    diverged = np.logical_or.accumulate(diverged, axis=1)
    errors[diverged] = np.inf

    return errors, preds


def off_attractor(preds: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Distance from each prediction to the nearest reference attractor point.
    A prediction of the mean of possible futures lies off the attractor, so
    this separates sharp predictions from blurred ones at equal error.

    Args:
        preds (np.ndarray): predictions of size (n, k_max, point_dim)
        reference (np.ndarray): attractor points of size (m, point_dim)

    Returns:
        np.ndarray: distances of size (n, k_max)
    """

    flat = preds.reshape(-1, preds.shape[-1])
    finite = np.isfinite(flat).all(axis=1)
    dist = np.full(len(flat), np.inf)
    ref = torch.from_numpy(reference).float()
    for start in range(0, len(flat), 4096):
        chunk = flat[start : start + 4096]
        ok = finite[start : start + 4096]
        if ok.any():
            d = (
                torch.cdist(
                    torch.from_numpy(chunk[ok]).float(),
                    ref,
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
                .min(dim=1)
                .values
            )
            dist[start : start + 4096][ok] = d.numpy()

    return dist.reshape(preds.shape[:2])


def sharpness(preds: np.ndarray, windows: torch.Tensor) -> np.ndarray:
    """
    Spread of predictions relative to the spread of true points at each
    horizon: std over windows, averaged over coordinates. A prediction of
    the mean gives values near 0, unlike off_attractor this also works for
    one-dimensional observations. Diverged windows are left out.

    Args:
        preds (np.ndarray): predictions of size (n, k_max, point_dim)
        windows (torch.Tensor): test windows of size (n, k_max + 1, point_dim)

    Returns:
        np.ndarray: ratios of size (k_max,)
    """

    targets = windows[:, 1:].numpy()
    ratios = []
    for k in range(preds.shape[1]):
        ok = np.isfinite(preds[:, k]).all(axis=1) & (
            np.abs(preds[:, k]) <= DIVERGENCE_THRESHOLD
        ).all(axis=1)
        ratios.append((preds[ok, k].std(axis=0) / targets[ok, k].std(axis=0)).mean())

    return np.array(ratios)


def mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean over seeds (first axis) and half-width of the 95% t-interval."""
    half = T_95_DF4 * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])

    return values.mean(axis=0), half


def to_lists(results: dict) -> dict:
    """Converts numpy arrays to lists for JSON."""
    if isinstance(results, dict):
        return {k: to_lists(v) for k, v in results.items()}

    return results.tolist()
