"""
Statistics shared by the experiments: t-intervals over seeds for any number
of seeds, paired log ratios and non-inferiority verdicts with a margin
(the protocol of e02, README there). e01 and e01b keep their own copies
(metrics.mean_ci with T_95_DF4), with which their results were produced.
"""

import math

import numpy as np


def t_cdf(x: float, df: int) -> float:
    """CDF of Student's t by Simpson integration of the density from 0 to |x|."""
    log_norm = (
        math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    )
    n = 2000
    h = abs(x) / n
    total = 0.0
    for i in range(n + 1):
        u = i * h
        weight = 1 if i in (0, n) else (4 if i % 2 else 2)
        total += weight * math.exp(log_norm - (df + 1) / 2 * math.log1p(u * u / df))
    half = total * h / 3

    return 0.5 + math.copysign(half, x)


def t_quantile(df: int, prob: float = 0.975) -> float:
    """Quantile of Student's t by bisection on t_cdf."""
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < prob:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2


def mean_ci(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean over seeds along axis and half-width of the two-sided 95% t-interval."""
    values = np.asarray(values)
    n = values.shape[axis]
    half = t_quantile(n - 1) * values.std(axis=axis, ddof=1) / np.sqrt(n)

    return values.mean(axis=axis), half


def paired_log_ratio(errors: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """log(errors / baseline) per seed; both arrays are ordered by the same seeds."""
    return np.log(np.asarray(errors) / np.asarray(baseline))


def verdict(mean: float, half: float, margin: float = math.log(1.02)) -> str:
    """
    Non-inferiority verdict of a log ratio to the baseline with a margin:
    "better" if the interval is below 0, "material harm" if it is above the
    margin, "small increase" if above 0 but not beyond the margin,
    "harmless" if below the margin, "inconclusive" otherwise.
    """

    if mean + half < 0:
        return "better"
    if mean - half > margin:
        return "material harm"
    if mean - half > 0:
        return "small increase"
    if mean + half < margin:
        return "harmless"
    return "inconclusive"
