import math

import numpy as np
import pytest

from anchor import stats


@pytest.mark.parametrize(
    "df, expected",
    [(4, 2.776), (9, 2.262), (29, 2.045), (1000, 1.962)],
)
def test_t_quantile_matches_tables(df, expected):
    assert stats.t_quantile(df) == pytest.approx(expected, abs=1e-3)


def test_mean_ci_uses_the_number_of_seeds():
    values = np.arange(5.0)
    mean, half = stats.mean_ci(values)
    assert mean == 2.0
    assert half == pytest.approx(2.776 * values.std(ddof=1) / math.sqrt(5), abs=1e-3)


def test_verdicts_follow_the_margin():
    margin = math.log(1.02)
    assert stats.verdict(-0.01, 0.005) == "better"
    assert stats.verdict(0.05, 0.01) == "material harm"
    assert stats.verdict(0.01, 0.005) == "small increase"
    assert stats.verdict(0.0, 0.01) == "harmless"
    assert stats.verdict(0.0, margin + 0.01) == "inconclusive"


def test_paired_log_ratio():
    assert np.allclose(stats.paired_log_ratio([2.0, 1.0], [1.0, 1.0]), [math.log(2), 0])
