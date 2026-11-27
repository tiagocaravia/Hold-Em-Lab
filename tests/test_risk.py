import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so "poker" imports work

import numpy as np
import pytest

from bots.risk import path, risk_stats, running


def test_risk_stats_basic():
    s = risk_stats([1, -1, 3, -1])
    assert s["mean"] == pytest.approx(0.5)
    assert s["sd"] == pytest.approx(np.std([1, -1, 3, -1], ddof=1))
    assert s["sharpe"] == pytest.approx(0.5 / s["sd"])
    assert s["bb100"] == pytest.approx(50)
    assert s["sd100"] == pytest.approx(s["sd"] * 10)
    assert s["sharpe100"] == pytest.approx(s["sharpe"] * 10)


def test_too_few_hands_is_nan():
    assert np.isnan(risk_stats([3.0])["sharpe"])


def test_running_matches_direct():
    rng = np.random.default_rng(0)
    x = rng.standard_t(3, 2000) + 0.1  # fat tails, like poker results
    mean, sd, sharpe, hw = running(x)
    for n in (10, 500, 2000):
        s = risk_stats(x[:n])
        assert mean[n - 1] == pytest.approx(s["mean"])
        assert sd[n - 1] == pytest.approx(s["sd"])
        assert sharpe[n - 1] == pytest.approx(s["sharpe"])
        assert hw[n - 1] == pytest.approx(s["sharpe_hw"], rel=1e-6)


def test_sharpe_interval_covers_truth():
    # The 95% interval should contain the true Sharpe ratio about 95% of the time
    rng = np.random.default_rng(1)
    true = 0.1
    hits = 0
    for _ in range(400):
        s = risk_stats(rng.normal(true, 1, 1000))
        hits += abs(s["sharpe"] - true) <= s["sharpe_hw"]
    assert 0.9 < hits / 400 < 0.99


def test_risk_path():
    assert path("arena.png") == "arena_risk.png"
    assert path("out/arena") == "out/arena_risk.png"
    assert path(None) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
