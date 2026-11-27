import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so "poker" imports work

import pytest

from poker.equity import estimate_equity, pot_share
from poker.prompts import cards_from as cards


def test_pot_share():
    assert pot_share(10, []) == 1.0
    assert pot_share(10, [5, 7]) == 1.0
    assert pot_share(5, [10]) == 0.0
    assert pot_share(10, [10]) == 0.5
    assert pot_share(10, [10, 10, 3]) == 1 / 3


def test_aces_vs_one_random_hand():
    random.seed(0)
    est = estimate_equity(cards("AS AH"), [], num_opponents=1, num_simulations=4000)
    assert abs(est.equity - 0.852) < 0.03  # known exact value ~85.2%
    assert abs(est.win + est.tie + est.loss - 1) < 1e-9


def test_made_royal_flush_always_wins():
    random.seed(0)
    est = estimate_equity(cards("AS KS"), cards("QS JS 10S 2D 3C"), num_opponents=3, num_simulations=500)
    assert est.equity == 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
