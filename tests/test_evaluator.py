import random

import pytest

from poker.card import parse_card
from poker.deck import Deck
from poker import evaluator
from poker.evaluator import best_hand, category, get_hand_rank, hand_value


def cards(s):
    return [parse_card(c) for c in s.split()]


@pytest.mark.parametrize("hand, expected", [
    ("AS KS QS JS 10S", 8),  # straight flush
    ("9C 9D 9H 9S 2C", 7),   # quads
    ("KC KD KH 4S 4C", 6),   # full house
    ("2H 7H 9H JH KH", 5),   # flush
    ("5C 6D 7H 8S 9C", 4),   # straight
    ("AC 2D 3H 4S 5C", 4),   # wheel
    ("QC QD QH 7S 2C", 3),   # trips
    ("JC JD 4H 4S 9C", 2),   # two pair
    ("10C 10D 4H 8S 2C", 1), # pair
    ("AC JD 8H 5S 3C", 0),   # high card
])
def test_categories(hand, expected):
    assert get_hand_rank(cards(hand))[0] == expected


def test_kickers_break_ties():
    assert get_hand_rank(cards("KC KD 5H 4S 3C")) > get_hand_rank(cards("2C 2D AH KS QC"))
    assert get_hand_rank(cards("AC AD KH 4S 3C")) > get_hand_rank(cards("AH AS QH JS 10C"))


def test_wheel_is_lowest_straight():
    assert get_hand_rank(cards("2C 3D 4H 5S 6C")) > get_hand_rank(cards("AC 2D 3H 4S 5C"))


def test_best_hand_picks_from_seven():
    assert best_hand(cards("AS KS QS JS 10S 2D 3C"))[0] == 8


@pytest.mark.skipif(evaluator._fast is None, reason="phevaluator not installed")
def test_fast_evaluator_agrees_with_reference():
    rng = random.Random(0)
    deck = Deck().cards
    for _ in range(2000):
        a, b = rng.sample(deck, 7), rng.sample(deck, 7)
        ref_a, ref_b = best_hand(a), best_hand(b)
        fast_a, fast_b = hand_value(a), hand_value(b)
        assert category(fast_a) == ref_a[0]
        assert (fast_a > fast_b) == (ref_a > ref_b)
        assert (fast_a == fast_b) == (ref_a == ref_b)
