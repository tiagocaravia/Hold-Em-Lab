import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so "poker" imports work

import pytest

from poker.card import Card, parse_card
from poker.deck import Deck


def test_parse_card():
    card = parse_card("as")
    assert card == Card("A", "Spades")
    assert parse_card(" 10H ") == Card("10", "Hearts")


def test_rank_value_orders_cards():
    assert Card("2", "Clubs").rank_value < Card("K", "Clubs").rank_value < Card("A", "Clubs").rank_value


def test_deck_has_52_unique_cards():
    cards = Deck().cards
    assert len(cards) == 52
    assert len({(c.rank, c.suit) for c in cards}) == 52


def test_deal_removes_cards():
    deck = Deck()
    deck.shuffle()
    hand = deck.deal(2)
    assert len(hand) == 2
    assert len(deck.cards) == 50
    assert all(c not in deck.cards for c in hand)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
