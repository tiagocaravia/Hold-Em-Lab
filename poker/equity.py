import math
import random
from dataclasses import dataclass

import numpy as np

from .deck import Deck
from .evaluator import hand_value
from .tables import hand_label

Z_95 = 1.96


def key(card):
    return (card.rank, card.suit)


# Opponent models: deal(deck, n) -> (hands still in, every card dealt)

class RandomOpponents: # Any two cards, nobody folds
    name = "random"

    def deal(self, deck, num_opponents):
        cards = random.sample(deck, 2 * num_opponents)
        return [cards[2 * i: 2 * i + 2] for i in range(num_opponents)], cards


class FoldingOpponents: # Any two cards, weak hands fold; their cards stay dead
    name = "real table"

    def __init__(self, keep):
        self.keep = keep  # hand labels that stay in, e.g. set(load_survivors(n)["Hand"])

    def deal(self, deck, num_opponents):
        hands, cards = RandomOpponents().deal(deck, num_opponents)
        return [h for h in hands if hand_label(*h) in self.keep], cards


class RangeOpponents: # Hands drawn from ranges: one for every seat, or one per seat; weights make some hands likelier
    def __init__(self, pool, name="range", weights=None, seats=None):
        # pool, weights: every seat's hands and how likely each is (None: all equally likely)
        # seats: [(pool, weights) or None, ...] one per seat instead (None: any two cards); the last covers any extra seats
        self.seats = seats or [(pool, weights)]
        self.name = name
        self._deck_id = None

    def deal(self, deck, num_opponents):
        if self._deck_id != id(deck):  # usually the same deck every trial: filter each range against it once
            live = {key(c) for c in deck}
            self._options = [None if seat is None else
                             [(h, 1.0 if seat[1] is None else seat[1][i]) for i, h in enumerate(seat[0])
                              if key(h[0]) in live and key(h[1]) in live and (seat[1] is None or seat[1][i] > 0)]
                             for seat in self.seats]
            self._deck_id = id(deck)
        hands, taken = [], set()
        for i in range(num_opponents):
            options = self._options[min(i, len(self._options) - 1)]
            if options is None:  # any two cards left
                hand = random.sample([c for c in deck if key(c) not in taken], 2)
            else:
                available = [(h, w) for h, w in options if key(h[0]) not in taken and key(h[1]) not in taken]
                if not available:
                    raise ValueError("Opponent range has no hands left after removing known cards.")
                weighted = self.seats[min(i, len(self.seats) - 1)][1] is not None
                hand = (random.choices([h for h, _ in available], weights=[w for _, w in available])[0] if weighted
                        else random.choice(available)[0])
            hands.append(hand)
            taken |= {key(hand[0]), key(hand[1])}
        return hands, [c for h in hands for c in h]


def pot_share(my_rank, opponent_ranks): # 1 win, 0 loss, 1/k split
    if not opponent_ranks:
        return 1.0
    best_opp = max(opponent_ranks)
    if my_rank > best_opp:
        return 1.0
    if my_rank < best_opp:
        return 0.0
    return 1 / (1 + sum(1 for r in opponent_ranks if r == best_opp))


@dataclass
class EquityEstimate:
    equity: float          # 0-1, average share of the pot
    half_width: float      # 95% CI half-width, 0-1
    sims: int
    model: str
    win: float = 0.0       # 0-1, includes pots everyone folded to you
    tie: float = 0.0
    loss: float = 0.0
    uncontested: float = 0.0
    opponents_left: float = 0.0

    def __str__(self):
        return f"{self.equity * 100:.1f}% ± {self.half_width * 100:.1f}"


def run_hands(hole_cards, board, num_opponents, num_simulations, model=None): # Pot share of each simulated hand
    # hole_cards=None deals us a random hand each time: what an average hand gets
    model = model or RandomOpponents()
    deck = [c for c in Deck().cards if c not in (hole_cards or []) + board]
    cards_needed = 5 - len(board)
    shares = np.empty(num_simulations)
    uncontested = opponents = 0
    for i in range(num_simulations):
        live, mine = deck, hole_cards
        if mine is None:
            mine = random.sample(deck, 2)
            live = [c for c in deck if c is not mine[0] and c is not mine[1]]
        hands, dealt = model.deal(live, num_opponents)
        used = {key(c) for c in dealt}
        runout = board + random.sample([c for c in live if key(c) not in used], cards_needed)
        shares[i] = pot_share(hand_value(mine + runout), [hand_value(h + runout) for h in hands])
        uncontested += not hands
        opponents += len(hands)
    return shares, uncontested / num_simulations, opponents / num_simulations


def estimate_equity(hole_cards, board, num_opponents, num_simulations=5000, model=None): # Equity + 95% CI vs a model
    model = model or RandomOpponents()
    shares, uncontested, opponents = run_hands(hole_cards, board, num_opponents, num_simulations, model)
    n = len(shares)
    half_width = Z_95 * shares.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return EquityEstimate(shares.mean(), half_width, n, model.name,
                          win=(shares == 1).mean(), tie=((shares > 0) & (shares < 1)).mean(),
                          loss=(shares == 0).mean(), uncontested=uncontested, opponents_left=opponents)
