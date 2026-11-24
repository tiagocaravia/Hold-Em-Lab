import os
import time
import warnings
from itertools import combinations
from multiprocessing import Pool as Workers

import numpy as np

from poker.deck import Deck
from poker.evaluator import best_hand
from poker.profiles import DRAWS, soft_keep
from poker.ranges import draw_type, preflop_percentiles
from poker.tables import hand_label

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "hands.npz")
CARDS = Deck().cards
COMBOS = np.array(list(combinations(range(52), 2)))  # all 1,326 two-card hands, as card ids
FLOPS, TURNS, RIVERS = 300, 6, 6  # 300 flops, 6 turns each, 6 rivers per turn: 10,800 full boards
PLAY_PCT = 0.45  # players see the flop with roughly their top 45% of hands


class Pool:
    def __init__(self, data):
        self.flops, self.turns, self.rivers = data["flops"], data["turns"], data["rivers"]
        self.rank = data["rank"]                    # (F, T, R, 1326) showdown order on that river; -1 = uses a board card
        self.strength = {                           # share of hands each hand beats (on the flop/turn: on average)
            "flop": data["flop_equity"], "turn": data["turn_equity"], "river": data["river_strength"]}
        self.draws = {"flop": data["flop_draw"], "turn": data["turn_draw"]}
        self.played = data["played"]                # how likely each hand is to be played preflop

    def deal(self, rng): # A full board plus two disjoint played hands
        f, t, r = rng.integers(FLOPS), rng.integers(TURNS), rng.integers(RIVERS)
        weights = np.where(self.rank[f, t, r] >= 0, self.played, 0.0)
        weights /= weights.sum()
        while True:
            a, b = rng.choice(len(COMBOS), size=2, p=weights)
            if not set(COMBOS[a]) & set(COMBOS[b]):
                return (f, t, r), a, b


def score_flop(seed): # Everything for one random flop: its turns and rivers, every hand's rank, and draws
    rng = np.random.default_rng(seed)
    flop = rng.choice(52, size=3, replace=False)
    return score_board(flop, None, None, rng)


def score_board(flop, turn, river, rng, turns=TURNS, rivers_per_turn=RIVERS): # Score a flop; unknown turn/river cards are sampled
    turn_cards = np.array([turn]) if turn is not None else rng.choice(np.setdiff1d(np.arange(52), flop), size=turns, replace=False)
    n_rivers = 1 if river is not None else rivers_per_turn
    rivers = np.zeros((len(turn_cards), n_rivers), dtype=np.int8)
    rank = np.full((len(turn_cards), n_rivers, len(COMBOS)), -1, dtype=np.int16)
    strength = np.full((len(turn_cards), n_rivers, len(COMBOS)), np.nan, dtype=np.float32)
    for t, turn_card in enumerate(turn_cards):
        rivers[t] = [river] if river is not None else rng.choice(
            np.setdiff1d(np.arange(52), np.append(flop, turn_card)), size=n_rivers, replace=False)
        for r, river_card in enumerate(rivers[t]):
            board = [*flop, turn_card, river_card]
            live = [k for k, (i, j) in enumerate(COMBOS) if i not in board and j not in board]
            cards = [CARDS[i] for i in board]
            ranks = [best_hand([CARDS[COMBOS[k][0]], CARDS[COMBOS[k][1]]] + cards) for k in live]
            ranks = [(x[0], tuple(x[1])) for x in ranks]
            order = {x: n for n, x in enumerate(sorted(set(ranks)))}
            dense = np.array([order[x] for x in ranks])
            ordered = np.sort(dense)
            below = np.searchsorted(ordered, dense, side="left")
            ties = np.searchsorted(ordered, dense, side="right") - below
            rank[t, r, live] = dense
            strength[t, r, live] = (below + 0.5 * ties) / len(live)
    draw = lambda board: np.array([DRAWS[draw_type([CARDS[i], CARDS[j]], [CARDS[b] for b in board])]
                                   if i not in board and j not in board else 0 for i, j in COMBOS], dtype=np.int8)
    return flop, turn_cards, rivers, rank, strength, draw(flop), np.array([draw([*flop, t]) for t in turn_cards])


def equities(strength): # Flop and turn equity: each hand's average strength over the runouts below it
    with warnings.catch_warnings():  # hands that use a board card have no equity there; that's expected
        warnings.simplefilter("ignore", RuntimeWarning)
        turn = np.nanmean(strength, axis=-2)
        flop = np.nanmean(strength.reshape(*strength.shape[:-3], -1, strength.shape[-1]), axis=-2)
    return flop, turn


def played_weights():
    pct = preflop_percentiles(9)
    return np.array([float(soft_keep(pct[hand_label(CARDS[i], CARDS[j])], PLAY_PCT)) for i, j in COMBOS])


def build(seed=1):
    start = time.time()
    with Workers() as workers:
        parts = workers.map(score_flop, range(seed * 100_000, seed * 100_000 + FLOPS), chunksize=4)
    flops, turns, rivers, rank, strength, flop_draw, turn_draw = (np.array(x) for x in zip(*parts))
    flop_equity, turn_equity = equities(strength)
    print(f"  scored {FLOPS * TURNS * RIVERS:,} boards in {time.time() - start:.0f}s")
    return dict(flops=flops, turns=turns, rivers=rivers, rank=rank, river_strength=strength.astype(np.float16),
                turn_equity=turn_equity, flop_equity=flop_equity, flop_draw=flop_draw, turn_draw=turn_draw,
                played=played_weights())


def load():
    if not os.path.exists(CACHE):
        print(f"First run: scoring every hand on {FLOPS * TURNS * RIVERS:,} boards "
              "(a minute or two, then it's cached)...")
        data = build()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, **data)
        return Pool(data)
    return Pool(dict(np.load(CACHE)))
