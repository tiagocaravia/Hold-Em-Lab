import os
import time
import warnings
from itertools import combinations
from multiprocessing import Pool as Workers

import numpy as np

from poker.deck import Deck
from poker.evaluator import _fast, card_id, hand_value, value_of_ids
from poker.profiles import DRAWS, soft_keep
from poker.ranges import draw_type, preflop_percentiles
from poker.tables import hand_label

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "hands.npz")
CARDS = Deck().cards
COMBOS = np.array(list(combinations(range(52), 2)))  # all 1,326 two-card hands, as card ids
COMBO_ID = np.zeros((52, 52), dtype=int)  # two card ids -> their hand's index, either order
COMBO_ID[COMBOS[:, 0], COMBOS[:, 1]] = COMBO_ID[COMBOS[:, 1], COMBOS[:, 0]] = np.arange(len(COMBOS))
LIB_ID = [card_id(c) for c in CARDS]  # our card ids -> the fast evaluator's card numbers
CONFLICT_PAIRS = np.array([[bool(set(a) & set(b)) for b in COMBOS] for a in COMBOS])  # hands that share a card
RANK = np.array([c.rank_value for c in CARDS])  # 0 = deuce ... 12 = ace
SUIT = np.array([["Clubs", "Diamonds", "Hearts", "Spades"].index(c.suit) for c in CARDS])


def _straight_tables(): # For every set of ranks (13-bit mask): is there a straight, and which ranks would make one
    def has(mask):
        bits = (mask << 1) | (mask >> 12 & 1)  # bit 0 is the ace playing low, bits 1-13 are deuce to ace
        return any(bits >> lo & 0b11111 == 0b11111 for lo in range(10))
    made = np.array([has(m) for m in range(1 << 13)])
    outs = np.array([0 if made[m] else sum(1 << r for r in range(13) if not m >> r & 1 and made[m | 1 << r])
                     for m in range(1 << 13)])
    return made, outs


MADE_STRAIGHT, STRAIGHT_OUTS = _straight_tables()


def draw_codes(board): # Every hand's draw on this board (DRAWS codes), all at once
    board = np.asarray(board)
    hole_rank, hole_suit = RANK[COMBOS], SUIT[COMBOS]
    board_mask = np.bitwise_or.reduce(1 << RANK[board])
    mask = board_mask | (1 << hole_rank[:, 0]) | (1 << hole_rank[:, 1])
    suit_count = np.bincount(SUIT[board], minlength=4)[None, :] + np.eye(4, dtype=int)[hole_suit].sum(axis=1)
    flush = ((suit_count == 4) & np.eye(4, dtype=bool)[hole_suit].any(axis=1)).any(axis=1)
    outs, new = STRAIGHT_OUTS[mask], STRAIGHT_OUTS[mask] & ~STRAIGHT_OUTS[board_mask]
    n_outs = np.array([bin(o).count("1") for o in range(1 << 13)])[outs]
    straight = np.where(n_outs >= 2, DRAWS["Straight draw"], DRAWS["Gutshot"])
    codes = np.where(flush, DRAWS["Flush draw"], np.where(~MADE_STRAIGHT[mask] & (new > 0), straight, 0))
    return np.where(np.isin(COMBOS, board).any(axis=1), 0, codes).astype(np.int8)
FLOPS, TURNS, RIVERS = 300, 6, 6  # 300 flops, 6 turns each, 6 rivers per turn: 10,800 full boards
PREFLOP_BOARDS = 4000             # independent random boards behind the preflop equity table
PLAY_PCT = 0.45  # players see the flop with roughly their top 45% of hands


class Pool:
    def __init__(self, data):
        self.flops, self.turns, self.rivers = data["flops"], data["turns"], data["rivers"]
        self.rank = data["rank"]                    # (F, T, R, 1326) showdown order on that river; -1 = uses a board card
        self.strength = {                           # share of hands each hand beats (on the flop/turn: on average)
            "flop": data["flop_equity"], "turn": data["turn_equity"], "river": data["river_strength"]}
        self.draws = {"flop": data["flop_draw"], "turn": data["turn_draw"]}
        self.played = data["played"]                # how likely each hand is to be played preflop (9-handed tables)
        self.preflop_vs = data["preflop_vs"]        # (1326, 1326) row hand's equity against column hand, all-in preflop
        self.strength["preflop"] = data["preflop_strength"]  # each hand's equity vs a random hand
        self.preflop_pct = data["preflop_pct"]      # share of hands at least as strong (0 = aces)
        self.preflop_ranks = data["preflop_ranks"]  # (boards, 1326) showdown order on independent random boards

    def deal(self, rng, players=2): # A full board plus disjoint random hands: (f, t, r), [hand of each player]
        f, t, r = rng.integers(FLOPS), rng.integers(TURNS), rng.integers(RIVERS)
        if players == 2:  # the original two-hand deal, so heads-up results stay comparable
            live = np.flatnonzero(self.rank[f, t, r] >= 0)
            while True:
                a, b = rng.choice(live, size=2, replace=False)
                if not set(COMBOS[a]) & set(COMBOS[b]):
                    return (f, t, r), [a, b]
        board = [*self.flops[f], self.turns[f, t], self.rivers[f, t, r]]
        cards = rng.choice(np.setdiff1d(np.arange(52), board), size=2 * players, replace=False)
        return (f, t, r), list(COMBO_ID[cards[0::2], cards[1::2]])


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
            rank[t, r], strength[t, r] = rank_board([*flop, turn_card, river_card])
    return flop, turn_cards, rivers, rank, strength, draw_codes(flop), np.array([draw_codes([*flop, t]) for t in turn_cards])


def rank_board(board): # Every hand's showdown order on a full board (-1 if it uses a board card), and strength
    live = [k for k, (i, j) in enumerate(COMBOS) if i not in board and j not in board]
    if _fast:  # compiled evaluator, on card numbers
        ids = [LIB_ID[b] for b in board]
        values = [value_of_ids([LIB_ID[COMBOS[k][0]], LIB_ID[COMBOS[k][1]], *ids]) for k in live]
    else:
        cards = [CARDS[i] for i in board]
        values = [hand_value([CARDS[COMBOS[k][0]], CARDS[COMBOS[k][1]]] + cards) for k in live]
        values = [(x[0], tuple(x[1])) for x in values]
    order = {x: n for n, x in enumerate(sorted(set(values)))}
    dense = np.array([order[x] for x in values])  # equal hands share a number
    ordered = np.sort(dense)
    below = np.searchsorted(ordered, dense, side="left")
    ties = np.searchsorted(ordered, dense, side="right") - below
    rank = np.full(len(COMBOS), -1, dtype=np.int16)
    strength = np.full(len(COMBOS), np.nan, dtype=np.float32)
    rank[live], strength[live] = dense, (below + 0.5 * ties) / len(live)
    return rank, strength


def rank_random_board(seed): # One independent random full board, ranked (for the preflop table)
    return rank_board(list(np.random.default_rng(seed).choice(52, size=5, replace=False)))[0]


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
    with Workers() as workers:  # independent random boards, so no flop is over-represented
        boards = np.array(workers.map(rank_random_board, range(seed * 10_000_000, seed * 10_000_000 + PREFLOP_BOARDS),
                                      chunksize=50))
    vs, preflop_strength, pct = preflop_tables(boards)
    print(f"  preflop equity of every hand vs every hand in {time.time() - start:.0f}s")
    return dict(preflop_ranks=boards.astype(np.int16), flops=flops, turns=turns, rivers=rivers, rank=rank, river_strength=strength.astype(np.float16),
                turn_equity=turn_equity, flop_equity=flop_equity, flop_draw=flop_draw, turn_draw=turn_draw,
                played=played_weights(), preflop_vs=vs, preflop_strength=preflop_strength, preflop_pct=pct)


def preflop_tables(ranks): # Every hand vs every hand, averaged over many random full boards
    wins = np.zeros((len(COMBOS), len(COMBOS)), dtype=np.float32)
    counts = np.zeros_like(wins)
    for r in ranks.astype(np.int32):
        valid = r >= 0
        both = valid[:, None] & valid[None, :]
        wins += both * ((r[:, None] > r[None, :]) + 0.5 * (r[:, None] == r[None, :]))
        counts += both
    vs = np.where(counts > 0, wins / np.maximum(counts, 1), 0.5)
    vs[CONFLICT_PAIRS] = 0.5  # hands sharing a card can't meet
    strength = np.array([vs[i, ~CONFLICT_PAIRS[i]].mean() for i in range(len(COMBOS))])
    pct = (np.argsort(np.argsort(-strength)) + 1) / len(COMBOS)  # 1/1326 for the best hand ... 1 for the worst
    return vs.astype(np.float16), strength, pct


def load():
    if not os.path.exists(CACHE):
        print(f"First run: scoring every hand on {FLOPS * TURNS * RIVERS:,} boards "
              "(seconds with phevaluator, a minute or two without; then it's cached)...")
        data = build()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, **data)
        return Pool(data)
    data = dict(np.load(CACHE))
    if "preflop_ranks" not in data:  # caches built before multiway play: add the preflop boards (same seeds)
        with Workers() as workers:
            data["preflop_ranks"] = np.array(workers.map(rank_random_board, range(10_000_000, 10_000_000 + PREFLOP_BOARDS),
                                                         chunksize=50)).astype(np.int16)
        np.savez_compressed(CACHE, **data)
    return Pool(data)
