import numpy as np
import random
from collections import Counter
from itertools import combinations

from .deck import Deck
from .equity import RangeOpponents, key, pot_share
from .evaluator import category, hand_value
from .profiles import DRAWS, NO_READ, bet_weights, call_weights, soft_keep
from .tables import hand_label, load_table

TABLE_SIZES = (3, 5, 7, 9)  # Preflop tables we have
CATEGORY = ["High card", "One pair", "Two pair", "Three of a kind", "Straight",
            "Flush", "Full house", "Four of a kind", "Straight flush"]

STREET_ACTIONS = ("check", "bet", "call", "raise")
SIZE_WORDS = {"third": 1 / 3, "half": 0.5, "pot": 1.0}


def parse_street_action(text): # "bet half" -> ("bet", 0.5); "check" -> ("check", None)
    words = text.strip().lower().split()
    if not words or words[0] not in STREET_ACTIONS:
        raise ValueError
    action, size = words[0], None
    if action == "bet":
        size = 0.5
        if len(words) > 1:
            w = words[1]
            if w in SIZE_WORDS:
                size = SIZE_WORDS[w]
            elif w.endswith("%"):
                size = float(w[:-1]) / 100
            elif w.endswith("x"):
                size = float(w[:-1])
            else:
                raise ValueError
    return action, size


def draw_type(combo, board): # Strongest draw the combo adds: flush, straight, gutshot or None
    if len(board) >= 5:
        return None
    cards = combo + board
    for suit in {c.suit for c in combo}:
        if sum(c.suit == suit for c in cards) == 4:
            return "Flush draw"
    mine, theirs = _straight_outs(cards), _straight_outs(board)
    new = mine - theirs
    if _has_straight(_values(cards)) or not new:
        return None
    return "Straight draw" if len(mine) >= 2 else "Gutshot"


def _values(cards):
    vals = {c.rank_value for c in cards}
    return vals | {-1} if 12 in vals else vals  # ace plays low too


def _has_straight(vals):
    return any(all(v + k in vals for k in range(5)) for v in range(-1, 9))


def _straight_outs(cards): # Ranks that would complete a straight
    vals = _values(cards)
    if _has_straight(vals):
        return set()
    outs = set()
    for r in range(13):
        if r not in vals and _has_straight(vals | {r} | ({-1} if r == 12 else set())):
            outs.add(r)
    return outs


PREFLOP_ACTIONS = {
    "none": "play", "check": None,
    "limp": "call", "call": "call",
    "raise": "raise", "open": "raise",
    "raise big": "big", "3bet": "big", "4bet": "big", "allin": "big", "all in": "big",
}


def short_label(a, b): # AKo, AA, JTs
    hi, lo = sorted((a, b), key=lambda c: c.rank_value, reverse=True)
    name = "".join("T" if c.rank == "10" else c.rank for c in (hi, lo))
    return name if hi.rank == lo.rank else name + ("s" if hi.suit == lo.suit else "o")


def preflop_percentiles(num_players): # Hand label -> share of combos at least as strong
    size = min(TABLE_SIZES, key=lambda n: abs(n - num_players))
    table = load_table(size).sort_values("equity", ascending=False)
    counts = Counter(hand_label(a, b) for a, b in combinations(Deck().cards, 2))
    percentiles, seen = {}, 0
    for label in table["Hand"]:
        seen += counts[label]
        percentiles[label] = seen / 1326
    return percentiles


class Range:
    def __init__(self, dead_cards):
        dead = {key(c) for c in dead_cards}
        self.combos = [list(c) for c in combinations(Deck().cards, 2)
                       if key(c[0]) not in dead and key(c[1]) not in dead]
        self.weights = np.ones(len(self.combos))
        self.history = []

    def total(self):
        return sum(self.weights)

    def preflop(self, action, num_players, t=NO_READ): # Narrow by their preflop action (t: their Profile)
        kind = PREFLOP_ACTIONS.get(action.strip().lower())
        if kind is None:
            return
        pct = preflop_percentiles(num_players)
        for i, (a, b) in enumerate(self.combos):
            p = pct[hand_label(a, b)]
            if kind == "play":
                w = soft_keep(p, t.play_pct)
            elif kind == "raise":
                w = soft_keep(p, t.raise_pct)
            elif kind == "big":
                w = soft_keep(p, t.big_raise_pct)
            else:
                w = soft_keep(p, t.call_pct) * (t.slowplay if p < t.big_raise_pct else 1)
            self.weights[i] *= w
        self.history.append("played a hand preflop" if kind == "play" else f"{action} preflop")

    def strengths(self, board): # Share of all live hands each combo beats or ties on this board
        on_board = {key(b) for b in board}
        is_live = lambda c: key(c[0]) not in on_board and key(c[1]) not in on_board
        rank_of = {i: hand_value(c + board) for i, c in enumerate(self.combos) if is_live(c)}
        ranks = sorted(rank_of.values())
        out = []
        for i in range(len(self.combos)):
            if i not in rank_of:
                out.append(None)
                continue
            r = rank_of[i]
            lo, hi = 0, len(ranks)
            while lo < hi:  # count of ranks <= r
                mid = (lo + hi) // 2
                if ranks[mid] <= r:
                    lo = mid + 1
                else:
                    hi = mid
            out.append(lo / len(ranks))
        return out

    def street_arrays(self, board): # Strength and draw code of every combo on this board; dead combos get weight 0
        s = self.strengths(board)
        dead = np.array([x is None for x in s])
        self.weights[dead] = 0.0
        strength = np.array([0.0 if x is None else x for x in s])
        draws = np.array([DRAWS[draw_type(c, board)] if len(board) < 5 else 0 for c in self.combos])
        return strength, draws

    def river_bet(self, board, pot_before, bet, t=NO_READ): # Narrow by a river bet: value + bluffs
        strength, draws = self.street_arrays(board)
        self.weights *= bet_weights(strength, draws, bet / pot_before, "river", t, base=self.weights)
        self.history.append(f"bet {bet:g} into {pot_before:g} on the river")

    def street_action(self, board, action, size=None, t=NO_READ): # Narrow by a flop/turn action
        strength, draws = self.street_arrays(board)
        street = "flop" if len(board) == 3 else "turn"
        if action == "bet":
            w = bet_weights(strength, draws, size or 0.5, street, t)
        elif action == "raise":  # a raise is a big bet: stronger value, fewer bluffs
            w = bet_weights(strength, draws, 2.0, street, t)
        elif action == "call":
            w = call_weights(strength, draws, t.bet_size, street, t, base=self.weights)
        else:  # check: everything they'd have bet, the other way round
            w = 1 - bet_weights(strength, draws, t.bet_size, street, t)
        self.weights *= w
        label = action if action != "bet" else f"bet {size or 0.5:.0%} pot"
        self.history.append(f"{label} on the {street}")

    def river_equity(self, hole, board): # Exact equity vs this range, heads-up
        mine = hand_value(hole + board)
        blocked = {key(c) for c in hole}
        num = den = 0.0
        for c, w in zip(self.combos, self.weights):
            if w == 0 or key(c[0]) in blocked or key(c[1]) in blocked:
                continue
            num += w * pot_share(mine, [hand_value(c + board)])
            den += w
        return num / den if den else float("nan")

    def breakdown(self, hole, board): # Hand categories, and how many beat you
        mine = hand_value(hole + board)
        blocked = {key(c) for c in hole}
        cats, beats, ties, total = {}, 0.0, 0.0, 0.0
        for c, w in zip(self.combos, self.weights):
            if w == 0 or key(c[0]) in blocked or key(c[1]) in blocked:
                continue
            r = hand_value(c + board)
            cats[CATEGORY[category(r)]] = cats.get(CATEGORY[category(r)], 0) + w
            total += w
            if r > mine:
                beats += w
            elif r == mine:
                ties += w
        cats = sorted(((k, v / total) for k, v in cats.items()), key=lambda kv: -kv[1])
        return cats, beats / total, ties / total

    def draws(self, hole, board): # Share of the range on each kind of draw
        blocked = {key(c) for c in hole}
        out, total = {}, 0.0
        for c, w in zip(self.combos, self.weights):
            if w == 0 or key(c[0]) in blocked or key(c[1]) in blocked:
                continue
            total += w
            d = draw_type(c, board)
            if d:
                out[d] = out.get(d, 0) + w
        return sorted(((k, v / total) for k, v in out.items()), key=lambda kv: -kv[1])

    def top_hands(self, n=6): # Most likely hand types
        types = {}
        for (a, b), w in zip(self.combos, self.weights):
            label = short_label(a, b)
            types[label] = types.get(label, 0) + w
        total = sum(types.values())
        return [(k, v / total) for k, v in sorted(types.items(), key=lambda kv: -kv[1])[:n]]


class RangeModel(RangeOpponents): # The first opponent from a narrowed Range; any others from `others` (an opponent model)
    def __init__(self, rng, others=None, name="ranged"):
        rest = others.seats[-1] if isinstance(others, RangeOpponents) else None  # None: any two cards
        super().__init__(None, name, seats=[(rng.combos, rng.weights), rest])
