import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread: much faster for one-spot predictions

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from poker.decisions import advise_call
from poker.profiles import AIR, NO_READ, bet_weights, call_weights

from .pool import COMBOS

STREETS = ("flop", "turn", "river")
BET_SIZES = (0.5, 0.75, 1.25)  # pot fractions the bots can bet
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "ml_model.joblib")
CALL_FEATURES = ["street", "hand strength", "our draw", "equity vs their range", "bet size (pot)",
                 "guided EV of calling", "pot (bb)", "stack behind (pots)", "bluff share (read)",
                 "their bets seen", "their call rate (read)", "hands of evidence"]
BET_FEATURES = ["street", "hand strength", "our draw", "equity vs their range", "bet size (pot)",
                "guided value of this option", "fold chance (guided)", "equity when called (guided)",
                "call rate at this size (read)", "times we bet this size", "pot (bb)", "stack behind (pots)",
                "hands of evidence"]
CALL_GUIDED = BET_GUIDED = 5  # where the hand-built value sits; the models learn corrections to it

# CONFLICT[i] marks every hand that shares a card with hand i
_onehot = np.zeros((len(COMBOS), 52), dtype=np.float32)
_onehot[np.arange(len(COMBOS)), COMBOS[:, 0]] = 1
_onehot[np.arange(len(COMBOS)), COMBOS[:, 1]] = 1
CONFLICT = (_onehot @ _onehot.T) > 0


@dataclass
class Spot: # What the bot can see on one street
    pool: object
    board: tuple   # (flop, turn, river) indexes into the pool
    street: str
    mine: int      # our hand (combo index)
    pot: float
    invested: float = 0.0  # what we've already put in this hand (sunk)
    behind: float = 0.0    # what we have left to bet

    def index(self):
        f, t, r = self.board
        return {"flop": (f,), "turn": (f, t), "river": (f, t, r)}[self.street]

    def cached(self, name, make): # Each street's arrays are computed once, however often they're used
        if name not in self.__dict__:
            self.__dict__[name] = make()
        return self.__dict__[name]

    def strengths(self): # Every hand's strength on this street (0 for hands that use a board card)
        return self.cached("_strengths", lambda: np.nan_to_num(self.pool.strength[self.street][self.index()].astype(float)))

    def draws(self):
        return self.cached("_draws", lambda: self.pool.draws[self.street][self.index()] if self.street != "river"
                           else np.zeros(len(COMBOS), int))

    def live(self): # Hands they could hold: not using a visible board card or one of ours
        return self.cached("_live", lambda: ~np.isnan(self.pool.strength[self.street][self.index()].astype(float))
                           & ~CONFLICT[self.mine])

    def order(self): # Their hands from strongest to weakest
        return self.cached("_order", lambda: np.argsort(-self.strengths()))

    def vs(self): # Our share of the pot against each of their possible hands, over the runouts still to come
        return self.cached("_vs", self._vs)

    def _vs(self):
        f, t, r = self.board
        ranks = {"flop": self.pool.rank[f], "turn": self.pool.rank[f, t], "river": self.pool.rank[f, t, r]}[self.street]
        ranks = ranks.reshape(-1, len(COMBOS)).astype(int)
        mine = ranks[:, self.mine][:, None]
        valid = (ranks >= 0) & (mine >= 0)
        share = np.where(valid, (mine > ranks) + 0.5 * (mine == ranks), 0.0).sum(axis=0)
        return share / np.maximum(valid.sum(axis=0), 1)


class Agent: # The hand-built bot: ranging + pot odds + reads. The ML bot's guidance, and the baselines
    def __init__(self, name, learns=True, mixes=False, oracle=False, prior_weight=8):
        self.name = name
        self.learns = learns
        self.mixes = mixes
        self.oracle = oracle
        self.k = prior_weight  # how many hands' worth of trust the no-read assumptions get

    # --- per opponent / per hand ---
    def new_opponent(self, profile): # Forget everything; the oracle peeks at the real profile
        self.truth = profile if self.oracle else None
        self.bets_seen = self.bluffs_seen = 0
        self.calls = self.folds = 0

    def new_hand(self, pool, board, mine):
        self.range = pool.played * Spot(pool, board, "flop", mine, 0).live()  # hands they'd see a flop with

    def belief(self):
        return self.truth or NO_READ

    # --- reads ---
    def bluff_share(self, prior): # Share of their bets we think are bluffs (None = use the profile as is)
        if self.truth or not self.learns:
            return None
        return (self.bluffs_seen + self.k * prior) / (self.bets_seen + self.k)

    def call_rate(self, size): # Share of their range we think calls our bet
        prior = 1 / (1 + size)  # no read: they defend just enough
        if not self.learns:
            return prior
        return (self.calls + self.k * prior) / (self.calls + self.folds + self.k)

    def evidence(self):
        return self.bets_seen + self.calls + self.folds if self.learns else 0

    # --- their actions narrow their range ---
    def their_bet_weights(self, spot, size):
        s, d, p = spot.strengths(), spot.draws(), self.belief()
        weights = bet_weights(s, d, size, spot.street, p, base=self.range * spot.live())
        prior = (self.range * weights * (s < AIR) * (d == 0)).sum() / max((self.range * weights).sum(), 1e-12)
        share = self.bluff_share(prior)
        if share is not None:
            weights = bet_weights(s, d, size, spot.street, p, base=self.range * spot.live(), share=share)
        return weights

    def see(self, spot, action, size=None): # They checked, bet (size = fraction of pot) or called
        live = self.range * spot.live()
        s, d, p = spot.strengths(), spot.draws(), self.belief()
        if action == "bet":
            w = self.their_bet_weights(spot, size)
        elif action == "call":
            w = call_weights(s, d, size, spot.street, p, base=live)
        else:  # check: what they wouldn't have bet at their usual size
            w = 1 - bet_weights(s, d, p.bet_size, spot.street, p, base=live)
        self.range = live * w

    # --- our decisions ---
    def equity(self, spot, weights=None): # Our equity vs a weighted range (default: their current range)
        weights = self.range * spot.live() if weights is None else weights
        total = weights.sum()
        return float((weights * spot.vs()).sum() / total) if total else 0.5

    def respond(self, spot, bet, rng): # Facing their bet: call (True) or fold (False). spot.pot is before the bet
        equity = self.equity(spot)  # their range already includes the bet
        advice = advise_call(equity, 0.0, spot.pot + bet, bet, bet_in_range=True)
        return rng.random() < advice.call_probability if self.mixes else equity >= advice.break_even

    def bet_value(self, spot, size): # What betting this size wins from here, this street only, in bb
        return self.bet_parts(spot, size)["value"]

    def bet_parts(self, spot, size): # The pieces: fold chance, equity when called, value
        live = self.range * spot.live()
        s, d = spot.strengths(), spot.draws()
        if self.truth:
            calls = live * call_weights(s, d, size, spot.street, self.truth, base=live)
        else:  # they call with their strongest hands, as many as the call rate says
            order = spot.order()
            cum = np.cumsum(live[order]) / max(live.sum(), 1e-12)
            calls = np.zeros_like(live)
            calls[order[cum <= self.call_rate(size)]] = live[order[cum <= self.call_rate(size)]]
        fold = 1 - calls.sum() / max(live.sum(), 1e-12)
        bet = size * spot.pot
        equity_called = self.equity(spot, calls) if calls.sum() else 0.5
        value = fold * spot.pot + (1 - fold) * (equity_called * (spot.pot + 2 * bet) - bet)
        return {"fold": fold, "equity_called": equity_called, "value": value}

    def act(self, spot, stack): # Our turn with no bet to face: a bet in bb, or None to check
        best, choice = self.equity(spot) * spot.pot, None  # checking: our share of the pot, roughly
        for size in BET_SIZES:
            if size * spot.pot > stack:
                continue
            value = self.bet_value(spot, size)
            if value > best:
                best, choice = value, size * spot.pot
        return choice

    # --- what we learn ---
    def saw_response(self, called):
        if called:
            self.calls += 1
        else:
            self.folds += 1

    def showdown(self, their_bets): # We saw their hand: count which of their bets were bluffs
        for strength, draw in their_bets:
            self.bets_seen += 1
            self.bluffs_seen += strength < AIR and draw == 0

    def finish_hand(self, net):
        pass


def new_model(): # Stops adding trees once a held-back 10% of the data stops improving: fewer trees, faster, less overfit
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=80,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=15, random_state=0)


class MLAgent(Agent): # Hand-built value of each option + a learned correction, on every street
    def __init__(self, name="ml", explore=0.1, models=None, seed=7):
        super().__init__(name, learns=True, mixes=False)
        self.explore = explore  # chance of trying a random action, to keep learning
        self.call_model, self.bet_model = models or (None, None)
        self.call_data, self.bet_data = [], []  # (features, what the rest of the hand won)
        self.rng = np.random.default_rng(seed)
        self.spot_log = None  # set to {"call": [], "bet": []} to record every spot it decides
        self.pending, self.last_size = [], None

    def new_opponent(self, profile):
        super().new_opponent(profile)
        self.size_reads = {}  # bet size -> [calls, folds] against this opponent

    def new_hand(self, pool, board, mine):
        super().new_hand(pool, board, mine)
        self.pending = []

    def basics(self, spot): # Features every decision shares
        street = STREETS.index(spot.street)
        draw = int(spot.draws()[spot.mine])
        return street, float(spot.strengths()[spot.mine]), draw, self.equity(spot)

    # --- facing their bet ---
    def call_features(self, spot, bet):
        street, strength, draw, equity = self.basics(spot)
        ratio = bet / spot.pot
        guided = equity * (spot.pot + 2 * bet) - bet  # what calling wins from here; folding wins 0
        prior = ratio / (1 + 2 * ratio)
        read = self.bluff_share(prior)
        return [street, strength, draw, equity, ratio, guided, spot.pot, spot.behind / spot.pot,
                prior if read is None else read, self.bets_seen, self.call_rate(0.75), self.evidence()]

    def respond(self, spot, bet, rng):
        x = self.call_features(spot, bet)
        if self.spot_log is not None:
            self.spot_log["call"].append(x)
        correction = self.call_model.predict([x])[0] if self.call_model else 0.0
        call = x[CALL_GUIDED] + correction > 0
        if self.rng.random() < self.explore:
            call = self.rng.random() < 0.5
        if call:  # folds have a known result (0 from here), so only calls are learned from
            self.pending.append(("call", x, spot.invested))
        return call

    # --- our bet ---
    def bet_features(self, spot, size, basics=None):
        street, strength, draw, equity = basics or self.basics(spot)
        base = [street, strength, draw, equity]
        tail = [spot.pot, spot.behind / spot.pot, self.evidence()]
        if not size:  # checking
            return base + [0.0, equity * spot.pot, 0.0, equity, 0.0, 0] + tail
        parts = self.bet_parts(spot, size)
        calls, folds = self.size_reads.get(size, (0, 0))
        rate = (calls + self.k * self.call_rate(size)) / (calls + folds + self.k)
        return base + [size, parts["value"], parts["fold"], parts["equity_called"], rate, calls + folds] + tail

    def act(self, spot, stack):
        options = [0.0] + [s for s in BET_SIZES if s * spot.pot <= stack]
        basics = self.basics(spot)
        rows = [self.bet_features(spot, size, basics) for size in options]
        if self.spot_log is not None:
            self.spot_log["bet"].append(rows if len(rows) == 4 else None)
        if self.rng.random() < self.explore:
            choice = int(self.rng.integers(len(options)))
        else:
            values = np.array([r[BET_GUIDED] for r in rows])
            if self.bet_model is not None:
                values = values + self.bet_model.predict(rows)
            choice = int(np.argmax(values))
        self.pending.append(("bet", rows[choice], spot.invested))
        self.last_size = options[choice]
        return options[choice] * spot.pot if options[choice] else None

    def saw_response(self, called):
        super().saw_response(called)
        reads = self.size_reads.setdefault(self.last_size, [0, 0])
        reads[0 if called else 1] += 1

    def finish_hand(self, net): # Each decision's reward: what the rest of the hand won (net + what was already in)
        for kind, x, invested in self.pending:
            (self.call_data if kind == "call" else self.bet_data).append((x, net + invested))
        self.pending = []

    # --- training ---
    def train(self):
        for kind, guided_at in (("call", CALL_GUIDED), ("bet", BET_GUIDED)):
            data = self.call_data if kind == "call" else self.bet_data
            if len(data) < 500:
                continue
            X = np.array([x for x, _ in data])
            y = np.array([won for _, won in data]) - X[:, guided_at]  # learn where the hand-built value is wrong
            with threadpool_limits(limits=os.cpu_count()):  # fitting is worth every core; predicting isn't
                setattr(self, f"{kind}_model", new_model().fit(X, y))

    def frozen(self, name="ml"): # A copy that only plays (no exploring, no data collection)
        bot = MLAgent(name, explore=0.0, models=(self.call_model, self.bet_model))
        bot.finish_hand = lambda net: None
        return bot


def lineup(): # The hand-built bot versions the arena compares
    return [
        Agent("no read", learns=False),
        Agent("learner", learns=True),
        Agent("learner + mixing", learns=True, mixes=True),
        Agent("oracle", oracle=True),
    ]
