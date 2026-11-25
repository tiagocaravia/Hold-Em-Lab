import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread: much faster for one-spot predictions

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from poker.decisions import advise_call
from poker.profiles import AIR, NO_READ, bet_weights, call_weights, preflop_weights, raise_weights

from .pool import COMBOS

STREETS = ("preflop", "flop", "turn", "river")
BET_SIZES = (0.5, 0.75, 1.25)    # bets, as a fraction of the pot
RAISE_SIZES = (0.6, 1.0)         # raises: how much more than a call, as a fraction of the pot after calling
PREFLOP_SIZES = {"vs_raise": 0.75, "vs_3bet": 1.0}  # open / raise a limp to 2.5bb; 3-bet to 3x
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "ml_model.joblib")
FEATURES = ["street", "hand strength", "our draw", "equity vs their range", "to call (pots)", "aggressive",
            "size (pots)", "guided value", "fold chance (guided)", "equity when called (guided)",
            "call rate at this size (read)", "times we bet this size", "pot (bb)", "stack behind (pots)",
            "bluff share (read)", "their call rate (read)", "hands of evidence",
            "their bluff share at this size (read)", "their bets seen at this size"]
GUIDED = FEATURES.index("guided value")  # the model learns a correction to this


def predict(model, rows): # Model corrections; older models trained on fewer inputs get just the ones they know
    rows = np.asarray(rows, dtype=float)
    return model.predict(rows[:, :model.n_features_in_])


def size_group(ratio): # Their bet as a fraction of the pot -> 0 small (< 0.6), 1 medium, 2 big (over pot)
    return 0 if ratio < 0.6 else 1 if ratio <= 1.0 else 2

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
    pot: float     # everything in the middle, including any bet we face
    invested: float = 0.0  # what we've already put in this hand (sunk)
    behind: float = 0.0    # what we have left to bet
    aggressions: int = 0   # bets and raises so far on this street

    def index(self):
        f, t, r = self.board
        return {"flop": (f,), "turn": (f, t), "river": (f, t, r)}[self.street]

    def cached(self, name, make): # Each street's arrays are computed once, however often they're used
        if name not in self.__dict__:
            self.__dict__[name] = make()
        return self.__dict__[name]

    def raw(self):
        return self.pool.strength["preflop"] if self.street == "preflop" else self.pool.strength[self.street][self.index()]

    def strengths(self): # Every hand's strength on this street (0 for hands that use a board card)
        return self.cached("_strengths", lambda: np.nan_to_num(self.raw().astype(float)))

    def draws(self):
        return self.cached("_draws", lambda: self.pool.draws[self.street][self.index()]
                           if self.street in ("flop", "turn") else np.zeros(len(COMBOS), int))

    def live(self): # Hands they could hold: not using a visible board card or one of ours
        return self.cached("_live", lambda: ~np.isnan(self.raw().astype(float)) & ~CONFLICT[self.mine])

    def order(self): # Their hands from strongest to weakest
        return self.cached("_order", lambda: np.argsort(-self.strengths()))

    def vs(self): # Our share of the pot against each of their possible hands, over the cards still to come
        return self.cached("_vs", self._vs)

    def _vs(self):
        if self.street == "preflop":
            return self.pool.preflop_vs[self.mine].astype(float)
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
        self.size_seen, self.size_bluffs = [0, 0, 0], [0, 0, 0]  # their bets seen at showdown, by size group

    def new_hand(self, pool, board, mine):
        self.range = (~CONFLICT[mine]).astype(float)  # any two cards they could hold

    def belief(self):
        return self.truth or NO_READ

    # --- reads ---
    def bluff_share(self, prior, size=None): # Share of their bets (of this size) we think are bluffs; None = profile as is
        if self.truth or not self.learns:
            return None
        overall = (self.bluffs_seen + self.k * prior) / (self.bets_seen + self.k)
        if size is None:
            return overall
        g = size_group(size)  # this size's own record, starting from the overall read
        return (self.size_bluffs[g] + self.k * overall) / (self.size_seen[g] + self.k)

    def call_rate(self, size): # Share of their range we think continues against our bet
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
        share = self.bluff_share(prior, size)
        if share is not None:
            weights = bet_weights(s, d, size, spot.street, p, base=self.range * spot.live(), share=share)
        return weights

    def see(self, spot, action, size=None, facing=0.0, situation=None):
        # They checked / bet / called / raised. size: their bet as a fraction of the pot before it;
        # facing: the bet they were facing (fraction of pot); situation: the preflop spot they were in
        live = self.range * spot.live()
        p = self.belief()
        if spot.street == "preflop":
            w = preflop_weights(spot.pool.preflop_pct, situation, p)
            w = w["raise"] if action in ("bet", "raise") else w["call"] if action == "call" else 1 - w["raise"]
        else:
            s, d = spot.strengths(), spot.draws()
            if action == "bet":
                w = self.their_bet_weights(spot, size)
            elif action == "check":
                w = 1 - bet_weights(s, d, p.bet_size, spot.street, p, base=live)
            elif action == "raise":
                w = raise_weights(s, d, facing, spot.street, p)
            else:  # call: they didn't raise, and their hand was good enough to call
                w = (1 - raise_weights(s, d, facing, spot.street, p)) * call_weights(s, d, facing, spot.street, p, base=live)
        self.range = live * w

    # --- valuing our options (what each wins from here, in bb; folding wins 0) ---
    def equity(self, spot, weights=None): # Our equity vs a weighted range (default: their current range)
        weights = self.range * spot.live() if weights is None else weights
        total = weights.sum()
        return float((weights * spot.vs()).sum() / total) if total else 0.5

    def passive_value(self, spot, to_call): # Check (to_call 0) or call, then no more betting
        return self.equity(spot) * (spot.pot + to_call) - to_call

    def continuing(self, spot, ratio, situation): # The part of their range that doesn't fold to our bet or raise
        live = self.range * spot.live()
        if spot.street == "preflop":
            w = preflop_weights(spot.pool.preflop_pct, situation, self.belief())
            return live * (w["call"] + w["raise"])
        if self.truth:
            s, d = spot.strengths(), spot.draws()
            r = raise_weights(s, d, ratio, spot.street, self.truth)
            return live * (r + (1 - r) * call_weights(s, d, ratio, spot.street, self.truth, base=live))
        order = spot.order()  # they continue with their strongest hands, as many as the call rate says
        cum = np.cumsum(live[order]) / max(live.sum(), 1e-12)
        keep = np.zeros_like(live)
        keep[order[cum <= self.call_rate(ratio)]] = live[order[cum <= self.call_rate(ratio)]]
        return keep

    def aggressive_parts(self, spot, to_call, size, situation=None): # Bet (to_call 0) or raise
        added = to_call + size * (spot.pot + to_call)        # what we put in now
        extra = added - to_call                              # what they must add to continue
        ratio = extra / (spot.pot + added)
        live = self.range * spot.live()
        cont = self.continuing(spot, ratio, situation)
        fold = 1 - cont.sum() / max(live.sum(), 1e-12)
        equity_called = self.equity(spot, cont) if cont.sum() else 0.5
        value = fold * spot.pot + (1 - fold) * (equity_called * (spot.pot + added + extra) - added)
        return {"added": added, "fold": fold, "equity_called": equity_called, "value": value, "ratio": ratio}

    def options(self, spot, to_call, can_raise, situation): # Every legal option: (move, chips put in, size, parts)
        stack = spot.behind
        opts = [("call" if to_call else "check", min(to_call, stack), 0.0, None)]
        if can_raise and stack > to_call:
            sizes = [PREFLOP_SIZES[situation]] if spot.street == "preflop" else (RAISE_SIZES if to_call else BET_SIZES)
            for size in sizes:
                parts = self.aggressive_parts(spot, to_call, size, situation)
                if parts["added"] <= stack:
                    opts.append(("raise" if to_call else "bet", parts["added"], size, parts))
        return opts

    def decide(self, spot, to_call, can_raise, situation, rng): # -> (move, chips put in now)
        # situation: for preflop raises, the spot they'd face ("vs_raise" or "vs_3bet")
        best, choice = (0.0, ("fold", 0.0)) if to_call else (-np.inf, None)
        for move, added, size, parts in self.options(spot, to_call, can_raise, situation):
            value = self.passive_value(spot, to_call) if parts is None else parts["value"]
            if value > best:
                best, choice = value, (move, added)
        if self.mixes and to_call and choice[0] in ("call", "fold"):  # mix calls and folds in close spots
            advice = advise_call(self.equity(spot), 0.0, spot.pot + to_call, to_call, bet_in_range=True)
            choice = ("call", min(to_call, spot.behind)) if rng.random() < advice.call_probability else ("fold", 0.0)
        self.last_street = spot.street
        return choice

    # --- what we learn ---
    def saw_response(self, called): # How they answered our bet or raise (after the flop only: preflop has its own rules)
        if getattr(self, "last_street", None) == "preflop":
            return
        if called:
            self.calls += 1
        else:
            self.folds += 1

    def showdown(self, their_bets): # We saw their hand: which of their bets (strength, draw, size) were bluffs
        for strength, draw, size in their_bets:
            bluff = strength < AIR and draw == 0
            self.bets_seen += 1
            self.bluffs_seen += bluff
            self.size_seen[size_group(size)] += 1
            self.size_bluffs[size_group(size)] += bluff

    def finish_hand(self, net):
        pass


def new_model(): # Stops adding trees once a held-back 10% of the data stops improving: fewer trees, faster, less overfit
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=80,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=15, random_state=0)


class MLAgent(Agent): # Hand-built value of each option + a learned correction, on every street
    def __init__(self, name="ml", explore=0.1, models=None, seed=7):
        super().__init__(name, learns=True, mixes=False)
        self.explore = explore  # chance of trying a random option, to keep learning
        self.model = models[0] if isinstance(models, tuple) else models
        self.data = []          # (features of the option taken, what the rest of the hand won)
        self.rng = np.random.default_rng(seed)
        self.spot_log = None    # set to [] to record every decision (for the training animation)
        self.pending, self.last_size = [], None

    def new_opponent(self, profile):
        super().new_opponent(profile)
        self.size_reads = {}  # (after the flop?, size) -> [calls, folds] against this opponent

    def new_hand(self, pool, board, mine):
        super().new_hand(pool, board, mine)
        self.pending = []

    def row(self, spot, to_call, size, parts, basics): # The features of one option
        street, strength, draw, equity = basics
        pot = spot.pot
        if parts is None:  # check or call
            guided, fold, equity_called, rate, seen = self.passive_value(spot, to_call), 0.0, equity, 0.0, 0
        else:
            calls, folds = self.size_reads.get((spot.street != "preflop", size), (0, 0))
            rate = (calls + self.k * self.call_rate(parts["ratio"])) / (calls + folds + self.k)
            guided, fold, equity_called, seen = parts["value"], parts["fold"], parts["equity_called"], calls + folds
        read = self.bluff_share(1 / 3)
        facing = to_call / max(pot - to_call, 1e-9) if to_call else None  # the bet we face, as a fraction of the pot
        size_read = self.bluff_share(1 / 3, facing) if facing else None
        size_seen = self.size_seen[size_group(facing)] if facing and self.learns else 0
        return [street, strength, draw, equity, to_call / pot, float(parts is not None), size, guided, fold,
                equity_called, rate, seen, pot, spot.behind / pot, 1 / 3 if read is None else read,
                self.call_rate(0.75), self.evidence(), (1 / 3 if size_read is None else size_read), size_seen]

    def decide(self, spot, to_call, can_raise, situation, rng):
        basics = (STREETS.index(spot.street), float(spot.strengths()[spot.mine]), int(spot.draws()[spot.mine]),
                  self.equity(spot))
        opts = self.options(spot, to_call, can_raise, situation)
        rows = [self.row(spot, to_call, size, parts, basics) for _, _, size, parts in opts]
        if self.spot_log is not None:
            self.spot_log.append((to_call > 0, spot.street, rows))
        values = np.array([r[GUIDED] for r in rows])
        if self.model is not None:
            values = values + predict(self.model, rows)
        if self.rng.random() < self.explore:  # try something random now and then (folding included)
            pick = int(self.rng.integers(len(opts) + (to_call > 0))) - (to_call > 0)
        else:
            pick = int(np.argmax(values))
            if to_call and values[pick] < 0:
                pick = -1  # folding (worth 0) beats every other option
        if pick < 0:
            return ("fold", 0.0)
        move, added, size, _ = opts[pick]
        self.last_street = spot.street
        self.pending.append((rows[pick], spot.invested))
        if move in ("bet", "raise"):
            self.last_size = (spot.street != "preflop", size)
        return (move, added)

    def saw_response(self, called):
        super().saw_response(called)
        reads = self.size_reads.setdefault(self.last_size, [0, 0])
        reads[0 if called else 1] += 1

    def finish_hand(self, net): # Each decision's reward: what the rest of the hand won (net + what was already in)
        for x, invested in self.pending:
            self.data.append((x, net + invested))
        self.pending = []

    def train(self):
        if len(self.data) < 500:
            return
        X = np.array([x for x, _ in self.data])
        y = np.array([won for _, won in self.data]) - X[:, GUIDED]  # learn where the hand-built value is wrong
        with threadpool_limits(limits=os.cpu_count()):  # fitting is worth every core; predicting isn't
            self.model = new_model().fit(X, y)

    def frozen(self, name="ml"): # A copy that only plays (no exploring, no data collection)
        bot = MLAgent(name, explore=0.0, models=self.model)
        bot.finish_hand = lambda net: None
        return bot


def lineup(): # The bots the arena compares: the simple baseline, the peeking oracle, and the trained ML bot if saved
    bots = [Agent("no read", learns=False), Agent("oracle", oracle=True)]
    if os.path.exists(MODEL_PATH):
        import joblib
        bots.append(MLAgent("ML bot", explore=0.0, models=joblib.load(MODEL_PATH)).frozen("ML bot"))
    return bots
