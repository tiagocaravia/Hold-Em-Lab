import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread: much faster for one-spot predictions

from dataclasses import dataclass, field

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
PREFLOP_BOARDS = 1000           # random boards behind preflop equity against several players (fewer = faster, rougher)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "ml_model.joblib")
FEATURES = ["street", "hand strength", "our draw", "equity vs their range", "to call (pots)", "aggressive",
            "size (pots)", "guided value", "fold chance (guided)", "equity when called (guided)",
            "call rate at this size (read)", "times we bet this size", "pot (bb)", "stack behind (pots)",
            "bluff share (read)", "their call rate (read)", "hands of evidence",
            "their bluff share at this size (read)", "their bets seen at this size",
            "opponents in hand", "players left to act", "in position"]
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
    opponents: tuple = ()  # seats still in the hand besides ours
    to_act: int = 0        # players still to act after us on this street
    in_position: bool = True  # we act last after the flop
    aggressor: object = None  # the seat whose bet we face (preflop: the big blind until someone raises)

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

    def ranks(self): # Every hand's showdown order on each runout still possible (rows), -1 if it uses a board card
        if self.street == "preflop":
            return self.pool.preflop_ranks[:PREFLOP_BOARDS]
        f, t, r = self.board
        ranks = {"flop": self.pool.rank[f], "turn": self.pool.rank[f, t], "river": self.pool.rank[f, t, r]}[self.street]
        return ranks.reshape(-1, len(COMBOS))

    def vs(self): # Our share of the pot against each of their possible hands, over the cards still to come
        return self.cached("_vs", self._vs)

    def _vs(self):
        if self.street == "preflop":
            return self.pool.preflop_vs[self.mine].astype(float)
        ranks = self.ranks().astype(int)
        mine = ranks[:, self.mine][:, None]
        valid = (ranks >= 0) & (mine >= 0)
        share = np.where(valid, (mine > ranks) + 0.5 * (mine == ranks), 0.0).sum(axis=0)
        return share / np.maximum(valid.sum(axis=0), 1)

    def outcomes(self): # Per runout: which of their hands we beat, tie, and could meet (for equity vs several players)
        return self.cached("_outcomes", self._outcomes)

    def _outcomes(self):
        ranks = self.ranks()
        mine = ranks[:, self.mine]
        ranks, mine = ranks[mine >= 0], mine[mine >= 0][:, None]  # runouts that don't use our cards
        valid = ranks >= 0
        return [((ranks < mine) & valid).astype(np.float32), ((ranks == mine) & valid).astype(np.float32),
                valid.astype(np.float32)]


def multiway_share(spot, ranges, folds=None): # Our expected share of the pot against several ranges at once
    # On each runout: the chance we beat or tie each opponent, combined across opponents (ties split the pot),
    # then averaged over runouts. folds[i]: chance opponent i folds first (then we beat them for sure)
    beat, tie, valid = spot.outcomes()
    weights = np.stack(ranges, axis=1).astype(np.float32)
    n = valid @ weights
    w, e = beat @ weights / np.maximum(n, 1e-12), tie @ weights / np.maximum(n, 1e-12)
    stay = np.ones(len(ranges), np.float32) if folds is None else 1 - np.asarray(folds, dtype=np.float32)
    w, e = 1 - stay + stay * w, stay * e
    ways = np.ones((len(w), 1))  # ways[:, k]: chance we're best with exactly k players tied
    for i in range(w.shape[1]):
        ways = np.pad(ways * w[:, i:i + 1], ((0, 0), (0, 1))) + np.pad(ways * e[:, i:i + 1], ((0, 0), (1, 0)))
    share = (ways / np.arange(1, ways.shape[1] + 1)).sum(axis=1).mean()
    if len(ranges) > 1:  # card removal: two players can't hold the same card, so "someone beats us" is likelier
        some = slice(None, None, max(1, len(w) // 64))  # than independent ranges say. A small, smooth correction:
        share += removal(beat[some], tie[some], valid[some], weights, stay, (1 - w - e)[some]).mean()  # 64 runouts do
    return float(np.clip(share, 0, 1))


def removal(beat, tie, valid, weights, stay, ahead): # Pairwise card-removal correction to our share, per runout
    # The chance two opponents both beat us, counting only hand pairs that share no card: exact, from card totals
    # (hands sharing a card with hand c = the hands holding either of its cards, minus c itself)
    hands, k = weights.shape
    by_card = (weights[:, :, None] * _onehot[:, None, :]).reshape(hands, -1)  # each player's weight on each card
    pairs = (weights[:, :, None] * weights[:, None, :]).reshape(hands, -1)    # both players holding the same hand

    def disjoint(mask):  # (runouts, hands) 0/1 -> (runouts, k, k): weight of the pairs of hands that share no card
        totals, cards = mask @ weights, (mask @ by_card).reshape(len(mask), k, 52)
        return (totals[:, :, None] * totals[:, None, :] - cards @ cards.transpose(0, 2, 1)
                + (mask @ pairs).reshape(len(mask), k, k))

    both = disjoint(valid - beat - tie) / np.maximum(disjoint(valid), 1e-12)
    out = np.zeros(len(beat))
    for i in range(k):
        for j in range(i + 1, k):
            rest = np.prod(np.delete(1 - ahead, [i, j], axis=1), axis=1)  # everyone else doesn't beat us
            out += (stay[i] * stay[j] * both[:, i, j] - ahead[:, i] * ahead[:, j]) * rest
    return out


@dataclass
class Read: # What we've learned about one opponent
    truth: object = None  # their real Profile (the oracle only)
    bets_seen: int = 0    # their bets we saw at showdown, and how many were bluffs
    bluffs_seen: int = 0
    calls: int = 0        # how they answered our bets
    folds: int = 0
    size_seen: list = field(default_factory=lambda: [0, 0, 0])  # the same, by size group
    size_bluffs: list = field(default_factory=lambda: [0, 0, 0])


class Agent: # The hand-built bot: ranging + pot odds + reads. The ML bot's guidance, and the baselines
    def __init__(self, name, learns=True, mixes=False, oracle=False, prior_weight=8):
        self.name = name
        self.learns = learns
        self.mixes = mixes
        self.oracle = oracle
        self.k = prior_weight  # how many hands' worth of trust the no-read assumptions get
        self.new_table({})

    # --- per table / per hand ---
    def new_table(self, profiles): # New opponents, {seat: Profile or None}: forget every read; the oracle peeks
        self.truths = dict(profiles) if self.oracle else {}
        self.reads = {}

    def read(self, who): # Reads are kept per seat (or per name), made the first time we meet them
        if who not in self.reads:
            self.reads[who] = Read(self.truths.get(who))
        return self.reads[who]

    def new_hand(self, pool, board, mine, me=0, seats=(0, 1)):
        self.me = me
        self.ranges = {s: (~CONFLICT[mine]).astype(float) for s in seats if s != me}  # any two cards they could hold

    def belief(self, who):
        return self.read(who).truth or NO_READ

    def opponents(self, spot): # Seats still in the hand
        return [s for s in (spot.opponents or self.ranges) if s in self.ranges]

    # --- reads ---
    def bluff_share(self, prior, size=None, who=None): # Share of their bets (of this size) we think are bluffs; None = profile as is
        r = self.read(who)
        if r.truth or not self.learns:
            return None
        overall = (r.bluffs_seen + self.k * prior) / (r.bets_seen + self.k)
        if size is None:
            return overall
        g = size_group(size)  # this size's own record, starting from the overall read
        return (r.size_bluffs[g] + self.k * overall) / (r.size_seen[g] + self.k)

    def call_rate(self, size, who=None): # Share of their range we think continues against our bet
        prior = 1 / (1 + size)  # no read: they defend just enough
        if not self.learns:
            return prior
        r = self.read(who)
        return (r.calls + self.k * prior) / (r.calls + r.folds + self.k)

    def evidence(self, who=None):
        r = self.read(who)
        return r.bets_seen + r.calls + r.folds if self.learns else 0

    # --- their actions narrow their range ---
    def their_bet_weights(self, spot, size, who):
        s, d, p, rng = spot.strengths(), spot.draws(), self.belief(who), self.ranges[who]
        weights = bet_weights(s, d, size, spot.street, p, base=rng * spot.live())
        prior = (rng * weights * (s < AIR) * (d == 0)).sum() / max((rng * weights).sum(), 1e-12)
        share = self.bluff_share(prior, size, who)
        if share is not None:
            weights = bet_weights(s, d, size, spot.street, p, base=rng * spot.live(), share=share)
        return weights

    def see(self, who, spot, action, size=None, facing=0.0, situation=None):
        # Seat `who` folded / checked / bet / called / raised. size: their bet as a fraction of the pot before it;
        # facing: the bet they were facing (fraction of pot); situation: the preflop spot they were in
        if action == "fold" or who not in self.ranges:
            self.ranges.pop(who, None)
            return
        live = self.ranges[who] * spot.live()
        p = self.belief(who)
        if spot.street == "preflop":
            w = preflop_weights(spot.pool.preflop_pct, situation, p)
            w = w["raise"] if action in ("bet", "raise") else w["call"] if action == "call" else 1 - w["raise"]
        else:
            s, d = spot.strengths(), spot.draws()
            if action == "bet":
                w = self.their_bet_weights(spot, size, who)
            elif action == "check":
                w = 1 - bet_weights(s, d, p.bet_size, spot.street, p, base=live)
            elif action == "raise":
                w = raise_weights(s, d, facing, spot.street, p)
            else:  # call: they didn't raise, and their hand was good enough to call
                w = (1 - raise_weights(s, d, facing, spot.street, p)) * call_weights(s, d, facing, spot.street, p, base=live)
        self.ranges[who] = live * w

    # --- valuing our options (what each wins from here, in bb; folding wins 0) ---
    def equity(self, spot, ranges=None, folds=None): # Our share of the pot vs their ranges (default: everyone still in)
        if ranges is None:
            ranges = [self.ranges[s] * spot.live() for s in self.opponents(spot)]
        if len(ranges) == 1 and folds is None:  # heads-up: exact, from the head-to-head table
            total = ranges[0].sum()
            return float((ranges[0] * spot.vs()).sum() / total) if total else 0.5
        return multiway_share(spot, ranges, folds) if ranges else 1.0

    def passive_value(self, spot, to_call): # Check (to_call 0) or call, then no more betting
        return self.equity(spot) * (spot.pot + to_call) - to_call

    def continuing(self, spot, ratio, situation, who): # The part of their range that doesn't fold to our bet or raise
        live = self.ranges[who] * spot.live()
        truth = self.read(who).truth
        if spot.street == "preflop":
            w = preflop_weights(spot.pool.preflop_pct, situation, self.belief(who))
            return live * (w["call"] + w["raise"])
        if truth:
            s, d = spot.strengths(), spot.draws()
            r = raise_weights(s, d, ratio, spot.street, truth)
            return live * (r + (1 - r) * call_weights(s, d, ratio, spot.street, truth, base=live))
        order = spot.order()  # they continue with their strongest hands, as many as the call rate says
        cum = np.cumsum(live[order]) / max(live.sum(), 1e-12)
        keep = np.zeros_like(live)
        kept = order[cum <= self.call_rate(ratio, who)]
        keep[kept] = live[kept]
        return keep

    def aggressive_parts(self, spot, to_call, size, situation=None): # Bet (to_call 0) or raise
        added = to_call + size * (spot.pot + to_call)        # what we put in now
        extra = added - to_call                              # what each caller must add to continue
        ratio = extra / (spot.pot + added)
        seats = self.opponents(spot)
        lives = [self.ranges[s] * spot.live() for s in seats]
        conts = [self.continuing(spot, ratio, situation, s) for s in seats]
        stay = [c.sum() / max(l.sum(), 1e-12) for c, l in zip(conts, lives)]  # chance each one continues
        if len(seats) == 1:
            fold = 1 - stay[0]
            equity_called = self.equity(spot, conts) if conts[0].sum() else 0.5
            callers = 1.0
        else:  # everyone folds, or we play the ones who continue (a folder is a hand we beat)
            fold = float(np.prod([1 - c for c in stay]))
            mixed = self.equity(spot, [c if c.sum() else l for c, l in zip(conts, lives)], folds=[1 - c for c in stay])
            equity_called = (mixed - fold) / (1 - fold) if fold < 1 - 1e-9 else 0.5
            callers = sum(stay) / (1 - fold) if fold < 1 - 1e-9 else 1.0
        value = fold * spot.pot + (1 - fold) * (equity_called * (spot.pot + added + extra * callers) - added)
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
    def saw_response(self, who, called): # How `who` answered our bet or raise (after the flop only: preflop has its own rules)
        if getattr(self, "last_street", None) == "preflop":
            return
        r = self.read(who)
        if called:
            r.calls += 1
        else:
            r.folds += 1

    def showdown(self, who, their_bets): # We saw their hand: which of their bets (strength, draw, size) were bluffs
        r = self.read(who)
        for strength, draw, size in their_bets:
            bluff = strength < AIR and draw == 0
            r.bets_seen += 1
            r.bluffs_seen += bluff
            r.size_seen[size_group(size)] += 1
            r.size_bluffs[size_group(size)] += bluff

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

    def new_table(self, profiles):
        super().new_table(profiles)
        self.size_reads = {}  # (seat, after the flop?, size) -> [calls, folds] against our bets of that size

    def new_hand(self, pool, board, mine, me=0, seats=(0, 1)):
        super().new_hand(pool, board, mine, me, seats)
        self.pending = []

    def row(self, spot, to_call, size, parts, basics): # The features of one option
        street, strength, draw, equity = basics
        pot = spot.pot
        seats = self.opponents(spot) or [None]
        mean = lambda f: sum(f(s) for s in seats) / len(seats)  # reads averaged over everyone still in
        if parts is None:  # check or call
            guided, fold, equity_called, rate, seen = self.passive_value(spot, to_call), 0.0, equity, 0.0, 0
        else:
            reads = {s: self.size_reads.get((s, spot.street != "preflop", size), (0, 0)) for s in seats}
            rate = mean(lambda s: (reads[s][0] + self.k * self.call_rate(parts["ratio"], s)) / (sum(reads[s]) + self.k))
            guided, fold, equity_called, seen = parts["value"], parts["fold"], parts["equity_called"], mean(lambda s: sum(reads[s]))
        facing = to_call / max(pot - to_call, 1e-9) if to_call else None  # the bet we face, as a fraction of the pot
        bettor = spot.aggressor if spot.aggressor in seats else seats[0]
        share = lambda s: self.bluff_share(1 / 3, who=s) or 1 / 3  # None (no read kept) -> the 1/3 prior
        read = share(bettor) if facing else mean(share)  # facing a bet: the bettor's read; otherwise everyone's
        size_read = self.bluff_share(1 / 3, facing, bettor) if facing else None
        size_seen = self.read(bettor).size_seen[size_group(facing)] if facing and self.learns else 0
        return [street, strength, draw, equity, to_call / pot, float(parts is not None), size, guided, fold,
                equity_called, rate, seen, pot, spot.behind / pot, read,
                mean(lambda s: self.call_rate(0.75, s)), mean(self.evidence), (1 / 3 if size_read is None else size_read),
                size_seen, len(seats), spot.to_act, float(spot.in_position)]

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

    def saw_response(self, who, called):
        super().saw_response(who, called)
        reads = self.size_reads.setdefault((who, *self.last_size), [0, 0])
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
