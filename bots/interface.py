import argparse
import os
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so "poker" imports work

import joblib
import numpy as np

from bots.ml_agent import MODEL_PATH, Agent, MLAgent, Spot
from bots.pool import CARDS, COMBOS, equities, load, score_board
from poker.card import parse_card
from poker.profiles import PRESETS

CARD_ID = {(c.rank, c.suit): i for i, c in enumerate(CARDS)}
STREET_OF = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
CARDS_ON = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
SITUATIONS = ("vs_raise", "vs_3bet")  # the preflop spot a raise leaves the other player in, by raises so far


@dataclass
class GameState: # What a game engine tells the bot when it's our turn. Amounts in big blinds; blinds 0.5 / 1
    hole: list                 # our cards, e.g. ["AS", "KH"] (10 is "10": "10H")
    board: list                # community cards so far ([] preflop)
    pot: float                 # everything in the middle, including any bet we face
    to_call: float = 0.0       # what we must add to call (0 if nothing)
    stack: float = 100.0       # what we have left
    actions: list = field(default_factory=list)  # every action this hand so far, in order, blinds not included:
                                                 # (street, "us" | a player's name, "fold" | "check" | "call" | "bet" | "raise", chips put in)
    opponent: str = "opponent" # heads-up: who we're playing ("them" in actions means this player)
    players: list = None       # several players: everyone dealt in, from the small blind round ("us" for us)


def card_ids(cards):
    return [CARD_ID[(c.rank, c.suit)] for c in (parse_card(x) for x in cards)]


def combo_of(ids):
    return int(np.flatnonzero((COMBOS == sorted(ids)).all(axis=1))[0])


class Board: # A one-board version of the pool, so the bots can play any real board
    def __init__(self, ids, base, rng, runouts):
        flop, turn, river = np.array(ids[:3]), (ids[3] if len(ids) > 3 else None), (ids[4] if len(ids) > 4 else None)
        _, _, _, rank, strength, flop_draw, turn_draw = score_board(flop, turn, river, rng, *runouts)
        flop_equity, turn_equity = equities(strength[None])
        self.rank = rank[None]
        self.strength = {"flop": flop_equity, "turn": turn_equity, "river": strength[None], "preflop": base.strength["preflop"]}
        self.draws = {"flop": flop_draw[None], "turn": turn_draw[None]}
        self.preflop_vs, self.preflop_pct = base.preflop_vs, base.preflop_pct


class Bot: # The plug-in bot: one act() call per decision. Reads are kept per player name, across hands
    def __init__(self, model_path=MODEL_PATH, runouts=(6, 6), seed=0, hand_built=False):
        self.model = joblib.load(model_path) if not hand_built and model_path and os.path.exists(model_path) else None
        self.kind = "ML bot" if self.model is not None else "hand-built bot"
        self.base = load()           # preflop equity tables
        self.runouts = runouts       # turns x rivers sampled on the flop (fewer = faster, rougher)
        self.rng = np.random.default_rng(seed)
        self.agent = MLAgent("bot", explore=0.0, models=self.model) if self.model is not None else Agent("bot")
        self.agent.finish_hand = lambda net: None
        self.hand = None

    def spot(self, street, pot, invested=0.0, behind=0.0, **seen):
        ids = self.hand["board"][:CARDS_ON[street]]
        if street == "preflop":
            pool = self.base
        else:
            if tuple(ids) not in self.hand["boards"]:  # score each board once per hand and street
                self.hand["boards"][tuple(ids)] = Board(ids, self.base, self.rng, self.runouts)
            pool = self.hand["boards"][tuple(ids)]
        return Spot(pool, (0, 0, 0), street, self.hand["mine"], pot, invested, behind, **seen)

    def start(self, state):
        players = state.players
        if not players:  # heads-up: the small blind acts first preflop
            players = ["us", state.opponent] if not state.actions or state.actions[0][1] == "us" else [state.opponent, "us"]
        put = {p: 0.0 for p in players}
        put[players[0]], put[players[1]] = 0.5, 1.0
        self.hand = {"hole": state.hole, "players": players, "mine": combo_of(card_ids(state.hole)), "board": [],
                     "boards": {}, "seen": 0, "street": "preflop", "put": put, "cur": dict(put), "aggressions": 0,
                     "last_bet": None, "live": set(players), "their_bets": {p: [] for p in players}}
        self.agent.new_hand(self.base, (0, 0, 0), self.hand["mine"], "us", tuple(players))

    def new_street(self, street):
        h = self.hand
        h["street"], h["cur"], h["aggressions"], h["last_bet"] = street, {p: 0.0 for p in h["players"]}, 0, None

    def replay(self, state): # Catch up on the actions since we last acted, as the arena would have shown them
        h = self.hand
        for street, who, move, chips in state.actions[h["seen"]:]:
            who = state.opponent if who == "them" else who
            if street != h["street"]:
                self.new_street(street)
            pot = sum(h["put"].values())
            to_call = max(h["cur"].values()) - h["cur"][who]
            if who != "us":
                if h["last_bet"] == "us":
                    self.agent.saw_response(who, move != "fold")
                aggressions = h["aggressions"]
                situation = ("vs_limp" if to_call == 0 else "open") if aggressions == 0 else SITUATIONS[aggressions - 1]
                self.agent.see(who, self.spot(street, pot), move, chips / pot, to_call / max(pot - to_call, 1e-9), situation)
                if street != "preflop" and move in ("bet", "raise"):
                    h["their_bets"][who].append((street, chips / pot))
            if move == "fold":
                h["live"].discard(who)
            h["put"][who] += chips
            h["cur"][who] += chips
            if move in ("bet", "raise"):
                h["aggressions"], h["last_bet"] = h["aggressions"] + 1, who
        h["seen"] = len(state.actions)

    def act(self, state): # -> ("fold" | "check" | "call" | "bet" | "raise", chips to put in now)
        if self.hand is None or self.hand["hole"] != state.hole or len(state.board) < len(self.hand["board"]):
            self.start(state)
        self.hand["board"] = card_ids(state.board)
        self.replay(state)
        h, street = self.hand, STREET_OF[len(state.board)]
        if street != h["street"]:
            self.new_street(street)
        players = h["players"]
        order = players[2:] + players[:2] if street == "preflop" and len(players) > 2 else \
            players if street == "preflop" or len(players) > 2 else players[::-1]  # this street's acting order
        others = tuple(p for p in players if p in h["live"] and p != "us")
        after = order[order.index("us") + 1:]
        postflop = players if len(players) > 2 else players[::-1]
        seen = dict(opponents=others, to_act=sum(p in others for p in after),
                    in_position=[p for p in postflop if p in h["live"]][-1] == "us",
                    aggressor=h["last_bet"] or (players[1] if street == "preflop" else None))
        spot = self.spot(street, state.pot, h["put"]["us"], state.stack, **seen)
        can_raise = h["aggressions"] < 2 and state.stack > state.to_call
        return self.agent.decide(spot, state.to_call, can_raise, SITUATIONS[min(h["aggressions"], 1)], self.rng)

    def end_hand(self, state=None, shown=None): # Learn from how the hand ended
        # state: the final GameState, to catch up on the last actions (say, folds to our bet)
        # shown: {player name: hole cards} for every hand shown down
        if self.hand is None:
            return
        if state is not None:  # after we fold the others play on: catch up on their board and actions too
            self.hand["board"] = card_ids(state.board)
            self.replay(state)
        for who, hole in (shown or {}).items():  # which of their bets were bluffs?
            theirs, bets = combo_of(card_ids(hole)), []
            for street, size in self.hand["their_bets"].get(who, []):
                spot = self.spot(street, 1.0)
                bets.append((float(spot.strengths()[theirs]), int(spot.draws()[theirs]), size))
            self.agent.showdown(who, bets)
        self.hand = None


class ThroughInterface: # Plays arena hands, but every decision goes through Bot.act(): the template for an adapter
    def __init__(self, bot):
        self.bot, self.name, self.thinking, self.decisions = bot, f"{bot.kind} (through the interface)", 0.0, 0

    def new_table(self, profiles):
        self.table = id(profiles)  # new opponents get new names, so the bot starts fresh reads on them

    def new_hand(self, pool, board, mine, me=0, seats=(0, 1)):
        f, t, r = board
        name = lambda s: "us" if s == me else f"seat {s} at table {self.table}"
        self.hole = [f"{CARDS[i].rank}{CARDS[i].suit[0]}" for i in COMBOS[mine]]
        self.cards = [f"{CARDS[i].rank}{CARDS[i].suit[0]}" for i in [*pool.flops[f], pool.turns[f, t], pool.rivers[f, t, r]]]
        self.name_of, self.players, self.actions, self.shown = name, [name(s) for s in seats], [], {}

    def see(self, who, spot, move, size=None, facing=0.0, situation=None): # Record their action as an engine would
        if move in ("bet", "raise"):
            chips = size * spot.pot
        elif move == "call":
            chips = spot.pot * facing / (1 + facing)  # facing = to_call / (pot - to_call)
        else:
            chips = 0.0
        self.actions.append((spot.street, self.name_of(who), move, chips))

    def state(self, street, pot=0.0, to_call=0.0, stack=0.0):
        return GameState(self.hole, self.cards[:CARDS_ON[street]], pot, to_call, stack, list(self.actions),
                         players=self.players)

    def decide(self, spot, to_call, can_raise, situation, rng):
        start = time.time()
        move, chips = self.bot.act(self.state(spot.street, spot.pot, to_call, spot.behind))
        self.thinking, self.decisions = self.thinking + time.time() - start, self.decisions + 1
        self.actions.append((spot.street, "us", move, chips))
        return move, chips

    def reveal(self, who, combo):
        self.shown[self.name_of(who)] = [f"{CARDS[i].rank}{CARDS[i].suit[0]}" for i in COMBOS[combo]]

    def saw_response(self, *args):
        pass

    def showdown(self, *args):
        pass

    def finish_hand(self, net):
        self.bot.end_hand(self.state("river"), self.shown)


def demo(hands, opponent, seed, hand_built=False, players=2):
    from bots.arena import bb100, play
    player = ThroughInterface(Bot(hand_built=hand_built))
    print(f"{player.name} vs {'a ' + opponent if players == 2 else f'{players - 1} {opponent}s'}, {hands} hands")
    results = play(player, load(), hands, hands, seed, {opponent: PRESETS[opponent]}, players=players)
    m, h = bb100([r[2] for r in results])
    print(f"  result: {m:+.0f} ± {h:.0f} bb/100;  {player.decisions} decisions, "
          f"{player.thinking / max(player.decisions, 1):.3f}s each on average")
    return results


def main():
    parser = argparse.ArgumentParser(description="The plug-in bot. Run to watch it play a preset opponent through its interface.")
    parser.add_argument("--hands", type=int, default=200, help="Hands to play in the demo")
    parser.add_argument("--opponent", default="regular", choices=list(PRESETS), help="Preset opponent")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--hand-built", action="store_true", help="Use the hand-built bot instead of the trained ML bot")
    parser.add_argument("--players", type=int, default=2, help="Players at the table, the bot included (2-9)")
    args = parser.parse_args()
    demo(args.hands, args.opponent, args.seed, args.hand_built, args.players)


if __name__ == "__main__":
    main()
