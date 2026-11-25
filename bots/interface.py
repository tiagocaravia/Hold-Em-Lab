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
                                                 # (street, "us" | "them", "fold" | "check" | "call" | "bet" | "raise", chips put in)
    opponent: str = "opponent" # who we're playing; reads are kept per name


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


class Bot: # The plug-in bot: one act() call per decision
    def __init__(self, model_path=MODEL_PATH, runouts=(6, 6), seed=0, hand_built=False):
        self.model = joblib.load(model_path) if not hand_built and model_path and os.path.exists(model_path) else None
        self.kind = "ML bot" if self.model is not None else "hand-built bot"
        self.base = load()           # preflop equity tables
        self.runouts = runouts       # turns x rivers sampled on the flop (fewer = faster, rougher)
        self.rng = np.random.default_rng(seed)
        self.agents = {}             # one bot per opponent, so reads are kept per player
        self.hand = None

    def agent(self, opponent):
        if opponent not in self.agents:
            agent = MLAgent("bot", explore=0.0, models=self.model) if self.model is not None else Agent("bot")
            agent.finish_hand = lambda net: None
            agent.new_opponent(None)
            self.agents[opponent] = agent
        return self.agents[opponent]

    def spot(self, street, pot, invested=0.0, behind=0.0):
        ids = self.hand["board"][:CARDS_ON[street]]
        if street == "preflop":
            pool = self.base
        else:
            if tuple(ids) not in self.hand["boards"]:  # score each board once per hand and street
                self.hand["boards"][tuple(ids)] = Board(ids, self.base, self.rng, self.runouts)
            pool = self.hand["boards"][tuple(ids)]
        return Spot(pool, (0, 0, 0), street, self.hand["mine"], pot, invested, behind)

    def start(self, state):
        small_blind = not state.actions or state.actions[0][1] == "us"  # the small blind acts first preflop
        put = {"us": 0.5 if small_blind else 1.0, "them": 1.0 if small_blind else 0.5}
        self.hand = {"hole": state.hole, "opponent": state.opponent, "mine": combo_of(card_ids(state.hole)),
                     "board": [], "boards": {}, "seen": 0, "street": "preflop", "put": put, "cur": dict(put),
                     "aggressions": 0, "we_just_bet": False, "their_bets": []}
        self.agent(state.opponent).new_hand(self.base, (0, 0, 0), self.hand["mine"])

    def replay(self, state): # Catch up on the actions since we last acted, as the arena would have shown them
        h, agent = self.hand, self.agent(state.opponent)
        for street, who, move, chips in state.actions[h["seen"]:]:
            if street != h["street"]:
                h["street"], h["cur"], h["aggressions"] = street, {"us": 0.0, "them": 0.0}, 0
            other = "them" if who == "us" else "us"
            pot = sum(h["put"].values())
            to_call = h["cur"][other] - h["cur"][who]
            if who == "them":
                if h["we_just_bet"]:
                    agent.saw_response(move != "fold")
                if move != "fold":
                    aggressions = h["aggressions"]
                    situation = ("vs_limp" if to_call == 0 else "open") if aggressions == 0 else SITUATIONS[aggressions - 1]
                    agent.see(self.spot(street, pot), move, chips / pot, to_call / max(pot - to_call, 1e-9), situation)
                    if street != "preflop" and move in ("bet", "raise"):
                        h["their_bets"].append((street, chips / pot))
            h["put"][who] += chips
            h["cur"][who] += chips
            h["aggressions"] += move in ("bet", "raise")
            h["we_just_bet"] = who == "us" and move in ("bet", "raise")
        h["seen"] = len(state.actions)

    def act(self, state): # -> ("fold" | "check" | "call" | "bet" | "raise", chips to put in now)
        if self.hand is None or self.hand["hole"] != state.hole or len(state.board) < len(self.hand["board"]):
            self.start(state)
        self.hand["board"] = card_ids(state.board)
        self.replay(state)
        street = STREET_OF[len(state.board)]
        if street != self.hand["street"]:
            self.hand["street"], self.hand["cur"], self.hand["aggressions"] = street, {"us": 0.0, "them": 0.0}, 0
        aggressions = self.hand["aggressions"]
        spot = self.spot(street, state.pot, self.hand["put"]["us"], state.stack)
        can_raise = aggressions < 2 and state.stack > state.to_call
        return self.agent(state.opponent).decide(spot, state.to_call, can_raise, SITUATIONS[min(aggressions, 1)], self.rng)

    def end_hand(self, opponent_folded=False, their_hole=None): # Learn from how the hand ended
        if self.hand is None:
            return
        agent = self.agent(self.hand["opponent"])
        if opponent_folded and self.hand["we_just_bet"]:
            agent.saw_response(False)
        elif their_hole:  # showdown: which of their bets were bluffs?
            theirs = combo_of(card_ids(their_hole))
            bets = []
            for street, size in self.hand["their_bets"]:
                spot = self.spot(street, 1.0)
                bets.append((float(spot.strengths()[theirs]), int(spot.draws()[theirs]), size))
            agent.showdown(bets)
        self.hand = None


class ThroughInterface: # Plays arena hands, but every decision goes through Bot.act(): the template for an adapter
    def __init__(self, bot):
        self.bot, self.name, self.thinking, self.decisions = bot, f"{bot.kind} (through the interface)", 0.0, 0

    def new_opponent(self, profile):
        self.opponent = f"{profile.name} {id(profile)}"

    def new_hand(self, pool, board, mine):
        f, t, r = board
        self.hole = [f"{CARDS[i].rank}{CARDS[i].suit[0]}" for i in COMBOS[mine]]
        self.cards = [f"{CARDS[i].rank}{CARDS[i].suit[0]}" for i in [*pool.flops[f], pool.turns[f, t], pool.rivers[f, t, r]]]
        self.actions = []

    def see(self, spot, move, size=None, facing=0.0, situation=None): # Record their action as an engine would
        if move in ("bet", "raise"):
            chips = size * spot.pot
        elif move == "call":
            chips = spot.pot * facing / (1 + facing)  # facing = to_call / (pot - to_call)
        else:
            chips = 0.0
        self.actions.append((spot.street, "them", move, chips))

    def decide(self, spot, to_call, can_raise, situation, rng):
        state = GameState(self.hole, self.cards[:CARDS_ON[spot.street]], spot.pot, to_call, spot.behind,
                          list(self.actions), self.opponent)
        start = time.time()
        move, chips = self.bot.act(state)
        self.thinking, self.decisions = self.thinking + time.time() - start, self.decisions + 1
        self.actions.append((spot.street, "us", move, chips))
        return move, chips

    def saw_response(self, called):
        pass

    def showdown(self, shown):
        pass

    def finish_hand(self, net):
        folded = bool(self.actions) and self.actions[-1][1] == "us" and self.actions[-1][2] in ("bet", "raise")
        self.bot.end_hand(opponent_folded=folded)


def demo(hands, opponent, seed, hand_built=False):
    from bots.arena import bb100, play
    player = ThroughInterface(Bot(hand_built=hand_built))
    print(f"{player.name} vs a {opponent}, {hands} hands")
    results = play(player, load(), hands, hands, seed, {opponent: PRESETS[opponent]})
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
    args = parser.parse_args()
    demo(args.hands, args.opponent, args.seed, args.hand_built)


if __name__ == "__main__":
    main()
