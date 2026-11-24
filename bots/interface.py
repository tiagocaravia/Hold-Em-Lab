import argparse
import os
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so "poker" imports work

import joblib
import numpy as np

from bots.ml_agent import MODEL_PATH, Agent, MLAgent, Spot
from bots.pool import CARDS, COMBOS, equities, played_weights, score_board
from poker.card import parse_card
from poker.profiles import PRESETS, bet_weights, call_weights

CARD_ID = {(c.rank, c.suit): i for i, c in enumerate(CARDS)}
STREET_OF = {3: "flop", 4: "turn", 5: "river"}


@dataclass
class GameState: # What a game engine tells the bot when it's our turn. Amounts in big blinds
    hole: list                 # our cards, e.g. ["AS", "KH"] (10 is "10": "10H")
    board: list                # community cards so far
    pot: float                 # everything in the middle, including any bet we face
    to_call: float = 0.0       # the bet we face (0 if none)
    stack: float = 100.0       # what we have left
    actions: list = field(default_factory=list)  # the opponent's actions this hand, in order:
                                                 # (street, "check" | "bet" | "call", size as a fraction of the pot before it)
    opponent: str = "opponent" # who we're playing; reads are kept per name


def card_ids(cards):
    return [CARD_ID[(c.rank, c.suit)] for c in (parse_card(x) for x in cards)]


class Board: # A one-board version of the pool, so the bots can play any real board
    def __init__(self, ids, played, rng, runouts):
        flop, turn, river = np.array(ids[:3]), (ids[3] if len(ids) > 3 else None), (ids[4] if len(ids) > 4 else None)
        _, _, _, rank, strength, flop_draw, turn_draw = score_board(flop, turn, river, rng, *runouts)
        flop_equity, turn_equity = equities(strength[None])
        self.rank = rank[None]
        self.strength = {"flop": flop_equity, "turn": turn_equity, "river": strength[None]}
        self.draws = {"flop": flop_draw[None], "turn": turn_draw[None]}
        self.played = played


class Bot: # The plug-in bot: one act() call per decision
    def __init__(self, model_path=MODEL_PATH, runouts=(6, 6), seed=0):
        self.models = joblib.load(model_path) if model_path and os.path.exists(model_path) else None
        self.kind = "ML bot" if self.models else "hand-built bot"
        self.runouts = runouts       # turns x rivers sampled on the flop (fewer = faster, rougher)
        self.played = played_weights()
        self.rng = np.random.default_rng(seed)
        self.agents = {}             # one bot per opponent, so reads are kept per player
        self.hand = None

    def agent(self, opponent):
        if opponent not in self.agents:
            agent = MLAgent("bot", explore=0.0, models=self.models) if self.models else Agent("bot")
            agent.finish_hand = lambda net: None
            agent.new_opponent(None)
            self.agents[opponent] = agent
        return self.agents[opponent]

    def board_for(self, ids): # Score a board once per hand and street
        key = tuple(ids)
        if key not in self.hand["boards"]:
            self.hand["boards"][key] = Board(ids, self.played, self.rng, self.runouts)
        return self.hand["boards"][key]

    def spot(self, state, ids, pot):
        mine = int(np.flatnonzero((COMBOS == sorted(card_ids(state.hole))).all(axis=1))[0])
        return Spot(self.board_for(ids), (0, 0, 0), STREET_OF[len(ids)], mine, pot, behind=state.stack)

    def act(self, state): # -> ("fold" | "check" | "call" | "bet", amount in bb)
        ids = card_ids(state.board)
        if not ids:
            return self.preflop(state)
        agent = self.agent(state.opponent)
        if self.hand is None or self.hand["hole"] != state.hole or len(ids) < len(self.hand["board"]):
            self.hand = {"hole": state.hole, "board": ids, "boards": {}, "seen": 0, "opponent": state.opponent,
                         "their_bets": []}
            agent.new_hand(self.board_for(ids[:3]), (0, 0, 0), self.spot(state, ids[:3], state.pot).mine)
        self.hand["board"] = ids
        for street, kind, size in state.actions[self.hand["seen"]:]:  # catch up on what they did since we last acted
            cards = ids[:{"flop": 3, "turn": 4, "river": 5}[street]]
            if kind == "call":
                agent.saw_response(True)
            if kind == "bet":
                self.hand["their_bets"].append(tuple(cards))
            agent.see(self.spot(state, cards, state.pot), kind, size)
        self.hand["seen"] = len(state.actions)

        if state.to_call > 0:
            before = state.pot - state.to_call
            call = agent.respond(self.spot(state, ids, before), state.to_call, self.rng)
            return ("call", min(state.to_call, state.stack)) if call else ("fold", 0.0)
        bet = agent.act(self.spot(state, ids, state.pot), state.stack)
        return ("bet", bet) if bet else ("check", 0.0)

    def preflop(self, state): # Placeholder until preflop play is trained: play the hands people play
        mine = int(np.flatnonzero((COMBOS == sorted(card_ids(state.hole))).all(axis=1))[0])
        if state.to_call == 0:
            return ("check", 0.0)
        return ("call", min(state.to_call, state.stack)) if self.played[mine] > 0.5 else ("fold", 0.0)

    def end_hand(self, opponent_folded=False, their_hole=None): # Learn from how the hand ended
        if self.hand is None:
            return
        agent = self.agent(self.hand["opponent"])
        if opponent_folded:
            agent.saw_response(False)
        elif their_hole:  # showdown: which of their bets were bluffs?
            theirs = int(np.flatnonzero((COMBOS == sorted(card_ids(their_hole))).all(axis=1))[0])
            bets = []
            for ids in self.hand["their_bets"]:
                street, board = STREET_OF[len(ids)], self.hand["boards"][ids]
                idx = {"flop": (0,), "turn": (0, 0), "river": (0, 0, 0)}[street]
                draw = int(board.draws[street][idx][theirs]) if street != "river" else 0
                bets.append((float(np.nan_to_num(board.strength[street][idx][theirs])), draw))
            agent.showdown(bets)
        self.hand = None


def demo(hands, opponent, seed): # A tiny engine that talks to the bot only through GameState: the template for an adapter
    rng = np.random.default_rng(seed)
    bot, profile = Bot(), PRESETS[opponent]
    print(f"{bot.kind} vs a {opponent}, {hands} hands, every decision through Bot.act()")
    total, decisions, thinking = 0.0, 0, 0.0
    for h in range(hands):
        deck = rng.permutation(52)
        ours, theirs, board = deck[:2], deck[2:4], deck[4:9]
        name = lambda ids: [f"{CARDS[i].rank}{CARDS[i].suit[0]}" for i in ids]
        pot, put_in, stack, actions = 6.0, 3.0, 97.0, []
        theirs_combo = int(np.flatnonzero((COMBOS == sorted(theirs)).all(axis=1))[0])
        result = None
        for street, n in (("flop", 3), ("turn", 4), ("river", 5)):
            their_board = Board(list(board[:n]), bot.played, rng, (2, 2))
            idx = (0,) if street == "flop" else (0, 0) if street == "turn" else (0, 0, 0)
            s = float(np.nan_to_num(their_board.strength[street][idx][theirs_combo]))
            d = int(their_board.draws[street][idx][theirs_combo]) if street != "river" else 0
            if rng.random() < float(bet_weights(np.array([s]), np.array([d]), profile.bet_size, street, profile)[0]):
                bet = min(profile.bet_size * pot, stack)
                actions.append((street, "bet", bet / pot))
                state = GameState(name(ours), name(board[:n]), pot + bet, bet, stack, list(actions), opponent)
                start = time.time(); move, amount = bot.act(state); thinking += time.time() - start; decisions += 1
                if move == "fold":
                    result = -put_in
                    break
                pot, put_in, stack = pot + 2 * bet, put_in + bet, stack - bet
                continue
            actions.append((street, "check", None))
            state = GameState(name(ours), name(board[:n]), pot, 0.0, stack, list(actions), opponent)
            start = time.time(); move, amount = bot.act(state); thinking += time.time() - start; decisions += 1
            if move == "bet":
                size = amount / pot
                if rng.random() < float(call_weights(np.array([s]), np.array([d]), size, street, profile)[0]):
                    actions.append((street, "call", size))
                    pot, put_in, stack = pot + 2 * amount, put_in + amount, stack - amount
                else:
                    bot.end_hand(opponent_folded=True)
                    result = pot - put_in
                    break
        if result is None:
            final = Board(list(board), bot.played, rng, (1, 1))
            ours_combo = int(np.flatnonzero((COMBOS == sorted(ours)).all(axis=1))[0])
            diff = int(final.rank[0, 0, 0][ours_combo]) - int(final.rank[0, 0, 0][theirs_combo])
            result = (pot if diff > 0 else pot / 2 if diff == 0 else 0) - put_in
            bot.end_hand(their_hole=name(theirs))
        bot.hand = None
        total += result
    print(f"  result: {total / hands * 100:+.0f} bb/100 over {hands} hands")
    print(f"  {decisions} decisions, {thinking / max(decisions, 1):.2f}s per decision on average")


def main():
    parser = argparse.ArgumentParser(description="The plug-in bot. Run to watch it play a preset opponent through its interface.")
    parser.add_argument("--hands", type=int, default=30, help="Hands to play in the demo")
    parser.add_argument("--opponent", default="regular", choices=list(PRESETS), help="Preset opponent")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    args = parser.parse_args()
    demo(args.hands, args.opponent, args.seed)


if __name__ == "__main__":
    main()
