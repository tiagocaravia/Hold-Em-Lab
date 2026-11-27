import argparse
import os
import sys

import joblib
import numpy as np

from bots.arena import ProfilePlayer, bb100, play_table
from bots.ml_agent import BET_SIZES, MODEL_PATH, PREFLOP_SIZES, RAISE_SIZES, Agent, MLAgent
from bots.pool import CARDS, COMBOS, load
from poker.evaluator import category, hand_value
from poker.profiles import TEST_PRESETS
from poker.ranges import CATEGORY
from poker.prompts import ask, interactive_loop

SUIT = {"Spades": "♠", "Hearts": "♥", "Diamonds": "♦", "Clubs": "♣"}
OPPONENTS = ["ml bot", "old ml bot", "hand-built bot", *TEST_PRESETS]
OLD_MODEL = os.path.join(os.path.dirname(MODEL_PATH), "ml_model_no_selfplay.joblib")  # trained before self-play
BET_NAMES = {0.5: "½ pot", 0.75: "¾ pot", 1.25: "1¼ pot", 0.6: "small raise", 1.0: "pot-sized raise"}
COLOR = sys.stdout.isatty()  # turned off by --no-color, or when output isn't a terminal
LINE = "─" * 58


def paint(text, code): # ANSI colour: 31 red, 32 green, 1 bold, 2 dim
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def show(ids): # Card ids -> "A♠ K♥", red suits in red
    out = []
    for i in ids:
        card = f"{CARDS[i].rank}{SUIT[CARDS[i].suit]}"
        out.append(paint(card, 31) if CARDS[i].suit in ("Hearts", "Diamonds") else card)
    return " ".join(out)


def hand_name(cards, board): # "two pair", "a flush"...
    value = hand_value([CARDS[i] for i in [*cards, *board]])
    return CATEGORY[category(value)].lower()


def positions(n, button): # Seat -> its position name this hand
    names = {}
    for seat in range(n):
        off = (seat - button) % n
        if n == 2:
            names[seat] = "button/SB" if off == 0 else "big blind"
        else:
            late = {n - 1: "CO", n - 2: "HJ", n - 3: "LJ"}
            names[seat] = {0: "button", 1: "small blind", 2: "big blind", 3: "UTG"}.get(off) or late.get(off) or f"UTG+{off - 3}"
    return names


class Human: # You, at the terminal: the engine asks you for each decision and tells you what everyone did
    name = "you"

    def __init__(self, names):
        self.names, self.hand_number = names, 0  # names: seat -> who sits there (you are seat 0)

    def new_table(self, profiles):
        pass

    def new_hand(self, pool, board, mine, me=0, seats=(0, 1)):
        f, t, r = board
        self.board = [*pool.flops[f], pool.turns[f, t], pool.rivers[f, t, r]]
        self.mine, self.street, self.hand_number = COMBOS[mine], None, self.hand_number + 1
        n, (sb, bb) = len(seats), seats[:2]  # seats run from the small blind
        self.pos = positions(n, seats[0] if n == 2 else seats[-1])
        self.total = {s: 0.5 if s == sb else 1.0 if s == bb else 0.0 for s in sorted(seats)}  # chips in this hand
        self.cur = dict(self.total)                                                 # chips in this street
        self.folded = set()
        print("\n" + paint(f"━━━━━━━━━━  Hand {self.hand_number}  ━━━━━━━━━━", 1))
        blinds = lambda s: "you post" if s == 0 else f"{self.names[s]} posts"
        first = blinds(sb)
        print(f"{first[0].upper()}{first[1:]} the small blind (0.5bb), {blinds(bb)} the big blind (1bb).")
        if n == 2:
            print("Preflop the small blind acts first; after the flop the big blind does."
                  if sb == 0 else f"Preflop {self.names[sb]} acts first; you're already in for 1bb.")
        else:
            print(f"You're {'on the ' if self.pos[0] == 'button' else 'in '}{self.pos[0]}."
                  + ("  The button acts last after the flop." if self.pos[0] == "button" else ""))
        print(f"Your cards: {paint(show(self.mine), 1)}")

    def street_banner(self, street):
        if street == self.street:
            return
        self.street = street
        if street != "preflop":
            self.cur = {s: 0.0 for s in self.cur}
            n = {"flop": 3, "turn": 4, "river": 5}[street]
            print(f"\n{paint(street.upper(), 1)}   {show(self.board[:n])}      pot {sum(self.total.values()):.1f}bb")

    def see(self, who, spot, move, size=None, facing=0.0, situation=None):
        self.street_banner(spot.street)
        if move in ("bet", "raise"):
            chips = size * spot.pot
        elif move == "call":
            chips = spot.pot * facing / (1 + facing)  # facing = to_call / (pot - to_call)
        else:
            chips = 0.0
        self.total[who] += chips
        self.cur[who] += chips
        if move == "raise" or (move == "bet" and spot.street == "preflop"):
            text = f"raises to {self.cur[who]:.1f}bb"
        elif move == "bet":
            text = f"bets {chips:.1f}bb"
        elif move == "call":
            text = "completes the small blind (calls 0.5bb)" if spot.street == "preflop" and self.cur[who] == 1.0 and \
                chips == 0.5 else f"calls {chips:.1f}bb"
        else:
            text = {"fold": "folds", "check": "checks"}[move]
        if move == "fold":
            self.folded.add(who)
        print(f"  {paint('▶', 2)} {self.names[who]} {text}")

    def table(self):
        n = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[self.street]
        board = show(self.board[:n]) if n else paint("(no cards yet)", 2)
        print(paint(LINE, 2))
        print(f"  {self.street.capitalize():<9} board: {board}      pot {paint(f'{sum(self.total.values()):.1f}bb', 1)}")
        for s in self.total:
            cards = show(self.mine) if s == 0 else "?? ??"
            line = (f"  {self.names[s]:<14} {self.pos[s]:<12} {cards:<7}  stack {100 - self.total[s]:5.1f}bb"
                    f"   in this street {self.cur[s]:.1f}bb")
            print(paint(f"  {self.names[s]:<14} {self.pos[s]:<12} folded", 2) if s in self.folded else
                  paint(line, 1) if s == 0 else line)
        print(paint(LINE, 2))

    def decide(self, spot, to_call, can_raise, situation, rng):
        self.street_banner(spot.street)
        self.table()
        options = {}
        if to_call:
            options["f"] = ("fold", 0.0, "fold")
            note = "  (complete the small blind)" if spot.street == "preflop" and to_call == 0.5 and self.cur[0] == 0.5 else ""
            options["c"] = ("call", to_call, f"call {to_call:.1f}bb{note}")
        else:
            options["k"] = ("check", 0.0, "check")
        if can_raise and spot.behind > to_call:
            if spot.street == "preflop":
                sizes = [PREFLOP_SIZES[situation]]
            else:
                sizes = RAISE_SIZES if to_call else BET_SIZES
            for n, size in enumerate(sizes, 1):
                added = min(to_call + size * (spot.pot + to_call), spot.behind)
                all_in = "  ALL IN" if added >= spot.behind else ""
                if to_call or spot.street == "preflop":
                    kind = "3-bet" if spot.street == "preflop" and situation == "vs_3bet" else "raise"
                    label = f"{kind} to {self.cur[0] + added:.1f}bb  (puts in {added:.1f}bb){all_in}"
                    move = "raise" if to_call else "bet"
                else:
                    label, move = f"bet {added:.1f}bb  ({BET_NAMES.get(size, '')}){all_in}", "bet"
                options[str(n)] = (move, added, label)
        prompt = f"{to_call:.1f}bb to call" if to_call else "nothing to call"
        print(f"  Your move ({prompt}):")
        for key, (_, _, label) in options.items():
            print(f"    {paint(key, 1)}  {label}")
        choice = ask("  >", None, str.lower, lambda c: None if c in options else f"Pick one of: {', '.join(options)}")
        move, added, _ = options[choice]
        self.total[0] += added
        self.cur[0] += added
        if move == "fold":
            self.folded.add(0)
        return move, added

    def reveal(self, who, their_combo):
        if not getattr(self, "shown", False):
            self.shown = True
            print(f"\n{paint('SHOWDOWN', 1)}   board {show(self.board)}")
            if 0 not in self.folded:
                print(f"  {'You':<14} {show(self.mine)}   {hand_name(self.mine, self.board)}")
        theirs = COMBOS[their_combo]
        print(f"  {self.names[who]:<14} {show(theirs)}   {hand_name(theirs, self.board)}")

    def saw_response(self, who, called):
        pass

    def showdown(self, who, their_bets):
        pass

    def finish_hand(self, net):
        self.shown = False
        if net > 0:
            print(paint(f"  You win {net:.1f}bb", 32))
        elif net < 0:
            print(paint(f"  You lose {-net:.1f}bb", 31))
        else:
            print("  You break even" if 0 in self.folded or len(self.total) > 2 else "  Split pot")


def opponent_for(kind, seats): # One bot you'll play; seats: everyone else at the table, so it keeps a read on each
    if kind in ("ml bot", "old ml bot"):
        path = MODEL_PATH if kind == "ml bot" else OLD_MODEL
        if not os.path.exists(path):
            sys.exit("No trained ML bot yet. Run: python3 bots/train.py" if kind == "ml bot" else
                     "No old ML bot saved (it's the one trained without self-play).")
        bot = MLAgent("ML bot" if kind == "ml bot" else "old ML bot", explore=0.0, models=joblib.load(path))
        bot.finish_hand = lambda net: None  # plays what it learned; doesn't retrain mid-session
    elif kind == "hand-built bot":
        bot = Agent("hand-built bot")
    else:
        return ProfilePlayer(TEST_PRESETS[kind])
    bot.new_table({s: None for s in seats})  # it starts with no read on anyone and builds them as you play
    return bot


def session(kinds, seed=None):
    pool = load()
    world, own = np.random.default_rng(seed), np.random.default_rng(None if seed is None else seed + 1)
    n = len(kinds) + 1
    bots = [opponent_for(k, [s for s in range(n) if s != seat]) for seat, k in enumerate(kinds, 1)]
    names = {0: "You"}
    for seat, bot in enumerate(bots, 1):
        same = sum(b.name == bot.name for b in bots) > 1
        names[seat] = f"{bot.name} {seat}" if same else bot.name
    you = Human(names)
    who = f"the {bots[0].name}" if n == 2 else ", ".join(names[s] for s in range(1, n))
    print(paint(f"\n{'Heads-up' if n == 2 else f'{n}-handed'} vs {who}", 1) + ": blinds 0.5/1bb, everyone starts "
          "each hand with 100bb, and the button moves every hand. Ctrl-C to stop.")
    nets = []
    try:
        while True:
            net = play_table([you, *bots], pool, world, own, button=len(nets) % n)[0]
            you.finish_hand(net)
            nets.append(net)
            total = sum(nets)
            print(paint(f"  Session: {total:+.1f}bb over {len(nets)} hand{'s' * (len(nets) > 1)}", 32 if total > 0 else 31 if total < 0 else 0))
            if input("  Enter for the next hand, q to stop: ").strip().lower() == "q":
                break
    except (KeyboardInterrupt, EOFError):  # stopping mid-hand still shows the summary
        print()
    if not nets:
        return
    total = sum(nets)
    m, h = bb100(nets) if len(nets) > 1 else (total * 100, float("nan"))
    print(f"\nFinal: {total:+.1f}bb over {len(nets)} hands = {m:+.0f} bb/100" +
          (f" ± {h:.0f} (95% CI)" if len(nets) > 1 else ""))
    if len(nets) > 1:
        verdict = ("you beat them" if m - h > 0 else "they beat you" if m + h < 0 else
                   "too close to call yet; it takes a few hundred hands to know")
        print(f"  → {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Play against the ML bot, the hand-built bot or presets, heads-up or at a full table.")
    parser.add_argument("--opponent", default=None, help=f"Who to play: one of {', '.join(OPPONENTS)}; "
                                                         "or several, comma-separated, one per seat (asked if left out)")
    parser.add_argument("--players", type=int, default=None, help="Players at the table, you included (2-9; asked if left out)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, to replay the same cards")
    parser.add_argument("--no-color", action="store_true", help="Plain text, no colours")
    args = parser.parse_args()
    global COLOR
    COLOR = COLOR and not args.no_color
    valid = lambda text: next((f"'{k}' isn't one of: {', '.join(OPPONENTS)}" for k in text.split(",")
                               if k.strip() not in OPPONENTS), None)
    players = args.players or ask("Players at the table, you included (2-9)", 2, int,
                                  lambda n: None if 2 <= n <= 9 else "Between 2 and 9.")
    if not 2 <= players <= 9:
        sys.exit("--players must be between 2 and 9")
    text = args.opponent or ask(f"Opponents ({', '.join(OPPONENTS)}; comma-separate to mix seats)", "ml bot",
                                str.lower, valid)
    if valid(text):
        sys.exit(valid(text))
    kinds = [k.strip() for k in text.split(",")]
    session([kinds[i % len(kinds)] for i in range(players - 1)], args.seed)


if __name__ == "__main__":
    interactive_loop(main)
