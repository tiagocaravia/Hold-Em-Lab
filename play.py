import argparse
import os
import sys

import joblib
import numpy as np

from bots.arena import ProfilePlayer, bb100, play_hand
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


class Human: # You, at the terminal: the engine asks you for each decision and tells you what the bot did
    name = "you"

    def __init__(self, opponent_name):
        self.them, self.hand_number, self.small_blind = opponent_name, 0, True

    def new_opponent(self, other):
        pass

    def new_hand(self, pool, board, mine):
        f, t, r = board
        self.board = [*pool.flops[f], pool.turns[f, t], pool.rivers[f, t, r]]
        self.mine, self.street, self.hand_number = COMBOS[mine], None, self.hand_number + 1
        me, them = (0.5, 1.0) if self.small_blind else (1.0, 0.5)
        self.total = {"you": me, "them": them}   # chips in this hand
        self.cur = dict(self.total)              # chips in this street
        print("\n" + paint(f"━━━━━━━━━━  Hand {self.hand_number}  ━━━━━━━━━━", 1))
        if self.small_blind:
            print(f"You post the small blind (0.5bb), {self.them} posts the big blind (1bb).")
            print("Preflop you act first: call 0.5bb more to stay in, raise, or fold.")
        else:
            print(f"{self.them} posts the small blind (0.5bb), you post the big blind (1bb).")
            print(f"Preflop {self.them} acts first; you're already in for 1bb.")
        print(f"Your cards: {paint(show(self.mine), 1)}")

    def street_banner(self, street):
        if street == self.street:
            return
        self.street = street
        if street != "preflop":
            self.cur = {"you": 0.0, "them": 0.0}
            n = {"flop": 3, "turn": 4, "river": 5}[street]
            pot = sum(self.total.values())
            print(f"\n{paint(street.upper(), 1)}   {show(self.board[:n])}      pot {pot:.1f}bb")

    def see(self, spot, move, size=None, facing=0.0, situation=None):
        self.street_banner(spot.street)
        if move in ("bet", "raise"):
            chips = size * spot.pot
        elif move == "call":
            chips = spot.pot * facing / (1 + facing)  # facing = to_call / (pot - to_call)
        else:
            chips = 0.0
        self.total["them"] += chips
        self.cur["them"] += chips
        if move == "raise" or (move == "bet" and spot.street == "preflop"):
            text = f"raises to {self.cur['them']:.1f}bb"
        elif move == "bet":
            text = f"bets {chips:.1f}bb"
        elif move == "call":
            text = "completes the small blind (calls 0.5bb)" if spot.street == "preflop" and chips == 0.5 else f"calls {chips:.1f}bb"
        else:
            text = "checks"
        print(f"  {paint('▶', 2)} {self.them} {text}")

    def table(self, to_call):
        pot = sum(self.total.values())
        n = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[self.street]
        board = show(self.board[:n]) if n else paint("(no cards yet)", 2)
        sb, bb = ("small blind", "big blind") if self.small_blind else ("big blind", "small blind")
        print(paint(LINE, 2))
        print(f"  {self.street.capitalize():<9} board: {board}      pot {paint(f'{pot:.1f}bb', 1)}")
        print(f"  You       {sb:<12} {show(self.mine):<7}  stack {100 - self.total['you']:5.1f}bb   in this street {self.cur['you']:.1f}bb")
        print(f"  {self.them:<9} {bb:<12} ?? ??    stack {100 - self.total['them']:5.1f}bb   in this street {self.cur['them']:.1f}bb")
        print(paint(LINE, 2))

    def decide(self, spot, to_call, can_raise, situation, rng):
        self.street_banner(spot.street)
        self.table(to_call)
        options = {}
        if to_call:
            options["f"] = ("fold", 0.0, "fold")
            note = "  (complete the small blind)" if spot.street == "preflop" and to_call == 0.5 and self.small_blind else ""
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
                    label = f"{kind} to {self.cur['you'] + added:.1f}bb  (puts in {added:.1f}bb){all_in}"
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
        self.total["you"] += added
        self.cur["you"] += added
        return move, added

    def opponent_folded(self):
        print(f"  {paint('▶', 2)} {self.them} folds")

    def reveal(self, their_combo):
        theirs = COMBOS[their_combo]
        print(f"\n{paint('SHOWDOWN', 1)}   board {show(self.board)}")
        print(f"  You       {show(self.mine)}   {hand_name(self.mine, self.board)}")
        print(f"  {self.them:<9} {show(theirs)}   {hand_name(theirs, self.board)}")

    def saw_response(self, called):
        pass

    def showdown(self, their_bets):
        pass

    def finish_hand(self, net):
        if net > 0:
            print(paint(f"  You win {net:.1f}bb", 32))
        elif net < 0:
            print(paint(f"  You lose {-net:.1f}bb", 31))
        else:
            print("  Split pot")


def opponent_for(kind): # The bot you'll play
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
    bot.new_opponent(None)  # it starts with no read on you and builds one as you play
    return bot


def session(kind, seed=None):
    pool = load()
    world, own = np.random.default_rng(seed), np.random.default_rng(None if seed is None else seed + 1)
    bot = opponent_for(kind)
    you = Human(bot.name)
    print(paint(f"\nHeads-up vs the {bot.name}", 1) + ": blinds 0.5/1bb, everyone starts each hand with 100bb, "
          "and the blinds swap every hand. Ctrl-C to stop.")
    nets = []
    try:
        while True:
            you.small_blind = len(nets) % 2 == 0
            net = play_hand(you, bot, pool, world, own, a_is_small_blind=you.small_blind)
            you.finish_hand(net)
            bot.finish_hand(-net)
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
        verdict = ("you beat it" if m - h > 0 else "it beat you" if m + h < 0 else
                   "too close to call yet; it takes a few hundred hands to know")
        print(f"  → {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Play heads-up against the ML bot, the hand-built bot or a preset.")
    parser.add_argument("--opponent", default=None, choices=OPPONENTS, help="Who to play (asked if left out)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, to replay the same cards")
    parser.add_argument("--no-color", action="store_true", help="Plain text, no colours")
    args = parser.parse_args()
    global COLOR
    COLOR = COLOR and not args.no_color
    kind = args.opponent or ask(f"Opponent ({', '.join(OPPONENTS)})", "ml bot", str.lower,
                                lambda k: None if k in OPPONENTS else f"One of: {', '.join(OPPONENTS)}")
    session(kind, args.seed)


if __name__ == "__main__":
    interactive_loop(main)
