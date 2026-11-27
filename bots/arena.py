import argparse
from dataclasses import replace
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so "poker" imports work

import numpy as np

from bots.ml_agent import PREFLOP_SIZES, STREETS, Spot, lineup
from bots.pool import load
from poker.profiles import PRESETS, bet_weights, call_weights, preflop_weights, raise_weights, random_profile
from poker.prompts import again, ask, interactive_loop, yes

BLINDS, STACK = (0.5, 1.0), 100.0  # small and big blind, starting stacks (bb)
BUCKETS = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000)]
SITUATIONS = ("vs_raise", "vs_3bet")   # the preflop spot a raise leaves the other player in, by raises so far


def situation_of(to_call, aggressions): # The preflop spot a player is in, by raises so far
    return ("vs_limp" if to_call == 0 else "open") if aggressions == 0 else SITUATIONS[aggressions - 1]


def their_move(profile, pool, board, theirs, street, pot, to_call, can_raise, aggressions, dice):
    # The opponent's choice, from the same rules the bots range with. Returns (move, chips put in now)
    u = next(dice)
    if street == "preflop":
        w = {k: float(np.ravel(v)[0]) for k, v in preflop_weights(
            np.array([pool.preflop_pct[theirs]]), situation_of(to_call, aggressions), profile).items()}
        if can_raise and u < w["raise"]:
            size = PREFLOP_SIZES[SITUATIONS[aggressions]]
            return ("raise" if to_call else "bet"), to_call + size * (pot + to_call)
        if u < w["raise"] + w["call"] or to_call == 0:
            return ("call" if to_call else "check"), to_call
        return "fold", 0.0
    idx = {"flop": board[:1], "turn": board[:2], "river": board}[street]
    s = np.array([float(np.nan_to_num(pool.strength[street][idx][theirs]))])
    d = np.array([int(pool.draws[street][idx][theirs]) if street != "river" else 0])
    if to_call == 0:
        if can_raise and u < float(bet_weights(s, d, profile.bet_size, street, profile)[0]):
            return "bet", profile.bet_size * pot
        return "check", 0.0
    facing = to_call / (pot - to_call)
    r = float(raise_weights(s, d, facing, street, profile)[0]) if can_raise else 0.0
    c = float(call_weights(s, d, facing, street, profile)[0])
    if u < r:
        return "raise", to_call + 1.0 * (pot + to_call)
    if u < r + (1 - r) * c:
        return "call", to_call
    return "fold", 0.0


class ProfilePlayer: # A rule-based opponent: plays its Profile with the shared rules
    def __init__(self, profile):
        self.profile, self.name = profile, profile.name

    def new_hand(self, pool, board, mine, me=0, seats=()):
        self.pool, self.board, self.mine = pool, board, mine

    def decide(self, spot, to_call, can_raise, situation, rng):
        move, added = their_move(self.profile, self.pool, self.board, self.mine, spot.street, spot.pot, to_call,
                                 can_raise, spot.aggressions, self.dice)
        return move, min(added, spot.behind)

    def new_table(self, profiles):
        pass

    def see(self, *args):
        pass

    def saw_response(self, *args):
        pass

    def showdown(self, *args):
        pass

    def finish_hand(self, net):
        pass


def payouts(put, live, rank): # What each seat wins: every layer of chips goes to the best live hand that put it in (side pots)
    win, below = [0.0] * len(put), 0.0
    levels = sorted({put[s] for s in live})
    for n, level in enumerate(levels):
        top = max(put) if n == len(levels) - 1 else level  # the last layer takes anything above it too
        layer = sum(min(x, top) - min(x, below) for x in put)
        eligible = [s for s in live if put[s] >= level]
        best = max(rank[s] for s in eligible)
        winners = [s for s in eligible if rank[s] == best]
        for s in winners:
            win[s] += layer / len(winners)
        below = level
    return win


def play_table(players, pool, world, own, button=0): # One hand at a table of 2-9 seats, preflop to river. Returns each seat's net bb
    # Any player works in any seat: a bot, a ProfilePlayer, or a person (play.py). Seats are 0..n-1; the button moves
    n = len(players)
    board, cards = pool.deal(world, n)
    dice = iter(world.random(20 * n))  # random choices for rule-based players, drawn up front so every bot faces the same ones
    ring = lambda first: [(first + k) % n for k in range(n)]
    sb, bb = (button, (button + 1) % n) if n == 2 else ((button + 1) % n, (button + 2) % n)  # heads-up the button is the small blind
    for s, player in enumerate(players):  # each player: its cards, its seat, and every seat from the small blind round
        player.dice = dice
        player.new_hand(pool, board, cards[s], s, tuple(ring(sb)))
    put = [0.0] * n
    put[sb], put[bb] = BLINDS
    behind = [STACK - x for x in put]
    live = list(range(n))           # seats that haven't folded
    shown = [[] for _ in range(n)]  # each seat's bets and raises after the flop, as (strength, draw, size): shown at showdown
    for street in STREETS:
        order = ring((bb + 1) % n) if street == "preflop" else ring((button + 1) % n)
        cur = list(put) if street == "preflop" else [0.0] * n  # chips in on this street
        needs = {s for s in live if behind[s] > 0}            # who still has to act
        if street != "preflop" and len(needs) < 2:
            continue                                          # everyone else is all in: just deal the cards
        aggressions, last_bet, k = 0, None, 0
        while needs and len(live) > 1:
            p = next(order[(k + j) % n] for j in range(n) if order[(k + j) % n] in needs)
            k = order.index(p) + 1
            others = [s for s in live if s != p]
            to_call = min(max(cur[s] for s in live) - cur[p], behind[p])
            pot = sum(put)
            can_raise = aggressions < 2 and any(behind[o] > 0 for o in others) and behind[p] > to_call
            last = [s for s in ring((button + 1) % n) if s in live][-1]
            aggressor = last_bet if last_bet is not None else bb if street == "preflop" else None
            spot = Spot(pool, board, street, cards[p], pot, put[p], behind[p], aggressions, tuple(others),
                        len(needs - {p}), p == last, aggressor)
            move, added = players[p].decide(spot, to_call, can_raise, SITUATIONS[min(aggressions, 1)], own)
            added = min(added, behind[p])
            if last_bet is not None and last_bet != p:  # how p answered the last bet or raise
                players[last_bet].saw_response(p, move != "fold")
            for o in range(n):  # everyone sees the action
                if o != p:
                    players[o].see(p, Spot(pool, board, street, cards[o], pot), move, added / pot,
                                   to_call / max(pot - to_call, 1e-9), situation_of(to_call, aggressions))
            needs.discard(p)
            if move == "fold":
                live.remove(p)
                continue
            if street != "preflop" and move in ("bet", "raise"):
                idx = {"flop": board[:1], "turn": board[:2], "river": board}[street]
                shown[p].append((float(np.nan_to_num(pool.strength[street][idx][cards[p]])),
                                 int(pool.draws[street][idx][cards[p]]) if street != "river" else 0, added / pot))
            put[p] += added
            cur[p] += added
            behind[p] -= added
            if move in ("bet", "raise"):
                aggressions, last_bet = aggressions + 1, p
                needs = {s for s in live if s != p and behind[s] > 0}
        if len(live) == 1:
            break
    if len(live) == 1:
        win = [0.0] * n
        win[live[0]] = sum(put)
    else:
        for x in live:
            for y in live:
                if x != y:
                    players[x].showdown(y, shown[y])
        for x, player in enumerate(players):
            if hasattr(player, "reveal"):  # players who want to see the other hands (a person, the interface)
                for y in live:
                    if y != x:
                        player.reveal(y, cards[y])
        win = payouts(put, live, [int(pool.rank[board][c]) for c in cards])
    return [w - x for w, x in zip(win, put)]


def play_hand(a, b, pool, world, own, a_is_small_blind): # Heads-up: one hand between two seats. Returns a's net bb
    return play_table([a, b], pool, world, own, button=0 if a_is_small_blind else 1)[0]


def play(agent, pool, hands, session_len, seed, presets=PRESETS, league=(), self_share=0.0, players=2):
    # One bot vs a new table every session: each other seat a rule-based profile, or (self_share of seats)
    # a frozen ML bot drawn from league, a list of trained models: self-play.
    # players: the table size, or (fewest, most) for a random size each session
    lo, hi = (players, players) if isinstance(players, int) else players
    world = np.random.default_rng(seed)    # deals and opponents' choices: identical for every bot
    own = np.random.default_rng(seed + 1)  # the bot's own mixing
    profile_rng = random.Random(seed)
    league_rng = random.Random(seed + 7)   # separate dice, so profile sessions stay the same with or without self-play
    size_rng = random.Random(seed + 13)
    results = []                           # (hand number within session, opponent type or table size, our net bb)
    for i in range(hands):
        if i % session_len == 0:
            n = size_rng.randint(lo, hi)
            table, profiles = [agent], {}
            for seat in range(1, n):
                profile = random_profile(profile_rng, presets)
                if league and league_rng.random() < self_share:
                    from bots.ml_agent import MLAgent
                    opponent = MLAgent("self-play", explore=0.0, models=league_rng.choice(league)).frozen("self-play")
                    opponent.new_table({s: None for s in range(n) if s != seat})  # it reads the table too
                    profile, profiles[seat] = replace(profile, name="self-play"), None
                else:
                    opponent, profiles[seat] = ProfilePlayer(profile), profile
                table.append(opponent)
            agent.new_table(profiles)
            label = profile.name if n == 2 else f"{n} players"
        net = play_table(table, pool, world, own, button=i % n)[0]
        agent.finish_hand(net)
        results.append((i % session_len, label, net))
    return results


_POOL = None  # each worker process loads the pool once


def _play_block(job): # One worker: play a block of sessions, return results and anything the bot learned from
    global _POOL
    make, hands, session_len, seed, presets, league, self_share, players = job
    if _POOL is None:
        _POOL = load()
    agent = make()
    if hasattr(agent, "rng"):
        agent.rng = np.random.default_rng(seed)  # each worker explores with its own dice
    results = play(agent, _POOL, hands, session_len, seed, presets, league, self_share, players)
    return results, getattr(agent, "data", [])


def play_parallel(workers, make, hands, session_len, seed, presets=PRESETS, league=(), self_share=0.0, players=2): # play() on every core
    # returns (results, what the bot learned from)
    # make: a function that builds a fresh bot (e.g. functools.partial(Agent, "no read", learns=False)); same seed -> same hands
    sessions = max(1, hands // session_len)
    blocks = min(os.cpu_count(), sessions)
    sizes = [(sessions // blocks + (i < sessions % blocks)) * session_len for i in range(blocks)]  # sessions shared evenly
    sizes[-1] += hands - sum(sizes)
    lo, hi = (players, players) if isinstance(players, int) else players
    jobs = [(make, n, session_len, seed * 1000 + i, presets, league, self_share, lo + i % (hi - lo + 1))
            for i, n in enumerate(sizes)]  # table sizes shared evenly across blocks
    parts = workers.map(_play_block, jobs)
    return [r for part in parts for r in part[0]], [x for part in parts for x in part[1]]


def bb100(nets): # Win rate in bb per 100 hands, with a 95% CI half-width
    n = len(nets)
    if n < 2:
        return float("nan"), float("nan")
    mean = sum(nets) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in nets) / (n - 1))
    return mean * 100, 1.96 * sd / math.sqrt(n) * 100


def report(all_results, session_len):
    names = list(all_results)
    print(f"\n{'':<18}" + "".join(f"{n:>20}" for n in names))

    print("Overall (bb/100)")
    row = "  all hands       "
    for n in names:
        m, h = bb100([r[2] for r in all_results[n]])
        row += f"{m:>+12.1f} ± {h:<5.1f}"
    print(row)

    print("\nBy opponent type")
    for kind in sorted({r[1] for r in all_results[names[0]]}):
        row = f"  {kind:<16}"
        for n in names:
            m, h = bb100([r[2] for r in all_results[n] if r[1] == kind])
            row += f"{m:>+12.1f} ± {h:<5.1f}"
        print(row)

    print("\nBy hands into a session with a new opponent (how fast it learns)")
    buckets = [(lo, hi) for lo, hi in BUCKETS if lo < session_len]
    for lo, hi in buckets:
        hi = min(hi, session_len)
        row = f"  hands {lo:>4}-{hi:<5}  "
        for n in names:
            m, h = bb100([r[2] for r in all_results[n] if lo <= r[0] < hi])
            row += f"{m:>+12.1f} ± {h:<5.1f}"
        print(row)

    base = all_results.get("no read")
    if not base:
        return
    for name in [n for n in all_results if n != "no read"]:
        print(f"\n{name}: edge over 'no read' on the same hands (bb/100)")
        first = None
        for lo, hi in buckets:
            hi = min(hi, session_len)
            diffs = [a[2] - b[2] for a, b in zip(all_results[name], base) if lo <= a[0] < hi]
            m, h = bb100(diffs)
            print(f"  hands {lo:>4}-{hi:<5}  {m:>+8.1f} ± {h:.1f}")
            if first is None and m - h > 0:
                first = lo
        m, h = bb100([a[2] - b[2] for a, b in zip(all_results[name], base)])
        when = f"clearly ahead (whole 95% CI above 0) from about hand {first}" if first is not None else "never clearly ahead"
        print(f"  overall {m:+.1f} ± {h:.1f}; {when} against a new opponent")


def plot(all_results, session_len, save=None):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Bot arena: flop to river vs random opponents", fontsize=13, fontweight="bold")
    colors = {name: plt.cm.tab10(i) for i, name in enumerate(all_results)}  # same color in both panels

    for name, results in all_results.items():
        total, curve = 0.0, []
        for _, _, net in results:
            total += net
            curve.append(total)
        ax1.plot(range(1, len(curve) + 1), curve, label=name, linewidth=1.4, color=colors[name])
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_xlabel("Hands played")
    ax1.set_ylabel("Total profit (bb)")
    ax1.set_title("Cumulative profit")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9)

    buckets = [(lo, min(hi, session_len)) for lo, hi in BUCKETS if lo < session_len]
    labels = [f"{lo}-{hi}" for lo, hi in buckets]
    base = all_results["no read"]
    for name, results in all_results.items():
        if name == "no read":
            continue
        stats = [bb100([a[2] - b[2] for a, b in zip(results, base) if lo <= a[0] < hi]) for lo, hi in buckets]
        ax2.errorbar(range(len(buckets)), [m for m, _ in stats], yerr=[h for _, h in stats],
                     marker="o", capsize=3, label=name, color=colors[name])
    ax2.axhline(0, color=colors["no read"], linewidth=1.2, label="no read")
    ax2.set_xticks(range(len(buckets)))
    ax2.set_xticklabels(labels, rotation=30)
    ax2.set_xlabel("Hands into a session with a new opponent")
    ax2.set_ylabel("Edge over 'no read' (bb/100, 95% CI)")
    ax2.set_title("How fast each bot figures out a new opponent")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
        print(f"\nSaved plot to {save}")
    else:
        plt.show()


def parse_players(text): # "6" -> 6, "2-6" -> (2, 6): a table size, or a range for a random size each session
    lo, _, hi = str(text).partition("-")
    players = int(lo) if not hi else (int(lo), int(hi))
    sizes = [players] if isinstance(players, int) else list(players)
    if not all(2 <= n <= 9 for n in sizes) or sizes[0] > sizes[-1]:
        raise ValueError("players must be between 2 and 9, e.g. 6 or 2-6")
    return players


def run(hands, session_len, seed, show_plot=True, save=None, players=2):
    pool = load()
    size = f"{players}" if isinstance(players, int) else f"{players[0]}-{players[1]}"
    print(f"\n=== Arena: {hands:,} hands at {size}-player tables, new opponents every {session_len:,} hands ===")
    all_results = {}
    for agent in lineup():
        start = time.time()
        all_results[agent.name] = play(agent, pool, hands, session_len, seed, players=players)
        print(f"  {agent.name:<18} done in {time.time() - start:.1f}s")
    report(all_results, session_len)
    if show_plot:
        plot(all_results, session_len, save)


def _players_error(text):
    try:
        parse_players(text)
    except ValueError:
        return "Between 2 and 9, e.g. 6 or 2-6."


def interactive():
    print("=== Bot arena ===  (enter keeps the value in brackets)\n")
    hands, session_len, seed, show, players = 100000, 1000, 1, True, "2"
    while True:
        players = ask("Players per table, the bot included (2-9, or a range like 2-6)", players, str,
                      lambda t: _players_error(t))
        hands = ask("Hands to play", hands, int, lambda n: None if n >= 100 else "At least 100.")
        session_len = ask("Hands per opponent before switching", session_len, int,
                          lambda n: None if n >= 10 else "At least 10.")
        seed = ask("Random seed", seed, int)
        show = yes("Show the plot? (y/n)", show)
        run(hands, session_len, seed, show, players=parse_players(players))
        if not again():
            break


def main():
    if len(sys.argv) == 1:
        return interactive_loop(interactive)
    parser = argparse.ArgumentParser(description="Play bot versions against random opponents and compare win rates. "
                                                 "Run with no options to be asked for each input.")
    parser.add_argument("--hands", type=int, default=100000, help="Hands each bot plays")
    parser.add_argument("--session", type=int, default=1000, help="Hands per opponent before a new one")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--save", default=None, help="Save the plot to this file instead of showing it")
    parser.add_argument("--no-plot", action="store_true", help="Skip the plot")
    parser.add_argument("--players", default="2", metavar="N or LO-HI",
                        help="Players per table, the bot included: 6, or 2-6 for a random size each session (2-9)")
    args = parser.parse_args()
    try:
        players = parse_players(args.players)
    except ValueError as e:
        sys.exit(f"--{e}")
    run(args.hands, args.session, args.seed, not args.no_plot, args.save, players)


if __name__ == "__main__":
    main()
