import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so "poker" imports work

import numpy as np

from bots.ml_agent import STREETS, Spot, lineup
from bots.pool import load
from poker.profiles import PRESETS, bet_weights, call_weights, random_profile
from poker.prompts import again, ask, interactive_loop, yes

POT, STACK = 6.0, 100.0  # bb in the pot on the flop (3 each), and starting stacks
BUCKETS = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000)]


def their_bets(profile, strength, draw, size, street, u): # Does the opponent bet? (same rules the bots range with)
    return u < float(bet_weights(np.array([strength]), np.array([draw]), size, street, profile)[0])


def their_calls(profile, strength, draw, size, street, u):
    return u < float(call_weights(np.array([strength]), np.array([draw]), size, street, profile)[0])


def play_hand(agent, profile, pool, world, own, we_act_first): # One hand, flop to river. Returns our net bb
    board, mine, theirs = pool.deal(world)
    dice = iter(world.random(12))   # the opponent's random choices, drawn up front so every bot faces the same ones
    agent.new_hand(pool, board, mine)
    pot, ours, behind = POT, POT / 2, STACK - POT / 2  # pot, what we've put in, what each player has left
    shown = []                                          # their bets as (strength, draw), learned from at showdown
    for street in STREETS:
        spot = Spot(pool, board, street, mine, pot, ours, behind)
        idx = spot.index()
        s = float(np.nan_to_num(pool.strength[street][idx][theirs]))
        d = int(pool.draws[street][idx][theirs]) if street != "river" else 0
        rolls = [next(dice) for _ in range(4)]
        order = ["us", "them"] if we_act_first else ["them", "us"]
        for turn, who in enumerate(order):  # the first player acts; if they check, the second can bet
            if behind <= 0:
                break
            if who == "us":
                bet = agent.act(spot, behind)
                if bet:
                    size = bet / pot
                    called = their_calls(profile, s, d, size, street, rolls[2 + turn])
                    agent.saw_response(called)
                    if not called:
                        return pot - ours
                    agent.see(spot, "call", size)
                    pot, ours, behind = pot + 2 * bet, ours + bet, behind - bet
                    break
                if turn == 0:
                    continue
            else:
                size = profile.bet_size
                bet = min(size * pot, behind)
                if their_bets(profile, s, d, size, street, rolls[turn]):
                    shown.append((s, d))
                    agent.see(spot, "bet", bet / pot)
                    if not agent.respond(spot, bet, own):
                        return -ours
                    pot, ours, behind = pot + 2 * bet, ours + bet, behind - bet
                    break
                agent.see(spot, "check")
    diff = int(pool.rank[board][mine]) - int(pool.rank[board][theirs])
    agent.showdown(shown)
    return (pot if diff > 0 else pot / 2 if diff == 0 else 0) - ours


def play(agent, pool, hands, session_len, seed, presets=PRESETS): # One bot vs a new random opponent every session
    world = np.random.default_rng(seed)    # deals and opponents' choices: identical for every bot
    own = np.random.default_rng(seed + 1)  # the bot's own mixing
    profile_rng = random.Random(seed)
    results = []                           # (hand number within session, opponent type, net bb)
    for i in range(hands):
        if i % session_len == 0:
            profile = random_profile(profile_rng, presets)
            agent.new_opponent(profile)
        net = play_hand(agent, profile, pool, world, own, we_act_first=i % 2 == 0)
        agent.finish_hand(net)
        results.append((i % session_len, profile.name, net))
    return results


_POOL = None  # each worker process loads the pool once


def _play_block(job): # One worker: play a block of sessions, return results and anything the bot learned from
    global _POOL
    make, hands, session_len, seed, presets = job
    if _POOL is None:
        _POOL = load()
    agent = make()
    if hasattr(agent, "rng"):
        agent.rng = np.random.default_rng(seed)  # each worker explores with its own dice
    results = play(agent, _POOL, hands, session_len, seed, presets)
    return results, getattr(agent, "call_data", []), getattr(agent, "bet_data", [])


def play_parallel(workers, make, hands, session_len, seed, presets=PRESETS): # play() split across every core
    # make: a function that builds a fresh bot (e.g. functools.partial(Agent, "learner")); same seed -> same hands
    sessions = max(1, hands // session_len)
    blocks = min(os.cpu_count(), sessions)
    sizes = [(sessions // blocks + (i < sessions % blocks)) * session_len for i in range(blocks)]  # sessions shared evenly
    sizes[-1] += hands - sum(sizes)
    parts = workers.map(_play_block, [(make, n, session_len, seed * 1000 + i, presets) for i, n in enumerate(sizes)])
    results = [r for part in parts for r in part[0]]
    return results, [x for part in parts for x in part[1]], [x for part in parts for x in part[2]]


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


def run(hands, session_len, seed, show_plot=True, save=None):
    pool = load()
    print(f"\n=== Arena: {hands:,} hands, new opponent every {session_len:,} hands ===")
    all_results = {}
    for agent in lineup():
        start = time.time()
        all_results[agent.name] = play(agent, pool, hands, session_len, seed)
        print(f"  {agent.name:<18} done in {time.time() - start:.1f}s")
    report(all_results, session_len)
    if show_plot:
        plot(all_results, session_len, save)


def interactive():
    print("=== Bot arena ===  (enter keeps the value in brackets)\n")
    hands, session_len, seed, show = 100000, 1000, 1, True
    while True:
        hands = ask("Hands to play", hands, int, lambda n: None if n >= 100 else "At least 100.")
        session_len = ask("Hands per opponent before switching", session_len, int,
                          lambda n: None if n >= 10 else "At least 10.")
        seed = ask("Random seed", seed, int)
        show = yes("Show the plot? (y/n)", show)
        run(hands, session_len, seed, show)
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
    args = parser.parse_args()
    run(args.hands, args.session, args.seed, not args.no_plot, args.save)


if __name__ == "__main__":
    main()
