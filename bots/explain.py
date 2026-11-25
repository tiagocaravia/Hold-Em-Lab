import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so "poker" imports work

import joblib
import numpy as np
from sklearn.inspection import permutation_importance

from bots.arena import play
from bots.ml_agent import FEATURES, GUIDED, MODEL_PATH, MLAgent, predict
from bots.pool import load
from poker.profiles import TEST_PRESETS

SHORT = {"equity vs their range": "equity vs range", "guided value": "guided value", "size (pots)": "size",
         "fold chance (guided)": "fold chance", "equity when called (guided)": "equity if called",
         "call rate at this size (read)": "size call rate", "times we bet this size": "size bets seen",
         "stack behind (pots)": "stack (pots)", "their call rate (read)": "call rate",
         "bluff share (read)": "bluff share", "to call (pots)": "to call"}  # short labels for the animation
BINS = np.linspace(0, 1, 11)  # hand strength buckets


def collect(model, hands, seed): # Real spots the trained bot faces, with a little exploring so every option shows up
    bot = MLAgent("explain", explore=0.15, models=model)
    play(bot, load(), hands, 1000, seed, TEST_PRESETS)
    return np.array([x for x, _ in bot.data])


def reliance(model, X, names=FEATURES): # How much the model's own predictions change when each input is shuffled
    X = np.asarray(X, dtype=float)[:, :model.n_features_in_]
    target = model.predict(X)
    result = permutation_importance(model, X, target, scoring="r2", n_repeats=5, random_state=0)
    order = np.argsort(result.importances_mean)
    return [names[i] for i in order], result.importances_mean[order], result.importances_std[order]


def explain(hands=20000, seed=99, save=None):
    if not os.path.exists(MODEL_PATH):
        sys.exit("No trained model yet. Run: python3 bots/train.py")
    model = joblib.load(MODEL_PATH)
    print(f"Playing {hands:,} hands with the trained bot to collect spots...")
    X = collect(model, hands, seed)
    print(f"  {len(X):,} decisions")

    import matplotlib.pyplot as plt
    labels, mean, std = reliance(model, X)
    used = mean > 0.005  # hide inputs it ignores
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.array(labels)[used], mean[used], xerr=std[used], color="#2a6f97", capsize=2, height=0.6)
    ax.set_title("What the ML bot relies on (every street, every option)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Change in its corrections when this input is scrambled (drop in R²)", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ignored = [l for l, u in zip(labels, used) if not u]
    if ignored:
        ax.text(0.98, 0.03, "ignores: " + ", ".join(ignored), transform=ax.transAxes, ha="right", fontsize=8, color="#666666")
    print("  most relied on: " + ", ".join(f"{l} ({m:.2f})" for l, m in zip(labels[::-1][:5], mean[::-1][:5])))
    print()
    print_tell(sizing_tell(model, collect_spots(load(), seed + 1)), "Does its bet size give its hand away?")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120)
        print(f"\nSaved to {save}")
    else:
        plt.show()


# --- the training animation ---

FREE = ["check", "bet ½ pot", "bet ¾ pot", "bet 1¼ pot"]
FACING = ["fold", "call", "raise"]
COLORS = {"check": "#c9c9c9", "bet ½ pot": "#9ecae1", "bet ¾ pot": "#4292c6", "bet 1¼ pot": "#08519c",
          "fold": "#e0e0e0", "call": "#74c476", "raise": "#d94801"}


def collect_spots(pool, seed, hands=6000): # A fixed set of decisions to replay every round
    bot = MLAgent("spots", explore=0.0)
    bot.spot_log = []
    play(bot, pool, hands, 1000, seed, TEST_PRESETS)
    groups = {"free": [], "facing": [], "preflop": []}
    for facing, street, rows in bot.spot_log:
        if street == "preflop":
            groups["preflop"].append(rows)
        elif facing:
            groups["facing"].append(rows)
        elif len(rows) == 4:  # checked to us with every bet size affordable
            groups["free"].append(rows)
    return groups


def choices(model, decisions, can_fold): # What the bot picks in each decision: index into its options (-1 = fold)
    flat = np.array([r for rows in decisions for r in rows])
    values = flat[:, GUIDED] + (predict(model, flat) if model is not None else 0)
    picks, i = [], 0
    for rows in decisions:
        v = values[i:i + len(rows)]
        best = int(np.argmax(v))
        picks.append(-1 if can_fold and v[best] < 0 else best)
        i += len(rows)
    return np.array(picks)


def policy(model, spots): # Share of each action by hand strength, for the three kinds of decision
    out = {}
    for kind, labels in (("free", FREE), ("facing", FACING), ("preflop", FACING)):
        decisions = spots[kind]
        picks = choices(model, decisions, can_fold=kind != "free")
        strength = np.array([rows[0][1] for rows in decisions])
        bins = np.clip(np.digitize(strength, BINS) - 1, 0, 9)
        if kind == "free":
            action = picks
        else:  # fold / call / any raise
            action = np.where(picks < 0, 0, np.where(picks == 0, 1, 2))
        out[kind] = np.array([[np.mean(action[bins == b] == a) if np.any(bins == b) else 0 for b in range(10)]
                              for a in range(len(labels))])
    return out


def sizing_tell(model, spots): # Does the bet size give the hand away? Strength and bluff share at each size
    decisions = spots["free"]
    picks = choices(model, decisions, can_fold=False)
    strength = np.array([rows[0][1] for rows in decisions])
    draw = np.array([rows[0][2] for rows in decisions])
    out = {}
    for a, label in enumerate(FREE):
        chosen = picks == a
        if chosen.any():
            out[label] = {"share": chosen.mean(), "strength": strength[chosen].mean(),
                          "bluffs": ((strength < 0.5) & (draw == 0))[chosen].mean()}
    return out


def print_tell(tell, title):
    print(f"  {title}")
    print(f"    {'when checked to':<16}{'how often':>10}{'avg strength':>14}{'bluffs':>9}")
    for label, t in tell.items():
        print(f"    {label:<16}{t['share']:>10.0%}{t['strength']:>14.2f}{t['bluffs']:>9.0%}")


def animate_training(frames, spots, baselines, path): # One frame per round: results, decisions, what it relies on
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    sample = np.array([r for rows in spots["free"] + spots["facing"] + spots["preflop"] for r in rows])
    sample = sample[np.random.default_rng(0).choice(len(sample), min(4000, len(sample)), replace=False)]
    for f in frames:
        f["policy"] = policy(f["models"], spots)
        if f["models"] is not None:
            f["reliance"] = reliance(f["models"], sample)
    rel_max = max(f["reliance"][1].max() for f in frames if "reliance" in f) * 1.1
    trained = [f["trained"] for f in frames[1:]]
    ylo = min(list(baselines.values()) + [f["test"][0] - f["test"][1] for f in frames[1:]]) - 5
    yhi = max(list(baselines.values()) + [f["test"][0] + f["test"][1] for f in frames[1:]]) + 5
    elo = min([-10] + [f["edge"][0] - f["edge"][1] for f in frames[1:]]) - 2
    ehi = max([5] + [f["edge"][0] + f["edge"][1] for f in frames[1:]]) + 3
    centers = (BINS[:-1] + BINS[1:]) / 2

    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    (ax_win, ax_free, ax_facing), (ax_edge, ax_pre, ax_rel) = axes

    def stacked(ax, shares, labels, title, xlabel):
        bottom = np.zeros(10)
        for a, label in enumerate(labels):
            ax.bar(centers, shares[a], width=0.09, bottom=bottom, color=COLORS[label], label=label)
            bottom += shares[a]
        ax.set(xlim=(0, 1), ylim=(0, 1), title=title, xlabel=xlabel, ylabel="Share of decisions")
        ax.legend(fontsize=7, loc="upper left", ncol=2)

    def draw(i):
        f = frames[i]
        for ax in axes.flat:
            ax.clear()
        fig.suptitle(f"Training the ML bot: {f['label']}", fontsize=14, fontweight="bold")
        done = frames[1:i + 1]

        for name, color in (("no read", "tab:blue"), ("oracle", "tab:purple")):
            ax_win.axhline(baselines[name], color=color, linestyle="--", linewidth=2, label=name)
        if done:
            ax_win.errorbar([d["trained"] for d in done], [d["test"][0] for d in done],
                            yerr=[d["test"][1] for d in done], color="black", marker="o", capsize=3, label="ML bot")
        ax_win.set(xlim=(0, max(trained) * 1.05), ylim=(ylo, yhi), xlabel="Training hands",
                   ylabel="bb/100 on held-out test", title="Win rate on opponents it hasn't seen")
        ax_win.legend(fontsize=8, loc="lower right")
        ax_win.grid(True, linestyle="--", alpha=0.4)

        ax_edge.axhline(0, color="tab:blue", linestyle="--", linewidth=2)
        if done:
            ax_edge.errorbar([d["trained"] for d in done], [d["edge"][0] for d in done],
                             yerr=[d["edge"][1] for d in done], color="black", marker="o", capsize=3)
        ax_edge.set(xlim=(0, max(trained) * 1.05), ylim=(elo, ehi), xlabel="Training hands", ylabel="bb/100 (95% CI)",
                    title="Edge over 'no read' (the simple hand-built bot)")
        ax_edge.grid(True, linestyle="--", alpha=0.4)

        p = f["policy"]
        stacked(ax_free, p["free"], FREE, "Checked to it (flop to river)", "Hand strength (share of hands it beats)")
        stacked(ax_facing, p["facing"], FACING, "Facing a bet (flop to river)", "Hand strength (share of hands it beats)")
        stacked(ax_pre, p["preflop"], FACING, "Preflop", "Hand strength (equity vs a random hand)")

        ax_rel.set_title("What it relies on")
        if "reliance" not in f:
            ax_rel.text(0.5, 0.5, "no model yet:\nfollowing the hand-built guidance", ha="center", va="center",
                        transform=ax_rel.transAxes, color="#666666")
            ax_rel.set_xticks([]), ax_rel.set_yticks([])
        else:
            labels, mean, _ = f["reliance"]
            ax_rel.barh([SHORT.get(l, l) for l in labels[-6:]], mean[-6:], color="#2a6f97")
            ax_rel.set_xlim(0, rel_max)
            ax_rel.set_xlabel("drop in R² when scrambled", fontsize=8)
            ax_rel.tick_params(labelsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    order = list(range(len(frames))) + [len(frames) - 1] * 2  # hold the last frame
    anim = FuncAnimation(fig, draw, frames=order, interval=1300)
    anim.save(path, writer="pillow", fps=0.8, dpi=80)
    plt.close(fig)
    print(f"Saved training animation to {path}")


def main():
    parser = argparse.ArgumentParser(description="Show which inputs the trained ML bot relies on.")
    parser.add_argument("--hands", type=int, default=20000, help="Hands to play to collect spots")
    parser.add_argument("--save", default=None, help="Save the figure to this file instead of showing it")
    args = parser.parse_args()
    explain(args.hands, save=args.save)


if __name__ == "__main__":
    main()
