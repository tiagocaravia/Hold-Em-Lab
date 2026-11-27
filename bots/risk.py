import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so "poker" imports work

import numpy as np

from poker.equity import Z_95

START = 100  # below this many hands the running numbers are mostly noise, so the plots start here


def sharpe_half_width(sharpe, skew, kurtosis, n): # 95% CI half-width of a Sharpe ratio, allowing for fat tails
    # Mertens (2002): Var(SR) = (1 - skew * SR + (kurtosis - 1) / 4 * SR^2) / n. Poker results are far from normal
    # (rare big pots), so the plain 1/n version would make the interval look too tight.
    return Z_95 * math.sqrt(max(1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe ** 2, 0) / n)


def risk_stats(nets): # Mean, standard deviation and Sharpe ratio of each hand's net bb
    x = np.asarray(nets, dtype=float)
    n = len(x)
    if n < 2:
        return dict(n=n, mean=float("nan"), sd=float("nan"), sharpe=float("nan"), sharpe_hw=float("nan"),
                    bb100=float("nan"), sd100=float("nan"), sharpe100=float("nan"), sharpe100_hw=float("nan"))
    mean, sd = x.mean(), x.std(ddof=1)
    sharpe = mean / sd if sd > 0 else float("nan")
    if sd > 0:
        z = (x - mean) / x.std()
        hw = sharpe_half_width(sharpe, (z ** 3).mean(), (z ** 4).mean(), n)
    else:
        hw = float("nan")
    return dict(n=n, mean=mean, sd=sd, sharpe=sharpe, sharpe_hw=hw,
                bb100=mean * 100, sd100=sd * 10, sharpe100=sharpe * 10, sharpe100_hw=hw * 10)


def running(nets): # Running mean, standard deviation and Sharpe ratio after each hand (and the Sharpe's 95% CI)
    x = np.asarray(nets, dtype=float)
    n = np.arange(1, len(x) + 1)
    s1, s2, s3, s4 = (np.cumsum(x ** k) for k in (1, 2, 3, 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        var = np.maximum(s2 / n - mean ** 2, 0)  # population variance, for the moments
        sd = np.sqrt(var * n / (n - 1))
        sharpe = mean / sd
        m3 = s3 / n - 3 * mean * s2 / n + 2 * mean ** 3
        m4 = s4 / n - 4 * mean * s3 / n + 6 * mean ** 2 * s2 / n - 3 * mean ** 4
        skew, kurt = m3 / var ** 1.5, m4 / var ** 2
        hw = Z_95 * np.sqrt(np.maximum(1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2, 0) / n)
    return mean, sd, sharpe, hw


def report(all_results): # σ (how much a hand swings) and Sharpe ratio (mean / σ, no risk-free rate) for each bot
    # per 100 hands the mean grows 100x and σ 10x (hands are independent), so Sharpe per 100 hands is 10x per hand
    names = list(all_results)
    print("\nRisk: how much each hand swings (σ, bb) and the win rate for that swing (Sharpe = mean / σ)")
    print(f"  {'':<16}{'bb/hand':>10}{'σ/hand':>9}{'σ/100':>8}{'Sharpe/hand':>20}{'Sharpe/100':>18}")
    rows = [("all hands", None)] + [(kind, kind) for kind in sorted({r[1] for r in all_results[names[0]]})]
    for label, kind in rows:
        print(f" {label}" if kind is None else f"  {label}")
        for name in names:
            s = risk_stats([r[2] for r in all_results[name] if kind is None or r[1] == kind])
            print(f"    {name:<14}{s['mean']:>+10.2f}{s['sd']:>9.1f}{s['sd100']:>8.0f}"
                  f"{s['sharpe']:>+11.3f} ± {s['sharpe_hw']:<6.3f}{s['sharpe100']:>+9.2f} ± {s['sharpe100_hw']:<5.2f}")
    print("  σ/100 = σ/hand × 10 and Sharpe/100 = Sharpe/hand × 10 (hands are independent). Intervals are 95%,"
          " widened for fat tails.")


def _colors(names):
    import matplotlib.pyplot as plt
    return {name: plt.cm.tab10(i) for i, name in enumerate(names)}  # same colors as the arena plot


def _bins(all_results, count=60): # Shared histogram bins, ignoring the rarest 0.5% at each end
    nets = np.concatenate([[r[2] for r in res] for res in all_results.values()])
    lo, hi = np.percentile(nets, [0.5, 99.5])
    return np.linspace(lo, hi, count + 1)


def _setup(all_results, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    ax_sd, ax_sh, ax_hist = axes
    ax_sd.set(xscale="log", xlabel="Hands played (log scale)", ylabel="σ per hand (bb)",
              title="Standard deviation per hand")
    ax_sh.set(xscale="log", xlabel="Hands played (log scale)", ylabel="Sharpe ratio per 100 hands (95% CI)",
              title="Sharpe ratio (mean / σ)")
    ax_sh.axhline(0, color="gray", linewidth=0.8)
    ax_hist.set(yscale="log", xlabel="Net per hand (bb)", ylabel="Share of hands (log scale)",
                title="Where the swings come from")
    for ax in axes:
        ax.grid(True, which="major", linestyle="--", alpha=0.4)
    return fig, axes


def _limits(curves, n_total): # Axis ranges that fit every bot from hand START on, so the animation doesn't jump
    start = min(START, n_total) - 1
    sds = np.concatenate([sd[start:] for _, sd, _, _ in curves.values()])
    lows = np.concatenate([(sh - hw)[start:] * 10 for _, _, sh, hw in curves.values()])
    highs = np.concatenate([(sh + hw)[start:] * 10 for _, _, sh, hw in curves.values()])
    pad = lambda lo, hi: (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    return pad(np.nanmin(sds), np.nanmax(sds)), pad(min(np.nanmin(lows), 0), max(np.nanmax(highs), 0))


def plot(all_results, save=None):
    import matplotlib.pyplot as plt
    n_total = len(next(iter(all_results.values())))
    if n_total < START:
        print(f"\nNeed at least {START} hands for the risk plot.")
        return
    colors = _colors(all_results)
    curves = {name: running([r[2] for r in res]) for name, res in all_results.items()}
    fig, (ax_sd, ax_sh, ax_hist) = _setup(all_results, "Risk: standard deviation and Sharpe ratio per hand")
    xs = np.arange(1, n_total + 1)
    keep = xs >= START
    bins = _bins(all_results)
    for name, (mean, sd, sharpe, hw) in curves.items():
        c = colors[name]
        ax_sd.plot(xs[keep], sd[keep], color=c, linewidth=1.4, label=f"{name} ({sd[-1]:.1f})")
        ax_sh.plot(xs[keep], sharpe[keep] * 10, color=c, linewidth=1.4, label=f"{name} ({sharpe[-1] * 10:+.2f})")
        ax_sh.fill_between(xs[keep], (sharpe - hw)[keep] * 10, (sharpe + hw)[keep] * 10, color=c, alpha=0.15,
                           linewidth=0)
        counts, _ = np.histogram([r[2] for r in all_results[name]], bins=bins)
        ax_hist.stairs(counts / n_total, bins, color=c, linewidth=1.4, label=name)
    (sd_lo, sd_hi), (sh_lo, sh_hi) = _limits(curves, n_total)
    ax_sd.set(xlim=(START, n_total), ylim=(sd_lo, sd_hi))
    ax_sh.set(xlim=(START, n_total), ylim=(sh_lo, sh_hi))
    for ax in (ax_sd, ax_sh, ax_hist):
        ax.legend(fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
        print(f"\nSaved risk plot to {save}")
    else:
        plt.show()


def animate(all_results, save=None, frames=120):
    # The same three panels, growing hand by hand (log-spaced frames): σ settling, the Sharpe ratio and its
    # interval narrowing, and the histogram of results filling in.
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    n_total = len(next(iter(all_results.values())))
    if n_total < START:
        print(f"\nNeed at least {START} hands for the risk animation.")
        return None
    colors = _colors(all_results)
    nets = {name: np.array([r[2] for r in res]) for name, res in all_results.items()}
    curves = {name: running(x) for name, x in nets.items()}
    idx = np.unique(np.geomspace(START, n_total, 800).astype(int)) - 1  # points to draw, log-spaced
    ends = np.unique(np.geomspace(START, n_total, frames).astype(int))
    ends = np.concatenate([ends, [n_total] * 25])  # hold the last frame
    bins = _bins(all_results)

    fig, (ax_sd, ax_sh, ax_hist) = _setup(all_results, "Risk as the hands add up")
    (sd_lo, sd_hi), (sh_lo, sh_hi) = _limits(curves, n_total)
    ax_sd.set(xlim=(START, n_total), ylim=(sd_lo, sd_hi))
    ax_sh.set(xlim=(START, n_total), ylim=(sh_lo, sh_hi))
    peak = max(np.histogram(x, bins=bins)[0].max() for x in nets.values()) / START
    floor = 0.5 / n_total
    ax_hist.set(xlim=(bins[0], bins[-1]), ylim=(floor, min(peak * 2, 1.0)))

    lines = {}
    for name in nets:
        c = colors[name]
        sd_line, = ax_sd.plot([], [], color=c, linewidth=1.6, label=name)
        sh_line, = ax_sh.plot([], [], color=c, linewidth=1.6, label=name)
        hist = ax_hist.stairs(np.full(len(bins) - 1, floor), bins, color=c, linewidth=1.4, label=name)
        lines[name] = [sd_line, sh_line, ax_sh.fill_between([], [], [], color=c, alpha=0.15), hist]
    for ax in (ax_sd, ax_sh, ax_hist):
        ax.legend(fontsize=9, loc="upper right")
    readout = ax_sh.text(0.02, 0.04, "", transform=ax_sh.transAxes, fontsize=9, family="monospace", va="bottom",
                         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))

    def draw(frame):
        n = ends[frame]
        shown = idx[idx < n]
        text = [f"after {n:,} hands      σ/hand   Sharpe/100"]
        for name, (mean, sd, sharpe, hw) in curves.items():
            sd_line, sh_line, band, hist = lines[name]
            sd_line.set_data(shown + 1, sd[shown])
            sh_line.set_data(shown + 1, sharpe[shown] * 10)
            band.remove()
            lines[name][2] = ax_sh.fill_between(shown + 1, (sharpe - hw)[shown] * 10, (sharpe + hw)[shown] * 10,
                                                color=colors[name], alpha=0.15, linewidth=0)
            counts, _ = np.histogram(nets[name][:n], bins=bins)
            hist.set_data(np.maximum(counts / n, floor))
            text.append(f"{name:<18}{sd[n - 1]:>7.1f}   {sharpe[n - 1] * 10:>+6.2f} ± {hw[n - 1] * 10:.2f}")
        readout.set_text("\n".join(text))

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=len(ends), interval=60)
    if save:
        anim.save(save, writer="pillow", fps=15, dpi=80)
        print(f"\nSaved risk animation to {save}")
    else:
        plt.show()
    return anim


def path(save): # arena.png -> arena_risk.png
    if not save:
        return None
    root, ext = os.path.splitext(save)
    return f"{root}_risk{ext or '.png'}"


if __name__ == "__main__":  # the numbers come from arena games, so running this runs the arena (same options)
    from bots.arena import main
    main()
