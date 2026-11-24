import random
from dataclasses import dataclass, replace

import numpy as np

AIR = 0.5     # hands weaker than this (share of hands beaten) count as missed
EDGE = 0.03   # how soft the cutoffs are
DRAWS = {None: 0, "Gutshot": 1, "Straight draw": 2, "Flush draw": 3}  # draw codes used everywhere


@dataclass
class Profile: # How a player plays. None = work it out from the bet size (the no-read default)
    name: str = "no read"
    # preflop: share of hands, best first
    play_pct: float = 0.45         # plays when we didn't see the action
    raise_pct: float = 0.20        # opens with a raise
    big_raise_pct: float = 0.06    # big raises / 3-bets
    call_pct: float = 0.40         # calls / limps with hands inside this
    slowplay: float = 0.35         # how often top hands just call
    # flop and turn
    flop_bluff: float = 0.40       # missed hands that bet the flop (c-bet bluffs)
    turn_bluff: float = 0.25       # missed hands that bet the turn
    semi_bluff: float = 0.60       # flush / open-ended straight draws that bet
    gutshot_bet: float = 0.30      # gutshots that bet
    early_medium_bet: float = 0.35 # in-between hands that bet the flop/turn
    draw_call: float = 0.85        # draws that call a bet
    float_call: float = 0.15       # missed hands that call a flop/turn bet anyway
    # river
    value_cut: float | None = None # bets for value above this strength; None: bigger bets need stronger hands
    bluff: float | None = None     # missed hands that bet the river; None: balanced
    medium_bet: float = 0.15       # in-between hands that bet the river
    # facing a bet, and sizing
    call_cut: float | None = None  # calls with hands above this strength; None: defends just enough
    bet_size: float = 0.75         # usual bet, as a fraction of the pot
    # raising a bet
    raise_extra: float = 0.12      # raises need this much more strength than a value bet
    raise_bluff: float = 0.05      # missed hands that raise anyway
    semi_raise: float = 0.15       # draws that raise


NO_READ = Profile()

PRESETS = {
    "nit": Profile("nit", play_pct=0.25, raise_pct=0.12, big_raise_pct=0.04, flop_bluff=0.15, turn_bluff=0.08,
                   semi_bluff=0.35, early_medium_bet=0.15, float_call=0.03, draw_call=0.6,
                   value_cut=0.85, bluff=0.05, medium_bet=0.05, call_cut=0.70,
                   raise_extra=0.15, raise_bluff=0.0, semi_raise=0.03),
    "regular": Profile("regular", value_cut=0.75, bluff=0.30, medium_bet=0.10, call_cut=0.45),
    "calling station": Profile("calling station", play_pct=0.70, raise_pct=0.10, flop_bluff=0.10,
                               turn_bluff=0.05, semi_bluff=0.20, early_medium_bet=0.30, float_call=0.45,
                               draw_call=1.0, value_cut=0.70, bluff=0.05, medium_bet=0.20,
                               call_cut=0.20, bet_size=0.5, raise_extra=0.2, raise_bluff=0.01, semi_raise=0.03),
    "maniac": Profile("maniac", play_pct=0.70, raise_pct=0.45, big_raise_pct=0.20, flop_bluff=0.80,
                      turn_bluff=0.65, semi_bluff=0.95, gutshot_bet=0.70, early_medium_bet=0.60,
                      float_call=0.30, value_cut=0.65, bluff=0.70, medium_bet=0.50, call_cut=0.35, bet_size=1.2,
                      raise_extra=0.05, raise_bluff=0.25, semi_raise=0.50),
}
HELD_OUT = {  # only used for testing bots, never for training
    "tricky": Profile("tricky", flop_bluff=0.55, turn_bluff=0.45, semi_bluff=0.40, early_medium_bet=0.05,
                      value_cut=0.82, bluff=0.45, medium_bet=0.02, call_cut=0.55, bet_size=1.5,
                      raise_extra=0.08, raise_bluff=0.15, semi_raise=0.30),
}
TEST_PRESETS = {**PRESETS, **HELD_OUT}


def random_profile(rng, presets=PRESETS): # A preset with its betting numbers nudged, so no two are identical
    base = rng.choice(list(presets.values()))
    nudge = lambda x, spread, lo, hi: min(hi, max(lo, x + rng.uniform(-spread, spread)))
    return replace(
        base,
        flop_bluff=nudge(base.flop_bluff, 0.10, 0, 1), turn_bluff=nudge(base.turn_bluff, 0.10, 0, 1),
        semi_bluff=nudge(base.semi_bluff, 0.10, 0, 1), value_cut=nudge(base.value_cut, 0.05, 0.55, 0.95),
        bluff=nudge(base.bluff, 0.10, 0, 1), medium_bet=nudge(base.medium_bet, 0.05, 0, 1),
        bet_size=nudge(base.bet_size, 0.15, 0.33, 2.0), call_cut=nudge(base.call_cut, 0.07, 0.05, 0.95),
        raise_bluff=nudge(base.raise_bluff, 0.05, 0, 1), semi_raise=nudge(base.semi_raise, 0.08, 0, 1),
        raise_pct=nudge(base.raise_pct, 0.05, 0.02, 0.9), play_pct=nudge(base.play_pct, 0.08, 0.1, 1.0),
    )


# --- the rules: how likely each hand is to take an action. Shared by advise.py's ranging,
# the bots' ranging, and the simulated opponents, so they all agree.
# strength: share of hands each one beats (array), draw: DRAWS codes (array), size: bet / pot

def soft_keep(x, cut): # 1 below the cut, fading to 0 above it
    return 1 / (1 + np.exp(np.clip((np.asarray(x, dtype=float) - cut) / EDGE, -50, 50)))


def value_cut(p, size, street): # Strength needed to bet for value
    if p.value_cut is not None:
        return p.value_cut if street == "river" else p.value_cut - 0.1
    base, slope = (0.65, 0.20) if street == "river" else (0.60, 0.15)
    return min(0.92, base + slope * size)


def bet_weights(strength, draw, size, street, p, base=None, share=None): # Chance each hand bets this size
    # share: make missed hands exactly this share of the bets (e.g. a learned bluff rate)
    strength, draw = np.asarray(strength, dtype=float), np.asarray(draw)
    drawing = np.where(draw >= 2, p.semi_bluff, np.where(draw == 1, p.gutshot_bet, 0.0))
    value = 1 - soft_keep(strength, value_cut(p, size, street))
    air = soft_keep(strength, AIR) * (draw == 0)
    middle = np.clip(1 - value - air, 0, None) * (draw == 0)
    if street == "river":
        medium, bluff = p.medium_bet, p.bluff
        if bluff is None and share is None:  # balanced: bluffs make up bet / (pot + 2 bet) of the bets
            share = size / (1 + 2 * size)
    else:
        medium, bluff = p.early_medium_bet, (p.flop_bluff if street == "flop" else p.turn_bluff)
    if share is not None:
        base = np.ones_like(strength) if base is None else base
        honest = value + middle * medium + (1 - value) * drawing
        air_mass = (base * air).sum()
        bluff = 0.0 if air_mass == 0 else min(1.0, share / (1 - share) * (base * honest).sum() / air_mass)
    return value + air * bluff + middle * medium + (1 - value) * drawing


def call_weights(strength, draw, size, street, p, base=None): # Chance each hand calls a bet of this size
    strength, draw = np.asarray(strength, dtype=float), np.asarray(draw)
    cut = p.call_cut
    if cut is None:  # defend just enough: the strongest 1 / (1 + size) of hands call
        base = np.ones_like(strength) if base is None else base
        order = np.argsort(-strength)
        share = np.cumsum(base[order]) / max(base.sum(), 1e-12)
        cut = strength[order][min(np.searchsorted(share, 1 / (1 + size)), len(order) - 1)]
    else:
        cut = min(0.98, cut + 0.15 * (size - 0.75))  # bigger bets need stronger hands
    strong = 1 - soft_keep(strength, cut)
    if street == "river":
        return strong
    drawing = np.where(draw > 0, p.draw_call, 0.0)
    missed = soft_keep(strength, AIR) * (draw == 0) * p.float_call
    return strong + (1 - strong) * np.maximum(drawing, missed)


def raise_weights(strength, draw, size, street, p): # Chance each hand raises a bet of this size (not on the river: bluffs only there)
    strength, draw = np.asarray(strength, dtype=float), np.asarray(draw)
    value = 1 - soft_keep(strength, min(0.97, value_cut(p, size, street) + p.raise_extra))
    air = soft_keep(strength, AIR) * (draw == 0)
    drawing = np.where(draw > 0, p.semi_raise, 0.0) if street != "river" else 0.0
    return np.clip(value + (1 - value) * (air * p.raise_bluff + drawing), 0, 1)


def preflop_weights(pct, situation, p): # Chance of each preflop action, by hand percentile (0 = best hand)
    # situation: "open" (small blind, first in), "vs_limp" (big blind after a limp),
    # "vs_raise" (facing a raise), "vs_3bet" (facing a re-raise; can only call or fold)
    pct = np.asarray(pct, dtype=float)
    if situation == "open":
        raises = soft_keep(pct, p.raise_pct)
        calls = (1 - raises) * soft_keep(pct, p.play_pct)
    elif situation == "vs_limp":
        raises, calls = soft_keep(pct, p.raise_pct), 0.0 * pct
    elif situation == "vs_raise":
        raises = soft_keep(pct, p.big_raise_pct)
        calls = (1 - raises) * soft_keep(pct, p.call_pct)
    else:  # vs_3bet
        raises, calls = 0.0 * pct, soft_keep(pct, max(p.big_raise_pct * 2, p.call_pct * 0.5))
    return {"raise": raises, "call": calls, "fold": np.clip(1 - raises - calls, 0, 1)}
