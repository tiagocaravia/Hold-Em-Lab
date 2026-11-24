# Hold'em Lab

A Texas Hold'em lab I built in Python. It estimates how often your hand wins with Monte Carlo simulation, turns that into decisions (play, call or fold) by working out what your opponent likely holds, and trains a machine-learning bot that learns to beat different kinds of players. It also checks its own math against exact answers.

```
 cards ──► EQUITY ──► RANGES ──► DECISIONS ──► BOT
          how often   what they   call / fold,   learns each opponent,
          you win     likely hold pot odds, EV   trained with ML on profit
          (Monte      (from their
           Carlo)      actions)

        + ACCURACY: is the equity right, and how many sims is enough?
```

---

## Start here

| I want to... | Run |
|---|---|
| See how often my hand wins | `python3 hand_analysis.py` |
| Know whether to play, call or fold | `python3 advise.py` |
| Watch bots play simulated opponents | `python3 bots/arena.py` |
| Train the ML bot | `python3 bots/train.py` |
| Check the sim against exact answers | `python3 accuracy/validate.py` |
| See how many sims are enough | `python3 accuracy/convergence.py` |

Every script asks for its inputs if you run it with no options. After each result it asks **Try another?**, and the next round starts with your previous answers in brackets, so you can press enter to keep them and change only what you want, e.g. the same hand with a different board or player count. Ctrl-C quits. You can also pass options (shown in each section) to skip the questions. Run everything from the project folder.

---

## Setup

Needs Python 3.10 or newer.

```bash
pip install matplotlib pandas scikit-learn
```

(numpy and Pillow, used by the convergence animation, come with these.)

---

## 1. Equity: how often you win

`hand_analysis.py` asks for your hole cards (like `AS KH`; use `10H` for the ten of hearts), the number of players including you, the number of simulations (enter for 10,000), and the community cards, if any.

* Win / tie / loss estimates for 2-9 players, from preflop through the river
* Two views of your hand (see below), each compared with what an average hand would get
* An equity-by-street plot that compares preflop, flop, turn and river equity

### Two ways to look at your hand

1. **vs random hands (nobody folds).** Everyone gets two random cards and stays in to showdown. This is the classic equity calculator.
2. **At a real table (weak hands fold).** Everyone is dealt random cards, then anyone holding an unplayable hand folds. Folded cards stay dead (they can't land on the board) and you play whoever is left. If everyone folds, you take the pot.

Each view is compared with **what an average hand gets in that same situation**. With nobody folding, that's 1 / players (11.1% at 9 players). At a real table it's higher, since an average hand faces fewer opponents and sometimes wins when everyone folds. It's measured by running the same sim with a random hand in your seat.

### How "playable" gets decided

The preflop tables are already included in `poker/Data/` as `Standard{n}.csv`, for 3, 5, 7 and 9 players. Each one has all 169 starting hands simulated against random opponents (10,000 sims per hand). `poker/Data/preflop_tables.py` builds them and looks hands up:

```bash
python3 poker/Data/preflop_tables.py                          # asks: build or find
python3 poker/Data/preflop_tables.py --find "J10 Suited" --players 9
python3 poker/Data/preflop_tables.py --build 6                # add a 6-player table
```

The real-table view needs a table for your player count; without one, `hand_analysis.py` shows the random-hands view and tells you to build it.

A hand counts as playable if its equity is at least the table's fair share, which is 1 / number of players. At 9 players that's about 11.1%, and 94 of the 169 hand types make the cut. It's a single cut, with no repeated rounds of tightening.

### The math

Every tool runs on one Monte Carlo engine (`poker/equity.py`). Equity is your average share of the pot, with tied pots split exactly among whoever tied, and it's shown with a 95% confidence interval. The two views differ only in the opponent model the engine is given: random hands, or random hands where the weak ones fold.

A few honest caveats: in `hand_analysis.py`, opponents' hands don't depend on how they bet, and folding only happens preflop based on the starting hand. "Playable" is judged purely by equity against random hands. Betting is handled in `advise.py` and the bots below.

### Example

* **Hole cards:** 7H 2S
* **Players:** 9
* **Simulations:** 3000
* **Community cards:** AH QD 2H 7D 7C

That makes sevens full of twos.

```
--- VS RANDOM HANDS (nobody folds) ---
Win:  88.4%
Tie:  1.4%
Loss: 10.1%
Equity: 89.1% ± 1.1  (an average hand here: 11.1%)

--- AT A REAL TABLE (weak hands fold, their cards stay dead) ---
Win:  92.3%  (0.6% because everyone folded)
Tie:  0.0%
Loss: 7.7%
Average opponents left in: 3.55
Equity: 92.3% ± 1.0  (an average hand here: 20.8%)
```

At a real table the loss rate drops: more than half of the eight opponents fold, so only about three or four are left to beat you. The hands that beat this one are AA, QQ, or the 7 of spades with an ace or a queen. Ties drop to 0.0% because the only hands that tie are the 7 of spades with a 2, and 7-2 isn't a hand people play.


The plot shows the same hand street by street, in both views. Each solid line has a matching baseline in the same color: the dashed line is an average hand with nobody folding (11.1%), and the dotted line is an average hand at a real table (about 18–22% here). Here 7-2 sits below the real-table average until the turn, then climbs well above it. The plot and the text come from the same run, so their numbers match.

<img width="899" height="552" alt="Screenshot 2026-11-21 at 2 30 57 AM" src="https://github.com/user-attachments/assets/56642ed5-2fb2-4f2a-a493-203611d33f1b" />


---

## 2. Decisions: play, call or fold

`advise.py` turns equity into a decision:

```bash
python3 advise.py                                                   # asks you questions
python3 advise.py --hole "AS 10S" --players 9                       # preflop: playable?
python3 advise.py --hole "KH KD" --board "KS 9H 4H 7C 2H" --pot 10 --bet pot --stack 40
python3 advise.py --hole "QH JH" --board "10H 4C 2H" --opponents 2 --pot 60 --bet 20 --bb 2
python3 advise.py --hole "9S 9D" --board "7H 4C 2D 3S JC" --pot 10 --bet pot --action "raise big"
python3 advise.py --hole "9S 9D" --board "7H 4C 2D 3S JC" --pot 10 --bet pot --action raise --flop "bet half" --turn check
```

* **Preflop:** PLAY / FOLD / CLOSE, comparing your equity with the fair share (1 / players), the same cutoff the preflop tables use.
* **Facing a bet:** CALL / FOLD / CLOSE. In CLOSE spots it also says how often to call (closer to break-even means closer to 50%), and `--roll` picks one for you at that rate.
* **Pot odds:** You need equity of at least call ÷ (pot + call) to break even, and EV(call) = equity × (pot + call) − call, with folding worth 0. The bet can be an amount or `third`, `half`, `pot`, `75%`, `2x`.
* **Units:** amounts are in big blinds, and EV is reported in big blinds. If you'd rather type chips, pass `--bb` with the big blind's size (`--bb 2` makes `--pot 60` mean 30bb). `2.5bb` always means big blinds.
* **Stack (optional):** `--stack` is your stack before calling. It shows what's left behind and the stack-to-pot ratio. If calling puts you all in, it uses only the part of the bet you can match, and notes that pot odds are exact since no more betting follows.
* **CLOSE** means the equity is within 3 points of break-even, or within its confidence interval. Those are the spots where the right answer depends on who you're playing.

Equity is shown against random hands and against played hands (the hands people actually play, since anyone still in the hand chose to play), with a 95% interval, and the verdict uses played hands when there's a table for that many players. Tied pots are split exactly.

**Big bets:** players who bet half the pot or more usually hold strong hands or bluffs, not the average playable hand. So for those the CLOSE band is wider and the output warns that the equity is likely too optimistic, unless ranging (below) has already accounted for the bet.

### Ranging: what they could have

`advise.py` narrows the opponent's possible hands based on what they did, then computes your equity against that range:

* **Preflop action** (`--action`, or asked): `raise` keeps roughly their top 20% of hands, `raise big` / `3bet` the top 6%, and `call` / `limp` hands inside the top 40% (with top hands less likely, since they'd usually raise). If you don't give one, it assumes they played a hand, roughly the top 45%.
* **Flop and turn actions** (`--flop`, `--turn`, or asked): `check`, `call`, `raise`, or `bet` with a size (`bet half`, `bet pot`, `bet 75%`, `bet 2x`). Draws (flush draws, open-ended straight draws, gutshots) are tracked separately, since players often bet them as semi-bluffs:

  | They... | Strong hands | Draws | In-between | Missed |
  |---|---|---|---|---|
  | bet | kept (bigger bets need stronger) | often | sometimes | sometimes (c-bet bluffs) |
  | check | rarely (they'd usually bet) | sometimes | kept | mostly kept |
  | call | sometimes (might have raised) | kept | kept | rarely |
  | raise | kept, stronger | sometimes | rarely | rarely |

  A bet you're facing on the flop or turn counts the same way.
* **River bet:** the bet you're facing narrows it further. Hands that beat most of the possible hands on this board bet for value (bigger bets need stronger hands), in-between hands rarely bet, and missed hands are kept as bluffs. With no read, bluffs are assumed to make up a balanced share of their bets: bet ÷ (pot + 2 × bet), which is a third for a pot-sized bet.
* **Output:** their most likely hands, what they're holding on the board (pairs, flushes, nothing), what they're drawing to before the river, how much of their range beats you, and your equity vs the range. On the river against one opponent that's exact, so no ± is shown.

For example, a big preflop raiser bets pot on a low river. Against random hands 9-9 looks like an easy call (75%), but their range is mostly AA–JJ plus AK/AQ that missed, so your equity drops to 31%, just under the 33% you need. The verdict is CLOSE, and it comes down to whether this player bluffs a balanced amount.

The assumptions are the no-read `Profile` in `poker/profiles.py`: how wide players raise, how often they c-bet, semi-bluff, trap and float. The same profiles drive the opponents in the bot arena.

**Why river bluff-catchers come out CLOSE with no read:** a balanced bluffing amount is, by definition, the amount that makes a bluff-catching hand break even. So with no read, hands that only beat bluffs land near the pot odds, and the earlier streets mostly change which value hands they have, not how many bluffs. That's the honest no-read answer. What decides those spots is how this particular player bluffs, which is what the bots learn.

---

## 3. Bots: learning to beat opponents

`bots/` plays simulated opponents, learns how each one plays, and trains a machine-learning bot on the results.

### The arena

```bash
python3 bots/arena.py                      # asks you questions
python3 bots/arena.py --hands 100000 --session 1000 --save arena.png
```

**The game:** heads-up from the flop to the river, with 6bb in the pot and 100bb stacks. On each street the first player checks or bets ½, ¾ or 1¼ pot; the other calls or folds, or bets if checked to. No raises yet. Players take turns acting first. Each opponent plays one session (1,000 hands by default) and is a randomly nudged version of a preset profile: nit, regular, calling station or maniac. The bot meets every new opponent knowing nothing about them.

**One set of rules:** opponents are `Profile`s from `poker/profiles.py`, the same description `advise.py` uses. Their choices come from the same betting and calling rules the bots and `advise.py` use to range them, so what a bot assumes about a player and how that player actually behaves line up.

**Real cards, fast:** the first run scores every possible hand on 10,800 real boards (300 flops × 6 turns × 6 rivers), in parallel, in about 90 seconds, cached in `bots/cache/`. On the flop and turn a hand's strength is its equity over the runouts still to come, and draws are flagged. The bots range their opponent on every street, blockers included. 100,000 hands take about 75 seconds per bot.

**Bot versions** (the hand-built bot in `bots/ml_agent.py`), all playing exactly the same deals and opponent choices:

* **no read:** the no-read assumptions, and never learns
* **learner:** learns from showdowns what share of this opponent's bets are bluffs, and how often they call its bets, starting from the no-read assumptions held as if it had seen about 8 hands
* **learner + mixing:** randomizes in CLOSE spots
* **oracle:** knows the opponent's exact profile, but still plays the hand-built rules

**Results** (100,000 hands, new opponent every 1,000):

| bb/100 | no read | learner | learner + mixing | oracle |
|---|---|---|---|---|
| all | +555 | +620 | +618 | +806 |
| vs nits | +57 | +63 | +63 | +226 |
| vs calling stations | +996 | +1,300 | +1,300 | +1,542 |

* The win rates are huge because these opponents are simple rules that never raise: over three streets with 100bb stacks they pay off value bets and bluff into calls. A bot that never bets and calls everything loses about 560 bb/100 against them, so the accounting is sound. The meaningful number is each bot's **edge over "no read" on the same hands**, which the arena reports with a 95% interval.
* The learner is +66 ± 12 bb/100 ahead of "no read" and clearly ahead from about **hand 25** against a new opponent, mostly against calling stations.
* The oracle is another +190 bb/100 ahead, especially against nits. That gap is what the ML bot gets to work on.

The plot shows cumulative profit for each bot, and each bot's edge over "no read" by how many hands into a new opponent it is.

### Training the ML bot

```bash
python3 bots/train.py                      # asks you questions
python3 bots/train.py --rounds 5 --hands 20000 --save training.png
python3 bots/train.py --rounds 5 --resume  # keep training the saved bot
```

**Saved after every round:** the bot, meaning its models, every hand it has learned from, and its learning curve, is saved to `bots/cache/ml_state.joblib`. So training can stop and pick up later with `--resume`, and an interrupted run keeps its progress. The models alone also go to `bots/cache/ml_model.joblib` for other tools to load. `bots/cache/` isn't committed to git.

**Fast:** hands are played on every CPU core at once. Each core plays its own block of opponents, with seeds fixed so every bot version still faces exactly the same hands. The models stop adding trees once more trees stop helping on held-back data, which keeps predictions fast and avoids fitting noise.

The ML bot (`MLAgent` in `bots/ml_agent.py`) plays every street with two gradient-boosted models (scikit-learn's `HistGradientBoostingRegressor`): one for calling a bet, one for checking or betting ½, ¾ or 1¼ pot. Each option's value is the hand-built value plus a **learned correction**, so the models learn where the guidance is wrong rather than relearning poker. Their inputs include the street, hand strength, equity vs the ranged opponent, the bet size, the pot and stack sizes, the hand-built value (and for bets, its fold chance and equity when called), and reads on this opponent, including how often they call each bet size.

**The reward** for each decision is what the rest of the hand won from that point: the hand's final profit plus what was already in the pot. So a flop bet gets credit or blame for how the whole hand played out, which is how it can learn things the hand-built value can't see, like a semi-bluff that wins on the river.

Training works in rounds:

1. **Play** hands against random opponents, logging every decision and its reward. Early rounds explore a lot (30% random actions), later ones less (5%).
2. **Retrain** both models on everything collected so far.
3. **Test** a frozen copy on held-out hands against held-out opponents, including a "tricky" type it never trains against.

Results (15 rounds × 50,000 hands = 750,000 training hands, about 15 minutes on 15 cores):

| Training hands | ML bot (bb/100) | Edge over the hand-built learner |
|---|---|---|
| 50,000 | +480 | −0.2 ± 25.7 |
| 200,000 | +522 | +42.3 ± 26.5 |
| 400,000 | +549 | +69.0 ± 27.2 |
| 600,000 | +567 | **+86.4 ± 26.8** |
| 750,000 | +559 | +79.3 ± 26.8 |

It matches the learner after 50,000 hands, pulls clearly ahead by 200,000, and levels off at about +80 bb/100 from around 600,000 hands. More hands alone won't add much now; the next gains have to come from better inputs or a richer game. It beats the learner against every opponent type, including "tricky", the type it never trained on (+289 vs +249). The oracle is still about 110 bb/100 ahead, mostly against nits and "tricky".

To watch it learn, add `--animate training.gif` (10 rounds of 10,000 hands works well: `--rounds 10 --hands 10000`). Each frame is one round and shows:

* its win rate and its edge over the hand-built learner so far;
* what it does when it acts first, at each hand strength (check, or bet ½, ¾ or 1¼ pot);
* how often it calls a bet, at each hand strength, next to the hand-built guidance it started from;
* which inputs each model relies on.

Every round replays the same fixed set of spots, so the changes come from what it learned, not from new cards.

### What the ML bot relies on

```bash
python3 bots/explain.py --save explain.png
```

The script plays 20,000 hands with the trained bot, then scrambles one input at a time and measures how much each model's corrections change. The more they change, the more the model relies on that input.

What it shows (100,000 training hands):

* **The betting model** relies most on its read of how often this opponent calls each bet size, then the hand-built value and the **street**: it learned to bet differently on the flop than on the river.
* **The calling model** relies most on hand strength, the pot size and its reads on the opponent.
* Draw flags barely matter, since flop and turn equity already include a draw's chances.

---

## 4. Accuracy: is the math right?

### Validation

`accuracy/validate.py` checks the Monte Carlo sim against the true equity, heads-up against one random hand:

```bash
python3 accuracy/validate.py                                   # asks you questions
python3 accuracy/validate.py --hole "AH KH" --board "QH JD 2C"
python3 accuracy/validate.py --hole "KS KD" --board ""         # preflop
python3 accuracy/validate.py --hole "7H 2S" --board "AH QD 2H 7D" --save validation.png
```

* **Flop, turn, river:** the true equity comes from enumerating every opponent hand and every runout (about 1.07 million matchups on the flop, about 35 seconds).
* **Preflop:** enumerating would mean about 2.1 billion matchups, so it uses published exact equities instead (AA, KK, QQ, JJ, TT, AKs, AQs, AKo). Only hands where both of these sources agree to two decimals are included:
  * [pailiku.com: Texas Hold'em matchup odds, exact enumerations](https://pailiku.com/matchups)
  * [pokercoaching.com: Best Texas Hold'em starting hands](https://pokercoaching.com/blog/best-texas-holdem-starting-hands/)

It then runs the sim several times and tracks a 95% confidence interval (estimate ± 1.96·σ/√n) as the number of simulations grows. It reports how many runs end with an interval that contains the true value (should be about 95%) and how many simulations you need for a given precision. Against a random hand, 10,000 sims gives roughly ±1 point. The plot shows the interval closing in on the true equity, and the interval width falling as 1/√n (4× the sims gives half the width).

The flop, turn and river checks use the same hand evaluator for both the enumeration and the sim, so they test the sampling and the interval math. The preflop check compares against outside numbers, so it also tests the evaluator.

### Convergence

`accuracy/convergence.py` shows how the estimate settles as the number of simulations grows. It works for any hand, player count and board:

```bash
python3 accuracy/convergence.py                                     # asks you questions
python3 accuracy/convergence.py --hole "AS KH" --players 6
python3 accuracy/convergence.py --hole "7H 2S" --players 9 --board "AH QD 2H" --save convergence.gif
```

It prints the running equity and its 95% confidence interval at checkpoints (100, 200, 500, 1,000, ...), how far the estimate moved since the last checkpoint, and how many sims you'd need for ±2, ±1, ±0.5 or ±0.1 points. The 95% confidence interval is estimate ± 1.96·σ/√n, so 4× the sims gives half the width.

It also plays an animation: the running estimate draws in with its confidence band shrinking around it, next to a bell curve of where the true equity probably is, which narrows into a spike. Use `--save file.gif` to save it, or `--no-animate` to skip it.

<img width="1299" height="559" alt="Screenshot 2026-11-19 at 4 46 56 PM" src="https://github.com/user-attachments/assets/ef600f3d-c15f-4d8f-86d3-4f4567a49cd7" />

---

## Project layout

```
holdem-lab/
├── README.md
├── hand_analysis.py           # 1. equity: how often your hand wins
├── advise.py                  # 2. decisions: play / call / fold, pot odds, ranging
├── bots/                      # 3. bots
│   ├── pool.py                # scores every hand on 10,800 real boards, flop to river, cached
│   ├── ml_agent.py            # the bot: hand-built versions + the ML bot
│   ├── arena.py               # plays bot versions against random opponents
│   ├── train.py               # trains the ML bot, tests on held-out opponents
│   └── explain.py             # which inputs the ML bot relies on, training animation
├── accuracy/                  # 4. accuracy
│   ├── validate.py            # sim vs exact / published equity
│   └── convergence.py         # how many sims is enough, with animation
└── poker/                     # the engine everything above uses
    ├── card.py, deck.py       # cards and deck
    ├── evaluator.py           # hand ranking
    ├── equity.py              # the Monte Carlo engine: equity + 95% CI vs an opponent model
    ├── tables.py              # preflop tables: load, playable hands, look up
    ├── profiles.py            # how a player plays (Profile, presets) + the shared betting/calling rules
    ├── ranges.py              # narrowing an opponent's range from their actions
    ├── decisions.py           # pot odds, EV and verdicts (tunable Settings)
    ├── visualizer.py          # equity-by-street plot
    ├── prompts.py             # the questions the scripts ask, with defaults
    └── Data/                  # preflop tables (Standard{n}.csv)
        └── preflop_tables.py  # builds a table or looks up a hand
```
