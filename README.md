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
| Play the trained bot, heads-up or at a full table | `python3 play.py` |
| Watch bots play simulated opponents | `python3 bots/arena.py` |
| Train the ML bot | `python3 bots/train.py` |
| Check the sim against exact answers | `python3 accuracy/validate.py` |
| See how many sims are enough | `python3 accuracy/convergence.py` |

Every script asks for its inputs if you run it with no options. After each result it asks **Try another?**, and the next round starts with your previous answers in brackets, so you can press enter to keep them and change only what you want, e.g. the same hand with a different board or player count. Ctrl-C quits. You can also pass options (shown in each section) to skip the questions. Run everything from the project folder.

---

## Setup

Needs Python 3.10 or newer.

```bash
pip install matplotlib pandas scikit-learn phevaluator
```

(numpy and Pillow, used by the convergence animation, come with these.)

`phevaluator` is a compiled hand evaluator, about 50× faster than the Python one in `poker/evaluator.py`. Everything uses it when it's installed and falls back to the Python evaluator when it isn't, with identical results: the two agreed on who wins and on the hand category in 300,000 random matchups. The Python evaluator stays as the readable reference.

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

Every tool runs on one Monte Carlo engine (`poker/equity.py`). Equity is your average share of the pot, with tied pots split exactly among whoever tied, and it's shown with a 95% confidence interval. The two views differ only in the opponent model the engine is given: random hands, or random hands where the weak ones fold. A third model, `RangeOpponents`, deals each opponent from a weighted range, one per seat if you like ("a tight player and two loose ones"). `advise.py` uses it for the ranged opponent, and it's the reference the bots' multiway equity was checked against (see [Limitations](#limitations)).

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

## Play against the bots

```bash
python3 play.py                                    # asks how many players and who
python3 play.py --opponent "ml bot"                # heads-up vs the trained ML bot
python3 play.py --players 6 --opponent "ml bot"    # you and five ML bots
python3 play.py --players 4 --opponent "ml bot,nit,maniac"   # mixed seats, in that order
```

In the terminal, heads-up or at a table of up to 9: blinds 0.5/1bb, 100bb stacks every hand, and the button moves every hand. You can play:

* **the ML bot:** the trained one saved in `bots/cache/`, trained with self-play and bet-size reads at tables of 2 to 9. It plays what it learned and doesn't retrain mid-session, but it builds reads on you as you play.
* **the old ML bot:** the heads-up version trained only against rule-based opponents, before self-play
* **the hand-built bot:** what the ML bot started from, so in effect the untrained version
* **any preset opponent:** nit, regular, calling station, maniac or tricky

Each hand starts by saying who posts which blind and where you're sitting. Before each decision you get a small table: the street, the board and the pot, then a row per player with their position (button, blinds, UTG, HJ, CO...), cards (theirs hidden), stack, and what each has put in this street. Players who folded are greyed out. Every bot keeps its own read on every other player, you included. Your options are listed one per line with what they mean, like `c  call 0.5bb  (complete the small blind)` or `1  raise to 2.5bb  (puts in 2.0bb)`. Red suits and wins and losses are coloured; `--no-color` turns that off. At the end, or when you press Ctrl-C, it shows your result in bb/100 with a 95% margin of error, and whether that's enough hands to say who's better. Usually that takes a few hundred.

The ML bot learned against rule-based opponents and itself, so a person who spots its habits (it calls a lot preflop) may be able to exploit it. That's part of the point of playing it.

---

## 3. Bots: learning to beat opponents

`bots/` plays simulated opponents, learns how each one plays, and trains a machine-learning bot on the results.

### The arena

```bash
python3 bots/arena.py                      # asks you questions
python3 bots/arena.py --hands 100000 --session 1000 --save arena.png
python3 bots/arena.py --hands 20000 --players 6        # 6-player tables
python3 bots/arena.py --hands 20000 --players 2-9      # a random table size each session
```

**The game:** no-limit hold'em, preflop to river, at tables of 2 to 9 (heads-up by default). Blinds are 0.5 and 1bb, stacks 100bb, and the button moves every hand. Preflop the player after the big blind acts first, and after the flop the first player after the button. Heads-up, the button is the small blind, so it acts first preflop and last after the flop. Side pots are handled, though with equal stacks they only matter once someone is all in.

Preflop a player can fold, limp, raise (to 2.5bb) or 3-bet (to 3×), then call or fold a 3-bet. After the flop, on each street a player checks or bets ½, ¾ or 1¼ pot, and facing a bet can fold, call or raise. That's two aggressive actions per street (bet and raise, or raise and 3-bet), with no 4-bets.

Each table plays one session (1,000 hands by default), and every other seat is a randomly nudged version of a preset profile: nit, regular, calling station or maniac. The bot meets every new opponent knowing nothing about them, and keeps a separate read and a separate range for each player at the table.

**One set of rules:** opponents are `Profile`s from `poker/profiles.py`, the same description `advise.py` uses. Their choices (preflop actions, bets, calls and raises) come from the same rules the bots and `advise.py` use to range them, so what a bot assumes about a player and how that player actually behaves line up.

**Real cards, fast:** the first run scores every possible hand on 10,800 real boards (300 flops × 6 turns × 6 rivers) and builds a preflop table of every hand's equity against every other hand from 4,000 independent random boards. That takes about 40 seconds in parallel with `phevaluator`, and it's cached in `bots/cache/`. The preflop table matches published equities to within about half a point (AA vs a random hand: 85.4%, published 85.2%). On the flop and turn a hand's strength is its equity over the runouts still to come, and draws are flagged. The bots range every opponent on every street, preflop included, blockers included. Heads-up, 100,000 hands take about 45 seconds per bot. A 6-player table is about 20× slower per hand, since every decision weighs several ranges at once.

**Bot versions,** all playing exactly the same deals and opponent choices:

* **no read:** the hand-built bot (in `bots/ml_agent.py`) with no-read assumptions. It never learns, and it's the yardstick everything is measured against.
* **oracle:** the hand-built bot allowed to **peek at each opponent's true profile**. It knows exactly how this player bets and calls, though not their cards. It shows how well the hand-built rules could do with perfect reads. It isn't a ceiling: a bot with better rules can beat it.
* **ML bot:** the trained bot, when one is saved.

The hand-built logic is kept even though the ML bot plays better. It's how the ML bot sees the game: every input it uses (equity vs a narrowed range, fold chances, reads) comes from it, and the ML bot learns *corrections* to its value of each option. It's also the fallback when no trained model is saved. Earlier versions with learned reads ("learner", "learner + mixing") were dropped from the comparison; the reads now live on as inputs to the ML bot.

**Results** (heads-up, 20,000 hands, new opponent every 1,000):

| bb/100 | no read | oracle | ML bot |
|---|---|---|---|
| all | +210 | +290 | **+405** |
| edge over no read | n/a | +79 ± 24 | **+195 ± 34** |

* Even "no read" profits, because these opponents are simple rules: they fold too much preflop and call too much after. A bot that checks when it can and folds to any bet loses about 38 bb/100 (its blinds), so the accounting is sound. The meaningful number is each bot's **edge over "no read" on the same hands**, which the arena reports with a 95% interval.
* The ML bot is far ahead of both hand-built versions, including the oracle.

The plot shows cumulative profit for each bot, and each bot's edge over "no read" by how many hands into a new opponent it is.

### Training the ML bot

```bash
python3 bots/train.py                      # asks you questions
python3 bots/train.py --rounds 5 --hands 20000 --save training.png
python3 bots/train.py --rounds 5 --resume  # keep training the saved bot
python3 bots/train.py --self-play 0        # train only against the rule-based opponents
python3 bots/train.py --players 2-9 --resume  # tables of a random size from 2 to 9 each session
```

**How it decides:** the ML bot (`MLAgent` in `bots/ml_agent.py`) makes every decision, from preflop to the river, with one gradient-boosted model (scikit-learn's `HistGradientBoostingRegressor`). For each legal option (check, call, or a bet or raise of each size) it takes the hand-built value of that option and adds a **learned correction**, then picks the best; folding is worth 0. So the model learns where the guidance is wrong rather than relearning poker.

Its 22 inputs include:

* the street, hand strength and draws
* equity against everyone still in, each with their own narrowed range
* the bet to call, the option's size, and the pot and stack sizes
* the hand-built value of the option (for bets and raises, also the chance everyone folds and the equity when called)
* reads on the players in the hand, such as how often they call each bet size and how often their bets of each size were bluffs
* the table: how many opponents are in the hand, how many are still to act, and whether it's in position

**The reward** for each decision is what the rest of the hand won from that point: the hand's final profit plus what was already in the pot. So a flop bet gets credit or blame for how the whole hand played out, which is how it can learn things the hand-built value can't see, like a semi-bluff that wins on the river.

**Training works in rounds:**

1. **Play** hands against random opponents, logging every decision and its reward. Early rounds explore a lot (30% random actions), later ones less (5%).
2. **Retrain** the model on everything collected so far.
3. **Test** a frozen copy on held-out hands against held-out opponents, including a "tricky" type it never trains against.

**Self-play (on by default):** half of the opponent seats are frozen copies of the bot itself, drawn from its last 5 versions, which are saved with it as a "league". The rest are rule-based profiles, so it stays grounded. Being hard to read doesn't matter against rule-based opponents, because they never read you. Against its own copies it does: they keep reads on it as hands go by, and the next version learns from whatever they punished.

**Saved after every round:** the bot is saved to `bots/cache/ml_state.joblib`, meaning its model, every decision it has learned from, and its learning curve. So training can stop and pick up later with `--resume`, and an interrupted run keeps its progress. The model alone also goes to `bots/cache/ml_model.joblib` for other tools to load. `bots/cache/` isn't committed to git. A saved bot trained on a different set of inputs can't be resumed, so training starts fresh.

**Fast:** hands are played on every CPU core at once. Each core plays its own block of tables, with seeds fixed so every bot version still faces exactly the same hands. With a range of table sizes, the sizes are shared evenly across cores. The model stops adding trees once more trees stop helping on held-back data, which keeps predictions fast and avoids fitting noise.

#### Results: heads-up

The first bot, trained only against the rule-based profiles (15 rounds × 50,000 hands = 750,000 hands, about 15 minutes on 15 cores):

| Training hands | ML bot (bb/100) | Edge over the hand-built bot |
|---|---|---|
| 50,000 | +99 | −81 ± 27 |
| 150,000 | +178 | −2 ± 30 |
| 250,000 | +289 | +108 ± 34 |
| 500,000 | +343 | +163 ± 35 |
| 750,000 | **+366** | **+185 ± 35** |

| Test opponents (bb/100) | ML bot | no read | oracle |
|---|---|---|---|
| all | **+366** | +173 | +231 |
| maniac | +693 | +216 | +284 |
| calling station | +555 | +353 | +487 |
| tricky (never trained on) | +208 | +122 | +176 |
| nit | +41 | +39 | +95 |

(The edge in the first table is over "learner", an earlier hand-built bot with learned reads that has since been dropped. It scored within a few bb/100 of "no read".)

It starts well behind the hand-built bots, passes them after about 200,000 hands, and is still creeping up at 750,000. It beats the oracle overall because the oracle still plays hand-built rules, and with preflop play and raises there's far more for a learned strategy to improve on. It plays far more hands preflop than the guidance would (mostly calling, raising only its best hands), and it punishes the maniac's raises. Nits are the one type where it barely beats "no read".

#### Results: tables of 2 to 6

The first multiway bot: 1,000,000 hands (20 rounds × 50,000) at a random table size from 2 to 6 each session, half the seats self-play, with the three table inputs added.

| Training hands | ML bot (bb/100) | Edge over "no read" |
|---|---|---|
| 50,000 | +194 | −57 ± 35 |
| 200,000 | +477 | +226 ± 51 |
| 450,000 | +724 | +473 ± 67 |
| 1,000,000 | **+750** | **+499 ± 69** |

* **It levelled off from about 450,000 hands.** Every later round sits within the margin of error.
* **Head to head against the heads-up bot** (positions rotating, both keeping reads):

  | Table | Multiway bot's result |
  |---|---|
  | heads-up, 40,000 hands | +5 ± 9 bb/100, no real difference |
  | 6 players, against five copies of the heads-up bot, 30,000 hands | **+95 ± 23 bb/100** |

  So learning bigger tables cost it nothing heads-up, and at 6 players it's clearly better.
* **These win rates aren't comparable with the heads-up ones:** with up to five weak opponents at the table there's more to win each hand. Compare bots on the same table sizes, or by their edge over "no read".

#### Results: tables of 2 to 9

The current bot: the 2–6 bot, trained 500,000 more hands (10 rounds × 50,000) at a random table size from 2 to 9, about 95 minutes on 15 cores.

* **On the held-out test it stayed flat** at about +440 to +460 bb/100 over "no read" in every round. This test includes 7 to 9-player tables, so it isn't comparable with the 2–6 numbers above.
* **Head to head it beat the 2–6 bot at every size tested:**

  | Table | 2–9 bot vs 2–6 bot |
  |---|---|
  | heads-up, 40,000 hands | +10 ± 5 bb/100 |
  | 6 players, against five copies of the 2–6 bot, 30,000 hands | +61 ± 27 bb/100 |
  | 9 players, against eight copies, 30,000 hands | +41 ± 26 bb/100 |

* **Final test by table size** (bb/100, 20,000 hands spread over every size, so the margins are wide):

  | Table | ML bot | no read | oracle |
  |---|---|---|---|
  | heads-up vs a regular | +380 ± 164 | +285 ± 129 | +277 ± 136 |
  | heads-up vs tricky (never trained on) | +197 ± 103 | +120 ± 90 | +178 ± 87 |
  | 3 players | +702 ± 176 | +333 ± 94 | +404 ± 123 |
  | 4 players | +756 ± 171 | +347 ± 101 | +458 ± 117 |
  | 5 players | +741 ± 199 | +238 ± 101 | +366 ± 126 |
  | 6 players | +960 ± 240 | +282 ± 118 | +396 ± 140 |
  | 7 players | +951 ± 337 | +380 ± 176 | +478 ± 198 |
  | 8 players | +825 ± 362 | +331 ± 187 | +388 ± 211 |
  | 9 players | +678 ± 469 | +75 ± 229 | +136 ± 238 |

The 2–6 bot is kept in `bots/cache/ml_model_6max.joblib`.

To watch it learn, add `--animate training.gif`. Each frame is one round and shows:

* its win rate and its edge over "no read" so far;
* what it does when checked to after the flop (check, or bet ½, ¾ or 1¼ pot), when facing a bet (fold, call or raise), and preflop, each by hand strength;
* which inputs the model relies on.

Every round replays the same fixed set of spots, so the changes come from what it learned, not from new cards.

### Plugging the bot into a game

`bots/interface.py` wraps the trained bot behind one call, so a game engine or a competition only needs a thin adapter:

```python
from bots.interface import Bot, GameState

bot = Bot()  # loads the trained ML bot from bots/cache/, or the hand-built bot if none is saved
move, chips = bot.act(GameState(
    hole=["AS", "KH"], board=["QH", "JD", "2C"], pot=12, to_call=4, stack=94, opponent="player 2",
    actions=[("preflop", "us", "raise", 2.0), ("preflop", "them", "call", 1.5), ("flop", "them", "bet", 4.0)]))
# -> e.g. ("call", 4.0); moves are "fold", "check", "call", "bet" or "raise", with the chips to put in now
bot.end_hand(shown={"player 2": ["9C", "9D"]})  # it learns from the hands shown down

# at a bigger table: name everyone, in order from the small blind, and use their names in the actions
state = GameState(hole=["AS", "KH"], board=[], pot=4.0, to_call=2.5, stack=99.0,
                  players=["ana", "ben", "us", "cy"], actions=[("preflop", "us", "raise", 2.5), ...])
bot.end_hand(final_state, shown={"cy": ["QC", "QS"]})  # the final state catches it up on the last actions
```

* **`GameState`:** your cards and the board as text, the pot (including any bet you face), what you must add to call, your stack, and **every action this hand so far** as (street, who, move, chips put in), blinds not included. Heads-up, who is "us" or "them"; at a bigger table it's "us" or a player's name, and `players` lists everyone from the small blind round. From that it works out the blinds, positions, who's still in and what each action meant, just as the arena would.
* **Every street, any board:** preflop uses the cached preflop table; after the flop it scores each new board as it appears, sampling the runouts on the flop. About 0.02 seconds per decision.
* **Reads are kept per player name,** so it keeps learning about each player across hands and tables.

**Tested against the arena:** played through the interface, the bot makes exactly the same preflop decisions as when it's played directly. The first differences show up on the flop, where the interface samples its own runouts instead of using the pool's.

`python3 bots/interface.py --opponent regular` (add `--players 6` for a table) plays arena hands with every decision going through `Bot.act()`. Its `ThroughInterface` class, which turns the engine's events into `GameState`s, is the template for a competition adapter.

### Self-play, and reading bet sizes (heads-up)

Three heads-up bots, compared on the same tests:

* **profiles only:** trained 750,000 hands against the rule-based profiles
* **self-play v1:** the same, but half its sessions against copies of itself
* **self-play + size reads:** self-play over 1,000,000 hands, plus reads of each opponent **by bet size**. For small, medium and big bets separately, it tracks how often that opponent's bets of that size turned out to be bluffs at showdown. Those reads feed both the hand-built ranging and the model.

| | profiles only | self-play v1 | self-play + size reads |
|---|---|---|---|
| Head to head vs "profiles only" | n/a | +163 ± 16 | **+163 ± 17** |
| Head to head vs "self-play v1" | n/a | n/a | **+28 ± 12** |
| Held-out rule-based opponents | +356 | +343 | +338 |
| How often it bets 1¼ pot when checked to | 20% | 23% | 37% |
| Average strength of its 1¼-pot bets | 0.76 | 0.73 | 0.68 |
| Share of its 1¼-pot bets that are bluffs | 1% | 1% | 6% |

(Head-to-head numbers are bb/100 over 40,000 hands with positions alternating; the others are bb/100 on 20,000 held-out hands.)

* **Each step beat the one before it** head to head, so self-play with size reads became the default, and the multiway bot was trained on top of it.
* **The sizing tell is smaller but not gone.** Its big bets are now more common and less one-sided: thinner value plus a few bluffs. 6% bluffs is still far from the roughly 36% a balanced player would have at that size, so folding to its big bets still works, just less well than before.
* **Against the simple rule-based bots all three are about the same,** since those never read bet sizes.
* Training levelled off at about +165 bb/100 over "no read" from around 650,000 hands.

`python3 bots/explain.py` prints the bet-size table for the saved bot. In `play.py`, "ml bot" is the current multiway bot, and "old ml bot" is the profiles-only one.

### What the ML bot relies on

```bash
python3 bots/explain.py --save explain.png
```

The script plays 20,000 heads-up hands with the trained bot, then scrambles one input at a time and measures how much the model's corrections change. The more they change, the more the model relies on that input.

What it showed for an early heads-up bot (100,000 training hands), when betting and calling were still two separate models:

* **Betting** relied most on its read of how often this opponent calls each bet size, then the hand-built value and the **street**: it learned to bet differently on the flop than on the river.
* **Calling** relied most on hand strength, the pot size and its reads on the opponent.
* Draw flags barely mattered, since flop and turn equity already include a draw's chances.

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

## Limitations

What this lab doesn't do, or does only roughly. Most of it comes down to one thing: the bots are measured against simple simulated players, not people.

### The game is simplified

* **Fixed bet sizes.** Preflop raises go to 2.5bb and 3-bets to 3×. After the flop, bets are ½, ¾ or 1¼ pot and raises come in two sizes. There are no 4-bets, no preflop shoves and no overbets beyond 1¼ pot.
* **At most two bets or raises per street**, so a betting war is always cut short.
* **Every hand starts at 100bb**, so there's no short-stack or deep-stack play, and no tournaments, ICM or rake.
* **The bots play on a fixed set of boards:** 300 flops, each with 6 turns and 6 rivers. Through `bots/interface.py` the bot handles any real board, but it's tested on these.

### The bots' opponents are simple

* **They're trained and tested against rule-based profiles and copies of themselves.** The profiles fold too much preflop, call too much after the flop and never 4-bet. The big bb/100 numbers mean "beats these opponents", not "plays well against people".
* **It isn't a game-theory-optimal strategy.** It's exploitative: it learns what works against the players it meets. A person who spots its habits (it calls a lot preflop, and its 1¼-pot bets are only about 6% bluffs) can exploit it.
* **Reads take time.** Against a new player it starts from no-read assumptions and needs a few hundred hands before its reads count for much. It barely beats "no read" against nits.
* **Big tables have the least training and testing.** 7 to 9-player tables only came in for the last 500,000 of its 1,500,000 training hands. Results there have the widest margins (about ±350–470 bb/100 in the final test).
* **At a bigger table some reads are averaged.** When there are several opponents in the hand, the model gets the average of their reads rather than each one's. The hand-built ranging still keeps a separate range per player.

### The math is approximate in places

* **The bots' flop and turn equity uses only the sampled runouts:** 36 per flop and 6 per turn. That keeps the arena fast and matches the boards it deals, but any single spot can be off. In a check of 12 spots against a 20,000-sim `RangeOpponents` run, the bot's multiway equity was within 0–1.6 points on the river and up to about 9 points off on the flop and turn.
* **Multiway equity treats opponents' ranges as independent,** with a pairwise card-removal correction. With several narrow ranges that's an approximation.
* **Preflop multiway equity** uses 1,000 random boards.
* **`hand_analysis.py`'s "real table"** folds by starting hand only. Opponents' hands don't depend on how they bet, and "playable" is a single cut on equity against random hands, without position.
* **Preflop tables exist for 3, 5, 7 and 9 players.** Other sizes need building first with `poker/Data/preflop_tables.py`.
* **`advise.py`'s ranging rules are hand-set assumptions** (the no-read `Profile`), not fitted to real players' data. With several opponents, only the first is narrowed by the actions you give; the others are drawn from the hands people play.

### The checks are limited

* **`accuracy/validate.py` checks heads-up against one random hand.** Flop to river it uses the same evaluator for the enumeration and the sim. Preflop it compares against published equities for 8 hands.
* **Results have wide margins.** Poker is noisy: separating two bots by 10 bb/100 takes tens of thousands of hands, so small differences in the tables above are within the margin of error.

---

## Project layout

```
holdem-lab/
├── README.md
├── hand_analysis.py           # 1. equity: how often your hand wins
├── advise.py                  # 2. decisions: play / call / fold, pot odds, ranging
├── play.py                    # play the trained bot, the hand-built bot or presets, 2 to 9 players
├── bots/                      # 3. bots
│   ├── pool.py                # scores every hand on 10,800 real boards, flop to river, cached
│   ├── ml_agent.py            # the bot: hand-built versions + the ML bot
│   ├── arena.py               # the table (2-9 seats, side pots); plays bot versions against random opponents
│   ├── train.py               # trains the ML bot, tests on held-out opponents
│   ├── explain.py             # which inputs the ML bot relies on, training animation
│   └── interface.py           # the plug-in bot: Bot.act(GameState) for game engines and competitions
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
