# Kev learns Tetris from outcomes

An experiment on top of Kev: can a Jev-style decision model learn a game with no teacher, only from what happens when it
plays? Kev's released checkpoints are trained with cross-entropy on labelled answers. Here the placement question is trained
by policy gradient on the lines it clears, and the "will this game clear 5 more lines?" question is trained on whether it
did. Same pointer head, same checkpoint format, a different teacher.

## The game as a decision

The board is the state. Every legal drop of the current piece is one option of a `choice` question (up to 34 on a 10x20
board). A `noul` question asks whether the game will clear at least 5 more lines before it ends.

```text
state:  board (20 rows of . and #), column heights, holes, current piece, next piece, lines so far
choice: A: T stem down, cols 3-5, top 4, clears 1, max 5
        B: T stem down, cols 4-6, top 5, 1 hole, max 6
        ...
noul:   Will this game clear at least 5 more lines before it ends?
```

`rich` options say what the placement does (the simulator supplies those facts, as application code would to Jev); `bare`
options only say where the piece goes, so the model has to work the consequences out from the board.

## Files

| file | what |
|---|---|
| `tetris/env.py` | placement-level Tetris: 7-bag pieces, straight drops, line clears, hole and height accounting, the classic linear heuristic |
| `tetris/encode.py` | the position as a System One request |
| `tetris/agents.py` | random, heuristic, a Kev checkpoint in-process, any System One endpoint over HTTP |
| `tetris/play.py` | play N games, paired seeds across agents, JSONL logs |
| `tetris/viewer.py` | browser view: board, candidate placements shaded by probability, lines-per-game history |
| `tetris/rl.py` | the trainer: PPO-clipped policy gradient on the placement question, outcome cross-entropy on the noul question |
| `tetris/modal_app.py` | train / play / pull on Modal GPUs, with a budget check |
| `tests/test_tetris.py` | environment, encoding and trainer plumbing, no weights |

## Commands

```bash
uv run python -m pytest tests/test_tetris.py -q
uv run python -m tetris.play --agent heuristic --games 20
uv run --extra serve python -m tetris.viewer --agent heuristic --port 8010          # open http://localhost:8010
uv run modal run tetris/modal_app.py::smoke                                          # one tiny iteration on an H100 (~2 min)
uv run modal run --detach tetris/modal_app.py::train --name rl-0.8b-v1 --args "--iters 60 --games 16 --max_pieces 150"
uv run modal run tetris/modal_app.py::pull --name rl-0.8b-v1                         # -> runs/tetris/rl-0.8b-v1
uv run --extra serve python -m tetris.viewer --agent model --run runs/tetris/rl-0.8b-v1/checkpoint
```

## How the trainer works

1. **Rollout.** 16 games run in lockstep. At each piece the model's placement probabilities are sampled (not argmaxed), so
   it tries things. Each decision records its log-probability and a reward: `LINE_SCORES[lines]` (1/3/5/8) minus 0.3 per
   hole made, minus 0.02 per unit the max height rises, minus 3 for topping out.
2. **Returns.** Rewards are summed backwards with discount 0.97, then standardised across the batch. A decision's advantage
   is how much better its return was than the batch average.
3. **Update.** Two passes over the decisions in microbatches of 8: PPO-clipped policy gradient (`clip=0.2`) on the
   placement question, an entropy bonus (0.01) so it keeps exploring, and cross-entropy on the noul question against the
   actual outcome (weight 0.5). LoRA (rank 16) and the pointer head train; the base stays frozen. The update runs the
   network in eval mode, so there is no dropout and PPO's ratio starts at exactly 1.
4. **Evaluate.** 8 greedy games on fixed seeds after every iteration, plus the Brier score of the noul forecasts.

`--bc_iters N` is the fallback: N iterations of imitation (cross-entropy on the heuristic's greedy choice) before RL.

## Results

All runs: Kev-0.8B (`jaredpalmer/kev-0.8b`), one H100 each, 16 games per iteration. "Lines" are the greedy policy's mean
over 8 fixed seeds (10000-10007) at the piece cap named; the heuristic and random player are scored on the same seeds.
Metrics per iteration are in `runs/tetris/<run>/metrics.jsonl`; the curves are `docs/tetris-curves-100.png` and
`docs/tetris-curves-200.png`.

| run | recipe | encoding | lines @100 pieces | lines @200 | lines @500 (median) | cost |
|---|---|---|---|---|---|---|
| random | | | 0 (dies at ~22 pieces) | | | |
| Kev-0.8B untrained | | rich | 0 (dies at ~17 pieces) | | | |
| v1 | RL from the untrained model, 10 iterations | rich | 0 (dies at 13 pieces) | | | $1.0 |
| v3 | 6 iterations of imitation (BC) | rich | 36.1 | 71.9 | 190 (mean 156; 3 of 8 games topped out) | $1.1 |
| v2 | 6 BC then RL at lr 2e-5, 2 epochs | rich | 35.8 after BC, 24.4 three RL iterations later | | | $1.6 |
| v4b | RL from v3 at lr 5e-6, clip 0.1, KL stop 0.02 | rich | | 73.4 → 75.9 after 2 iterations (running) | | ~$1/iteration... |
| v5 + v5b | 24 iterations of imitation | bare | 26.5 | | | $2.0 |
| heuristic | | | 36.3 | 73–80 | 195 | |

A 10-wide board clears at most 0.4 lines per piece, so 40 @100, 80 @200 and 200 @500 are the ceilings.

## What happened and why (notes)

**Note 1: the untrained model is a coin flip.** Kev-0.8B zero-shot spreads its probability almost evenly over the 20-34
placements (entropy 2.9 nats; uniform over 20 is 3.0). It clears 0 lines and dies in ~17 pieces, like the random player.
Its "will this clear 5 more lines?" answer sits near 0.4 whatever the board. Nothing in its training looked like this.

**Note 2: pure RL from that start reward-hacked within 10 iterations (run v1).** The reward was lines (+1/3/5/8) minus
0.3 per hole, minus 0.02 per height unit, minus 3 for topping out. A near-random policy never sees a line clear, so the
only gradient comes from the penalties. The cheapest way to collect fewer penalties is to die sooner: games went from 22
pieces to 13 while the per-move reward "improved" from -1.04 to -0.69, and the entropy fell as it committed to that. The
PPO clip fraction sat at 0.4-0.6 (healthy is ~0.1-0.2), a sign the policy was moving much faster than the data supported.
Stopped at $1. Two fixes are standard and both are in the trainer: shape less (lower hole/height penalties, or an alive
bonus) and start from a policy that sees the reward at all (`--bc_iters`).

**Note 3: six iterations of imitation make the reward visible (run v2).** Cross-entropy on the heuristic's greedy choice,
over games the heuristic played at softmax temperature 1, took the model from 0 to 33 lines per 100 pieces after one
iteration (96 optimizer steps, 2.5 minutes). With `rich` options that is mostly reading: "clears 1" and "2 holes" are in
the text. The interesting question is what RL does from there.

**Note 4: RL on top of imitation needs a short leash (runs v2 and v4b).** At lr 2e-5 with two passes over each
iteration's ~1,600 decisions (v2), the greedy policy lost a third of its lines in three iterations while the clip
fraction sat at 0.3-0.5 and the entropy rose: the updates moved the policy further than the advantages justified. With
lr 5e-6, clip 0.1, one pass, and an early stop once the running approximate KL between the old and new policy passes
0.02 (v4b), the same start improved from 71.9 to 75.9 lines per 200 pieces in two iterations, against a ceiling of 80,
and the noul question's Brier score fell from 0.17 to 0.02 because it now sees many games that do and do not clear 5
more lines. The rules of thumb (clip fraction near 0.1-0.2, a KL target) exist because this failure is common.

**Note 5: without the hints, imitation is 10x slower (runs v5, v5b).** With `bare` options ("T stem down, cols 3-5"),
the model has to read the 20-row board to know what a placement does. The same 6 BC iterations that reached 36 lines
with `rich` options reached 3.4; after 24 iterations it was at 26.5 lines per 100 pieces and still climbing. The
pointer head can learn the geometry, but the text facts in `rich` were doing most of the work early on, which is
exactly how Jev-style systems are meant to be used: the application computes what it can and asks the model to decide.
