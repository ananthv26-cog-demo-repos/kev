"""Kev learns Tetris from outcomes: policy-gradient fine-tuning of the adapter and pointer head, no teacher labels.

Each iteration plays a batch of games with the current policy (sampling placements from its probabilities), turns what
happened into per-decision returns, and takes PPO-clipped policy-gradient steps on the placement question. The Noul
"will this game clear 5 more lines?" question is trained on what actually happened (cross-entropy on the outcome), so
its probability is a calibrated forecast, not a guess. Together that is the open version of "decisions trained against
outcomes": the same pointer head, the same checkpoint format as kev.train, a different teacher.

    # smoke (CPU, a minute): tiny board, two games, one iteration
    uv run python -m tetris.rl --init_from jaredpalmer/kev-0.8b --device cpu --iters 1 --games 2 --max_pieces 3 --width 6 --height 8 --eval_games 1 --out runs/tetris/smoke
    # real (H100): see tetris/modal_app.py
    uv run python -m tetris.rl --init_from jaredpalmer/kev-0.8b --device cuda --iters 60 --games 16 --max_pieces 150 --out runs/tetris/rl-0.8b

Optional warm start: `--bc_iters N` first imitates the linear heuristic (tetris.env.heuristic_score) for N iterations,
cross-entropy on its greedy choice over rollouts it plays at softmax temperature 1. Pure RL from the released checkpoint is
the default; the warm start is the fallback if sparse early reward stalls it (a random player clears ~0 lines).
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from kev.api import SystemOneRequest, to_record
from kev.checkpoint import Checkpoint, Meta
from kev.device import select
from kev.model import DecisionModel, load_tokenizer
from kev.train import finish_checkpoint

from .agents import HeuristicAgent
from .encode import MAX_BRANCH_TOKENS, SURVIVE_LINES, request_of
from .env import LINE_SCORES, Game, heuristic_score

GPU_HOURLY = {"H100": 3.95, "H200": 4.54, "B200": 6.25, "T4": 0.59, "L4": 0.80, "A10G": 1.10}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", default="jaredpalmer/kev-0.8b", help="released checkpoint (adapter + head) to start from; its base/LoRA/head settings are reused")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto"); ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"], help="autocast dtype on CUDA")
    ap.add_argument("--seed", type=int, default=0)
    # environment
    ap.add_argument("--width", type=int, default=10); ap.add_argument("--height", type=int, default=20)
    ap.add_argument("--max_pieces", type=int, default=150, help="episode cap; a game that survives this long is a success")
    ap.add_argument("--encoding", default="rich", choices=["rich", "bare"])
    # rollouts
    ap.add_argument("--iters", type=int, default=60); ap.add_argument("--games", type=int, default=16, help="games per iteration, played in one batch")
    ap.add_argument("--bc_iters", type=int, default=0, help="behaviour-cloning iterations on the heuristic before RL (0 = pure RL)")
    # reward
    ap.add_argument("--w_lines", type=float, default=1.0, help="x LINE_SCORES[lines] (1/3/5/8)")
    ap.add_argument("--w_holes", type=float, default=0.3, help="penalty per hole a placement makes")
    ap.add_argument("--w_height", type=float, default=0.02, help="penalty per unit the max height rises")
    ap.add_argument("--w_death", type=float, default=3.0, help="penalty for topping out")
    ap.add_argument("--gamma", type=float, default=0.97)
    # optimisation
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--head_lr", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=2, help="PPO passes over each iteration's decisions")
    ap.add_argument("--microbatch", type=int, default=8, help="decisions per forward/backward")
    ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--target_kl", type=float, default=0.02, help="stop an iteration's updates once the approx KL(old || new), averaged over the microbatches so far, passes this (0 = never)")
    ap.add_argument("--w_survive", type=float, default=0.5, help="weight of the Noul outcome loss")
    ap.add_argument("--checkpointing", type=int, default=0)
    # evaluation / bookkeeping
    ap.add_argument("--eval_games", type=int, default=8, help="greedy games on fixed seeds after every iteration")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--gpu", default="", help="label for the cost estimate (H100, ...)")
    return ap.parse_args(argv)


# --- rollouts -----------------------------------------------------------------------------------------------------------

class Step:
    __slots__ = ("game_id", "rec", "enc", "action", "logp", "reward", "ret", "survive_label", "lines_before", "n_options", "teacher")

    def __init__(self, game_id, rec, enc, action, logp, reward, lines_before, n_options, teacher=None):
        self.game_id, self.rec, self.enc, self.action, self.logp, self.reward = game_id, rec, enc, action, logp, reward
        self.lines_before, self.n_options, self.teacher = lines_before, n_options, teacher
        self.ret = self.survive_label = None


def encode_games(model, tok, games, encoding):
    """-> (records, encodings, placements per game) for the games still running."""
    recs, encs, plist = [], [], []
    for g in games:
        ps = g.placements()
        req, _ = request_of(g, ps, encoding)
        rec, _ = to_record(SystemOneRequest.model_validate(req))
        recs.append(rec); encs.append(model.encode(tok, rec, max_branch=MAX_BRANCH_TOKENS)); plist.append(ps)
    return recs, encs, plist


def step_reward(a, g, p):
    r = a.w_lines * LINE_SCORES[p.lines] - a.w_holes * p.holes_made
    r -= a.w_height * max(0, max(p.heights_after) - max(g.heights()))
    if p.game_over: r -= a.w_death
    return r


@torch.no_grad()
def rollout(a, model, tok, seeds, rng, autocast, actor="policy", greedy=False):
    """Play len(seeds) games in lockstep. actor="policy": the model samples (or argmaxes) its placement question;
    actor="heuristic": the linear heuristic plays at softmax temperature 1 and its greedy choice is kept as the teacher label.
    -> (steps, finished games)."""
    model.eval()
    games = [Game(a.width, a.height, seed=s, max_pieces=a.max_pieces) for s in seeds]
    steps, teacher = [], HeuristicAgent(temperature=1.0, seed=rng.randrange(1 << 30))
    while True:
        live = [(i, g) for i, g in enumerate(games) if not g.over]
        if not live: break
        recs, encs, plist = encode_games(model, tok, [g for _, g in live], a.encoding)
        with autocast:
            out = model.forward_batch(encs)
        for (i, g), rec, enc, ps, logits in zip(live, recs, encs, plist, out):
            z = logits[0].float()
            logp_all = F.log_softmax(z, -1)
            if actor == "heuristic":
                action, _ = teacher.act(g, ps)
                best = max(range(len(ps)), key=lambda j: heuristic_score(ps[j]))
            else:
                action = int(torch.argmax(z)) if greedy else int(torch.multinomial(logp_all.exp(), 1))
                best = None
            steps.append(Step(i, rec, enc, action, float(logp_all[action]), step_reward(a, g, ps[action]), g.lines, len(ps), best))
            g.step(ps[action])
    # returns and outcome labels, per game
    by_game = {}
    for s in steps: by_game.setdefault(s.game_id, []).append(s)
    for i, ss in by_game.items():
        G = 0.0
        for s in reversed(ss):
            G = s.reward + a.gamma * G; s.ret = G
            s.survive_label = int(games[i].lines - s.lines_before >= SURVIVE_LINES)
    return steps, games


# --- update --------------------------------------------------------------------------------------------------------------

def policy_loss(a, model, batch, adv, autocast, mode):
    """PPO-clipped policy gradient on the placement question (or cross-entropy on the teacher's choice in BC mode) plus the
    outcome loss on the Noul question. -> (loss, stats)."""
    with autocast:
        out = model.forward_batch([s.enc for s in batch])
    pg = ent = sv = clipped = kl = 0.0; n = len(batch)
    pg_terms, ent_terms, sv_terms = [], [], []
    for s, logits, A in zip(batch, out, adv):
        z = logits[0].float(); logp = F.log_softmax(z, -1); p = logp.exp()
        H = -(p * logp).sum()
        if mode == "bc":
            pg_terms.append(-logp[s.teacher])
        else:
            ratio = torch.exp(logp[s.action] - s.logp)
            unclipped, clip = ratio * A, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * A
            pg_terms.append(-torch.min(unclipped, clip))
            clipped += float((ratio < 1 - a.clip) | (ratio > 1 + a.clip))
            kl += float(s.logp - logp[s.action].detach())   # approx KL(old || new) at the taken action
        ent_terms.append(H)
        if len(logits) > 1:
            sv_terms.append(F.cross_entropy(logits[1].float()[None], torch.tensor([s.survive_label], device=z.device)))
    pg = torch.stack(pg_terms).mean(); ent = torch.stack(ent_terms).mean()
    sv = torch.stack(sv_terms).mean() if sv_terms else torch.zeros((), device=pg.device)
    loss = pg - a.entropy * ent + a.w_survive * sv
    return loss, {"pg": pg.item(), "entropy": ent.item(), "survive_ce": sv.item(), "clipfrac": clipped / n, "approx_kl": kl / n}


def update(a, model, opt, steps, autocast, rng, mode):
    """One iteration's optimisation: `epochs` shuffled passes over the decisions in microbatches."""
    model.eval()   # no LoRA dropout: the rollout and the update see the same network (PPO's ratio starts at 1)
    rets = torch.tensor([s.ret for s in steps])
    adv_all = ((rets - rets.mean()) / (rets.std() + 1e-6)).tolist() if len(steps) > 1 else [0.0]
    stats, count, stopped = {"pg": 0.0, "entropy": 0.0, "survive_ce": 0.0, "clipfrac": 0.0, "approx_kl": 0.0, "grad_norm": 0.0}, 0, False
    order = list(range(len(steps))); kl_sum = 0.0
    for _ in range(a.epochs):
        rng.shuffle(order)
        for k in range(0, len(order), a.microbatch):
            idx = order[k: k + a.microbatch]
            loss, st = policy_loss(a, model, [steps[i] for i in idx], [adv_all[i] for i in idx], autocast, mode)
            kl_sum += st["approx_kl"]
            if mode == "rl" and a.target_kl and count >= 8 and kl_sum / (count + 1) > a.target_kl:
                stopped = True; break          # the policy has moved as far as this iteration's data supports (SB3-style early stop, on the running mean: one microbatch of 8 is too noisy)
            opt.zero_grad(set_to_none=True); loss.backward()
            st["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0))
            opt.step()
            for key in stats: stats[key] += st[key]
            count += 1
        if stopped: break
    return {k: v / max(count, 1) for k, v in stats.items()} | {"optimizer_steps": count, "kl_stopped": stopped}


# --- evaluation ------------------------------------------------------------------------------------------------------------

def evaluate(a, model, tok, autocast, seeds):
    steps, games = rollout(a, model, tok, seeds, random.Random(0), autocast, greedy=True)
    return {"eval_lines_mean": statistics.mean(g.lines for g in games), "eval_lines_max": max(g.lines for g in games),
            "eval_pieces_mean": statistics.mean(g.pieces for g in games), "eval_survived": sum(g.pieces >= a.max_pieces for g in games) / len(games),
            "eval_survive_brier": survive_brier(model, steps, autocast)}, games


@torch.no_grad()
def survive_brier(model, steps, autocast):
    """Brier score of the Noul forecasts against what happened, on the iteration's own rollouts (recomputed in eval mode)."""
    tot, n = 0.0, 0
    for k in range(0, len(steps), 16):
        batch = steps[k: k + 16]
        with autocast:
            out = model.forward_batch([s.enc for s in batch])
        for s, logits in zip(batch, out):
            if len(logits) > 1:
                p = float(F.softmax(logits[1].float(), -1)[1]); tot += (p - s.survive_label) ** 2; n += 1
    return tot / n if n else None


# --- main --------------------------------------------------------------------------------------------------------------------

def build_model(a, dev):
    ck = Checkpoint(a.init_from); m = ck.meta
    tok = load_tokenizer(m.base, revision=m.base_revision)
    model = DecisionModel(m.base, tok, dev, lora=m.lora, revision=m.base_revision, head_dim=m.head_dim, option_isolation=m.option_isolation,
                          special_embeddings=m.special_embeddings, dtype=torch.float32)
    meta = Meta(base=m.base, base_revision=m.base_revision, lora=m.lora, head_dim=m.head_dim, option_isolation=m.option_isolation,
                special_embeddings=m.special_embeddings, weights_dtype="fp32", holdout=[], weights="lora")
    init = ck.warm_start(model, meta)
    if a.checkpointing: model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    return tok, model, meta, init


def save(out, model, meta, tok, a, extra):
    out.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(out)
    meta.head = model.head.state_dict(); meta.temperature = 1.0
    meta.extra = {"args": vars(a), "trainer": "tetris.rl", **extra}
    finish_checkpoint(out, meta, tok)


def main(argv=None):
    a = parse_args(argv)
    dev = select(a.device, dtype=torch.bfloat16 if a.dtype == "bf16" else torch.float32)
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    if dev == "cuda": torch.backends.cuda.matmul.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if (a.dtype == "bf16" and dev == "cuda") else torch.autocast("cpu", enabled=False)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok, model, meta, init = build_model(a, dev)
    print(f"device={dev} init={init['resolved']} trainable={sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M", flush=True)
    head_ids = {id(p) for p in model.head.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in model.trainable_parameters() if id(p) not in head_ids], "lr": a.lr},
                             {"params": list(model.head.parameters()), "lr": a.head_lr or a.lr}], weight_decay=0.0)
    (out / "training_config.json").write_text(json.dumps({"args": vars(a), "init_source": init}, indent=1))
    eval_seeds = [10_000 + i for i in range(a.eval_games)]
    log = open(out / "metrics.jsonl", "w"); t_start = time.time()   # one run, one log: a reused --out starts over
    total = a.bc_iters + a.iters
    for it in range(1, total + 1):
        mode = "bc" if it <= a.bc_iters else "rl"
        t0 = time.time()
        seeds = [rng.randrange(1 << 30) for _ in range(a.games)]
        steps, games = rollout(a, model, tok, seeds, rng, autocast, actor="heuristic" if mode == "bc" else "policy")
        t_roll = time.time() - t0
        st = update(a, model, opt, steps, autocast, rng, mode)
        t_upd = time.time() - t0 - t_roll
        row = {"iter": it, "mode": mode, "decisions": len(steps), "train_lines_mean": statistics.mean(g.lines for g in games), "train_lines_max": max(g.lines for g in games),
               "train_pieces_mean": statistics.mean(g.pieces for g in games), "reward_mean": statistics.mean(s.reward for s in steps),
               "return_mean": statistics.mean(s.ret for s in steps), **st, "survive_brier": survive_brier(model, steps, autocast),
               "rollout_seconds": round(t_roll, 1), "update_seconds": round(t_upd, 1)}
        if a.eval_games and (it % a.eval_every == 0 or it == total):
            ev, _ = evaluate(a, model, tok, autocast, eval_seeds); row.update(ev)
        row["elapsed_seconds"] = round(time.time() - t_start)
        if a.gpu in GPU_HOURLY: row["est_cost_usd"] = round(GPU_HOURLY[a.gpu] * row["elapsed_seconds"] / 3600, 2)
        log.write(json.dumps(row) + "\n"); log.flush()
        print(json.dumps(row), flush=True)
        if a.save_every and it % a.save_every == 0:
            save(out / "checkpoint", model, meta, tok, a, {"iteration": it, "init_source": init})
    save(out / "checkpoint", model, meta, tok, a, {"iteration": total, "init_source": init})
    print("saved", out / "checkpoint", flush=True)


if __name__ == "__main__":
    main()
