"""Play N games with an agent and report lines per game. Seeds are shared across agents, so two agents see the same piece
sequences (paired comparison).

    uv run python -m tetris.play --agent heuristic --games 20
    uv run python -m tetris.play --agent model --run jaredpalmer/kev-0.8b --games 5 --device cpu
    uv run python -m tetris.play --agent remote --url http://127.0.0.1:8009 --games 5
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from .agents import make_agent
from .env import Game


def play_game(agent, seed, width=10, height=20, max_pieces=500, log=None):
    g = Game(width, height, seed=seed, max_pieces=max_pieces)
    survive_preds = []
    while not g.over:
        ps = g.placements()
        i, info = agent.act(g, ps)
        if info.get("survive") is not None: survive_preds.append((g.lines, info["survive"]))
        if log is not None:
            log.append({"pieces": g.pieces, "piece": g.current, "choice": ps[i].key, "lines": ps[i].lines, "probs": info.get("probs"), "survive": info.get("survive")})
        g.step(ps[i])
    return g, survive_preds


def summarize(games):
    lines = [g.lines for g in games]; pieces = [g.pieces for g in games]
    return {"games": len(games), "lines_mean": statistics.mean(lines), "lines_median": statistics.median(lines), "lines_max": max(lines),
            "pieces_mean": statistics.mean(pieces), "score_mean": statistics.mean(g.score for g in games)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="heuristic", choices=["random", "heuristic", "remote", "model"])
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b"); ap.add_argument("--url", default="http://127.0.0.1:8009")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--encoding", default="rich", choices=["rich", "bare"])
    ap.add_argument("--sample", type=int, default=0, help="1: sample a placement from the probabilities instead of taking the argmax")
    ap.add_argument("--temperature", type=float, default=0.0, help="heuristic: softmax temperature over its scores (0 = greedy)")
    ap.add_argument("--games", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=10); ap.add_argument("--height", type=int, default=20); ap.add_argument("--max_pieces", type=int, default=500)
    ap.add_argument("--out", default="", help="write per-game results (and per-move logs) as JSONL here")
    a = ap.parse_args()
    agent = make_agent(a.agent, run=a.run, url=a.url, device=a.device, encoding=a.encoding, sample=bool(a.sample), temperature=a.temperature, seed=a.seed)
    games, rows, t0 = [], [], time.time()
    for k in range(a.games):
        log = []
        g, preds = play_game(agent, a.seed + k, a.width, a.height, a.max_pieces, log)
        games.append(g)
        brier = None
        if preds:
            from .encode import SURVIVE_LINES
            brier = statistics.mean((p - float(g.lines - l >= SURVIVE_LINES)) ** 2 for l, p in preds)
        rows.append({"seed": a.seed + k, "lines": g.lines, "pieces": g.pieces, "score": g.score, "survive_brier": brier, "moves": log})
        print(f"game {k}: {g.lines} lines in {g.pieces} pieces" + (f", survive Brier {brier:.3f}" if brier is not None else ""), flush=True)
    s = summarize(games); s["seconds"] = round(time.time() - t0, 1); s["agent"] = a.agent; s["encoding"] = a.encoding
    print(json.dumps(s))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"summary": s}) + "\n")


if __name__ == "__main__":
    main()
