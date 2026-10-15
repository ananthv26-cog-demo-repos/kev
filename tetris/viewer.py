"""Watch an agent play in the browser: the board, the next piece, every candidate placement with its probability, and the
lines-per-game history. One FastAPI process runs the game loop in a thread and serves a single page that polls it.

    uv run --extra serve python -m tetris.viewer --agent heuristic --port 8010
    uv run --extra serve python -m tetris.viewer --agent remote --url http://127.0.0.1:8009 --port 8010
    uv run --extra serve python -m tetris.viewer --agent model --run runs/tetris/rl-0.8b/checkpoint --port 8010
"""
import argparse
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .agents import make_agent
from .env import Game

PAGE = Path(__file__).with_name("viewer.html")


class Loop:
    def __init__(self, agent, width, height, max_pieces, seed, delay):
        self.agent, self.W, self.H, self.max_pieces, self.seed, self.delay = agent, width, height, max_pieces, seed, delay
        self.lock = threading.Lock()
        self.paused = False
        self.history = []            # lines per finished game
        self.game = None
        self.last = {}               # the latest decision: placements, probabilities, the pick
        self.thread = threading.Thread(target=self.run, daemon=True); self.thread.start()

    def run(self):
        k = 0
        while True:
            g = Game(self.W, self.H, seed=self.seed + k, max_pieces=self.max_pieces)
            with self.lock: self.game = g; self.last = {}
            while not g.over:
                while self.paused: time.sleep(0.1)
                ps = g.placements()
                t0 = time.time()
                i, info = self.agent.act(g, ps)
                probs = info.get("probs") or _uniform_or_scores(info, len(ps))
                last = {"piece": g.current, "pick": ps[i].key, "think_ms": round((time.time() - t0) * 1000),
                        "survive": info.get("survive"), "state_tokens": info.get("state_tokens") or info.get("input_tokens"),
                        "candidates": [{"key": p.key, "cells": [[r, c] for r, c in p.cells], "p": probs[j], "lines": p.lines, "holes": p.holes_made} for j, p in enumerate(ps)]}
                with self.lock: self.last = last
                time.sleep(self.delay)
                with self.lock: g.step(ps[i])
            with self.lock: self.history.append({"seed": g.seed, "lines": g.lines, "pieces": g.pieces})
            k += 1
            time.sleep(min(2.0, self.delay * 4))

    def state(self):
        with self.lock:
            g = self.game
            snap = g.snapshot() if g else None
            return {"agent": self.agent.name, "game": snap, "last": self.last, "history": self.history[-50:], "paused": self.paused,
                    "games": len(self.history), "mean_lines": round(sum(h["lines"] for h in self.history) / len(self.history), 2) if self.history else None}


def _uniform_or_scores(info, n):
    if "scores" in info:
        m = max(info["scores"]); import math
        w = [math.exp(s - m) for s in info["scores"]]; z = sum(w)
        return [x / z for x in w]
    return [1 / n] * n


def build(loop: Loop):
    app = FastAPI(title="kev tetris")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE.read_text(encoding="utf-8")

    @app.get("/api/state")
    def state():
        return JSONResponse(loop.state())

    @app.post("/api/pause")
    def pause():
        loop.paused = not loop.paused
        return {"paused": loop.paused}

    @app.post("/api/delay/{ms}")
    def delay(ms: int):
        loop.delay = max(0, ms) / 1000
        return {"delay_ms": ms}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="heuristic", choices=["random", "heuristic", "remote", "model"])
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b"); ap.add_argument("--url", default="http://127.0.0.1:8009")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--encoding", default="rich", choices=["rich", "bare"])
    ap.add_argument("--sample", type=int, default=0); ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--width", type=int, default=10); ap.add_argument("--height", type=int, default=20)
    ap.add_argument("--max_pieces", type=int, default=500); ap.add_argument("--delay", type=float, default=0.4, help="seconds to show each decision before it is played")
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8010)
    a = ap.parse_args()
    agent = make_agent(a.agent, run=a.run, url=a.url, device=a.device, encoding=a.encoding, sample=bool(a.sample), temperature=a.temperature, seed=a.seed)
    import uvicorn
    uvicorn.run(build(Loop(agent, a.width, a.height, a.max_pieces, a.seed, a.delay)), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
