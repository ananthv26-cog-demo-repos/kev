"""Training curves from one or more runs' metrics.jsonl: lines per game (eval, greedy) and the noul Brier score.

    uv run python -m tetris.plot runs/tetris/bc-rl-0.8b-v2 runs/tetris/rl-0.8b-v1 --out docs/tetris-curves.png
"""
import argparse
import json
from pathlib import Path


def read(run):
    rows = [json.loads(l) for l in open(Path(run) / "metrics.jsonl") if l.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--out", default="docs/tetris-curves.png")
    ap.add_argument("--baseline", type=float, default=None, help="heuristic lines per game at the same piece cap, drawn as a dashed line")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for run in a.runs:
        rows = read(run); name = Path(run).name
        it = [r["iter"] for r in rows]
        ev = [(r["iter"], r["eval_lines_mean"]) for r in rows if "eval_lines_mean" in r]
        axes[0].plot([x for x, _ in ev], [y for _, y in ev], marker=".", label=name)
        axes[1].plot(it, [r["train_pieces_mean"] for r in rows], marker=".", label=name)
        br = [(r["iter"], r["eval_survive_brier"]) for r in rows if r.get("eval_survive_brier") is not None]
        axes[2].plot([x for x, _ in br], [y for _, y in br], marker=".", label=name)
        bc = [r["iter"] for r in rows if r.get("mode") == "bc"]
        if bc:
            for ax in axes: ax.axvspan(min(bc) - 0.5, max(bc) + 0.5, color="#ddd", alpha=0.5, lw=0)
    if a.baseline is not None: axes[0].axhline(a.baseline, ls="--", color="k", lw=1, label="heuristic")
    axes[0].set_title("lines per game (greedy eval)"); axes[1].set_title("pieces survived per training game"); axes[2].set_title("Brier of 'clears 5 more lines?' (lower is better)")
    for ax in axes: ax.set_xlabel("iteration"); ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout(); Path(a.out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, dpi=120)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
