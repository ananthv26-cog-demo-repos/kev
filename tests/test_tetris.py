"""Tetris environment, encoding and agents, with no model weights. Run: uv run python -m pytest tests/test_tetris.py -q"""
import json

import torch

from tetris.agents import HeuristicAgent, RandomAgent
from tetris.encode import describe, request_of
from tetris.env import PIECES, Game, bumpiness, column_heights, count_holes, heuristic_score
from tetris.play import play_game


def test_pieces_have_four_cells_and_distinct_rotations():
    for name, rots in PIECES.items():
        for shape in rots:
            assert len(shape) == 4 and len(set(shape)) == 4 and min(r for r, _ in shape) == 0 and min(c for _, c in shape) == 0
        assert len({tuple(sorted(s)) for s in rots}) == len(rots), name


def test_placements_are_legal_and_rest_on_the_stack():
    g = Game(seed=3)
    for _ in range(30):
        ps = g.placements()
        assert ps and len(ps) <= 34
        assert len({p.key for p in ps}) == len(ps)
        for p in ps:
            for r, c in p.cells:
                assert 0 <= c < g.W and r < g.H
                if r >= 0: assert g.board[r][c] == 0                      # never overlaps the stack
            if not p.game_over:
                # resting: at least one cell sits on the floor or on a filled cell
                assert any(r + 1 == g.H or (r + 1 >= 0 and g.board[r + 1][c]) for r, c in p.cells)
        g.step(ps[0])
        if g.over: break


def test_line_clear_and_hole_accounting():
    g = Game(seed=0)
    g.board[-1] = [1] * 9 + [0]                       # one gap in the bottom row, under column 9
    g.current = "I"
    upright = [p for p in g.placements() if p.rotation == 1 and p.col == 9][0]
    assert upright.lines == 1 and upright.holes_made == 0
    assert sum(sum(row) for row in upright.board_after) == 3   # 9 + 4 - 10 cleared
    g.current = "O"
    over_gap = [p for p in g.placements() if p.col == 8][0]
    assert over_gap.lines == 0 and over_gap.holes_made == 1    # the O covers the gap
    assert count_holes(over_gap.board_after) == 1


def test_game_ends_when_a_piece_sticks_out_or_at_max_pieces():
    g = Game(seed=1, max_pieces=5)
    while not g.over: g.step(g.placements()[0])
    assert g.pieces == 5 and g.over
    g = Game(seed=1, max_pieces=10000)
    for r in range(1, g.H): g.board[r] = [1] * 9 + [0]       # a near-full stack
    g.current = "I"
    tall = [p for p in g.placements() if p.rotation == 1 and p.col == 0][0]
    assert tall.game_over
    g.step(tall); assert g.over


def test_heights_bumpiness_and_heuristic_prefer_clearing():
    board = [[0] * 10 for _ in range(20)]
    board[-1] = [1] * 9 + [0]; board[-2][0] = 1
    assert column_heights(board) == [2] + [1] * 8 + [0] and bumpiness(column_heights(board)) == 2
    g = Game(seed=0); g.board = board; g.current = "I"
    ps = g.placements()
    best = max(ps, key=heuristic_score)
    assert best.lines == 1 and best.col == 9 and best.rotation == 1


def test_request_shape_and_descriptions():
    g = Game(seed=2)
    ps = g.placements()
    req, keys = request_of(g, ps, "rich")
    assert list(req["questions"]["placement"]["criteria"]) == keys == [p.key for p in ps]
    assert req["questions"]["survive"]["type"] == "noul" and req["state"]["current_piece"] == g.current
    json.dumps(req)                                        # serialisable as an API request
    rich, bare = describe(g, ps[0], "rich"), describe(g, ps[0], "bare")
    assert bare in rich and "height" in rich and "height" not in bare
    from kev.api import SystemOneRequest, to_record
    rec, meta = to_record(SystemOneRequest.model_validate(req))
    assert [m["type"] for m in meta] == ["choice", "noul"] and len(rec["questions"][0]["options"]) == len(ps)


def test_heuristic_beats_random_on_the_same_seeds():
    h = [play_game(HeuristicAgent(), seed, max_pieces=60)[0].lines for seed in range(3)]
    r = [play_game(RandomAgent(seed), seed, max_pieces=60)[0].lines for seed in range(3)]
    assert sum(h) > sum(r) and all(play_game(HeuristicAgent(), s, max_pieces=60)[0].pieces == 60 for s in range(3))


class _StubModel(torch.nn.Module):
    """Scores options with a tiny bag-of-tokens model; the same interface tetris.rl uses on a DecisionModel."""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(2048, 8); self.w = torch.nn.Linear(8, 1)

    def encode(self, tok, rec):
        return {"opts": [[hash(o) % 2048 for o in q["options"]] for q in rec["questions"]]}

    def forward_batch(self, encs):
        return [[self.w(self.emb(torch.tensor(ids))).squeeze(-1) for ids in e["opts"]] for e in encs]

    def trainable_parameters(self):
        return list(self.parameters())


def test_rl_rollout_returns_and_update_run_on_a_stub():
    import random
    from tetris import rl
    a = rl.parse_args(["--out", "/tmp/x", "--width", "6", "--height", "8", "--max_pieces", "4", "--games", "2", "--epochs", "1", "--microbatch", "3"])
    model = _StubModel(); auto = torch.autocast("cpu", enabled=False)
    steps, games = rl.rollout(a, model, None, [1, 2], random.Random(0), auto)
    assert len(steps) == sum(g.pieces for g in games) and all(s.ret is not None and s.survive_label in (0, 1) for s in steps)
    last = [s for s in steps if s.game_id == 0][-1]
    assert last.ret == last.reward                                  # the final decision's return is its own reward
    assert all(0 <= s.action < s.n_options and s.logp <= 0 for s in steps)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    st = rl.update(a, model, opt, steps, auto, random.Random(0), "rl")
    assert st["optimizer_steps"] == -(-len(steps) // 3) and st["entropy"] > 0 and 0 <= st["clipfrac"] <= 1
    bc_steps, _ = rl.rollout(a, model, None, [3], random.Random(0), auto, actor="heuristic")
    assert all(s.teacher is not None for s in bc_steps)
    rl.update(a, model, opt, bc_steps, auto, random.Random(0), "bc")
    assert rl.survive_brier(model, steps, auto) is not None
