"""Placement-level Tetris: one decision per piece, the options are every (rotation, column) the piece can be dropped into.

A board is a list of rows, top row first, each a list of 0/1. Pieces fall straight down (no sliding or tucking), which
keeps the option list small (up to 34 placements on a 10-wide board) and every option legal by construction. The game ends
when a placed piece sticks out above the top row, or after `max_pieces` pieces, so an episode always terminates.

`Placement` carries what the placement does to the board (lines cleared, holes made, resulting heights), both for the
heuristic agent and for the option descriptions the decision model reads (tetris.encode).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# cells as (row, col) offsets, row 0 = top of the piece's bounding box, for every distinct rotation
PIECES = {
    "I": [[(0, 0), (0, 1), (0, 2), (0, 3)], [(0, 0), (1, 0), (2, 0), (3, 0)]],
    "O": [[(0, 0), (0, 1), (1, 0), (1, 1)]],
    "T": [[(0, 0), (0, 1), (0, 2), (1, 1)], [(0, 0), (1, 0), (2, 0), (1, 1)], [(1, 0), (1, 1), (1, 2), (0, 1)], [(0, 1), (1, 1), (2, 1), (1, 0)]],
    "S": [[(0, 1), (0, 2), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1), (2, 1)]],
    "Z": [[(0, 0), (0, 1), (1, 1), (1, 2)], [(0, 1), (1, 0), (1, 1), (2, 0)]],
    "J": [[(0, 0), (1, 0), (1, 1), (1, 2)], [(0, 0), (0, 1), (1, 0), (2, 0)], [(0, 0), (0, 1), (0, 2), (1, 2)], [(0, 1), (1, 1), (2, 1), (2, 0)]],
    "L": [[(0, 2), (1, 0), (1, 1), (1, 2)], [(0, 0), (1, 0), (2, 0), (2, 1)], [(0, 0), (0, 1), (0, 2), (1, 0)], [(0, 0), (0, 1), (1, 1), (2, 1)]],
}
# how each rotation reads to a person (and to the model), in PIECES order
ROTATION_NAMES = {"I": ["flat", "upright"], "O": ["square"], "T": ["stem down", "stem right", "stem up", "stem left"],
                  "S": ["flat", "upright"], "Z": ["flat", "upright"],
                  "J": ["hook up-left", "foot left", "hook down-right", "hook up-right"],
                  "L": ["hook up-right", "foot right", "hook down-left", "hook up-left"]}
LINE_SCORES = {0: 0, 1: 1, 2: 3, 3: 5, 4: 8}   # classic-ish: multi-line clears are worth more than their lines


@dataclass
class Placement:
    rotation: int
    col: int                    # leftmost column of the piece's bounding box
    cells: list                 # absolute (row, col) cells the piece occupies after dropping; a row < 0 is above the board
    lines: int                  # lines this placement clears
    holes_made: int             # empty cells newly covered from above
    landing_height: float       # height of the piece's centre above the floor (Dellacherie's landing height)
    heights_after: list         # column heights after the placement and its clears
    board_after: list = field(repr=False)
    game_over: bool = False

    @property
    def key(self):
        return f"r{self.rotation}c{self.col}"

    @property
    def columns(self):
        cols = sorted({c for _, c in self.cells})
        return cols[0], cols[-1]


def column_heights(board):
    H, W = len(board), len(board[0])
    out = []
    for c in range(W):
        h = 0
        for r in range(H):
            if board[r][c]:
                h = H - r; break
        out.append(h)
    return out


def count_holes(board):
    H, W = len(board), len(board[0])
    holes = 0
    for c in range(W):
        covered = False
        for r in range(H):
            if board[r][c]: covered = True
            elif covered: holes += 1
    return holes


def bumpiness(heights):
    return sum(abs(a - b) for a, b in zip(heights, heights[1:]))


def wells(heights):
    """Sum of well depths (a column lower than both neighbours; walls count as neighbours of infinite height)."""
    total = 0
    for i, h in enumerate(heights):
        left = heights[i - 1] if i > 0 else 10 ** 6
        right = heights[i + 1] if i + 1 < len(heights) else 10 ** 6
        d = min(left, right) - h
        if d > 0: total += d
    return total


class Game:
    def __init__(self, width=10, height=20, seed=0, max_pieces=500):
        self.W, self.H, self.max_pieces = width, height, max_pieces
        self.rng = random.Random(seed)
        self.seed = seed
        self.board = [[0] * width for _ in range(height)]
        self.bag = []
        self.current, self.next = self._draw(), self._draw()
        self.lines = self.score = self.pieces = 0
        self.over = False
        self.history = []   # keys of the placements taken

    def _draw(self):
        if not self.bag:
            self.bag = list(PIECES); self.rng.shuffle(self.bag)   # 7-bag randomiser, as modern Tetris
        return self.bag.pop()

    def heights(self):
        return column_heights(self.board)

    def holes(self):
        return count_holes(self.board)

    def placements(self):
        """Every legal drop of the current piece, in (rotation, column) order."""
        out = []
        heights = self.heights()
        for rot, shape in enumerate(PIECES[self.current]):
            w = max(c for _, c in shape) + 1
            h = max(r for r, _ in shape) + 1
            for col in range(self.W - w + 1):
                out.append(self._drop(rot, shape, w, h, col, heights))
        return out

    def _drop(self, rot, shape, w, h, col, heights):
        # the piece's bottom profile per column: the lowest cell row offset in that column
        bottoms = {}
        for r, c in shape: bottoms[c] = max(bottoms.get(c, -1), r)
        # top-of-box row so the piece rests on the stack: for each column the piece's lowest cell lands on the column's top
        top = min(self.H - heights[col + c] - 1 - b for c, b in bottoms.items())
        cells = [(top + r, col + c) for r, c in shape]
        game_over = any(r < 0 for r, _ in cells)
        board = [row[:] for row in self.board]
        for r, c in cells:
            if 0 <= r < self.H: board[r][c] = 1
        holes_before = count_holes(self.board)
        full = [i for i, row in enumerate(board) if all(row)]
        for i in full:
            del board[i]; board.insert(0, [0] * self.W)
        after = column_heights(board)
        landing = self.H - (top + h) + h / 2   # centre of the piece's bounding box above the floor
        return Placement(rotation=rot, col=col, cells=cells, lines=len(full), holes_made=max(0, _holes_without_clears(self.board, cells, self.H, self.W) - holes_before),
                         landing_height=landing, heights_after=after, board_after=board, game_over=game_over)

    def step(self, p: Placement):
        """Apply a placement. Returns (lines cleared, game over)."""
        if self.over: raise ValueError("game is over")
        self.board = p.board_after
        self.lines += p.lines; self.score += LINE_SCORES[p.lines]; self.pieces += 1
        self.history.append(p.key)
        self.current, self.next = self.next, self._draw()
        if p.game_over or self.pieces >= self.max_pieces: self.over = True
        return p.lines, self.over

    def render(self, empty=".", full="#"):
        return "\n".join("".join(full if x else empty for x in row) for row in self.board)

    def snapshot(self):
        return {"board": [row[:] for row in self.board], "current": self.current, "next": self.next, "lines": self.lines, "score": self.score,
                "pieces": self.pieces, "over": self.over, "heights": self.heights(), "holes": self.holes()}


def _holes_without_clears(board, cells, H, W):
    """Holes the placement makes, counted before any line clears (a cleared line can hide a hole it created)."""
    b = [row[:] for row in board]
    for r, c in cells:
        if 0 <= r < H: b[r][c] = 1
    return count_holes(b)


def heuristic_score(p: Placement, weights=(-0.510066, 0.760666, -0.35663, -0.184483)):
    """The widely used linear evaluation (aggregate height, lines, holes, bumpiness; Yiyuan Lee's GA weights). Higher is better."""
    wh, wl, wo, wb = weights
    return wh * sum(p.heights_after) + wl * p.lines + wo * count_holes(p.board_after) + wb * bumpiness(p.heights_after) + (-1000 if p.game_over else 0)
