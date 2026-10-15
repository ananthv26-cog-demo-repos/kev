"""The Tetris position as a System One request: the board is the state, the legal placements are the options of one
Choice question, and a Noul question asks whether the game will still clear more lines (trained against what happened).

Two option encodings:
  rich  each option says what it does (columns, landing row, lines it clears, holes it makes, resulting height). The model
        has to read and weigh the facts; the facts come from the simulator, as application code would supply them to Jev.
  bare  each option says only where the piece goes. The model has to work out the consequences from the board itself.
"""
from .env import Game, Placement, ROTATION_NAMES

MOVE_INSTRUCTIONS = ("You are playing Tetris. Choose the placement for the current piece. Prefer placements that clear lines, "
                     "keep the stack low and flat, and create no holes (empty cells covered from above).")
SURVIVE_INSTRUCTIONS = "Will this game clear at least {n} more lines before it ends?"
SURVIVE_LINES = 5


def render_board(game: Game):
    rows = []
    for i, row in enumerate(game.board):
        rows.append("".join("#" if x else "." for x in row))
    return "\n".join(rows)


def describe(game: Game, p: Placement, encoding="rich"):
    name = ROTATION_NAMES[game.current][p.rotation]
    lo, hi = p.columns
    cols = f"column {lo}" if lo == hi else f"columns {lo}-{hi}"
    parts = [f"{game.current} {name}, {cols}"]
    if encoding == "bare":
        return ", ".join(parts)
    if p.game_over:
        parts.append("does not fit: game over")
        return ", ".join(parts)
    parts.append(f"top at height {max(game.H - r for r, _ in p.cells)}")
    if p.lines: parts.append(f"clears {p.lines} line{'s' if p.lines > 1 else ''}")
    if p.holes_made: parts.append(f"makes {p.holes_made} hole{'s' if p.holes_made > 1 else ''}")
    parts.append(f"max height after {max(p.heights_after)}")
    return ", ".join(parts)


def state_of(game: Game):
    heights = game.heights()
    return {"game": "tetris", "board_top_to_bottom": render_board(game), "column_heights": " ".join(map(str, heights)),
            "holes": game.holes(), "current_piece": game.current, "next_piece": game.next, "lines_cleared": game.lines, "pieces_placed": game.pieces}


def request_of(game: Game, placements, encoding="rich", survive=True):
    """-> (System One request dict, placement keys in option order)."""
    criteria = {p.key: describe(game, p, encoding) for p in placements}
    questions = {"placement": {"type": "choice", "instructions": MOVE_INSTRUCTIONS, "criteria": criteria}}
    if survive:
        questions["survive"] = {"type": "noul", "instructions": SURVIVE_INSTRUCTIONS.format(n=SURVIVE_LINES)}
    return {"state": state_of(game), "model": "kev-latest", "questions": questions}, [p.key for p in placements]
