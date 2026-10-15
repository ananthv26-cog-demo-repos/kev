"""The Tetris position as a System One request: the board is the state, the legal placements are the options of one
Choice question, and a Noul question asks whether the game will still clear more lines (trained against what happened).

Two option encodings:
  rich  each option says what it does (columns, landing row, lines it clears, holes it makes, resulting height). The model
        has to read and weigh the facts; the facts come from the simulator, as application code would supply them to Jev.
  bare  each option says only where the piece goes. The model has to work out the consequences from the board itself.
"""
from .env import Game, Placement, ROTATION_NAMES

MOVE_INSTRUCTIONS = ("Tetris: choose the placement for the current piece. Options give columns, the height of the piece's top, lines "
                     "cleared, holes made and the stack's max height after. Prefer clearing lines, a low flat stack, and no holes.")
MAX_BRANCH_TOKENS = 1536   # in-process row limit: the pathological boards a random player builds (16 holes in one option) pass 1,024
OPTION_KEYS = [chr(ord("A") + i) for i in range(26)] + [chr(ord("a") + i) for i in range(26)]   # option names: 34 placements at most
SURVIVE_INSTRUCTIONS = "Will this game clear at least {n} more lines before it ends?"
SURVIVE_LINES = 5


def render_board(game: Game):
    rows = []
    for i, row in enumerate(game.board):
        rows.append("".join("#" if x else "." for x in row))
    return "\n".join(rows)


def describe(game: Game, p: Placement, encoding="rich"):
    """Terse on purpose: 34 options have to fit one question row. kev.serve's limit is 1,024 tokens (kev.model.MAX_BRANCH);
    in-process callers (tetris.rl, ModelAgent) allow MAX_BRANCH_TOKENS. A 10x20 board with this wording peaks near 1,000."""
    name = ROTATION_NAMES[game.current][p.rotation]
    lo, hi = p.columns
    cols = f"col {lo}" if lo == hi else f"cols {lo}-{hi}"
    parts = [f"{game.current} {name}, {cols}"]
    if encoding == "bare":
        return ", ".join(parts)
    if p.game_over:
        parts.append("does not fit: game over")
        return ", ".join(parts)
    parts.append(f"top {max(game.H - r for r, _ in p.cells)}")
    if p.lines: parts.append(f"clears {p.lines}")
    if p.holes_made: parts.append(f"{p.holes_made} hole{'s' if p.holes_made > 1 else ''}")
    parts.append(f"max {max(p.heights_after)}")
    return ", ".join(parts)


def state_of(game: Game):
    heights = game.heights()
    return {"game": "tetris", "board_top_to_bottom": render_board(game), "column_heights": " ".join(map(str, heights)),
            "holes": game.holes(), "current_piece": game.current, "next_piece": game.next, "lines_cleared": game.lines, "pieces_placed": game.pieces}


def request_of(game: Game, placements, encoding="rich", survive=True):
    """-> (System One request dict, placement keys in option order)."""
    criteria = {OPTION_KEYS[i]: describe(game, p, encoding) for i, p in enumerate(placements)}
    questions = {"placement": {"type": "choice", "instructions": MOVE_INSTRUCTIONS, "criteria": criteria}}
    if survive:
        questions["survive"] = {"type": "noul", "instructions": SURVIVE_INSTRUCTIONS.format(n=SURVIVE_LINES)}
    return {"state": state_of(game), "model": "kev-latest", "questions": questions}, list(criteria)
