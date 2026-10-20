"""De-bias the fighter-1/fighter-2 column ordering in the raw fight stats.

The raw data lists the winner as f1 for every fight before 2010, and even
afterwards f1 is the more experienced fighter more often than chance. Any
model trained on fixed f1/f2 columns can partly learn "which slot" instead of
"which fighter is better".

The fix here relabels each fight's fighters as fighter_a / fighter_b using a
coin flip, so slot no longer correlates with outcome. The coin flip is a hash
of the fight's own stable id (`fights_url`) rather than a positional random
draw, which matters for the pipeline as new fights get appended over time:

    rng.random(len(df))   -> row 5's outcome depends on how many rows exist
    hash(fight_id + seed) -> row 5's outcome depends only on that fight

That makes the transform idempotent and stable across reruns: appending new
rows, or reordering existing ones, never changes the A/B assignment already
given to a fight that was processed before.
"""

import hashlib

import numpy as np
import pandas as pd

from ufc_predictor.config import RANDOMIZE_SEED


def stable_swap_mask(ids: pd.Series, seed: int = RANDOMIZE_SEED) -> pd.Series:
    """Deterministic per-row coin flip, keyed by a stable id rather than row position."""

    def _flip(fight_id: str) -> bool:
        # sha256 turns "fight_id::seed" into 32 random-looking bytes, always
        # the same bytes for the same input. We only need one of them: take
        # the first byte (a number 0-255) and check even/odd - that's the
        # coin flip. Same fight_id + same seed -> always the same flip.
        digest = hashlib.sha256(f"{fight_id}::{seed}".encode()).digest()
        return digest[0] % 2 == 0

    # astype(str) first so every id is a plain string before it goes into the
    # f-string above, regardless of what type it originally was.
    return ids.astype(str).map(_flip)


def find_side_column_pairs(columns) -> list[tuple[str, str, str]]:
    """Pair up every f1_X / f2_X column, returning (suffix, f1_col, f2_col).

    Derived from the column names themselves rather than hand-listed, so a
    renamed or added column can't silently fall outside the swap and corrupt
    a row by moving identity without its matching stats.
    """
    # set() for fast "does this column exist" checks below; sorted() so the
    # order we loop through columns is always the same, since sets don't
    # remember the order things were added in.
    columns = set(columns)
    pairs = []
    for col in sorted(columns):
        if not col.startswith("f1_"):
            continue
        suffix = col[len("f1_"):]
        f2_col = f"f2_{suffix}"
        if f2_col in columns:  # only pair up f1_X if a matching f2_X exists
            pairs.append((suffix, col, f2_col))
    return pairs


def randomize_fighter_sides(
    df: pd.DataFrame,
    id_col: str = "fights_url",
    seed: int = RANDOMIZE_SEED,
) -> pd.DataFrame:
    """Relabel f1_*/f2_* columns as fighter_a_*/fighter_b_*, swapped per-fight.

    Expects `df` to already have a `winner_id` column (the pre-swap winning
    fighter's id, e.g. derived from the raw `result` column) so the label can
    be computed once, before any columns move. A nullable Int64 `winner_id`
    with NA means the fight had no winner (a draw or no-contest), and
    `fighter_a_won` comes out NA for those rows rather than False - they are
    kept as fighter history but can never be training rows.
    """
    pairs = find_side_column_pairs(df.columns)
    swap = stable_swap_mask(df[id_col], seed=seed)

    out = df.copy()
    for _, f1_col, f2_col in pairs:
        a_col = f"fighter_a_{f1_col[len('f1_'):]}"
        b_col = f"fighter_b_{f1_col[len('f1_'):]}"
        out[a_col] = np.where(swap, df[f2_col], df[f1_col])
        out[b_col] = np.where(swap, df[f1_col], df[f2_col])

    out = out.drop(columns=[c for pair in pairs for c in pair[1:]])
    out["fighter_a_won"] = (out["fighter_a_id"] == out["winner_id"]).astype("boolean")
    return out
