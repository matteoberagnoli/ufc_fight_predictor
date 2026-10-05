"""Three versions of the ratio features, to check whether shrinkage helps the model.

  raw        no shrinkage (alpha = 0). Accuracy is NaN when a fighter never
             attempted something (0/0), and tiny samples give extreme values.
  zero_fill  raw, with every 0/0 accuracy set to 0 before the A - B difference.
  shrunk     the pipeline's version: every ratio pulled towards the league
             value, with alphas learned on the training years.

The model handles NaN natively, so "raw" is a genuine contender. Shrinkage was
chosen because it predicts each fighter's *next fight* better; whether it also
helps predict *who wins* is a separate question, and this is how it gets
answered - on the validation years only.
"""

import pandas as pd

from ufc_predictor.features.pipeline import build_featured_table
from ufc_predictor.features.rolling import PAIRED_STATS, ROLLING_WINDOW, no_shrinkage


def zero_fill_undefined_accuracy(featured: pd.DataFrame) -> pd.DataFrame:
    """Set accuracy to 0 wherever it is 0/0, then rebuild the differences.

    Expects raw (unshrunk) features. A 0/0 accuracy is one that is missing
    while the matching attempts-per-minute is exactly 0: the fighter has
    history, but never attempted the move. Debutants have no history at all,
    so their attempts are missing too, and they stay NaN.
    """
    out = featured.copy()
    for landed, attempted in PAIRED_STATS.items():
        for d in ("", "_absorbed"):
            for window in ("career_avg", f"last{ROLLING_WINDOW}_avg"):
                accuracy = f"{landed}{d}_accuracy_{window}"
                attempts = f"{attempted}{d}_per_min_{window}"
                for side in ("fighter_a", "fighter_b"):
                    a, r = f"{side}_prior_{accuracy}", f"{side}_prior_{attempts}"
                    out.loc[out[a].isna() & out[r].eq(0), a] = 0.0
                out[f"diff_prior_{accuracy}"] = (
                    out[f"fighter_a_prior_{accuracy}"] - out[f"fighter_b_prior_{accuracy}"]
                )
    return out


def build_variants(fights: pd.DataFrame, alphas: dict, fighters: pd.DataFrame,
                   elo_k: float | None) -> dict:
    """{"raw", "zero_fill", "shrunk"} -> a full featured table for each.

    Only the ratio-feature treatment differs between them. Both tables come
    from features.pipeline.build_featured_table, the same function that builds
    the main table, so static attributes, weight class and Elo are identical
    in all three and the comparison isolates shrinkage and nothing else.
    """
    raw = build_featured_table(fights, fighters, no_shrinkage(), elo_k=elo_k)
    return {
        "raw": raw,
        "zero_fill": zero_fill_undefined_accuracy(raw),
        "shrunk": build_featured_table(fights, fighters, alphas, elo_k=elo_k),
    }
