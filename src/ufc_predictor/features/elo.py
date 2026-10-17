"""Elo ratings: one number per fighter, corrected after every fight.

Two formulas do all the work:

    forecast    E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))     P(A wins), from the gap
    correction  R_A_new = R_A + K * (S_A - E_A)                S_A: 1 win, 0 loss, 0.5 draw

Fighter B moves by exactly the opposite amount. A big upset (S - E large)
moves both ratings a lot; an expected result barely moves them. Worked
example from our data, K = 64: at UFC 69 Matt Serra (1538) had a 28% chance
against Georges St-Pierre (1701), won, and both ratings moved by 46 points.

It is the only feature that knows *who* a fighter beat. Win rate counts a win
over a champion and a win over a debutant the same; Elo weights each result
by the opponent's rating.

The rules that keep it honest, each one a lesson from earlier in the project:

  * store first, then update - the rating saved on a fight's row is the one
    from *before* that fight. The Elo version of rolling.py's shift(1).
  * every fight since 1994 counts, including pre-2014 fights and draws; only
    a no-contest leaves both ratings untouched.
  * fights are processed in (event_date, fights_url) order, sorted here, not
    trusted from the input: fights.parquet is date-sorted but its order
    within a date is arbitrary, and 31 fighters fought twice in one night in
    the 1994-1999 tournaments.

K - how far one fight moves a rating - is the one real choice, and it is
chosen by how much the rating helps *the model* (modeling.tune.choose_elo_k),
not by how well Elo forecasts on its own. Tested before building this: the K
that forecasts best alone (~63) added nothing to the model; a faster rating
(K ~150-300, a memory of roughly three fights) helped in 8 of 8 model seeds.
"""

import numpy as np
import pandas as pd

from ufc_predictor.config import ERA_START, VALIDATION_START
from ufc_predictor.data.io import load_processed_json, save_processed_json
from ufc_predictor.features.shrinkage import _grid

# Everyone's first rating. Cosmetic: only gaps between ratings matter, so
# starting everyone at 1000 or 2200 would give identical features. Debutants
# get it too - UFC debutants win only ~42%, but a lower start for *every*
# newcomer just shifts the whole scale; the is_debut flags carry that effect.
ELO_START = 1500.0

# A 400-point gap means 10-to-1 odds. A convention from chess, never tuned:
# changing it only rescales the ratings, which K would then absorb.
ELO_SCALE = 400.0

# K values tried by choose_elo_k: the same ~26%-steps log grid the shrinkage
# alphas use, from very slow (10: a memory of ~50 fights) to very fast (1000).
ELO_K_GRID = _grid(10, 1000)

# What each outcome is worth to fighter A. "no_contest" is deliberately absent:
# it scores nothing and moves nothing.
SCORE_FOR_A = {"fighter_a": 1.0, "fighter_b": 0.0, "draw": 0.5}

ELO_COLUMNS = ("fighter_a_prior_elo", "fighter_b_prior_elo")


def expected_score(r_a, r_b):
    """Formula 1: the probability that A beats B, from the two ratings."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / ELO_SCALE))


def win_score(method: str, weights: dict | None) -> float:
    """What a win is worth to the winner, by how it came; the loser gets 1 minus it.

    Without `weights` every win is worth 1, the classic Elo. `weights` maps
    "unanimous" and "narrow" (split or majority decision) to a value between
    0.5 (no better than a draw) and 1 (as convincing as a finish). A finish,
    and anything else (a DQ), is always worth 1.
    """
    if weights is None:
        return 1.0
    return float(weights.get(method, 1.0)) if method in ("unanimous", "narrow") else 1.0


def elo_ratings(fights: pd.DataFrame, k: float, weights: dict | None = None) -> pd.DataFrame:
    """Both fighters' ratings going into every fight, aligned to `fights`' index.

    Needs `event_date`, `fights_url`, `fighter_a_id`, `fighter_b_id` and
    `outcome` on every row - all fights, not just the decided ones, since a
    draw is history too - and `method` when `weights` is given (see
    win_score). The input's row order does not matter.
    """
    if not fights.index.is_unique:
        raise ValueError("elo_ratings needs a unique index to put each rating back on its row")
    if fights[["fighter_a_id", "fighter_b_id"]].isna().any().any():
        raise ValueError("every fight needs both fighter ids")
    unknown = set(fights["outcome"].dropna().unique()) - set(SCORE_FOR_A) - {"no_contest"}
    if unknown or fights["outcome"].isna().any():
        raise ValueError(f"unrecognised or missing outcome(s): {sorted(map(str, unknown))}")

    ordered = fights.sort_values(["event_date", "fights_url"], kind="stable")
    rating: dict = {}
    before_a = np.empty(len(ordered))
    before_b = np.empty(len(ordered))

    methods = ordered["method"] if weights is not None else pd.Series("other", index=ordered.index)
    rows = zip(ordered["fighter_a_id"], ordered["fighter_b_id"], ordered["outcome"], methods)
    for i, (a, b, outcome, method) in enumerate(rows):
        r_a, r_b = rating.get(a, ELO_START), rating.get(b, ELO_START)
        before_a[i], before_b[i] = r_a, r_b     # <- stored before this fight changes anything
        if outcome == "no_contest":
            continue
        score_a = SCORE_FOR_A[outcome]
        if outcome != "draw":
            w = win_score(method, weights)
            score_a = w if outcome == "fighter_a" else 1.0 - w
        change = k * (score_a - expected_score(r_a, r_b))
        rating[a] = r_a + change
        rating[b] = r_b - change                # whatever A gains, B loses

    out = pd.DataFrame({ELO_COLUMNS[0]: before_a, ELO_COLUMNS[1]: before_b}, index=ordered.index)
    return out.reindex(fights.index)


def attach_elo_features(table: pd.DataFrame, k: float, weights: dict | None = None) -> pd.DataFrame:
    """Add fighter_a_prior_elo, fighter_b_prior_elo and diff_prior_elo.

    Works on the fights table or the featured table - both keep every fight.
    Running it again replaces the columns, so a table can be re-rated with a
    different K.
    """
    out = table.copy()
    ratings = elo_ratings(table, k, weights)
    for column in ELO_COLUMNS:
        out[column] = ratings[column]
    out["diff_prior_elo"] = out[ELO_COLUMNS[0]] - out[ELO_COLUMNS[1]]
    return out


def elo_forecast_log_loss(fights: pd.DataFrame, k: float,
                          start: str = ERA_START, until: str = VALIDATION_START) -> float:
    """How well Elo forecasts on its own: log loss of Formula 1, decided fights in [start, until).

    Every forecast uses only earlier fights, so this is an honest score
    without touching later years. For reference only: K is chosen by what
    helps the model, which is a different question (see the module docstring).
    """
    ratings = elo_ratings(fights, k)
    period = ((fights["event_date"] >= pd.Timestamp(start))
              & (fights["event_date"] < pd.Timestamp(until))
              & fights["fighter_a_won"].notna())
    p = expected_score(ratings.loc[period, ELO_COLUMNS[0]], ratings.loc[period, ELO_COLUMNS[1]])
    p = np.clip(p.to_numpy(), 1e-6, 1 - 1e-6)
    y = fights.loc[period, "fighter_a_won"].to_numpy(dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# The Elo settings searched by 05_elo.ipynb (modeling.tune.choose_elo_settings).
# K from slow to fast; the two win weights from 1 (as convincing as a finish) down
# to 0.5 (no better than a draw), a narrow win never counting more than a unanimous one.
ELO_SEARCH_KS = (39.81, 63.1, 100.0, 125.9, 158.5, 199.5, 251.2, 316.2, 398.1)
ELO_WEIGHT_PAIRS = tuple(
    (u, n) for u in (1.0, 0.9, 0.8, 0.7, 0.6) for n in (1.0, 0.9, 0.75, 0.65, 0.55, 0.5) if n <= u
)


def save_elo_params(k: float, curve: list, seeds, until: str = VALIDATION_START,
                    start: str = ERA_START, processed_dir=None, note: str | None = None,
                    weights: dict | None = None) -> None:
    """Save K with how it was chosen, including the whole K -> score curve.

    The curve is kept because the choice sits on a broad plateau: seeing that
    neighbouring K values score within ~0.001 of each other is what tells you
    the exact K barely matters.

    `note` records anything the curve alone would get wrong - e.g. on Day 12
    the saved K is not the curve's minimum, because a check with the final
    model's own settings overruled the search (05_elo.ipynb, section 4).
    `weights` are the win weights (see win_score); None is the classic Elo.
    The curve is a list of (K, score) pairs or of dicts (K, weights, score).
    """
    chosen_by = {
        "method": "model CV log loss, expanding window inside the training years",
        "fitted_on": {"from": start, "until": until},
        "model_seeds": list(seeds),
    }
    if note is not None:
        chosen_by["note"] = note
    save_processed_json(
        {
            "k": float(k),
            "start_rating": ELO_START,
            "scale": ELO_SCALE,
            "weights": None if weights is None else {key: float(v) for key, v in weights.items()},
            "chosen_by": chosen_by,
            "curve": [{key: float(v) for key, v in point.items()} if isinstance(point, dict)
                      else [float(point[0]), float(point[1])] for point in curve],
        },
        "elo_params",
        processed_dir=processed_dir,
    )


def load_elo_params(processed_dir=None) -> float:
    return load_processed_json("elo_params", processed_dir=processed_dir)["k"]


def load_elo_weights(processed_dir=None) -> dict | None:
    """The saved win weights, or None for the classic Elo (also for files saved before weights existed)."""
    return load_processed_json("elo_params", processed_dir=processed_dir).get("weights")
