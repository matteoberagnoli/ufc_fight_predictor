"""Pre-fight rolling features, computed only from each fighter's history
strictly before the fight being predicted.

This is what closes leakage traps #2 and #3 from ROADMAP.md: the per-fight
stat columns in fights.parquet (strikes, takedowns, control time...) are
outcomes of the fight they're attached to, so they can never be used as
features for that same fight. They're only safe as *raw material*, rolled up
over a fighter's earlier fights.

The approach:
  1. Reshape fights.parquet from one-row-per-fight into one-row-per-fighter-
     appearance ("long format") - each fight contributes two rows, one for
     fighter_a and one for fighter_b.
  2. Sort each fighter's rows chronologically.
  3. Shift every stat down by one fight *before* computing any average or
     count. This is the actual leakage guard: it removes the current fight's
     own row from view before anything is aggregated, so a fighter's Nth
     fight can only ever see fights 1..N-1.
  4. Merge the resulting prior-fight features back onto the wide fight table
     for both sides, and take fighter_a-minus-fighter_b differences.
"""

import numpy as np
import pandas as pd

ROLLING_WINDOW = 3

# raw per-fight stat columns rolled into prior-fight averages. Each is taken
# both as the fighter's own output ("landed") and what their opponent did to
# them in the same fight ("absorbed") - offense and defense are different
# skills, worth keeping separate rather than averaging away.
ROLLED_STATS = [
    "sig_strikes",
    "sig_strike_atts",
    "takedowns",
    "takedown_atts",
    "knockdowns",
    "submissions",
    "ctrl_time",
]


def _side_history(fights: pd.DataFrame, side: str) -> pd.DataFrame:
    """One row per fighter appearance, from `side`'s point of view."""
    other = "fighter_b" if side == "fighter_a" else "fighter_a"

    out = pd.DataFrame(
        {
            "fighter_id": fights[f"{side}_id"],
            "opponent_id": fights[f"{other}_id"],
            "fights_url": fights["fights_url"],
            "event_date": fights["event_date"],
            "won": fights["fighter_a_won"] if side == "fighter_a" else ~fights["fighter_a_won"],
        }
    )
    for stat in ROLLED_STATS:
        out[stat] = fights[f"{side}_{stat}"]
        out[f"{stat}_absorbed"] = fights[f"{other}_{stat}"]
    return out


def build_fighter_history(fights: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (fighter, fight), every fighter appearance.

    Sorted chronologically within each fighter so later steps can rely on
    row order alone (no re-sorting needed once this is built).
    """
    long = pd.concat(
        [_side_history(fights, "fighter_a"), _side_history(fights, "fighter_b")],
        ignore_index=True,
    )
    return long.sort_values(
        ["fighter_id", "event_date", "fights_url"]
    ).reset_index(drop=True)


def _streak_after_each_fight(won: pd.Series) -> pd.Series:
    """Running win/loss streak *after* each fight: +2 = won last 2, -1 = lost last 1."""
    streak = []
    current = 0
    for did_win in won:
        if did_win:
            current = current + 1 if current >= 0 else 1
        else:
            current = current - 1 if current <= 0 else -1
        streak.append(current)
    return pd.Series(streak, index=won.index)


def _add_prior_features_for_one_fighter(g: pd.DataFrame) -> pd.DataFrame:
    """g is one fighter's rows, already in chronological order."""
    g = g.copy()
    n = len(g)

    # names here deliberately don't repeat "prior" - the {side}_prior_ prefix
    # added later in attach_prior_features() already says that for every
    # column in this list, so e.g. this becomes fighter_a_prior_win_rate,
    # not fighter_a_prior_prior_win_rate.
    g["fight_count"] = np.arange(n)
    g["is_debut"] = g["fight_count"] == 0

    wins_so_far = g["won"].astype(int).shift(1).cumsum()
    g["wins"] = wins_so_far.fillna(0)
    g["losses"] = g["fight_count"] - g["wins"]
    g["win_rate"] = g["wins"] / g["fight_count"].replace(0, np.nan)

    streak_after = _streak_after_each_fight(g["won"])
    g["streak"] = streak_after.shift(1).fillna(0)

    g["days_since_last_fight"] = g["event_date"].diff().dt.days

    for stat in ROLLED_STATS + [f"{s}_absorbed" for s in ROLLED_STATS]:
        prior = g[stat].shift(1)  # <- the leakage guard: today's own row is gone
        g[f"{stat}_career_avg"] = prior.expanding().mean()
        g[f"{stat}_last{ROLLING_WINDOW}_avg"] = prior.rolling(
            ROLLING_WINDOW, min_periods=1
        ).mean()

    return g


def add_prior_features(history: pd.DataFrame) -> pd.DataFrame:
    """Add every prior-fight feature to a fighter-history long table."""
    return history.groupby("fighter_id", group_keys=False).apply(_add_prior_features_for_one_fighter, include_groups=False).join(history[["fighter_id"]])


PRIOR_FEATURE_COLUMNS = (
    ["fight_count", "is_debut", "wins", "losses",
     "win_rate", "streak", "days_since_last_fight"]
    + [f"{s}_career_avg" for s in ROLLED_STATS]
    + [f"{s}_absorbed_career_avg" for s in ROLLED_STATS]
    + [f"{s}_last{ROLLING_WINDOW}_avg" for s in ROLLED_STATS]
    + [f"{s}_absorbed_last{ROLLING_WINDOW}_avg" for s in ROLLED_STATS]
)


def attach_prior_features(fights: pd.DataFrame) -> pd.DataFrame:
    """Add fighter_a_prior_*/fighter_b_prior_* columns and their diffs to fights."""
    history = build_fighter_history(fights)
    history = add_prior_features(history)

    out = fights.copy()
    for side in ("fighter_a", "fighter_b"):
        cols = ["fighter_id", "fights_url"] + PRIOR_FEATURE_COLUMNS
        side_features = history.loc[history["fighter_id"].notna(), cols].rename(
            columns={c: f"{side}_prior_{c}" for c in PRIOR_FEATURE_COLUMNS}
            | {"fighter_id": f"{side}_id"}
        )
        out = out.merge(side_features, on=[f"{side}_id", "fights_url"], how="left", validate="one_to_one")

    diff_cols = {}
    for col in PRIOR_FEATURE_COLUMNS:
        a_col, b_col = f"fighter_a_prior_{col}", f"fighter_b_prior_{col}"
        if pd.api.types.is_bool_dtype(out[a_col]):
            continue
        diff_cols[f"diff_prior_{col}"] = out[a_col] - out[b_col]

    return pd.concat([out, pd.DataFrame(diff_cols, index=out.index)], axis=1)
