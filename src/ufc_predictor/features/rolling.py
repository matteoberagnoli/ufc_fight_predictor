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
  3. Only ever look at earlier rows. This is the actual leakage guard, and it
     appears in two places: the record features shift every column down by
     one fight before counting, and the ratio features add up running totals
     that stop at the previous fight (shrinkage.prior_totals). Either way a
     fighter's Nth fight can only ever see fights 1..N-1.
  4. Every feature that is a total divided by an amount of evidence - accuracy,
     per-minute rates, control share, win rate, average fight time - is shrunk
     towards the league value, so a tiny sample can't produce an extreme
     number. See features/shrinkage.py for why and how.
  5. Merge the resulting prior-fight features back onto the wide fight table
     for both sides, and take fighter_a-minus-fighter_b differences.

Draws and no-contests are part of this history, even though they can never
be modelling rows: the strikes landed and the cage time elapsed either way.
Only `wins`, `losses`, `win_rate` and `streak` treat them specially - see
`_fighter_score` and `_streak_after_each_fight`.
"""

import numpy as np
import pandas as pd

from ufc_predictor.features import shrinkage
from ufc_predictor.features.shrinkage import RatioFeature

ROLLING_WINDOW = 3

# Stats that come as a landed/attempted pair, as {landed: attempted}.
#
# For these we keep the *attempts* and an *accuracy* ratio rather than landed
# and attempted separately. That's a reparameterisation, not a loss - landed =
# attempts x accuracy recovers the original exactly - but it separates "how
# much they throw" from "how well it lands", which the raw pair tangles
# together. Landed totals are still added up (accuracy needs them); they're
# just not carried onto the wide fight table.
#
# Note the location stats are exact decompositions of sig_strikes, verified on
# the raw data: sig_strikes == head + body + leg == distance + clinch + ground,
# on 100% of rows. They're two ways of slicing the same total (by target, and
# by position), so expect exact collinearity with the sig_strikes columns.
# The *mix* is real signal - a leg-kicker and a head-hunter fight differently -
# but it matters for linear models, which can't sit on a dependent basis.
PAIRED_STATS = {
    "sig_strikes": "sig_strike_atts",
    "head_strikes": "head_strike_atts",
    "body_strikes": "body_strike_atts",
    "leg_strikes": "leg_strike_atts",
    "dist_strikes": "dist_strike_atts",
    "clinchs": "clinch_atts",
    "grounds": "ground_atts",
    "takedowns": "takedown_atts",
    "tot_strikes": "tot_strike_atts",
}

# Stats with no attempted counterpart - rolled as-is, both as the fighter's
# own output and as what opponents did to them ("absorbed").
UNPAIRED_STATS = ["knockdowns", "submissions", "ctrl_time", "reversals"]

# Identical for both fighters (it's the bout's duration, verified equal on 100%
# of rows), so an "absorbed" mirror would be pure duplication. Rolled once per
# fighter: their average fight length, a fast-finisher / goes-the-distance
# signal in its own right, and the denominator any per-minute rate needs.
SHARED_STATS = ["total_fight_time"]

# stats turned into a per-minute rate (see ratio_features) rather than left as
# a raw per-fight count - a count means nothing without the fight's length.
RATE_STATS = list(PAIRED_STATS.values()) + ["knockdowns", "submissions", "reversals"]

# turned into a share of fight time instead of a per-minute rate - only
# ctrl_time, since it shares units (seconds) with total_fight_time.
FRACTION_STATS = ["ctrl_time"]

# every raw per-fight column pulled into the long table
ROLLED_STATS = (
    list(PAIRED_STATS) + list(PAIRED_STATS.values()) + UNPAIRED_STATS
)

MIRRORED_STATS = ROLLED_STATS  # these get an "_absorbed" counterpart too


def _fighter_score(fights: pd.DataFrame, side: str) -> pd.Series:
    """How this fight went for `side`: 1.0 won, 0.0 lost, 0.5 drew, NaN no-contest.

    One float column carries all four outcomes, which is what lets the same
    value serve two jobs. The record counts read it as categories (wins are
    the 1.0s, losses the 0.0s). The shrunk `win_rate` feature reads it as a
    number to add up, so a draw contributes half a win - and a no-contest,
    being NaN, drops out of both the top and the bottom of the ratio, since
    `shrinkage._ingredients` masks the evidence wherever the total is
    missing. That is "a draw is half a win, a no-contest never happened",
    expressed once.
    """
    won = fights["fighter_a_won"]
    mine = won if side == "fighter_a" else ~won
    score = mine.astype("float64")          # True -> 1.0, False -> 0.0, NA -> NaN
    score[fights["outcome"] == "draw"] = 0.5
    return score


def _side_history(fights: pd.DataFrame, side: str) -> pd.DataFrame:
    """One row per fighter appearance, from `side`'s point of view."""
    other = "fighter_b" if side == "fighter_a" else "fighter_a"

    out = pd.DataFrame(
        {
            "fighter_id": fights[f"{side}_id"],
            "opponent_id": fights[f"{other}_id"],
            "fights_url": fights["fights_url"],
            "event_date": fights["event_date"],
            "won": _fighter_score(fights, side),
        }
    )
    for stat in MIRRORED_STATS:
        out[stat] = fights[f"{side}_{stat}"]
        out[f"{stat}_absorbed"] = fights[f"{other}_{stat}"]
    for stat in SHARED_STATS:
        out[stat] = fights[f"{side}_{stat}"]

    out["is_title"] = fights["weight_class"].str.contains("Title", case=False, na=False)
    # A 3-round fight cannot exceed 900s (3 x 5 min), so anything longer
    # proves 5-round scheduling even for a non-title bout - non-title main
    # events have been eligible for 5 rounds since ~2012, and 162 of the
    # 7,697 fights are exactly this case. This only catches fights that ran
    # past the 3-round ceiling: a 5-round-scheduled fight that ends by a
    # first-round finish looks identical to a 3-round fight ending the same
    # way, so title-status-plus-duration is a lower bound on true 5-round
    # fights, not an exact count.
    out["is_five_round"] = out["is_title"] | (out["total_fight_time"] > 900)

    # the "evidence" columns the shrunk ratio features divide by (see ratio_features)
    out["minutes"] = out["total_fight_time"] / 60
    out["fights"] = 1.0
    out["ctrl_minutes"] = out["ctrl_time"] / 60
    out["ctrl_minutes_absorbed"] = out["ctrl_time_absorbed"] / 60
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


def _streak_after_each_fight(score: pd.Series) -> pd.Series:
    """Running win/loss streak *after* each fight: +2 = won last 2, -1 = lost last 1.

    A draw resets the streak to 0 - the fighter genuinely failed to win, so
    the run is over. A no-contest leaves it untouched: it is usually an
    overturned result or an accidental foul, and wiping out real momentum
    for it would say something the fight never showed.
    """
    streak = []
    current = 0
    for value in score:
        if pd.isna(value):                      # no-contest: carry on unchanged
            pass
        elif value == 1.0:
            current = current + 1 if current >= 0 else 1
        elif value == 0.0:
            current = current - 1 if current <= 0 else -1
        else:                                   # draw
            current = 0
        streak.append(current)
    return pd.Series(streak, index=score.index)


# counts and facts about a fighter's record - the evidence itself, not an
# estimate made from it, so there is nothing to shrink
RECORD_COLUMNS = [
    "fight_count", "is_debut", "wins", "losses", "streak",
    "days_since_last_fight", "title_fights", "five_round_fights",
]


def _record_features_for_one_fighter(g: pd.DataFrame) -> pd.DataFrame:
    """g is one fighter's rows, already in chronological order."""
    n = len(g)

    # names here deliberately don't repeat "prior" - the {side}_prior_ prefix
    # added later in attach_prior_features() already says that for every
    # column in this list, so e.g. this becomes fighter_a_prior_wins,
    # not fighter_a_prior_prior_wins.
    # every prior appearance counts as experience, including draws and
    # no-contests - the fighter was in the cage
    fight_count = pd.Series(np.arange(n), index=g.index)
    prior = lambda flag: flag.astype(int).shift(1).cumsum().fillna(0)  # noqa: E731
    wins = prior(g["won"].eq(1.0))      # <- shift(1): today's result isn't known yet
    losses = prior(g["won"].eq(0.0))    # counted directly: fight_count also holds draws/NCs
    streak_after = _streak_after_each_fight(g["won"])
    title_fights = prior(g["is_title"])
    five_round_fights = prior(g["is_five_round"])

    return pd.DataFrame(
        {
            "fight_count": fight_count,
            "is_debut": fight_count == 0,
            "wins": wins,
            "losses": losses,
            "streak": streak_after.shift(1).fillna(0),
            "days_since_last_fight": g["event_date"].diff().dt.days,
            "title_fights": title_fights,
            "five_round_fights": five_round_fights,
        },
        index=g.index,
    )


def _volume_column_name(stat: str, direction: str, window: str) -> str:
    """Name of the volume column carried to the wide table for one stat.

    RATE_STATS/FRACTION_STATS are normalised by fight length in both
    directions and replace the raw per-fight count - a raw count means
    nothing without knowing how long the fight ran. Anything else stays a
    plain per-fight average.
    """
    if stat in RATE_STATS:
        return f"{stat}{direction}_per_min_{window}"
    if stat in FRACTION_STATS:
        return f"{stat}{direction}_frac_{window}"
    return f"{stat}{direction}_{window}"


def _feature_names_for_window(window: str) -> list[str]:
    """Columns carried onto the wide table for one rolling window.

    Volume (normalised where applicable - see _volume_column_name) and
    accuracy for the paired stats (not landed - see PAIRED_STATS), the
    unpaired stats the same way, and the shared ones without an absorbed
    mirror.
    """
    names = []
    for attempted in PAIRED_STATS.values():
        names.append(_volume_column_name(attempted, "", window))
        names.append(_volume_column_name(attempted, "_absorbed", window))
    for landed in PAIRED_STATS:
        names += [f"{landed}_accuracy_{window}", f"{landed}_absorbed_accuracy_{window}"]
    for stat in UNPAIRED_STATS:
        names.append(_volume_column_name(stat, "", window))
        names.append(_volume_column_name(stat, "_absorbed", window))
    names += [f"{stat}_{window}" for stat in SHARED_STATS]
    return names


PRIOR_FEATURE_COLUMNS = (
    ["fight_count", "is_debut", "wins", "losses",
     "win_rate", "streak", "days_since_last_fight",
     "title_fights", "five_round_fights"]
    + _feature_names_for_window("career_avg")
    + _feature_names_for_window(f"last{ROLLING_WINDOW}_avg")
)


def ratio_features() -> list[RatioFeature]:
    """Every feature that is a total divided by an amount of evidence - all shrunk.

    One spec per wide-table column, so each column gets its own alpha. The
    absorbed direction uses the league value of the fighter's own direction:
    every strike one fighter lands is one the other absorbs, so league-wide
    the two are the same number.
    """
    specs = []
    for window, last_n in (("career_avg", None), (f"last{ROLLING_WINDOW}_avg", ROLLING_WINDOW)):
        for landed, attempted in PAIRED_STATS.items():
            for d in ("", "_absorbed"):
                specs.append(RatioFeature(
                    f"{landed}{d}_accuracy_{window}", f"{landed}{d}", f"{attempted}{d}",
                    landed, attempted, "attempts", last_n,
                ))
        for stat in RATE_STATS:
            for d in ("", "_absorbed"):
                specs.append(RatioFeature(
                    _volume_column_name(stat, d, window), f"{stat}{d}", "minutes",
                    stat, "minutes", "minutes", last_n,
                ))
        # control time is the one FRACTION_STAT: minutes in control / minutes fought
        for d in ("", "_absorbed"):
            specs.append(RatioFeature(
                _volume_column_name("ctrl_time", d, window), f"ctrl_minutes{d}", "minutes",
                "ctrl_minutes", "minutes", "minutes", last_n,
            ))
        specs.append(RatioFeature(
            f"total_fight_time_{window}", "total_fight_time", "fights",
            "total_fight_time", "fights", "fights", last_n,
        ))
    # a win depends on the opponent, so its alpha is chosen by who wins, not by
    # the fighter alone (see shrinkage.choose_alphas)
    specs.append(RatioFeature(
        "win_rate", "won", "fights", "won", "fights", "fights", judged_by="who_wins",
    ))
    return specs


RATIO_FEATURES = ratio_features()


def no_shrinkage() -> dict[str, float]:
    """alpha = 0 for every ratio feature: the raw ratios, with 0/0 left as NaN."""
    return {spec.name: 0.0 for spec in RATIO_FEATURES}


def add_prior_features(history: pd.DataFrame, alphas: dict[str, float]) -> pd.DataFrame:
    """Add every prior-fight feature to a fighter-history long table."""
    record = history.groupby("fighter_id", group_keys=False).apply(
        _record_features_for_one_fighter, include_groups=False
    )
    ratios = shrinkage.shrunk_features(history, RATIO_FEATURES, alphas)
    return pd.concat([history, record, ratios], axis=1)


def fit_alphas(fights: pd.DataFrame, *, until: str) -> dict[str, float]:
    """Learn one shrinkage alpha per ratio feature from fights before `until`.

    Pass the start of the period the model must not see: VALIDATION_START
    while choosing features, TEST_START for the final model. Fights before
    ERA_START still count as history but aren't scored.
    """
    return shrinkage.choose_alphas(
        build_fighter_history(fights), RATIO_FEATURES, until, fights=fights
    )


def attach_prior_features(fights: pd.DataFrame, alphas: dict[str, float]) -> pd.DataFrame:
    """Add fighter_a_prior_*/fighter_b_prior_* columns and their diffs to fights.

    `alphas` says how strongly each ratio feature is shrunk - from fit_alphas,
    or shrinkage.load_alphas for a saved set; no_shrinkage() gives raw ratios.
    """
    history = build_fighter_history(fights)
    history = add_prior_features(history, alphas)

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
