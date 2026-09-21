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

# Stats that come as a landed/attempted pair, as {landed: attempted}.
#
# For these we keep the *attempts* and an *accuracy* ratio rather than landed
# and attempted separately. That's a reparameterisation, not a loss - landed =
# attempts x accuracy recovers the original exactly - but it separates "how
# much they throw" from "how well it lands", which the raw pair tangles
# together. The landed averages are still computed (accuracy needs them) and
# stay in the long table; they're just not carried onto the wide fight table.
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

# stats normalised to a per-minute rate (see _rate_features) rather than
# left as a raw per-fight count.
RATE_STATS = list(PAIRED_STATS.values()) + ["knockdowns", "submissions", "reversals"]

# absorbed-side stats normalised as a fraction of fight time instead - only
# ctrl_time, since it shares units (seconds) with total_fight_time.
FRACTION_STATS = ["ctrl_time"]

# every raw per-fight column pulled into the long table
ROLLED_STATS = (
    list(PAIRED_STATS) + list(PAIRED_STATS.values()) + UNPAIRED_STATS
)

MIRRORED_STATS = ROLLED_STATS  # these get an "_absorbed" counterpart too


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
    fight_count = pd.Series(np.arange(n), index=g.index)
    wins = g["won"].astype(int).shift(1).cumsum().fillna(0)
    streak_after = _streak_after_each_fight(g["won"])
    title_fights = g["is_title"].astype(int).shift(1).cumsum().fillna(0)
    five_round_fights = g["is_five_round"].astype(int).shift(1).cumsum().fillna(0)

    new = {
        "fight_count": fight_count,
        "is_debut": fight_count == 0,
        "wins": wins,
        "losses": fight_count - wins,
        "win_rate": wins / fight_count.replace(0, np.nan),
        "streak": streak_after.shift(1).fillna(0),
        "days_since_last_fight": g["event_date"].diff().dt.days,
        "title_fights": title_fights,
        "five_round_fights": five_round_fights,
    }

    rolled = (
        MIRRORED_STATS
        + [f"{s}_absorbed" for s in MIRRORED_STATS]
        + SHARED_STATS
    )
    for stat in rolled:
        prior = g[stat].shift(1)  # <- the leakage guard: today's own row is gone
        new[f"{stat}_career_avg"] = prior.expanding().mean()
        new[f"{stat}_last{ROLLING_WINDOW}_avg"] = prior.rolling(
            ROLLING_WINDOW, min_periods=1
        ).mean()

    new.update(_accuracy_ratios(new))
    new.update(_rate_features(g, ""))
    new.update(_rate_features(g, "_absorbed"))
    return pd.concat([g, pd.DataFrame(new, index=g.index)], axis=1)


def _rate_features(g: pd.DataFrame, direction: str) -> dict:
    """Normalise volume counts by fight length, for one direction.

    `direction=""` is the fighter's own output; `direction="_absorbed"` is
    what opponents did to them. A raw count (e.g. sig_strike_atts_career_avg
    = 40) means nothing on its own - 40 strikes thrown in a typical 3-minute
    fight is very different from 40 over 15 minutes, whichever side threw
    them.

    RATE_STATS (attempt counts, knockdowns, submissions, reversals) become a
    per-minute rate. ctrl_time is handled separately as a fraction: it's
    already in the same units as total_fight_time (seconds), so dividing
    gives "share of the fight spent controlled/controlling" directly, which
    is more interpretable than a per-minute figure would be.

    The denominator is re-derived per stat rather than reusing one global
    rolled duration, and that detail matters. `expanding().mean()` silently
    skips NaNs, so if a stat is missing for some bouts (ctrl_time is absent
    for ~1.5% of fighter-appearances) its average covers fewer fights than a
    duration average taken over all of them - and dividing the two mismatched
    averages produced control-time fractions above 1.0. Masking the duration
    to exactly the bouts where the stat is present keeps numerator and
    denominator over the same fights.
    """
    out = {}
    prior_duration = g["total_fight_time"].shift(1)

    for window in ("career_avg", f"last{ROLLING_WINDOW}_avg"):
        def mean_over(series: pd.Series) -> pd.Series:
            if window == "career_avg":
                return series.expanding().mean()
            return series.rolling(ROLLING_WINDOW, min_periods=1).mean()

        for stat in RATE_STATS + FRACTION_STATS:
            prior_stat = g[f"{stat}{direction}"].shift(1)
            paired_duration = prior_duration.where(prior_stat.notna())

            numerator = mean_over(prior_stat)
            denominator = mean_over(paired_duration).replace(0, np.nan)

            if stat in FRACTION_STATS:
                out[f"{stat}{direction}_frac_{window}"] = numerator / denominator
            else:
                out[f"{stat}{direction}_per_min_{window}"] = numerator / (denominator / 60)
    return out


def _accuracy_ratios(rolled: dict) -> dict:
    """Landed/attempted ratios for both directions, from already-rolled averages.

    Computed as (average landed) / (average attempted) rather than the average
    of per-fight ratios. That's deliberate: it's the volume-weighted accuracy
    over the fighter's prior bouts, so a fight where they threw two strikes and
    landed both doesn't count as much as a fight where they threw eighty.

    Both directions are built. `*_accuracy` is the fighter's own - how well
    their offence lands. `*_absorbed_accuracy` is how often opponents land what
    they throw *at* this fighter, which is the real defensive-vulnerability
    signal: raw absorbed volume conflates it with fight length and opponent
    output, and the two turn out to be almost uncorrelated in this data.
    """
    out = {}
    for window in ("career_avg", f"last{ROLLING_WINDOW}_avg"):
        for landed, attempted in PAIRED_STATS.items():
            for suffix in ("", "_absorbed"):
                num = rolled[f"{landed}{suffix}_{window}"]
                den = rolled[f"{attempted}{suffix}_{window}"]
                out[f"{landed}{suffix}_accuracy_{window}"] = num / den.replace(0, np.nan)
    return out


def add_prior_features(history: pd.DataFrame) -> pd.DataFrame:
    """Add every prior-fight feature to a fighter-history long table."""
    return history.groupby("fighter_id", group_keys=False).apply(_add_prior_features_for_one_fighter, include_groups=False).join(history[["fighter_id"]])


def _volume_column_name(stat: str, direction: str, window: str) -> str:
    """Name of the volume column carried to the wide table for one stat.

    RATE_STATS/FRACTION_STATS are normalised by fight length in both
    directions (see _rate_features) and replace the raw per-fight count -
    a raw count means nothing without knowing how long the fight ran.
    Anything else stays a plain per-fight average.
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
