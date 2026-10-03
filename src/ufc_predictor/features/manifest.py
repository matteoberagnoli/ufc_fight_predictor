"""Declares which columns may become model features, and which never may.

`fights_featured.parquet` carries 366 columns and they are not interchangeable.
46 of them are the raw per-fight statistics of the bout being predicted, and
they sit one word away from their safe counterparts:

    fighter_a_sig_strikes                          <- the answer sheet
    fighter_a_prior_sig_strikes_accuracy_career_avg <- a feature

So `X = df.drop(columns=[...])` is not a safe way to build a design matrix.
Everything here exists so that `X` is built from an explicit allow-list
instead, with two guards that fail loudly rather than silently:

  * leakage     - a raw post-fight column can never be selected
  * completeness - every column in the table must be classified, so adding a
                   column without classifying it breaks the tests rather than
                   letting it drift into or out of `X` unnoticed

Note the two reasons a column is excluded are not the same thing. The 46 raw
post-fight columns are a *safety* exclusion. The ~206 per-side columns are a
*modelling* one - they are the fighter_a_*/fighter_b_* halves each `diff_*` is
computed from, so carrying both the difference and its components is redundant,
and per-side columns break the symmetry the diffs were built to give. Those are
perfectly safe, and the choice is reversible by editing a group.

Nothing is dropped from the parquet: this module only governs what reaches a
model. The raw columns stay because they are what you check a row against by
hand when a prediction looks wrong.
"""

import pandas as pd

from ufc_predictor.features.rolling import MIRRORED_STATS, SHARED_STATS

LABEL_COLUMN = "fighter_a_won"

# `outcome` says the same thing as the label and more ("draw" / "no_contest"
# as well as which corner won), so it is the answer sheet too and must never
# reach `X`. It is kept on the table because it is what `modeling.split`
# filters on and what makes a draw readable when checking a row by hand.
LABEL_COLUMNS = (LABEL_COLUMN, "outcome")

METADATA_COLUMNS = [
    "event_date",
    "event_url",
    "fights_url",
    "weight_class",
    "fighter_a_id",
    "fighter_b_id",
    "fighter_a_name",
    "fighter_b_name",
]

# Statistic families, keyed by stem - the underlying stat name with all
# prefixes and suffixes stripped off. Deliberately keyed on stems rather than
# full column names: there are ~22 stems and they only change when the raw
# data schema does, whereas column names multiply with every new window or
# ratio (37 -> 102 diff features in a single day's work). A new feature then
# lands in the right group automatically, and one that cannot be classified
# trips the completeness guard instead of silently vanishing from `X`.
STRIKING_STEMS = {
    "sig_strikes", "sig_strike_atts",
    "head_strikes", "head_strike_atts",
    "body_strikes", "body_strike_atts",
    "leg_strikes", "leg_strike_atts",
    "dist_strikes", "dist_strike_atts",
    "clinchs", "clinch_atts",
    "grounds", "ground_atts",
    "tot_strikes", "tot_strike_atts",
}
GRAPPLING_STEMS = {"takedowns", "takedown_atts", "submissions", "reversals", "ctrl_time"}
DAMAGE_STEMS = {"knockdowns"}
PACE_STEMS = {"total_fight_time"}

RECORD_STEMS = {
    "fight_count", "wins", "losses", "win_rate", "streak",
    "days_since_last_fight", "title_fights", "five_round_fights",
}

# order matters: longer suffixes first, so "_accuracy_career_avg" is stripped
# before the bare "_career_avg" can match part of it
_SUFFIXES = (
    "_accuracy_career_avg", "_accuracy_last3_avg",
    "_per_min_career_avg", "_per_min_last3_avg",
    "_frac_career_avg", "_frac_last3_avg",
    "_career_avg", "_last3_avg",
)

_DIFF_PRIOR = "diff_prior_"
_DIFF_STATIC = "diff_static_"


def _decompose(column: str) -> tuple[str, bool]:
    """A `diff_prior_*` column -> (stem, was it the absorbed direction)."""
    body = column[len(_DIFF_PRIOR):]
    absorbed = "_absorbed" in body
    stem = body.split("_absorbed")[0] if absorbed else body
    for suffix in _SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)], absorbed
    return stem, absorbed


def _group_of(column: str) -> str | None:
    """Which feature group a column belongs to, or None if it is not a feature."""
    if column in ("fighter_a_prior_is_debut", "fighter_b_prior_is_debut"):
        return "cold_start"
    if column == "same_stance" or column.startswith(_DIFF_STATIC):
        return "physical"
    if not column.startswith(_DIFF_PRIOR):
        return None

    stem, absorbed = _decompose(column)
    if stem in RECORD_STEMS:
        return "record"
    if stem in PACE_STEMS:
        return "pace"
    # the absorbed direction is defence whatever the underlying stat is -
    # being hit, taken down or controlled are all the same kind of signal
    if absorbed:
        return "defence"
    if stem in STRIKING_STEMS:
        return "striking"
    if stem in GRAPPLING_STEMS:
        return "grappling"
    if stem in DAMAGE_STEMS:
        return "damage"
    return None


GROUP_NAMES = (
    "record", "physical", "cold_start", "pace",
    "striking", "grappling", "damage", "defence",
)


def feature_groups(fights: pd.DataFrame) -> dict[str, list[str]]:
    """Map each group name to the columns of `fights` belonging to it."""
    groups: dict[str, list[str]] = {name: [] for name in GROUP_NAMES}
    for column in fights.columns:
        group = _group_of(column)
        if group is not None:
            groups[group].append(column)
    return {name: sorted(cols) for name, cols in groups.items()}


def leaky_columns(fights: pd.DataFrame) -> list[str]:
    """Raw per-fight stats of the bout being predicted - never safe as features.

    Derived from rolling.py's own stat lists rather than restated here, so the
    two cannot drift apart.
    """
    raw_stems = set(MIRRORED_STATS) | set(SHARED_STATS)
    return sorted(
        c for c in fights.columns
        for side in ("fighter_a", "fighter_b")
        if c.startswith(f"{side}_") and c[len(side) + 1:] in raw_stems
    )


def feature_columns(fights: pd.DataFrame, groups=None) -> list[str]:
    """The allow-list that becomes `X`. Pass `groups` to select a subset."""
    available = feature_groups(fights)
    selected = list(GROUP_NAMES) if groups is None else list(groups)

    unknown = set(selected) - set(GROUP_NAMES)
    if unknown:
        raise ValueError(f"unknown feature group(s): {sorted(unknown)}")

    columns = [c for name in selected for c in available[name]]

    leaks = sorted(set(columns) & set(leaky_columns(fights)))
    if leaks:
        raise ValueError(
            f"{len(leaks)} raw post-fight column(s) reached the feature list: {leaks[:5]}"
        )
    return sorted(columns)


def unclassified_columns(fights: pd.DataFrame) -> list[str]:
    """Columns belonging to no declared bucket.

    The completeness guard. Should always be empty: a non-empty result means a
    column was added without deciding whether it is a feature, metadata, the
    label, or unsafe - and silence there is how a column slips into or out of
    `X` unnoticed.
    """
    classified = set(feature_columns(fights))
    classified |= set(leaky_columns(fights))
    classified |= set(METADATA_COLUMNS)
    classified |= set(LABEL_COLUMNS)
    # the per-side halves the diffs are built from: safe, deliberately unused
    classified |= {
        c for c in fights.columns
        if c.startswith(("fighter_a_prior_", "fighter_b_prior_"))
    }
    classified |= {
        f"fighter_{side}_{attr}"
        for side in ("a", "b")
        for attr in ("age_during", "height_cm", "reach_cm", "stance", "ape_index_cm")
    }
    return sorted(set(fights.columns) - classified)


def make_xy(fights: pd.DataFrame, groups=None) -> tuple[pd.DataFrame, pd.Series]:
    """Split a featured fight table into a design matrix and the label.

    Every row must have a winner. Draws and no-contests have no answer to
    learn, so passing the full featured table is a mistake - run it through
    `modeling.split.drop_undecided` (or `chronological_split`, which does it
    for you) first.
    """
    undecided = fights[LABEL_COLUMN].isna().sum()
    if undecided:
        raise ValueError(
            f"{undecided} row(s) have no winner (draws/no-contests). "
            "Use modeling.split.drop_undecided or chronological_split first."
        )
    columns = feature_columns(fights, groups=groups)
    return fights[columns].copy(), fights[LABEL_COLUMN].astype(bool)


def nan_report(fights: pd.DataFrame) -> pd.DataFrame:
    """Per-group column count and missingness, for choosing what to train on.

    `rows_with_nan` is the share of fights where *at least one* column in the
    group is missing - the figure that matters, since a single NaN is enough to
    drop a row from a model that cannot handle them.
    """
    groups = feature_groups(fights)
    rows = [
        {
            "group": name,
            "columns": len(cols),
            "rows_with_nan": fights[cols].isna().any(axis=1).mean() if cols else 0.0,
        }
        for name, cols in groups.items()
    ]
    report = pd.DataFrame(rows).sort_values("rows_with_nan").reset_index(drop=True)

    everything = feature_columns(fights)
    total = pd.DataFrame([{
        "group": "ALL",
        "columns": len(everything),
        "rows_with_nan": fights[everything].isna().any(axis=1).mean(),
    }])
    return pd.concat([report, total], ignore_index=True)
