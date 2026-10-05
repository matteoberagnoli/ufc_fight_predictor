"""Build the clean, de-biased fight-level dataset from raw source tables.

This is the single place that owns the transform raw CSVs -> one row per
fight, event date attached, fighter sides de-biased. Both the exploratory
notebook and any future incremental refresh (new fights appended to
data/raw) call `build_fight_dataset` rather than reimplementing this logic.

**Every fight is kept, including draws and no-contests.** They can't be
training rows - there is no winner to predict - but they are still part of
each fighter's history: the strikes landed, the cage time elapsed, the
fighter wasn't resting. Dropping them here would erase 143 fights from
2,495 fighters' records and leave 20.6% of modelling-era fights built on an
incomplete history. Choosing which fights become modelling rows belongs in
`modeling/split.py` (`drop_undecided`), downstream of the features, for the
same reason the 2014 era cut lives there.
"""

import numpy as np
import pandas as pd

from ufc_predictor.config import RANDOMIZE_SEED
from ufc_predictor.data.io import load_events, load_fight_stats
from ufc_predictor.data.randomize import randomize_fighter_sides

# raw `result` values that are not a winner's id, mapped to the outcome they mean
UNDECIDED_RESULTS = {"nc": "no_contest", "d": "draw"}

OUTCOMES = ("fighter_a", "fighter_b", "draw", "no_contest")


def attach_event_dates(fights: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    #merge fights with events to get event date and drop the link.
    merged = fights.merge(
        events[["url_link", "event_date"]],
        left_on="event_url",
        right_on="url_link",
        how="left",
        validate="many_to_one",
    )
    if merged["event_date"].isna().any():
        missing = merged.loc[merged["event_date"].isna(), "event_url"].nunique()
        raise ValueError(f"{missing} fight(s) reference an event_url missing from events.csv")
    return merged.drop(columns=["url_link"])


def split_result_column(fights: pd.DataFrame) -> pd.DataFrame:
    """Split the raw `result` column into a winner id and an undecided kind.

    Despite the name, `result` holds the winning fighter's id as a string -
    1,747 distinct values, one per fighter who has ever won - except for the
    143 fights that had no winner, where it is "d" or "nc" instead. So this
    is a rename plus a cast plus a special case, not a computation.

    `winner_id` is a nullable Int64: a plain int64 column has no way to say
    "no winner". `undecided_kind` is "draw" or "no_contest" for those fights
    and NA for the rest; it is what lets the record features treat the two
    differently further down.
    """
    out = fights.rename(columns={"result": "winner_id"}, errors="raise")

    # A missing result has to be caught here, explicitly. `astype(str)` below
    # turns NaN into the string "nan", which `pd.to_numeric` then parses
    # happily back into NaN - so the errors="raise" guard would never fire,
    # and the fight would end up with no winner id, no undecided kind, and an
    # `outcome` of "fighter_b" from add_outcome_column's default: a row whose
    # two label columns contradict each other, silently.
    missing = out["winner_id"].isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} fight(s) have no `result` value. Every fight must "
            "name a winner or be marked a draw/no-contest."
        )

    raw = out["winner_id"].astype(str)
    undecided = raw.isin(UNDECIDED_RESULTS)

    out["winner_id"] = pd.to_numeric(raw.where(~undecided), errors="raise").astype("Int64")
    out["undecided_kind"] = raw.map(UNDECIDED_RESULTS).astype("string")
    return out


def add_outcome_column(fights: pd.DataFrame) -> pd.DataFrame:
    """One column holding the complete result, side-relative.

    Must run after `randomize_fighter_sides`, since "fighter_a" only means
    something once the coin flip has decided who that is. `fighter_a_won`
    is the model's label and says True/False/NA; `outcome` additionally
    separates a draw from a no-contest, which `win_rate` and `streak` treat
    differently (see features/rolling.py).
    """
    out = fights.copy()
    kind, won = out["undecided_kind"], out["fighter_a_won"]
    out["outcome"] = pd.Series(
        np.select(
            [kind.notna(), won.fillna(False).to_numpy(dtype=bool)],
            [kind.astype(object), "fighter_a"],
            default="fighter_b",
        ),
        index=out.index,
        dtype="string",
    )
    return out


def drop_label_source_columns(fights: pd.DataFrame) -> pd.DataFrame:
    """Drop the raw label ingredients once `outcome`/`fighter_a_won` exist.

    Both are now redundant: the winner is recoverable at any time as
    `fighter_a_id if fighter_a_won else fighter_b_id`, and `undecided_kind`
    is `outcome` for the fights that had no winner. Keeping them around
    risks a future `X = fights_featured.drop(columns=[...])` accidentally
    leaving the label in the features.
    """
    return fights.drop(columns=["winner_id", "undecided_kind"])


def build_fight_dataset(
    raw_dir=None,
    seed: int = RANDOMIZE_SEED,
) -> pd.DataFrame:
    """Raw CSVs -> one row per fight, chronological, sides de-biased.

    Draws and no-contests are kept, with `fighter_a_won` as NA. Use
    `modeling.split.drop_undecided` to get modelling rows.
    """
    kwargs = {} if raw_dir is None else {"raw_dir": raw_dir}
    events = load_events(**kwargs)
    fights = load_fight_stats(**kwargs)

    fights = attach_event_dates(fights, events)
    fights = split_result_column(fights)
    fights = randomize_fighter_sides(fights, seed=seed)
    fights = add_outcome_column(fights)
    fights = drop_label_source_columns(fights)

    fights = fights.sort_values("event_date").reset_index(drop=True)
    return fights
