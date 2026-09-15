"""Build the clean, de-biased fight-level dataset from raw source tables.

This is the single place that owns the transform raw CSVs -> one row per
fight, event date attached, no-contests/draws dropped, fighter sides
de-biased. Both the exploratory notebook and any future incremental refresh
(new fights appended to data/raw) call `build_fight_dataset` rather than
reimplementing this logic.
"""

import pandas as pd

from ufc_predictor.config import RANDOMIZE_SEED
from ufc_predictor.data.io import load_events, load_fight_stats
from ufc_predictor.data.randomize import randomize_fighter_sides

UNDECIDED_RESULTS = {"nc", "d"}


def attach_event_dates(fights: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
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


def drop_undecided(fights: pd.DataFrame) -> pd.DataFrame:
    return fights[~fights["result"].isin(UNDECIDED_RESULTS)].copy()


def add_winner_id(fights: pd.DataFrame) -> pd.DataFrame:
    out = fights.copy()
    out["winner_id"] = out["result"].astype("int64")
    return out


def drop_label_source_columns(fights: pd.DataFrame) -> pd.DataFrame:
    """Drop `result`/`winner_id` once `fighter_a_won` has been derived from them.

    Both are now redundant: `winner_id` is recoverable at any time as
    `fighter_a_id if fighter_a_won else fighter_b_id`, and `result` was only
    ever `winner_id` as a string. Keeping them around risks a future `X =
    fights_featured.drop(columns=[...])` accidentally leaving the label in
    the features.
    """
    return fights.drop(columns=["result", "winner_id"])


def build_fight_dataset(
    raw_dir=None,
    seed: int = RANDOMIZE_SEED,
) -> pd.DataFrame:
    """Raw CSVs -> one row per decided fight, chronological, sides de-biased."""
    kwargs = {} if raw_dir is None else {"raw_dir": raw_dir}
    events = load_events(**kwargs)
    fights = load_fight_stats(**kwargs)

    fights = attach_event_dates(fights, events)
    fights = drop_undecided(fights)
    fights = add_winner_id(fights)
    fights = randomize_fighter_sides(fights, seed=seed)
    fights = drop_label_source_columns(fights)

    fights = fights.sort_values("event_date").reset_index(drop=True)
    return fights
