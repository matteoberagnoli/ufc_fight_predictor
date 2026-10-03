"""Choosing which fights become modelling rows.

The era filter must run on the *featured* table - after rolling features have
been computed on the full history - never on fights.parquet before them.

    features first, then filter -> a veteran's pre-2014 fights still count
                                   towards their record and career averages
    filter first, then features -> every fighter's history restarts in 2014,
                                   so a 15-fight veteran looks like a debutant

The second version doesn't crash. It just quietly makes the rows wrong,
which is why the filter lives here, downstream of the feature pipeline,
rather than in data/build.py.

Draws and no-contests are dropped here too, for the same reason: they have
no winner to predict, so they cannot be modelling rows - but they are real
fights, and `data/build.py` keeps them so they still count as each
fighter's history. Dropping them any earlier would erase 143 fights from
2,495 fighters' records.

Within the era, rows are split by date into train / validation / test. The
validation period exists so that design decisions (which features, which
model settings) are made without ever looking at the test years - every
decision taken by peeking at the test set makes its final score less honest.
"""

import pandas as pd

from ufc_predictor.config import ERA_START, TEST_START, VALIDATION_START


def drop_undecided(fights: pd.DataFrame) -> pd.DataFrame:
    """Keep only fights with a winner. Pass the featured table, not raw fights.

    A draw or no-contest has no answer to learn, so it cannot be a training
    or test row. It stays in `fights.parquet` and in every fighter's rolled
    history; it just never becomes a row of `X`.
    """
    return fights[fights["fighter_a_won"].notna()].reset_index(drop=True)


def filter_era(fights: pd.DataFrame, start: str = ERA_START) -> pd.DataFrame:
    """Keep only fights on or after `start`. Pass the featured table, not raw fights."""
    return fights[fights["event_date"] >= pd.Timestamp(start)].reset_index(drop=True)


def chronological_split(
    fights: pd.DataFrame,
    era_start: str = ERA_START,
    validation_start: str = VALIDATION_START,
    test_start: str = TEST_START,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Featured table -> (train, validation, test), split by date, never at random.

    Each boundary date belongs to the later period. Fights before `era_start`,
    and draws and no-contests at any date, are in no period at all - they
    still count as fighter history, because the features were computed
    before this split.
    """
    starts = [pd.Timestamp(d) for d in (era_start, validation_start, test_start)]
    if not starts[0] < starts[1] < starts[2]:
        raise ValueError(
            f"period starts must be in order, got era={era_start}, "
            f"validation={validation_start}, test={test_start}"
        )

    era = drop_undecided(filter_era(fights, start=era_start))
    dates = era["event_date"]
    train = era[dates < starts[1]]
    validation = era[(dates >= starts[1]) & (dates < starts[2])]
    test = era[dates >= starts[2]]
    return tuple(part.reset_index(drop=True) for part in (train, validation, test))
