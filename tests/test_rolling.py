import numpy as np
import pandas as pd
import pytest

from ufc_predictor.features.rolling import (
    _streak_after_each_fight,
    add_prior_features,
    attach_prior_features,
    build_fighter_history,
)


@pytest.fixture
def toy_fights() -> pd.DataFrame:
    # fighter 1 vs a new opponent each time: W, L, W, W (as fighter_a every time)
    # fighter 99 only shows up as the opponent, never as the fighter being tracked
    dates = pd.to_datetime(["2020-01-01", "2020-03-01", "2020-05-01", "2020-09-01"])
    return pd.DataFrame(
        {
            "fights_url": ["f/1", "f/2", "f/3", "f/4"],
            "event_date": dates,
            "fighter_a_id": [1, 1, 1, 1],
            "fighter_b_id": [90, 91, 92, 93],
            "fighter_a_won": [True, False, True, True],
            **{
                f"fighter_{side}_{stat}": [10, 10, 10, 10]
                for side in ("a", "b")
                for stat in (
                    "sig_strikes", "sig_strike_atts", "takedowns", "takedown_atts",
                    "knockdowns", "submissions", "ctrl_time",
                )
            },
        }
    )


def test_streak_after_each_fight_tracks_sign_and_length():
    won = pd.Series([True, True, False, False, False, True])
    assert list(_streak_after_each_fight(won)) == [1, 2, -1, -2, -3, 1]


def test_build_fighter_history_has_two_rows_per_fight(toy_fights):
    history = build_fighter_history(toy_fights)
    assert len(history) == 2 * len(toy_fights)


def test_prior_features_only_see_earlier_fights(toy_fights):
    history = build_fighter_history(toy_fights)
    featured = add_prior_features(history)
    fighter_1 = featured[featured["fighter_id"] == 1].sort_values("event_date")

    # W, L, W, W -> prior record before each fight:
    assert list(fighter_1["wins"]) == [0, 1, 1, 2]
    assert list(fighter_1["losses"]) == [0, 0, 1, 1]
    assert list(fighter_1["is_debut"]) == [True, False, False, False]
    # streak entering each fight: debut(0), after W->1, after L->-1, after W->1
    assert list(fighter_1["streak"]) == [0, 1, -1, 1]


def test_debut_fighter_has_no_prior_history(toy_fights):
    # fighters 90-93 each appear exactly once, always as a debut
    history = build_fighter_history(toy_fights)
    featured = add_prior_features(history)
    opponents = featured[featured["fighter_id"].isin([90, 91, 92, 93])]
    assert opponents["is_debut"].all()
    assert (opponents["fight_count"] == 0).all()
    assert opponents["sig_strikes_career_avg"].isna().all()


def test_attach_prior_features_leakage_audit(toy_fights):
    """The actual point of this module: assert no feature for a given fight
    reflects anything from that fight or a later one.

    We do this by recomputing, from scratch and independently (a plain
    Python loop over each fighter's chronological fights, not reusing any
    of rolling.py's own aggregation code), what fighter 1's prior win/loss
    record *should* be at each of their fights, and checking it against
    what attach_prior_features actually produced.
    """
    featured = attach_prior_features(toy_fights)
    fighter_1_fights = featured[featured["fighter_a_id"] == 1].sort_values("event_date")

    outcomes = [True, False, True, True]  # ground truth, independent of the module
    expected_wins, expected_losses = [], []
    w, l = 0, 0
    for outcome in outcomes:
        expected_wins.append(w)
        expected_losses.append(l)
        if outcome:
            w += 1
        else:
            l += 1

    assert list(fighter_1_fights["fighter_a_prior_wins"]) == [float(x) for x in expected_wins]
    assert list(fighter_1_fights["fighter_a_prior_losses"]) == [float(x) for x in expected_losses]


def test_diff_columns_are_a_minus_b(toy_fights):
    featured = attach_prior_features(toy_fights)
    row = featured.iloc[3]  # fighter 1's 4th fight: prior_wins=2, prior_losses=1
    assert row["diff_prior_wins"] == (
        row["fighter_a_prior_wins"] - row["fighter_b_prior_wins"]
    )


def test_attach_prior_features_row_count_unchanged(toy_fights):
    featured = attach_prior_features(toy_fights)
    assert len(featured) == len(toy_fights)
