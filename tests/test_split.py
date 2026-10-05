import pandas as pd
import pytest

from ufc_predictor.config import ERA_START, TEST_START, VALIDATION_START
from ufc_predictor.data.io import load_processed
from ufc_predictor.features.rolling import (
    MIRRORED_STATS, SHARED_STATS, attach_prior_features, no_shrinkage,
)
from ufc_predictor.modeling.split import chronological_split, drop_undecided, filter_era


def _toy_fights() -> pd.DataFrame:
    # fighter 1 fights twice before the cutoff and once after it
    dates = pd.to_datetime(["2012-01-01", "2013-01-01", "2015-01-01"])
    return pd.DataFrame(
        {
            "fights_url": ["f/1", "f/2", "f/3"],
            "event_date": dates,
            "fighter_a_id": [1, 1, 1],
            "fighter_b_id": [90, 91, 92],
            "fighter_a_won": pd.array([True, True, False], dtype="boolean"),
        "outcome": ["fighter_a", "fighter_a", "fighter_b"],
            "weight_class": ["Lightweight Bout"] * 3,
            **{
                f"fighter_{side}_{stat}": [10, 10, 10]
                for side in ("a", "b")
                for stat in MIRRORED_STATS + SHARED_STATS
            },
        }
    )


def test_filter_keeps_only_fights_from_the_start_date():
    out = filter_era(_toy_fights(), start="2014-01-01")
    assert list(out["fights_url"]) == ["f/3"]
    assert out.index.tolist() == [0]


def test_start_date_is_inclusive():
    out = filter_era(_toy_fights(), start="2015-01-01")
    assert list(out["fights_url"]) == ["f/3"]


def test_default_start_comes_from_config():
    fights = _toy_fights()
    assert filter_era(fights).equals(filter_era(fights, start=ERA_START))


def test_pre_era_fights_still_count_as_history():
    """The reason the filter runs after features, not before.

    Fighter 1 went 2-0 before 2014. Their 2015 fight must show that record -
    not a debut, which is what filtering before feature computation produces.
    """
    right = filter_era(attach_prior_features(_toy_fights(), no_shrinkage()), start="2014-01-01")
    row = right.iloc[0]
    assert row["fighter_a_prior_fight_count"] == 2
    assert row["fighter_a_prior_wins"] == 2
    assert not row["fighter_a_prior_is_debut"]

    # and the wrong order really is wrong
    wrong = attach_prior_features(filter_era(_toy_fights(), start="2014-01-01"), no_shrinkage())
    assert wrong.iloc[0]["fighter_a_prior_is_debut"]


def _dated(dates: list[str], outcomes: list[str] | None = None) -> pd.DataFrame:
    outcomes = outcomes or ["fighter_a"] * len(dates)
    won = pd.array([{"fighter_a": True, "fighter_b": False}.get(o, pd.NA) for o in outcomes],
                   dtype="boolean")
    return pd.DataFrame({"fights_url": [f"f/{i}" for i in range(len(dates))],
                         "event_date": pd.to_datetime(dates),
                         "fighter_a_won": won,
                         "outcome": outcomes})


def test_split_puts_every_era_fight_in_exactly_one_period():
    fights = _dated(["2013-06-01", "2015-01-01", "2019-12-31", "2020-01-01",
                     "2021-07-01", "2022-01-01", "2024-03-01"])
    train, validation, test = chronological_split(
        fights, era_start="2014-01-01", validation_start="2020-01-01", test_start="2022-01-01"
    )
    assert list(train["fights_url"]) == ["f/1", "f/2"]
    assert list(validation["fights_url"]) == ["f/3", "f/4"]   # 2020-01-01 is validation
    assert list(test["fights_url"]) == ["f/5", "f/6"]         # 2022-01-01 is test
    # the 2013 fight is history only - in no period
    assert "f/0" not in set(train["fights_url"]) | set(validation["fights_url"]) | set(test["fights_url"])


def test_periods_run_forward_in_time():
    fights = _dated(["2014-02-01", "2018-05-05", "2020-03-03", "2021-11-11", "2023-01-01"])
    train, validation, test = chronological_split(fights)
    assert train["event_date"].max() < validation["event_date"].min()
    assert validation["event_date"].max() < test["event_date"].min()


def test_split_defaults_come_from_config():
    fights = _dated(["2015-01-01", "2020-06-01", "2023-06-01"])
    default = chronological_split(fights)
    explicit = chronological_split(fights, era_start=ERA_START,
                                   validation_start=VALIDATION_START, test_start=TEST_START)
    assert all(a.equals(b) for a, b in zip(default, explicit))


def test_out_of_order_dates_are_rejected():
    with pytest.raises(ValueError, match="in order"):
        chronological_split(_dated(["2015-01-01"]),
                            validation_start="2022-01-01", test_start="2020-01-01")


def test_real_data_split_sizes():
    try:
        fights = load_processed("fights")
    except (FileNotFoundError, OSError):
        pytest.skip("processed data not built yet - run notebook 02")
    train, validation, test = chronological_split(fights)
    # train and validation are the same fights as with the previous data source.
    # The test period runs to the end of the data, so it grows with each refresh:
    # 1,471 fights to 2024-11-09, 2,448 once 2025-26 was added (data source v18).
    assert (len(train), len(validation)) == (2862, 941)
    assert len(test) >= 1471
    assert test["event_date"].min() >= pd.Timestamp("2022-01-01")



def test_drop_undecided_keeps_only_fights_with_a_winner():
    fights = _dated(["2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01"],
                    outcomes=["fighter_a", "draw", "no_contest", "fighter_b"])
    assert list(drop_undecided(fights)["fights_url"]) == ["f/0", "f/3"]


def test_split_excludes_draws_and_no_contests():
    fights = _dated(["2015-01-01", "2016-01-01", "2020-06-01", "2023-06-01"],
                    outcomes=["fighter_a", "draw", "no_contest", "fighter_b"])
    train, validation, test = chronological_split(fights)
    assert list(train["fights_url"]) == ["f/0"]     # the 2016 draw is gone
    assert list(validation["fights_url"]) == []     # the 2020 no-contest is gone
    assert list(test["fights_url"]) == ["f/3"]


def test_undecided_fights_still_count_as_history():
    """The mirror of the era test: a draw is not a modelling row, but it IS history.

    Fighter 1 wins, then draws, then fights again. The last fight must know
    about both earlier bouts - two fights of experience, not one.
    """
    fights = _toy_fights()
    fights["outcome"] = ["fighter_a", "draw", "fighter_b"]
    fights["fighter_a_won"] = pd.array([True, pd.NA, False], dtype="boolean")

    featured = attach_prior_features(fights, no_shrinkage())
    last = drop_undecided(featured).iloc[-1]
    assert last["fighter_a_prior_fight_count"] == 2
    assert last["fighter_a_prior_wins"] == 1
    assert last["fighter_a_prior_losses"] == 0      # a draw is neither
