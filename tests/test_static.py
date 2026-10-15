import numpy as np
import pandas as pd
import pytest

from ufc_predictor.features.static import attach_stance_features, attach_static_features


@pytest.fixture
def toy_featured() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fights_url": ["f/1", "f/2", "f/3", "f/4"],
            "fighter_a_id": [1, 2, 2, 1],
            "fighter_b_id": [2, 3, 4, 5],
            "fighter_a_age_during": [28.0, 31.0, 31.0, 28.0],
            "fighter_b_age_during": [30.0, 25.0, 29.0, 33.0],
            "fighter_a_height_cm": [180.0, 175.0, 175.0, 180.0],
            "fighter_b_height_cm": [175.0, 190.0, 178.0, 182.0],
        }
    )


@pytest.fixture
def toy_fighters() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fighter_id": [1, 2, 3, 4, 5],
            "fighter_reach_cm": [183.0, 178.0, np.nan, 180.0, 190.0],
            # fighter 4 shares fighter 2's stance (Southpaw); fighter 5 has no
            # stance recorded, for the "can't tell" case
            "fighter_stance": ["Orthodox", "Southpaw", "Orthodox", "Southpaw", np.nan],
        }
    )


def test_attach_static_features_row_count_unchanged(toy_featured, toy_fighters):
    out = attach_static_features(toy_featured, toy_fighters)
    assert len(out) == len(toy_featured)


def test_ape_index_is_reach_minus_height(toy_featured, toy_fighters):
    out = attach_static_features(toy_featured, toy_fighters)
    row = out.iloc[0]
    assert row["fighter_a_ape_index_cm"] == pytest.approx(183.0 - 180.0)
    assert row["fighter_b_ape_index_cm"] == pytest.approx(178.0 - 175.0)
    assert row["diff_static_ape_index_cm"] == pytest.approx(
        row["fighter_a_ape_index_cm"] - row["fighter_b_ape_index_cm"]
    )


def test_ape_index_is_nan_when_reach_missing(toy_featured, toy_fighters):
    out = attach_static_features(toy_featured, toy_fighters)
    row = out.iloc[1]  # fighter_b=3, reach is NaN
    assert pd.isna(row["fighter_b_ape_index_cm"])
    assert pd.isna(row["diff_static_ape_index_cm"])


def test_same_stance_true_when_stances_match(toy_featured, toy_fighters):
    out = attach_static_features(toy_featured, toy_fighters)
    # fight 3: fighter_a=2 (Southpaw), fighter_b=4 (Southpaw)
    assert out.iloc[2]["same_stance"] == True  # noqa: E712


def test_same_stance_false_when_stances_differ(toy_featured, toy_fighters):
    out = attach_static_features(toy_featured, toy_fighters)
    # fight 1: Orthodox vs Southpaw
    assert out.iloc[0]["same_stance"] == False  # noqa: E712


def test_same_stance_is_na_when_either_stance_unknown(toy_featured, toy_fighters):
    out = attach_static_features(toy_featured, toy_fighters)
    # fight 4: fighter_b=5 has no recorded stance
    assert pd.isna(out.iloc[3]["same_stance"])


def test_reach_and_stance_joined_correctly(toy_featured, toy_fighters):
    out = attach_static_features(toy_featured, toy_fighters)
    # fight 1: fighter_a=1 (reach 183, Orthodox), fighter_b=2 (reach 178, Southpaw)
    row = out.iloc[0]
    assert row["fighter_a_reach_cm"] == 183.0
    assert row["fighter_a_stance"] == "Orthodox"
    assert row["fighter_b_reach_cm"] == 178.0
    assert row["fighter_b_stance"] == "Southpaw"


def test_missing_reach_stays_nan_not_dropped(toy_featured, toy_fighters):
    out = attach_static_features(toy_featured, toy_fighters)
    # fight 2: fighter_b=3, whose reach is NaN in the fixture
    row = out.iloc[1]
    assert pd.isna(row["fighter_b_reach_cm"])
    assert len(out) == len(toy_featured)  # row wasn't dropped just because reach is missing


def test_diff_columns_are_a_minus_b(toy_featured, toy_fighters):
    out = attach_static_features(toy_featured, toy_fighters)
    row = out.iloc[0]
    assert row["diff_static_age_gap"] == 28.0 - 30.0
    assert row["diff_static_height_cm"] == 180.0 - 175.0
    assert row["diff_static_reach_cm"] == 183.0 - 178.0


def test_same_fighter_gets_same_static_values_in_different_fights(toy_featured, toy_fighters):
    # fighter 2 is fighter_b in fight 1 and fighter_a in fight 2 - same person,
    # same reach/stance either side of the swap
    out = attach_static_features(toy_featured, toy_fighters)
    assert out.iloc[0]["fighter_b_reach_cm"] == out.iloc[1]["fighter_a_reach_cm"]
    assert out.iloc[0]["fighter_b_stance"] == out.iloc[1]["fighter_a_stance"]


def _stances(a, b) -> pd.DataFrame:
    return pd.DataFrame({"fighter_a_stance": a, "fighter_b_stance": b})


def test_stance_features_say_which_corner_has_the_rarer_stance():
    out = attach_stance_features(_stances(
        ["Southpaw", "Orthodox", "Southpaw", "Orthodox", "Switch", "Switch"],
        ["Orthodox", "Southpaw", "Southpaw", "Orthodox", "Orthodox", "Southpaw"],
    ))
    assert out["diff_static_southpaw"].tolist() == [1, -1, 0, 0, 0, -1]
    assert out["diff_static_switch"].tolist() == [0, 0, 0, 0, 1, 1]


def test_stance_features_are_nan_when_either_stance_is_unknown():
    out = attach_stance_features(_stances(["Southpaw", None, None], [None, "Orthodox", None]))
    assert out["diff_static_southpaw"].isna().all() and out["diff_static_switch"].isna().all()


def test_rare_stances_count_as_neither():
    out = attach_stance_features(_stances(["Open Stance", "Sideways"], ["Orthodox", "Southpaw"]))
    assert out["diff_static_southpaw"].tolist() == [0, -1]
    assert out["diff_static_switch"].tolist() == [0, 0]


def test_stance_features_flip_when_the_corners_swap():
    """Swap the stances and every value changes sign - which is exactly what
    model.mirror does to a diff_ column, so the two always agree."""
    a = ["Southpaw", "Orthodox", "Switch", "Southpaw", None]
    b = ["Orthodox", "Switch", "Southpaw", "Southpaw", "Orthodox"]
    forward, swapped = attach_stance_features(_stances(a, b)), attach_stance_features(_stances(b, a))
    for column in ("diff_static_southpaw", "diff_static_switch"):
        pd.testing.assert_series_equal(swapped[column], -forward[column] + 0.0, check_names=False)


def test_stance_features_are_physical_features():
    from ufc_predictor.features.manifest import _group_of
    assert _group_of("diff_static_southpaw") == _group_of("diff_static_switch") == "physical"
