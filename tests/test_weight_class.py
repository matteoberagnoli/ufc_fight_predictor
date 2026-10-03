import numpy as np
import pandas as pd
import pytest

from ufc_predictor.features.weight_class import (
    NO_DIVISION,
    attach_weight_class_features,
    division_from_weight_class,
    is_womens_bout,
    weight_limit_lbs,
)


def test_every_real_weight_class_string_is_recognised():
    strings = pd.Series([
        "Lightweight Bout",
        "UFC Lightweight Title Bout",
        "Ultimate Fighter 15 Lightweight Tournament Title Bout",
        "Women's Strawweight Bout",
        "UFC Women's Bantamweight Title Bout",
    ])
    assert division_from_weight_class(strings).tolist() == [
        "lightweight", "lightweight", "lightweight", "strawweight", "bantamweight",
    ]


def test_light_heavyweight_is_not_swallowed_by_heavyweight():
    """The trap the module exists to avoid: every light heavyweight string
    also contains the word "heavyweight"."""
    strings = pd.Series(["Light Heavyweight Bout", "UFC Light Heavyweight Title Bout", "Heavyweight Bout"])
    assert division_from_weight_class(strings).tolist() == [
        "light_heavyweight", "light_heavyweight", "heavyweight",
    ]


def test_catch_weight_and_open_weight_have_no_division():
    strings = pd.Series(["Catch Weight Bout", "Open Weight Bout", "UFC 2 Tournament Title Bout"])
    assert (division_from_weight_class(strings) == NO_DIVISION).all()


def test_weight_limit_matches_the_real_ufc_divisions():
    division = pd.Series(["strawweight", "flyweight", "lightweight", "heavyweight", NO_DIVISION])
    assert weight_limit_lbs(division).tolist()[:4] == [115.0, 125.0, 155.0, 265.0]
    assert np.isnan(weight_limit_lbs(division).iloc[4])


def test_is_womens_bout_catches_the_catchweight_gap():
    """A catchweight bout carries no gender word, so it has to be resolved
    from the fighters' other fights."""
    history = pd.DataFrame({
        "weight_class": ["Women's Bantamweight Bout", "Catch Weight Bout", "Lightweight Bout"],
        "fighter_a_id": [1, 1, 5],
        "fighter_b_id": [2, 3, 6],
    })
    assert is_womens_bout(history).tolist() == [True, True, False]


def test_is_womens_bout_on_a_slice_needs_full_history():
    history = pd.DataFrame({
        "weight_class": ["Women's Bantamweight Bout", "Catch Weight Bout"],
        "fighter_a_id": [1, 1], "fighter_b_id": [2, 3],
    })
    catchweight_only = history.iloc[[1]]
    assert not is_womens_bout(catchweight_only).iloc[0]            # her other fight isn't visible
    assert is_womens_bout(catchweight_only, history=history).iloc[0]


def test_attach_weight_class_features_adds_exactly_three_columns():
    featured = pd.DataFrame({
        "weight_class": ["Lightweight Bout", "Women's Strawweight Bout"],
        "fighter_a_id": [1, 2], "fighter_b_id": [3, 4],
    })
    out = attach_weight_class_features(featured)
    assert set(out.columns) - set(featured.columns) == {"division", "weight_class_lbs", "is_womens_bout"}
    assert out["weight_class_lbs"].tolist() == [155.0, 115.0]
    assert out["is_womens_bout"].tolist() == [False, True]
    assert out["is_womens_bout"].dtype == "boolean"


def test_every_fight_gets_exactly_one_division():
    """On the real table: every fight classified, nothing double-counted."""
    from ufc_predictor.data.io import load_processed
    try:
        featured = load_processed("fights_featured")
    except (FileNotFoundError, OSError):
        pytest.skip("processed data not built yet")
    out = attach_weight_class_features(featured)
    assert out["division"].notna().all()
    assert out.groupby("division", observed=True)["weight_class_lbs"].nunique().le(1).all()
    assert len(out) == len(featured)
