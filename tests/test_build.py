import json

import pandas as pd
import pytest

from ufc_predictor.config import RAW_DIR
from ufc_predictor.data.build import build_fight_dataset
from ufc_predictor.features.manifest import LABEL_COLUMN
from ufc_predictor.modeling.split import drop_undecided


@pytest.fixture(scope="module")
def fights() -> pd.DataFrame:
    return build_fight_dataset()


def test_build_fight_dataset_keeps_every_raw_fight(fights):
    """Draws and no-contests stay: they are history, even if never modelling rows.

    Dropping them here was the original mistake. It erased 143 fights from
    264 fighters' records, which left 20.6% of modelling-era fights built on
    an incomplete history - a fighter's layoff stretched across a bout they
    actually had, and its strikes and cage time gone from their averages.
    """
    raw_count = pd.read_csv("data/raw/ufc_event_fight_stats.csv").shape[0]
    source = json.loads((RAW_DIR / "source" / "SOURCE.json").read_text())
    assert len(fights) == raw_count == source["fights"]


def test_undecided_fights_have_no_label_but_a_readable_outcome(fights):
    undecided = fights[fights[LABEL_COLUMN].isna()]
    raw_result = pd.read_csv("data/raw/ufc_event_fight_stats.csv")["result"]
    assert len(undecided) == raw_result.isin(["d", "nc"]).sum() > 0
    assert undecided["outcome"].value_counts().to_dict() == {
        "no_contest": (raw_result == "nc").sum(), "draw": (raw_result == "d").sum()}
    # and every decided fight names the corner that won
    decided = fights[fights[LABEL_COLUMN].notna()]
    assert set(decided["outcome"]) == {"fighter_a", "fighter_b"}
    assert decided[LABEL_COLUMN].eq(decided["outcome"] == "fighter_a").all()


def test_label_is_nullable_so_a_draw_cannot_pass_as_a_loss(fights):
    # a plain bool column has no way to say "no winner" - it would call all
    # 143 of them a fighter_b win
    assert fights[LABEL_COLUMN].dtype == "boolean"


def test_build_fight_dataset_is_chronological(fights):
    assert fights["event_date"].is_monotonic_increasing


def test_build_fight_dataset_label_balance_holds_in_every_era(fights):
    decided = drop_undecided(fights)
    assert 0.45 < decided[LABEL_COLUMN].mean() < 0.55

    # this is the specific bug being fixed: the source lists the winner as f1
    # in every decided fight (the previous source did so before 2010).
    # Individual years are too small a sample to expect exactly 50/50, but
    # each era as a whole should be nowhere near 100%.
    for era in (decided["event_date"].dt.year < 2010, decided["event_date"].dt.year >= 2010):
        assert era.sum() > 1000
        assert 0.40 < decided.loc[era, LABEL_COLUMN].mean() < 0.60


def test_build_fight_dataset_is_reproducible():
    pd.testing.assert_frame_equal(build_fight_dataset(), build_fight_dataset())


def test_build_fight_dataset_drops_redundant_label_columns(fights):
    # result/winner_id/undecided_kind are redundant with fighter_a_won +
    # outcome + the two fighter ids, and keeping them around risks a future
    # accidental leak into X
    for column in ("result", "winner_id", "undecided_kind"):
        assert column not in fights.columns
    assert LABEL_COLUMN in fights.columns
    assert "outcome" in fights.columns


def test_a_missing_result_is_rejected_loudly():
    """`astype(str)` turns NaN into "nan", which pd.to_numeric parses back to
    NaN - so without an explicit check the fight would silently end up with
    fighter_a_won = <NA> but outcome = "fighter_b", two label columns
    disagreeing. Reachable the moment a scrape returns an incomplete row."""
    from ufc_predictor.data.build import split_result_column

    fights = pd.DataFrame({"fights_url": ["f1"], "result": [None]})
    with pytest.raises(ValueError, match="no `result` value"):
        split_result_column(fights)


def test_a_result_that_is_neither_an_id_nor_a_known_code_is_rejected(fights):
    from ufc_predictor.data.build import split_result_column

    bad = pd.DataFrame({"fights_url": ["f1"], "result": ["draw"]})   # not "d"
    with pytest.raises(ValueError):
        split_result_column(bad)
