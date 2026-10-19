import numpy as np
import pandas as pd
import pytest

from ufc_predictor.config import RAW_DIR
from ufc_predictor.data.build import build_fight_dataset
from ufc_predictor.data.io import load_fight_stats, load_fighters
from ufc_predictor.features import manifest
from ufc_predictor.features.pipeline import build_featured_table
from ufc_predictor.features.rolling import no_shrinkage
from ufc_predictor.modeling.final import UPCOMING, _batches, training_rows, upcoming_featured, upcoming_rows
from ufc_predictor.modeling.model import mirror

FIGHTERS = pd.DataFrame({
    "fighter_id": [1, 2, 3], "fighter_name": ["Ann Able", "Bea Bold", "Bea Bold"],
    "fighter_dob": ["1990-06-15", "1995-01-01", "1980-01-01"], "fighter_height_cm": [170.0, 165.0, 160.0],
    "fighter_reach_cm": [175.0, 168.0, 160.0], "fighter_stance": ["Orthodox", "Southpaw", "Orthodox"],
})


def test_an_upcoming_fight_has_no_stats_and_no_result():
    stats, events = upcoming_rows([{"fighter_a": "ann able", "fighter_b": 2, "date": "2026-11-01",
                                    "weight_class": "Women's Flyweight Bout"}], FIGHTERS)
    row = stats.iloc[0]
    assert (row["f1_id"], row["f2_id"], row["result"], row["method"]) == (1, 2, "nc", "other")
    assert row["fights_url"].startswith(UPCOMING) and np.isnan(row["f1_sig_strikes"])
    assert row["f1_age_during"] == 36 and row["f2_age_during"] == 31
    assert events["event_date"].iloc[0] == "2026-11-01"


def test_unknown_or_ambiguous_names_are_refused():
    with pytest.raises(ValueError, match="no fighter"):
        upcoming_rows([{"fighter_a": "Nobody", "fighter_b": 1, "date": "2026-11-01", "weight_class": "x"}], FIGHTERS)
    with pytest.raises(ValueError, match="pass the id"):
        upcoming_rows([{"fighter_a": "Bea Bold", "fighter_b": 1, "date": "2026-11-01", "weight_class": "x"}], FIGHTERS)


def test_a_fighter_cannot_be_on_both_sides():
    with pytest.raises(ValueError, match="both sides"):
        upcoming_rows([{"fighter_a": "Ann Able", "fighter_b": 1, "date": "2026-11-01", "weight_class": "x"}], FIGHTERS)


def test_bouts_sharing_a_fighter_are_built_in_separate_passes():
    # a card where everyone fights once is one pass; a round robin of 1, 2, 3 needs three
    assert _batches([(1, 2), (3, 4), (5, 6)]) == [[0, 1, 2]]
    assert _batches([(1, 2), (1, 3), (2, 3), (4, 5)]) == [[0, 3], [1], [2]]
    batches = _batches([(1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (1, 5)])
    assert sorted(i for b in batches for i in b) == list(range(6))
    pairs = [(1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (1, 5)]
    for b in batches:
        ids = [f for i in b for f in pairs[i]]
        assert len(ids) == len(set(ids))


def test_training_rows_stop_before_until_and_skip_undecided():
    f = pd.DataFrame({"event_date": pd.to_datetime(["2013-06-01", "2016-01-01", "2016-02-01", "2025-01-01"]),
                      "fighter_a_won": pd.array([True, None, False, True], dtype="boolean")})
    assert training_rows(f, "2025-01-01")["event_date"].tolist() == [pd.Timestamp("2016-02-01")]


def test_an_upcoming_fight_gets_exactly_the_features_the_real_fight_had(tmp_path):
    """Take a real 2024 fight out of the raw data, add it back as 'upcoming' on the same night:
    its 40 inputs must equal the real row's (corners may be swapped by the coin flip)."""
    try:
        stats = load_fight_stats()
    except (FileNotFoundError, OSError):
        pytest.skip("raw data not present")
    fighters = load_fighters()
    real_featured = build_featured_table(build_fight_dataset(), fighters, no_shrinkage(), elo_k=158.5,
                                         elo_weights={"unanimous": 0.7, "narrow": 0.55})
    real = real_featured[(real_featured["event_date"] == "2024-06-29")
                         & real_featured["fighter_a_won"].notna()].iloc[0]
    raw = stats[stats["fights_url"] != real["fights_url"]]
    for name in ("ufc_events.csv", "ufc_fighters.csv"):
        (tmp_path / name).write_bytes((RAW_DIR / name).read_bytes())
    raw.to_csv(tmp_path / "ufc_event_fight_stats.csv", index=False)
    bout = {"fighter_a": int(real["fighter_a_id"]), "fighter_b": int(real["fighter_b_id"]),
            "date": "2024-06-29", "weight_class": real["weight_class"]}
    rows, _ = upcoming_featured([bout], no_shrinkage(), 158.5, {"unanimous": 0.7, "narrow": 0.55}, raw_dir=tmp_path)
    cols = manifest.FEATURES
    got = rows[list(cols)]
    if rows["fighter_a_id"].iloc[0] != real["fighter_a_id"]:
        got = mirror(got)
    expected = real_featured.loc[[real.name], list(cols)]
    pd.testing.assert_frame_equal(got.reset_index(drop=True).astype(float),
                                  expected.reset_index(drop=True).astype(float), check_exact=False, atol=1e-9)


def test_other_bouts_in_the_same_call_do_not_change_a_bouts_inputs():
    """The same fighter in an earlier bout (same night or weeks before) must not count as a fight they
    already had: the bout's 40 inputs equal those it gets when predicted alone, and stay in order."""
    try:
        stats = load_fight_stats()
    except (FileNotFoundError, OSError):
        pytest.skip("raw data not present")
    recent = stats.dropna(subset=["f1_id", "f2_id"]).iloc[-1]
    a, b = int(recent["f1_id"]), int(recent["f2_id"])
    third = int(stats.loc[~stats["f1_id"].isin([a, b]), "f1_id"].dropna().iloc[-1])
    elo = (158.5, {"unanimous": 0.7, "narrow": 0.55})
    bout = {"fighter_a": a, "fighter_b": b, "date": "2026-12-05", "weight_class": "Lightweight Bout"}
    alone, _ = upcoming_featured([bout], no_shrinkage(), *elo)
    for earlier_date in ("2026-12-05", "2026-11-01"):
        other = {"fighter_a": third, "fighter_b": a, "date": earlier_date, "weight_class": "Lightweight Bout"}
        rows, first_ids = upcoming_featured([other, bout], no_shrinkage(), *elo)
        assert first_ids.tolist() == [third, a]
        got = rows.iloc[[1]][list(manifest.FEATURES)]
        if rows["fighter_a_id"].iloc[1] != alone["fighter_a_id"].iloc[0]:
            got = mirror(got)
        pd.testing.assert_frame_equal(got.reset_index(drop=True).astype(float),
                                      alone[list(manifest.FEATURES)].reset_index(drop=True).astype(float),
                                      check_exact=False, atol=1e-9)
