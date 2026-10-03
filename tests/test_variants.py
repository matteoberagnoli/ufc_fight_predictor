import numpy as np
import pandas as pd

from ufc_predictor.features.rolling import MIRRORED_STATS, SHARED_STATS, attach_prior_features, no_shrinkage
from ufc_predictor.modeling.variants import zero_fill_undefined_accuracy


def _raw_features() -> pd.DataFrame:
    """Fighter 1 never attempts a takedown, but lands 5 of 10 strikes every fight."""
    fights = pd.DataFrame({
        "fights_url": ["f/1", "f/2", "f/3", "f/4"],
        "event_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        "fighter_a_id": [1, 1, 1, 1],
        "fighter_b_id": [90, 91, 92, 93],
        "fighter_a_won": pd.array([True, False, True, True], dtype="boolean"),
        "outcome": ["fighter_a", "fighter_b", "fighter_a", "fighter_a"],
        "weight_class": ["Lightweight Bout"] * 4,
        **{f"fighter_{s}_{stat}": [0.0] * 4 for s in ("a", "b") for stat in MIRRORED_STATS},
        **{f"fighter_{s}_{stat}": [600.0] * 4 for s in ("a", "b") for stat in SHARED_STATS},
    })
    fights["fighter_a_sig_strikes"] = 5.0
    fights["fighter_a_sig_strike_atts"] = 10.0
    return attach_prior_features(fights, no_shrinkage())


def test_never_attempted_becomes_zero_but_debut_stays_unknown():
    raw = _raw_features()
    filled = zero_fill_undefined_accuracy(raw)
    td = filled["fighter_a_prior_takedowns_accuracy_career_avg"]
    assert raw["fighter_a_prior_takedowns_accuracy_career_avg"].iloc[1:].isna().all()   # 0/0 before
    assert np.isnan(td.iloc[0])                    # debut: no history, nothing to fill
    assert td.iloc[1:].tolist() == [0.0, 0.0, 0.0]


def test_real_accuracy_is_left_alone():
    filled = zero_fill_undefined_accuracy(_raw_features())
    assert filled["fighter_a_prior_sig_strikes_accuracy_career_avg"].iloc[1:].tolist() == [0.5, 0.5, 0.5]


def test_differences_are_rebuilt_after_filling():
    filled = zero_fill_undefined_accuracy(_raw_features())
    col = "takedowns_accuracy_career_avg"
    expected = filled[f"fighter_a_prior_{col}"] - filled[f"fighter_b_prior_{col}"]
    pd.testing.assert_series_equal(filled[f"diff_prior_{col}"], expected, check_names=False)
