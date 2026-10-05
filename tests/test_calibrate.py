import numpy as np
import pytest
from scipy.special import expit

from ufc_predictor.modeling.calibrate import (
    fit_stretch,
    load_calibration,
    save_calibration,
    stretch,
    stretch_interval,
)
from ufc_predictor.modeling.evaluate import score

P = np.array([0.05, 0.3, 0.45, 0.5, 0.55, 0.7, 0.95])


def _fights(n=20_000, seed=0):
    """True chances, and outcomes drawn from them."""
    rng = np.random.default_rng(seed)
    true_p = expit(rng.normal(0, 1, n))
    won = rng.random(n) < true_p
    return true_p, won


def test_fifty_fifty_stays_fifty_fifty():
    for a in (0.5, 1.0, 2.0):
        assert stretch(0.5, a) == pytest.approx(0.5)


def test_a_stretch_of_one_changes_nothing():
    np.testing.assert_allclose(stretch(P, 1.0), P)


def test_the_two_corners_still_add_up_to_one():
    """No shift term, so corner A gets no built-in advantage."""
    np.testing.assert_allclose(stretch(1 - P, 1.7) + stretch(P, 1.7), 1.0)


def test_bigger_a_means_bolder_predictions():
    bold, timid = stretch(0.7, 2.0), stretch(0.7, 0.5)
    assert timid < 0.7 < bold


def test_picks_and_ranking_never_change():
    true_p, won = _fights(2_000)
    before, after = score(won, true_p), score(won, stretch(true_p, 1.8))
    assert after["accuracy"] == before["accuracy"]
    assert after["auc"] == pytest.approx(before["auc"])


def test_fit_recovers_a_known_squash():
    """Predictions squashed towards 50% with a = 0.5 need a stretch of 2 to undo."""
    true_p, won = _fights()
    squashed = stretch(true_p, 0.5)
    assert fit_stretch(won, squashed) == pytest.approx(2.0, abs=0.15)


def test_honest_predictions_need_no_stretch():
    true_p, won = _fights()
    assert fit_stretch(won, true_p) == pytest.approx(1.0, abs=0.08)


def test_the_interval_brackets_the_fit():
    true_p, won = _fights(2_000)
    a = fit_stretch(won, true_p)
    low, high = stretch_interval(won, true_p, n_boot=100)
    assert low < a < high


def test_saved_calibration_round_trips_inside_the_given_folder(tmp_path):
    save_calibration(1.12, (0.92, 1.33), n_fights=1904, decision="kept", processed_dir=tmp_path)
    assert (tmp_path / "calibration.json").exists()
    assert load_calibration(processed_dir=tmp_path) == 1.12
