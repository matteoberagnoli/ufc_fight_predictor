import numpy as np
import pandas as pd
import pytest

from ufc_predictor.modeling.model import (
    corner_disagreement,
    fit_model,
    mirror,
    predict_proba,
    to_matrix,
)


def _toy(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Fights where a hidden skill gap decides the winner, mostly."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1, n)
    X = pd.DataFrame({
        "diff_prior_skill": skill,
        "diff_static_age": rng.normal(0, 5, n),
        "fighter_a_prior_is_debut": rng.random(n) < 0.2,
        "fighter_b_prior_is_debut": rng.random(n) < 0.2,
        "same_stance": pd.array(rng.random(n) < 0.6, dtype="boolean"),
    })
    y = pd.Series(rng.random(n) < 1 / (1 + np.exp(-2 * skill)))
    return X, y


def test_mirror_negates_diffs_swaps_pairs_and_keeps_symmetric_features():
    X, _ = _toy(6)
    m = mirror(X)
    assert (m["diff_prior_skill"] == -X["diff_prior_skill"]).all()
    assert (m["fighter_a_prior_is_debut"] == X["fighter_b_prior_is_debut"]).all()
    assert (m["fighter_b_prior_is_debut"] == X["fighter_a_prior_is_debut"]).all()
    assert m["same_stance"].equals(X["same_stance"])


def test_mirroring_twice_gives_back_the_original():
    X, _ = _toy(20)
    pd.testing.assert_frame_equal(mirror(mirror(X)), X)


def test_mirror_refuses_a_feature_it_cannot_place():
    """A new feature must be declared corner-free, a diff, or a pair - never guessed."""
    X, _ = _toy(5)
    with pytest.raises(ValueError, match="don't know how to mirror"):
        mirror(X.assign(mystery=1.0))


def test_mirror_refuses_a_side_column_without_its_partner():
    X, _ = _toy(5)
    with pytest.raises(ValueError, match="no fighter_b_prior_is_debut"):
        mirror(X.drop(columns=["fighter_b_prior_is_debut"]))


def test_to_matrix_turns_everything_into_plain_floats():
    X, _ = _toy(5)
    X.loc[0, "same_stance"] = pd.NA
    out = to_matrix(X)
    assert (out.dtypes == "float64").all()
    assert np.isnan(out.loc[0, "same_stance"])


def test_predictions_agree_whichever_fighter_is_called_a():
    X, y = _toy()
    model = fit_model(X, y)
    np.testing.assert_allclose(predict_proba(model, X) + predict_proba(model, mirror(X)), 1.0)


def test_corner_disagreement_is_a_share():
    X, y = _toy()
    assert 0.0 <= corner_disagreement(fit_model(X, y), X) <= 1.0


def test_fitting_twice_gives_identical_predictions():
    X, y = _toy()
    np.testing.assert_array_equal(predict_proba(fit_model(X, y), X),
                                  predict_proba(fit_model(X, y), X))


def test_model_picks_up_the_signal():
    X, y = _toy(2_000)
    p = predict_proba(fit_model(X, y), X)
    assert np.corrcoef(p, X["diff_prior_skill"])[0, 1] > 0.5
