import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from ufc_predictor.modeling.explain import (
    _partner_index, group_interactions, group_shap, shap_interaction_values, shap_values, symmetric_score,
)
from ufc_predictor.modeling.model import fit_models, mirror, predict_proba_mean

SMALL = {"max_iter": 40, "max_features": 0.6}


def _toy(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Fights decided mostly by a skill gap, a little by a debut; one column is always 0."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1, n)
    a_debut, b_debut = rng.random(n) < 0.2, rng.random(n) < 0.2
    X = pd.DataFrame({
        "diff_prior_skill": skill,
        "diff_static_age": rng.normal(0, 5, n),
        "diff_prior_nothing": np.zeros(n),
        "fighter_a_prior_is_debut": a_debut,
        "fighter_b_prior_is_debut": b_debut,
        "same_stance": pd.array(rng.random(n) < 0.6, dtype="boolean"),
    })
    logit = 2 * skill - 0.8 * a_debut + 0.8 * b_debut
    y = pd.Series(rng.random(n) < 1 / (1 + np.exp(-logit)))
    return X, y


@pytest.fixture(scope="module")
def fitted():
    X, y = _toy()
    return X, fit_models(X, y, params=SMALL, seeds=range(3))


def test_values_add_up_to_the_symmetric_score(fitted):
    X, models = fitted
    e = shap_values(models, X)
    assert e.values.shape == X.shape
    assert np.allclose(e.values.sum(axis=1) + e.base_values, symmetric_score(models, X))


def test_the_baseline_is_exactly_50_50(fitted):
    """The corner lean of a single model call cancels: no information means 0 log-odds."""
    X, models = fitted
    assert (shap_values(models, X).base_values == 0).all()


def test_swapping_corners_negates_the_explanation(fitted):
    """The same fight seen from the other corner: every piece flips sign, and
    fighter A's debut flag swaps places with fighter B's."""
    X, models = fitted
    e = shap_values(models, X).values
    e_swapped = shap_values(models, mirror(X)).values
    assert np.allclose(e_swapped, -e[:, _partner_index(X.columns)])


def test_a_column_that_never_varies_gets_nothing(fitted):
    X, models = fitted
    e = shap_values(models, X)
    assert np.allclose(e.values[:, list(X.columns).index("diff_prior_nothing")], 0)


def test_the_signal_gets_the_credit(fitted):
    """Skill decides these fights, so it should carry by far the most weight -
    and pushing towards A exactly when A is the more skilled fighter."""
    X, models = fitted
    e = shap_values(models, X)
    importance = np.abs(e.values).mean(axis=0)
    skill = list(X.columns).index("diff_prior_skill")
    assert importance.argmax() == skill
    assert np.corrcoef(X["diff_prior_skill"], e.values[:, skill])[0, 1] > 0.9


def test_the_explained_score_matches_the_real_prediction(fitted):
    """The model averages probabilities, SHAP explains averaged scores; they must stay close."""
    X, models = fitted
    S = symmetric_score(models, X)
    p = predict_proba_mean(models, X)
    # the gap grows with confidence; these toy fights reach 90%, real ones rarely
    # pass 75% (measured on 2023-24 validation: at most 0.008)
    assert np.abs(expit(S) - p).max() < 0.05
    assert ((S > 0) == (p > 0.5)).all()


def test_one_model_is_its_own_average(fitted):
    X, models = fitted
    alone = shap_values(models[:1], X).values
    together = shap_values(models, X).values
    each = [shap_values([m], X).values for m in models]
    assert not np.allclose(alone, together)            # the seeds really differ
    assert np.allclose(together, np.mean(each, axis=0))


def test_groups_add_up_exactly(fitted):
    X, models = fitted
    e = shap_values(models, X)
    groups = {"skill": ["diff_prior_skill", "diff_prior_nothing"],
              "physical": ["diff_static_age", "same_stance"],
              "cold_start": ["fighter_a_prior_is_debut", "fighter_b_prior_is_debut"]}
    g = group_shap(e, groups)
    assert g.feature_names == ["skill", "physical", "cold_start"]
    assert np.allclose(g.values.sum(axis=1), e.values.sum(axis=1))
    debut = [list(X.columns).index(c) for c in groups["cold_start"]]
    assert np.allclose(g.values[:, 2], e.values[:, debut].sum(axis=1))


@pytest.mark.parametrize("groups", [
    {"skill": ["diff_prior_skill"]},                                                    # most features missing
    {"all": ["diff_prior_skill", "diff_static_age", "diff_prior_nothing", "fighter_a_prior_is_debut",
             "fighter_b_prior_is_debut", "same_stance"], "again": ["diff_prior_skill"]},   # one counted twice
])
def test_groups_must_cover_every_feature_once(fitted, groups):
    X, models = fitted
    with pytest.raises(ValueError, match="exactly once"):
        group_shap(shap_values(models[:1], X), groups)


GROUPS = {"skill": ["diff_prior_skill", "diff_prior_nothing"],
          "physical": ["diff_static_age", "same_stance"],
          "cold_start": ["fighter_a_prior_is_debut", "fighter_b_prior_is_debut"]}


@pytest.fixture(scope="module")
def interactions(fitted):
    X, models = fitted
    return shap_interaction_values(models, X)


def test_interaction_rows_add_up_to_the_shap_values(fitted, interactions):
    X, models = fitted
    assert interactions.shape == (len(X), X.shape[1], X.shape[1])
    assert np.allclose(interactions.sum(axis=2), shap_values(models, X).values)
    assert np.allclose(interactions, interactions.transpose(0, 2, 1))       # [i, j] == [j, i]


def test_swapping_corners_negates_the_interactions(fitted, interactions):
    """The swapped fight's table: every entry flips sign, A's and B's debut flags trade places on both axes."""
    X, models = fitted
    p = _partner_index(X.columns)
    assert np.allclose(shap_interaction_values(models, mirror(X)), -interactions[:, p][:, :, p])


def test_a_column_that_never_varies_interacts_with_nothing(fitted, interactions):
    X, _ = fitted
    nothing = list(X.columns).index("diff_prior_nothing")
    assert np.allclose(interactions[:, nothing, :], 0) and np.allclose(interactions[:, :, nothing], 0)


def test_trees_with_one_split_cannot_interact():
    """A tree with two leaves looks at one feature, so every off-diagonal entry is exactly 0."""
    X, y = _toy()
    models = fit_models(X, y, params=SMALL | {"max_leaf_nodes": 2}, seeds=range(2))
    values = shap_interaction_values(models, X)
    off = values.copy()
    off[:, np.arange(X.shape[1]), np.arange(X.shape[1])] = 0
    assert np.allclose(off, 0)
    assert not np.allclose(values, 0)


def test_a_real_interaction_is_found():
    """Fights where skill counts more when the age gap is big (logit = skill x |age gap|):
    skill x age must be the strongest pair, and with a big age gap it pushes towards the
    more skilled fighter - the shape of "striking counts more when the defence gap is big"."""
    X, _ = _toy(n=2000, seed=1)
    rng = np.random.default_rng(1)
    logit = 1.5 * X["diff_prior_skill"] * np.abs(X["diff_static_age"]) / 5
    y = pd.Series(rng.random(len(X)) < 1 / (1 + np.exp(-logit)))
    models = fit_models(X, y, params={"max_iter": 150, "max_leaf_nodes": 8}, seeds=range(2))
    values = shap_interaction_values(models, X)
    names = list(X.columns)
    strength = np.abs(values).mean(axis=0)
    strength[np.arange(len(names)), np.arange(len(names))] = 0
    i, j = np.unravel_index(strength.argmax(), strength.shape)
    assert {names[i], names[j]} == {"diff_prior_skill", "diff_static_age"}
    skill, age = names.index("diff_prior_skill"), names.index("diff_static_age")
    push = 2 * values[:, skill, age] * np.sign(X["diff_prior_skill"])      # towards the more skilled fighter
    big_gap = np.abs(X["diff_static_age"]) > np.abs(X["diff_static_age"]).median()
    assert push[big_gap].mean() > 0 > push[~big_gap].mean()


def test_group_interactions_add_both_halves_and_skip_main_effects(fitted, interactions):
    X, _ = fitted
    table = group_interactions(interactions, X.columns, GROUPS)
    assert list(table.index) == list(table.columns) == list(GROUPS)
    assert np.allclose(table, table.T)
    names = list(X.columns)
    skill, age = names.index("diff_prior_skill"), names.index("diff_static_age")
    stance, nothing = names.index("same_stance"), names.index("diff_prior_nothing")
    by_hand = (interactions[:, skill, age] + interactions[:, age, skill]
               + interactions[:, skill, stance] + interactions[:, stance, skill]
               + interactions[:, nothing, age] + interactions[:, age, nothing]
               + interactions[:, nothing, stance] + interactions[:, stance, nothing])
    assert np.isclose(table.loc["skill", "physical"], np.abs(by_hand).mean())
    within = interactions[:, age, stance] + interactions[:, stance, age]
    assert np.isclose(table.loc["physical", "physical"], np.abs(within).mean())


def test_group_interactions_need_every_feature_once(fitted, interactions):
    X, _ = fitted
    with pytest.raises(ValueError, match="exactly once"):
        group_interactions(interactions, X.columns, {"skill": ["diff_prior_skill"]})
