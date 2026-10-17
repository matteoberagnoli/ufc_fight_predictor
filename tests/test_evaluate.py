import numpy as np
import pandas as pd
import pytest

from ufc_predictor.modeling.evaluate import (
    breakdown,
    calibration_table,
    compare,
    compare_log_loss,
    fit_baseline,
    group_importance,
    has_debutant,
    is_womens_fight,
    per_fight_log_loss,
    predict_baseline,
    score,
    scorecard,
)
from ufc_predictor.modeling.model import fit_model


def test_a_coin_flip_scores_exactly_what_it_should():
    s = score([True, False, True, False], np.full(4, 0.5))
    assert s["accuracy"] == 0.5                       # every 50/50 call counts as half right
    assert s["log_loss"] == pytest.approx(np.log(2))  # 0.693
    assert s["auc"] == 0.5


def test_perfect_predictions_score_perfectly():
    s = score([True, False, True, False], [0.9, 0.2, 0.8, 0.1])
    assert s["accuracy"] == 1.0 and s["auc"] == 1.0


def test_confident_mistakes_cost_more_log_loss_than_timid_ones():
    y = [True, True]
    assert score(y, [0.1, 0.9])["log_loss"] > score(y, [0.4, 0.9])["log_loss"]


def test_auc_is_undefined_when_every_fight_ended_the_same_way():
    assert np.isnan(score([True, True], [0.1, 0.9])["auc"])


def test_scorecard_reports_mens_fights_separately():
    y = [True, False, True, False]
    mens = [True, True, False, False]
    card = scorecard({"model": [0.9, 0.1, 0.1, 0.9]}, y, mens)
    assert card.loc["model", "men_accuracy"] == 1.0   # both men's fights right
    assert card.loc["model", "all_accuracy"] == 0.5   # both women's fights wrong


def test_has_debutant_is_true_when_either_corner_debuts():
    fights = pd.DataFrame({"fighter_a_prior_is_debut": [True, False, False, True],
                           "fighter_b_prior_is_debut": [False, True, False, True]})
    assert has_debutant(fights).tolist() == [True, True, False, True]


def test_breakdown_scores_each_group_on_its_own_fights():
    y = [True, False, True, False, True]
    debutant = [False, False, False, True, True]
    p = [0.9, 0.1, 0.8, 0.9, 0.1]                     # right without debutants, wrong with them
    card = breakdown({"model": p, "coin flip": [0.5] * 5}, y, debutant)
    assert card.loc[("all", "model"), "fights"] == 5
    assert card.loc[("no debutant", "model"), "accuracy"] == 1.0
    assert card.loc[("with a debutant", "model"), "accuracy"] == 0.0
    assert card.loc[("with a debutant", "coin flip"), "log_loss"] == pytest.approx(0.6931, abs=1e-4)


def test_breakdown_keep_restricts_every_group():
    y = [True, False, True, False]
    debutant = [False, True, False, True]
    card = breakdown({"model": [0.9, 0.1, 0.1, 0.9]}, y, debutant, keep=[True, True, False, False])
    assert card.loc[("all", "model"), "fights"] == 2
    assert card.loc[("no debutant", "model"), "fights"] == 1
    assert card.loc[("with a debutant", "model"), "fights"] == 1


def test_compare_counts_fixes_and_breaks():
    y = [True, True, False, False]
    out = compare(y, p_new=[0.9, 0.9, 0.9, 0.1], p_old=[0.1, 0.9, 0.1, 0.1])
    assert (out["fixes"], out["breaks"]) == (1, 1)


def test_catchweight_bouts_between_women_count_as_womens_fights():
    history = pd.DataFrame({
        "weight_class": ["Women's Strawweight Bout", "Catch Weight Bout", "Lightweight Bout"],
        "fighter_a_id": [1, 1, 5],
        "fighter_b_id": [2, 3, 6],
    })
    assert is_womens_fight(history).tolist() == [True, True, False]
    # on a slice, the career that proves fighter 1 is a woman is only visible via history
    catchweight = history.iloc[[1]]
    assert not is_womens_fight(catchweight).iloc[0]
    assert is_womens_fight(catchweight, history=history).iloc[0]


def test_baseline_has_no_corner_preference():
    X = pd.DataFrame({"diff_static_age": [-5.0, -2, 3, 6, -4, 1, 2, -1]})
    y = [True, True, False, False, False, True, True, False]
    model = fit_baseline(X, y, "diff_static_age")
    p = predict_baseline(model, X, "diff_static_age")
    q = predict_baseline(model, -X, "diff_static_age")
    np.testing.assert_allclose(p + q, 1.0)


def test_baseline_treats_a_missing_value_as_no_difference():
    X = pd.DataFrame({"diff_static_age": [-5.0, -2, 3, 6, -4, 1, 2, -1]})
    y = [True, True, False, False, False, True, True, False]
    model = fit_baseline(X, y, "diff_static_age")
    p = predict_baseline(model, pd.DataFrame({"diff_static_age": [np.nan]}), "diff_static_age")
    assert p[0] == pytest.approx(0.5)


def _skill_and_noise(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1, n)
    X = pd.DataFrame({"diff_prior_skill": skill, "diff_prior_noise": rng.normal(0, 1, n)})
    return X, pd.Series(rng.random(n) < 1 / (1 + np.exp(-2 * skill)))


def test_group_importance_finds_the_family_that_matters():
    X_train, y_train = _skill_and_noise(1_500, seed=1)
    X_held_out, y_held_out = _skill_and_noise(1_500, seed=2)      # fights it never trained on
    groups = {"skill": ["diff_prior_skill"], "noise": ["diff_prior_noise"]}
    table = group_importance(fit_model(X_train, y_train), X_held_out, y_held_out, groups, repeats=5)
    rises = table.set_index("group")["log_loss_rise"]
    assert rises["skill"] > 0.3          # scrambling the real signal hurts a lot
    assert abs(rises["noise"]) < 0.02    # scrambling noise barely registers


def test_group_importance_on_training_fights_mostly_measures_memorising():
    """Why importance must be measured on held-out fights: on training fights,
    the noise column looks important because the model memorised it."""
    X, y = _skill_and_noise(1_500, seed=1)
    table = group_importance(fit_model(X, y), X, y, {"noise": ["diff_prior_noise"]}, repeats=5)
    assert table["log_loss_rise"].iloc[0] > 0.02


def test_calibration_table_covers_every_fight_in_rising_order():
    rng = np.random.default_rng(0)
    p = rng.random(1_000)
    y = rng.random(1_000) < p                          # honest by construction
    table = calibration_table(y, p, bins=5)
    assert table["fights"].sum() == 1_000
    assert table["predicted"].is_monotonic_increasing
    assert (table["actual"] - table["predicted"]).abs().max() < 0.1


def test_per_fight_log_loss_averages_to_log_loss():
    y = np.array([True, False, True, True])
    p = np.array([0.7, 0.4, 0.2, 0.55])
    assert per_fight_log_loss(y, p).mean() == pytest.approx(score(y, p)["log_loss"])
    assert per_fight_log_loss(y, p)[0] == pytest.approx(-np.log(0.7))
    assert per_fight_log_loss(y, p)[1] == pytest.approx(-np.log(0.6))   # B won: A's 40% means 60% on B


def test_compare_log_loss_of_identical_predictions_is_no_difference():
    rng = np.random.default_rng(0)
    y, p = rng.random(200) < 0.5, rng.uniform(0.3, 0.7, 200)
    result = compare_log_loss(y, p, p)
    assert result["difference"] == 0 and result["p_value"] == 1.0


def test_compare_log_loss_difference_is_the_log_loss_gap_and_flips_sign():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.2, 0.8, 500)
    y = rng.random(500) < truth
    sharp, vague = truth, np.full(500, 0.5)
    better = compare_log_loss(y, sharp, vague)
    assert better["difference"] == pytest.approx(score(y, sharp)["log_loss"] - score(y, vague)["log_loss"])
    assert better["difference"] < 0 and better["p_value"] < 0.01
    worse = compare_log_loss(y, vague, sharp)
    assert worse["difference"] == pytest.approx(-better["difference"])
    assert worse["p_value"] == pytest.approx(better["p_value"])


def test_compare_log_loss_sees_what_compare_cannot():
    """More confident on the same picks: no pick changes, so McNemar sees nothing,
    but log loss - and this test - do."""
    rng = np.random.default_rng(2)
    truth = rng.uniform(0.55, 0.85, 400)
    y = rng.random(400) < truth
    timid, bold = np.full(400, 0.55), truth
    assert compare(y, bold, timid) == {"fixes": 0, "breaks": 0, "p_value": 1.0}
    assert compare_log_loss(y, bold, timid)["p_value"] < 0.05
