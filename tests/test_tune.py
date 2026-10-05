import numpy as np
import optuna
import pandas as pd
import pytest

from ufc_predictor.modeling.model import MODEL_PARAMS, make_model
from ufc_predictor.modeling.tune import (
    check_no_test_years,
    choose_elo_k,
    cv_log_loss,
    expanding_folds,
    load_params,
    out_of_fold_predictions,
    save_params,
    suggest_params,
    tune,
    tuning_table,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _toy(n_per_year: int = 120, years=(2014, 2015, 2016, 2017), seed: int = 0) -> pd.DataFrame:
    """A featured-looking table: a hidden skill gap decides most fights.

    Shaped like the real one as `make_xy` expects it - a label column, a
    couple of diff features and a mirrorable pair - so the fold machinery is
    exercised against the same contract the real table meets.

    The column names matter: `make_xy` builds `X` from an allow-list keyed on
    stat stems, so a made-up name like `diff_prior_skill` is silently not a
    feature at all. `win_rate` is a real RECORD_STEM, so this frame really
    does reach the model.
    """
    rng = np.random.default_rng(seed)
    n = n_per_year * len(years)
    skill = rng.normal(0, 1, n)
    return pd.DataFrame({
        "event_date": pd.to_datetime(
            [f"{y}-{1 + i % 12:02d}-10" for y in years for i in range(n_per_year)]
        ),
        "fighter_a_won": pd.array(rng.random(n) < 1 / (1 + np.exp(-2 * skill)), dtype="boolean"),
        "diff_prior_win_rate_career_avg": skill,
        "diff_static_age": rng.normal(0, 5, n),
        "fighter_a_prior_is_debut": rng.random(n) < 0.2,
        "fighter_b_prior_is_debut": rng.random(n) < 0.2,
    })


def test_the_toy_frame_really_reaches_the_model():
    """Guards the fixture itself: a mistyped feature name would make every
    other test in this file pass against an empty-signal `X`."""
    from ufc_predictor.features.manifest import make_xy
    X, _ = make_xy(_toy(10, years=(2014,)))
    assert "diff_prior_win_rate_career_avg" in X.columns


# ------------------------------------------------------------------ the folds

def test_folds_only_ever_train_on_earlier_years():
    """The whole point: never learn from the future to predict the past."""
    fights = _toy()
    for train_rows, score_rows in expanding_folds(fights, n_folds=3):
        trained_on = fights.iloc[train_rows]["event_date"]
        scored_on = fights.iloc[score_rows]["event_date"]
        assert trained_on.max() < scored_on.min()


def test_folds_expand_and_score_each_year_once():
    fights = _toy()
    folds = expanding_folds(fights, n_folds=3)

    assert len(folds) == 3
    scored_years = [fights.iloc[rows]["event_date"].dt.year.unique().tolist() for _, rows in folds]
    assert scored_years == [[2015], [2016], [2017]]
    # each fold trains on everything before its scoring year, so they grow
    assert [len(train) for train, _ in folds] == [120, 240, 360]


def test_a_fold_never_scores_a_fight_it_trained_on():
    fights = _toy()
    for train_rows, score_rows in expanding_folds(fights, n_folds=3):
        assert not set(train_rows) & set(score_rows)


def test_folds_refuse_to_use_every_year_as_a_scoring_year():
    """4 folds over 4 years would leave the first one with nothing to train on."""
    with pytest.raises(ValueError, match="more than 4 years"):
        expanding_folds(_toy(), n_folds=4)


def test_folds_need_at_least_two():
    with pytest.raises(ValueError, match="at least 2 folds"):
        expanding_folds(_toy(), n_folds=1)


# ---------------------------------------------------------------- the guard

def test_tuning_refuses_data_reaching_the_test_years():
    fights = _toy(years=(2019, 2020, 2021, 2022))
    with pytest.raises(ValueError, match="test years"):
        check_no_test_years(fights)
    with pytest.raises(ValueError, match="test years"):
        tune(fights, n_trials=1)


def test_training_years_pass_the_guard():
    check_no_test_years(_toy(years=(2014, 2015, 2016)))   # does not raise


# ---------------------------------------------------------------- the scoring

def test_cv_log_loss_reports_one_entry_per_fold_and_counts_every_fight_once():
    fights = _toy()
    out = cv_log_loss(fights, MODEL_PARAMS, n_folds=3)

    assert len(out["folds"]) == 3
    assert [f["scoring_year"] for f in out["folds"]] == [2015, 2016, 2017]
    assert out["scored_fights"] == sum(f["scored_fights"] for f in out["folds"]) == 360
    assert 0.0 < out["log_loss"] < 2.0


def test_cv_log_loss_beats_a_coin_flip_when_there_is_real_signal():
    """A sanity check on the whole loop: 0.693 is what guessing 50% scores."""
    assert cv_log_loss(_toy(n_per_year=400), MODEL_PARAMS, n_folds=3)["log_loss"] < 0.69


def test_cv_log_loss_is_deterministic():
    fights = _toy()
    assert (cv_log_loss(fights, MODEL_PARAMS, n_folds=3)["log_loss"]
            == cv_log_loss(fights, MODEL_PARAMS, n_folds=3)["log_loss"])


SUBSAMPLED = MODEL_PARAMS | {"max_features": 0.5}   # where the model seed actually matters


def test_several_seeds_score_the_average_of_single_seed_runs():
    fights = _toy()
    singles = [cv_log_loss(fights, SUBSAMPLED, n_folds=3, seeds=(s,))["log_loss"] for s in (0, 1, 2)]
    together = cv_log_loss(fights, SUBSAMPLED, n_folds=3, seeds=(0, 1, 2))
    assert together["log_loss"] == pytest.approx(np.mean(singles))
    assert together["log_loss_by_seed"] == pytest.approx(singles)


def test_the_model_seed_matters_when_features_are_subsampled():
    """Why seeds exist at all: with max_features < 1, two seeds give two scores."""
    fights = _toy()
    a, b = (cv_log_loss(fights, SUBSAMPLED, n_folds=3, seeds=(s,))["log_loss"] for s in (0, 1))
    assert a != b


def test_out_of_fold_predictions_cover_every_scored_fight_once():
    fights = _toy()
    y, p, year = out_of_fold_predictions(fights, SUBSAMPLED, seeds=(0, 1), n_folds=3)
    assert len(y) == len(p) == len(year) == 360
    assert sorted(set(year)) == [2015, 2016, 2017]
    y2, p2, _ = out_of_fold_predictions(fights, SUBSAMPLED, seeds=(0, 1), n_folds=3)
    np.testing.assert_array_equal(p, p2)


def test_out_of_fold_predictions_match_cv_log_loss():
    """Same folds, same model: one seed's predictions score exactly what cv_log_loss reports."""
    fights = _toy()
    y, p, _ = out_of_fold_predictions(fights, SUBSAMPLED, seeds=(0,), n_folds=3)
    from sklearn.metrics import log_loss
    expected = cv_log_loss(fights, SUBSAMPLED, n_folds=3, seeds=(0,))["log_loss"]
    assert log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[False, True]) == pytest.approx(expected)


# ------------------------------------------------------------- the search

def test_tune_is_reproducible():
    """Same seed, same answer - otherwise a saved model cannot be rebuilt."""
    fights = _toy()
    first = tune(fights, n_trials=3, n_folds=2, seed=7, model_seeds=(0,))
    second = tune(fights, n_trials=3, n_folds=2, seed=7, model_seeds=(0,))
    assert first.best_params == second.best_params
    assert first.best_value == second.best_value


def test_tune_tries_the_hand_chosen_settings_first():
    """Trial 0 is MODEL_PARAMS, so the search can always be compared against it."""
    study = tune(_toy(), n_trials=2, n_folds=2, model_seeds=(0,))
    trial_zero = study.trials[0].params
    assert trial_zero["learning_rate"] == MODEL_PARAMS["learning_rate"]
    assert trial_zero["max_iter"] == MODEL_PARAMS["max_iter"]
    assert trial_zero["max_leaf_nodes"] == MODEL_PARAMS["max_leaf_nodes"]
    assert trial_zero["max_features"] == 1.0


def test_tune_returns_the_best_trial_it_saw():
    study = tune(_toy(), n_trials=4, n_folds=2, model_seeds=(0,))
    assert study.best_value == min(t.value for t in study.trials if t.value is not None)


def test_suggested_settings_build_a_usable_model():
    """Every corner of the search space must be a model sklearn accepts."""
    def objective(trial):
        make_model(suggest_params(trial))
        return 0.0

    optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0)).optimize(objective, n_trials=20)


def test_tuning_table_is_sorted_best_first():
    table = tuning_table(tune(_toy(), n_trials=4, n_folds=2, model_seeds=(0,)))
    assert table["log_loss"].is_monotonic_increasing
    assert "max_features" in table.columns


# ------------------------------------------------------------- choosing Elo's K

def _toy_league(n_fighters=30, per_year=60, years=range(2014, 2020), seed=0) -> pd.DataFrame:
    """Fights where a hidden strength decides most results - what Elo can pick up."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 1, n_fighters)
    rows = []
    for year in years:
        for i in range(per_year):
            a, b = (int(x) for x in rng.choice(n_fighters, size=2, replace=False))
            a_won = bool(rng.random() < 1 / (1 + np.exp(strength[b] - strength[a])))
            rows.append({
                "event_date": pd.Timestamp(f"{year}-{1 + i % 12:02d}-{1 + i // 12:02d}"),
                "fights_url": f"{year}/{i:03d}",
                "fighter_a_id": a, "fighter_b_id": b,
                "outcome": "fighter_a" if a_won else "fighter_b",
                "fighter_a_won": a_won,
                "diff_static_age": rng.normal(0, 5),        # a feature with no signal
            })
    table = pd.DataFrame(rows)
    table["fighter_a_won"] = table["fighter_a_won"].astype("boolean")
    return table


def test_choose_elo_k_is_deterministic_and_reports_every_k():
    table = _toy_league()
    first = choose_elo_k(table, MODEL_PARAMS, grid=(16, 64, 256), seeds=(0,), n_folds=2)
    second = choose_elo_k(table, MODEL_PARAMS, grid=(16, 64, 256), seeds=(0,), n_folds=2)
    best, curve = first
    assert first == second
    assert [k for k, _ in curve] == [16, 64, 256]
    assert best == min(curve, key=lambda point: point[1])[0]


def test_the_elo_column_really_reaches_the_model():
    """If diff_prior_elo were silently left out of X, every K would score the same."""
    _, curve = choose_elo_k(_toy_league(), MODEL_PARAMS, grid=(16, 64, 256), seeds=(0,), n_folds=2)
    assert len({score for _, score in curve}) == 3


# ------------------------------------------------------------- saving

def test_saved_params_round_trip(tmp_path, monkeypatch):
    import ufc_predictor.data.io as io_module
    monkeypatch.setattr(io_module, "PROCESSED_DIR", tmp_path)

    study = tune(_toy(), n_trials=2, n_folds=2, model_seeds=(0, 1))
    save_params(study)

    # the file must land in the temporary folder. Without this line the test
    # once passed while silently overwriting the real data/processed/model_params.json
    assert (tmp_path / "model_params.json").exists()
    assert load_params() == study.best_params
    saved = io_module.load_processed_json("model_params")
    assert saved["cv_log_loss"] == study.best_value
    assert saved["chosen_by"]["n_folds"] == 2
    assert saved["chosen_by"]["model_seeds"] == [0, 1]


# ------------------------------------------- the one change made to model.py

def test_params_override_the_defaults_without_losing_the_fixed_ones():
    model = make_model({"max_iter": 7})
    assert model.max_iter == 7
    assert model.learning_rate == MODEL_PARAMS["learning_rate"]   # untouched default
    assert model.early_stopping is False                          # never reverts to "auto"
    assert model.random_state == 0


def test_make_model_without_params_is_unchanged():
    plain, explicit = make_model(), make_model(None)
    assert plain.get_params() == explicit.get_params()
    for key, value in MODEL_PARAMS.items():
        assert plain.get_params()[key] == value
