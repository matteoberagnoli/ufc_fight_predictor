"""Searching for better model settings, without spending the validation years.

`model.MODEL_PARAMS` was chosen by judgement and never measured. This module
measures. It uses Optuna, which spends its guesses where the good scores have
been rather than walking a fixed grid - with ~6 settings to choose at once, a
grid fine enough to be useful would be far too big to run.

Two decisions shape everything here.

**Where each trial is scored.** Not on the validation years. Those are
~1,000 fights, which carry about +/-1.6 points of accuracy noise, so trying
60 settings against them and keeping the best would mostly find whichever
setting got lucky on those particular fights - and would spend the one clean
check we have left before the test years. Instead each trial is scored by
*expanding-window cross-validation inside the training years*:

    fold 1   train 2014-2018        ->  score on 2019
    fold 2   train 2014-2019        ->  score on 2020
    fold 3   train 2014-2020        ->  score on 2021
    fold 4   train 2014-2021        ->  score on 2022

(the last four training years, whatever the split in config.py makes them).
Always past -> future, never the reverse, because a model that trains on 2022
to predict 2019 is being asked an easier question than the real one. Four
folds give ~1,900 scored fights instead of ~1,000, and a setting has to work
in four different years rather than one, so a fluke is far less likely to
win. The validation years then stay untouched, available to check the tuned
model honestly.

**Log loss, not accuracy.** Accuracy only moves when a fight crosses the 50%
line, so it steps in chunks and two genuinely different settings often tie.
Log loss uses the full probability of every fight, so it responds to a model
getting *more sure* about fights it already called right. It is the number
`evaluate.score` already treats as the headline.

Scoring uses `model.predict_proba`, which averages the prediction over both
corner orderings - the same way the model is used everywhere else. Tuning
against a differently-made prediction than the one you ship would optimise
the wrong thing.

One honest caveat: the shrinkage alphas in `alphas.json` were fitted on
the training years as a whole, so a fold scored on 2019 uses features built
with an alpha that saw 2020-2022. That is a small leak - one scalar per feature,
shared by every trial, so it cannot favour one setting over another - but it
means fold scores here read very slightly better than a true 2016 forecast
would. Refitting alphas per fold would remove it, at several times the
runtime.
"""

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

from ufc_predictor.config import TEST_START, VALIDATION_START
from ufc_predictor.data.io import load_processed_json, save_processed_json
from ufc_predictor.features.elo import ELO_K_GRID, attach_elo_features
from ufc_predictor.features.manifest import make_xy
from ufc_predictor.modeling.model import MODEL_PARAMS, fit_model, predict_proba
from ufc_predictor.modeling.split import chronological_split

# Fixed so a tuning run can be repeated and give the same answer. Optuna's
# search is random; without a seed, re-running would quietly pick different
# settings and the model would stop being reproducible.
TUNE_SEED = 1926

N_FOLDS = 4

# Model seeds each tuning trial is averaged over. Three is a compromise: the
# seed-to-seed spread shrinks by ~1.7x, at three times the runtime.
TUNE_MODEL_SEEDS = (0, 1, 2)


def expanding_folds(fights: pd.DataFrame, n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split training fights into (train rows, score rows) pairs, past -> future.

    The last `n_folds` calendar years each become a scoring year in turn, and
    every fight before that year is the training part. Returns positional
    indices into `fights`.

    More folds means more scored fights, but the earliest fold then trains on
    less: with 4 folds the first one learns from ~2,350 fights (2014-2018), which may
    flatter settings that suit small data. 3 folds trades scored fights for
    more realistic training sizes.
    """
    if n_folds < 2:
        raise ValueError(f"need at least 2 folds to compare settings, got {n_folds}")

    years = sorted(fights["event_date"].dt.year.unique())
    if len(years) <= n_folds:
        raise ValueError(
            f"{n_folds} folds needs more than {n_folds} years of fights, "
            f"but the data covers {len(years)}: {years}"
        )

    year = fights["event_date"].dt.year.to_numpy()
    folds = []
    for scoring_year in years[-n_folds:]:
        train = np.flatnonzero(year < scoring_year)
        score = np.flatnonzero(year == scoring_year)
        folds.append((train, score))
    return folds


def check_no_test_years(fights: pd.DataFrame, test_start: str = TEST_START) -> None:
    """Refuse to tune on anything from the test years.

    Tuning is a decision made by looking at scores, so any fight it sees is
    spent. The test years must stay unseen until the model is final, and this
    is the guard that makes that structural rather than a thing to remember.
    """
    latest = fights["event_date"].max()
    if latest >= pd.Timestamp(test_start):
        raise ValueError(
            f"tuning data runs to {latest.date()}, which reaches the test years "
            f"(from {test_start}). Pass the training split only."
        )


def suggest_params(trial: optuna.Trial) -> dict:
    """One candidate set of settings, drawn by Optuna.

    Ranges are centred on the hand-chosen MODEL_PARAMS and wide enough on both
    sides that the search can tell us those were wrong. `max_iter` and
    `learning_rate` are searched together because they trade off directly:
    halve the learning rate and you need roughly twice the trees.

    `max_features` is the one setting the first model never used. At 1.0 every
    tree weighs all 107 features at every split, so the trees end up alike -
    each one reaching for the same strongest feature first. Below 1.0 each
    split sees a random subset, which forces variety between trees. With far
    more features than the ~4,300 training rows can firmly support, this is the setting
    most likely to help.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 1000, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 63, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 120, log=True),
        # from "effectively off" to "strong"; log scale because the difference
        # between 0.01 and 0.1 matters as much as between 1 and 10
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 30.0, log=True),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


def _clipped_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[False, True]))


def _one_seed(fights: pd.DataFrame, params: dict,
              folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, list]:
    """Every fold once, with one fixed model seed: (actual, predicted, per-fold log loss)."""
    actual, predicted, per_fold = [], [], []
    for train_rows, score_rows in folds:
        X_train, y_train = make_xy(fights.iloc[train_rows])
        X_score, y_score = make_xy(fights.iloc[score_rows])

        p = predict_proba(fit_model(X_train, y_train, params=params), X_score)
        y = np.asarray(y_score, dtype=bool)
        actual.append(y)
        predicted.append(p)
        per_fold.append(_clipped_log_loss(y, p))
    return np.concatenate(actual), np.concatenate(predicted), per_fold


def cv_log_loss(fights: pd.DataFrame, params: dict,
                folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
                n_folds: int = N_FOLDS, seeds=(0,)) -> dict:
    """Score one set of settings by expanding-window CV. Lower is better.

    `log_loss` is pooled: every scored fight across every fold counts once,
    so a fold with more fights carries more weight - which is right, since
    those extra fights are extra evidence, not a separate question.

    `seeds` are model seeds (`random_state`, overriding any in `params`). With
    `max_features` below 1 each split looks at a random subset of features, so
    the same settings score differently from seed to seed - measured: a
    spread of about 0.001 across 8 seeds, as large as many of the differences
    worth testing. The score is the average over `seeds`, so comparing two
    things on several seeds compares the things, not the luck of one draw.
    """
    folds = expanding_folds(fights, n_folds=n_folds) if folds is None else folds

    runs = [_one_seed(fights, params | {"random_state": s}, folds) for s in seeds]
    by_seed = [_clipped_log_loss(y, p) for y, p, _ in runs]
    accuracy = [float(((p > 0.5) == y).mean()) for y, p, _ in runs]

    per_fold = []
    for i, (train_rows, score_rows) in enumerate(folds):
        per_fold.append({
            "scoring_year": int(fights.iloc[score_rows]["event_date"].dt.year.iloc[0]),
            "train_fights": len(train_rows),
            "scored_fights": len(score_rows),
            "log_loss": float(np.mean([fold_scores[i] for _, _, fold_scores in runs])),
        })

    return {
        "log_loss": float(np.mean(by_seed)),
        "log_loss_by_seed": by_seed,
        "accuracy": float(np.mean(accuracy)),
        "scored_fights": int(len(runs[0][0])),
        "folds": per_fold,
    }


def out_of_fold_predictions(fights: pd.DataFrame, params: dict, seeds=range(8),
                            n_folds: int = N_FOLDS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(actual, predicted, year) for every scored fight of the expanding-window CV.

    Each prediction comes from a model that never saw that fight's year, and
    is averaged over `seeds` - the same kind of prediction the final model
    makes, which is what a calibrator has to be fitted on. Predictions on a
    model's own training fights look more confident and more correct than
    they really are, so they would teach a calibrator the wrong lesson.
    """
    check_no_test_years(fights)
    folds = expanding_folds(fights, n_folds=n_folds)
    runs = [_one_seed(fights, params | {"random_state": s}, folds) for s in seeds]
    actual = runs[0][0]
    predicted = np.mean([p for _, p, _ in runs], axis=0)
    year = np.concatenate([fights.iloc[rows]["event_date"].dt.year.to_numpy() for _, rows in folds])
    return actual, predicted, year


def tune(fights: pd.DataFrame, n_trials: int = 60, n_folds: int = N_FOLDS,
         seed: int = TUNE_SEED, model_seeds=TUNE_MODEL_SEEDS,
         timeout: float | None = None, show_progress: bool = False) -> optuna.Study:
    """Search for the settings with the lowest cross-validated log loss.

    Pass the *training* split only - `check_no_test_years` refuses anything
    reaching the test years, and scoring on the validation years is
    deliberately not what happens here (see the module docstring).

    `seed` fixes Optuna's own search; `model_seeds` are the model seeds each
    trial is averaged over. Scored on one model seed, the search rewards
    settings that got a lucky draw: the first tuning run's 0.6705 turned out
    to be the best of 8 seeds for those settings, whose average is 0.6722.

    A few seconds per trial per model seed, so 60 trials on 3 seeds is
    roughly 10-15 minutes. `timeout` (seconds) stops the search early and
    keeps the best found so far.
    """
    check_no_test_years(fights)
    folds = expanding_folds(fights, n_folds=n_folds)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        return cv_log_loss(fights, params, folds=folds, seeds=model_seeds)["log_loss"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name="hist_gradient_boosting",
    )
    # kept on the study so save_params can record what the score was averaged over
    study.set_user_attr("model_seeds", list(model_seeds))
    study.set_user_attr("n_folds", n_folds)
    # the hand-chosen settings go in as trial 0, so the search starts from a
    # known-decent point and the report can always say whether tuning beat it
    study.enqueue_trial({
        key: MODEL_PARAMS[key] for key in
        ("learning_rate", "max_iter", "max_leaf_nodes", "l2_regularization")
    } | {"min_samples_leaf": 20, "max_features": 1.0})

    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   show_progress_bar=show_progress)
    return study


def choose_elo_k(featured: pd.DataFrame, params: dict, grid=ELO_K_GRID,
                 seeds=range(5), n_folds: int = N_FOLDS) -> tuple[float, list]:
    """Pick Elo's K by how much the rating helps the model: (best K, [(K, score), ...]).

    For each K the Elo columns are attached to the full featured table (all
    fights, so the ratings carry the whole history), the training split is
    taken, and the model is scored by expanding-window CV averaged over
    `seeds`. Lower is better.

    Judged by the model, not by Elo's own forecasts, because the two disagree:
    the K that forecasts best alone (~63) added nothing to the model, which
    already knows career-level quality from win rate and 100+ stats. What it
    lacked was a fast rating - recent results weighted by opponent quality.

    The curve is returned as well as the winner: it is a broad plateau, and
    neighbouring K values within ~0.001 of each other are a tie, not a ranking.
    Choosing K on the folds that later score it flatters it slightly - the
    same caveat as the alphas - so the validation years get the last word.
    """
    curve = []
    folds = None
    for k in grid:
        train, _, _ = chronological_split(attach_elo_features(featured, k))
        check_no_test_years(train)
        folds = expanding_folds(train, n_folds=n_folds) if folds is None else folds
        curve.append((float(k), cv_log_loss(train, params, folds=folds, seeds=seeds)["log_loss"]))
    best_k = min(curve, key=lambda point: point[1])[0]
    return best_k, curve


def tuning_table(study: optuna.Study, top: int = 10) -> pd.DataFrame:
    """The best trials, best first - for seeing how flat or peaked the search was.

    If the top ten scores are nearly identical, the settings barely matter and
    the "winner" is mostly noise; prefer the simplest of them.
    """
    rows = [
        {"trial": t.number, "log_loss": t.value, **t.params}
        for t in study.trials
        if t.value is not None
    ]
    return (pd.DataFrame(rows).sort_values("log_loss")
            .head(top).reset_index(drop=True).round(5))


def save_params(study: optuna.Study, validation_start: str = VALIDATION_START,
                processed_dir=None) -> None:
    """Save the winning settings with how they were found.

    A score is only meaningful alongside what it was measured on, so the fold
    scheme, model seeds and trial count are stored next to the settings
    rather than left in a notebook that gets re-run.
    """
    save_processed_json(
        {
            "params": study.best_params,
            "cv_log_loss": float(study.best_value),
            "chosen_by": {
                "method": "expanding-window CV inside the training years",
                "n_folds": study.user_attrs["n_folds"],
                "model_seeds": study.user_attrs["model_seeds"],
                "n_trials": len(study.trials),
                "seed": TUNE_SEED,
                "never_saw_fights_from": validation_start,
            },
        },
        "model_params",
        processed_dir=processed_dir,
    )


def load_params(processed_dir=None) -> dict:
    return load_processed_json("model_params", processed_dir=processed_dir)["params"]
