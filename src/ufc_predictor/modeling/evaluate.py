"""How good is a set of predictions, and what is the model relying on?

Three numbers, always on fights the model never trained on:

  accuracy  share of fights where the predicted favourite won. Easy to read,
            but it treats a 51% call and a 90% call the same.
  log loss  how surprised the predictions were by what actually happened,
            on average. Lower is better, and confident mistakes cost the most.
            A coin flip (0.5 for every fight) scores 0.693. The main number.
  AUC       how well the predictions rank fights: take one fight fighter A
            won and one they lost; the share of such pairs where the won fight
            got the higher probability. 0.5 is a coin flip, 1.0 is perfect.

Men's fights are reported first because they are what this project is for.
The model still trains on all fights: tested on men's validation fights, the
pooled model did better than one trained on men's fights only.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from ufc_predictor.modeling.model import predict_proba, to_matrix


def score(y, p) -> dict:
    """Accuracy, log loss and AUC for one set of predictions."""
    y = np.asarray(y, dtype=bool)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    right = np.where(p == 0.5, 0.5, (p > 0.5) == y)   # a 50/50 call counts as half right
    both_results = y.any() and not y.all()            # AUC needs at least one win and one loss
    return {
        "fights": len(y),
        "accuracy": float(right.mean()),
        "log_loss": float(log_loss(y, p, labels=[False, True])),
        "auc": float(roc_auc_score(y, p)) if both_results else float("nan"),
    }


def scorecard(predictions: dict, y, mens) -> pd.DataFrame:
    """One row per set of predictions: men's fights first (the headline), then all fights."""
    y, mens = np.asarray(y, dtype=bool), np.asarray(mens, dtype=bool)
    rows = []
    for name, p in predictions.items():
        p = np.asarray(p, dtype=float)
        men, every = score(y[mens], p[mens]), score(y, p)
        rows.append({
            "model": name,
            "men_accuracy": men["accuracy"], "men_log_loss": men["log_loss"], "men_auc": men["auc"],
            "all_accuracy": every["accuracy"], "all_log_loss": every["log_loss"], "all_auc": every["auc"],
        })
    return pd.DataFrame(rows).set_index("model").round(4)


def compare(y, p_new, p_old) -> dict:
    """Which fights does the new set call right that the old one got wrong, and vice versa?

    `p_value` is McNemar's exact test: the chance of a split at least this
    lopsided if the two were really equally good. Below about 0.05, the
    difference is unlikely to be luck.
    """
    y = np.asarray(y, dtype=bool)
    new_right = (np.asarray(p_new) > 0.5) == y
    old_right = (np.asarray(p_old) > 0.5) == y
    fixes, breaks = int((new_right & ~old_right).sum()), int((old_right & ~new_right).sum())
    p_value = binomtest(fixes, fixes + breaks, 0.5).pvalue if fixes + breaks else 1.0
    return {"fixes": fixes, "breaks": breaks, "p_value": float(p_value)}


# re-exported here so evaluation code has one obvious place to import it from;
# the real logic lives in features/weight_class.py, next to is_womens_bout's
# sibling facts (division, weight_class_lbs), which are built from the same string
from ufc_predictor.features.weight_class import is_womens_bout as is_womens_fight  # noqa: E402,F401


def fit_baseline(X: pd.DataFrame, y, column: str) -> LogisticRegression:
    """A one-feature baseline: logistic regression on a single difference column.

    One coefficient and no intercept. With no intercept there is no built-in
    preference for either corner, so the baseline is fair to both corners by
    construction, like the model. C=inf means no regularisation: with one
    coefficient and thousands of fights there is nothing to hold back. A
    missing value (a debutant) is treated as "no difference", 0.
    """
    model = LogisticRegression(C=np.inf, fit_intercept=False)
    return model.fit(_one_column(X, column), np.asarray(y, dtype=bool))


def predict_baseline(model: LogisticRegression, X: pd.DataFrame, column: str) -> np.ndarray:
    return model.predict_proba(_one_column(X, column))[:, 1]


def _one_column(X: pd.DataFrame, column: str) -> np.ndarray:
    return X[[column]].astype("float64").fillna(0.0).to_numpy()


def group_importance(model, X: pd.DataFrame, y, groups: dict, repeats: int = 20,
                     seed: int = 0) -> pd.DataFrame:
    """How much worse log loss gets when one feature family is scrambled.

    For each group, shuffle its columns across fights - all of them with the
    same shuffle, so the family stays internally consistent - then predict
    again and measure how much log loss rises. Repeated `repeats` times. A
    family the model relies on hurts a lot when scrambled; one it ignores
    barely registers.

    Use fights the model did NOT train on. On its own training fights,
    scrambling even a pure-noise column hurts, because the model has partly
    memorised that noise - which measures overfitting, not importance.
    """
    X = to_matrix(X)
    rng = np.random.default_rng(seed)
    base = score(y, predict_proba(model, X))["log_loss"]
    rows = []
    for name, columns in groups.items():
        if not columns:
            continue
        rises = []
        for _ in range(repeats):
            scrambled = X.copy()
            scrambled[columns] = X[columns].to_numpy()[rng.permutation(len(X))]
            rises.append(score(y, predict_proba(model, scrambled))["log_loss"] - base)
        rows.append({"group": name, "columns": len(columns),
                     "log_loss_rise": float(np.mean(rises)), "spread": float(np.std(rises))})
    return pd.DataFrame(rows).sort_values("log_loss_rise", ascending=False).reset_index(drop=True)


def calibration_table(y, p, bins: int = 5) -> pd.DataFrame:
    """Are the probabilities honest?

    Fights are put into `bins` equal-sized groups by predicted probability.
    If the model says 70% for a group, fighter A should win about 70% of
    those fights.
    """
    y, p = np.asarray(y, dtype=bool), np.asarray(p, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    group = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    table = pd.DataFrame({"group": group, "predicted": p, "actual": y})
    return (table.groupby("group")
            .agg(fights=("predicted", "size"), predicted=("predicted", "mean"), actual=("actual", "mean"))
            .reset_index(drop=True).round(3))
