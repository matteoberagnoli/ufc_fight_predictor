"""The first model: gradient-boosted trees, made fair to both corners.

Fighter A and fighter B are assigned by a coin flip, so the same fight can be
shown to the model two ways round. Trees don't know that: a split on
`diff_x > 0.3` has no automatic mirror at `diff_x < -0.3`. In testing, the
plain model picked a different winner for 34.8% of validation fights depending
on which corner the coin flip had chosen. So every prediction here is made
both ways round and averaged (`predict_proba`). That makes the answers
consistent by construction - p(A wins) + p(B wins) is exactly 1 - and it was
also more accurate: on validation it fixed 106 fights and broke 68 (McNemar
p = 0.005). Training on mirrored copies as well was tried, and did worse.

The model is scikit-learn's HistGradientBoostingClassifier: the same family
as XGBoost and LightGBM (small trees built one after another, each correcting
the ones before), it handles NaN natively, and it needs nothing beyond
scikit-learn. XGBoost itself needs the OpenMP runtime, which this machine
doesn't have; swapping it in later only means changing `make_model`.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# Not tuned: cautious settings for a first model with ~2,900 training rows and
# 105 features. Tuning them is a job for the validation years, later.
MODEL_PARAMS = {
    "max_iter": 300,           # how many trees
    "learning_rate": 0.05,     # how much each tree may correct the ones before it
    "max_leaf_nodes": 15,      # small trees: each one learns a simple pattern
    "l2_regularization": 1.0,  # discourages extreme values in the leaves
    "early_stopping": False,   # its built-in check would use a random split, not a time-based one
    "random_state": 0,         # nothing random happens with these settings; fixed anyway
}

# Features that read the same whichever fighter is called A. Every other
# feature must be a fighter_a-minus-fighter_b difference (negated when the
# corners swap) or one half of a fighter_a_/fighter_b_ pair (the halves swap).
SYMMETRIC_FEATURES = frozenset({"same_stance", "weight_class_lbs", "is_womens_bout"})


def to_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Every feature as plain float64: True/False become 1/0, pandas' <NA> becomes NaN.

    The model then never depends on how pandas happened to store a column.
    """
    return X.astype("float64")


def mirror(X: pd.DataFrame) -> pd.DataFrame:
    """The same fights with the corners swapped.

    diff_* columns flip sign, fighter_a_*/fighter_b_* columns trade places, and
    the features in SYMMETRIC_FEATURES stay as they are. Anything else raises:
    otherwise a new feature could be mirrored wrongly without anyone noticing.
    """
    out = X.copy()
    for column in X.columns:
        if column.startswith("diff_"):
            out[column] = -X[column]
        elif column.startswith(("fighter_a_", "fighter_b_")):
            side, other = ("fighter_a_", "fighter_b_") if column.startswith("fighter_a_") else ("fighter_b_", "fighter_a_")
            partner = other + column[len(side):]
            if partner not in X.columns:
                raise ValueError(f"{column} has no {partner} to swap with")
            out[column] = X[partner]
        elif column not in SYMMETRIC_FEATURES:
            raise ValueError(
                f"don't know how to mirror {column!r}: it is not a diff_ column, not a "
                "fighter_a_/fighter_b_ pair, and not listed in SYMMETRIC_FEATURES"
            )
    return out


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**MODEL_PARAMS)


def fit_model(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    """Train on each fight once, as the coin flip presented it."""
    return make_model().fit(to_matrix(X), np.asarray(y, dtype=bool))


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """P(fighter A wins), asked both ways round and averaged.

    Asked as-is, the model says p. Asked with the corners swapped, it says q
    for the *other* fighter - which means 1 - q for fighter A. The answer is
    the average of p and 1 - q. For the swapped fight it comes out as exactly
    1 minus this one, so the two corners can never disagree.
    """
    X = to_matrix(X)
    as_is = model.predict_proba(X)[:, 1]
    swapped = model.predict_proba(mirror(X))[:, 1]
    return (as_is + (1 - swapped)) / 2


def corner_disagreement(model, X: pd.DataFrame) -> float:
    """Share of fights where the model, asked one way only, contradicts itself.

    A contradiction: asked as-is it picks fighter A, and asked with the corners
    swapped it picks the fighter now in slot A - i.e. the other one.
    """
    X = to_matrix(X)
    as_is = model.predict_proba(X)[:, 1] > 0.5
    swapped = model.predict_proba(mirror(X))[:, 1] > 0.5
    return float((as_is == swapped).mean())
