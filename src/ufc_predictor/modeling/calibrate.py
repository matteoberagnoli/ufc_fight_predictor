"""Calibration: making the model's percentages mean what they say.

If the model says 70% for a hundred fights, fighter A should win about seventy
of them. The model turns a score into a percentage with the logistic curve,
p = 1 / (1 + e^-score). Calibration learns one number, the stretch `a`, and
redraws that curve:

    p_new = 1 / (1 + e^-(a * score))        where score = logit(p)

    a > 1   bolder: percentages pushed away from 50%   (fixes under-confidence)
    a < 1   more cautious: pulled towards 50%           (fixes over-confidence)
    a = 1   no change

Two things it deliberately cannot do:

  * shift the curve. A shift would favour whoever sits in corner A, and
    corner A is a coin flip. Without one, stretch(1 - p) = 1 - stretch(p): the
    two corners still add up to exactly 1.
  * change a pick. The stretch keeps the order of predictions and leaves 50%
    at 50%, so the favourite stays the favourite. Accuracy and AUC can't move;
    only log loss can.

`a` is fitted on out-of-fold predictions from the training years
(modeling.tune.out_of_fold_predictions), never on the model's own training
fights and never on the validation years. Measured at planning time: a = 1.12,
90% range 0.92-1.33 - at most mildly under-confident, so always report the
range with it.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from ufc_predictor.data.io import load_processed_json, save_processed_json

_EPS = 1e-6


def stretch(p, a: float) -> np.ndarray:
    """Redraw the probability curve with stretch `a`."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return expit(a * logit(p))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_stretch(y, p, bounds=(0.1, 10.0)) -> float:
    """The stretch that makes `p` best match what happened (lowest log loss)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    result = minimize_scalar(lambda a: _log_loss(y, stretch(p, a)), bounds=bounds, method="bounded")
    return float(result.x)


def stretch_interval(y, p, n_boot: int = 400, seed: int = 0) -> tuple[float, float]:
    """90% bootstrap range for the stretch: how far chance alone moves it.

    Resample the fights with replacement, refit, repeat. With ~1,900 fights
    the range is about +-0.2 - wide enough that an `a` near 1 may just mean
    "no correction needed".
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    fits = [fit_stretch(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    low, high = np.percentile(fits, [5, 95])
    return float(low), float(high)


def save_calibration(a: float, interval, n_fights: int, decision: str,
                     processed_dir=None) -> None:
    """Save the stretch in use, its range, and why it was (or wasn't) kept."""
    save_processed_json(
        {
            "a": float(a),
            "interval_90": [float(interval[0]), float(interval[1])],
            "fitted_on": {"fights": int(n_fights),
                          "method": "out-of-fold predictions, expanding-window CV inside the training years"},
            "decision": decision,
        },
        "calibration",
        processed_dir=processed_dir,
    )


def load_calibration(processed_dir=None) -> float:
    return load_processed_json("calibration", processed_dir=processed_dir)["a"]
