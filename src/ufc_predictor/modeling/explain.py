"""Explaining predictions: how much each feature pushed each fight's prediction.

SHAP splits one prediction into one piece per feature:

    score = baseline + piece from Elo + piece from age + ...   (108 pieces)

Each piece is that feature's SHAP value for that fight. Positive pushes
towards fighter A, negative towards fighter B. The units are log-odds, the
scale the trees add up on: near 50%, +0.1 is about +2.5 percentage points.
SHAP never looks at who won - it explains the prediction, not the result - so
looking at validation fights through it spends nothing.

Two things make our model different from the textbook case, and both are
handled here so the explanation matches what the model actually predicts:

  * Every fight is predicted both ways round (model.predict_proba). One
    model call, asked as-is, carries a small lean towards one corner; the
    swapped call carries the opposite lean. So the explained quantity is the
    *symmetric score*

        S = (score as-is - score with corners swapped) / 2

    and its SHAP values are (SHAP as-is - SHAP swapped) / 2, with the swapped
    values put back under the columns they describe (a fighter_a_ column of
    the swapped fight holds fighter B's information). The corner lean cancels,
    so the baseline is exactly 0: a fight with no information is 50-50.
  * The final predictor is 8 models, one per seed (model.fit_models). Their
    SHAP values are averaged, which is exact: SHAP values add up.

S is not quite the logit of the final probability, because the model averages
*probabilities*, not scores. The two stay very close: on the 2023-24
validation fights sigmoid(S) was at most 0.8 percentage points from
predict_proba_mean (0.06 on average) and picked the same winner in every
fight. `symmetric_score` returns S so that can be re-checked. Calibration's stretch would multiply every piece by
the same a = 1.05, so it is left out.
"""

import numpy as np
import pandas as pd
import shap

from ufc_predictor.modeling.model import mirror, to_matrix


def _partner_index(columns) -> np.ndarray:
    """Position of each column's mirror image: its fighter_a_/fighter_b_ partner, or itself."""
    position = {c: i for i, c in enumerate(columns)}
    partner = []
    for c in columns:
        if c.startswith("fighter_a_"):
            partner.append(position["fighter_b_" + c[len("fighter_a_"):]])
        elif c.startswith("fighter_b_"):
            partner.append(position["fighter_a_" + c[len("fighter_b_"):]])
        else:
            partner.append(position[c])
    return np.array(partner)


def _raw_shap(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values of one model call, in log-odds: shape (fights, features)."""
    return shap.TreeExplainer(model)(X).values


def shap_values(models: list, X: pd.DataFrame) -> shap.Explanation:
    """SHAP values of the symmetric score, averaged over the models.

    Returns a shap.Explanation, so shap.plots.beeswarm / bar / scatter /
    waterfall take it directly. Its rows add up exactly to symmetric_score.
    """
    X = to_matrix(X)
    swapped_X = mirror(X)
    partner = _partner_index(X.columns)
    per_model = [(_raw_shap(m, X) - _raw_shap(m, swapped_X)[:, partner]) / 2 for m in models]
    return shap.Explanation(
        values=np.mean(per_model, axis=0),
        base_values=np.zeros(len(X)),
        data=X.to_numpy(),
        feature_names=list(X.columns),
    )


def symmetric_score(models: list, X: pd.DataFrame) -> np.ndarray:
    """What the SHAP values add up to: (score as-is - score swapped) / 2, averaged over the models."""
    X = to_matrix(X)
    swapped_X = mirror(X)
    return np.mean([(m.decision_function(X) - m.decision_function(swapped_X)) / 2 for m in models], axis=0)


def group_shap(explanation: shap.Explanation, groups: dict) -> shap.Explanation:
    """One SHAP value per feature group: the sum of its features' values.

    Correlated features split credit between them - the 32 striking features
    partly measure the same thing, so each looks small even if striking as a
    whole matters. Summing within a group (manifest.feature_groups) adds that
    credit back up, and it is exact: the groups still add up to the score.
    Every feature must be in exactly one group. A group has no single feature
    value, so `data` is left empty: use the bar plot, not the beeswarm.
    """
    names = list(explanation.feature_names)
    grouped = [c for columns in groups.values() for c in columns]
    if sorted(grouped) != sorted(names):
        missing = set(names) - set(grouped)
        extra = [c for c in set(grouped) if grouped.count(c) > 1 or c not in names]
        raise ValueError(f"groups must cover every feature exactly once; missing {sorted(missing)}, extra {sorted(extra)}")
    position = {c: i for i, c in enumerate(names)}
    values = np.column_stack([
        explanation.values[:, [position[c] for c in columns]].sum(axis=1) for columns in groups.values()
    ])
    return shap.Explanation(values=values, base_values=explanation.base_values, feature_names=list(groups))
