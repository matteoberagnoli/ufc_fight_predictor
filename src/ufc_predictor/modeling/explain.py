"""Explaining predictions: how much each feature pushed each fight's prediction.

SHAP splits one prediction into one piece per feature:

    score = baseline + piece from Elo + piece from age + ...   (40 pieces)

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

Interactions (`shap_interaction_values`) split each piece once more: into the
feature's push on its own (its *main effect*) and the extra push it gets only
because of another feature's value - "Elo pushes harder because the age gap
points the same way". Per fight that is a features x features table: the
diagonal holds the main effects, entries [i, j] and [j, i] each hold half of
the pair's interaction, and each row adds up exactly to that feature's SHAP
value. The same two adjustments apply, on both axes of the table.
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


def shap_interaction_values(models: list, X: pd.DataFrame) -> np.ndarray:
    """SHAP interaction values of the symmetric score, averaged over the models.

    Shape (fights, features, features). The diagonal is each feature's main
    effect; [i, j] + [j, i] is the interaction of the pair; row i adds up to
    feature i's value in shap_values. Same recipe as shap_values, with the
    swapped fight's table put back under the right columns on both axes.
    """
    X = to_matrix(X)
    swapped_X = mirror(X)
    partner = _partner_index(X.columns)
    total = np.zeros((len(X), X.shape[1], X.shape[1]))
    for m in models:                      # a running sum: 8 stored tables of 4,000 fights would be large
        explainer = shap.TreeExplainer(m)
        swapped = explainer.shap_interaction_values(swapped_X)[:, partner][:, :, partner]
        total += (explainer.shap_interaction_values(X) - swapped) / 2
    return total / len(models)


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
    _check_groups(names, groups)
    position = {c: i for i, c in enumerate(names)}
    values = np.column_stack([
        explanation.values[:, [position[c] for c in columns]].sum(axis=1) for columns in groups.values()
    ])
    return shap.Explanation(values=values, base_values=explanation.base_values, feature_names=list(groups))


def group_interactions(values: np.ndarray, feature_names, groups: dict) -> pd.DataFrame:
    """How strongly each pair of feature groups interacts: the average size of the push.

    `values` comes from shap_interaction_values. For two groups, the entries
    linking their features are added up per fight (both halves, [i, j] and
    [j, i]), then averaged ignoring direction. The diagonal is the interaction
    *within* a group, between its own features; main effects are left out.
    """
    names = list(feature_names)
    _check_groups(names, groups)
    position = {c: i for i, c in enumerate(names)}
    off = values.copy()
    off[:, np.arange(len(names)), np.arange(len(names))] = 0      # main effects are not interactions
    index = {g: [position[c] for c in columns] for g, columns in groups.items()}
    table = pd.DataFrame(0.0, index=list(groups), columns=list(groups))
    for g, rows in index.items():
        for h, cols in index.items():
            block = off[:, rows][:, :, cols].sum(axis=(1, 2))
            # within a group the block already holds both halves; between groups add the mirror block
            pair = block if g == h else block + off[:, cols][:, :, rows].sum(axis=(1, 2))
            table.loc[g, h] = np.abs(pair).mean()
    return table


def _check_groups(names: list, groups: dict) -> None:
    """Every feature in exactly one group, or a ValueError naming what is wrong."""
    grouped = [c for columns in groups.values() for c in columns]
    if sorted(grouped) != sorted(names):
        missing = set(names) - set(grouped)
        extra = [c for c in set(grouped) if grouped.count(c) > 1 or c not in names]
        raise ValueError(f"groups must cover every feature exactly once; missing {sorted(missing)}, extra {sorted(extra)}")
