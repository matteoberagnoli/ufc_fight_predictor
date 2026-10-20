"""The whole feature chain in one place: fights -> the featured table.

Every table the project trains on is built by this one function - the main
table in 02_first_features.ipynb and every comparison arm in
modeling/variants.py. Before it existed, the chain was written out twice, and
during development one copy silently missed the weight-class step: a "shrinkage
comparison" was really comparing 107 features against 105. With one shared
chain, two tables can only differ in what is passed in.
"""

import pandas as pd

from ufc_predictor.features.elo import attach_elo_features
from ufc_predictor.features.rolling import attach_prior_features
from ufc_predictor.features.static import attach_static_features
from ufc_predictor.features.weight_class import attach_weight_class_features


def build_featured_table(fights: pd.DataFrame, fighters: pd.DataFrame, alphas: dict,
                         *, elo_k: float | None, elo_weights: dict | None) -> pd.DataFrame:
    """Prior-fight features, static attributes, weight class, then Elo.

    `alphas` controls shrinkage (rolling.no_shrinkage() gives raw ratios).
    `elo_k` is Elo's K from features.elo.load_elo_params(); None leaves the
    Elo columns out, which is only needed before any K has been chosen. It has
    to be passed by name, so leaving Elo out is always a visible decision.
    `elo_weights` (features.elo.load_elo_weights()) is required for the same
    reason: forgetting it would silently build the classic Elo. None means
    classic: every win counts 1.
    """
    featured = attach_prior_features(fights, alphas)
    featured = attach_static_features(featured, fighters)
    featured = attach_weight_class_features(featured)
    if elo_k is not None:
        featured = attach_elo_features(featured, elo_k, elo_weights)
    return featured
