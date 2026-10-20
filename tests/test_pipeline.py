import pandas as pd

from ufc_predictor.features.elo import ELO_COLUMNS
from ufc_predictor.features.pipeline import build_featured_table
from ufc_predictor.features.rolling import MIRRORED_STATS, SHARED_STATS, no_shrinkage
from ufc_predictor.modeling.variants import build_variants

ELO = list(ELO_COLUMNS) + ["diff_prior_elo"]


def _fights() -> pd.DataFrame:
    """Four fights for fighter 1, enough raw columns for the whole chain to run."""
    fights = pd.DataFrame({
        "fights_url": ["f/1", "f/2", "f/3", "f/4"],
        "method": ["finish", "narrow", "narrow", "unanimous"],
        "event_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        "fighter_a_id": [1, 1, 1, 1],
        "fighter_b_id": [90, 91, 92, 90],
        "fighter_a_won": pd.array([True, False, None, True], dtype="boolean"),
        "outcome": ["fighter_a", "fighter_b", "draw", "fighter_a"],
        "weight_class": ["Lightweight Bout"] * 4,
        "fighter_a_age_during": [30.0] * 4, "fighter_b_age_during": [28.0] * 4,
        "fighter_a_height_cm": [180.0] * 4, "fighter_b_height_cm": [175.0] * 4,
        **{f"fighter_{s}_{stat}": [1.0] * 4 for s in ("a", "b") for stat in MIRRORED_STATS},
        **{f"fighter_{s}_{stat}": [600.0] * 4 for s in ("a", "b") for stat in SHARED_STATS},
    })
    return fights


FIGHTERS = pd.DataFrame({
    "fighter_id": [1, 90, 91, 92],
    "fighter_reach_cm": [185.0, 178.0, 180.0, 175.0],
    "fighter_stance": ["Orthodox", "Southpaw", "Orthodox", "Orthodox"],
})


def test_elo_adds_its_three_columns_and_changes_nothing_else():
    without = build_featured_table(_fights(), FIGHTERS, no_shrinkage(), elo_k=None, elo_weights=None)
    with_elo = build_featured_table(_fights(), FIGHTERS, no_shrinkage(), elo_k=64, elo_weights=None)
    assert not set(ELO) & set(without.columns)
    assert list(with_elo.columns) == list(without.columns) + ELO
    pd.testing.assert_frame_equal(with_elo.drop(columns=ELO), without)


def test_every_comparison_arm_gets_the_same_elo():
    """An arm that silently skips a step fakes a comparison (see features/pipeline.py)."""
    arms = build_variants(_fights(), no_shrinkage(), FIGHTERS, elo_k=64, elo_weights={"unanimous": 0.7, "narrow": 0.55})
    reference = arms["shrunk"][ELO]
    for name, table in arms.items():
        pd.testing.assert_frame_equal(table[ELO], reference, obj=name)


def test_win_weights_reach_the_featured_table():
    classic = build_featured_table(_fights(), FIGHTERS, no_shrinkage(), elo_k=64, elo_weights=None)
    weighted = build_featured_table(_fights(), FIGHTERS, no_shrinkage(), elo_k=64,
                                    elo_weights={"unanimous": 0.7, "narrow": 0.55})
    # fight 2 is a narrow loss for fighter 1: it costs less, so his rating going into fight 3 is higher
    assert weighted["fighter_a_prior_elo"].iloc[2] > classic["fighter_a_prior_elo"].iloc[2]
    pd.testing.assert_frame_equal(weighted.drop(columns=ELO), classic.drop(columns=ELO))
