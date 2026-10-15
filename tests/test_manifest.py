import pandas as pd
import pytest

from ufc_predictor.config import VALIDATION_START
from ufc_predictor.data.build import build_fight_dataset
from ufc_predictor.data.io import load_fighters, load_processed
from ufc_predictor.features import manifest as m
from ufc_predictor.features.rolling import ROLLING_WINDOW, attach_prior_features, fit_alphas
from ufc_predictor.features.static import attach_static_features
from ufc_predictor.modeling.split import drop_undecided


@pytest.fixture(scope="module")
def featured() -> pd.DataFrame:
    """The real featured table - read if it exists, rebuilt if not.

    Reading takes 0.04s against 40s to rebuild, which is the difference
    between a suite people run constantly and one they avoid. The fallback
    keeps it working on a fresh clone, where data/processed is absent because
    it is gitignored as regenerable.
    """
    try:
        return load_processed("fights_featured")
    except (FileNotFoundError, OSError):
        fights = build_fight_dataset()
        out = attach_prior_features(fights, fit_alphas(fights, until=VALIDATION_START))
        return attach_static_features(out, load_fighters())


def test_every_column_is_classified(featured):
    """The completeness guard, and the one actually earning its keep.

    A column added without deciding what it is - feature, metadata, label or
    unsafe - should fail here rather than silently drift into or out of X.
    """
    assert m.unclassified_columns(featured) == []


def test_buckets_do_not_overlap(featured):
    features = set(m.feature_columns(featured))
    leaky = set(m.leaky_columns(featured))
    metadata = set(m.METADATA_COLUMNS)

    assert features & leaky == set()
    assert features & metadata == set()
    assert m.LABEL_COLUMN not in features


def test_leaky_columns_are_the_raw_per_fight_stats(featured):
    leaky = m.leaky_columns(featured)
    assert len(leaky) == 46  # 23 stems x two corners

    # the footgun this module exists for: these two differ by one word
    assert "fighter_a_sig_strikes" in leaky
    assert "fighter_a_prior_sig_strikes_accuracy_career_avg" not in leaky


def test_make_xy_splits_features_from_label(featured):
    decided = drop_undecided(featured)
    X, y = m.make_xy(decided)
    assert len(X) == len(y) == len(decided)
    assert m.LABEL_COLUMN not in X.columns
    assert y.name == m.LABEL_COLUMN
    assert y.dtype == bool          # no NA survives, so a plain bool is safe
    assert not set(X.columns) & set(m.leaky_columns(featured))


def test_make_xy_refuses_rows_with_no_winner(featured):
    """A draw has no answer to learn, so it must never reach X silently.

    The featured table deliberately holds them (they are fighter history),
    which makes passing it straight to make_xy an easy mistake - hence a
    loud failure rather than 143 rows of NA label.
    """
    with pytest.raises(ValueError, match="no winner"):
        m.make_xy(featured)


def test_outcome_is_treated_as_a_label_not_a_feature(featured):
    # `outcome` says everything fighter_a_won does and more, so it is the
    # answer sheet too
    assert "outcome" in m.LABEL_COLUMNS
    assert "outcome" not in m.feature_columns(featured)
    assert "outcome" not in m.METADATA_COLUMNS


def test_groups_compose_as_their_union(featured):
    groups = m.feature_groups(featured)
    selected = m.feature_columns(featured, groups=["record", "physical"])
    assert set(selected) == set(groups["record"]) | set(groups["physical"])
    assert len(selected) < len(m.feature_columns(featured))


def test_unknown_group_is_rejected(featured):
    with pytest.raises(ValueError, match="unknown feature group"):
        m.feature_columns(featured, groups=["striking", "not_a_group"])


def test_leakage_guard_fires_if_a_raw_column_is_ever_misclassified(featured, monkeypatch):
    """Simulate the future change this guard exists to catch.

    Feature names (`diff_*`, `same_stance`, `is_debut`) and leaky names
    (`fighter_{a,b}_{raw stem}`) are structurally disjoint today, so the guard
    cannot fire on the real table. It becomes live the moment `_group_of`
    starts classifying per-side columns as features - which is exactly what
    happens if the diffs-only decision is revisited. Force that here.
    """
    real_group_of = m._group_of

    def misclassifying(column: str):
        if column == "fighter_a_sig_strikes":  # a genuinely leaky column
            return "striking"
        return real_group_of(column)

    monkeypatch.setattr(m, "_group_of", misclassifying)
    monkeypatch.setattr(m, "FEATURES", m.FEATURES + ("fighter_a_sig_strikes",))

    with pytest.raises(ValueError, match="raw post-fight column"):
        m.feature_columns(featured)


def test_nan_report_covers_every_group_and_the_total(featured):
    report = m.nan_report(featured)
    assert set(report["group"]) == set(m.GROUP_NAMES) | {"ALL"}

    # the whole point of the report: adding groups can only cost rows, never
    # recover them, so the total must be at least as bad as the worst group
    worst_group = report[report["group"] != "ALL"]["rows_with_nan"].max()
    total = report.loc[report["group"] == "ALL", "rows_with_nan"].iloc[0]
    assert total >= worst_group

    assert report.loc[report["group"] == "cold_start", "rows_with_nan"].iloc[0] == 0.0


def test_stem_decomposition_recovers_the_underlying_statistic():
    assert m._decompose("diff_prior_head_strike_atts_absorbed_per_min_career_avg") == (
        "head_strike_atts", True,
    )
    assert m._decompose(f"diff_prior_leg_strikes_accuracy_last{ROLLING_WINDOW}_avg") == ("leg_strikes", False)
    # what is thrown at a fighter is the defending direction, however it is named
    assert m._decompose("diff_prior_head_strikes_hit_rate_against_career_avg") == ("head_strikes", True)
    assert m._decompose(f"diff_prior_takedowns_success_rate_against_last{ROLLING_WINDOW}_avg") == ("takedowns", True)
    assert m._decompose("diff_prior_ctrl_time_absorbed_frac_career_avg") == ("ctrl_time", True)
    # bare window suffix, no ratio decoration
    assert m._decompose("diff_prior_total_fight_time_career_avg") == ("total_fight_time", False)


def test_absorbed_direction_always_lands_in_defence(featured):
    groups = m.feature_groups(featured)
    for column in m.feature_columns(featured):
        if column.startswith("diff_prior_") and ("_absorbed" in column or "_against_" in column):
            assert column in groups["defence"], column


# ------------------------------------------------------------ the explicit list

def test_features_are_the_40_chosen_in_the_review():
    """FEATURES and the review's record of the final model must never drift apart."""
    import csv
    from ufc_predictor.config import PROJECT_ROOT
    with open(PROJECT_ROOT / "reviews" / "final_inputs_2026-10-08.csv") as f:
        recorded = [row["final_name"] for row in csv.DictReader(f)]
    assert len(m.FEATURES) == len(set(m.FEATURES)) == 40
    assert set(m.FEATURES) == set(recorded)


def test_every_feature_is_in_the_table_and_in_a_group(featured):
    assert [c for c in m.FEATURES if c not in featured.columns] == []
    grouped = [c for cols in m.feature_groups(featured).values() for c in cols]
    assert sorted(grouped) == sorted(m.FEATURES)


def test_x_is_exactly_the_chosen_features(featured):
    X, _ = m.make_xy(drop_undecided(featured))
    assert sorted(X.columns) == sorted(m.FEATURES)
    assert not set(X.columns) & set(m.unselected_columns(featured))


def test_unselected_columns_are_safe_and_not_features(featured):
    unselected = set(m.unselected_columns(featured))
    assert unselected                                       # the career windows, at least
    assert not unselected & set(m.leaky_columns(featured))
    assert "diff_prior_sig_strikes_accuracy_career_avg" in unselected
    assert "diff_static_ape_index_cm" in unselected         # discarded in the review


def test_a_feature_outside_every_group_fails_loudly(featured, monkeypatch):
    monkeypatch.setattr(m, "FEATURES", m.FEATURES + ("fights_url",))
    with pytest.raises(ValueError, match="belongs to no feature group"):
        m.feature_groups(featured)
