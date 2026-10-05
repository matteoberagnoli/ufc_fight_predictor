import numpy as np
import pandas as pd
import pytest

from ufc_predictor.features.elo import (
    ELO_START,
    attach_elo_features,
    elo_forecast_log_loss,
    elo_ratings,
    expected_score,
    load_elo_params,
    save_elo_params,
)
from ufc_predictor.features.manifest import (
    _group_of,
    feature_columns,
    feature_groups,
    unclassified_columns,
)
from ufc_predictor.modeling.model import mirror

K = 64


def _fights(*rows) -> pd.DataFrame:
    """rows: (date, url, fighter_a, fighter_b, outcome)."""
    df = pd.DataFrame(rows, columns=["event_date", "fights_url", "fighter_a_id", "fighter_b_id", "outcome"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    won = df["outcome"].map({"fighter_a": True, "fighter_b": False})
    df["fighter_a_won"] = pd.array(won, dtype="boolean")
    return df


def _made_up_league(n_fighters=12, n_fights=200, seed=0) -> pd.DataFrame:
    """Random pairings and random results, draws and no-contests included."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_fights):
        a, b = rng.choice(n_fighters, size=2, replace=False)
        outcome = rng.choice(["fighter_a", "fighter_b", "draw", "no_contest"], p=[0.45, 0.45, 0.05, 0.05])
        rows.append((pd.Timestamp("2015-01-01") + pd.Timedelta(days=7 * i), f"f/{i:04d}", int(a), int(b), outcome))
    return _fights(*rows)


# ---------------------------------------------------------------- the formulas

def test_expected_score_basics():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)        # 400 points = 10-to-1
    assert expected_score(1600, 1500) + expected_score(1500, 1600) == pytest.approx(1.0)


def test_the_ufc_69_example():
    # Serra 1538.06 vs St-Pierre 1701.45: 28% for Serra; his win is worth 46.0 at K = 64
    p_serra = expected_score(1538.056661, 1701.445009)
    assert p_serra == pytest.approx(0.2808, abs=1e-4)
    assert K * (1 - p_serra) == pytest.approx(46.03, abs=0.01)
    assert K * p_serra == pytest.approx(17.97, abs=0.01)    # what St-Pierre would have gained


# ------------------------------------------------------ store first, then update

def test_ratings_are_stored_before_the_fight_changes_them():
    fights = _fights(
        ("2015-01-01", "f/1", 1, 2, "fighter_a"),
        ("2015-02-01", "f/2", 1, 2, "fighter_b"),
    )
    r = elo_ratings(fights, K)
    # first fight: both debut at 1500. Second: only the first fight's +-32 is in
    assert r.iloc[0].tolist() == [ELO_START, ELO_START]
    assert r.iloc[1].tolist() == [1532.0, 1468.0]


def test_points_balance():
    """Whatever one fighter gains, the other loses - so the total never moves."""
    league = _made_up_league()
    # one last card where everyone appears, ending in no-contests: it reveals
    # every fighter's final rating without changing any of them
    last = league["event_date"].max() + pd.Timedelta(days=7)
    probe = _fights(*[(last, f"z/{i}", 2 * i, 2 * i + 1, "no_contest") for i in range(6)])
    r = elo_ratings(pd.concat([league, probe], ignore_index=True), K)
    final = r.iloc[-6:].to_numpy().sum()
    assert final == pytest.approx(12 * ELO_START)


def test_a_draw_between_equals_changes_nothing():
    fights = _fights(
        ("2015-01-01", "f/1", 1, 2, "draw"),
        ("2015-02-01", "f/2", 1, 2, "fighter_a"),
    )
    assert elo_ratings(fights, K).iloc[1].tolist() == [ELO_START, ELO_START]


def test_a_draw_moves_the_higher_rated_fighter_down():
    fights = _fights(
        ("2015-01-01", "f/1", 1, 2, "fighter_a"),    # 1532 vs 1468
        ("2015-02-01", "f/2", 1, 2, "draw"),
        ("2015-03-01", "f/3", 1, 2, "fighter_a"),
    )
    r = elo_ratings(fights, K)
    shift = K * (0.5 - expected_score(1532, 1468))
    assert r.iloc[2].tolist() == pytest.approx([1532 + shift, 1468 - shift])
    assert shift < 0


def test_a_no_contest_changes_neither_rating():
    fights = _fights(
        ("2015-01-01", "f/1", 1, 2, "fighter_a"),
        ("2015-02-01", "f/2", 1, 2, "no_contest"),
        ("2015-03-01", "f/3", 1, 2, "fighter_b"),
    )
    r = elo_ratings(fights, K)
    assert r.iloc[2].tolist() == r.iloc[1].tolist() == [1532.0, 1468.0]


# ---------------------------------------------------------------- leakage

def test_rewriting_a_later_result_changes_no_rating_before_it():
    league = _made_up_league()
    before = elo_ratings(league, K)
    j = 120
    rewritten = league.copy()
    rewritten.loc[j, "outcome"] = "fighter_b" if league.loc[j, "outcome"] == "fighter_a" else "fighter_a"
    after = elo_ratings(rewritten, K)
    # every rating up to and including fight j is identical - j's own row too,
    # because its rating was stored before its result was used
    pd.testing.assert_frame_equal(before.iloc[: j + 1], after.iloc[: j + 1])
    assert not before.iloc[j + 1:].equals(after.iloc[j + 1:])


# ---------------------------------------------------------------- order

def test_input_order_does_not_matter():
    league = _made_up_league()
    shuffled = league.sample(frac=1.0, random_state=3)
    a = elo_ratings(league, K)
    b = elo_ratings(shuffled, K).loc[league.index]
    pd.testing.assert_frame_equal(a, b)


def test_two_fights_on_one_night_go_in_fights_url_order():
    # early UFC tournaments: fighter 1 fights twice on one date
    fights = _fights(
        ("1994-03-11", "url/2", 1, 3, "fighter_a"),     # listed first, but happens second
        ("1994-03-11", "url/1", 1, 2, "fighter_a"),
    )
    r = elo_ratings(fights, K)
    assert r.loc[1].tolist() == [ELO_START, ELO_START]     # url/1 goes first
    assert r.loc[0, "fighter_a_prior_elo"] == 1532.0        # url/2 sees fighter 1's first win


# ---------------------------------------------------------------- loud failures

def test_unknown_or_missing_outcomes_are_refused():
    bad = _fights(("2015-01-01", "f/1", 1, 2, "fighter_c"))
    with pytest.raises(ValueError, match="outcome"):
        elo_ratings(bad, K)
    missing = _fights(("2015-01-01", "f/1", 1, 2, "fighter_a"))
    missing["outcome"] = pd.array([pd.NA], dtype="string")
    with pytest.raises(ValueError, match="outcome"):
        elo_ratings(missing, K)


# ---------------------------------------------------------------- attaching

def test_attach_adds_the_difference_and_can_be_rerun():
    league = _made_up_league()
    once = attach_elo_features(league, 32)
    twice = attach_elo_features(once, 128)               # re-rated with another K
    assert (twice["diff_prior_elo"] == twice["fighter_a_prior_elo"] - twice["fighter_b_prior_elo"]).all()
    pd.testing.assert_frame_equal(twice, attach_elo_features(league, 128))


def test_elo_on_its_own_scores_by_hand():
    fights = _fights(
        ("2015-06-01", "f/1", 1, 2, "fighter_a"),
        ("2016-06-01", "f/2", 1, 2, "fighter_a"),       # the only fight in the scored period
    )
    score = elo_forecast_log_loss(fights, K, start="2016-01-01", until="2017-01-01")
    assert score == pytest.approx(-np.log(expected_score(1532, 1468)))


# ---------------------------------------------- fitting the rest of the project

def _with_elo_columns() -> pd.DataFrame:
    return attach_elo_features(_made_up_league(), K)


def test_the_manifest_files_elo_under_strength():
    table = _with_elo_columns()
    assert _group_of("diff_prior_elo") == "strength"
    assert feature_groups(table)["strength"] == ["diff_prior_elo"]
    assert feature_columns(table) == ["diff_prior_elo"]        # the per-side halves stay out
    assert unclassified_columns(table) == []


def test_mirror_negates_the_elo_difference():
    X = _with_elo_columns()[["diff_prior_elo"]]
    pd.testing.assert_series_equal(mirror(X)["diff_prior_elo"], -X["diff_prior_elo"])


def test_saved_k_round_trips_inside_the_given_folder(tmp_path):
    curve = [(100.0, 0.672), (158.5, 0.669), (199.5, 0.670)]
    save_elo_params(158.5, curve, seeds=range(5), processed_dir=tmp_path)
    assert (tmp_path / "elo_params.json").exists()
    assert load_elo_params(processed_dir=tmp_path) == 158.5
