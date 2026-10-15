import numpy as np
import pandas as pd
import pytest

from ufc_predictor.features.rolling import (
    MIRRORED_STATS,
    PRIOR_FEATURE_COLUMNS,
    RATIO_FEATURES,
    RECORD_COLUMNS,
    SHARED_STATS,
    _streak_after_each_fight,
    add_prior_features,
    attach_prior_features,
    build_fighter_history,
    no_shrinkage,
)


def _same_alpha_everywhere(alpha: float) -> dict[str, float]:
    return {spec.name: alpha for spec in RATIO_FEATURES}


@pytest.fixture
def toy_fights() -> pd.DataFrame:
    # fighter 1 vs a new opponent each time: W, L, W, W (as fighter_a every time)
    # fighter 99 only shows up as the opponent, never as the fighter being tracked
    dates = pd.to_datetime(["2020-01-01", "2020-03-01", "2020-05-01", "2020-09-01"])
    return pd.DataFrame(
        {
            "fights_url": ["f/1", "f/2", "f/3", "f/4"],
            "event_date": dates,
            "fighter_a_id": [1, 1, 1, 1],
            "fighter_b_id": [90, 91, 92, 93],
            "fighter_a_won": pd.array([True, False, True, True], dtype="boolean"),
            "outcome": ["fighter_a", "fighter_b", "fighter_a", "fighter_a"],
            "weight_class": ["Lightweight Bout"] * 4,
            **{
                f"fighter_{side}_{stat}": [10, 10, 10, 10]
                for side in ("a", "b")
                for stat in MIRRORED_STATS + SHARED_STATS
            },
        }
    )


def test_streak_after_each_fight_tracks_sign_and_length():
    won = pd.Series([True, True, False, False, False, True])
    assert list(_streak_after_each_fight(won)) == [1, 2, -1, -2, -3, 1]


def test_build_fighter_history_has_two_rows_per_fight(toy_fights):
    history = build_fighter_history(toy_fights)
    assert len(history) == 2 * len(toy_fights)


def test_prior_features_only_see_earlier_fights(toy_fights):
    history = build_fighter_history(toy_fights)
    featured = add_prior_features(history, no_shrinkage())
    fighter_1 = featured[featured["fighter_id"] == 1].sort_values("event_date")

    # W, L, W, W -> prior record before each fight:
    assert list(fighter_1["wins"]) == [0, 1, 1, 2]
    assert list(fighter_1["losses"]) == [0, 0, 1, 1]
    assert list(fighter_1["fight_count"]) == [0, 1, 2, 3]
    assert list(fighter_1["is_debut"]) == [True, False, False, False]
    # streak entering each fight: debut(0), after W->1, after L->-1, after W->1
    assert list(fighter_1["streak"]) == [0, 1, -1, 1]


def test_debut_fighter_has_no_prior_history(toy_fights):
    # fighters 90-93 each appear exactly once, always as a debut. Even with
    # shrinkage switched on, a debutant gets NaN, not the league value: there
    # is no history to shrink, and "no history" is information in itself.
    history = build_fighter_history(toy_fights)
    featured = add_prior_features(history, _same_alpha_everywhere(10.0))
    opponents = featured[featured["fighter_id"].isin([90, 91, 92, 93])]
    assert opponents["is_debut"].all()
    assert (opponents["fight_count"] == 0).all()
    ratio_columns = [spec.name for spec in RATIO_FEATURES]
    assert opponents[ratio_columns].isna().all().all()


def test_attach_prior_features_leakage_audit(toy_fights):
    """The actual point of this module: assert no feature for a given fight
    reflects anything from that fight or a later one.

    We do this by recomputing, from scratch and independently (a plain
    Python loop over each fighter's chronological fights, not reusing any
    of rolling.py's own aggregation code), what fighter 1's prior win/loss
    record *should* be at each of their fights, and checking it against
    what attach_prior_features actually produced.
    """
    featured = attach_prior_features(toy_fights, no_shrinkage())
    fighter_1_fights = featured[featured["fighter_a_id"] == 1].sort_values("event_date")

    outcomes = [True, False, True, True]  # ground truth, independent of the module
    expected_wins, expected_losses = [], []
    w, l = 0, 0
    for outcome in outcomes:
        expected_wins.append(w)
        expected_losses.append(l)
        if outcome:
            w += 1
        else:
            l += 1

    assert list(fighter_1_fights["fighter_a_prior_wins"]) == [float(x) for x in expected_wins]
    assert list(fighter_1_fights["fighter_a_prior_losses"]) == [float(x) for x in expected_losses]


def test_diff_columns_are_a_minus_b(toy_fights):
    featured = attach_prior_features(toy_fights, no_shrinkage())
    row = featured.iloc[3]  # fighter 1's 4th fight: prior_wins=2, prior_losses=1
    assert row["diff_prior_wins"] == (
        row["fighter_a_prior_wins"] - row["fighter_b_prior_wins"]
    )


def test_attach_prior_features_row_count_unchanged(toy_fights):
    featured = attach_prior_features(toy_fights, no_shrinkage())
    assert len(featured) == len(toy_fights)


def _fighter_1(stats: dict | None = None, durations=(300, 600, 900, 1200),
               alphas: dict | None = None,
               outcomes=("fighter_a", "fighter_b", "fighter_a", "fighter_a")) -> pd.DataFrame:
    """Fighter 1's four fights (always as fighter_a, against four debutants).

    Every per-fight stat is 0 unless set in `stats`, by wide-table column:
    {"fighter_a_sig_strikes": [...]} is what fighter 1 did, and
    {"fighter_b_knockdowns": [...]} is what their opponents did to them.
    `outcomes` says how each fight ended, so a "draw" or "no_contest" can be
    put anywhere in the four.
    """
    won = pd.array([{"fighter_a": True, "fighter_b": False}.get(o, pd.NA) for o in outcomes],
                   dtype="boolean")
    fights = pd.DataFrame({
        "fights_url": ["f/1", "f/2", "f/3", "f/4"],
        "event_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        "fighter_a_id": [1, 1, 1, 1],
        "fighter_b_id": [90, 91, 92, 93],
        "fighter_a_won": won,
        "outcome": list(outcomes),
        "weight_class": ["Lightweight Bout"] * 4,
        **{f"fighter_{side}_{stat}": [0.0] * 4 for side in ("a", "b") for stat in MIRRORED_STATS},
        **{f"fighter_{side}_total_fight_time": list(durations) for side in ("a", "b")},
    })
    for column, values in (stats or {}).items():
        fights[column] = values
    featured = add_prior_features(build_fighter_history(fights), alphas or no_shrinkage())
    return featured[featured["fighter_id"] == 1].reset_index(drop=True)


# With no shrinkage (alpha = 0) the ratios are the plain pooled ratios, which
# is what these hand-computed checks pin down. Shrinkage itself is tested in
# test_shrinkage.py and at the end of this file.

def test_accuracy_is_volume_weighted_not_mean_of_per_fight_ratios():
    # 10/100, then 30/100, then 60/100. Career accuracy entering each fight is
    # the pooled ratio over prior bouts, not the average of each bout's ratio.
    out = _fighter_1({"fighter_a_sig_strikes": [10, 30, 60, 99],
                      "fighter_a_sig_strike_atts": [100, 100, 100, 100]})
    acc = out["sig_strikes_accuracy_career_avg"]

    assert pd.isna(acc.iloc[0])  # debut - no prior bouts to compute from
    assert acc.iloc[1] == pytest.approx(10 / 100)
    assert acc.iloc[2] == pytest.approx(40 / 200)
    assert acc.iloc[3] == pytest.approx(100 / 300)


def test_average_fight_time_uses_only_prior_bouts():
    avg = _fighter_1()["total_fight_time_career_avg"]

    assert pd.isna(avg.iloc[0])
    assert avg.iloc[1] == pytest.approx(300)
    assert avg.iloc[2] == pytest.approx((300 + 600) / 2)
    assert avg.iloc[3] == pytest.approx((300 + 600 + 900) / 3)


def test_without_shrinkage_zero_attempts_is_nan_not_inf():
    assert _fighter_1()["sig_strikes_accuracy_career_avg"].isna().all()


def test_absorbed_rates_are_per_minute_of_prior_cage_time():
    out = _fighter_1({"fighter_b_knockdowns": [1, 1, 0, 5]})   # knocked down by opponents
    rate = out["knockdowns_absorbed_per_min_career_avg"]

    assert pd.isna(rate.iloc[0])
    assert rate.iloc[1] == pytest.approx(1 / (300 / 60))
    assert rate.iloc[2] == pytest.approx(2 / ((300 + 600) / 60))
    assert rate.iloc[3] == pytest.approx(2 / ((300 + 600 + 900) / 60))


def test_control_time_is_a_fraction_of_fight_time():
    out = _fighter_1({"fighter_b_ctrl_time": [150, 300, 0, 999]})   # controlled by opponents
    frac = out["ctrl_time_absorbed_frac_career_avg"]

    assert pd.isna(frac.iloc[0])
    assert frac.iloc[1] == pytest.approx(150 / 300)
    assert frac.iloc[2] == pytest.approx((150 + 300) / (300 + 600))
    assert frac.iloc[3] == pytest.approx((150 + 300 + 0) / (300 + 600 + 900))


def test_rate_denominator_ignores_bouts_where_the_stat_is_missing():
    """Regression: ctrl_time is absent for ~1.5% of fighter-appearances while
    total_fight_time never is. Rolling the two independently let the numerator
    average over fewer bouts than the denominator, which produced control-time
    fractions above 1.0 - impossible, since a fighter cannot be controlled for
    longer than the fight lasted.
    """
    out = _fighter_1({"fighter_b_ctrl_time": [np.nan, 531, 256, 3]},
                     durations=(193, 853, 1500, 802))
    frac = out["ctrl_time_absorbed_frac_career_avg"]

    # entering fight 3, only fight 2 recorded control time: 531s of an 853s bout.
    # the 193s of fight 1 must not reach the denominator.
    assert frac.iloc[2] == pytest.approx(531 / 853)
    assert frac.dropna().le(1.0).all()


def test_offensive_rate_matches_absorbed_pattern():
    """Item 2: the fighter's own output gets the same per-minute/fraction
    treatment as what they absorb, not just a raw per-fight count.
    """
    out = _fighter_1({"fighter_a_knockdowns": [1, 1, 0, 5],     # own knockdowns scored
                      "fighter_a_ctrl_time": [150, 300, 0, 999]})  # own control time

    kd_rate = out["knockdowns_per_min_career_avg"]
    assert pd.isna(kd_rate.iloc[0])
    assert kd_rate.iloc[3] == pytest.approx((1 + 1 + 0) / ((300 + 600 + 900) / 60))

    ctrl_frac = out["ctrl_time_frac_career_avg"]
    assert ctrl_frac.iloc[3] == pytest.approx((150 + 300 + 0) / (300 + 600 + 900))
    assert ctrl_frac.dropna().le(1.0).all()


def test_wide_table_keeps_attempts_and_accuracy_but_not_landed(toy_fights):
    featured = attach_prior_features(toy_fights, no_shrinkage())
    # the reparameterisation: rate + accuracy replace landed + attempts.
    # sig_strike_atts is a RATE_STAT, so attempts is normalised to a
    # per-minute figure rather than kept as a raw per-fight count.
    assert "fighter_a_prior_sig_strike_atts_per_min_career_avg" in featured.columns
    assert "fighter_a_prior_sig_strikes_accuracy_career_avg" in featured.columns
    assert "fighter_a_prior_sig_strikes_career_avg" not in featured.columns
    assert "fighter_a_prior_sig_strike_atts_career_avg" not in featured.columns


def test_both_offensive_and_defensive_accuracy_are_built(toy_fights):
    featured = attach_prior_features(toy_fights, no_shrinkage())
    assert "fighter_a_prior_sig_strikes_accuracy_career_avg" in featured.columns
    assert "fighter_a_prior_sig_strikes_hit_rate_against_career_avg" in featured.columns


def test_fight_time_has_no_absorbed_mirror(toy_fights):
    # it's the bout's duration - identical for both fighters, so an absorbed
    # variant would be pure duplication
    featured = attach_prior_features(toy_fights, no_shrinkage())
    assert "fighter_a_prior_total_fight_time_career_avg" in featured.columns
    assert "fighter_a_prior_total_fight_time_absorbed_career_avg" not in featured.columns


def test_offensive_volume_no_longer_a_raw_count_on_wide_table(toy_fights):
    featured = attach_prior_features(toy_fights, no_shrinkage())
    assert "fighter_a_prior_knockdowns_per_min_career_avg" in featured.columns
    assert "fighter_a_prior_knockdowns_career_avg" not in featured.columns
    assert "fighter_a_prior_ctrl_time_frac_career_avg" in featured.columns
    assert "fighter_a_prior_ctrl_time_career_avg" not in featured.columns


def test_is_title_detected_from_weight_class():
    fights = pd.DataFrame({
        "fights_url": ["f/1", "f/2"],
        "event_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "fighter_a_id": [1, 1],
        "fighter_b_id": [2, 3],
        "fighter_a_won": pd.array([True, True], dtype="boolean"),
        "outcome": ["fighter_a", "fighter_a"],
        "weight_class": ["UFC Lightweight Title Bout", "Lightweight Bout"],
        **{
            f"fighter_{side}_{stat}": [0, 0]
            for side in ("a", "b")
            for stat in MIRRORED_STATS + SHARED_STATS
        },
    })
    history = build_fighter_history(fights)
    assert history.loc[history["fights_url"] == "f/1", "is_title"].iloc[0] == True  # noqa: E712
    assert history.loc[history["fights_url"] == "f/2", "is_title"].iloc[0] == False  # noqa: E712


def test_is_five_round_from_long_non_title_fight():
    """A 3-round fight cannot exceed 900s, so a non-title fight running
    longer proves 5-round scheduling even without the title flag.
    """
    fights = pd.DataFrame({
        "fights_url": ["f/1", "f/2"],
        "event_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "fighter_a_id": [1, 1],
        "fighter_b_id": [2, 3],
        "fighter_a_won": pd.array([True, True], dtype="boolean"),
        "outcome": ["fighter_a", "fighter_a"],
        "weight_class": ["Lightweight Bout", "Lightweight Bout"],
        **{
            f"fighter_{side}_total_fight_time": [1500, 800]
            for side in ("a", "b")
        },
        **{
            f"fighter_{side}_{stat}": [0, 0]
            for side in ("a", "b")
            for stat in MIRRORED_STATS
        },
    })
    history = build_fighter_history(fights)
    assert history.loc[history["fights_url"] == "f/1", "is_five_round"].iloc[0] == True  # noqa: E712
    assert history.loc[history["fights_url"] == "f/2", "is_five_round"].iloc[0] == False  # noqa: E712


def test_prior_title_and_five_round_counts_are_cumulative(toy_fights):
    # override weight_class: fighter 1's 2nd fight is a title bout
    toy_fights = toy_fights.copy()
    toy_fights["weight_class"] = [
        "Lightweight Bout", "UFC Lightweight Title Bout",
        "Lightweight Bout", "Lightweight Bout",
    ]
    featured = attach_prior_features(toy_fights, no_shrinkage())
    fighter_1 = featured[featured["fighter_a_id"] == 1].sort_values("event_date")

    # entering fights 1,2,3,4: 0,0,1,1 prior title fights
    assert list(fighter_1["fighter_a_prior_title_fights"]) == [0.0, 0.0, 1.0, 1.0]
    # title fights are always 5-round, so the same pattern holds here
    assert list(fighter_1["fighter_a_prior_five_round_fights"]) == [0.0, 0.0, 1.0, 1.0]



# ------------------------------------------------------------------ shrinkage

def test_every_ratio_column_on_the_wide_table_is_shrunk():
    # 91 "total divided by evidence" columns get shrinkage; the 8 record
    # columns (counts and facts) don't, and nothing is left unaccounted for
    shrunk = {spec.name for spec in RATIO_FEATURES}
    assert len(shrunk) == len(RATIO_FEATURES) == 91
    assert shrunk == set(PRIOR_FEATURE_COLUMNS) - set(RECORD_COLUMNS)


def test_attach_refuses_to_run_without_an_alpha_for_every_ratio(toy_fights):
    with pytest.raises(ValueError, match="missing"):
        attach_prior_features(toy_fights, {})


def test_with_shrinkage_zero_attempts_gets_the_league_value():
    # fighter 1 never throws; the debutants they face land 5 of 10. With no
    # evidence of their own, fighter 1's accuracy is the league's - not NaN.
    out = _fighter_1({"fighter_b_sig_strikes": [5, 5, 5, 5],
                      "fighter_b_sig_strike_atts": [10, 10, 10, 10]},
                     alphas=_same_alpha_everywhere(10.0))
    acc = out["sig_strikes_accuracy_career_avg"]
    assert pd.isna(acc.iloc[0])                       # debut stays NaN
    assert acc.iloc[1:].tolist() == pytest.approx([0.5, 0.5, 0.5])


def test_win_rate_is_pulled_towards_one_half():
    # W, L, W, W with alpha = 2: two imaginary fights at 50%, then the real record
    out = _fighter_1(alphas=_same_alpha_everywhere(0.0) | {"win_rate": 2.0})
    win_rate = out["win_rate"]
    assert pd.isna(win_rate.iloc[0])
    assert win_rate.iloc[1:].tolist() == pytest.approx([(1 + 1) / (1 + 2),     # 1-0
                                                        (1 + 1) / (2 + 2),     # 1-1
                                                        (2 + 1) / (3 + 2)])    # 2-1



# ------------------------------------------------- draws and no-contests

def test_draw_counts_as_half_a_win_and_resets_the_streak():
    """W, W, draw, W. The draw is not a loss, but it ends the run."""
    out = _fighter_1(outcomes=("fighter_a", "fighter_a", "draw", "fighter_a"),
                     alphas=_same_alpha_everywhere(0.0))

    assert list(out["wins"]) == [0, 1, 2, 2]        # the draw adds no win
    assert list(out["losses"]) == [0, 0, 0, 0]      # and no loss
    assert list(out["fight_count"]) == [0, 1, 2, 3]  # but it is still experience
    assert list(out["streak"]) == [0, 1, 2, 0]      # +2 going in, reset after the draw

    # win_rate entering fight 4: 2 wins + half a draw over 3 fights
    assert out["win_rate"].iloc[3] == pytest.approx(2.5 / 3)


def test_no_contest_is_experience_but_leaves_the_record_alone():
    """W, W, no-contest, W. Octagon time, but nothing else changes."""
    out = _fighter_1(outcomes=("fighter_a", "fighter_a", "no_contest", "fighter_a"),
                     alphas=_same_alpha_everywhere(0.0))

    assert list(out["wins"]) == [0, 1, 2, 2]
    assert list(out["losses"]) == [0, 0, 0, 0]
    assert list(out["fight_count"]) == [0, 1, 2, 3]   # it happened
    assert list(out["streak"]) == [0, 1, 2, 2]        # ...but the run carries through

    # entering fight 4: 2 wins from 2 scored fights - the no-contest is not counted
    assert out["win_rate"].iloc[3] == pytest.approx(1.0)


def test_undecided_fights_still_feed_the_volume_stats():
    """The strikes landed in a no-contest are real and must count."""
    out = _fighter_1({"fighter_a_sig_strikes": [10, 10, 60, 0],
                      "fighter_a_sig_strike_atts": [100, 100, 100, 0]},
                     outcomes=("fighter_a", "fighter_a", "no_contest", "fighter_a"))
    # entering fight 4, accuracy pools all three prior bouts including the NC
    assert out["sig_strikes_accuracy_career_avg"].iloc[3] == pytest.approx(80 / 300)


def test_undecided_fights_count_towards_the_layoff():
    out = _fighter_1(outcomes=("fighter_a", "draw", "no_contest", "fighter_a"))
    # the gap is to the previous fight, whatever its result: 2020-03-01 -> 2020-04-01
    assert out["days_since_last_fight"].iloc[3] == 31
