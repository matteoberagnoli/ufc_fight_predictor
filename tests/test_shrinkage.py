import numpy as np
import pandas as pd
import pytest

from ufc_predictor.features.shrinkage import (
    ALPHA_GRIDS,
    RatioFeature,
    _ranking_score,
    check_alphas,
    choose_alphas,
    league_rate,
    prior_totals,
    shrink,
)

ACCURACY = RatioFeature("acc", "landed", "attempts", "landed", "attempts", "attempts")
WIN_RATE = RatioFeature("win_rate", "won", "fights", "won", "fights", "fights", judged_by="who_wins")


# ---------------------------------------------------------------- the formula

def test_shrink_matches_the_worked_examples():
    # takedowns with a league rate of 36.5% and alpha = 10: 0/0, 1/1 and 30/60
    league = pd.Series([0.365] * 3)
    out = shrink(pd.Series([0.0, 1.0, 30.0]), pd.Series([0.0, 1.0, 60.0]), league, alpha=10)
    assert out.tolist() == pytest.approx([0.365, 4.65 / 11, 33.65 / 70])


def test_alpha_zero_is_the_raw_ratio():
    no_league = pd.Series([np.nan, np.nan])
    out = shrink(pd.Series([30.0, 0.0]), pd.Series([60.0, 0.0]), no_league, alpha=0)
    assert out.iloc[0] == 0.5          # the league isn't used at all...
    assert np.isnan(out.iloc[1])       # ...and 0/0 stays undefined


def test_grids_step_evenly_on_a_log_scale():
    grid = ALPHA_GRIDS["attempts"]
    assert (grid[0], grid[-1], len(grid)) == (1, 10_000, 41)
    ratios = np.array(grid[1:]) / np.array(grid[:-1])
    assert ratios == pytest.approx(10 ** 0.1, rel=1e-3)     # each value ~26% above the last


# ------------------------------------------------------- the building blocks

def test_prior_totals_only_add_up_earlier_fights():
    # two fighters, rows sorted by fighter then date, as build_fighter_history leaves them
    fighter = pd.Series([1, 1, 1, 1, 1, 2, 2])
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0])

    career = prior_totals(values, fighter)
    last3 = prior_totals(values, fighter, last_n=3)

    assert np.isnan(career.iloc[0]) and np.isnan(career.iloc[5])   # first fights: no history
    assert career.iloc[1:5].tolist() == [1, 3, 6, 10]
    assert last3.iloc[1:5].tolist() == [1, 3, 6, 9]    # entering fight 5: 2+3+4, fight 1 dropped out
    assert career.iloc[6] == 10                         # fighter 2 never sees fighter 1's fights


def test_prior_totals_count_a_missing_value_as_zero():
    out = prior_totals(pd.Series([np.nan, 5.0, 1.0]), pd.Series([1, 1, 1]))
    assert out.tolist()[1:] == [0, 5]


def test_league_rate_uses_only_strictly_earlier_dates():
    history = pd.DataFrame({
        "event_date": pd.to_datetime(["2015-01-01", "2015-01-01", "2016-01-01",
                                      "2016-01-01", "2018-06-01", "2018-06-01"]),
        "landed": [1.0, 3.0, 5.0, 5.0, 9.0, 9.0],
        "attempts": [10.0] * 6,
    })
    rate = league_rate(history, "landed", "attempts", years=3)
    assert rate.iloc[:2].isna().all()               # nothing before the first card
    assert rate.iloc[2] == pytest.approx(4 / 20)    # 2016 sees the 2015 card, not its own
    # 2018-06 sees 2016 but not 2015, which is more than 3 years back
    assert rate.iloc[4] == pytest.approx(10 / 20)


def test_league_rate_ignores_fights_where_the_total_was_not_recorded():
    history = pd.DataFrame({
        "event_date": pd.to_datetime(["2015-01-01", "2015-01-01", "2016-01-01"]),
        "landed": [np.nan, 3.0, 0.0],
        "attempts": [10.0, 10.0, 10.0],
    })
    # the first fight's 10 attempts must not reach the bottom of the ratio
    assert league_rate(history, "landed", "attempts").iloc[2] == pytest.approx(3 / 10)


def test_ranking_score_basics():
    won = np.array([True, True, False, False])
    assert _ranking_score(np.array([4, 3, 2, 1]), won) == 1.0    # winners always ranked higher
    assert _ranking_score(np.array([1, 2, 3, 4]), won) == 0.0    # always the wrong way round
    assert _ranking_score(np.array([1, 1, 1, 1]), won) == 0.5    # no information


def test_check_alphas_rejects_missing_and_unexpected_names():
    check_alphas([ACCURACY], {"acc": 10.0})
    with pytest.raises(ValueError, match="missing"):
        check_alphas([ACCURACY], {})
    with pytest.raises(ValueError, match="unexpected"):
        check_alphas([ACCURACY], {"acc": 10.0, "stale_column": 3.0})


# ------------------------------------------ choosing alpha: does the test work?

def _made_up_fighters(fight_noise: float, n_fighters: int = 5_000, n_fights: int = 10,
                      skill_spread: float = 30, league: float = 0.37) -> pd.DataFrame:
    """Fighters whose true accuracy we set ourselves, so the right alpha is known.

    True skill comes from a spread worth `skill_spread` attempts. Without fight
    noise, the best possible alpha is exactly that. With fight noise (each fight's
    real accuracy wobbles around the fighter's skill, like facing different
    opponents) the evidence is weaker, so the best alpha is bigger.
    """
    rng = np.random.default_rng(1926)
    skill = rng.beta(skill_spread * league, skill_spread * (1 - league), n_fighters)
    wobble = rng.normal(0, fight_noise, (n_fighters, n_fights))
    rate = 1 / (1 + np.exp(-(np.log(skill / (1 - skill))[:, None] + wobble)))
    attempts = rng.poisson(70, (n_fighters, n_fights))
    landed = rng.binomial(attempts, rate)
    return pd.DataFrame({
        "fighter_id": np.repeat(np.arange(n_fighters), n_fights),
        "fights_url": [f"{i}-{j}" for i in range(n_fighters) for j in range(n_fights)],
        # everyone's n-th fight on the same day, a month apart
        "event_date": pd.Timestamp("2014-01-01")
        + pd.to_timedelta(np.tile(np.arange(n_fights) * 30, n_fighters), unit="D"),
        "landed": landed.ravel().astype(float),
        "attempts": attempts.ravel().astype(float),
    })


def _alpha_for(history: pd.DataFrame, until: str = "2100-01-01") -> float:
    return choose_alphas(history, [ACCURACY], until=until, start="2000-01-01")["acc"]


def test_choose_alphas_finds_the_right_answer_when_it_is_known():
    # the right answer is 30; the nearest grid values are 25.12 and 31.62
    assert 25 <= _alpha_for(_made_up_fighters(fight_noise=0)) <= 40


def test_fight_to_fight_noise_calls_for_more_shrinkage():
    calm = _alpha_for(_made_up_fighters(fight_noise=0))
    noisy = _alpha_for(_made_up_fighters(fight_noise=0.5))
    assert noisy > 3 * calm


def test_choose_alphas_is_deterministic():
    history = _made_up_fighters(fight_noise=0.5, n_fighters=1_000)
    assert _alpha_for(history) == _alpha_for(history)


def test_choose_alphas_never_looks_at_or_after_until():
    history = _made_up_fighters(fight_noise=0.5, n_fighters=1_000)
    until = "2014-06-01"
    before = _alpha_for(history, until=until)

    rewritten = history.copy()
    later = rewritten["event_date"] >= pd.Timestamp(until)
    rewritten.loc[later, "landed"] = rewritten.loc[later, "attempts"]   # rewrite the future
    assert _alpha_for(rewritten, until=until) == before


# ---------------------------------------------------- choosing alpha by who wins

def _made_up_league(n_fighters: int = 400, n_rounds: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fighters with a hidden strength; the stronger one usually - not always - wins."""
    rng = np.random.default_rng(7)
    strength = rng.normal(0, 1, n_fighters)
    history, fights = [], []
    for r in range(n_rounds):
        date = pd.Timestamp("2014-01-01") + pd.Timedelta(days=30 * r)
        order = rng.permutation(n_fighters)
        for k in range(0, n_fighters, 2):
            a, b = int(order[k]), int(order[k + 1])
            a_won = bool(rng.random() < 1 / (1 + np.exp(strength[b] - strength[a])))
            url = f"r{r}-{k}"
            fights.append({"fights_url": url, "event_date": date, "fighter_a_id": a,
                           "fighter_b_id": b, "fighter_a_won": a_won})
            history += [{"fighter_id": a, "fights_url": url, "event_date": date, "won": a_won},
                        {"fighter_id": b, "fights_url": url, "event_date": date, "won": not a_won}]
    history = pd.DataFrame(history).assign(fights=1.0)
    history = history.sort_values(["fighter_id", "event_date", "fights_url"]).reset_index(drop=True)
    return history, pd.DataFrame(fights)


def test_who_wins_picks_a_moderate_alpha_that_beats_both_extremes():
    history, fights = _made_up_league()
    alpha = choose_alphas(history, [WIN_RATE], until="2100-01-01",
                          fights=fights, start="2000-01-01")["win_rate"]
    grid = ALPHA_GRIDS["fights"]
    assert grid[0] < alpha < grid[-1]


def test_who_wins_needs_the_fights_table():
    history, _ = _made_up_league(n_fighters=20, n_rounds=2)
    with pytest.raises(ValueError, match="fights="):
        choose_alphas(history, [WIN_RATE], until="2100-01-01", start="2000-01-01")
