"""Shrinkage: honest estimates from small samples.

Many features are a total divided by an amount of evidence:

    accuracy              landed   / attempts
    per-minute rates      events   / minutes fought
    control share         control  / minutes fought
    win rate              wins     / fights
    average fight time    seconds  / fights

With little evidence the raw ratio is mostly luck. A fighter who landed their
only takedown attempt reads 100%; one knockdown in an 8-second fight reads as
a knockdown every 8 seconds. Shrinkage pretends every fighter already has
`alpha` units of imaginary evidence at the league rate, then adds their real
record on top:

    shrunk = (total + alpha * league) / (evidence + alpha)

Little real evidence -> close to the league. Lots -> close to their own
record. With no evidence at all (never attempted a takedown) the result is
exactly the league value, which also removes the 0/0 NaNs. Debutants stay NaN:
with no history there is nothing to shrink, and "no history" is itself
information the model can use.

alpha is never typed by hand. `choose_alphas` tries every value in a fixed
grid and keeps the one whose estimates best predict what fighters actually did
next - using only fights before a cutoff date. Because it is learned from data
it follows the model's rules: fitted on the training years only, and saved
next to what was built with it (`save_alphas`).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ufc_predictor.config import ERA_START, LEAGUE_RATE_YEARS
from ufc_predictor.data.io import load_processed_json, save_processed_json


@dataclass(frozen=True)
class RatioFeature:
    """One shrunk column: shrink(total / evidence) over a fighter's earlier fights."""

    name: str             # output column, e.g. "takedowns_success_rate_against_career_avg"
    total: str            # history column added up on top, e.g. "takedowns_absorbed"
    evidence: str         # history column added up underneath, e.g. "takedown_atts_absorbed"
    league_total: str     # the fighter's-own-side version of the same stat: league-wide,
    league_evidence: str  # what one fighter lands is exactly what the other absorbs
    unit: str             # what alpha is counted in: "attempts", "minutes" or "fights"
    last_n: int | None = None       # None = whole career so far, 3 = last three fights
    judged_by: str = "next_fight"   # how alpha is chosen - see choose_alphas


def _grid(low: float, high: float) -> tuple[float, ...]:
    """10 values per tenfold step, each ~26% bigger than the last.

    Rounded to 4 significant digits so a saved alpha reads 125.9 rather than
    125.89254117941673. The grid is what it is; rounding it changes nothing.
    """
    steps = 10 * round(np.log10(high / low))
    return tuple(float(f"{x:.4g}") for x in np.geomspace(low, high, steps + 1))


ALPHA_GRIDS = {
    "attempts": _grid(1, 10_000),
    "minutes": _grid(0.1, 100_000),
    "fights": _grid(0.01, 1_000),
}


def league_rate(history: pd.DataFrame, total: str, evidence: str,
                years: int = LEAGUE_RATE_YEARS) -> pd.Series:
    """League-wide total / evidence over roughly the `years` before each fight.

    Only strictly earlier dates count: the fight itself, and every other fight
    on the same card, are left out. Fights where the total wasn't recorded are
    left out of the evidence too, so both sides cover the same fights.
    """
    t = history[total].astype(float)
    e = history[evidence].astype(float).where(t.notna())
    daily = (
        pd.DataFrame({"t": t.fillna(0), "e": e.fillna(0), "date": history["event_date"]})
        .groupby("date")[["t", "e"]].sum().sort_index()
    )
    days = daily.index.to_numpy()
    running_t = np.concatenate([[0.0], daily["t"].cumsum().to_numpy()])
    running_e = np.concatenate([[0.0], daily["e"].cumsum().to_numpy()])

    fight_day = history["event_date"].to_numpy()
    end = np.searchsorted(days, fight_day, side="left")      # first day on/after the fight: excluded
    start = np.searchsorted(days, fight_day - np.timedelta64(365 * years, "D"), side="left")
    t_sum = running_t[end] - running_t[start]
    e_sum = running_e[end] - running_e[start]
    rate = np.divide(t_sum, e_sum, out=np.full(len(e_sum), np.nan), where=e_sum > 0)
    return pd.Series(rate, index=history.index)


def prior_totals(values: pd.Series, fighter: pd.Series, last_n: int | None = None) -> pd.Series:
    """Sum of `values` over each fighter's earlier fights - never the current one.

    `last_n=None` adds up the whole career so far; `last_n=3` only the last
    three fights. NaN on a fighter's first fight: there is nothing to add up.
    Missing values count as zero, so mask the evidence where the total is
    missing (as _ingredients does) to keep both sides on the same fights.

    Rows must be sorted by fighter, then date - build_fighter_history does this.
    """
    x = values.astype(float).fillna(0)
    running = x.groupby(fighter).cumsum()                # up to and including this fight
    before = running.groupby(fighter).shift(1)           # <- the leakage guard: stop one fight earlier
    if last_n is None:
        return before
    dropped = running.groupby(fighter).shift(last_n + 1).fillna(0)   # everything older than the window
    return before - dropped


def shrink(prior_total: pd.Series, prior_evidence: pd.Series, league: pd.Series,
           alpha: float) -> pd.Series:
    """(total + alpha * league) / (evidence + alpha). alpha = 0 gives the raw ratio."""
    top = prior_total if alpha == 0 else prior_total + alpha * league
    bottom = prior_evidence + alpha
    return top / bottom.where(bottom > 0)                # 0/0 (only possible at alpha = 0) -> NaN


def check_alphas(specs: list[RatioFeature], alphas: dict[str, float]) -> None:
    """Fail loudly unless there is exactly one alpha per spec - e.g. a stale saved file."""
    names = {spec.name for spec in specs}
    missing, unexpected = sorted(names - set(alphas)), sorted(set(alphas) - names)
    if missing or unexpected:
        raise ValueError(
            f"alphas don't match the shrunk features: missing {missing[:5]}, "
            f"unexpected {unexpected[:5]}"
        )


def _ingredients(history: pd.DataFrame, spec: RatioFeature, leagues: dict):
    """Everything the formula needs for one spec.

    One code path for both building features and choosing alpha, so alpha is
    always tuned on exactly the column the model will see.
    """
    total = history[spec.total].astype(float)
    evidence = history[spec.evidence].astype(float).where(total.notna())   # same fights on both sides
    key = (spec.league_total, spec.league_evidence)
    if key not in leagues:
        leagues[key] = league_rate(history, *key)
    fighter = history["fighter_id"]
    return (
        prior_totals(total, fighter, spec.last_n),
        prior_totals(evidence, fighter, spec.last_n),
        leagues[key],
        total,
        evidence,
    )


def shrunk_features(history: pd.DataFrame, specs: list[RatioFeature],
                    alphas: dict[str, float]) -> pd.DataFrame:
    """One shrunk column per spec, lined up with `history`."""
    check_alphas(specs, alphas)
    leagues, columns = {}, {}
    for spec in specs:
        prior_t, prior_e, league, _, _ = _ingredients(history, spec, leagues)
        columns[spec.name] = shrink(prior_t, prior_e, league, alphas[spec.name])
    return pd.DataFrame(columns, index=history.index)


def choose_alphas(history: pd.DataFrame, specs: list[RatioFeature], until: str,
                  fights: pd.DataFrame | None = None,
                  start: str = ERA_START) -> dict[str, float]:
    """Pick alpha for every spec from fights in [start, until) - nothing after `until`.

    judged_by="next_fight" (every stat): in each fighter-fight of the period,
    the shrunk estimate from the fighter's earlier fights is a prediction of
    what they do in this fight. It is scored by the squared miss, weighted by
    this fight's evidence (attempts or minutes), summed over the period. The
    alpha with the smallest total wins.

    judged_by="who_wins" (win rate): nobody wins on their own, so a record is
    judged against the opponent's. Draws and no-contests are skipped - there
    is no winner to separate - though they still count as history. The alpha whose fighter_a-minus-fighter_b
    difference best separates the fights fighter_a won from the ones they lost
    wins (see _ranking_score).

    Fights before `start` still count as history; they just aren't scored.
    Deterministic: a fixed grid, fixed scores, and ties go to the smaller alpha.
    """
    dates = history["event_date"]
    in_period = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(until))
    leagues, alphas = {}, {}
    for spec in specs:
        prior_t, prior_e, league, total, evidence = _ingredients(history, spec, leagues)
        grid = ALPHA_GRIDS[spec.unit]
        if spec.judged_by == "who_wins":
            alphas[spec.name] = _best_alpha_who_wins(
                history, fights, prior_t, prior_e, league, grid, start, until
            )
        else:
            rows = in_period & prior_t.notna() & prior_e.notna() & league.notna() & (evidence > 0)
            alphas[spec.name] = _best_alpha_next_fight(
                prior_t[rows], prior_e[rows], league[rows], total[rows], evidence[rows], grid
            )
    return alphas


def _best_alpha_next_fight(prior_t, prior_e, league, total, evidence, grid) -> float:
    T, E, L = prior_t.to_numpy(), prior_e.to_numpy(), league.to_numpy()
    w = evidence.to_numpy()        # this fight's attempts/minutes: more evidence, more weight
    y = total.to_numpy() / w       # what the fighter actually did in this fight
    misses = [np.sum(w * (y - (T + a * L) / (E + a)) ** 2) for a in grid]
    return grid[int(np.argmin(misses))]


def _best_alpha_who_wins(history, fights, prior_t, prior_e, league, grid, start, until) -> float:
    if fights is None:
        raise ValueError("judging by who wins needs the wide fights table: pass fights=")
    fight_dates = fights["event_date"]
    # a draw or no-contest has no winner to separate, so it cannot be scored
    period = fights[
        (fight_dates >= pd.Timestamp(start))
        & (fight_dates < pd.Timestamp(until))
        & fights["fighter_a_won"].notna()
    ]

    # line each history row up with its fight's fighter_a / fighter_b slot
    known = history["fighter_id"].notna()
    by_fight_and_fighter = pd.MultiIndex.from_arrays(
        [history.loc[known, "fights_url"], history.loc[known, "fighter_id"].astype(float)]
    )

    def slot(series: pd.Series, side: str) -> np.ndarray:
        lookup = pd.Series(series[known].to_numpy(), index=by_fight_and_fighter)
        wanted = pd.MultiIndex.from_arrays(
            [period["fights_url"], period[f"{side}_id"].astype(float)]
        )
        return lookup.reindex(wanted).to_numpy()

    a = [slot(s, "fighter_a") for s in (prior_t, prior_e, league)]
    b = [slot(s, "fighter_b") for s in (prior_t, prior_e, league)]
    usable = ~np.isnan(np.column_stack(a + b)).any(axis=1)     # both fighters have history
    (ta, ea, la), (tb, eb, lb) = [x[usable] for x in a], [x[usable] for x in b]
    won = period["fighter_a_won"].astype(bool).to_numpy()[usable]

    scores = [
        _ranking_score((ta + g * la) / (ea + g) - (tb + g * lb) / (eb + g), won) for g in grid
    ]
    return grid[int(np.argmax(scores))]


def _ranking_score(score: np.ndarray, won: np.ndarray) -> float:
    """How well `score` separates the fights fighter_a won from the ones they lost.

    Take every pair made of one won fight and one lost fight: the share of
    pairs where the won fight has the higher score, ties counting half.
    1.0 is perfect; 0.5 is no better than a coin flip.
    """
    n_won = int(won.sum())
    n_lost = len(won) - n_won
    if n_won == 0 or n_lost == 0:
        return 0.5
    ranks = pd.Series(score).rank().to_numpy()      # ties share the average rank
    return float((ranks[won].sum() - n_won * (n_won + 1) / 2) / (n_won * n_lost))


def describe_alphas(specs: list[RatioFeature], alphas: dict[str, float]) -> pd.DataFrame:
    """alphas as a readable table: feature, alpha, and the unit alpha is counted in."""
    return pd.DataFrame(
        [{"feature": s.name, "alpha": alphas[s.name], "unit": s.unit} for s in specs]
    )


def save_alphas(alphas: dict[str, float], until: str, start: str = ERA_START) -> None:
    """Save alphas with the period they were learned from.

    Changing alpha changes the features of every past fight, so a trained
    model must always travel with the alphas its features were built with.
    """
    save_processed_json(
        {
            "fitted_on": {"from": start, "until": until},
            "league_rate_years": LEAGUE_RATE_YEARS,
            "alphas": alphas,
        },
        "alphas",
    )


def load_alphas() -> dict[str, float]:
    return load_processed_json("alphas")["alphas"]
