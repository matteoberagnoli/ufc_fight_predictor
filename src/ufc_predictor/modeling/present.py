"""Presenting the finished model: rankings and fight previews built from its own pieces.

Nothing here feeds back into the model. It reads what the model already uses -
the Elo rating and the 40 inputs - and lays it out for people.

Pound-for-pound (`p4p`): every active fighter ranked by their Elo after their
last fight. Men and women are ranked separately, since they never meet. Elo
measures results against the opponents a fighter actually faced, with each win
weighted by how it came; it knows nothing about weight, so "pound for pound" is
literal here: one scale for every division.

Inactivity costs ranking points. From 8 months without a
fight the rating used for the ranking falls by 0.5 x (months - 8)^2 points:
8 at 12 months, 50 at 18, 128 at 24. The size matches what the data shows:
fighters returning after 12-18 months won 2.9 points less often than their Elo
expected, after 18-24 months 11.8 points less (about 80 Elo points). After 18
months without a fight a fighter is not ranked at all: retirement leaves no
results to learn from, so it needs a rule. This is for the ranking only; the
model's own Elo input is unchanged (it already sees days_since_last_fight).
"""
import pandas as pd

from ufc_predictor.features.elo import current_ratings

ACTIVE_MONTHS = 18     # fought within this long before the ranking date
MIN_FIGHTS = 3         # a rating needs a few results before it means much
DECAY_START_MONTHS = 8
DECAY_PER_MONTH_SQUARED = 0.5
DAYS_PER_MONTH = 30.44


def inactivity_penalty(months_inactive) -> pd.Series:
    """Ranking points lost to inactivity: 0 before 8 months, then 0.5 x (months - 8)^2."""
    extra = (pd.Series(months_inactive, dtype=float) - DECAY_START_MONTHS).clip(lower=0)
    return DECAY_PER_MONTH_SQUARED * extra ** 2


def fighter_records(fights: pd.DataFrame) -> pd.DataFrame:
    """One row per fighter: UFC record, fights, last fight date, last division, women's or not."""
    sides = []
    for side, other in (("fighter_a", "fighter_b"), ("fighter_b", "fighter_a")):
        sides.append(pd.DataFrame({
            "fighter_id": fights[f"{side}_id"], "name": fights[f"{side}_name"],
            "date": fights["event_date"], "url": fights["fights_url"], "division": fights["division"],
            "womens": fights["is_womens_bout"].astype(bool),
            "win": fights["outcome"].eq(side), "loss": fights["outcome"].eq(other),
            "draw": fights["outcome"].eq("draw"),
        }))
    long = pd.concat(sides, ignore_index=True).sort_values(["fighter_id", "date", "url"])
    last = long.groupby("fighter_id").tail(1).set_index("fighter_id")
    totals = long.groupby("fighter_id")[["win", "loss", "draw"]].sum()
    out = last[["name", "date", "division", "womens"]].rename(columns={"date": "last_fight"})
    out["fights"] = long.groupby("fighter_id").size()
    out["record"] = (totals["win"].astype(str) + "-" + totals["loss"].astype(str)
                     + totals["draw"].map(lambda d: f"-{d}" if d else ""))
    return out


def p4p(fights: pd.DataFrame, elo_k: float, elo_weights: dict | None, top: int = 15,
        as_of=None, active_months: int = ACTIVE_MONTHS, min_fights: int = MIN_FIGHTS) -> dict:
    """{"men": table, "women": table}: active fighters ranked by Elo minus the inactivity penalty.

    `fights` is the full fights table (every fight since 1994, with `division`
    and `is_womens_bout`, i.e. the featured table). `as_of` defaults to the last
    event in the data. Active: fought within `active_months` of it, with at least
    `min_fights` UFC fights. `rating` is what the ranking uses; `elo` is the
    model's rating before the penalty.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else fights["event_date"].max()
    fights = fights[fights["event_date"] <= as_of]
    table = fighter_records(fights).join(current_ratings(fights, elo_k, elo_weights))
    table["months_inactive"] = (as_of - table["last_fight"]).dt.days / DAYS_PER_MONTH
    table["penalty"] = inactivity_penalty(table["months_inactive"]).to_numpy()
    table["rating"] = table["elo"] - table["penalty"]
    active = table[(table["last_fight"] >= as_of - pd.DateOffset(months=active_months))
                   & (table["fights"] >= min_fights)]
    out = {}
    for label, is_women in (("men", False), ("women", True)):
        ranked = active[active["womens"] == is_women].sort_values("rating", ascending=False).head(top).copy()
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        for column in ("rating", "elo", "penalty"):
            ranked[column] = ranked[column].round(0).astype(int)
        ranked["months_inactive"] = ranked["months_inactive"].round(1)
        ranked["last_fight"] = ranked["last_fight"].dt.date
        out[label] = ranked[["rank", "name", "rating", "elo", "penalty", "months_inactive",
                             "division", "record", "last_fight"]].reset_index(drop=True)
    return out
