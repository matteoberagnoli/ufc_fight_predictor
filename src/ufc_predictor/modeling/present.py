"""Presenting the finished model: rankings and fight previews built from its own pieces.

Nothing here feeds back into the model. It reads what the model already uses -
the Elo rating and the 40 inputs - and lays it out for people.

Pound-for-pound (`p4p`): every active fighter ranked by their Elo after their
last fight. Men and women are ranked separately, since they never meet. Elo
measures results against the opponents a fighter actually faced, with each win
weighted by how it came; it knows nothing about weight, so "pound for pound" is
literal here: one scale for every division.
"""
import pandas as pd

from ufc_predictor.features.elo import current_ratings

ACTIVE_MONTHS = 24     # fought within this long before the last event in the data
MIN_FIGHTS = 3         # a rating needs a few results before it means much


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
    """{"men": table, "women": table}: active fighters ranked by their current Elo.

    `fights` is the full fights table (every fight since 1994, with `division`
    and `is_womens_bout`, i.e. the featured table). `as_of` defaults to the last
    event in the data. Active: fought within `active_months` of it, with at least
    `min_fights` UFC fights.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else fights["event_date"].max()
    fights = fights[fights["event_date"] <= as_of]
    table = fighter_records(fights).join(current_ratings(fights, elo_k, elo_weights))
    active = table[(table["last_fight"] >= as_of - pd.DateOffset(months=active_months))
                   & (table["fights"] >= min_fights)]
    out = {}
    for label, is_women in (("men", False), ("women", True)):
        ranked = active[active["womens"] == is_women].sort_values("elo", ascending=False).head(top).copy()
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        ranked["elo"] = ranked["elo"].round(0).astype(int)
        ranked["last_fight"] = ranked["last_fight"].dt.date
        out[label] = ranked[["rank", "name", "elo", "division", "record", "last_fight"]].reset_index(drop=True)
    return out
