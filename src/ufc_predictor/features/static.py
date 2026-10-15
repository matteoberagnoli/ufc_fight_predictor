"""Static (time-invariant) fighter attributes: age, height, reach, stance.

Age and height are already present per-fight in fights.parquet
(`fighter_a_age_during`, `fighter_a_height_cm`, ...), sourced from
ufc_event_fight_stats.csv - they're the fighter's actual age/height *at that
specific fight*, not a scraped-today snapshot, so they're already safe and
just need their diffs computed.

Reach and stance aren't in the fight table at all and have to be joined in
from ufc_fighters.csv. Unlike the career-aggregate columns flagged in
ROADMAP.md (wins, SLpM, ...), reach and stance don't change over a fighter's
career, so joining the fighters.csv value is safe regardless of when the
fight took place - there's no "as of" date attached to a person's arm span.

Weight is deliberately excluded: a fighter's competing weight changes across
their career as they move divisions, so unlike reach/stance a single joined
value wouldn't be safe. `weight_class` on the fight itself already captures
what they actually weighed in at for each bout, which is the correct source
if weight is ever needed.

Two derived columns on top of the joined values:

`ape_index_cm` (reach minus height) is a distinct scouting concept from
either alone - two fighters can share identical reach but very different
frames depending on height, and "long arms for your height" is specifically
what matters for range control.

`same_stance` is a fact about the *matchup*, not either fighter alone, so it
isn't diffed the way a number would be - "is this an open-stance fight" is
identical information regardless of which fighter got randomly labelled A or
B. What it can't say is *which* corner holds the rarer stance in a mixed
fight. A flag like "fighter_a is the southpaw facing an orthodox opponent"
would bake in an assumed direction of advantage, so `attach_stance_features`
uses signed differences instead (see there): they say who is the southpaw
and leave the direction of any edge for the model to discover.

Tested on Day 9 (08_feature_questions.ipynb) and **not adopted**: with its
own tuned settings, the model with the stance columns was slightly worse in
cross-validation (better in 1 of 8 seeds) and slightly better on validation,
both by noise-sized amounts. That matches the raw data - southpaws beat
orthodox fighters 52.2% of the time in 2014-2022, inside the coin-flip range.
So `attach_stance_features` is not part of features/pipeline.py; it stays
here so the comparison can be re-run.
"""

import pandas as pd

# The stances that give a corner something its opponent may rarely face.
# Orthodox is the default (74% of appearances); "Open Stance" and "Sideways"
# are too rare to learn from (30 appearances between them) and count as neither.
SIDED_STANCES = ("Southpaw", "Switch")


def attach_static_features(featured: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    """Add joined reach/stance and age/height/reach/ape-index diffs, plus same_stance."""
    out = featured.copy()

    for side in ("fighter_a", "fighter_b"):
        side_static = fighters[["fighter_id", "fighter_reach_cm", "fighter_stance"]].rename(
            columns={
                "fighter_id": f"{side}_id",
                "fighter_reach_cm": f"{side}_reach_cm",
                "fighter_stance": f"{side}_stance",
            }
        )
        out = out.merge(side_static, on=f"{side}_id", how="left", validate="many_to_one")
        out[f"{side}_ape_index_cm"] = out[f"{side}_reach_cm"] - out[f"{side}_height_cm"]

    # named "age gap" (Day 12): it is a difference, and the model never sees either age itself
    out["diff_static_age_gap"] = out["fighter_a_age_during"] - out["fighter_b_age_during"]
    out["diff_static_height_cm"] = out["fighter_a_height_cm"] - out["fighter_b_height_cm"]
    out["diff_static_reach_cm"] = out["fighter_a_reach_cm"] - out["fighter_b_reach_cm"]
    out["diff_static_ape_index_cm"] = out["fighter_a_ape_index_cm"] - out["fighter_b_ape_index_cm"]

    out["same_stance"] = (out["fighter_a_stance"] == out["fighter_b_stance"]).astype("boolean")
    out.loc[out["fighter_a_stance"].isna() | out["fighter_b_stance"].isna(), "same_stance"] = pd.NA

    return out


def attach_stance_features(featured: pd.DataFrame) -> pd.DataFrame:
    """Add diff_static_southpaw and diff_static_switch: which corner has the rarer stance.

    Each is fighter A's indicator minus fighter B's:

        A southpaw, B orthodox    diff_static_southpaw = +1
        A orthodox, B southpaw                         = -1
        both or neither                                =  0

    A signed difference carries the direction (*who* is the southpaw) without
    assuming which way any edge goes: the model learns whether +1 helps. It
    also flips sign when the corners swap, like every other diff_ column, so
    model.mirror handles it with no special case. Unknown stance on either
    side gives NaN, the same rule as same_stance.

    Needs fighter_a_stance / fighter_b_stance, which attach_static_features adds.
    """
    out = featured.copy()
    a, b = out["fighter_a_stance"], out["fighter_b_stance"]
    known = a.notna() & b.notna()
    for stance in SIDED_STANCES:
        diff = a.eq(stance).astype("float64") - b.eq(stance).astype("float64")
        out[f"diff_static_{stance.lower()}"] = diff.where(known)
    return out
