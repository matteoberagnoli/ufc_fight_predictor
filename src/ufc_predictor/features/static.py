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
"""

import pandas as pd


def attach_static_features(featured: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    """Add joined reach/stance and age/height/reach diffs to a featured fight table."""
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

    out["diff_static_age"] = out["fighter_a_age_during"] - out["fighter_b_age_during"]
    out["diff_static_height_cm"] = out["fighter_a_height_cm"] - out["fighter_b_height_cm"]
    out["diff_static_reach_cm"] = out["fighter_a_reach_cm"] - out["fighter_b_reach_cm"]

    return out
