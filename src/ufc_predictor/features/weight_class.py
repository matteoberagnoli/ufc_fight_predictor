"""Clean division and gender out of the raw `weight_class` string.

`weight_class` is already on the fight table and already pre-fight-safe - it's
set when the bout is made, not a result of it - but it is 108 distinct, messy
strings: "Lightweight Bout", "UFC Lightweight Title Bout", "Ultimate Fighter 15
Lightweight Tournament Title Bout" are all the lightweight division. Two clean
facts are pulled out of it:

`weight_class_lbs` - the division's upper weight limit (115 for strawweight up
to 265 for heavyweight), a single ordinal number instead of nine category
names. Found by checking which division name appears in the string. The only
ordering trap: "heavyweight" is a substring of "light heavyweight", so that one
has to be checked first - the same trap, and the same fix, as the `_SUFFIXES`
list in manifest.py.

67 "Catch Weight", 45 "Open Weight" and a handful of pre-weight-class-era bouts
(120 fights, 1.5%, almost all before 2014) have no fixed division. They get
`weight_class_lbs = NaN` rather than a guess - there is no honest number to put
there - which a tree-based model can place wherever it needs to.

`is_womens_bout` - "Women's" is in the string for most of them, but 8
catchweight bouts between two women have no gender word at all. Resolved the
same way a fighter's history fixes any other gap: anyone who has ever fought in
a women's division is a woman, so a catchweight bout between two such fighters
is a women's bout too.

Both are facts about the *bout*, not about either corner - fighter A and
fighter B always compete at the same weight, under the same rules. Like
`same_stance`, neither is a `diff_`; both read the same whichever fighter the
coin flip called A.
"""

import pandas as pd

# heaviest first: "heavyweight" is a substring of "light heavyweight", so it
# must be checked after, or every light heavyweight bout would match heavyweight
DIVISIONS = (
    ("heavyweight", 265),
    ("light heavyweight", 205),
    ("middleweight", 185),
    ("welterweight", 170),
    ("lightweight", 155),
    ("featherweight", 145),
    ("bantamweight", 135),
    ("flyweight", 125),
    ("strawweight", 115),
)
# re-sorted so the substring trap above can never bite, whatever order
# DIVISIONS is edited into: longest name first, always.
_DIVISIONS_SAFE_ORDER = tuple(sorted(DIVISIONS, key=lambda d: -len(d[0])))

NO_DIVISION = "no_division"


def division_from_weight_class(weight_class: pd.Series) -> pd.Series:
    """The division name found in each string, or "no_division" if none is.

    First match wins, longest name first, so "Light Heavyweight Bout" is
    claimed by "light heavyweight" before "heavyweight" gets a chance to match
    the same string - every light heavyweight string contains "heavyweight"
    too, so without this order (and without stopping at the first match) every
    light heavyweight bout would be overwritten as heavyweight.
    """
    lower = weight_class.str.lower()
    division = pd.Series(NO_DIVISION, index=weight_class.index, dtype="object")
    unclaimed = pd.Series(True, index=weight_class.index)
    for name, _ in _DIVISIONS_SAFE_ORDER:
        matched = unclaimed & lower.str.contains(name, regex=False)
        division = division.where(~matched, name.replace(" ", "_"))
        unclaimed &= ~matched
    return division.astype("string")


def weight_limit_lbs(division: pd.Series) -> pd.Series:
    """Each division's upper weight limit in pounds; NaN for `no_division`."""
    limits = {name.replace(" ", "_"): float(lbs) for name, lbs in DIVISIONS}
    return division.map(limits).astype("float64")


def is_womens_bout(fights: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.Series:
    """True for bouts between women.

    `weight_class` alone misses 8 women's catchweight bouts ("Catch Weight
    Bout" carries no gender word). A fight also counts as women's when either
    fighter has ever fought in a women's division. Pass the full featured
    table as `history` when `fights` is only a slice (the validation years,
    say), so whole careers are visible.
    """
    history = fights if history is None else history
    womens_division = history["weight_class"].str.contains("Women", na=False)
    women = set(history.loc[womens_division, "fighter_a_id"]) | set(history.loc[womens_division, "fighter_b_id"])
    return fights["fighter_a_id"].isin(women) | fights["fighter_b_id"].isin(women)


def attach_weight_class_features(featured: pd.DataFrame) -> pd.DataFrame:
    """Add `division` (readable, metadata), `weight_class_lbs` and `is_womens_bout` (features)."""
    out = featured.copy()
    out["division"] = division_from_weight_class(out["weight_class"])
    out["weight_class_lbs"] = weight_limit_lbs(out["division"])
    out["is_womens_bout"] = is_womens_bout(out).astype("boolean")
    return out
