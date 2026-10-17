"""Where the raw data comes from, and how to refresh it.

Source: the Kaggle dataset `scarekrow/ufc-data` - every UFC fight on
ufcstats.com from UFC 2 (1994) to the latest card, kept current by its
maintainer's scrape after every event. Chosen on Day 9 after comparing about
a hundred Kaggle UFC datasets: it reproduces all 7,840 fights of the
previous source (dates, strikes, knockdowns and fight times identical;
control time within a few seconds), adds 82 early fights that source was
missing, and adds every fight since November 2024.

ufcstats.com itself can no longer be scraped with plain requests - it serves
a JavaScript "Loading..." page to anything that isn't a real browser - so a
refresh means downloading the newest version of the Kaggle dataset:

    refresh()            newest version: download, check, convert, write
    refresh(version=18)  a pinned version, to rebuild an older state exactly

The rest of the pipeline never sees the source's own format.
`to_pipeline_tables` converts it into the three tables data/io.py has always
read (ufc_event_fight_stats.csv, ufc_events.csv, ufc_fighters.csv), with the
same columns as before. Only what the pipeline uses is carried over - which
also keeps out two column families that would leak the future: the
fighters' *current* rankings and *current* career totals, both captured at
scrape time, not at the time of each fight.

Conversion rules worth knowing:

  * the source lists the winner first in every decided fight. That is the
    fighter-ordering trap data/randomize.py exists for, and its coin flip
    neutralises it exactly as it did for the old source's pre-2010 fights.
  * URLs are written as http://www.ufcstats.com/... - the old source's form.
    The coin flip hashes the fight URL, so keeping the form keeps every
    fight's flip the same as before.
  * fighter ids are the first 13 hex digits of the ufcstats fighter id, read
    as a number: stable across refreshes, no lookup table to maintain. 13,
    not more: the largest such number (4.5e15) still fits exactly in a
    float64, and any column with a blank in it becomes float64 - with 15
    digits, winner ids silently rounded and stopped matching their fighter.
  * strikes by target (head/body/leg) and position (distance/clinch/ground)
    are the sums of the source's round-by-round columns - verified equal to
    the old source's totals on every shared fight.
  * control time is "not recorded" (blank), not zero, when every round's
    control time is blank - the 1994-99 fights. Shrinkage treats a blank as
    missing evidence and a zero as real evidence, so the difference matters.
  * no winner + a decision, or + time expiring (the early superfights) = draw;
    no winner + anything else (overturned, failed drug test, accidental
    foul) = no-contest.
"""

import glob
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_predictor.config import RAW_DIR

KAGGLE_DATASET = "scarekrow/ufc-data"
SOURCE_FILE_PATTERN = "UFC_full_data*.csv"

ROUNDS = range(1, 6)

# The columns the pipeline reads, in the order the previous source had them.
FIGHT_STATS_COLUMNS = [
    "f1_id", "f2_id", "f1_name", "f2_name", "weight_class",
    "f1_age_during", "f2_age_during", "f1_height_cm", "f2_height_cm",
    "f1_knockdowns", "f2_knockdowns", "f1_sig_strike_atts", "f2_sig_strike_atts",
    "f1_sig_strikes", "f2_sig_strikes", "f1_tot_strike_atts", "f2_tot_strike_atts",
    "f1_tot_strikes", "f2_tot_strikes", "f1_takedown_atts", "f2_takedown_atts",
    "f1_takedowns", "f2_takedowns", "f1_submissions", "f2_submissions",
    "f1_reversals", "f2_reversals", "f1_ctrl_time", "f2_ctrl_time",
    "f1_head_strike_atts", "f2_head_strike_atts", "f1_head_strikes", "f2_head_strikes",
    "f1_body_strike_atts", "f2_body_strike_atts", "f1_body_strikes", "f2_body_strikes",
    "f1_leg_strike_atts", "f2_leg_strike_atts", "f1_leg_strikes", "f2_leg_strikes",
    "f1_dist_strike_atts", "f2_dist_strike_atts", "f1_dist_strikes", "f2_dist_strikes",
    "f1_clinch_atts", "f2_clinch_atts", "f1_clinchs", "f2_clinchs",
    "f1_ground_atts", "f2_ground_atts", "f1_grounds", "f2_grounds",
    "f1_total_fight_time", "f2_total_fight_time", "result", "method", "fights_url", "event_url",
]

# How a fight ended, in the four groups the Elo rating can weight differently
# (features/elo.py). Post-fight information about the fight itself: it updates
# ratings *after* a fight and must never be a model input (manifest.py lists
# it with the raw post-fight columns). DQ and the winnerless endings stay "other".
METHODS = {
    "KO/TKO": "finish", "Submission": "finish", "TKO - Doctor's Stoppage": "finish",
    "Decision - Unanimous": "unanimous",
    "Decision - Split": "narrow", "Decision - Majority": "narrow",
}

# fight totals: our stat name -> the source's column suffix
TOTALS = {
    "knockdowns": "knockdowns",
    "sig_strike_atts": "sig_strikes_att", "sig_strikes": "sig_strikes_succ",
    "tot_strike_atts": "total_strikes_att", "tot_strikes": "total_strikes_succ",
    "takedown_atts": "takedown_att", "takedowns": "takedown_succ",
    "submissions": "submission_att", "reversals": "reversals",
}

# summed over rounds: our stat name -> the source's per-round column suffix
ROUND_SUMS = {
    "head_strike_atts": "head_att", "head_strikes": "head_succ",
    "body_strike_atts": "body_att", "body_strikes": "body_succ",
    "leg_strike_atts": "leg_att", "leg_strikes": "leg_succ",
    "dist_strike_atts": "distance_att", "dist_strikes": "distance_succ",
    "clinch_atts": "clinch_att", "clinchs": "clinch_succ",
    "ground_atts": "ground_att", "grounds": "ground_succ",
}

BIO = ["name", "fighter_dob", "fighter_height_cm", "fighter_weight_lbs", "fighter_reach_cm", "fighter_stance"]


def required_columns() -> list[str]:
    """Every source column the conversion reads. A refresh fails loudly if one disappears."""
    columns = ["fight_url", "event_url", "event_name", "event_date", "weight_class", "title_fight",
               "winner", "result", "result_details", "finish_round", "finish_time"]
    for k in (1, 2):
        columns += [f"f_{k}_url", f"f_{k}_name"] + [f"f_{k}_{c}" for c in BIO if c != "name"]
        columns += [f"f_{k}_{c}" for c in TOTALS.values()] + [f"f_{k}_ctrl_time_sec"]
        columns += [f"f_{k}_r{r}_{c}" for r in ROUNDS for c in ROUND_SUMS.values()]
        columns += [f"f_{k}_r{r}_ctrl" for r in ROUNDS]
    return columns


def check_columns(wide: pd.DataFrame) -> None:
    missing = sorted(set(required_columns()) - set(wide.columns))
    if missing:
        raise ValueError(
            f"the source's format has changed: {len(missing)} column(s) the conversion needs are "
            f"gone, e.g. {missing[:5]}. Update data/source.py before refreshing."
        )


def _ufcstats_hash(url: pd.Series) -> pd.Series:
    return url.astype(str).str.strip().str.rstrip("/").str.split("/").str[-1]


def _canonical_url(url: pd.Series, kind: str) -> pd.Series:
    """http://www.ufcstats.com/<kind>-details/<id> - the form the coin flip has always hashed."""
    return f"http://www.ufcstats.com/{kind}-details/" + _ufcstats_hash(url)


ID_HEX_DIGITS = 13          # 16**13 = 4.5e15 < 2**53: exact even as a float64


def _numeric_id(url: pd.Series) -> pd.Series:
    """The first 13 hex digits of the ufcstats id as a number - exact even as a float64."""
    return _ufcstats_hash(url).str[:ID_HEX_DIGITS].map(lambda h: int(h, 16)).astype("int64")


def _seconds(clock: pd.Series) -> pd.Series:
    parts = clock.astype(str).str.split(":", expand=True)
    return pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")


def _age_in_years(dob: pd.Series, on: pd.Series) -> pd.Series:
    """Whole years old on the fight date (birthday not yet reached -> one year younger)."""
    dob = pd.to_datetime(dob, errors="coerce")
    years = on.dt.year - dob.dt.year
    before_birthday = (on.dt.month < dob.dt.month) | ((on.dt.month == dob.dt.month) & (on.dt.day < dob.dt.day))
    return (years - before_birthday.astype(int)).where(dob.notna()).astype("float64")


def _weight_class_label(weight_class: pd.Series, title: pd.Series) -> pd.Series:
    """The old source's wording, which features/weight_class.py and the title flag parse."""
    title = title.astype(bool)
    return np.where(title, "UFC " + weight_class + " Title Bout", weight_class + " Bout")


def to_pipeline_tables(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Source table -> (fight stats, events, fighters) in the pipeline's input format."""
    check_columns(wide)
    if wide["fight_url"].duplicated().any():
        raise ValueError("the source lists a fight twice")
    date_ = pd.to_datetime(wide["event_date"])

    out = pd.DataFrame(index=wide.index)
    ids = {}
    for k in (1, 2):
        side = f"f{k}"
        ids[k] = _numeric_id(wide[f"f_{k}_url"])
        out[f"{side}_id"] = ids[k]
        out[f"{side}_name"] = wide[f"f_{k}_name"].astype(str).str.strip()
        out[f"{side}_age_during"] = _age_in_years(wide[f"f_{k}_fighter_dob"], date_)
        out[f"{side}_height_cm"] = wide[f"f_{k}_fighter_height_cm"].astype("float64")
        for ours, theirs in TOTALS.items():
            out[f"{side}_{ours}"] = wide[f"f_{k}_{theirs}"]
        # "not recorded" when no round has a control time (the 1994-99 fights)
        recorded = wide[[f"f_{k}_r{r}_ctrl" for r in ROUNDS]].notna().any(axis=1)
        out[f"{side}_ctrl_time"] = wide[f"f_{k}_ctrl_time_sec"].astype("float64").where(recorded)
        for ours, theirs in ROUND_SUMS.items():
            out[f"{side}_{ours}"] = wide[[f"f_{k}_r{r}_{theirs}" for r in ROUNDS]].sum(axis=1, min_count=1)

    fight_time = (wide["finish_round"] - 1) * 300 + _seconds(wide["finish_time"])
    out["f1_total_fight_time"] = out["f2_total_fight_time"] = fight_time
    out["weight_class"] = _weight_class_label(wide["weight_class"], wide["title_fight"])

    winner = wide["winner"].astype(str).str.strip().where(wide["winner"].notna())
    first, second = winner == out["f1_name"], winner == out["f2_name"]
    if (winner.notna() & ~first & ~second).any():
        raise ValueError("a winner's name matches neither fighter")
    # no winner + a decision, or + time running out (the early superfights,
    # e.g. Shamrock vs Gracie II), is a draw; any other no-winner result
    # (overturned, failed drug test, accidental foul) is a no-contest
    draw = (wide["result"].astype(str).str.startswith("Decision")
            | wide["result_details"].astype(str).str.contains("Time Expired", case=False))
    out["result"] = np.select(
        [first, second, winner.isna() & draw],
        [ids[1].astype(str), ids[2].astype(str), "d"],
        default="nc",
    )
    out["method"] = wide["result"].map(METHODS).fillna("other")
    out["fights_url"] = _canonical_url(wide["fight_url"], "fight")
    out["event_url"] = _canonical_url(wide["event_url"], "event")
    fight_stats = out[FIGHT_STATS_COLUMNS]

    events = (
        pd.DataFrame({"event_id": _numeric_id(wide["event_url"]), "event_name": wide["event_name"],
                      "event_date": date_.dt.strftime("%Y-%m-%d"), "url_link": out["event_url"]})
        .drop_duplicates("url_link").sort_values("event_date", ascending=False).reset_index(drop=True)
    )

    # one row per fighter; bios are profile values, so take each fighter's latest fight
    per_side = []
    for k in (1, 2):
        side = wide[[f"f_{k}_url", f"f_{k}_name"] + [f"f_{k}_{c}" for c in BIO if c != "name"]].copy()
        side.columns = ["fighter_url", "fighter_name"] + [c for c in BIO if c != "name"]
        side["event_date"] = date_
        per_side.append(side)
    bios = pd.concat(per_side, ignore_index=True).sort_values("event_date")
    fighters = bios.groupby("fighter_url", sort=False).last().reset_index()
    fighters.insert(0, "fighter_id", _numeric_id(fighters["fighter_url"]))
    fighters["fighter_url"] = _canonical_url(fighters["fighter_url"], "fighter")
    fighters = fighters.drop(columns="event_date").sort_values("fighter_name").reset_index(drop=True)
    if fighters["fighter_id"].duplicated().any():
        raise ValueError("two fighters share a numeric id - lengthen the id in _numeric_id")
    return fight_stats, events, fighters


def history_changes(new: pd.DataFrame, old: pd.DataFrame, ctrl_tolerance: float = 10.0) -> dict:
    """What a refresh would change: fights added, removed, and changed.

    Compares every column on fights present in both tables, allowing
    control time to move by `ctrl_tolerance` seconds (ufcstats revises it
    slightly now and then). Names are left out: ufcstats corrects
    spellings, which changes nothing the pipeline uses.
    """
    key = "fights_url"
    new_i, old_i = new.set_index(key), old.set_index(key)
    shared = new_i.index.intersection(old_i.index)
    # a column the old tables don't have yet is new information, not a change to history
    compare = [c for c in FIGHT_STATS_COLUMNS
               if c not in (key, "f1_name", "f2_name") and c in new_i.columns and c in old_i.columns]
    changed = set()
    for column in compare:
        a, b = new_i.loc[shared, column], old_i.loc[shared, column]
        if column.endswith("ctrl_time"):
            differs = ~(np.isclose(a.astype(float), b.astype(float), atol=ctrl_tolerance) | (a.isna() & b.isna()))
        else:
            differs = ~((a.astype(str) == b.astype(str)) | (a.isna() & b.isna()))
        changed |= set(differs[differs].index)
    return {
        "added": sorted(set(new_i.index) - set(old_i.index)),
        "removed": sorted(set(old_i.index) - set(new_i.index)),
        "changed": sorted(changed),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refresh(version: int | None = None, raw_dir: Path = RAW_DIR, allow_history_changes: bool = False) -> dict:
    """Download the source (newest, or `version`), convert it, and write data/raw.

    Writes the untouched download to data/raw/source/ together with
    SOURCE.json (dataset, version, download date, last fight, checksum), and
    the three converted tables next to it. Committing data/raw pins exactly
    which version every result was built from.

    If the current tables came from the same source, past fights must not
    change: a refresh that would rewrite history stops with a report, unless
    `allow_history_changes=True` after looking at it.
    """
    import kagglehub

    handle = KAGGLE_DATASET if version is None else f"{KAGGLE_DATASET}/versions/{version}"
    folder = Path(kagglehub.dataset_download(handle))
    got_version = int(folder.name) if folder.name.isdigit() else version
    matches = glob.glob(str(folder / SOURCE_FILE_PATTERN))
    if len(matches) != 1:
        raise ValueError(f"expected one {SOURCE_FILE_PATTERN} in the download, found {matches}")
    wide = pd.read_csv(matches[0], low_memory=False)
    fight_stats, events, fighters = to_pipeline_tables(wide)

    raw_dir = Path(raw_dir)
    source_dir = raw_dir / "source"
    manifest_path = source_dir / "SOURCE.json"
    report = {"added": [], "removed": [], "changed": []}
    same_source = manifest_path.exists() and json.loads(manifest_path.read_text()).get("dataset") == KAGGLE_DATASET
    if same_source:
        report = history_changes(fight_stats, pd.read_csv(raw_dir / "ufc_event_fight_stats.csv"))
        if (report["removed"] or report["changed"]) and not allow_history_changes:
            raise ValueError(
                f"this refresh would change {len(report['changed'])} past fight(s) and remove "
                f"{len(report['removed'])}, e.g. {(report['changed'] + report['removed'])[:3]}. "
                "Look at them, then pass allow_history_changes=True if they are genuine corrections."
            )

    source_dir.mkdir(parents=True, exist_ok=True)
    for old in source_dir.glob(SOURCE_FILE_PATTERN):
        old.unlink()
    copied = Path(shutil.copy2(matches[0], source_dir / Path(matches[0]).name))
    fight_stats.to_csv(raw_dir / "ufc_event_fight_stats.csv", index=False)
    events.to_csv(raw_dir / "ufc_events.csv", index=False)
    fighters.to_csv(raw_dir / "ufc_fighters.csv", index=False)

    summary = {
        "dataset": KAGGLE_DATASET,
        "version": got_version,
        "downloaded_on": date.today().isoformat(),
        "file": copied.name,
        "sha256": _sha256(copied),
        "fights": int(len(fight_stats)),
        "events": int(len(events)),
        "fighters": int(len(fighters)),
        "first_event": str(events["event_date"].min()),
        "last_event": str(events["event_date"].max()),
    }
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary | {"fights_added": len(report["added"]), "history_checked": same_source}
