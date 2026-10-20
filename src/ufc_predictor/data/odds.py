"""Betting odds: the market's pre-fight probability for each fight, as a benchmark.

The betting market is the hardest benchmark there is: thousands of people
with money on the line, pricing in everything public. This module turns
bookmakers' odds into one market probability per fight, matched exactly to
our fights. It is a yardstick only, never a model feature.

Source: the Kaggle dataset `jerzyszocik/ufc-betting-odds-daily-dataset`,
refreshed daily. Only the rows that carry the ufcstats fight URL and both
fighters' ufcstats URLs are used (36% of the file), so odds join our fights
by id - no fuzzy name matching. Those rows are of two kinds:

  * archive rows (source "zewnetrzne", Polish for "external"): one line per
    fight from 2010-03 to 2024-12, imported in one batch. Cross-checked:
    96% are identical to the odds in Kaggle's mdabbert/ultimate-ufc-dataset,
    which scrapes bestfightodds.com. Its documentation doesn't say when the
    line is taken, but on the fights both datasets cover, it matches the
    bookmakers' consensus from the final day before the fight best (2.2
    points apart on average, against 5.5 points two weeks out): near-closing
    lines, the standard benchmark. Their `adding_date` is the import date,
    not when the odds were quoted.
  * snapshot rows: many bookmakers' prices scraped again and again as fight
    night approaches. With URLs they exist only for the August and
    September 2025 events (51 fights). Only quotes taken before the event's
    date (UTC) count - a quote from fight day could already know a late
    withdrawal or the result - and each bookmaker's last such quote is used.
    When a fight has snapshots, their consensus is used rather than the
    archive line.

The other 64% of the file - snapshots for events from 2025-09-13 on - has no
URLs and is dropped here. (A single Polish bookmaker's 150 lines for
2025-03 to 2025-07 were imported after those fights and are dropped too:
their timing is unknown.) So this source covers 2010-2024 fully and almost
nothing after: of the 950 test-year fights (data v20), 51 get odds. For the test years,
mdabbert/ultimate-ufc-dataset has odds for 98% of its fights to 2026-03-28
(about two thirds of the test fights); using it means matching on date and
fighter names, which is not built yet.

From odds to a probability, for one bookmaker's quote:

  1. decimal odds o1, o2 -> raw probabilities 1/o1 and 1/o2
  2. those add up to a bit more than 1 (about 1.036 here): that excess is
     the bookmaker's margin, the "overround"
  3. divide each by their sum, so they add up to exactly 1

Several bookmakers' probabilities for the same fight are then averaged.
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_predictor.config import RAW_DIR
from ufc_predictor.data.source import _canonical_url, _numeric_id

ODDS_DATASET = "jerzyszocik/ufc-betting-odds-daily-dataset"
ODDS_FILE = "UFC_betting_odds.csv"
ARCHIVE_SOURCE = "zewnetrzne"
ODDS_COLUMNS = ["fight_url", "fighter_1_url", "fighter_2_url", "odds_1", "odds_2",
                "event_date", "adding_date", "source"]
MARKET_COLUMNS = ["fights_url", "event_date", "fighter_1_id", "fighter_2_id",
                  "p_fighter_1", "bookmakers", "overround", "kind"]


def _usable_quotes(odds: pd.DataFrame) -> pd.DataFrame:
    """Quotes with both fighters known and real decimal odds, each turned into a vig-free probability."""
    missing = sorted(set(ODDS_COLUMNS) - set(odds.columns))
    if missing:
        raise ValueError(f"the odds file's format has changed: missing {missing}")
    q = odds[ODDS_COLUMNS].copy()
    q = q[q["fighter_1_url"].notna() & q["fighter_2_url"].notna()
          & q["odds_1"].gt(1) & q["odds_2"].gt(1)].copy()
    q["fights_url"] = _canonical_url(q["fight_url"], "fight")
    id_1, id_2 = _numeric_id(q["fighter_1_url"]), _numeric_id(q["fighter_2_url"])
    raw_1, raw_2 = 1 / q["odds_1"], 1 / q["odds_2"]
    q["overround"] = raw_1 + raw_2
    p_1 = raw_1 / q["overround"]
    # one reference fighter per fight (the smaller id), so quotes that list the
    # two fighters in different orders can be averaged
    q["fighter_1_id"], q["fighter_2_id"] = np.minimum(id_1, id_2), np.maximum(id_1, id_2)
    q["p_fighter_1"] = np.where(id_1 <= id_2, p_1, 1 - p_1)
    q["event_date"] = pd.to_datetime(q["event_date"])
    q["adding_date"] = pd.to_datetime(q["adding_date"], utc=True, format="mixed")
    return q


def market_probabilities(odds: pd.DataFrame) -> pd.DataFrame:
    """One row per fight: the market's vig-free probability that fighter_1 wins.

    fighter_1 is the fighter with the smaller id. `bookmakers` is how many
    quotes were averaged, `kind` whether they were snapshots or the archive line.
    """
    q = _usable_quotes(odds)
    archive = q[q["source"] == ARCHIVE_SOURCE]
    snapshots = q[q["source"] != ARCHIVE_SOURCE]
    fight_day = snapshots["event_date"].dt.tz_localize("UTC")
    snapshots = snapshots[snapshots["adding_date"] < fight_day]
    # each bookmaker's last quote before fight day - the whole row, so odds and
    # time always come from the same quote
    last = snapshots.sort_values("adding_date", kind="stable").groupby(["fights_url", "source"]).tail(1)

    keys = ["fights_url", "event_date", "fighter_1_id", "fighter_2_id"]
    from_snapshots = last.groupby(keys, as_index=False).agg(
        p_fighter_1=("p_fighter_1", "mean"), bookmakers=("source", "size"), overround=("overround", "mean"))
    from_snapshots["kind"] = "snapshot"
    from_archive = archive.groupby(keys, as_index=False).agg(
        p_fighter_1=("p_fighter_1", "mean"), bookmakers=("source", "size"), overround=("overround", "mean"))
    from_archive["kind"] = "archive"
    from_archive = from_archive[~from_archive["fights_url"].isin(from_snapshots["fights_url"])]

    market = pd.concat([from_snapshots, from_archive], ignore_index=True)
    if market["fights_url"].duplicated().any():
        raise ValueError("a fight got two different pairs of fighters across quotes")
    return market[MARKET_COLUMNS].sort_values(["event_date", "fights_url"]).reset_index(drop=True)


def market_for(fights: pd.DataFrame, market: pd.DataFrame) -> pd.Series:
    """The market's probability that fighter A wins, for each row of `fights`.

    Matched on the fight URL, then turned round to the coin flip's corners: if
    our fighter A is the market's fighter_2, A's probability is 1 - p. NaN
    where there are no odds, or where the odds name different fighters.
    """
    m = fights[["fights_url", "fighter_a_id", "fighter_b_id"]].merge(
        market[["fights_url", "fighter_1_id", "fighter_2_id", "p_fighter_1"]],
        on="fights_url", how="left", validate="many_to_one")
    as_listed = m["fighter_a_id"].eq(m["fighter_1_id"]) & m["fighter_b_id"].eq(m["fighter_2_id"])
    turned = m["fighter_a_id"].eq(m["fighter_2_id"]) & m["fighter_b_id"].eq(m["fighter_1_id"])
    p = np.where(as_listed, m["p_fighter_1"], np.where(turned, 1 - m["p_fighter_1"], np.nan))
    return pd.Series(p, index=fights.index, name="p_market")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refresh_odds(version: int | None = None, raw_dir: Path = RAW_DIR) -> dict:
    """Download the odds (newest, or `version`) and write data/raw/odds/.

    Writes market.csv (one row per fight, small enough to commit) and
    ODDS.json (dataset, version, download date, checksum of the raw file).
    The raw download itself (30+ MB, growing daily) stays in kagglehub's
    cache; the version number is enough to fetch it again exactly.
    """
    import kagglehub

    handle = ODDS_DATASET if version is None else f"{ODDS_DATASET}/versions/{version}"
    folder = Path(kagglehub.dataset_download(handle))
    got_version = int(folder.name) if folder.name.isdigit() else version
    path = folder / ODDS_FILE
    market = market_probabilities(pd.read_csv(path, low_memory=False))

    out = Path(raw_dir) / "odds"
    out.mkdir(parents=True, exist_ok=True)
    market.to_csv(out / "market.csv", index=False)
    summary = {
        "dataset": ODDS_DATASET,
        "version": got_version,
        "downloaded_on": date.today().isoformat(),
        "raw_file": ODDS_FILE,
        "raw_sha256": _sha256(path),
        "fights": int(len(market)),
        "first_event": str(market["event_date"].min().date()),
        "last_event": str(market["event_date"].max().date()),
        "by_kind": {k: int(n) for k, n in market["kind"].value_counts().items()},
    }
    (out / "ODDS.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def load_market(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    market = pd.read_csv(Path(raw_dir) / "odds" / "market.csv")
    market["event_date"] = pd.to_datetime(market["event_date"])
    return market
