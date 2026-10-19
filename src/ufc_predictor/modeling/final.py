"""The finished model: fit on every decided fight up to a date, and predict fights not yet fought.

Two uses, one code path:

  * the one-time test look (10_test_evaluation.ipynb): `fit_final(..., until=TEST_START)`
    learns everything from 2014-2024 - shrinkage alphas, calibration, the 8 models -
    and is scored once on 2025 onwards;
  * predicting upcoming fights: `fit_final` on everything to the last event, then
    `predict_upcoming`.

The recipe itself (inputs, window, Elo settings, model settings) is frozen elsewhere:
manifest.FEATURES, rolling.ROLLING_WINDOW, elo_params.json and model_params.json.
This module only decides *which fights* it learns from.

An upcoming fight is added to the raw tables as a fight with no statistics and no
result (a no-contest), dated on fight night. Every feature is built only from earlier
fights, so the row gets each fighter's history exactly as it stands, and a no-contest
moves no rating and no average. The same feature code then serves past and future
fights alike: there is no second implementation to drift.

A fake fight is still a row in the fighter's history, though: a later fake fight of the
same fighter would count it as their last fight (layoff 0 days) and as one of the 8 fights
in their window. So bouts that share a fighter are built in separate passes, each pass
holding every fighter at most once (review 2026-10-18, finding F3).
"""
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ufc_predictor.config import ERA_START, PROCESSED_DIR, RAW_DIR
from ufc_predictor.data.build import build_fight_dataset
from ufc_predictor.data.io import load_fight_stats, load_fighters
from ufc_predictor.data.source import FIGHT_STATS_COLUMNS, _age_in_years
from ufc_predictor.features import manifest
from ufc_predictor.features.pipeline import build_featured_table
from ufc_predictor.features.rolling import fit_alphas
from ufc_predictor.modeling.calibrate import fit_stretch, stretch
from ufc_predictor.modeling.model import fit_models, predict_proba_mean
from ufc_predictor.modeling.split import drop_undecided, filter_era
from ufc_predictor.modeling.tune import out_of_fold_predictions

UPCOMING = "upcoming/"


def training_rows(featured: pd.DataFrame, until) -> pd.DataFrame:
    """Decided fights from ERA_START up to (not including) `until`."""
    rows = drop_undecided(filter_era(featured, start=ERA_START))
    return rows[rows["event_date"] < pd.Timestamp(until)].reset_index(drop=True)


def fit_final(fights: pd.DataFrame, fighters: pd.DataFrame, *, until, params: dict,
              elo_k: float, elo_weights: dict | None, seeds=range(8), calibrate: bool = True) -> dict:
    """Everything learned from fights before `until`: alphas, table, calibration, models.

    The stretch is fitted on out-of-fold predictions of the last four years before
    `until` (the guard in tune refuses test years, so for a model trained on
    everything pass `calibrate=False` and reuse the stretch from the test look).
    """
    alphas = fit_alphas(fights, until=str(pd.Timestamp(until).date()))
    featured = build_featured_table(fights, fighters, alphas, elo_k=elo_k, elo_weights=elo_weights)
    train = training_rows(featured, until)
    a = 1.0
    if calibrate:
        y_oof, p_oof, _ = out_of_fold_predictions(train, params, seeds=seeds)
        a = fit_stretch(y_oof, p_oof)
    X, y = manifest.make_xy(train)
    return {
        "models": fit_models(X, y, params, seeds=seeds), "stretch": float(a), "alphas": alphas,
        "elo_k": float(elo_k), "elo_weights": elo_weights, "params": dict(params),
        "until": str(pd.Timestamp(until).date()), "training_fights": int(len(train)),
        "features": list(X.columns), "featured": featured,
    }


def predict(bundle: dict, rows: pd.DataFrame) -> np.ndarray:
    """P(fighter A wins) for featured rows: 8 models, both corners, then the stretch."""
    return stretch(predict_proba_mean(bundle["models"], rows[bundle["features"]]), bundle["stretch"])


def save_bundle(bundle: dict, path=None) -> Path:
    path = Path(path) if path is not None else PROCESSED_DIR / "final_model.joblib"
    joblib.dump({k: v for k, v in bundle.items() if k != "featured"}, path)
    return path


def load_bundle(path=None) -> dict:
    return joblib.load(Path(path) if path is not None else PROCESSED_DIR / "final_model.joblib")


def _fighter_id(fighters: pd.DataFrame, who) -> int:
    if isinstance(who, (int, np.integer)):
        if who not in set(fighters["fighter_id"]):
            raise ValueError(f"no fighter with id {who}")
        return int(who)
    hits = fighters[fighters["fighter_name"].str.casefold() == str(who).strip().casefold()]
    if len(hits) != 1:
        found = "no fighter" if hits.empty else f"{len(hits)} fighters ({hits['fighter_id'].tolist()}) - pass the id"
        raise ValueError(f"{who!r}: {found}. Only fighters with a UFC record can be predicted.")
    return int(hits["fighter_id"].iloc[0])


def _batches(ids: list[tuple[int, int]]) -> list[list[int]]:
    """Group bouts so that no fighter appears twice in a group.

    `ids` holds each bout's two fighter ids; the result holds positions in `ids`.
    Each bout goes into the first group that has neither of its fighters yet, so a
    normal card (everyone fights once) stays one group.
    """
    batches, taken = [], []
    for i, pair in enumerate(ids):
        for batch, fighters_in in zip(batches, taken):
            if not fighters_in & set(pair):
                batch.append(i)
                fighters_in.update(pair)
                break
        else:
            batches.append([i])
            taken.append(set(pair))
    return batches


def upcoming_rows(bouts: list[dict], fighters: pd.DataFrame,
                  positions: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw fight-stats and event rows for fights not yet fought.

    Each bout: {"fighter_a": name or id, "fighter_b": ..., "date": "YYYY-MM-DD",
    "weight_class": e.g. "Lightweight Bout" or "UFC Women's Flyweight Title Bout"}.
    No statistics and result "nc": history for nobody, features for itself.
    `positions` numbers the bouts in the url (default 0, 1, 2...), which keeps the
    caller's order when bouts are built in separate passes.
    """
    positions = range(len(bouts)) if positions is None else positions
    stats, events = [], []
    for i, bout in zip(positions, bouts):
        date = pd.Timestamp(bout["date"])
        row = dict.fromkeys(FIGHT_STATS_COLUMNS, np.nan)
        if _fighter_id(fighters, bout["fighter_a"]) == _fighter_id(fighters, bout["fighter_b"]):
            raise ValueError(f"{bout['fighter_a']!r}: the same fighter is on both sides")
        for side, key in (("f1", "fighter_a"), ("f2", "fighter_b")):
            fid = _fighter_id(fighters, bout[key])
            bio = fighters.loc[fighters["fighter_id"] == fid].iloc[0]
            row[f"{side}_id"], row[f"{side}_name"] = fid, bio["fighter_name"]
            row[f"{side}_age_during"] = float(_age_in_years(pd.Series([bio["fighter_dob"]]), pd.Series([date])).iloc[0])
            row[f"{side}_height_cm"] = bio["fighter_height_cm"]
        event_url = f"{UPCOMING}event/{date.date()}"
        row.update({"weight_class": bout["weight_class"], "result": "nc", "method": "other",
                    "fights_url": f"{UPCOMING}{i:03d}/{row['f1_id']}-{row['f2_id']}", "event_url": event_url})
        stats.append(row)
        events.append({"event_id": 0, "event_name": "Upcoming", "event_date": str(date.date()), "url_link": event_url})
    return pd.DataFrame(stats)[FIGHT_STATS_COLUMNS], pd.DataFrame(events).drop_duplicates("url_link")


def upcoming_featured(bouts: list[dict], alphas: dict, elo_k: float, elo_weights: dict | None,
                      raw_dir=None) -> tuple[pd.DataFrame, pd.Series]:
    """The featured rows of fights not yet fought, in the order of `bouts`, and each bout's first-listed id."""
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DIR
    fighters = load_fighters(raw_dir)
    new_stats, _ = upcoming_rows(bouts, fighters)       # checks every bout before the slow part
    old_stats, old_events = load_fight_stats(raw_dir), pd.read_csv(raw_dir / "ufc_events.csv")
    parts = []
    # one pass per group of bouts that share no fighter: a fake fight must not enter
    # another fake fight's history (see the module docstring)
    for batch in _batches(list(zip(new_stats["f1_id"], new_stats["f2_id"]))):
        stats, events = upcoming_rows([bouts[i] for i in batch], fighters, positions=batch)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pd.concat([old_stats, stats], ignore_index=True).to_csv(tmp / "ufc_event_fight_stats.csv", index=False)
            # read as plain text: load_events parses dates, and a mix of parsed and new string dates won't re-parse
            pd.concat([old_events, events], ignore_index=True).to_csv(tmp / "ufc_events.csv", index=False)
            shutil.copy(raw_dir / "ufc_fighters.csv", tmp / "ufc_fighters.csv")
            fights = build_fight_dataset(raw_dir=tmp)
        featured = build_featured_table(fights, fighters, alphas, elo_k=elo_k, elo_weights=elo_weights)
        parts.append(featured[featured["fights_url"].str.startswith(UPCOMING)])
    rows = pd.concat(parts).sort_values("fights_url")
    return rows, new_stats["f1_id"]


def predict_upcoming(bouts: list[dict], bundle: dict, raw_dir=None) -> pd.DataFrame:
    """Win probabilities for fights not yet fought, for the fighter listed first in each bout."""
    rows, first_ids = upcoming_featured(bouts, bundle["alphas"], bundle["elo_k"], bundle["elo_weights"], raw_dir)
    p_a = predict(bundle, rows)                       # rows are in the order of `bouts` (url carries the index)
    out = []
    for (_, row), p, first_id in zip(rows.iterrows(), p_a, first_ids):
        # the coin flip may have put the first-listed fighter in corner B
        first_is_a = row["fighter_a_id"] == first_id
        side, other = ("fighter_a", "fighter_b") if first_is_a else ("fighter_b", "fighter_a")
        out.append({"date": row["event_date"].date(), "fighter": row[f"{side}_name"], "opponent": row[f"{other}_name"],
                    "p_win": float(p if first_is_a else 1 - p),
                    "fighter_debut": bool(row[f"{side}_prior_is_debut"]),
                    "opponent_debut": bool(row[f"{other}_prior_is_debut"])})
    return pd.DataFrame(out)
