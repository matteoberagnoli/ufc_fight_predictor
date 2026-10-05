"""Reading raw source tables and writing processed pipeline outputs."""

import json

import pandas as pd

from ufc_predictor.config import PROCESSED_DIR, RAW_DIR


def load_events(raw_dir=RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "ufc_events.csv")
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def load_fight_stats(raw_dir=RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "ufc_event_fight_stats.csv")


def load_fighters(raw_dir=RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "ufc_fighters.csv")


def _processed(processed_dir):
    """The folder to use: the one passed in, or PROCESSED_DIR looked up *now*.

    A default written as `processed_dir=PROCESSED_DIR` is fixed once, when the
    function is defined - so a test that points PROCESSED_DIR at a temporary
    folder would still write to the real one. That happened: a test overwrote
    data/processed/model_params.json. Looking it up at call time means
    redirecting PROCESSED_DIR really redirects every save and load.
    """
    return PROCESSED_DIR if processed_dir is None else processed_dir


def save_processed(df: pd.DataFrame, name: str, processed_dir=None) -> None:
    processed_dir = _processed(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_dir / f"{name}.parquet", index=False)


def load_processed(name: str, processed_dir=None) -> pd.DataFrame:
    return pd.read_parquet(_processed(processed_dir) / f"{name}.parquet")


def save_processed_json(obj: dict, name: str, processed_dir=None) -> None:
    # sorted keys and a fixed layout: the same content always gives the same file
    processed_dir = _processed(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def load_processed_json(name: str, processed_dir=None) -> dict:
    with open(_processed(processed_dir) / f"{name}.json") as f:
        return json.load(f)
