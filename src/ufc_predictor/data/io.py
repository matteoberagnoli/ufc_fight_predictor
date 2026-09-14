"""Reading raw source tables and writing processed pipeline outputs."""

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


def save_processed(df: pd.DataFrame, name: str, processed_dir=PROCESSED_DIR) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_dir / f"{name}.parquet", index=False)


def load_processed(name: str, processed_dir=PROCESSED_DIR) -> pd.DataFrame:
    return pd.read_parquet(processed_dir / f"{name}.parquet")
