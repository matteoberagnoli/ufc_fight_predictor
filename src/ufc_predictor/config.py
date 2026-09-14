from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Seed for the deterministic fighter-side randomization (see data.randomize).
# Fixed so re-running the pipeline never reshuffles an already-processed fight.
RANDOMIZE_SEED = 1926
