from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Seed for the deterministic fighter-side randomization (see data.randomize).
# Fixed so re-running the pipeline never reshuffles an already-processed fight.
RANDOMIZE_SEED = 1926

# First date of the modelling era. MMA before ~2014 was a different sport
# (rules, judging, depth of talent, anti-doping), so older fights are not used
# as training rows - but they still count as fighter history (see modeling.split).
ERA_START = "2014-01-01"

# Chronological split of the modelling era. Never random: a random split would
# let the model learn from 2023 fights to predict 2016 ones.
#   train       ERA_START -> VALIDATION_START   fit the model (and shrinkage alphas)
#   validation  VALIDATION_START -> TEST_START  make every design decision here
#   test        TEST_START -> end of data       look once, at the very end
VALIDATION_START = "2020-01-01"
TEST_START = "2022-01-01"

# Shrinkage (see features/shrinkage.py): each estimate is pulled towards the
# league-wide value over this many years before the fight. Tested: 1, 3 or 5
# years change the results by less than 0.1 point, so it's not a sensitive choice.
LEAGUE_RATE_YEARS = 3
