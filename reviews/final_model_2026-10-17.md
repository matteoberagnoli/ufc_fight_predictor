# The final model — frozen recipe (2026-10-17)

Everything below was decided on fights before 2025, by cross-validation inside the training
years (2019-2022 scored) with the validation years (2023-24) as a secondary check. From this
commit on, nothing about the recipe changes because of a score. The test years (2025 on) are
scored once, in `10_test_evaluation.ipynb`.

## Data
- Kaggle `scarekrow/ufc-data`, **version 20**: 8,937 fights, 1994-03-11 to 2026-10-10
  (`data/raw/source/SOURCE.json`).
- Modelling rows: decided fights from 2014 on. Every fight since 1994, draws and no-contests
  included, counts as fighter history.

## Inputs
- **40 inputs**, `manifest.FEATURES`, identical to `reviews/final_inputs_2026-10-08.csv`.
- Every rolling statistic is one window over the **last 8 fights** (`rolling.ROLLING_WINDOW`).
- Ratios shrunk towards the league average; alphas fitted on the training period only
  (`rolling.fit_alphas(fights, until=<training cut>)`).

## Elo
- **K = 158.5**, start 1500, scale 400.
- A win counts by how it came: **finish 1.0, unanimous decision 0.7, split or majority
  decision 0.55**; the loser gets 1 minus that. Draw 0.5. No-contests move nothing.
- The fight's `method` is used only for these updates, never as an input.
- Chosen in `05_elo.ipynb`: 180 settings searched with the model's own settings; adopted over the
  classic Elo (K 251.2) because it was better in 8 of 8 seeds and on validation.

## Model
- scikit-learn `HistGradientBoostingClassifier`, settings (`model_params.json`): learning rate
  0.0113, 950 trees, 5 leaves, at least 28 fights per leaf, L2 0.0034, 55% of the inputs per
  split, no early stopping.
- **8 seeds**, each fight predicted **both ways round** and averaged.
- **Calibration**: one stretch `a`, fitted on out-of-fold predictions of the last 4 years of the
  training period (on 2014-2022 data: a = 1.077, 90% range 0.95-1.25).

## Development scores (2014-2022 training)
| | Cross-validation 2019-22 | Men's validation 2023-24 (837) |
|---|---|---|
| log loss | 0.65477 (8 seeds) | 0.6389 (calibrated) |
| accuracy | | 64.6% |
| AUC | | 0.689 |

No debutant / with a debutant (men's validation, calibrated): log loss 0.6372 / 0.6455.

## How the test years are scored (`10_test_evaluation.ipynb`, once)
1. Refit on 2014-2024 with this recipe (alphas and calibration refitted on those years).
2. Score the 2025-on fights a single time. Men's fights first, then all; each as all fights /
   no debutant / with a debutant (`evaluate.breakdown`); baselines: coin flip, win rate only,
   age only.
3. Save every test prediction (`reviews/test_predictions_2026-10-17.csv`) and the scores
   (`reviews/test_result_2026-10-17.json`). The betting market is scored on these same
   predictions later, once its test-year odds can be matched.

## After the test look (`11_final_model.ipynb`)
Refit on every decided fight to 2026-10-10; calibration stretch reused from the test look.
`modeling.final.predict_upcoming` predicts fights not yet fought.
