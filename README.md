# ufc_fight_predictor
Personal project - ML model to predict ufc fights outcome

## Read more

Four pages explain the project without reading the code:

- [Teo's UFC ML Predictor](https://claude.ai/artifact/EDZr7qCv1WQA2xRxNoTyRt): the overview, with the test results and the pound-for-pound ranking.
- [UFC Pipeline Walkthrough](https://claude.ai/artifact/3uXZoZbc7T3zVsma3K91QU): how the Kaggle download becomes the 40 model inputs, step by step.
- [The Modelling Half](https://claude.ai/artifact/J1VyvnhyvJb9qY4yWG3fVU): the model, how its score becomes a probability, how it is judged, tuned and explained.
- [Volkanovski vs Evloev](https://claude.ai/artifact/58U9xDqvk6oPbDmzkkdSjh): the model's preview of one fight (UFC 333).

## How this was built

I built this project with [Claude Code](https://claude.com/claude-code), Anthropic's AI
coding assistant, as a pair programmer: it wrote much of the code with me, under my direction.
The questions, the modelling decisions (which inputs to keep, how to split the years, which
test decides, etc.), the modelling workflow, and the checks on the results are mine. Each decision is recorded with its
evidence in `reviews/` and the notebooks.

## Data credits

This project would not exist without two public Kaggle datasets. Thank you to their authors.

| Data | Dataset | Used for |
|---|---|---|
| Fights, fighters, events (1994 to 2026) | [`scarekrow/ufc-data`](https://www.kaggle.com/datasets/scarekrow/ufc-data) by scarekrow | every input of the model |
| Betting odds (2010 to 2025) | [`jerzyszocik/ufc-betting-odds-daily-dataset`](https://www.kaggle.com/datasets/jerzyszocik/ufc-betting-odds-daily-dataset) by jerzyszocik (CC0) | the market benchmark only, never a model input |

The MIT licence in `LICENSE` covers the code in this repository. It does **not** cover the data
in `data/raw/`, which belongs to its original authors. Please keep the fight data non-commercial
if you reuse it.
