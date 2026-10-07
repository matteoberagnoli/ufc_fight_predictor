import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.data.odds import load_market, market_for, market_probabilities, refresh_odds
from ufc_predictor.data.source import _numeric_id

FIGHTER = "http://ufcstats.com/fighter-details/{}"
HASH = {"ann": "1aaaaaaaaaaaaaaa", "bea": "2bbbbbbbbbbbbbbb", "cat": "3ccccccccccccccc"}
ID = {name: int(h[:13], 16) for name, h in HASH.items()}


def _quote(fight, f1, f2, o1, o2, source="zewnetrzne", event="2024-03-02", added="2025-07-27 09:00"):
    return {"fight_url": f"http://ufcstats.com/fight-details/{fight}", "fighter_1_url": FIGHTER.format(HASH[f1]),
            "fighter_2_url": FIGHTER.format(HASH[f2]), "fighter_1": f1, "fighter_2": f2, "odds_1": o1, "odds_2": o2,
            "event_date": event, "adding_date": added, "source": source, "region": "us"}


def _p(o1, o2):
    return (1 / o1) / (1 / o1 + 1 / o2)


def test_the_bookmakers_margin_is_removed():
    market = market_probabilities(pd.DataFrame([_quote("f1", "ann", "bea", 1.5, 2.5)]))
    row = market.iloc[0]
    assert row["p_fighter_1"] == pytest.approx(0.625)             # 0.667 / (0.667 + 0.4)
    assert row["overround"] == pytest.approx(1 / 1.5 + 1 / 2.5)
    assert row["kind"] == "archive" and row["bookmakers"] == 1


def test_fighter_1_is_the_smaller_id_whatever_order_the_quote_used():
    listed = market_probabilities(pd.DataFrame([_quote("f1", "ann", "bea", 1.5, 2.5)])).iloc[0]
    turned = market_probabilities(pd.DataFrame([_quote("f1", "bea", "ann", 2.5, 1.5)])).iloc[0]
    assert listed["fighter_1_id"] == turned["fighter_1_id"] == ID["ann"]
    assert listed["p_fighter_1"] == pytest.approx(turned["p_fighter_1"])


def test_urls_match_the_pipeline_form():
    market = market_probabilities(pd.DataFrame([_quote("abc123", "ann", "bea", 1.5, 2.5)]))
    assert market.loc[0, "fights_url"] == "http://www.ufcstats.com/fight-details/abc123"


def test_snapshots_use_each_bookmakers_last_quote_before_fight_day():
    quotes = pd.DataFrame([
        _quote("f1", "ann", "bea", 1.8, 2.0, source="BookA", added="2024-02-20 10:00"),   # superseded
        _quote("f1", "ann", "bea", 1.5, 2.5, source="BookA", added="2024-03-01 10:00"),   # BookA's last
        _quote("f1", "bea", "ann", 3.0, 1.4, source="BookB", added="2024-02-28 10:00"),   # listed the other way
        _quote("f1", "ann", "bea", 1.1, 7.0, source="BookB", added="2024-03-02 21:00"),   # on fight day: ignored
        _quote("f1", "ann", "bea", 9.0, 1.05, source="zewnetrzne"),                       # archive: not needed
    ])
    row = market_probabilities(quotes).iloc[0]
    assert row["kind"] == "snapshot" and row["bookmakers"] == 2
    assert row["p_fighter_1"] == pytest.approx((_p(1.5, 2.5) + _p(1.4, 3.0)) / 2)


def test_the_archive_line_is_used_when_there_are_no_usable_snapshots():
    quotes = pd.DataFrame([
        _quote("f1", "ann", "bea", 1.5, 2.5),
        _quote("f1", "ann", "bea", 1.1, 7.0, source="BookB", added="2024-03-03 09:00"),   # after the fight
    ])
    row = market_probabilities(quotes).iloc[0]
    assert row["kind"] == "archive" and row["p_fighter_1"] == pytest.approx(0.625)


def test_unusable_quotes_are_dropped():
    quotes = pd.DataFrame([
        _quote("f1", "ann", "bea", 1.5, 2.5),
        _quote("f2", "ann", "cat", np.nan, 2.5),
        _quote("f3", "bea", "cat", 1.0, 9.0),        # decimal odds of 1 pay nothing: not a real price
    ])
    quotes.loc[len(quotes)] = _quote("f4", "ann", "cat", 1.5, 2.5) | {"fighter_2_url": np.nan}
    assert market_probabilities(quotes)["fights_url"].str.endswith("f1").tolist() == [True]


def test_a_changed_format_fails_loudly():
    with pytest.raises(ValueError, match="format has changed"):
        market_probabilities(pd.DataFrame([_quote("f1", "ann", "bea", 1.5, 2.5)]).drop(columns=["odds_2"]))


def _fights(a, b):
    return pd.DataFrame({"fights_url": ["http://www.ufcstats.com/fight-details/f1"],
                         "fighter_a_id": [ID[a]], "fighter_b_id": [ID[b]]})


def test_market_for_turns_the_probability_to_our_corners():
    market = market_probabilities(pd.DataFrame([_quote("f1", "ann", "bea", 1.5, 2.5)]))
    assert market_for(_fights("ann", "bea"), market).iloc[0] == pytest.approx(0.625)
    assert market_for(_fights("bea", "ann"), market).iloc[0] == pytest.approx(0.375)


def test_market_for_is_nan_without_odds_or_with_other_fighters():
    market = market_probabilities(pd.DataFrame([_quote("f1", "ann", "bea", 1.5, 2.5)]))
    assert np.isnan(market_for(_fights("ann", "cat"), market).iloc[0])
    other_fight = _fights("ann", "bea").assign(fights_url="http://www.ufcstats.com/fight-details/zzz")
    assert np.isnan(market_for(other_fight, market).iloc[0])


def test_ids_are_the_pipelines_ids():
    assert _numeric_id(pd.Series([FIGHTER.format(HASH["ann"])])).iloc[0] == ID["ann"]


def test_refresh_writes_the_market_and_pins_the_version(tmp_path, monkeypatch):
    folder = tmp_path / "download" / "259"
    folder.mkdir(parents=True)
    pd.DataFrame([_quote("f1", "ann", "bea", 1.5, 2.5)]).to_csv(folder / "UFC_betting_odds.csv", index=False)
    monkeypatch.setitem(sys.modules, "kagglehub", types.SimpleNamespace(dataset_download=lambda handle: str(folder)))
    summary = refresh_odds(raw_dir=tmp_path / "raw")
    manifest = json.loads((tmp_path / "raw" / "odds" / "ODDS.json").read_text())
    assert manifest["version"] == summary["version"] == 259 and manifest["fights"] == 1
    assert load_market(tmp_path / "raw")["p_fighter_1"].iloc[0] == pytest.approx(0.625)
