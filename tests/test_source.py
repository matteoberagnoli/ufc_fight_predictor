import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.data.build import build_fight_dataset
from ufc_predictor.data.source import (
    FIGHT_STATS_COLUMNS,
    check_columns,
    history_changes,
    refresh,
    required_columns,
    to_pipeline_tables,
)

URL = "http://ufcstats.com/{kind}-details/{h}"
HASH = {"ann": "aaaaaaaaaaaaaaa1", "bea": "bbbbbbbbbbbbbbb2", "cat": "ccccccccccccccc3", "dee": "ddddddddddddddd4"}


def _fight(i, f1, f2, winner, result, details="", title=False, weight="Lightweight",
           date="2020-06-15", finish_round=3, finish_time="5:00", rounds=3, ctrl=(30, 10)):
    """One source row, every required column filled in."""
    row = {c: np.nan for c in required_columns()}
    row.update({
        "fight_url": URL.format(kind="fight", h=f"f{i:015d}"), "event_url": URL.format(kind="event", h=f"e{i:015d}"),
        "event_name": f"UFC {i}", "event_date": date, "weight_class": weight, "title_fight": title,
        "winner": winner, "result": result, "result_details": details,
        "finish_round": finish_round, "finish_time": finish_time,
    })
    for k, name in ((1, f1), (2, f2)):
        row.update({
            f"f_{k}_url": URL.format(kind="fighter", h=HASH[name]), f"f_{k}_name": name.title(),
            f"f_{k}_fighter_dob": "1990-06-15", f"f_{k}_fighter_height_cm": 180.0, f"f_{k}_fighter_weight_lbs": 155.0,
            f"f_{k}_fighter_reach_cm": 185.0, f"f_{k}_fighter_stance": "Orthodox",
            f"f_{k}_knockdowns": 0, f"f_{k}_sig_strikes_att": 50, f"f_{k}_sig_strikes_succ": 20,
            f"f_{k}_total_strikes_att": 60, f"f_{k}_total_strikes_succ": 25, f"f_{k}_takedown_att": 2,
            f"f_{k}_takedown_succ": 1, f"f_{k}_submission_att": 0, f"f_{k}_reversals": 0,
            f"f_{k}_ctrl_time_sec": sum(ctrl) if ctrl else 0,
        })
        for r in range(1, rounds + 1):
            for c in ("head", "body", "leg", "distance", "clinch", "ground"):
                row[f"f_{k}_r{r}_{c}_att"] = 4
                row[f"f_{k}_r{r}_{c}_succ"] = r          # round r lands r strikes of each kind
            row[f"f_{k}_r{r}_ctrl"] = (ctrl[r - 1] if ctrl and r <= len(ctrl) else 0)
    return row


def _wide():
    return pd.DataFrame([
        _fight(1, "ann", "bea", "Ann", "Decision - Unanimous", title=True, weight="Women's Strawweight"),
        _fight(2, "ann", "cat", np.nan, "Decision - Split", date="2021-06-14"),
        _fight(3, "bea", "cat", np.nan, "Overturned", details="Failed Drug Test by Cat",
               date="1995-04-07", rounds=0, ctrl=None, finish_round=1, finish_time="2:30"),
        _fight(4, "cat", "dee", np.nan, "Other", details="Time Expired", date="1995-05-01"),
    ])


def test_output_has_exactly_the_columns_the_pipeline_reads():
    stats, _, _ = to_pipeline_tables(_wide())
    assert list(stats.columns) == FIGHT_STATS_COLUMNS


def test_results_winner_draw_no_contest():
    stats, _, _ = to_pipeline_tables(_wide())
    ann = int(HASH["ann"][:13], 16)
    assert stats["result"].tolist() == [str(ann), "d", "nc", "d"]   # time expired = draw


def test_strikes_by_target_and_position_are_round_sums():
    stats, _, _ = to_pipeline_tables(_wide())
    assert stats.loc[0, "f1_head_strikes"] == 1 + 2 + 3
    assert stats.loc[0, "f2_ground_atts"] == 4 * 3
    assert np.isnan(stats.loc[2, "f1_head_strikes"])               # no round data: not recorded, not 0


def test_control_time_blank_when_no_round_recorded_it():
    stats, _, _ = to_pipeline_tables(_wide())
    assert stats.loc[0, "f1_ctrl_time"] == 40
    assert np.isnan(stats.loc[2, "f1_ctrl_time"])


def test_weight_class_wording_and_fight_time():
    stats, _, _ = to_pipeline_tables(_wide())
    assert stats.loc[0, "weight_class"] == "UFC Women's Strawweight Title Bout"
    assert stats.loc[1, "weight_class"] == "Lightweight Bout"
    assert stats.loc[0, "f1_total_fight_time"] == 900
    assert stats.loc[2, "f1_total_fight_time"] == 150


def test_age_counts_whole_years_up_to_the_birthday():
    stats, _, _ = to_pipeline_tables(_wide())
    assert stats.loc[0, "f1_age_during"] == 30        # fight on the 30th birthday
    assert stats.loc[1, "f1_age_during"] == 30        # 2021-06-14: the 31st birthday is tomorrow


def test_ids_survive_a_trip_through_a_float_column():
    stats, _, fighters = to_pipeline_tables(_wide())
    ids = pd.concat([stats["f1_id"], stats["f2_id"], fighters["fighter_id"]])
    assert (ids.astype("float64").astype("int64") == ids).all()


def test_urls_keep_the_form_the_coin_flip_hashes():
    stats, events, fighters = to_pipeline_tables(_wide())
    assert stats["fights_url"].str.startswith("http://www.ufcstats.com/fight-details/").all()
    assert set(events["url_link"]) == set(stats["event_url"])
    assert fighters["fighter_url"].str.startswith("http://www.ufcstats.com/fighter-details/").all()


def test_one_row_per_fighter_with_the_latest_bio():
    wide = _wide()
    wide.loc[1, "f_1_fighter_reach_cm"] = 190.0      # Ann's profile updated by her 2021 fight
    _, _, fighters = to_pipeline_tables(wide)
    assert fighters["fighter_id"].is_unique and len(fighters) == 4
    assert fighters.set_index("fighter_name").loc["Ann", "fighter_reach_cm"] == 190.0


def test_a_missing_column_fails_loudly():
    with pytest.raises(ValueError, match="format has changed"):
        check_columns(_wide().drop(columns=["f_1_r2_head_succ"]))


def test_a_winner_matching_neither_fighter_fails_loudly():
    wide = _wide()
    wide.loc[0, "winner"] = "Somebody Else"
    with pytest.raises(ValueError, match="neither fighter"):
        to_pipeline_tables(wide)


def test_the_converted_tables_build_into_fights(tmp_path):
    """The real contract: the output must go through build_fight_dataset unchanged."""
    # ten extra decided fights, winners alternating, so the coin flip puts the
    # winner in corner A for some and corner B for others - a label that is
    # always False cannot pass by luck
    extra = [_fight(10 + i, "ann", "bea", ("Ann", "Bea")[i % 2], "KO/TKO", date=f"2022-01-{i + 1:02d}")
             for i in range(10)]
    stats, events, fighters = to_pipeline_tables(pd.concat([_wide(), pd.DataFrame(extra)], ignore_index=True))
    stats.to_csv(tmp_path / "ufc_event_fight_stats.csv", index=False)
    events.to_csv(tmp_path / "ufc_events.csv", index=False)
    fighters.to_csv(tmp_path / "ufc_fighters.csv", index=False)
    fights = build_fight_dataset(raw_dir=tmp_path)
    assert len(fights) == 14
    assert fights["outcome"].value_counts().to_dict().get("draw") == 2
    assert fights["fighter_a_won"].isna().sum() == 3      # two draws, one no-contest
    # and the label names the right fighter, whichever corner the coin flip put
    # the winner in. (15-digit ids once broke exactly this: rounded in a float
    # column, no winner id matched its fighter, and every label came out False.)
    winners = {**{f"f{10 + i:015d}": ("Ann", "Bea")[i % 2] for i in range(10)}, "f000000000000001": "Ann"}
    decided = fights[fights["fighter_a_won"].notna()]
    expected = decided["fighter_a_name"] == decided["fights_url"].str.split("/").str[-1].map(winners)
    assert decided["fighter_a_won"].astype(bool).tolist() == expected.tolist()
    assert decided["fighter_a_won"].any() and not decided["fighter_a_won"].all()


# ---------------------------------------------------------------- refreshing

def test_history_changes_spots_edits_but_tolerates_control_time_wobble():
    old, _, _ = to_pipeline_tables(_wide())
    new = old.copy()
    new.loc[0, "f1_ctrl_time"] += 3                    # ufcstats revision: tolerated
    new.loc[1, "f1_name"] = "Ann Renamed"              # spelling fix: ignored
    assert history_changes(new, old) == {"added": [], "removed": [], "changed": []}
    new.loc[2, "f1_sig_strikes"] = 99                  # a real change
    report = history_changes(new.iloc[:3], old)
    assert report["changed"] == [old.loc[2, "fights_url"]]
    assert report["removed"] == [old.loc[3, "fights_url"]]


def _fake_kaggle(monkeypatch, tmp_path, wide, version):
    folder = tmp_path / "download" / str(version)
    folder.mkdir(parents=True)
    wide.to_csv(folder / "UFC_full_data_silver_v2.csv", index=False)
    monkeypatch.setitem(sys.modules, "kagglehub", types.SimpleNamespace(dataset_download=lambda handle: str(folder)))


def test_refresh_writes_tables_and_pins_the_version(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _fake_kaggle(monkeypatch, tmp_path, _wide(), version=18)
    summary = refresh(raw_dir=raw)
    manifest = json.loads((raw / "source" / "SOURCE.json").read_text())
    assert manifest["version"] == 18 and manifest["fights"] == 4
    assert (raw / "ufc_event_fight_stats.csv").exists() and (raw / "source" / "UFC_full_data_silver_v2.csv").exists()
    assert summary["history_checked"] is False             # first download from this source


def test_refresh_refuses_to_rewrite_history(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _fake_kaggle(monkeypatch, tmp_path, _wide(), version=18)
    refresh(raw_dir=raw)
    edited = _wide()
    edited.loc[0, "f_1_sig_strikes_succ"] = 49             # a past fight silently changed upstream
    _fake_kaggle(monkeypatch, tmp_path, edited, version=19)
    with pytest.raises(ValueError, match="change 1 past fight"):
        refresh(raw_dir=raw)
    assert refresh(raw_dir=raw, allow_history_changes=True)["version"] == 19
