import pandas as pd

from ufc_predictor.data.build import build_fight_dataset


def test_build_fight_dataset_drops_only_undecided_results():
    raw_count = pd.read_csv("data/raw/ufc_event_fight_stats.csv").shape[0]
    df = build_fight_dataset()
    # 143 draws/no-contests removed, everything else kept
    assert len(df) == raw_count - 143


def test_build_fight_dataset_is_chronological():
    df = build_fight_dataset()
    assert df["event_date"].is_monotonic_increasing


def test_build_fight_dataset_label_balance_holds_in_every_era():
    df = build_fight_dataset()

    overall_rate = df["fighter_a_won"].mean()
    assert 0.45 < overall_rate < 0.55

    # this is the specific bug being fixed: raw data has f1 (= winner) for
    # every fight before 2010. Individual years are too small a sample to
    # expect exactly 50/50 (e.g. 1998 has 11 fights total), but the pre-2010
    # block as a whole (1,169 fights) should no longer be anywhere near 100%.
    pre_2010 = df[df["event_date"].dt.year < 2010]
    assert len(pre_2010) > 1000
    assert 0.40 < pre_2010["fighter_a_won"].mean() < 0.60


def test_build_fight_dataset_is_reproducible():
    first = build_fight_dataset()
    second = build_fight_dataset()
    pd.testing.assert_frame_equal(first, second)
