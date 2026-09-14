import pandas as pd
import pytest

from ufc_predictor.data.randomize import (
    find_side_column_pairs,
    randomize_fighter_sides,
    stable_swap_mask,
)


@pytest.fixture
def toy_fights() -> pd.DataFrame:
    # 3 fights, f1 always the winner - mimics the pre-2010 slice of the real data.
    return pd.DataFrame(
        {
            "fights_url": ["fight/aaa", "fight/bbb", "fight/ccc"],
            "f1_id": [10, 20, 30],
            "f2_id": [11, 21, 31],
            "f1_sig_strikes": [50, 12, 80],
            "f2_sig_strikes": [30, 40, 5],
            "weight_class": ["Lightweight", "Welterweight", "Heavyweight"],
            "winner_id": [10, 20, 30],
        }
    )


def test_find_side_column_pairs_matches_only_full_pairs():
    columns = ["f1_id", "f2_id", "f1_strikes", "f2_strikes", "f1_orphan", "weight_class"]
    pairs = find_side_column_pairs(columns)
    suffixes = {s for s, _, _ in pairs}
    assert suffixes == {"id", "strikes"}
    assert "orphan" not in suffixes


def test_swap_mask_is_deterministic_and_not_constant():
    ids = pd.Series([f"fight/{i}" for i in range(200)])
    mask_a = stable_swap_mask(ids, seed=42)
    mask_b = stable_swap_mask(ids, seed=42)
    assert (mask_a == mask_b).all()
    # a real coin flip shouldn't land all-heads or all-tails over 200 draws
    assert 0 < mask_a.mean() < 1


def test_swap_mask_depends_on_id_not_position():
    # same ids, different order -> each id still gets the same flip
    ids = pd.Series(["fight/1", "fight/2", "fight/3"])
    shuffled = pd.Series(["fight/3", "fight/1", "fight/2"])
    by_id = dict(zip(ids, stable_swap_mask(ids, seed=7)))
    by_id_shuffled = dict(zip(shuffled, stable_swap_mask(shuffled, seed=7)))
    assert by_id == by_id_shuffled


def test_randomize_preserves_fighter_identity_set(toy_fights):
    out = randomize_fighter_sides(toy_fights)
    for i in range(len(toy_fights)):
        original = {toy_fights.loc[i, "f1_id"], toy_fights.loc[i, "f2_id"]}
        after = {out.loc[i, "fighter_a_id"], out.loc[i, "fighter_b_id"]}
        assert original == after


def test_randomize_keeps_identity_and_stats_glued_together(toy_fights):
    # whichever slot a fighter's id lands in, their own stat travels with it
    out = randomize_fighter_sides(toy_fights)
    for i in range(len(toy_fights)):
        if out.loc[i, "fighter_a_id"] == toy_fights.loc[i, "f1_id"]:
            assert out.loc[i, "fighter_a_sig_strikes"] == toy_fights.loc[i, "f1_sig_strikes"]
            assert out.loc[i, "fighter_b_sig_strikes"] == toy_fights.loc[i, "f2_sig_strikes"]
        else:
            assert out.loc[i, "fighter_a_sig_strikes"] == toy_fights.loc[i, "f2_sig_strikes"]
            assert out.loc[i, "fighter_b_sig_strikes"] == toy_fights.loc[i, "f1_sig_strikes"]


def test_randomize_label_matches_swap(toy_fights):
    out = randomize_fighter_sides(toy_fights)
    for i in range(len(toy_fights)):
        expected = out.loc[i, "fighter_a_id"] == toy_fights.loc[i, "winner_id"]
        assert out.loc[i, "fighter_a_won"] == expected


def test_randomize_breaks_the_all_f1_wins_artifact(toy_fights):
    # before: f1 (= winner) sits in the same slot for every row (the actual bug)
    assert (toy_fights["f1_id"] == toy_fights["winner_id"]).all()
    # after: winner isn't pinned to fighter_a for every row anymore.
    # seed pinned explicitly (not the config default) and verified to produce
    # a mix for this 3-row fixture - with only 3 coin flips, an unpinned seed
    # can land on all-heads or all-tails by chance and make this test flaky.
    out = randomize_fighter_sides(toy_fights, seed=0)
    assert not out["fighter_a_won"].all()


def test_randomize_is_stable_when_new_fights_are_appended(toy_fights):
    out_before = randomize_fighter_sides(toy_fights)
    appended = pd.concat(
        [
            toy_fights,
            pd.DataFrame(
                {
                    "fights_url": ["fight/ddd"],
                    "f1_id": [40],
                    "f2_id": [41],
                    "f1_sig_strikes": [15],
                    "f2_sig_strikes": [15],
                    "weight_class": ["Bantamweight"],
                    "winner_id": [41],
                }
            ),
        ],
        ignore_index=True,
    )
    out_after = randomize_fighter_sides(appended)
    common = out_after.iloc[: len(toy_fights)]
    pd.testing.assert_frame_equal(
        out_before.reset_index(drop=True), common.reset_index(drop=True)
    )
