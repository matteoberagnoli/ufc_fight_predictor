import pandas as pd

from ufc_predictor.modeling.present import fighter_records, p4p


def _fights():
    """Fighter 1 beats everyone; 2 and 3 trade wins; 9 is a woman; 7 stopped fighting in 2019."""
    rows = [
        ("2025-01-01", 1, 2, "fighter_a", False), ("2025-02-01", 1, 3, "fighter_a", False),
        ("2025-03-01", 2, 3, "fighter_a", False), ("2025-04-01", 3, 2, "fighter_a", False),
        ("2025-05-01", 1, 2, "fighter_a", False), ("2025-06-01", 3, 2, "draw", False),
        ("2019-01-01", 7, 2, "fighter_a", False), ("2019-02-01", 7, 3, "fighter_a", False),
        ("2019-03-01", 7, 1, "fighter_a", False),
        ("2025-01-15", 9, 8, "fighter_a", True), ("2025-02-15", 9, 8, "fighter_a", True),
        ("2025-03-15", 8, 9, "fighter_b", True),
    ]
    f = pd.DataFrame(rows, columns=["event_date", "fighter_a_id", "fighter_b_id", "outcome", "is_womens_bout"])
    f["event_date"] = pd.to_datetime(f["event_date"])
    f["fights_url"] = [f"f/{i:02d}" for i in range(len(f))]
    f["fighter_a_name"] = "F" + f["fighter_a_id"].astype(str)
    f["fighter_b_name"] = "F" + f["fighter_b_id"].astype(str)
    f["division"] = "lightweight"
    f["method"] = "finish"
    return f


def test_records_count_wins_losses_and_draws():
    rec = fighter_records(_fights())
    assert rec.loc[1, "record"] == "3-1"      # 3 wins in 2025, lost to 7 in 2019
    assert rec.loc[3, "record"] == "1-3-1"
    assert rec.loc[9, "womens"] and not rec.loc[1, "womens"]


def test_p4p_ranks_active_fighters_by_rating_men_and_women_apart():
    ranks = p4p(_fights(), 64, None)
    men, women = ranks["men"], ranks["women"]
    assert men["name"].tolist()[0] == "F1"
    assert "F7" not in men["name"].tolist()          # last fought in 2019: not active
    assert men["elo"].is_monotonic_decreasing
    assert women["name"].tolist() == ["F9", "F8"]
    assert men["rank"].tolist() == list(range(1, len(men) + 1))


def test_p4p_needs_enough_fights():
    assert p4p(_fights(), 64, None, min_fights=4)["women"].empty
