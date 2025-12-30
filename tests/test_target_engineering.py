import pandas as pd

from src.target_engineering import compute_rfm, merge_target_into_features


def test_compute_rfm_simple_case_with_snapshot():
    df = pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C2"],
            "TransactionId": ["T1", "T2", "T3"],
            "Value": [10, 20, 5],
            "TransactionStartTime": ["2018-01-01", "2018-01-03", "2018-01-02"],
        }
    )

    # Snapshot is 2018-01-04 so:
    # C1 last txn = 2018-01-03 => Recency=1, Freq=2, Monetary=30
    # C2 last txn = 2018-01-02 => Recency=2, Freq=1, Monetary=5
    rfm = compute_rfm(df, snapshot_date=pd.Timestamp("2018-01-04"))
    rfm = rfm.set_index("CustomerId")

    assert int(rfm.loc["C1", "Recency"]) == 1
    assert int(rfm.loc["C1", "Frequency"]) == 2
    assert float(rfm.loc["C1", "Monetary"]) == 30.0

    assert int(rfm.loc["C2", "Recency"]) == 2
    assert int(rfm.loc["C2", "Frequency"]) == 1
    assert float(rfm.loc["C2", "Monetary"]) == 5.0


def test_merge_target_into_features_adds_is_high_risk_and_fills_missing():
    features = pd.DataFrame({"CustomerId": ["C1", "C2"], "TransactionCount": [2, 1]})
    labeled = pd.DataFrame({"CustomerId": ["C1"], "is_high_risk": [1]})

    out = merge_target_into_features(features, labeled)
    assert "is_high_risk" in out.columns
    assert out.loc[out["CustomerId"] == "C1", "is_high_risk"].iloc[0] == 1
    assert out.loc[out["CustomerId"] == "C2", "is_high_risk"].iloc[0] == 0


