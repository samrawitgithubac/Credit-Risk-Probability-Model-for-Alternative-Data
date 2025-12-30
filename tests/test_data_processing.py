import pandas as pd

from src.data_processing import build_feature_table


def test_build_feature_table_expected_core_columns():
    df = pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C2"],
            "TransactionId": ["T1", "T2", "T3"],
            "Amount": [10.0, -2.0, 5.0],
            "Value": [10, 2, 5],
            "TransactionStartTime": ["2018-01-01 10:00:00", "2018-01-02 11:00:00", "2018-01-03 12:00:00"],
            "CurrencyCode": ["UGX", "UGX", "UGX"],
            "ProviderId": ["P1", "P1", "P2"],
            "ProductId": ["PR1", "PR2", "PR1"],
            "ProductCategory": ["airtime", "airtime", "airtime"],
            "ChannelId": ["web", "web", "android"],
        }
    )

    feats = build_feature_table(df)

    expected = {
        "CustomerId",
        "TotalTransactionAmount",
        "AverageTransactionAmount",
        "StdTransactionAmount",
        "TransactionCount",
        "TransactionMonth",
        "TransactionYear",
    }
    assert expected.issubset(set(feats.columns))


def test_build_feature_table_transaction_count_and_sum():
    df = pd.DataFrame(
        {
            "CustomerId": ["C1", "C1"],
            "TransactionId": ["T1", "T2"],
            "Amount": [10.0, 5.0],
            "Value": [10, 5],
            "TransactionStartTime": ["2018-01-01 10:00:00", "2018-01-02 11:00:00"],
        }
    )

    feats = build_feature_table(df)
    row = feats.loc[feats["CustomerId"] == "C1"].iloc[0]
    assert row["TransactionCount"] == 2
    assert row["TotalTransactionAmount"] == 15.0


