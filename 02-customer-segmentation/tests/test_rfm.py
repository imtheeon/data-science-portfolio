import pandas as pd

from analysis.rfm import compute_rfm


def test_compute_rfm_basic():
    df = pd.DataFrame(
        {
            "Invoice": ["1", "1", "2", "3"],
            "Customer ID": [100, 100, 100, 200],
            "InvoiceDate": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-10", "2024-01-15"]
            ),
            "Quantity": [2, 1, 3, 5],
            "Price": [10.0, 5.0, 2.0, 4.0],
        }
    )
    snapshot = pd.Timestamp("2024-01-16")

    rfm = compute_rfm(df, snapshot_date=snapshot)

    cust100 = rfm[rfm["CustomerID"] == 100].iloc[0]
    cust200 = rfm[rfm["CustomerID"] == 200].iloc[0]

    assert cust100["Recency"] == 6          # snapshot - 2024-01-10
    assert cust100["Frequency"] == 2         # 2 distinct invoices
    assert cust100["Monetary"] == 2 * 10.0 + 1 * 5.0 + 3 * 2.0  # 31.0

    assert cust200["Recency"] == 1           # snapshot - 2024-01-15
    assert cust200["Frequency"] == 1
    assert cust200["Monetary"] == 5 * 4.0    # 20.0


def test_compute_rfm_excludes_cancelled_and_missing_customer():
    df = pd.DataFrame(
        {
            "Invoice": ["1", "C2", "3"],  # "C..." = cancellation/return
            "Customer ID": [100, 100, None],
            "InvoiceDate": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "Quantity": [2, -1, 5],
            "Price": [10.0, 10.0, 4.0],
        }
    )
    rfm = compute_rfm(df, snapshot_date=pd.Timestamp("2024-01-04"))
    assert len(rfm) == 1
    assert rfm.iloc[0]["CustomerID"] == 100
    assert rfm.iloc[0]["Frequency"] == 1
    assert rfm.iloc[0]["Monetary"] == 20.0
