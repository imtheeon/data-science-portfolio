"""RFM (Recency, Frequency, Monetary) feature engineering for customer
segmentation, computed from real invoice-level transaction data.

Cleaning rules
--------------
- Rows with a missing Customer ID are dropped (can't attribute to a segment).
- Cancelled/returned orders (Invoice starting with "C") are excluded from
  Frequency and Monetary — they represent returns, not purchases.
- Rows with non-positive Quantity are dropped.

Deliberately NOT filtered (left as-is, a scope decision rather than an
oversight):
- Exact-duplicate line-item rows are not de-duplicated. In this dataset
  duplicates typically represent genuine repeated line items (e.g. the same
  SKU rung up twice on one invoice) rather than data-entry errors, so
  removing them would silently understate real Frequency/Monetary.
- Rows with Price == 0 (e.g. free promotional items, samples) are not
  excluded. They contribute 0 to Monetary but still count toward Frequency
  via their invoice, which is treated as acceptable for this analysis.
"""

from __future__ import annotations

import pandas as pd


def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["Customer ID"])
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["Price"]

    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    grouped = df.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda dates: (snapshot_date - dates.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    grouped = grouped.reset_index().rename(columns={"Customer ID": "CustomerID"})
    return grouped
