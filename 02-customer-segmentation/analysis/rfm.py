"""RFM (Recency, Frequency, Monetary) feature engineering for customer
segmentation, computed from real invoice-level transaction data.

Cleaning rules
--------------
- Rows with a missing Customer ID are dropped (can't attribute to a segment).
- Cancelled/returned orders (Invoice starting with "C") are excluded from
  Frequency and Monetary — they represent returns, not purchases.
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
