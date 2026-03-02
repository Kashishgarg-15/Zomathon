"""
Shared utilities for parsing & loading data.
"""
import re
import pandas as pd
from data_curation.config import RAW_CSV


def load_raw_data() -> pd.DataFrame:
    """Load the raw CSV and do minimal cleaning."""
    df = pd.read_csv(RAW_CSV)

    # Parse datetime
    df["order_datetime"] = pd.to_datetime(
        df["Order Placed At"], format="%I:%M %p, %B %d %Y", errors="coerce"
    )

    # Numeric distance (km)
    df["distance_km"] = df["Distance"].apply(_parse_distance)

    return df


def _parse_distance(val) -> float:
    """Convert '3km', '<1km', '2km' → float."""
    if pd.isna(val):
        return None
    s = str(val).lower().replace("km", "").strip()
    if s.startswith("<"):
        return 0.5  # conservative estimate for <1km
    try:
        return float(s)
    except ValueError:
        return None


def parse_items(items_str: str) -> list[dict]:
    """
    Parse 'Items in order' column.
    '1 x Biryani, 2 x Raita' → [{'item': 'Biryani', 'qty': 1}, {'item': 'Raita', 'qty': 2}]
    """
    if pd.isna(items_str):
        return []
    # Split on comma followed by "N x" to avoid consuming the next item's prefix
    parts = re.split(r",\s*(?=\d+\s*x\s)", items_str)
    results = []
    for part in parts:
        match = re.match(r"(\d+)\s*x\s*(.+)", part.strip())
        if match:
            results.append({"item": match.group(2).strip(), "qty": int(match.group(1))})
    return results


def get_unique_items(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique item names across all orders."""
    items = set()
    for items_str in df["Items in order"].dropna():
        for parsed in parse_items(items_str):
            items.add(parsed["item"])
    return sorted(items)


def get_items_per_order(df: pd.DataFrame) -> pd.Series:
    """Return a Series (aligned to df index) with item names as lists."""
    return df["Items in order"].apply(
        lambda x: [p["item"] for p in parse_items(x)] if pd.notna(x) else []
    )
