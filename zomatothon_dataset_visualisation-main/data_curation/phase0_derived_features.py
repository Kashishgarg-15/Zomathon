"""
Phase 0 — Derived Features (zero cost, pure pandas)
====================================================
Produces:
  • orders_enriched.csv   — order-level features (temporal, cart stats)
  • user_profiles.csv     — per-customer aggregated behavior
  • item_stats.csv        — per-item aggregated statistics
  • copurchase_matrix.csv — item-pair co-occurrence with lift & confidence
"""
import itertools
from collections import Counter

import numpy as np
import pandas as pd

from data_curation.config import (
    COPURCHASE_CSV,
    ITEM_STATS_CSV,
    ORDERS_ENRICHED_CSV,
    USER_PROFILES_CSV,
)
from data_curation.utils import get_items_per_order, load_raw_data, parse_items


# ─────────────────────────────────────────────────────────────────────────
#  Temporal features
# ─────────────────────────────────────────────────────────────────────────
def _assign_meal_period(hour: int) -> str:
    """Map hour (0-23) → meal period label."""
    if 6 <= hour < 10:
        return "breakfast"
    elif 10 <= hour < 14:
        return "lunch"
    elif 14 <= hour < 17:
        return "snack"
    elif 17 <= hour < 22:
        return "dinner"
    else:
        return "late_night"


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based columns from order_datetime."""
    dt = df["order_datetime"]
    df["order_hour"]     = dt.dt.hour
    df["order_dow"]      = dt.dt.dayofweek          # 0=Mon … 6=Sun
    df["order_day_name"] = dt.dt.day_name()
    df["is_weekend"]     = df["order_dow"].isin([5, 6]).astype(int)
    df["meal_period"]    = df["order_hour"].apply(
        lambda h: _assign_meal_period(h) if pd.notna(h) else "unknown"
    )
    df["order_date"]     = dt.dt.date
    return df


# ─────────────────────────────────────────────────────────────────────────
#  Cart-level features
# ─────────────────────────────────────────────────────────────────────────
def add_cart_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features derived from items-in-order."""
    parsed = df["Items in order"].apply(
        lambda x: parse_items(x) if pd.notna(x) else []
    )
    df["num_items"]       = parsed.apply(lambda lst: sum(d["qty"] for d in lst))
    df["num_unique_items"] = parsed.apply(lambda lst: len(set(d["item"] for d in lst)))
    df["is_single_item"]  = (df["num_unique_items"] == 1).astype(int)

    # Price per item (rough)
    df["price_per_item"] = np.where(
        df["num_items"] > 0,
        df["Bill subtotal"] / df["num_items"],
        np.nan,
    )

    # Discount features
    df["total_discount"] = (
        df["Restaurant discount (Promo)"].fillna(0)
        + df["Restaurant discount (Flat offs, Freebies & others)"].fillna(0)
        + df["Gold discount"].fillna(0)
        + df["Brand pack discount"].fillna(0)
    )
    df["discount_pct"] = np.where(
        df["Bill subtotal"] > 0,
        df["total_discount"] / df["Bill subtotal"] * 100,
        0,
    )
    df["has_discount"] = (df["total_discount"] > 0).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────
#  User profiles
# ─────────────────────────────────────────────────────────────────────────
def build_user_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-customer behavioral features."""
    grp = df.groupby("Customer ID")

    profiles = pd.DataFrame({
        "order_count":           grp["Order ID"].nunique(),
        "avg_order_value":       grp["Total"].mean(),
        "std_order_value":       grp["Total"].std().fillna(0),
        "avg_items_per_order":   grp["num_items"].mean(),
        "avg_unique_items":      grp["num_unique_items"].mean(),
        "single_item_ratio":     grp["is_single_item"].mean(),
        "avg_discount_pct":      grp["discount_pct"].mean(),
        "total_spend":           grp["Total"].sum(),
        "preferred_restaurant":  grp["Restaurant name"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown"),
        "preferred_meal_period": grp["meal_period"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown"),
        "weekend_ratio":         grp["is_weekend"].mean(),
    })

    # Diversity score: unique items ordered / total items ordered
    item_lists = df.groupby("Customer ID").apply(
        lambda g: get_items_per_order(g), include_groups=False
    )
    # Flatten per customer
    def _diversity(item_series):
        all_items = []
        for lst in item_series:
            all_items.extend(lst)
        if len(all_items) == 0:
            return 0.0
        return len(set(all_items)) / len(all_items)

    profiles["diversity_score"] = item_lists.apply(_diversity)

    # Recency: days between first and last order
    profiles["first_order"] = grp["order_datetime"].min()
    profiles["last_order"]  = grp["order_datetime"].max()
    profiles["tenure_days"] = (profiles["last_order"] - profiles["first_order"]).dt.days

    # Classify user as cold-start or returning
    profiles["is_cold_start"] = (profiles["order_count"] == 1).astype(int)

    # Veg preference proxy — will be refined after Phase 1
    # (for now, just a placeholder)
    profiles["veg_preference"] = np.nan

    profiles = profiles.reset_index()
    return profiles


# ─────────────────────────────────────────────────────────────────────────
#  Item statistics
# ─────────────────────────────────────────────────────────────────────────
def build_item_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-item aggregated statistics from order data."""
    rows = []
    for _, row in df.iterrows():
        parsed = parse_items(row["Items in order"]) if pd.notna(row["Items in order"]) else []
        n_items = len(parsed)
        for p in parsed:
            rows.append({
                "item":             p["item"],
                "qty":              p["qty"],
                "order_id":         row["Order ID"],
                "restaurant":       row["Restaurant name"],
                "bill_subtotal":    row["Bill subtotal"],
                "n_items_in_order": n_items,
                "meal_period":      row.get("meal_period", "unknown"),
                "is_weekend":       row.get("is_weekend", 0),
            })

    item_df = pd.DataFrame(rows)

    stats = item_df.groupby("item").agg(
        order_frequency    = ("order_id", "nunique"),
        total_qty_sold     = ("qty", "sum"),
        restaurant         = ("restaurant", lambda x: x.mode().iloc[0]),
        avg_bill_when_present = ("bill_subtotal", "mean"),
        solo_order_count   = ("n_items_in_order", lambda x: (x == 1).sum()),
        multi_order_count  = ("n_items_in_order", lambda x: (x > 1).sum()),
    ).reset_index()

    stats["solo_order_ratio"] = stats["solo_order_count"] / stats["order_frequency"]
    stats["popularity_rank"]  = stats["order_frequency"].rank(ascending=False, method="min").astype(int)

    # Price bucket (percentile-based from avg bill ÷ items)
    avg_price = item_df.groupby("item").apply(
        lambda g: (g["bill_subtotal"] / g["n_items_in_order"]).mean(), include_groups=False
    ).rename("avg_price_proxy")
    stats = stats.merge(avg_price, left_on="item", right_index=True, how="left")

    # Meal period affinity
    meal_pivot = item_df.groupby(["item", "meal_period"]).size().unstack(fill_value=0)
    if not meal_pivot.empty:
        meal_pivot = meal_pivot.div(meal_pivot.sum(axis=1), axis=0)
        stats["peak_meal_period"] = meal_pivot.idxmax(axis=1).reindex(stats["item"]).values

    return stats


# ─────────────────────────────────────────────────────────────────────────
#  Co-purchase matrix
# ─────────────────────────────────────────────────────────────────────────
def build_copurchase_matrix(df: pd.DataFrame, min_support: int = 5) -> pd.DataFrame:
    """
    Build item-pair co-occurrence with support, confidence, and lift.
    Only multi-item orders contribute.
    """
    item_lists = get_items_per_order(df)
    multi_orders = item_lists[item_lists.apply(len) > 1]
    total_orders = len(df)

    # Item frequency
    item_freq = Counter()
    pair_freq = Counter()

    for items in multi_orders:
        unique = list(set(items))
        for it in unique:
            item_freq[it] += 1
        for a, b in itertools.combinations(sorted(unique), 2):
            pair_freq[(a, b)] += 1

    rows = []
    for (a, b), count in pair_freq.items():
        if count < min_support:
            continue
        support    = count / total_orders
        conf_a_b   = count / item_freq[a] if item_freq[a] > 0 else 0
        conf_b_a   = count / item_freq[b] if item_freq[b] > 0 else 0
        lift       = (count * total_orders) / (item_freq[a] * item_freq[b]) if (item_freq[a] * item_freq[b]) > 0 else 0
        rows.append({
            "item_a":       a,
            "item_b":       b,
            "co_count":     count,
            "support":      round(support, 6),
            "confidence_a_given_b": round(conf_a_b, 4),
            "confidence_b_given_a": round(conf_b_a, 4),
            "lift":         round(lift, 4),
        })

    co_df = pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)
    return co_df


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────
def run_phase0() -> dict:
    """Execute Phase 0 and save all outputs. Returns summary dict."""
    print("=" * 60)
    print("PHASE 0 — Derived Features")
    print("=" * 60)

    # Load
    print("[1/5] Loading raw data …")
    df = load_raw_data()
    print(f"       {len(df):,} orders loaded")

    # Temporal
    print("[2/5] Adding temporal features …")
    df = add_temporal_features(df)

    # Cart
    print("[3/5] Adding cart features …")
    df = add_cart_features(df)
    df.to_csv(ORDERS_ENRICHED_CSV, index=False)
    print(f"       → Saved {ORDERS_ENRICHED_CSV.name}")

    # User profiles
    print("[4/5] Building user profiles …")
    users = build_user_profiles(df)
    users.to_csv(USER_PROFILES_CSV, index=False)
    print(f"       → Saved {USER_PROFILES_CSV.name}  ({len(users):,} users)")

    # Item stats
    print("[5/5] Building item statistics & co-purchase matrix …")
    items = build_item_stats(df)
    items.to_csv(ITEM_STATS_CSV, index=False)
    print(f"       → Saved {ITEM_STATS_CSV.name}  ({len(items):,} items)")

    co = build_copurchase_matrix(df, min_support=3)
    co.to_csv(COPURCHASE_CSV, index=False)
    print(f"       → Saved {COPURCHASE_CSV.name}  ({len(co):,} item pairs)")

    summary = {
        "orders": len(df),
        "users": len(users),
        "unique_items": len(items),
        "copurchase_pairs": len(co),
        "cold_start_users": int(users["is_cold_start"].sum()),
        "cold_start_pct": round(users["is_cold_start"].mean() * 100, 1),
        "single_item_orders_pct": round(df["is_single_item"].mean() * 100, 1),
    }
    print(f"\n  Summary: {summary}")
    print("Phase 0 complete ✓\n")
    return summary


if __name__ == "__main__":
    run_phase0()
