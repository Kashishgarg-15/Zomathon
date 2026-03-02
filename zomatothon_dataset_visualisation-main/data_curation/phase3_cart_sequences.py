"""
Phase 3 — Cart Sequence Simulation
===================================
Generates progressive cart states from multi-item orders using
category-based canonical ordering (Main → Side → Drink → Dessert).

Within the same category, items are ordered by price descending
(expensive items are typically chosen first, cheaper items are add-ons).

Produces: cart_sequences.csv
Columns:  order_id, stage, current_cart, next_item_added, next_item_category,
          next_item_role, cart_value_at_stage, items_in_cart, completeness_at_stage
"""
import json

import numpy as np
import pandas as pd

from data_curation.config import (
    CART_SEQUENCES_CSV,
    CATEGORY_ORDER,
    COMPLETENESS_WEIGHTS,
    ITEM_ATTRIBUTES_CSV,
    ITEM_STATS_CSV,
    ORDERS_ENRICHED_CSV,
)
from data_curation.utils import parse_items


def _category_sort_key(item: str, item_cat: dict, item_price: dict) -> tuple:
    """
    Sort key: (category_rank, -price).
    Main courses first, then sides, drinks, desserts.
    Within a category, expensive items first.
    """
    cat = item_cat.get(item, "main")
    try:
        cat_rank = CATEGORY_ORDER.index(cat)
    except ValueError:
        cat_rank = len(CATEGORY_ORDER)  # unknown → last
    price = item_price.get(item, 0)
    return (cat_rank, -price)


def _completeness_at_stage(items_in_cart: list[str], item_cat: dict) -> float:
    """Compute completeness score for a partial cart."""
    cats = set(item_cat.get(it, "unknown") for it in items_in_cart)
    score = 0.0
    if "main" in cats or "combo" in cats:
        score += COMPLETENESS_WEIGHTS["main"]
    if "side" in cats or "snack" in cats:
        score += COMPLETENESS_WEIGHTS["side"]
    if "drink" in cats:
        score += COMPLETENESS_WEIGHTS["drink"]
    if "dessert" in cats:
        score += COMPLETENESS_WEIGHTS["dessert"]
    return round(score, 3)


def run_phase3() -> dict:
    """Generate cart sequences from multi-item orders. Returns summary dict."""
    print("=" * 60)
    print("PHASE 3 — Cart Sequence Simulation")
    print("=" * 60)

    # Load data
    print("[1/3] Loading data …")
    orders = pd.read_csv(ORDERS_ENRICHED_CSV)
    attrs  = pd.read_csv(ITEM_ATTRIBUTES_CSV)
    items_stats = pd.read_csv(ITEM_STATS_CSV)

    # Build lookups
    item_cat   = dict(zip(attrs["item"], attrs["category"]))
    item_role  = dict(zip(attrs["item"], attrs["typical_role"]))
    item_price = dict(zip(items_stats["item"], items_stats["avg_price_proxy"]))

    # ── Generate sequences ──────────────────────────────────────────────
    print("[2/3] Generating cart sequences …")
    sequences = []
    multi_count = 0
    single_count = 0

    for _, row in orders.iterrows():
        parsed = parse_items(row["Items in order"]) if pd.notna(row["Items in order"]) else []
        item_names = [p["item"] for p in parsed]

        if len(item_names) <= 1:
            single_count += 1
            # Single-item orders: record as stage 1 with no next item
            # (these are inference targets, not training data)
            if item_names:
                sequences.append({
                    "order_id":              row["Order ID"],
                    "stage":                 1,
                    "current_cart":          json.dumps(item_names),
                    "next_item_added":       "",             # no next item
                    "next_item_category":    "",
                    "next_item_role":        "",
                    "cart_value_at_stage":   row.get("Bill subtotal", 0),
                    "items_in_cart":         1,
                    "completeness_at_stage": _completeness_at_stage(item_names, item_cat),
                    "is_final":              1,
                    "restaurant":            row["Restaurant name"],
                    "meal_period":           row.get("meal_period", "unknown"),
                })
            continue

        multi_count += 1

        # Sort items by category order, then by price descending
        sorted_items = sorted(
            item_names,
            key=lambda it: _category_sort_key(it, item_cat, item_price)
        )

        # Generate progressive stages
        # Stage 1: first item only → next item is second
        # Stage N-1: all but last → next item is last
        # Stage N (final): all items → no next
        bill = row.get("Bill subtotal", 0)
        n = len(sorted_items)

        for stage_idx in range(n):
            cart_so_far = sorted_items[:stage_idx + 1]
            next_item = sorted_items[stage_idx + 1] if stage_idx + 1 < n else ""
            is_final = 1 if stage_idx + 1 >= n else 0

            # Estimate cart value at this stage (proportional to items so far)
            est_value = bill * (stage_idx + 1) / n if n > 0 else 0

            sequences.append({
                "order_id":              row["Order ID"],
                "stage":                 stage_idx + 1,
                "current_cart":          json.dumps(cart_so_far),
                "next_item_added":       next_item,
                "next_item_category":    item_cat.get(next_item, "") if next_item else "",
                "next_item_role":        item_role.get(next_item, "") if next_item else "",
                "cart_value_at_stage":   round(est_value, 2),
                "items_in_cart":         len(cart_so_far),
                "completeness_at_stage": _completeness_at_stage(cart_so_far, item_cat),
                "is_final":              is_final,
                "restaurant":            row["Restaurant name"],
                "meal_period":           row.get("meal_period", "unknown"),
            })

    seq_df = pd.DataFrame(sequences)

    # ── Statistics ──────────────────────────────────────────────────────
    print("[3/3] Computing statistics …")
    seq_df.to_csv(CART_SEQUENCES_CSV, index=False)

    # Training-relevant stages (non-final, multi-item only)
    trainable = seq_df[(seq_df["is_final"] == 0) & (seq_df["next_item_added"] != "")]

    summary = {
        "total_sequences": len(seq_df),
        "multi_item_orders": multi_count,
        "single_item_orders": single_count,
        "trainable_stages": len(trainable),
        "avg_stages_per_multi_order": round(trainable.groupby("order_id").size().mean(), 2) if len(trainable) > 0 else 0,
        "stage_completeness_dist": {
            f"stage_{s}": round(g["completeness_at_stage"].mean(), 3)
            for s, g in trainable.groupby("stage")
            if s <= 5
        },
    }

    print(f"\n  Multi-item orders: {multi_count:,}")
    print(f"  Single-item orders (inference targets): {single_count:,}")
    print(f"  Trainable stages: {len(trainable):,}")
    print(f"\n  → Saved {CART_SEQUENCES_CSV.name}")
    print("Phase 3 complete ✓\n")
    return summary


if __name__ == "__main__":
    run_phase3()
