"""
Phase 2 — Meal Completeness Scoring
====================================
Uses item_attributes.csv (from Phase 1) to score each order's meal completeness
and flag missing components.

Produces: order_completeness.csv
Columns:  Order ID, completeness_score, has_main, has_side, has_drink,
          has_dessert, missing_categories, recommendation_priority
"""
import pandas as pd
import numpy as np

from data_curation.config import (
    COMPLETENESS_CSV,
    COMPLETENESS_WEIGHTS,
    ITEM_ATTRIBUTES_CSV,
    ORDERS_ENRICHED_CSV,
)
from data_curation.utils import parse_items


def run_phase2() -> dict:
    """Score each order's meal completeness. Returns summary dict."""
    print("=" * 60)
    print("PHASE 2 — Meal Completeness Scoring")
    print("=" * 60)

    # Load enriched orders + item attributes
    print("[1/3] Loading data …")
    orders = pd.read_csv(ORDERS_ENRICHED_CSV)
    attrs  = pd.read_csv(ITEM_ATTRIBUTES_CSV)

    # Build item → category lookup
    item_cat = dict(zip(attrs["item"], attrs["category"]))
    item_veg = dict(zip(attrs["item"], attrs["veg_nonveg"]))

    # ── Score each order ────────────────────────────────────────────────
    print("[2/3] Scoring orders …")
    results = []
    for _, row in orders.iterrows():
        parsed = parse_items(row["Items in order"]) if pd.notna(row["Items in order"]) else []
        item_names = [p["item"] for p in parsed]

        # Get categories present
        categories_present = set()
        veg_types = set()
        for it in item_names:
            cat = item_cat.get(it, "unknown")
            categories_present.add(cat)
            vt = item_veg.get(it, "unknown")
            veg_types.add(vt)

        # Component flags
        has_main    = int("main" in categories_present or "combo" in categories_present)
        has_side    = int("side" in categories_present or "snack" in categories_present)
        has_drink   = int("drink" in categories_present)
        has_dessert = int("dessert" in categories_present)

        # Completeness score (weighted)
        score = (
            COMPLETENESS_WEIGHTS["main"]    * has_main
            + COMPLETENESS_WEIGHTS["side"]  * has_side
            + COMPLETENESS_WEIGHTS["drink"] * has_drink
            + COMPLETENESS_WEIGHTS["dessert"] * has_dessert
        )

        # Missing categories
        missing = []
        if not has_main:    missing.append("main")
        if not has_side:    missing.append("side")
        if not has_drink:   missing.append("drink")
        if not has_dessert: missing.append("dessert")

        # Recommendation priority: lower completeness = higher priority
        # Also boost single-item orders
        is_single = 1 if len(item_names) <= 1 else 0
        priority = (1 - score) + 0.2 * is_single

        # Veg classification of order
        if veg_types == {"veg"}:
            order_veg = "veg"
        elif "non-veg" in veg_types:
            order_veg = "non-veg"
        elif "egg" in veg_types:
            order_veg = "egg"
        else:
            order_veg = "unknown"

        results.append({
            "Order ID":               row["Order ID"],
            "completeness_score":     round(score, 3),
            "has_main":               has_main,
            "has_side":               has_side,
            "has_drink":              has_drink,
            "has_dessert":            has_dessert,
            "missing_categories":     "|".join(missing) if missing else "none",
            "num_missing":            len(missing),
            "recommendation_priority": round(priority, 3),
            "order_veg_type":         order_veg,
            "categories_present":     "|".join(sorted(categories_present)),
        })

    comp_df = pd.DataFrame(results)

    # ── Summary stats ───────────────────────────────────────────────────
    print("[3/3] Computing statistics …")
    comp_df.to_csv(COMPLETENESS_CSV, index=False)

    avg_score = comp_df["completeness_score"].mean()
    fully_complete = (comp_df["completeness_score"] >= 1.0).sum()
    needs_rec = (comp_df["completeness_score"] < 0.8).sum()

    # Most common missing component
    all_missing = []
    for m in comp_df["missing_categories"]:
        if m != "none":
            all_missing.extend(m.split("|"))
    from collections import Counter
    missing_counts = Counter(all_missing).most_common()

    summary = {
        "avg_completeness_score": round(avg_score, 3),
        "fully_complete_orders": int(fully_complete),
        "fully_complete_pct": round(fully_complete / len(comp_df) * 100, 1),
        "needs_recommendation": int(needs_rec),
        "needs_recommendation_pct": round(needs_rec / len(comp_df) * 100, 1),
        "missing_component_freq": dict(missing_counts),
        "veg_order_split": comp_df["order_veg_type"].value_counts().to_dict(),
    }

    print(f"\n  Avg completeness: {avg_score:.3f}")
    print(f"  Fully complete:   {fully_complete} ({fully_complete/len(comp_df)*100:.1f}%)")
    print(f"  Need add-ons:     {needs_rec} ({needs_rec/len(comp_df)*100:.1f}%)")
    print(f"  Missing freq:     {dict(missing_counts)}")
    print(f"\n  → Saved {COMPLETENESS_CSV.name}")
    print("Phase 2 complete ✓\n")
    return summary


if __name__ == "__main__":
    run_phase2()
