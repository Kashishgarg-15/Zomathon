"""
Phase 4 — Training Data Construction
=====================================
Combines all previous phases into a model-ready feature matrix.

For each trainable cart stage (from Phase 3), creates:
  • 1 positive sample  (the item actually added next)
  • N negative samples (other items from the same restaurant NOT chosen)

Feature groups:
  1. Cart context       — items_in_cart, cart_value, completeness, missing categories
  2. Candidate item     — category, veg, cuisine, price, popularity, role
  3. User profile       — avg_order_value, avg_items, diversity, veg_preference
  4. Temporal           — meal_period, is_weekend, order_hour
  5. Co-purchase signal — lift, co-occurrence with current cart items

Produces: training_data.csv
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from data_curation.config import (
    CART_SEQUENCES_CSV,
    COMPLETENESS_CSV,
    COPURCHASE_CSV,
    ITEM_ATTRIBUTES_CSV,
    ITEM_STATS_CSV,
    NEG_SAMPLE_RATIO,
    ORDERS_ENRICHED_CSV,
    TRAINING_DATA_CSV,
    USER_PROFILES_CSV,
)


def _build_copurchase_lookup(co_df: pd.DataFrame) -> dict:
    """Build fast lookup: (item_a, item_b) → {lift, confidence, co_count}."""
    lookup = {}
    for _, row in co_df.iterrows():
        a, b = row["item_a"], row["item_b"]
        entry = {
            "lift":       row["lift"],
            "co_count":   row["co_count"],
            "conf_a_b":   row["confidence_a_given_b"],
            "conf_b_a":   row["confidence_b_given_a"],
        }
        lookup[(a, b)] = entry
        lookup[(b, a)] = {
            "lift":     entry["lift"],
            "co_count": entry["co_count"],
            "conf_a_b": entry["conf_b_a"],
            "conf_b_a": entry["conf_a_b"],
        }
    return lookup


def _copurchase_features(cart_items: list[str], candidate: str, lookup: dict) -> dict:
    """Aggregate co-purchase signals between candidate and all items in cart."""
    lifts = []
    co_counts = []
    confidences = []

    for cart_item in cart_items:
        key = (cart_item, candidate)
        if key in lookup:
            entry = lookup[key]
            lifts.append(entry["lift"])
            co_counts.append(entry["co_count"])
            confidences.append(entry["conf_a_b"])

    return {
        "max_lift":         max(lifts) if lifts else 0.0,
        "avg_lift":         np.mean(lifts) if lifts else 0.0,
        "total_co_count":   sum(co_counts),
        "max_confidence":   max(confidences) if confidences else 0.0,
        "copurchase_pairs": len(lifts),
    }


def run_phase4() -> dict:
    """Build the final training dataset. Returns summary dict."""
    print("=" * 60)
    print("PHASE 4 — Training Data Construction")
    print("=" * 60)

    # ── Load all artifacts ──────────────────────────────────────────────
    print("[1/4] Loading artifacts …")
    orders      = pd.read_csv(ORDERS_ENRICHED_CSV)
    users       = pd.read_csv(USER_PROFILES_CSV)
    item_attrs  = pd.read_csv(ITEM_ATTRIBUTES_CSV)
    item_stats  = pd.read_csv(ITEM_STATS_CSV)
    co_df       = pd.read_csv(COPURCHASE_CSV)
    completeness = pd.read_csv(COMPLETENESS_CSV)
    sequences   = pd.read_csv(CART_SEQUENCES_CSV)

    # ── Build lookups ───────────────────────────────────────────────────
    print("[2/4] Building lookups …")

    # Item attributes
    item_attr_dict = {}
    for _, row in item_attrs.iterrows():
        item_attr_dict[row["item"]] = {
            "category":       row["category"],
            "veg_nonveg":     row["veg_nonveg"],
            "cuisine":        row["cuisine"],
            "typical_role":   row["typical_role"],
            "flavor_profile": row["flavor_profile"],
        }

    # Item stats
    item_stat_dict = {}
    for _, row in item_stats.iterrows():
        item_stat_dict[row["item"]] = {
            "popularity_rank":   row["popularity_rank"],
            "order_frequency":   row["order_frequency"],
            "solo_order_ratio":  row["solo_order_ratio"],
            "avg_price_proxy":   row["avg_price_proxy"],
        }

    # User profiles
    user_dict = {}
    for _, row in users.iterrows():
        user_dict[row["Customer ID"]] = {
            "user_order_count":     row["order_count"],
            "user_avg_order_value": row["avg_order_value"],
            "user_avg_items":       row["avg_items_per_order"],
            "user_diversity":       row["diversity_score"],
            "user_weekend_ratio":   row["weekend_ratio"],
            "user_single_item_ratio": row["single_item_ratio"],
            "user_is_cold_start":   row["is_cold_start"],
        }

    # Order → user mapping
    order_user = dict(zip(orders["Order ID"], orders["Customer ID"]))
    order_hour = dict(zip(orders["Order ID"], orders.get("order_hour", pd.Series(dtype=float))))
    order_weekend = dict(zip(orders["Order ID"], orders.get("is_weekend", pd.Series(dtype=int))))
    order_restaurant = dict(zip(orders["Order ID"], orders["Restaurant name"]))

    # Restaurant → menu items
    restaurant_menu: dict[str, set[str]] = defaultdict(set)
    for item, row in item_attrs.iterrows():
        restaurant_menu[row["restaurant"]].add(row["item"])

    # Also populate from item_stats
    for _, row in item_stats.iterrows():
        restaurant_menu[row["restaurant"]].add(row["item"])

    # Co-purchase lookup
    co_lookup = _build_copurchase_lookup(co_df)

    # Completeness info
    comp_dict = {}
    for _, row in completeness.iterrows():
        comp_dict[row["Order ID"]] = {
            "missing_categories": row["missing_categories"],
            "order_veg_type":     row["order_veg_type"],
        }

    # ── Filter trainable stages ─────────────────────────────────────────
    print("[3/4] Generating training samples …")
    trainable = sequences[
        (sequences["is_final"] == 0) & (sequences["next_item_added"] != "")
    ].copy()
    print(f"       {len(trainable):,} trainable stages")

    training_rows = []
    pos_count = 0
    neg_count = 0

    for _, stage in trainable.iterrows():
        order_id = stage["order_id"]
        cart_items = json.loads(stage["current_cart"])
        next_item = stage["next_item_added"]
        restaurant = stage["restaurant"]
        customer_id = order_user.get(order_id, "")

        # ── Shared context features ─────────────────────────────────────
        context = {
            "order_id":              order_id,
            "stage":                 stage["stage"],
            "items_in_cart":         stage["items_in_cart"],
            "cart_value":            stage["cart_value_at_stage"],
            "completeness":          stage["completeness_at_stage"],
            "meal_period":           stage["meal_period"],
            "order_hour":            order_hour.get(order_id, -1),
            "is_weekend":            order_weekend.get(order_id, 0),
            "restaurant":            restaurant,
        }

        # Missing categories
        comp_info = comp_dict.get(order_id, {})
        missing = comp_info.get("missing_categories", "")
        context["missing_main"]    = int("main" in str(missing))
        context["missing_side"]    = int("side" in str(missing))
        context["missing_drink"]   = int("drink" in str(missing))
        context["missing_dessert"] = int("dessert" in str(missing))

        # User features
        user_feats = user_dict.get(customer_id, {
            "user_order_count": 0, "user_avg_order_value": 0,
            "user_avg_items": 0, "user_diversity": 0,
            "user_weekend_ratio": 0, "user_single_item_ratio": 0,
            "user_is_cold_start": 1,
        })

        # Cart category composition
        cart_cats = [item_attr_dict.get(it, {}).get("category", "unknown") for it in cart_items]
        context["cart_has_main"]    = int("main" in cart_cats or "combo" in cart_cats)
        context["cart_has_side"]    = int("side" in cart_cats or "snack" in cart_cats)
        context["cart_has_drink"]   = int("drink" in cart_cats)
        context["cart_has_dessert"] = int("dessert" in cart_cats)

        # ── Helper to build candidate features ──────────────────────────
        def _make_row(candidate_item: str, label: int) -> dict:
            attr = item_attr_dict.get(candidate_item, {})
            stat = item_stat_dict.get(candidate_item, {})
            co_feats = _copurchase_features(cart_items, candidate_item, co_lookup)

            row = {**context, **user_feats}
            # Candidate features
            row["candidate_item"]       = candidate_item
            row["cand_category"]        = attr.get("category", "unknown")
            row["cand_veg_nonveg"]      = attr.get("veg_nonveg", "unknown")
            row["cand_cuisine"]         = attr.get("cuisine", "unknown")
            row["cand_typical_role"]    = attr.get("typical_role", "unknown")
            row["cand_flavor_profile"]  = attr.get("flavor_profile", "unknown")
            row["cand_popularity_rank"] = stat.get("popularity_rank", 999)
            row["cand_order_frequency"] = stat.get("order_frequency", 0)
            row["cand_solo_ratio"]      = stat.get("solo_order_ratio", 0)
            row["cand_avg_price"]       = stat.get("avg_price_proxy", 0)

            # Does candidate fill a missing slot?
            cand_cat = attr.get("category", "")
            row["fills_missing_slot"] = int(cand_cat in str(missing))

            # Veg compatibility
            order_veg = comp_info.get("order_veg_type", "unknown")
            cand_veg  = attr.get("veg_nonveg", "unknown")
            row["veg_compatible"] = int(
                order_veg == "unknown"
                or cand_veg == "unknown"
                or order_veg == cand_veg
                or (order_veg == "non-veg")  # non-veg eaters eat anything
            )

            # Co-purchase signals
            row["max_lift"]         = co_feats["max_lift"]
            row["avg_lift"]         = co_feats["avg_lift"]
            row["total_co_count"]   = co_feats["total_co_count"]
            row["max_confidence"]   = co_feats["max_confidence"]
            row["copurchase_pairs"] = co_feats["copurchase_pairs"]

            row["label"] = label
            return row

        # ── Positive sample ─────────────────────────────────────────────
        training_rows.append(_make_row(next_item, label=1))
        pos_count += 1

        # ── Negative samples ────────────────────────────────────────────
        # Pool: all items from same restaurant, excluding items already in cart + next_item
        exclude = set(cart_items) | {next_item}
        neg_pool = [it for it in restaurant_menu.get(restaurant, set()) if it not in exclude]

        # Sample up to NEG_SAMPLE_RATIO negatives
        if len(neg_pool) > NEG_SAMPLE_RATIO:
            rng = np.random.RandomState(hash(order_id) % (2**31))
            neg_sample = rng.choice(neg_pool, size=NEG_SAMPLE_RATIO, replace=False).tolist()
        else:
            neg_sample = neg_pool

        for neg_item in neg_sample:
            training_rows.append(_make_row(neg_item, label=0))
            neg_count += 1

    # ── Build final DataFrame ───────────────────────────────────────────
    print("[4/4] Saving training data …")
    train_df = pd.DataFrame(training_rows)

    # Shuffle
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df.to_csv(TRAINING_DATA_CSV, index=False)

    summary = {
        "total_samples":   len(train_df),
        "positive_samples": pos_count,
        "negative_samples": neg_count,
        "pos_neg_ratio":    f"1:{neg_count/pos_count:.1f}" if pos_count > 0 else "N/A",
        "feature_columns":  len(train_df.columns),
        "unique_orders":    train_df["order_id"].nunique(),
        "unique_candidates": train_df["candidate_item"].nunique(),
    }

    print(f"\n  Total samples:  {len(train_df):,}")
    print(f"  Positive:       {pos_count:,}")
    print(f"  Negative:       {neg_count:,}")
    print(f"  Ratio:          {summary['pos_neg_ratio']}")
    print(f"  Features:       {len(train_df.columns)} columns")
    print(f"\n  → Saved {TRAINING_DATA_CSV.name}")
    print("Phase 4 complete ✓\n")
    return summary


if __name__ == "__main__":
    run_phase4()
