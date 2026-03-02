"""
Phase 4.5 — Data Augmentation
==============================
Augments the training dataset to address two bottlenecks:
  1. Sparse multi-item signal (45% single-item orders produce zero positives)
  2. Thin user profiles (66.5% cold-start users, personalization signal too weak)

Three augmentation strategies:
  A) Item-Swap: replace items in real multi-item orders with same-category alternatives
  B) Collaborative Synthesis: extend light users' orders using heavy-user co-purchase patterns
  C) Soft-Label: generate hypothetical add-ons for single-item orders via co-purchase matrix

Every augmented row carries:
  • aug_type:    original | item_swap | collaborative | soft_label
  • sample_weight: 1.0 (original), 0.85 (item_swap), 0.75 (collaborative), 0.5 (soft_label)

Produces: augmented_training_data.csv
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from data_curation.config import (
    AUGMENTED_TRAINING_CSV,
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


# ─────────────────────────────────────────────────────────────────────────
#  Strategy A: Item-Swap Augmentation
# ─────────────────────────────────────────────────────────────────────────
def _build_category_pools(attrs: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """
    Build restaurant → category → [items] lookup.
    Used for swapping items within the same restaurant & category.
    """
    pools: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for _, row in attrs.iterrows():
        pools[row["restaurant"]][row["category"]].append(row["item"])
    return pools


def augment_item_swap(
    sequences: pd.DataFrame,
    attrs: pd.DataFrame,
    rng: np.random.RandomState,
    swap_ratio: float = 0.5,
    max_swaps_per_order: int = 2,
) -> list[dict]:
    """
    For each multi-item trainable stage, randomly swap one item in the cart
    with another from the same restaurant+category, keeping the next_item label.
    Also swap the next_item with a same-category alternative sometimes.

    Returns list of augmented stage dicts (same schema as cart_sequences rows).
    """
    cat_pools = _build_category_pools(attrs)
    item_cat = dict(zip(attrs["item"], attrs["category"]))
    item_rest = dict(zip(attrs["item"], attrs["restaurant"]))

    trainable = sequences[
        (sequences["is_final"] == 0)
        & (sequences["next_item_added"] != "")
        & (sequences["items_in_cart"] >= 1)
    ]

    augmented = []
    for _, stage in trainable.iterrows():
        if rng.random() > swap_ratio:
            continue

        cart_items = json.loads(stage["current_cart"])
        next_item = stage["next_item_added"]
        restaurant = stage["restaurant"]
        swaps_done = 0

        new_cart = list(cart_items)
        new_next = next_item

        # Try swapping a random cart item
        if len(new_cart) > 0 and swaps_done < max_swaps_per_order:
            idx = rng.randint(0, len(new_cart))
            old_item = new_cart[idx]
            old_cat = item_cat.get(old_item, "main")
            pool = [
                it for it in cat_pools.get(restaurant, {}).get(old_cat, [])
                if it != old_item and it != new_next
            ]
            if pool:
                new_cart[idx] = rng.choice(pool)
                swaps_done += 1

        # Sometimes swap the next_item too (30% chance)
        if rng.random() < 0.3 and swaps_done < max_swaps_per_order:
            next_cat = item_cat.get(next_item, "main")
            pool = [
                it for it in cat_pools.get(restaurant, {}).get(next_cat, [])
                if it != next_item and it not in new_cart
            ]
            if pool:
                new_next = rng.choice(pool)
                swaps_done += 1

        if swaps_done > 0:
            aug_stage = {
                "order_id":              stage["order_id"],
                "stage":                 stage["stage"],
                "current_cart":          json.dumps(new_cart),
                "next_item_added":       new_next,
                "next_item_category":    item_cat.get(new_next, ""),
                "next_item_role":        stage.get("next_item_role", ""),
                "cart_value_at_stage":   stage["cart_value_at_stage"],
                "items_in_cart":         len(new_cart),
                "completeness_at_stage": stage["completeness_at_stage"],
                "is_final":              0,
                "restaurant":            restaurant,
                "meal_period":           stage["meal_period"],
                "aug_type":              "item_swap",
            }
            augmented.append(aug_stage)

    return augmented


# ─────────────────────────────────────────────────────────────────────────
#  Strategy B: Collaborative Synthesis
# ─────────────────────────────────────────────────────────────────────────
def augment_collaborative(
    orders: pd.DataFrame,
    users: pd.DataFrame,
    co_df: pd.DataFrame,
    attrs: pd.DataFrame,
    rng: np.random.RandomState,
) -> list[dict]:
    """
    For light users (2-4 orders), synthesize plausible larger orders by:
    1. Taking their actual single-item orders
    2. Adding the top co-purchased item(s) from the co-purchase matrix
    3. Creating new cart stages from these synthetic orders

    Returns augmented stage dicts.
    """
    item_cat = dict(zip(attrs["item"], attrs["category"]))

    # Build co-purchase top-N per item
    top_copurchase: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for _, row in co_df.iterrows():
        top_copurchase[row["item_a"]].append((row["item_b"], row["lift"]))
        top_copurchase[row["item_b"]].append((row["item_a"], row["lift"]))
    # Sort by lift descending, keep top 5
    for item in top_copurchase:
        top_copurchase[item] = sorted(top_copurchase[item], key=lambda x: -x[1])[:5]

    # Find light & regular users
    target_users = set(
        users[users["order_count"].between(2, 10)]["Customer ID"]
    )

    # Get their single-item orders
    from data_curation.utils import parse_items
    single_orders = orders[
        (orders["Customer ID"].isin(target_users))
        & (orders["num_unique_items"] == 1)
    ]

    augmented = []
    for _, row in single_orders.iterrows():
        parsed = parse_items(row["Items in order"]) if pd.notna(row["Items in order"]) else []
        if not parsed:
            continue
        anchor = parsed[0]["item"]
        restaurant = row["Restaurant name"]

        # Get top co-purchases for this anchor
        candidates = top_copurchase.get(anchor, [])
        if not candidates:
            continue

        # Pick 1-2 add-ons (weighted random from top co-purchases)
        n_addons = min(rng.choice([1, 2], p=[0.6, 0.4]), len(candidates))
        chosen = []
        for cand_item, cand_lift in candidates[:n_addons + 2]:
            # Only add if different category from anchor and existing cart
            cand_cat = item_cat.get(cand_item, "unknown")
            anchor_cat = item_cat.get(anchor, "unknown")
            existing_cats = {anchor_cat} | {item_cat.get(c, "") for c in chosen}
            if cand_cat != anchor_cat and cand_cat not in existing_cats:
                chosen.append(cand_item)
            if len(chosen) >= n_addons:
                break

        if not chosen:
            continue

        # Create progressive cart stages
        full_order = [anchor] + chosen
        for stage_idx in range(len(full_order) - 1):
            cart = full_order[:stage_idx + 1]
            next_item = full_order[stage_idx + 1]
            augmented.append({
                "order_id":              f"syn_{row['Order ID']}_{stage_idx}",
                "stage":                 stage_idx + 1,
                "current_cart":          json.dumps(cart),
                "next_item_added":       next_item,
                "next_item_category":    item_cat.get(next_item, ""),
                "next_item_role":        "",
                "cart_value_at_stage":   row.get("Bill subtotal", 0) * (stage_idx + 1) / len(full_order),
                "items_in_cart":         len(cart),
                "completeness_at_stage": 0.0,  # will be recomputed
                "is_final":              0,
                "restaurant":            restaurant,
                "meal_period":           row.get("meal_period", "unknown"),
                "aug_type":              "collaborative",
                # Carry user ID for feature lookup
                "_customer_id":          row["Customer ID"],
                "_order_hour":           row.get("order_hour", -1),
                "_is_weekend":           row.get("is_weekend", 0),
            })

    return augmented


# ─────────────────────────────────────────────────────────────────────────
#  Strategy C: Soft-Label from Single-Item Orders
# ─────────────────────────────────────────────────────────────────────────
def augment_soft_label(
    orders: pd.DataFrame,
    co_df: pd.DataFrame,
    attrs: pd.DataFrame,
    rng: np.random.RandomState,
    sample_frac: float = 0.6,
) -> list[dict]:
    """
    For single-item orders, generate a 'what would we recommend?' hypothetical.
    Uses top co-purchase as the soft-positive, with label weight = 0.5.

    Returns augmented stage dicts.
    """
    item_cat = dict(zip(attrs["item"], attrs["category"]))

    # Build top-1 co-purchase per item
    best_addon: dict[str, tuple[str, float]] = {}
    for _, row in co_df.iterrows():
        for anchor, addon in [(row["item_a"], row["item_b"]), (row["item_b"], row["item_a"])]:
            if anchor not in best_addon or row["lift"] > best_addon[anchor][1]:
                # Prefer different category
                if item_cat.get(addon, "") != item_cat.get(anchor, ""):
                    best_addon[anchor] = (addon, row["lift"])

    from data_curation.utils import parse_items
    single_orders = orders[orders["num_unique_items"] == 1]

    # Sample a fraction
    if sample_frac < 1.0:
        single_orders = single_orders.sample(frac=sample_frac, random_state=rng)

    augmented = []
    for _, row in single_orders.iterrows():
        parsed = parse_items(row["Items in order"]) if pd.notna(row["Items in order"]) else []
        if not parsed:
            continue
        anchor = parsed[0]["item"]

        if anchor not in best_addon:
            continue

        addon_item, addon_lift = best_addon[anchor]
        augmented.append({
            "order_id":              f"soft_{row['Order ID']}",
            "stage":                 1,
            "current_cart":          json.dumps([anchor]),
            "next_item_added":       addon_item,
            "next_item_category":    item_cat.get(addon_item, ""),
            "next_item_role":        "",
            "cart_value_at_stage":   row.get("Bill subtotal", 0),
            "items_in_cart":         1,
            "completeness_at_stage": 0.0,
            "is_final":              0,
            "restaurant":            row.get("Restaurant name", ""),
            "meal_period":           row.get("meal_period", "unknown"),
            "aug_type":              "soft_label",
            "_customer_id":          row["Customer ID"],
            "_order_hour":           row.get("order_hour", -1),
            "_is_weekend":           row.get("is_weekend", 0),
        })

    return augmented


# ─────────────────────────────────────────────────────────────────────────
#  Training sample builder (reused from Phase 4 logic)
# ─────────────────────────────────────────────────────────────────────────
def _build_copurchase_lookup(co_df: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in co_df.iterrows():
        a, b = row["item_a"], row["item_b"]
        entry = {"lift": row["lift"], "co_count": row["co_count"],
                 "conf_a_b": row["confidence_a_given_b"], "conf_b_a": row["confidence_b_given_a"]}
        lookup[(a, b)] = entry
        lookup[(b, a)] = {"lift": entry["lift"], "co_count": entry["co_count"],
                          "conf_a_b": entry["conf_b_a"], "conf_b_a": entry["conf_a_b"]}
    return lookup


def _copurchase_features(cart_items: list, candidate: str, lookup: dict) -> dict:
    lifts, co_counts, confidences = [], [], []
    for ci in cart_items:
        key = (ci, candidate)
        if key in lookup:
            e = lookup[key]
            lifts.append(e["lift"]); co_counts.append(e["co_count"]); confidences.append(e["conf_a_b"])
    return {"max_lift": max(lifts) if lifts else 0.0, "avg_lift": np.mean(lifts) if lifts else 0.0,
            "total_co_count": sum(co_counts), "max_confidence": max(confidences) if confidences else 0.0,
            "copurchase_pairs": len(lifts)}


def _stage_to_training_rows(
    stage: dict,
    item_attr_dict: dict,
    item_stat_dict: dict,
    user_dict: dict,
    co_lookup: dict,
    comp_dict: dict,
    restaurant_menu: dict,
    order_user: dict,
    order_hour: dict,
    order_weekend: dict,
    rng: np.random.RandomState,
) -> list[dict]:
    """Convert a single augmented stage dict into training rows (1 positive + N negatives)."""
    order_id = stage["order_id"]
    cart_items = json.loads(stage["current_cart"])
    next_item = stage["next_item_added"]
    restaurant = stage["restaurant"]
    aug_type = stage.get("aug_type", "original")

    # Weight by augmentation type
    weight_map = {"original": 1.0, "item_swap": 0.85, "collaborative": 0.75, "soft_label": 0.5}
    sample_weight = weight_map.get(aug_type, 1.0)

    # Customer ID — augmented stages may carry it directly
    customer_id = stage.get("_customer_id", order_user.get(order_id, ""))

    # Context
    context = {
        "order_id":     order_id,
        "stage":        stage["stage"],
        "items_in_cart": stage["items_in_cart"],
        "cart_value":   stage["cart_value_at_stage"],
        "completeness": stage["completeness_at_stage"],
        "meal_period":  stage["meal_period"],
        "order_hour":   stage.get("_order_hour", order_hour.get(order_id, -1)),
        "is_weekend":   stage.get("_is_weekend", order_weekend.get(order_id, 0)),
        "restaurant":   restaurant,
    }

    # Missing categories (from completeness or infer from cart)
    comp_info = comp_dict.get(order_id, {})
    missing_str = comp_info.get("missing_categories", "")

    # For synthetic orders, compute missing from cart directly
    if aug_type in ("collaborative", "soft_label", "item_swap"):
        cart_cats = set(item_attr_dict.get(it, {}).get("category", "unknown") for it in cart_items)
        missing_parts = []
        if "main" not in cart_cats and "combo" not in cart_cats: missing_parts.append("main")
        if "side" not in cart_cats and "snack" not in cart_cats: missing_parts.append("side")
        if "drink" not in cart_cats: missing_parts.append("drink")
        if "dessert" not in cart_cats: missing_parts.append("dessert")
        missing_str = "|".join(missing_parts) if missing_parts else "none"

    context["missing_main"]    = int("main" in str(missing_str))
    context["missing_side"]    = int("side" in str(missing_str))
    context["missing_drink"]   = int("drink" in str(missing_str))
    context["missing_dessert"] = int("dessert" in str(missing_str))

    # Cart composition
    cart_cats = [item_attr_dict.get(it, {}).get("category", "unknown") for it in cart_items]
    context["cart_has_main"]    = int("main" in cart_cats or "combo" in cart_cats)
    context["cart_has_side"]    = int("side" in cart_cats or "snack" in cart_cats)
    context["cart_has_drink"]   = int("drink" in cart_cats)
    context["cart_has_dessert"] = int("dessert" in cart_cats)

    # User features
    user_feats = user_dict.get(customer_id, {
        "user_order_count": 0, "user_avg_order_value": 0,
        "user_avg_items": 0, "user_diversity": 0,
        "user_weekend_ratio": 0, "user_single_item_ratio": 0,
        "user_is_cold_start": 1,
    })

    def _make_row(candidate_item: str, label: float) -> dict:
        attr = item_attr_dict.get(candidate_item, {})
        stat = item_stat_dict.get(candidate_item, {})
        co_feats = _copurchase_features(cart_items, candidate_item, co_lookup)

        row = {**context, **user_feats}
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

        cand_cat = attr.get("category", "")
        row["fills_missing_slot"] = int(cand_cat in str(missing_str))

        order_veg = comp_info.get("order_veg_type", "unknown")
        cand_veg  = attr.get("veg_nonveg", "unknown")
        row["veg_compatible"] = int(
            order_veg == "unknown" or cand_veg == "unknown"
            or order_veg == cand_veg or order_veg == "non-veg"
        )

        row["max_lift"]         = co_feats["max_lift"]
        row["avg_lift"]         = co_feats["avg_lift"]
        row["total_co_count"]   = co_feats["total_co_count"]
        row["max_confidence"]   = co_feats["max_confidence"]
        row["copurchase_pairs"] = co_feats["copurchase_pairs"]
        row["label"]            = label
        row["aug_type"]         = aug_type
        row["sample_weight"]    = sample_weight if label > 0 else 1.0  # negatives always weight 1.0
        return row

    rows = []

    # Positive
    label_val = 1.0 if aug_type != "soft_label" else 0.5
    rows.append(_make_row(next_item, label_val))

    # Negatives
    exclude = set(cart_items) | {next_item}
    neg_pool = [it for it in restaurant_menu.get(restaurant, set()) if it not in exclude]
    n_neg = min(NEG_SAMPLE_RATIO, len(neg_pool))
    if n_neg > 0:
        chosen_negs = rng.choice(neg_pool, size=n_neg, replace=False).tolist()
        for neg in chosen_negs:
            rows.append(_make_row(neg, 0))

    return rows


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────
def run_phase4_5() -> dict:
    """Run all augmentation strategies and merge with original training data."""
    print("=" * 60)
    print("PHASE 4.5 — Data Augmentation")
    print("=" * 60)

    rng = np.random.RandomState(42)

    # Load artifacts
    print("[1/6] Loading artifacts …")
    orders     = pd.read_csv(ORDERS_ENRICHED_CSV)
    users      = pd.read_csv(USER_PROFILES_CSV)
    attrs      = pd.read_csv(ITEM_ATTRIBUTES_CSV)
    item_stats = pd.read_csv(ITEM_STATS_CSV)
    co_df      = pd.read_csv(COPURCHASE_CSV)
    comp       = pd.read_csv(COMPLETENESS_CSV)
    sequences  = pd.read_csv(CART_SEQUENCES_CSV)
    orig_train = pd.read_csv(TRAINING_DATA_CSV)

    # Add aug metadata to original
    orig_train["aug_type"] = "original"
    orig_train["sample_weight"] = 1.0

    # Build lookups
    print("[2/6] Building lookups …")
    item_attr_dict = {r["item"]: {k: r[k] for k in ["category","veg_nonveg","cuisine","typical_role","flavor_profile"]}
                      for _, r in attrs.iterrows()}
    item_stat_dict = {r["item"]: {k: r[k] for k in ["popularity_rank","order_frequency","solo_order_ratio","avg_price_proxy"]}
                      for _, r in item_stats.iterrows()}
    user_dict = {r["Customer ID"]: {
        "user_order_count": r["order_count"], "user_avg_order_value": r["avg_order_value"],
        "user_avg_items": r["avg_items_per_order"], "user_diversity": r["diversity_score"],
        "user_weekend_ratio": r["weekend_ratio"], "user_single_item_ratio": r["single_item_ratio"],
        "user_is_cold_start": r["is_cold_start"],
    } for _, r in users.iterrows()}

    order_user = dict(zip(orders["Order ID"], orders["Customer ID"]))
    order_hour = dict(zip(orders["Order ID"], orders.get("order_hour", pd.Series(dtype=float))))
    order_weekend = dict(zip(orders["Order ID"], orders.get("is_weekend", pd.Series(dtype=int))))

    restaurant_menu: dict[str, set[str]] = defaultdict(set)
    for _, r in attrs.iterrows(): restaurant_menu[r["restaurant"]].add(r["item"])
    for _, r in item_stats.iterrows(): restaurant_menu[r["restaurant"]].add(r["item"])

    co_lookup = _build_copurchase_lookup(co_df)
    comp_dict = {r["Order ID"]: {"missing_categories": r["missing_categories"], "order_veg_type": r["order_veg_type"]}
                 for _, r in comp.iterrows()}

    # ── Strategy A: Item-Swap ───────────────────────────────────────────
    print("[3/6] Strategy A — Item-Swap augmentation …")
    swap_stages = augment_item_swap(sequences, attrs, rng, swap_ratio=0.5)
    print(f"       Generated {len(swap_stages):,} swapped stages")

    swap_rows = []
    for s in swap_stages:
        swap_rows.extend(_stage_to_training_rows(
            s, item_attr_dict, item_stat_dict, user_dict, co_lookup,
            comp_dict, restaurant_menu, order_user, order_hour, order_weekend, rng
        ))
    swap_df = pd.DataFrame(swap_rows)
    print(f"       → {len(swap_df):,} training rows ({(swap_df['label']>0).sum():,} pos)")

    # ── Strategy B: Collaborative Synthesis ─────────────────────────────
    print("[4/6] Strategy B — Collaborative synthesis …")
    collab_stages = augment_collaborative(orders, users, co_df, attrs, rng)
    print(f"       Generated {len(collab_stages):,} collaborative stages")

    collab_rows = []
    for s in collab_stages:
        collab_rows.extend(_stage_to_training_rows(
            s, item_attr_dict, item_stat_dict, user_dict, co_lookup,
            comp_dict, restaurant_menu, order_user, order_hour, order_weekend, rng
        ))
    collab_df = pd.DataFrame(collab_rows)
    print(f"       → {len(collab_df):,} training rows ({(collab_df['label']>0).sum():,} pos)")

    # ── Strategy C: Soft-Label ──────────────────────────────────────────
    print("[5/6] Strategy C — Soft-label for single-item orders …")
    soft_stages = augment_soft_label(orders, co_df, attrs, rng, sample_frac=0.6)
    print(f"       Generated {len(soft_stages):,} soft-label stages")

    soft_rows = []
    for s in soft_stages:
        soft_rows.extend(_stage_to_training_rows(
            s, item_attr_dict, item_stat_dict, user_dict, co_lookup,
            comp_dict, restaurant_menu, order_user, order_hour, order_weekend, rng
        ))
    soft_df = pd.DataFrame(soft_rows)
    print(f"       → {len(soft_df):,} training rows ({(soft_df['label']>0).sum():,} soft-pos)")

    # ── Merge all ───────────────────────────────────────────────────────
    print("[6/6] Merging & saving …")
    all_dfs = [orig_train]
    if len(swap_df) > 0:
        all_dfs.append(swap_df)
    if len(collab_df) > 0:
        all_dfs.append(collab_df)
    if len(soft_df) > 0:
        all_dfs.append(soft_df)

    merged = pd.concat(all_dfs, ignore_index=True)

    # Shuffle
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    merged.to_csv(AUGMENTED_TRAINING_CSV, index=False)

    # Summary
    aug_counts = merged["aug_type"].value_counts().to_dict()
    pos_by_type = merged[merged["label"] > 0].groupby("aug_type").size().to_dict()

    summary = {
        "total_samples": len(merged),
        "original_samples": aug_counts.get("original", 0),
        "item_swap_samples": aug_counts.get("item_swap", 0),
        "collaborative_samples": aug_counts.get("collaborative", 0),
        "soft_label_samples": aug_counts.get("soft_label", 0),
        "positives_by_type": pos_by_type,
        "total_positives": int((merged["label"] > 0).sum()),
        "total_negatives": int((merged["label"] == 0).sum()),
        "augmentation_multiplier": round(len(merged) / len(orig_train), 2),
    }

    print(f"\n  Original:      {aug_counts.get('original', 0):>8,}")
    print(f"  Item-swap:     {aug_counts.get('item_swap', 0):>8,}")
    print(f"  Collaborative: {aug_counts.get('collaborative', 0):>8,}")
    print(f"  Soft-label:    {aug_counts.get('soft_label', 0):>8,}")
    print(f"  ─────────────────────────")
    print(f"  Total:         {len(merged):>8,}  ({summary['augmentation_multiplier']}x)")
    print(f"  Total pos:     {summary['total_positives']:>8,}")
    print(f"\n  → Saved {AUGMENTED_TRAINING_CSV.name}")
    print("Phase 4.5 complete ✓\n")
    return summary


if __name__ == "__main__":
    run_phase4_5()
