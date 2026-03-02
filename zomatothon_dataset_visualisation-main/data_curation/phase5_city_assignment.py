"""
Phase 5 — City Assignment & Geographic Enrichment
===================================================
Assigns users to 6 Indian cities based on behavioral clustering,
then builds inverse mappings (city → popular items, city → cuisine affinity).

Steps:
  1. Define city profiles (food culture priors)
  2. Score each user against city profiles → assign best match
  3. Apply subtle post-assignment adjustments (peak hours, price, distance)
  4. Build inverse mappings: city_item_popularity, city_cuisine_affinity

Produces:
  • city_profiles.csv          — 6 cities with behavioral parameters
  • user_city_assignment.csv   — user → city + confidence
  • city_item_popularity.csv   — per-city item rankings + local favorites + lift
  • city_cuisine_affinity.csv  — per-city cuisine distribution
  • orders_with_city.csv       — orders with city column + adjusted features
"""
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from data_curation.config import (
    CITY_CUISINE_AFFINITY_CSV,
    CITY_ITEM_POPULARITY_CSV,
    CITY_PROFILES_CSV,
    ITEM_ATTRIBUTES_CSV,
    ORDERS_ENRICHED_CSV,
    ORDERS_WITH_CITY_CSV,
    USER_CITY_CSV,
    USER_PROFILES_CSV,
)
from data_curation.utils import parse_items


# ─────────────────────────────────────────────────────────────────────────
#  City profiles: real-world food culture priors
# ─────────────────────────────────────────────────────────────────────────
CITY_PROFILES = {
    "Delhi": {
        "cuisine_weights":    {"North Indian": 0.35, "Pan-Indian": 0.30, "Mughlai": 0.15, "Street Food": 0.10, "Continental": 0.10},
        "veg_ratio":          0.40,    # ~40% veg orders
        "peak_meal":          "dinner",
        "late_night_affinity": 0.30,
        "avg_order_value_z":  0.2,     # slightly above avg
        "cart_size_z":        0.0,     # average
        "discount_affinity":  0.0,     # average
        "weekend_boost":      0.0,     # average
        "description":        "North Indian dominant, strong late-night culture, moderate-high spend",
    },
    "Mumbai": {
        "cuisine_weights":    {"Pan-Indian": 0.30, "Continental": 0.25, "Street Food": 0.15, "Italian": 0.15, "North Indian": 0.15},
        "veg_ratio":          0.45,
        "peak_meal":          "late_night",
        "late_night_affinity": 0.40,
        "avg_order_value_z":  0.5,     # highest spend
        "cart_size_z":        0.3,     # bigger carts
        "discount_affinity":  -0.2,    # less discount driven
        "weekend_boost":      0.2,     # strong weekend
        "description":        "Most diverse cuisine, highest spend, strong late-night + weekend spike",
    },
    "Bangalore": {
        "cuisine_weights":    {"Continental": 0.30, "Pan-Indian": 0.25, "Italian": 0.20, "North Indian": 0.15, "American": 0.10},
        "veg_ratio":          0.42,
        "peak_meal":          "lunch",
        "late_night_affinity": 0.15,
        "avg_order_value_z":  0.1,
        "cart_size_z":        0.1,
        "discount_affinity":  0.0,
        "weekend_boost":      0.1,
        "description":        "Tech crowd, pizza/continental heavy, strong lunch ordering",
    },
    "Hyderabad": {
        "cuisine_weights":    {"North Indian": 0.40, "Pan-Indian": 0.25, "Mughlai": 0.15, "Continental": 0.10, "Street Food": 0.10},
        "veg_ratio":          0.30,    # most non-veg
        "peak_meal":          "dinner",
        "late_night_affinity": 0.20,
        "avg_order_value_z":  -0.2,    # price sensitive
        "cart_size_z":        -0.3,    # smaller carts (biryani alone is a meal)
        "discount_affinity":  0.1,
        "weekend_boost":      0.0,
        "description":        "Biryani capital, non-veg dominant, single-item large portions",
    },
    "Pune": {
        "cuisine_weights":    {"Street Food": 0.30, "Pan-Indian": 0.25, "Continental": 0.15, "North Indian": 0.15, "American": 0.15},
        "veg_ratio":          0.45,
        "peak_meal":          "snack",
        "late_night_affinity": 0.15,
        "avg_order_value_z":  -0.4,    # most budget conscious
        "cart_size_z":        -0.2,
        "discount_affinity":  0.4,     # highest discount usage
        "weekend_boost":      -0.1,
        "description":        "Young/student city, budget-conscious, street food + quick bites",
    },
    "Kolkata": {
        "cuisine_weights":    {"Street Food": 0.25, "Pan-Indian": 0.25, "Continental": 0.20, "North Indian": 0.15, "American": 0.15},
        "veg_ratio":          0.35,
        "peak_meal":          "snack",
        "late_night_affinity": 0.20,
        "avg_order_value_z":  -0.1,
        "cart_size_z":        0.0,
        "discount_affinity":  0.15,
        "weekend_boost":      0.1,
        "description":        "Street food + Chinese crossover, evening snack culture",
    },
}


# ─────────────────────────────────────────────────────────────────────────
#  Step 1: Score users against city profiles
# ─────────────────────────────────────────────────────────────────────────
def _score_user_city(
    user_row: pd.Series,
    user_cuisine_dist: dict[str, float],
    city: str,
    profile: dict,
    global_stats: dict,
) -> float:
    """Score how well a user's behavior matches a city profile. Higher = better fit."""
    score = 0.0

    # 1. Cuisine similarity (cosine-like)
    cuisine_score = 0.0
    city_cuisines = profile["cuisine_weights"]
    for cuisine, city_w in city_cuisines.items():
        user_w = user_cuisine_dist.get(cuisine, 0.0)
        cuisine_score += city_w * user_w
    score += cuisine_score * 3.0  # heavily weight cuisine match

    # 2. Meal period match
    user_peak = user_row.get("preferred_meal_period", "dinner")
    if user_peak == profile["peak_meal"]:
        score += 1.5
    elif profile["peak_meal"] == "late_night" and user_row.get("weekend_ratio", 0) > 0.4:
        score += 0.5  # partial credit for late-night affinity

    # 3. Spending pattern
    user_aov = user_row.get("avg_order_value", 0)
    global_aov = global_stats["avg_order_value"]
    global_std = global_stats["std_order_value"]
    if global_std > 0:
        user_z = (user_aov - global_aov) / global_std
        score -= abs(user_z - profile["avg_order_value_z"]) * 0.5

    # 4. Cart size pattern
    user_cart = user_row.get("avg_items_per_order", 0)
    global_cart = global_stats["avg_cart_size"]
    global_cart_std = global_stats["std_cart_size"]
    if global_cart_std > 0:
        user_cart_z = (user_cart - global_cart) / global_cart_std
        score -= abs(user_cart_z - profile["cart_size_z"]) * 0.3

    # 5. Discount affinity
    user_disc = user_row.get("avg_discount_pct", 0)
    global_disc = global_stats["avg_discount_pct"]
    global_disc_std = global_stats["std_discount_pct"]
    if global_disc_std > 0:
        user_disc_z = (user_disc - global_disc) / global_disc_std
        score -= abs(user_disc_z - profile["discount_affinity"]) * 0.3

    # 6. Single-item ratio (Hyderabad signal)
    user_single = user_row.get("single_item_ratio", 0)
    if profile["cart_size_z"] < -0.2:
        score += user_single * 0.5  # high single-item → Hyderabad-like

    # 7. Weekend ratio
    user_wknd = user_row.get("weekend_ratio", 0)
    if user_wknd > 0.4 and profile["weekend_boost"] > 0:
        score += 0.3

    return score


def _get_user_cuisine_dist(
    user_id: str,
    orders: pd.DataFrame,
    item_attr: dict[str, str],
) -> dict[str, float]:
    """Get a user's cuisine distribution from their orders."""
    user_orders = orders[orders["Customer ID"] == user_id]
    cuisines = Counter()
    total = 0
    for _, row in user_orders.iterrows():
        parsed = parse_items(row["Items in order"]) if pd.notna(row["Items in order"]) else []
        for p in parsed:
            c = item_attr.get(p["item"], "Pan-Indian")
            cuisines[c] += 1
            total += 1
    if total == 0:
        return {}
    return {c: cnt / total for c, cnt in cuisines.items()}


# ─────────────────────────────────────────────────────────────────────────
#  Step 2: Post-assignment adjustments
# ─────────────────────────────────────────────────────────────────────────
def _adjust_orders(orders: pd.DataFrame, user_city: dict[str, str], rng: np.random.RandomState) -> pd.DataFrame:
    """Apply subtle city-specific perturbations to make distributions diverge realistically."""
    df = orders.copy()
    df["city"] = df["Customer ID"].map(user_city).fillna("Delhi")

    # Hour shifts by city (in minutes)
    hour_shift = {
        "Delhi": 0, "Mumbai": 30, "Bangalore": -60,
        "Hyderabad": -15, "Pune": -30, "Kolkata": 15,
    }

    # Price multipliers
    price_mult = {
        "Delhi": 1.0, "Mumbai": 1.08, "Bangalore": 1.02,
        "Hyderabad": 0.92, "Pune": 0.88, "Kolkata": 0.95,
    }

    # Discount inflation
    disc_adjust = {
        "Delhi": 0, "Mumbai": -2, "Bangalore": 0,
        "Hyderabad": 1, "Pune": 5, "Kolkata": 2,
    }

    # Distance multiplier
    dist_mult = {
        "Delhi": 1.0, "Mumbai": 1.15, "Bangalore": 1.10,
        "Hyderabad": 0.90, "Pune": 0.85, "Kolkata": 0.95,
    }

    for city in hour_shift:
        mask = df["city"] == city

        # Hour shift (add noise ±15 min on top)
        if "order_datetime" in df.columns:
            shift_mins = hour_shift[city] + rng.randint(-15, 16, size=mask.sum())
            td = pd.to_timedelta(shift_mins, unit="m")
            datetimes = pd.to_datetime(df.loc[mask, "order_datetime"], errors="coerce")
            df.loc[mask, "order_datetime"] = (datetimes + td).astype(str)
            # Update order_hour
            new_hours = (datetimes + td).dt.hour
            df.loc[mask, "order_hour"] = new_hours.values

        # Price (subtle ±5% noise on top)
        noise = 1 + rng.normal(0, 0.03, size=mask.sum())
        mult = price_mult[city] * noise
        for col in ["Bill subtotal", "Total"]:
            if col in df.columns:
                df.loc[mask, col] = (df.loc[mask, col] * mult).round(2)

        # Discount
        if "discount_pct" in df.columns:
            adj = disc_adjust[city] + rng.normal(0, 1, size=mask.sum())
            df.loc[mask, "discount_pct"] = (df.loc[mask, "discount_pct"] + adj).clip(0, 100).round(1)

        # Distance
        if "distance_km" in df.columns:
            d_noise = 1 + rng.normal(0, 0.05, size=mask.sum())
            df.loc[mask, "distance_km"] = (df.loc[mask, "distance_km"] * dist_mult[city] * d_noise).round(1)

    return df


# ─────────────────────────────────────────────────────────────────────────
#  Step 3: Inverse mappings
# ─────────────────────────────────────────────────────────────────────────
def _build_city_item_popularity(
    orders_city: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-city item rankings with local-favorite detection."""
    # Count items per city
    city_item_counts: dict[str, Counter] = defaultdict(Counter)
    national_counts = Counter()

    for _, row in orders_city.iterrows():
        city = row["city"]
        parsed = parse_items(row["Items in order"]) if pd.notna(row["Items in order"]) else []
        for p in parsed:
            city_item_counts[city][p["item"]] += 1
            national_counts[p["item"]] += 1

    total_national = sum(national_counts.values())

    rows = []
    for city, item_counter in city_item_counts.items():
        total_city = sum(item_counter.values())
        for item, count in item_counter.most_common():
            city_share = count / total_city if total_city > 0 else 0
            national_share = national_counts[item] / total_national if total_national > 0 else 0
            city_lift = city_share / national_share if national_share > 0 else 0

            rows.append({
                "city":             city,
                "item":             item,
                "city_order_count": count,
                "city_share":       round(city_share, 4),
                "national_share":   round(national_share, 4),
                "city_lift":        round(city_lift, 3),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Add city rank
    df["city_rank"] = df.groupby("city")["city_order_count"].rank(ascending=False, method="min").astype(int)

    # National rank
    nat_rank = df.drop_duplicates("item").set_index("item")
    nat_rank["national_rank"] = nat_rank["national_share"].rank(ascending=False, method="min").astype(int)
    df = df.merge(nat_rank[["national_rank"]], left_on="item", right_index=True, how="left")

    # Local favorite: city_rank significantly better than national_rank
    df["is_local_favorite"] = (
        (df["city_lift"] > 1.3) & (df["city_rank"] <= 20)
    ).astype(int)

    return df.sort_values(["city", "city_rank"]).reset_index(drop=True)


def _build_city_cuisine_affinity(
    orders_city: pd.DataFrame,
    item_cuisine: dict[str, str],
) -> pd.DataFrame:
    """Build per-city cuisine distribution."""
    city_cuisine_counts: dict[str, Counter] = defaultdict(Counter)

    for _, row in orders_city.iterrows():
        city = row["city"]
        parsed = parse_items(row["Items in order"]) if pd.notna(row["Items in order"]) else []
        for p in parsed:
            cuisine = item_cuisine.get(p["item"], "Pan-Indian")
            city_cuisine_counts[city][cuisine] += 1

    rows = []
    for city, counter in city_cuisine_counts.items():
        total = sum(counter.values())
        for cuisine, count in counter.most_common():
            rows.append({
                "city":    city,
                "cuisine": cuisine,
                "count":   count,
                "share":   round(count / total, 4) if total > 0 else 0,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["rank"] = df.groupby("city")["count"].rank(ascending=False, method="min").astype(int)
    return df.sort_values(["city", "rank"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────
def run_phase5() -> dict:
    """Assign cities, build inverse mappings. Returns summary dict."""
    print("=" * 60)
    print("PHASE 5 — City Assignment & Geographic Enrichment")
    print("=" * 60)

    rng = np.random.RandomState(123)

    # Load
    print("[1/6] Loading data …")
    orders = pd.read_csv(ORDERS_ENRICHED_CSV)
    users  = pd.read_csv(USER_PROFILES_CSV)
    attrs  = pd.read_csv(ITEM_ATTRIBUTES_CSV)

    item_cuisine = dict(zip(attrs["item"], attrs["cuisine"]))

    # Global stats for z-scoring
    global_stats = {
        "avg_order_value": users["avg_order_value"].mean(),
        "std_order_value": users["avg_order_value"].std(),
        "avg_cart_size":   users["avg_items_per_order"].mean(),
        "std_cart_size":   users["avg_items_per_order"].std(),
        "avg_discount_pct": users["avg_discount_pct"].mean(),
        "std_discount_pct": users["avg_discount_pct"].std(),
    }

    # ── Save city profiles ──────────────────────────────────────────────
    print("[2/6] Saving city profiles …")
    cp_rows = []
    for city, p in CITY_PROFILES.items():
        cp_rows.append({
            "city": city,
            "peak_meal": p["peak_meal"],
            "late_night_affinity": p["late_night_affinity"],
            "avg_order_value_z": p["avg_order_value_z"],
            "cart_size_z": p["cart_size_z"],
            "discount_affinity": p["discount_affinity"],
            "veg_ratio": p["veg_ratio"],
            "weekend_boost": p["weekend_boost"],
            "top_cuisines": json.dumps(p["cuisine_weights"]),
            "description": p["description"],
        })
    pd.DataFrame(cp_rows).to_csv(CITY_PROFILES_CSV, index=False)

    # ── Score & assign ──────────────────────────────────────────────────
    print("[3/6] Scoring users against city profiles …")
    cities = list(CITY_PROFILES.keys())

    # Pre-compute cuisine distributions for all users
    # (batch: group orders by customer, parse items, tally cuisines)
    user_cuisine_dists: dict[str, dict[str, float]] = {}
    for cid, grp in orders.groupby("Customer ID"):
        cuisines = Counter()
        total = 0
        for items_str in grp["Items in order"].dropna():
            for p in parse_items(items_str):
                c = item_cuisine.get(p["item"], "Pan-Indian")
                cuisines[c] += 1
                total += 1
        user_cuisine_dists[cid] = {c: cnt / total for c, cnt in cuisines.items()} if total > 0 else {}

    assignments = []
    for _, user_row in users.iterrows():
        cid = user_row["Customer ID"]
        cuisine_dist = user_cuisine_dists.get(cid, {})

        scores = {}
        for city in cities:
            scores[city] = _score_user_city(
                user_row, cuisine_dist, city, CITY_PROFILES[city], global_stats
            )

        # Normalize to 0-1 range for confidence
        max_score = max(scores.values())
        min_score = min(scores.values())
        score_range = max_score - min_score if max_score != min_score else 1.0

        best_city = max(scores, key=scores.get)
        confidence = (scores[best_city] - min_score) / score_range

        # Add stochastic noise for cold-start users to prevent all going to one city
        if user_row["is_cold_start"] == 1 and confidence < 0.5:
            # Softmax-weighted random selection
            score_arr = np.array([scores[c] for c in cities])
            # Temperature-based softmax
            temp = 1.5
            exp_scores = np.exp((score_arr - score_arr.max()) / temp)
            probs = exp_scores / exp_scores.sum()
            best_city = rng.choice(cities, p=probs)
            confidence = scores[best_city] / max_score if max_score > 0 else 0

        assignments.append({
            "Customer ID": cid,
            "city":        best_city,
            "confidence":  round(confidence, 3),
            "order_count": user_row["order_count"],
            **{f"score_{c}": round(scores[c], 3) for c in cities},
        })

    assign_df = pd.DataFrame(assignments)
    assign_df.to_csv(USER_CITY_CSV, index=False)

    user_city_map = dict(zip(assign_df["Customer ID"], assign_df["city"]))

    city_dist = assign_df["city"].value_counts()
    print(f"       City distribution:")
    for c in cities:
        n = city_dist.get(c, 0)
        print(f"         {c:12s}: {n:>5,} users ({n/len(assign_df)*100:.1f}%)")

    # ── Apply adjustments ───────────────────────────────────────────────
    print("[4/6] Applying city-specific adjustments …")
    orders_city = _adjust_orders(orders, user_city_map, rng)
    orders_city.to_csv(ORDERS_WITH_CITY_CSV, index=False)
    print(f"       → Saved {ORDERS_WITH_CITY_CSV.name}")

    # ── Inverse mapping: city → items ───────────────────────────────────
    print("[5/6] Building city → item popularity …")
    city_items = _build_city_item_popularity(orders_city)
    city_items.to_csv(CITY_ITEM_POPULARITY_CSV, index=False)
    n_favorites = city_items["is_local_favorite"].sum()
    print(f"       → {len(city_items):,} city-item pairs, {n_favorites} local favorites")

    # ── Inverse mapping: city → cuisine ─────────────────────────────────
    print("[6/6] Building city → cuisine affinity …")
    city_cuisine = _build_city_cuisine_affinity(orders_city, item_cuisine)
    city_cuisine.to_csv(CITY_CUISINE_AFFINITY_CSV, index=False)
    print(f"       → {len(city_cuisine):,} city-cuisine pairs")

    # ── Summary ─────────────────────────────────────────────────────────
    summary = {
        "cities":             len(cities),
        "users_assigned":     len(assign_df),
        "city_distribution":  city_dist.to_dict(),
        "local_favorites":    int(n_favorites),
        "city_item_pairs":    len(city_items),
        "city_cuisine_pairs": len(city_cuisine),
        "avg_confidence":     round(assign_df["confidence"].mean(), 3),
    }

    print(f"\n  Avg confidence: {summary['avg_confidence']:.3f}")
    print(f"  Local favorites found: {n_favorites}")
    print(f"\n  → Saved {USER_CITY_CSV.name}, {CITY_ITEM_POPULARITY_CSV.name}, {CITY_CUISINE_AFFINITY_CSV.name}")
    print("Phase 5 complete ✓\n")
    return summary


if __name__ == "__main__":
    run_phase5()
