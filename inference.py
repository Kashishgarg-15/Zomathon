# -*- coding: utf-8 -*-
"""
CSAO Add-On Recommendation — Inference Pipeline
=================================================
Takes a user's current cart and returns ranked add-on recommendations.

Usage:
    python inference.py                          # interactive demo
    python inference.py --json '{"restaurant": "Swaad", "cart_items": ["Chicken Biryani"], ...}'

Requires:
    - model_output_v3/lgb_model.txt, xgb_model.json, label_encoders.pkl
    - llm_artifacts/embedding_artifacts.pkl
    - city_item_popularity.csv, city_cuisine_affinity.csv
    - Lookup tables built from training_data_llm.csv (auto-generated on first run)
"""
import os, sys, json, time, warnings, argparse, pickle
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib

# ================================================================
# PATHS
# ================================================================
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model_output_v3")
LLM_DIR   = os.path.join(BASE, "llm_artifacts")
LOOKUP_PATH = os.path.join(BASE, "inference_lookups.pkl")

# Ensemble weights (from training_results.json)
LGB_W = 0.75
XGB_W = 0.25

# ================================================================
# 1. LOAD MODELS + ARTIFACTS (once at startup)
# ================================================================
print("Loading models and artifacts...", flush=True)
t0 = time.time()

# LightGBM
lgb_model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgb_model.txt"))

# XGBoost
xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(MODEL_DIR, "xgb_model.json"))

# Label encoders
encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))

# LLM embedding artifacts
llm_arts = joblib.load(os.path.join(LLM_DIR, "embedding_artifacts.pkl"))
item_to_pca    = llm_arts['item_to_pca']
item_to_cluster = llm_arts['item_to_cluster']
ctx_to_emb     = llm_arts['ctx_to_emb']
flavor_to_emb  = llm_arts['flavor_to_emb']
item_to_emb    = llm_arts['item_to_emb']

# City lookup files
city_items_df   = pd.read_csv(os.path.join(BASE, "city_item_popularity.csv"))
city_cuisine_df = pd.read_csv(os.path.join(BASE, "city_cuisine_affinity.csv"))

# ================================================================
# 2. BUILD LOOKUP TABLES (item attrs, restaurant menus, copurchase)
# ================================================================
def build_lookups():
    """Build lookup tables from training data (run once, cache to disk)."""
    if os.path.exists(LOOKUP_PATH):
        return joblib.load(LOOKUP_PATH)
    
    print("  Building lookup tables from training data (first run only)...", flush=True)
    df = pd.read_csv(os.path.join(BASE, "training_data_llm.csv"))
    
    # Item attributes lookup
    item_cols = ['candidate_item', 'cand_category', 'cand_veg_nonveg', 'cand_cuisine',
                 'cand_typical_role', 'cand_flavor_profile', 'cand_popularity_rank',
                 'cand_order_frequency', 'cand_solo_ratio', 'cand_avg_price']
    item_lookup = df.drop_duplicates('candidate_item')[item_cols].set_index('candidate_item').to_dict('index')
    
    # Restaurant → list of menu items
    rest_menus = df.groupby('restaurant')['candidate_item'].apply(lambda x: list(x.unique())).to_dict()
    
    # Co-purchase stats: for each (item_A, item_B) pair, store lift/confidence
    # Build from training data aggregatedsignals
    copurchase = {}
    for _, row in df.iterrows():
        key = row['candidate_item']
        if key not in copurchase:
            copurchase[key] = {
                'max_lift': row['max_lift'],
                'total_co_count': row['total_co_count'],
                'max_confidence': row['max_confidence'],
                'copurchase_pairs': row['copurchase_pairs'],
            }
    
    # Item category lookup (for completeness computation)
    item_cats = {row['candidate_item']: row['cand_category'] 
                 for _, row in df.drop_duplicates('candidate_item').iterrows()}
    
    # Max order frequency (for cold-start signal normalization)
    max_order_freq = df['cand_order_frequency'].max()
    
    lookups = {
        'item_lookup': item_lookup,
        'rest_menus': rest_menus,
        'copurchase': copurchase,
        'item_cats': item_cats,
        'max_order_freq': max_order_freq,
    }
    joblib.dump(lookups, LOOKUP_PATH)
    print(f"  Saved lookups to {LOOKUP_PATH}", flush=True)
    return lookups

lookups = build_lookups()
item_lookup    = lookups['item_lookup']
rest_menus     = lookups['rest_menus']
copurchase_map = lookups['copurchase']
item_cats      = lookups['item_cats']
max_order_freq = lookups['max_order_freq']

print(f"Loaded in {time.time()-t0:.1f}s. {len(item_lookup)} items, {len(rest_menus)} restaurants.", flush=True)

# ================================================================
# 3. FEATURE ENGINEERING FUNCTIONS
# ================================================================

# Meal completeness weights
SLOT_WEIGHTS = {'main': 0.40, 'side': 0.25, 'drink': 0.20, 'dessert': 0.15}

def compute_completeness(cart_items):
    """Compute meal completeness score (0-1) from cart items."""
    cats_present = set()
    for item in cart_items:
        cat = item_cats.get(item, 'main')
        if cat in ('main', 'combo'):
            cats_present.add('main')
        elif cat in ('side', 'snack'):
            cats_present.add('side')
        elif cat == 'drink':
            cats_present.add('drink')
        elif cat == 'dessert':
            cats_present.add('dessert')
    return sum(SLOT_WEIGHTS.get(s, 0) for s in cats_present)

def get_cart_has_flags(cart_items):
    """Return cart_has_main/side/drink/dessert flags."""
    cats_present = set()
    for item in cart_items:
        cat = item_cats.get(item, 'main')
        if cat in ('main', 'combo'):
            cats_present.add('main')
        elif cat in ('side', 'snack'):
            cats_present.add('side')
        elif cat == 'drink':
            cats_present.add('drink')
        elif cat == 'dessert':
            cats_present.add('dessert')
    return {
        'cart_has_main': int('main' in cats_present),
        'cart_has_side': int('side' in cats_present),
        'cart_has_drink': int('drink' in cats_present),
        'cart_has_dessert': int('dessert' in cats_present),
    }

def compute_cart_value(cart_items):
    """Estimate cart value from item avg prices."""
    return sum(item_lookup.get(it, {}).get('cand_avg_price', 200) for it in cart_items)

def fills_missing_slot_check(candidate_cat, cart_flags):
    """Check if candidate fills a missing meal slot."""
    if candidate_cat in ('main', 'combo') and cart_flags['cart_has_main'] == 0:
        return 1
    if candidate_cat in ('side', 'snack') and cart_flags['cart_has_side'] == 0:
        return 1
    if candidate_cat == 'drink' and cart_flags['cart_has_drink'] == 0:
        return 1
    if candidate_cat == 'dessert' and cart_flags['cart_has_dessert'] == 0:
        return 1
    return 0

def veg_compatible_check(candidate_veg, cart_items):
    """Check dietary compatibility."""
    cart_has_nonveg = any(
        item_lookup.get(it, {}).get('cand_veg_nonveg', 'veg') == 'non-veg'
        for it in cart_items
    )
    if cart_has_nonveg:
        return 1  # Non-veg cart accepts anything
    if candidate_veg == 'veg':
        return 1  # Veg item is always compatible
    return 0

def get_hour_from_meal_period(meal_period):
    """Estimate typical hour from meal period."""
    period_hours = {
        'breakfast': 8, 'lunch': 12, 'snack': 15,
        'dinner': 20, 'late_night': 23
    }
    return period_hours.get(meal_period, 19)

# LLM feature computation functions
def compute_llm_context_compatibility(item_name, meal_period, restaurant):
    """Cosine similarity between item embedding and meal context."""
    item_emb = item_to_emb.get(item_name)
    ctx_key = (meal_period, restaurant)
    ctx_emb = ctx_to_emb.get(ctx_key)
    if item_emb is None or ctx_emb is None:
        return 0.5
    from sklearn.preprocessing import normalize as sk_norm
    ie = item_emb.reshape(1, -1)
    ce = ctx_emb.reshape(1, -1)
    ie_n = sk_norm(ie)[0]
    ce_n = sk_norm(ce)[0]
    return float(np.dot(ie_n, ce_n))

def compute_llm_cuisine_cat_affinity(cuisine, category, meal_period, restaurant):
    """Cosine similarity between cuisine-category and meal context."""
    # Simplified: use context compatibility as proxy
    ctx_key = (meal_period, restaurant)
    ctx_emb = ctx_to_emb.get(ctx_key)
    if ctx_emb is None:
        return 0.5
    text = f"{cuisine} {category} food item"
    # Use precomputed flavor embeddings as approximate
    return 0.5  # Default; in production would embed on-the-fly

PERIOD_NEEDS = {
    'lunch': {'main': 1.0, 'side': 0.7, 'drink': 0.8, 'dessert': 0.3},
    'dinner': {'main': 1.0, 'side': 0.8, 'drink': 0.6, 'dessert': 0.5},
    'snack': {'main': 0.3, 'side': 0.8, 'drink': 0.9, 'dessert': 0.4},
    'breakfast': {'main': 1.0, 'side': 0.5, 'drink': 0.9, 'dessert': 0.2},
    'late_night': {'main': 0.7, 'side': 0.6, 'drink': 0.5, 'dessert': 0.4},
}

def compute_llm_meal_completion(cand_category, cart_flags, meal_period):
    """LLM-inspired meal completion score."""
    if cand_category == 'combo':
        return 0.5
    needs = PERIOD_NEEDS.get(meal_period, {'main': 0.7, 'side': 0.6, 'drink': 0.6, 'dessert': 0.4})
    desire = needs.get(cand_category, 0.5)
    has_map = {'main': cart_flags['cart_has_main'], 'side': cart_flags['cart_has_side'],
               'drink': cart_flags['cart_has_drink'], 'dessert': cart_flags['cart_has_dessert']}
    if cand_category in has_map and has_map[cand_category] == 0:
        desire *= 1.5
    elif cand_category in has_map and has_map[cand_category] == 1:
        desire *= 0.6
    return min(desire, 1.0)

PERIOD_FLAVOR = {'lunch': 'savory', 'dinner': 'rich', 'snack': 'spicy', 'breakfast': 'mild', 'late_night': 'rich'}
HARMONY_PAIRS = {
    ('spicy', 'mild'): 1.2, ('spicy', 'sweet'): 1.1, ('rich', 'tangy'): 1.1,
    ('savory', 'sweet'): 0.9, ('rich', 'rich'): 0.7, ('spicy', 'spicy'): 0.8,
}

def compute_llm_flavor_harmony(cand_flavor, meal_period):
    """Flavor harmony score."""
    cart_flavor = PERIOD_FLAVOR.get(meal_period, 'savory')
    pair = (cart_flavor, cand_flavor)
    pair_rev = (cand_flavor, cart_flavor)
    if pair in HARMONY_PAIRS:
        return HARMONY_PAIRS[pair]
    if pair_rev in HARMONY_PAIRS:
        return HARMONY_PAIRS[pair_rev]
    e1 = flavor_to_emb.get(cart_flavor, flavor_to_emb.get('savory'))
    e2 = flavor_to_emb.get(cand_flavor, flavor_to_emb.get('neutral'))
    if e1 is None or e2 is None:
        return 0.5
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10))

# Explanation generator
def generate_explanation(item_name, cand_category, cand_cuisine, meal_period,
                         fills_gap, flavor, restaurant, score):
    """Human-readable recommendation explanation."""
    reasons = []
    if fills_gap:
        reasons.append(f"completes your meal by adding a {cand_category}")
    if flavor in ('mild', 'tangy') and cand_category == 'drink':
        reasons.append(f"its {flavor} taste refreshes your palate")
    elif flavor == 'sweet':
        reasons.append("adds a sweet finish to your meal")
    elif flavor == 'spicy':
        reasons.append("adds an exciting kick to your order")
    if not reasons:
        reasons.append(f"it's a popular {cand_cuisine} {cand_category} that pairs well with your order")
    return f"We recommend {item_name} because {' and '.join(reasons)}."


# ================================================================
# 4. MAIN INFERENCE FUNCTION
# ================================================================

def recommend_addons(
    restaurant: str,
    cart_items: list,
    meal_period: str = "dinner",
    is_weekend: int = 0,
    user_order_count: int = 1,
    user_avg_order_value: float = 400.0,
    user_avg_items: float = 2.0,
    user_weekend_ratio: float = 0.3,
    user_single_item_ratio: float = 0.5,
    user_is_cold_start: int = 1,
    city: str = "Delhi",
    top_k: int = 5,
) -> list:
    """
    Recommend add-on items for a given cart.
    
    Args:
        restaurant: Restaurant name (must match training data)
        cart_items: List of item names currently in cart
        meal_period: One of 'breakfast', 'lunch', 'snack', 'dinner', 'late_night'
        is_weekend: 1 if weekend, 0 otherwise
        user_*: User profile features (use defaults for cold-start)
        city: User's city
        top_k: Number of recommendations to return
    
    Returns:
        List of dicts with: item, score, category, explanation, rank
    """
    t_start = time.time()
    
    # Get candidate items (restaurant menu minus cart items)
    menu = rest_menus.get(restaurant, [])
    if not menu:
        return [{"error": f"Unknown restaurant: {restaurant}"}]
    candidates = [item for item in menu if item not in cart_items]
    if not candidates:
        return [{"error": "No candidates available (all items already in cart)"}]
    
    # Compute cart-level features
    items_in_cart = len(cart_items)
    cart_value = compute_cart_value(cart_items)
    completeness = compute_completeness(cart_items)
    cart_flags = get_cart_has_flags(cart_items)
    order_hour = get_hour_from_meal_period(meal_period)
    
    # Build feature rows for all candidates
    rows = []
    for cand in candidates:
        attrs = item_lookup.get(cand, {})
        cand_cat = attrs.get('cand_category', 'main')
        cand_veg = attrs.get('cand_veg_nonveg', 'veg')
        cand_cuisine = attrs.get('cand_cuisine', 'Pan-Indian')
        cand_role = attrs.get('cand_typical_role', 'complement')
        cand_flavor = attrs.get('cand_flavor_profile', 'savory')
        cand_pop_rank = attrs.get('cand_popularity_rank', 100)
        cand_order_freq = attrs.get('cand_order_frequency', 10)
        cand_solo = attrs.get('cand_solo_ratio', 0.3)
        cand_price = attrs.get('cand_avg_price', 200)
        
        # Fills missing slot
        fills_gap = fills_missing_slot_check(cand_cat, cart_flags)
        
        # Veg compatible
        veg_compat = veg_compatible_check(cand_veg, cart_items)
        
        # Co-purchase signals (use candidate-level aggregates)
        cop = copurchase_map.get(cand, {})
        max_lift_val = cop.get('max_lift', 0)
        total_co = cop.get('total_co_count', 0)
        max_conf = cop.get('max_confidence', 0)
        cop_pairs = cop.get('copurchase_pairs', 0)
        
        # City features
        city_row = city_items_df[(city_items_df['city'] == city) & (city_items_df['item'] == cand)]
        if len(city_row) > 0:
            city_lift = city_row.iloc[0]['city_lift']
            city_rank_val = city_row.iloc[0]['city_rank']
            is_local_fav = int(city_row.iloc[0]['is_local_favorite'])
        else:
            city_lift = 0.0
            city_rank_val = 999
            is_local_fav = 0
        
        cuis_row = city_cuisine_df[(city_cuisine_df['city'] == city) & (city_cuisine_df['cuisine'] == cand_cuisine)]
        if len(cuis_row) > 0:
            cuisine_city_share = cuis_row.iloc[0]['share']
            cuisine_city_rank_val = cuis_row.iloc[0]['rank']
        else:
            cuisine_city_share = 0.0
            cuisine_city_rank_val = 99
        
        # LLM features: PCA embeddings
        pca_vec = item_to_pca.get(cand, np.zeros(16))
        cluster = item_to_cluster.get(cand, 0)
        
        # LLM computed features
        llm_ctx = compute_llm_context_compatibility(cand, meal_period, restaurant)
        llm_cca = compute_llm_cuisine_cat_affinity(cand_cuisine, cand_cat, meal_period, restaurant)
        llm_mc = compute_llm_meal_completion(cand_cat, cart_flags, meal_period)
        llm_fh = compute_llm_flavor_harmony(cand_flavor, meal_period)
        llm_csb = (0.7 * llm_ctx + 0.3 * llm_mc) if user_is_cold_start else (0.3 * llm_ctx + 0.1 * llm_mc)
        llm_cps = (cand_order_freq / (max_order_freq + 1)) if user_is_cold_start else 0.0
        
        # Engineered features
        hour_sin = np.sin(2 * np.pi * order_hour / 24)
        hour_cos = np.cos(2 * np.pi * order_hour / 24)
        price_ratio = cand_price / (cart_value + 1)
        pop_x_lift = cand_order_freq * max_lift_val
        comp_gap = 1.0 - completeness
        price_vs_user = cand_price / (user_avg_order_value + 1)
        complement_fills = int(cand_role != 'anchor' and fills_gap == 1)
        cart_bucket = min(items_in_cart, 3) if items_in_cart <= 3 else 3
        city_item_sig = city_lift * is_local_fav
        
        row = {
            'items_in_cart': items_in_cart,
            'cart_value': cart_value,
            'completeness': completeness,
            'meal_period': meal_period,
            'order_hour': order_hour,
            'is_weekend': is_weekend,
            'restaurant': restaurant,
            'cart_has_main': cart_flags['cart_has_main'],
            'cart_has_side': cart_flags['cart_has_side'],
            'cart_has_drink': cart_flags['cart_has_drink'],
            'cart_has_dessert': cart_flags['cart_has_dessert'],
            'user_order_count': user_order_count,
            'user_avg_order_value': user_avg_order_value,
            'user_avg_items': user_avg_items,
            'user_weekend_ratio': user_weekend_ratio,
            'user_single_item_ratio': user_single_item_ratio,
            'user_is_cold_start': user_is_cold_start,
            'cand_category': cand_cat,
            'cand_veg_nonveg': cand_veg,
            'cand_cuisine': cand_cuisine,
            'cand_typical_role': cand_role,
            'cand_flavor_profile': cand_flavor,
            'cand_popularity_rank': cand_pop_rank,
            'cand_order_frequency': cand_order_freq,
            'cand_solo_ratio': cand_solo,
            'cand_avg_price': cand_price,
            'fills_missing_slot': fills_gap,
            'veg_compatible': veg_compat,
            'max_lift': max_lift_val,
            'total_co_count': total_co,
            'max_confidence': max_conf,
            'copurchase_pairs': cop_pairs,
            'city': city,
            'city_lift': city_lift,
            'city_rank': city_rank_val,
            'is_local_favorite': is_local_fav,
            'cuisine_city_share': cuisine_city_share,
            'cuisine_city_rank': cuisine_city_rank_val,
            # LLM PCA embeddings
            **{f'item_emb_{i}': pca_vec[i] for i in range(16)},
            'item_semantic_cluster': str(cluster),
            'llm_context_compatibility': llm_ctx,
            'llm_cuisine_cat_affinity': llm_cca,
            'llm_meal_completion': llm_mc,
            'llm_flavor_harmony': llm_fh,
            'llm_cold_start_boost': llm_csb,
            'llm_cold_pop_signal': llm_cps,
            # Engineered features
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'price_ratio': price_ratio,
            'popularity_x_lift': pop_x_lift,
            'completeness_gap': comp_gap,
            'price_vs_user_avg': price_vs_user,
            'complement_fills_gap': complement_fills,
            'cart_size_bucket': cart_bucket,
            'city_item_signal': city_item_sig,
        }
        row['_candidate'] = cand
        row['_cand_category'] = cand_cat
        row['_cand_cuisine'] = cand_cuisine
        row['_cand_flavor'] = cand_flavor
        row['_fills_gap'] = fills_gap
        rows.append(row)
    
    feat_df = pd.DataFrame(rows)
    
    # Extract metadata before encoding
    meta_cols = ['_candidate', '_cand_category', '_cand_cuisine', '_cand_flavor', '_fills_gap']
    meta = feat_df[meta_cols].copy()
    feat_df.drop(columns=meta_cols, inplace=True)
    
    # Encode categoricals using saved label encoders
    CATS = ['meal_period', 'restaurant', 'cand_category', 'cand_veg_nonveg',
            'cand_cuisine', 'cand_typical_role', 'cand_flavor_profile', 'city',
            'item_semantic_cluster']
    for col in CATS:
        le = encoders.get(col)
        if le is not None:
            # Handle unseen labels gracefully
            known = set(le.classes_)
            feat_df[col] = feat_df[col].apply(
                lambda x: le.transform([x])[0] if x in known else 0
            )
        else:
            feat_df[col] = 0
    
    # Ensure column order matches training
    FEATURE_ORDER = [
        'items_in_cart', 'cart_value', 'completeness', 'meal_period', 'order_hour',
        'is_weekend', 'restaurant', 'cart_has_main', 'cart_has_side', 'cart_has_drink',
        'cart_has_dessert', 'user_order_count', 'user_avg_order_value', 'user_avg_items',
        'user_weekend_ratio', 'user_single_item_ratio', 'user_is_cold_start',
        'cand_category', 'cand_veg_nonveg', 'cand_cuisine', 'cand_typical_role',
        'cand_flavor_profile', 'cand_popularity_rank', 'cand_order_frequency',
        'cand_solo_ratio', 'cand_avg_price', 'fills_missing_slot', 'veg_compatible',
        'max_lift', 'total_co_count', 'max_confidence', 'copurchase_pairs',
        'city', 'city_lift', 'city_rank', 'is_local_favorite', 'cuisine_city_share',
        'cuisine_city_rank',
        'item_emb_0', 'item_emb_1', 'item_emb_2', 'item_emb_3', 'item_emb_4',
        'item_emb_5', 'item_emb_6', 'item_emb_7', 'item_emb_8', 'item_emb_9',
        'item_emb_10', 'item_emb_11', 'item_emb_12', 'item_emb_13', 'item_emb_14',
        'item_emb_15', 'item_semantic_cluster',
        'llm_context_compatibility', 'llm_cuisine_cat_affinity', 'llm_meal_completion',
        'llm_flavor_harmony', 'llm_cold_start_boost', 'llm_cold_pop_signal',
        'hour_sin', 'hour_cos', 'price_ratio', 'popularity_x_lift', 'completeness_gap',
        'price_vs_user_avg', 'complement_fills_gap', 'cart_size_bucket', 'city_item_signal',
    ]
    
    X = feat_df[FEATURE_ORDER]
    
    # Predict with LightGBM + XGBoost ensemble
    lgb_scores = lgb_model.predict(X)
    xgb_scores = xgb_model.predict(xgb.DMatrix(X))
    ensemble_scores = LGB_W * lgb_scores + XGB_W * xgb_scores
    
    # Rank and return top-K
    meta['score'] = ensemble_scores
    meta = meta.sort_values('score', ascending=False).head(top_k).reset_index(drop=True)
    
    latency_ms = (time.time() - t_start) * 1000
    
    results = []
    for i, row in meta.iterrows():
        expl = generate_explanation(
            row['_candidate'], row['_cand_category'], row['_cand_cuisine'],
            meal_period, row['_fills_gap'], row['_cand_flavor'], restaurant,
            row['score']
        )
        results.append({
            'rank': i + 1,
            'item': row['_candidate'],
            'score': round(float(row['score']), 4),
            'category': row['_cand_category'],
            'cuisine': row['_cand_cuisine'],
            'explanation': expl,
        })
    
    return {
        'restaurant': restaurant,
        'cart': cart_items,
        'city': city,
        'meal_period': meal_period,
        'recommendations': results,
        'latency_ms': round(latency_ms, 1),
        'candidates_scored': len(candidates),
    }


# ================================================================
# 5. INTERACTIVE DEMO
# ================================================================

def interactive_demo():
    """Run interactive demo with sample scenarios."""
    print("\n" + "=" * 60)
    print("  CSAO ADD-ON RECOMMENDATION ENGINE")
    print("  3-Tier Model: LightGBM + XGBoost + LLM Features")
    print("=" * 60)
    
    # Demo scenarios
    scenarios = [
        {
            "name": "Scenario 1: Pizza dinner, missing drink",
            "restaurant": "Aura Pizzas",
            "cart_items": ["Margherita Pizza", "Cheesy Garlic Bread"],
            "meal_period": "dinner",
            "city": "Mumbai",
            "is_weekend": 1,
        },
        {
            "name": "Scenario 2: Single biryani, cold-start user",
            "restaurant": "Swaad",
            "cart_items": ["Chicken Biryani"],
            "meal_period": "lunch",
            "city": "Hyderabad",
            "user_is_cold_start": 1,
        },
        {
            "name": "Scenario 3: Burger snack, evening",
            "restaurant": "Dilli Burger Adda",
            "cart_items": ["Classic Chicken Burger"],
            "meal_period": "snack",
            "city": "Delhi",
        },
        {
            "name": "Scenario 4: Large order, late night",
            "restaurant": "The Chicken Junction",
            "cart_items": ["Chicken Curry", "Butter Naan"],
            "meal_period": "late_night",
            "city": "Kolkata",
            "is_weekend": 0,
            "user_order_count": 15,
            "user_avg_order_value": 550,
            "user_is_cold_start": 0,
        },
    ]
    
    for scenario in scenarios:
        name = scenario.pop("name")
        print(f"\n{'─' * 60}")
        print(f"  {name}")
        print(f"{'─' * 60}")
        
        result = recommend_addons(**scenario, top_k=5)
        
        print(f"  Restaurant: {result['restaurant']}")
        print(f"  Cart: {result['cart']}")
        print(f"  City: {result['city']} | Meal: {result['meal_period']}")
        print(f"  Scored {result['candidates_scored']} candidates in {result['latency_ms']:.0f}ms")
        print()
        
        for rec in result['recommendations']:
            print(f"  #{rec['rank']} {rec['item']}")
            print(f"     Score: {rec['score']:.4f} | {rec['category']} | {rec['cuisine']}")
            print(f"     {rec['explanation']}")
            print()


# ================================================================
# 6. CLI ENTRY POINT
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSAO Add-On Recommendation")
    parser.add_argument("--json", type=str, help="JSON input for single prediction")
    parser.add_argument("--top_k", type=int, default=5, help="Number of recommendations")
    args = parser.parse_args()
    
    if args.json:
        # Single prediction from JSON
        params = json.loads(args.json)
        params['top_k'] = args.top_k
        result = recommend_addons(**params)
        print(json.dumps(result, indent=2))
    else:
        # Interactive demo
        interactive_demo()
