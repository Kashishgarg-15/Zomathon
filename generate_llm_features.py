# -*- coding: utf-8 -*-
"""
Tier 3: LLM-Enhanced Feature Engineering
==========================================
Uses sentence-transformers to create semantic item embeddings,
then derives features:
  1. Item embedding PCA components (captures semantic meaning of item names)
  2. Cuisine-role semantic similarity (how well candidate fits meal context)
  3. LLM-inspired meal completion score (rule-based + embedding-informed)
  4. Item semantic cluster assignment (groups similar items)

Output: training_data_llm.csv  (original data + new LLM features)
"""
import os, sys, time, json, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

t0 = time.time()
print("=" * 60, flush=True)
print("Tier 3: LLM-Enhanced Feature Engineering", flush=True)
print("=" * 60, flush=True)

# ================================================================
# 1. LOAD DATA
# ================================================================
print("\nStep 1: Loading data...", flush=True)
df = pd.read_csv("training_data_with_city.csv")
print(f"  Shape: {df.shape}", flush=True)

# ================================================================
# 2. GENERATE ITEM EMBEDDINGS
# ================================================================
print("\nStep 2: Generating semantic embeddings for items...", flush=True)
print("  Loading all-MiniLM-L6-v2 model...", flush=True)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Get unique items + create rich text descriptions
unique_items = df['candidate_item'].unique()
print(f"  {len(unique_items)} unique candidate items", flush=True)

# For each item, create a rich description using available metadata
item_meta = df.groupby('candidate_item').agg({
    'cand_category': 'first',
    'cand_veg_nonveg': 'first',
    'cand_cuisine': 'first',
    'cand_typical_role': 'first',
    'cand_flavor_profile': 'first',
    'cand_avg_price': 'mean',
}).reset_index()

# Build rich text descriptions for embedding (LLM-style item representation)
item_texts = {}
for _, row in item_meta.iterrows():
    text = (
        f"{row['candidate_item']} - "
        f"a {row['cand_veg_nonveg']} {row['cand_category']} item from {row['cand_cuisine']} cuisine, "
        f"typically served as {row['cand_typical_role']}, "
        f"with {row['cand_flavor_profile']} flavor profile"
    )
    item_texts[row['candidate_item']] = text

texts_ordered = [item_texts[item] for item in unique_items]
print(f"  Sample description: '{texts_ordered[0]}'", flush=True)

# Generate embeddings (batch)
print("  Encoding items...", flush=True)
embeddings = model.encode(texts_ordered, show_progress_bar=True, batch_size=64)
print(f"  Embedding shape: {embeddings.shape}", flush=True)

# Create item -> embedding mapping
item_to_emb = {item: emb for item, emb in zip(unique_items, embeddings)}

# ================================================================
# 3. PCA ON ITEM EMBEDDINGS (384-dim -> 16-dim)
# ================================================================
print("\nStep 3: PCA dimensionality reduction...", flush=True)
N_COMPONENTS = 16
pca = PCA(n_components=N_COMPONENTS, random_state=42)
emb_pca = pca.fit_transform(embeddings)
print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}", flush=True)

item_to_pca = {item: pca_vec for item, pca_vec in zip(unique_items, emb_pca)}

# Add PCA components as features
print("  Adding PCA embedding features to dataset...", flush=True)
pca_cols = [f'item_emb_{i}' for i in range(N_COMPONENTS)]
pca_matrix = np.array([item_to_pca[item] for item in df['candidate_item'].values])
for i, col in enumerate(pca_cols):
    df[col] = pca_matrix[:, i]

# ================================================================
# 4. SEMANTIC CLUSTERING (K-Means on embeddings)
# ================================================================
print("\nStep 4: Semantic item clustering...", flush=True)
N_CLUSTERS = 12
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
clusters = km.fit_predict(normalize(embeddings))
item_to_cluster = {item: int(c) for item, c in zip(unique_items, clusters)}
df['item_semantic_cluster'] = df['candidate_item'].map(item_to_cluster)

# Distribution
print(f"  {N_CLUSTERS} clusters. Distribution:", flush=True)
for cid in range(N_CLUSTERS):
    items_in = [it for it, cl in item_to_cluster.items() if cl == cid]
    print(f"    Cluster {cid}: {len(items_in)} items - e.g. {items_in[:3]}", flush=True)

# ================================================================
# 5. MEAL CONTEXT EMBEDDINGS + COMPATIBILITY SCORE
# ================================================================
print("\nStep 5: Computing meal-context compatibility scores...", flush=True)

# Create "meal context" embeddings based on meal_period + restaurant + cart composition
# This simulates what an LLM would reason about meal compatibility
meal_contexts = {}
for _, row in df.drop_duplicates(['meal_period', 'restaurant']).iterrows():
    ctx_key = (row['meal_period'], row['restaurant'])
    text = f"A {row['meal_period']} meal at {row['restaurant']}"
    meal_contexts[ctx_key] = text

ctx_keys = list(meal_contexts.keys())
ctx_texts = [meal_contexts[k] for k in ctx_keys]
ctx_embeddings = model.encode(ctx_texts, batch_size=64)
ctx_to_emb = {k: emb for k, emb in zip(ctx_keys, ctx_embeddings)}

# For each row, compute cosine similarity between candidate embedding and meal context
print("  Computing per-row context compatibility...", flush=True)
cand_embs = np.array([item_to_emb[item] for item in df['candidate_item'].values])
cand_embs_norm = normalize(cand_embs)

# Build context embedding array (vectorized via tuple key mapping)
ctx_keys_arr = list(zip(df['meal_period'].values, df['restaurant'].values))
ctx_emb_arr = normalize(np.array([ctx_to_emb[k] for k in ctx_keys_arr]))

# Cosine similarity
df['llm_context_compatibility'] = np.sum(cand_embs_norm * ctx_emb_arr, axis=1)

# ================================================================
# 6. CUISINE-CATEGORY SEMANTIC AFFINITY
# ================================================================
print("\nStep 6: Computing cuisine-category semantic affinity...", flush=True)

# Create embeddings for all cuisine-category combinations
cuisine_cat_texts = {}
for cuisine in df['cand_cuisine'].unique():
    for cat in df['cand_category'].unique():
        key = (cuisine, cat)
        cuisine_cat_texts[key] = f"{cuisine} {cat} food item"

cc_keys = list(cuisine_cat_texts.keys())
cc_texts = [cuisine_cat_texts[k] for k in cc_keys]
cc_embeddings = model.encode(cc_texts, batch_size=64)
cc_to_emb = {k: emb for k, emb in zip(cc_keys, cc_embeddings)}

# For each candidate, compute how well its cuisine-category matches the meal context
cc_keys_arr = list(zip(df['cand_cuisine'].values, df['cand_category'].values))
cc_emb_arr = normalize(np.array([cc_to_emb[k] for k in cc_keys_arr]))

df['llm_cuisine_cat_affinity'] = np.sum(cc_emb_arr * ctx_emb_arr, axis=1)

# ================================================================
# 7. MEAL COMPLETION REASONING (LLM-inspired rule-based)
# ================================================================
print("\nStep 7: Meal completion reasoning features...", flush=True)

# Simulate what an LLM would reason about meal completeness:
# "This cart is missing a drink, and the candidate IS a drink -> high completion score"
# This goes beyond the simple fills_missing_slot binary by adding semantic nuance.

# Meal completeness desire by period (LLM-style prior knowledge)
period_needs = {
    'Lunch': {'main': 1.0, 'side': 0.7, 'drink': 0.8, 'dessert': 0.3},
    'Dinner': {'main': 1.0, 'side': 0.8, 'drink': 0.6, 'dessert': 0.5},
    'Snack': {'main': 0.3, 'side': 0.8, 'drink': 0.9, 'dessert': 0.4},
    'Breakfast': {'main': 1.0, 'side': 0.5, 'drink': 0.9, 'dessert': 0.2},
    'Late Night': {'main': 0.7, 'side': 0.6, 'drink': 0.5, 'dessert': 0.4},
}
# Default for unknown periods
default_needs = {'main': 0.7, 'side': 0.6, 'drink': 0.6, 'dessert': 0.4}

def meal_completion_score(row):
    """LLM-inspired: how much does this candidate complete the meal?"""
    needs = period_needs.get(row['meal_period'], default_needs)
    cat = row['cand_category']
    if cat == 'combo':
        return 0.5  # combos always moderately desired
    
    # Base desire for this category in this meal period
    desire = needs.get(cat, 0.5)
    
    # Amplify if cart is missing this slot
    has_map = {
        'main': row['cart_has_main'],
        'side': row['cart_has_side'],
        'drink': row['cart_has_drink'],
        'dessert': row['cart_has_dessert'],
    }
    if cat in has_map and has_map[cat] == 0:
        desire *= 1.5  # bonus for filling gap
    elif cat in has_map and has_map[cat] == 1:
        desire *= 0.6  # penalty for redundancy
    
    return min(desire, 1.0)

# Vectorized meal completion (avoid slow .apply)
llm_mc = np.full(len(df), 0.5)  # default for combos
non_combo = df['cand_category'] != 'combo'
for cat, has_col in [('main', 'cart_has_main'), ('side', 'cart_has_side'),
                      ('drink', 'cart_has_drink'), ('dessert', 'cart_has_dessert')]:
    mask_cat = (df['cand_category'] == cat)
    for period, needs in period_needs.items():
        mask_period = (df['meal_period'] == period)
        mask = mask_cat & mask_period & non_combo
        if mask.sum() == 0:
            continue
        desire = needs.get(cat, 0.5)
        vals = np.where(
            df.loc[mask, has_col] == 0,
            np.minimum(desire * 1.5, 1.0),  # gap bonus
            desire * 0.6                     # redundancy penalty
        )
        llm_mc[mask.values] = vals
df['llm_meal_completion'] = llm_mc

# ================================================================
# 8. FLAVOR HARMONY SCORE (embedding-based)
# ================================================================
print("\nStep 8: Flavor harmony scoring...", flush=True)

# Use embeddings to compute flavor compatibility
flavor_texts = {
    'rich': "rich creamy buttery heavy indulgent food",
    'spicy': "hot spicy chili pepper fiery food",
    'tangy': "sour tangy citrus lime vinegar food",
    'mild': "mild gentle subtle delicate bland food",
    'sweet': "sweet sugary dessert candy honey food",
    'savory': "savory umami salty meaty brothy food",
    'neutral': "plain neutral basic standard food",
}

flavor_embs = model.encode(list(flavor_texts.values()))
flavor_to_emb = {f: emb for f, emb in zip(flavor_texts.keys(), flavor_embs)}

# Define pairing harmony (LLM knowledge about flavor pairings)
harmony_pairs = {
    ('spicy', 'mild'): 1.2,   # cooling with heat
    ('spicy', 'sweet'): 1.1,  # sweet+heat
    ('rich', 'tangy'): 1.1,   # cutting richness
    ('savory', 'sweet'): 0.9, # meh
    ('rich', 'rich'): 0.7,    # too heavy
    ('spicy', 'spicy'): 0.8,  # too much heat
}

# For each order, approximate cart flavor from meal_period heuristics
# Since we don't have cart item names, use meal-period as proxy for typical cart flavor
period_dominant_flavor = {
    'Lunch': 'savory',
    'Dinner': 'rich',
    'Snack': 'spicy',
    'Breakfast': 'mild',
    'Late Night': 'rich',
}

def flavor_harmony(row):
    cart_flavor = period_dominant_flavor.get(row['meal_period'], 'savory')
    cand_flavor = row['cand_flavor_profile']
    # Check known harmonies
    pair = (cart_flavor, cand_flavor)
    pair_rev = (cand_flavor, cart_flavor)
    if pair in harmony_pairs:
        return harmony_pairs[pair]
    if pair_rev in harmony_pairs:
        return harmony_pairs[pair_rev]
    # Use embedding cosine similarity as fallback
    e1 = flavor_to_emb.get(cart_flavor, flavor_to_emb['savory'])
    e2 = flavor_to_emb.get(cand_flavor, flavor_to_emb['neutral'])
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10))

# Vectorized flavor harmony
llm_fh = np.zeros(len(df))
for period, dom_flav in period_dominant_flavor.items():
    for cand_flav in df['cand_flavor_profile'].unique():
        mask = (df['meal_period'] == period) & (df['cand_flavor_profile'] == cand_flav)
        if mask.sum() == 0:
            continue
        pair = (dom_flav, cand_flav)
        pair_rev = (cand_flav, dom_flav)
        if pair in harmony_pairs:
            llm_fh[mask.values] = harmony_pairs[pair]
        elif pair_rev in harmony_pairs:
            llm_fh[mask.values] = harmony_pairs[pair_rev]
        else:
            e1 = flavor_to_emb.get(dom_flav, flavor_to_emb['savory'])
            e2 = flavor_to_emb.get(cand_flav, flavor_to_emb['neutral'])
            llm_fh[mask.values] = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10))
df['llm_flavor_harmony'] = llm_fh

# ================================================================
# 9. COLD-START ENHANCED FEATURES
# ================================================================
print("\nStep 9: Cold-start enhancement features...", flush=True)

# For cold-start users, we rely MORE on item semantics and less on user history
# Create a "confidence" feature that blends user signals and semantic signals
df['llm_cold_start_boost'] = np.where(
    df['user_is_cold_start'] == 1,
    # Cold users: boost semantic signals
    0.7 * df['llm_context_compatibility'] + 0.3 * df['llm_meal_completion'],
    # Warm users: lower boost (they have history features)
    0.3 * df['llm_context_compatibility'] + 0.1 * df['llm_meal_completion']
)

# For cold-start: item popularity becomes more important
df['llm_cold_pop_signal'] = np.where(
    df['user_is_cold_start'] == 1,
    df['cand_order_frequency'] / (df['cand_order_frequency'].max() + 1),
    0.0
)

# ================================================================
# 10. SAVE ENRICHED DATASET
# ================================================================
print("\nStep 10: Saving enriched dataset...", flush=True)

new_features = pca_cols + [
    'item_semantic_cluster',
    'llm_context_compatibility',
    'llm_cuisine_cat_affinity',
    'llm_meal_completion',
    'llm_flavor_harmony',
    'llm_cold_start_boost',
    'llm_cold_pop_signal',
]
print(f"  New LLM features ({len(new_features)}):", flush=True)
for f in new_features:
    print(f"    {f}: range [{df[f].min():.4f}, {df[f].max():.4f}]", flush=True)

df.to_csv("training_data_llm.csv", index=False)
print(f"  Saved training_data_llm.csv: {df.shape}", flush=True)

# Also save the embedding artifacts for inference
import joblib
os.makedirs("llm_artifacts", exist_ok=True)
joblib.dump({
    'item_to_emb': item_to_emb,
    'item_to_pca': item_to_pca,
    'item_to_cluster': item_to_cluster,
    'pca_model': pca,
    'kmeans_model': km,
    'ctx_to_emb': ctx_to_emb,
    'flavor_to_emb': flavor_to_emb,
}, os.path.join("llm_artifacts", "embedding_artifacts.pkl"))
print("  Saved llm_artifacts/embedding_artifacts.pkl", flush=True)

# ================================================================
# 11. LLM EXPLANATION GENERATOR (Framework)
# ================================================================
print("\n" + "=" * 60, flush=True)
print("LLM Explanation Generator (Framework):", flush=True)
print("-" * 60, flush=True)

# This shows how an LLM could generate human-readable explanations
# In production, you'd call Groq/GPT API here
def generate_explanation(candidate_item, cand_category, cand_cuisine,
                         meal_period, fills_gap, flavor, restaurant):
    """
    Template-based explanation generator.
    In production, replace with LLM API call:
      prompt = f"User ordered {meal_period} at {restaurant}. 
                 Explain why {candidate_item} ({cand_cuisine} {cand_category})
                 would be a good add-on."
    """
    reasons = []
    if fills_gap:
        reasons.append(f"completes your meal by adding a {cand_category}")
    if flavor in ['mild', 'tangy'] and cand_category == 'drink':
        reasons.append(f"its {flavor} taste refreshes your palate")
    elif flavor == 'sweet':
        reasons.append("adds a sweet finish to your meal")
    elif flavor == 'spicy':
        reasons.append("adds an exciting kick to your order")
    
    if not reasons:
        reasons.append(f"it's a popular {cand_cuisine} {cand_category} that pairs well with your order")
    
    return f"We recommend {candidate_item} because {'and '.join(reasons)}."

# Demo explanations
demo_rows = df.drop_duplicates('candidate_item').head(5)
for _, row in demo_rows.iterrows():
    expl = generate_explanation(
        row['candidate_item'], row['cand_category'], row['cand_cuisine'],
        row['meal_period'], row['fills_missing_slot'], row['cand_flavor_profile'],
        row['restaurant']
    )
    print(f"  {expl}", flush=True)

elapsed = time.time() - t0
print(f"\n{'=' * 60}", flush=True)
print(f"DONE! {elapsed / 60:.1f} min", flush=True)
print(f"  Original features: 43", flush=True)
print(f"  New LLM features: {len(new_features)}", flush=True)
print(f"  Total features: {df.shape[1]}", flush=True)
print(f"  Output: training_data_llm.csv ({df.shape[0]:,} x {df.shape[1]})", flush=True)
print(f"{'=' * 60}", flush=True)
