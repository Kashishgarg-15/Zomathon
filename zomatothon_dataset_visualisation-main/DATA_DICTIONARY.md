# Data Dictionary — CSAO Add-On Recommendation Pipeline

> Complete reference for every output file, feature, and curation strategy used in the data pipeline.
> Source dataset: **21,321 food delivery orders** from 6 restaurants (Delhi NCR).
> Pipeline runtime: ~100 s on CPU.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [File Inventory](#2-file-inventory)
3. [Primary Training File — `augmented_training_data.csv`](#3-primary-training-file--augmented_training_datacsv)
4. [Lookup / Reference Files for Training & Inference](#4-lookup--reference-files-for-training--inference)
   - 4.1 [item_attributes.csv](#41-item_attributescsv)
   - 4.2 [item_stats.csv](#42-item_statscsv)
   - 4.3 [user_profiles.csv](#43-user_profilescsv)
   - 4.4 [copurchase_matrix.csv](#44-copurchase_matrixcsv)
   - 4.5 [order_completeness.csv](#45-order_completenesscsv)
   - 4.6 [cart_sequences.csv](#46-cart_sequencescsv)
   - 4.7 [orders_enriched.csv](#47-orders_enrichedcsv)
5. [Geographic / City Files](#5-geographic--city-files)
   - 5.1 [city_profiles.csv](#51-city_profilescsv)
   - 5.2 [user_city_assignment.csv](#52-user_city_assignmentcsv)
   - 5.3 [city_item_popularity.csv](#53-city_item_popularitycsv)
   - 5.4 [city_cuisine_affinity.csv](#54-city_cuisine_affinitycsv)
   - 5.5 [orders_with_city.csv](#55-orders_with_citycsv)
6. [Augmentation Strategies Deep Dive](#6-augmentation-strategies-deep-dive)
7. [Feature Engineering Strategies](#7-feature-engineering-strategies)
8. [How to Use for Model Training](#8-how-to-use-for-model-training)
9. [How to Use for Inference](#9-how-to-use-for-inference)

---

## 1. Pipeline Overview

```
Raw CSV (21,321 orders)
  │
  ├─ Phase 0 ── Derived Features (temporal, cart, user profiles, co-purchase)
  ├─ Phase 1 ── LLM Item Enrichment (Groq llama-3.3-70b: category, cuisine, role)
  ├─ Phase 2 ── Meal Completeness Scoring (weighted slot detection)
  ├─ Phase 3 ── Cart Sequence Simulation (canonical ordering → progressive stages)
  ├─ Phase 4 ── Training Data Construction (positive + 5× negative sampling)
  ├─ Phase 4.5 ── Data Augmentation (item-swap, collaborative, soft-label)
  └─ Phase 5 ── City Assignment & Geographic Enrichment (6 Indian cities)
```

| Phase | Cost | Key Output | Rows |
|-------|------|-----------|------|
| 0 | Zero (pandas) | orders_enriched, user_profiles, item_stats, copurchase_matrix | 21,321 / 11,607 / 244 / 1,681 |
| 1 | 6 Groq API calls | item_attributes | 244 |
| 2 | Zero | order_completeness | 21,321 |
| 3 | Zero | cart_sequences | 38,232 |
| 4 | Zero | training_data | 101,466 |
| 4.5 | Zero | augmented_training_data | **212,880** |
| 5 | Zero | city outputs (5 files) | various |

---

## 2. File Inventory

All files are in `data_curation/output/`.

| File | Rows | Cols | Role | Required for Training | Required for Inference |
|------|------|------|------|:----:|:----:|
| `augmented_training_data.csv` | 212,880 | 44 | **Primary training matrix** | ✅ | — |
| `training_data.csv` | 101,466 | 42 | Original (non-augmented) training matrix | Optional | — |
| `item_attributes.csv` | 244 | 8 | Item metadata (LLM-generated) | ✅ lookup | ✅ lookup |
| `item_stats.csv` | 244 | 11 | Item popularity & pricing | ✅ lookup | ✅ lookup |
| `user_profiles.csv` | 11,607 | 18 | User behavioral aggregates | ✅ lookup | ✅ lookup |
| `copurchase_matrix.csv` | 1,681 | 7 | Item-pair co-occurrence | ✅ lookup | ✅ lookup |
| `order_completeness.csv` | 21,321 | 11 | Meal completeness per order | ✅ lookup | ✅ (runtime) |
| `cart_sequences.csv` | 38,232 | 12 | Progressive cart stages | Intermediate | — |
| `orders_enriched.csv` | 21,341 | 44 | Enriched order-level features | Intermediate | — |
| `city_profiles.csv` | 6 | 10 | City food culture priors | Reference | ✅ lookup |
| `user_city_assignment.csv` | 11,607 | 10 | User → city mapping | ✅ lookup | ✅ lookup |
| `city_item_popularity.csv` | 975 | 9 | Per-city item rankings | ✅ lookup | ✅ lookup |
| `city_cuisine_affinity.csv` | 41 | 5 | Per-city cuisine distribution | Reference | ✅ lookup |
| `orders_with_city.csv` | 21,341 | 45 | Orders with city + adjustments | Reference | — |
| `pipeline_log.json` | — | — | Execution metadata | — | — |

---

## 3. Primary Training File — `augmented_training_data.csv`

**212,880 rows × 44 columns.** Each row represents one **(cart-stage, candidate-item)** pair.

### Label semantics

| `label` | `aug_type` | Meaning |
|---------|-----------|---------|
| `1` | `original` | User actually added this item next (ground truth) |
| `1` | `item_swap` | Same-category item swapped in (synthetic positive, weight 0.85) |
| `1` | `collaborative` | Co-purchase-based synthetic add-on (weight 0.75) |
| `0.5` | `soft_label` | Hypothetical best add-on for a single-item order (weight 0.50) |
| `0` | any | Negative sample — item from same restaurant menu NOT chosen (weight 1.0) |

### Full Column Reference

#### A. Context Features (describe the cart at this stage)

| # | Column | Type | Source | Description | Curation Strategy |
|---|--------|------|--------|-------------|-------------------|
| 1 | `order_id` | str | Phase 3 | Order identifier (prefixed `syn_` for collaborative, `soft_` for soft-label augmented orders) | Direct from raw data; synthetic IDs for augmented rows |
| 2 | `stage` | int | Phase 3 | Stage number within the cart building sequence (1 = first item in cart) | Cart sequence simulation: items sorted by category order (Main→Side→Drink→Dessert), then by descending price within each category |
| 3 | `items_in_cart` | int | Phase 3 | Number of items currently in the cart at this stage | Count of items added so far in the canonical ordering |
| 4 | `cart_value` | float | Phase 3 | Estimated monetary value of the cart at this stage (₹) | Proportional split of `Bill subtotal` across the number of items at this stage |
| 5 | `completeness` | float | Phase 2/3 | Meal completeness score at this stage (0.0–1.0) | Weighted sum: main=0.40, side=0.25, drink=0.20, dessert=0.15. Recalculated at each stage from categories present in the partial cart |
| 6 | `meal_period` | str | Phase 0 | Time-of-day category: `breakfast` (6–10), `lunch` (10–14), `snack` (14–17), `dinner` (17–22), `late_night` (22–6) | Derived from `order_hour` using fixed bin boundaries |
| 7 | `order_hour` | int | Phase 0 | Hour of day the order was placed (0–23) | Extracted from `Order Placed At` datetime field |
| 8 | `is_weekend` | int | Phase 0 | 1 if Saturday or Sunday, 0 otherwise | Derived from `order_dow` (day of week) |
| 9 | `restaurant` | str | Raw | Restaurant name | Direct from raw data |

#### B. Gap-Detection Features (what the cart is missing)

| # | Column | Type | Source | Description | Curation Strategy |
|---|--------|------|--------|-------------|-------------------|
| 10 | `missing_main` | int | Phase 2 | 1 if the cart lacks a main course / combo | Checks if neither `main` nor `combo` category is present among cart items (using LLM-classified categories) |
| 11 | `missing_side` | int | Phase 2 | 1 if the cart lacks a side dish / snack | Checks if neither `side` nor `snack` category is present |
| 12 | `missing_drink` | int | Phase 2 | 1 if the cart lacks a drink | Category presence check |
| 13 | `missing_dessert` | int | Phase 2 | 1 if the cart lacks a dessert | Category presence check |

#### C. Cart Composition Features

| # | Column | Type | Source | Description | Curation Strategy |
|---|--------|------|--------|-------------|-------------------|
| 14 | `cart_has_main` | int | Phase 4 | 1 if cart contains a main or combo item | Inverse of `missing_main`; computed from LLM-categorized items currently in the cart |
| 15 | `cart_has_side` | int | Phase 4 | 1 if cart contains a side or snack item | Inverse of `missing_side` |
| 16 | `cart_has_drink` | int | Phase 4 | 1 if cart contains a drink | Category presence check |
| 17 | `cart_has_dessert` | int | Phase 4 | 1 if cart contains a dessert | Category presence check |

#### D. User Profile Features

| # | Column | Type | Source | Description | Curation Strategy |
|---|--------|------|--------|-------------|-------------------|
| 18 | `user_order_count` | int | Phase 0 | Total number of orders by this user | Count of distinct `Order ID` per `Customer ID` |
| 19 | `user_avg_order_value` | float | Phase 0 | Average total bill across all user's orders (₹) | `mean(Total)` grouped by customer |
| 20 | `user_avg_items` | float | Phase 0 | Average number of items per order | `mean(num_items)` grouped by customer |
| 21 | `user_diversity` | float | Phase 0 | Item diversity score (0–1): unique items / total items ordered | Higher = user tries many different items; lower = habitual orderer |
| 22 | `user_weekend_ratio` | float | Phase 0 | Fraction of user's orders placed on weekends (0–1) | `mean(is_weekend)` grouped by customer |
| 23 | `user_single_item_ratio` | float | Phase 0 | Fraction of user's orders that contain only 1 unique item (0–1) | `mean(is_single_item)` grouped by customer. High ratio = user rarely adds items (prime target for recommendations) |
| 24 | `user_is_cold_start` | int | Phase 0 | 1 if user has only 1 order in the dataset, 0 otherwise | Binary flag; 66.5% of users are cold-start |

#### E. Candidate Item Features (describe the item being scored)

| # | Column | Type | Source | Description | Curation Strategy |
|---|--------|------|--------|-------------|-------------------|
| 25 | `candidate_item` | str | Phase 4 | Name of the candidate item being evaluated | Positive: the item actually added next. Negative: random item from same restaurant's menu |
| 26 | `cand_category` | str | Phase 1 | Food category: `main`, `side`, `drink`, `dessert`, `snack`, `combo` | LLM-classified using Groq llama-3.3-70b-versatile with restaurant context. Validated against allowed set |
| 27 | `cand_veg_nonveg` | str | Phase 1 | Dietary type: `veg`, `non-veg`, `egg` | LLM-classified. Validated against allowed set |
| 28 | `cand_cuisine` | str | Phase 1 | Cuisine type (e.g., `North Indian`, `Continental`, `Italian`, `Street Food`, `Pan-Indian`) | LLM-classified. Free-text cuisine labels normalized by the LLM |
| 29 | `cand_typical_role` | str | Phase 1 | Item role: `anchor` (meal center), `complement` (add-on), `impulse` (low-cost extra) | LLM-classified. 164 anchors, 71 complements, 9 impulse items |
| 30 | `cand_flavor_profile` | str | Phase 1 | Flavor: `spicy`, `mild`, `sweet`, `savory`, `tangy`, `rich`, `neutral` | LLM-classified |
| 31 | `cand_popularity_rank` | int | Phase 0 | Item popularity rank (1 = most ordered) | Rank by `order_frequency` (number of distinct orders containing this item) |
| 32 | `cand_order_frequency` | int | Phase 0 | Number of orders containing this item | Count of distinct `Order ID` where item appears |
| 33 | `cand_solo_ratio` | float | Phase 0 | Fraction of time this item is ordered alone (0–1) | `solo_order_count / order_frequency`. High = item is a standalone meal (e.g., biryani); Low = item is typically part of a combo |
| 34 | `cand_avg_price` | float | Phase 0 | Average price proxy for this item (₹) | `mean(Bill subtotal / num_items_in_order)` across all orders containing this item |

#### F. Compatibility Features

| # | Column | Type | Source | Description | Curation Strategy |
|---|--------|------|--------|-------------|-------------------|
| 35 | `fills_missing_slot` | int | Phase 4 | 1 if the candidate's category matches a missing meal component | Cross-references candidate's LLM-classified category against the order's `missing_categories` string |
| 36 | `veg_compatible` | int | Phase 4 | 1 if the candidate is veg-compatible with the order | Compatible if: order is non-veg (eats anything), either is unknown, or categories match. Prevents recommending non-veg to a veg order |

#### G. Co-Purchase Signal Features

| # | Column | Type | Source | Description | Curation Strategy |
|---|--------|------|--------|-------------|-------------------|
| 37 | `max_lift` | float | Phase 0/4 | Maximum association rule lift between the candidate and ANY item currently in the cart | Lift = P(A∩B) / (P(A)×P(B)). Computed from 1,681 co-purchase pairs (min_support=3). Higher lift = items appear together more than expected by chance |
| 38 | `avg_lift` | float | Phase 0/4 | Average lift across all co-purchase pairs between candidate and cart items | Mean of individual pairwise lifts |
| 39 | `total_co_count` | int | Phase 0/4 | Total number of co-occurrences between the candidate and all cart items | Sum of raw co-occurrence counts from multi-item orders |
| 40 | `max_confidence` | float | Phase 0/4 | Maximum conditional probability P(candidate | cart_item) | Directional association: "if someone ordered cart_item, what fraction also ordered candidate?" |
| 41 | `copurchase_pairs` | int | Phase 0/4 | Number of items in the cart that have a co-purchase relationship with the candidate | 0 = no historical co-purchase signal (cold pair) |

#### H. Label & Metadata

| # | Column | Type | Source | Description | Curation Strategy |
|---|--------|------|--------|-------------|-------------------|
| 42 | `label` | float | Phase 4/4.5 | Target variable. `1.0` = positive (item was/would be added), `0.5` = soft positive (hypothetical), `0` = negative (item not chosen) | 1:5 positive:negative ratio per stage. Negatives sampled from same restaurant's menu, excluding items already in cart + the positive item |
| 43 | `aug_type` | str | Phase 4.5 | Augmentation source: `original`, `item_swap`, `collaborative`, `soft_label` | See [Section 6](#6-augmentation-strategies-deep-dive) |
| 44 | `sample_weight` | float | Phase 4.5 | Loss weight for training: `1.0` (original/negatives), `0.85` (item-swap), `0.75` (collaborative), `0.50` (soft-label) | Decreasing confidence by augmentation fidelity. Use as `sample_weight` parameter in XGBoost/LightGBM or weighted BCE in neural models |

---

## 4. Lookup / Reference Files for Training & Inference

### 4.1 `item_attributes.csv`

**244 rows × 8 columns** — One row per unique menu item.

| Column | Type | Description | Curation Strategy |
|--------|------|-------------|-------------------|
| `item` | str | Exact item name as appears in orders | Parsed from order strings using regex `re.split(r",\s*(?=\d+\s*x\s)")` |
| `category` | str | `main` / `side` / `drink` / `dessert` / `snack` / `combo` | **LLM-classified** (Groq llama-3.3-70b-versatile). Items batched by restaurant (50/batch). Validated against allowed set — fallback to `main` if invalid |
| `veg_nonveg` | str | `veg` / `non-veg` / `egg` | **LLM-classified**. Validated — fallback to `non-veg` |
| `cuisine` | str | Cuisine label (e.g., `North Indian`, `Italian`) | **LLM-classified**. Free-text labels from the model |
| `typical_role` | str | `anchor` / `complement` / `impulse` | **LLM-classified**. Anchor = meal center (pizza, biryani), Complement = add-on (raita, fries), Impulse = low-cost extra (brownie, coke). Validated — fallback to `anchor` |
| `flavor_profile` | str | `spicy` / `mild` / `sweet` / `savory` / `tangy` / `rich` / `neutral` | **LLM-classified** |
| `pairs_well_with` | str | Comma-separated pairing suggestions (3–5 items) | **LLM-generated**. E.g., "raita, cold drink, salad, naan" |
| `restaurant` | str | Primary restaurant serving this item | Mode restaurant from order data |

**Distribution:** 155 main, 71 side, 9 combo, 7 drink, 2 dessert | 127 non-veg, 116 veg, 1 egg | 164 anchor, 71 complement, 9 impulse

---

### 4.2 `item_stats.csv`

**244 rows × 11 columns** — Aggregated statistics per item.

| Column | Type | Description | Curation Strategy |
|--------|------|-------------|-------------------|
| `item` | str | Item name | — |
| `order_frequency` | int | Distinct orders containing this item | Count of unique `Order ID`s |
| `total_qty_sold` | int | Total quantity sold across all orders | Sum of parsed quantities |
| `restaurant` | str | Mode restaurant | Most frequent restaurant for this item |
| `avg_bill_when_present` | float | Average bill subtotal when item is in the order (₹) | Indicates price tier context |
| `solo_order_count` | int | Orders where this was the only item | Indicates standalone meal viability |
| `multi_order_count` | int | Orders where this appeared alongside other items | Indicates complementary potential |
| `solo_order_ratio` | float | `solo_order_count / order_frequency` (0–1) | High = standalone meal (biryani); Low = add-on item |
| `popularity_rank` | int | Rank by order frequency (1 = most popular) | Descending rank |
| `avg_price_proxy` | float | Estimated per-item price (₹) | `mean(Bill subtotal / num_items_in_order)` for orders containing this item |
| `peak_meal_period` | str | Meal period when item is most ordered | Mode of meal_period across orders containing this item |

---

### 4.3 `user_profiles.csv`

**11,607 rows × 18 columns** — One row per unique customer.

| Column | Type | Description | Curation Strategy |
|--------|------|-------------|-------------------|
| `Customer ID` | str | Unique customer identifier | Direct from raw data |
| `order_count` | int | Total orders placed | Count of unique `Order ID` |
| `avg_order_value` | float | Mean `Total` bill (₹) | Aggregated mean |
| `std_order_value` | float | Std dev of `Total` bill | Spending consistency indicator |
| `avg_items_per_order` | float | Mean item count per order | Parsed item count averaged |
| `avg_unique_items` | float | Mean unique item names per order | Measures within-order variety |
| `single_item_ratio` | float | Fraction of orders with 1 unique item | Key cold-start signal |
| `avg_discount_pct` | float | Mean discount percentage across orders | `total_discount / Bill subtotal × 100` |
| `total_spend` | float | Lifetime spend (₹) | Sum of `Total` |
| `preferred_restaurant` | str | Most-ordered-from restaurant | Mode |
| `preferred_meal_period` | str | Most common ordering time | Mode of `meal_period` |
| `weekend_ratio` | float | Fraction of weekend orders | Mean of `is_weekend` |
| `diversity_score` | float | Unique items / total items ordered (0–1) | Higher = adventurous eater |
| `first_order` | datetime | Date of first order | Min `order_datetime` |
| `last_order` | datetime | Date of last order | Max `order_datetime` |
| `tenure_days` | int | Days between first and last order | Retention metric |
| `is_cold_start` | int | 1 if `order_count == 1` | 66.5% of users |
| `veg_preference` | float | Placeholder (NaN) | Reserved for Phase 1 refinement |

---

### 4.4 `copurchase_matrix.csv`

**1,681 rows × 7 columns** — Item-pair association rules from multi-item orders.

| Column | Type | Description | Curation Strategy |
|--------|------|-------------|-------------------|
| `item_a` | str | First item in the pair (alphabetically sorted) | Canonical pair ordering (A < B alphabetically) |
| `item_b` | str | Second item | — |
| `co_count` | int | Number of orders containing both items | Raw co-occurrence count; min_support threshold = 3 |
| `support` | float | `co_count / total_orders` | Joint probability; typically very low (sparse) |
| `confidence_a_given_b` | float | P(A appears | B appears) = `co_count / freq(A)` | Directional: "given item A was ordered, how often was B also ordered?" |
| `confidence_b_given_a` | float | P(B appears | A appears) = `co_count / freq(B)` | Reverse direction |
| `lift` | float | `co_count × total_orders / (freq(A) × freq(B))` | >1 = positive association (items complement each other), <1 = substitutes, =1 = independent |

---

### 4.5 `order_completeness.csv`

**21,321 rows × 11 columns** — Meal completeness evaluation per order.

| Column | Type | Description | Curation Strategy |
|--------|------|-------------|-------------------|
| `Order ID` | str | Order identifier | — |
| `completeness_score` | float | Weighted completeness (0–1) | Weights: main=0.40, side=0.25, drink=0.20, dessert=0.15. A "complete meal" has all four → 1.0 |
| `has_main` | int | 1 if order contains main or combo | LLM-classified category check |
| `has_side` | int | 1 if order contains side or snack | — |
| `has_drink` | int | 1 if order contains a drink | — |
| `has_dessert` | int | 1 if order contains a dessert | — |
| `missing_categories` | str | Pipe-separated missing components (e.g., `side\|drink\|dessert`) or `none` | Determines what to recommend |
| `num_missing` | int | Number of missing categories (0–4) | — |
| `recommendation_priority` | float | Priority score: `(1 - completeness) + 0.2 × is_single`. Higher = more urgently needs a recommendation | Composite signal for ranking which orders to target first |
| `order_veg_type` | str | `veg` / `non-veg` / `egg` / `unknown` | Derived from all items' veg_nonveg. If any item is non-veg, the order is non-veg |
| `categories_present` | str | Pipe-separated categories found in order | Sorted alphabetically |

**Stats:** Avg completeness = 0.453, 99% of orders need add-ons. Missing: dessert 99.5%, drink 98%, side 65%.

---

### 4.6 `cart_sequences.csv`

**38,232 rows × 12 columns** — Progressive cart states used to generate training data.

| Column | Type | Description | Curation Strategy |
|--------|------|-------------|-------------------|
| `order_id` | str | Order identifier | — |
| `stage` | int | Stage number (1-indexed) | Stage 1 = first item; Stage N = final cart state |
| `current_cart` | str (JSON) | JSON array of item names at this stage | E.g., `["Margherita Pizza", "Garlic Bread"]` |
| `next_item_added` | str | Item added at the next stage (empty string if `is_final=1`) | The label for positive training samples |
| `next_item_category` | str | Category of `next_item_added` | From LLM classification |
| `next_item_role` | str | Role of `next_item_added` | anchor / complement / impulse |
| `cart_value_at_stage` | float | Estimated cart value (₹) at this stage | Proportional allocation of bill |
| `items_in_cart` | int | Item count at this stage | — |
| `completeness_at_stage` | float | Meal completeness at this stage (0–1) | Recalculated from categories present in partial cart |
| `is_final` | int | 1 if this is the final cart state (no more items added) | Final stages are NOT used for training (no positive label) |
| `restaurant` | str | Restaurant name | — |
| `meal_period` | str | Meal period of the order | — |

**Ordering strategy:** Items sorted by **category order** (Main → Side → Drink → Dessert → Snack → Combo), then by **descending price** within each category. This simulates how users typically build a meal: choose the main dish first, then add complementary items.

**Stage counts:** 11,694 multi-item orders → 16,911 trainable stages + 9,627 single-item orders (inference targets only).

---

### 4.7 `orders_enriched.csv`

**21,341 rows × 44 columns** — Order-level data with all Phase 0 derived features.

| Feature Group | Columns | Curation Strategy |
|---------------|---------|-------------------|
| **Original raw** | `Restaurant ID`, `Restaurant name`, `Subzone`, `City`, `Order ID`, `Order Placed At`, `Order Status`, `Delivery`, `Distance`, `Items in order`, `Instructions`, `Discount construct`, `Bill subtotal`, `Packaging charges`, `Restaurant discount (Promo)`, `Restaurant discount (Flat offs, Freebies & others)`, `Gold discount`, `Brand pack discount`, `Total`, `Rating`, `Review`, `Cancellation / Rejection reason`, `Restaurant compensation (Cancellation)`, `Restaurant penalty (Rejection)`, `KPT duration (minutes)`, `Rider wait time (minutes)`, `Order Ready Marked`, `Customer complaint tag`, `Customer ID` | Direct from raw CSV |
| **Temporal** | `order_datetime` (parsed datetime), `distance_km` (float from Distance), `order_hour`, `order_dow`, `order_day_name`, `is_weekend`, `meal_period`, `order_date` | Extracted from `Order Placed At` datetime |
| **Cart** | `num_items`, `num_unique_items`, `is_single_item`, `price_per_item` | Parsed from `Items in order` via regex |
| **Discount** | `total_discount`, `discount_pct`, `has_discount` | Sum of all discount columns; percentage relative to subtotal |

---

## 5. Geographic / City Files

### 5.1 `city_profiles.csv`

**6 rows × 10 columns** — Behavioral templates for each city.

| Column | Type | Description |
|--------|------|-------------|
| `city` | str | City name: Delhi, Mumbai, Bangalore, Hyderabad, Pune, Kolkata |
| `peak_meal` | str | Dominant meal period for this city |
| `late_night_affinity` | float | Propensity for late-night ordering (0–1) |
| `avg_order_value_z` | float | Z-score offset for average order value |
| `cart_size_z` | float | Z-score offset for cart size |
| `discount_affinity` | float | Z-score offset for discount usage |
| `veg_ratio` | float | Expected fraction of veg orders |
| `weekend_boost` | float | Weekend ordering boost factor |
| `top_cuisines` | str (JSON) | JSON dict of cuisine → weight |
| `description` | str | Human-readable city food profile |

**Strategy:** Hardcoded from real-world Indian food culture knowledge. Used as priors for behavior-first clustering.

---

### 5.2 `user_city_assignment.csv`

**11,607 rows × 10 columns** — User-to-city mapping.

| Column | Type | Description |
|--------|------|-------------|
| `Customer ID` | str | User identifier |
| `city` | str | Assigned city |
| `confidence` | float | Assignment confidence (0–1) |
| `order_count` | int | User's order count |
| `score_Delhi` … `score_Kolkata` | float | Raw match score for each city |

**Strategy:** Multi-factor scoring: cuisine similarity (×3 weight), meal period match, spending z-score proximity, cart size proximity, discount affinity proximity, single-item ratio, weekend ratio. Cold-start users (confidence < 0.5) use softmax-temperature random assignment (temp=1.5) to prevent all going to one city.

**Distribution:** Hyderabad 33.3%, Mumbai 24.6%, Delhi 22.5%, Bangalore 9.2%, Kolkata 7.2%, Pune 3.1%.

---

### 5.3 `city_item_popularity.csv`

**975 rows × 9 columns** — Per-city item rankings.

| Column | Type | Description |
|--------|------|-------------|
| `city` | str | City name |
| `item` | str | Item name |
| `city_order_count` | int | Times ordered in this city |
| `city_share` | float | Item's share of all orders in this city |
| `national_share` | float | Item's share of all orders nationally |
| `city_lift` | float | `city_share / national_share`. >1.3 = locally popular |
| `city_rank` | int | Rank within this city (1 = most popular) |
| `national_rank` | int | Rank across all cities |
| `is_local_favorite` | int | 1 if `city_lift > 1.3` AND `city_rank ≤ 20` |

**Stats:** 33 local favorites identified across 6 cities.

---

### 5.4 `city_cuisine_affinity.csv`

**41 rows × 5 columns** — Per-city cuisine distribution.

| Column | Type | Description |
|--------|------|-------------|
| `city` | str | City name |
| `cuisine` | str | Cuisine type |
| `count` | int | Number of item-orders for this cuisine in this city |
| `share` | float | Fraction of this city's orders going to this cuisine |
| `rank` | int | Rank within this city |

---

### 5.5 `orders_with_city.csv`

**21,341 rows × 45 columns** — Full enriched orders with city column + city-specific adjustments.

Same as `orders_enriched.csv` plus:

| Column | Type | Description |
|--------|------|-------------|
| `city` | str | Assigned city for this order's customer |

**Post-assignment adjustments applied:**

| Adjustment | Delhi | Mumbai | Bengaluru | Hyderabad | Pune | Kolkata |
|-----------|-------|--------|-----------|-----------|------|---------|
| Hour shift (min) | 0 | +30 | −60 | −15 | −30 | +15 |
| Price multiplier | 1.00 | 1.08 | 1.02 | 0.92 | 0.88 | 0.95 |
| Discount inflation | 0% | −2% | 0% | +1% | +5% | +2% |
| Distance multiplier | 1.00 | 1.15 | 1.10 | 0.90 | 0.85 | 0.95 |

Each adjustment has ±3–5% Gaussian noise added for realism.

---

## 6. Augmentation Strategies Deep Dive

### Problem Statement

| Bottleneck | Stat | Impact |
|-----------|------|--------|
| Single-item orders | 45.2% of orders | Produce 0 positive training samples |
| Cold-start users | 66.5% of users | No personalization signal |
| Gold data (returning + multi-item) | Only 34% of orders | Limits model generalization |

### Strategy A — Item-Swap Augmentation

**What:** Take real multi-item orders, randomly swap 1–2 items with same-category alternatives from the same restaurant.

| Parameter | Value |
|-----------|-------|
| `swap_ratio` | 0.5 (50% of eligible stages get a swap) |
| `max_swaps_per_order` | 2 |
| Next-item swap probability | 30% |
| Sample weight | **0.85** |

**Why it works:** Generates plausible "what if the user chose a different pizza but still added garlic bread?" scenarios. Same-category constraint ensures realistic substitutions. Zero API cost.

**Result:** 8,332 swapped stages → 49,992 training rows.

---

### Strategy B — Collaborative Synthesis

**What:** For light users (2–10 orders) with single-item orders, extend their order using the top co-purchased items from the co-purchase matrix.

| Parameter | Value |
|-----------|-------|
| Target users | 2–10 orders (light + regular returners) |
| Add-ons per order | 1–2 (60%/40% probability) |
| Category diversification | Add-on must be a different category than anchor and other add-ons |
| Top-N co-purchases used | Up to 5 per item |
| Sample weight | **0.75** |

**Why it works:** Leverages heavy users' multi-item behavior to bootstrap light users' profiles. The co-purchase matrix ensures add-ons are items that genuinely appear together in real orders.

**Result:** 4,570 collaborative stages → 27,420 training rows.

---

### Strategy C — Soft-Label

**What:** For single-item orders, generate a hypothetical "what would we recommend?" using the best co-purchased item from a different category.

| Parameter | Value |
|-----------|-------|
| `sample_frac` | 0.6 (sample 60% of single-item orders) |
| Label value | **0.5** (soft positive, not hard 1) |
| Selection criterion | Highest lift co-purchase from a *different* category |
| Sample weight | **0.50** |

**Why it works:** Converts previously-wasted single-item orders into weak training signal. The soft label (0.5) and low weight (0.5) prevent these hypotheticals from dominating training. Different-category constraint ensures meal complementarity.

**Result:** 5,667 soft-label stages → 34,002 training rows.

---

### Augmentation Summary

| Source | Stages | Training Rows | Positive Rows | Weight |
|--------|--------|---------------|---------------|--------|
| Original | 16,911 | 101,466 | 16,911 | 1.00 |
| Item-Swap | 8,332 | 49,992 | 8,332 | 0.85 |
| Collaborative | 4,570 | 27,420 | 4,570 | 0.75 |
| Soft-Label | 5,667 | 34,002 | 5,667 | 0.50 |
| **Total** | **35,480** | **212,880** | **35,480** | — |

**Multiplier:** 2.1× data volume increase.

---

## 7. Feature Engineering Strategies

### Summary of All Strategies Used

| Strategy | Features Affected | Method | Cost |
|----------|-------------------|--------|------|
| **Temporal extraction** | `order_hour`, `order_dow`, `is_weekend`, `meal_period` | Datetime parsing + bin mapping | Zero |
| **Cart parsing** | `num_items`, `num_unique_items`, `is_single_item`, `price_per_item` | Regex `re.split(r",\s*(?=\d+\s*x\s)")` to split multi-item strings | Zero |
| **Discount aggregation** | `total_discount`, `discount_pct`, `has_discount` | Sum of 4 discount columns; ratio to subtotal | Zero |
| **User profiling** | 7 user features | GroupBy aggregation on Customer ID | Zero |
| **LLM classification** | `category`, `veg_nonveg`, `cuisine`, `typical_role`, `flavor_profile`, `pairs_well_with` | Groq llama-3.3-70b-versatile, batched by restaurant (50 items/batch), structured JSON output | 6 API calls |
| **Meal completeness** | `completeness_score`, `has_*`, `missing_*`, `recommendation_priority` | Weighted slot detection using LLM categories | Zero |
| **Association rules** | `lift`, `confidence`, `support`, `co_count` | Standard market basket analysis on multi-item orders (min_support=3) | Zero |
| **Cart sequence simulation** | `stage`, `current_cart`, `next_item_added`, `completeness_at_stage` | Canonical ordering (category priority + price desc) to simulate progressive cart building | Zero |
| **Cross-features** | `fills_missing_slot`, `veg_compatible`, `cart_has_*` | Runtime cross-referencing between candidate item attributes and current cart state | Zero |
| **Co-purchase aggregation** | `max_lift`, `avg_lift`, `total_co_count`, `max_confidence`, `copurchase_pairs` | Aggregate association rules between candidate and ALL cart items | Zero |
| **City assignment** | `city`, per-city adjustments | Multi-factor user-city scoring with 7 behavioral dimensions + softmax cold-start handling | Zero |
| **Inverse geographic mapping** | `city_lift`, `city_rank`, `is_local_favorite` | Per-city item popularity normalized against national baseline | Zero |

---

## 8. How to Use for Model Training

### Recommended approach

```python
import pandas as pd

# ── Load primary training file ──
train = pd.read_csv("data_curation/output/augmented_training_data.csv")

# ── Separate features and labels ──
LABEL_COL  = "label"
WEIGHT_COL = "sample_weight"
META_COLS  = ["order_id", "candidate_item", "aug_type", "sample_weight"]

# Categorical columns (need encoding)
CAT_COLS = [
    "meal_period", "restaurant",
    "cand_category", "cand_veg_nonveg", "cand_cuisine",
    "cand_typical_role", "cand_flavor_profile",
]

# Numeric columns (ready to use)
NUM_COLS = [c for c in train.columns if c not in META_COLS + CAT_COLS + [LABEL_COL]]

X = train[NUM_COLS + CAT_COLS]
y = train[LABEL_COL]
w = train[WEIGHT_COL]

# ── Encode categoricals ──
# Option A: LabelEncoder / OrdinalEncoder for tree models
# Option B: OneHotEncoder for linear / neural models

# ── Train with sample weights ──
# XGBoost example:
import xgboost as xgb
dtrain = xgb.DMatrix(X_encoded, label=y, weight=w)
```

### Feature types for encoding

| Encoding | Columns |
|----------|---------|
| **Numeric (use as-is)** | `stage`, `items_in_cart`, `cart_value`, `completeness`, `order_hour`, `is_weekend`, `missing_main`, `missing_side`, `missing_drink`, `missing_dessert`, `cart_has_main`, `cart_has_side`, `cart_has_drink`, `cart_has_dessert`, `user_order_count`, `user_avg_order_value`, `user_avg_items`, `user_diversity`, `user_weekend_ratio`, `user_single_item_ratio`, `user_is_cold_start`, `cand_popularity_rank`, `cand_order_frequency`, `cand_solo_ratio`, `cand_avg_price`, `fills_missing_slot`, `veg_compatible`, `max_lift`, `avg_lift`, `total_co_count`, `max_confidence`, `copurchase_pairs` |
| **Categorical (encode)** | `meal_period`, `restaurant`, `cand_category`, `cand_veg_nonveg`, `cand_cuisine`, `cand_typical_role`, `cand_flavor_profile` |
| **ID / Meta (drop)** | `order_id`, `candidate_item`, `aug_type`, `sample_weight`, `label` |

### Optional: Enrich with city features

```python
# Join city to training data via order_id → user → city
user_city = pd.read_csv("data_curation/output/user_city_assignment.csv")
city_items = pd.read_csv("data_curation/output/city_item_popularity.csv")

# Add city_lift and is_local_favorite as features
# (requires joining on user's city + candidate_item)
```

---

## 9. How to Use for Inference

At inference time (recommending add-ons for a live order), construct a feature row for each candidate item:

1. **Cart context:** Build from the user's current cart (`items_in_cart`, `cart_value`, `completeness`, `cart_has_*`, `missing_*`)
2. **Candidate features:** Look up from `item_attributes.csv` + `item_stats.csv`
3. **User features:** Look up from `user_profiles.csv` (or use cold-start defaults if new user)
4. **Co-purchase features:** Compute from `copurchase_matrix.csv` (aggregate signals between candidate and cart items)
5. **Compatibility:** Compute `fills_missing_slot` and `veg_compatible` at runtime
6. **City features (optional):** Look up `city_lift` and `is_local_favorite` from `city_item_popularity.csv`

**Candidate pool:** All items from the same restaurant's menu, excluding items already in the cart.

**Output:** Rank candidates by predicted score; present top-K as "You might also like" suggestions.

---

*Generated from the CSAO data curation pipeline. See `data_curation/run_pipeline.py` for execution.*
