"""
Shared configuration for the data curation pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "Food_Delivary_dataSet"
RAW_CSV  = DATA_DIR / "order_history_kaggle_data.csv"
OUTPUT_DIR = ROOT_DIR / "data_curation" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── API ──────────────────────────────────────────────────────────────────
load_dotenv(ROOT_DIR / ".env")
GROQ_API_KEY = os.getenv("GROQ_API", "").strip()
GROQ_MODEL   = "llama-3.3-70b-versatile"   # fast + cheap on Groq

# ── Constants ────────────────────────────────────────────────────────────
MEAL_PERIOD_BINS = {
    "breakfast":  (6, 10),
    "lunch":      (11, 14),
    "snack":      (14, 17),
    "dinner":     (18, 22),
    "late_night": (22, 6),   # wraps around midnight
}

# Canonical meal component ordering for cart sequences
CATEGORY_ORDER = ["main", "side", "drink", "dessert", "snack", "combo"]

# Completeness weights
COMPLETENESS_WEIGHTS = {
    "main":    0.40,
    "side":    0.25,
    "drink":   0.20,
    "dessert": 0.15,
}

# Negative sampling ratio for training data (negatives per positive)
NEG_SAMPLE_RATIO = 5

# ── Output file names ───────────────────────────────────────────────────
ITEM_ATTRIBUTES_CSV   = OUTPUT_DIR / "item_attributes.csv"
ORDERS_ENRICHED_CSV   = OUTPUT_DIR / "orders_enriched.csv"
USER_PROFILES_CSV     = OUTPUT_DIR / "user_profiles.csv"
ITEM_STATS_CSV        = OUTPUT_DIR / "item_stats.csv"
COPURCHASE_CSV        = OUTPUT_DIR / "copurchase_matrix.csv"
COMPLETENESS_CSV      = OUTPUT_DIR / "order_completeness.csv"
CART_SEQUENCES_CSV    = OUTPUT_DIR / "cart_sequences.csv"
TRAINING_DATA_CSV     = OUTPUT_DIR / "training_data.csv"
AUGMENTED_TRAINING_CSV = OUTPUT_DIR / "augmented_training_data.csv"
PIPELINE_LOG          = OUTPUT_DIR / "pipeline_log.json"

# Phase 5 — City assignment outputs
CITY_PROFILES_CSV       = OUTPUT_DIR / "city_profiles.csv"
USER_CITY_CSV           = OUTPUT_DIR / "user_city_assignment.csv"
CITY_ITEM_POPULARITY_CSV = OUTPUT_DIR / "city_item_popularity.csv"
CITY_CUISINE_AFFINITY_CSV = OUTPUT_DIR / "city_cuisine_affinity.csv"
ORDERS_WITH_CITY_CSV    = OUTPUT_DIR / "orders_with_city.csv"
