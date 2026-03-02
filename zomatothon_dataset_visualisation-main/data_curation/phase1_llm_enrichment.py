"""
Phase 1 — LLM Item Enrichment via Groq
=======================================
Classifies ~234 unique items by batching them per restaurant (6 calls)
plus one catch-all call.  Uses structured JSON output.

Produces: item_attributes.csv
Columns:  item, category, veg_nonveg, cuisine, typical_role,
          flavor_profile, pairs_well_with, restaurant
"""
import json
import time
import traceback

import pandas as pd
from groq import Groq

from data_curation.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    ITEM_ATTRIBUTES_CSV,
)
from data_curation.utils import get_unique_items, load_raw_data, parse_items


# ─────────────────────────────────────────────────────────────────────────
#  Prompt templates
# ─────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Indian food analyst. You will be given a list of food items from a restaurant menu on Zomato.

For each item, return a JSON array where each element has EXACTLY these fields:
- "item": the exact item name as given
- "category": one of ["main", "side", "drink", "dessert", "snack", "combo"]
- "veg_nonveg": one of ["veg", "non-veg", "egg"]
- "cuisine": most fitting cuisine (e.g. "North Indian", "Chinese", "Italian", "Continental", "American", "Pan-Indian", "Mughlai", "Street Food")
- "typical_role": one of ["anchor", "complement", "impulse"]
  - anchor: items people build a meal around (biryani, pizza, burger)
  - complement: items added alongside an anchor (raita, fries, garlic bread)
  - impulse: low-cost items added on impulse (cold drink, brownie, extra cheese)
- "flavor_profile": one of ["spicy", "mild", "sweet", "savory", "tangy", "rich", "neutral"]
- "pairs_well_with": a comma-separated string of 3-5 generic food categories/items that go well with this item (e.g. "raita, cold drink, salad, naan")

RULES:
1. Return ONLY a valid JSON array — no markdown, no explanation, no code blocks.
2. Every item in the input MUST appear in the output.
3. Use your food knowledge — the restaurant names give context about the menu."""


def _build_user_prompt(restaurant: str, items: list[str]) -> str:
    items_str = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(items))
    return f"""Restaurant: {restaurant}

Menu items ({len(items)} items):
{items_str}

Return the JSON array for ALL {len(items)} items."""


# ─────────────────────────────────────────────────────────────────────────
#  LLM call with retry
# ─────────────────────────────────────────────────────────────────────────
def _call_groq(client: Groq, restaurant: str, items: list[str], max_retries: int = 3) -> list[dict]:
    """Call Groq API for a batch of items, parse JSON, retry on failure."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_user_prompt(restaurant, items)},
                ],
                temperature=0.1,
                max_tokens=8000,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()

            # Parse — handle both raw array and {"items": [...]} wrapper
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                # Look for the array inside the dict
                for key in ("items", "menu_items", "data", "results"):
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break
                else:
                    # If dict has item-like keys, wrap in list
                    if "item" in parsed:
                        parsed = [parsed]
                    else:
                        raise ValueError(f"Unexpected JSON structure: {list(parsed.keys())}")

            if not isinstance(parsed, list):
                raise ValueError(f"Expected list, got {type(parsed)}")

            # Validate all items are present
            returned_items = {d.get("item", "").strip() for d in parsed}
            missing = [it for it in items if it not in returned_items]
            if missing and attempt < max_retries - 1:
                print(f"    ⚠ Missing {len(missing)} items, retrying …")
                time.sleep(2)
                continue

            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            print(f"    ⚠ Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                traceback.print_exc()
                return []
        except Exception as e:
            print(f"    ⚠ API error attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                traceback.print_exc()
                return []

    return []


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────
def run_phase1() -> dict:
    """Enrich all unique items via Groq LLM. Returns summary dict."""
    print("=" * 60)
    print("PHASE 1 — LLM Item Enrichment")
    print("=" * 60)

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API key not found in .env — cannot run Phase 1")

    client = Groq(api_key=GROQ_API_KEY)

    # Load data & group items by restaurant
    df = load_raw_data()
    restaurant_items: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        rest = row["Restaurant name"]
        parsed = parse_items(row["Items in order"]) if pd.notna(row["Items in order"]) else []
        for p in parsed:
            restaurant_items.setdefault(rest, set()).add(p["item"])

    total_unique = sum(len(v) for v in restaurant_items.values())
    print(f"  {total_unique} unique items across {len(restaurant_items)} restaurants\n")

    all_results = []
    for i, (restaurant, items_set) in enumerate(sorted(restaurant_items.items()), 1):
        items = sorted(items_set)
        print(f"  [{i}/{len(restaurant_items)}] {restaurant} ({len(items)} items) …")

        # Split into batches of 50 for safety (most restaurants have <80 items)
        for batch_start in range(0, len(items), 50):
            batch = items[batch_start:batch_start + 50]
            results = _call_groq(client, restaurant, batch)
            for r in results:
                r["restaurant"] = restaurant
            all_results.extend(results)
            print(f"         Got {len(results)} classifications")

            # Rate limiting — Groq free tier is generous but let's be safe
            if batch_start + 50 < len(items):
                time.sleep(2)

        # Pause between restaurants
        time.sleep(1)

    # Build DataFrame
    attr_df = pd.DataFrame(all_results)

    # Normalize columns
    expected_cols = ["item", "category", "veg_nonveg", "cuisine",
                     "typical_role", "flavor_profile", "pairs_well_with", "restaurant"]
    for col in expected_cols:
        if col not in attr_df.columns:
            attr_df[col] = "unknown"

    # Deduplicate (same item might appear in multiple restaurants)
    attr_df = attr_df.drop_duplicates(subset=["item"], keep="first")

    # Validate categories
    valid_categories = {"main", "side", "drink", "dessert", "snack", "combo"}
    attr_df["category"] = attr_df["category"].str.lower().str.strip()
    attr_df.loc[~attr_df["category"].isin(valid_categories), "category"] = "main"

    valid_veg = {"veg", "non-veg", "egg"}
    attr_df["veg_nonveg"] = attr_df["veg_nonveg"].str.lower().str.strip()
    attr_df.loc[~attr_df["veg_nonveg"].isin(valid_veg), "veg_nonveg"] = "non-veg"

    valid_roles = {"anchor", "complement", "impulse"}
    attr_df["typical_role"] = attr_df["typical_role"].str.lower().str.strip()
    attr_df.loc[~attr_df["typical_role"].isin(valid_roles), "typical_role"] = "anchor"

    attr_df = attr_df[expected_cols].reset_index(drop=True)
    attr_df.to_csv(ITEM_ATTRIBUTES_CSV, index=False)

    summary = {
        "items_classified": len(attr_df),
        "api_calls": len(restaurant_items),
        "categories": attr_df["category"].value_counts().to_dict(),
        "veg_split": attr_df["veg_nonveg"].value_counts().to_dict(),
        "roles": attr_df["typical_role"].value_counts().to_dict(),
    }
    print(f"\n  Summary: {json.dumps(summary, indent=2)}")
    print(f"  → Saved {ITEM_ATTRIBUTES_CSV.name}")
    print("Phase 1 complete ✓\n")
    return summary


if __name__ == "__main__":
    run_phase1()
