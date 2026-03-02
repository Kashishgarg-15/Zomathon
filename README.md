# Cart Super Add-On (CSAO) Rail Recommendation System

**Zomathon Hackathon Submission** — ML-powered cross-sell recommendations for Zomato's Cart Page.

## Key Results

| Metric | Value |
|--------|-------|
| AUC | **0.902** |
| NDCG@10 | **0.876** |
| Hit@10 | 0.999 |
| P95 Latency | 18.8ms |
| AOV Lift | +Rs 103/order (+23.8%) |
| vs Best Baseline | +19.5% AUC, +11.6% NDCG@10 |

## Quick Start

**Start here:** Open [`CSAO_Recommendation_System.ipynb`](CSAO_Recommendation_System.ipynb) — the complete walkthrough notebook covering data exploration, feature engineering, model training, evaluation, and business impact.

**Submission PDF:** [`analysis_output/submission.pdf`](analysis_output/submission.pdf) (599 KB, includes embedded plots)

---

## Repository Structure

```
├── CSAO_Recommendation_System.ipynb    # >>> START HERE — Full pipeline notebook
├── README.md                           # This file
├── DATA_DICTIONARY.md                  # Complete data pipeline documentation
│
├── train_model_v3.py                   # GBDT training (LightGBM + XGBoost + Optuna)
├── train_dcn_v2.py                     # DCN-v2 deep learning model
├── final_ensemble_v2.py                # 3-model ensemble (LGB + XGB + DCN-v2)
├── generate_llm_features.py            # Sentence-transformer feature engineering
├── baseline_and_analysis.py            # 4 baselines + business impact analysis
├── inference_benchmark.py              # Production latency benchmark
├── inference.py                        # Production inference engine
├── generate_pdf.py                     # Converts submission.md to PDF with embedded plots
│
├── model_output_v3/                    # GBDT model results & plots
├── model_output_dcn_v2/                # DCN-v2 model results & plots
├── model_output_final_v2/              # Final ensemble results & plots
├── analysis_output/                    # Baselines, latency, submission PDF
│
├── city_cuisine_affinity.csv           # Per-city cuisine distributions
├── city_item_popularity.csv            # Per-city item rankings
└── user_city_assignment.csv            # User-to-city mapping
```

## Approach Summary

### 1. Dataset Curation (No dataset provided — built from scratch)
- Started with 21,321 raw orders from 6 Delhi NCR restaurants
- 6-phase pipeline: Derived features → LLM item enrichment → Meal completeness → Cart simulation → Training construction → City enrichment
- Final: **212,880 training rows** with 7 augmentation types and soft labels

### 2. Feature Engineering (70 features)
- **38 base features**: Cart context, user behavior, candidate item stats, co-purchase signals, temporal
- **9 engineered interactions**: price_ratio, hour_sin/cos, completeness_gap, etc.
- **23 LLM-derived semantic features**: Sentence-transformer embeddings (all-MiniLM-L6-v2), PCA reduction, K-means clustering, compatibility scores

### 3. Model Architecture
```
3-Model Ensemble: Score = 0.40×LGB + 0.40×XGB + 0.20×DCN-v2
├── LightGBM  (leaf-wise, 30 Optuna trials)
├── XGBoost   (level-wise, 25 Optuna trials)
└── DCN-v2    (2 Cross + [512,256,128] Deep, 10 Optuna trials)
```

### 4. Evaluation vs Baselines

| Model | AUC | NDCG@10 |
|-------|-----|---------|
| Random Baseline | 0.497 | 0.621 |
| Popularity Baseline | 0.733 | 0.699 |
| Heuristic Scoring | 0.590 | 0.710 |
| Co-purchase Signal | 0.755 | 0.785 |
| **Our 3-Model Ensemble** | **0.902** | **0.876** |

### 5. Production Readiness
- **P95 latency**: 18.8ms (SLA < 300ms → PASS)
- **Throughput**: 512 req/sec (8 workers)
- **Fallback cascade**: Full Ensemble → GBDT-only → Popularity → No rail
- **A/B testing framework**: Shadow mode → Canary (10%) → 50/50 split → Full rollout

## Tech Stack
- Python, LightGBM, XGBoost, PyTorch (DCN-v2), Optuna, sentence-transformers
- Groq API (llama-3.3-70b) for item enrichment during data curation