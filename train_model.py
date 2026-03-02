"""
CSAO Rail Recommendation System - Full Training Pipeline
========================================================
Model: LightGBM + XGBoost (ensemble)
Dataset: training_data_with_city.csv (212,880 x 43)
Target: label (0.0 / 0.5 / 1.0)
"""

import sys
import io
# Log to file for Windows compatibility
log_file = open('training_log.txt', 'w', encoding='utf-8')
class TeeLog:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = TeeLog(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace'), log_file)
sys.stderr = TeeLog(io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace'), log_file)

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, log_loss
)
import optuna
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
import os
import time

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)

OUT_DIR = "model_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Loading data...")
t0 = time.time()

df = pd.read_csv("training_data_with_city.csv")
print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} cols in {time.time()-t0:.1f}s")

# ════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING — Additional Features
# ════════════════════════════════════════════════════════════════
print("\nSTEP 2: Engineering additional features...")

# 2a. Cyclical hour encoding (captures 23:00 ≈ 0:00 proximity)
df['hour_sin'] = np.sin(2 * np.pi * df['order_hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['order_hour'] / 24)

# 2b. Price ratio: candidate price relative to cart value
df['price_ratio'] = df['cand_avg_price'] / (df['cart_value'] + 1)

# 2c. Popularity × co-purchase interaction
df['popularity_x_lift'] = df['cand_order_frequency'] * df['max_lift']

# 2d. Cart completeness gap (how much room for add-ons)
df['completeness_gap'] = 1.0 - df['completeness']

# 2e. User spending vs candidate price alignment
df['price_vs_user_avg'] = df['cand_avg_price'] / (df['user_avg_order_value'] + 1)

# 2f. Is candidate a complement/impulse AND fills a missing slot?
df['complement_fills_gap'] = ((df['cand_typical_role'] != 'anchor') & 
                               (df['fills_missing_slot'] == 1)).astype(int)

# 2g. Cart size category
df['cart_size_bucket'] = pd.cut(df['items_in_cart'], bins=[0, 1, 2, 3, 100], 
                                 labels=['single', 'pair', 'triple', 'large']).astype(str)

# 2h. City-item popularity signal strength
df['city_item_signal'] = df['city_lift'] * df['is_local_favorite']

print(f"  Added 9 new features -> {df.shape[1]} total columns")

# ════════════════════════════════════════════════════════════════
# 3. DEFINE COLUMN ROLES
# ════════════════════════════════════════════════════════════════
LABEL_COL = 'label'
WEIGHT_COL = 'sample_weight'
GROUP_COL = 'order_id'  # for group-based split & ranking

META_COLS = ['order_id', 'candidate_item', 'aug_type', 'sample_weight', 'label']

CAT_COLS = ['meal_period', 'restaurant', 'cand_category', 'cand_veg_nonveg',
            'cand_cuisine', 'cand_typical_role', 'cand_flavor_profile', 'city',
            'cart_size_bucket']

NUM_COLS = [c for c in df.columns if c not in META_COLS + CAT_COLS]

FEATURE_COLS = NUM_COLS + CAT_COLS
print(f"\n  Features: {len(NUM_COLS)} numeric + {len(CAT_COLS)} categorical = {len(FEATURE_COLS)} total")

# ════════════════════════════════════════════════════════════════
# 4. ENCODE CATEGORICALS
# ════════════════════════════════════════════════════════════════
print("\nSTEP 3: Encoding categoricals...")

label_encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} classes")

# Save encoders
joblib.dump(label_encoders, os.path.join(OUT_DIR, 'label_encoders.pkl'))

# ════════════════════════════════════════════════════════════════
# 5. TEMPORAL-AWARE GROUP SPLIT (no data leakage)
# ════════════════════════════════════════════════════════════════
print("\nSTEP 4: Train/Val/Test split (group-based, no leakage)...")

# Extract base order_id (strip syn_/soft_ prefixes) for proper grouping
df['base_order_id'] = df['order_id'].str.replace(r'^(syn_|soft_)', '', regex=True)

# Assign unique group IDs for splitting
unique_orders = df['base_order_id'].unique()
np.random.shuffle(unique_orders)

n = len(unique_orders)
train_orders = set(unique_orders[:int(0.70 * n)])
val_orders = set(unique_orders[int(0.70 * n):int(0.85 * n)])
test_orders = set(unique_orders[int(0.85 * n):])

train_mask = df['base_order_id'].isin(train_orders)
val_mask = df['base_order_id'].isin(val_orders)
test_mask = df['base_order_id'].isin(test_orders)

X_train = df.loc[train_mask, FEATURE_COLS]
y_train = df.loc[train_mask, LABEL_COL]
w_train = df.loc[train_mask, WEIGHT_COL]

X_val = df.loc[val_mask, FEATURE_COLS]
y_val = df.loc[val_mask, LABEL_COL]
w_val = df.loc[val_mask, WEIGHT_COL]

X_test = df.loc[test_mask, FEATURE_COLS]
y_test = df.loc[test_mask, LABEL_COL]
w_test = df.loc[test_mask, WEIGHT_COL]

# Save test order_ids and candidate items for ranking evaluation
test_order_ids = df.loc[test_mask, 'order_id'].values
test_candidates = df.loc[test_mask, 'candidate_item'].values

print(f"  Train: {X_train.shape[0]:,} rows ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"  Val:   {X_val.shape[0]:,} rows ({X_val.shape[0]/len(df)*100:.1f}%)")
print(f"  Test:  {X_test.shape[0]:,} rows ({X_test.shape[0]/len(df)*100:.1f}%)")

# Verify no leakage
train_base = set(df.loc[train_mask, 'base_order_id'])
val_base = set(df.loc[val_mask, 'base_order_id'])
test_base = set(df.loc[test_mask, 'base_order_id'])
assert len(train_base & val_base) == 0, "Train-Val leakage!"
assert len(train_base & test_base) == 0, "Train-Test leakage!"
assert len(val_base & test_base) == 0, "Val-Test leakage!"
print("  ✅ No data leakage between splits")

# ════════════════════════════════════════════════════════════════
# 6. LIGHTGBM — HYPERPARAMETER TUNING WITH OPTUNA
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: LightGBM Hyperparameter Tuning (Optuna, 40 trials)...")
print("=" * 70)

lgb_train = lgb.Dataset(X_train, label=y_train, weight=w_train, 
                         categorical_feature=CAT_COLS, free_raw_data=False)
lgb_val = lgb.Dataset(X_val, label=y_val, weight=w_val,
                       categorical_feature=CAT_COLS, free_raw_data=False)


def lgb_objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': SEED,
        'feature_pre_filter': False,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 6.0),
    }
    
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(
        params, lgb_train, 
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )
    
    y_pred = model.predict(X_val)
    y_val_binary = (y_val > 0.25).astype(int)
    auc = roc_auc_score(y_val_binary, y_pred, sample_weight=w_val)
    return auc


t1 = time.time()
study_lgb = optuna.create_study(direction='maximize', 
                                 study_name='lgb_csao',
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(lgb_objective, n_trials=40, show_progress_bar=False)

print(f"\n  Best LightGBM AUC: {study_lgb.best_value:.6f}")
print(f"  Tuning time: {time.time()-t1:.1f}s")
print(f"  Best params: {json.dumps(study_lgb.best_params, indent=2)}")

# ════════════════════════════════════════════════════════════════
# 7. TRAIN FINAL LIGHTGBM WITH BEST PARAMS
# ════════════════════════════════════════════════════════════════
print("\nSTEP 6: Training final LightGBM model...")

best_lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'seed': SEED,
    'feature_pre_filter': False,
    **study_lgb.best_params
}

callbacks = [lgb.early_stopping(80, verbose=False), lgb.log_evaluation(100)]
lgb_model = lgb.train(
    best_lgb_params, lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_val],
    callbacks=callbacks,
)

print(f"  Best iteration: {lgb_model.best_iteration}")
lgb_model.save_model(os.path.join(OUT_DIR, 'lgb_model.txt'))

# ════════════════════════════════════════════════════════════════
# 8. XGBOOST — HYPERPARAMETER TUNING
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: XGBoost Hyperparameter Tuning (Optuna, 40 trials)...")
print("=" * 70)


def xgb_objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'seed': SEED,
        'verbosity': 0,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 2.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 6.0),
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, enable_categorical=False)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val, enable_categorical=False)
    
    model = xgb.train(
        params, dtrain,
        num_boost_round=1000,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    y_pred = model.predict(dval)
    y_val_binary = (y_val > 0.25).astype(int)
    auc = roc_auc_score(y_val_binary, y_pred, sample_weight=w_val)
    return auc


t2 = time.time()
study_xgb = optuna.create_study(direction='maximize',
                                 study_name='xgb_csao',
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(xgb_objective, n_trials=40, show_progress_bar=False)

print(f"\n  Best XGBoost AUC: {study_xgb.best_value:.6f}")
print(f"  Tuning time: {time.time()-t2:.1f}s")
print(f"  Best params: {json.dumps(study_xgb.best_params, indent=2)}")

# ════════════════════════════════════════════════════════════════
# 9. TRAIN FINAL XGBOOST
# ════════════════════════════════════════════════════════════════
print("\nSTEP 8: Training final XGBoost model...")

best_xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'seed': SEED,
    'verbosity': 0,
    **study_xgb.best_params
}

dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)
dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test)

xgb_model = xgb.train(
    best_xgb_params, dtrain,
    num_boost_round=2000,
    evals=[(dval, 'val')],
    early_stopping_rounds=80,
    verbose_eval=200
)

xgb_model.save_model(os.path.join(OUT_DIR, 'xgb_model.json'))

# ════════════════════════════════════════════════════════════════
# 10. ENSEMBLE PREDICTIONS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 9: Generating Ensemble Predictions...")
print("=" * 70)

lgb_pred_test = lgb_model.predict(X_test)
xgb_pred_test = xgb_model.predict(dtest)

lgb_pred_val = lgb_model.predict(X_val)
xgb_pred_val = xgb_model.predict(dval)

# Find optimal ensemble weight on validation set
best_w, best_auc = 0.5, 0.0
y_val_binary = (y_val > 0.25).astype(int)
for w in np.arange(0.1, 0.95, 0.05):
    ens_pred = w * lgb_pred_val + (1 - w) * xgb_pred_val
    auc = roc_auc_score(y_val_binary, ens_pred, sample_weight=w_val)
    if auc > best_auc:
        best_w, best_auc = w, auc

print(f"  Optimal ensemble weight: LGB={best_w:.2f}, XGB={1-best_w:.2f}")
print(f"  Validation ensemble AUC: {best_auc:.6f}")

# Final ensemble on test
ensemble_pred = best_w * lgb_pred_test + (1 - best_w) * xgb_pred_test

# ════════════════════════════════════════════════════════════════
# 11. COMPREHENSIVE EVALUATION ON TEST SET
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 10: Comprehensive Evaluation on Test Set")
print("=" * 70)

y_test_binary = (y_test > 0.25).astype(int)


def compute_all_metrics(y_true, y_pred, w, model_name):
    """Compute AUC, Precision@K, Recall@K, NDCG@K, Hit-Rate@K"""
    auc = roc_auc_score(y_true, y_pred, sample_weight=w)
    ap = average_precision_score(y_true, y_pred, sample_weight=w)
    ll = log_loss(y_true, np.clip(y_pred, 1e-7, 1-1e-7), sample_weight=w)
    
    print(f"\n  {'─'*50}")
    print(f"  {model_name}")
    print(f"  {'─'*50}")
    print(f"  AUC-ROC:            {auc:.6f}")
    print(f"  Average Precision:  {ap:.6f}")
    print(f"  Log Loss:           {ll:.6f}")
    
    return {'auc': auc, 'avg_precision': ap, 'log_loss': ll}


def ranking_metrics_per_group(order_ids, y_true, y_pred, Ks=[3, 5, 10]):
    """Compute Precision@K, Recall@K, NDCG@K, HitRate@K grouped by order."""
    results_df = pd.DataFrame({
        'order_id': order_ids, 'y_true': y_true, 'y_pred': y_pred
    })
    
    metrics = {f'precision@{k}': [] for k in Ks}
    metrics.update({f'recall@{k}': [] for k in Ks})
    metrics.update({f'ndcg@{k}': [] for k in Ks})
    metrics.update({f'hit_rate@{k}': [] for k in Ks})
    
    for oid, grp in results_df.groupby('order_id'):
        if grp['y_true'].sum() == 0:
            continue  # skip groups with no positives
        
        sorted_grp = grp.sort_values('y_pred', ascending=False)
        relevance = sorted_grp['y_true'].values
        
        for k in Ks:
            topk = relevance[:k]
            n_relevant = (grp['y_true'] == 1).sum()
            
            # Precision@K
            prec = topk.sum() / k
            metrics[f'precision@{k}'].append(prec)
            
            # Recall@K
            rec = topk.sum() / max(n_relevant, 1)
            metrics[f'recall@{k}'].append(rec)
            
            # Hit Rate@K (at least 1 positive in top-K)
            hit = 1 if topk.sum() > 0 else 0
            metrics[f'hit_rate@{k}'].append(hit)
            
            # NDCG@K
            dcg = np.sum(relevance[:k] / np.log2(np.arange(2, k + 2)))
            ideal = np.sort(grp['y_true'].values)[::-1]
            idcg = np.sum(ideal[:k] / np.log2(np.arange(2, k + 2)))
            ndcg = dcg / max(idcg, 1e-10)
            metrics[f'ndcg@{k}'].append(ndcg)
    
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics


# Individual model metrics
lgb_metrics = compute_all_metrics(y_test_binary, lgb_pred_test, w_test, "LightGBM")
xgb_metrics = compute_all_metrics(y_test_binary, xgb_pred_test, w_test, "XGBoost")
ens_metrics = compute_all_metrics(y_test_binary, ensemble_pred, w_test, "Ensemble (LGB+XGB)")

# Ranking metrics
print("\n  Ranking Metrics (per-order grouped):")
for name, preds in [("LightGBM", lgb_pred_test), ("XGBoost", xgb_pred_test), ("Ensemble", ensemble_pred)]:
    rank_m = ranking_metrics_per_group(test_order_ids, y_test_binary.values, preds)
    print(f"\n  {name}:")
    for k, v in sorted(rank_m.items()):
        print(f"    {k:20s}: {v:.4f}")
    if name == "Ensemble":
        ensemble_ranking = rank_m

# ════════════════════════════════════════════════════════════════
# 12. SEGMENT-LEVEL ERROR ANALYSIS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 11: Segment-Level Error Analysis")
print("=" * 70)

test_df = df.loc[test_mask].copy()
test_df['pred'] = ensemble_pred
test_df['y_binary'] = y_test_binary.values

# Decode categoricals for readable segments
for col in ['meal_period', 'restaurant', 'city', 'cand_category']:
    test_df[col + '_name'] = label_encoders[col].inverse_transform(test_df[col])

# Segment analysis function
def segment_auc(df_seg, col, pred_col='pred', true_col='y_binary'):
    print(f"\n  By {col}:")
    for name, grp in df_seg.groupby(col):
        if grp[true_col].nunique() < 2:
            print(f"    {name:25s}: n={len(grp):>6,}  (skipped - single class)")
            continue
        auc = roc_auc_score(grp[true_col], grp[pred_col])
        pos_rate = grp[true_col].mean()
        print(f"    {name:25s}: AUC={auc:.4f}  n={len(grp):>6,}  pos_rate={pos_rate:.3f}")

segment_auc(test_df, 'meal_period_name')
segment_auc(test_df, 'restaurant_name')
segment_auc(test_df, 'city_name')
segment_auc(test_df, 'cand_category_name')

# Cold start vs returning
print("\n  By user_is_cold_start:")
for cs in [0, 1]:
    grp = test_df[test_df['user_is_cold_start'] == cs]
    if grp['y_binary'].nunique() < 2:
        continue
    label = "Cold-start" if cs else "Returning"
    auc = roc_auc_score(grp['y_binary'], grp['pred'])
    print(f"    {label:25s}: AUC={auc:.4f}  n={len(grp):>6,}")

# By augmentation type
print("\n  By aug_type:")
for at in test_df['aug_type'].unique():
    grp = test_df[test_df['aug_type'] == at]
    if grp['y_binary'].nunique() < 2:
        continue
    auc = roc_auc_score(grp['y_binary'], grp['pred'])
    print(f"    {at:25s}: AUC={auc:.4f}  n={len(grp):>6,}")

# ════════════════════════════════════════════════════════════════
# 13. FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 12: Feature Importance")
print("=" * 70)

# LightGBM importance
lgb_imp = pd.DataFrame({
    'feature': FEATURE_COLS,
    'gain': lgb_model.feature_importance(importance_type='gain'),
    'split': lgb_model.feature_importance(importance_type='split')
}).sort_values('gain', ascending=False)

print("\n  Top 20 Features (LightGBM - Gain):")
for i, row in lgb_imp.head(20).iterrows():
    print(f"    {row['feature']:35s}: gain={row['gain']:>10.1f}  splits={row['split']:>6d}")

# Save importance
lgb_imp.to_csv(os.path.join(OUT_DIR, 'feature_importance_lgb.csv'), index=False)

# ════════════════════════════════════════════════════════════════
# 14. PLOTS
# ════════════════════════════════════════════════════════════════
print("\nSTEP 13: Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Feature Importance (top 20)
top20 = lgb_imp.head(20).sort_values('gain')
axes[0, 0].barh(top20['feature'], top20['gain'], color='steelblue')
axes[0, 0].set_title('Top 20 Features (LightGBM Gain)', fontsize=12)
axes[0, 0].set_xlabel('Gain')

# Plot 2: Score Distribution by Label
for label_val, color, name in [(0, 'red', 'Negative'), (1, 'green', 'Positive')]:
    mask = y_test_binary == label_val
    axes[0, 1].hist(ensemble_pred[mask], bins=50, alpha=0.5, color=color, 
                     label=name, density=True)
axes[0, 1].set_title('Ensemble Score Distribution', fontsize=12)
axes[0, 1].set_xlabel('Predicted Score')
axes[0, 1].legend()

# Plot 3: Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test_binary, ensemble_pred)
axes[1, 0].plot(rec, prec, color='darkorange', lw=2)
axes[1, 0].set_title(f'Precision-Recall Curve (AP={ens_metrics["avg_precision"]:.4f})', fontsize=12)
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')

# Plot 4: Ranking Metrics Bar Chart
rank_names = list(ensemble_ranking.keys())
rank_vals = list(ensemble_ranking.values())
colors = ['#2196F3' if '@3' in n else '#4CAF50' if '@5' in n else '#FF9800' for n in rank_names]
axes[1, 1].barh(rank_names, rank_vals, color=colors)
axes[1, 1].set_title('Ensemble Ranking Metrics', fontsize=12)
axes[1, 1].set_xlim(0, 1)
for i, v in enumerate(rank_vals):
    axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'evaluation_plots.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {OUT_DIR}/evaluation_plots.png")

# ════════════════════════════════════════════════════════════════
# 15. SAVE ALL RESULTS
# ════════════════════════════════════════════════════════════════
print("\nSTEP 14: Saving all results...")

results = {
    'dataset': {
        'total_rows': int(df.shape[0]),
        'features': len(FEATURE_COLS),
        'train_size': int(X_train.shape[0]),
        'val_size': int(X_val.shape[0]),
        'test_size': int(X_test.shape[0]),
    },
    'lightgbm': {
        'best_iteration': lgb_model.best_iteration,
        'best_params': study_lgb.best_params,
        'test_metrics': lgb_metrics,
    },
    'xgboost': {
        'best_params': study_xgb.best_params,
        'test_metrics': xgb_metrics,
    },
    'ensemble': {
        'lgb_weight': best_w,
        'xgb_weight': 1 - best_w,
        'test_metrics': ens_metrics,
        'ranking_metrics': ensemble_ranking,
    },
    'feature_cols': FEATURE_COLS,
    'cat_cols': CAT_COLS,
    'num_cols': NUM_COLS,
}

with open(os.path.join(OUT_DIR, 'training_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save test predictions for further analysis
pred_df = pd.DataFrame({
    'order_id': test_order_ids,
    'candidate_item': test_candidates,
    'y_true': y_test.values,
    'y_binary': y_test_binary.values,
    'lgb_pred': lgb_pred_test,
    'xgb_pred': xgb_pred_test,
    'ensemble_pred': ensemble_pred,
})
pred_df.to_csv(os.path.join(OUT_DIR, 'test_predictions.csv'), index=False)

print(f"\n  Saved to {OUT_DIR}/:")
print(f"    - lgb_model.txt (LightGBM model)")
print(f"    - xgb_model.json (XGBoost model)")
print(f"    - label_encoders.pkl (categorical encoders)")
print(f"    - training_results.json (all metrics & params)")
print(f"    - test_predictions.csv (predictions for analysis)")
print(f"    - feature_importance_lgb.csv")
print(f"    - evaluation_plots.png")

total_time = time.time() - t0
print(f"\n{'=' * 70}")
print(f"PIPELINE COMPLETE — Total time: {total_time/60:.1f} minutes")
print(f"{'=' * 70}")
