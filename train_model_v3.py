# -*- coding: utf-8 -*-
"""CSAO Rail v3 - LLM Features + 30 LGB + 25 XGB Optuna trials"""
import os, sys, time, json, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, log_loss
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

SEED = 42
np.random.seed(SEED)
OUT = "model_output_v3"
os.makedirs(OUT, exist_ok=True)

L = []
def log(m=""):
    print(m, flush=True)
    L.append(str(m))

# ----------------------------------------------------------------
def compute_ndcg(order_ids, y_true, y_pred, k=10):
    tmp = pd.DataFrame({'o': order_ids, 'y': y_true, 'p': y_pred})
    vals = []
    for _, g in tmp.groupby('o'):
        if g['y'].sum() == 0: continue
        s = g.sort_values('p', ascending=False)
        rel = s['y'].values
        tk = rel[:k]; n = len(tk)
        dcg = np.sum(tk / np.log2(np.arange(2, n + 2)))
        irel = np.sort(g['y'].values)[::-1][:k]; ni = len(irel)
        idcg = np.sum(irel / np.log2(np.arange(2, ni + 2)))
        vals.append(dcg / max(idcg, 1e-10))
    return float(np.mean(vals)) if vals else 0.0

def ranking_metrics(oids, yt, yp, Ks=[3, 5, 10]):
    r = pd.DataFrame({'o': oids, 't': yt, 'p': yp})
    M = {f'{m}@{k}': [] for k in Ks for m in ['prec', 'rec', 'ndcg', 'hit']}
    for _, g in r.groupby('o'):
        if g['t'].sum() == 0: continue
        s = g.sort_values('p', ascending=False)
        rel = s['t'].values; np_ = (g['t'] == 1).sum()
        for k in Ks:
            tk = rel[:k]; n_tk = len(tk)
            M[f'prec@{k}'].append(tk.sum() / k)
            M[f'rec@{k}'].append(tk.sum() / max(np_, 1))
            M[f'hit@{k}'].append(1 if tk.sum() > 0 else 0)
            disc = np.log2(np.arange(2, n_tk + 2))
            dcg = np.sum(tk / disc)
            irel = np.sort(g['t'].values)[::-1][:k]; n_ir = len(irel)
            idisc = np.log2(np.arange(2, n_ir + 2))
            idcg = np.sum(irel / idisc)
            M[f'ndcg@{k}'].append(dcg / max(idcg, 1e-10))
    return {k: float(np.mean(v)) for k, v in M.items()}

# ================================================================
# STEP 1: LOAD LLM-ENRICHED DATA
# ================================================================
log("=" * 60)
log("STEP 1: Loading LLM-enriched data...")
t0 = time.time()

if os.path.exists("training_data_llm.csv"):
    df = pd.read_csv("training_data_llm.csv")
    log(f"  Using LLM-enriched data: {df.shape}")
    HAS_LLM = True
else:
    df = pd.read_csv("training_data_with_city.csv")
    log(f"  Using original data (no LLM features): {df.shape}")
    HAS_LLM = False

# ================================================================
# STEP 2: FEATURE ENGINEERING
# ================================================================
log("\nSTEP 2: Feature engineering...")
df['hour_sin'] = np.sin(2 * np.pi * df['order_hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['order_hour'] / 24)
df['price_ratio'] = df['cand_avg_price'] / (df['cart_value'] + 1)
df['popularity_x_lift'] = df['cand_order_frequency'] * df['max_lift']
df['completeness_gap'] = 1.0 - df['completeness']
df['price_vs_user_avg'] = df['cand_avg_price'] / (df['user_avg_order_value'] + 1)
df['complement_fills_gap'] = ((df['cand_typical_role'] != 'anchor') & (df['fills_missing_slot'] == 1)).astype(int)
df['cart_size_bucket'] = pd.cut(df['items_in_cart'], bins=[0,1,2,3,100], labels=[0,1,2,3]).astype(int)
df['city_item_signal'] = df['city_lift'] * df['is_local_favorite']
log(f"  Added 9 engineered features -> {df.shape[1]} cols")

# ================================================================
# STEP 3: COLUMNS
# ================================================================
META = ['order_id', 'candidate_item', 'aug_type', 'sample_weight', 'label']
CATS = ['meal_period', 'restaurant', 'cand_category', 'cand_veg_nonveg',
        'cand_cuisine', 'cand_typical_role', 'cand_flavor_profile', 'city']
if HAS_LLM and 'item_semantic_cluster' in df.columns:
    CATS.append('item_semantic_cluster')
    df['item_semantic_cluster'] = df['item_semantic_cluster'].astype(str)
    log("  Added item_semantic_cluster as categorical")

FEATS = [c for c in df.columns if c not in META]
NUMS = [c for c in FEATS if c not in CATS]
log(f"  {len(NUMS)} numeric + {len(CATS)} cat = {len(FEATS)} features")

# ================================================================
# STEP 4: ENCODE CATEGORICALS
# ================================================================
log("\nSTEP 3: Encoding categoricals...")
encoders = {}
for col in CATS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    log(f"  {col}: {len(le.classes_)} classes")
joblib.dump(encoders, os.path.join(OUT, 'label_encoders.pkl'))

# ================================================================
# STEP 5: TEMPORAL SPLIT
# ================================================================
log("\nSTEP 4: TEMPORAL Train/Val/Test split...")
df['_base'] = df['order_id'].str.replace(r'^(syn_|soft_)', '', regex=True)

def extract_order_num(oid):
    parts = oid.split('_')
    for p in reversed(parts):
        if p.isdigit(): return int(p)
    return 0

base_ids_sorted = sorted(df['_base'].unique(), key=extract_order_num)
n = len(base_ids_sorted)
train_ids = set(base_ids_sorted[:int(0.70 * n)])
val_ids   = set(base_ids_sorted[int(0.70 * n):int(0.85 * n)])
test_ids  = set(base_ids_sorted[int(0.85 * n):])

masks = {
    'train': df['_base'].isin(train_ids),
    'val':   df['_base'].isin(val_ids),
    'test':  df['_base'].isin(test_ids),
}

X = {k: df.loc[m, FEATS] for k, m in masks.items()}
y = {k: df.loc[m, 'label'] for k, m in masks.items()}
w = {k: df.loc[m, 'sample_weight'] for k, m in masks.items()}
yb = {k: (v > 0.25).astype(int) for k, v in y.items()}

val_oids  = df.loc[masks['val'], 'order_id'].values
test_oids = df.loc[masks['test'], 'order_id'].values
test_cands = df.loc[masks['test'], 'candidate_item'].values

for k in ['train', 'val', 'test']:
    log(f"  {k}: {len(X[k]):,}")

train_max = max(extract_order_num(x) for x in train_ids)
val_min = min(extract_order_num(x) for x in val_ids)
val_max = max(extract_order_num(x) for x in val_ids)
test_min = min(extract_order_num(x) for x in test_ids)
log(f"  Temporal: train[..{train_max}] < val[{val_min}..{val_max}] < test[{test_min}..]")

# ================================================================
# STEP 6: LIGHTGBM OPTUNA (30 trials, NDCG@10)
# ================================================================
log("\n" + "=" * 60)
log("STEP 5: LightGBM tuning (30 trials, optimizing NDCG@10)...")
t1 = time.time()

lgb_tr = lgb.Dataset(X['train'], label=y['train'], weight=w['train'], free_raw_data=False)
lgb_vl = lgb.Dataset(X['val'], label=y['val'], weight=w['val'], free_raw_data=False)

def lgb_obj(trial):
    p = {
        'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'seed': SEED,
        'feature_pre_filter': False, 'force_col_wise': True,
        'learning_rate': trial.suggest_float('lr', 0.02, 0.15),
        'num_leaves': trial.suggest_int('nl', 20, 127),
        'max_depth': trial.suggest_int('md', 4, 10),
        'min_child_samples': trial.suggest_int('mcs', 20, 100),
        'colsample_bytree': trial.suggest_float('cs', 0.6, 1.0),
        'subsample': trial.suggest_float('ss', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('ra', 1e-3, 5.0, log=True),
        'reg_lambda': trial.suggest_float('rl', 1e-3, 5.0, log=True),
        'scale_pos_weight': trial.suggest_float('spw', 1.5, 6.0),
    }
    m = lgb.train(p, lgb_tr, num_boost_round=300, valid_sets=[lgb_vl],
                  callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
    preds = m.predict(X['val'])
    return compute_ndcg(val_oids, yb['val'].values, preds, k=10)

study1 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study1.optimize(lgb_obj, n_trials=30)
log(f"  Best LGB val NDCG@10: {study1.best_value:.6f} ({time.time()-t1:.0f}s)")
log(f"  Params: {study1.best_params}")

# Final LightGBM
log("\nSTEP 6: Final LightGBM (up to 1000 rounds)...")
bp1 = {
    'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'seed': SEED,
    'feature_pre_filter': False, 'force_col_wise': True,
    **{k: study1.best_params[k] for k in study1.best_params},
}
# Rename keys for LGB
lgb_param_map = {'lr': 'learning_rate', 'nl': 'num_leaves', 'md': 'max_depth',
                 'mcs': 'min_child_samples', 'cs': 'colsample_bytree', 'ss': 'subsample',
                 'ra': 'reg_alpha', 'rl': 'reg_lambda', 'spw': 'scale_pos_weight'}
bp1_clean = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'seed': SEED,
             'feature_pre_filter': False, 'force_col_wise': True}
for short, full in lgb_param_map.items():
    bp1_clean[full] = study1.best_params[short]

lgb_final = lgb.train(bp1_clean, lgb_tr, num_boost_round=1000, valid_sets=[lgb_vl],
                       callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
log(f"  Best iter: {lgb_final.best_iteration}")
lgb_final.save_model(os.path.join(OUT, 'lgb_model.txt'))

# ================================================================
# STEP 7: XGBOOST OPTUNA (25 trials, NDCG@10)
# ================================================================
log("\n" + "=" * 60)
log("STEP 7: XGBoost tuning (25 trials, optimizing NDCG@10)...")
t2 = time.time()

dt_train = xgb.DMatrix(X['train'], label=y['train'], weight=w['train'])
dt_val = xgb.DMatrix(X['val'], label=y['val'], weight=w['val'])

def xgb_obj(trial):
    p = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
        'seed': SEED, 'verbosity': 0,
        'learning_rate': trial.suggest_float('lr', 0.02, 0.15),
        'max_depth': trial.suggest_int('md', 4, 10),
        'min_child_weight': trial.suggest_int('mcw', 5, 60),
        'colsample_bytree': trial.suggest_float('cs', 0.6, 1.0),
        'subsample': trial.suggest_float('ss', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('ra', 1e-3, 5.0, log=True),
        'reg_lambda': trial.suggest_float('rl', 1e-3, 5.0, log=True),
        'scale_pos_weight': trial.suggest_float('spw', 1.5, 6.0),
    }
    m = xgb.train(p, dt_train, num_boost_round=200, evals=[(dt_val, 'v')],
                  early_stopping_rounds=20, verbose_eval=False)
    preds = m.predict(dt_val)
    return compute_ndcg(val_oids, yb['val'].values, preds, k=10)

study2 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study2.optimize(xgb_obj, n_trials=25)
log(f"  Best XGB val NDCG@10: {study2.best_value:.6f} ({time.time()-t2:.0f}s)")
log(f"  Params: {study2.best_params}")

# Final XGBoost
log("\nSTEP 8: Final XGBoost (up to 1000 rounds)...")
bp2 = {
    'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
    'seed': SEED, 'verbosity': 0,
    'learning_rate': study2.best_params['lr'],
    'max_depth': study2.best_params['md'],
    'min_child_weight': study2.best_params['mcw'],
    'colsample_bytree': study2.best_params['cs'],
    'subsample': study2.best_params['ss'],
    'reg_alpha': study2.best_params['ra'],
    'reg_lambda': study2.best_params['rl'],
    'scale_pos_weight': study2.best_params['spw'],
}
dte = xgb.DMatrix(X['test'], label=y['test'], weight=w['test'])
xgb_final = xgb.train(bp2, dt_train, num_boost_round=1000, evals=[(dt_val, 'val')],
                       early_stopping_rounds=50, verbose_eval=100)
xgb_final.save_model(os.path.join(OUT, 'xgb_model.json'))

# ================================================================
# STEP 9: ENSEMBLE
# ================================================================
log("\n" + "=" * 60)
log("STEP 9: Ensemble (optimize by val NDCG@10)...")
lp_t = lgb_final.predict(X['test'])
xp_t = xgb_final.predict(dte)
lp_v = lgb_final.predict(X['val'])
xp_v = xgb_final.predict(dt_val)

bw, bn = 0.5, 0.0
for ww in np.arange(0.1, 0.95, 0.05):
    blend = ww * lp_v + (1 - ww) * xp_v
    ndcg_val = compute_ndcg(val_oids, yb['val'].values, blend, k=10)
    if ndcg_val > bn:
        bw, bn = ww, ndcg_val
log(f"  Weight: LGB={bw:.2f} XGB={1-bw:.2f}, Val NDCG@10={bn:.6f}")
ep = bw * lp_t + (1 - bw) * xp_t

# ================================================================
# STEP 10: FULL EVALUATION
# ================================================================
log("\n" + "=" * 60)
log("STEP 10: Evaluation on TEST set...")

def eval_model(name, pred):
    a = roc_auc_score(yb['test'], pred, sample_weight=w['test'])
    ap = average_precision_score(yb['test'], pred, sample_weight=w['test'])
    ll = log_loss(yb['test'], np.clip(pred, 1e-7, 1 - 1e-7), sample_weight=w['test'])
    log(f"  {name:12s}  AUC={a:.6f}  AP={ap:.6f}  LL={ll:.6f}")
    return {'auc': float(a), 'avg_precision': float(ap), 'log_loss': float(ll)}

lm = eval_model("LightGBM", lp_t)
xm = eval_model("XGBoost", xp_t)
enm = eval_model("Ensemble", ep)

er = ranking_metrics(test_oids, yb['test'].values, ep)
log("\n  Ranking metrics (Ensemble):")
for k, v in sorted(er.items()):
    log(f"    {k:12s}: {v:.4f}")

er_lgb = ranking_metrics(test_oids, yb['test'].values, lp_t)
er_xgb = ranking_metrics(test_oids, yb['test'].values, xp_t)

# ================================================================
# STEP 11: SEGMENT ANALYSIS
# ================================================================
log("\n" + "=" * 60)
log("STEP 11: Segment analysis...")
tdf = df.loc[masks['test']].copy()
tdf['pred'] = ep
tdf['yb'] = yb['test'].values

for c in ['meal_period', 'restaurant', 'city', 'cand_category']:
    tdf[c + '_d'] = encoders[c].inverse_transform(tdf[c])
    log(f"\n  By {c}:")
    for nm, g in tdf.groupby(c + '_d'):
        if g['yb'].nunique() < 2: continue
        a = roc_auc_score(g['yb'], g['pred'])
        nd = compute_ndcg(g['order_id'].values, g['yb'].values, g['pred'].values)
        log(f"    {str(nm):25s}: AUC={a:.4f} NDCG@10={nd:.4f} n={len(g):>6,}")

log("\n  By cold_start:")
for cs in [0, 1]:
    g = tdf[tdf['user_is_cold_start'] == cs]
    if g['yb'].nunique() < 2: continue
    tag = 'Returning' if cs == 0 else 'Cold-start'
    a = roc_auc_score(g['yb'], g['pred'])
    nd = compute_ndcg(g['order_id'].values, g['yb'].values, g['pred'].values)
    log(f"    {tag:25s}: AUC={a:.4f} NDCG@10={nd:.4f} n={len(g):>6,}")

log("\n  By aug_type:")
for at in tdf['aug_type'].unique():
    g = tdf[tdf['aug_type'] == at]
    if g['yb'].nunique() < 2: continue
    a = roc_auc_score(g['yb'], g['pred'])
    log(f"    {str(at):25s}: AUC={a:.4f} n={len(g):>6,}")

# ================================================================
# STEP 12: FEATURE IMPORTANCE
# ================================================================
log("\n" + "=" * 60)
log("STEP 12: Feature importance (top 25)...")
fi = pd.DataFrame({
    'feature': FEATS,
    'gain': lgb_final.feature_importance('gain'),
    'split': lgb_final.feature_importance('split')
}).sort_values('gain', ascending=False)
for _, r in fi.head(25).iterrows():
    log(f"    {r['feature']:35s}: gain={r['gain']:>10.1f}  splits={r['split']:>5d}")
fi.to_csv(os.path.join(OUT, 'feature_importance.csv'), index=False)

# Show LLM feature ranks
if HAS_LLM:
    llm_feats = [f for f in fi['feature'] if f.startswith('llm_') or f.startswith('item_emb_')]
    log(f"\n  LLM feature ranks:")
    for feat in llm_feats:
        rank = fi.index.get_loc(fi[fi['feature'] == feat].index[0]) + 1
        gain = fi.loc[fi['feature'] == feat, 'gain'].values[0]
        log(f"    #{rank:3d}: {feat:35s} gain={gain:>10.1f}")

# ================================================================
# STEP 13: PLOTS
# ================================================================
log("\nSTEP 13: Plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

t25 = fi.head(25).sort_values('gain')
colors = ['#F44336' if (f.startswith('llm_') or f.startswith('item_emb_')) else '#2196F3'
          for f in t25['feature']]
axes[0, 0].barh(t25['feature'], t25['gain'], color=colors)
axes[0, 0].set_title('Top 25 Features (red=LLM)')

for lv, c, nm in [(0, 'red', 'Neg'), (1, 'green', 'Pos')]:
    axes[0, 1].hist(ep[yb['test'] == lv], bins=50, alpha=0.5, color=c, label=nm, density=True)
axes[0, 1].set_title('Score Distribution'); axes[0, 1].legend()

pr, rc, _ = precision_recall_curve(yb['test'], ep)
axes[1, 0].plot(rc, pr, color='darkorange', lw=2)
axes[1, 0].set_title(f'PR Curve (AP={enm["avg_precision"]:.4f})')

rn, rv = list(er.keys()), list(er.values())
co = ['#2196F3' if '@3' in n else '#4CAF50' if '@5' in n else '#FF9800' for n in rn]
axes[1, 1].barh(rn, rv, color=co)
axes[1, 1].set_xlim(0, 1); axes[1, 1].set_title('Ranking Metrics')
for i, v in enumerate(rv):
    axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'evaluation_plots.png'), dpi=150, bbox_inches='tight')
log("  Saved plots")

# ================================================================
# STEP 14: SAVE
# ================================================================
log("\nSTEP 14: Saving...")
res = {
    'version': 'v3',
    'improvements': ['llm_features', '30_lgb_trials', '25_xgb_trials', 'temporal_split', 'ndcg10_optuna'],
    'has_llm_features': HAS_LLM,
    'data': {
        'rows': int(df.shape[0]), 'feats': len(FEATS),
        'train': len(X['train']), 'val': len(X['val']), 'test': len(X['test']),
    },
    'lgb': {
        'iter': lgb_final.best_iteration, 'metrics': lm, 'ranking': er_lgb,
        'optuna': {'trials': 30, 'objective': 'ndcg@10', 'best_val': study1.best_value}
    },
    'xgb': {
        'metrics': xm, 'ranking': er_xgb,
        'optuna': {'trials': 25, 'objective': 'ndcg@10', 'best_val': study2.best_value}
    },
    'ensemble': {
        'lgb_w': float(bw), 'xgb_w': float(1 - bw),
        'metrics': enm, 'ranking': er
    },
    'features': FEATS, 'cats': CATS,
}
with open(os.path.join(OUT, 'training_results.json'), 'w') as f:
    json.dump(res, f, indent=2, default=str)

pd.DataFrame({
    'order_id': test_oids, 'candidate_item': test_cands,
    'y_true': y['test'].values, 'y_binary': yb['test'].values,
    'lgb': lp_t, 'xgb': xp_t, 'ensemble': ep
}).to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)

pd.DataFrame({
    'order_id': val_oids,
    'y_true': y['val'].values, 'y_binary': yb['val'].values,
    'lgb': lp_v, 'xgb': xp_v
}).to_csv(os.path.join(OUT, 'val_predictions.csv'), index=False)

with open(os.path.join(OUT, 'full_log.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))

log(f"\n{'=' * 60}")
log(f"DONE! {(time.time() - t0) / 60:.1f} min total")
log(f"Total Optuna trials: 55 (30 LGB + 25 XGB)")
log(f"Saved: {os.listdir(OUT)}")
log(f"{'=' * 60}")

with open(os.path.join(OUT, 'full_log.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
