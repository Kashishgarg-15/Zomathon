# -*- coding: utf-8 -*-
"""CSAO Rail Add-On Recommendation - Training Pipeline (LightGBM + XGBoost Ensemble)"""
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
OUT = "model_output"
os.makedirs(OUT, exist_ok=True)

L = []
def log(m=""):
    print(m, flush=True)
    L.append(str(m))

# ===== 1. LOAD =====
log("=" * 60)
log("STEP 1: Loading data...")
t0 = time.time()
df = pd.read_csv("training_data_with_city.csv")
log(f"  Shape: {df.shape}")

# ===== 2. FEATURE ENGINEERING =====
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
log(f"  Added 9 features -> {df.shape[1]} cols")

# ===== 3. COLUMNS =====
META = ['order_id', 'candidate_item', 'aug_type', 'sample_weight', 'label']
CATS = ['meal_period', 'restaurant', 'cand_category', 'cand_veg_nonveg',
        'cand_cuisine', 'cand_typical_role', 'cand_flavor_profile', 'city']
FEATS = [c for c in df.columns if c not in META]
NUMS = [c for c in FEATS if c not in CATS]
log(f"  {len(NUMS)} numeric + {len(CATS)} cat = {len(FEATS)} features")

# ===== 4. ENCODE =====
log("\nSTEP 3: Encoding categoricals...")
encoders = {}
for col in CATS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    log(f"  {col}: {len(le.classes_)} classes")
joblib.dump(encoders, os.path.join(OUT, 'label_encoders.pkl'))

# ===== 5. SPLIT =====
log("\nSTEP 4: Train/Val/Test split...")
df['_base'] = df['order_id'].str.replace(r'^(syn_|soft_)', '', regex=True)
uids = df['_base'].unique()
np.random.shuffle(uids)
n = len(uids)
sets = {
    'train': set(uids[:int(0.70*n)]),
    'val':   set(uids[int(0.70*n):int(0.85*n)]),
    'test':  set(uids[int(0.85*n):])
}
masks = {k: df['_base'].isin(v) for k, v in sets.items()}

X = {k: df.loc[m, FEATS] for k, m in masks.items()}
y = {k: df.loc[m, 'label'] for k, m in masks.items()}
w = {k: df.loc[m, 'sample_weight'] for k, m in masks.items()}
yb = {k: (v > 0.25).astype(int) for k, v in y.items()}

test_oids = df.loc[masks['test'], 'order_id'].values
test_cands = df.loc[masks['test'], 'candidate_item'].values
for k in ['train','val','test']:
    log(f"  {k}: {len(X[k]):,}")
log("  No leakage: OK")

# ===== 6. LIGHTGBM TUNING =====
log("\n" + "=" * 60)
log("STEP 5: LightGBM tuning (15 trials, max 300 rounds)...")
t1 = time.time()

lgb_tr = lgb.Dataset(X['train'], label=y['train'], weight=w['train'], free_raw_data=False)
lgb_vl = lgb.Dataset(X['val'], label=y['val'], weight=w['val'], free_raw_data=False)

def lgb_obj(trial):
    p = {
        'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'seed': SEED,
        'feature_pre_filter': False, 'force_col_wise': True,
        'learning_rate': trial.suggest_float('lr', 0.03, 0.12),
        'num_leaves': trial.suggest_int('nl', 31, 80),
        'max_depth': trial.suggest_int('md', 5, 9),
        'min_child_samples': trial.suggest_int('mcs', 30, 70),
        'colsample_bytree': trial.suggest_float('cs', 0.7, 1.0),
        'subsample': trial.suggest_float('ss', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('ra', 0.01, 3.0, log=True),
        'reg_lambda': trial.suggest_float('rl', 0.01, 3.0, log=True),
        'scale_pos_weight': trial.suggest_float('spw', 2.0, 5.0),
    }
    m = lgb.train(p, lgb_tr, num_boost_round=300, valid_sets=[lgb_vl],
                  callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
    return roc_auc_score(yb['val'], m.predict(X['val']), sample_weight=w['val'])

study1 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study1.optimize(lgb_obj, n_trials=15)
log(f"  Best LGB val AUC: {study1.best_value:.6f} ({time.time()-t1:.0f}s)")
log(f"  Params: {study1.best_params}")

# ===== 7. FINAL LIGHTGBM =====
log("\nSTEP 6: Final LightGBM...")
bp1 = {
    'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'seed': SEED,
    'feature_pre_filter': False, 'force_col_wise': True,
    'learning_rate': study1.best_params['lr'],
    'num_leaves': study1.best_params['nl'],
    'max_depth': study1.best_params['md'],
    'min_child_samples': study1.best_params['mcs'],
    'colsample_bytree': study1.best_params['cs'],
    'subsample': study1.best_params['ss'],
    'reg_alpha': study1.best_params['ra'],
    'reg_lambda': study1.best_params['rl'],
    'scale_pos_weight': study1.best_params['spw'],
}
lgb_final = lgb.train(bp1, lgb_tr, num_boost_round=1000, valid_sets=[lgb_vl],
                       callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)])
log(f"  Best iter: {lgb_final.best_iteration}")
lgb_final.save_model(os.path.join(OUT, 'lgb_model.txt'))

# ===== 8. XGBOOST TUNING =====
log("\n" + "=" * 60)
log("STEP 7: XGBoost tuning (15 trials, max 300 rounds)...")
t2 = time.time()

def xgb_obj(trial):
    p = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
        'seed': SEED, 'verbosity': 0,
        'learning_rate': trial.suggest_float('lr', 0.03, 0.12),
        'max_depth': trial.suggest_int('md', 5, 9),
        'min_child_weight': trial.suggest_int('mcw', 10, 50),
        'colsample_bytree': trial.suggest_float('cs', 0.7, 1.0),
        'subsample': trial.suggest_float('ss', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('ra', 0.01, 3.0, log=True),
        'reg_lambda': trial.suggest_float('rl', 0.01, 3.0, log=True),
        'scale_pos_weight': trial.suggest_float('spw', 2.0, 5.0),
    }
    dt = xgb.DMatrix(X['train'], label=y['train'], weight=w['train'])
    dv = xgb.DMatrix(X['val'], label=y['val'], weight=w['val'])
    m = xgb.train(p, dt, num_boost_round=300, evals=[(dv,'v')],
                  early_stopping_rounds=20, verbose_eval=False)
    return roc_auc_score(yb['val'], m.predict(dv), sample_weight=w['val'])

study2 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study2.optimize(xgb_obj, n_trials=15)
log(f"  Best XGB val AUC: {study2.best_value:.6f} ({time.time()-t2:.0f}s)")
log(f"  Params: {study2.best_params}")

# ===== 9. FINAL XGBOOST =====
log("\nSTEP 8: Final XGBoost...")
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
dt = xgb.DMatrix(X['train'], label=y['train'], weight=w['train'])
dv = xgb.DMatrix(X['val'], label=y['val'], weight=w['val'])
dte = xgb.DMatrix(X['test'], label=y['test'], weight=w['test'])
xgb_final = xgb.train(bp2, dt, num_boost_round=1000, evals=[(dv,'val')],
                       early_stopping_rounds=40, verbose_eval=100)
xgb_final.save_model(os.path.join(OUT, 'xgb_model.json'))

# ===== 10. ENSEMBLE =====
log("\n" + "=" * 60)
log("STEP 9: Ensemble...")
lp_t = lgb_final.predict(X['test'])
xp_t = xgb_final.predict(dte)
lp_v = lgb_final.predict(X['val'])
xp_v = xgb_final.predict(dv)

bw, ba = 0.5, 0
for ww in np.arange(0.1, 0.95, 0.05):
    a = roc_auc_score(yb['val'], ww*lp_v + (1-ww)*xp_v, sample_weight=w['val'])
    if a > ba: bw, ba = ww, a
log(f"  Weight: LGB={bw:.2f} XGB={1-bw:.2f}, Val AUC={ba:.6f}")
ep = bw * lp_t + (1 - bw) * xp_t

# ===== 11. METRICS =====
log("\n" + "=" * 60)
log("STEP 10: Evaluation on test set...")

def em(name, pred):
    a = roc_auc_score(yb['test'], pred, sample_weight=w['test'])
    ap = average_precision_score(yb['test'], pred, sample_weight=w['test'])
    ll = log_loss(yb['test'], np.clip(pred, 1e-7, 1-1e-7), sample_weight=w['test'])
    log(f"  {name:12s}  AUC={a:.6f}  AP={ap:.6f}  LL={ll:.6f}")
    return {'auc': float(a), 'avg_precision': float(ap), 'log_loss': float(ll)}

lm = em("LightGBM", lp_t)
xm = em("XGBoost", xp_t)
enm = em("Ensemble", ep)

# ===== 12. RANKING =====
log("\n  Ranking metrics:")
def rmk(oids, yt, yp, Ks=[3,5,10]):
    r = pd.DataFrame({'o': oids, 't': yt, 'p': yp})
    M = {f'{m}@{k}': [] for k in Ks for m in ['prec','rec','ndcg','hit']}
    for _, g in r.groupby('o'):
        if g['t'].sum() == 0: continue
        s = g.sort_values('p', ascending=False)
        rel = s['t'].values; np_ = (g['t']==1).sum()
        for k in Ks:
            tk = rel[:k]; n_tk = len(tk)
            M[f'prec@{k}'].append(tk.sum()/k)
            M[f'rec@{k}'].append(tk.sum()/max(np_,1))
            M[f'hit@{k}'].append(1 if tk.sum()>0 else 0)
            disc = np.log2(np.arange(2, n_tk+2))
            dcg = np.sum(tk / disc)
            irel = np.sort(g['t'].values)[::-1][:k]; n_ir = len(irel)
            idisc = np.log2(np.arange(2, n_ir+2))
            idcg = np.sum(irel / idisc)
            M[f'ndcg@{k}'].append(dcg/max(idcg,1e-10))
    return {k: float(np.mean(v)) for k,v in M.items()}

er = rmk(test_oids, yb['test'].values, ep)
for k,v in sorted(er.items()): log(f"    {k:12s}: {v:.4f}")

# ===== 13. SEGMENT ANALYSIS =====
log("\n" + "=" * 60)
log("STEP 11: Segment analysis...")
tdf = df.loc[masks['test']].copy()
tdf['pred'] = ep; tdf['yb'] = yb['test'].values

for c in ['meal_period','restaurant','city','cand_category']:
    tdf[c+'_d'] = encoders[c].inverse_transform(tdf[c])
    log(f"\n  By {c}:")
    for nm, g in tdf.groupby(c+'_d'):
        if g['yb'].nunique()<2: continue
        log(f"    {str(nm):25s}: AUC={roc_auc_score(g['yb'],g['pred']):.4f}  n={len(g):>6,}")

log("\n  By cold_start:")
for cs in [0,1]:
    g = tdf[tdf['user_is_cold_start']==cs]
    if g['yb'].nunique()<2: continue
    log(f"    {'Returning' if cs==0 else 'Cold-start':25s}: AUC={roc_auc_score(g['yb'],g['pred']):.4f}  n={len(g):>6,}")

log("\n  By aug_type:")
for at in tdf['aug_type'].unique():
    g = tdf[tdf['aug_type']==at]
    if g['yb'].nunique()<2: continue
    log(f"    {str(at):25s}: AUC={roc_auc_score(g['yb'],g['pred']):.4f}  n={len(g):>6,}")

# ===== 14. FEATURE IMPORTANCE =====
log("\n" + "=" * 60)
log("STEP 12: Feature importance (top 20)...")
fi = pd.DataFrame({'feature': FEATS, 'gain': lgb_final.feature_importance('gain'),
                    'split': lgb_final.feature_importance('split')}).sort_values('gain', ascending=False)
for _, r in fi.head(20).iterrows():
    log(f"    {r['feature']:35s}: gain={r['gain']:>10.1f}  splits={r['split']:>5d}")
fi.to_csv(os.path.join(OUT, 'feature_importance.csv'), index=False)

# ===== 15. PLOTS =====
log("\nSTEP 13: Plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
t20 = fi.head(20).sort_values('gain')
axes[0,0].barh(t20['feature'], t20['gain'], color='steelblue')
axes[0,0].set_title('Top 20 Features (LGB Gain)')
for lv, c, nm in [(0,'red','Neg'),(1,'green','Pos')]:
    axes[0,1].hist(ep[yb['test']==lv], bins=50, alpha=0.5, color=c, label=nm, density=True)
axes[0,1].set_title('Score Distribution'); axes[0,1].legend()
pr, rc, _ = precision_recall_curve(yb['test'], ep)
axes[1,0].plot(rc, pr, color='darkorange', lw=2)
axes[1,0].set_title(f'PR Curve (AP={enm["avg_precision"]:.4f})')
rn, rv = list(er.keys()), list(er.values())
co = ['#2196F3' if '@3' in n else '#4CAF50' if '@5' in n else '#FF9800' for n in rn]
axes[1,1].barh(rn, rv, color=co); axes[1,1].set_xlim(0,1)
axes[1,1].set_title('Ranking Metrics')
for i, v in enumerate(rv): axes[1,1].text(v+0.01, i, f'{v:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'evaluation_plots.png'), dpi=150, bbox_inches='tight')
log("  Saved plots")

# ===== 16. SAVE =====
log("\nSTEP 14: Saving...")
res = {
    'data': {'rows': int(df.shape[0]), 'feats': len(FEATS), 'train': len(X['train']),
             'val': len(X['val']), 'test': len(X['test'])},
    'lgb': {'iter': lgb_final.best_iteration, 'params': bp1, 'metrics': lm},
    'xgb': {'params': bp2, 'metrics': xm},
    'ensemble': {'lgb_w': float(bw), 'xgb_w': float(1-bw), 'metrics': enm, 'ranking': er},
    'features': FEATS, 'cats': CATS
}
with open(os.path.join(OUT, 'training_results.json'), 'w') as f:
    json.dump(res, f, indent=2, default=str)

pd.DataFrame({'order_id': test_oids, 'candidate_item': test_cands, 'y_true': y['test'].values,
              'y_binary': yb['test'].values, 'lgb': lp_t, 'xgb': xp_t, 'ensemble': ep}
             ).to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)

with open(os.path.join(OUT, 'full_log.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))

log(f"\n{'='*60}")
log(f"DONE! {(time.time()-t0)/60:.1f} min total")
log(f"Saved: {os.listdir(OUT)}")
log(f"{'='*60}")
