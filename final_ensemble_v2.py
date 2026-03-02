# -*- coding: utf-8 -*-
"""
Final 4-Model Ensemble v2: LGB_v3 + XGB_v3 + DCN-v2(Optuna) + segment analysis
================================================================================
Reads test predictions from GBDT v3 (LLM features) and DCN-v2 (Optuna-tuned),
finds optimal weights, produces evaluation + full segment analysis.
"""
import os, json, warnings, time
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "model_output_final_v2"
os.makedirs(OUT, exist_ok=True)

L = []
def log(m=""):
    print(m, flush=True)
    L.append(str(m))

t0 = time.time()

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
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

class FastNDCG:
    def __init__(self, order_ids, y_true, k=10):
        self.k = k
        df = pd.DataFrame({'o': order_ids, 'y': y_true}).reset_index(drop=True)
        self.groups = []
        for _, g in df.groupby('o'):
            if g['y'].sum() == 0: continue
            idx = g.index.values
            rel = g['y'].values
            irel = np.sort(rel)[::-1][:k]
            idcg = np.sum(irel / np.log2(np.arange(2, len(irel) + 2)))
            self.groups.append((idx, rel, idcg))
    def __call__(self, y_pred):
        k = self.k
        vals = []
        for idx, rel, idcg in self.groups:
            scores = y_pred[idx]
            order = np.argsort(-scores)
            tk = rel[order][:k]; n = len(tk)
            dcg = np.sum(tk / np.log2(np.arange(2, n + 2)))
            vals.append(dcg / max(idcg, 1e-10))
        return float(np.mean(vals))

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

# ------------------------------------------------------------------
# 1. LOAD PREDICTIONS
# ------------------------------------------------------------------
log("=" * 60)
log("Loading predictions from all pipelines...")

# Try v3 first, fall back to v2
gbdt_dir = "model_output_v3" if os.path.exists("model_output_v3/test_predictions.csv") else "model_output_v2"
dcn_dir = "model_output_dcn_v2" if os.path.exists("model_output_dcn_v2/test_predictions.csv") else "model_output_dcn"

log(f"  GBDT source: {gbdt_dir}")
log(f"  DCN source: {dcn_dir}")

gbdt_test = pd.read_csv(f"{gbdt_dir}/test_predictions.csv")
dcn_test  = pd.read_csv(f"{dcn_dir}/test_predictions.csv")
gbdt_val = pd.read_csv(f"{gbdt_dir}/val_predictions.csv")
dcn_val  = pd.read_csv(f"{dcn_dir}/val_predictions.csv")

log(f"  GBDT test: {len(gbdt_test)}, val: {len(gbdt_val)}")
log(f"  DCN  test: {len(dcn_test)}, val: {len(dcn_val)}")

# Verify alignment
assert (gbdt_test['order_id'] == dcn_test['order_id']).all(), "Test order mismatch!"
assert (gbdt_val['order_id'] == dcn_val['order_id']).all(), "Val order mismatch!"
log("  Alignment: OK")

# ------------------------------------------------------------------
# 2. EXTRACT PREDICTIONS
# ------------------------------------------------------------------
t_oids = gbdt_test['order_id'].values
t_yb = gbdt_test['y_binary'].values
t_lgb = gbdt_test['lgb'].values
t_xgb = gbdt_test['xgb'].values
t_dcn = dcn_test['dcn_pred'].values

v_oids = gbdt_val['order_id'].values
v_yb = gbdt_val['y_binary'].values.astype(int)
v_lgb = gbdt_val['lgb'].values
v_xgb = gbdt_val['xgb'].values
v_dcn = dcn_val['dcn_pred'].values

# ------------------------------------------------------------------
# 3. OPTIMIZE 3-MODEL WEIGHTS
# ------------------------------------------------------------------
log("\n" + "=" * 60)
log("Optimizing 3-model ensemble weights (val NDCG@10)...")

fast_ndcg = FastNDCG(v_oids, v_yb, k=10)
log(f"  {len(fast_ndcg.groups)} valid groups indexed")

best_w = (0.5, 0.3, 0.2)
best_ndcg = 0.0
n_combos = 0

for w_lgb in np.arange(0.05, 0.90, 0.05):
    for w_xgb in np.arange(0.05, 0.90 - w_lgb, 0.05):
        w_dcn = 1.0 - w_lgb - w_xgb
        if w_dcn < 0.05: continue
        blend = w_lgb * v_lgb + w_xgb * v_xgb + w_dcn * v_dcn
        ndcg = fast_ndcg(blend)
        n_combos += 1
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_w = (w_lgb, w_xgb, w_dcn)

w_lgb, w_xgb, w_dcn = best_w
log(f"  Searched {n_combos} combinations")
log(f"  Best weights: LGB={w_lgb:.2f} XGB={w_xgb:.2f} DCN={w_dcn:.2f}")
log(f"  Val NDCG@10: {best_ndcg:.6f}")

# 2-model GBDT-only
best_w2, best_ndcg2 = 0.5, 0.0
for w in np.arange(0.1, 0.95, 0.05):
    ndcg = fast_ndcg(w * v_lgb + (1 - w) * v_xgb)
    if ndcg > best_ndcg2:
        best_ndcg2 = ndcg; best_w2 = w
log(f"  Best GBDT-only: LGB={best_w2:.2f} XGB={1-best_w2:.2f}, Val NDCG@10={best_ndcg2:.6f}")

# ------------------------------------------------------------------
# 4. EVALUATE ALL VARIANTS ON TEST
# ------------------------------------------------------------------
log("\n" + "=" * 60)
log("Test Set Evaluation:")
log("-" * 60)

final_3m = w_lgb * t_lgb + w_xgb * t_xgb + w_dcn * t_dcn
final_2m = best_w2 * t_lgb + (1 - best_w2) * t_xgb

models = {
    'LightGBM': t_lgb,
    'XGBoost': t_xgb,
    'DCN-v2': t_dcn,
    'GBDT Ens (LGB+XGB)': final_2m,
    '3-Model Ens (LGB+XGB+DCN)': final_3m,
}

results = {}
for name, preds in models.items():
    auc = roc_auc_score(t_yb, preds)
    ap = average_precision_score(t_yb, preds)
    ll = log_loss(t_yb, np.clip(preds, 1e-7, 1 - 1e-7))
    rm = ranking_metrics(t_oids, t_yb, preds)
    log(f"\n  {name}:")
    log(f"    AUC={auc:.6f}  AP={ap:.6f}  LL={ll:.6f}")
    log(f"    NDCG@3={rm['ndcg@3']:.4f}  @5={rm['ndcg@5']:.4f}  @10={rm['ndcg@10']:.4f}")
    log(f"    Rec@3={rm['rec@3']:.4f}   @5={rm['rec@5']:.4f}   @10={rm['rec@10']:.4f}")
    log(f"    Hit@3={rm['hit@3']:.4f}   @5={rm['hit@5']:.4f}   @10={rm['hit@10']:.4f}")
    results[name] = {'auc': auc, 'avg_precision': ap, 'log_loss': ll, 'ranking': rm}

# Comparison table
log("\n" + "=" * 60)
log("COMPARISON TABLE:")
log(f"  {'Model':30s} {'AUC':>8s} {'AP':>8s} {'NDCG@5':>8s} {'NDCG@10':>8s} {'Rec@3':>8s} {'Hit@5':>8s}")
log("-" * 90)
for name, r in results.items():
    rm = r['ranking']
    log(f"  {name:30s} {r['auc']:8.4f} {r['avg_precision']:8.4f} {rm['ndcg@5']:8.4f} {rm['ndcg@10']:8.4f} {rm['rec@3']:8.4f} {rm['hit@5']:8.4f}")

# ------------------------------------------------------------------
# 5. SEGMENT-LEVEL ANALYSIS (the missing piece!)
# ------------------------------------------------------------------
log("\n" + "=" * 60)
log("SEGMENT-LEVEL ANALYSIS (3-Model Ensemble):")
log("-" * 60)

# Need original data for segment columns
src = "training_data_llm.csv" if os.path.exists("training_data_llm.csv") else "training_data_with_city.csv"
df_full = pd.read_csv(src, usecols=['order_id', 'meal_period', 'restaurant', 'city',
                                     'cand_category', 'user_is_cold_start', 'aug_type',
                                     'cand_veg_nonveg', 'cand_cuisine'])

# Merge segment info with test predictions
tdf = pd.DataFrame({
    'order_id': t_oids,
    'y_binary': t_yb,
    'pred_3m': final_3m,
    'pred_lgb': t_lgb,
    'pred_dcn': t_dcn,
})

# Match test rows by order
df_test_meta = df_full.loc[df_full['order_id'].isin(set(t_oids))].copy()
# Must align exactly with prediction order
df_test_meta = df_test_meta.reset_index(drop=True)
tdf = tdf.reset_index(drop=True)

# Since both come from the same temporal split, rows should align
# But let's merge properly to be safe
tdf['_idx'] = range(len(tdf))
# The test predictions were saved in the same order as the test split
# We add segment columns directly from the full dataset's test split

# Re-read from the GBDT test predictions which has candidate_item for ordering
if 'candidate_item' in gbdt_test.columns:
    tdf['candidate_item'] = gbdt_test['candidate_item'].values

# Merge segment info
df_full_dedup = df_full.drop_duplicates(subset=['order_id']).set_index('order_id')
for col in ['meal_period', 'restaurant', 'city', 'user_is_cold_start', 'aug_type']:
    if col in df_full.columns:
        # Get per-row info (not per-order, since multiple candidates per order)
        pass

# Better approach: join on position (same temporal split = same row order)
# Read test data directly
if os.path.exists("training_data_llm.csv"):
    df_raw = pd.read_csv("training_data_llm.csv")
else:
    df_raw = pd.read_csv("training_data_with_city.csv")

df_raw['_base'] = df_raw['order_id'].str.replace(r'^(syn_|soft_)', '', regex=True)
def extract_order_num(oid):
    parts = oid.split('_')
    for p in reversed(parts):
        if p.isdigit(): return int(p)
    return 0

base_ids_sorted = sorted(df_raw['_base'].unique(), key=extract_order_num)
n = len(base_ids_sorted)
test_ids = set(base_ids_sorted[int(0.85 * n):])
test_mask = df_raw['_base'].isin(test_ids)
df_test = df_raw.loc[test_mask].copy().reset_index(drop=True)

assert len(df_test) == len(tdf), f"Mismatch: df_test={len(df_test)}, tdf={len(tdf)}"

# Add segment columns
for col in ['meal_period', 'restaurant', 'city', 'cand_category',
            'user_is_cold_start', 'aug_type', 'cand_veg_nonveg', 'cand_cuisine']:
    tdf[col] = df_test[col].values

# Now do segment analysis
segment_results = {}

def segment_analysis(col_name, display_name):
    log(f"\n  By {display_name}:")
    seg_data = {}
    for nm, g in tdf.groupby(col_name):
        if g['y_binary'].nunique() < 2: continue
        auc_3m = roc_auc_score(g['y_binary'], g['pred_3m'])
        ndcg_3m = compute_ndcg(g['order_id'].values, g['y_binary'].values, g['pred_3m'].values)
        auc_lgb = roc_auc_score(g['y_binary'], g['pred_lgb'])
        ndcg_lgb = compute_ndcg(g['order_id'].values, g['y_binary'].values, g['pred_lgb'].values)
        auc_dcn = roc_auc_score(g['y_binary'], g['pred_dcn'])
        log(f"    {str(nm):25s}: 3M_AUC={auc_3m:.4f} LGB_AUC={auc_lgb:.4f} DCN_AUC={auc_dcn:.4f} "
            f"3M_NDCG@10={ndcg_3m:.4f} n={len(g):>6,}")
        seg_data[str(nm)] = {
            'n': len(g),
            '3m_auc': auc_3m, 'lgb_auc': auc_lgb, 'dcn_auc': auc_dcn,
            '3m_ndcg10': ndcg_3m, 'lgb_ndcg10': ndcg_lgb,
        }
    segment_results[display_name] = seg_data

segment_analysis('meal_period', 'meal_period')
segment_analysis('restaurant', 'restaurant')
segment_analysis('city', 'city')
segment_analysis('cand_category', 'cand_category')
segment_analysis('cand_cuisine', 'cand_cuisine')
segment_analysis('cand_veg_nonveg', 'veg_nonveg')

# Cold start
log(f"\n  By user_is_cold_start:")
cs_data = {}
for cs in [0, 1]:
    g = tdf[tdf['user_is_cold_start'] == cs]
    if g['y_binary'].nunique() < 2: continue
    tag = 'Returning' if cs == 0 else 'Cold-start'
    auc_3m = roc_auc_score(g['y_binary'], g['pred_3m'])
    ndcg_3m = compute_ndcg(g['order_id'].values, g['y_binary'].values, g['pred_3m'].values)
    auc_lgb = roc_auc_score(g['y_binary'], g['pred_lgb'])
    auc_dcn = roc_auc_score(g['y_binary'], g['pred_dcn'])
    log(f"    {tag:25s}: 3M_AUC={auc_3m:.4f} LGB_AUC={auc_lgb:.4f} DCN_AUC={auc_dcn:.4f} "
        f"3M_NDCG@10={ndcg_3m:.4f} n={len(g):>6,}")
    cs_data[tag] = {
        'n': len(g),
        '3m_auc': auc_3m, 'lgb_auc': auc_lgb, 'dcn_auc': auc_dcn,
        '3m_ndcg10': ndcg_3m,
    }
segment_results['user_is_cold_start'] = cs_data

# Aug type
log(f"\n  By aug_type:")
at_data = {}
for at, g in tdf.groupby('aug_type'):
    if g['y_binary'].nunique() < 2: continue
    auc_3m = roc_auc_score(g['y_binary'], g['pred_3m'])
    log(f"    {str(at):25s}: AUC={auc_3m:.4f} n={len(g):>6,}")
    at_data[str(at)] = {'n': len(g), 'auc': auc_3m}
segment_results['aug_type'] = at_data

# ------------------------------------------------------------------
# 6. PLOTS
# ------------------------------------------------------------------
log("\nGenerating comparison plots...")

fig, axes = plt.subplots(2, 3, figsize=(24, 14))

# 6a. AUC comparison
names_short = ['LGB', 'XGB', 'DCN-v2', 'GBDT\nEns', '3-Model\nEns']
aucs = [results[n]['auc'] for n in models.keys()]
colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50', '#F44336']
axes[0, 0].bar(names_short, aucs, color=colors)
axes[0, 0].set_ylim(min(aucs) - 0.02, max(aucs) + 0.01)
axes[0, 0].set_title('AUC Comparison')
for i, v in enumerate(aucs):
    axes[0, 0].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=10)

# 6b. NDCG comparison
ndcg_data = {}
for k in [3, 5, 10]:
    ndcg_data[f'@{k}'] = [results[n]['ranking'][f'ndcg@{k}'] for n in models.keys()]
x = np.arange(len(names_short)); w = 0.25
for i, (k, vals) in enumerate(ndcg_data.items()):
    axes[0, 1].bar(x + i * w, vals, w, label=f'NDCG{k}',
                   color=['#2196F3', '#4CAF50', '#FF9800'][i])
axes[0, 1].set_xticks(x + w); axes[0, 1].set_xticklabels(names_short)
axes[0, 1].set_title('NDCG Comparison'); axes[0, 1].legend()
axes[0, 1].set_ylim(0.75, 1.0)

# 6c. PR Curves
for name, preds, c in [('LGB', t_lgb, '#2196F3'), ('XGB', t_xgb, '#FF9800'),
                        ('DCN', t_dcn, '#9C27B0'), ('3-Ens', final_3m, '#F44336')]:
    pr, rc, _ = precision_recall_curve(t_yb, preds)
    axes[0, 2].plot(rc, pr, label=name, color=c, lw=2)
axes[0, 2].set_title('Precision-Recall Curves')
axes[0, 2].set_xlabel('Recall'); axes[0, 2].set_ylabel('Precision'); axes[0, 2].legend()

# 6d. Score distributions
for lv, col, lab in [(0, 'red', 'Neg'), (1, 'green', 'Pos')]:
    axes[1, 0].hist(final_3m[t_yb == lv], bins=50, alpha=0.5, color=col, label=lab, density=True)
axes[1, 0].set_title('3-Model Ensemble Score Distribution'); axes[1, 0].legend()

# 6e. Segment AUC by meal_period
if 'meal_period' in segment_results:
    segs = segment_results['meal_period']
    seg_names = list(segs.keys())
    seg_aucs_3m = [segs[s]['3m_auc'] for s in seg_names]
    seg_aucs_lgb = [segs[s]['lgb_auc'] for s in seg_names]
    sx = np.arange(len(seg_names))
    axes[1, 1].bar(sx - 0.15, seg_aucs_lgb, 0.3, label='LGB', color='#2196F3')
    axes[1, 1].bar(sx + 0.15, seg_aucs_3m, 0.3, label='3-Model', color='#F44336')
    axes[1, 1].set_xticks(sx); axes[1, 1].set_xticklabels(seg_names, rotation=30)
    axes[1, 1].set_title('AUC by Meal Period'); axes[1, 1].legend()
    axes[1, 1].set_ylim(0.80, 0.95)

# 6f. Segment AUC by cold_start
if 'user_is_cold_start' in segment_results:
    cs = segment_results['user_is_cold_start']
    cs_names = list(cs.keys())
    cs_aucs_3m = [cs[s]['3m_auc'] for s in cs_names]
    cs_aucs_lgb = [cs[s]['lgb_auc'] for s in cs_names]
    cs_aucs_dcn = [cs[s]['dcn_auc'] for s in cs_names]
    sx = np.arange(len(cs_names))
    axes[1, 2].bar(sx - 0.2, cs_aucs_lgb, 0.2, label='LGB', color='#2196F3')
    axes[1, 2].bar(sx, cs_aucs_dcn, 0.2, label='DCN', color='#9C27B0')
    axes[1, 2].bar(sx + 0.2, cs_aucs_3m, 0.2, label='3-Model', color='#F44336')
    axes[1, 2].set_xticks(sx); axes[1, 2].set_xticklabels(cs_names)
    axes[1, 2].set_title('AUC by Cold-Start Status'); axes[1, 2].legend()
    axes[1, 2].set_ylim(0.80, 0.95)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'final_comparison.png'), dpi=150, bbox_inches='tight')
log("  Saved comparison plots")

# ------------------------------------------------------------------
# 7. SAVE
# ------------------------------------------------------------------
log("\nSaving final results...")

final_results = {
    'ensemble_weights': {
        '3model': {'lgb': float(w_lgb), 'xgb': float(w_xgb), 'dcn': float(w_dcn)},
        '2model': {'lgb': float(best_w2), 'xgb': float(1 - best_w2)},
    },
    'val_ndcg10': {'3model': best_ndcg, '2model': best_ndcg2},
    'test_results': results,
    'segment_analysis': segment_results,
    'data_sources': {'gbdt': gbdt_dir, 'dcn': dcn_dir},
}

with open(os.path.join(OUT, 'final_results.json'), 'w') as f:
    json.dump(final_results, f, indent=2, default=float)

pd.DataFrame({
    'order_id': t_oids,
    'candidate_item': gbdt_test['candidate_item'].values,
    'y_true': gbdt_test['y_true'].values,
    'y_binary': t_yb,
    'lgb': t_lgb,
    'xgb': t_xgb,
    'dcn': t_dcn,
    'ensemble_2m': final_2m,
    'ensemble_3m': final_3m,
}).to_csv(os.path.join(OUT, 'final_predictions.csv'), index=False)

with open(os.path.join(OUT, 'full_log.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))

elapsed = time.time() - t0
log(f"\nResults saved to {OUT}/")
log(f"Elapsed: {elapsed:.1f}s")
log("=" * 60)
log("DONE!")

with open(os.path.join(OUT, 'full_log.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
