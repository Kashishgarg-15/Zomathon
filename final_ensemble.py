# -*- coding: utf-8 -*-
"""
Final 3-Model Ensemble: LightGBM + XGBoost + DCN-v2
====================================================
Reads test predictions from GBDT v2 and DCN-v2 pipelines,
finds optimal weights, and produces the definitive evaluation.
"""
import os, json, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "model_output_final"
os.makedirs(OUT, exist_ok=True)

def log(m=""):
    print(m, flush=True)

# ------------------------------------------------------------------
# HELPER: ranking metrics
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


class FastNDCG:
    """Pre-indexes group structure for fast repeated NDCG@k evaluation."""
    def __init__(self, order_ids, y_true, k=10):
        self.k = k
        df = pd.DataFrame({'o': order_ids, 'y': y_true}).reset_index(drop=True)
        self.groups = []
        for _, g in df.groupby('o'):
            if g['y'].sum() == 0:
                continue
            idx = g.index.values
            rel = g['y'].values
            irel = np.sort(rel)[::-1][:k]
            ni = len(irel)
            idcg = np.sum(irel / np.log2(np.arange(2, ni + 2)))
            self.groups.append((idx, rel, idcg))

    def __call__(self, y_pred):
        k = self.k
        vals = []
        for idx, rel, idcg in self.groups:
            scores = y_pred[idx]
            order = np.argsort(-scores)
            tk = rel[order][:k]
            n = len(tk)
            dcg = np.sum(tk / np.log2(np.arange(2, n + 2)))
            vals.append(dcg / max(idcg, 1e-10))
        return float(np.mean(vals))

# ------------------------------------------------------------------
# 1. LOAD PREDICTIONS
# ------------------------------------------------------------------
log("=" * 60)
log("Loading predictions from both pipelines...")

gbdt_test = pd.read_csv("model_output_v2/test_predictions.csv")
dcn_test  = pd.read_csv("model_output_dcn/test_predictions.csv")

gbdt_val = pd.read_csv("model_output_v2/val_predictions.csv")
dcn_val  = pd.read_csv("model_output_dcn/val_predictions.csv")

log(f"  GBDT test: {len(gbdt_test)}, val: {len(gbdt_val)}")
log(f"  DCN  test: {len(dcn_test)}, val: {len(dcn_val)}")

# Verify alignment (same order_ids)
assert (gbdt_test['order_id'] == dcn_test['order_id']).all(), "Test order mismatch!"
assert (gbdt_val['order_id'] == dcn_val['order_id']).all(), "Val order mismatch!"
log("  Alignment: OK")

# ------------------------------------------------------------------
# 2. EXTRACT PREDICTIONS
# ------------------------------------------------------------------
# Test set
t_oids = gbdt_test['order_id'].values
t_yb = gbdt_test['y_binary'].values
t_lgb = gbdt_test['lgb'].values
t_xgb = gbdt_test['xgb'].values
t_ens = gbdt_test['ensemble'].values
t_dcn = dcn_test['dcn_pred'].values

# Val set
v_oids = gbdt_val['order_id'].values
v_yb = gbdt_val['y_binary'].values.astype(int)
v_lgb = gbdt_val['lgb'].values
v_xgb = gbdt_val['xgb'].values
v_dcn = dcn_val['dcn_pred'].values

# ------------------------------------------------------------------
# 3. FIND OPTIMAL 3-MODEL WEIGHTS ON VAL
# ------------------------------------------------------------------
log("\n" + "=" * 60)
log("Optimizing 3-model ensemble weights (val NDCG@10)...")
log("  Pre-indexing group structure for fast evaluation...")

fast_ndcg = FastNDCG(v_oids, v_yb, k=10)
log(f"  {len(fast_ndcg.groups)} valid groups indexed")

best_w = (0.5, 0.3, 0.2)
best_ndcg = 0.0
n_combos = 0

for w_lgb in np.arange(0.1, 0.85, 0.05):
    for w_xgb in np.arange(0.05, 0.85 - w_lgb, 0.05):
        w_dcn = 1.0 - w_lgb - w_xgb
        if w_dcn < 0.05: continue
        blend = w_lgb * v_lgb + w_xgb * v_xgb + w_dcn * v_dcn
        ndcg = fast_ndcg(blend)
        n_combos += 1
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_w = (w_lgb, w_xgb, w_dcn)

w_lgb, w_xgb, w_dcn = best_w
log(f"  Searched {n_combos} weight combinations")
log(f"  Best weights: LGB={w_lgb:.2f} XGB={w_xgb:.2f} DCN={w_dcn:.2f}")
log(f"  Val NDCG@10: {best_ndcg:.6f}")

# Also find best 2-model weights (GBDT ensemble only, no DCN)
best_w2 = 0.5
best_ndcg2 = 0.0
for w in np.arange(0.1, 0.95, 0.05):
    blend = w * v_lgb + (1 - w) * v_xgb
    ndcg = fast_ndcg(blend)
    if ndcg > best_ndcg2:
        best_ndcg2 = ndcg
        best_w2 = w
log(f"  Best GBDT-only: LGB={best_w2:.2f} XGB={1-best_w2:.2f}, Val NDCG@10={best_ndcg2:.6f}")

# ------------------------------------------------------------------
# 4. EVALUATE ALL MODEL VARIANTS ON TEST
# ------------------------------------------------------------------
log("\n" + "=" * 60)
log("Test Set Evaluation:")
log("-" * 60)

# Create final ensemble predictions
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
    results[name] = {
        'auc': auc, 'avg_precision': ap, 'log_loss': ll,
        'ranking': rm,
    }

# ------------------------------------------------------------------
# 5. COMPARISON TABLE
# ------------------------------------------------------------------
log("\n" + "=" * 60)
log("COMPARISON TABLE:")
log(f"  {'Model':30s} {'AUC':>8s} {'AP':>8s} {'NDCG@5':>8s} {'NDCG@10':>8s} {'Rec@3':>8s} {'Hit@5':>8s}")
log("-" * 90)
for name, r in results.items():
    rm = r['ranking']
    log(f"  {name:30s} {r['auc']:8.4f} {r['avg_precision']:8.4f} {rm['ndcg@5']:8.4f} {rm['ndcg@10']:8.4f} {rm['rec@3']:8.4f} {rm['hit@5']:8.4f}")

# ------------------------------------------------------------------
# 6. PLOTS
# ------------------------------------------------------------------
log("\nGenerating comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 6a. AUC comparison bar chart
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

x = np.arange(len(names_short))
w = 0.25
for i, (k, vals) in enumerate(ndcg_data.items()):
    axes[0, 1].bar(x + i * w, vals, w, label=f'NDCG{k}',
                   color=['#2196F3', '#4CAF50', '#FF9800'][i])
axes[0, 1].set_xticks(x + w)
axes[0, 1].set_xticklabels(names_short)
axes[0, 1].set_title('NDCG Comparison')
axes[0, 1].legend()
axes[0, 1].set_ylim(0.75, 1.0)

# 6c. PR Curves for all models
for name, preds, c in [('LGB', t_lgb, '#2196F3'), ('XGB', t_xgb, '#FF9800'),
                        ('DCN', t_dcn, '#9C27B0'), ('3-Ens', final_3m, '#F44336')]:
    pr, rc, _ = precision_recall_curve(t_yb, preds)
    axes[1, 0].plot(rc, pr, label=name, color=c, lw=2)
axes[1, 0].set_title('Precision-Recall Curves')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()

# 6d. Score distributions overlay
for lv, col, lab in [(0, 'red', 'Neg'), (1, 'green', 'Pos')]:
    axes[1, 1].hist(final_3m[t_yb == lv], bins=50, alpha=0.5, color=col, label=lab, density=True)
axes[1, 1].set_title('3-Model Ensemble Score Distribution')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'final_comparison.png'), dpi=150, bbox_inches='tight')
log("  Saved comparison plots")

# ------------------------------------------------------------------
# 7. SAVE
# ------------------------------------------------------------------
log("\nSaving final results...")

final_results = {
    'ensemble_weights': {
        '3model': {'lgb': w_lgb, 'xgb': w_xgb, 'dcn': w_dcn},
        '2model': {'lgb': best_w2, 'xgb': 1 - best_w2},
    },
    'val_ndcg10': {'3model': best_ndcg, '2model': best_ndcg2},
    'test_results': results,
}

with open(os.path.join(OUT, 'final_results.json'), 'w') as f:
    json.dump(final_results, f, indent=2, default=float)

# Save final predictions
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

log(f"\nResults saved to {OUT}/")
log("=" * 60)
log("DONE!")
