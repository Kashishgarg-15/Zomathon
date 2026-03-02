# -*- coding: utf-8 -*-
"""
Tier 2 v2: DCN-v2 with Optuna Hyperparameter Tuning + LLM Features
=====================================================================
Adds Optuna HPO (10 trials) to find optimal DCN-v2 architecture.
Uses LLM-enriched dataset (training_data_llm.csv).
"""
import os, sys, time, json, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = "model_output_dcn_v2"
os.makedirs(OUT, exist_ok=True)

L = []
def log(m=""):
    print(m, flush=True)
    L.append(str(m))

log(f"Device: {DEVICE}")

# ================================================================
# HELPER: NDCG@K
# ================================================================
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
# 1. LOAD LLM-ENRICHED DATA
# ================================================================
log("=" * 60)
log("STEP 1: Loading LLM-enriched data...")
t0 = time.time()

# Use LLM-enriched data if available, otherwise fall back to original
if os.path.exists("training_data_llm.csv"):
    df = pd.read_csv("training_data_llm.csv")
    log(f"  Using LLM-enriched data: {df.shape}")
else:
    df = pd.read_csv("training_data_with_city.csv")
    log(f"  Using original data (no LLM features): {df.shape}")

# Feature engineering (same as before)
df['hour_sin'] = np.sin(2 * np.pi * df['order_hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['order_hour'] / 24)
df['price_ratio'] = df['cand_avg_price'] / (df['cart_value'] + 1)
df['popularity_x_lift'] = df['cand_order_frequency'] * df['max_lift']
df['completeness_gap'] = 1.0 - df['completeness']
df['price_vs_user_avg'] = df['cand_avg_price'] / (df['user_avg_order_value'] + 1)
df['complement_fills_gap'] = ((df['cand_typical_role'] != 'anchor') & (df['fills_missing_slot'] == 1)).astype(int)
df['cart_size_bucket'] = pd.cut(df['items_in_cart'], bins=[0,1,2,3,100], labels=[0,1,2,3]).astype(int)
df['city_item_signal'] = df['city_lift'] * df['is_local_favorite']

META = ['order_id', 'candidate_item', 'aug_type', 'sample_weight', 'label']
CATS = ['meal_period', 'restaurant', 'cand_category', 'cand_veg_nonveg',
        'cand_cuisine', 'cand_typical_role', 'cand_flavor_profile', 'city']
# Include item_semantic_cluster as categorical if present
if 'item_semantic_cluster' in df.columns:
    CATS.append('item_semantic_cluster')
    df['item_semantic_cluster'] = df['item_semantic_cluster'].astype(str)

FEATS = [c for c in df.columns if c not in META]
NUMS = [c for c in FEATS if c not in CATS]
log(f"  {len(NUMS)} numeric + {len(CATS)} cat = {len(FEATS)} features")

# ================================================================
# 2. ENCODE CATEGORICALS
# ================================================================
log("\nSTEP 2: Encoding...")
cat_card = {}
encoders = {}
for col in CATS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    cat_card[col] = len(le.classes_)
    log(f"  {col}: {cat_card[col]} classes")
joblib.dump(encoders, os.path.join(OUT, 'label_encoders.pkl'))

# Normalize numeric features
num_means = df[NUMS].mean()
num_stds = df[NUMS].std().replace(0, 1)
df[NUMS] = (df[NUMS] - num_means) / num_stds
joblib.dump({'means': num_means, 'stds': num_stds}, os.path.join(OUT, 'num_scaler.pkl'))

# ================================================================
# 3. TEMPORAL SPLIT
# ================================================================
log("\nSTEP 3: Temporal split...")
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
for k in ['train', 'val', 'test']:
    log(f"  {k}: {masks[k].sum():,}")

val_oids  = df.loc[masks['val'], 'order_id'].values
test_oids = df.loc[masks['test'], 'order_id'].values
test_cands = df.loc[masks['test'], 'candidate_item'].values

# ================================================================
# 4. PYTORCH DATASET
# ================================================================
class CSAODataset(Dataset):
    def __init__(self, dataframe, mask, num_cols, cat_cols):
        sub = dataframe.loc[mask]
        self.nums = torch.tensor(sub[num_cols].values, dtype=torch.float32)
        self.cats = torch.tensor(sub[cat_cols].values, dtype=torch.long)
        self.labels = torch.tensor((sub['label'].values > 0.25).astype(np.float32), dtype=torch.float32)
        self.weights = torch.tensor(sub['sample_weight'].values, dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.nums[idx], self.cats[idx], self.labels[idx], self.weights[idx]

train_ds = CSAODataset(df, masks['train'], NUMS, CATS)
val_ds   = CSAODataset(df, masks['val'], NUMS, CATS)
test_ds  = CSAODataset(df, masks['test'], NUMS, CATS)

# ================================================================
# 5. MODEL DEFINITION
# ================================================================
class CrossNetworkV2(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.W = nn.ParameterList([nn.Parameter(torch.randn(input_dim, input_dim) * 0.01) for _ in range(num_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])
    def forward(self, x0):
        x = x0
        for i in range(self.num_layers):
            xw = torch.matmul(x, self.W[i])
            x = x0 * (xw + self.b[i]) + x
        return x

class DCNV2(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, emb_dim=8,
                 cross_layers=3, deep_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, min(emb_dim, (card + 1) // 2 + 1))
            for card in cat_cardinalities
        ])
        total_emb = sum(min(emb_dim, (c + 1) // 2 + 1) for c in cat_cardinalities)
        input_dim = num_numeric + total_emb
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.cross_net = CrossNetworkV2(input_dim, cross_layers)
        deep_layers = []
        prev_dim = input_dim
        for d in deep_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout),
            ])
            prev_dim = d
        self.deep_net = nn.Sequential(*deep_layers)
        self.head = nn.Sequential(
            nn.Linear(input_dim + deep_dims[-1], 64), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )
    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embs, dim=1)
        x0 = torch.cat([x_num, x_emb], dim=1)
        x0 = self.bn_input(x0)
        x_cross = self.cross_net(x0)
        x_deep = self.deep_net(x0)
        x_combined = torch.cat([x_cross, x_deep], dim=1)
        return self.head(x_combined).squeeze(1)

# ================================================================
# 6. TRAINING FUNCTION
# ================================================================
def train_model(config, train_ds, val_ds, val_oids, cat_cards, num_numeric,
                max_epochs=25, patience=6, verbose=False):
    """Train a DCN-v2 model with given config, return best val NDCG@10."""
    torch.manual_seed(SEED)
    
    batch_size = config['batch_size']
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)
    
    model = DCNV2(
        num_numeric=num_numeric,
        cat_cardinalities=cat_cards,
        emb_dim=config['emb_dim'],
        cross_layers=config['cross_layers'],
        deep_dims=config['deep_dims'],
        dropout=config['dropout']
    ).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    
    best_ndcg, best_epoch, wait = 0.0, 0, 0
    best_state = None
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, n_samples = 0.0, 0
        for x_num, x_cat, labels, weights in train_dl:
            x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
            labels, weights = labels.to(DEVICE), weights.to(DEVICE)
            logits = model(x_num, x_cat)
            loss = (criterion(logits, labels) * weights).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels); n_samples += len(labels)
        scheduler.step()
        
        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_num, x_cat, labels, weights in val_dl:
                x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
                probs = torch.sigmoid(model(x_num, x_cat)).cpu().numpy()
                all_preds.extend(probs); all_labels.extend(labels.numpy())
        
        val_preds = np.array(all_preds)
        val_labels = np.array(all_labels)
        val_ndcg = compute_ndcg(val_oids, val_labels, val_preds, k=10)
        
        if verbose:
            val_auc = roc_auc_score(val_labels, val_preds)
            log(f"  Epoch {epoch:2d} | loss={total_loss/n_samples:.4f} | val_AUC={val_auc:.4f} | val_NDCG@10={val_ndcg:.4f}")
        
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg; best_epoch = epoch; wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    
    return best_ndcg, best_epoch, best_state, model

# ================================================================
# 7. OPTUNA HYPERPARAMETER OPTIMIZATION
# ================================================================
log("\n" + "=" * 60)
log("STEP 4: Optuna HPO for DCN-v2 (10 trials)...")
log("-" * 60)

cat_cards = [cat_card[c] for c in CATS]
num_numeric = len(NUMS)
N_TRIALS = 10

def objective(trial):
    config = {
        'emb_dim': trial.suggest_int('emb_dim', 6, 16),
        'cross_layers': trial.suggest_int('cross_layers', 2, 5),
        'deep_dims': {
            'small': [128, 64],
            'medium': [256, 128, 64],
            'large': [512, 256, 128],
        }[trial.suggest_categorical('deep_arch', ['small', 'medium', 'large'])],
        'dropout': trial.suggest_float('dropout', 0.1, 0.4),
        'lr': trial.suggest_float('lr', 5e-4, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [1024, 2048, 4096]),
    }
    
    best_ndcg, best_ep, _, _ = train_model(
        config, train_ds, val_ds, val_oids, cat_cards, num_numeric,
        max_epochs=20, patience=5, verbose=False
    )
    
    log(f"  Trial {trial.number:2d}: NDCG@10={best_ndcg:.6f} (ep={best_ep}) | "
        f"arch={trial.params['deep_arch']} cross={config['cross_layers']} "
        f"emb={config['emb_dim']} dropout={config['dropout']:.2f} lr={config['lr']:.4f}")
    
    return best_ndcg

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
t_opt = time.time()
study.optimize(objective, n_trials=N_TRIALS)
opt_time = time.time() - t_opt

log(f"\n  Optuna finished in {opt_time/60:.1f} min")
log(f"  Best trial: {study.best_trial.number}")
log(f"  Best NDCG@10: {study.best_value:.6f}")
log(f"  Best params: {study.best_params}")

# ================================================================
# 8. RETRAIN BEST MODEL WITH FULL EPOCHS
# ================================================================
log("\n" + "=" * 60)
log("STEP 5: Retraining best model (40 epochs, patience=8)...")

bp = study.best_params
best_config = {
    'emb_dim': bp['emb_dim'],
    'cross_layers': bp['cross_layers'],
    'deep_dims': {
        'small': [128, 64],
        'medium': [256, 128, 64],
        'large': [512, 256, 128],
    }[bp['deep_arch']],
    'dropout': bp['dropout'],
    'lr': bp['lr'],
    'weight_decay': bp['weight_decay'],
    'batch_size': bp['batch_size'],
}

best_ndcg, best_epoch, best_state, model = train_model(
    best_config, train_ds, val_ds, val_oids, cat_cards, num_numeric,
    max_epochs=40, patience=8, verbose=True
)

log(f"\n  Best epoch: {best_epoch}, Best val NDCG@10: {best_ndcg:.4f}")
torch.save(best_state, os.path.join(OUT, 'dcn_best.pt'))

# ================================================================
# 9. TEST EVALUATION
# ================================================================
log("\n" + "=" * 60)
log("STEP 6: Test evaluation...")

model.load_state_dict(best_state)
model.to(DEVICE)
model.eval()

test_dl = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=0)
all_preds, all_labels, all_weights = [], [], []
with torch.no_grad():
    for x_num, x_cat, labels, weights in test_dl:
        x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
        probs = torch.sigmoid(model(x_num, x_cat)).cpu().numpy()
        all_preds.extend(probs); all_labels.extend(labels.numpy()); all_weights.extend(weights.numpy())

test_preds = np.array(all_preds)
test_labels = np.array(all_labels)
test_weights = np.array(all_weights)

dcn_auc = roc_auc_score(test_labels, test_preds, sample_weight=test_weights)
dcn_ap = average_precision_score(test_labels, test_preds, sample_weight=test_weights)
dcn_ll = log_loss(test_labels, np.clip(test_preds, 1e-7, 1-1e-7), sample_weight=test_weights)
log(f"  AUC={dcn_auc:.6f}  AP={dcn_ap:.6f}  LL={dcn_ll:.6f}")

er = ranking_metrics(test_oids, test_labels, test_preds)
log("\n  Ranking metrics:")
for k, v in sorted(er.items()):
    log(f"    {k:12s}: {v:.4f}")

# ================================================================
# 10. SEGMENT ANALYSIS
# ================================================================
log("\n" + "=" * 60)
log("STEP 7: Segment analysis...")
tdf = df.loc[masks['test']].copy()
tdf['pred'] = test_preds
tdf['yb'] = test_labels

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

# Check for aug_type
if 'aug_type' in tdf.columns:
    log("\n  By aug_type:")
    for at, g in tdf.groupby('aug_type'):
        if g['yb'].nunique() < 2: continue
        a = roc_auc_score(g['yb'], g['pred'])
        log(f"    {str(at):25s}: AUC={a:.4f} n={len(g):>6,}")

# ================================================================
# 11. PLOTS
# ================================================================
log("\nSTEP 8: Plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Placeholder history from final training (re-run with tracking)
axes[0, 0].text(0.5, 0.5, f'Best NDCG@10: {best_ndcg:.4f}\nBest Epoch: {best_epoch}',
                ha='center', va='center', fontsize=14, transform=axes[0, 0].transAxes)
axes[0, 0].set_title('DCN-v2 (Optuna-tuned)')

# Optuna trial values
trial_nums = [t.number for t in study.trials]
trial_vals = [t.value for t in study.trials]
axes[0, 1].bar(trial_nums, trial_vals, color='steelblue')
axes[0, 1].set_title('Optuna Trial Values (NDCG@10)')
axes[0, 1].set_xlabel('Trial'); axes[0, 1].set_ylabel('Val NDCG@10')

# Score distribution
for lv, c, nm in [(0, 'red', 'Neg'), (1, 'green', 'Pos')]:
    axes[1, 0].hist(test_preds[test_labels == lv], bins=50, alpha=0.5, color=c, label=nm, density=True)
axes[1, 0].set_title('Score Distribution (DCN-v2 Optuna)'); axes[1, 0].legend()

# Ranking metrics bar
rn, rv = list(er.keys()), list(er.values())
co = ['#2196F3' if '@3' in n else '#4CAF50' if '@5' in n else '#FF9800' for n in rn]
axes[1, 1].barh(rn, rv, color=co)
axes[1, 1].set_xlim(0, 1); axes[1, 1].set_title('Ranking Metrics')
for i, v in enumerate(rv):
    axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'dcn_v2_plots.png'), dpi=150, bbox_inches='tight')
log("  Saved plots")

# ================================================================
# 12. SAVE
# ================================================================
log("\nSTEP 9: Saving...")

results = {
    'model': 'DCN-v2 (Optuna-tuned)',
    'optuna': {
        'n_trials': N_TRIALS,
        'best_trial': study.best_trial.number,
        'best_params': study.best_params,
        'optimization_time_min': opt_time / 60,
    },
    'architecture': {
        'num_numeric': len(NUMS),
        'cat_embeddings': {c: cat_card[c] for c in CATS},
        'best_config': {k: str(v) for k, v in best_config.items()},
    },
    'training': {
        'best_epoch': best_epoch,
        'best_val_ndcg10': best_ndcg,
    },
    'test_metrics': {'auc': dcn_auc, 'avg_precision': dcn_ap, 'log_loss': dcn_ll},
    'ranking': er,
    'features': FEATS,
    'cats': CATS,
    'num_features': NUMS,
}
with open(os.path.join(OUT, 'dcn_v2_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save predictions
pd.DataFrame({
    'order_id': test_oids,
    'candidate_item': test_cands,
    'y_true': df.loc[masks['test'], 'label'].values,
    'y_binary': test_labels,
    'dcn_pred': test_preds,
}).to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)

# Val predictions
model.eval()
val_dl = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=0)
val_preds_final = []
with torch.no_grad():
    for x_num, x_cat, labels, weights in val_dl:
        x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
        probs = torch.sigmoid(model(x_num, x_cat)).cpu().numpy()
        val_preds_final.extend(probs)
val_preds_final = np.array(val_preds_final)

pd.DataFrame({
    'order_id': val_oids,
    'y_binary': df.loc[masks['val'], 'label'].values > 0.25,
    'dcn_pred': val_preds_final,
}).to_csv(os.path.join(OUT, 'val_predictions.csv'), index=False)

with open(os.path.join(OUT, 'full_log.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))

elapsed = time.time() - t0
log(f"\n{'=' * 60}")
log(f"DONE! {elapsed/60:.1f} min | Best NDCG@10={best_ndcg:.4f}")
log(f"Files: {os.listdir(OUT)}")
log(f"{'=' * 60}")

with open(os.path.join(OUT, 'full_log.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
