# -*- coding: utf-8 -*-
"""
Tier 2: Deep Cross Network v2 (DCN-v2) for CSAO Rail Recommendation
=====================================================================
- Cross Network captures explicit feature interactions (e.g., "dessert candidate + missing dessert + dinner")
- Deep Network captures implicit non-linear patterns
- Combined with weighted BCE loss using sample_weight
- Same temporal split & features as GBDT pipeline
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
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = "model_output_dcn"
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
# 1. LOAD & ENGINEER (same as GBDT v2)
# ================================================================
log("=" * 60)
log("STEP 1: Loading data & feature engineering...")
t0 = time.time()
df = pd.read_csv("training_data_with_city.csv")
log(f"  Shape: {df.shape}")

# Same 9 engineered features
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
FEATS = [c for c in df.columns if c not in META]
NUMS = [c for c in FEATS if c not in CATS]
log(f"  {len(NUMS)} numeric + {len(CATS)} cat = {len(FEATS)} features")

# ================================================================
# 2. ENCODE CATEGORICALS
# ================================================================
log("\nSTEP 2: Encoding...")
cat_card = {}  # cardinality for embedding dims
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
# 3. TEMPORAL SPLIT (same as v2)
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
test_oids  = df.loc[masks['test'], 'order_id'].values
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

BATCH = 2048
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH*2, shuffle=False, num_workers=0)
test_dl  = DataLoader(test_ds,  batch_size=BATCH*2, shuffle=False, num_workers=0)

log(f"\n  Batches: train={len(train_dl)} val={len(val_dl)} test={len(test_dl)}")

# ================================================================
# 5. DCN-v2 MODEL
# ================================================================
class CrossNetworkV2(nn.Module):
    """DCN-v2 Cross Network: learns explicit feature crosses."""
    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.W = nn.ParameterList([nn.Parameter(torch.randn(input_dim, input_dim) * 0.01) for _ in range(num_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x0):
        x = x0
        for i in range(self.num_layers):
            # x_{l+1} = x0 * (W_l * x_l + b_l) + x_l
            xw = torch.matmul(x, self.W[i])  # (batch, dim)
            x = x0 * (xw + self.b[i]) + x
        return x


class DCNV2(nn.Module):
    """
    Deep & Cross Network v2 for CSAO recommendation.
    Architecture:
      - Categorical embeddings -> concat with numeric -> input x0
      - Cross Network (3 layers): explicit feature interactions
      - Deep Network (MLP): implicit interactions
      - Combine cross + deep -> final prediction
    """
    def __init__(self, num_numeric, cat_cardinalities, emb_dim=8,
                 cross_layers=3, deep_dims=[256, 128, 64], dropout=0.2):
        super().__init__()

        # Categorical embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, min(emb_dim, (card + 1) // 2 + 1))
            for card in cat_cardinalities
        ])
        total_emb = sum(min(emb_dim, (c + 1) // 2 + 1) for c in cat_cardinalities)
        input_dim = num_numeric + total_emb

        # Batch norm for input
        self.bn_input = nn.BatchNorm1d(input_dim)

        # Cross Network
        self.cross_net = CrossNetworkV2(input_dim, cross_layers)

        # Deep Network
        deep_layers = []
        prev_dim = input_dim
        for d in deep_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, d),
                nn.BatchNorm1d(d),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = d
        self.deep_net = nn.Sequential(*deep_layers)

        # Final combination: cross_output + deep_output -> 1
        self.head = nn.Sequential(
            nn.Linear(input_dim + deep_dims[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x_num, x_cat):
        # Embed categoricals
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embs, dim=1)

        # Concat numeric + embeddings
        x0 = torch.cat([x_num, x_emb], dim=1)
        x0 = self.bn_input(x0)

        # Cross path
        x_cross = self.cross_net(x0)

        # Deep path
        x_deep = self.deep_net(x0)

        # Combine
        x_combined = torch.cat([x_cross, x_deep], dim=1)
        out = self.head(x_combined).squeeze(1)
        return out


# ================================================================
# 6. TRAINING
# ================================================================
log("\n" + "=" * 60)
log("STEP 4: Building DCN-v2...")

cat_cards = [cat_card[c] for c in CATS]
model = DCNV2(
    num_numeric=len(NUMS),
    cat_cardinalities=cat_cards,
    emb_dim=10,
    cross_layers=3,
    deep_dims=[256, 128, 64],
    dropout=0.25
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
log(f"  Parameters: {total_params:,}")
log(f"  Architecture:\n{model}")

# Weighted BCE
criterion = nn.BCEWithLogitsLoss(reduction='none')

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)

EPOCHS = 40
PATIENCE = 8
best_ndcg = 0.0
best_epoch = 0
wait = 0
history = {'train_loss': [], 'val_auc': [], 'val_ndcg10': []}

log(f"\nSTEP 5: Training DCN-v2 ({EPOCHS} epochs, patience={PATIENCE})...")
log("-" * 60)

for epoch in range(1, EPOCHS + 1):
    # ---- Train ----
    model.train()
    total_loss, n_samples = 0.0, 0
    for x_num, x_cat, labels, weights in train_dl:
        x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
        labels, weights = labels.to(DEVICE), weights.to(DEVICE)

        logits = model(x_num, x_cat)
        loss_per_sample = criterion(logits, labels)
        loss = (loss_per_sample * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        n_samples += len(labels)

    train_loss = total_loss / n_samples
    scheduler.step()

    # ---- Validate ----
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_num, x_cat, labels, weights in val_dl:
            x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
            logits = model(x_num, x_cat)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())

    val_preds = np.array(all_preds)
    val_labels = np.array(all_labels)
    val_auc = roc_auc_score(val_labels, val_preds)
    val_ndcg = compute_ndcg(val_oids, val_labels, val_preds, k=10)

    history['train_loss'].append(train_loss)
    history['val_auc'].append(val_auc)
    history['val_ndcg10'].append(val_ndcg)

    lr_now = optimizer.param_groups[0]['lr']
    log(f"  Epoch {epoch:2d} | loss={train_loss:.4f} | val_AUC={val_auc:.4f} | val_NDCG@10={val_ndcg:.4f} | lr={lr_now:.6f}")

    if val_ndcg > best_ndcg:
        best_ndcg = val_ndcg
        best_epoch = epoch
        wait = 0
        torch.save(model.state_dict(), os.path.join(OUT, 'dcn_best.pt'))
    else:
        wait += 1
        if wait >= PATIENCE:
            log(f"  Early stopping at epoch {epoch} (best={best_epoch}, NDCG@10={best_ndcg:.4f})")
            break

log(f"\n  Best epoch: {best_epoch}, best val NDCG@10: {best_ndcg:.4f}")

# ================================================================
# 7. EVALUATION ON TEST SET
# ================================================================
log("\n" + "=" * 60)
log("STEP 6: Loading best model & evaluating on test...")

model.load_state_dict(torch.load(os.path.join(OUT, 'dcn_best.pt'), weights_only=True))
model.eval()

all_preds, all_labels, all_weights = [], [], []
with torch.no_grad():
    for x_num, x_cat, labels, weights in test_dl:
        x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
        logits = model(x_num, x_cat)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.numpy())
        all_weights.extend(weights.numpy())

test_preds = np.array(all_preds)
test_labels = np.array(all_labels)
test_weights = np.array(all_weights)

# Classification metrics
dcn_auc = roc_auc_score(test_labels, test_preds, sample_weight=test_weights)
dcn_ap = average_precision_score(test_labels, test_preds, sample_weight=test_weights)
dcn_ll = log_loss(test_labels, np.clip(test_preds, 1e-7, 1-1e-7), sample_weight=test_weights)
log(f"  DCN-v2     AUC={dcn_auc:.6f}  AP={dcn_ap:.6f}  LL={dcn_ll:.6f}")

# Ranking metrics
er = ranking_metrics(test_oids, test_labels, test_preds)
log("\n  Ranking metrics (DCN-v2):")
for k, v in sorted(er.items()):
    log(f"    {k:12s}: {v:.4f}")

# ================================================================
# 8. SEGMENT ANALYSIS
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
        log(f"    {str(nm):25s}: AUC={roc_auc_score(g['yb'], g['pred']):.4f}  n={len(g):>6,}")

log("\n  By cold_start:")
for cs in [0, 1]:
    g = tdf[tdf['user_is_cold_start'] == cs]
    if g['yb'].nunique() < 2: continue
    tag = 'Returning' if cs == 0 else 'Cold-start'
    log(f"    {tag:25s}: AUC={roc_auc_score(g['yb'], g['pred']):.4f}  n={len(g):>6,}")

# ================================================================
# 9. PLOTS
# ================================================================
log("\nSTEP 8: Plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(history['train_loss'], label='Train Loss', color='steelblue')
axes[0, 0].set_title('Training Loss')
axes[0, 0].set_xlabel('Epoch'); axes[0, 0].legend()

axes[0, 1].plot(history['val_auc'], label='Val AUC', color='green')
axes[0, 1].plot(history['val_ndcg10'], label='Val NDCG@10', color='orange')
axes[0, 1].axvline(x=best_epoch-1, color='red', linestyle='--', alpha=0.5, label='Best')
axes[0, 1].set_title('Validation Metrics')
axes[0, 1].set_xlabel('Epoch'); axes[0, 1].legend()

for lv, c, nm in [(0, 'red', 'Neg'), (1, 'green', 'Pos')]:
    axes[1, 0].hist(test_preds[test_labels == lv], bins=50, alpha=0.5, color=c, label=nm, density=True)
axes[1, 0].set_title('Score Distribution (DCN-v2)')
axes[1, 0].legend()

rn, rv = list(er.keys()), list(er.values())
co = ['#2196F3' if '@3' in n else '#4CAF50' if '@5' in n else '#FF9800' for n in rn]
axes[1, 1].barh(rn, rv, color=co)
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_title('Ranking Metrics (DCN-v2)')
for i, v in enumerate(rv):
    axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'dcn_plots.png'), dpi=150, bbox_inches='tight')
log("  Saved plots")

# ================================================================
# 10. SAVE
# ================================================================
log("\nSTEP 9: Saving...")

results = {
    'model': 'DCN-v2',
    'architecture': {
        'num_numeric': len(NUMS),
        'cat_embeddings': {c: cat_card[c] for c in CATS},
        'cross_layers': 3,
        'deep_dims': [256, 128, 64],
        'dropout': 0.25,
        'total_params': total_params,
    },
    'training': {
        'epochs_run': epoch,
        'best_epoch': best_epoch,
        'best_val_ndcg10': best_ndcg,
        'patience': PATIENCE,
        'batch_size': BATCH,
        'lr': 1e-3,
        'weight_decay': 1e-4,
    },
    'test_metrics': {
        'auc': dcn_auc,
        'avg_precision': dcn_ap,
        'log_loss': dcn_ll,
    },
    'ranking': er,
    'features': FEATS,
    'cats': CATS,
    'num_features': NUMS,
}

with open(os.path.join(OUT, 'dcn_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save test predictions for final ensemble
pd.DataFrame({
    'order_id': test_oids,
    'candidate_item': test_cands,
    'y_true': df.loc[masks['test'], 'label'].values,
    'y_binary': test_labels,
    'dcn_pred': test_preds,
}).to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)

# Save val predictions for final ensemble tuning
model.eval()
val_preds_final = []
with torch.no_grad():
    for x_num, x_cat, labels, weights in val_dl:
        x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
        logits = model(x_num, x_cat)
        probs = torch.sigmoid(logits).cpu().numpy()
        val_preds_final.extend(probs)
val_preds_final = np.array(val_preds_final)

pd.DataFrame({
    'order_id': val_oids,
    'y_binary': df.loc[masks['val'], 'label'].values > 0.25,
    'dcn_pred': val_preds_final,
}).to_csv(os.path.join(OUT, 'val_predictions.csv'), index=False)

with open(os.path.join(OUT, 'full_log.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))

log(f"\n{'=' * 60}")
log(f"DONE! {(time.time() - t0) / 60:.1f} min | Best NDCG@10={best_ndcg:.4f}")
log(f"Files: {os.listdir(OUT)}")
log(f"{'=' * 60}")

with open(os.path.join(OUT, 'full_log.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
