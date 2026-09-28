"""
run_zero_leakage_audit.py
Forensic verification of data leakage across:
1. Full 30-dim features vs 26-dim Clean features
2. Pure behavioral topology vs Full heterogeneous graph with typology/ring/case edges
"""

import sys
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

print("="*70, flush=True)
print("FORENSIC CODE & LEAKAGE AUDIT", flush=True)
print("="*70, flush=True)

path = "data/graph/hetero_graph_14node.pt"
data = torch.load(path, map_location="cpu", weights_only=False)

X_30 = data["account"].x.numpy()
y = data["account"].y.numpy()
train_mask = data["account"].train_mask.numpy()
val_mask = data["account"].val_mask.numpy()
test_mask = data["account"].test_mask.numpy()

# Clean 26 columns (strictly dropping historical fraud counters 10-13)
clean_cols = [c for c in range(30) if c not in [10, 11, 12, 13]]
X_26 = X_30[:, clean_cols]

print(f"Total Accounts: {len(y):,}", flush=True)
print(f"Total Fraud Accounts: {y.sum():,} ({y.mean()*100:.2f}%)", flush=True)
print(f"Account Feature Dims: Full={X_30.shape[1]}, Clean={X_26.shape[1]}", flush=True)

# 1. Evaluate Tabular with 30 dims (Leaked) vs 26 dims (Clean)
print("\n--- 1. TABULAR LEAKAGE AUDIT ---", flush=True)

# 30-dim RF
rf_30 = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
rf_30.fit(X_30[train_mask], y[train_mask])
p30 = rf_30.predict_proba(X_30[test_mask])[:, 1]
f30 = f1_score(y[test_mask], (p30 >= 0.5).astype(int))
print(f"[Random Forest - 30 Dims (With Historical Fraud Dims 10-13)]: F1 = {f30:.4f}, ROC-AUC = {roc_auc_score(y[test_mask], p30):.4f}", flush=True)

# 26-dim RF
rf_26 = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
rf_26.fit(X_26[train_mask], y[train_mask])
p26 = rf_26.predict_proba(X_26[test_mask])[:, 1]
f26 = f1_score(y[test_mask], (p26 >= 0.5).astype(int))
print(f"[Random Forest - 26 Dims (Clean, Zero-Leakage)]:            F1 = {f26:.4f}, ROC-AUC = {roc_auc_score(y[test_mask], p26):.4f}", flush=True)

# 26-dim Boosted Trees
hgb_26 = HistGradientBoostingClassifier(max_iter=50, max_depth=6, random_state=42)
hgb_26.fit(X_26[train_mask], y[train_mask])
phgb = hgb_26.predict_proba(X_26[test_mask])[:, 1]
fhgb = f1_score(y[test_mask], (phgb >= 0.5).astype(int))
print(f"[HistGradientBoosting - 26 Dims (Clean, Zero-Leakage)]:     F1 = {fhgb:.4f}, ROC-AUC = {roc_auc_score(y[test_mask], phgb):.4f}", flush=True)

# 2. Graph Topology Edge Audit
print("\n--- 2. GRAPH TOPOLOGY EDGE AUDIT ---", flush=True)
edge_types = list(data.edge_index_dict.keys())
print(f"Total Edge Types in Graph Tensor: {len(edge_types)}", flush=True)
for s, r, d in edge_types:
    n_edges = data.edge_index_dict[(s, r, d)].shape[1]
    is_target_related = ("typology" in d or "ring" in d or "case" in d)
    tag = " [TARGET-DERIVED EDGE]" if is_target_related else " [BEHAVIORAL/ENTITY EDGE]"
    print(f"  ({s} -> {r} -> {d}): {n_edges:,} edges {tag}", flush=True)

# 3. Clean GraphSAGE on Pure Behavioral Topology + Clean 26 Dims
print("\n--- 3. CLEAN GRAPHSAGE AUDIT (26 Dims + Pure P2P Transfer Topology) ---", flush=True)
class FastSAGE(nn.Module):
    def __init__(self, in_ch, hid=64, out_ch=2):
        super().__init__()
        self.c1 = SAGEConv(in_ch, hid)
        self.c2 = SAGEConv(hid, out_ch)
    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = self.c2(x, edge_index)
        return x

edge_index_p2p = data['account', 'transfers_to', 'account'].edge_index
x_clean_t = torch.from_numpy(X_26).float()
y_t = torch.from_numpy(y).long()
tr_m = torch.from_numpy(train_mask)
te_m = torch.from_numpy(test_mask)

model_sage = FastSAGE(in_ch=26, hid=64, out_ch=2)
opt = torch.optim.Adam(model_sage.parameters(), lr=0.02)
crit = nn.CrossEntropyLoss()

for epoch in range(1, 16):
    model_sage.train()
    opt.zero_grad()
    out = model_sage(x_clean_t, edge_index_p2p)
    loss = crit(out[tr_m], y_t[tr_m])
    loss.backward()
    opt.step()

model_sage.eval()
with torch.no_grad():
    out_eval = model_sage(x_clean_t, edge_index_p2p)
    probs = F.softmax(out_eval[te_m], dim=1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    y_test_np = y[test_mask]
    pr_v, rc_v, _ = precision_recall_curve(y_test_np, probs)
    print(f"[GraphSAGE - Clean 26 Dims + Pure P2P (Zero-Leakage)]:\n  ROC-AUC: {roc_auc_score(y_test_np, probs):.4f} | PR-AUC: {auc(rc_v, pr_v):.4f} | F1: {f1_score(y_test_np, preds):.4f} | Prec: {precision_score(y_test_np, preds):.4f} | Rec: {recall_score(y_test_np, preds):.4f}", flush=True)

print("="*70, flush=True)
