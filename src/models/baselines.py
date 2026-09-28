"""
Step 10: Baseline ML & GNN Model Benchmark Suite
==================================================
Evaluates the graphforANN 20M dataset across:

  [1] Random Forest          (Traditional ML - tabular)
  [2] XGBoost                (Traditional ML - boosted)
  [3] Hetero-GraphSAGE       (GNN - node classification)

Metrics:
  - ROC-AUC, PR-AUC, F1-Score, Precision, Recall
  - Training time
  - Inference throughput (transactions/sec)

Outputs:
  - data/graph/benchmark_results.json   (for dashboard)
  - data/graph/benchmark_results.csv    (for paper/report)

Usage:
  python src/models/baselines.py
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, precision_score, recall_score
)

# -------------------------
# Try XGBoost (optional)
# -------------------------
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("   [WARN] XGBoost not installed. Skipping XGBoost baseline.")

# -------------------------
# Try PyTorch Geometric
# -------------------------
try:
    from torch_geometric.nn import SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("   [WARN] PyTorch Geometric not installed. Skipping GNN baseline.")


# =========================================================
# GNN Model Definition
# =========================================================
if HAS_PYG:
    class GraphSAGEFraudNet(nn.Module):
        """
        2-layer GraphSAGE for fraud node classification.
        Operates on the account-account transfer graph.
        """
        def __init__(self, in_channels: int, hidden: int = 128, out_channels: int = 2, dropout: float = 0.3):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
            self.dropout = dropout
            self.classifier = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden // 2, out_channels)
            )

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            return self.classifier(x)


# =========================================================
# Benchmark Runner
# =========================================================
def run_benchmark(graph_path: str = None, results_dir: str = None):
    base = Path(__file__).resolve().parents[2]
    graph_path  = graph_path  or str(base / "data" / "graph" / "hetero_graph_14node.pt")
    results_dir = results_dir or str(base / "data" / "graph")
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("   STEP 10: BASELINE BENCHMARK SUITE")
    print("=" * 70)

    if not os.path.exists(graph_path):
        print(f"   [ERROR] Graph file not found: {graph_path}")
        print("   Please run Step 8 (graph builder) first.")
        return

    print(f"   Loading graph from: {graph_path}")
    data = torch.load(graph_path, map_location='cpu', weights_only=False)

    X           = data['account'].x.numpy()
    y           = data['account'].y.numpy()
    train_mask  = data['account'].train_mask.numpy()
    val_mask    = data['account'].val_mask.numpy()
    test_mask   = data['account'].test_mask.numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"\n   Dataset Stats:")
    print(f"     Train: {len(X_train):,}  (fraud: {y_train.sum():,} = {y_train.mean()*100:.1f}%)")
    print(f"     Val  : {len(X_val):,}  (fraud: {y_val.sum():,} = {y_val.mean()*100:.1f}%)")
    print(f"     Test : {len(X_test):,}  (fraud: {y_test.sum():,} = {y_test.mean()*100:.1f}%)")
    print()

    results = []

    # =====================================================
    # Model 1: Random Forest
    # =====================================================
    print("   [1/3] Random Forest Classifier...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    train_time = time.time() - t0

    t1 = time.time()
    rf_probs = rf.predict_proba(X_test)[:, 1]
    infer_time = time.time() - t1
    rf_preds = (rf_probs >= 0.5).astype(int)

    pr_vals, rc_vals, _ = precision_recall_curve(y_test, rf_probs)
    results.append({
        "model":      "Random Forest",
        "type":       "Traditional ML",
        "roc_auc":    round(float(roc_auc_score(y_test, rf_probs)), 4),
        "pr_auc":     round(float(auc(rc_vals, pr_vals)), 4),
        "f1":         round(float(f1_score(y_test, rf_preds, zero_division=0)), 4),
        "precision":  round(float(precision_score(y_test, rf_preds, zero_division=0)), 4),
        "recall":     round(float(recall_score(y_test, rf_preds, zero_division=0)), 4),
        "train_time_s": round(train_time, 2),
        "infer_tps":  round(len(X_test) / infer_time, 0),
    })
    print(f"       ROC-AUC={results[-1]['roc_auc']} | PR-AUC={results[-1]['pr_auc']} "
          f"| F1={results[-1]['f1']} | Train={train_time:.1f}s")

    # =====================================================
    # Model 2: XGBoost
    # =====================================================
    if HAS_XGB:
        print("\n   [2/3] XGBoost Classifier...")
        scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
        t0 = time.time()
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        train_time = time.time() - t0

        t1 = time.time()
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        infer_time = time.time() - t1
        xgb_preds = (xgb_probs >= 0.5).astype(int)

        pr_vals, rc_vals, _ = precision_recall_curve(y_test, xgb_probs)
        results.append({
            "model":      "XGBoost",
            "type":       "Traditional ML",
            "roc_auc":    round(float(roc_auc_score(y_test, xgb_probs)), 4),
            "pr_auc":     round(float(auc(rc_vals, pr_vals)), 4),
            "f1":         round(float(f1_score(y_test, xgb_preds, zero_division=0)), 4),
            "precision":  round(float(precision_score(y_test, xgb_preds, zero_division=0)), 4),
            "recall":     round(float(recall_score(y_test, xgb_preds, zero_division=0)), 4),
            "train_time_s": round(train_time, 2),
            "infer_tps":  round(len(X_test) / infer_time, 0),
        })
        print(f"       ROC-AUC={results[-1]['roc_auc']} | PR-AUC={results[-1]['pr_auc']} "
              f"| F1={results[-1]['f1']} | Train={train_time:.1f}s")
    else:
        results.append({
            "model": "XGBoost", "type": "Traditional ML",
            "roc_auc": None, "pr_auc": None, "f1": None,
            "precision": None, "recall": None, "train_time_s": None, "infer_tps": None,
            "note": "xgboost not installed"
        })

    # =====================================================
    # Model 3: Hetero-GraphSAGE GNN
    # =====================================================
    if HAS_PYG:
        print("\n   [3/3] Hetero-GraphSAGE GNN...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"       Device: {device}")

        # Get account->account edge_index
        edge_key = ('account', 'transfers_to', 'account')
        if edge_key in data.edge_index_dict:
            edge_index = data[edge_key].edge_index.to(device)
        else:
            # Fallback: empty edges
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        x_all = data['account'].x.to(device)
        y_all = data['account'].y.to(device)
        tmask = torch.tensor(train_mask, device=device)
        vmask = torch.tensor(val_mask,   device=device)
        emask = torch.tensor(test_mask,  device=device)

        in_ch = x_all.shape[1]
        model = GraphSAGEFraudNet(in_ch, hidden=128, out_channels=2, dropout=0.3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        # Class-balanced weights
        n_neg = int((y_all.cpu() == 0).sum())
        n_pos = int((y_all.cpu() == 1).sum())
        weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)

        EPOCHS = 30
        t0 = time.time()
        best_val_auc = 0
        best_state   = None

        for ep in range(1, EPOCHS + 1):
            model.train()
            optimizer.zero_grad()
            out  = model(x_all, edge_index)
            loss = criterion(out[tmask], y_all[tmask])
            loss.backward()
            optimizer.step()

            if ep % 5 == 0:
                model.eval()
                with torch.no_grad():
                    logits  = model(x_all, edge_index)
                    val_prb = F.softmax(logits[vmask], dim=1)[:, 1].cpu().numpy()
                    val_lbl = y_all[vmask].cpu().numpy()
                try:
                    val_auc = roc_auc_score(val_lbl, val_prb)
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                    print(f"       Epoch {ep:3d}/{EPOCHS} | loss={loss.item():.4f} | val_auc={val_auc:.4f}")
                except Exception:
                    pass

        train_time = time.time() - t0
        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        t1 = time.time()
        with torch.no_grad():
            logits = model(x_all, edge_index)
            gnn_probs = F.softmax(logits[emask], dim=1)[:, 1].cpu().numpy()
            gnn_preds = torch.argmax(logits[emask], dim=1).cpu().numpy()
        infer_time = time.time() - t1

        y_test_gnn = y_all[emask].cpu().numpy()
        pr_vals, rc_vals, _ = precision_recall_curve(y_test_gnn, gnn_probs)
        results.append({
            "model":      "GraphSAGE (GNN)",
            "type":       "Graph Neural Network",
            "roc_auc":    round(float(roc_auc_score(y_test_gnn, gnn_probs)), 4),
            "pr_auc":     round(float(auc(rc_vals, pr_vals)), 4),
            "f1":         round(float(f1_score(y_test_gnn, gnn_preds, zero_division=0)), 4),
            "precision":  round(float(precision_score(y_test_gnn, gnn_preds, zero_division=0)), 4),
            "recall":     round(float(recall_score(y_test_gnn, gnn_preds, zero_division=0)), 4),
            "train_time_s": round(train_time, 2),
            "infer_tps":  round(emask.sum().item() / infer_time, 0),
        })
        print(f"       ROC-AUC={results[-1]['roc_auc']} | PR-AUC={results[-1]['pr_auc']} "
              f"| F1={results[-1]['f1']} | Train={train_time:.1f}s")
    else:
        results.append({
            "model": "GraphSAGE (GNN)", "type": "Graph Neural Network",
            "roc_auc": None, "pr_auc": None, "f1": None,
            "precision": None, "recall": None, "train_time_s": None, "infer_tps": None,
            "note": "torch_geometric not installed"
        })

    # =====================================================
    # Save Results
    # =====================================================
    json_path = os.path.join(results_dir, "benchmark_results.json")
    csv_path  = os.path.join(results_dir, "benchmark_results.csv")

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("   BENCHMARK SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"   {r['model']:25s} | ROC-AUC={r['roc_auc']} | PR-AUC={r['pr_auc']} | F1={r['f1']}")
    print("=" * 70)
    print(f"   [SAVED] {json_path}")
    print(f"   [SAVED] {csv_path}")
    return results


if __name__ == "__main__":
    run_benchmark()
