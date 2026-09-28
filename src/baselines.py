import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero

class HeteroGraphSAGE(nn.Module):
    """Heterogeneous GraphSAGE model for account node classification."""
    def __init__(self, hidden_channels=64, out_channels=2):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict['account'], edge_index_dict[('account', 'transfers', 'account')])
        x = F.relu(x)
        x = self.conv2(x, edge_index_dict[('account', 'transfers', 'account')])
        x = F.relu(x)
        out = self.classifier(x)
        return out

def evaluate_baselines(data_path="graphforANN/data/20m_dataset/hetero_graph_20m.pt"):
    print("\n==========================================================================")
    print("   RUNNING BENCHMARK EVALUATION SUITE: TRADITIONAL ML VS. GRAPH GNNs")
    print("==========================================================================")

    if not os.path.exists(data_path):
        print(f"Graph file {data_path} not found. Please run hetero_builder.py first.")
        return

    data = torch.load(data_path)

    X = data['account'].x.numpy()
    y = data['account'].y.numpy()

    train_mask = data['account'].train_mask.numpy()
    val_mask = data['account'].val_mask.numpy()
    test_mask = data['account'].test_mask.numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f" Dataset stats -> Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f" Fraud ratio   -> Train: {y_train.mean()*100:.2f}%, Test: {y_test.mean()*100:.2f}%\n")

    results = []

    # 1. Baseline: Random Forest
    print(" [1/3] Training Random Forest Baseline...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)

    rf_roc = roc_auc_score(y_test, rf_probs)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, rf_probs)
    rf_pr_auc = auc(recall_vals, precision_vals)
    rf_f1 = f1_score(y_test, rf_preds)

    results.append({
        'Model': 'Random Forest',
        'Type': 'Traditional ML',
        'ROC-AUC': round(rf_roc, 4),
        'PR-AUC': round(rf_pr_auc, 4),
        'F1-Score': round(rf_f1, 4)
    })
    print(f"       -> Random Forest: ROC-AUC = {rf_roc:.4f}, PR-AUC = {rf_pr_auc:.4f}, F1 = {rf_f1:.4f}")

    # 2. Baseline: PyTorch Hetero-GraphSAGE
    print("\n [2/3] Training PyTorch Geometric Hetero-GraphSAGE GNN...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    in_channels = X.shape[1]
    model = HeteroGraphSAGE(hidden_channels=64, out_channels=2).to(device)
    
    # Simple PyTorch training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    acc_x = data['account'].x.to(device)
    acc_y = data['account'].y.to(device)
    edge_index = data['account', 'transfers', 'account'].edge_index.to(device)

    model.train()
    for epoch in range(1, 21):
        optimizer.zero_grad()
        # Direct SageConv on transfers
        out = model({'account': acc_x}, {('account', 'transfers', 'account'): edge_index})
        loss = criterion(out[train_mask], acc_y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model({'account': acc_x}, {('account', 'transfers', 'account'): edge_index})
        probs = F.softmax(out[test_mask], dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(out[test_mask], dim=1).cpu().numpy()

    gnn_roc = roc_auc_score(y_test, probs)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
    gnn_pr_auc = auc(recall_vals, precision_vals)
    gnn_f1 = f1_score(y_test, preds)

    results.append({
        'Model': 'Hetero-GraphSAGE',
        'Type': 'Graph Neural Network',
        'ROC-AUC': round(gnn_roc, 4),
        'PR-AUC': round(gnn_pr_auc, 4),
        'F1-Score': round(gnn_f1, 4)
    })
    print(f"       -> Hetero-GraphSAGE: ROC-AUC = {gnn_roc:.4f}, PR-AUC = {gnn_pr_auc:.4f}, F1 = {gnn_f1:.4f}")

    # Summary
    print("\n==========================================================================")
    print("   BENCHMARK EVALUATION SUMMARY")
    print("==========================================================================")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print("==========================================================================")

if __name__ == "__main__":
    evaluate_baselines()
