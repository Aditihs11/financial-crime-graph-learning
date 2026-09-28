"""
run_advanced_pipeline.py
========================
Executes the Adaptive Multi-Entity Context Fusion (AMECF) model with Continual Learning (EWC).

This script simulates a Continual Learning environment:
 - Task A: Train on standard dataset
 - EWC Consolidation: Compute Fisher Information Matrix
 - Task B: Train on a simulated "future" dataset (novel fraud typologies) using EWC penalty.

Usage:
    python run_advanced_pipeline.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models.context_fusion import AdaptiveContextFusionNet
from src.models.continual_learning import EWC

def run_advanced():
    graph_path = ROOT / "data" / "graph" / "hetero_graph_14node.pt"
    if not graph_path.exists():
        print(f"[ERROR] {graph_path} not found. Run builder first.")
        return

    print("=" * 70)
    print("   LOADING GRAPH FOR ADVANCED PIPELINE")
    print("=" * 70)
    data = torch.load(str(graph_path), map_location='cpu', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # ── FAST CPU DEMONSTRATION PRUNING ──
    # To prevent HGTConv from bottlenecking the CPU on 9.6M edges, we sample 
    # a highly dense subgraph of edges for this fast demonstration.
    print("   [INFO] Pruning edges for ultra-fast CPU demonstration...")
    for k in data.edge_index_dict.keys():
        num_edges = data.edge_index_dict[k].size(1)
        max_edges = min(15000, num_edges)
        data.edge_index_dict[k] = data.edge_index_dict[k][:, :max_edges]
    
    # Send data to device
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    y_all = data['account'].y.to(device)
    
    train_mask = data['account'].train_mask.to(device)
    train_idx = train_mask.nonzero(as_tuple=True)[0]
    
    perm = torch.randperm(train_idx.size(0))
    train_idx = train_idx[perm]
    
    split_point = int(0.7 * train_idx.size(0))
    idx_A = train_idx[:split_point]
    idx_B = train_idx[split_point:]
    
    mask_A = torch.zeros_like(train_mask, dtype=torch.bool)
    mask_A[idx_A] = True
    mask_B = torch.zeros_like(train_mask, dtype=torch.bool)
    mask_B[idx_B] = True
    
    val_mask = data['account'].val_mask.to(device)
    test_mask = data['account'].test_mask.to(device)
    
    print(f"   Task A Nodes: {mask_A.sum().item():,}")
    print(f"   Task B Nodes: {mask_B.sum().item():,}")
    
    node_features_dict = {k: v.shape[1] for k, v in data.x_dict.items()}
    model = AdaptiveContextFusionNet(
        hidden_channels=32, 
        out_channels=2, 
        num_heads=1, 
        num_layers=1, 
        metadata=data.metadata(), 
        node_features_dict=node_features_dict
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("\n" + "=" * 70)
    print("   PHASE 1: TRAINING TASK A (Base Knowledge) - FAST GRAPH")
    print("=" * 70)
    
    EPOCHS_A = 5
    model.train()
    for ep in range(1, EPOCHS_A + 1):
        optimizer.zero_grad()
        out = model(x_dict, edge_index_dict)
        loss = criterion(out[mask_A], y_all[mask_A])
        loss.backward()
        optimizer.step()
        
        print(f"   Epoch {ep}/{EPOCHS_A} | Loss A: {loss.item():.4f}")
            
    print("\n" + "=" * 70)
    print("   COMPUTING FISHER INFORMATION MATRIX (EWC)")
    print("=" * 70)
    
    data.x_dict = x_dict
    data.edge_index_dict = edge_index_dict
    data['account'].train_mask = mask_A
    
    ewc = EWC(model, data, device)
    print("   Fisher Information Matrix computed.")
    
    print("\n" + "=" * 70)
    print("   PHASE 2: TRAINING TASK B WITH EWC PENALTY (Continual Learning)")
    print("=" * 70)
    
    EWC_WEIGHT = 400.0
    EPOCHS_B = 5
    
    model.train()
    for ep in range(1, EPOCHS_B + 1):
        optimizer.zero_grad()
        out = model(x_dict, edge_index_dict)
        
        loss_B = criterion(out[mask_B], y_all[mask_B])
        loss_ewc = ewc.penalty(model)
        loss = loss_B + EWC_WEIGHT * loss_ewc
        
        loss.backward()
        optimizer.step()
        
        print(f"   Epoch {ep}/{EPOCHS_B} | Loss B: {loss_B.item():.4f} | EWC Penalty: {loss_ewc.item():.6f}")

    print("\n" + "=" * 70)
    print("   EVALUATION ON TEST SET")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
        preds = torch.argmax(out[test_mask], dim=1)
        correct = (preds == y_all[test_mask]).sum().item()
        acc = correct / test_mask.sum().item()
        
        tp = ((preds == 1) & (y_all[test_mask] == 1)).sum().item()
        fp = ((preds == 1) & (y_all[test_mask] == 0)).sum().item()
        fn = ((preds == 0) & (y_all[test_mask] == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"   Test Accuracy: {acc*100:.2f}%")
    print(f"   Precision:     {precision:.4f}")
    print(f"   Recall:        {recall:.4f}")
    print(f"   F1-Score:      {f1:.4f}")
    print("\n   PIPELINE COMPLETE.")

if __name__ == "__main__":
    run_advanced()
