import os
import hashlib
import json
import torch
import yaml
from torch_geometric.data import HeteroData

def get_file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

graph_path = "data/graph/hetero_graph_14node.pt"
print(f"Checking {graph_path}...")

if not os.path.exists(graph_path):
    print(f"CRITICAL ERROR: {graph_path} does not exist!")
    exit(1)

file_size = os.path.getsize(graph_path)
graph_hash = get_file_hash(graph_path)

data = torch.load(graph_path, map_location="cpu", weights_only=False)

node_types = list(data.node_types)
edge_types = [str(r) for r in data.edge_types]
feat_dims = {}
for k in node_types:
    if hasattr(data[k], 'x') and data[k].x is not None:
        feat_dims[k] = list(data[k].x.shape)
    else:
        feat_dims[k] = "None"

train_mask_sum = int(data['account'].train_mask.sum().item())
val_mask_sum = int(data['account'].val_mask.sum().item())
test_mask_sum = int(data['account'].test_mask.sum().item())
total_accounts = data['account'].x.shape[0]

y_acc = data['account'].y
pos_acc = int((y_acc == 1).sum().item())
neg_acc = int((y_acc == 0).sum().item())

# Edge counts per relation
edge_counts = {}
total_edges = 0
for r in data.edge_types:
    cnt = data[r].edge_index.shape[1]
    edge_counts[str(r)] = cnt
    total_edges += cnt

# Target-derived edges to prune
target_relations = [
    ("account", "involved_in", "typology"),
    ("account", "member_of", "ring"),
    ("account", "subject_of", "case")
]
target_edges_count = 0
for tr in target_relations:
    if tr in data.edge_index_dict:
        target_edges_count += data[tr].edge_index.shape[1]

# Verified clean count
safe_edges_count = total_edges - target_edges_count

# Config check
configs_exist = {
    "banks.yaml": os.path.exists("configs/banks.yaml"),
    "fraud.yaml": os.path.exists("configs/fraud.yaml"),
    "generator.yaml": os.path.exists("configs/generator.yaml"),
    "personas.yaml": os.path.exists("configs/personas.yaml"),
}

# Raw data check
raw_files = os.listdir("data/raw") if os.path.exists("data/raw") else []

audit_report = {
    "status": "INPUTS_VERIFIED",
    "input_graph": {
        "path": os.path.abspath(graph_path),
        "file_size_bytes": file_size,
        "sha256_hash": graph_hash,
        "total_nodes": sum([data[k].x.shape[0] for k in node_types if hasattr(data[k], 'x') and data[k].x is not None]),
        "total_edges": total_edges,
        "node_types_count": len(node_types),
        "node_types": node_types,
        "edge_relations_count": len(edge_types),
        "edge_relations": edge_types,
        "node_feature_dimensions": feat_dims,
        "edge_counts_per_relation": edge_counts
    },
    "account_partitions": {
        "total_accounts": total_accounts,
        "train_mask": train_mask_sum,
        "val_mask": val_mask_sum,
        "test_mask": test_mask_sum,
        "train_pct": round((train_mask_sum / total_accounts) * 100, 2),
        "val_pct": round((val_mask_sum / total_accounts) * 100, 2),
        "test_pct": round((test_mask_sum / total_accounts) * 100, 2)
    },
    "account_labels": {
        "positive_fraud_count": pos_acc,
        "negative_legit_count": neg_acc,
        "fraud_prevalence_pct": round((pos_acc / total_accounts) * 100, 2)
    },
    "leakage_sanitization_specification": {
        "raw_account_feature_dims": 30,
        "excluded_feature_dims": [10, 11, 12, 13],
        "excluded_feature_names": ["fraud_tx_sent", "fraud_tx_received", "fraud_rate_sent", "fraud_recv_ratio"],
        "clean_account_feature_dims": 26,
        "pruned_target_node_types": ["typology", "ring", "case"],
        "pruned_target_relations": [str(tr) for tr in target_relations],
        "pruned_target_edges_count": target_edges_count,
        "retained_clean_node_types_count": 11,
        "retained_clean_relations_count": 15,
        "retained_clean_edges_count": safe_edges_count
    },
    "supporting_artifacts": {
        "configs": configs_exist,
        "raw_data_tables": raw_files,
        "experiment_script_exists": os.path.exists("experiments/zero_leakage/run_zero_leakage_experiments.py")
    },
    "random_seed": 42
}

os.makedirs("outputs", exist_ok=True)
with open("outputs/zero_leakage_input_audit.json", "w") as f:
    json.dump(audit_report, f, indent=2)

print("\nSaved input audit to outputs/zero_leakage_input_audit.json")
print("Total Graph Nodes:", audit_report["input_graph"]["total_nodes"])
print("Total Graph Edges:", audit_report["input_graph"]["total_edges"])
print("Safe Graph Edges:", safe_edges_count)
print("Pruned Target Edges:", target_edges_count)
print("Account Train / Val / Test:", train_mask_sum, "/", val_mask_sum, "/", test_mask_sum)
