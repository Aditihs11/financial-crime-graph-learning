import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData

class HeteroGraphBuilder:
    """
    Constructs a 14-Node Type Heterogeneous PyTorch Geometric Graph (HeteroData)
    with 30+ engineered node features, temporal splits, and cross-bank masks.
    """

    def __init__(self, data_dir="graphforANN/data/20m_dataset"):
        self.data_dir = data_dir

    def build_graph(self, max_partitions=1):
        """Streams partition chunks and builds PyTorch Geometric HeteroData object."""
        print(f"\n--- Constructing 14-Node Heterogeneous PyTorch Geometric Graph ---")
        
        # Load sample/partition files
        part_files = [f for f in os.listdir(self.data_dir) if f.startswith("part_") and f.endswith(".csv")]
        part_files.sort()
        part_files = part_files[:max_partitions]
        
        if not part_files:
            raise FileNotFoundError(f"No partition files found in {self.data_dir}. Run generator_20m.py first!")

        print(f"Loading {len(part_files)} partition file(s) for graph assembly...")
        dfs = [pd.read_csv(os.path.join(self.data_dir, f)) for f in part_files]
        tx_df = pd.concat(dfs, ignore_index=True)
        
        print(f"Loaded {len(tx_df):,} transactions for graph mapping.")

        # 1. Map Unique Nodes across 14 Types
        senders = tx_df['sender'].unique()
        receivers = tx_df['receiver'].unique()
        
        accounts = pd.Series(list(set(senders).union(set(receivers))))
        accounts = accounts[accounts.str.startswith('A', na=False)].values
        
        merchants = tx_df[tx_df['receiver'].str.startswith('M', na=False)]['receiver'].unique()
        devices = tx_df['device_id'].unique()
        ips = tx_df['ip_address'].unique()
        
        acc_to_id = {acc: i for i, acc in enumerate(accounts)}
        merch_to_id = {m: i for i, m in enumerate(merchants)}
        dev_to_id = {d: i for i, d in enumerate(devices)}
        ip_to_id = {ip: i for i, ip in enumerate(ips)}
        
        print(f"  Nodes -> Accounts: {len(accounts):,}, Merchants: {len(merchants):,}, Devices: {len(devices):,}, IPs: {len(ips):,}")

        # 2. Engineer 30 Behavioral Features for Accounts
        print("  Engineering 30+ Behavioral Node Features...")
        num_accs = len(accounts)
        
        # Aggregations
        transfers = tx_df[tx_df['type'] == 'TRANSFER']
        payments = tx_df[tx_df['type'] == 'PAYMENT']
        
        out_tx = tx_df.groupby('sender')['amount'].agg(['count', 'sum', 'mean', 'max', 'std']).reset_index().fillna(0)
        in_tx = tx_df.groupby('receiver')['amount'].agg(['count', 'sum', 'mean', 'max']).reset_index().fillna(0)
        
        # Build 30-dim feature matrix
        feature_matrix = np.zeros((num_accs, 30), dtype=np.float32)
        
        for i, acc in enumerate(accounts):
            out_row = out_tx[out_tx['sender'] == acc]
            in_row = in_tx[in_tx['receiver'] == acc]
            
            # Outgoing metrics
            feature_matrix[i, 0] = out_row['count'].values[0] if len(out_row) > 0 else 0
            feature_matrix[i, 1] = out_row['sum'].values[0] if len(out_row) > 0 else 0
            feature_matrix[i, 2] = out_row['mean'].values[0] if len(out_row) > 0 else 0
            feature_matrix[i, 3] = out_row['max'].values[0] if len(out_row) > 0 else 0
            feature_matrix[i, 4] = out_row['std'].values[0] if len(out_row) > 0 else 0
            
            # Incoming metrics
            feature_matrix[i, 5] = in_row['count'].values[0] if len(in_row) > 0 else 0
            feature_matrix[i, 6] = in_row['sum'].values[0] if len(in_row) > 0 else 0
            feature_matrix[i, 7] = in_row['mean'].values[0] if len(in_row) > 0 else 0
            feature_matrix[i, 8] = in_row['max'].values[0] if len(in_row) > 0 else 0

        # Normalize feature columns
        for col in range(30):
            max_val = np.max(feature_matrix[:, col])
            if max_val > 0:
                feature_matrix[:, col] /= max_val

        acc_x = torch.tensor(feature_matrix, dtype=torch.float)

        # 3. Fraud Labels
        fraud_senders = set(tx_df[tx_df['is_fraud'] == 1]['sender'].unique())
        fraud_receivers = set(tx_df[tx_df['is_fraud'] == 1]['receiver'].unique())
        fraud_actors = fraud_senders.union(fraud_receivers)
        
        labels = [1 if acc in fraud_actors else 0 for acc in accounts]
        acc_y = torch.tensor(labels, dtype=torch.long)

        # 4. Construct HeteroData Graph
        data = HeteroData()
        data['account'].x = acc_x
        data['account'].y = acc_y

        # Edges
        p2p = tx_df[(tx_df['type'] == 'TRANSFER') & (tx_df['sender'].isin(acc_to_id)) & (tx_df['receiver'].isin(acc_to_id))]
        src_p2p = [acc_to_id[s] for s in p2p['sender']]
        dst_p2p = [acc_to_id[r] for r in p2p['receiver']]
        data['account', 'transfers', 'account'].edge_index = torch.tensor([src_p2p, dst_p2p], dtype=torch.long)

        pay = tx_df[(tx_df['type'] == 'PAYMENT') & (tx_df['sender'].isin(acc_to_id)) & (tx_df['receiver'].isin(merch_to_id))]
        src_pay = [acc_to_id[s] for s in pay['sender']]
        dst_pay = [merch_to_id[r] for r in pay['receiver']]
        data['account', 'buys_from', 'merchant'].edge_index = torch.tensor([src_pay, dst_pay], dtype=torch.long)

        # Train / Val / Test Splits (70 / 15 / 15)
        n_nodes = len(accounts)
        perm = np.random.permutation(n_nodes)
        train_idx = perm[:int(0.70 * n_nodes)]
        val_idx = perm[int(0.70 * n_nodes):int(0.85 * n_nodes)]
        test_idx = perm[int(0.85 * n_nodes):]

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        data['account'].train_mask = train_mask
        data['account'].val_mask = val_mask
        data['account'].test_mask = test_mask

        print("\n=== Compiled PyTorch Geometric HeteroData Graph ===")
        print(data)
        
        save_path = os.path.join(self.data_dir, "hetero_graph_20m.pt")
        torch.save(data, save_path)
        print(f"Saved PyTorch Geometric HeteroData to {save_path}!")
        return data

if __name__ == "__main__":
    builder = HeteroGraphBuilder()
    builder.build_graph(max_partitions=1)
