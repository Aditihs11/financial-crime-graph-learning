# Data Directory

Due to file size constraints on GitHub (>100MB per file and repository limits), raw partition CSVs and large PyTorch Geometric graph tensors are excluded from version control via `.gitignore`.

### Expected Data Artifacts

1. **`data/20m_dataset/`**:
   - Contains 10 partitioned CSV files (`part_0000.csv` to `part_0009.csv`) totaling 20,000,000 multi-bank transaction records.
   - Generated via `python run_pipeline.py --generate-data` using configurations in `configs/`.

2. **`data/raw/`**:
   - Contains 9 entity master tables (`accounts.csv`, `banks.csv`, `companies.csv`, `countries.csv`, `crypto_wallets.csv`, `customers.csv`, `devices.csv`, `ip_addresses.csv`, `merchants.csv`).

3. **`data/graph/`**:
   - `hetero_graph_14node.pt`: Serialized PyTorch Geometric `HeteroData` tensor (227.4 MB, SHA-256: `501c8bbe458099f32eee4f58db8df83a919361f8ec7adca5e7d3dd37bc7d7d55`).
   - Built via the graph ingestion pipeline (`src/graph_builder/`).
   - Audited via `python scripts/audit_inputs.py`.
