# graphforANN — Technical Architecture Documentation

## 1. System Overview

graphforANN is a modular, pipeline-based research system for training and evaluating Graph Neural Networks on Anti-Money Laundering (AML) tasks. The core contribution is the generation of a realistic, publication-quality synthetic financial transaction graph at 20M scale with ground-truth fraud labels.

---

## 2. Module Descriptions

### 2.1 Configuration Layer (`configs/`)

All system parameters are externalized to YAML files, eliminating hardcoded constants.

| File             | Purpose                                          |
|------------------|--------------------------------------------------|
| `generator.yaml` | Scale (20M), seed (42), partition count (10)    |
| `personas.yaml`  | 8 persona archetypes with behavioral parameters |
| `fraud.yaml`     | 12 fraud typology injection parameters          |
| `banks.yaml`     | 10 bank silo names, weights, and parameters     |

### 2.2 Generator Layer (`src/generator/`)

| Module             | Responsibility                                     |
|--------------------|----------------------------------------------------|
| `master_tables.py` | Generates 9 normalized relational entity tables   |
| `personas.py`      | Assigns personas to customers, governs behavior   |
| `behaviour.py`     | Simulates normal spending patterns per persona    |
| `stream.py`        | Streaming 20M transaction writer (10 partitions)  |

**Key Design Decisions:**
- Streaming architecture: generates one partition (~2M rows) at a time to maintain ≤4GB RAM usage
- Temporal realism: Gaussian hour-of-day distributions per persona
- Bank silo isolation: each account belongs to one bank silo

### 2.3 Fraud Engine (`src/fraud/`)

| Module       | Responsibility                                      |
|--------------|-----------------------------------------------------|
| `manager.py` | Plugin-based fraud scenario dispatcher             |

The fraud engine is scenario-driven — each typology is implemented as an independent injection function. This makes adding new fraud patterns trivial (add a new function, register it in the manager).

**Injection Logic:**
1. Select N% of accounts as fraud candidates (configurable per typology)
2. Generate synthetic fraud transaction sequences for those accounts
3. Set `is_fraud=1` labels on all generated fraud transactions
4. Write fraud transactions into the partition stream

### 2.4 Graph Builder (`src/graph/builder.py`) — Step 8

Converts raw CSV partitions into a PyTorch Geometric `HeteroData` object.

**Graph Schema (14 Nodes, 18 Edges):**
```
Nodes:  account, merchant, atm, device, ip_address, bank, channel, customer, merchant_sector, typology, ring, case, country, crypto_wallet
Edges:  transfers_to, buys_from, withdraws_from, uses_device, from_ip, belongs_to, transacts_via, owned_by, operates_in, involved_in, member_of, subject_of, geolocated_in, has_wallet, located_in, assigned_to, accepts_crypto
```

**Stratified 70/15/15 Split:**
- Applies class-stratified (fraud/legit) split for unbiased evaluation
- Uses `numpy.random.default_rng(42)` for reproducibility

### 2.5 Feature Engineering (`src/features/engineer.py`) — Step 9

Generates 30-dimensional behavioral feature vectors per account node.

**Feature Categories:**
- **Volume features (0–7):** Transaction count, total, mean, max (in/out)
- **Connectivity (8–9):** Unique counterparties
- **Fraud exposure (10–13):** Raw fraud counts and rates
- **Temporal (14–18):** Night ratio, weekend ratio, velocity, inter-gap stats
- **Structural (19–22):** Cross-bank ratio, ATM/transfer/payment ratios
- **Statistical (23–25):** Std dev, skewness, 75th percentile
- **Device/IP (26–27):** Device and IP diversity
- **AML signals (28–29):** Round-amount ratio, structuring score

**Normalization:** All features are min-max normalized to [0, 1] per dimension.

### 2.6 Model Benchmark Suite (`src/models/baselines.py`) — Step 10

Three-way comparison:

| Model          | Library          | Class Balancing       | Epochs |
|----------------|------------------|-----------------------|--------|
| Random Forest  | scikit-learn     | `class_weight=balanced` | N/A  |
| XGBoost        | xgboost          | `scale_pos_weight`    | 300    |
| GraphSAGE      | PyTorch Geometric| `CrossEntropyLoss(weight)` | 30 |

**Evaluation Metrics:**
- ROC-AUC (primary)
- PR-AUC (fraud recall focus)
- F1, Precision, Recall
- Training time (seconds)
- Inference throughput (transactions/second)

### 2.7 Validation Suite (`src/validation/validate.py`) — Step 7

7-point automated integrity check:

1. Zero duplicate `transaction_id`s
2. 100% merchant sector coverage
3. 100% timestamp completeness
4. Zero negative amounts
5. All `device_id`s bound to master table
6. Fraud ratio in [4.5%, 5.5%]
7. All 10 banks present

Outputs: `[PASS]` / `[FAIL]` ASCII markers (Windows `cp1252` compatible).

### 2.8 Research Dashboard (`app.py`) — Step 11

6-page Streamlit application:

| Page                    | Content                                         |
|-------------------------|-------------------------------------------------|
| Overview & Metrics      | System KPIs, pipeline diagram, scale targets   |
| Dataset Explorer        | Partition viewer, amount distributions, time   |
| Fraud Typology Atlas    | 12 typology details, donut chart, indicators   |
| Persona Profiles        | 8 persona archetypes, risk comparison chart    |
| Validation Report       | 7-check status, SAR case log                   |
| Model Benchmarks        | ROC-AUC bar, radar chart, speed vs accuracy    |

---

## 3. Data Flow

```
configs/*.yaml
      ↓
src/generator/master_tables.py  →  data/raw/*.csv
      ↓
src/generator/stream.py         →  data/20m_dataset/part_000N.csv
      ↓ (parallel)
src/fraud/manager.py            →  fraud labels injected inline
      ↓
src/validation/validate.py      →  7/7 PASS verification
      ↓
src/graph/builder.py
  + src/features/engineer.py    →  data/graph/hetero_graph_14node.pt
      ↓
src/models/baselines.py         →  data/graph/benchmark_results.json
      ↓
app.py (Streamlit)              →  Research Dashboard
```

---

## 4. Reproducibility

All random seeds are set to `42` across:
- Python `random` module
- NumPy (`np.random.seed(42)`)
- NumPy Generator (`np.random.default_rng(42)`)
- PyTorch (`torch.manual_seed(42)`)
- XGBoost (`random_state=42`)
- scikit-learn (`random_state=42`)

---

## 5. Limitations & Future Work

| Limitation                    | Future Enhancement                              |
|-------------------------------|-------------------------------------------------|
| 2M rows used for graph (2 parts) | Full 20M graph with graph sampling (SAINT) |
| GraphSAGE only                | Add GAT, HGT, RGCN comparisons                 |
| Static graph                  | Temporal GNN (TGN) implementation              |
| Simulated data                | Blend with public datasets (Elliptic, PAYSIM)  |
| Federated learning stub       | Full FedAvg across 10 bank silos               |

---

## 6. Known Issues

- Windows `cp1252` encoding: use `encoding='utf-8'` when reading CSVs in non-Windows environments
- `torch.load()` requires `weights_only=False` for custom `HeteroData` objects (PyTorch 2.x)
- XGBoost `use_label_encoder` deprecation: parameter removed in XGBoost >= 1.6

---

*Documentation for graphforANN Phase 2 — Aryan Gowda, 2026*
