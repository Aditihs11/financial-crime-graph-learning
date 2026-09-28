# Financial Crime Graph Learning (AMECF)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.3+-3C2179)](https://pyg.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Research-00C853)](#project-status--what-works-today)
[![Audit](https://img.shields.io/badge/Leakage%20Audit-100%25%20Zero--Leakage-blue)](#scientific-audit--zero-leakage-methodology)

> **Adaptive Multi-Entity Context Fusion (AMECF) & Continual Graph Learning for Multi-Bank Financial Crime Intelligence**

---

## Executive Summary

Detecting financial crime, money laundering syndicates, and fraudulent transaction networks across modern commercial banking ecosystems is fundamentally a **heterogeneous graph problem**. Criminal rings actively fragment transactions across payment channels, shell companies, synthetic identities, mule accounts, ATMs, and OTC crypto gateways to evade isolated rule-based filters.

This repository presents **AMECF (Adaptive Multi-Entity Context Fusion)**—a heterogeneous graph learning framework designed for multi-bank financial transaction networks. The system operates on a synthetic ecosystem of **20,000,000 multi-bank transactions** spanning 10 commercial bank silos, constructing a unified heterogeneous graph to uncover complex relational topologies and multi-hop layering schemes.

---

## Project Status: What Works Today

| Component | Status | Description |
|---|:---:|---|
| **Synthetic Multi-Bank Data Engine** | **Operational** | Generates 20M transactions across 10 bank silos, 8 customer archetypes, and 12 crime typologies. |
| **Heterogeneous Graph Pipeline** | **Operational** | Ingests multi-table relational schema into PyTorch Geometric `HeteroData` tensors (1.33M nodes, 9.4M clean edges). |
| **Automated Data Validation Suite** | **Operational** | 7-point automated integrity verification (`src/validation/validate.py`) covering IDs, timestamps, foreign keys, and ratios. |
| **Scientific Leakage Audit Engine** | **Operational** | Audits tensor checksums, features, and relations (`scripts/audit_inputs.py` $\rightarrow$ `outputs/zero_leakage_input_audit.json`). |
| **Sanitized Zero-Leakage Dataset** | **Operational** | Pruned 4 target-derived account features (leaving 26 clean behavioral dims) and 278k target-derived relation edges. |
| **Clean Tabular & Spatial Baselines** | **Operational** | Fully benchmarked Random Forest, HistGradientBoosting, and P2P GraphSAGE on clean 26-dim features. |
| **FastAPI REST Service** | **Operational** | 21 production REST endpoints for health, telemetry, AML risk scoring, path tracing, and SAR narrative generation. |
| **Interactive Streamlit Dashboard** | **Operational** | Multi-page investigative UI featuring cross-bank corridor maps, node topology exploration, and copilot reasoning. |

---

## Scientific Audit & Zero-Leakage Methodology

In preliminary experiments across literature and early iterations of this project, tabular baselines (Random Forest, XGBoost) and preliminary GNNs exhibited suspiciously high performance ($\text{ROC-AUC} \ge 0.9945$). 

A comprehensive forensic audit of the data generation pipeline revealed **three severe vectors of ground-truth target leakage**:

```
[Target Leakage Forensic Audit]
 ├── 1. Account Feature Leakage
 │    ├── Dim 10: fraud_tx_sent       (Ground-truth fraudulent transfer count)
 │    ├── Dim 11: fraud_tx_received   (Ground-truth inbound fraud transfer count)
 │    ├── Dim 12: fraud_rate_sent     (Ground-truth outbound fraud ratio)
 │    └── Dim 13: fraud_recv_ratio    (Ground-truth inbound fraud ratio)
 ├── 2. Auxiliary Entity Leakage
 │    ├── merchant, atm, device, ip_address embedded historical fraud rates
 └── 3. Graph Topological Leakage
      ├── ('account', 'involved_in', 'typology') -> 92,115 edges to ground-truth crime IDs
      ├── ('account', 'member_of', 'ring')       -> 93,401 edges to crime ring IDs
      └── ('account', 'subject_of', 'case')      -> 93,401 edges to filed SAR cases
```

### Remediation Protocol Enforced
To establish genuine scientific provenance, we applied strict zero-leakage constraints:
1. **Feature Vector Sanitization:** Excluded dimensions `[10, 11, 12, 13]`. Retained **26 pure behavioral dimensions** (transaction velocities, amount statistics, structuring ratios, off-hours frequency, counterparty diversity).
2. **Auxiliary Entity Sanitization:** Removed all embedded fraud rates from merchants, ATMs, devices, and IPs.
3. **Graph Topology Sanitization:** Pruned all 3 target-derived node types (`typology`, `ring`, `case`) and all 3 target-derived edge relations (`involved_in`, `member_of`, `subject_of`).
4. **Net Topology:** Reduced graph from 14 node types / 18 relations down to **11 legitimate entity types / 15 safe structural relations** (pruning 278,917 target edges).

---

## Verified Empirical Benchmark Results

The following table presents the **actual, verified experimental results** obtained under strict zero-leakage conditions on the 487,815-account dataset (341,470 train / 73,172 val / 73,173 test):

| Model Architecture | Input Representation | Leakage Status | ROC-AUC | PR-AUC | F1-Score | Recall | Precision | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Legacy)** | 30 Dims (w/ fraud counters) | **LEAKED** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Discredited |
| **XGBoost (Legacy)** | 30 Dims (w/ fraud counters) | **LEAKED** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Discredited |
| **Hetero-GraphSAGE (Legacy)** | 30 Dims + 18 Relations | **LEAKED** | 0.9945 | 0.9890 | 0.9945 | 0.9891 | 1.0000 | Discredited |
| **Random Forest (Clean)** | **26 Clean Behavioral Dims** | **ZERO-LEAKAGE** | **0.7231** | **0.4911** | **0.4047** | 0.3204 | 0.5489 | **Verified** |
| **HistGradientBoosting (Clean)**| **26 Clean Behavioral Dims** | **ZERO-LEAKAGE** | **0.7230** | **0.4909** | **0.4010** | 0.3168 | 0.5458 | **Verified** |
| **GraphSAGE P2P (Clean)** | **26 Dims + P2P Account Graph**| **ZERO-LEAKAGE** | **0.6876** | **0.4560** | **0.3560** | 0.2810 | 0.4850 | **Verified** |

> **Key Scientific Insight:** The drop in baseline tree ROC-AUC from $1.0000$ to $0.7231$ upon removing ground-truth counters confirms the elimination of target leakage. Detecting complex money laundering without label leakage is a non-trivial challenge that purely tabular baselines cannot solve alone.

---

## Experiments in Progress

The following advanced graph experiments are currently being evaluated and optimized:

1. **Multi-Relational Heterogeneous GNN (15 Relations):**
   - Evaluating multi-hop cross-entity relational message passing across all 11 clean entity types (`account`, `merchant`, `atm`, `device`, `ip_address`, `bank`, `channel`, `customer`, `merchant_sector`, `country`, `crypto_wallet`).
2. **AMECF (Heterogeneous Graph Transformer - HGT):**
   - Incorporating entity-specific projection heads, multi-head relational attention, and LayerNorm residual connections to fuse cross-bank context.
3. **Elastic Weight Consolidation (EWC) Continual Learning:**
   - Sequential training on Task A (traditional smurfing/structuring) followed by Task B (rapid multi-hop crypto layering).
   - Evaluating stability-plasticity hyperparameter sweep: $\lambda \in \{0.0, 100.0, 400.0, 5000.0\}$ using the Fisher Information diagonal:
     $$\mathcal{L}_{EWC}(\theta) = \mathcal{L}_{\text{Task B}}(\theta) + \sum_{i} \frac{\lambda}{2} F_i (\theta_i - \theta_{A,i}^*)^2$$

*Note on Compute Environment:* Full-batch attention across 1.33 million nodes in pure-Python CPU PyTorch Geometric requires substantial compute time. We are finalizing optimized mini-batch neighbor sampling (`NeighborLoader`) to accelerate GPU and CPU convergence.

---

## Repository Structure

```
financial-crime-graph-learning/
├── configs/                              # Simulation and model configuration YAMLs
│   ├── banks.yaml                        # 10 Simulated commercial banks
│   ├── fraud.yaml                        # 12 Crime typologies & parameter rules
│   ├── generator.yaml                    # Transaction volume, velocity, & amount distributions
│   └── personas.yaml                     # Legitimate behavioral archetypes
├── experiments/
│   └── zero_leakage/
│       └── run_zero_leakage_experiments.py # Master zero-leakage benchmark suite
├── outputs/                              # Benchmark artifacts and validation logs
│   ├── zero_leakage_input_audit.json    # Audited tensor checksums & split parameters
│   └── zero_leakage/                     # Experiment metrics, provenance, and plots
├── research_paper/                       # IEEE formatted LaTeX manuscript files
│   ├── main.tex                          # Primary LaTeX document
│   ├── references.bib                    # 16 academic citations
│   └── sections/                         # 12 modular sections
├── scripts/
│   ├── audit_inputs.py                   # Standalone input tensor verification script
│   └── generate_figures.py               # Publication figure generation script
├── src/                                  # Core library source code
│   ├── data_generator/                   # Synthetic multi-bank transaction engine
│   ├── graph_builder/                    # PyG HeteroData graph construction
│   ├── models/                           # AMECF and GNN model architectures
│   └── validation/                       # 7-point automated validation checks
├── backend/                              # FastAPI REST service
├── frontend/                             # Streamlit intelligence platform
├── requirements.txt                      # Project Python dependencies
└── README.md                             # Project documentation
```

---

## Installation & How to Run

### 1. Prerequisites & Environment Setup
Clone the repository and create an isolated Python 3.10+ virtual environment:

```bash
git clone https://github.com/Aditihs11/financial-crime-graph-learning.git
cd financial-crime-graph-learning

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Verify Data & Graph Inputs
Run the input audit tool to verify graph tensor integrity, entity partitions, and zero-leakage constraints:

```bash
python scripts/audit_inputs.py
```

Expected output:
```text
Checking data/graph/hetero_graph_14node.pt...
Saved input audit to outputs/zero_leakage_input_audit.json
Total Graph Nodes: 1333398
Total Graph Edges: 9698248
Safe Graph Edges: 9419331
Pruned Target Edges: 278917
Account Train / Val / Test: 341470 / 73172 / 73173
```

### 3. Run the Zero-Leakage Benchmark Suite
Execute the verified clean benchmark suite (Random Forest, Gradient Boosting, GraphSAGE P2P):

```bash
python experiments/zero_leakage/run_zero_leakage_experiments.py
```

Results and evaluation curves are automatically exported to `outputs/zero_leakage/`.

### 4. Run Automated Data Validation Checks
Validate dataset consistency, foreign key constraints, and merchant completeness:

```bash
python -m src.validation.validate
```

### 5. Launch the Intelligence Platform (Backend + UI)
Launch the FastAPI REST server and Streamlit dashboard:

```bash
python start_system.py
```
- **REST API Documentation:** `http://127.0.0.1:8000/docs`
- **Investigative Dashboard:** `http://localhost:8501`

---

## License & Academic Citation

This project is released under the **MIT License**.

If you use this work, codebase, or methodology in your research, please cite:

```bibtex
@article{amecf_graph_learning_2026,
  author    = {Aditi et al.},
  title     = {Adaptive Multi-Entity Context Fusion and Continual Graph Learning for Multi-Bank Financial Crime Detection},
  journal   = {IEEE Transactions / Conference Proceedings},
  year      = {2026}
}
```
