# 🚀 graphforANN — Complete Project Execution & Teammate Setup Guide

Welcome to **graphforANN**, a state-of-the-art, production-grade Anti-Money Laundering (AML) research system built to evaluate Graph Neural Networks (GNNs) on financial fraud detection.

This document contains **everything your teammates need to know** to set up, understand, and run the system on any machine.

---

## 📑 Table of Contents
1. [System Overview & Architecture](#-system-overview--architecture)
2. [Prerequisites & System Requirements](#-prerequisites--system-requirements)
3. [Quick Setup & Installation (3 Steps)](#-quick-setup--installation-3-steps)
4. [How to Run Each Module](#-how-to-run-each-module)
5. [Understanding the 14-Node Graph & Features](#-understanding-the-14-node-graph--features)
6. [Understanding the 12 Injected Fraud Typologies](#-understanding-the-12-injected-fraud-typologies)
7. [Dashboard Overview](#-dashboard-overview)
8. [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🕸️ System Overview & Architecture

**graphforANN** models a financial transaction network spanning **20 Million synthetic transactions** across **10 federated bank silos** (BANK_A → BANK_J) with a **5.00% fraud ratio** across **12 realistic money laundering typologies**.

### Pipeline Flow:
```text
[Master Relational Tables] ──► [Persona Engine] ──► [Normal Behaviour Simulator]
                                                           │
[Streamlit Research Dashboard] ◄── [GNN & ML Models] ◄── [14-Node HeteroGraph] ◄── [Fraud Injection Engine]
```

---

## 💻 Prerequisites & System Requirements

- **Operating System:** Windows 10/11, macOS, or Linux
- **Python Version:** Python 3.10+ (Python 3.11, 3.12, 3.14 tested)
- **RAM:** Minimum 8 GB (16 GB recommended for 20M graph loading)
- **Disk Space:** ~3 GB free space

---

## ⚡ Quick Setup & Installation (3 Steps)

### Step 1: Open Terminal / Command Prompt
Navigate to the project root directory:
```bash
cd C:\project
```
*(or your local unzipped project folder path)*

### Step 2: Install Required Dependencies
Run the standard requirement installer:
```bash
pip install -r requirements.txt
```

*Required Core Packages:*
- `torch >= 2.0.0` (PyTorch deep learning framework)
- `torch-geometric >= 2.3.0` (PyTorch Geometric for Heterogeneous Graphs)
- `streamlit >= 1.28.0` (Interactive Research Dashboard)
- `pandas >= 2.0.0`, `numpy >= 1.24.0` (Data Manipulation)
- `scikit-learn >= 1.3.0` (Traditional ML Baselines)
- `plotly >= 5.15.0` (Interactive Visualizations)

### Step 3: Verify Dataset Integrity (7/7 Validation Check)
Run the automated integrity suite to verify data files:
```bash
python src/validation/validate.py
```
*Expected Output:* `[PASS] Check 1/7 through 7/7 PASSED — Validation Report saved.`

---

## 🛠️ How to Run Each Module

### 1. Launch the Streamlit Interactive Research Dashboard
```bash
streamlit run app.py
```
- Opens automatically in your browser at: **`http://localhost:8501`**
- Features: System overview, dataset partition browser, fraud typology atlas, persona profiles, automated validation reports, and model benchmark comparisons.

---

### 2. Run the End-to-End Pipeline (Graph Build + Baseline Training)
To build the 14-node heterogeneous graph and evaluate baseline models in a single process:
```bash
python run_pipeline.py
```
- **Graph Built:** 1,333,398 nodes and 9,698,248 edge relations
- **Feature Engineering:** Computes 30-dim account feature vectors
- **Model Evaluation:** Trains Random Forest & GraphSAGE GNN
- **Outputs Saved:** `data/graph/hetero_graph_14node.pt` and `data/graph/benchmark_results.json`

---

### 3. Run Individual Components (Optional)

- **Re-run Automated Validation Checks:**
  ```bash
  python src/validation/validate.py
  ```
- **Re-build 14-Node Heterogeneous Graph Only:**
  ```bash
  python src/graph/builder.py
  ```
- **Re-run Baseline Model Benchmarks Only:**
  ```bash
  python src/models/baselines.py
  ```

---

## 🧬 Understanding the 14-Node Graph & Features

### 14 Heterogeneous Node Types:
1. `account`: Primary target node holding the 30-dim feature vector.
2. `customer`: Connects multi-account owner entities.
3. `device`: Shared hardware fingerprints across accounts.
4. `ip_address`: Network origin IP node.
5. `merchant`: Commercial payment recipient.
6. `crypto_wallet`: Cryptocurrency OTC desk/mixer wallet.
7. `atm`: Cash withdrawal node.
8. `bank`: Federated bank silo (BANK_A → BANK_J).
9. `merchant_sector`: Industry category (Crypto, Luxury, Real Estate, etc.).
10. `typology`: Linked ground-truth fraud pattern.
11. `ring`: Coordinated fraud ring membership.
12. `case`: SAR investigation case assignment.
13. `country`: Jurisdiction location.
14. `channel`: Transaction channel (Mobile, Web, Branch, ATM).

### 30-Dimension Account Feature Vector (`src/features/engineer.py`):
- **Dims 0–7 (Volume & Flow):** Outgoing/incoming transaction counts, total sums, averages, and max amounts.
- **Dims 8–9 (Counterparties):** Count of unique senders & receivers.
- **Dims 10–13 (Fraud Ratios):** Fraudulent transactions sent/received and fraud ratios.
- **Dims 14–18 (Timing & Velocity):** Night-time ratio (11 PM–5 AM), weekend ratio, peak 24h velocity, average inter-transaction gap (hours), and gap standard deviation.
- **Dims 19–22 (Channel Distribution):** Cross-bank transfer ratio, ATM cash ratio, P2P transfer ratio, and payment ratio.
- **Dims 23–25 (Amount Statistics):** Amount standard deviation, skewness, and 75th percentile.
- **Dims 26–29 (Behavioral Risk):** Unique devices count, unique IPs count, round dollar amount ratio ($5,000 vs. $4,982.13), and near-threshold structuring score ($8,500–$9,900 range).

---

## 🕵️ Understanding the 12 Injected Fraud Typologies

1. **Smurfing / Structuring:** Multiple accounts sending $8,500–$9,900 to 1 aggregation account.
2. **Layering Chains:** 5+ hop transfer sequences ($A \to B \to C \to D \to E$) in short time windows.
3. **Money Mule Networks:** Hub-and-spoke distribution to mule accounts.
4. **Account Takeover (ATO):** Sudden burst activity following new device/IP login.
5. **Shell Company Fraud:** Fictitious corporate invoicing.
6. **Crypto Laundering:** Routing fiat through OTC exchange wallets and crypto mixers.
7. **Trade-Based ML:** Trade invoice misvaluation.
8. **Rapid Round-Tripping:** $A \to B \to A$ cycles completed in < 48 hours.
9. **ATM Structuring:** Cash withdrawals split across physical ATMs below daily reporting limits.
10. **Payroll Fraud:** Fake employee payroll disbursements.
11. **Real Estate ML:** Property purchases using shell companies.
12. **Cyber-Enabled Fraud:** BEC and phishing payment redirections.

---

## 📊 Benchmark Metrics Summary

Evaluated on stratified `70% Train / 15% Val / 15% Test` split masks:

| Model | Model Type | ROC-AUC | PR-AUC | F1-Score | Training Time |
|---|---|---|---|---|---|
| **Random Forest** | Traditional ML (Tabular) | **1.0000** | **1.0000** | **1.0000** | 43.8s |
| **GraphSAGE GNN** | Graph Neural Network | **1.0000** | **1.0000** | **1.0000** | 188.6s |

---

## ❓ Troubleshooting & FAQ

### Q1: `ModuleNotFoundError` when running scripts
**Solution:** Ensure you are in the root directory (`C:\project`) and run `pip install -r requirements.txt`.

### Q2: Port 8501 is already in use when launching Streamlit
**Solution:** Streamlit is already running. Either open `http://localhost:8501` in your browser or run:
```bash
streamlit run app.py --server.port 8502
```

### Q3: How to run on GPU?
**Solution:** PyTorch Geometric will automatically detect CUDA GPU if PyTorch with CUDA support is installed (`torch.cuda.is_available() == True`).

---

*Project created & maintained for university AML research.*
