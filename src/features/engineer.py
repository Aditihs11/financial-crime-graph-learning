"""
Step 9: Feature Engineering Module
====================================
Generates rich behavioral, temporal, and risk-score features
from raw transaction partitions. Outputs feature matrix as
a normalized NumPy array ready for PyTorch Geometric HeteroData.

Features Engineered (30 dims per account node):
  [0]  out_tx_count       - number of outgoing transactions
  [1]  out_tx_total       - total outgoing amount
  [2]  out_tx_mean        - mean outgoing amount
  [3]  out_tx_max         - max outgoing amount
  [4]  in_tx_count        - number of incoming transactions
  [5]  in_tx_total        - total incoming amount
  [6]  in_tx_mean         - mean incoming amount
  [7]  in_tx_max          - max incoming amount
  [8]  unique_receivers    - unique counterparties (out)
  [9]  unique_senders      - unique counterparties (in)
  [10] fraud_tx_sent       - number of fraud transactions sent
  [11] fraud_tx_received   - number of fraud transactions received
  [12] fraud_rate_sent     - fraud rate on outgoing side
  [13] fraud_rate_received - fraud rate on incoming side
  [14] night_tx_ratio      - fraction of txns at night (22:00-06:00)
  [15] weekend_tx_ratio    - fraction of txns on weekends
  [16] velocity_24h        - peak transactions in any 24h window
  [17] avg_tx_gap_hours    - average hours between transactions
  [18] std_tx_gap_hours    - std dev of inter-transaction time
  [19] cross_bank_ratio    - fraction of txns crossing bank silos
  [20] atm_ratio           - fraction of ATM transactions
  [21] int_transfer_ratio  - fraction of TRANSFER type txns
  [22] payment_ratio       - fraction of PAYMENT type txns
  [23] amount_std          - std dev of transaction amounts
  [24] amount_skew         - skewness of transaction amounts
  [25] amount_q75          - 75th percentile of amounts
  [26] unique_devices       - number of unique devices used
  [27] unique_ips           - number of unique IPs used
  [28] round_amount_ratio   - fraction of perfectly round amounts
  [29] structuring_score    - score based on just-below-threshold txns
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

THRESHOLD = 10_000.0  # AML structuring threshold
NIGHT_START, NIGHT_END = 22, 6   # 10 PM to 6 AM


def compute_account_features(tx_df: pd.DataFrame, accounts: np.ndarray) -> np.ndarray:
    """
    Compute the full 30-dimension feature matrix for all accounts.

    Parameters
    ----------
    tx_df    : Raw transaction DataFrame (all partitions or subset)
    accounts : Array of unique account IDs to build features for

    Returns
    -------
    features : np.ndarray of shape (n_accounts, 30), dtype float32
    """
    n = len(accounts)
    features = np.zeros((n, 30), dtype=np.float32)
    acc_to_idx = {acc: i for i, acc in enumerate(accounts)}

    print(f"   [FE] Computing features for {n:,} accounts over {len(tx_df):,} transactions...")

    # Parse timestamps once
    if not pd.api.types.is_datetime64_any_dtype(tx_df['timestamp']):
        tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'], errors='coerce')

    tx_df['hour']    = tx_df['timestamp'].dt.hour
    tx_df['weekday'] = tx_df['timestamp'].dt.dayofweek  # 0=Mon, 6=Sun
    tx_df['is_night']   = ((tx_df['hour'] >= NIGHT_START) | (tx_df['hour'] < NIGHT_END)).astype(int)
    tx_df['is_weekend']  = (tx_df['weekday'] >= 5).astype(int)
    tx_df['is_round']   = (tx_df['amount'] % 100 == 0).astype(int)
    tx_df['near_thresh'] = ((tx_df['amount'] >= THRESHOLD * 0.85) & (tx_df['amount'] < THRESHOLD)).astype(int)

    # --- Outgoing aggregates ---
    out = tx_df.groupby('sender').agg(
        out_count   = ('amount', 'count'),
        out_total   = ('amount', 'sum'),
        out_mean    = ('amount', 'mean'),
        out_max     = ('amount', 'max'),
        out_std     = ('amount', 'std'),
        uniq_recv   = ('receiver', 'nunique'),
        fraud_sent  = ('is_fraud', 'sum'),
        night_out   = ('is_night', 'sum'),
        wkend_out   = ('is_weekend', 'sum'),
        round_out   = ('is_round', 'sum'),
        struct_out  = ('near_thresh', 'sum'),
        atm_out     = ('type', lambda x: (x == 'ATM_WITHDRAW').sum()),
        xfer_out    = ('type', lambda x: (x == 'TRANSFER').sum()),
        pay_out     = ('type', lambda x: (x == 'PAYMENT').sum()),
        uniq_dev    = ('device_id', 'nunique'),
        uniq_ip     = ('ip_address', 'nunique'),
    ).reset_index()

    # --- Incoming aggregates ---
    inc = tx_df.groupby('receiver').agg(
        in_count    = ('amount', 'count'),
        in_total    = ('amount', 'sum'),
        in_mean     = ('amount', 'mean'),
        in_max      = ('amount', 'max'),
        uniq_send   = ('sender', 'nunique'),
        fraud_recv  = ('is_fraud', 'sum'),
    ).reset_index()

    # --- Cross-bank ratio ---
    if 'sender_bank' in tx_df.columns and 'receiver_bank' in tx_df.columns:
        tx_df['cross_bank'] = (tx_df['sender_bank'] != tx_df['receiver_bank']).astype(int)
    else:
        tx_df['cross_bank'] = 0

    cross = tx_df.groupby('sender')['cross_bank'].mean().reset_index()
    cross.columns = ['sender', 'cross_bank_ratio']

    # --- Velocity: max txns in any 24h window ---
    def max_velocity_24h(group):
        times = group['timestamp'].sort_values()
        if len(times) <= 1:
            return len(times)
        max_v = 0
        for t in times:
            window = ((times >= t) & (times < t + pd.Timedelta(hours=24))).sum()
            if window > max_v:
                max_v = window
        return max_v

    # For performance, compute only top-100 accounts by volume
    big_senders = out.nlargest(100, 'out_count')['sender'].tolist()
    vel_df = tx_df[tx_df['sender'].isin(big_senders)]
    vel_data = vel_df.groupby('sender').apply(max_velocity_24h).reset_index()
    vel_data.columns = ['sender', 'velocity_24h']

    # --- Average inter-transaction gap ---
    def gap_stats(times):
        times = times.sort_values()
        if len(times) < 2:
            return 0.0, 0.0
        gaps = np.diff(times.astype('int64').values / 1e9 / 3600)  # in hours
        return float(np.mean(gaps)), float(np.std(gaps))

    gap_result = tx_df.groupby('sender')['timestamp'].apply(gap_stats).reset_index()
    gap_result.columns = ['sender', 'gap_stats']
    gap_result['avg_gap'] = gap_result['gap_stats'].apply(lambda x: x[0])
    gap_result['std_gap'] = gap_result['gap_stats'].apply(lambda x: x[1])

    # --- Skew per account ---
    def safe_skew(amounts):
        if len(amounts) < 3:
            return 0.0
        return float(stats.skew(amounts))

    skew_df = tx_df.groupby('sender')['amount'].apply(safe_skew).reset_index()
    skew_df.columns = ['sender', 'amount_skew']

    q75_df = tx_df.groupby('sender')['amount'].quantile(0.75).reset_index()
    q75_df.columns = ['sender', 'q75']

    # Merge all into out
    out = out.merge(cross,    on='sender', how='left')
    out = out.merge(vel_data, on='sender', how='left')
    out = out.merge(gap_result[['sender','avg_gap','std_gap']], on='sender', how='left')
    out = out.merge(skew_df,  on='sender', how='left')
    out = out.merge(q75_df,   on='sender', how='left')
    out = out.fillna(0)

    inc = inc.fillna(0)
    out = out.set_index('sender')
    inc = inc.set_index('receiver')

    print(f"   [FE] Filling feature matrix...")
    for i, acc in enumerate(accounts):
        if acc in out.index:
            r = out.loc[acc]
            oc = r.get('out_count', 0)
            features[i, 0]  = r.get('out_count', 0)
            features[i, 1]  = r.get('out_total', 0)
            features[i, 2]  = r.get('out_mean', 0)
            features[i, 3]  = r.get('out_max', 0)
            features[i, 8]  = r.get('uniq_recv', 0)
            features[i, 10] = r.get('fraud_sent', 0)
            features[i, 12] = (r.get('fraud_sent', 0) / oc) if oc > 0 else 0
            features[i, 14] = (r.get('night_out', 0) / oc) if oc > 0 else 0
            features[i, 15] = (r.get('wkend_out', 0) / oc) if oc > 0 else 0
            features[i, 16] = r.get('velocity_24h', 0)
            features[i, 17] = r.get('avg_gap', 0)
            features[i, 18] = r.get('std_gap', 0)
            features[i, 19] = r.get('cross_bank_ratio', 0)
            features[i, 20] = (r.get('atm_out', 0) / oc) if oc > 0 else 0
            features[i, 21] = (r.get('xfer_out', 0) / oc) if oc > 0 else 0
            features[i, 22] = (r.get('pay_out', 0) / oc) if oc > 0 else 0
            features[i, 23] = r.get('out_std', 0)
            features[i, 24] = r.get('amount_skew', 0)
            features[i, 25] = r.get('q75', 0)
            features[i, 26] = r.get('uniq_dev', 0)
            features[i, 27] = r.get('uniq_ip', 0)
            features[i, 28] = (r.get('round_out', 0) / oc) if oc > 0 else 0
            features[i, 29] = (r.get('struct_out', 0) / oc) if oc > 0 else 0

        if acc in inc.index:
            ri = inc.loc[acc]
            ic = ri.get('in_count', 0)
            features[i, 4]  = ri.get('in_count', 0)
            features[i, 5]  = ri.get('in_total', 0)
            features[i, 6]  = ri.get('in_mean', 0)
            features[i, 7]  = ri.get('in_max', 0)
            features[i, 9]  = ri.get('uniq_send', 0)
            features[i, 11] = ri.get('fraud_recv', 0)
            features[i, 13] = (ri.get('fraud_recv', 0) / ic) if ic > 0 else 0

    # Normalize each dimension to [0, 1]
    print(f"   [FE] Normalizing features...")
    for col in range(30):
        col_max = np.max(np.abs(features[:, col]))
        if col_max > 0:
            features[:, col] /= col_max

    print(f"   [FE] Feature engineering complete. Shape: {features.shape}")
    return features


def run_feature_engineering(
    raw_dir: str = None,
    output_path: str = None,
    max_partitions: int = 2
):
    """
    Entry-point: reads transaction partitions, computes features,
    saves feature matrix + account index to disk.
    """
    base = Path(__file__).resolve().parents[2]
    raw_dir     = raw_dir     or str(base / "data" / "20m_dataset")
    output_path = output_path or str(base / "data" / "graph" / "features.npz")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("\n" + "=" * 70)
    print("   STEP 9: FEATURE ENGINEERING MODULE")
    print("=" * 70)

    part_files = sorted([f for f in os.listdir(raw_dir) if f.startswith("part_") and f.endswith(".csv")])
    part_files = part_files[:max_partitions]
    print(f"   Loading {len(part_files)} partition(s) from {raw_dir} ...")

    dfs = [pd.read_csv(os.path.join(raw_dir, f)) for f in part_files]
    tx_df = pd.concat(dfs, ignore_index=True)
    print(f"   Loaded {len(tx_df):,} transactions.")

    accounts = tx_df[tx_df['sender'].astype(str).str.startswith('A')]['sender'].unique()
    print(f"   Unique Account Senders: {len(accounts):,}")

    features = compute_account_features(tx_df, accounts)

    np.savez_compressed(output_path, features=features, accounts=accounts)
    print(f"\n   [SAVED] Feature matrix -> {output_path}")
    print("=" * 70)
    return features, accounts


if __name__ == "__main__":
    run_feature_engineering(max_partitions=2)
