import os
import argparse
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def run_streaming_generator(total_txs=20000000, chunk_size=2000000, out_dir="graphforANN/data/raw"):
    os.makedirs(out_dir, exist_ok=True)
    num_chunks = (total_txs + chunk_size - 1) // chunk_size

    print(f"\n==========================================================================")
    print(f" [Step 6] Streaming 20 Million Transactions in {num_chunks} Partition Chunks")
    print(f" Output Location: {out_dir}")
    print(f"==========================================================================")

    # Load master entities
    acc_df = pd.read_csv(os.path.join(out_dir, "accounts.csv"))
    merch_df = pd.read_csv(os.path.join(out_dir, "merchants.csv"))
    dev_df = pd.read_csv(os.path.join(out_dir, "devices.csv"))
    ip_df = pd.read_csv(os.path.join(out_dir, "ip_addresses.csv"))

    accounts = acc_df['account_id'].values
    merchants = merch_df['merchant_id'].values
    merch_sectors = merch_df.set_index('merchant_id')['sector'].to_dict()
    devices = dev_df['device_id'].values
    ips = ip_df['ip_address'].values
    banks = acc_df.set_index('account_id')['bank_id'].to_dict()

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 7, 28)
    total_seconds = int((end_date - start_date).total_seconds())

    rng = np.random.RandomState(42)
    tx_counter = 0

    for c in range(num_chunks):
        t0 = time.time()
        n_this_chunk = min(chunk_size, total_txs - tx_counter)

        time_offsets = rng.randint(0, total_seconds, size=n_this_chunk)
        timestamps_pd = pd.to_datetime(start_date) + pd.to_timedelta(time_offsets, unit='s')
        timestamps = timestamps_pd.strftime("%Y-%m-%d %H:%M:%S").tolist()

        senders = rng.choice(accounts, size=n_this_chunk)
        receivers = rng.choice(accounts, size=n_this_chunk)
        tx_types = rng.choice(['TRANSFER', 'PAYMENT', 'CASH_OUT', 'CASH_IN'], size=n_this_chunk, p=[0.45, 0.40, 0.08, 0.07])
        amounts = np.round(rng.exponential(scale=450.0, size=n_this_chunk) + 4.0, 2)

        merch_picks = rng.choice(merchants, size=n_this_chunk)
        dev_picks = rng.choice(devices, size=n_this_chunk)
        ip_picks = rng.choice(ips, size=n_this_chunk)

        is_frauds = (rng.rand(n_this_chunk) < 0.05).astype(int)
        case_ids = ["NORMAL"] * n_this_chunk
        ring_ids = ["NORMAL"] * n_this_chunk

        for i in range(n_this_chunk):
            if tx_types[i] == 'PAYMENT':
                receivers[i] = merch_picks[i]
            elif tx_types[i] == 'CASH_OUT':
                receivers[i] = "ATM_CASH"
            elif tx_types[i] == 'CASH_IN':
                senders[i] = "ATM_CASH"

            if is_frauds[i] == 1:
                case_ids[i] = f"CASE-{rng.randint(1000, 1500)}"
                ring_ids[i] = f"RING_{rng.randint(0, 500):04d}"

        chunk_df = pd.DataFrame({
            'transaction_id': [f"TX_{tx_counter + i:09d}" for i in range(n_this_chunk)],
            'timestamp': timestamps,
            'sender': senders,
            'receiver': receivers,
            'amount': amounts,
            'type': tx_types,
            'bank_id': [banks.get(s, 'BANK_A') for s in senders],
            'device_id': dev_picks,
            'ip_address': ip_picks,
            'is_fraud': is_frauds,
            'case_id': case_ids,
            'ring_id': ring_ids
        })
        chunk_df['merchant_sector'] = chunk_df['receiver'].map(lambda r: merch_sectors.get(r, 'N/A'))

        part_file = os.path.join(out_dir, f"part_{c:04d}.csv")
        chunk_df.to_csv(part_file, index=False)
        tx_counter += n_this_chunk
        t1 = time.time()
        print(f"  [Chunk {c+1}/{num_chunks}] Saved {n_this_chunk:,} rows to part_{c:04d}.csv ({t1 - t0:.2f}s)")

    print(f"\n [Step 6 Complete] Successfully streamed {tx_counter:,} transactions to {out_dir}!")

if __name__ == "__main__":
    run_streaming_generator(total_txs=1000000, chunk_size=1000000)
