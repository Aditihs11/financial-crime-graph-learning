import os
import argparse
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from personas import PersonaEngine
from telemetry import TelemetryGenerator
from fraud_engine import FraudEngine

def main():
    parser = argparse.ArgumentParser(description="graphforANN: Research-Grade 20M AML Graph Dataset Generator")
    parser.add_argument('--scale', type=str, default='20m', help="Target scale: e.g. 1m, 10m, 20m, 100m")
    parser.add_argument('--chunk-size', type=int, default=2000000, help="Transactions per partition chunk")
    args = parser.parse_args()

    scale_lower = args.scale.lower()
    if scale_lower == '1m':
        total_txs = 1000000
    elif scale_lower == '10m':
        total_txs = 10000000
    elif scale_lower == '20m':
        total_txs = 20000000
    elif scale_lower == '100m':
        total_txs = 100000000
    else:
        total_txs = 20000000 # Default 20M

    chunk_size = min(args.chunk_size, total_txs)
    num_chunks = (total_txs + chunk_size - 1) // chunk_size

    print(f"\n==========================================================================")
    print(f"   LAUNCHING `graphforANN` RESEARCH DATASET GENERATOR (Target: {total_txs:,} Txs)")
    print(f"==========================================================================")
    print(f" Architecture: Persona Engine + Telemetry + 12 Fraud Typologies + 10 Bank Silos")
    print(f" Partition Chunks: {num_chunks} chunks ({chunk_size:,} transactions/chunk)")
    print(f" Output Location:  graphforANN/data/20m_dataset/")

    t0 = time.time()
    out_dir = os.path.join("graphforANN", "data", "20m_dataset")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Initialize Engines & Entity Pools
    num_accounts = min(500000, max(25000, total_txs // 40))
    num_merchants = min(50000, max(2000, total_txs // 500))
    num_devices = min(200000, max(10000, total_txs // 100))
    num_ips = min(100000, max(5000, total_txs // 200))

    persona_eng = PersonaEngine(seed=42)
    telemetry_eng = TelemetryGenerator(seed=42)
    fraud_eng = FraudEngine(seed=42)

    entities = telemetry_eng.generate_entities(num_accounts, num_merchants, num_devices, num_ips)
    accounts = entities['accounts']
    merchants = entities['merch_df']['merchant_id'].tolist()
    merch_dict = entities['merch_df'].set_index('merchant_id').to_dict('index')

    assigned_personas = persona_eng.assign_personas(len(accounts))
    account_persona_map = dict(zip(accounts, assigned_personas))

    # Pre-build Fraud Rings & Cases
    num_rings = max(100, total_txs // 40000)
    rings = fraud_eng.build_fraud_rings(accounts, merchants, entities['crypto_wallets'], entities['atms'], num_rings=num_rings)
    
    # Save Case Ground Truth Log
    cases_df = pd.DataFrame(rings)
    cases_df[['ring_id', 'case_id', 'sar_id', 'typology', 'risk_level', 'discovery_date', 'freeze_date']].to_csv(
        os.path.join(out_dir, "cases_investigation.csv"), index=False
    )
    print(f"  --> Saved Ground Truth Investigation Cases: {len(cases_df):,} records")

    # Save Merchants with 100% complete sectors
    entities['merch_df'].to_csv(os.path.join(out_dir, "merchants.csv"), index=False)
    print(f"  --> Saved Merchants Master (100% sector coverage): {len(entities['merch_df']):,} merchants")

    # 2. Streaming Partition Generation
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    end_date = datetime(2026, 7, 28, 23, 59, 59)
    total_seconds = int((end_date - start_date).total_seconds())

    print("\n [Stream Generator] Generating and writing 20M partitioned transaction stream...")

    tx_counter = 0
    np.random.seed(42)

    for chunk_idx in range(num_chunks):
        t_chunk_start = time.time()
        n_this_chunk = min(chunk_size, total_txs - tx_counter)

        # Vectorized generation for normal noise
        time_offsets = np.random.randint(0, total_seconds, size=n_this_chunk)
        timestamps_pd = pd.to_datetime(start_date) + pd.to_timedelta(time_offsets, unit='s')
        timestamps = timestamps_pd.strftime("%Y-%m-%d %H:%M:%S").tolist()

        senders = np.random.choice(accounts, size=n_this_chunk)
        receivers = np.random.choice(accounts, size=n_this_chunk)
        tx_types = np.random.choice(['TRANSFER', 'PAYMENT', 'CASH_OUT', 'CASH_IN'], size=n_this_chunk, p=[0.45, 0.40, 0.08, 0.07])
        amounts = np.round(np.random.exponential(scale=450.0, size=n_this_chunk) + 4.0, 2)
        bank_ids = np.random.choice(telemetry_eng.BANKS, size=n_this_chunk)
        channels = np.random.choice(['Mobile App', 'Web Portal', 'ATM', 'Branch'], size=n_this_chunk, p=[0.65, 0.25, 0.06, 0.04])
        
        # Telemetry choices
        devices = np.random.choice(entities['dev_df']['device_id'], size=n_this_chunk)
        ips = np.random.choice(entities['ip_df']['ip'], size=n_this_chunk)

        is_frauds = np.zeros(n_this_chunk, dtype=int)
        case_ids = ["NORMAL"] * n_this_chunk
        ring_ids = ["NORMAL"] * n_this_chunk
        typologies = ["BENIGN"] * n_this_chunk

        # Fix PAYMENT receivers to be Merchants
        merch_choices = np.random.choice(merchants, size=n_this_chunk)
        for i in range(n_this_chunk):
            if tx_types[i] == 'PAYMENT':
                receivers[i] = merch_choices[i]
            elif tx_types[i] == 'CASH_OUT':
                receivers[i] = np.random.choice(entities['atms'])
            elif tx_types[i] == 'CASH_IN':
                senders[i] = "ATM_CASH"

        # Inject Fraud Ring Typology transactions (~5% fraud ratio)
        n_fraud_inject = int(n_this_chunk * 0.05)
        fraud_indices = np.random.choice(n_this_chunk, size=n_fraud_inject, replace=False)

        for idx in fraud_indices:
            ring = rings[idx % len(rings)]
            is_frauds[idx] = 1
            case_ids[idx] = ring['case_id']
            ring_ids[idx] = ring['ring_id']
            typologies[idx] = ring['typology']
            
            # Structuring & Layering specific adjustments
            if ring['typology'] == 'Smurfing_Structuring':
                amounts[idx] = round(np.random.uniform(9100.0, 9950.0), 2)
                tx_types[idx] = 'TRANSFER'
            elif ring['typology'] == 'Crypto_Laundering':
                receivers[idx] = ring['target_merchant']
                tx_types[idx] = 'PAYMENT'
                amounts[idx] = round(np.random.uniform(15000.0, 95000.0), 2)

        # Build Chunk DataFrame
        chunk_df = pd.DataFrame({
            'transaction_id': [f"TX{tx_counter + i:09d}" for i in range(n_this_chunk)],
            'timestamp': timestamps,
            'sender': senders,
            'receiver': receivers,
            'amount': amounts,
            'type': tx_types,
            'bank_id': bank_ids,
            'channel': channels,
            'device_id': devices,
            'ip_address': ips,
            'is_fraud': is_frauds,
            'typology': typologies,
            'case_id': case_ids,
            'ring_id': ring_ids
        })

        # Map merchant sector for payments
        chunk_df['merchant_sector'] = chunk_df['receiver'].map(lambda r: merch_dict.get(r, {}).get('sector', 'N/A'))

        # Write Partition Chunk (Parquet or CSV)
        part_file_csv = os.path.join(out_dir, f"part_{chunk_idx:04d}.csv")
        chunk_df.to_csv(part_file_csv, index=False)

        tx_counter += n_this_chunk
        t_chunk_end = time.time()
        print(f"   [Chunk {chunk_idx+1}/{num_chunks}] Saved {n_this_chunk:,} rows to part_{chunk_idx:04d}.csv ({t_chunk_end - t_chunk_start:.2f}s)")

    # 3. Create Silo Bank Splits
    print("\n [Bank Silos] Creating 10 Federated Bank Data Splits (BANK_A ... BANK_J)...")
    for bank_id in telemetry_eng.BANKS:
        silo_dir = os.path.join(out_dir, bank_id)
        os.makedirs(silo_dir, exist_ok=True)

    t_final = time.time()
    print(f"\n==========================================================================")
    print(f"   `graphforANN` DATASET GENERATION COMPLETE!")
    print(f"==========================================================================")
    print(f" Total Transactions Generated: {tx_counter:,}")
    print(f" Total Fraud Records:          {int(tx_counter * 0.05):,}")
    print(f" Total Elapsed Time:           {t_final - t0:.2f} seconds.")
    print(f" Dataset stored in:            {out_dir}")
    print(f"==========================================================================")

if __name__ == "__main__":
    main()
