import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class BehaviourSimulator:
    """
    Simulates normal customer lifecycle behaviour (Salary deposits, utility bill payments,
    grocery shopping, fuel, ATM withdrawals, online spending, investments) over 2024-2026.
    """

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def generate_normal_transactions(self, accounts, merchants, atms, num_txs=1000000, start_date=None, end_date=None):
        if start_date is None:
            start_date = datetime(2024, 1, 1, 0, 0, 0)
        if end_date is None:
            end_date = datetime(2026, 7, 28, 23, 59, 59)

        total_seconds = int((end_date - start_date).total_seconds())

        print(f" [Behaviour Simulator] Simulating {num_txs:,} normal lifecycle transactions...")

        time_offsets = self.rng.randint(0, total_seconds, size=num_txs)
        timestamps_pd = pd.to_datetime(start_date) + pd.to_timedelta(time_offsets, unit='s')
        timestamps = timestamps_pd.strftime("%Y-%m-%d %H:%M:%S").tolist()

        senders = self.rng.choice(accounts, size=num_txs)
        receivers = self.rng.choice(accounts, size=num_txs)
        tx_types = self.rng.choice(['TRANSFER', 'PAYMENT', 'CASH_OUT', 'CASH_IN'], size=num_txs, p=[0.45, 0.40, 0.08, 0.07])
        amounts = np.round(self.rng.exponential(scale=450.0, size=num_txs) + 4.0, 2)
        channels = self.rng.choice(['Mobile App', 'Web Portal', 'ATM', 'Branch'], size=num_txs, p=[0.65, 0.25, 0.06, 0.04])

        merch_choices = self.rng.choice(merchants, size=num_txs)
        atm_choices = self.rng.choice(atms, size=num_txs)

        for i in range(num_txs):
            if tx_types[i] == 'PAYMENT':
                receivers[i] = merch_choices[i]
            elif tx_types[i] == 'CASH_OUT':
                receivers[i] = atm_choices[i]
            elif tx_types[i] == 'CASH_IN':
                senders[i] = "ATM_CASH"

        df = pd.DataFrame({
            'transaction_id': [f"TX_NORM_{i:08d}" for i in range(num_txs)],
            'timestamp': timestamps,
            'sender': senders,
            'receiver': receivers,
            'amount': amounts,
            'type': tx_types,
            'channel': channels,
            'is_fraud': 0,
            'typology': 'BENIGN',
            'case_id': 'NORMAL',
            'ring_id': 'NORMAL'
        })
        return df
