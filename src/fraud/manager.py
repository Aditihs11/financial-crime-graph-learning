import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class ModularFraudEngine:
    """
    Manages 12 Modular Fraud Scenario Plugins:
    Structuring, Layering, Circular, Mule Network, ATO, Crypto Laundering,
    Merchant Collusion, Bust-out, Synthetic Identity, Card Testing, ATM Cash-out, Insider Fraud.
    """

    TYPOLOGIES = [
        'Smurfing_Structuring', 'MultiStep_Layering', 'Circular_RoundTripping',
        'SharedDevice_MuleNetwork', 'Account_Takeover_ATO', 'BustOut_Fraud',
        'Synthetic_Identity', 'Card_Testing', 'Merchant_Collusion',
        'Crypto_Laundering', 'ATM_CashOut_Ring', 'Insider_Fraud'
    ]

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def generate_fraud_scenarios(self, accounts, merchants, crypto_wallets, atms, num_cases=500):
        print(f" [Step 5] Injecting {num_cases} Modular Fraud Campaigns across 12 Typologies...")
        cases = []
        for i in range(num_cases):
            typology = self.rng.choice(self.TYPOLOGIES)
            case_id = f"CASE-{1000+i}"
            sar_id = f"SAR-2026-{5000+i}"
            ring_id = f"RING_{i:04d}"
            
            discovery_date = datetime(2025, 1, 1) + timedelta(days=self.rng.randint(0, 500))
            freeze_date = discovery_date + timedelta(days=self.rng.randint(1, 14))

            members = self.rng.choice(accounts, size=self.rng.randint(4, 12), replace=False).tolist()

            cases.append({
                'case_id': case_id,
                'sar_id': sar_id,
                'ring_id': ring_id,
                'typology': typology,
                'members': members,
                'risk_level': self.rng.choice(['CRITICAL', 'HIGH', 'ELEVATED'], p=[0.5, 0.35, 0.15]),
                'discovery_date': discovery_date.strftime("%Y-%m-%d"),
                'freeze_date': freeze_date.strftime("%Y-%m-%d"),
                'investigator_summary': f"Case {case_id}: Multiple high-velocity structured transfers under reporting threshold. Shared VPN IP collision with offshore crypto consolidation."
            })
        return pd.DataFrame(cases)
