import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class FraudEngine:
    """
    Injects 12 realistic Evolving Financial Crime Typologies with graph topology rings
    and complete Investigation Ground Truth Metadata (Case IDs, SAR IDs, Ring IDs, Discovery/Freeze timestamps).
    """
    
    TYPOLOGIES = [
        'Smurfing_Structuring',
        'MultiStep_Layering',
        'Circular_RoundTripping',
        'SharedDevice_MuleNetwork',
        'Account_Takeover_ATO',
        'BustOut_Fraud',
        'Synthetic_Identity',
        'Card_Testing',
        'Merchant_Collusion',
        'Crypto_Laundering',
        'ATM_CashOut_Ring',
        'Insider_Fraud'
    ]

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.cases = []
        self.rings = []

    def build_fraud_rings(self, accounts, merchants, crypto_wallets, atms, num_rings=500):
        """Construct explicit topological fraud rings with ground truth metadata."""
        print(f" [Fraud Engine] Pre-building {num_rings} topological Fraud Rings across 12 Typologies...")
        
        rings = []
        case_counter = 1001
        sar_counter = 5001
        
        for ring_id in range(num_rings):
            typology = self.rng.choice(self.TYPOLOGIES)
            ring_size = self.rng.randint(4, 15)
            ring_members = self.rng.choice(accounts, size=ring_size, replace=False).tolist()
            
            case_id = f"CASE-{case_counter}"
            sar_id = f"SAR-2026-{sar_counter}"
            case_counter += 1
            sar_counter += 1
            
            discovery_date = datetime(2025, 1, 1) + timedelta(days=self.rng.randint(0, 500))
            freeze_date = discovery_date + timedelta(days=self.rng.randint(1, 14))
            
            target_merchant = self.rng.choice(merchants)
            target_crypto = self.rng.choice(crypto_wallets) if crypto_wallets else "0x_CRYPTO_MOCK"
            target_atm = self.rng.choice(atms)
            
            rings.append({
                'ring_id': f"RING_{ring_id:04d}",
                'case_id': case_id,
                'sar_id': sar_id,
                'typology': typology,
                'members': ring_members,
                'target_merchant': target_merchant,
                'target_crypto': target_crypto,
                'target_atm': target_atm,
                'discovery_date': discovery_date,
                'freeze_date': freeze_date,
                'risk_level': self.rng.choice(['CRITICAL', 'HIGH', 'ELEVATED'], p=[0.5, 0.35, 0.15])
            })
            
        self.rings = rings
        return rings

    def inject_typology_transactions(self, ring, start_base_time):
        """Generates topological fraudulent transaction sequences for a ring."""
        typology = ring['typology']
        members = ring['members']
        txs = []
        
        # Base time for ring activity
        t_curr = start_base_time + timedelta(hours=self.rng.randint(0, 48))
        
        if typology == 'Smurfing_Structuring':
            # Fan-out under $10,000 threshold ($9,100-$9,950)
            master = members[0]
            mules = members[1:]
            for m in mules:
                t_curr += timedelta(minutes=self.rng.randint(2, 15))
                amount = round(self.rng.uniform(9100.0, 9950.0), 2)
                txs.append({
                    'timestamp': t_curr,
                    'sender': master,
                    'receiver': m,
                    'amount': amount,
                    'type': 'TRANSFER',
                    'is_fraud': 1,
                    'typology': typology,
                    'case_id': ring['case_id'],
                    'ring_id': ring['ring_id']
                })
                
        elif typology == 'MultiStep_Layering':
            # Sequential chain A -> B -> C -> D -> Target
            for i in range(len(members) - 1):
                t_curr += timedelta(minutes=self.rng.randint(5, 30))
                amount = round(self.rng.uniform(15000.0, 85000.0), 2)
                txs.append({
                    'timestamp': t_curr,
                    'sender': members[i],
                    'receiver': members[i+1],
                    'amount': amount,
                    'type': 'TRANSFER',
                    'is_fraud': 1,
                    'typology': typology,
                    'case_id': ring['case_id'],
                    'ring_id': ring['ring_id']
                })
                
        elif typology == 'Circular_RoundTripping':
            # Closed cycle A -> B -> C -> ... -> A
            for i in range(len(members)):
                t_curr += timedelta(minutes=self.rng.randint(10, 45))
                next_m = members[(i+1) % len(members)]
                amount = round(self.rng.uniform(25000.0, 120000.0), 2)
                txs.append({
                    'timestamp': t_curr,
                    'sender': members[i],
                    'receiver': next_m,
                    'amount': amount,
                    'type': 'TRANSFER',
                    'is_fraud': 1,
                    'typology': typology,
                    'case_id': ring['case_id'],
                    'ring_id': ring['ring_id']
                })
                
        elif typology == 'Crypto_Laundering':
            # Mule -> Crypto Exchange
            for m in members:
                t_curr += timedelta(minutes=self.rng.randint(3, 20))
                amount = round(self.rng.uniform(10000.0, 95000.0), 2)
                txs.append({
                    'timestamp': t_curr,
                    'sender': m,
                    'receiver': ring['target_merchant'], # Crypto Exchange Merchant
                    'amount': amount,
                    'type': 'PAYMENT',
                    'is_fraud': 1,
                    'typology': typology,
                    'case_id': ring['case_id'],
                    'ring_id': ring['ring_id']
                })
                
        else: # Default fallback for ATO, BustOut, CashOut, etc.
            for m in members:
                t_curr += timedelta(minutes=self.rng.randint(1, 60))
                amount = round(self.rng.uniform(5000.0, 50000.0), 2)
                txs.append({
                    'timestamp': t_curr,
                    'sender': m,
                    'receiver': ring['target_atm'] if 'ATM' in typology else members[0],
                    'amount': amount,
                    'type': 'CASH_OUT' if 'ATM' in typology else 'TRANSFER',
                    'is_fraud': 1,
                    'typology': typology,
                    'case_id': ring['case_id'],
                    'ring_id': ring['ring_id']
                })
                
        return txs
