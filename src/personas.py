import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class PersonaEngine:
    """
    Simulates realistic human & organizational financial behaviour.
    Generates event timelines with salary cycles, bill payments, weekend/night spending,
    and distinct volume/amount distributions for 8 customer personas.
    """
    
    PERSONAS = {
        'Student': {
            'age_range': (18, 25),
            'monthly_income': (300, 1500),
            'tx_frequency_per_month': (30, 80),
            'preferred_channels': ['Mobile App'],
            'sectors': ['Food', 'Retail', 'Gaming', 'Education'],
            'night_prob': 0.25,
            'weekend_prob': 0.40,
            'avg_amount_range': (5, 80)
        },
        'Salaried_Employee': {
            'age_range': (22, 60),
            'monthly_income': (3500, 15000),
            'tx_frequency_per_month': (25, 60),
            'preferred_channels': ['Mobile App', 'Web Portal'],
            'sectors': ['Retail', 'Food', 'Healthcare', 'Utilities', 'Fuel', 'Entertainment'],
            'night_prob': 0.10,
            'weekend_prob': 0.35,
            'avg_amount_range': (20, 500)
        },
        'Business_Owner': {
            'age_range': (28, 65),
            'monthly_income': (15000, 150000),
            'tx_frequency_per_month': (60, 200),
            'preferred_channels': ['Web Portal', 'Branch'],
            'sectors': ['Wholesale', 'Services', 'Retail', 'Logistics', 'Tax'],
            'night_prob': 0.05,
            'weekend_prob': 0.15,
            'avg_amount_range': (200, 15000)
        },
        'Retiree': {
            'age_range': (60, 85),
            'monthly_income': (1200, 5000),
            'tx_frequency_per_month': (10, 30),
            'preferred_channels': ['ATM', 'Branch', 'Web Portal'],
            'sectors': ['Healthcare', 'Retail', 'Food', 'Utilities'],
            'night_prob': 0.02,
            'weekend_prob': 0.15,
            'avg_amount_range': (15, 300)
        },
        'Freelancer': {
            'age_range': (20, 45),
            'monthly_income': (2000, 10000),
            'tx_frequency_per_month': (20, 70),
            'preferred_channels': ['Mobile App', 'Web Portal'],
            'sectors': ['Software', 'Services', 'Retail', 'Food', 'Coworking'],
            'night_prob': 0.20,
            'weekend_prob': 0.30,
            'avg_amount_range': (30, 800)
        },
        'HNWI': {
            'age_range': (35, 75),
            'monthly_income': (50000, 500000),
            'tx_frequency_per_month': (20, 90),
            'preferred_channels': ['Web Portal', 'Branch'],
            'sectors': ['Luxury Retail', 'Real Estate', 'Investments', 'Travel'],
            'night_prob': 0.12,
            'weekend_prob': 0.25,
            'avg_amount_range': (1000, 50000)
        },
        'Money_Mule': {
            'age_range': (18, 40),
            'monthly_income': (500, 3000),
            'tx_frequency_per_month': (5, 20),
            'preferred_channels': ['Mobile App', 'ATM'],
            'sectors': ['Retail', 'Food'],
            'night_prob': 0.30,
            'weekend_prob': 0.30,
            'avg_amount_range': (10, 200)
        },
        'Shell_Company': {
            'age_range': (0, 5), # Account age
            'monthly_income': (100000, 2000000),
            'tx_frequency_per_month': (50, 300),
            'preferred_channels': ['Web Portal'],
            'sectors': ['Consulting', 'Import_Export', 'Financial_Services'],
            'night_prob': 0.10,
            'weekend_prob': 0.05,
            'avg_amount_range': (5000, 100000)
        }
    }

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def assign_personas(self, num_accounts):
        """Assign customer personas with realistic population proportions."""
        persona_names = list(self.PERSONAS.keys())
        # Realistic economic weights
        weights = [0.22, 0.45, 0.10, 0.12, 0.06, 0.02, 0.02, 0.01]
        assigned = self.rng.choice(persona_names, size=num_accounts, p=weights)
        return assigned

    def generate_baseline_events(self, account_id, persona_name, start_date, end_date):
        """Generate realistic temporal baseline event timeline for a given persona."""
        cfg = self.PERSONAS[persona_name]
        days = (end_date - start_date).days
        months = max(1, days // 30)
        
        events = []
        
        # 1. Monthly Recurring Salary / Income Deposit (around 1st of month)
        for m in range(months):
            salary_date = start_date + timedelta(days=m * 30 + self.rng.randint(0, 3))
            amount = round(self.rng.uniform(*cfg['monthly_income']), 2)
            events.append({
                'timestamp': salary_date.replace(hour=9, minute=self.rng.randint(0, 59)),
                'event_type': 'INCOME',
                'amount': amount,
                'channel': 'Direct Deposit',
                'category': 'Salary/Income'
            })
            
            # 2. Monthly Utility & Rent Bills (around 5th-10th of month)
            bill_date = start_date + timedelta(days=m * 30 + self.rng.randint(4, 10))
            bill_amount = round(amount * self.rng.uniform(0.20, 0.40), 2)
            events.append({
                'timestamp': bill_date.replace(hour=14, minute=self.rng.randint(0, 59)),
                'event_type': 'BILL_PAYMENT',
                'amount': bill_amount,
                'channel': cfg['preferred_channels'][0],
                'category': 'Rent & Utilities'
            })

        # 3. Regular daily/weekly discretionary spending
        num_spending_txs = self.rng.randint(cfg['tx_frequency_per_month'][0] * months, 
                                            cfg['tx_frequency_per_month'][1] * months)
        
        for _ in range(num_spending_txs):
            random_days = self.rng.randint(0, days)
            tx_time = start_date + timedelta(days=int(random_days))
            
            # Night or day spending offset
            if self.rng.rand() < cfg['night_prob']:
                hour = self.rng.choice([22, 23, 0, 1, 2, 3, 4])
            else:
                hour = self.rng.randint(8, 21)
                
            tx_time = tx_time.replace(hour=hour, minute=self.rng.randint(0, 59), second=self.rng.randint(0, 59))
            amount = round(self.rng.uniform(*cfg['avg_amount_range']), 2)
            sector = self.rng.choice(cfg['sectors'])
            channel = self.rng.choice(cfg['preferred_channels'])
            
            events.append({
                'timestamp': tx_time,
                'event_type': 'PAYMENT',
                'amount': amount,
                'channel': channel,
                'category': sector
            })
            
        return events
