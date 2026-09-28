import os
import pandas as pd
import numpy as np

class MasterTableGenerator:
    """
    Generates clean Master Entity Tables with unique primary keys and valid foreign key relationships:
    - customers.csv
    - accounts.csv
    - merchants.csv
    - devices.csv
    - ip_addresses.csv
    - banks.csv
    - companies.csv
    - crypto_wallets.csv
    - countries.csv
    """

    def __init__(self, raw_dir="graphforANN/data/raw", seed=42):
        self.raw_dir = raw_dir
        self.rng = np.random.RandomState(seed)
        os.makedirs(self.raw_dir, exist_ok=True)

    def generate_all_master_tables(self, num_accounts=500000, num_merchants=50000, num_devices=200000, num_ips=100000):
        print("\n [Step 2] Generating Normalized Master Entity Tables...")

        # 1. Countries Master Table
        countries = ['USA', 'India', 'UK', 'Singapore', 'Cayman Islands', 'Panama', 'Switzerland', 'Germany', 'Japan', 'UAE']
        high_risk_flags = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
        countries_df = pd.DataFrame({
            'country_id': [f"CTRY_{i+1:02d}" for i in range(len(countries))],
            'country_name': countries,
            'is_high_risk': high_risk_flags
        })
        countries_df.to_csv(os.path.join(self.raw_dir, "countries.csv"), index=False)
        print(f"  -> Generated countries.csv ({len(countries_df)} rows)")

        # 2. Banks Master Table (10 Banks)
        banks = [f"BANK_{chr(65+i)}" for i in range(10)]
        banks_df = pd.DataFrame({
            'bank_id': banks,
            'bank_name': [f"Global Bank {chr(65+i)}" for i in range(10)],
            'swift_code': [f"GBANK{chr(65+i)}XX" for i in range(10)],
            'kyc_strictness': self.rng.choice(['HIGH', 'MEDIUM', 'LOW'], size=10, p=[0.6, 0.3, 0.1])
        })
        banks_df.to_csv(os.path.join(self.raw_dir, "banks.csv"), index=False)
        print(f"  -> Generated banks.csv ({len(banks_df)} rows)")

        # 3. Customers Master Table
        cust_ids = [f"CUST_{i:07d}" for i in range(num_accounts)]
        cust_df = pd.DataFrame({
            'customer_id': cust_ids,
            'age': self.rng.randint(18, 85, size=num_accounts),
            'risk_rating': np.round(self.rng.uniform(0.05, 0.40, size=num_accounts), 2),
            'sar_history_count': self.rng.choice([0, 1, 2], size=num_accounts, p=[0.98, 0.015, 0.005])
        })
        cust_df.to_csv(os.path.join(self.raw_dir, "customers.csv"), index=False)
        print(f"  -> Generated customers.csv ({len(cust_df):,} rows)")

        # 4. Accounts Master Table
        accounts_df = pd.DataFrame({
            'account_id': [f"A{i:07d}" for i in range(num_accounts)],
            'customer_id': cust_ids,
            'bank_id': self.rng.choice(banks, size=num_accounts),
            'account_type': self.rng.choice(['SAVINGS', 'CHECKING', 'CORPORATE'], size=num_accounts, p=[0.6, 0.35, 0.05]),
            'account_age_months': self.rng.randint(1, 120, size=num_accounts)
        })
        accounts_df.to_csv(os.path.join(self.raw_dir, "accounts.csv"), index=False)
        print(f"  -> Generated accounts.csv ({len(accounts_df):,} rows)")

        # 5. Merchants Master Table (100% complete sector coverage)
        sectors = [
            'Retail', 'Grocery', 'Fuel', 'Food_Dining', 'Healthcare',
            'Insurance', 'Travel_Airline', 'Utilities', 'Government_Tax',
            'Electronics', 'ECommerce', 'Casino_Gambling', 'Crypto_Exchange',
            'Shell_Company', 'Luxury_Goods', 'Gaming'
        ]
        merchants_df = pd.DataFrame({
            'merchant_id': [f"M{i:05d}" for i in range(num_merchants)],
            'merchant_name': [f"Merchant Global Corp {i:05d}" for i in range(num_merchants)],
            'sector': self.rng.choice(sectors, size=num_merchants),
            'risk_rating': self.rng.randint(5, 95, size=num_merchants),
            'country': self.rng.choice(countries, size=num_merchants)
        })
        merchants_df.to_csv(os.path.join(self.raw_dir, "merchants.csv"), index=False)
        print(f"  -> Generated merchants.csv ({len(merchants_df):,} rows)")

        # 6. Devices Master Table
        devices_df = pd.DataFrame({
            'device_id': [f"D{i:06d}" for i in range(num_devices)],
            'os_type': self.rng.choice(['Android', 'iOS', 'Web_Windows', 'Web_MacOS'], size=num_devices),
            'is_jailbroken': (self.rng.rand(num_devices) < 0.04).astype(int),
            'is_vpn': (self.rng.rand(num_devices) < 0.08).astype(int)
        })
        devices_df.to_csv(os.path.join(self.raw_dir, "devices.csv"), index=False)
        print(f"  -> Generated devices.csv ({len(devices_df):,} rows)")

        # 7. IP Addresses Master Table
        ips_df = pd.DataFrame({
            'ip_address': [f"{self.rng.randint(1,255)}.{self.rng.randint(0,255)}.{self.rng.randint(0,255)}.{self.rng.randint(1,255)}" for _ in range(num_ips)],
            'isp_type': self.rng.choice(['Residential', 'Mobile_Carrier', 'Corporate', 'VPN_Proxy', 'Tor_Exit'], size=num_ips),
            'country': self.rng.choice(countries, size=num_ips)
        })
        ips_df.to_csv(os.path.join(self.raw_dir, "ip_addresses.csv"), index=False)
        print(f"  -> Generated ip_addresses.csv ({len(ips_df):,} rows)")

        # 8. Companies & Crypto Wallets Master Tables
        companies_df = pd.DataFrame({
            'company_id': [f"COMP_{i:05d}" for i in range(10000)],
            'company_name': [f"Shell Company Inc {i:05d}" for i in range(10000)],
            'registration_country': self.rng.choice(['Panama', 'Cayman Islands', 'USA', 'UK'], size=10000)
        })
        companies_df.to_csv(os.path.join(self.raw_dir, "companies.csv"), index=False)

        crypto_df = pd.DataFrame({
            'wallet_address': [f"0x{self.rng.randint(10000000, 99999999):08x}" for _ in range(20000)],
            'chain': self.rng.choice(['BTC', 'ETH', 'USDT'], size=20000),
            'risk_level': self.rng.choice(['HIGH', 'CRITICAL', 'NORMAL'], size=20000)
        })
        crypto_df.to_csv(os.path.join(self.raw_dir, "crypto_wallets.csv"), index=False)

        print(f" [Step 2 Complete] All 9 Master Tables generated successfully in {self.raw_dir}!\n")

if __name__ == "__main__":
    gen = MasterTableGenerator()
    gen.generate_all_master_tables(num_accounts=100000, num_merchants=10000, num_devices=50000, num_ips=25000)
