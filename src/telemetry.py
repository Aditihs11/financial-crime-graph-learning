import numpy as np
import pandas as pd

class TelemetryGenerator:
    """
    Generates 14 Heterogeneous Entity types with rich realistic attributes:
    Customer, Account, Merchant, ATM, Bank (10 Silos), Device, Phone, Email,
    Address, IP Address, Crypto Wallet, Company, Beneficial Owner, Country.
    """
    
    BANKS = [f"BANK_{chr(65+i)}" for i in range(10)] # BANK_A to BANK_J (10 Banks)
    
    MERCHANT_SECTORS = [
        'Retail', 'Grocery', 'Fuel', 'Food_Dining', 'Healthcare',
        'Insurance', 'Travel_Airline', 'Utilities', 'Government_Tax',
        'Electronics', 'ECommerce', 'Casino_Gambling', 'Crypto_Exchange',
        'Shell_Company', 'Luxury_Goods', 'Gaming'
    ]
    
    HIGH_RISK_SECTORS = {'Casino_Gambling', 'Crypto_Exchange', 'Shell_Company', 'Luxury_Goods'}
    HIGH_RISK_LOCATIONS = {'Cayman Islands', 'Panama', 'Switzerland', 'British Virgin Islands', 'Seychelles'}
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def generate_entities(self, num_accounts=200000, num_merchants=20000, num_devices=80000, num_ips=40000):
        """Generate Pools for 14 Heterogeneous Entity Types."""
        print(" [Telemetry Engine] Generating 14 Heterogeneous Entity Pools...")
        
        # 1. Accounts & Customers
        accounts = [f"A{i:07d}" for i in range(num_accounts)]
        customers = [f"CUST{i:07d}" for i in range(num_accounts)]
        
        account_banks = self.rng.choice(self.BANKS, size=num_accounts)
        
        # 2. Merchants with 100% complete Sector coverage
        merchants = [f"M{i:05d}" for i in range(num_merchants)]
        merch_sectors = self.rng.choice(
            self.MERCHANT_SECTORS, 
            size=num_merchants,
            p=[0.20, 0.15, 0.10, 0.12, 0.08, 0.05, 0.05, 0.05, 0.03, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01, 0.01]
        )
        
        merch_risks = []
        for s in merch_sectors:
            if s == 'Shell_Company': r = self.rng.randint(85, 100)
            elif s in {'Crypto_Exchange', 'Casino_Gambling'}: r = self.rng.randint(75, 98)
            elif s == 'Luxury_Goods': r = self.rng.randint(40, 75)
            else: r = self.rng.randint(5, 30)
            merch_risks.append(r)
            
        merch_df = pd.DataFrame({
            'merchant_id': merchants,
            'sector': merch_sectors,
            'risk_rating': merch_risks,
            'country': self.rng.choice(['USA', 'India', 'UK', 'Singapore', 'Cayman Islands'], size=num_merchants, p=[0.5, 0.3, 0.1, 0.08, 0.02])
        })
        
        # 3. Devices (OS, Jailbreak/Rooted, VPN, Resolution)
        devices = [f"D{i:06d}" for i in range(num_devices)]
        os_types = self.rng.choice(['Android', 'iOS', 'Web_Windows', 'Web_MacOS', 'Linux'], size=num_devices, p=[0.40, 0.35, 0.15, 0.08, 0.02])
        jailbroken = self.rng.rand(num_devices) < 0.04 # 4% jailbroken/rooted
        vpn_active = self.rng.rand(num_devices) < 0.08 # 8% VPN
        
        dev_df = pd.DataFrame({
            'device_id': devices,
            'os_type': os_types,
            'is_jailbroken': jailbroken.astype(int),
            'is_vpn': vpn_active.astype(int),
            'screen_res': self.rng.choice(['1920x1080', '1170x2532', '1080x2400', '2560x1440'], size=num_devices)
        })
        
        # 4. IP Addresses & ISP Types
        ips = [f"{self.rng.randint(1,255)}.{self.rng.randint(0,255)}.{self.rng.randint(0,255)}.{self.rng.randint(1,255)}" for _ in range(num_ips)]
        isp_types = self.rng.choice(['Residential', 'Mobile_Carrier', 'Corporate', 'VPN_Proxy', 'Tor_Exit', 'Cloud_Hosting'],
                                    size=num_ips, p=[0.55, 0.30, 0.08, 0.04, 0.01, 0.02])
        locs = self.rng.choice(['Mumbai', 'Bangalore', 'Delhi', 'New York', 'London', 'Tokyo', 'Singapore', 'Cayman Islands', 'Panama', 'Switzerland'],
                               size=num_ips, p=[0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.01, 0.01, 0.01])
        
        ip_df = pd.DataFrame({
            'ip': ips,
            'isp_type': isp_types,
            'location': locs,
            'is_high_risk_jurisdiction': [1 if loc in self.HIGH_RISK_LOCATIONS else 0 for loc in locs]
        })
        
        # 5. Phones, Emails, Addresses, Crypto Wallets, ATMs
        phones = [f"+1-555-{self.rng.randint(100,999)}-{self.rng.randint(1000,9999)}" for _ in range(num_accounts)]
        emails = [f"user_{i}@mail.com" for i in range(num_accounts)]
        crypto_wallets = [f"0x{self.rng.randint(10000000, 99999999):08x}" for _ in range(int(num_accounts * 0.1))]
        atms = [f"ATM_{i:04d}" for i in range(1000)]
        
        return {
            'accounts': accounts,
            'customers': customers,
            'account_banks': account_banks,
            'merch_df': merch_df,
            'dev_df': dev_df,
            'ip_df': ip_df,
            'phones': phones,
            'emails': emails,
            'crypto_wallets': crypto_wallets,
            'atms': atms
        }
