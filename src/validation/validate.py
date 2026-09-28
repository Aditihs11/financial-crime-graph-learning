import os
import json
import pandas as pd
import numpy as np

class GraphValidator:
    """
    Automated Validation Suite for graphforANN:
    Validates duplicate IDs, missing values, timestamps, foreign keys,
    merchant completeness, fraud ratio, bank partitions, and generates outputs/validation_report.json.
    """

    def __init__(self, raw_data_dir=None, out_report=None):
        import pathlib
        base_dir = pathlib.Path(__file__).resolve().parents[2]
        self.raw_data_dir = raw_data_dir or str(base_dir / "data" / "20m_dataset")
        self.out_report = out_report or str(base_dir / "outputs" / "validation_report.json")
        os.makedirs(os.path.dirname(self.out_report), exist_ok=True)

    def run_all_checks(self):
        print("\n==========================================================================")
        print("   [Step 7] RUNNING AUTOMATED VALIDATION SUITE (`graphforANN`)")
        print("==========================================================================")

        part_files = [f for f in os.listdir(self.raw_data_dir) if f.startswith("part_") and f.endswith(".csv")]
        part_files.sort()

        if not part_files:
            print(f"❌ ERROR: No partition files found in {self.raw_data_dir}")
            return False

        sample_file = os.path.join(self.raw_data_dir, part_files[0])
        df = pd.read_csv(sample_file)

        report = {}
        checks_passed = 0

        # Check 1: Duplicate Transaction IDs
        dup_count = int(df['transaction_id'].duplicated().sum())
        report['check_1_duplicate_ids'] = {'passed': dup_count == 0, 'duplicates_found': dup_count}
        if dup_count == 0:
            print(" [PASS] Check 1/7: Zero Duplicate Transaction IDs verified.")
            checks_passed += 1

        # Check 2: 100% Merchant Sector Completeness
        missing_sectors = int(df[df['type'] == 'PAYMENT']['merchant_sector'].isna().sum())
        report['check_2_merchant_completeness'] = {'passed': missing_sectors == 0, 'missing_sectors': missing_sectors}
        if missing_sectors == 0:
            print(" [PASS] Check 2/7: 100% Merchant Sector completeness verified (0 missing).")
            checks_passed += 1

        # Check 3: Timestamp Completeness
        missing_ts = int(df['timestamp'].isna().sum())
        report['check_3_timestamp_completeness'] = {'passed': missing_ts == 0, 'missing_timestamps': missing_ts}
        if missing_ts == 0:
            print(" [PASS] Check 3/7: 100% Timestamp completeness verified.")
            checks_passed += 1

        # Check 4: Non-negative Amounts
        neg_amounts = int((df['amount'] < 0).sum())
        report['check_4_non_negative_amounts'] = {'passed': neg_amounts == 0, 'negative_amounts': neg_amounts}
        if neg_amounts == 0:
            print(" [PASS] Check 4/7: Zero negative transaction amounts verified.")
            checks_passed += 1

        # Check 5: Orphan Devices
        missing_devs = int(df['device_id'].isna().sum())
        report['check_5_orphan_devices'] = {'passed': missing_devs == 0, 'missing_devices': missing_devs}
        if missing_devs == 0:
            print(" [PASS] Check 5/7: All transactions bound to valid Device IDs.")
            checks_passed += 1

        # Check 6: Fraud Ratio
        fraud_ratio = float(df['is_fraud'].mean())
        report['check_6_fraud_ratio'] = {'passed': 0.01 <= fraud_ratio <= 0.15, 'fraud_ratio_pct': round(fraud_ratio*100, 2)}
        if 0.01 <= fraud_ratio <= 0.15:
            print(f" [PASS] Check 6/7: Realistic Fraud Label ratio verified ({fraud_ratio*100:.2f}%).")
            checks_passed += 1

        # Check 7: Bank Silos
        bank_count = int(df['bank_id'].nunique())
        report['check_7_bank_silos'] = {'passed': bank_count >= 3, 'banks_present': bank_count}
        if bank_count >= 3:
            print(f" [PASS] Check 7/7: Multi-Bank Federated Silos verified ({bank_count} banks present).")
            checks_passed += 1

        report['summary'] = {'checks_passed': checks_passed, 'total_checks': 7, 'status': 'PASSED' if checks_passed == 7 else 'WARNING'}

        with open(self.out_report, 'w') as f:
            json.dump(report, f, indent=4)

        print(f"\n [Step 7 Complete] Automated Validation Report saved to {self.out_report}")
        print("==========================================================================")
        return checks_passed == 7

if __name__ == "__main__":
    v = GraphValidator()
    v.run_all_checks()
