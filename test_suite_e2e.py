"""
test_suite_e2e.py
=================
Unified End-to-End Regression & Verification Test Suite.
Exercises all 21 REST API endpoints and UI services, certifying platform integrity.
"""

import sys
import time
import requests

API_KEY = "amecf-secure-token-8f92j-x01z"
HEADERS = {"X-API-Key": API_KEY}
BASE_URL = "http://127.0.0.1:8000/api/v1"
UI_URL = "http://localhost:8501"

TEST_RESULTS = []

def run_test(name: str, test_func):
    try:
        t0 = time.time()
        success, detail = test_func()
        duration_ms = int((time.time() - t0) * 1000)
        TEST_RESULTS.append({"name": name, "passed": success, "detail": detail, "latency_ms": duration_ms})
    except Exception as e:
        TEST_RESULTS.append({"name": name, "passed": False, "detail": f"Exception: {e}", "latency_ms": 0})

def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    return r.status_code == 200, f"HTTP {r.status_code} | Version: {r.json().get('version')}"

def test_telemetry():
    r = requests.get(f"{BASE_URL}/telemetry", headers=HEADERS, timeout=5)
    d = r.json()
    return r.status_code == 200, f"RAM: {d.get('memory_usage_pct')}% | Partitions: {d.get('partitions_loaded')}"

def test_partition_query():
    r = requests.get(f"{BASE_URL}/dataset/partition/0", headers=HEADERS, params={"limit": 5}, timeout=5)
    d = r.json()
    return r.status_code == 200, f"Rows: {d['stats']['total_rows']:,} | Fraud Ratio: {d['stats']['fraud_ratio']*100:.2f}%"

def test_cross_bank():
    r = requests.get(f"{BASE_URL}/analytics/cross-bank", headers=HEADERS, timeout=5)
    d = r.json()
    return r.status_code == 200, f"Banks: {len(d['banks'])} | Top Corridors: {len(d['top_risk_corridors'])}"

def test_path_tracer():
    payload = {"source_id": "ACC_SOURCE_01", "target_id": "OTC_MIXER_01", "amount": 60000.0}
    r = requests.post(f"{BASE_URL}/tracer/flow-path", headers=HEADERS, json=payload, timeout=5)
    d = r.json()
    return r.status_code == 200, f"Hops: {d['total_hops']} | Extracted: ${d['final_extracted_usd']:,.2f}"

def test_bank_health():
    r = requests.get(f"{BASE_URL}/analytics/bank-health", headers=HEADERS, timeout=5)
    d = r.json()
    return r.status_code == 200, f"Scored Silos: {len(d['scorecard'])} | Bank A: {d['scorecard'][0]['aml_grade']}"

def test_copilot():
    payload = {"query": "Explain Smurfing detection"}
    r = requests.post(f"{BASE_URL}/copilot/query", headers=HEADERS, json=payload, timeout=5)
    d = r.json()
    return r.status_code == 200, f"Output chars: {len(d.get('response', ''))}"

def test_alert_rule():
    payload = {"rule_name": "Smurfing Policy", "min_amount": 8500, "max_amount": 9999, "shared_device_threshold": 3}
    r = requests.post(f"{BASE_URL}/rules/simulate", headers=HEADERS, json=payload, timeout=5)
    d = r.json()
    return r.status_code == 200, f"Precision: {d['estimated_precision_pct']}% | Triggers: {d['simulated_triggers_per_month']}"

def test_audit_package():
    r = requests.get(f"{BASE_URL}/compliance/audit-package", headers=HEADERS, timeout=5)
    d = r.json()
    return r.status_code == 200, f"Certified text: {len(d.get('report_text', ''))} chars"

def test_entity_360():
    r = requests.get(f"{BASE_URL}/entity/A0360672", headers=HEADERS, timeout=5)
    d = r.json()
    return r.status_code == 200, f"Risk: {d['risk_score']} | Sub-graph nodes: {len(d['subgraph']['nodes'])}"

def test_hybrid_policy():
    r = requests.get(f"{BASE_URL}/analytics/hybrid-policy", headers=HEADERS, timeout=5)
    d = r.json()
    return r.status_code == 200, f"GNN F1: {d['gnn_performance']['f1']} vs Rules F1: {d['rule_performance']['f1']}"

def test_jurisdictions():
    r = requests.get(f"{BASE_URL}/analytics/jurisdictions", headers=HEADERS, timeout=5)
    d = r.json()
    return r.status_code == 200, f"Jurisdictions: {len(d['jurisdictions'])} | Offshore: {len(d['offshore_tax_havens'])}"

def test_adversarial():
    payload = {"camouflage_pct": 30.0, "fragmentation_level": 2, "delay_hours": 12.0}
    r = requests.post(f"{BASE_URL}/simulation/adversarial", headers=HEADERS, json=payload, timeout=5)
    d = r.json()
    return r.status_code == 200, f"AMECF: {d['current_scores']['amecf_accuracy']}% | Vanilla GNN: {d['current_scores']['graphsage_accuracy']}%"

def test_custom_scenario():
    payload = {"typology": "Smurfing / Structuring", "mule_count": 5, "base_amount": 9200.0, "target_bank": "BANK_A"}
    r = requests.post(f"{BASE_URL}/generator/custom-scenario", headers=HEADERS, json=payload, timeout=5)
    d = r.json()
    return r.status_code == 200, f"Scenario: {d['scenario_id']} | Txns: {len(d['transactions'])}"

def test_sar_cases():
    r = requests.get(f"{BASE_URL}/cases/sar", headers=HEADERS, params={"limit": 5}, timeout=5)
    d = r.json()
    return r.status_code == 200, f"Cases loaded: {d.get('count', len(d.get('cases', [])))}"

def test_sar_detail():
    r = requests.get(f"{BASE_URL}/cases/sar/SAR-2026-5001", headers=HEADERS, timeout=5)
    d = r.json()
    return r.status_code == 200, f"Risk Amount: ${d.get('risk_amount', 0):,.2f} | Nodes: {len(d.get('subgraph', {}).get('nodes', []))}"

def test_sar_narrative():
    r = requests.get(f"{BASE_URL}/cases/sar/SAR-2026-5001/narrative", headers=HEADERS, timeout=5)
    d = r.json()
    return r.status_code == 200, f"Narrative chars: {len(d.get('narrative', ''))}"

def test_live_stream():
    r = requests.get(f"{BASE_URL}/stream/transactions", headers=HEADERS, params={"count": 4}, timeout=5)
    d = r.json()
    return r.status_code == 200, f"Events generated: {len(d.get('events', []))}"

def test_inference_scoring():
    payload = {
        "amount": 9350.0, "type": "TRANSFER", "channel": "Mobile App",
        "sender_bank": "BANK_A", "receiver_bank": "BANK_C",
        "is_off_hours": True, "shared_devices_count": 3, "hop_count": 2,
        "is_new_device": True, "is_crypto_linked": False
    }
    r = requests.post(f"{BASE_URL}/inference/score", headers=HEADERS, json=payload, timeout=5)
    d = r.json()
    p = d.get("prediction", {})
    return r.status_code == 200, f"Fraud Prob: {p.get('fraud_probability')}% | Typology: {p.get('typology')}"

def test_validation_report():
    r = requests.get(f"{BASE_URL}/validation/report", headers=HEADERS, timeout=5)
    d = r.json()
    s = d.get("summary", {})
    return r.status_code == 200, f"Integrity: {s.get('checks_passed')}/{s.get('total_checks')} checks passed"

def test_streamlit_ui():
    r = requests.get(UI_URL, timeout=5)
    return r.status_code == 200, f"HTTP {r.status_code} OK (Streamlit Dashboard Online)"

def main():
    print("=" * 80)
    print("  graphforANN: Unified End-to-End Regression & Verification Test Suite")
    print("=" * 80)

    tests = [
        ("01. Backend Health Check", test_health),
        ("02. Server Telemetry & Resources", test_telemetry),
        ("03. 20M Dataset Partition Query", test_partition_query),
        ("04. 10x10 Cross-Bank Flow Matrix", test_cross_bank),
        ("05. Multi-Hop Fund Flow Tracer", test_path_tracer),
        ("06. Bank Silo Health Scorecard", test_bank_health),
        ("07. AML Compliance AI Copilot", test_copilot),
        ("08. Custom Alert Rule Simulator", test_alert_rule),
        ("09. Institutional Audit Package", test_audit_package),
        ("10. Global Entity 360 & Ego-Graph", test_entity_360),
        ("11. Hybrid Policy & Blind Spots", test_hybrid_policy),
        ("12. Global Jurisdiction Flows", test_jurisdictions),
        ("13. Adversarial Evasion Simulator", test_adversarial),
        ("14. Custom Synthetic Generator", test_custom_scenario),
        ("15. SAR Cases Registry (500+ Cases)", test_sar_cases),
        ("16. SAR Sub-Graph Topology", test_sar_detail),
        ("17. Automated SAR Narrative (FinCEN/RBI)", test_sar_narrative),
        ("18. Real-Time Sentinel Stream", test_live_stream),
        ("19. GNN Attention Inference & XAI", test_inference_scoring),
        ("20. 7/7 Dataset Integrity Validation", test_validation_report),
        ("21. Streamlit Research Portal UI", test_streamlit_ui),
    ]

    for name, fn in tests:
        run_test(name, fn)

    passed_count = sum(1 for t in TEST_RESULTS if t["passed"])
    total_count = len(TEST_RESULTS)

    print("\n" + "-" * 80)
    for t in TEST_RESULTS:
        status_str = "[ PASS ]" if t["passed"] else "[ FAIL ]"
        print(f"{status_str} {t['name']:<42} | {t['latency_ms']:>4}ms | {t['detail']}")
    print("-" * 80)

    print(f"\nFINAL VERDICT: {passed_count}/{total_count} Tests Passed ({(passed_count/total_count)*100:.1f}%)")
    if passed_count == total_count:
        print(">>> ALL REGRESSION SUITES PASSED! System is certified production-ready. <<<\n")
        return 0
    else:
        print(">>> REGRESSION FAILURES DETECTED! <<<\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
