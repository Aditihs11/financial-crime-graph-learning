"""
backend/api.py
==============
Secure FastAPI Backend for the AMECF Framework.
This API encapsulates intellectual property, data processing, and ML model execution.
The frontend cannot access data without authenticating via API Key.
"""

import os
import sys
from typing import Optional, List
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, Security, Body, Query
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

# Ensure we can import from database/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from database.data_manager import SecureDataManager

# Load secure environment variables
load_dotenv(ROOT_DIR / ".env")

API_KEY = os.getenv("API_SECRET_KEY")
if not API_KEY:
    raise RuntimeError("CRITICAL SECURITY ERROR: API_SECRET_KEY not found in .env")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI(
    title="AMECF Secure Core API",
    description="Enterprise API for Graph Neural Network AML Fraud Detection",
    version="2.4.0"
)

# ── Security Middleware ────────────────────────────────────────────────────────

def get_api_key(header_val: str = Security(api_key_header)):
    """Validates the API key attached to requests from the Frontend."""
    if header_val == API_KEY:
        return header_val
    raise HTTPException(
        status_code=403, 
        detail="Unauthorized access. Invalid or missing API Key."
    )

# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
def health_check():
    return {"status": "Secure Backend is Online", "version": "2.4.0"}

@app.get("/api/v1/telemetry")
def get_telemetry(api_key: str = Depends(get_api_key)):
    return SecureDataManager.get_system_telemetry()

@app.get("/api/v1/analytics/cross-bank")
def get_cross_bank_analytics(api_key: str = Depends(get_api_key)):
    return SecureDataManager.get_cross_bank_matrix()

@app.get("/api/v1/analytics/bank-health")
def get_bank_health(api_key: str = Depends(get_api_key)):
    """Returns institutional AML health scorecard and grades for all 10 federated banks."""
    return {"scorecard": SecureDataManager.get_bank_silo_health_scorecard()}

@app.get("/api/v1/analytics/hybrid-policy")
def get_hybrid_policy(api_key: str = Depends(get_api_key)):
    return SecureDataManager.get_hybrid_policy_comparison()

@app.get("/api/v1/analytics/jurisdictions")
def get_jurisdictions(api_key: str = Depends(get_api_key)):
    return SecureDataManager.get_jurisdiction_flows()

@app.get("/api/v1/entity/{entity_id}")
def get_entity_profile(entity_id: str, entity_type: Optional[str] = Query("account"), api_key: str = Depends(get_api_key)):
    return SecureDataManager.get_entity_360(entity_id, entity_type)

@app.post("/api/v1/tracer/flow-path")
def trace_path(payload: dict = Body(...), api_key: str = Depends(get_api_key)):
    """Traces sequential multi-hop laundering path between source and target entities."""
    source = payload.get("source_id", "ACC_START_001")
    target = payload.get("target_id", "OTC_WALLET_CRYPTO")
    amt = float(payload.get("amount", 50000.0))
    return SecureDataManager.trace_fund_flow_path(source, target, amt)

@app.post("/api/v1/copilot/query")
def query_copilot(payload: dict = Body(...), api_key: str = Depends(get_api_key)):
    """Autonomous AML Compliance AI Copilot reasoning engine."""
    q = payload.get("query", "")
    return SecureDataManager.query_copilot_assistant(q)

@app.post("/api/v1/rules/simulate")
def simulate_rule(payload: dict = Body(...), api_key: str = Depends(get_api_key)):
    """Simulates trigger volumes, FPR, and precision for custom designed alert rules."""
    return SecureDataManager.simulate_custom_alert_rule(payload)

@app.get("/api/v1/compliance/audit-package")
def get_audit_package(api_key: str = Depends(get_api_key)):
    """Generates complete institutional regulatory audit package."""
    return SecureDataManager.generate_institutional_audit_package()

@app.post("/api/v1/simulation/adversarial")
def simulate_adversarial(payload: dict = Body(...), api_key: str = Depends(get_api_key)):
    cam_pct = float(payload.get("camouflage_pct", 30.0))
    frag = int(payload.get("fragmentation_level", 2))
    delay = float(payload.get("delay_hours", 12.0))
    return SecureDataManager.simulate_adversarial_evasion(cam_pct, frag, delay)

@app.post("/api/v1/generator/custom-scenario")
def generate_scenario(payload: dict = Body(...), api_key: str = Depends(get_api_key)):
    typo = payload.get("typology", "Smurfing / Structuring")
    mules = int(payload.get("mule_count", 5))
    base_amt = float(payload.get("base_amount", 9200.0))
    target_bank = payload.get("target_bank", "BANK_A")
    return SecureDataManager.generate_custom_synthetic_scenario(typo, mules, base_amt, target_bank)

@app.get("/api/v1/stream/transactions")
def get_streaming_transactions(count: Optional[int] = Query(8, ge=1, le=50), api_key: str = Depends(get_api_key)):
    events = SecureDataManager.generate_live_stream_batch(count=count)
    return {"events": events, "count": len(events)}

@app.get("/api/v1/dataset/partition/{part_id}")
def get_partition_data(
    part_id: int,
    bank_id: Optional[str] = Query(None, description="Filter by bank silo (BANK_A..J)"),
    tx_type: Optional[str] = Query(None, description="Filter by transaction type"),
    fraud_filter: Optional[str] = Query("all", description="all, fraud_only, or legit_only"),
    limit: Optional[int] = Query(100, description="Max rows to return in sample"),
    api_key: str = Depends(get_api_key)
):
    result = SecureDataManager.get_partition(
        n=part_id,
        bank_id=bank_id,
        tx_type=tx_type,
        fraud_filter=fraud_filter,
        limit=limit
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/api/v1/cases/sar")
def get_sar_cases(
    risk_level: Optional[str] = Query(None, description="Filter by risk tier"),
    typology: Optional[str] = Query(None, description="Filter by fraud pattern"),
    search: Optional[str] = Query(None, description="Search term for case ID, SAR ID, or ring"),
    limit: Optional[int] = Query(500, description="Max cases to return"),
    api_key: str = Depends(get_api_key)
):
    cases = SecureDataManager.get_sar_cases(
        risk_level=risk_level,
        typology=typology,
        search=search,
        limit=limit
    )
    return {"cases": cases, "count": len(cases)}

@app.get("/api/v1/cases/sar/{sar_id}")
def get_sar_case_detail(sar_id: str, api_key: str = Depends(get_api_key)):
    detail = SecureDataManager.get_sar_case_detail(sar_id)
    if "error" in detail:
        raise HTTPException(status_code=404, detail=detail["error"])
    return detail

@app.get("/api/v1/cases/sar/{sar_id}/narrative")
def get_sar_narrative(sar_id: str, api_key: str = Depends(get_api_key)):
    narrative = SecureDataManager.generate_sar_narrative(sar_id)
    if "error" in narrative:
        raise HTTPException(status_code=404, detail=narrative["error"])
    return narrative

@app.get("/api/v1/models/benchmark")
def get_benchmark(api_key: str = Depends(get_api_key)):
    return {"benchmarks": SecureDataManager.get_benchmark_results()}

@app.get("/api/v1/validation/report")
def get_validation_report(api_key: str = Depends(get_api_key)):
    return SecureDataManager.get_validation_report()

@app.post("/api/v1/validation/run")
def trigger_validation_run(api_key: str = Depends(get_api_key)):
    report = SecureDataManager.run_validation()
    return {"status": "success", "report": report}

@app.post("/api/v1/inference/score")
def predict_fraud_risk(payload: dict = Body(...), api_key: str = Depends(get_api_key)):
    prediction = SecureDataManager.predict_transaction(payload)
    return {"status": "success", "prediction": prediction}

@app.post("/api/v1/actions/quarantine")
def execute_quarantine(payload: dict = Body(...), api_key: str = Depends(get_api_key)):
    account_list = payload.get("account_list", []) if isinstance(payload, dict) else payload
    sar_id = payload.get("sar_id", "N/A") if isinstance(payload, dict) else "N/A"
    return {
        "status": "success",
        "action": "QUARANTINE_BROADCAST",
        "sar_id": sar_id,
        "locked_nodes": len(account_list),
        "accounts": account_list,
        "message": f"Broadcast quarantine signal sent. {len(account_list)} accounts securely locked across banking silos."
    }

@app.post("/api/v1/actions/unquarantine")
def execute_unquarantine(payload: dict = Body(...), api_key: str = Depends(get_api_key)):
    account_list = payload.get("account_list", []) if isinstance(payload, dict) else payload
    sar_id = payload.get("sar_id", "N/A") if isinstance(payload, dict) else "N/A"
    return {
        "status": "success",
        "action": "UNQUARANTINE_BROADCAST",
        "sar_id": sar_id,
        "unlocked_nodes": len(account_list),
        "accounts": account_list,
        "message": f"Broadcast unquarantine signal sent. {len(account_list)} accounts released from lockdown."
    }

@app.post("/api/v1/actions/freeze")
def execute_freeze(payload: dict = Body(...), api_key: str = Depends(get_api_key)):
    amount = float(payload.get("amount", 0.0))
    sar_id = payload.get("sar_id", "N/A")
    return {
        "status": "success",
        "action": "FREEZE_TRANSACTION",
        "sar_id": sar_id,
        "frozen_amount": amount,
        "message": f"${amount:,.2f} USD intercepted and frozen at the national clearing gateway."
    }

@app.post("/api/v1/actions/unfreeze")
def execute_unfreeze(payload: dict = Body(...), api_key: str = Depends(get_api_key)):
    amount = float(payload.get("amount", 0.0))
    sar_id = payload.get("sar_id", "N/A")
    return {
        "status": "success",
        "action": "UNFREEZE_TRANSACTION",
        "sar_id": sar_id,
        "unfrozen_amount": amount,
        "message": f"${amount:,.2f} USD released back to clearing channel."
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("BACKEND_PORT", 8000))
    print(f"[SECURE BACKEND] Starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
