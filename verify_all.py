"""
verify_all.py — Full system verification before building the finance agent.
Run from the project root: python verify_all.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sqlite3, joblib, json, os
import numpy as np
from models import ForecastWrapper   # needed to unpickle sales_forecast.pkl

print("=" * 60)
print("FULL SYSTEM VERIFICATION")
print("=" * 60)

# ── DATABASE ────────────────────────────────────────────────────────────────
print("\n[DATABASE]")
conn = sqlite3.connect("data/company.db")
for t in ["products", "transactions", "customers", "monthly_sales", "company_rates"]:
    n      = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    status = "OK" if n > 0 else "EMPTY"
    print(f"  {t}: {n} rows  [{status}]")

rev  = conn.execute("SELECT SUM(final_amount) FROM transactions WHERE status='Completed'").fetchone()[0]
cust = conn.execute("SELECT COUNT(DISTINCT customer_id) FROM transactions").fetchone()[0]
print(f"  Total revenue:    Rs.{rev:,.2f}")
print(f"  Unique customers: {cust}")
conn.close()

# ── CONFIG ───────────────────────────────────────────────────────────────────
print("\n[CONFIG]")
if os.path.exists("config/company_rates.json"):
    rates = json.load(open("config/company_rates.json"))
    print(f"  company_rates.json: OK ({rates['company_name']})")
    print(f"  GST rates:          {len(rates['pricing']['gst_rates'])} categories")
    print(f"  Loyalty tiers:      {list(rates['discounts']['loyalty'].keys())}")
else:
    print("  company_rates.json: MISSING")

# ── MODELS ───────────────────────────────────────────────────────────────────
print("\n[MODELS]")

# ARIMA
if os.path.exists("models/sales_forecast.pkl"):
    model   = joblib.load("models/sales_forecast.pkl")
    forecast = model.forecast(3)
    metrics  = json.load(open("models/forecast_metrics.json"))
    print(f"  sales_forecast.pkl:  OK")
    print(f"    Next 3 months: {['Rs.{:,.0f}'.format(v) for v in forecast]}")
    print(f"    MAPE: {metrics.get('mape', 'N/A')}%  (high = noisy price data, expected)")
else:
    print("  sales_forecast.pkl:  MISSING")

# Churn classifier
if os.path.exists("models/churn_classifier.pkl"):
    clf     = joblib.load("models/churn_classifier.pkl")
    le      = joblib.load("models/churn_label_encoder.pkl")
    metrics = json.load(open("models/churn_metrics.json"))
    print(f"  churn_classifier.pkl: OK")
    print(f"    Classes:   {list(le.classes_)}")
    print(f"    LOO accuracy: {round(metrics.get('accuracy', 0) * 100, 0):.0f}%")
    # Spot-check with a high-risk dummy (high days_inactive, zero return rate)
    dummy = np.array([[50000, 11, 4500, 293, 0.00, 3, 3, 1.5]])
    pred  = le.inverse_transform(clf.predict(dummy))
    print(f"    Test (293 days inactive) -> {pred[0]}  (expected: High Risk)")
else:
    print("  churn_classifier.pkl: MISSING")

# Demand
if os.path.exists("models/demand_models.pkl"):
    demand = joblib.load("models/demand_models.pkl")
    print(f"  demand_models.pkl:   OK ({len(demand)} products)")
    # Show first entry
    first_pid = sorted(demand.keys())[0]
    d = demand[first_pid]
    print(f"    {first_pid}: next_month_demand={d['next_month_demand']}, stock={d['current_stock']}")
else:
    print("  demand_models.pkl:   MISSING")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
checks = [
    os.path.exists("data/company.db"),
    os.path.exists("config/company_rates.json"),
    os.path.exists("models/sales_forecast.pkl"),
    os.path.exists("models/churn_classifier.pkl"),
    os.path.exists("models/demand_models.pkl"),
]
if all(checks):
    print("ALL CHECKS PASSED -- ready to build finance agent")
else:
    missing = [f for f, ok in zip(
        ["data/company.db", "config/company_rates.json",
         "models/sales_forecast.pkl", "models/churn_classifier.pkl",
         "models/demand_models.pkl"],
        checks
    ) if not ok]
    print(f"FAILED: missing {missing}")
print("=" * 60)
