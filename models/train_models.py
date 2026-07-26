"""
train_models.py — Train all ML models for the TechMart India finance agent.

Trains in sequence:
  4A — ARIMA sales forecast (pmdarima.auto_arima, fallback: statsmodels ARIMA(1,1,1))
  4B — Customer churn classifier (RandomForest + LeaveOneOut CV)
  4C — Product demand forecasting (LinearRegression per product)

Saves 7 files to models/:
  sales_forecast.pkl, forecast_metrics.json
  churn_classifier.pkl, churn_label_encoder.pkl,
  churn_feature_columns.pkl, churn_metrics.json
  demand_models.pkl

Run:
    python models/train_models.py
"""

import json
import os
import sqlite3
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

_BASE    = os.path.dirname(os.path.abspath(__file__))
# Add project root to sys.path so `from models import ForecastWrapper` works
# when this script is run directly (python models/train_models.py).
import sys as _sys
_sys.path.insert(0, os.path.dirname(_BASE))
from models import ForecastWrapper   # stable pickle path: models.ForecastWrapper

DB_PATH  = os.path.join(_BASE, "..", "data", "company.db")
MDL_DIR  = _BASE  # save all model files here


def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4A — ARIMA Sales Forecast
# ─────────────────────────────────────────────────────────────────────────────

def train_arima() -> dict:
    print("\n--- 4A: ARIMA Sales Forecast ---")

    ms = _query("SELECT month, total_revenue FROM monthly_sales ORDER BY month")
    revenue = ms["total_revenue"].values.astype(float)
    months  = ms["month"].tolist()

    # Train/test split: last 3 months held out for test
    n_test  = min(3, max(1, len(revenue) // 5))
    n_train = len(revenue) - n_test
    train, test = revenue[:n_train], revenue[n_train:]
    print(f"  Train: {n_train} months ({months[0]} to {months[n_train-1]})")
    print(f"  Test:  {n_test} months ({months[n_train]} to {months[-1]})")

    # Try pmdarima auto_arima first; fall back to statsmodels ARIMA(1,1,1)
    best_order = (1, 1, 1)
    try:
        import pmdarima as pm
        _raw = pm.auto_arima(
            train,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        best_order = _raw.order
        model = ForecastWrapper(_raw, best_order)
        print(f"  auto_arima best order: {best_order}")
    except Exception as e:
        print(f"  pmdarima failed ({e}) — using statsmodels ARIMA(1,1,1)")
        from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
        _raw  = SM_ARIMA(train, order=(1, 1, 1)).fit()
        model = ForecastWrapper(_raw, (1, 1, 1))

    # Test metrics
    preds = np.array(model.forecast(3)).flatten()
    actual = test[:3]
    mae  = float(np.mean(np.abs(actual - preds)))
    mape = float(np.mean(np.abs((actual - preds) / actual)) * 100)
    rmse = float(np.sqrt(np.mean((actual - preds) ** 2)))

    print(f"\n  Test period results:")
    for i, (m, a, p) in enumerate(zip(months[n_train:], actual, preds)):
        print(f"    {m}  actual=Rs.{a:,.0f}  predicted=Rs.{p:,.0f}  err={abs(a-p)/a*100:.1f}%")
    print(f"\n  MAE:  Rs.{mae:,.0f}")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  RMSE: Rs.{rmse:,.0f}")

    # Retrain on ALL 18 months for production
    try:
        import pmdarima as pm
        _raw_full = pm.auto_arima(
            revenue,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        best_order = _raw_full.order
        model = ForecastWrapper(_raw_full, best_order)
    except Exception:
        from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
        _raw_full = SM_ARIMA(revenue, order=(1, 1, 1)).fit()
        model     = ForecastWrapper(_raw_full, (1, 1, 1))

    # 3-month forward forecast
    next3 = list(np.array(model.forecast(3)).flatten())
    print(f"\n  Next 3 months forecast: {['Rs.{:,.0f}'.format(v) for v in next3]}")

    # Save
    pkl_path     = os.path.join(MDL_DIR, "sales_forecast.pkl")
    metrics_path = os.path.join(MDL_DIR, "forecast_metrics.json")
    joblib.dump(model, pkl_path)
    metrics = {
        "mae":         round(mae, 2),
        "mape":        round(mape, 2),
        "rmse":        round(rmse, 2),
        "best_order":  list(best_order),
        "train_months": 18,
        "last_month":  months[-1],
    }
    json.dump(metrics, open(metrics_path, "w"), indent=2)
    print(f"\n  Saved: {pkl_path}")
    print(f"  Saved: {metrics_path}")

    return {"best_order": best_order, "mae": mae, "mape": mape, "rmse": rmse, "next3": next3}


# ─────────────────────────────────────────────────────────────────────────────
# 4B — Customer Churn Classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_churn() -> dict:
    print("\n--- 4B: Customer Churn Classifier ---")

    custs = _query("SELECT * FROM customers ORDER BY customer_id")
    txns  = _query("SELECT * FROM transactions")

    feature_rows = []
    for _, c in custs.iterrows():
        cid = c["customer_id"]
        ct  = txns[txns["customer_id"] == cid]

        total_spend = float(ct[ct["status"] == "Completed"]["final_amount"].sum())
        order_count = int((ct["status"] == "Completed").sum())
        aov         = total_spend / order_count if order_count > 0 else 0.0
        days_inact  = int(c["days_inactive"])
        return_rate = float((ct["status"] == "Returned").sum() / max(len(ct), 1))
        uniq_cats   = int(ct["category"].nunique())
        uniq_prods  = int(ct["product_id"].nunique())
        avg_qty     = float(ct["quantity"].mean()) if len(ct) > 0 else 0.0

        feature_rows.append({
            "customer_id":       cid,
            "customer_name":     c["customer_name"],
            "tier":              c["tier"],
            "total_spend":       total_spend,
            "order_count":       order_count,
            "avg_order_value":   aov,
            "days_inactive":     days_inact,
            "return_rate":       return_rate,
            "unique_categories": uniq_cats,
            "unique_products":   uniq_prods,
            "avg_quantity":      avg_qty,
        })

    feat_df = pd.DataFrame(feature_rows)

    # Label creation: based on days_inactive + return_rate
    def _label(row):
        if row["days_inactive"] > 90 or row["return_rate"] > 0.25:
            return "High Risk"
        elif row["days_inactive"] > 45 or row["return_rate"] > 0.15:
            return "Medium Risk"
        else:
            return "Low Risk"

    feat_df["churn_label"] = feat_df.apply(_label, axis=1)

    feature_cols = [
        "total_spend", "order_count", "avg_order_value",
        "days_inactive", "return_rate", "unique_categories",
        "unique_products", "avg_quantity",
    ]

    print("\n  Feature matrix:")
    for _, r in feat_df.iterrows():
        print(
            f"    {r['customer_id']} ({r['tier']:<10})  "
            f"orders={r['order_count']:>3}  "
            f"days_inactive={r['days_inactive']:>4}  "
            f"return_rate={r['return_rate']:.2f}  "
            f"label={r['churn_label']}"
        )

    X  = feat_df[feature_cols].values
    le = LabelEncoder()
    y  = le.fit_transform(feat_df["churn_label"].values)

    print(f"\n  Classes: {list(le.classes_)}")

    # LeaveOneOut cross-validation
    loo       = LeaveOneOut()
    clf_cv    = RandomForestClassifier(n_estimators=100, random_state=42)
    loo_preds = []
    for train_idx, test_idx in loo.split(X):
        clf_cv.fit(X[train_idx], y[train_idx])
        loo_preds.append(clf_cv.predict(X[test_idx])[0])

    accuracy = float(np.mean(np.array(loo_preds) == y))
    print(f"  LOO cross-val accuracy: {accuracy * 100:.0f}%")

    # Final model on all 10 customers
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    importances = dict(zip(feature_cols, np.round(clf.feature_importances_, 4).tolist()))

    print("\n  Per-customer predictions (final model):")
    for _, r in feat_df.iterrows():
        xi   = np.array([[r[c] for c in feature_cols]])
        pred = le.inverse_transform(clf.predict(xi))[0]
        prob = clf.predict_proba(xi)[0]
        print(f"    {r['customer_id']} {r['customer_name']:<16} -> {pred}  (proba: {dict(zip(le.classes_, np.round(prob, 2)))})")

    print(f"\n  Feature importances: {importances}")

    # Save
    pkl_clf  = os.path.join(MDL_DIR, "churn_classifier.pkl")
    pkl_le   = os.path.join(MDL_DIR, "churn_label_encoder.pkl")
    pkl_feat = os.path.join(MDL_DIR, "churn_feature_columns.pkl")
    met_path = os.path.join(MDL_DIR, "churn_metrics.json")

    joblib.dump(clf,          pkl_clf)
    joblib.dump(le,           pkl_le)
    joblib.dump(feature_cols, pkl_feat)
    json.dump({
        "accuracy":            round(accuracy, 4),
        "classes":             list(le.classes_),
        "feature_importances": importances,
        "n_customers":         len(feat_df),
    }, open(met_path, "w"), indent=2)

    for p in [pkl_clf, pkl_le, pkl_feat, met_path]:
        print(f"  Saved: {p}")

    return {"accuracy": accuracy, "classes": list(le.classes_), "importances": importances}


# ─────────────────────────────────────────────────────────────────────────────
# 4C — Product Demand Forecasting
# ─────────────────────────────────────────────────────────────────────────────

def train_demand() -> dict:
    print("\n--- 4C: Product Demand Forecasting ---")

    # Monthly quantity per product
    txns  = _query(
        "SELECT product_id, product_name, month, SUM(quantity) as qty "
        "FROM transactions WHERE status='Completed' "
        "GROUP BY product_id, month ORDER BY product_id, month"
    )
    prods = _query("SELECT product_id, product_name, stock_quantity FROM products")
    stock = {r["product_id"]: int(r["stock_quantity"]) for _, r in prods.iterrows()}

    # All months in order
    all_months  = sorted(txns["month"].unique().tolist())
    month_index = {m: i for i, m in enumerate(all_months)}

    demand_dict = {}
    reorder_needed = []

    print(f"\n  {'Product':<12} {'Name':<30} {'PredDemand':>10} {'Stock':>8} {'Reorder':>8}")
    for pid in sorted(txns["product_id"].unique()):
        pdf      = txns[txns["product_id"] == pid].copy()
        if len(pdf) < 6:
            continue   # not enough history for meaningful regression

        pdf["midx"] = pdf["month"].map(month_index)
        X_reg = pdf["midx"].values.reshape(-1, 1)
        y_reg = pdf["qty"].values.astype(float)

        reg = LinearRegression()
        reg.fit(X_reg, y_reg)

        # Predict the next month (index = len(all_months))
        next_idx     = np.array([[len(all_months)]])
        pred_demand  = max(1, int(round(float(reg.predict(next_idx)[0]))))
        curr_stock   = stock.get(pid, 0)
        reorder      = pred_demand > curr_stock

        pname = pdf["product_name"].iloc[0]
        print(
            f"  {pid:<12} {pname:<30} {pred_demand:>10} {curr_stock:>8} "
            f"{'YES' if reorder else 'no':>8}"
        )

        demand_dict[pid] = {
            "model":             reg,
            "next_month_demand": pred_demand,
            "current_stock":     curr_stock,
            "reorder_needed":    reorder,
            "product_name":      pname,
        }

        if reorder:
            reorder_needed.append(pid)

    print(f"\n  Products needing reorder: {reorder_needed or 'none'}")

    pkl_path = os.path.join(MDL_DIR, "demand_models.pkl")
    joblib.dump(demand_dict, pkl_path)
    print(f"  Saved: {pkl_path}")

    return {"reorder_needed": reorder_needed, "n_models": len(demand_dict)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    arima_res  = train_arima()
    churn_res  = train_churn()
    demand_res = train_demand()

    print("\n" + "=" * 60)
    print("=== MODEL TRAINING COMPLETE ===")

    print(f"\nARIMA Forecast:")
    print(f"  Best order: {tuple(arima_res['best_order'])}")
    print(f"  Test MAE:   Rs.{arima_res['mae']:,.0f}")
    print(f"  Test MAPE:  {arima_res['mape']:.1f}%")
    print(f"  Next 3 months: {['Rs.{:,.0f}'.format(v) for v in arima_res['next3']]}")

    print(f"\nChurn Classifier:")
    print(f"  Cross-val accuracy: {churn_res['accuracy'] * 100:.0f}%")
    print(f"  Classes: {churn_res['classes']}")
    imps = sorted(churn_res['importances'].items(), key=lambda x: -x[1])
    print(f"  Top features: {imps[:3]}")

    print(f"\nDemand Forecast:")
    print(f"  Models trained: {demand_res['n_models']}")
    print(f"  Products needing reorder: {demand_res['reorder_needed'] or 'none'}")

    model_files = [
        "models/sales_forecast.pkl",
        "models/forecast_metrics.json",
        "models/churn_classifier.pkl",
        "models/churn_label_encoder.pkl",
        "models/churn_feature_columns.pkl",
        "models/churn_metrics.json",
        "models/demand_models.pkl",
    ]
    print("\nFiles saved:")
    for f in model_files:
        exists = "OK" if os.path.exists(f) else "MISSING"
        print(f"  {f}  [{exists}]")

    print("=" * 60)


if __name__ == "__main__":
    main()
