"""
generate_data.py — Produce realistic TechMart India e-commerce data.

Outputs data/company_data.xlsx with 5 sheets:
  - products      (20 rows)   product catalogue with costs, margins, GST
  - transactions  (600+ rows) 18-month history with seasonal Diwali/year-end spikes
  - monthly_sales (18 rows)   aggregated revenue & KPIs per month
  - customers     (10 rows)   customer master with tier, lifetime value, churn signals
  - company_rates (15 rows)   TechMart pricing rules (GST, loyalty, bulk, shipping)

Run:
    pip install pandas openpyxl numpy
    python data/generate_data.py
"""

import calendar
import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

TODAY       = date(2026, 7, 1)          # fixed for reproducibility
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "company_data.xlsx")


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTS  (20 rows across 5 categories)
# selling_price = base_cost × (1 + margin/100) × (1 + tax/100)
# mrp           = selling_price × random(1.08–1.18)
# ─────────────────────────────────────────────────────────────────────────────

_PRODUCT_CATALOG = [
    # (name,                       category,          base_cost, margin_pct, tax_pct)
    ("Samsung Galaxy S24",         "Electronics",       18_000,    22,        18),
    ("Apple AirPods Pro",          "Electronics",       12_000,    25,        18),
    ("LG 43-inch Smart TV",        "Electronics",       28_000,    20,        18),
    ("Lenovo IdeaPad Laptop",      "Electronics",       42_000,    18,        18),
    ("JBL Bluetooth Speaker",      "Electronics",        3_500,    30,        18),
    ("Men's Formal Shirt",         "Clothing",              800,    45,         5),
    ("Women's Cotton Kurta",       "Clothing",              600,    50,         5),
    ("Levis 511 Jeans",            "Clothing",            2_200,    40,         5),
    ("Puma Winter Jacket",         "Clothing",            3_500,    38,         5),
    ("Nike Running Shoes",         "Clothing",            4_500,    35,         5),
    ("Prestige Rice Cooker",       "Home & Kitchen",      2_800,    32,        12),
    ("Philips Air Fryer",          "Home & Kitchen",      6_500,    28,        12),
    ("Story@Home Bedsheet Set",    "Home & Kitchen",        900,    45,        12),
    ("Dyson V8 Vacuum Cleaner",    "Home & Kitchen",     28_000,    22,        12),
    ("Yoga Mat Premium",           "Sports",                800,    40,        12),
    ("Nivia Football",             "Sports",                500,    38,        12),
    ("Cosco Cricket Kit",          "Sports",              3_500,    30,        12),
    ("Lakme Face Serum",           "Beauty",              1_200,    50,        18),
    ("Philips Hair Dryer",         "Beauty",              2_500,    42,        18),
    ("Fogg Signature Perfume",     "Beauty",                800,    55,        18),
]

_SUPPLIERS = [
    "TechVision Pvt Ltd", "FashionHub Exports", "HomeComfort Ltd",
    "SportsPrime India",  "BeautyBazaar Co",    "EliteTrade Corp",
]

# Fixed launch dates so the output is reproducible
_LAUNCH_DAYS_AGO = [
    365, 400, 290, 500, 180, 270, 310, 200, 350, 240,
    420, 150, 600, 730, 100, 200, 330, 90,  260, 380,
]


def _build_products() -> pd.DataFrame:
    rows = []
    for i, ((name, cat, cost, margin, tax), days_ago) in enumerate(
        zip(_PRODUCT_CATALOG, _LAUNCH_DAYS_AGO), start=1
    ):
        pid          = f"PRD{i:03d}"
        selling      = round(cost * (1 + margin / 100) * (1 + tax / 100), 2)
        mrp          = round(selling * (1 + random.uniform(0.08, 0.18)), 2)
        launch       = TODAY - timedelta(days=days_ago)
        supplier_idx = (i - 1) % len(_SUPPLIERS)
        rows.append({
            "product_id":      pid,
            "product_name":    name,
            "category":        cat,
            "base_cost":       cost,
            "margin_pct":      margin,
            "tax_pct":         tax,
            "selling_price":   selling,
            "mrp":             mrp,
            "stock_quantity":  random.randint(15, 500),
            "reorder_level":   random.randint(10,  50),
            "supplier":        _SUPPLIERS[supplier_idx],
            "launch_date":     launch.isoformat(),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOMERS  (10 rows)
# ─────────────────────────────────────────────────────────────────────────────

_CUSTOMERS = [
    # (id,       name,               email,                         phone,        city,        tier,       join_days_ago)
    ("CUST001", "Arjun Sharma",    "arjun.sharma@email.com",     "9876543210", "Mumbai",    "Gold",      400),
    ("CUST002", "Priya Patel",     "priya.patel@email.com",      "9876543211", "Delhi",     "Silver",    350),
    ("CUST003", "Rahul Verma",     "rahul.verma@email.com",      "9876543212", "Bangalore", "Platinum",  600),
    ("CUST004", "Sneha Gupta",     "sneha.gupta@email.com",      "9876543213", "Chennai",   "Bronze",    200),
    ("CUST005", "Vikram Singh",    "vikram.singh@email.com",     "9876543214", "Hyderabad", "Gold",      500),
    ("CUST006", "Ananya Iyer",     "ananya.iyer@email.com",      "9876543215", "Pune",      "Silver",    280),
    ("CUST007", "Kiran Reddy",     "kiran.reddy@email.com",      "9876543216", "Mumbai",    "Bronze",    180),
    ("CUST008", "Divya Nair",      "divya.nair@email.com",       "9876543217", "Delhi",     "Silver",    450),
    ("CUST009", "Rohit Joshi",     "rohit.joshi@email.com",      "9876543218", "Bangalore", "Bronze",    300),
    ("CUST010", "Meera Krishnan",  "meera.krishnan@email.com",   "9876543219", "Chennai",   "Gold",      550),
]

# Base monthly purchase frequency weights — higher tier customers buy more often.
# CUST004/CUST007/CUST009 are Bronze with low weight (churn candidates for ML model).
_CUST_FREQ: dict[str, int] = {
    "CUST001": 8,   "CUST002": 5,  "CUST003": 12,  "CUST004": 3,
    "CUST005": 9,   "CUST006": 6,  "CUST007": 2,   "CUST008": 7,
    "CUST009": 3,   "CUST010": 10,
}


# ─────────────────────────────────────────────────────────────────────────────
# TRANSACTIONS  (600+ rows)
# Date range: Jan 2025 → Jun 2026  (18 months)
# ─────────────────────────────────────────────────────────────────────────────

# (month_key, seasonal_multiplier)
# Oct/Nov: Diwali spike; Mar: year-end; Feb: post-holiday dip
_MONTHS = [
    ("2025-01", 0.82), ("2025-02", 0.70), ("2025-03", 1.25),
    ("2025-04", 0.80), ("2025-05", 0.88), ("2025-06", 0.90),
    ("2025-07", 0.85), ("2025-08", 0.92), ("2025-09", 1.00),
    ("2025-10", 1.60), ("2025-11", 1.90), ("2025-12", 1.15),
    ("2026-01", 0.88), ("2026-02", 0.75), ("2026-03", 1.30),
    ("2026-04", 0.88), ("2026-05", 0.95), ("2026-06", 1.05),
]

# 0.8 % MoM growth trend
_GROWTH = [1 + 0.008 * i for i in range(18)]

_PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash"]

# Discount distribution: 70 % have 0 %, rest spread across 5–15 %
_DISCOUNTS = [0] * 70 + [5] * 12 + [8] * 8 + [10] * 7 + [15] * 3

# Status distribution: 90 % Completed, 7 % Returned, 3 % Cancelled
_STATUSES = (["Completed"] * 90) + (["Returned"] * 7) + (["Cancelled"] * 3)

# Churn signal: Bronze customers stop buying from month index 10 onward (Nov 2025+)
# This creates a realistic decreasing-purchase signal for the ML churn model.
_CHURN_CUSTOMERS = {"CUST004", "CUST007", "CUST009"}
_CHURN_FROM_IDX  = 10   # 2025-11 onward


def _build_transactions(products_df: pd.DataFrame) -> pd.DataFrame:
    prod_list = products_df.to_dict("records")

    # Weighted customer pool for random selection
    cust_pool: list[str] = []
    for cid, *_ in _CUSTOMERS:
        cust_pool.extend([cid] * _CUST_FREQ[cid])

    # Lookup name from id
    cust_name_map = {row[0]: row[1] for row in _CUSTOMERS}

    rows: list[dict] = []
    txn_idx = 1

    for m_idx, (month_key, seasonal) in enumerate(_MONTHS):
        year, mon    = int(month_key[:4]), int(month_key[5:])
        days_in_mon  = calendar.monthrange(year, mon)[1]

        # Base 30 transactions/month × seasonal × growth trend
        n_txn = int(30 * seasonal * _GROWTH[m_idx]) + random.randint(-3, 3)
        n_txn = max(n_txn, 5)   # floor to avoid 0-transaction months

        # ── Special: bulk corporate order in October 2025 (anomaly for ARIMA) ─
        if month_key == "2025-10":
            laptop = next(p for p in prod_list if "Laptop" in p["product_name"])
            qty    = 10
            sp     = laptop["selling_price"]
            ta     = round(qty * sp, 2)
            disc   = 15
            fa     = round(ta * (1 - disc / 100), 2)
            rows.append({
                "transaction_id": f"TXN{txn_idx:04d}",
                "customer_id":    "CUST003",
                "customer_name":  "Rahul Verma",
                "product_id":     laptop["product_id"],
                "product_name":   laptop["product_name"],
                "category":       laptop["category"],
                "quantity":       qty,
                "unit_price":     sp,
                "total_amount":   ta,
                "discount_pct":   disc,
                "final_amount":   fa,
                "payment_method": "Net Banking",
                "date":           date(year, mon, 15).isoformat(),
                "month":          month_key,
                "status":         "Completed",
            })
            txn_idx += 1

        for _ in range(n_txn):
            cust_id = random.choice(cust_pool)

            # Churn signal: Bronze customers rarely purchase after Nov 2025
            if cust_id in _CHURN_CUSTOMERS and m_idx >= _CHURN_FROM_IDX:
                if random.random() < 0.80:   # 80 % chance to skip this transaction
                    continue

            prod      = random.choice(prod_list)
            sp        = prod["selling_price"]
            qty       = random.randint(1, 6)
            ta        = round(qty * sp, 2)
            disc      = random.choice(_DISCOUNTS)
            fa        = round(ta * (1 - disc / 100), 2)
            txn_date  = date(year, mon, random.randint(1, days_in_mon))
            status    = random.choice(_STATUSES)

            rows.append({
                "transaction_id": f"TXN{txn_idx:04d}",
                "customer_id":    cust_id,
                "customer_name":  cust_name_map[cust_id],
                "product_id":     prod["product_id"],
                "product_name":   prod["product_name"],
                "category":       prod["category"],
                "quantity":       qty,
                "unit_price":     sp,
                "total_amount":   ta,
                "discount_pct":   disc,
                "final_amount":   fa,
                "payment_method": random.choice(_PAYMENT_METHODS),
                "date":           txn_date.isoformat(),
                "month":          month_key,
                "status":         status,
            })
            txn_idx += 1

    df = pd.DataFrame(rows)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MONTHLY SALES  (18 rows — aggregated from completed transactions)
# ─────────────────────────────────────────────────────────────────────────────

def _build_monthly_sales(txn_df: pd.DataFrame) -> pd.DataFrame:
    completed = txn_df[txn_df["status"] == "Completed"]
    rows: list[dict] = []

    for month_key, _ in _MONTHS:
        m_comp  = completed[completed["month"] == month_key]
        m_all   = txn_df[txn_df["month"] == month_key]
        revenue = round(m_comp["final_amount"].sum(), 2)
        orders  = len(m_comp)
        aov     = round(revenue / orders, 2) if orders > 0 else 0.0
        returns = int((m_all["status"] == "Returned").sum())

        if orders > 0:
            top_cat = m_comp.groupby("category")["final_amount"].sum().idxmax()
        else:
            top_cat = "N/A"

        rows.append({
            "month":             month_key,
            "total_revenue":     revenue,
            "total_orders":      orders,
            "avg_order_value":   aov,
            "unique_customers":  m_comp["customer_id"].nunique(),
            "returns_count":     returns,
            "top_category":      top_cat,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOMERS  (enriched with aggregated stats from transactions)
# ─────────────────────────────────────────────────────────────────────────────

def _build_customers(txn_df: pd.DataFrame) -> pd.DataFrame:
    completed = txn_df[txn_df["status"] == "Completed"]
    rows: list[dict] = []

    for cid, name, email, phone, city, tier, join_days_ago in _CUSTOMERS:
        cust_txn     = completed[completed["customer_id"] == cid]
        total_spend  = round(cust_txn["final_amount"].sum(), 2)
        total_orders = len(cust_txn)
        join_date    = TODAY - timedelta(days=join_days_ago)

        # Days since last purchase (None if never purchased)
        if total_orders > 0:
            last_purchase = cust_txn["date"].max()
            days_inactive = (TODAY - date.fromisoformat(last_purchase)).days
        else:
            last_purchase = None
            days_inactive = join_days_ago

        rows.append({
            "customer_id":      cid,
            "customer_name":    name,
            "email":            email,
            "phone":            phone,
            "city":             city,
            "tier":             tier,
            "join_date":        join_date.isoformat(),
            "last_purchase":    last_purchase,
            "total_spend":      total_spend,
            "total_orders":     total_orders,
            "days_inactive":    days_inactive,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# COMPANY RATES  (15 rows — TechMart pricing config, also written to JSON later)
# ─────────────────────────────────────────────────────────────────────────────

def _build_company_rates() -> pd.DataFrame:
    rates = [
        ("bulk_discount_threshold",   50_000, "Orders above this value qualify for bulk discount (₹)"),
        ("bulk_discount_pct",              8, "Bulk order discount percentage"),
        ("loyalty_discount_bronze",        0, "Bronze tier loyalty discount %"),
        ("loyalty_discount_silver",        5, "Silver tier loyalty discount %"),
        ("loyalty_discount_gold",          8, "Gold tier loyalty discount %"),
        ("loyalty_discount_platinum",     12, "Platinum tier loyalty discount %"),
        ("seasonal_markup_pct",            5, "Additional markup during festival seasons (Oct-Nov)"),
        ("gst_electronics",               18, "GST rate for Electronics (%)"),
        ("gst_clothing",                   5, "GST rate for Clothing (%)"),
        ("gst_home_kitchen",              12, "GST rate for Home & Kitchen (%)"),
        ("gst_beauty",                    18, "GST rate for Beauty (%)"),
        ("gst_sports",                    12, "GST rate for Sports (%)"),
        ("shipping_free_above",          999, "Order value above which shipping is free (₹)"),
        ("shipping_flat_rate",            49, "Flat shipping charge for orders below free threshold (₹)"),
        ("return_window_days",             7, "Days within which returns are accepted"),
    ]
    return pd.DataFrame(rates, columns=["rate_name", "rate_value", "description"])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Generating TechMart India company data ...")

    products      = _build_products()
    transactions  = _build_transactions(products)
    monthly_sales = _build_monthly_sales(transactions)
    customers     = _build_customers(transactions)
    company_rates = _build_company_rates()

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        products.to_excel(      writer, sheet_name="products",      index=False)
        transactions.to_excel(  writer, sheet_name="transactions",  index=False)
        monthly_sales.to_excel( writer, sheet_name="monthly_sales", index=False)
        customers.to_excel(     writer, sheet_name="customers",     index=False)
        company_rates.to_excel( writer, sheet_name="company_rates", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    comp   = transactions[transactions["status"] == "Completed"]
    ret    = transactions[transactions["status"] == "Returned"]
    canc   = transactions[transactions["status"] == "Cancelled"]

    print(f"\n{'='*60}")
    print(f"  Products:             {len(products)}")
    print(f"  Total transactions:   {len(transactions)}")
    print(f"    Completed:          {len(comp)}")
    print(f"    Returned:           {len(ret)}")
    print(f"    Cancelled:          {len(canc)}")
    print(f"  Date range:           {transactions['date'].min()} -> {transactions['date'].max()}")
    print(f"  Unique customers:     {transactions['customer_id'].nunique()}")
    print(f"  Total revenue (comp): Rs.{comp['final_amount'].sum():,.2f}")
    print(f"\n  Monthly revenue (completed orders):")
    for _, row in monthly_sales.iterrows():
        bar = "#" * int(row["total_revenue"] / 40_000)
        print(f"    {row['month']}  Rs.{row['total_revenue']:>12,.2f}  {bar}")

    print(f"\n  Customer summary:")
    print(f"  {'ID':<8} {'Name':<18} {'Tier':<10} {'Orders':>6} {'Spend':>14} {'DaysInactive':>12}")
    for _, c in customers.iterrows():
        print(
            f"  {c['customer_id']:<8} {c['customer_name']:<18} {c['tier']:<10} "
            f"{c['total_orders']:>6} Rs.{c['total_spend']:>10,.2f} {c['days_inactive']:>12}"
        )

    print(f"\n  Saved -> {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
