import os, sqlite3
import pandas as pd

EXCEL = "data/company_data.xlsx"
DB    = "data/company.db"

for f in [DB, "data/shipments.db", "data/transactions.db"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Deleted: {f}")

xl = pd.ExcelFile(EXCEL)
df_products  = pd.read_excel(xl, "products")
df_txns      = pd.read_excel(xl, "transactions")
df_customers = pd.read_excel(xl, "customers")
df_monthly   = pd.read_excel(xl, "monthly_sales")
df_rates     = pd.read_excel(xl, "company_rates")

conn = sqlite3.connect(DB)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA foreign_keys=ON")

conn.executescript("""
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS monthly_sales;
DROP TABLE IF EXISTS company_rates;

CREATE TABLE products (
    product_id TEXT PRIMARY KEY, product_name TEXT NOT NULL,
    category TEXT NOT NULL, base_cost REAL NOT NULL,
    margin_pct REAL NOT NULL, tax_pct REAL NOT NULL,
    selling_price REAL NOT NULL, mrp REAL NOT NULL,
    stock_quantity INTEGER NOT NULL, reorder_level INTEGER NOT NULL,
    supplier TEXT NOT NULL, launch_date TEXT NOT NULL
);
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY, customer_name TEXT NOT NULL,
    email TEXT, phone TEXT, city TEXT,
    tier TEXT DEFAULT 'Bronze', join_date TEXT,
    last_purchase TEXT, total_spend REAL DEFAULT 0,
    total_orders INTEGER DEFAULT 0, days_inactive INTEGER DEFAULT 0
);
CREATE TABLE transactions (
    transaction_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL,
    customer_name TEXT NOT NULL, product_id TEXT NOT NULL,
    product_name TEXT NOT NULL, category TEXT NOT NULL,
    quantity INTEGER NOT NULL, unit_price REAL NOT NULL,
    total_amount REAL NOT NULL, discount_pct REAL DEFAULT 0,
    final_amount REAL NOT NULL, payment_method TEXT NOT NULL,
    date TEXT NOT NULL, month TEXT NOT NULL,
    status TEXT DEFAULT 'Completed'
);
CREATE TABLE monthly_sales (
    month TEXT PRIMARY KEY, total_revenue REAL NOT NULL,
    total_orders INTEGER NOT NULL, avg_order_value REAL NOT NULL,
    unique_customers INTEGER NOT NULL, returns_count INTEGER DEFAULT 0,
    top_category TEXT
);
CREATE TABLE company_rates (
    rate_name TEXT PRIMARY KEY,
    rate_value REAL NOT NULL, description TEXT
);
""")
conn.commit()

for _, r in df_products.iterrows():
    conn.execute("INSERT OR REPLACE INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
        str(r["product_id"]), str(r["product_name"]), str(r["category"]),
        float(r["base_cost"]), float(r["margin_pct"]), float(r["tax_pct"]),
        float(r["selling_price"]), float(r["mrp"]),
        int(r["stock_quantity"]), int(r["reorder_level"]),
        str(r["supplier"]), str(r["launch_date"])[:10],
    ))

for _, r in df_customers.iterrows():
    conn.execute("INSERT OR REPLACE INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
        str(r["customer_id"]), str(r["customer_name"]),
        str(r.get("email","")), str(r.get("phone","")),
        str(r.get("city","")), str(r.get("tier","Bronze")),
        str(r.get("join_date",""))[:10],
        str(r.get("last_purchase",""))[:10],
        float(r.get("total_spend",0)),
        int(r.get("total_orders",0)),
        int(r.get("days_inactive",0)),
    ))

for _, r in df_txns.iterrows():
    conn.execute("INSERT OR REPLACE INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        str(r["transaction_id"]), str(r["customer_id"]), str(r["customer_name"]),
        str(r["product_id"]), str(r["product_name"]), str(r["category"]),
        int(r["quantity"]), float(r["unit_price"]),
        float(r["total_amount"]), float(r.get("discount_pct",0)),
        float(r["final_amount"]), str(r["payment_method"]),
        str(r["date"])[:10], str(r["month"])[:7],
        str(r.get("status","Completed")),
    ))

seen = set()
for _, r in df_monthly.iterrows():
    m = str(r["month"])[:7]
    if m in seen: continue
    seen.add(m)
    conn.execute("INSERT OR REPLACE INTO monthly_sales VALUES (?,?,?,?,?,?,?)", (
        m, float(r["total_revenue"]), int(r["total_orders"]),
        float(r["avg_order_value"]), int(r["unique_customers"]),
        int(r.get("returns_count",0)), str(r.get("top_category","")),
    ))

for _, r in df_rates.iterrows():
    conn.execute("INSERT OR REPLACE INTO company_rates VALUES (?,?,?)", (
        str(r["rate_name"]), float(r["rate_value"]),
        str(r.get("description","")),
    ))

conn.commit()

print("\n=== DB VERIFICATION ===")
for t in ["products","transactions","customers","monthly_sales","company_rates"]:
    n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t:<20}: {n} rows")

print("\n=== PRODUCT CHECK ===")
for r in conn.execute("SELECT product_id, product_name, selling_price FROM products ORDER BY product_id").fetchall():
    print(f"  {r[0]} | {r[1]:<22} | Rs.{r[2]:,.2f}")

print("\n=== CUSTOMER CHECK ===")
for r in conn.execute("SELECT customer_id, customer_name, tier, days_inactive FROM customers ORDER BY customer_id").fetchall():
    churn = " <- CHURN RISK" if r[3] > 45 else ""
    print(f"  {r[0]} | {r[1]:<16} | {r[2]:<9} | {r[3]}d{churn}")

conn.close()
print("\nDB replacement complete.")
