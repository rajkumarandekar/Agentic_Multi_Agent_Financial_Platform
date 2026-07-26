import sqlite3
import os

DB = "data/company.db"

if not os.path.exists(DB):
    print("ERROR: data/company.db not found")
    exit()

conn = sqlite3.connect(DB)

print("=== PRODUCTS ===")
for r in conn.execute("SELECT product_id, product_name, category, base_cost, margin_pct, tax_pct, selling_price FROM products ORDER BY product_id").fetchall():
    print(f"{r[0]}|{r[1]}|{r[2]}|{r[3]}|{r[4]}|{r[5]}|{r[6]}")

print("\n=== CUSTOMERS ===")
for r in conn.execute("SELECT customer_id, customer_name, tier, city, total_spend, total_orders, days_inactive FROM customers ORDER BY customer_id").fetchall():
    print(f"{r[0]}|{r[1]}|{r[2]}|{r[3]}|{r[4]}|{r[5]}|{r[6]}")

print("\n=== COMPANY RATES ===")
for r in conn.execute("SELECT rate_name, rate_value FROM company_rates").fetchall():
    print(f"{r[0]}|{r[1]}")

print("\n=== MONTHLY SALES ===")
for r in conn.execute("SELECT month, total_revenue, total_orders FROM monthly_sales ORDER BY month").fetchall():
    print(f"{r[0]}|{r[1]}|{r[2]}")

conn.close()
print("\nDone - paste this output to Claude")
