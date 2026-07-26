"""
setup_db.py — Load company_data.xlsx into data/company.db (SQLite).

Drops and recreates company.db on each run (idempotent).
WAL mode and foreign keys are enabled for better concurrency and integrity.

Run:
    python data/setup_db.py
"""

import os
import sqlite3

import pandas as pd

_BASE    = os.path.dirname(os.path.abspath(__file__))
XLSX     = os.path.join(_BASE, "company_data.xlsx")
DB_PATH  = os.path.join(_BASE, "company.db")

_SCHEMA = """
CREATE TABLE products (
    product_id      TEXT PRIMARY KEY,
    product_name    TEXT NOT NULL,
    category        TEXT NOT NULL,
    base_cost       REAL NOT NULL,
    margin_pct      REAL NOT NULL,
    tax_pct         REAL NOT NULL,
    selling_price   REAL NOT NULL,
    mrp             REAL NOT NULL,
    stock_quantity  INTEGER NOT NULL,
    reorder_level   INTEGER NOT NULL,
    supplier        TEXT NOT NULL,
    launch_date     TEXT NOT NULL
);

CREATE TABLE transactions (
    transaction_id  TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL,
    customer_name   TEXT NOT NULL,
    product_id      TEXT NOT NULL,
    product_name    TEXT NOT NULL,
    category        TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    unit_price      REAL NOT NULL,
    total_amount    REAL NOT NULL,
    discount_pct    REAL DEFAULT 0,
    final_amount    REAL NOT NULL,
    payment_method  TEXT NOT NULL,
    date            TEXT NOT NULL,
    month           TEXT NOT NULL,
    status          TEXT DEFAULT 'Completed',
    FOREIGN KEY (product_id)  REFERENCES products(product_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE customers (
    customer_id     TEXT PRIMARY KEY,
    customer_name   TEXT NOT NULL,
    email           TEXT,
    phone           TEXT,
    city            TEXT,
    tier            TEXT DEFAULT 'Bronze',
    join_date       TEXT,
    last_purchase   TEXT,
    total_spend     REAL DEFAULT 0,
    total_orders    INTEGER DEFAULT 0,
    days_inactive   INTEGER DEFAULT 0
);

CREATE TABLE monthly_sales (
    month               TEXT PRIMARY KEY,
    total_revenue       REAL NOT NULL,
    total_orders        INTEGER NOT NULL,
    avg_order_value     REAL NOT NULL,
    unique_customers    INTEGER NOT NULL,
    returns_count       INTEGER DEFAULT 0,
    top_category        TEXT
);

CREATE TABLE company_rates (
    rate_name   TEXT PRIMARY KEY,
    rate_value  REAL NOT NULL,
    description TEXT
);
"""

_EXPECTED = {
    "products":      20,
    "transactions": 550,  # may vary slightly — just check > 0
    "customers":     10,
    "monthly_sales": 18,
    "company_rates": 15,
}


def _read_excel() -> dict[str, pd.DataFrame]:
    print(f"Reading {XLSX} ...")
    xl = pd.ExcelFile(XLSX)
    sheets = {}
    for name in ["products", "transactions", "monthly_sales", "customers", "company_rates"]:
        df = xl.parse(name)
        sheets[name] = df
        print(f"  {name}: {len(df)} rows x {len(df.columns)} cols")
    return sheets


def _coerce_types(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Convert Excel floats to ints where the schema uses INTEGER."""
    int_cols = {
        "products":      ["stock_quantity", "reorder_level"],
        "transactions":  ["quantity"],
        "customers":     ["total_orders", "days_inactive"],
        "monthly_sales": ["total_orders", "unique_customers", "returns_count"],
    }
    for table, cols in int_cols.items():
        for col in cols:
            if col in sheets[table].columns:
                sheets[table][col] = sheets[table][col].fillna(0).astype(int)

    # Ensure NaN→None for nullable TEXT columns (last_purchase, etc.)
    for name, df in sheets.items():
        sheets[name] = df.where(pd.notna(df), other=None)

    return sheets


def _create_db(sheets: dict[str, pd.DataFrame]) -> None:
    # Fresh start
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"\nRemoved existing {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    # Performance + integrity pragmas
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Create schema (split on semicolons, skip empty statements)
    for stmt in _SCHEMA.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)

    conn.commit()
    print("Schema created.")

    # Insert data (foreign-key order: products and customers before transactions)
    load_order = ["products", "customers", "transactions", "monthly_sales", "company_rates"]
    for table in load_order:
        df = sheets[table]
        df.to_sql(table, conn, if_exists="append", index=False)
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        expected = _EXPECTED.get(table, 1)
        status   = "OK" if count >= expected else f"!! expected ~{expected}"
        print(f"  {table}: {count} rows  {status}")

    conn.commit()
    conn.close()


def main() -> None:
    sheets = _read_excel()
    sheets = _coerce_types(sheets)
    print()
    _create_db(sheets)

    # Final verification count
    conn = sqlite3.connect(DB_PATH)
    print("\n=== DATABASE SETUP COMPLETE ===")
    for table in ["products", "transactions", "customers", "monthly_sales", "company_rates"]:
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n} rows")
    conn.close()
    print(f"\n  Saved -> {DB_PATH}")


if __name__ == "__main__":
    main()
