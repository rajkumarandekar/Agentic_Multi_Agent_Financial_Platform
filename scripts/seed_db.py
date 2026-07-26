"""
Create and seed the financial transactions SQLite database.

Run from the project root:
    python scripts/seed_db.py
"""

import os
import random
import sqlite3
from datetime import datetime, timedelta

DB_PATH = os.getenv("DB_PATH", "data/shipments.db")
random.seed(42)

CUSTOMERS = {
    "CUST001": {"name": "Arjun Sharma", "balance": 50_000.0, "salary": 80_000.0},
    "CUST002": {"name": "Priya Patel",  "balance": 75_000.0, "salary": 100_000.0},
}

POOLS = {
    "Groceries":     ["BigBasket", "DMart", "Reliance Fresh"],
    "Dining":        ["Zomato", "Swiggy", "Starbucks"],
    "Transport":     ["Uber", "Ola", "Metro"],
    "Shopping":      ["Amazon", "Flipkart", "Myntra"],
    "Entertainment": ["Netflix", "Movies"],
    "Utilities":     ["Electricity Board", "Mobile Recharge", "Internet"],
    "Health":        ["Pharmacy"],
    "Fitness":       ["Gym Membership", "Yoga Class"],
    "Beauty":        ["Nykaa"],
    "Supplements":   ["HealthKart"],
}


def _p(cat): return random.choice(POOLS[cat])
def _a(lo, hi): return round(random.uniform(lo, hi), 2)


def _day_txns(cid, d):
    wd = d.weekday()
    is_weekday = wd < 5
    is_month_start = d.day <= 3
    txns = []

    if is_month_start:
        txns += [
            ("Utilities", "Electricity Board",  _a(800,  1500)),
            ("Utilities", "Mobile Recharge",     _a(299,   599)),
            ("Utilities", "Internet",            _a(499,   999)),
            ("Entertainment", "Netflix",         _a(199,   499)),
        ]
        if cid == "CUST001":
            txns.append(("Fitness", "Gym Membership", _a(1500, 2500)))
        else:
            txns.append(("Fitness", "Yoga Class",     _a(1500, 2000)))
            txns.append(("Health",  "HealthKart",     _a(500,   800)))

    if is_weekday:
        if cid == "CUST001":
            txns.append(("Transport", _p("Transport"), _a(80, 300)))
        elif random.random() < 0.25:
            txns.append(("Transport", _p("Transport"), _a(50, 150)))
        txns.append(("Dining", _p("Dining"), _a(150, 600)))
        if random.random() < 0.5:
            txns.append(("Groceries", _p("Groceries"), _a(200, 800)))
        if cid == "CUST002" and random.random() < 0.15:
            txns.append(("Shopping", "Myntra", _a(500, 3000)))
    else:
        txns.append(("Dining",        _p("Dining"),        _a(500, 1500)))
        txns.append(("Groceries",     _p("Groceries"),     _a(800, 2000)))
        if random.random() < 0.65:
            txns.append(("Entertainment", "Movies",        _a(300,  800)))
        if random.random() < 0.4:
            txns.append(("Shopping", _p("Shopping"),       _a(500, 4000)))
        if cid == "CUST002" and random.random() < 0.35:
            txns.append(("Beauty", "Nykaa",                _a(500, 2500)))

    # Arjun: chicken Tuesday, eggs+milk Monday
    if cid == "CUST001":
        if wd == 1:
            txns.append(("Groceries", "Licious",  _a(300, 600)))
        if wd == 0:
            txns.append(("Groceries", "DMart",    _a(180, 320)))

    if random.random() < 0.06:
        txns.append(("Health", "Pharmacy", _a(200, 800)))

    return txns


def generate_rows():
    rows = []
    txn_id = 1
    end   = datetime.today().date()
    start = end - timedelta(days=60)

    for cid, info in CUSTOMERS.items():
        balance = info["balance"]
        d = start
        while d <= end:
            if d.day == 1:
                balance += info["salary"]
                rows.append((f"TXN{txn_id:04d}", cid, info["name"], d.isoformat(),
                              "Salary", "Employer", -round(info["salary"], 2), round(balance, 2)))
                txn_id += 1

            for cat, merchant, amount in _day_txns(cid, d):
                if balance - amount < 5_000:
                    continue
                balance -= amount
                rows.append((f"TXN{txn_id:04d}", cid, info["name"], d.isoformat(),
                              cat, merchant, round(amount, 2), round(balance, 2)))
                txn_id += 1

            d += timedelta(days=1)

    return rows


def seed():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS transactions")
    cur.execute("DROP TABLE IF EXISTS shipments")
    cur.execute("""
        CREATE TABLE transactions (
            transaction_id TEXT PRIMARY KEY,
            customer_id    TEXT NOT NULL,
            customer_name  TEXT NOT NULL,
            date           TEXT NOT NULL,
            category       TEXT NOT NULL,
            merchant       TEXT NOT NULL,
            amount         REAL NOT NULL,
            balance        REAL NOT NULL
        )
    """)
    rows = generate_rows()
    cur.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    print(f"[seed_db] Seeded {len(rows)} transactions -> {DB_PATH}")


if __name__ == "__main__":
    seed()
