"""
Create and seed the shipments SQLite database.

Run from the project root:
    python scripts/seed_db.py
"""

import os
import sqlite3

DB_PATH = os.getenv("DB_PATH", "data/shipments.db")

SHIPMENTS = [
    (1,  "SHP-001", "New York",      "Los Angeles",  "delivered",  3.5,  "2024-01-15"),
    (2,  "SHP-002", "Chicago",       "Houston",      "in_transit", 12.0, "2024-01-18"),
    (3,  "SHP-003", "Phoenix",       "Philadelphia", "delivered",  0.8,  "2024-01-20"),
    (4,  "SHP-004", "San Antonio",   "San Diego",    "pending",    5.2,  "2024-01-22"),
    (5,  "SHP-005", "Dallas",        "San Jose",     "delivered",  9.1,  "2024-01-25"),
    (6,  "SHP-006", "Austin",        "Jacksonville", "in_transit", 2.3,  "2024-01-28"),
    (7,  "SHP-007", "Fort Worth",    "Columbus",     "delivered",  7.7,  "2024-02-01"),
    (8,  "SHP-008", "Charlotte",     "Indianapolis", "cancelled",  1.4,  "2024-02-03"),
    (9,  "SHP-009", "San Francisco", "Seattle",      "delivered",  6.6,  "2024-02-05"),
    (10, "SHP-010", "Denver",        "Nashville",    "in_transit", 4.9,  "2024-02-08"),
    (11, "SHP-011", "Oklahoma City", "El Paso",      "delivered",  11.2, "2024-02-10"),
    (12, "SHP-012", "Las Vegas",     "Louisville",   "pending",    3.3,  "2024-02-12"),
    (13, "SHP-013", "Memphis",       "Baltimore",    "delivered",  8.8,  "2024-02-14"),
    (14, "SHP-014", "Boston",        "Milwaukee",    "in_transit", 0.5,  "2024-02-17"),
    (15, "SHP-015", "Portland",      "Albuquerque",  "delivered",  14.0, "2024-02-20"),
]


def seed() -> None:
    """Create the shipments table and insert sample rows."""
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS shipments (
            id              INTEGER PRIMARY KEY,
            tracking_number TEXT    UNIQUE NOT NULL,
            origin          TEXT    NOT NULL,
            destination     TEXT    NOT NULL,
            status          TEXT    NOT NULL,
            weight_kg       REAL    NOT NULL,
            shipped_date    TEXT    NOT NULL
        )
    """)
    cur.executemany(
        "INSERT OR IGNORE INTO shipments VALUES (?,?,?,?,?,?,?)",
        SHIPMENTS,
    )
    conn.commit()
    conn.close()
    print(f"[seed_db] Seeded {len(SHIPMENTS)} shipments → {DB_PATH}")


if __name__ == "__main__":
    seed()
