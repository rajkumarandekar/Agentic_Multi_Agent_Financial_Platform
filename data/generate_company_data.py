"""
generate_company_data.py — synthetic TechMart India dataset.

Phase 1 extension (see project chat history): the original 20 products and
10 customers are preserved EXACTLY as they were (same IDs, names, costs,
margins, tiers) — every existing test that references "PRD001 = Laptop =
Rs.47,146.48" or "CUST003 = Vikram Singh = Platinum" depends on this and
must keep passing. Everything beyond that (100 more products, 240 more
customers, 3 years of daily transactions instead of 18 months, credit
limits, a seeded loans table) is purely additive.

Why 3 years, not 18 months: forecasting at day/month/year granularity needs
real historical seasonality to learn from. Seasonality here is a "month of
calendar year" multiplier that repeats across all 3 years (not a fixed
18-month curve), which is what actually gives a forecasting model (ARIMA,
Prophet) genuine year-over-year signal instead of a single unrepeated trend.

Run:
    python data/generate_company_data.py
"""
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

TODAY = datetime(2026, 7, 25)


# ── Products: original 20, UNCHANGED ─────────────────────────────────────────

products_data = [
    ("PRD001","Laptop",          "Electronics",   33774, 18.3, 18),
    ("PRD002","Wireless Earbuds","Electronics",    8912, 22.2, 18),
    ("PRD003","Smartphone",      "Electronics",   30256, 25.2, 18),
    ("PRD004","Tablet",          "Electronics",   41424, 18.1, 18),
    ("PRD005","Cotton Shirt",    "Clothing",       1200, 44.4,  5),
    ("PRD006","Denim Jeans",     "Clothing",       4314, 39.1,  5),
    ("PRD007","Kurta Set",       "Clothing",       4879, 35.7,  5),
    ("PRD008","Running Shoes",   "Clothing",       1365, 31.0,  5),
    ("PRD009","Air Purifier",    "Home & Kitchen", 6322, 29.5, 12),
    ("PRD010","Mixer Grinder",   "Home & Kitchen",10382, 32.0, 12),
    ("PRD011","Pressure Cooker", "Home & Kitchen",11208, 26.6, 12),
    ("PRD012","Water Purifier",  "Home & Kitchen",12800, 32.8, 12),
    ("PRD013","Face Serum",      "Beauty",          479, 39.1, 18),
    ("PRD014","Sunscreen SPF50", "Beauty",         1368, 39.1, 18),
    ("PRD015","Hair Oil Set",    "Beauty",         1755, 37.5, 18),
    ("PRD016","Skincare Kit",    "Beauty",         2993, 31.4, 18),
    ("PRD017","Yoga Mat",        "Sports",         8087, 24.2, 12),
    ("PRD018","Dumbbells Set",   "Sports",         3113, 25.5, 12),
    ("PRD019","Cricket Bat",     "Sports",         7871, 23.4, 12),
    ("PRD020","Fitness Tracker", "Sports",         9741, 27.6, 12),
]

# ── New products: 3 new categories + deepen the original 5 ──────────────────
# Each category is templated (base name + variant words) rather than hand-typed
# one by one — 140 products would be unmaintainable as a flat literal list.
_CATEGORY_GST = {
    "Electronics": 18, "Clothing": 5, "Home & Kitchen": 12, "Beauty": 18,
    "Sports": 12, "Groceries": 5, "Books & Stationery": 5, "Toys & Games": 12,
}
_CATEGORY_COST_RANGE = {
    "Electronics": (3000, 60000), "Clothing": (400, 6000),
    "Home & Kitchen": (800, 18000), "Beauty": (200, 3500),
    "Sports": (500, 12000), "Groceries": (50, 1200),
    "Books & Stationery": (80, 2500), "Toys & Games": (300, 5000),
}
_CATEGORY_MARGIN_RANGE = (15.0, 45.0)

_NEW_PRODUCT_NAMES = {
    "Electronics": [
        "Bluetooth Speaker", "Smart Watch", "4K Monitor", "Gaming Mouse", "Mechanical Keyboard",
        "Power Bank 20000mAh", "Wireless Charger", "Noise Cancelling Headphones", "Webcam HD",
        "Router Wi-Fi 6", "External SSD 1TB", "Action Camera", "Smart Plug", "LED Desk Lamp",
        "Portable Projector",
    ],
    "Clothing": [
        "Formal Blazer", "Silk Saree", "Winter Jacket", "Rain Poncho", "Track Pants",
        "Formal Trousers", "Cotton Saree", "Woolen Sweater", "Leather Belt", "Baseball Cap",
        "Ethnic Kurti", "Nightwear Set", "Sports Bra", "Thermal Wear", "Scarf",
    ],
    "Home & Kitchen": [
        "Induction Cooktop", "Non-Stick Cookware Set", "Electric Kettle", "Vacuum Cleaner",
        "Ceiling Fan", "Table Lamp", "Bedsheet Set", "Curtain Set", "Storage Rack",
        "Coffee Maker", "Toaster", "Rice Cooker", "OTG Oven", "Room Heater", "Water Bottle Set",
    ],
    "Beauty": [
        "Lipstick Set", "Foundation", "Perfume", "Deodorant Pack", "Shampoo Set",
        "Conditioner", "Face Wash", "Body Lotion", "Nail Polish Set", "Hair Straightener",
        "Trimmer", "Makeup Kit", "Beard Oil", "Talcum Powder", "Lip Balm Set",
    ],
    "Sports": [
        "Badminton Racket", "Football", "Basketball", "Skipping Rope", "Resistance Bands",
        "Gym Gloves", "Cycling Helmet", "Swimming Goggles", "Table Tennis Kit", "Chess Set",
        "Camping Tent", "Hiking Backpack", "Water Bottle Sports", "Knee Support", "Yoga Block",
    ],
    "Groceries": [
        "Basmati Rice 5kg", "Wheat Flour 5kg", "Cooking Oil 1L", "Toor Dal 1kg", "Sugar 1kg",
        "Tea Leaves 500g", "Instant Coffee", "Spice Mix Combo", "Salt 1kg", "Ghee 500ml",
        "Breakfast Cereal", "Biscuit Pack", "Namkeen Combo", "Pickle Jar", "Honey 500g",
    ],
    "Books & Stationery": [
        "Notebook Set", "Ballpoint Pen Pack", "Fiction Novel", "Self-Help Book", "Kids Storybook",
        "Sketchbook", "Highlighter Set", "Geometry Box", "Desk Organizer", "Sticky Notes Pack",
        "Fountain Pen", "Exam Prep Guide", "Comic Book", "Diary", "Whiteboard Marker Set",
    ],
    "Toys & Games": [
        "Building Blocks Set", "Remote Control Car", "Puzzle 1000pc", "Board Game Classic",
        "Action Figure", "Soft Toy Bear", "Drone Mini", "Art & Craft Kit", "Doll House",
        "Educational Tablet Toy", "Card Game Pack", "Water Gun", "Kite Set", "Musical Toy",
        "Building Robot Kit",
    ],
}

_new_rows: list[tuple] = []
_next_id = 21
for category, names in _NEW_PRODUCT_NAMES.items():
    lo, hi = _CATEGORY_COST_RANGE[category]
    gst = _CATEGORY_GST[category]
    for name in names:
        cost   = round(random.uniform(lo, hi), 0)
        margin = round(random.uniform(*_CATEGORY_MARGIN_RANGE), 1)
        _new_rows.append((f"PRD{_next_id:03d}", name, category, cost, margin, gst))
        _next_id += 1

all_products_data = products_data + _new_rows

suppliers = ["VendorPrime","SupplyHub India","TradeMart Co","GlobalSource","QuickShip Ltd"]

rows = []
for pid, name, cat, base, margin, gst in all_products_data:
    selling = round(base * (1 + margin/100) * (1 + gst/100), 2)
    mrp     = round(selling * random.uniform(1.08, 1.18), 2)
    rows.append({
        "product_id": pid, "product_name": name, "category": cat,
        "base_cost": base, "margin_pct": margin, "tax_pct": gst,
        "selling_price": selling, "mrp": mrp,
        "stock_quantity": random.randint(30, 400),
        "reorder_level":  random.randint(10, 50),
        "supplier":    random.choice(suppliers),
        "launch_date": (datetime(2023,1,1) + timedelta(days=random.randint(0,900))).strftime("%Y-%m-%d"),
    })
df_products = pd.DataFrame(rows)

print(f"Products: {len(df_products)} across {df_products['category'].nunique()} categories")


# ── Customers: original 10, UNCHANGED ────────────────────────────────────────

customers_data = [
    ("CUST001","Arjun Mehta",  "arjun@email.com", "9876543210","Mumbai",   "Gold",    "2024-03-15"),
    ("CUST002","Priya Patel",  "priya@email.com", "9876543211","Delhi",    "Silver",  "2024-05-20"),
    ("CUST003","Vikram Singh", "vikram@email.com","9876543212","Bangalore","Platinum","2024-01-10"),
    ("CUST004","Deepa Nair",   "deepa@email.com", "9876543213","Chennai",  "Bronze",  "2024-08-01"),
    ("CUST005","Rahul Gupta",  "rahul@email.com", "9876543214","Hyderabad","Gold",    "2024-04-12"),
    ("CUST006","Ananya Sharma","ananya@email.com","9876543215","Pune",     "Silver",  "2024-06-25"),
    ("CUST007","Karthik Rajan","karthik@email.com","9876543216","Chennai", "Bronze",  "2024-09-05"),
    ("CUST008","Sneha Reddy",  "sneha@email.com", "9876543217","Hyderabad","Gold",    "2024-02-18"),
    ("CUST009","Amit Joshi",   "amit@email.com",  "9876543218","Mumbai",   "Bronze",  "2024-07-30"),
    ("CUST010","Meera Iyer",   "meera@email.com", "9876543219","Bangalore","Silver",  "2024-05-05"),
]

# ── New customers: 240 more, generated ───────────────────────────────────────
_FIRST_NAMES = [
    "Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Reyansh","Krishna","Ishaan","Rohan",
    "Kabir","Aryan","Dhruv","Kartik","Rudra","Yash","Ansh","Devansh","Shaurya","Atharv",
    "Ananya","Diya","Saanvi","Aadhya","Myra","Anika","Ira","Pari","Riya","Siya",
    "Aisha","Kavya","Navya","Sara","Tara","Zara","Meera","Nisha","Pooja","Ritu",
    "Rakesh","Suresh","Manoj","Vijay","Sanjay","Ramesh","Dinesh","Naresh","Ajay","Vinay",
    "Lakshmi","Kavita","Sunita","Anita","Rekha","Geeta","Seema","Neha","Shreya","Divya",
]
_LAST_NAMES = [
    "Mehta","Patel","Singh","Nair","Gupta","Sharma","Rajan","Reddy","Joshi","Iyer",
    "Kumar","Verma","Yadav","Chauhan","Malhotra","Kapoor","Bose","Das","Rao","Pillai",
    "Agarwal","Jain","Bhatt","Trivedi","Shetty","Nayak","Menon","Pandey","Mishra","Saxena",
]
_CITIES = [
    "Mumbai","Delhi","Bangalore","Chennai","Hyderabad","Pune","Kolkata","Ahmedabad",
    "Jaipur","Lucknow","Surat","Indore","Nagpur","Bhopal","Coimbatore",
]
_TIER_WEIGHTS = [("Bronze", 0.40), ("Silver", 0.32), ("Gold", 0.20), ("Platinum", 0.08)]

_new_customers: list[tuple] = []
_seen_names = {f"{f} {l}" for f, l in [(c[1].split()[0], c[1].split()[1]) for c in customers_data]}
_cid = 11
while len(_new_customers) < 240:
    fname = random.choice(_FIRST_NAMES)
    lname = random.choice(_LAST_NAMES)
    full  = f"{fname} {lname}"
    if full in _seen_names:
        continue
    _seen_names.add(full)
    tier   = random.choices([t for t, _ in _TIER_WEIGHTS], weights=[w for _, w in _TIER_WEIGHTS])[0]
    city   = random.choice(_CITIES)
    email  = f"{fname.lower()}.{lname.lower()}{_cid}@email.com"
    phone  = f"9{random.randint(100000000, 999999999)}"
    join   = (datetime(2023,1,1) + timedelta(days=random.randint(0, 900))).strftime("%Y-%m-%d")
    _new_customers.append((f"CUST{_cid:03d}", full, email, phone, city, tier, join))
    _cid += 1

all_customers_data = customers_data + _new_customers
print(f"Customers: {len(all_customers_data)}")

CHURN_CUSTOMERS = {"CUST004","CUST007","CUST009"}
CHURN_FROM = "2026-05"   # last ~3 months of the data window

# Each customer gets a persistent "engagement weight" that drives how often
# they're picked for a transaction across the WHOLE window -- heavy shoppers
# stay heavy, light shoppers stay light, all 3.5 years. Without this, every
# transaction is an independent uniform coin-flip across all customers, so a
# customer's total_spend/order_count/days_inactive end up uncorrelated with
# each other -- which is exactly why the churn classifier could only hit 41%
# (barely above chance) once label leakage was removed: there was no real
# behavioral signal in the data to learn from, only noise.
all_customer_ids = [c[0] for c in all_customers_data]
_engagement = {c[0]: round(np.random.lognormal(mean=0.0, sigma=1.0), 3) for c in all_customers_data}

# ~30% of customers are "decliners", drawn preferentially from the lower half
# of engagement -- i.e. customers who were ALREADY light/occasional shoppers
# are the ones most likely to go quiet in the recent months. This is what
# gives the churn model a genuine, learnable signal: low total_spend/
# order_count/unique_products (visible to the model) now actually correlates
# with high days_inactive (part of the label), instead of days_inactive being
# independent noise.
_decline_pool = sorted(all_customer_ids, key=lambda cid: _engagement[cid])[: int(len(all_customer_ids) * 0.5)]
DECLINE_CUSTOMERS = set(random.sample(_decline_pool, int(len(all_customer_ids) * 0.30)))


# ── 42-month window (2023-01 through 2026-06), seasonality repeats by
#    calendar month across all 3 years -- this is what gives a forecasting
#    model real year-over-year signal instead of one unrepeated trend. ────────

# 43 entries, not 42: the last one is the CURRENT (partial) month, so recent
# transactions land close to TODAY. Without this, the window would end a
# full month early and EVERY customer -- not just the intentional churners --
# would show an inflated days_inactive purely from the calendar gap between
# "last full month in the data" and "today" (a real bug this caught: 8 of 10
# original customers showed up as churned before this fix, not the intended 3).
months = []
start = datetime(2023, 1, 1)
for i in range(43):
    y = start.year + (start.month - 1 + i) // 12
    m = (start.month - 1 + i) % 12 + 1
    months.append(f"{y}-{m:02d}")
CURRENT_MONTH = months[-1]
DECLINE_FROM  = months[-4]   # last ~4 months of the window

# Multiplier keyed by CALENDAR month (01-12), not by absolute month -- repeats
# every year. Oct/Nov bump (festival season, Diwali), summer dip, etc.
_SEASONAL_BY_CAL_MONTH = {
    "01": 1.00, "02": 0.95, "03": 1.10, "04": 1.00, "05": 0.90, "06": 0.95,
    "07": 1.00, "08": 1.05, "09": 1.10, "10": 1.55, "11": 1.85, "12": 1.25,
}
# Mild year-over-year growth on top of seasonality, so the series has a trend
# component too, not just repeating seasonality.
_YEAR_GROWTH = {2023: 1.00, 2024: 1.08, 2025: 1.15, 2026: 1.22}

BASE_TXNS_PER_MONTH = 115   # tuned so 42 months lands near ~5,000 transactions

all_products_rows = rows  # already built above, list of dicts

txns = []
tid  = 1
_txn_weights = [_engagement[c[0]] for c in all_customers_data]
for month_str in months:
    year, mon = map(int, month_str.split("-"))
    cal_mult  = _SEASONAL_BY_CAL_MONTH[f"{mon:02d}"]
    yr_mult   = _YEAR_GROWTH[year]
    count     = int(BASE_TXNS_PER_MONTH * cal_mult * yr_mult)
    days_in_month = 28 if mon == 2 else (30 if mon in (4,6,9,11) else 31)

    # The current (partial) month only has days up to "yesterday" -- both the
    # day range AND the transaction count are scaled down proportionally, so
    # this month isn't artificially as dense as a completed one.
    if month_str == CURRENT_MONTH:
        days_in_month = max(1, TODAY.day - 1)
        count = int(count * days_in_month / 30)

    for _ in range(count):
        cust = random.choices(all_customers_data, weights=_txn_weights, k=1)[0]
        cid  = cust[0]
        if cid in CHURN_CUSTOMERS and month_str >= CHURN_FROM:
            if random.random() < 0.80:
                continue
        if cid in DECLINE_CUSTOMERS and month_str >= DECLINE_FROM:
            if random.random() < 0.85:
                continue
        prod       = random.choice(all_products_rows)
        qty        = random.choices([1,1,1,2,2,3], k=1)[0]
        unit_price = prod["selling_price"]
        total      = round(unit_price * qty, 2)
        disc       = random.choice([0,0,0,0,5,8,10])
        final      = round(total * (1 - disc/100), 2)
        day        = random.randint(1, days_in_month)
        date_str   = f"{year}-{mon:02d}-{day:02d}"
        r2         = random.random()
        status     = "Completed" if r2 < 0.90 else ("Returned" if r2 < 0.97 else "Cancelled")
        txns.append({
            "transaction_id": f"TXN{tid:04d}",
            "customer_id": cid, "customer_name": cust[1],
            "product_id": prod["product_id"],
            "product_name": prod["product_name"],
            "category": prod["category"],
            "quantity": qty, "unit_price": unit_price,
            "total_amount": total, "discount_pct": disc,
            "final_amount": final,
            "payment_method": random.choice(["UPI","Credit Card","Debit Card","Net Banking","Cash"]),
            "date": date_str, "month": month_str, "status": status,
        })
        tid += 1
df_txns = pd.DataFrame(txns)
print(f"Transactions: {len(df_txns)} across {len(months)} months ({months[0]} to {months[-1]})")


# ── Customer aggregates + credit fields ──────────────────────────────────────

_CREDIT_LIMIT_BY_TIER = {"Bronze": 15000, "Silver": 35000, "Gold": 75000, "Platinum": 150000}

completed = df_txns[df_txns["status"] == "Completed"]
cust_rows = []
for c in all_customers_data:
    cid = c[0]
    ct  = completed[completed["customer_id"] == cid]
    total_spend  = round(ct["final_amount"].sum(), 2)
    total_orders = len(ct)
    last_purchase = ct["date"].max() if len(ct) > 0 else ""
    days_inactive = (TODAY - datetime.strptime(last_purchase, "%Y-%m-%d")).days if last_purchase else 999

    tier = c[5]
    base_limit = _CREDIT_LIMIT_BY_TIER[tier]
    credit_limit = round(base_limit * random.uniform(0.85, 1.15), 2)
    # ~15% of customers run a high outstanding balance (credit-risk flavor);
    # the rest carry a light, healthy balance.
    if random.random() < 0.15:
        outstanding_balance = round(credit_limit * random.uniform(0.6, 0.95), 2)
    else:
        outstanding_balance = round(credit_limit * random.uniform(0.0, 0.35), 2)

    cust_rows.append({
        "customer_id": c[0], "customer_name": c[1], "email": c[2],
        "phone": c[3], "city": c[4], "tier": tier, "join_date": c[6],
        "last_purchase": last_purchase,
        "total_spend": total_spend, "total_orders": total_orders,
        "days_inactive": days_inactive,
        "credit_limit": credit_limit, "outstanding_balance": outstanding_balance,
    })
df_customers = pd.DataFrame(cust_rows)

print(f"\nOriginal 10 customers (unchanged identity fields):")
for _, r in df_customers.head(10).iterrows():
    churn = " <- CHURN" if r["days_inactive"] > 45 else ""
    print(f"  {r['customer_id']} | {r['customer_name']:<16} | {r['tier']:<9} | {r['days_inactive']}d{churn}")


# ── Monthly sales (all months) ────────────────────────────────────────────────

monthly_rows = []
for m in months:
    mc = completed[completed["month"] == m]
    if len(mc) == 0:
        continue
    monthly_rows.append({
        "month": m,
        "total_revenue":   round(mc["final_amount"].sum(), 2),
        "total_orders":    len(mc),
        "avg_order_value": round(mc["final_amount"].mean(), 2),
        "unique_customers": mc["customer_id"].nunique(),
        "returns_count":   len(df_txns[(df_txns["month"]==m) & (df_txns["status"]=="Returned")]),
        "top_category":    mc.groupby("category")["final_amount"].sum().idxmax(),
    })
df_monthly = pd.DataFrame(monthly_rows)
print(f"\nMonthly sales: {len(df_monthly)} rows")


# ── Company rates (extended with the 3 new categories' GST) ────────────────

rates = [
    ("gst_electronics",18,"GST rate for Electronics"),
    ("gst_clothing",5,"GST rate for Clothing"),
    ("gst_home_kitchen",12,"GST rate for Home & Kitchen"),
    ("gst_beauty",18,"GST rate for Beauty"),
    ("gst_sports",12,"GST rate for Sports"),
    ("gst_groceries",5,"GST rate for Groceries"),
    ("gst_books_stationery",5,"GST rate for Books & Stationery"),
    ("gst_toys_games",12,"GST rate for Toys & Games"),
    ("bulk_discount_threshold",50000,"Order total above which bulk discount applies"),
    ("bulk_discount_pct",8,"Bulk discount percentage"),
    ("loyalty_discount_silver",5,"Silver tier loyalty discount"),
    ("loyalty_discount_gold",8,"Gold tier loyalty discount"),
    ("loyalty_discount_platinum",12,"Platinum tier loyalty discount"),
    ("seasonal_markup_pct",5,"Festival season markup"),
    ("employee_discount_pct",20,"Employee discount"),
    ("shipping_free_above",999,"Free shipping threshold"),
    ("shipping_flat_rate",49,"Flat shipping rate"),
    ("return_window_days",7,"Return window in days"),
]
df_rates = pd.DataFrame(rates, columns=["rate_name","rate_value","description"])


# ── Loans (seed data for Phase 2's Credit agent) ─────────────────────────────
# A handful of realistic loan records so the SQL agent has something real to
# query immediately, and the Credit agent's history-aware tools have example
# rows to work with. The Credit agent will create MORE of these at runtime;
# this is just a believable starting point, not the primary data source.

def _emi(principal: float, annual_rate: float, months: int) -> float:
    r = annual_rate / 12 / 100
    if r == 0:
        return round(principal / months, 2)
    return round(principal * r * (1 + r) ** months / ((1 + r) ** months - 1), 2)

_LOAN_RATE_BY_TIER = {"Bronze": 18.0, "Silver": 15.0, "Gold": 12.0, "Platinum": 10.0}
_customers_by_id = {c[0]: c for c in all_customers_data}

loan_rows = []
sample_customers = random.sample(all_customer_ids, 30)
for i, cid in enumerate(sample_customers, start=1):
    cust = _customers_by_id[cid]
    tier = cust[5]
    principal = round(random.choice([10000, 15000, 20000, 30000, 50000, 75000]), 2)
    rate      = _LOAN_RATE_BY_TIER[tier]
    tenure    = random.choice([6, 12, 18, 24])
    emi       = _emi(principal, rate, tenure)
    status    = random.choices(["Approved","Pending","Rejected"], weights=[0.7,0.2,0.1])[0]
    created   = (datetime(2025,1,1) + timedelta(days=random.randint(0, 570))).strftime("%Y-%m-%d")
    loan_rows.append({
        "loan_id": f"LOAN{i:04d}", "customer_id": cid,
        "principal": principal, "interest_rate": rate, "tenure_months": tenure,
        "monthly_emi": emi, "status": status, "created_at": created,
    })
df_loans = pd.DataFrame(loan_rows)
print(f"Loans (seed): {len(df_loans)} rows")


with pd.ExcelWriter("data/company_data.xlsx", engine="openpyxl") as writer:
    df_products.to_excel(writer, sheet_name="products",      index=False)
    df_txns.to_excel(    writer, sheet_name="transactions",  index=False)
    df_monthly.to_excel( writer, sheet_name="monthly_sales", index=False)
    df_customers.to_excel(writer, sheet_name="customers",    index=False)
    df_rates.to_excel(   writer, sheet_name="company_rates", index=False)
    df_loans.to_excel(   writer, sheet_name="loans",         index=False)

print(f"\nSaved: data/company_data.xlsx")
print(f"Products: {len(df_products)} | Transactions: {len(df_txns)} | "
      f"Customers: {len(df_customers)} | Months: {len(df_monthly)} | Loans: {len(df_loans)}")
