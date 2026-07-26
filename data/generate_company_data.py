import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

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

suppliers = ["VendorPrime","SupplyHub India","TradeMart Co","GlobalSource","QuickShip Ltd"]

rows = []
for pid, name, cat, base, margin, gst in products_data:
    selling = round(base * (1 + margin/100) * (1 + gst/100), 2)
    mrp     = round(selling * random.uniform(1.08, 1.18), 2)
    rows.append({
        "product_id": pid, "product_name": name, "category": cat,
        "base_cost": base, "margin_pct": margin, "tax_pct": gst,
        "selling_price": selling, "mrp": mrp,
        "stock_quantity": random.randint(30, 400),
        "reorder_level":  random.randint(10, 50),
        "supplier":    random.choice(suppliers),
        "launch_date": (datetime(2024,1,1) + timedelta(days=random.randint(0,400))).strftime("%Y-%m-%d"),
    })
df_products = pd.DataFrame(rows)

print("=== PRODUCT VERIFICATION ===")
for _, r in df_products.iterrows():
    print(f"{r['product_id']} | {r['product_name']:<22} | Rs.{r['selling_price']:>10,.2f}")

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

CHURN_CUSTOMERS = {"CUST004","CUST007","CUST009"}
CHURN_FROM = "2025-11"

months = []
start = datetime(2025, 1, 1)
for i in range(18):
    m = start + timedelta(days=i*30)
    months.append(m.strftime("%Y-%m"))

seasonal = {
    "2025-01":1.0,"2025-02":0.95,"2025-03":1.15,
    "2025-04":1.0,"2025-05":0.9, "2025-06":0.95,
    "2025-07":1.0,"2025-08":1.05,"2025-09":1.1,
    "2025-10":1.60,"2025-11":1.90,"2025-12":1.3,
    "2026-01":1.05,"2026-02":1.0, "2026-03":1.2,
    "2026-04":1.1, "2026-05":1.15,"2026-06":1.2,
}

txns = []
tid  = 1
for month_str in months:
    year, mon = map(int, month_str.split("-"))
    mult  = seasonal.get(month_str, 1.0)
    count = int(28 * mult)
    for _ in range(count):
        cust = random.choice(customers_data)
        cid  = cust[0]
        if cid in CHURN_CUSTOMERS and month_str >= CHURN_FROM:
            if random.random() < 0.80:
                continue
        prod      = random.choice(rows)
        qty       = random.choices([1,1,1,2,2,3], k=1)[0]
        unit_price = prod["selling_price"]
        total     = round(unit_price * qty, 2)
        disc      = random.choice([0,0,0,0,5,8,10])
        final     = round(total * (1 - disc/100), 2)
        day       = random.randint(1, 28)
        date_str  = f"{year}-{mon:02d}-{day:02d}"
        r2        = random.random()
        status    = "Completed" if r2 < 0.90 else ("Returned" if r2 < 0.97 else "Cancelled")
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

completed = df_txns[df_txns["status"] == "Completed"]
cust_rows = []
for c in customers_data:
    cid = c[0]
    ct  = completed[completed["customer_id"] == cid]
    total_spend  = round(ct["final_amount"].sum(), 2)
    total_orders = len(ct)
    last_purchase = ct["date"].max() if len(ct) > 0 else ""
    days_inactive = (datetime(2026,7,3) - datetime.strptime(last_purchase, "%Y-%m-%d")).days if last_purchase else 999
    cust_rows.append({
        "customer_id": c[0], "customer_name": c[1], "email": c[2],
        "phone": c[3], "city": c[4], "tier": c[5], "join_date": c[6],
        "last_purchase": last_purchase,
        "total_spend": total_spend, "total_orders": total_orders,
        "days_inactive": days_inactive,
    })
df_customers = pd.DataFrame(cust_rows)

print("\n=== CUSTOMERS ===")
for _, r in df_customers.iterrows():
    churn = " <- CHURN" if r["days_inactive"] > 45 else ""
    print(f"{r['customer_id']} | {r['customer_name']:<16} | {r['tier']:<9} | {r['days_inactive']}d{churn}")

monthly_rows = []
seen_months  = set()
for m in months:
    if m in seen_months: continue
    seen_months.add(m)
    mc = completed[completed["month"] == m]
    if len(mc) == 0: continue
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

rates = [
    ("gst_electronics",18,"GST rate for Electronics"),
    ("gst_clothing",5,"GST rate for Clothing"),
    ("gst_home_kitchen",12,"GST rate for Home & Kitchen"),
    ("gst_beauty",18,"GST rate for Beauty"),
    ("gst_sports",12,"GST rate for Sports"),
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

with pd.ExcelWriter("data/company_data.xlsx", engine="openpyxl") as writer:
    df_products.to_excel(writer, sheet_name="products",      index=False)
    df_txns.to_excel(    writer, sheet_name="transactions",  index=False)
    df_monthly.to_excel( writer, sheet_name="monthly_sales", index=False)
    df_customers.to_excel(writer, sheet_name="customers",    index=False)
    df_rates.to_excel(   writer, sheet_name="company_rates", index=False)

print(f"\nSaved: data/company_data.xlsx")
print(f"Products: {len(df_products)} | Transactions: {len(df_txns)} | Customers: {len(df_customers)}")
