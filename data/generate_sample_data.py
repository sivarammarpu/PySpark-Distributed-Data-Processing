"""
data/generate_sample_data.py
=============================
Generates realistic sample datasets for local development and testing.

Outputs
-------
- data/raw/sales_data.csv   : 50 000 sales transaction rows
- data/raw/events.json      : 10 000 user-event rows (newline-delimited JSON)
"""

import csv
import json
import os
import random
from datetime import date, timedelta

# ── Reproducibility ────────────────────────────────────────────────────────
random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────
REGIONS       = ["NORTH", "SOUTH", "EAST", "WEST", "CENTRAL"]
PRODUCTS      = [
    "Laptop", "Monitor", "Keyboard", "Mouse", "Headset",
    "Webcam", "Desk Chair", "Standing Desk", "USB Hub", "Printer",
]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "Digital Wallet", "Cash", "Bank Transfer"]
DEVICES         = ["mobile", "desktop", "tablet"]
ACTIONS         = ["view", "click", "purchase", "add_to_cart", "search", "logout"]
COUNTRIES       = ["US", "UK", "DE", "IN", "CA", "AU", "FR", "JP", "BR", "SG"]

START_DATE = date(2022, 1, 1)
END_DATE   = date(2024, 12, 31)
DATE_RANGE = (END_DATE - START_DATE).days


def random_date() -> str:
    return (START_DATE + timedelta(days=random.randint(0, DATE_RANGE))).isoformat()


# ════════════════════════════════════════════════════════════════════════════
# CSV — Sales Transactions
# ════════════════════════════════════════════════════════════════════════════

def generate_sales_csv(path: str, n_rows: int = 50_000) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    headers = [
        "order_id", "order_date", "region", "customer_id",
        "product", "quantity", "unit_price", "discount",
        "payment_method", "customer_age",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for i in range(1, n_rows + 1):
            # Introduce ~0.3 % corrupt records to test PERMISSIVE mode
            if random.random() < 0.003:
                f.write(f"ORD{i:07d},INVALID_DATE,{random.choice(REGIONS)},,BAD_ROW\n")
                continue

            writer.writerow({
                "order_id":       f"ORD{i:07d}",
                "order_date":     random_date(),
                "region":         random.choice(REGIONS),
                "customer_id":    f"CUST{random.randint(1, 5000):05d}",
                "product":        random.choice(PRODUCTS),
                "quantity":       random.randint(1, 20),
                "unit_price":     round(random.uniform(9.99, 2999.99), 2),
                "discount":       round(random.choice([0, 0.05, 0.10, 0.15, 0.20, 0.25]), 2),
                "payment_method": random.choice(PAYMENT_METHODS),
                "customer_age":   random.randint(18, 70),
            })

    print(f"[✔] Sales CSV  → {path}  ({n_rows:,} rows)")


# ════════════════════════════════════════════════════════════════════════════
# JSON — User Events
# ════════════════════════════════════════════════════════════════════════════

def generate_events_json(path: str, n_rows: int = 10_000) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for i in range(1, n_rows + 1):
            event = {
                "event_id":   f"EVT{i:08d}",
                "timestamp":  f"{random_date()}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}",
                "user_id":    f"USR{random.randint(1, 2000):05d}",
                "session_id": f"SES{random.randint(1, 10000):06d}",
                "action":     random.choice(ACTIONS),
                "device":     random.choice(DEVICES),
                "country":    random.choice(COUNTRIES),
                "page":       random.choice(["/home", "/product", "/cart", "/checkout", "/profile"]),
                "duration_s": random.randint(1, 600),
            }
            f.write(json.dumps(event) + "\n")

    print(f"[✔] Events JSON → {path}  ({n_rows:,} rows)")


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    generate_sales_csv(os.path.join(base, "raw", "sales_data.csv"), n_rows=50_000)
    generate_events_json(os.path.join(base, "raw", "events.json"),  n_rows=10_000)
    print("\nSample data generation complete.")
    print("Run:  python main.py  to start the pipeline.")
