import os
import csv
from builtins import float
from crm.models import Client
from datetime import datetime
import re

from datetime import datetime
import pandas as pd

def parse_date(date_str):
    """Parse datetime strings like '2018-12-11 00:24:04' or return None."""
    if not date_str or str(date_str).strip() in ["", "nan", "NaN", "None"]:
        return None

    s = str(date_str).strip()

    # Try the most common datetime pattern first (your CSV format)
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").date()
    except ValueError:
        pass

    # Try a few fallback patterns just in case
    for fmt in ("%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue

    # Last resort: pandas auto-detection
    parsed = pd.to_datetime(s, errors="coerce")
    if pd.notnull(parsed):
        return parsed.date()

    return None


def load_clients_once(csv_path):
    """Load clients only if the table is empty."""
    if Client.objects.exists():
        print("⚠️ Clients already exist — skipping load.")
        return

    print("📦 Loading clients from CSV...")

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        clients = []
        for row in reader:
            client = Client(
                customer_id=row.get('customer_id') or row.get('id'),
                name=row.get('name'),
                gender=row.get('gender'),
                region=row.get('region'),
                age=int(float(row.get('age', 0))) if row.get('age') else None,
                source=row.get('source'),
                email=row.get('email'),
                phone=row.get('phone'),
                income_segment=row.get('income_segment'),
                loyalty_status=row.get('loyalty_status'),
                is_active=row.get('is_active', 'True').lower() in ['true', '1', 'yes'],
                registration_date=parse_date(row.get('registration_date')),
                total_spent=float(row.get('total_spent', 0)),
                avg_order_value=float(row.get('avg_order_value', 0)),
                total_orders=int(float(row.get('total_orders', 0))),
                total_items=int(float(row.get('total_items', 0))),
                avg_discount=float(row.get('avg_discount', 0)),
                avg_review_rating=float(row.get('avg_review_rating', 0)),
                avg_seller_rating=float(row.get('avg_seller_rating', 0)),
                avg_delivery_days=float(row.get('avg_delivery_days', 0)),
                total_returns=int(float(row.get('total_returns', 0))),
                return_ratio=float(row.get('return_ratio', 0)),
                total_previous_returns=int(float(row.get('total_previous_returns', 0))),
                is_prime_member=row.get('is_prime_member', 'False').lower() in ['true', '1', 'yes'],
                customer_tenure_days=float(row.get('customer_tenure_days', 0)),
                last_order_date=parse_date(row.get('last_order_date')),
                first_order_date=parse_date(row.get('first_order_date')),
                recency_days=float(row.get('recency_days', 0)),
                tenure_days=float(row.get('tenure_days', 0)),
                frequency=float(row.get('frequency', 0)),
            )
            clients.append(client)

        Client.objects.bulk_create(clients, ignore_conflicts=True)
        print(f"✅ Loaded {len(clients)} clients successfully.")