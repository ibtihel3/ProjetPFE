# fastapi_app.py
import os
import django
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# ==========================================================
# Django ORM setup
# ==========================================================
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bowise_crm.settings")
django.setup()

#import Django model after setup()
from crm.models import Client

app = FastAPI(title="BoWise Marketing Notification API")

# ==========================================================
# Risk detection & marketing strategies
# ==========================================================
INACTIVE_DAYS = 90
LOW_SPEND_QUANTILE = 0.3
LOW_ORDER_THRESHOLD = 1

def suggest_strategies(client, spend_threshold):
    """Return personalized marketing suggestions for a client."""
    strategies = []

    # --- Recency-based ---
    if client.get("recency_days", 0) > INACTIVE_DAYS:
        strategies.append("Send a reactivation email or WhatsApp message with a personalized discount code.")

    # --- Spending-based ---
    if client.get("total_spent", 0) < spend_threshold:
        strategies.append("Offer a loyalty reward to encourage additional purchases.")

    # --- Orders-based ---
    if client.get("total_orders", 0) <= LOW_ORDER_THRESHOLD:
        strategies.append("Provide a 'buy more, save more' deal to increase order frequency.")

    # --- Loyalty status ---
    if str(client.get("loyalty_status", "")).lower() not in ["vip", "premium"]:
        strategies.append("Promote loyalty program benefits or early access deals.")

    # --- Income segment customization ---
    income = str(client.get("income_segment", "")).lower()
    if income in ["high", "upper"]:
        strategies.append("Highlight premium collections or limited editions.")
    elif income in ["low", "budget"]:
        strategies.append("Emphasize affordable bundles and free delivery offers.")

    # --- Engagement status ---
    if not client.get("is_active", True):
        strategies.append("Send a 'we miss you' campaign to bring the client back.")

    return strategies


# ==========================================================
# API endpoint — marketing notifications
# ==========================================================
@app.get("/notify/at_risk_clients")
def get_at_risk_clients():
    try:
        # --- Load data directly from Django ORM ---
        clients = list(Client.objects.all().values(
            "customer_id", "name", "region", "income_segment",
            "loyalty_status", "total_spent", "total_orders",
            "avg_order_value", "is_active"
        ))

        if not clients:
            return {"count": 0, "notifications": []}

        df_clients = pd.DataFrame(clients)

        # Handle missing numeric data
        for col in ["total_spent", "total_orders", "avg_order_value"]:
            df_clients[col] = pd.to_numeric(df_clients[col], errors="coerce").fillna(0)

        # --- Compute spend threshold dynamically ---
        spend_threshold = df_clients["total_spent"].quantile(LOW_SPEND_QUANTILE)

        # --- Identify at-risk clients ---
        # You can refine this rule as needed
        at_risk = df_clients[
            (df_clients["total_orders"] <= LOW_ORDER_THRESHOLD)
            | (df_clients["total_spent"] < spend_threshold)
            | (df_clients["is_active"] == False)
        ]

        # --- Build structured notifications ---
        notifications = []
        for _, client in at_risk.iterrows():
            strategies = suggest_strategies(client, spend_threshold)
            notifications.append({
                "client_id": client.get("customer_id"),
                "name": client.get("name"),
                "region": client.get("region"),
                "risk_level": "High" if client.get("total_orders") <= 2 else "Moderate",
                "total_orders": int(client.get("total_orders", 0)),
                "total_spent": float(client.get("total_spent", 0)),
                "recommendations": strategies
})


        return {"count": len(notifications), "notifications": notifications}

    except Exception as e:
        # Force a valid JSON response even on error
        return JSONResponse(
            content={"error": str(e), "notifications": []},
            status_code=500
        )


# ==========================================================
# Root route (simple health check)
# ==========================================================
@app.get("/")
def root():
    return {"message": "BoWise Notification API is running 🚀"}


@app.get("/test")
def test_api():
    return {"status": "ok", "message": "FastAPI is mounted correctly 🚀"}
