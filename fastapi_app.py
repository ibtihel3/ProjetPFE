from fastapi import FastAPI
import pandas as pd
import numpy as np

app = FastAPI(title="BoWise Marketing Notification API")

# ==========================================================
# 1️⃣ Load your CSV (uploaded client dataset)
# ==========================================================
# Adjust the path to match your project structure if needed
CSV_PATH = "data/dash_plotly/clients_dashboard_ready.csv"
df_clients = pd.read_csv(CSV_PATH)

# --- Preview ---
print(f"✅ Loaded {len(df_clients)} clients from {CSV_PATH}")

# ==========================================================
# 2️⃣ Define churn / risk detection rules
# ==========================================================
INACTIVE_DAYS = 60
LOW_SPEND_QUANTILE = 0.3
LOW_ORDER_THRESHOLD = 2

# Compute thresholds dynamically
spend_threshold = df_clients["total_spent"].quantile(LOW_SPEND_QUANTILE)

# ==========================================================
# 3️⃣ Marketing strategy rules
# ==========================================================
def suggest_strategies(client):
    strategies = []

    # --- Recency-based ---
    if client.get("recency_days", 0) > INACTIVE_DAYS:
        strategies.append("Send a reactivation email or WhatsApp message with a personalized discount code.")

    # --- Spending-based ---
    if client.get("total_spent", 0) < spend_threshold:
        strategies.append("Offer a loyalty reward to encourage additional purchases.")

    # --- Orders-based ---
    if client.get("avg_order_value", 0) <= LOW_ORDER_THRESHOLD:
        strategies.append("Provide a 'buy more, save more' deal to increase order frequency.")

    # --- Loyalty status ---
    if str(client.get("loyalty_status", "")).lower() not in ["vip", "premium"]:
        strategies.append("Promote loyalty program benefits or early access deals.")

    # --- Income segment customization ---
    if str(client.get("income_segment", "")).lower() in ["high", "upper"]:
        strategies.append("Highlight premium collections or limited editions.")
    elif str(client.get("income_segment", "")).lower() in ["low", "budget"]:
        strategies.append("Emphasize affordable bundles and free delivery offers.")

    # --- Engagement status ---
    if not client.get("is_active", True):
        strategies.append("Send a 'we miss you' campaign to bring the client back.")

    return strategies


# ==========================================================
# 4️⃣ API endpoint — marketing notifications
# ==========================================================
@app.get("/notify/at_risk_clients")
def get_at_risk_clients():
    """
    Detect clients at churn risk and suggest personalized marketing strategies.
    """
    at_risk = df_clients[
        (df_clients["recency_days"] > INACTIVE_DAYS)
        | (df_clients["total_spent"] < spend_threshold)
        | (df_clients["avg_order_value"] <= LOW_ORDER_THRESHOLD)
    ]

    notifications = []
    for _, client in at_risk.iterrows():
        recs = suggest_strategies(client)
        risk_level = "High" if len(recs) >= 3 else "Moderate"

        notifications.append({
            "client_id": client.get("customer_id"),
            "name": client.get("name"),
            "risk_level": risk_level,
            "recommendations": recs
        })

    return {"count": len(notifications), "notifications": notifications}


# ==========================================================
# 5️⃣ Root route (simple health check)
# ==========================================================
@app.get("/")
def root():
    return {"message": "BoWise Notification API is running 🚀"}
