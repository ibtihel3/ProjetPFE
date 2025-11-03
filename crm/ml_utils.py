import joblib
import os
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

from .models import Product

# === Load Gradient Boosting model ===
MODEL_PATH1 = os.path.join("model", "gradient_boosting_model.pkl")
model1 = joblib.load(MODEL_PATH1)
# === Load LightGBM model ===
MODEL_PATH2 = os.path.join("model", "xgb_model.pkl")
model2 = joblib.load(MODEL_PATH2)


# === Encoders (must match training setup) ===
le_category = LabelEncoder()
le_category.classes_ = np.array(["beauty", "clothing", "electronics", "home"])
le_status = LabelEncoder()
le_status.classes_ = np.array(["active", "archived", "draft", "inactive"])

# === Sentiment ===
def get_sentiment(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0

# === Preprocessing Products ===
def preprocess(product):
    review_sentiment = get_sentiment(product.about_product or "")
    category_encoded = le_category.transform([product.category.lower()])[0] if product.category else 0
    status_encoded = le_status.transform([product.status.lower()])[0] if product.status else 0

    return [
        product.actual_price,
        product.discounted_price,
        category_encoded,
        product.rating_count or 0,
        review_sentiment,
        product.stock or 0,
        product.rating or 0,
    ]

# === Preprocessing Clients ===
def preprocess_client(client):
    """Extracts all 16 features in the same order as the LightGBM model was trained."""
    return [
        client.total_spent,
        client.avg_order_value,
        client.total_orders,
        client.total_items,
        client.avg_discount,
        client.avg_review_rating,
        client.avg_seller_rating,
        client.avg_delivery_days,
        client.total_returns,
        client.return_ratio,
        client.total_previous_returns,
        1 if client.is_prime_member else 0,
        client.customer_tenure_days,
        client.recency_days,
        client.tenure_days,
        client.frequency,
    ]

# === Predict discount ===
def predict_discount(product):
    X1 = [preprocess(product)]
    return float(model1.predict(X1)[0])

# === Predict clv ===
def predict_clv(clients):
    X = np.array([preprocess_client(clients)])
    return float(model2.predict(X)[0])

# === Notifications ====
def generate_notifications(df):
    notifications = []

    for _, row in df.iterrows():
        if row['recency_days'] > 120 and row['is_active'] == 0:
            notifications.append({
                'customer_id': row['customer_id'],
                'name': row['name'],
                'segment': row['segment'],
                'type': 'Reactivation',
                'priority': 'High',
                'message': f"{row['name']} has been inactive for over 4 months. Send a re-engagement offer."
            })
        elif row['clv'] > 1500 and row['frequency'] > 0.4:
            notifications.append({
                'customer_id': row['customer_id'],
                'name': row['name'],
                'segment': row['segment'],
                'type': 'Reward',
                'priority': 'Medium',
                'message': f"{row['name']} is a loyal VIP. Offer a loyalty coupon."
            })
        elif row['tenure_days'] < 30:
            notifications.append({
                'customer_id': row['customer_id'],
                'name': row['name'],
                'segment': row['segment'],
                'type': 'Welcome',
                'priority': 'Low',
                'message': f"Welcome {row['name']}! Send onboarding email."
            })
    return pd.DataFrame(notifications)