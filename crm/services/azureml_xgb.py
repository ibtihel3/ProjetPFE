import os, json, time, requests
from typing import Dict, Any, List, Optional
from django.conf import settings

def client_to_payload_dict(c) -> Dict[str, Any]:
    return {
        "total_spent": c.total_spent,
        "avg_order_value": c.avg_order_value,
        "total_orders": c.total_orders,
        "total_items": c.total_items,
        "avg_discount": c.avg_discount,
        "avg_review_rating": c.avg_review_rating,
        "avg_seller_rating": c.avg_seller_rating,
        "avg_delivery_days": c.avg_delivery_days,
        "total_returns": c.total_returns,
        "return_ratio": c.return_ratio,
        "total_previous_returns": c.total_previous_returns,
        "is_prime_member": bool(getattr(c, "is_prime_member", False)),
        "customer_tenure_days": c.customer_tenure_days,
        "recency_days": c.recency_days,
        "tenure_days": c.tenure_days,
        "frequency": c.frequency,
    }

def predict_clients_via_api(clients: List[Dict[str, Any]], timeout: float = 8.0, retries: int = 1) -> List[float]:
    url = getattr(settings, "AZURE_ML_XGB_URL", None)
    key = getattr(settings, "AZURE_ML_XGB_KEY", None)
    if not url or not key:
        raise RuntimeError("Missing AZURE_ML_XGB_URL or AZURE_ML_XGB_KEY")

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"clients": clients}

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=timeout)
            if r.status_code == 401:
                raise RuntimeError("Unauthorized (401) — check AZURE_ML_XGB_KEY")
            r.raise_for_status()
            out = r.json()
            if "error" in out:
                raise RuntimeError(f"Endpoint error: {out['error']}")
            preds = out.get("predictions")
            if preds is None:
                raise RuntimeError(f"Bad response (no 'predictions'): {out}")
            return preds
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
            else:
                raise last_err
