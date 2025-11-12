import os, json, time, requests
from typing import Dict, Any, List, Optional
from django.conf import settings

def product_to_payload_dict(p) -> Dict[str, Any]:
    """
    Build the JSON object expected by your product endpoint score.py.
    Keys match what the scoring script preprocesses:
      actual_price, discounted_price, category, rating_count,
      about_product, stock, rating, (optional) status
    """
    d = {
        "actual_price": getattr(p, "actual_price", 0.0),
        "discounted_price": getattr(p, "discounted_price", 0.0),
        "category": (getattr(p, "category", None) or ""),
        "rating_count": getattr(p, "rating_count", 0) or 0,
        "about_product": getattr(p, "about_product", "") or "",
        "stock": getattr(p, "stock", 0) or 0,
        "rating": getattr(p, "rating", 0.0) or 0.0,
    }
    # include status if your model/object has it
    if hasattr(p, "status"):
        d["status"] = getattr(p, "status") or ""
    return d

def predict_products_via_api(products: List[Dict[str, Any]], timeout: float = 8.0, retries: int = 1) -> List[float]:
    """
    products: list of dicts built with product_to_payload_dict
    returns: list of predictions (floats)
    """
    url = getattr(settings, "AZURE_ML_GB_PROD_URL", None)
    key = getattr(settings, "AZURE_ML_GB_PROD_KEY", None)
    if not url or not key:
        raise RuntimeError("Missing AZURE_ML_GB_PROD_URL or AZURE_ML_GB_PROD_KEY")

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"products": products}

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=timeout)
            if r.status_code == 401:
                raise RuntimeError("Unauthorized (401) — check AZURE_ML_GB_PROD_KEY")
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
