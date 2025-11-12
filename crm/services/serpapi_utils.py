import os, requests

def get_serpapi_prices(query, country="fr", language="fr"):
    """
    Fetch product prices from Google Shopping via SerpApi.
    Returns sorted list of offers with title, source, price, currency, and link.
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise ValueError("SERPAPI_KEY not set in environment variables.")

    params = {
        "engine": "google_shopping",
        "q": query,
        "gl": country.lower(),
        "hl": language.lower(),
        "api_key": api_key,
    }

    r = requests.get("https://serpapi.com/search", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    offers = []
    for item in data.get("shopping_results", []):
        price = item.get("extracted_price")
        if not price:
            continue
        offers.append({
            "title": item.get("title"),
            "source": item.get("source") or "Unknown",
            "price": float(price),
            "currency": "EUR",
            "link": item.get("link"),
        })
    return sorted(offers, key=lambda x: x["price"])
