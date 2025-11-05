import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from bowise_crm import settings
from .models import Product, Client
from .forms import ProductForm, ClientForm
import pandas as pd
from .ml_utils import predict_discount
from .ml_utils import predict_clv
import io
import json
import pandas as pd
from datetime import datetime
from textblob import TextBlob
import sqlite3
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Q
from .models import Product
from .models import Client



# ==========================
# 🔐 AUTHENTICATION VIEWS
# ==========================

def login_view(request):
    """Business owner login page."""
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            messages.error(request, "❌ Invalid username or password.")
    return render(request, "login.html")


def logout_view(request):
    """Logout user and redirect to login page."""
    logout(request)
    return redirect("login")


# ==========================
# 🏠 DASHBOARD HOME
# ==========================
import time
@login_required
def home(request):
    """Main dashboard home view."""
    context = {"server_session_id": int(time.time())}  # resets chatbot eash time
    return render(request, "home.html")

# ==========================
# 🛍️ PRODUCT CRUD
# ==========================

@login_required
def product_list(request):
    products_qs = Product.objects.all().order_by("-last_updated_date")
    paginator = Paginator(products_qs, 5)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)
    return render(request, "crm/product_list.html", {"page_obj": page_obj})


@login_required
def product_create(request):
    """Add a new product."""
    if request.method == "POST":
        form = ProductForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "✅ Product added successfully!")
            return redirect("product_list")
        else:
            messages.error(request, "⚠️ Invalid product data.")
    else:
        form = ProductForm()
    return render(request, "crm/product_form.html", {"form": form})


@login_required
def product_edit(request, pk):
    """Edit an existing product."""
    product = get_object_or_404(Product, pk=pk)
    if request.method == "POST":
        form = ProductForm(request.POST, instance=product)
        if form.is_valid():
            form.save()
            messages.success(request, "✅ Product updated successfully!")
            return redirect("product_list")
        else:
            messages.error(request, "⚠️ Error updating product.")
    else:
        form = ProductForm(instance=product)
    return render(request, "crm/product_form.html", {"form": form})


@login_required
def product_delete(request, pk):
    """Delete a product."""
    product = get_object_or_404(Product, pk=pk)
    product.delete()
    messages.success(request, "🗑️ Product deleted successfully.")
    return redirect("product_list")

# ==========================
# 🛍️ Client CRUD
# ==========================

@login_required
def client_list(request):
    clients_qs = Client.objects.all().order_by("-registration_date")
    paginator = Paginator(clients_qs, 5)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)
    return render(request, "crm/client_list.html", {"page_obj": page_obj})


@login_required
def client_create(request):
    """Add a new client."""
    if request.method == "POST":
        form = ClientForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "✅ Client added successfully!")
            return redirect("client_list")
        else:
            messages.error(request, "⚠️ Invalid client data.")
    else:
        form = ClientForm()
    return render(request, "crm/client_form.html", {"form": form})


@login_required
def client_edit(request, pk):
    """Edit an existing client."""
    clients = get_object_or_404(Client, pk=pk)
    if request.method == "POST":
        form = ClientForm(request.POST, instance=clients)
        if form.is_valid():
            form.save()
            messages.success(request, "✅ Client updated successfully!")
            return redirect("client_list")
        else:
            messages.error(request, "⚠️ Error updating client.")
    else:
        form = ClientForm(instance=clients)
    return render(request, "crm/client_form.html", {"form": form})


@login_required
def client_delete(request, pk):
    """Delete a client."""
    clients = get_object_or_404(Client, pk=pk)
    clients.delete()
    messages.success(request, "🗑️ Client deleted successfully.")
    return redirect("client_list")


# ==========================
# 📥 CSV IMPORT
# ==========================

@login_required
def import_csv(request):
    """Import or update crm from a CSV file."""
    if request.method == "POST" and request.FILES.get("csv_file"):
        file = request.FILES["csv_file"]
        df = pd.read_csv(file)

        # ✅ Required columns
        expected_cols = [
            "product_id", "product_name", "category",
            "discounted_price", "actual_price", "discount_percentage",
            "rating", "rating_count", "about_product",
            "stock", "creation_date", "status"
        ]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            messages.error(request, f"❌ Missing columns in CSV: {', '.join(missing)}")
            return redirect("import_csv")

        imported = 0
        for _, row in df.iterrows():
            Product.objects.update_or_create(
                product_id=row["product_id"],
                defaults={
                    "product_name": row["product_name"],
                    "category": row["category"],
                    "discounted_price": row["discounted_price"],
                    "actual_price": row["actual_price"],
                    "discount_percentage": row.get("discount_percentage", None),
                    "rating": row.get("rating", None),
                    "rating_count": row.get("rating_count", 0),
                    "about_product": row.get("about_product", ""),
                    "stock": row.get("stock", 0),
                    "status": row.get("status", "active"),
                },
            )
            imported += 1

        messages.success(request, f"✅ {imported} crm imported/updated successfully!")
        return redirect("product_list")

    return render(request, "crm/import_csv.html")



# ==========================
# Discount Predictions
# ==========================


@login_required
def predict_for_product(request, pk):
    """Predict discount for one product using ML model."""
    product = get_object_or_404(Product, pk=pk)
    product.discount_percentage = predict_discount(product)
    product.save()
    messages.success(request, f"🔮 Predicted discount: {product.discount_percentage:.2f}% for {product.product_name}")
    return redirect("product_list")

@login_required
def predict_all_discounts(request):
    """Predict discounts for all products missing a value."""
    products = Product.objects.filter(discount_percentage__isnull=True)
    count = 0
    for p in products:
        p.discount_percentage = predict_discount(p)
        p.save()
        count += 1
    messages.success(request, f"🔮 Predicted discounts for {count} products .")
    return redirect("product_list")



# ==========================
# CLV Predictions
# ==========================



@login_required
def predict_for_client(request, pk):
    """Predict clv for one client using ML model."""
    clients = get_object_or_404(Client, pk=pk)
    clients.predicted_clv = predict_clv(clients)
    clients.save()
    messages.success(request, f"🔮 Predicted clv: {clients.predicted_clv:.2f} for {clients.customer_id}")
    return redirect("client_list")

@login_required
def predict_all_clv(request):
    """Predict CLV for all clients using ML model."""
    clients = Client.objects.all()
    count = 0
    for c in clients:
        c.predicted_clv = predict_clv(c)
        c.save()
        count += 1
        print(f"✅ Predicted CLV for {c.customer_id}: {c.predicted_clv:.2f}")

    messages.success(request, f"🔮 Predicted CLV for {count} clients.")
    print(f"🎯 CLV prediction completed for {count} clients.")
    return redirect("client_list")



# ==========================
# AI Assistant
# ==========================



# === Configure Groq ===
client = Groq(api_key="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")

# ===== Load & Index Product Data Once =====
print("⏳ Building TF-IDF index ")
import subprocess

conn = sqlite3.connect("db.sqlite3")

df_products = pd.read_sql_query("SELECT * FROM crm_product", conn)
df_clients = pd.read_sql_query("SELECT * FROM crm_client", conn)

conn.close()

print(f"✅ Loaded {len(df_products)} products and {len(df_clients)} clients.")


# Combine relevant text columns for search

# Products
df_products["search_text"] = df_products.apply(
    lambda x: f"{x['product_name']} {x['category']} {x['about_product']} {x['rating']} {x['rating_count']}", axis=1
)

# Clients
df_clients["search_text"] = df_clients.apply(
    lambda x: f"{x['name']} {x['region']} {x['gender']} "
              f"Income:{x['income_segment']} Loyalty:{x['loyalty_status']} "
              f"Spent:{x['total_spent']} Orders:{x['total_orders']} Active:{x['is_active']}",
    axis=1
)
# Fit vectorizers separately
vectorizer_prod = TfidfVectorizer(stop_words="english").fit(df_products["search_text"])
vectorizer_cli = TfidfVectorizer(stop_words="english").fit(df_clients["search_text"])

# Transform data
tfidf_products = vectorizer_prod.transform(df_products["search_text"])
tfidf_clients = vectorizer_cli.transform(df_clients["search_text"])

print("✅ Indexed both products and clients.")


# ===== Helper: Retrieve Most Relevant Rows =====
def get_relevant_products(query, top_n=30):
    q_vec = vectorizer_prod.transform([query])
    sims = cosine_similarity(q_vec, tfidf_products).flatten()
    top_idx = sims.argsort()[-top_n:][::-1]
    return df_products.iloc[top_idx], sims[top_idx]

def get_relevant_clients(query, top_n=30):
    q_vec = vectorizer_cli.transform([query])
    sims = cosine_similarity(q_vec, tfidf_clients).flatten()
    top_idx = sims.argsort()[-top_n:][::-1]
    return df_clients.iloc[top_idx], sims[top_idx]

def get_best_context(query):
    prod_df, prod_scores = get_relevant_products(query)
    cli_df, cli_scores = get_relevant_clients(query)

    if max(cli_scores) > max(prod_scores):
        return "client", cli_df
    return "product", prod_df


# ===== Chatbot Endpoint =====
@csrf_exempt
@login_required
def chatbot(request):
    if request.method != "POST":
        return JsonResponse({"response": "Use POST with a 'message' field."})

    user_message = request.POST.get("message", "")
    if not user_message:
        return JsonResponse({"response": "Please enter a question."})

    # Step 1: Retrieve relevant data
    relevant_products, prod_scores = get_relevant_products(user_message)
    relevant_clients, cli_scores = get_relevant_clients(user_message)

    top_prod_score = float(prod_scores.max()) if len(prod_scores) > 0 else 0
    top_cli_score = float(cli_scores.max()) if len(cli_scores) > 0 else 0
    min_relevance = 0.2

    # Step 2: Detect context
    client_keywords = ["client", "customer", "user", "buyer", "loyalty", "region", "income", "segment"]
    marketing_keywords = [
        "improve", "boost", "increase", "strategy", "sell more", "marketing",
        "campaign", "promotion", "discount", "sale", "performance", "growth",
        "visibility", "conversion", "retention"
    ]

    # Context decision logic
    if any(word in user_message.lower() for word in marketing_keywords):
        context_type = "marketing_advice"
        relevant_df_prod = df_products.copy()
        relevant_df_cli = df_clients.copy()
    elif any(word in user_message.lower() for word in client_keywords):
        context_type = "client"
        relevant_df_prod = None
        relevant_df_cli = relevant_clients
    elif top_prod_score < min_relevance and top_cli_score < min_relevance:
        context_type = "fallback"
        relevant_df_prod = df_products.sample(min(10, len(df_products)))
        relevant_df_cli = None
    else:
        context_type = "product" if top_prod_score >= top_cli_score else "client"
        relevant_df_prod = relevant_products if context_type == "product" else None
        relevant_df_cli = relevant_clients if context_type == "client" else None

    # Step 3: Prepare data for context
    csv_products = relevant_df_prod.to_csv(index=False) if relevant_df_prod is not None else ""
    csv_clients = relevant_df_cli.to_csv(index=False) if relevant_df_cli is not None else ""

    # Step 4: Build system prompt
    if context_type == "client":
        system_prompt = f"""
You are an AI assistant analyzing CLIENT data for a business owner.
Answer clearly based on the following CSV data only:

{csv_clients}

If the question is not about clients, respond:
"I can only provide insights related to your client data."
"""
    elif context_type == "product":
        system_prompt = f"""
You are an AI assistant analyzing PRODUCT data for a business owner.
Use the data below to provide accurate, structured answers.

{csv_products}

If the question is not about products, say:
"I can only provide insights related to your product data."
"""
    elif context_type == "marketing_advice":

        # Sample only a few representative rows to reduce token count
        prod_sample = df_products.sample(min(30, len(df_products)))[
            ["product_name", "category", "discounted_price", "actual_price",
             "discount_percentage", "rating", "rating_count", "stock"]
        ]
        cli_sample = df_clients.sample(min(30, len(df_clients)))[
            ["name", "region", "income_segment", "loyalty_status",
             "total_spent", "total_orders", "is_active"]
        ]

        system_prompt = (
            "You are an AI marketing strategist assistant integrated into a CRM system.\n\n"
            "Your job is to analyze sales performance and give clear, actionable recommendations.\n"
            "Use these datasets to identify opportunities and suggest business actions:\n"
            "- Which products are underperforming?\n"
            "- What can be promoted, discounted, or bundled?\n"
            "- Which clients or segments can be reactivated?\n\n"
            "If numeric reasoning is needed (e.g., lowest rating or stock), do simple comparisons.\n\n"
            "Client Data (sampled):\n"
            f"{cli_sample.to_csv(index=False)}\n\n"
            "Product Data (sampled):\n"
            f"{prod_sample.to_csv(index=False)}\n\n"
            "Always answer with practical marketing advice, not generic text.\n"
            "If the data is unclear, say: 'I need more detailed sales information to advise further.'"
        )

    else:  # fallback
        system_prompt = f"""
You are an AI assistant helping a business owner by suggesting products from their stock 
that could be relevant to the user's request.

The user may ask questions that are not about existing products or clients.

If you can connect the question to any of the products below, recommend 3–5 that might be useful or related.
If not, answer: "I can only provide suggestions or information about your current stock and client base."

Available products:
{csv_products}
"""

    # Step 5: Call Groq model
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.45,
            max_tokens=1200,
        )
        answer = completion.choices[0].message.content
        return JsonResponse({"response": answer})

    except Exception as e:
        return JsonResponse({"response": f"⚠️ Error: {str(e)}"})




from django.shortcuts import render

@login_required
def chatbot_page(request):
    """Render the chatbot UI page."""
    return render(request, "crm/chatbot.html")

# ================================
# Similarity System - text based
# ================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

@login_required
def similar_products(request, pk):
    """Show crm similar to the selected product using NLP similarity."""
    from .models import Product

    # Load all crm
    products = Product.objects.all()
    if not products:
        messages.error(request, "No crm in database.")
        return redirect("product_list")

    # Convert to DataFrame for NLP
    df = pd.DataFrame(list(products.values()))
    if "about_product" not in df.columns:
        messages.error(request, "Missing product description data.")
        return redirect("product_list")

    # Create a text corpus combining relevant columns
    df["text"] = df.apply(
        lambda x: f"{x['product_name']} {x['category']} {x.get('about_product','')}", axis=1
    )

    # TF-IDF encoding
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["text"])

    # Find index of selected product
    idx = df.index[df["id"] == pk][0]

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()

    # Get top 5 similar products (excluding itself)
    similar_idx = cosine_sim.argsort()[-6:][::-1]  # top 5 + self
    similar_idx = [i for i in similar_idx if i != idx]
    similar_df = df.iloc[similar_idx][:5]

    # Get Django Product objects for template
    similar_objs = Product.objects.filter(id__in=similar_df["id"].tolist())

    return render(request, "crm/similar_products.html", {
        "product": Product.objects.get(pk=pk),
        "similar_products": similar_objs,
    })



# ==========================
# Reviews analysis
# ==========================

def _label_sentiment(score: float) -> str:
    if score > 0.1:
        return "Positive"
    if score < -0.1:
        return "Negative"
    return "Neutral"


def _safe_date(s):
    try:
        return datetime.fromisoformat(str(s))
    except Exception:
        return None


@login_required
def reviews_dashboard(request):
    """
    Professional review analytics:
      - analyze ALL reviews (about_product) with TextBlob
      - filters: category, status, min_rating, date range
      - charts + KPIs
      - paginated detailed table
    """

    # --------- Filters from query params ----------
    category = request.GET.get("category")  # exact category string
    status = request.GET.get("status")      # e.g. active/inactive/draft
    min_rating = request.GET.get("min_rating")
    date_from = request.GET.get("date_from")  # YYYY-MM-DD
    date_to = request.GET.get("date_to")      # YYYY-MM-DD

    qs = Product.objects.exclude(
        Q(about_product__isnull=True) | Q(about_product__exact="")
    )

    if category and category.lower() != "all":
        qs = qs.filter(category=category)

    if status and status.lower() != "all":
        qs = qs.filter(status=status)

    if min_rating:
        try:
            qs = qs.filter(rating__gte=float(min_rating))
        except Exception:
            pass

    if date_from:
        dtf = _safe_date(date_from)
        if dtf:
            qs = qs.filter(creation_date__gte=dtf.date())

    if date_to:
        dtt = _safe_date(date_to)
        if dtt:
            qs = qs.filter(creation_date__lte=dtt.date())

    # Nothing to analyze?
    if not qs.exists():
        # For filter dropdowns
        categories = list(Product.objects.values_list("category", flat=True).distinct())
        statuses = list(Product.objects.values_list("status", flat=True).distinct())
        return render(
            request,
            "crm/reviews_dashboard.html",
            {
                "no_data": True,
                "categories": categories,
                "statuses": statuses,
            },
        )

    # --------- DataFrame for ALL (filtered) reviews ----------
    df = pd.DataFrame(list(qs.values(
        "product_id", "product_name", "category", "status", "rating",
        "rating_count", "about_product", "creation_date", "last_updated_date",
        "discounted_price", "actual_price", "discount_percentage", "stock"
    )))

    # Sentiment on ALL rows
    df["sentiment"] = df["about_product"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df["sentiment_label"] = df["sentiment"].apply(_label_sentiment)

    # --------- KPIs / Charts data ----------
    total_reviews = int(len(df))  # ✅ ALL analyzed reviews
    avg_sentiment = round(df["sentiment"].mean(), 3)
    sentiment_counts = df["sentiment_label"].value_counts().to_dict()

    # Average sentiment by category
    cat_df = (
        df.groupby("category")["sentiment"]
        .mean()
        .reset_index()
        .sort_values("sentiment", ascending=False)
    )
    category_sentiment = cat_df.to_dict(orient="records")

    # Top positive / negative (for insights cards)
    top_positive = (
        df.sort_values("sentiment", ascending=False)
        .head(3)[["product_name", "category", "sentiment"]]
        .to_dict(orient="records")
    )
    top_negative = (
        df.sort_values("sentiment", ascending=True)
        .head(3)[["product_name", "category", "sentiment"]]
        .to_dict(orient="records")
    )

    # --------- Paginated review table (ALL rows) ----------
    page_number = request.GET.get("page", 1)
    paginator = Paginator(
        df[["product_name", "category", "sentiment_label", "sentiment", "about_product"]]
        .sort_values("sentiment", ascending=True),  # show most negative first
        10,  # reviews per page
    )
    page_obj = paginator.get_page(page_number)

    # Dropdown source lists
    categories = list(Product.objects.values_list("category", flat=True).distinct())
    statuses = list(Product.objects.values_list("status", flat=True).distinct())

    context = {
        # Filters
        "categories": categories,
        "statuses": statuses,
        "selected_category": category or "all",
        "selected_status": status or "all",
        "selected_min_rating": min_rating or "",
        "selected_date_from": date_from or "",
        "selected_date_to": date_to or "",

        # KPIs
        "total_reviews": total_reviews,          # ✅ now reflects ALL analyzed rows
        "avg_sentiment": avg_sentiment,
        "sentiment_counts": json.dumps(sentiment_counts),
        "category_sentiment": json.dumps(category_sentiment),
        "top_positive": top_positive,
        "top_negative": top_negative,

        # Table
        "page_obj": page_obj,
    }
    return render(request, "crm/reviews_dashboard.html", context)


@login_required
def reviews_export(request):
    """Export the FULL analyzed dataset (with sentiment) as CSV for BI/Excel."""
    qs = Product.objects.exclude(
        Q(about_product__isnull=True) | Q(about_product__exact="")
    )
    if not qs.exists():
        return HttpResponse("No reviews to export.", content_type="text/plain")

    df = pd.DataFrame(list(qs.values(
        "product_id", "product_name", "category", "status", "rating",
        "rating_count", "about_product", "creation_date", "last_updated_date",
        "discounted_price", "actual_price", "discount_percentage", "stock"
    )))
    df["sentiment"] = df["about_product"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df["sentiment_label"] = df["sentiment"].apply(_label_sentiment)

    buff = io.StringIO()
    df.to_csv(buff, index=False)
    buff.seek(0)

    resp = HttpResponse(buff.getvalue(), content_type="text/csv")
    resp["Content-Disposition"] = 'attachment; filename="reviews_sentiment_export.csv"'
    return resp



# ==========================
# Similar Image system
# ==========================
import base64
import traceback
import sqlite3
import pandas as pd
from groq import Groq
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required


@login_required
@csrf_exempt
def similar_from_image(request):
    """Upload an image → LLaMA predicts correct category using real dataset context."""
    if request.method != "POST" or "image" not in request.FILES:
        messages.error(request, "Please upload a product image.")
        return redirect("product_list")

    try:
        # === Encode uploaded image ===
        image_file: InMemoryUploadedFile = request.FILES["image"]
        image_bytes = image_file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # === Load categories + sample descriptions from DB ===
        conn = sqlite3.connect("db.sqlite3")
        df = pd.read_sql_query(
            "SELECT category, about_product FROM crm_product WHERE about_product IS NOT NULL AND TRIM(about_product) != ''",
            conn,
        )
        conn.close()

        if df.empty:
            messages.error(request, "No data found in database.")
            return redirect("product_list")

        # Extract first 3 levels of category hierarchy
        def extract_first_three_levels(cat: str) -> str:
            if not cat or "|" not in cat:
                return cat.strip() if cat else ""
            parts = cat.split("|")
            return "|".join(parts[:3]).strip()

        df["main_category"] = df["category"].astype(str).apply(extract_first_three_levels)

        # Get one representative product per main category
        examples = (
            df.groupby("main_category")["about_product"]
            .apply(lambda x: x.iloc[0])
            .reset_index()
        )

        # Build contextual text
        context = "\n\n".join(
            f"Category: {row['main_category']}\nExample: {row['about_product']}"
            for _, row in examples.iterrows()
        )

        # === Build system prompt with grounded examples ===
        system_prompt = f"""
You are an AI vision assistant for an e-commerce platform.

Below are examples of product categories with real product descriptions:

{context}

Your task:
- Look carefully at the uploaded image.
- Identify which ONE of these categories best matches what you see.
- Return ONLY the category name exactly as written above (no punctuation, no explanation).
Be precise and consistent with the existing data structure.
"""

        client = Groq(api_key="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")

        # === Ask model to classify ===
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Which category does this product belong to?\n"
                               f"<image src='data:image/jpeg;base64,{image_b64}' />"
                },
            ],
            temperature=0.2,
            max_tokens=200,
        )

        predicted_category = completion.choices[0].message.content.strip()
        print(f"🧭 Predicted category: {predicted_category}")

        # === Retrieve all products in that category ===
        from .models import Product
        similar_objs = Product.objects.filter(category__startswith=predicted_category)

        if not similar_objs.exists():
            # fallback: match on top 2 levels if nothing found
            fallback = "|".join(predicted_category.split("|")[:2])
            similar_objs = Product.objects.filter(category__startswith=fallback)

        if not similar_objs.exists():
            messages.warning(
                request,
                f"No products found under category '{predicted_category}'.",
            )
            return redirect("product_list")

        # === Render page ===
        return render(
            request,
            "crm/similar_products.html",
            {
                "product": None,
                "query_description": f"Predicted Category: {predicted_category}",
                "similar_products": similar_objs,
            },
        )

    except Exception as e:
        print("⚠️ ERROR in similar_from_image:", traceback.format_exc())
        messages.error(request, f"Error while analyzing image: {str(e)}")
        return redirect("product_list")

# ==============================================
# NEWSLETTER (Email + WhatsApp via Twilio)
# ==============================================
from django.core.mail import send_mail
from django.conf import settings
from twilio.rest import Client as TwilioClient
import sqlite3

@login_required
def send_newsletter(request):
    """Send a newsletter (email + WhatsApp) to all CRM clients."""
    if request.method == "POST":
        subject = request.POST.get("subject")
        message = request.POST.get("message")
        action = request.POST.get("action") # detects which button was clicked

        if not subject or not message:
            messages.error(request, "❌ Please fill both subject and message.")
            return redirect("send_newsletter")

        # === 1️⃣ Fetch emails & phone numbers from crm_client ===
        conn = sqlite3.connect("db.sqlite3")
        df_clients = pd.read_sql_query("SELECT * FROM crm_clientnewsletter", conn)
        conn.close()

        emails = df_clients["email"].dropna().unique().tolist()
        phones = df_clients["phone"].dropna().unique().tolist() if "phone" in df_clients.columns else []

        # === If user clicked “Send Email” ===
        if action == "email":
            sent_emails = 0
            for email in emails:
                try:
                    send_mail(
                        subject,
                        message,
                        settings.DEFAULT_FROM_EMAIL,
                        [email],
                        fail_silently=False,
                    )
                    sent_emails += 1
                except Exception as e:
                    print(f"⚠️ Email error for {email}: {e}")

            messages.success(
                request,
                f"✅ Newsletter sent successfully to {sent_emails} email(s)."
            )

        # === If user clicked “Send WhatsApp Message” ===
        elif action == "whatsapp":
            twilio_sid = "AC02f4e34953e5a229eda0fafce078d262"
            twilio_auth = "40321a0abcc307fcc9b1904db82c482b"
            twilio_whatsapp = "whatsapp:+14155238886"  # Twilio sandbox number

            client = TwilioClient(twilio_sid, twilio_auth)
            sent_whatsapp = 0

            for phone in phones:
                try:
                    if not str(phone).startswith("+"):
                        continue  # skip invalid numbers
                    client.messages.create(
                        from_=twilio_whatsapp,
                        body=message,
                        to=f"whatsapp:{phone}",
                    )
                    sent_whatsapp += 1
                except Exception as e:
                    print(f"⚠️ WhatsApp error for {phone}: {e}")

            messages.success(
                request,
                f"✅ WhatsApp newsletter sent to {sent_whatsapp} contact(s)."
            )

        return redirect("send_newsletter")  # or render again if you prefer

    return render(request, "newsletter_form.html")

# ==============================================
# Automated emails (Birthday & Risk Clients only)
# ==============================================
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import MessageTemplate


@login_required
def manage_templates(request):
    """Add or update templates for Birthday and At-Risk Client automated emails."""
    templates = MessageTemplate.objects.filter(event_type__in=["birthday", "risk"]).order_by("-id")

    if request.method == "POST":
        event_type = request.POST.get("event_type")
        subject = request.POST.get("subject")
        message = request.POST.get("message")
        channel = request.POST.get("channel")
        active = bool(request.POST.get("active"))

        if not subject or not message:
            messages.error(request, "❌ Please fill both subject and message.")
            return redirect("manage_templates")

        MessageTemplate.objects.create(
            event_type=event_type,
            subject=subject,
            message=message,
            channel="email",
            active=active
        )

        messages.success(
            request,
            f"✅ New {event_type.title()} template saved ({channel})."
        )
        return redirect("manage_templates")

    return render(request, "crm/manage_templates.html", {"templates": templates})


from django.http import JsonResponse
from .models import MessageTemplate
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
@require_POST
def toggle_template_status(request):
    """Toggle the 'active' status of a message template dynamically."""
    template_id = request.POST.get("id")
    try:
        tpl = MessageTemplate.objects.get(id=template_id)
        tpl.active = not tpl.active
        tpl.save()
        return JsonResponse({
            "success": True,
            "active": tpl.active,
            "status": "Active" if tpl.active else "Inactive"
        })
    except MessageTemplate.DoesNotExist:
        return JsonResponse({"success": False, "error": "Template not found."})


@login_required
def view_template(request, pk):
    tpl = get_object_or_404(MessageTemplate, pk=pk)
    return render(request, "crm/view_template.html", {"tpl": tpl})


@login_required
def edit_template(request, pk):
    tpl = get_object_or_404(MessageTemplate, pk=pk)

    if request.method == "POST":
        tpl.subject = request.POST.get("subject")
        tpl.message = request.POST.get("message")
        tpl.active = bool(request.POST.get("active"))
        tpl.save()
        messages.success(request, "✅ Template updated successfully.")
        return redirect("manage_templates")

    return render(request, "crm/edit_template.html", {"tpl": tpl})


@login_required
def delete_template(request, pk):
    tpl = get_object_or_404(MessageTemplate, pk=pk)
    tpl.delete()
    messages.success(request, "🗑 Template deleted successfully.")
    return redirect("manage_templates")



# ==============================================
# Fuzzy search with words
# ==============================================

from django.http import JsonResponse
from django.db.models import Q
from .models import Product

def ajax_search_products(request):
    query = request.GET.get("q", "").strip()
    category_filter = request.GET.get("category", "")
    min_price = request.GET.get("min_price", "")
    max_price = request.GET.get("max_price", "")
    results = []

    if query:
        words = query.split()
        filters = Q()
        for word in words:
            filters &= (
                Q(product_name__icontains=word)
                | Q(category__icontains=word)
                | Q(about_product__icontains=word)
            )

        # Apply category filter
        if category_filter:
            filters &= Q(category__iexact=category_filter)

        # Apply price filters
        if min_price:
            filters &= Q(unit_price__gte=min_price)
        if max_price:
            filters &= Q(unit_price__lte=max_price)

        products = Product.objects.filter(filters)[:20]

        results = [
            {
                "name": p.product_name,
                "category": p.category,
                "actual_price": p.actual_price,
                "discounted_price": p.discounted_price,
            }
            for p in products
        ]

    return JsonResponse({"results": results})


# ==============================================
# FastAPI : Alert Notification for at risk clients
# ==============================================
import requests
from django.shortcuts import render
from django.contrib.auth.decorators import login_required

@login_required
def notifications(request):
    """Fetch and filter at-risk clients from the FastAPI marketing endpoint."""
    notifications = []
    count = 0
    error = None

    # --- Get filter parameters from query string ---
    selected_region = request.GET.get("region", "All")
    selected_risk = request.GET.get("risk", "All")

    try:
        api_url = "http://127.0.0.1:8000/api/notify/at_risk_clients"
        response = requests.get(api_url, timeout=10)
        print("🔍 Status:", response.status_code)
        print("🔍 Response text:", response.text[:200])

        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text[:100]}")

        data = response.json()
        notifications = data.get("notifications", [])
        count = data.get("count", len(notifications))

        # --- Apply filters ---
        if selected_region != "All":
            notifications = [n for n in notifications if str(n.get("region", "")).lower() == selected_region.lower()]

        if selected_risk != "All":
            notifications = [n for n in notifications if n.get("risk_level", "").lower() == selected_risk.lower()]

        count = len(notifications)

        # --- Extract unique regions for dropdown ---
        regions = sorted(set([n.get("region", "Unknown") for n in data.get("notifications", [])]))

    except Exception as e:
        error = str(e)
        regions = []
        print("⚠️ Error:", error)

    return render(
        request,
        "crm/notifications.html",
        {
            "notifications": notifications,
            "count": count,
            "error": error,
            "regions": regions,
            "selected_region": selected_region,
            "selected_risk": selected_risk,
        },
    )
