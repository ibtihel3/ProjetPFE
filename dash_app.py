import pandas as pd
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
# ==============================
# Load Data
# ==============================
clients = pd.read_csv("data/dash_plotly/clients_dashboard_ready.csv")
products = pd.read_csv("data/processed/products.csv")
kpis = pd.read_csv("data/dash_plotly/segment_kpis.csv")
rfm = pd.read_csv("data/dash_plotly/rfm_scores.csv")
cohorts = pd.read_csv("data/dash_plotly/cohorts.csv")
clients_overview = pd.read_csv("data/dash_plotly/clients_overview.csv")
products_overview = pd.read_csv("data/dash_plotly/products_overview.csv")
transactions = pd.read_csv("data/processed/transactions.csv")
revenue_category = pd.read_csv("data/dash_plotly/sales_by_category.csv")
top_products = pd.read_csv("data/dash_plotly/top_rated_products.csv")
discount_impact = pd.read_csv("data/dash_plotly/discount_impact.csv")

print("✅ All dashboard datasets loaded successfully.")
print(f"Clients: {clients.shape}, KPIs: {kpis.shape}, RFM: {rfm.shape}, Cohorts: {cohorts.shape}")

# ==============================
# App Setup
# ==============================
from django_plotly_dash import DjangoDash
app = DjangoDash("dash_app", external_stylesheets=[dbc.themes.BOOTSTRAP])

# ==============================
# Layout
# ==============================
app.layout = html.Div([
    html.H1("📊 BoWise CRM Analytics Dashboard",
            style={"textAlign": "center", "margin": "20px 0"}),

    dcc.Tabs(
        id="tabs",
        value="segmentation",
        children=[
            dcc.Tab(label="🧍 Clients Overview", value="clients"),
            dcc.Tab(label="📊 Segmentation", value="segmentation"),
            dcc.Tab(label="💄 Products Overview", value="products"),
            dcc.Tab(label="📈 RFM Analysis", value="rfm"),
            dcc.Tab(label="📊 Cohort Analysis", value="cohort"),
        ],
    ),
    html.Div(id="tabs-content", style={"padding": "0", "margin": "0"})
], style={"width": "100%", "height": "100vh", "padding": "0", "margin": "0"})

# ============================================================
# CLIENT OVERVIEW
# ============================================================
active_clients = clients_overview[clients_overview["is_active"] == True].shape[0]
new_clients = clients_overview.shape[0]  # refine if you track registration month
aov = round(clients_overview["avg_order_value"].mean(), 2)
retained = round((clients_overview["frequency"] > 1).mean() * 100, 2)
clients_by_source = clients_overview[clients_overview["source"] == True].shape[0]

geo_fig = px.bar(
    clients_overview.groupby("region")["customer_id"].count().reset_index(),
    x="region", y="customer_id", color="region",
    title="🌍 Clients by Region",
    text_auto=True
)
source_fig = px.bar(
    clients_overview.groupby("source")["customer_id"].count().reset_index(),
    x="source", y="customer_id", color="source",
    title=" 🛒 Clients by Source",
    text_auto=True
)
# ============================================================
# PRODUCT OVERVIEW
# ============================================================

#top rated
top_fig = px.bar(
    top_products,
    x="product_id", y="rating", color="rating",
    title="🏆 Top-Rated Products", text_auto=True
)
#  REVENUE PER CATEGORY
rev_fig = px.bar(
    revenue_category.sort_values("actual_price", ascending=True),
    x="actual_price",
    y="category",
    orientation="h",
    color="actual_price",
    title="💵 Revenue by Category",
    color_continuous_scale=px.colors.sequential.Teal,
    text="actual_price",
)

rev_fig.update_traces(
    texttemplate="$%{text:,.0f}",   # clean text format inside bars
    textposition="inside",
    insidetextanchor="middle",
    hovertemplate="<b>%{y}</b><br>Revenue: $%{x:,.2f}<extra></extra>",
)

rev_fig.update_layout(
    xaxis_title="Total Revenue ($)",
    yaxis=dict(showticklabels=False),  # hides category names for clean look
    coloraxis_showscale=False,         # hides color legend
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
    margin=dict(t=60, b=40, l=40, r=40),
    hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial"),
)



#  STOCK LEVELS
stock_fig = px.bar(
    products_overview.sort_values(by="stock", ascending=False).head(10),
    x="stock",
    y="product_name",
    orientation="h",
    color="category",
    title="📦 Stock Levels by Product",
    color_discrete_sequence=px.colors.qualitative.Vivid,
    hover_data={"product_name": True, "stock": ":,.0f", "category": True}
)

stock_fig.update_traces(
    text=None,
    hovertemplate="<b>%{customdata[0]}</b><br>Stock: %{x:,}<br>Category: %{customdata[2]}<extra></extra>"
)

stock_fig.update_layout(
    xaxis_title="Stock Quantity",
    yaxis=dict(showticklabels=False),   # hides product names
    showlegend=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=60, b=40, l=40, r=40),
    hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial")
)
# discount impact
disc_fig = px.scatter(
    discount_impact,
    x="discount_percentage", y="avg_rating",
    color="avg_stock", size="product_count",
    title="📊 Discount Impact on Ratings & Stock",
    hover_data=["avg_price"]
)

# ==============================
# Tab Callbacks
# ==============================

@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    # --- Clients Overview ---
    if tab == "clients":
        return html.Div([
            html.H4("Client Overview KPIs", style={"textAlign": "center"}),

            dbc.Row([
                dbc.Col(html.H5(f"🧍 Active Clients: {active_clients}")),
                dbc.Col(html.H5(f"🆕 New Clients: {new_clients}")),
                dbc.Col(html.H5(f"💰 Avg. Order Value: {aov}")),
                dbc.Col(html.H5(f"📈 Retention: {retained}%")),
            ], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(figure=geo_fig), md=12)]),
            dbc.Row([dbc.Col(dcc.Graph(figure=source_fig), md=12)]),
        ])

    # --- SEGMENTATION TAB ---
    elif tab == "segmentation":
        seg_counts = clients["segment_label"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        fig1 = px.bar(
            seg_counts, x="Segment", y="Count", color="Segment",
            title="Customer Count by Segment", text_auto=True
        )

        fig2 = px.box(
            clients, x="segment_label", y="total_spent",
            color="segment_label",
            title="Spending Distribution per Segment"
        )

        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig1), md=6),
                dbc.Col(dcc.Graph(figure=fig2), md=6)
            ])
        ])
    # --- Products Overview ---
    elif tab == "products":
        return html.Div([
            html.H4("Product Overview KPIs", style={"textAlign": "center"}),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=top_fig), md=6),
                dbc.Col(dcc.Graph(figure=rev_fig), md=6),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=stock_fig), md=6),
                dbc.Col(dcc.Graph(figure=disc_fig), md=6),
            ])
        ])
    # --- RFM ANALYSIS TAB ---
    elif tab == "rfm":
        # Scatter: Frequency vs Monetary (RFM bubble)
        rfm["RFM_Score"] = pd.to_numeric(rfm["RFM_Score"], errors="coerce")
        rfm_fig = px.scatter(
            rfm,
            x="frequency", y="monetary",
            color="Segment", size="RFM_Score",
            hover_name="customer_id",
            hover_data={
                "recency_days": True, "R": True, "F": True, "M": True
            },
            title="Customer Value Distribution (RFM Segments)"
        )

        # Bar: average RFM Score by Segment
        rfm_bar = px.bar(
            rfm.groupby("Segment")["RFM_Score"].mean().reset_index(),
            x="Segment", y="RFM_Score", color="Segment",
            text_auto=True,
            title="Average RFM Score per Segment"
        )

        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=rfm_fig), md=7),
                dbc.Col(dcc.Graph(figure=rfm_bar), md=5),
            ])
        ])

    # --- COHORT ANALYSIS TAB ---
    elif tab == "cohort":
        # Make sure the cohort_month column is string-like
        if "cohort_month" in cohorts.columns:
            cohorts["cohort_month"] = cohorts["cohort_month"].astype(str)
            cohorts_heatmap = cohorts.set_index("cohort_month")

            # Drop any non-numeric columns if present
            cohorts_heatmap = cohorts_heatmap.select_dtypes(include=["number"])
        else:
            cohorts_heatmap = cohorts.copy()

        fig = px.imshow(
            cohorts_heatmap,
            color_continuous_scale="YlGnBu",
            labels=dict(
                x="Months Since First Purchase",
                y="Cohort Month",
                color="Retention Rate"
            ),
            title="Customer Retention Heatmap (Cohort Analysis)",
            aspect="auto"
        )

        return html.Div([dcc.Graph(figure=fig)])


print("✅ Dash app successfully imported and registered!")


