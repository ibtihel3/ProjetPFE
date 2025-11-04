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
        value="clients",
        children=[
            dcc.Tab(label="🧍 Clients Overview", value="clients"),
            dcc.Tab(label="💄 Products Overview", value="products"),
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
rev_fig = px.pie(
    revenue_category,
    names="category",
    values="actual_price",
    hole=0.45,  # donut style
    title="💵 Revenue Share by Category",
    color_discrete_sequence=px.colors.sequential.Teal,
)

# --- Simplify visual text ---
rev_fig.update_traces(
    textinfo="none",   # no label , no percentage on chart
    textfont_size=12,
    hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.2f}<extra></extra>",
)

# --- Layout cleanup ---
rev_fig.update_layout(
    showlegend=False,  # 👈 hides the legend for ultra-clean look
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=60, b=40, l=40, r=40),
    title_x=0.5,
)




#  STOCK LEVELS
html.Div([
    html.H4("📦 Product Overview", style={"textAlign": "center"}),

    # Dropdown for sorting logic
    html.Div([
        html.Label("View by:", style={"fontWeight": "bold", "marginRight": "10px"}),
        dcc.Dropdown(
            id="stock-view",
            options=[
                {"label": "Top 10 by Stock", "value": "stock_desc"},
                {"label": "Low Stock (Restock Soon)", "value": "stock_asc"},
                {"label": "Top Rated Products", "value": "rating"},
                {"label": "High Revenue Products", "value": "revenue"}
            ],
            value="stock_desc",  # default
            clearable=False,
            style={"width": "300px"}
        )
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    # Chart
    dcc.Graph(id="stock-graph"),
])


@app.callback(
    Output("stock-graph", "figure"),
    Input("stock-view", "value")
)
def update_stock_chart(view_option):
    df_display = products_overview.copy()

    # Add computed revenue column if missing (Always compute revenue safely)
    df_display["revenue"] = (
            df_display.get("actual_price", 0) * df_display.get("stock", 0)
    )

    # Select sorting method
    if view_option == "stock_desc":
        df_display = df_display.sort_values(by="stock", ascending=False).head(10)
        title = "📦 Top 10 Products by Stock"
    elif view_option == "stock_asc":
        df_display = df_display.sort_values(by="stock", ascending=True).head(10)
        title = "⚠️ Low Stock Products (Restock Soon)"
    elif view_option == "rating":
    # Detect best available rating column
        rating_col = next(
            (col for col in ["rating", "avg_rating", "review_rating", "rating_value"] if col in df_display.columns),
            None
        )

        if rating_col:
            df_display = df_display.sort_values(by=rating_col, ascending=False).head(10)
            title = "⭐ Top Rated Products"
        else:
            df_display["rating_placeholder"] = 0
            df_display = df_display.head(10)
            title = "⭐ Top Rated Products ( Some products are not rated ! )"
    elif view_option == "revenue":
        df_display = df_display.sort_values(by="revenue", ascending=False).head(10)
        title = "💰 Top Revenue-Generating Products"
    else:
        df_display = df_display.head(10)
        title = "📦 Stock Levels by Product"

    # Truncate long product names
    df_display["product_display"] = df_display["product_name"].apply(
        lambda x: x if len(str(x)) <= 40 else str(x)[:37] + "..."
    )

    # Create chart
    fig = px.bar(
        df_display,
        x="stock",
        y="product_name",
        orientation="h",
        color="category",
        title=title,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        text="stock"
    )

    fig.update_traces(
        customdata=df_display[["product_display", "category"]],
        texttemplate="%{text:,}",
        textposition="outside",
        hovertemplate="<b>%{customdata[0]}</b><br>"
                      "Stock: %{x:,}<br>"
                      "Category: %{customdata[1]}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="Stock Quantity",
        yaxis_title="",
        yaxis=dict(showticklabels=False),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.95)",
            font_size=11,
            font_family="Arial",
            align="left",
            namelength=0
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40, l=80, r=40),
        title_x=0.5
    )

    return fig



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
    if tab == "clients":
        # === Clients Overview KPIs ===
        client_kpi_section = html.Div([
            html.H4("Client Overview KPIs", style={"textAlign": "center"}),

            dbc.Row([
                dbc.Col(html.H5(f"🧍 Active Clients: {active_clients}")),
                dbc.Col(html.H5(f"🆕 New Clients: {new_clients}")),
                dbc.Col(html.H5(f"💰 Avg. Order Value: {aov}")),
                dbc.Col(html.H5(f"📈 Retention: {retained}%")),
            ], className="mb-4"),

            dbc.Row([dbc.Col(dcc.Graph(figure=geo_fig), md=12)]),
            dbc.Row([dbc.Col(dcc.Graph(figure=source_fig), md=12)]),
        ], className="mb-5")

        # === Segmentation ===
        seg_counts = clients["segment_label"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        # Define a consistent color mapping for all segment labels
        segment_labels = seg_counts["Segment"].unique().tolist()

        # You can use any Plotly qualitative palette you like
        colors = px.colors.qualitative.Set3  # or "Vivid", "Pastel", "Bold", etc.
        color_map = {segment: colors[i % len(colors)] for i, segment in enumerate(segment_labels)}

        # --- Bar chart: count per segment ---
        seg_fig1 = px.bar(
            seg_counts,
            x="Segment",
            y="Count",
            color="Segment",
            title="🧮 Customer Count by Segment",
            text_auto=True,
            color_discrete_map=color_map  # 👈 enforce same colors
        )

        # --- Pie chart: spending share per segment ---
        seg_fig2 = px.pie(
            clients,
            names="segment_label",
            values="total_spent",
            title="💰 Spending Share per Segment",
            color="segment_label",
            color_discrete_map=color_map  # 👈 same color mapping
        )

        # Optional for better labels
        seg_fig2.update_traces(textinfo="percent+label")

        segmentation_section = html.Div([
            html.H4("Customer Segmentation", style={"textAlign": "center"}),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=seg_fig1), md=6),
                dbc.Col(dcc.Graph(figure=seg_fig2), md=6)
            ])
        ], className="mb-5")

        # === RFM Analysis ===
        rfm["RFM_Score"] = pd.to_numeric(rfm["RFM_Score"], errors="coerce")

        rfm_avg = rfm.groupby("Segment", as_index=False).agg({
            "monetary": "mean",
            "frequency": "mean",
            "recency_days": "mean"
        })

        rfm_fig = px.bar(
            rfm_avg.sort_values("monetary", ascending=True),
            y="Segment",
            x="monetary",
            orientation="h",
            color="Segment",
            title="💵 Average Spending per RFM Segment",
            text_auto=".2f",
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        rfm_fig.update_layout(
            xaxis_title="Average Monetary Value ($)",
            yaxis_title="RFM Segment",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )

        rfm_bar = px.bar(
            rfm.groupby("Segment")["RFM_Score"].mean().reset_index(),
            x="Segment", y="RFM_Score", color="Segment",
            text_auto=True,
            title="Average RFM Score per Segment"
        )

        rfm_section = html.Div([
            html.H4("RFM Analysis", style={"textAlign": "center"}),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=rfm_fig), md=7),
                dbc.Col(dcc.Graph(figure=rfm_bar), md=5),
            ])
        ], className="mb-5")

        # === Cohort Analysis ===
        if "cohort_month" in cohorts.columns:
            cohorts["cohort_month"] = cohorts["cohort_month"].astype(str)
            cohorts_heatmap = cohorts.set_index("cohort_month")
            cohorts_heatmap = cohorts_heatmap.select_dtypes(include=["number"])
        else:
            cohorts_heatmap = cohorts.copy()

        cohort_fig = px.imshow(
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

        cohort_section = html.Div([
            html.H4("Cohort Analysis", style={"textAlign": "center"}),
            dbc.Row([dbc.Col(dcc.Graph(figure=cohort_fig), md=12)])
        ])

        # === Combine all client sections ===
        return html.Div([
            client_kpi_section,
            segmentation_section,
            rfm_section,
            cohort_section
        ])

    elif tab == "products":
        # === Product Overview ===
        return html.Div([
            html.H4("Product Overview KPIs", style={"textAlign": "center"}),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=top_fig), md=6),
                dbc.Col(dcc.Graph(figure=rev_fig), md=6),
            ], className="mb-4"),
        # --- Dynamic Stock View Section ---
        html.Div([
            html.Label("View products by:", style={"fontWeight": "bold", "marginRight": "10px"}),
            dcc.Dropdown(
                id="stock-view",
                options=[
                    {"label": "Top 10 by Stock", "value": "stock_desc"},
                    {"label": "Low Stock (Restock Soon)", "value": "stock_asc"},
                    {"label": "Top Rated Products", "value": "rating"},
                    {"label": "High Revenue Products", "value": "revenue"}
                ],
                value="stock_desc",
                clearable=False,
                style={"width": "300px", "margin": "0 auto"}
            ),
            dcc.Graph(id="stock-graph")
        ], style={"textAlign": "center", "marginTop": "20px"}),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=disc_fig), md=12),
            ])
        ])


print("✅ Dash app successfully imported and registered!")


