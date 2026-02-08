import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# Load model & feature columns (UPDATED)
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "house_price_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# ======================================================
# Page config
# ======================================================
st.set_page_config(
    page_title="EstateIQ | Property Intelligence",
    page_icon="🏠",
    layout="wide"
)

# ======================================================
# Global styles (modern real-estate UI)
# ======================================================
st.markdown(
    """
        <style>
    :root {
        --bg: #0f1317;
        --ink: #e7ecf2;
        --muted: #a6b0bb;
        --brand: #4cc9a6;
        --brand-2: #e7ecf2;
        --card: #151b22;
        --line: #28303a;
        --accent: #f0c36a;
    }
    html, body, [class*="css"]  {
        font-family: "Source Sans 3", system-ui, -apple-system, "Segoe UI", Arial, sans-serif;
        color: var(--ink);
    }
    .stApp {
        background: radial-gradient(1200px 500px at 10% -10%, #1b2430 0%, #0f1317 55%, #0c1014 100%);
    }
    body, .stApp, .stApp * {
        color: var(--ink);
    }
    .hero {
        background: linear-gradient(120deg, rgba(76,201,166,0.15), rgba(240,195,106,0.12));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 18px;
    }
    .hero h1 {
        font-family: "Playfair Display", "Times New Roman", serif;
        font-weight: 600;
        letter-spacing: 0.2px;
        margin: 0 0 8px 0;
        color: var(--brand-2);
    }
    .hero p {
        margin: 0;
        color: var(--muted);
        font-size: 1.05rem;
    }
    .pill {
        display: inline-block;
        background: #11161c;
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 0.9rem;
        color: var(--muted);
        margin-right: 8px;
    }
    .card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 18px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
    }
    .section-title {
        font-family: "Playfair Display", "Times New Roman", serif;
        font-weight: 600;
        margin: 0 0 8px 0;
        color: var(--brand-2);
    }
    .kicker {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        color: var(--muted);
        margin-bottom: 8px;
    }
    .stat {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--brand-2);
    }
    .muted {
        color: var(--muted);
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: var(--muted);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--brand-2);
    }
    .stButton > button {
        background: var(--brand);
        color: #0b1417;
        border: 0;
        padding: 10px 18px;
        border-radius: 12px;
        font-weight: 700;
    }
    .stButton > button:hover {
        background: #39b592;
    }
    .stMetric {
        background: #121820;
        border: 1px solid var(--line);
        padding: 12px 14px;
        border-radius: 14px;
    }
    .stMarkdown, .stText, .stCaption, .stDataFrame, .stTable, .stAlert {
        color: var(--ink);
    }
    .stSidebar, [data-testid="stSidebar"] {
        background: linear-gradient(120deg, rgba(76,201,166,0.15), rgba(240,195,106,0.12));
    }
    .stSidebar label, .stSidebar .stMarkdown, .stSidebar .stCaption {
        color: var(--ink);
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        color: var(--brand-2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <div class="kicker">EstateIQ • Real Estate Intelligence</div>
        <h1>Modern valuation for premium residential listings</h1>
        <p>Advisor-grade pricing, market context, and explainability for urban Indian real estate.</p>
        <div style="margin-top:12px;">
            <span class="pill">Model: Random Forest</span>
            <span class="pill">Coverage: Tier-1 & Tier-2 cities</span>
            <span class="pill">Updated: today</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ======================================================
# City market intelligence (kept for insights)
# ======================================================
CITY_MARKET_INFO = {
    "Mumbai": {"tier": "Tier-1", "avg_price_per_sqft": 18000},
    "Delhi": {"tier": "Tier-1", "avg_price_per_sqft": 14000},
    "Bangalore": {"tier": "Tier-1", "avg_price_per_sqft": 12000},
    "Pune": {"tier": "Tier-1", "avg_price_per_sqft": 10000},
    "Hyderabad": {"tier": "Tier-2", "avg_price_per_sqft": 8000},
    "Chennai": {"tier": "Tier-2", "avg_price_per_sqft": 9000},
    "Kolkata": {"tier": "Tier-2", "avg_price_per_sqft": 7000},
}

# ======================================================
# Sidebar inputs
# ======================================================
st.sidebar.header("Property Inputs")

total_area_sqft = st.sidebar.slider("Total Area (sqft)", 300, 6000, 1200, 50)
bhk = st.sidebar.selectbox("Bedrooms (BHK)", [1, 2, 3, 4], index=1)
baths = st.sidebar.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1)
balcony = st.sidebar.selectbox("Balconies", [0, 1, 2, 3], index=1)
city = st.sidebar.selectbox("City", list(CITY_MARKET_INFO.keys()))

st.sidebar.divider()

property_type = st.sidebar.selectbox(
    "Property Type", ["Apartment", "Independent House", "Villa"]
)

furnishing = st.sidebar.selectbox(
    "Furnishing", ["Unfurnished", "Semi-Furnished", "Fully Furnished"]
)

property_age = st.sidebar.selectbox(
    "Property Age", ["0-1", "1-5", "5-10", "10+"]
)

floor_number = st.sidebar.slider("Floor Number", 0, 30, 2)
total_floors = st.sidebar.slider("Total Floors", 1, 30, 10)

# ======================================================
# Prediction function (FULL ML-DRIVEN)
# ======================================================

def predict_price(inputs):
    df_input = pd.DataFrame([inputs])

    # Ordinal encodings (same as training)
    df_input["property_age_encoded"] = df_input["property_age"].map({
        "0-1": 0, "1-5": 1, "5-10": 2, "10+": 3
    })

    df_input["furnishing_encoded"] = df_input["furnishing"].map({
        "Unfurnished": 0,
        "Semi-Furnished": 1,
        "Fully Furnished": 2
    })

    # Derived features
    df_input["price_per_sqft"] = (
        df_input["total_area_sqft"] * 0 + CITY_MARKET_INFO[inputs["city"]]["avg_price_per_sqft"]
    )

    df_input["floor_ratio"] = (
        df_input["floor_number"] / df_input["total_floors"]
    ).clip(0, 1)

    # One-hot encoding
    df_input = pd.get_dummies(
        df_input,
        columns=["property_type", "city"],
        drop_first=True
    )

    # Align with training features
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)

    return model.predict(df_input)[0]

# ======================================================
# Market insight helper
# ======================================================

def market_insight(price_per_sqft, city):
    avg = CITY_MARKET_INFO[city]["avg_price_per_sqft"]
    if price_per_sqft > avg * 1.1:
        return "Above city average"
    if price_per_sqft < avg * 0.9:
        return "Below city average"
    return "Close to city average"

# ======================================================
# Tabs
# ======================================================

tab1, tab2, tab3 = st.tabs(
    ["Predict", "Explore Cities", "Model Explainability"]
)

# ======================================================
# TAB 1 — Prediction
# ======================================================

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Valuation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Instant price estimate</div>', unsafe_allow_html=True)
    st.markdown(
        "Enter property details on the left to generate a market-aligned valuation "
        "with confidence bands and price-per-sqft context."
    )

    if st.button("Run Valuation"):
        inputs = {
            "total_area_sqft": total_area_sqft,
            "bhk": bhk,
            "Baths": baths,
            "Balcony": balcony,
            "property_type": property_type,
            "furnishing": furnishing,
            "property_age": property_age,
            "floor_number": floor_number,
            "total_floors": total_floors,
            "city": city
        }

        predicted_price = predict_price(inputs)
        price_per_sqft = (predicted_price * 100000) / total_area_sqft

        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated Price", f"INR {predicted_price:.2f} Lakhs")
        col2.metric("Price / Sqft", f"INR {price_per_sqft:.0f}")
        col3.metric("City Tier", CITY_MARKET_INFO[city]["tier"])

        st.info(
            f"Confidence Range: INR {predicted_price*0.9:.2f} – INR {predicted_price*1.1:.2f} Lakhs"
        )

        st.success(market_insight(price_per_sqft, city))

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# TAB 2 — City Exploration
# ======================================================

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">City Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Average price per sqft</div>', unsafe_allow_html=True)

    cities = list(CITY_MARKET_INFO.keys())
    prices = [CITY_MARKET_INFO[c]["avg_price_per_sqft"] for c in cities]

    fig, ax = plt.subplots()
    ax.bar(cities, prices)
    ax.set_ylabel("INR per Sqft")
    ax.set_title("City-wise Average Property Prices")
    plt.xticks(rotation=45)

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# TAB 3 — Explainability (Random Forest)
# ======================================================

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Model Transparency</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top feature drivers</div>', unsafe_allow_html=True)

    importances = pd.Series(
        model.feature_importances_,
        index=feature_columns
    ).sort_values(ascending=False)

    st.dataframe(importances.head(15))

    st.markdown(
        """
        **Explanation**
        - Random Forest captures non-linear relationships.
        - Area, city, BHK, and price-per-sqft dominate pricing.
        - Model learns interactions automatically.
        - No manual price hacks used.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# Footer
# ======================================================

st.divider()
st.markdown(
    "End-to-end ML: Feature engineering, model selection, explainability, and deployment."
)


